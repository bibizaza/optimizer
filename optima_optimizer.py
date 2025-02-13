# File: optima_optimizer.py

import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import io
import time
import traceback
from dateutil.relativedelta import relativedelta
import concurrent.futures

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

# Module-level imports from our project modules:
# (Make sure the paths match your actual folder structure)
from modules.optimization.cvxpy_optimizer import parametric_max_sharpe_aclass_subtype
from modules.backtesting.rolling_monthly import rolling_backtest_monthly_param_sharpe
from modules.analytics.returns_cov import compute_performance_metrics
from modules.analytics.weight_display import display_instrument_weight_diff, display_class_weight_diff
from modules.analytics.constraints import get_main_constraints

def parse_excel(file, streamlit_sheet="streamlit", histo_sheet="Histo_Price"):
    df_instruments = pd.read_excel(file, sheet_name=streamlit_sheet, header=0)
    df_prices_raw  = pd.read_excel(file, sheet_name=histo_sheet, header=0)
    if df_prices_raw.columns[0] != "Date":
        df_prices_raw.rename(columns={df_prices_raw.columns[0]: "Date"}, inplace=True)
    df_prices_raw["Date"] = pd.to_datetime(df_prices_raw["Date"], errors="coerce")
    df_prices_raw.dropna(subset=["Date"], inplace=True)
    df_prices_raw.set_index("Date", inplace=True)
    df_prices_raw.sort_index(inplace=True)
    df_prices_raw = df_prices_raw.apply(pd.to_numeric, errors="coerce")
    return df_instruments, df_prices_raw

def clean_df_prices(df_prices: pd.DataFrame, min_coverage=0.8) -> pd.DataFrame:
    """
    Removes rows where the fraction of non-null cells < min_coverage.
    Then ffill/bfill to fill remaining NaNs.
    """
    df_prices = df_prices.copy()
    coverage = df_prices.notna().sum(axis=1)
    n_cols = df_prices.shape[1]
    thr = n_cols * min_coverage
    df_prices = df_prices[coverage >= thr]
    df_prices = df_prices.sort_index().fillna(method="ffill").fillna(method="bfill")
    return df_prices

def build_old_portfolio_line(df_instruments: pd.DataFrame, df_prices: pd.DataFrame) -> pd.Series:
    """
    Creates a time series for the old portfolio's performance, given #Quantity.
    Normalizes it to 1 at the first date.
    """
    df_prices = df_prices.sort_index().fillna(method="ffill").fillna(method="bfill")
    ticker_qty = {}
    for _, row in df_instruments.iterrows():
        tkr = row["#ID"]
        qty = row["#Quantity"]
        ticker_qty[tkr] = qty

    col_list = df_prices.columns
    old_shares = np.array([ticker_qty.get(c, 0.0) for c in col_list])
    vals = [np.sum(old_shares * r.values) for _, r in df_prices.iterrows()]
    sr = pd.Series(vals, index=df_prices.index)
    if len(sr) > 0 and sr.iloc[0] <= 0:
        sr.iloc[0] = 1.0
    if len(sr) > 0:
        sr = sr / sr.iloc[0]
    sr.name = "Old_Ptf"
    return sr

def main():
    st.title("Optima with Security-Type Constraints")

    # 1) Data Source Selection
    approach_data = st.radio("Data Source Approach", 
                             ["One-time Convert Excel->Parquet", "Use Excel for Analysis", "Use Parquet for Analysis"])
    if approach_data == "One-time Convert Excel->Parquet":
        st.info("Excel-to-Parquet converter not shown here.")
        st.stop()

    if approach_data == "Use Excel for Analysis":
        excel_file = st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
        if not excel_file:
            st.stop()
        df_instruments, df_prices = parse_excel(excel_file)
    else:
        st.info("Upload instruments.parquet and prices.parquet")
        fi = st.file_uploader("Upload instruments.parquet", type=["parquet"])
        fp = st.file_uploader("Upload prices.parquet", type=["parquet"])
        if not fi or not fp:
            st.stop()
        df_instruments = pd.read_parquet(fi)
        df_prices = pd.read_parquet(fp)

    coverage = st.slider("Min coverage fraction", 0.0, 1.0, 0.8, 0.05)
    df_prices_clean = clean_df_prices(df_prices, coverage)
    st.write(f"**Clean data**: shape={df_prices_clean.shape}, from {df_prices_clean.index.min()} to {df_prices_clean.index.max()}")

    # 2) Get constraints from the UI
    main_constr = get_main_constraints(df_instruments, df_prices_clean)
    user_start = main_constr["user_start"]
    constraint_mode = main_constr["constraint_mode"]
    buffer_pct = main_constr["buffer_pct"]
    class_sum_constraints = main_constr["class_sum_constraints"]  # a dict: {className: {min_class_weight, max_class_weight}}
    subtype_constraints = main_constr["subtype_constraints"]      # a dict: {(class, secType): {min_instrument, max_instrument}}
    daily_rf = main_constr["daily_rf"]
    cost_type = main_constr["cost_type"]
    transaction_cost_value = main_constr["transaction_cost_value"]
    trade_buffer_pct = main_constr["trade_buffer_pct"]

    # 3) Build old portfolio weights
    df_instruments["Value"] = df_instruments["#Quantity"] * df_instruments["#Last_Price"]
    tot_val = df_instruments["Value"].sum()
    if tot_val <= 0:
        tot_val = 1.0
        df_instruments.loc[df_instruments.index[0], "Value"] = 1.0
    df_instruments["Weight_Old"] = df_instruments["Value"] / tot_val

    # 4) If 'keep_current' mode => override class_sum_constraints with old portfolio +/- buffer
    if constraint_mode == "keep_current":
        class_old_w = df_instruments.groupby("#Asset")["Weight_Old"].sum()
        for cl in df_instruments["#Asset"].unique():
            oldw = class_old_w.get(cl, 0.0)
            mn = max(0.0, oldw - buffer_pct)
            mx = min(1.0, oldw + buffer_pct)
            class_sum_constraints[cl] = {"min_class_weight": mn, "max_class_weight": mx}

    # 5) Subset the price data from user_start
    df_sub = df_prices_clean.loc[pd.Timestamp(user_start):]
    if len(df_sub) < 2:
        st.error("Not enough data from the selected start date.")
        st.stop()

    col_tickers = df_sub.columns.tolist()
    have_sec_type = ("#Security_Type" in df_instruments.columns)

    # Build lists: asset_cls_list, sec_type_list for each ticker
    asset_cls_list = []
    sec_type_list = []
    for tk in col_tickers:
        row_ = df_instruments[df_instruments["#ID"] == tk]
        if not row_.empty:
            asset_cls_list.append(row_["#Asset"].iloc[0])
            if have_sec_type:
                stp = row_["#Security_Type"].iloc[0]
                if pd.isna(stp):
                    stp = "Unknown"
                sec_type_list.append(stp)
            else:
                sec_type_list.append("Unknown")
        else:
            asset_cls_list.append("Unknown")
            sec_type_list.append("Unknown")

    approach = st.radio("Analysis Approach", ["Manual Single Rolling", "Grid Search", "Bayesian Optimization"], index=0)

    if approach == "Manual Single Rolling":
        rebal_freq = st.selectbox("Rebalance Frequency (months)", [1, 3, 6], index=0)
        lookback_m = st.selectbox("Lookback Window (months)", [3, 6, 12], index=0)
        window_days = lookback_m * 21

        reg_cov = st.checkbox("Regularize Cov?", False)
        do_ledoitwolf = st.checkbox("Use LedoitWolf Cov?", False)
        do_ewm = st.checkbox("Use EWM Cov?", False)
        ewm_alpha = st.slider("EWM alpha", 0.0, 1.0, 0.06, 0.01)

        st.write("**Mean & Cov Shrink**")
        do_shrink_means = st.checkbox("Shrink Means?", True)
        alpha_shrink = st.slider("Alpha (for means)", 0.0, 1.0, 0.3, 0.05)
        do_shrink_cov = st.checkbox("Shrink Cov (diagonal)?", True)
        beta_shrink = st.slider("Beta (for cov)", 0.0, 1.0, 0.2, 0.05)

        n_points_man = st.number_input("Frontier #points", 5, 100, 15, step=5)

        if st.button("Run Rolling (Manual)"):

            def param_sharpe_fn(sub_ret: pd.DataFrame):
                # Calls your parametric_max_sharpe_aclass_subtype solver
                w_opt, summary = parametric_max_sharpe_aclass_subtype(
                    df_returns = sub_ret,
                    tickers = col_tickers,
                    asset_classes = asset_cls_list,
                    security_types = sec_type_list,
                    class_constraints = class_sum_constraints,
                    subtype_constraints = subtype_constraints,
                    daily_rf = daily_rf,
                    no_short = True,
                    n_points = n_points_man,
                    regularize_cov = reg_cov,
                    shrink_means = do_shrink_means,
                    alpha = alpha_shrink,
                    shrink_cov = do_shrink_cov,
                    beta = beta_shrink,
                    use_ledoitwolf = do_ledoitwolf,
                    do_ewm = do_ewm,
                    ewm_alpha = ewm_alpha
                )
                return w_opt, summary

            # Use rolling backtest
            from modules.backtesting.rolling_monthly import rolling_backtest_monthly_param_sharpe
            sr_line, final_w, old_w_last, final_rebal_date, df_rebal = rolling_backtest_monthly_param_sharpe(
                df_prices = df_sub,
                df_instruments = df_instruments,
                param_sharpe_fn = param_sharpe_fn,
                start_date = df_sub.index[0],
                end_date = df_sub.index[-1],
                months_interval = rebal_freq,
                window_days = window_days,
                transaction_cost_value = transaction_cost_value,
                transaction_cost_type = cost_type,
                trade_buffer_pct = trade_buffer_pct
            )
            st.write("### Rebalance Debug Table")
            st.dataframe(df_rebal)

            old_line = build_old_portfolio_line(df_instruments, df_sub)
            idx_all = old_line.index.union(sr_line.index)
            old_line_u = old_line.reindex(idx_all, method="ffill")
            new_line_u = sr_line.reindex(idx_all, method="ffill")
            old0 = old_line_u.iloc[0]
            new0 = new_line_u.iloc[0]
            df_cum = pd.DataFrame({
                "Old(%)": (old_line_u/old0 - 1)*100,
                "New(%)": (new_line_u/new0 - 1)*100
            }, index=idx_all)
            st.line_chart(df_cum)

            perf_old = compute_performance_metrics(old_line_u * old0, daily_rf=daily_rf)
            perf_new = compute_performance_metrics(new_line_u * new0, daily_rf=daily_rf)
            df_perf = pd.DataFrame({"Old": perf_old, "New": perf_new})
            st.write("**Performance**:")
            st.dataframe(df_perf)

            display_instrument_weight_diff(df_instruments, col_tickers, final_w)
            display_class_weight_diff(df_instruments, col_tickers, asset_cls_list, final_w)

    elif approach == "Grid Search":
        st.subheader("Grid Search (Parallel) => security-type constraints")
        frontier_points_list = st.multiselect("Frontier Points (n_points)", [5,10,15,20,30], [5,10,15])
        alpha_list = st.multiselect("Alpha values", [0,0.1,0.2,0.3,0.4,0.5], [0,0.1,0.3])
        beta_list = st.multiselect("Beta values", [0,0.1,0.2,0.3,0.4,0.5], [0.1,0.2])
        rebal_freq_list = st.multiselect("Rebalance Frequency (months)", [1,3,6], [1,3])
        lookback_list = st.multiselect("Lookback (months)", [3,6,12], [3,6])
        max_workers = st.number_input("Max Workers", 1, 64, 4, step=1)

        if st.button("Run Grid Search"):
            if (not frontier_points_list or not alpha_list or not beta_list
                or not rebal_freq_list or not lookback_list):
                st.error("Please select at least one value in each parameter.")
            else:
                from modules.backtesting.rolling_monthly import rolling_grid_search
                df_gs = rolling_grid_search(
                    df_prices = df_sub,
                    df_instruments = df_instruments,
                    asset_cls_list = asset_cls_list,     # pass it here
                    sec_type_list = sec_type_list,       # pass it here
                    class_sum_constraints = class_sum_constraints,
                    subtype_constraints = subtype_constraints,
                    daily_rf = daily_rf,
                    frontier_points_list = frontier_points_list,
                    alpha_list = alpha_list,
                    beta_list = beta_list,
                    rebal_freq_list = rebal_freq_list,
                    lookback_list = lookback_list,
                    transaction_cost_value = transaction_cost_value,
                    transaction_cost_type = cost_type,
                    trade_buffer_pct = trade_buffer_pct,
                    do_shrink_means = True,
                    do_shrink_cov = True,
                    reg_cov = False,
                    do_ledoitwolf = False,
                    do_ewm = False,
                    ewm_alpha = 0.06,
                    max_workers = max_workers
                )
                st.dataframe(df_gs)
                best_ = df_gs.sort_values("Sharpe Ratio", ascending=False).head(5)
                st.write("**Top 5**:")
                st.dataframe(best_)

    else:
        st.subheader("Bayesian => security-type constraints")
        from modules.backtesting.rolling_bayesian import rolling_bayesian_optimization
        # pass the same lists & constraints
        rolling_bayesian_optimization(
            df_prices = df_sub,
            df_instruments = df_instruments,
            asset_cls_list = asset_cls_list,
            sec_type_list = sec_type_list,
            class_sum_constraints = class_sum_constraints,
            subtype_constraints = subtype_constraints,
            daily_rf = daily_rf,
            transaction_cost_value = transaction_cost_value,
            transaction_cost_type = cost_type,
            trade_buffer_pct = trade_buffer_pct
        )

if __name__ == "__main__":
    main()
