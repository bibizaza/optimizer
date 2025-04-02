# File: optima_optimizer.py

import streamlit as st
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

# scikit-optimize (if used for hyperparam)
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

# Import your custom excel_loader
from modules.data_loading.excel_loader import parse_excel

# Constraints
from modules.analytics.constraints import get_main_constraints

# Rolling monthly logic => 4 approaches
from modules.backtesting.rolling_monthly import (
    rolling_backtest_monthly_param_sharpe,
    rolling_backtest_monthly_direct_sharpe,
    rolling_backtest_monthly_param_cvar,
    rolling_backtest_monthly_direct_cvar,
    backtest_buy_and_hold_that_drifts,
    backtest_strategic_rebalanced,
    compute_cost_impact,
    rolling_grid_search
)

# Extended metrics
from modules.analytics.extended_metrics import compute_extended_metrics

# Excel export
from modules.export.export_backtest_to_excel import export_backtest_results_to_excel


###############################################################################
# Optional: If you want multi-portfolio display in the same file,
# otherwise keep it in display_utils.py
###############################################################################
def display_multi_portfolio_metrics(metrics_map: dict):
    """
    metrics_map is { "Optimized": {...}, "OldDrift": {...}, "Strategic": {...}, ... }
    Each value is a dict from compute_extended_metrics.

    We'll display 3 sections (Performance, Risk, Ratios),
    with each portfolio as a column in the DataFrame.
    """
    # Decide which metrics go in each category
    performance_keys= ["Total Return","Annual Return","Annual Vol","Sharpe"]
    risk_keys       = ["MaxDD","TimeToRecovery","VaR_1M99","CVaR_1M99"]
    ratio_keys      = ["Skew","Kurtosis","Sortino","Calmar","Omega"]

    # Build a helper to create a DataFrame with one row per metric,
    # one column per portfolio name in metrics_map.
    def build_dataframe_for_keys(metric_keys, metrics_map):
        port_names = list(metrics_map.keys())
        rows = []
        for mk in metric_keys:
            row = [mk]  # first col is metric name
            for pname in port_names:
                row.append(metrics_map[pname].get(mk, 0.0))
            rows.append(row)
        df_ = pd.DataFrame(rows, columns=["Metric"] + port_names)
        df_.set_index("Metric", inplace=True)
        return df_

    # Formatting function for multi-column data
    def format_ext_metrics_multi(df_):
        pct_metrics= ["Total Return","Annual Return","Annual Vol","MaxDD","VaR_1M99","CVaR_1M99"]
        dfx = df_.copy()
        for mk in dfx.index:
            for c_ in dfx.columns:
                val = dfx.loc[mk, c_]
                if mk in pct_metrics:
                    dfx.loc[mk, c_] = f"{val*100:.2f}%"
                elif mk=="TimeToRecovery":
                    dfx.loc[mk, c_] = f"{val:.0f}"
                else:
                    dfx.loc[mk, c_] = f"{val:.3f}"
        return dfx

    # Build & display each category
    for (title, keys) in [
        ("Performance", performance_keys),
        ("Risk",        risk_keys),
        ("Ratios",      ratio_keys)
    ]:
        st.write(f"### Extended Metrics - {title}")
        df_cat = build_dataframe_for_keys(keys, metrics_map)
        st.dataframe(format_ext_metrics_multi(df_cat))


###############################################################################
# 1) Data Loading & Constraints
###############################################################################
def sidebar_data_and_constraints():
    st.sidebar.title("Data Loading")
    approach_data = st.sidebar.radio(
        "Data Source Approach",
        ["One-time Convert Excel->Parquet", "Use Excel for Analysis", "Use Parquet for Analysis"],
        index=1
    )

    df_instruments = pd.DataFrame()
    df_prices = pd.DataFrame()
    coverage = 0.8
    main_constr = {}

    if approach_data == "One-time Convert Excel->Parquet":
        st.sidebar.info("Excel->Parquet converter not shown here.")
        return df_instruments, df_prices, coverage, main_constr

    elif approach_data == "Use Excel for Analysis":
        excel_file = st.sidebar.file_uploader("Upload Excel (.xlsx or .xlsm)", type=["xlsx","xlsm"])
        if excel_file:
            df_instruments, df_prices = parse_excel(excel_file)
            if not df_prices.empty:
                coverage = st.sidebar.slider("Min coverage fraction", 0.0, 1.0, 0.8, 0.05)
                st.sidebar.markdown("---")
                st.sidebar.subheader("Constraints & Costs")
                main_constr = get_main_constraints(df_instruments, df_prices)
            else:
                st.sidebar.warning("No valid data in Excel.")
    else:
        st.sidebar.info("Upload instruments.parquet & prices.parquet")
        fi = st.sidebar.file_uploader("Upload instruments.parquet", type=["parquet"])
        fp = st.sidebar.file_uploader("Upload prices.parquet", type=["parquet"])
        if fi and fp:
            df_instruments = pd.read_parquet(fi)
            df_prices = pd.read_parquet(fp)
            if not df_prices.empty:
                coverage = st.sidebar.slider("Min coverage fraction", 0.0, 1.0, 0.8, 0.05)
                st.sidebar.markdown("---")
                st.sidebar.subheader("Constraints & Costs")
                main_constr = get_main_constraints(df_instruments, df_prices)
            else:
                st.sidebar.warning("No valid data in Parquet files.")

    return df_instruments, df_prices, coverage, main_constr


###############################################################################
# 2) Utility: Clean df_prices
###############################################################################
def clean_df_prices(df_prices: pd.DataFrame, min_coverage=0.8) -> pd.DataFrame:
    df_prices = df_prices.copy()
    coverage = df_prices.notna().sum(axis=1)
    n_cols = df_prices.shape[1]
    threshold = n_cols * min_coverage
    df_prices = df_prices[coverage >= threshold]
    df_prices = df_prices.sort_index().fillna(method="ffill").fillna(method="bfill")
    df_prices.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_prices.fillna(method="ffill", inplace=True)
    df_prices.fillna(method="bfill", inplace=True)
    return df_prices


###############################################################################
# 3) Main app
###############################################################################
def main():
    st.title("Optimize Your Portfolio (Markowitz & CVaR)")

    # A) Load data & constraints
    df_instruments, df_prices, coverage, main_constr = sidebar_data_and_constraints()
    if df_instruments.empty or df_prices.empty or not main_constr:
        st.stop()

    # B) Clean & subset
    df_prices_clean = clean_df_prices(df_prices, coverage)
    user_start = main_constr["user_start"]
    df_sub = df_prices_clean.loc[pd.Timestamp(user_start):]
    if len(df_sub) < 2:
        st.error("Not enough data after your chosen start date.")
        st.stop()

    # Extract constraints
    class_sum_constraints = main_constr["class_sum_constraints"]
    subtype_constraints   = main_constr["subtype_constraints"]
    daily_rf              = main_constr["daily_rf"]
    cost_type             = main_constr["cost_type"]
    transaction_cost_val  = main_constr["transaction_cost_value"]
    trade_buffer_pct      = main_constr["trade_buffer_pct"]

    # If #Quantity exist => build Weight_Old
    if "#Quantity" in df_instruments.columns and "#Last_Price" in df_instruments.columns:
        df_instruments["Value"] = df_instruments["#Quantity"] * df_instruments["#Last_Price"]
        tot_val = df_instruments["Value"].sum()
        if tot_val <= 0:
            tot_val = 1.0
        df_instruments["Weight_Old"] = df_instruments["Value"] / tot_val
    else:
        df_instruments["Weight_Old"] = 0.0

    # C) Let user pick => Portfolio Optimization or Hyperparam
    top_choice = st.radio("Analysis Type:", ["Portfolio Optimization","Hyperparameter Optimization"], index=0)
    if top_choice == "Portfolio Optimization":

        solver_choice = st.selectbox(
            "Solver Approach:",
            [
                "Parametric (Markowitz)",
                "Direct (Markowitz)",
                "Parametric (CVaR)",
                "Direct (CVaR)"
            ],
            index=0
        )

        with st.expander("Rolling Parameters", expanded=False):
            rebal_freq = st.selectbox("Rebalance Frequency (months)", [1,3,6], index=0)
            lookback_m = st.selectbox("Lookback Window (months)", [3,6,12], index=0)
            window_days = lookback_m * 21

            # Some default init
            reg_cov         = False
            do_ledoitwolf   = False
            do_ewm          = False
            ewm_alpha       = 0.0
            do_shrink_means = False
            alpha_shrink    = 0.0
            do_shrink_cov   = False
            beta_shrink     = 0.0
            n_points_man    = 0

            cvar_alpha_in   = 0.95
            cvar_limit_in   = 0.10
            cvar_freq_user  = "daily"

            if solver_choice in ["Parametric (Markowitz)", "Direct (Markowitz)"]:
                reg_cov = st.checkbox("Regularize Cov?", False)
                do_ledoitwolf = st.checkbox("Use LedoitWolf Cov?", False)
                do_ewm = st.checkbox("Use EWM Cov?", False)
                ewm_alpha = st.slider("EWM alpha", 0.0,1.0,0.06,0.01)

                st.write("**Mean & Cov Shrink**")
                do_shrink_means = st.checkbox("Shrink Means?", True)
                alpha_shrink    = st.slider("Alpha (for means)", 0.0,1.0,0.3,0.01)
                do_shrink_cov   = st.checkbox("Shrink Cov (diagonal)?", True)
                beta_shrink     = st.slider("Beta (for cov)", 0.0,1.0,0.2,0.01)

                if solver_choice == "Parametric (Markowitz)":
                    n_points_man = st.number_input("Frontier #points (Param Only)", 5,100,15, step=5)
            else:
                cvar_alpha_in  = st.slider("CVaR alpha", 0.0,0.9999,0.95,0.01)
                cvar_limit_in  = st.slider("CVaR limit (Direct approach)", 0.0,1.0,0.10,0.01)
                cvar_freq_user = st.selectbox("CVaR Frequency", ["daily","weekly","monthly","annual"], index=0)

            run_rolling = st.button("Run Rolling")

        if run_rolling:
            # Build ticker lists
            col_tickers = df_sub.columns.tolist()
            have_sec = ("#Security_Type" in df_instruments.columns)
            asset_cls_list = []
            sec_type_list  = []
            for tk in col_tickers:
                row_ = df_instruments[df_instruments["#ID"] == tk]
                if not row_.empty:
                    asset_cls_list.append(row_["#Asset_Class"].iloc[0])
                    if have_sec:
                        stp = row_["#Security_Type"].iloc[0]
                        sec_type_list.append(stp if not pd.isna(stp) else "Unknown")
                    else:
                        sec_type_list.append("Unknown")
                else:
                    asset_cls_list.append("Unknown")
                    sec_type_list.append("Unknown")

            # Define solver callables
            def param_sharpe_fn(sub_ret: pd.DataFrame):
                from modules.optimization.cvxpy_parametric import parametric_max_sharpe_aclass_subtype
                w_opt, summary = parametric_max_sharpe_aclass_subtype(
                    df_returns=sub_ret,
                    tickers=col_tickers,
                    asset_classes=asset_cls_list,
                    security_types=sec_type_list,
                    class_constraints=class_sum_constraints,
                    subtype_constraints=subtype_constraints,
                    daily_rf=daily_rf,
                    no_short=True,
                    n_points=n_points_man,
                    regularize_cov=reg_cov,
                    shrink_means=do_shrink_means,
                    alpha=alpha_shrink,
                    shrink_cov=do_shrink_cov,
                    beta=beta_shrink,
                    use_ledoitwolf=do_ledoitwolf,
                    do_ewm=do_ewm,
                    ewm_alpha=ewm_alpha
                )
                return w_opt, summary

            def direct_sharpe_fn(sub_ret: pd.DataFrame):
                from modules.optimization.cvxpy_direct import direct_max_sharpe_aclass_subtype
                w_opt, summary = direct_max_sharpe_aclass_subtype(
                    df_returns=sub_ret,
                    tickers=col_tickers,
                    asset_classes=asset_cls_list,
                    security_types=sec_type_list,
                    class_constraints=class_sum_constraints,
                    subtype_constraints=subtype_constraints,
                    daily_rf=daily_rf,
                    no_short=True,
                    regularize_cov=reg_cov,
                    shrink_means=do_shrink_means,
                    alpha=alpha_shrink,
                    shrink_cov=do_shrink_cov,
                    beta=beta_shrink,
                    use_ledoitwolf=do_ledoitwolf,
                    do_ewm=do_ewm,
                    ewm_alpha=ewm_alpha
                )
                return w_opt, summary

            def param_cvar_fn(sub_ret: pd.DataFrame, old_w: np.ndarray):
                from modules.optimization.cvxpy_parametric_cvar import dynamic_min_cvar_with_fallback
                w_opt, summary = dynamic_min_cvar_with_fallback(
                    df_returns=sub_ret,
                    tickers=col_tickers,
                    asset_classes=asset_cls_list,
                    security_types=sec_type_list,
                    class_constraints=class_sum_constraints,
                    subtype_constraints=subtype_constraints,
                    old_w=old_w,
                    cvar_alpha=cvar_alpha_in,
                    no_short=True,
                    daily_rf=daily_rf,
                    freq_choice=cvar_freq_user,
                    clamp_factor=1.5,
                    max_weight_each=1.0,
                    max_iter=10
                )
                return w_opt, summary

            def direct_cvar_fn(sub_ret: pd.DataFrame):
                from modules.optimization.cvxpy_direct_cvar import direct_max_return_cvar_constraint_aclass_subtype
                w_opt, summary = direct_max_return_cvar_constraint_aclass_subtype(
                    df_returns=sub_ret,
                    tickers=col_tickers,
                    asset_classes=asset_cls_list,
                    security_types=sec_type_list,
                    class_constraints=class_sum_constraints,
                    subtype_constraints=subtype_constraints,
                    cvar_alpha=cvar_alpha_in,
                    cvar_limit=cvar_limit_in,
                    daily_rf=daily_rf,
                    no_short=True,
                    freq_choice=cvar_freq_user
                )
                return w_opt, summary

            # Run rolling
            if solver_choice == "Parametric (Markowitz)":
                sr_opt, final_w, _, _, df_rebal, extm_opt = rolling_backtest_monthly_param_sharpe(
                    df_prices=df_sub,
                    df_instruments=df_instruments,
                    param_sharpe_fn=param_sharpe_fn,
                    start_date=df_sub.index[0],
                    end_date=df_sub.index[-1],
                    months_interval=rebal_freq,
                    window_days=window_days,
                    transaction_cost_value=transaction_cost_val,
                    transaction_cost_type=cost_type,
                    trade_buffer_pct=trade_buffer_pct,
                    daily_rf=daily_rf
                )
            elif solver_choice == "Direct (Markowitz)":
                sr_opt, final_w, _, _, df_rebal, extm_opt = rolling_backtest_monthly_direct_sharpe(
                    df_prices=df_sub,
                    df_instruments=df_instruments,
                    direct_sharpe_fn=direct_sharpe_fn,
                    start_date=df_sub.index[0],
                    end_date=df_sub.index[-1],
                    months_interval=rebal_freq,
                    window_days=window_days,
                    transaction_cost_value=transaction_cost_val,
                    transaction_cost_type=cost_type,
                    trade_buffer_pct=trade_buffer_pct,
                    daily_rf=daily_rf
                )
            elif solver_choice == "Parametric (CVaR)":
                sr_opt, final_w, _, _, df_rebal, extm_opt = rolling_backtest_monthly_param_cvar(
                    df_prices=df_sub,
                    df_instruments=df_instruments,
                    param_cvar_fn=param_cvar_fn,
                    start_date=df_sub.index[0],
                    end_date=df_sub.index[-1],
                    months_interval=rebal_freq,
                    window_days=window_days,
                    transaction_cost_value=transaction_cost_val,
                    transaction_cost_type=cost_type,
                    trade_buffer_pct=trade_buffer_pct,
                    daily_rf=daily_rf
                )
            else:
                sr_opt, final_w, _, _, df_rebal, extm_opt = rolling_backtest_monthly_direct_cvar(
                    df_prices=df_sub,
                    df_instruments=df_instruments,
                    direct_cvar_fn=direct_cvar_fn,
                    start_date=df_sub.index[0],
                    end_date=df_sub.index[-1],
                    months_interval=rebal_freq,
                    window_days=window_days,
                    transaction_cost_value=transaction_cost_val,
                    transaction_cost_type=cost_type,
                    trade_buffer_pct=trade_buffer_pct,
                    daily_rf=daily_rf
                )

            extm_opt = extm_opt or {}

            # Build old drift if #Quantity
            sr_drift = pd.Series(dtype=float)
            extm_drift = {}
            if "#Quantity" in df_instruments.columns:
                ticker_qty_dict = {}
                for _, row in df_instruments.iterrows():
                    ticker_qty_dict[row["#ID"]] = row["#Quantity"]
                if ticker_qty_dict:
                    sr_drift = backtest_buy_and_hold_that_drifts(
                        df_prices=df_sub,
                        start_date=df_sub.index[0],
                        end_date=df_sub.index[-1],
                        ticker_qty=ticker_qty_dict
                    )
                    if not sr_drift.empty:
                        extm_drift = compute_extended_metrics(sr_drift, daily_rf=daily_rf)

            # Build old strategic => Weight_Old
            sr_strat = pd.Series(dtype=float)
            extm_strat = {}
            if "Weight_Old" in df_instruments.columns:
                w_old_map = {}
                for _, row in df_instruments.iterrows():
                    w_old_map[row["#ID"]] = row["Weight_Old"]
                w_arr = []
                for c in df_sub.columns:
                    w_arr.append(w_old_map.get(c, 0.0))
                w_arr = np.array(w_arr, dtype=float)
                sum_ = w_arr.sum()
                if sum_ > 1e-9:
                    w_arr /= sum_
                    sr_strat = backtest_strategic_rebalanced(
                        df_prices=df_sub,
                        start_date=df_sub.index[0],
                        end_date=df_sub.index[-1],
                        strategic_weights=w_arr,
                        months_interval=12,
                        transaction_cost_value=transaction_cost_val,
                        transaction_cost_type=cost_type,
                        daily_rf=daily_rf
                    )
                    if not sr_strat.empty:
                        extm_strat = compute_extended_metrics(sr_strat, daily_rf=daily_rf)

            # Store in session
            st.session_state["rolling_results"] = {
                "sr_opt":    sr_opt,
                "df_rebal":  df_rebal,
                "extm_opt":  extm_opt,

                "sr_drift":  sr_drift,
                "extm_drift":extm_drift,

                "sr_strat":  sr_strat,
                "extm_strat":extm_strat
            }

        # If we have results => show them
        if "rolling_results" in st.session_state:
            res = st.session_state["rolling_results"]
            sr_opt    = res["sr_opt"]
            sr_drift  = res["sr_drift"]
            sr_strat  = res["sr_strat"]

            df_rebal  = res["df_rebal"]
            extm_opt  = res["extm_opt"]
            extm_drift= res["extm_drift"]
            extm_strat= res["extm_strat"]

            st.write("## Rolling Backtest Comparison")
            show_opt   = st.checkbox("Optimized", value=(not sr_opt.empty))
            show_drift = st.checkbox("Old Drift (#Quantity)", value=(not sr_drift.empty))
            show_strat = st.checkbox("Old Strategic (Weight_Old)", value=(not sr_strat.empty))

            df_plot = pd.DataFrame()
            if show_opt and not sr_opt.empty:
                df_plot["Optimized"] = sr_opt / sr_opt.iloc[0] - 1.0
            if show_drift and not sr_drift.empty:
                df_plot["OldDrift"]  = sr_drift / sr_drift.iloc[0] - 1.0
            if show_strat and not sr_strat.empty:
                df_plot["OldStrategic"] = sr_strat / sr_strat.iloc[0] - 1.0

            if not df_plot.empty:
                st.line_chart(df_plot * 100.0)
            else:
                st.warning("No lines selected or data is missing.")

            # Transaction costs for optimized
            if not df_rebal.empty and not sr_opt.empty:
                final_opt_val = sr_opt.iloc[-1]
                cost_stats = compute_cost_impact(df_rebal, final_opt_val)
                st.write("### Transaction Cost Impact (Optimized Only)")
                st.dataframe(pd.DataFrame([cost_stats]))

            # Build metrics_map but only for checkboxes that are on
            metrics_map = {}
            if show_opt and not sr_opt.empty:
                metrics_map["Optimized"] = extm_opt
            if show_drift and not sr_drift.empty:
                metrics_map["Old Drift"] = extm_drift
            if show_strat and not sr_strat.empty:
                metrics_map["Old Strategic"] = extm_strat

            if metrics_map:
                st.write("## Extended Metrics: Selected Portfolios")
                display_multi_portfolio_metrics(metrics_map)
            else:
                st.info("No portfolios selected for metrics.")

            # Excel export (example)
            excel_bytes = export_backtest_results_to_excel(
                sr_line_new=sr_opt,
                sr_line_old=sr_drift,
                df_rebal=df_rebal,
                ext_metrics_new=extm_opt,
                ext_metrics_old=extm_drift
            )
            st.download_button(
                "Download Backtest Excel",
                data=excel_bytes,
                file_name="backtest_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        st.write("Hyperparameter Optimization placeholder.")


if __name__=="__main__":
    main()
