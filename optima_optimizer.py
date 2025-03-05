# File: optima_optimizer.py

import streamlit as st
import pandas as pd
import numpy as np
import traceback
import io
import time
from dateutil.relativedelta import relativedelta
import concurrent.futures

# scikit-optimize
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

# Constraints
from modules.analytics.constraints import get_main_constraints

# Weight display
from modules.analytics.weight_display import (
    display_instrument_weight_diff,
    display_class_weight_diff
)

# Extended metrics
from modules.analytics.extended_metrics import compute_extended_metrics

# Rolling monthly logic
from modules.backtesting.rolling_monthly import (
    rolling_backtest_monthly_param_sharpe,
    rolling_backtest_monthly_direct_sharpe,
    compute_cost_impact,
    rolling_grid_search
)

# Interval / drawdown
from modules.backtesting.rolling_intervals import display_interval_bars_and_stats
from modules.backtesting.max_drawdown import (
    plot_drawdown_series,
    show_max_drawdown_comparison
)

# Bayesian
from modules.backtesting.rolling_bayesian import rolling_bayesian_optimization

# Solvers
from modules.optimization.cvxpy_optimizer import (
    parametric_max_sharpe_aclass_subtype,
    direct_max_sharpe_aclass_subtype
)
# Optional efficient frontier
from modules.optimization.efficient_frontier import (
    compute_efficient_frontier_12m,
    interpolate_frontier_for_vol,
    plot_frontier_comparison
)


##########################################################################
# parse_excel helper
##########################################################################
def parse_excel(file, streamlit_sheet="streamlit", histo_sheet="Histo_Price"):
    df_instruments = pd.read_excel(file, sheet_name=streamlit_sheet, header=0)
    df_prices_raw = pd.read_excel(file, sheet_name=histo_sheet, header=0)
    if df_prices_raw.columns[0] != "Date":
        df_prices_raw.rename(columns={df_prices_raw.columns[0]: "Date"}, inplace=True)
    df_prices_raw["Date"] = pd.to_datetime(df_prices_raw["Date"], errors="coerce")
    df_prices_raw.dropna(subset=["Date"], inplace=True)
    df_prices_raw.set_index("Date", inplace=True)
    df_prices_raw.sort_index(inplace=True)
    df_prices_raw = df_prices_raw.apply(pd.to_numeric, errors="coerce")
    return df_instruments, df_prices_raw


##########################################################################
# sidebar_data_and_constraints
##########################################################################
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
        excel_file = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
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


##########################################################################
# clean_df_prices & build_old_portfolio_line
##########################################################################
def clean_df_prices(df_prices: pd.DataFrame, min_coverage=0.8) -> pd.DataFrame:
    df_prices = df_prices.copy()
    coverage = df_prices.notna().sum(axis=1)
    n_cols = df_prices.shape[1]
    threshold = n_cols * min_coverage
    df_prices = df_prices[coverage >= threshold]
    df_prices = df_prices.sort_index().fillna(method="ffill").fillna(method="bfill")
    return df_prices


def build_old_portfolio_line(df_instruments: pd.DataFrame, df_prices: pd.DataFrame) -> pd.Series:
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
    if len(sr)>0 and sr.iloc[0]<=0:
        sr.iloc[0]=1.0
    if len(sr)>0:
        sr= sr/ sr.iloc[0]
    sr.name= "Old_Ptf"
    return sr


##########################################################################
# main app
##########################################################################
def main():
    st.title("Optima Rolling Backtest + Extended Metrics (Param or Direct)")

    # 1) Data + constraints
    df_instruments, df_prices, coverage, main_constr = sidebar_data_and_constraints()
    if df_instruments.empty or df_prices.empty or not main_constr:
        st.stop()

    # Extract
    user_start = main_constr["user_start"]
    constraint_mode = main_constr["constraint_mode"]
    buffer_pct = main_constr["buffer_pct"]
    class_sum_constraints = main_constr["class_sum_constraints"]
    subtype_constraints = main_constr["subtype_constraints"]
    daily_rf = main_constr["daily_rf"]
    cost_type = main_constr["cost_type"]
    transaction_cost_value = main_constr["transaction_cost_value"]
    trade_buffer_pct = main_constr["trade_buffer_pct"]

    # 2) Clean data
    df_prices_clean = clean_df_prices(df_prices, coverage)
    st.write(f"**Clean data** => shape={df_prices_clean.shape}, range=({df_prices_clean.index.min()}..{df_prices_clean.index.max()})")

    # DEBUG / CLEAN => drop columns that remain fully NaN, replace inf
    nan_cols = [c for c in df_prices_clean.columns if df_prices_clean[c].isna().all()]
    if nan_cols:
        st.write("DEBUG => dropping columns fully NaN =>", nan_cols)
        df_prices_clean.drop(columns=nan_cols, inplace=True)

    df_prices_clean.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_prices_clean.fillna(method="ffill", inplace=True)
    df_prices_clean.fillna(method="bfill", inplace=True)

    # 3) Build old portfolio
    df_instruments["Value"] = df_instruments["#Quantity"]* df_instruments["#Last_Price"]
    tot_val = df_instruments["Value"].sum()
    if tot_val<=0:
        tot_val=1.0
        if not df_instruments.empty:
            df_instruments.loc[df_instruments.index[0], "Value"]=1.0
    df_instruments["Weight_Old"] = df_instruments["Value"]/ tot_val

    if constraint_mode=="keep_current":
        class_old_w = df_instruments.groupby("#Asset")["Weight_Old"].sum()
        for cl in df_instruments["#Asset"].unique():
            oldw= class_old_w.get(cl, 0.0)
            mn= max(0.0, oldw- buffer_pct)
            mx= min(1.0, oldw+ buffer_pct)
            class_sum_constraints[cl] = {
                "min_class_weight": mn,
                "max_class_weight": mx
            }

    # subset from user_start
    df_sub = df_prices_clean.loc[pd.Timestamp(user_start):]
    if len(df_sub)<2:
        st.error("Not enough data after your chosen start date.")
        st.stop()

    col_tickers = df_sub.columns.tolist()
    have_sec = ("#Security_Type" in df_instruments.columns)
    asset_cls_list= []
    sec_type_list= []
    for tk in col_tickers:
        row_ = df_instruments[df_instruments["#ID"]== tk]
        if not row_.empty:
            asset_cls_list.append(row_["#Asset"].iloc[0])
            if have_sec:
                stp= row_["#Security_Type"].iloc[0]
                if pd.isna(stp):
                    stp= "Unknown"
                sec_type_list.append(stp)
            else:
                sec_type_list.append("Unknown")
        else:
            asset_cls_list.append("Unknown")
            sec_type_list.append("Unknown")

    approach= st.radio(
        "Analysis Approach",
        [
            "Manual Single Rolling (Parametric)",
            "Manual Single Rolling (Direct)",
            "Grid Search",
            "Bayesian Optimization"
        ], index=0
    )

    ########################################################################
    # A) Manual Single Rolling
    ########################################################################
    if approach in ["Manual Single Rolling (Parametric)", "Manual Single Rolling (Direct)"]:
        with st.expander("Manual Rolling Parameters", expanded=False):
            rebal_freq= st.selectbox("Rebalance Frequency (months)", [1,3,6], index=0)
            lookback_m= st.selectbox("Lookback Window (months)", [3,6,12], index=0)
            window_days= lookback_m*21

            reg_cov= st.checkbox("Regularize Cov?", False)
            do_ledoitwolf= st.checkbox("Use LedoitWolf Cov?", False)
            do_ewm= st.checkbox("Use EWM Cov?", False)
            ewm_alpha= st.slider("EWM alpha", 0.0,1.0,0.06,0.01)

            st.write("**Mean & Cov Shrink**")
            do_shrink_means= st.checkbox("Shrink Means?", True)
            alpha_shrink= st.slider("Alpha (for means)",0.0,1.0,0.3,0.05)
            do_shrink_cov= st.checkbox("Shrink Cov (diagonal)?", True)
            beta_shrink= st.slider("Beta (for cov)",0.0,1.0,0.2,0.05)

            n_points_man= st.number_input("Frontier #points (Param Only)", 5,100,15, step=5)
            run_button= st.button("Run Rolling (Manual)")

        if run_button:
            def param_sharpe_fn(sub_ret: pd.DataFrame):
                from modules.optimization.cvxpy_optimizer import parametric_max_sharpe_aclass_subtype
                w_opt, summary= parametric_max_sharpe_aclass_subtype(
                    df_returns= sub_ret,
                    tickers= col_tickers,
                    asset_classes= asset_cls_list,
                    security_types= sec_type_list,
                    class_constraints= class_sum_constraints,
                    subtype_constraints= subtype_constraints,
                    daily_rf= daily_rf,
                    no_short= True,
                    n_points= n_points_man,
                    regularize_cov= reg_cov,
                    shrink_means= do_shrink_means,
                    alpha= alpha_shrink,
                    shrink_cov= do_shrink_cov,
                    beta= beta_shrink,
                    use_ledoitwolf= do_ledoitwolf,
                    do_ewm= do_ewm,
                    ewm_alpha= ewm_alpha
                )
                return w_opt, summary

            def direct_sharpe_fn(sub_ret: pd.DataFrame):
                from modules.optimization.cvxpy_optimizer import direct_max_sharpe_aclass_subtype
                w_opt, summary= direct_max_sharpe_aclass_subtype(
                    df_returns= sub_ret,
                    tickers= col_tickers,
                    asset_classes= asset_cls_list,
                    security_types= sec_type_list,
                    class_constraints= class_sum_constraints,
                    subtype_constraints= subtype_constraints,
                    daily_rf= daily_rf,
                    no_short= True,
                    regularize_cov= reg_cov,
                    shrink_means= do_shrink_means,
                    alpha= alpha_shrink,
                    shrink_cov= do_shrink_cov,
                    beta= beta_shrink,
                    use_ledoitwolf= do_ledoitwolf,
                    do_ewm= do_ewm,
                    ewm_alpha= ewm_alpha
                )
                return w_opt, summary

            if approach=="Manual Single Rolling (Parametric)":
                sr_line, final_w, old_w_last, final_rebal_date, df_rebal, ext_metrics_new= \
                    rolling_backtest_monthly_param_sharpe(
                        df_prices= df_sub,
                        df_instruments= df_instruments,
                        param_sharpe_fn= param_sharpe_fn,
                        start_date= df_sub.index[0],
                        end_date= df_sub.index[-1],
                        months_interval= rebal_freq,
                        window_days= window_days,
                        transaction_cost_value= transaction_cost_value,
                        transaction_cost_type= cost_type,
                        trade_buffer_pct= trade_buffer_pct,
                        daily_rf= daily_rf
                    )
            else:
                sr_line, final_w, old_w_last, final_rebal_date, df_rebal, ext_metrics_new= \
                    rolling_backtest_monthly_direct_sharpe(
                        df_prices= df_sub,
                        df_instruments= df_instruments,
                        direct_sharpe_fn= direct_sharpe_fn,
                        start_date= df_sub.index[0],
                        end_date= df_sub.index[-1],
                        months_interval= rebal_freq,
                        window_days= window_days,
                        transaction_cost_value= transaction_cost_value,
                        transaction_cost_type= cost_type,
                        trade_buffer_pct= trade_buffer_pct,
                        daily_rf= daily_rf
                    )

            # Build old line
            old_line= build_old_portfolio_line(df_instruments, df_sub)
            idx_all= old_line.index.union(sr_line.index)
            old_line_u= old_line.reindex(idx_all, method="ffill")
            new_line_u= sr_line.reindex(idx_all, method="ffill")
            old0= old_line_u.iloc[0]
            new0= new_line_u.iloc[0]

            # Plot
            df_cum= pd.DataFrame({
                "Old(%)": (old_line_u/old0 -1)*100,
                "New(%)": (new_line_u/new0 -1)*100
            }, index= idx_all)
            st.line_chart(df_cum)

            # Extended metrics
            ext_metrics_old= compute_extended_metrics(old_line_u, daily_rf=daily_rf)

            performance_keys= ["Total Return","Annual Return","Annual Vol","Sharpe"]
            risk_keys= ["MaxDD","TimeToRecovery","VaR_1M99","CVaR_1M99"]
            ratio_keys= ["Skew","Kurtosis","Sortino","Calmar","Omega"]

            def build_metric_df(metric_keys, old_m, new_m):
                rows=[]
                for mk in metric_keys:
                    val_old= old_m.get(mk,0.0)
                    val_new= new_m.get(mk,0.0)
                    rows.append((mk, val_old, val_new))
                df_out= pd.DataFrame(rows, columns=["Metric","Old","New"])
                df_out.set_index("Metric", inplace=True)
                return df_out

            df_perf_table= build_metric_df(performance_keys, ext_metrics_old, ext_metrics_new)
            df_risk_table= build_metric_df(risk_keys,        ext_metrics_old, ext_metrics_new)
            df_ratio_table= build_metric_df(ratio_keys,      ext_metrics_old, ext_metrics_new)

            def format_ext(df_):
                def format_val(mk,val):
                    pct_metrics= ["Total Return","Annual Return","Annual Vol","MaxDD","VaR_1M99","CVaR_1M99"]
                    if mk in pct_metrics:
                        return f"{val*100:.2f}%"
                    elif mk=="TimeToRecovery":
                        return f"{val:.0f}"
                    else:
                        return f"{val:.3f}"
                dfx= df_.copy()
                for mk in dfx.index:
                    for colx in ["Old","New"]:
                        rawv= dfx.loc[mk,colx]
                        dfx.loc[mk,colx]= format_val(mk,rawv)
                return dfx

            st.write("### Extended Metrics - Performance")
            st.dataframe(format_ext(df_perf_table))
            st.write("### Extended Metrics - Risk")
            st.dataframe(format_ext(df_risk_table))
            st.write("### Extended Metrics - Ratios")
            st.dataframe(format_ext(df_ratio_table))

            from modules.analytics.weight_display import (
                display_instrument_weight_diff,
                display_class_weight_diff
            )
            display_instrument_weight_diff(df_instruments, col_tickers, final_w)
            display_class_weight_diff(df_instruments, col_tickers, asset_cls_list, final_w)

            final_val= sr_line.iloc[-1]
            cost_stats= compute_cost_impact(df_rebal, final_val)
            st.write("### Transaction Cost Impact")
            df_cost_stats= pd.DataFrame([cost_stats])
            st.dataframe(df_cost_stats.style.format({
                "Total Cost":"{:.4f}",
                "Cost as % of Final Value":"{:.2%}",
                "Avg Cost per Rebalance":"{:.4f}"
            }))

            # intervals
            rebal_dates= df_rebal["Date"].unique().tolist()
            rebal_dates.sort()
            first_day= sr_line.index[0]
            if len(rebal_dates)==0 or rebal_dates[0]> first_day:
                rebal_dates= [first_day]+ rebal_dates
            last_day= sr_line.index[-1]
            if rebal_dates[-1]< last_day:
                rebal_dates.append(last_day)

            st.write("### Interval Performance")
            from modules.backtesting.rolling_intervals import display_interval_bars_and_stats
            display_interval_bars_and_stats(
                sr_line_old= old_line_u,
                sr_line_new= sr_line,
                rebal_dates= rebal_dates,
                label_old="Old",
                label_new="New",
                display_mode="grouped"
            )

            st.write("### Drawdown Over Time")
            from modules.backtesting.max_drawdown import plot_drawdown_series, show_max_drawdown_comparison
            df_compare= pd.DataFrame({
                "Old_Ptf": old_line_u* old0,
                "New_Ptf": new_line_u* new0
            }, index= old_line_u.index).dropna()
            fig_dd= plot_drawdown_series(df_compare)
            st.plotly_chart(fig_dd)
            df_dd_comp= show_max_drawdown_comparison(df_compare)
            st.write("### Max Drawdown Comparison")
            st.dataframe(df_dd_comp.style.format("{:.2%}"))

            st.write("### 12-Month Final Frontier")
            from modules.optimization.efficient_frontier import (
                compute_efficient_frontier_12m,
                interpolate_frontier_for_vol,
                plot_frontier_comparison
            )
            fvol, fret = compute_efficient_frontier_12m(
                df_prices= df_sub,
                df_instruments= df_instruments,
                n_points= 50,
                clamp_min_return= 0.0,
                remove_dominated= True,
                do_shrink_means= do_shrink_means,
                alpha= alpha_shrink,
                do_shrink_cov= do_shrink_cov,
                beta= beta_shrink,
                use_ledoitwolf= do_ledoitwolf,
                do_ewm= do_ewm,
                ewm_alpha= ewm_alpha,
                regularize_cov= reg_cov,
                class_constraints= class_sum_constraints,
                col_tickers= col_tickers,
                final_w= final_w
            )
            if len(fvol)==0:
                st.warning("No feasible 12-month frontier.")
            else:
                if len(df_sub)>252:
                    df_12m= df_sub.iloc[-252:].copy()
                else:
                    df_12m= df_sub.copy()
                ret_12m= df_12m.pct_change().fillna(0.0)
                mean_12m= ret_12m.mean().values
                cov_12m= ret_12m.cov().values

                def daily_vol_ret(w):
                    vol_= np.sqrt(w@ cov_12m@ w)
                    re_= mean_12m@ w
                    return vol_, re_

                old_map={}
                for i, tk in enumerate(col_tickers):
                    row_= df_instruments[df_instruments["#ID"]==tk]
                    if not row_.empty:
                        old_map[i]= row_["Weight_Old"].iloc[0]
                    else:
                        old_map[i]=0.0
                sum_old= sum(list(old_map.values()))
                if sum_old<=0:
                    sum_old=1.0
                w_old= np.array([old_map[i] for i in range(len(col_tickers))])/ sum_old

                old_vol, old_ret= daily_vol_ret(w_old)
                new_vol, new_ret= daily_vol_ret(final_w)

                same_v, same_r= interpolate_frontier_for_vol(fvol, fret, old_vol)
                figf= plot_frontier_comparison(
                    fvol, fret,
                    old_vol, old_ret,
                    new_vol, new_ret,
                    same_v, same_r,
                    title= "12-Month Frontier: Old vs New"
                )
                st.plotly_chart(figf)


    ########################################################################
    # B) Grid Search
    ########################################################################
    elif approach=="Grid Search":
        st.subheader("Grid Search => param or direct. pass use_direct_solver if needed.")
        frontier_points_list= st.multiselect("Frontier Points (n_points)", [5,10,15,20,30],[5,10,15])
        alpha_list= st.multiselect("Alpha values (for mean)", [0,0.1,0.2,0.3,0.4,0.5],[0,0.1,0.3])
        beta_list= st.multiselect("Beta values (for cov)", [0,0.1,0.2,0.3,0.4,0.5],[0.1,0.2])
        rebal_freq_list= st.multiselect("Rebalance freq (months)", [1,3,6],[1,3])
        lookback_list= st.multiselect("Lookback (months)", [3,6,12],[3,6])
        max_workers= st.number_input("Max Workers",1,64,4,step=1)
        use_direct_gs= st.checkbox("Use Direct Solver in Grid Search?", value=False)

        if st.button("Run Grid Search"):
            if (not frontier_points_list or not alpha_list or not beta_list
                or not rebal_freq_list or not lookback_list):
                st.error("Select at least one param in each list.")
            else:
                df_gs= rolling_grid_search(
                    df_prices= df_sub,
                    df_instruments= df_instruments,
                    asset_cls_list= asset_cls_list,
                    sec_type_list= sec_type_list,
                    class_sum_constraints= class_sum_constraints,
                    subtype_constraints= subtype_constraints,
                    daily_rf= daily_rf,
                    frontier_points_list= frontier_points_list,
                    alpha_list= alpha_list,
                    beta_list= beta_list,
                    rebal_freq_list= rebal_freq_list,
                    lookback_list= lookback_list,
                    transaction_cost_value= transaction_cost_value,
                    transaction_cost_type= cost_type,
                    trade_buffer_pct= trade_buffer_pct,
                    use_michaud= False,
                    n_boot= 10,
                    do_shrink_means= True,
                    do_shrink_cov= True,
                    reg_cov= False,
                    do_ledoitwolf= False,
                    do_ewm= False,
                    ewm_alpha= 0.06,
                    max_workers= max_workers,
                    use_direct_solver= use_direct_gs
                )
                st.dataframe(df_gs)
                if "Sharpe Ratio" in df_gs.columns:
                    best_ = df_gs.sort_values("Sharpe Ratio", ascending=False).head(5)
                    st.write("**Top 5 combos by Sharpe**")
                    st.dataframe(best_)

    ########################################################################
    # C) Bayesian
    ########################################################################
    else:
        st.subheader("Bayesian => param or direct => ignoring n_points for direct approach.")
        df_bayes= rolling_bayesian_optimization(
            df_prices= df_sub,
            df_instruments= df_instruments,
            asset_cls_list= asset_cls_list,
            sec_type_list= sec_type_list,
            class_sum_constraints= class_sum_constraints,
            subtype_constraints= subtype_constraints,
            daily_rf= daily_rf,
            transaction_cost_value= transaction_cost_value,
            transaction_cost_type= cost_type,
            trade_buffer_pct= trade_buffer_pct
        )
        if not df_bayes.empty:
            st.write("Bayesian Search Results:")
            st.dataframe(df_bayes)


if __name__=="__main__":
    main()
