# File: app.py

import streamlit as st
import pandas as pd
import numpy as np
import time
from dateutil.relativedelta import relativedelta

# scikit-optimize (if used for hyperparam)
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

# --- Import our custom excel_loader ---
from modules.data_loading.excel_loader import parse_excel

# Constraints => keep_current or custom
from modules.analytics.constraints import get_main_constraints

# Rolling monthly logic => 4 approaches + strategic
from modules.backtesting.rolling_monthly import (
    rolling_backtest_monthly_param_sharpe,
    rolling_backtest_monthly_direct_sharpe,
    rolling_backtest_monthly_param_cvar,
    rolling_backtest_monthly_direct_cvar,
    rolling_backtest_monthly_strategic,
    compute_cost_impact,
    rolling_grid_search
)

# Interval / drawdown
from modules.backtesting.rolling_intervals import display_interval_bars_and_stats
from modules.backtesting.max_drawdown import plot_drawdown_series, show_max_drawdown_table

# Bayesian
from modules.backtesting.rolling_bayesian import rolling_bayesian_optimization

# Markowitz solvers
from modules.optimization.cvxpy_parametric import parametric_max_sharpe_aclass_subtype
from modules.optimization.cvxpy_direct import direct_max_sharpe_aclass_subtype

# CVaR solvers
from modules.optimization.cvxpy_parametric_cvar import dynamic_min_cvar_with_fallback
from modules.optimization.cvxpy_direct_cvar import direct_max_return_cvar_constraint_aclass_subtype

# Optional efficient frontier for Markowitz only
from modules.optimization.efficient_frontier import (
    compute_efficient_frontier_12m,
    interpolate_frontier_for_vol,
    plot_frontier_comparison
)

# Weight / extended metrics
from modules.analytics.weight_display import (
    display_three_portfolio_class_weights,
    display_new_vs_old_instrument_tables
)
from modules.analytics.extended_metrics import compute_extended_metrics

# For "Extended Metrics" display
from modules.analytics.display_utils import display_extended_metrics

# risk contribution
from modules.analytics.risk_contributions import display_risk_contributions

# Excel export
from modules.export.export_backtest_to_excel import export_backtest_results_to_excel


###############################################################################
# data & constraints => from constraints.py
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
# Coverage cleaning & old portfolio line
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

def build_df_drift_weights(df_instruments: pd.DataFrame, df_sub: pd.DataFrame) -> np.ndarray:
    """
    Compute final drift weights (buy & hold) from #Quantity * last_price.
    """
    qties = df_instruments["#Quantity"].fillna(0.0).values
    last_prices = df_sub.iloc[-1].fillna(0.0).values
    val = qties * last_prices
    total_val = val.sum()
    if total_val <= 0:
        total_val = 1.0
    return val / total_val

def build_old_portfolio_line(df_instruments: pd.DataFrame, df_prices: pd.DataFrame) -> pd.Series:
    """
    If #Quantity present => build old drift line => no rebal.
    """
    df_prices = df_prices.sort_index()
    ticker_qty = {}
    for _, row in df_instruments.iterrows():
        tkr = row["#ID"]
        qty = row["#Quantity"]
        ticker_qty[tkr] = qty
    col_list = df_prices.columns
    old_shares = np.array([ticker_qty.get(c, 0.0) for c in col_list])
    vals = [np.sum(old_shares * row.values) for _, row in df_prices.iterrows()]
    sr = pd.Series(vals, index=df_prices.index)
    if len(sr) > 0 and sr.iloc[0] <= 0:
        sr.iloc[0] = 1.0
    if len(sr) > 0:
        sr = sr / sr.iloc[0]
    sr.name = "Old_Drift"
    return sr


###############################################################################
# Main app
###############################################################################
def main():
    st.title("Optimize Your Portfolio (Markowitz & CVaR)")

    # 1) Data + constraints
    df_instruments, df_prices, coverage, main_constr = sidebar_data_and_constraints()
    if df_instruments.empty or df_prices.empty or not main_constr:
        st.stop()

    # 2) Clean data
    df_prices_clean = clean_df_prices(df_prices, coverage)

    # 3) Build old portfolio weighting if columns exist
    if "#Quantity" in df_instruments.columns and "#Last_Price" in df_instruments.columns:
        df_instruments["Value"] = df_instruments["#Quantity"] * df_instruments["#Last_Price"]
        tot_val = df_instruments["Value"].sum()
        if tot_val <= 0:
            tot_val = 1.0
        df_instruments["Weight_Old"] = df_instruments["Value"] / tot_val
    else:
        df_instruments["Weight_Old"] = 0.0

    user_start = main_constr["user_start"]
    df_sub = df_prices_clean.loc[pd.Timestamp(user_start):]
    if len(df_sub) < 2:
        st.error("Not enough data after your chosen start date.")
        st.stop()

    col_tickers = df_sub.columns.tolist()
    have_sec    = ("#Security_Type" in df_instruments.columns)
    asset_cls_list = []
    sec_type_list  = []
    for tk in col_tickers:
        row_ = df_instruments[df_instruments["#ID"] == tk]
        if not row_.empty:
            asset_cls_list.append(row_["#Asset_Class"].iloc[0])
            if have_sec:
                stp = row_["#Security_Type"].iloc[0]
                if pd.isna(stp):
                    stp = "Unknown"
                sec_type_list.append(stp)
            else:
                sec_type_list.append("Unknown")
        else:
            asset_cls_list.append("Unknown")
            sec_type_list.append("Unknown")

    # constraints
    class_sum_constraints = main_constr["class_sum_constraints"]
    subtype_constraints   = main_constr["subtype_constraints"]
    daily_rf              = main_constr["daily_rf"]
    cost_type             = main_constr["cost_type"]
    transaction_cost_value= main_constr["transaction_cost_value"]
    trade_buffer_pct      = main_constr["trade_buffer_pct"]

    # Build old drift line
    old_drift_line = build_old_portfolio_line(df_instruments, df_sub)
    drift_final_w  = build_df_drift_weights(df_instruments, df_sub)
    if "results" not in st.session_state:
        st.session_state["results"] = {}
    st.session_state["results"]["w_drift_final"] = drift_final_w

    # Build old strategic line => rolling_backtest_monthly_strategic
    sr_strat   = pd.Series(dtype=float)
    strat_extm = {}
    w_s_final  = np.zeros(len(col_tickers))
    if df_instruments["Weight_Old"].sum() > 1e-12:
        sr_s, w_s_final_, w_s_old, dt_s, df_sreb, extm_s = rolling_backtest_monthly_strategic(
            df_prices         = df_sub,
            df_instruments    = df_instruments,
            start_date        = df_sub.index[0],
            end_date          = df_sub.index[-1],
            months_interval   = 12,
            transaction_cost_value = transaction_cost_value,
            transaction_cost_type  = cost_type,
            daily_rf               = daily_rf
        )
        sr_strat   = sr_s
        strat_extm = extm_s
        w_s_final  = w_s_final_
    st.session_state["results"]["w_strat_final"] = w_s_final

    # -------------------------------------------------------------------
    # Radio => "Portfolio Optimization" or "Hyperparameter Optimization"
    # -------------------------------------------------------------------
    top_choices_list = ["Portfolio Optimization","Hyperparameter Optimization"]
    if "top_choice" not in st.session_state:
        st.session_state["top_choice"] = top_choices_list[0]
    old_top = st.session_state["top_choice"]
    top_choice = st.radio("Analysis Type:", top_choices_list, index=top_choices_list.index(old_top))

    if top_choice != old_top:
        st.session_state["top_choice"] = top_choice
        if "results" in st.session_state:
            del st.session_state["results"]

    # =========================================================
    # ================ PORTFOLIO OPTIMIZATION =================
    # =========================================================
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
            window_days= lookback_m*21

            if solver_choice in ["Parametric (Markowitz)", "Direct (Markowitz)"]:
                st.subheader("Markowitz Cov + Mean Settings")
                # param => n_points
                n_points_man= 0
                if solver_choice=="Parametric (Markowitz)":
                    n_points_man = st.number_input("Frontier #points", 5,100,15,5)

                st.write("**Cov Estimator**")
                cov_estimator = st.selectbox("Cov Est", ["sample","ewma","dcc_garch","mcd"], 0)
                ewm_alpha = 0.06
                garch_p   = 1
                garch_q   = 1
                garch_dist= "normal"
                dcc_alpha = 0.05
                dcc_beta  = 0.90
                if cov_estimator=="ewma":
                    ewm_alpha = st.slider("EWMA alpha", 0.0,1.0,0.06,0.01)
                elif cov_estimator=="dcc_garch":
                    garch_p   = st.number_input("garch p",1,5,1)
                    garch_q   = st.number_input("garch q",1,5,1)
                    garch_dist= st.selectbox("garch dist",["normal","t"],0)
                    dcc_alpha = st.slider("DCC alpha",0.0,1.0,0.05,0.01)
                    dcc_beta  = st.slider("DCC beta", 0.0,1.0,0.90,0.01)

                st.write("**Cov Shrink**")
                shrinkage = st.selectbox("Shrink method", ["none","ledoitwolf","diagonal"], 0)
                diag_shrink_beta= 0.2
                if shrinkage=="diagonal":
                    diag_shrink_beta= st.slider("Diag Beta",0.0,1.0,0.2,0.01)
                regularize_cov= st.checkbox("Regularize Cov?", False)

                st.write("**Mean improvement**")
                mean_tech = st.selectbox(
                    "Mean tech", 
                    ["none", "shrink_to_grand_mean", "shrink_to_zero"], 
                    index=0
                )
                alpha_mean_shrink = 0.0
                if mean_tech in ("shrink_to_grand_mean", "shrink_to_zero"):
                    alpha_mean_shrink = st.slider("Alpha mean shrink", 0.0, 1.0, 0.3, 0.01)


                # cvar => None
                cvar_alpha_in  = None
                cvar_limit_in  = None
                cvar_freq_user = None

            else:
                # CVaR only => minimal UI
                st.subheader("CVaR Options")
                cvar_alpha_in  = st.slider("CVaR alpha",0.0,1.0,0.95,0.01)
                cvar_limit_in  = st.slider("CVaR limit (Direct only)",0.0,1.0,0.10,0.01)
                cvar_freq_user = st.selectbox("CVaR Frequency", ["daily","weekly","monthly","annual"],0)

                # Markowitz => None
                n_points_man= None
                cov_estimator= None
                ewm_alpha= None
                garch_p= None
                garch_q= None
                garch_dist= None
                dcc_alpha= None
                dcc_beta= None
                shrinkage= None
                diag_shrink_beta= None
                regularize_cov= None
                shrink_means= None
                alpha_mean_shrink= None

            run_rolling = st.button("Run Rolling")

        if run_rolling:
            # 1) param Markowitz
            def param_sharpe_fn(sub_ret):
                w_, summ_ = parametric_max_sharpe_aclass_subtype(
                    df_returns=sub_ret,
                    tickers=col_tickers,
                    asset_classes=asset_cls_list,
                    security_types=sec_type_list,
                    class_constraints=class_sum_constraints,
                    subtype_constraints=subtype_constraints,
                    daily_rf=daily_rf,
                    no_short=True,
                    n_points=n_points_man if n_points_man else 15,

                    cov_estimator=cov_estimator,
                    shrinkage=shrinkage,
                    ewm_alpha=ewm_alpha if ewm_alpha else 0.06,
                    diag_shrink_beta=diag_shrink_beta if diag_shrink_beta else 0.0,
                    regularize_cov=regularize_cov if regularize_cov else False,
                    garch_p=garch_p if garch_p else 1,
                    garch_q=garch_q if garch_q else 1,
                    garch_dist=garch_dist if garch_dist else "normal",
                    dcc_alpha=dcc_alpha if dcc_alpha else 0.05,
                    dcc_beta=dcc_beta if dcc_beta else 0.90,
                    mean_tech=mean_tech,
                    alpha_mean_shrink=alpha_mean_shrink if alpha_mean_shrink else 0.0
                )
                return w_, summ_

            # 2) direct Markowitz
            def direct_sharpe_fn(sub_ret):
                w_, summ_ = direct_max_sharpe_aclass_subtype(
                    df_returns=sub_ret,
                    tickers=col_tickers,
                    asset_classes=asset_cls_list,
                    security_types=sec_type_list,
                    class_constraints=class_sum_constraints,
                    subtype_constraints=subtype_constraints,
                    daily_rf=daily_rf,
                    no_short=True,

                    cov_estimator=cov_estimator,
                    shrinkage=shrinkage,
                    ewm_alpha=ewm_alpha if ewm_alpha else 0.06,
                    diag_shrink_beta=diag_shrink_beta if diag_shrink_beta else 0.0,
                    regularize_cov=regularize_cov if regularize_cov else False,
                    garch_p=garch_p if garch_p else 1,
                    garch_q=garch_q if garch_q else 1,
                    garch_dist=garch_dist if garch_dist else "normal",
                    dcc_alpha=dcc_alpha if dcc_alpha else 0.05,
                    dcc_beta=dcc_beta if dcc_beta else 0.90,
                    mean_tech=mean_tech,
                    alpha_mean_shrink=alpha_mean_shrink if alpha_mean_shrink else 0.0
                )
                return w_, summ_

            # 3) param cvar
            def param_cvar_fn(sub_ret, old_w):
                w_, summ_ = dynamic_min_cvar_with_fallback(
                    df_returns=sub_ret,
                    tickers=col_tickers,
                    asset_classes=asset_cls_list,
                    security_types=sec_type_list,
                    class_constraints=class_sum_constraints,
                    subtype_constraints=subtype_constraints,
                    old_w=old_w,
                    cvar_alpha=cvar_alpha_in if cvar_alpha_in else 0.95,
                    no_short=True,
                    daily_rf=daily_rf,
                    freq_choice=cvar_freq_user if cvar_freq_user else "daily",
                    clamp_factor=1.5,
                    max_weight_each=1.0,
                    max_iter=10
                )
                return w_, summ_

            # 4) direct cvar
            def direct_cvar_fn(sub_ret):
                w_, summ_ = direct_max_return_cvar_constraint_aclass_subtype(
                    df_returns=sub_ret,
                    tickers=col_tickers,
                    asset_classes=asset_cls_list,
                    security_types=sec_type_list,
                    class_constraints=class_sum_constraints,
                    subtype_constraints=subtype_constraints,
                    cvar_alpha=cvar_alpha_in if cvar_alpha_in else 0.95,
                    cvar_limit=cvar_limit_in if cvar_limit_in else 0.10,
                    daily_rf=daily_rf,
                    no_short=True,
                    freq_choice=cvar_freq_user if cvar_freq_user else "daily"
                )
                return w_, summ_

            # run the selected approach
            if solver_choice=="Parametric (Markowitz)":
                sr_new, w_final, w_old_fin, rebal_date, df_rebal, extm_new = rolling_backtest_monthly_param_sharpe(
                    df_prices=df_sub,
                    df_instruments=df_instruments,
                    param_sharpe_fn=param_sharpe_fn,
                    start_date=df_sub.index[0],
                    end_date=df_sub.index[-1],
                    months_interval=rebal_freq,
                    window_days=window_days,
                    transaction_cost_value=transaction_cost_value,
                    transaction_cost_type=cost_type,
                    trade_buffer_pct=trade_buffer_pct,
                    daily_rf=daily_rf
                )
            elif solver_choice=="Direct (Markowitz)":
                sr_new, w_final, w_old_fin, rebal_date, df_rebal, extm_new = rolling_backtest_monthly_direct_sharpe(
                    df_prices=df_sub,
                    df_instruments=df_instruments,
                    direct_sharpe_fn=direct_sharpe_fn,
                    start_date=df_sub.index[0],
                    end_date=df_sub.index[-1],
                    months_interval=rebal_freq,
                    window_days=window_days,
                    transaction_cost_value=transaction_cost_value,
                    transaction_cost_type=cost_type,
                    trade_buffer_pct=trade_buffer_pct,
                    daily_rf=daily_rf
                )
            elif solver_choice=="Parametric (CVaR)":
                sr_new, w_final, w_old_fin, rebal_date, df_rebal, extm_new = rolling_backtest_monthly_param_cvar(
                    df_prices=df_sub,
                    df_instruments=df_instruments,
                    param_cvar_fn=param_cvar_fn,
                    start_date=df_sub.index[0],
                    end_date=df_sub.index[-1],
                    months_interval=rebal_freq,
                    window_days=window_days,
                    transaction_cost_value=transaction_cost_value,
                    transaction_cost_type=cost_type,
                    trade_buffer_pct=trade_buffer_pct,
                    daily_rf=daily_rf
                )
            else:  # Direct (CVaR)
                sr_new, w_final, w_old_fin, rebal_date, df_rebal, extm_new = rolling_backtest_monthly_direct_cvar(
                    df_prices=df_sub,
                    df_instruments=df_instruments,
                    direct_cvar_fn=direct_cvar_fn,
                    start_date=df_sub.index[0],
                    end_date=df_sub.index[-1],
                    months_interval=rebal_freq,
                    window_days=window_days,
                    transaction_cost_value=transaction_cost_value,
                    transaction_cost_type=cost_type,
                    trade_buffer_pct=trade_buffer_pct,
                    daily_rf=daily_rf
                )

            # Extended metrics => old drift
            sr_drift_local = old_drift_line.reindex(sr_new.index, method="ffill")
            extm_drift = compute_extended_metrics(sr_drift_local, daily_rf=daily_rf)

            # store
            st.session_state["results"]["w_new_final"] = w_final
            st.session_state["results"].update({
                "sr_new": sr_new,
                "extm_new": extm_new,
                "df_rebal": df_rebal,
                "sr_drift": sr_drift_local,
                "extm_drift": extm_drift,
                "sr_strat": sr_strat.reindex(sr_new.index, method="ffill"),
                "extm_strat": strat_extm
            })

    # =========================================================
    # ============== HYPERPARAMETER OPTIMIZATION =============
    # =========================================================
    elif top_choice == "Hyperparameter Optimization":
        hyper_choice = st.radio("Hyperparameter Method:", ["Grid Search","Bayesian"], index=0)

        if hyper_choice == "Grid Search":
            st.subheader("Grid => param or direct Markowitz => pass use_direct if needed.")
            # old logic => just for Markowitz
            frontier_points_list= st.multiselect("Frontier Points (n_points)", [5,10,15,20,30],[5,10,15])
            alpha_list= st.multiselect("Alpha (means)", [0,0.1,0.2,0.3,0.4,0.5],[0,0.1,0.3])
            beta_list= st.multiselect("Beta (cov)", [0,0.1,0.2,0.3,0.4,0.5],[0.1,0.2])
            rebal_freq_list= st.multiselect("Rebalance freq (months)", [1,3,6],[1,3])
            lookback_list= st.multiselect("Lookback (months)", [3,6,12],[3,6])
            max_workers= st.number_input("Max Workers",1,64,4,step=1)
            use_direct_gs= st.checkbox("Use Direct Solver in Grid Search?", value=False)

            if st.button("Run Grid Search"):
                if not frontier_points_list or not alpha_list or not beta_list \
                   or not rebal_freq_list or not lookback_list:
                    st.error("Select at least one param in each list.")
                else:
                    # call rolling_grid_search
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
                        use_michaud=False,
                        n_boot=10,
                        do_shrink_means=True,
                        do_shrink_cov=True,
                        reg_cov=False,
                        do_ledoitwolf=False,
                        do_ewm=False,
                        ewm_alpha=0.06,
                        max_workers= max_workers,
                        use_direct_solver= use_direct_gs
                    )
                    st.dataframe(df_gs)
                    if "Sharpe Ratio" in df_gs.columns:
                        best_ = df_gs.sort_values("Sharpe Ratio", ascending=False).head(5)
                        st.write("**Top 5 combos by Sharpe**")
                        st.dataframe(best_)
        else:
            # Bayesian => param or direct Markowitz => ignoring n_points for direct
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

    # =========================================================
    # ============== If results => show analysis UI ===========
    # =========================================================
    if "results" in st.session_state and "sr_new" in st.session_state["results"]:
        r_ = st.session_state["results"]
        sr_new   = r_["sr_new"]
        sr_drift = r_["sr_drift"]
        sr_strat = r_["sr_strat"]

        st.write("### Show/Hide Portfolios")
        c_new   = st.checkbox("New Optimized", value=True)
        c_drift = st.checkbox("Old Drift",     value=True)
        c_strat = st.checkbox("Old Strategic", value=True)

        import plotly.graph_objects as go
        fig = go.Figure()
        if c_new and not sr_new.empty:
            fig.add_trace(go.Scatter(
                x= sr_new.index,
                y= (sr_new/sr_new.iloc[0]-1)*100,
                name="New Optimized",
                line=dict(color="#1f77b4", width=2)
            ))
        if c_drift and not sr_drift.empty:
            fig.add_trace(go.Scatter(
                x= sr_drift.index,
                y= (sr_drift/sr_drift.iloc[0]-1)*100,
                name="Old Drift",
                line=dict(color="grey", width=2)
            ))
        if c_strat and not sr_strat.empty:
            fig.add_trace(go.Scatter(
                x= sr_strat.index,
                y= (sr_strat/sr_strat.iloc[0]-1)*100,
                name="Old Strategic",
                line=dict(color="lightblue", width=2)
            ))
        fig.update_layout(
            title="Portfolio Comparison (Cumulative Returns)",
            xaxis_title="Date",
            yaxis_title="Return (%)"
        )
        st.plotly_chart(fig)

        st.write("### Extended Metrics (Side-by-Side)")
        metrics_map = {}
        if c_new and not sr_new.empty:
            metrics_map["New Optimized"] = r_["extm_new"]
        if c_drift and not sr_drift.empty:
            metrics_map["Old Drift"]     = r_["extm_drift"]
        if c_strat and not sr_strat.empty:
            metrics_map["Old Strategic"]= r_["extm_strat"]

        if metrics_map:
            display_extended_metrics(metrics_map)
        else:
            st.info("No portfolios selected => no extended metrics.")

        df_rebal = r_["df_rebal"]
        if not df_rebal.empty:
            final_val = sr_new.iloc[-1] if not sr_new.empty else 0.0
            cost_stats= compute_cost_impact(df_rebal, final_val)
            st.write("### Transaction Cost Impact (New Optimized)")
            st.dataframe(pd.DataFrame([cost_stats]))

        # Interval Perf
        st.write("## Interval Performance (Multi-Portfolios)")
        if not df_rebal.empty and "Date" in df_rebal.columns:
            rebal_dates = sorted(df_rebal["Date"].unique())
            if sr_new.index[0] not in rebal_dates:
                rebal_dates = [sr_new.index[0]] + rebal_dates
            if rebal_dates[-1] < sr_new.index[-1]:
                rebal_dates.append(sr_new.index[-1])
        else:
            rebal_dates = [sr_new.index[0], sr_new.index[-1]]

        display_interval_bars_and_stats(
            sr_new= sr_new,
            sr_drift= sr_drift,
            sr_strat= sr_strat,
            c_new= c_new,
            c_drift= c_drift,
            c_strat= c_strat,
            rebal_dates= rebal_dates,
            label_new="New Optimized",
            label_drift="Old Drift",
            label_strat="Old Strategic",
            color_new="#1f77b4",
            color_drift="grey",
            color_strat="lightblue"
        )

        # Drawdown
        st.write("### Drawdown Over Time")
        df_draw = pd.DataFrame()
        if c_new and not sr_new.empty:
            df_draw["New Optimized"] = sr_new
        if c_drift and not sr_drift.empty:
            df_draw["Old Drift"]     = sr_drift
        if c_strat and not sr_strat.empty:
            df_draw["Old Strategic"]= sr_strat
        if df_draw.empty:
            st.info("No portfolios selected for drawdown.")
        else:
            df_draw.dropna(how="all", inplace=True)
            fig_dd = plot_drawdown_series(df_draw, custom_title="Drawdown Over Time")
            st.plotly_chart(fig_dd)
            df_mdd = show_max_drawdown_table(df_draw)
            st.write("### Max Drawdown Comparison")
            st.dataframe(df_mdd.style.format("{:.2%}"))

        # Retrieve final weights from session
        w_new   = r_.get("w_new_final",   np.zeros(len(col_tickers)))
        w_drift = r_.get("w_drift_final", np.zeros(len(col_tickers)))
        w_strat = r_.get("w_strat_final", np.zeros(len(col_tickers)))

        # Show asset-class summary
        display_three_portfolio_class_weights(
            df_instruments, col_tickers, asset_cls_list,
            w_new=w_new,
            w_drift=w_drift,
            w_strat=w_strat
        )

        # Then show instrument-level side-by-side
        display_new_vs_old_instrument_tables(
            df_instruments=df_instruments,
            col_tickers=col_tickers,
            w_new=w_new,
            w_drift=w_drift,
            w_strat=w_strat
        )

        # Risk contributions
        df_daily_rets = df_sub.pct_change().dropna()
        alpha = 0.95
        display_risk_contributions(
            df_returns=df_daily_rets,
            col_tickers=col_tickers,
            weights=w_new,
            alpha=alpha,
            show_var=True,
            show_cvar=True
        )

if __name__=="__main__":
    main()
