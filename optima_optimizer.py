# File: optima_optimizer.py

import streamlit as st
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

# Data loading & constraints
from modules.data_loading.excel_loader import parse_excel
from modules.analytics.constraints import get_main_constraints

# Rolling monthly logic => 4 approaches
from modules.backtesting.rolling_monthly import (
    rolling_backtest_monthly_param_sharpe,
    rolling_backtest_monthly_direct_sharpe,
    rolling_backtest_monthly_param_cvar,
    rolling_backtest_monthly_direct_cvar,
    backtest_buy_and_hold_that_drifts,
    backtest_strategic_rebalanced,
    compute_cost_impact
)

# Interval logic for interval returns
from modules.backtesting.rolling_intervals import (
    compute_predefined_pairs_diff_stats,
    compute_multi_interval_returns,
    plot_multi_interval_bars
)

# Extended metrics
from modules.analytics.extended_metrics import compute_extended_metrics

# Excel export
from modules.export.export_backtest_to_excel import export_backtest_results_to_excel

# Drawdown logic
from modules.backtesting.max_drawdown import compute_drawdown_series
import plotly.express as px

# Hyperparameter UI module (the code you provided)
from modules.ui.hyperparam_ui import show_hyperparam_ui


###############################################################################
# Final weighting helpers
###############################################################################
def compute_final_aclass_weights(
    shares: np.ndarray,
    df_prices: pd.DataFrame,
    asset_cls_list: list[str]
) -> pd.Series:
    """
    For the final day in df_prices, multiply shares by prices, sum => total,
    and produce fraction by asset class.
    """
    if shares is None or len(shares) == 0 or df_prices.empty:
        return pd.Series(dtype=float)

    last_day = df_prices.index[-1]
    px_last = df_prices.loc[last_day].fillna(0.0).values
    vals = shares * px_last
    total_val = vals.sum()
    if total_val <= 1e-9:
        return pd.Series(dtype=float)

    df_map = pd.DataFrame({"Aclass": asset_cls_list, "Value": vals})
    group_ = df_map.groupby("Aclass")["Value"].sum()
    w_ = group_ / group_.sum()
    return w_


def compute_final_instrument_weights(
    shares: np.ndarray,
    df_prices: pd.DataFrame
) -> pd.Series:
    """
    Same idea, but at the ticker level instead of asset class.
    We return a Series of final weight by ticker name.
    """
    if shares is None or len(shares) == 0 or df_prices.empty:
        return pd.Series(dtype=float)

    last_day = df_prices.index[-1]
    px_last = df_prices.loc[last_day].fillna(0.0).values
    vals = shares * px_last
    total_val = vals.sum()
    if total_val <= 1e-9:
        return pd.Series(dtype=float)

    tickers = df_prices.columns
    w_ = vals / total_val
    return pd.Series(data=w_, index=tickers)


###############################################################################
# Display multi-portfolio metrics
###############################################################################
def display_multi_portfolio_metrics(metrics_map: dict):
    """
    Display multiple portfolios' extended metrics side-by-side in tables:
    Performance, Risk, Ratios.
    """
    performance_keys = ["Total Return", "Annual Return", "Annual Vol", "Sharpe"]
    risk_keys        = ["MaxDD", "TimeToRecovery", "VaR_1M99", "CVaR_1M99"]
    ratio_keys       = ["Skew", "Kurtosis", "Sortino", "Calmar", "Omega"]

    def build_dataframe_for_keys(metric_keys, metrics_map):
        pnames = list(metrics_map.keys())
        rows = []
        for mk in metric_keys:
            row = [mk]
            for pn in pnames:
                row.append(metrics_map[pn].get(mk, 0.0))
            rows.append(row)
        df_ = pd.DataFrame(rows, columns=["Metric"] + pnames)
        df_.set_index("Metric", inplace=True)
        return df_

    def format_ext_metrics_multi(df_):
        pct_metrics = ["Total Return", "Annual Return", "Annual Vol", "MaxDD", "VaR_1M99", "CVaR_1M99"]
        dfx = df_.copy()
        for mk in dfx.index:
            for c_ in dfx.columns:
                val = dfx.loc[mk, c_]
                if mk in pct_metrics:
                    dfx.loc[mk, c_] = f"{val * 100:.2f}%"
                elif mk == "TimeToRecovery":
                    dfx.loc[mk, c_] = f"{val:.0f}"
                else:
                    dfx.loc[mk, c_] = f"{val:.3f}"
        return dfx

    def display_category(title, keys):
        if not metrics_map:
            return
        df_cat = build_dataframe_for_keys(keys, metrics_map)
        st.write(f"### Extended Metrics - {title}")
        st.dataframe(format_ext_metrics_multi(df_cat))

    display_category("Performance", performance_keys)
    display_category("Risk",        risk_keys)
    display_category("Ratios",      ratio_keys)


###############################################################################
# Drawdown over time for multiple portfolios
###############################################################################
def plot_multi_drawdown(df_absolute: pd.DataFrame):
    if df_absolute.empty:
        return None

    dd_map = {}
    for c in df_absolute.columns:
        from modules.backtesting.max_drawdown import compute_drawdown_series
        dd_map[c + " (DD)"] = compute_drawdown_series(df_absolute[c])

    df_dd = pd.DataFrame(dd_map, index=df_absolute.index)
    fig = px.line(
        df_dd,
        x=df_dd.index,
        y=df_dd.columns,
        title="Historical Drawdown Over Time (Multiple)",
        labels={"value": "Drawdown", "index": "Date", "variable": "Portfolio"}
    )
    fig.update_yaxes(tickformat=".2%")
    return fig


###############################################################################
# Data loading & constraints
###############################################################################
from modules.data_loading.excel_loader import parse_excel
from modules.analytics.constraints import get_main_constraints

def sidebar_data_and_constraints():
    """
    Let user pick how to load the data: 
    - One-time Convert Excel->Parquet
    - Use Excel for Analysis
    - Use Parquet for Analysis
    Then set constraints from get_main_constraints.
    """
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

def clean_df_prices(df_prices: pd.DataFrame, coverage=0.8) -> pd.DataFrame:
    """
    Remove rows with insufficient coverage, then forward/backward fill, 
    and remove inf. 
    """
    df_prices = df_prices.copy()
    coverage_count = df_prices.notna().sum(axis=1)
    n_cols = df_prices.shape[1]
    threshold = n_cols * coverage

    df_prices = df_prices[coverage_count >= threshold]
    df_prices = df_prices.sort_index().fillna(method="ffill").fillna(method="bfill")
    df_prices.replace([np.inf, -np.inf], np.nan, inplace=True)
    df_prices.fillna(method="ffill", inplace=True)
    df_prices.fillna(method="bfill", inplace=True)
    return df_prices


###############################################################################
# main
###############################################################################
def main():
    st.title("Portfolio Optimization Example (+ GARCH Cov Option)")

    # 1) Data & constraints
    df_instruments, df_prices, coverage, main_constr = sidebar_data_and_constraints()
    if df_instruments.empty or df_prices.empty or not main_constr:
        st.stop()

    # 2) Clean data
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

    # If #Quantity => build an 'old' weighting
    if "#Quantity" in df_instruments.columns and "#Last_Price" in df_instruments.columns:
        df_instruments["Value"] = df_instruments["#Quantity"] * df_instruments["#Last_Price"]
        tot_val = df_instruments["Value"].sum()
        if tot_val <= 0:
            tot_val = 1.0
        df_instruments["Weight_Old"] = df_instruments["Value"] / tot_val
    else:
        df_instruments["Weight_Old"] = 0.0

    # Sort
    df_sub = df_sub.sort_index()

    # Build lists
    col_tickers = df_sub.columns.tolist()
    have_sec = ("#Security_Type" in df_instruments.columns)

    asset_cls_list = []
    sec_type_list  = []
    for tk in col_tickers:
        row_ = df_instruments[df_instruments["#ID"] == tk]
        if not row_.empty:
            asset_cls_list.append(row_["#Asset_Class"].iloc[0])
            if have_sec:
                stp_ = row_["#Security_Type"].iloc[0]
                if pd.isna(stp_):
                    sec_type_list.append("Unknown")
                else:
                    sec_type_list.append(stp_)
            else:
                sec_type_list.append("Unknown")
        else:
            asset_cls_list.append("Unknown")
            sec_type_list.append("Unknown")

    # Top choice => "Portfolio Optimization" or "Hyperparameter Optimization"
    top_choice = st.radio(
        "Analysis Type:",
        ["Portfolio Optimization", "Hyperparameter Optimization"],
        index=0
    )

    if top_choice == "Portfolio Optimization":
        solver_choice = st.selectbox(
            "Solver Approach:",
            [
                "Parametric (Markowitz)",
                "Direct (Markowitz)",
                "Parametric (CVaR)",
                "Direct (CVaR)"
            ],
            index=1  # Default => Direct (Markowitz)
        )

        with st.expander("Rolling Parameters", expanded=False):
            rebal_freq = st.selectbox("Rebalance Frequency (months)", [1, 3, 6], index=0)
            lookback_m = st.selectbox("Lookback Window (months)", [3, 6, 12], index=0)
            window_days = lookback_m * 21

            # Cov approach => sample, ewm, ledoit, or garch
            cov_approach = st.selectbox(
                "Covariance Approach",
                ["sample", "ewm", "ledoit", "garch"],
                index=0
            )

            reg_cov = False
            do_ledoitwolf = False
            do_ewm = False
            ewm_alpha = 0.0
            do_shrink_means = False
            alpha_shrink = 0.0
            do_shrink_cov = False
            beta_shrink = 0.0
            n_points_man = 0

            # placeholders for CVaR
            cvar_alpha_in = 0.95
            cvar_limit_in = 0.10
            cvar_freq_user= "daily"

            if solver_choice in ["Parametric (Markowitz)", "Direct (Markowitz)"]:
                reg_cov = st.checkbox("Regularize Cov?", False)
                do_ledoitwolf = st.checkbox("Use LedoitWolf Cov?", False)
                do_ewm = st.checkbox("Use EWM Cov?", False)
                ewm_alpha = st.slider("EWM alpha", 0.0, 1.0, 0.06, 0.01)

                st.write("**Mean & Cov Shrink**")
                do_shrink_means = st.checkbox("Shrink Means?", True)
                alpha_shrink = st.slider("Alpha (for means)", 0.0, 1.0, 0.3, 0.01)
                do_shrink_cov = st.checkbox("Shrink Cov (diagonal)?", True)
                beta_shrink = st.slider("Beta (for cov)", 0.0, 1.0, 0.2, 0.01)

                if solver_choice == "Parametric (Markowitz)":
                    n_points_man = st.number_input("Frontier #points (Param Only)", 5, 100, 15, step=5)
            else:
                # CVaR approach
                cvar_alpha_in = st.slider("CVaR alpha", 0.0, 0.9999, 0.95, 0.01)
                cvar_limit_in = st.slider("CVaR limit (Direct approach)", 0.0, 1.0, 0.10, 0.01)
                cvar_freq_user= st.selectbox("CVaR Frequency", ["daily", "weekly", "monthly", "annual"], index=0)

            run_rolling = st.button("Run Rolling")

        if run_rolling:
            # define solver callables
            def direct_sharpe_fn(sub_ret: pd.DataFrame):
                """
                Direct Markowitz (Sharpe) solver with user-specified 'cov_approach'.
                """
                from modules.optimization.cvxpy_direct import direct_max_sharpe_aclass_subtype
                w_opt, summary = direct_max_sharpe_aclass_subtype(
                    df_returns          = sub_ret,
                    tickers             = col_tickers,
                    asset_classes       = asset_cls_list,
                    security_types      = sec_type_list,
                    class_constraints   = class_sum_constraints,
                    subtype_constraints= subtype_constraints,
                    daily_rf           = daily_rf,
                    no_short           = True,
                    cov_approach       = cov_approach,
                    regularize_cov     = reg_cov,
                    shrink_means       = do_shrink_means,
                    alpha              = alpha_shrink,
                    shrink_cov         = do_shrink_cov,
                    beta               = beta_shrink,
                    use_ledoitwolf     = do_ledoitwolf,
                    do_ewm             = do_ewm,
                    ewm_alpha          = ewm_alpha
                )
                return w_opt, summary

            def param_sharpe_fn(sub_ret: pd.DataFrame):
                # Placeholder => param Markowitz 
                pass

            def param_cvar_fn(sub_ret: pd.DataFrame, old_w: np.ndarray):
                # Placeholder => param CVaR
                pass

            def direct_cvar_fn(sub_ret: pd.DataFrame):
                # Placeholder => direct CVaR
                pass

            # pick solver approach
            if solver_choice == "Direct (Markowitz)":
                sr_opt, final_w, final_sh_opt, old_w_last, final_rebal_date, df_rebal, extm_opt = (
                    rolling_backtest_monthly_direct_sharpe(
                        df_prices              = df_sub,
                        df_instruments         = df_instruments,
                        direct_sharpe_fn       = direct_sharpe_fn,
                        start_date             = df_sub.index[0],
                        end_date               = df_sub.index[-1],
                        months_interval        = rebal_freq,
                        window_days            = window_days,
                        transaction_cost_value = transaction_cost_val,
                        transaction_cost_type  = cost_type,
                        trade_buffer_pct       = trade_buffer_pct,
                        daily_rf               = daily_rf
                    )
                )
            elif solver_choice == "Parametric (Markowitz)":
                sr_opt, final_w, final_sh_opt, old_w_last, final_rebal_date, df_rebal, extm_opt = (
                    rolling_backtest_monthly_param_sharpe(
                        df_prices              = df_sub,
                        df_instruments         = df_instruments,
                        param_sharpe_fn        = param_sharpe_fn,
                        start_date             = df_sub.index[0],
                        end_date               = df_sub.index[-1],
                        months_interval        = rebal_freq,
                        window_days            = window_days,
                        transaction_cost_value = transaction_cost_val,
                        transaction_cost_type  = cost_type,
                        trade_buffer_pct       = trade_buffer_pct,
                        daily_rf               = daily_rf
                    )
                )
            elif solver_choice == "Parametric (CVaR)":
                # placeholders
                sr_opt       = pd.Series(dtype=float)
                final_sh_opt = np.array([])
                df_rebal     = pd.DataFrame()
                extm_opt     = {}
                st.warning("Param(CVaR) not implemented. Provide param_cvar_fn(...) if needed.")
            else:
                # Direct (CVaR)
                sr_opt       = pd.Series(dtype=float)
                final_sh_opt = np.array([])
                df_rebal     = pd.DataFrame()
                extm_opt     = {}
                st.warning("Direct(CVaR) not implemented. Provide direct_cvar_fn(...) if needed.")

            extm_opt = extm_opt or {}

            # Build old drift
            sr_drift, shares_drift = pd.Series(dtype=float), np.array([])
            extm_drift = {}
            if "#Quantity" in df_instruments.columns:
                qty_map = {}
                for _, row in df_instruments.iterrows():
                    qty_map[row["#ID"]] = row["#Quantity"]
                if qty_map:
                    sr_drift_temp, shares_drift_temp = backtest_buy_and_hold_that_drifts(
                        df_prices  = df_sub,
                        start_date = df_sub.index[0],
                        end_date   = df_sub.index[-1],
                        ticker_qty = qty_map
                    )
                    sr_drift, shares_drift = sr_drift_temp, shares_drift_temp
                    if not sr_drift.empty:
                        extm_drift = compute_extended_metrics(sr_drift, daily_rf=daily_rf)

            # Build old strategic
            sr_strat, shares_strat = pd.Series(dtype=float), np.array([])
            extm_strat = {}
            if "Weight_Old" in df_instruments.columns:
                w_map = {}
                for _, row in df_instruments.iterrows():
                    w_map[row["#ID"]] = row["Weight_Old"]
                arr_ = []
                for c in df_sub.columns:
                    arr_.append(w_map.get(c, 0.0))
                arr_ = np.array(arr_, dtype=float)
                s_ = arr_.sum()
                if s_ > 1e-9:
                    arr_ /= s_
                    sr_strat_temp, shares_strat_temp = backtest_strategic_rebalanced(
                        df_prices              = df_sub,
                        start_date             = df_sub.index[0],
                        end_date               = df_sub.index[-1],
                        strategic_weights      = arr_,
                        months_interval        = 12,
                        transaction_cost_value = transaction_cost_val,
                        transaction_cost_type  = cost_type,
                        daily_rf               = daily_rf
                    )
                    sr_strat, shares_strat = sr_strat_temp, shares_strat_temp
                    if not sr_strat.empty:
                        extm_strat = compute_extended_metrics(sr_strat, daily_rf=daily_rf)

            # Store in session
            st.session_state["results"] = {
                "sr_opt":       sr_opt,
                "extm_opt":     extm_opt,
                "final_sh_opt": final_sh_opt,

                "sr_drift":     sr_drift,
                "extm_drift":   extm_drift,
                "shares_drift": shares_drift,

                "sr_strat":     sr_strat,
                "extm_strat":   extm_strat,
                "shares_strat": shares_strat,

                "df_rebal":     df_rebal,
                "asset_cls_list": asset_cls_list,
                "sec_type_list":  sec_type_list
            }

        # Show results if they exist
        if "results" in st.session_state:
            r_ = st.session_state["results"]
            sr_opt       = r_["sr_opt"]
            extm_opt     = r_["extm_opt"]
            final_sh_opt = r_["final_sh_opt"]

            sr_drift     = r_["sr_drift"]
            extm_drift   = r_["extm_drift"]
            shares_drift = r_["shares_drift"]

            sr_strat     = r_["sr_strat"]
            extm_strat   = r_["extm_strat"]
            shares_strat = r_["shares_strat"]

            df_rebal     = r_["df_rebal"]
            asset_cls_list = r_.get("asset_cls_list", [])

            st.write("## Rolling Backtest Comparison")
            show_opt   = st.checkbox("Optimized",   value=(not sr_opt.empty))
            show_drift = st.checkbox("Old Drift",   value=(not sr_drift.empty))
            show_strat = st.checkbox("Old Strategic", value=(not sr_strat.empty))

            # Build a DF for the performance chart
            df_plot = pd.DataFrame()
            if show_opt   and not sr_opt.empty:
                df_plot["Optimized"]     = sr_opt / sr_opt.iloc[0] - 1
            if show_drift and not sr_drift.empty:
                df_plot["Old Drift"]     = sr_drift / sr_drift.iloc[0] - 1
            if show_strat and not sr_strat.empty:
                df_plot["Old Strat"]     = sr_strat / sr_strat.iloc[0] - 1

            if not df_plot.empty:
                st.line_chart(df_plot * 100)
            else:
                st.warning("No lines selected for the chart.")

            # Extended metrics
            metrics_map = {}
            if show_opt   and not sr_opt.empty:
                metrics_map["Optimized"]     = extm_opt
            if show_drift and not sr_drift.empty:
                metrics_map["Old Drift"]     = extm_drift
            if show_strat and not sr_strat.empty:
                metrics_map["Old Strategic"] = extm_strat

            if metrics_map:
                st.write("## Extended Metrics for Selected Portfolios")
                display_multi_portfolio_metrics(metrics_map)

            # Interval Analysis
            st.write("## Interval Analysis")
            rebal_dates = []
            if df_rebal is not None and not df_rebal.empty and "Date" in df_rebal.columns:
                rebal_dates = sorted(df_rebal["Date"].unique())
                if len(rebal_dates) < 1 or (not sr_opt.empty and rebal_dates[0] > sr_opt.index[0]):
                    # add sr_opt.index[0] if missing
                    if not sr_opt.empty:
                        rebal_dates = [sr_opt.index[0]] + rebal_dates
                if not sr_opt.empty and rebal_dates[-1] < sr_opt.index[-1]:
                    rebal_dates.append(sr_opt.index[-1])

            portfolio_map = {}
            if show_opt   and not sr_opt.empty:
                portfolio_map["Optimized"]      = sr_opt
            if show_drift and not sr_drift.empty:
                portfolio_map["Old Drift"]      = sr_drift
            if show_strat and not sr_strat.empty:
                portfolio_map["Old Strategic"]  = sr_strat

            from modules.backtesting.rolling_intervals import (
                compute_multi_interval_returns,
                plot_multi_interval_bars,
                compute_predefined_pairs_diff_stats
            )

            if (len(rebal_dates) < 2) or (len(portfolio_map) < 2):
                st.info("Insufficient rebal dates or <2 portfolios => skip interval analysis.")
            else:
                df_multi = compute_multi_interval_returns(portfolio_map, rebal_dates)
                if df_multi.empty:
                    st.warning("No intervals computed.")
                else:
                    fig_ = plot_multi_interval_bars(df_multi)
                    if fig_ is not None:
                        st.plotly_chart(fig_)

                df_pairs = compute_predefined_pairs_diff_stats(portfolio_map, rebal_dates)
                if df_pairs.empty:
                    st.info("No pairwise stats because we don't have the needed pairs.")
                else:
                    st.write("### Pairwise Interval Stats (User-Defined Directions)")
                    st.dataframe(df_pairs)

            # Transaction cost (for the Optimized portfolio)
            if df_rebal is not None and not df_rebal.empty and not sr_opt.empty:
                final_val = sr_opt.iloc[-1]
                cost_stats = compute_cost_impact(df_rebal, final_val)
                st.write("### Transaction Cost Impact (Optimized Only)")
                st.dataframe(pd.DataFrame([cost_stats]))

            # Multi-drawdown
            st.write("## Drawdown Over Time (Selected Portfolios)")
            df_dd_plot = pd.DataFrame()
            if show_opt   and not sr_opt.empty:
                df_dd_plot["Optimized"]     = sr_opt
            if show_drift and not sr_drift.empty:
                df_dd_plot["Old Drift"]     = sr_drift
            if show_strat and not sr_strat.empty:
                df_dd_plot["Old Strategic"] = sr_strat

            if df_dd_plot.empty:
                st.warning("No portfolios selected for drawdown.")
            else:
                fig_dd = plot_multi_drawdown(df_dd_plot)
                if fig_dd is not None:
                    st.plotly_chart(fig_dd)

            # highlight for "Diff" columns
            def highlight_diff(val):
                if val > 0:
                    return "color: green"
                elif val < 0:
                    return "color: red"
                else:
                    return "color: black"

            # Final Asset-Class Weights
            st.write("## Final Asset-Class Comparison")
            w_opt_aclass   = pd.Series(dtype=float)
            w_drift_aclass = pd.Series(dtype=float)
            w_strat_aclass = pd.Series(dtype=float)

            if show_opt and not sr_opt.empty and (len(final_sh_opt) == df_sub.shape[1]):
                w_opt_aclass = compute_final_aclass_weights(final_sh_opt, df_sub, asset_cls_list)

            if show_drift and not sr_drift.empty and (len(shares_drift) == df_sub.shape[1]):
                w_drift_aclass = compute_final_aclass_weights(shares_drift, df_sub, asset_cls_list)

            if show_strat and not sr_strat.empty and (len(shares_strat) == df_sub.shape[1]):
                w_strat_aclass = compute_final_aclass_weights(shares_strat, df_sub, asset_cls_list)

            if not w_opt_aclass.empty and not w_drift_aclass.empty:
                st.write("### Optimized vs Old Drift (Asset-Class Weights)")
                df_cmp_ac = pd.DataFrame({"Optimized": w_opt_aclass, "Old Drift": w_drift_aclass}).fillna(0.0)
                df_cmp_ac["Diff"] = df_cmp_ac["Optimized"] - df_cmp_ac["Old Drift"]
                st.dataframe(
                    df_cmp_ac.style.format("{:.2%}").applymap(highlight_diff, subset=["Diff"])
                )

            if not w_opt_aclass.empty and not w_strat_aclass.empty:
                st.write("### Optimized vs Old Strategic (Asset-Class Weights)")
                df_cmp_ac2 = pd.DataFrame({"Optimized": w_opt_aclass, "Old Strategic": w_strat_aclass}).fillna(0.0)
                df_cmp_ac2["Diff"] = df_cmp_ac2["Optimized"] - df_cmp_ac2["Old Strategic"]
                st.dataframe(
                    df_cmp_ac2.style.format("{:.2%}").applymap(highlight_diff, subset=["Diff"])
                )

            # Final Instrument Weights
            st.write("## Final Instrument Comparison")
            w_opt_instr   = pd.Series(dtype=float)
            w_drift_instr = pd.Series(dtype=float)
            w_strat_instr = pd.Series(dtype=float)

            if show_opt and not sr_opt.empty and (len(final_sh_opt) == df_sub.shape[1]):
                w_opt_instr = compute_final_instrument_weights(final_sh_opt, df_sub)

            if show_drift and not sr_drift.empty and (len(shares_drift) == df_sub.shape[1]):
                w_drift_instr = compute_final_instrument_weights(shares_drift, df_sub)

            if show_strat and not sr_strat.empty and (len(shares_strat) == df_sub.shape[1]):
                w_strat_instr = compute_final_instrument_weights(shares_strat, df_sub)

            # Compare instruments: Optimized vs Old Drift
            if not w_opt_instr.empty and not w_drift_instr.empty:
                st.write("### Optimized vs Old Drift (Instrument Weights)")
                df_cmp_instr = pd.DataFrame({"Optimized": w_opt_instr, "Old Drift": w_drift_instr}).fillna(0.0)
                df_cmp_instr["Diff"] = df_cmp_instr["Optimized"] - df_cmp_instr["Old Drift"]
                st.dataframe(
                    df_cmp_instr.style.format("{:.2%}").applymap(highlight_diff, subset=["Diff"])
                )

            # Compare instruments: Optimized vs Old Strategic
            if not w_opt_instr.empty and not w_strat_instr.empty:
                st.write("### Optimized vs Old Strategic (Instrument Weights)")
                df_cmp_instr2 = pd.DataFrame({"Optimized": w_opt_instr, "Old Strategic": w_strat_instr}).fillna(0.0)
                df_cmp_instr2["Diff"] = df_cmp_instr2["Optimized"] - df_cmp_instr2["Old Strategic"]
                st.dataframe(
                    df_cmp_instr2.style.format("{:.2%}").applymap(highlight_diff, subset=["Diff"])
                )

            # Excel export
            excel_bytes = export_backtest_results_to_excel(
                sr_line_new      = sr_opt,
                sr_line_old      = sr_drift,
                df_rebal         = df_rebal,
                ext_metrics_new  = extm_opt,
                ext_metrics_old  = {}
            )
            st.download_button(
                label="Download Backtest Excel",
                data=excel_bytes,
                file_name="backtest_results.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
            )

    else:
        # "Hyperparameter Optimization"
        st.write("## Hyperparameter Optimization")
        # We call your show_hyperparam_ui, which you gave in modules/ui/hyperparam_ui.py
        show_hyperparam_ui(
            df_sub              = df_sub,
            df_instruments      = df_instruments,
            asset_cls_list      = asset_cls_list,
            sec_type_list       = sec_type_list,
            class_sum_constraints = class_sum_constraints,
            subtype_constraints = subtype_constraints,
            daily_rf            = daily_rf,
            transaction_cost_value = transaction_cost_val,
            transaction_cost_type  = cost_type,
            trade_buffer_pct    = trade_buffer_pct
        )


if __name__ == "__main__":
    main()