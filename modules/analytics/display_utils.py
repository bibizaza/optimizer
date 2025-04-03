# File: modules/analytics/display_utils.py

import streamlit as st
import pandas as pd
import numpy as np

###############################################################################
# Existing utility (unchanged)
###############################################################################
def build_metric_df(metric_keys, old_metrics, new_metrics):
    rows = []
    for mk in metric_keys:
        val_old = old_metrics.get(mk, 0.0)
        val_new = new_metrics.get(mk, 0.0)
        rows.append((mk, val_old, val_new))
    df_out = pd.DataFrame(rows, columns=["Metric","Old","New"])
    df_out.set_index("Metric", inplace=True)
    return df_out

def format_ext_metrics(df_):
    """
    Format a DataFrame that has index=metric_name, columns=["Old","New"].
    Certain metrics are shown as percentages, etc.
    """
    def format_val(mk, val):
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
            rawv= dfx.loc[mk, colx]
            dfx.loc[mk, colx] = format_val(mk, rawv)
    return dfx

def display_extended_metrics(old_metrics: dict, new_metrics: dict):
    """
    Show performance, risk, ratio metrics in separate tables.
    old_metrics, new_metrics come from e.g. compute_extended_metrics(...)
    """
    performance_keys= ["Total Return","Annual Return","Annual Vol","Sharpe"]
    risk_keys       = ["MaxDD","TimeToRecovery","VaR_1M99","CVaR_1M99"]
    ratio_keys      = ["Skew","Kurtosis","Sortino","Calmar","Omega"]

    df_perf = build_metric_df(performance_keys, old_metrics, new_metrics)
    st.write("### Extended Metrics - Performance")
    st.dataframe(format_ext_metrics(df_perf))

    df_risk = build_metric_df(risk_keys, old_metrics, new_metrics)
    st.write("### Extended Metrics - Risk")
    st.dataframe(format_ext_metrics(df_risk))

    df_ratio = build_metric_df(ratio_keys, old_metrics, new_metrics)
    st.write("### Extended Metrics - Ratios")
    st.dataframe(format_ext_metrics(df_ratio))


###############################################################################
# NEW: Multi-Portfolio Display
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
        # metrics_map keys = portfolio names
        port_names = list(metrics_map.keys())
        rows = []
        for mk in metric_keys:
            row = [mk]  # first col is the metric name
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

    # Now build & display each category
    for (title, keys) in [
        ("Performance", performance_keys),
        ("Risk",        risk_keys),
        ("Ratios",      ratio_keys)
    ]:
        st.write(f"### Extended Metrics - {title}")
        df_cat = build_dataframe_for_keys(keys, metrics_map)
        st.dataframe(format_ext_metrics_multi(df_cat))
