# File: modules/analytics/display_utils.py

import streamlit as st
import pandas as pd
import numpy as np

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

    # same groupings as your code
    performance_keys= ["Total Return","Annual Return","Annual Vol","Sharpe"]
    risk_keys       = ["MaxDD","TimeToRecovery","VaR_1M99","CVaR_1M99"]
    ratio_keys      = ["Skew","Kurtosis","Sortino","Calmar","Omega"]

    # 1) Performance
    df_perf = build_metric_df(performance_keys, old_metrics, new_metrics)
    st.write("### Extended Metrics - Performance")
    st.dataframe(format_ext_metrics(df_perf))

    # 2) Risk
    df_risk = build_metric_df(risk_keys, old_metrics, new_metrics)
    st.write("### Extended Metrics - Risk")
    st.dataframe(format_ext_metrics(df_risk))

    # 3) Ratios
    df_ratio = build_metric_df(ratio_keys, old_metrics, new_metrics)
    st.write("### Extended Metrics - Ratios")
    st.dataframe(format_ext_metrics(df_ratio))