# File: modules/analytics/display_utils.py

import streamlit as st
import pandas as pd
import numpy as np

def display_extended_metrics(metrics_map: dict):
    """
    Display extended metrics for one or more portfolios side-by-side.

    Parameters
    ----------
    metrics_map : dict
        A dictionary of the form:
            {
              "Portfolio Label" : { "Total Return": float, "Annual Return": float, ... },
              "Another Label"   : { "Total Return": float, "Annual Return": float, ... },
              ...
            }
        Each inner dictionary is typically the output of compute_extended_metrics(...).
    """

    # Define which metrics go in each category
    performance_keys = ["Total Return","Annual Return","Annual Vol","Sharpe"]
    risk_keys        = ["MaxDD","TimeToRecovery","VaR_1M99","CVaR_1M99"]
    ratio_keys       = ["Skew","Kurtosis","Sortino","Calmar","Omega"]

    # Helper to format each numeric cell
    def format_val(metric_name: str, val: float) -> str:
        pct_metrics = ["Total Return","Annual Return","Annual Vol","MaxDD","VaR_1M99","CVaR_1M99"]
        if metric_name in pct_metrics:
            return f"{val*100:.2f}%"
        elif metric_name == "TimeToRecovery":
            return f"{val:.0f}"
        else:
            return f"{val:.3f}"

    def make_table_for_category(metric_keys_list):
        """
        Build a DataFrame for one category of metrics (e.g., performance, risk, ratio).
        Rows = metric names
        Columns = each portfolio label from metrics_map
        """
        rows = []
        for mk in metric_keys_list:
            row_dict = {"Metric": mk}
            for portfolio_label, mdict in metrics_map.items():
                row_dict[portfolio_label] = mdict.get(mk, 0.0)
            rows.append(row_dict)

        df_cat = pd.DataFrame(rows)
        df_cat.set_index("Metric", inplace=True)

        # Format each cell
        for mk in df_cat.index:
            for col_ in df_cat.columns:
                raw_val = df_cat.loc[mk, col_]
                df_cat.loc[mk, col_] = format_val(mk, raw_val)

        return df_cat

    # 1) Performance category
    st.write("### Extended Metrics - Performance")
    df_perf = make_table_for_category(performance_keys)
    st.dataframe(df_perf)

    # 2) Risk category
    st.write("### Extended Metrics - Risk")
    df_risk = make_table_for_category(risk_keys)
    st.dataframe(df_risk)

    # 3) Ratios category
    st.write("### Extended Metrics - Ratios")
    df_ratio = make_table_for_category(ratio_keys)
    st.dataframe(df_ratio)