# modules/backtesting/max_drawdown.py

import pandas as pd
import plotly.express as px

def compute_drawdown_series(series_abs: pd.Series) -> pd.Series:
    """
    Computes the daily drawdown series from the start of `series_abs`.
    Drawdown at time t is ( series_abs[t] / running_max[t] ) - 1.
    Typically negative or zero.
    """
    if len(series_abs) < 2:
        return pd.Series([0.0]*len(series_abs), index=series_abs.index)

    running_max = series_abs.cummax()
    drawdown = (series_abs / running_max) - 1.0
    return drawdown


def compute_max_drawdown(series_abs: pd.Series) -> float:
    """
    The single maximum drawdown (a negative number, e.g. -0.25 => -25%).
    If < 2 points, returns 0.0 by default.
    """
    if len(series_abs) < 2:
        return 0.0

    dd_series = compute_drawdown_series(series_abs)
    return dd_series.min()  # negative, e.g. -0.30 => -30%


def show_max_drawdown_comparison(df_compare: pd.DataFrame) -> pd.DataFrame:
    """
    df_compare must have "Old_Ptf" and "New_Ptf" columns, each a timeseries of absolute values.
    Returns a small one-row DataFrame:
      Old MaxDD   New MaxDD
    """
    required_cols = ["Old_Ptf", "New_Ptf"]
    for col in required_cols:
        if col not in df_compare.columns:
            raise ValueError(f"df_compare is missing required column '{col}'")

    old_dd = compute_max_drawdown(df_compare["Old_Ptf"])
    new_dd = compute_max_drawdown(df_compare["New_Ptf"])

    return pd.DataFrame({
        "Old MaxDD": [old_dd],
        "New MaxDD": [new_dd]
    })


def plot_drawdown_series(df_compare: pd.DataFrame):
    """
    Plots the historical drawdown lines for "Old_Ptf" and "New_Ptf".
    Returns a Plotly figure with lines for Old Drawdown vs. New Drawdown.
    """
    needed_cols = ["Old_Ptf", "New_Ptf"]
    for c in needed_cols:
        if c not in df_compare.columns:
            raise ValueError(f"df_compare missing column '{c}'")

    old_dd = compute_drawdown_series(df_compare["Old_Ptf"])
    new_dd = compute_drawdown_series(df_compare["New_Ptf"])

    df_dd = pd.DataFrame({
        "Old Drawdown": old_dd,
        "New Drawdown": new_dd
    }, index=df_compare.index)

    fig = px.line(
        df_dd,
        x=df_dd.index,
        y=df_dd.columns,
        title="Historical Drawdown Over Time (Old vs New)",
        labels={"value":"Drawdown", "index":"Date", "variable":"Portfolio"}
    )
    # Format Y-axis as a percentage
    fig.update_yaxes(tickformat=".2%")
    return fig