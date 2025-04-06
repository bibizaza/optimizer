# File: modules/backtesting/max_drawdown.py

import pandas as pd
import plotly.express as px

def compute_drawdown_series(series_abs: pd.Series) -> pd.Series:
    """
    Computes the daily drawdown series from the start of `series_abs`.
    Drawdown at time t is ( series_abs[t] / running_max[t] ) - 1,
    which is typically <= 0.  If there's not enough data, we return zeros.
    """
    if len(series_abs) < 2:
        return pd.Series([0.0]*len(series_abs), index=series_abs.index)

    running_max = series_abs.cummax()
    drawdown = (series_abs / running_max) - 1.0
    return drawdown


def compute_max_drawdown(series_abs: pd.Series) -> float:
    """
    The single maximum drawdown (a negative number).
    If < 2 points, returns 0.0 by default.
    """
    if len(series_abs) < 2:
        return 0.0
    dd_series = compute_drawdown_series(series_abs)
    return dd_series.min()  # e.g., -0.30 => -30%


def build_drawdown_df(df_abs: pd.DataFrame) -> pd.DataFrame:
    """
    Given a DataFrame of absolute values, with each column
    representing a portfolio, compute the daily drawdown for each.
    Returns a new DataFrame with the same shape & index:
      e.g. columns = ["New Optimized","Old Drift","Old Strategic"]
    If a column has < 2 points, that column is all zeros.
    """
    dd_map = {}
    for col in df_abs.columns:
        dd_map[col] = compute_drawdown_series(df_abs[col])
    df_dd = pd.DataFrame(dd_map, index=df_abs.index)
    return df_dd


def plot_drawdown_series(df_abs: pd.DataFrame, custom_title: str = "Historical Drawdown"):
    """
    Build a multi-line drawdown chart for whichever columns are in df_abs.
    Each column in df_abs is a separate portfolio's absolute values.
    We'll compute each column's drawdown, then plot them all in one figure.
    """
    if df_abs.empty:
        raise ValueError("df_abs is empty, no data to plot drawdown.")

    # Compute drawdown for each column
    df_dd = build_drawdown_df(df_abs)

    # For labeling in the figure, we keep the original column names
    fig = px.line(
        df_dd,
        x=df_dd.index,
        y=df_dd.columns,
        title=custom_title,
        labels={"value": "Drawdown", "index": "Date", "variable": "Portfolio"}
    )
    # format y-axis as a percentage
    fig.update_yaxes(tickformat=".2%")
    return fig


def show_max_drawdown_table(df_abs: pd.DataFrame) -> pd.DataFrame:
    """
    Return a small DataFrame with each column's MaxDD as a single row:
      e.g. "New Optimized": -0.31,
           "Old Drift":     -0.25,
           "Old Strategic": -0.40
    If a column has < 2 points, we consider its maxDD= 0.
    """
    if df_abs.empty:
        return pd.DataFrame()

    results = {}
    for col in df_abs.columns:
        mdd = compute_max_drawdown(df_abs[col])
        results[col] = mdd

    # Make a single-row DataFrame
    df_out = pd.DataFrame([results], index=["MaxDD"])
    return df_out