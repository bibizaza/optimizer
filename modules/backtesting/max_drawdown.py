# modules/backtesting/max_drawdown.py

import pandas as pd
import plotly.express as px

def compute_drawdown_series(series_abs: pd.Series) -> pd.Series:
    """
    Computes the daily drawdown series from the start of `series_abs`.
    Drawdown at time t is ( current_value / running_max ) - 1.

    Parameters
    ----------
    series_abs : pd.Series
        Daily absolute portfolio values. Index = dates, values = portfolio value.

    Returns
    -------
    drawdown_series : pd.Series
        A timeseries of drawdown (negative or zero).
        If we have fewer than 2 points, the entire series is 0.0 or NaN accordingly.
    """
    if len(series_abs) < 2:
        # Not enough data => return a series of zeros or something
        return pd.Series([0.0]*len(series_abs), index=series_abs.index)

    running_max = series_abs.cummax()
    drawdown = (series_abs / running_max) - 1.0
    return drawdown


def compute_max_drawdown(series_abs: pd.Series) -> float:
    """
    Computes the *maximum* drawdown from the start of `series_abs` until the end.
    Typically you pass in a daily absolute portfolio value timeseries.
    The maximum drawdown is the *minimum* of the drawdown series.
    e.g. -0.25 means a 25% drop from a peak.

    If the series has fewer than 2 points, returns 0.0 by default.
    """
    if len(series_abs) < 2:
        return 0.0

    dd_series = compute_drawdown_series(series_abs)
    max_dd = dd_series.min()  # typically negative
    return max_dd


def show_max_drawdown_comparison(df_compare: pd.DataFrame) -> pd.DataFrame:
    """
    Given a df_compare with "Old_Ptf" and "New_Ptf" columns,
    computes their *final* max drawdowns (a single number),
    and returns a small DataFrame with one row: ["Old MaxDD", "New MaxDD"].
    """
    columns_needed = ["Old_Ptf", "New_Ptf"]
    for col in columns_needed:
        if col not in df_compare.columns:
            raise ValueError(f"df_compare missing required column '{col}'")

    old_dd = compute_max_drawdown(df_compare["Old_Ptf"])
    new_dd = compute_max_drawdown(df_compare["New_Ptf"])

    return pd.DataFrame({
        "Old MaxDD": [old_dd],
        "New MaxDD": [new_dd]
    })


def plot_drawdown_series(df_compare: pd.DataFrame):
    """
    Plots the *historical* drawdown lines for both "Old_Ptf" and "New_Ptf".
    The resulting figure shows how each portfolio's drawdown evolves over time.

    Parameters
    ----------
    df_compare : pd.DataFrame
        Must have "Old_Ptf" and "New_Ptf" columns. Index = dates.

    Returns
    -------
    fig : plotly.graph_objs._figure.Figure
        The Plotly figure with the 2 drawdown lines over time.
    """
    needed_cols = ["Old_Ptf", "New_Ptf"]
    for c in needed_cols:
        if c not in df_compare.columns:
            raise ValueError(f"df_compare missing column {c}")

    # 1) compute drawdown series for old & new
    old_drawdown = compute_drawdown_series(df_compare["Old_Ptf"])
    new_drawdown = compute_drawdown_series(df_compare["New_Ptf"])

    # 2) combine in a DataFrame for plotting
    df_dd = pd.DataFrame({
        "Old Drawdown": old_drawdown,
        "New Drawdown": new_drawdown
    }, index=df_compare.index)

    # 3) create a line chart with plotly express
    #    Because these are negative or 0, we'll just do a standard line plot
    fig = px.line(
        df_dd,
        x=df_dd.index,
        y=df_dd.columns,
        title="Historical Drawdown Over Time (Old vs New)",
        labels={"value":"Drawdown", "index":"Date", "variable":"Portfolio"}
    )

    # By default, the lines will be labeled by the column names: "Old Drawdown", "New Drawdown"
    # Optionally, we can set a y-axis format, etc.
    fig.update_yaxes(tickformat=".2%")  # to show as percents e.g. -20%
    return fig