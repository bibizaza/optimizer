# modules/backtesting/max_drawdown.py

import pandas as pd
import plotly.express as px

def plot_drawdown_series(df_abs: pd.DataFrame, custom_title: str = "Drawdown Over Time"):
    """
    Plots the historical drawdown lines for all columns in df_abs.
    Each column is assumed to be an absolute (normalized) portfolio series.

    - We compute the running max for each column => drawdown series => (value / running_max) - 1.
    - We apply a consistent color scheme for columns named:
        "New Optimized" => #1f77b4
        "Old Drift"      => grey
        "Old Strategic"  => lightblue
      All others => #666666 fallback.

    Returns: a Plotly figure object.
    """
    if df_abs.empty:
        raise ValueError("plot_drawdown_series: df_abs is empty.")

    # 1) Compute daily drawdown for each column
    df_dd = pd.DataFrame(index=df_abs.index)
    for col in df_abs.columns:
        running_max = df_abs[col].cummax()
        df_dd[col] = (df_abs[col] / running_max) - 1.0

    # 2) Assign colors matching your app’s color scheme
    color_map = {
        "New Optimized": "#D6B77D",
        "Old Drift": "grey",
        "Old Strategic": "#00B0F0",
    }
    color_sequence = [color_map.get(col, "#666666") for col in df_dd.columns]

    # 3) Plot with Plotly Express
    fig = px.line(
        df_dd,
        x=df_dd.index,
        y=df_dd.columns,
        title=custom_title,
        labels={"value": "Drawdown", "index": "Date", "variable": "Portfolio"},
        color_discrete_sequence=color_sequence
    )
    # Format Y-axis as percentage
    fig.update_yaxes(tickformat=".2%")
    return fig


def show_max_drawdown_table(df_abs: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with each column's max drawdown (the minimum of the drawdown series).
    The returned DataFrame has:
        Index => the portfolio name (the column name),
        Column => "Max Drawdown" as a float (negative).
    """
    if df_abs.empty:
        # Return an empty DataFrame or raise an error
        return pd.DataFrame()

    # Compute daily drawdown
    df_dd = pd.DataFrame(index=df_abs.index)
    for col in df_abs.columns:
        running_max = df_abs[col].cummax()
        df_dd[col] = (df_abs[col] / running_max) - 1.0

    # Gather each column's min => max drawdown
    rows = []
    for col in df_dd.columns:
        md = df_dd[col].min()  # e.g. -0.35 => -35%
        rows.append({"Portfolio": col, "Max Drawdown": md})

    df_out = pd.DataFrame(rows).set_index("Portfolio")
    return df_out
