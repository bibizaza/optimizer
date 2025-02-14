# modules/backtesting/rolling_intervals.py

import pandas as pd
import numpy as np
import plotly.express as px

def compute_interval_returns(
    sr_line_old: pd.Series,
    sr_line_new: pd.Series,
    rebal_dates: list[pd.Timestamp],
    label_old: str = "Old",
    label_new: str = "New"
) -> pd.DataFrame:
    """
    Given two daily cumulative performance lines (sr_line_old, sr_line_new)
    and a list of sorted rebal_dates, compute the interval returns for each portfolio
    between each pair of consecutive rebalancing dates.

    Parameters
    ----------
    sr_line_old : pd.Series
        Daily cumulative performance for the old portfolio (e.g. from rolling backtest),
        indexed by date.
    sr_line_new : pd.Series
        Same but for the new/optimized portfolio.
    rebal_dates : list of pd.Timestamp
        The sorted list of rebalancing dates used in the rolling backtest.
        Typically something like [start_day, rebal1, rebal2, ..., final_day].
        Must exist in sr_line_* indexes, or at least be consistent with them.
    label_old : str
        The label for the old portfolio (default "Old").
    label_new : str
        The label for the new portfolio (default "New").

    Returns
    -------
    df_intervals : pd.DataFrame
        Columns:
          - "Interval Start"
          - "Interval End"
          - f"{label_old}(%)"
          - f"{label_new}(%)"
          - "Diff(%)" => (new - old) * 100
    """
    rows = []
    # Ensure rebal_dates is sorted and unique
    rebal_dates_sorted = sorted(list(set(rebal_dates)))
    # We'll also ensure any final date is included if needed
    # but typically your rolling backtest already has final date as last rebalance.

    for i in range(len(rebal_dates_sorted) - 1):
        start_d = rebal_dates_sorted[i]
        end_d   = rebal_dates_sorted[i + 1]

        # subset old/new lines to [start_d, end_d]
        sub_old = sr_line_old.loc[start_d:end_d]
        sub_new = sr_line_new.loc[start_d:end_d]

        # If missing data or less than 2 points, skip
        if len(sub_old) < 2 or len(sub_new) < 2:
            continue

        # interval return => last/first - 1
        ret_old = sub_old.iloc[-1] / sub_old.iloc[0] - 1
        ret_new = sub_new.iloc[-1] / sub_new.iloc[0] - 1

        rows.append({
            "Interval Start": start_d,
            "Interval End":   end_d,
            f"{label_old}(%)": ret_old * 100,
            f"{label_new}(%)": ret_new * 100,
            "Diff(%)": (ret_new - ret_old) * 100
        })

    df_intervals = pd.DataFrame(rows)
    return df_intervals

def plot_interval_bars(
    df_intervals: pd.DataFrame,
    label_old: str = "Old(%)",
    label_new: str = "New(%)",
    display_mode: str = "difference"
):
    """
    Create a Plotly bar chart from the df_intervals produced by compute_interval_returns.

    Parameters
    ----------
    df_intervals : pd.DataFrame
        Must have columns like ["Interval Start", "Interval End", label_old, label_new, "Diff(%)"].
    label_old : str
        The column name for old portfolio returns in percent.
    label_new : str
        Column name for new portfolio returns in percent.
    display_mode : str
        "difference" => shows one bar per interval => "Diff(%)"
        "grouped"    => shows two bars => old vs. new.

    Returns
    -------
    fig : plotly.graph_objects.Figure
        A bar chart figure ready for st.plotly_chart(fig).
    """
    import plotly.express as px

    if display_mode == "difference":
        # Single bar for each interval => difference
        if "Diff(%)" not in df_intervals.columns:
            raise ValueError("df_intervals must have 'Diff(%)' column for difference mode.")

        df_plot = df_intervals.copy()
        df_plot["Interval"] = df_plot["Interval Start"].astype(str)  # or combine Start/End
        fig = px.bar(
            df_plot,
            x="Interval",
            y="Diff(%)",
            title="Interval Return Difference (New - Old)",
            labels={"Interval": "Rebalance Interval", "Diff(%)": "Return Difference (%)"}
        )
        fig.update_layout(xaxis=dict(type="category"))
        return fig

    elif display_mode == "grouped":
        # side-by-side bars => old vs new
        missing_cols = []
        for c in [label_old, label_new]:
            if c not in df_intervals.columns:
                missing_cols.append(c)
        if missing_cols:
            raise ValueError(f"df_intervals missing columns: {missing_cols}")

        df_plot = df_intervals.copy()
        df_plot["Interval"] = df_plot["Interval Start"].astype(str)
        # melt to have variable column = "Portfolio", value column = "Return(%)"
        df_melt = df_plot.melt(
            id_vars=["Interval", "Interval End"],
            value_vars=[label_old, label_new],
            var_name="Portfolio",
            value_name="Return(%)"
        )
        fig = px.bar(
            df_melt,
            x="Interval",
            y="Return(%)",
            color="Portfolio",
            barmode="group",
            title="Interval Returns: Old vs. New",
            labels={"Interval": "Rebalance Interval", "Return(%)": "Interval Return (%)"}
        )
        fig.update_layout(xaxis=dict(type="category"))
        return fig

    else:
        raise ValueError("display_mode must be 'difference' or 'grouped'")