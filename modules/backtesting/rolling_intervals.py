# File: modules/backtesting/rolling_intervals.py

import streamlit as st
import pandas as pd
import plotly.express as px

def compute_interval_returns(
    sr_line_old: pd.Series,
    sr_line_new: pd.Series,
    rebal_dates: list[pd.Timestamp],
    label_old: str = "Old",
    label_new: str = "New"
) -> pd.DataFrame:
    """
    Computes interval returns for old vs new between consecutive rebalancing dates.
    Each row => "Interval Start", "Interval End", label_old+"(%)", label_new+"(%)", "Diff(%)"
    """
    rows = []
    rebal_dates_sorted = sorted(list(set(rebal_dates)))

    for i in range(len(rebal_dates_sorted) - 1):
        start_d = rebal_dates_sorted[i]
        end_d   = rebal_dates_sorted[i + 1]

        sub_old = sr_line_old.loc[start_d:end_d]
        sub_new = sr_line_new.loc[start_d:end_d]

        if len(sub_old) < 2 or len(sub_new) < 2:
            continue

        ret_old = sub_old.iloc[-1] / sub_old.iloc[0] - 1
        ret_new = sub_new.iloc[-1] / sub_new.iloc[0] - 1

        rows.append({
            "Interval Start": start_d,
            "Interval End":   end_d,
            f"{label_old}(%)": ret_old * 100,
            f"{label_new}(%)": ret_new * 100,
            "Diff(%)": (ret_new - ret_old) * 100
        })

    return pd.DataFrame(rows)

def plot_interval_bars(
    df_intervals: pd.DataFrame,
    label_old: str = "Old(%)",
    label_new: str = "New(%)",
    display_mode: str = "grouped"
):
    """
    Create a Plotly bar chart for interval returns.

    display_mode => "difference" => one bar per interval => "Diff(%)"
                 => "grouped"    => side-by-side bars for old vs new
    """
    if display_mode == "difference":
        if "Diff(%)" not in df_intervals.columns:
            raise ValueError("df_intervals must have 'Diff(%)' column for difference mode.")
        df_plot = df_intervals.copy()
        df_plot["Interval"] = df_plot["Interval Start"].astype(str)
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

def compute_interval_stats(
    df_intervals: pd.DataFrame,
    diff_col: str = "Diff(%)"
) -> dict:
    """
    Compute summary metrics from df_intervals (which has 'Diff(%)'):
      - Number of intervals
      - Win Count
      - Win Rate(%)
      - Average Diff(%)
      - Median Diff(%)
      - Max Diff(%)
      - Min Diff(%)
      - Average Positive Diff(%)
      - Average Negative Diff(%)
    """
    if diff_col not in df_intervals.columns:
        raise ValueError(f"df_intervals must have '{diff_col}' column for stats.")

    diffs = df_intervals[diff_col]
    n_intervals = len(diffs)
    if n_intervals == 0:
        return {
            "Number of Intervals": 0,
            "Win Count": 0,
            "Win Rate(%)": 0.0,
            "Average Diff(%)": 0.0,
            "Median Diff(%)": 0.0,
            "Max Diff(%)": 0.0,
            "Min Diff(%)": 0.0,
            "Average Positive Diff(%)": 0.0,
            "Average Negative Diff(%)": 0.0
        }

    wins = diffs[diffs > 0]
    lose = diffs[diffs < 0]
    n_win = len(wins)
    win_rate = 100.0 * n_win / n_intervals

    avg_diff = diffs.mean()
    med_diff = diffs.median()
    max_diff = diffs.max()
    min_diff = diffs.min()

    avg_positive = wins.mean() if len(wins) > 0 else 0.0
    avg_negative = lose.mean() if len(lose) > 0 else 0.0

    stats = {
        "Number of Intervals": n_intervals,
        "Win Count": n_win,
        "Win Rate(%)": round(win_rate, 2),
        "Average Diff(%)": round(avg_diff, 3),
        "Median Diff(%)": round(med_diff, 3),
        "Max Diff(%)": round(max_diff, 3),
        "Min Diff(%)": round(min_diff, 3),
        "Average Positive Diff(%)": round(avg_positive, 3),
        "Average Negative Diff(%)": round(avg_negative, 3)
    }
    return stats

def display_interval_bars_and_stats(
    sr_line_old: pd.Series,
    sr_line_new: pd.Series,
    rebal_dates: list[pd.Timestamp],
    label_old: str = "Old",
    label_new: str = "New",
    display_mode: str = "grouped"
):
    """
    High-level function that:
     1) Computes interval returns,
     2) Plots a bar chart,
     3) Shows summary stats (win rate, avg diff, etc.) in a small table below.

    This keeps optima_optimizer.py simpler:
     - Just call display_interval_bars_and_stats(...) once you have sr_line_old, sr_line_new, rebal_dates
    """
    # 1) Compute intervals
    df_intervals = compute_interval_returns(sr_line_old, sr_line_new, rebal_dates, label_old, label_new)

    # 2) Build bar chart
    fig = plot_interval_bars(
        df_intervals=df_intervals,
        label_old=f"{label_old}(%)",
        label_new=f"{label_new}(%)",
        display_mode=display_mode
    )
    st.plotly_chart(fig)

    # 3) Summaries
    stats_dict = compute_interval_stats(df_intervals, diff_col="Diff(%)")
    df_stats = pd.DataFrame([stats_dict])
    st.write("### Interval Stats")
    st.table(df_stats)