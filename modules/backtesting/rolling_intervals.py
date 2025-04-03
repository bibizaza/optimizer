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
    Where "Diff(%)" = (new - old) * 100 in each interval.
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

def compute_interval_stats(df_intervals: pd.DataFrame, diff_col: str = "Diff(%)") -> dict:
    """
    Summarize the 'Diff(%)' column:
      - # intervals
      - # wins (Diff>0)
      - Win rate
      - Average diff
      - Median diff
      - etc.
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

def compute_predefined_pairs_diff_stats(
    portfolio_map: dict[str, pd.Series],
    rebal_dates: list[pd.Timestamp]
) -> pd.DataFrame:
    """
    The user wants EXACT pairs in a certain direction:
      1) "Optimized vs Old Drift" => Diff = (Optimized - Drift)
      2) "Optimized vs Old Strategic" => Diff = (Optimized - Strat)
      3) "Old Drift vs Old Strategic" => Diff = (Drift - Strat)

    So we define a small list of possible pairs in the desired direction:
      [ (p1_name, p2_name, "Label for row", ... ), ... ]

    We only compute stats if both p1_name and p2_name are in portfolio_map.
    The 'Diff(%)' in the intervals will be sr_line_new - sr_line_old => (p2 - p1).
    """
    pairs_setup = [
        ("Old Drift", "Optimized",    "Optimized vs Old Drift"),      # Diff = (Optimized - Drift)
        ("Old Strategic", "Optimized","Optimized vs Old Strategic"),  # Diff = (Optimized - Old Strategic)
        ("Old Strategic", "Old Drift","Old Drift vs Old Strategic"),  # Diff = (Drift - Strategic)
    ]

    rows = []
    for p1_name, p2_name, row_label in pairs_setup:
        if (p1_name in portfolio_map) and (p2_name in portfolio_map):
            sr1 = portfolio_map[p1_name]  # "old"
            sr2 = portfolio_map[p2_name]  # "new"
            df_int = compute_interval_returns(sr1, sr2, rebal_dates,
                                              label_old=p1_name, label_new=p2_name)
            stats_ = compute_interval_stats(df_int, diff_col="Diff(%)")
            rowd = {"Pair": row_label}
            rowd.update(stats_)
            rows.append(rowd)

    if not rows:
        return pd.DataFrame()
    return pd.DataFrame(rows)

def compute_multi_interval_returns(
    portfolio_map: dict[str, pd.Series],
    rebal_dates: list[pd.Timestamp]
) -> pd.DataFrame:
    """
    For a set of portfolios (≥2) => produce DF with columns:
      ["Interval Start","Interval End","Portfolio","Return(%)"]
    for a grouped bar chart. We do not compute diffs here.
    """
    rows = []
    rebal_dates_sorted = sorted(list(set(rebal_dates)))
    for i in range(len(rebal_dates_sorted)-1):
        start_d = rebal_dates_sorted[i]
        end_d   = rebal_dates_sorted[i+1]
        for pname, sr_ in portfolio_map.items():
            sub_ = sr_.loc[start_d:end_d]
            if len(sub_)<2:
                continue
            r_ = sub_.iloc[-1]/ sub_.iloc[0]-1
            rows.append({
                "Interval Start": start_d,
                "Interval End":   end_d,
                "Portfolio":      pname,
                "Return(%)":      r_*100.0
            })
    return pd.DataFrame(rows)

def plot_multi_interval_bars(df_intervals: pd.DataFrame):
    """
    Grouped bar chart => X=Interval, color=Portfolio, Y=Return(%).
    """
    if df_intervals.empty:
        return None
    df_ = df_intervals.copy()
    df_["Interval"] = df_["Interval Start"].dt.strftime("%Y-%m-%d")

    fig = px.bar(
        df_,
        x="Interval",
        y="Return(%)",
        color="Portfolio",
        barmode="group",
        title="Interval Returns (Multi-Portfolio)",
        labels={"Interval":"Rebalance Interval","Return(%)":"Interval Return (%)"}
    )
    fig.update_layout(xaxis=dict(type="category"))
    return fig
