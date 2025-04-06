# File: modules/backtesting/rolling_intervals.py

import streamlit as st
import pandas as pd
import plotly.express as px

def get_interval_returns_for_line(
    sr_line: pd.Series,
    rebal_dates: list[pd.Timestamp],
    portfolio_label: str = "Portfolio"
) -> pd.DataFrame:
    """
    For a single timeseries 'sr_line', compute the interval returns between consecutive
    rebal_dates. Return a DataFrame with columns:
      ["Interval Start","Interval End","Portfolio","Return(%)"].
    """
    rows = []
    sorted_dates = sorted(list(set(rebal_dates)))
    for i in range(len(sorted_dates) - 1):
        start_d = sorted_dates[i]
        end_d   = sorted_dates[i + 1]

        sub = sr_line.loc[start_d:end_d]
        if len(sub) < 2:
            continue

        ret_ = sub.iloc[-1] / sub.iloc[0] - 1
        rows.append({
            "Interval Start": start_d,
            "Interval End":   end_d,
            "Portfolio":      portfolio_label,
            "Return(%)":      ret_ * 100.0
        })

    return pd.DataFrame(rows)


def compute_diff_stats_between_two(
    sr_a: pd.Series,
    sr_b: pd.Series,
    rebal_dates: list[pd.Timestamp],
    label_a: str,
    label_b: str
) -> dict:
    """
    Compute the interval differences (A minus B) for each interval, then
    build a dict with Win Rate, Average Diff, etc.
    We'll store these in one row of a final "comparison" table.
    """
    sorted_dates = sorted(list(set(rebal_dates)))
    diffs = []
    for i in range(len(sorted_dates) - 1):
        start_d = sorted_dates[i]
        end_d   = sorted_dates[i + 1]

        sub_a = sr_a.loc[start_d:end_d]
        sub_b = sr_b.loc[start_d:end_d]
        if len(sub_a) < 2 or len(sub_b) < 2:
            continue

        ret_a = sub_a.iloc[-1] / sub_a.iloc[0] - 1
        ret_b = sub_b.iloc[-1] / sub_b.iloc[0] - 1
        diffs.append(ret_a - ret_b)

    if len(diffs) == 0:
        return {
            "Comparison": f"{label_a} vs {label_b}",
            "Intervals": 0,
            "Win Rate(%)": 0.00,
            "Avg Diff(%)": 0.00,
            "Median Diff(%)": 0.00,
            "Max Diff(%)": 0.00,
            "Min Diff(%)": 0.00
        }

    diffs_series = pd.Series(diffs)
    n = len(diffs_series)
    # "Win Rate" => how often A minus B is > 0
    wins = diffs_series[diffs_series > 0].count()
    win_rate = (wins / n) * 100.0

    stats_row = {
        "Comparison":       f"{label_a} vs {label_b}",
        "Intervals":        n,
        "Win Rate(%)":      round(win_rate, 2),
        "Avg Diff(%)":      round(diffs_series.mean() * 100.0, 2),
        "Median Diff(%)":   round(diffs_series.median() * 100.0, 2),
        "Max Diff(%)":      round(diffs_series.max() * 100.0, 2),
        "Min Diff(%)":      round(diffs_series.min() * 100.0, 2)
    }
    return stats_row


def display_interval_bars_and_stats(
    sr_new: pd.Series,
    sr_drift: pd.Series,
    sr_strat: pd.Series,
    c_new: bool,
    c_drift: bool,
    c_strat: bool,
    rebal_dates: list[pd.Timestamp],
    label_new="New",
    label_drift="Old Drift",
    label_strat="Old Strategic",
    color_new="#1f77b4",
    color_drift="grey",
    color_strat="lightblue"
):
    """
    Multi-portfolio interval analysis:
      1) For each *checked* portfolio, compute interval returns => single grouped bar chart
      2) Build a table of difference stats for each pair that is selected:
         - e.g. (New vs Drift), (New vs Strat), (Drift vs Strat).
         - Only show pairs that both exist.

    'rebal_dates' => the list of interval boundaries
    """

    # 1) Build a dict for whichever lines are visible.
    lines = {}
    colors = {}

    if c_new and not sr_new.empty:
        lines[label_new] = sr_new
        colors[label_new] = color_new
    if c_drift and not sr_drift.empty:
        lines[label_drift] = sr_drift
        colors[label_drift] = color_drift
    if c_strat and not sr_strat.empty:
        lines[label_strat] = sr_strat
        colors[label_strat] = color_strat

    if len(lines) < 1:
        st.info("No portfolios selected => no interval performance.")
        return

    # 2) Compute interval returns for each selected portfolio
    all_rows = []
    for lbl, sr_ in lines.items():
        df_ = get_interval_returns_for_line(sr_, rebal_dates, portfolio_label=lbl)
        all_rows.append(df_)

    if not all_rows:
        st.info("No intervals to compute (empty).")
        return

    df_all = pd.concat(all_rows, ignore_index=True)

    # 3) Plot a grouped bar => "Interval Start" on X, "Return(%)" on Y, color="Portfolio"
    df_all["Interval"] = df_all["Interval Start"].astype(str)
    fig = px.bar(
        df_all,
        x="Interval",
        y="Return(%)",
        color="Portfolio",
        barmode="group",
        labels={
            "Interval": "Rebalance Interval",
            "Return(%)": "Interval Return (%)"
        },
        title="Interval Returns (Grouped)",
    )
    # optionally fix the color mapping
    fig.update_layout(
        xaxis=dict(type="category"),
        legend_title_text="Portfolio"
    )
    # apply custom colors if desired
    color_map = {}
    for lbl in lines.keys():
        color_map[lbl] = colors.get(lbl, None)  # default None => px auto
    fig.for_each_trace(
        lambda t: t.update(marker_color=color_map.get(t.name, t.marker.color))
    )

    st.plotly_chart(fig)

    # 4) Now build the difference table (3 lines => new vs drift, new vs strat, drift vs strat)
    #    only for pairs that exist in 'lines'
    comparison_rows = []
    if label_new in lines and label_drift in lines:
        row_ = compute_diff_stats_between_two(
            sr_a=lines[label_new],
            sr_b=lines[label_drift],
            rebal_dates=rebal_dates,
            label_a=label_new,
            label_b=label_drift
        )
        comparison_rows.append(row_)

    if label_new in lines and label_strat in lines:
        row_ = compute_diff_stats_between_two(
            sr_a=lines[label_new],
            sr_b=lines[label_strat],
            rebal_dates=rebal_dates,
            label_a=label_new,
            label_b=label_strat
        )
        comparison_rows.append(row_)

    if label_drift in lines and label_strat in lines:
        row_ = compute_diff_stats_between_two(
            sr_a=lines[label_drift],
            sr_b=lines[label_strat],
            rebal_dates=rebal_dates,
            label_a=label_drift,
            label_b=label_strat
        )
        comparison_rows.append(row_)

    if comparison_rows:
        df_cmp = pd.DataFrame(comparison_rows)
        st.write("### Interval Comparison Stats (Pairwise)")
        st.dataframe(df_cmp)
    else:
        st.write("No pairwise comparisons to show (fewer than 2 portfolios).")