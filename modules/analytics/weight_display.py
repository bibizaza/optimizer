# File: modules/analytics/weight_display.py

import pandas as pd
import numpy as np
import streamlit as st
from collections import defaultdict

def display_three_portfolio_class_weights(
    df_instruments: pd.DataFrame,
    col_tickers: list[str],
    asset_classes: list[str],
    w_new: np.ndarray,
    w_drift: np.ndarray,
    w_strat: np.ndarray
):
    """
    Builds and displays a single table showing each asset class
    and the corresponding total weights in the three portfolios:
      - New Optimized  (w_new)
      - Old Drift      (w_drift)
      - Old Strategic  (w_strat)
    """
    n_tickers = len(col_tickers)
    if (len(w_new) != n_tickers
        or len(w_drift) != n_tickers
        or len(w_strat) != n_tickers
        or len(asset_classes) != n_tickers):
        st.error("Length mismatch in display_three_portfolio_class_weights.")
        return

    # Summation by asset class
    sums_map = defaultdict(lambda: [0.0, 0.0, 0.0])  # key=asset_class => [sum_new, sum_drift, sum_strat]
    for i, cls in enumerate(asset_classes):
        sums_map[cls][0] += w_new[i]
        sums_map[cls][1] += w_drift[i]
        sums_map[cls][2] += w_strat[i]

    rows = []
    for cls_name, (val_new, val_drift, val_strat) in sums_map.items():
        rows.append({
            "Asset Class":     cls_name,
            "New Optimized":   val_new,
            "Old Drift":       val_drift,
            "Old Strategic":   val_strat
        })

    # Sort by class name
    rows.sort(key=lambda x: x["Asset Class"])

    df_cls = pd.DataFrame(rows)
    df_cls.set_index("Asset Class", inplace=True)

    st.write("### Asset-Class Weights (All Three Portfolios)")
    st.dataframe(df_cls.style.format("{:.2%}"))


def display_instrument_weights_two_portfolios(
    df_instruments: pd.DataFrame,
    col_tickers: list[str],
    w_port1: np.ndarray,
    w_port2: np.ndarray,
    label_port1: str,
    label_port2: str
):
    """
    Displays a table comparing instrument-level weights between two portfolios:
      - e.g. (New Optimized) vs. (Old Drift)
      or    (New Optimized) vs. (Old Strategic)

    Columns will be:
      Ticker | Name | Asset Class | [label_port1] | [label_port2] | Diff

    The 'Diff' column is color-coded: green if positive, red if negative.
    """
    if (len(w_port1) != len(col_tickers)) or (len(w_port2) != len(col_tickers)):
        st.error(f"Length mismatch in display_instrument_weights_two_portfolios for {label_port1} vs {label_port2}")
        return

    # Build quick lookups for instrument name, asset class, etc.
    name_map = {}
    asset_map = {}

    for _, row_ in df_instruments.iterrows():
        tkr = row_["#ID"]
        name_map[tkr] = row_.get("#Name", "")
        asset_map[tkr] = row_.get("#Asset_Class", "Unknown")

    rows = []
    for i, tkr in enumerate(col_tickers):
        val1 = w_port1[i]
        val2 = w_port2[i]
        diff = val1 - val2

        instr_name  = name_map.get(tkr, "")
        instr_asset = asset_map.get(tkr, "Unknown")

        rows.append({
            "Ticker":         tkr,
            "Name":           instr_name,
            "Asset Class":    instr_asset,
            label_port1:      val1,
            label_port2:      val2,
            "Diff":           diff
        })

    df_comp = pd.DataFrame(rows)

    def color_diff(val):
        if val > 0:   return "color:green"
        elif val < 0: return "color:red"
        else:         return "color:black"

    st.write(f"### Instrument Weights: {label_port1} vs. {label_port2}")
    st.dataframe(
        df_comp.style
        .applymap(color_diff, subset=["Diff"])
        .format({
            label_port1: "{:.2%}",
            label_port2: "{:.2%}",
            "Diff":      "{:.2%}"
        })
    )


def display_new_vs_old_instrument_tables(
    df_instruments: pd.DataFrame,
    col_tickers: list[str],
    w_new: np.ndarray,
    w_drift: np.ndarray,
    w_strat: np.ndarray
):
    """
    A convenience function that calls display_instrument_weights_two_portfolios
    twice:
      1) New Optimized vs Old Drift
      2) New Optimized vs Old Strategic
    """
    # 1) New vs. Old Drift
    display_instrument_weights_two_portfolios(
        df_instruments=df_instruments,
        col_tickers=col_tickers,
        w_port1=w_new,
        w_port2=w_drift,
        label_port1="New Optimized",
        label_port2="Old Drift"
    )

    # 2) New vs. Old Strategic
    display_instrument_weights_two_portfolios(
        df_instruments=df_instruments,
        col_tickers=col_tickers,
        w_port1=w_new,
        w_port2=w_strat,
        label_port1="New Optimized",
        label_port2="Old Strategic"
    )
