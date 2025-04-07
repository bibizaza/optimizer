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

    Requirements:
      - df_instruments has a row per ticker, with:
          "#ID" => ticker
          "Weight_Old" => old weighting (not used here, just for reference if needed)
      - col_tickers => list of ticker names in the same order as w_new, w_drift, w_strat
      - asset_classes => same length as col_tickers, each element is the asset class of that ticker
      - w_new, w_drift, w_strat => final weight vectors for each portfolio
        (all shape=(len(col_tickers),))

    We sum weights by asset class and display them side-by-side as percentages.
    """

    # 0) Basic checks
    n_tickers = len(col_tickers)
    if (len(w_new) != n_tickers or
        len(w_drift) != n_tickers or
        len(w_strat) != n_tickers or
        len(asset_classes) != n_tickers):
        st.error("Length mismatch in display_three_portfolio_class_weights.")
        return

    # 1) Summation by asset class
    sums_map = defaultdict(lambda: [0.0, 0.0, 0.0])  # key=class => [sum_new, sum_drift, sum_strat]
    for i, cls in enumerate(asset_classes):
        sums_map[cls][0] += w_new[i]
        sums_map[cls][1] += w_drift[i]
        sums_map[cls][2] += w_strat[i]

    # 2) Build rows => one row per unique asset class
    rows = []
    for cls_name, (val_new, val_drift, val_strat) in sums_map.items():
        rows.append({
            "Asset Class":    cls_name,
            "New Optimized":  val_new,
            "Old Drift":      val_drift,
            "Old Strategic":  val_strat
        })

    # sort by asset class name
    rows.sort(key=lambda x: x["Asset Class"])

    df_cls = pd.DataFrame(rows)
    df_cls.set_index("Asset Class", inplace=True)

    # 3) Display in Streamlit => percent columns
    st.write("### Asset-Class Weights (All Three Portfolios)")
    st.dataframe(
        df_cls.style.format("{:.2%}")
    )
