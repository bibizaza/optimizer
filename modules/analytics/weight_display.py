# modules/analytics/weight_display.py

import pandas as pd
import numpy as np
import streamlit as st

def display_instrument_weight_diff(
    df_instruments: pd.DataFrame,
    col_tickers: list[str],
    new_w: np.ndarray
):
    """
    Display a color-coded table of old vs new weights at the instrument level,
    with columns: Ticker, Name, Asset Class, Old W, New W, Diff.

    Requirements:
      - df_instruments must have:
         "#ID" => the ticker
         "#Name" => the instrument name
         "#Asset_Class" => the asset class
         "Weight_Old" => the old weight
      - col_tickers => list of tickers in the same order as new_w
      - new_w => shape=(len(col_tickers),) the new final weights
    """

    # 1) Build lookups (ticker => old_weight, name, asset_class)
    old_weight_map= {}
    name_map= {}
    asset_map= {}

    for _, row_ in df_instruments.iterrows():
        tkr= row_["#ID"]
        old_weight_map[tkr]= row_["Weight_Old"]
        # if your sheet has "#Name" => store it
        name_map[tkr] = row_.get("#Name","")
        # if your sheet has "#Asset" => store it
        asset_map[tkr] = row_.get("#Asset_Class","Unknown")

    # 2) Build a table (DataFrame) row by row
    rows= []
    for i, tkr in enumerate(col_tickers):
        old_w= old_weight_map.get(tkr,0.0)
        new_wi= new_w[i]
        diff= new_wi - old_w

        instr_name= name_map.get(tkr,"")
        instr_asset= asset_map.get(tkr,"Unknown")

        rows.append([
            tkr,
            instr_name,
            instr_asset,
            old_w,
            new_wi,
            diff
        ])

    df_instr= pd.DataFrame(
        rows, 
        columns=["Ticker","Name","Asset Class","Old W","New W","Diff"]
    )

    # 3) Color-coded style for the "Diff" column
    def color_diff(val):
        if val>0: return "color:green"
        elif val<0: return "color:red"
        else: return "color:black"

    st.write("### Instrument-Level Weight Changes")
    st.dataframe(
        df_instr.style
        .applymap(color_diff, subset=["Diff"])
        .format({"Old W":"{:.2%}","New W":"{:.2%}","Diff":"{:.2%}"})
    )


def display_class_weight_diff(
    df_instruments: pd.DataFrame,
    col_tickers: list[str],
    asset_classes: list[str],
    new_w: np.ndarray
):
    """
    Display a color-coded table of old vs new asset-class weights, 
    difference in red/green. Sums old/new by 'Asset Class'.
    Requirements:
      - df_instruments => must have "#ID" => the ticker, "Weight_Old" => old weight
      - col_tickers => same order as new_w
      - asset_classes => same order as col_tickers
      - new_w => final new weights
    """
    from collections import defaultdict

    old_weight_map= {}
    for _, row_ in df_instruments.iterrows():
        tkr= row_["#ID"]
        old_weight_map[tkr]= row_["Weight_Old"]

    sum_old= defaultdict(float)
    sum_new= defaultdict(float)

    for i, tkr in enumerate(col_tickers):
        cl= asset_classes[i]
        w_old= old_weight_map.get(tkr, 0.0)
        w_new= new_w[i]
        sum_old[cl]+= w_old
        sum_new[cl]+= w_new

    # build final table => one row per class
    all_cls= sorted(set(asset_classes))
    rows= []
    for cl in all_cls:
        o= sum_old.get(cl,0.0)
        n= sum_new.get(cl,0.0)
        diff= n - o
        rows.append([cl, o, n, diff])

    df_cls= pd.DataFrame(rows, columns=["Asset Class","Old W","New W","Diff"])

    def color_diff(val):
        if val>0: return "color:green"
        elif val<0: return "color:red"
        else: return "color:black"

    st.write("### Asset-Class Weight Changes")
    st.dataframe(
        df_cls.style
        .applymap(color_diff, subset=["Diff"])
        .format({"Old W":"{:.2%}","New W":"{:.2%}","Diff":"{:.2%}"})
    )