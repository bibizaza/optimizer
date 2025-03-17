# File: modules/analytics/constraints.py

import streamlit as st
import pandas as pd
import numpy as np

def get_main_constraints(df_instruments: pd.DataFrame, df_prices: pd.DataFrame) -> dict:
    """
    We ensure keep_current => class_sum_constraints for each asset class is
    old_weight ± buffer. This must include the 'Cash' class if #Asset == 'Cash'.
    """
    st.sidebar.markdown("### Constraints Configuration")

    earliest = df_prices.index.min()
    latest   = df_prices.index.max()
    if pd.isna(earliest) or pd.isna(latest):
        earliest = pd.Timestamp("2000-01-01")
        latest   = pd.Timestamp("2025-01-01")

    earliest_valid = earliest + pd.Timedelta(days=365)
    if earliest_valid > latest:
        earliest_valid = earliest

    user_start_date = st.sidebar.date_input(
        "Start Date",
        min_value=earliest.date(),
        value=min(earliest_valid, latest).date(),
        max_value=latest.date()
    )
    user_start = pd.Timestamp(user_start_date)

    constraint_mode = st.sidebar.selectbox("Constraint Mode", ["custom", "keep_current"], index=0)
    buffer_pct = 0.0
    class_sum_constraints = {}
    subtype_constraints   = {}

    st.sidebar.write(f"Mode: {constraint_mode}")
    have_sec_type = ("#Security_Type" in df_instruments.columns)
    all_classes   = df_instruments["#Asset_Class"].unique()

    # Make sure we compute Weight_Old if needed
    if "#Quantity" in df_instruments.columns and "#Last_Price" in df_instruments.columns:
        df_instruments["Value"] = df_instruments["#Quantity"] * df_instruments["#Last_Price"]
        tot_val = df_instruments["Value"].sum()
        if tot_val<=0:
            tot_val=1.0
        df_instruments["Weight_Old"] = df_instruments["Value"]/ tot_val
    else:
        df_instruments["Weight_Old"] = 0.0

    if constraint_mode == "custom":
        st.sidebar.info("Custom: specify class-level min/max + subtype constraints.")
        for cl in all_classes:
            with st.sidebar.expander(f"Asset Class: {cl}", expanded=False):
                st.markdown("**Class-Level Constraints**")
                c1, c2 = st.columns(2)
                with c1:
                    mn_cls = st.number_input(f"Min class sum (%) for {cl}", 0.0, 100.0, 0.0, step=5.0)
                with c2:
                    mx_cls = st.number_input(f"Max class sum (%) for {cl}", 0.0, 100.0, 100.0, step=5.0)
                class_sum_constraints[cl] = {
                    "min_class_weight": mn_cls/100.0,
                    "max_class_weight": mx_cls/100.0
                }
                # Subtype
                if have_sec_type:
                    df_cl = df_instruments[df_instruments["#Asset_Class"]==cl]
                    st.markdown("**Per-Instrument (Security-Type) Constraints**")
                    stypes = df_cl["#Security_Type"].dropna().unique()
                    for stp in stypes:
                        st.write(f"{cl} - {stp}")
                        cA, cB = st.columns(2)
                        with cA:
                            min_inst = st.number_input(f"Min inst (%) for {cl}-{stp}", 0.0, 100.0, 0.0, step=1.0)
                        with cB:
                            max_inst = st.number_input(f"Max inst (%) for {cl}-{stp}", 0.0, 100.0, 10.0, step=1.0)
                        subtype_constraints[(cl, stp)] = {
                            "min_instrument": min_inst/100.0,
                            "max_instrument": max_inst/100.0
                        }
    else:
        st.sidebar.info("Keep Current: class weights = old +/- buffer. Subtype optional.")
        buff_in = st.sidebar.number_input("Buffer (%) around old class weight", 0.0, 100.0, 5.0, step=1.0)
        buffer_pct = buff_in/100.0

        class_old_w = df_instruments.groupby("#Asset_Class")["Weight_Old"].sum()
        for cl in all_classes:
            oldw = class_old_w.get(cl, 0.0)
            mn = max(0.0, oldw - buffer_pct)
            mx = min(1.0, oldw + buffer_pct)
            class_sum_constraints[cl] = {
                "min_class_weight": mn,
                "max_class_weight": mx
            }
            with st.sidebar.expander(f"Asset Class: {cl} (keep_current)", expanded=False):
                st.write(f"Old weight = {oldw*100:.2f}%, buffer = ±{buffer_pct*100:.2f}% => range = [{mn*100:.2f}..{mx*100:.2f}]%")
                if have_sec_type:
                    df_cl = df_instruments[df_instruments["#Asset_Class"]==cl]
                    st.markdown("**Security-Type Constraints**")
                    stypes = df_cl["#Security_Type"].dropna().unique()
                    for stp in stypes:
                        st.write(f"{cl} - {stp}")
                        cA, cB = st.columns(2)
                        with cA:
                            min_inst = st.number_input(f"Min inst (%) for {cl}-{stp}", 0.0, 100.0, 0.0, step=1.0)
                        with cB:
                            max_inst = st.number_input(f"Max inst (%) for {cl}-{stp}", 0.0, 100.0, 10.0, step=1.0)
                        subtype_constraints[(cl, stp)] = {
                            "min_instrument": min_inst/100.0,
                            "max_instrument": max_inst/100.0
                        }

    daily_rf = st.sidebar.number_input("Daily RF (%)", 0.0, 100.0, 0.0, 0.5)/100.0

    cost_type = st.sidebar.selectbox("Transaction Cost Type", ["percentage","ticket_fee"], index=0)
    if cost_type=="percentage":
        cost_val = st.sidebar.number_input("Cost (%)", 0.0, 100.0, 1.0, step=0.5)/100.0
        transaction_cost_value = cost_val
    else:
        transaction_cost_value = st.sidebar.number_input("Ticket Fee", 0.0, 1e9, 10.0, step=10.0)

    tb_in = st.sidebar.number_input("Trade Buffer (%)", 0.0, 50.0, 1.0, step=1.0)
    trade_buffer_pct = tb_in/100.0

    return {
        "user_start": user_start,
        "constraint_mode": constraint_mode,
        "buffer_pct": buffer_pct,
        "class_sum_constraints": class_sum_constraints,
        "subtype_constraints": subtype_constraints,
        "daily_rf": daily_rf,
        "cost_type": cost_type,
        "transaction_cost_value": transaction_cost_value,
        "trade_buffer_pct": trade_buffer_pct
    }