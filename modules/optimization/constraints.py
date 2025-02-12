#modules/optimization/constraints

import streamlit as st
import pandas as pd

def get_main_constraints(df_instruments: pd.DataFrame, df_prices: pd.DataFrame):
    """
    A helper function that displays/collects the main constraints for both
    Manual Single Rolling and Grid Search approaches. Returns a dictionary
    of the user-selected (or default) parameters.

    New in this version:
    - We allow 'min_instrument_weight' in both "custom" and "keep_current" modes,
      the same way we already have a 'max_instrument_weight'.
    """

    # 1) Basic date logic
    earliest = df_prices.index.min()
    earliest_valid = earliest + pd.Timedelta(days=365)
    user_start = st.date_input(
        "Start Date",
        min_value=earliest,
        value=earliest_valid,
        max_value=df_prices.index.max()
    )
    if pd.Timestamp(user_start) < earliest_valid:
        st.error("Start date must be at least 1 year after the first date.")
        st.stop()

    # 2) Constraint mode
    constraint_mode = st.selectbox("Constraint Mode", ["custom", "keep_current"])
    user_custom_constraints = {}
    buffer_pct = 0.0
    all_classes = df_instruments["#Asset"].unique()

    # 3) Collect constraints
    if constraint_mode == "custom":
        st.write("Enter class constraints (min/max class weight, min/max instrument weight).")
        for cl in all_classes:
            c1, c2, c3, c4 = st.columns(4)
            with c1:
                min_val = st.number_input(f"{cl} min class %", 0.0, 100.0, 0.0, step=5.0, key=f"{cl}_min_class")
            with c2:
                max_val = st.number_input(f"{cl} max class %", 0.0, 100.0, 100.0, step=5.0, key=f"{cl}_max_class")
            with c3:
                min_inst = st.number_input(f"{cl} min inst%", 0.0, 100.0, 0.0, step=5.0, key=f"{cl}_min_inst")
            with c4:
                max_inst = st.number_input(f"{cl} max inst%", 0.0, 100.0, 100.0, step=5.0, key=f"{cl}_max_inst")

            user_custom_constraints[cl] = {
                "min_class_weight": min_val / 100.0,
                "max_class_weight": max_val / 100.0,
                "min_instrument_weight": min_inst / 100.0,
                "max_instrument_weight": max_inst / 100.0
            }

    else:
        # "keep_current" style => user can set a buffer for class weights,
        # plus min/max instrument weights if desired
        buff_in = st.number_input("Buffer (%) around old class weights", 0.0, 100.0, 5.0)
        buffer_pct = buff_in / 100.0
        st.write("For each class, define min/max instrument weight if you like.")
        for cl in all_classes:
            c1, c2 = st.columns(2)
            with c1:
                min_inst = st.number_input(f"{cl} min inst%", 0.0, 100.0, 0.0, step=5.0, key=f"{cl}_min_inst_keep")
            with c2:
                max_inst = st.number_input(f"{cl} max inst%", 0.0, 100.0, 100.0, step=5.0, key=f"{cl}_max_inst_keep")

            user_custom_constraints[cl] = {
                # We'll fill these in dynamically in your "build_final_class_constraints"
                # But we do have the instrument constraints here:
                "min_instrument_weight": min_inst / 100.0,
                "max_instrument_weight": max_inst / 100.0
            }

    # 4) Daily RF input
    rf_in = st.number_input("Daily RF (%)", 0.0, 100.0, 0.0)
    daily_rf = rf_in / 100.0

    # 5) Transaction cost details
    cost_type = st.selectbox("Transaction Cost Type", ["percentage", "ticket_fee"], index=0)
    if cost_type == "percentage":
        cost_in = st.number_input("Cost (%)", 0.0, 100.0, 1.0, step=0.5)
        transaction_cost_value = cost_in / 100.0
    else:
        transaction_cost_value = st.number_input("Ticket Fee", 0.0, 1e9, 10.0, step=10.0)

    # 6) Trade buffer
    tb_in = st.number_input("Trade Buffer (%)", 0.0, 50.0, 1.0)
    trade_buffer_pct = tb_in / 100.0

    return {
        "user_start": user_start,
        "constraint_mode": constraint_mode,
        "buffer_pct": buffer_pct,
        "user_custom_constraints": user_custom_constraints,
        "daily_rf": daily_rf,
        "cost_type": cost_type,
        "transaction_cost_value": transaction_cost_value,
        "trade_buffer_pct": trade_buffer_pct
    }
