# modules/analytics/constraints.py
import streamlit as st
import pandas as pd

def get_main_constraints(df_instruments: pd.DataFrame, df_prices: pd.DataFrame) -> dict:
    """
    Displays the UI to collect constraint parameters.
    
    Returns a dictionary with:
      - user_start: the chosen start date (Timestamp)
      - constraint_mode: "custom" or "keep_current"
      - buffer_pct: (for keep_current) as a decimal
      - class_sum_constraints: a dictionary mapping each asset class to a dict with 
            "min_class_weight" and "max_class_weight" (in custom mode only)
      - subtype_constraints: a dictionary mapping (asset_class, security_type) to a dict with
            "min_instrument" and "max_instrument" (applied individually to each instrument)
      - daily_rf: daily risk‑free rate (decimal)
      - cost_type: "percentage" or "ticket_fee"
      - transaction_cost_value: the cost value (decimal)
      - trade_buffer_pct: trade buffer (as a decimal)
    """
    # Optionally inject some CSS to style the expanders (you can adjust colors as needed)
    st.markdown(
        """
        <style>
        div[data-testid="stExpander"] > div > div[role="button"] {
            background-color: #e6f7ff;
            color: #003366;
            font-weight: bold;
            padding: 5px;
            border-radius: 5px;
        }
        </style>
        """,
        unsafe_allow_html=True
    )
    
    # Date selection
    earliest = df_prices.index.min()
    latest = df_prices.index.max()
    earliest_valid = earliest + pd.Timedelta(days=365)
    user_start_date = st.date_input(
        "Start Date",
        min_value=earliest.date(),
        value=min(earliest_valid, latest).date(),
        max_value=latest.date()
    )
    user_start = pd.Timestamp(user_start_date)
    if user_start < earliest_valid:
        st.error("Start date must be at least 1 year after the first date.")
        st.stop()
    
    # Constraint mode selection
    constraint_mode = st.selectbox("Constraint Mode", ["custom", "keep_current"], index=0)
    st.write(f"Selected Constraint Mode: {constraint_mode}")
    
    have_sec_type = ("#Security_Type" in df_instruments.columns)
    all_classes = df_instruments["#Asset"].unique()
    
    class_sum_constraints = {}
    subtype_constraints = {}
    buffer_pct = 0.0
    
    if constraint_mode == "custom":
        st.info("Custom: Specify class‑level min/max and per‑instrument (security‑type) constraints.")
        for cl in all_classes:
            with st.expander(f"Asset Class: {cl}", expanded=False):
                st.markdown("**Class‑Level Constraints**")
                col1, col2 = st.columns(2)
                with col1:
                    mn_cls = st.number_input(f"MIN class sum (%)", 0.0, 100.0, 0.0, step=5.0, key=f"{cl}_min_class")
                with col2:
                    mx_cls = st.number_input(f"MAX class sum (%)", 0.0, 100.0, 100.0, step=5.0, key=f"{cl}_max_class")
                class_sum_constraints[cl] = {
                    "min_class_weight": mn_cls / 100.0,
                    "max_class_weight": mx_cls / 100.0
                }
                if have_sec_type:
                    df_cl = df_instruments[df_instruments["#Asset"] == cl]
                    st.markdown("**Per‑Instrument (Security‑Type) Constraints**")
                    stypes = df_cl["#Security_Type"].dropna().unique()
                    if len(stypes) > 0:
                        for stp in stypes:
                            st.write(f"Security Type: **{stp}**")
                            colA, colB = st.columns(2)
                            with colA:
                                min_inst = st.number_input(f"MIN inst (%)", 0.0, 100.0, 0.0, step=1.0, key=f"{cl}_{stp}_min")
                            with colB:
                                max_inst = st.number_input(f"MAX inst (%)", 0.0, 100.0, 10.0, step=1.0, key=f"{cl}_{stp}_max")
                            subtype_constraints[(cl, stp)] = {
                                "min_instrument": min_inst / 100.0,
                                "max_instrument": max_inst / 100.0
                            }
                    else:
                        st.caption("No sub‑types found for this asset class.")
                else:
                    st.caption("No #Security_Type column; skipping per‑instrument constraints.")
    else:
        st.info("Keep Current: Class weights are computed automatically from the current allocation ± buffer. Specify only per‑instrument constraints.")
        buff_in = st.number_input("Buffer (%) around old class weight", 0.0, 100.0, 5.0, key="keep_buffer")
        buffer_pct = buff_in / 100.0
        for cl in all_classes:
            with st.expander(f"Asset Class: {cl} (Keep Current)", expanded=False):
                st.markdown("Class‑level weights will be determined automatically from the current portfolio allocation and the buffer.")
                if have_sec_type:
                    df_cl = df_instruments[df_instruments["#Asset"] == cl]
                    st.markdown("**Per‑Instrument (Security‑Type) Constraints**")
                    stypes = df_cl["#Security_Type"].dropna().unique()
                    if len(stypes) > 0:
                        for stp in stypes:
                            st.write(f"Security Type: **{stp}**")
                            colA, colB = st.columns(2)
                            with colA:
                                min_inst = st.number_input(f"MIN inst (%)", 0.0, 100.0, 0.0, step=1.0, key=f"{cl}_{stp}_min_keep")
                            with colB:
                                max_inst = st.number_input(f"MAX inst (%)", 0.0, 100.0, 10.0, step=1.0, key=f"{cl}_{stp}_max_keep")
                            subtype_constraints[(cl, stp)] = {
                                "min_instrument": min_inst / 100.0,
                                "max_instrument": max_inst / 100.0
                            }
                    else:
                        st.caption(f"No sub‑types found for asset class {cl}.")
                else:
                    st.caption(f"No #Security_Type column for asset class {cl}; skipping per‑instrument constraints.")
            # In keep_current mode, class‑level constraints are not user-specified.
    
    daily_rf = st.number_input("Daily RF (%)", 0.0, 100.0, 0.0) / 100.0

    cost_type = st.selectbox("Transaction Cost Type", ["percentage", "ticket_fee"], index=0)
    if cost_type == "percentage":
        cost_val = st.number_input("Cost (%)", 0.0, 100.0, 1.0, step=0.5) / 100.0
        transaction_cost_value = cost_val
    else:
        transaction_cost_value = st.number_input("Ticket Fee", 0.0, 1e9, 10.0, step=10.0)

    tb_in = st.number_input("Trade Buffer (%)", 0.0, 50.0, 1.0)
    trade_buffer_pct = tb_in / 100.0

    return {
        "user_start": pd.Timestamp(user_start),
        "constraint_mode": constraint_mode,
        "buffer_pct": buffer_pct,
        "class_sum_constraints": class_sum_constraints,
        "subtype_constraints": subtype_constraints,
        "daily_rf": daily_rf,
        "cost_type": cost_type,
        "transaction_cost_value": transaction_cost_value,
        "trade_buffer_pct": trade_buffer_pct
    }
