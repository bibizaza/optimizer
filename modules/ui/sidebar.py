# File: modules/ui/sidebar.py

import streamlit as st
import pandas as pd
import numpy as np

# This is your constraints code, but we forcibly adapt it
# so it uses st.sidebar.* calls, ensuring it appears on the sidebar
def get_main_constraints_sidebar(df_instruments: pd.DataFrame, df_prices: pd.DataFrame) -> dict:
    """
    The same logic as your get_main_constraints, but now
    forcibly using st.sidebar for the UI.
    """
    # We do the earliest, latest from df_prices
    earliest = df_prices.index.min()
    latest   = df_prices.index.max()

    # If the user has less than 2 rows, or earliest is NaN => fallback
    if pd.isna(earliest) or pd.isna(latest):
        earliest = pd.Timestamp("2000-01-01")
        latest   = pd.Timestamp("2025-01-01")

    earliest_valid = earliest + pd.Timedelta(days=365)
    if earliest_valid > latest:
        earliest_valid = earliest  # fallback

    user_start_date = st.sidebar.date_input(
        "Start Date",
        min_value=earliest.date(),
        value=min(earliest_valid, latest).date(),
        max_value=latest.date()
    )
    user_start = pd.Timestamp(user_start_date)

    # Constraint mode
    constraint_mode = st.sidebar.selectbox("Constraint Mode", ["custom", "keep_current"], index=0)
    st.sidebar.write(f"Selected Constraint Mode: {constraint_mode}")

    have_sec_type = ("#Security_Type" in df_instruments.columns)
    all_classes = df_instruments["#Asset_Class"].unique()

    class_sum_constraints = {}
    subtype_constraints = {}
    buffer_pct = 0.0

    if constraint_mode == "custom":
        st.sidebar.info("Custom: specify class-level min/max + subtype constraints.")
        for cl in all_classes:
            with st.sidebar.expander(f"Asset Class: {cl}", expanded=False):
                st.markdown("**Class-Level Constraints**")
                col1, col2 = st.columns(2)
                with col1:
                    mn_cls = st.number_input(f"Min class sum (%) for {cl}", 0.0, 100.0, 0.0, step=5.0)
                with col2:
                    mx_cls = st.number_input(f"Max class sum (%) for {cl}", 0.0, 100.0, 100.0, step=5.0)
                class_sum_constraints[cl] = {
                    "min_class_weight": mn_cls / 100.0,
                    "max_class_weight": mx_cls / 100.0
                }
                if have_sec_type:
                    df_cl = df_instruments[df_instruments["#Asset_Class"] == cl]
                    st.markdown("**Per-Instrument (Security-Type) Constraints**")
                    stypes = df_cl["#Security_Type"].dropna().unique()
                    if len(stypes) > 0:
                        for stp in stypes:
                            st.write(f"Security Type: **{stp}**")
                            colA, colB = st.columns(2)
                            with colA:
                                min_inst = st.number_input(f"Min inst (%) for {cl}-{stp}", 0.0, 100.0, 0.0, step=1.0)
                            with colB:
                                max_inst = st.number_input(f"Max inst (%) for {cl}-{stp}", 0.0, 100.0, 10.0, step=1.0)
                            subtype_constraints[(cl, stp)] = {
                                "min_instrument": min_inst / 100.0,
                                "max_instrument": max_inst / 100.0
                            }
                    else:
                        st.caption("No sub-types found for this asset class.")
    else:
        st.sidebar.info("Keep Current: class weights = old +/- buffer. Only security-type constraints optional.")
        buff_in = st.sidebar.number_input("Buffer (%) around old class weight", 0.0, 100.0, 5.0)
        buffer_pct = buff_in / 100.0
        for cl in all_classes:
            with st.sidebar.expander(f"Asset Class: {cl} (keep_current)", expanded=False):
                st.markdown("Class-level weights auto from old portfolio +/- buffer.")
                if have_sec_type:
                    df_cl = df_instruments[df_instruments["#Asset_Class"] == cl]
                    st.markdown("**Security-Type Constraints**")
                    stypes = df_cl["#Security_Type"].dropna().unique()
                    if len(stypes) > 0:
                        for stp in stypes:
                            st.write(f"Security Type: **{stp}**")
                            colA, colB = st.columns(2)
                            with colA:
                                min_inst = st.number_input(f"Min inst (%) for {cl}-{stp}", 0.0, 100.0, 0.0, step=1.0)
                            with colB:
                                max_inst = st.number_input(f"Max inst (%) for {cl}-{stp}", 0.0, 100.0, 10.0, step=1.0)
                            subtype_constraints[(cl, stp)] = {
                                "min_instrument": min_inst/100.0,
                                "max_instrument": max_inst/100.0
                            }

    # daily_rf
    daily_rf_input = st.sidebar.number_input("Daily RF (%)", 0.0, 100.0, 0.0, 0.5)/100.0

    # cost type
    cost_type = st.sidebar.selectbox("Transaction Cost Type", ["percentage", "ticket_fee"], index=0)
    if cost_type == "percentage":
        cost_val = st.sidebar.number_input("Cost (%)", 0.0, 100.0, 1.0, step=0.5)/100.0
        transaction_cost_value = cost_val
    else:
        transaction_cost_value = st.sidebar.number_input("Ticket Fee", 0.0, 1e9, 10.0, step=10.0)

    # trade buffer
    tb_in = st.sidebar.number_input("Trade Buffer (%)", 0.0, 50.0, 1.0)
    trade_buffer_pct = tb_in/100.0

    return {
        "user_start": user_start,
        "constraint_mode": constraint_mode,
        "buffer_pct": buffer_pct,
        "class_sum_constraints": class_sum_constraints,
        "subtype_constraints": subtype_constraints,
        "daily_rf": daily_rf_input,
        "cost_type": cost_type,
        "transaction_cost_value": transaction_cost_value,
        "trade_buffer_pct": trade_buffer_pct
    }


def get_sidebar_inputs() -> dict:
    """
    In the sidebar:
      1) Data approach + file upload
      2) If data loaded => coverage + constraints
    Returns a dict with data + constraints.
    """
    st.sidebar.title("Data Loading")

    approach_data = st.sidebar.radio(
        "Data Source Approach",
        ["One-time Convert Excel->Parquet", "Use Excel for Analysis", "Use Parquet for Analysis"],
        index=1
    )

    df_instruments = pd.DataFrame()
    df_prices = pd.DataFrame()

    coverage = 0.0
    # defaults for constraints
    user_start = pd.Timestamp("2000-01-01")
    constraint_mode = "custom"
    buffer_pct = 0.0
    class_sum_constraints = {}
    subtype_constraints = {}
    daily_rf = 0.0
    cost_type = "percentage"
    transaction_cost_value = 0.0
    trade_buffer_pct = 0.0

    if approach_data == "One-time Convert Excel->Parquet":
        st.sidebar.info("Converter not implemented.")
        return {
            "approach_data": approach_data,
            "df_instruments": df_instruments,
            "df_prices": df_prices,
            "coverage": coverage,
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

    elif approach_data == "Use Excel for Analysis":
        excel_file = st.sidebar.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
        if excel_file:
            # parse
            df_instruments, df_prices = parse_excel(excel_file)
            if not df_prices.empty:
                # now show coverage + constraints
                st.sidebar.markdown("---")
                st.sidebar.subheader("Coverage & Constraints")
                coverage = st.sidebar.slider("Min coverage fraction", 0.0,1.0,0.8,0.05)

                # call get_main_constraints_sidebar
                constr_dict = get_main_constraints_sidebar(df_instruments, df_prices)
                user_start             = constr_dict["user_start"]
                constraint_mode        = constr_dict["constraint_mode"]
                buffer_pct             = constr_dict["buffer_pct"]
                class_sum_constraints  = constr_dict["class_sum_constraints"]
                subtype_constraints    = constr_dict["subtype_constraints"]
                daily_rf               = constr_dict["daily_rf"]
                cost_type              = constr_dict["cost_type"]
                transaction_cost_value = constr_dict["transaction_cost_value"]
                trade_buffer_pct       = constr_dict["trade_buffer_pct"]
            else:
                st.sidebar.warning("Excel file loaded but no valid price data found.")
        else:
            st.sidebar.info("Waiting for Excel...")

    else:
        # "Use Parquet for Analysis"
        fi = st.sidebar.file_uploader("Upload instruments.parquet", type=["parquet"], key="inst_parquet")
        fp = st.sidebar.file_uploader("Upload prices.parquet", type=["parquet"], key="prices_parquet")
        if fi and fp:
            df_instruments = pd.read_parquet(fi)
            df_prices      = pd.read_parquet(fp)
            if not df_prices.empty:
                st.sidebar.markdown("---")
                st.sidebar.subheader("Coverage & Constraints")
                coverage = st.sidebar.slider("Min coverage fraction", 0.0,1.0,0.8,0.05)

                # call get_main_constraints_sidebar
                constr_dict = get_main_constraints_sidebar(df_instruments, df_prices)
                user_start             = constr_dict["user_start"]
                constraint_mode        = constr_dict["constraint_mode"]
                buffer_pct             = constr_dict["buffer_pct"]
                class_sum_constraints  = constr_dict["class_sum_constraints"]
                subtype_constraints    = constr_dict["subtype_constraints"]
                daily_rf               = constr_dict["daily_rf"]
                cost_type              = constr_dict["cost_type"]
                transaction_cost_value = constr_dict["transaction_cost_value"]
                trade_buffer_pct       = constr_dict["trade_buffer_pct"]
            else:
                st.sidebar.warning("Parquet files loaded but no valid data found.")
        else:
            st.sidebar.info("Waiting for Parquet files...")

    return {
        "approach_data": approach_data,
        "df_instruments": df_instruments,
        "df_prices": df_prices,
        "coverage": coverage,

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
