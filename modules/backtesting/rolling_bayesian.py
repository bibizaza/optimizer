# File: modules/backtesting/rolling_bayesian.py

import streamlit as st
import pandas as pd
import numpy as np
import time

# scikit-optimize
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

from modules.backtesting.rolling_gridsearch import run_one_combo

def rolling_bayesian_optimization(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    asset_cls_list: list[str],
    sec_type_list: list[str],
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float,
    transaction_cost_value: float,
    transaction_cost_type: str,
    trade_buffer_pct: float
) -> pd.DataFrame:
    """
    Bayesian Optimization over multiple parameters, including do_ewm & ewm_alpha.

    1) Asks user for param search ranges in Streamlit.
    2) Runs scikit-optimize's gp_minimize => calls run_one_combo(...) each iteration.
    3) Shows progress, final best parameters, and DataFrame of tries.

    If user does not click 'Run Bayesian Optimization', returns an empty DataFrame.
    """
    st.write("## Bayesian Optimization")

    # How many total evaluations to do
    n_calls = st.number_input("Number of Bayesian evaluations (n_calls)", 5, 100, 20, step=5)

    st.write("### Parameter Ranges")
    # 1) n_points range
    c1, c2 = st.columns(2)
    with c1:
        min_npts = st.number_input("Min n_points", 1, 999, 5, step=5)
    with c2:
        max_npts = st.number_input("Max n_points", 1, 999, 100, step=5)

    # 2) alpha range
    alpha_min = st.slider("Alpha min (mean shrink)", 0.0, 1.0, 0.0, 0.05)
    alpha_max = st.slider("Alpha max (mean shrink)", 0.0, 1.0, 1.0, 0.05)

    # 3) beta range
    beta_min = st.slider("Beta min (cov shrink)", 0.0, 1.0, 0.0, 0.05)
    beta_max = st.slider("Beta max (cov shrink)", 0.0, 1.0, 1.0, 0.05)

    # 4) possible rebal frequencies
    freq_choices = st.multiselect("Possible Rebal Frequencies (months)", [1,3,6], default=[1,3,6])
    if not freq_choices:
        freq_choices = [1]
    # 5) possible lookback windows
    lb_choices = st.multiselect("Possible Lookback Windows (months)", [3,6,12], default=[3,6,12])
    if not lb_choices:
        lb_choices = [3]

    st.write("### EWM Covariance")
    # 6) do_ewm => True/False
    ewm_bool_choices = st.multiselect("Use EWM Cov?", [False, True], default=[False, True])
    if not ewm_bool_choices:
        ewm_bool_choices = [False]
    # 7) ewm_alpha range
    ewm_alpha_min = st.slider("EWM alpha min", 0.0, 1.0, 0.0, 0.05)
    ewm_alpha_max = st.slider("EWM alpha max", 0.0, 1.0, 1.0, 0.05)

    # We'll store all tries in a list
    tries_list = []

    # Build param space for scikit-optimize. Each point x => [n_points, alpha_, beta_, freq_, lb_, do_ewm_, ewm_alpha_]
    space = [
        Integer(int(min_npts), int(max_npts), name="n_points"),
        Real(alpha_min, alpha_max, name="alpha_"),
        Real(beta_min,  beta_max,  name="beta_"),
        Categorical(freq_choices, name="freq_"),
        Categorical(lb_choices,   name="lb_"),
        Categorical(ewm_bool_choices, name="do_ewm_"),
        Real(ewm_alpha_min, ewm_alpha_max, name="ewm_alpha_")
    ]

    # UI for progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    start_time = time.time()

    def on_step(res):
        # Called after each iteration completes
        done = len(res.x_iters)
        pct = int(done * 100 / n_calls)
        elapsed = time.time() - start_time
        progress_text.text(f"Progress: {pct}% complete. Elapsed: {elapsed:.1f}s")
        progress_bar.progress(pct)

    def objective(x):
        """
        x => [n_points, alpha_, beta_, freq_, lb_, do_ewm_, ewm_alpha_]
        """
        combo = tuple(x)  # to store in logs
        n_points_ = x[0]
        alpha_ = x[1]
        beta_ = x[2]
        freq_ = x[3]
        lb_ = x[4]
        do_ewm_ = x[5]
        ewm_alpha_ = x[6]

        # Send them all to run_one_combo
        result = run_one_combo(
            df_prices = df_prices,
            df_instruments = df_instruments,
            asset_cls_list = asset_cls_list,
            sec_type_list = sec_type_list,
            class_sum_constraints = class_sum_constraints,
            subtype_constraints = subtype_constraints,
            daily_rf = daily_rf,
            combo = (n_points_, alpha_, beta_, freq_, lb_),
            transaction_cost_value = transaction_cost_value,
            transaction_cost_type = transaction_cost_type,
            trade_buffer_pct = trade_buffer_pct,
            use_michaud = False,
            n_boot = 10,
            do_shrink_means = True,
            do_shrink_cov = True,
            reg_cov = False,
            do_ledoitwolf = False,
            do_ewm = do_ewm_,
            ewm_alpha = ewm_alpha_
        )

        # Keep track of results in tries_list
        tries_list.append({
            "n_points":     n_points_,
            "alpha":        alpha_,
            "beta":         beta_,
            "rebal_freq":   freq_,
            "lookback_m":   lb_,
            "do_ewm":       do_ewm_,
            "ewm_alpha":    ewm_alpha_,
            "Sharpe Ratio": result["Sharpe Ratio"],
            "Annual Ret":   result["Annual Ret"],
            "Annual Vol":   result["Annual Vol"]
        })

        # We want to maximize Sharpe => so we minimize negative Sharpe
        return -result["Sharpe Ratio"]

    # If user doesn't click => do nothing
    if not st.button("Run Bayesian Optimization"):
        return pd.DataFrame()

    from skopt import gp_minimize

    with st.spinner("Running Bayesian..."):
        res = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=42,  # fix seed for reproducibility if desired
            callback=[on_step]
        )

    df_out = pd.DataFrame(tries_list)
    # Show best result
    if not df_out.empty:
        best_idx = df_out["Sharpe Ratio"].idxmax()
        best_row = df_out.loc[best_idx]
        st.write("**Best Found**:", dict(best_row))
        st.dataframe(df_out)

    return df_out