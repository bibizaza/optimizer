# File: modules/backtesting/rolling_bayesian.py

import streamlit as st
import pandas as pd
import numpy as np
import time

# scikit-optimize
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

# This 'run_one_combo' is the same one used in rolling_gridsearch. 
# We rely on it to handle param vs direct via `use_direct_solver`.
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
    Bayesian Optimization over multiple parameters. 
    We add an option for 'Use Direct Solver' => ignoring n_points in that case.

    Steps:
      1) UI to pick param search ranges, plus a checkbox 'Use Direct Solver?'.
      2) scikit-optimize => gp_minimize => in objective(...) we call run_one_combo(..., use_direct_solver=?)
      3) We store tries in tries_list => build df_out => present best combo.

    If the user doesn't click 'Run Bayesian Optimization', we return an empty DataFrame.
    """

    st.write("## Bayesian Optimization")

    # 1) Let user pick direct or param approach
    use_direct_solver = st.checkbox("Use Direct Solver for Bayesian?", value=False)

    # 2) # of Bayesian evaluations
    n_calls = st.number_input("Number of Bayesian evaluations (n_calls)", 5, 500, 20, step=5)

    st.write("### Parameter Ranges")
    # For the param approach, we do need n_points. We'll pass it but if use_direct_solver=True, 
    # run_one_combo will ignore n_points for the direct approach.
    c1, c2 = st.columns(2)
    with c1:
        min_npts = st.number_input("Min n_points", 1, 999, 5, step=5)
    with c2:
        max_npts = st.number_input("Max n_points", 1, 999, 100, step=5)

    alpha_min = st.slider("Alpha min (mean shrink)", 0.0, 1.0, 0.0, 0.05)
    alpha_max = st.slider("Alpha max (mean shrink)", 0.0, 1.0, 1.0, 0.05)

    beta_min = st.slider("Beta min (cov shrink)", 0.0, 1.0, 0.0, 0.05)
    beta_max = st.slider("Beta max (cov shrink)", 0.0, 1.0, 1.0, 0.05)

    # Possible rebal freq
    freq_choices = st.multiselect("Possible Rebal Frequencies (months)", [1,3,6], default=[1,3,6])
    if not freq_choices:
        freq_choices = [1]

    # Possible lookback windows
    lb_choices = st.multiselect("Possible Lookback Windows (months)", [3,6,12], default=[3,6,12])
    if not lb_choices:
        lb_choices = [3]

    st.write("### EWM Covariance")
    ewm_bool_choices = st.multiselect("Use EWM Cov?", [False, True], default=[False, True])
    if not ewm_bool_choices:
        ewm_bool_choices = [False]
    ewm_alpha_min = st.slider("EWM alpha min", 0.0, 1.0, 0.0, 0.05)
    ewm_alpha_max = st.slider("EWM alpha max", 0.0, 1.0, 1.0, 0.05)

    # We'll store all tries in a list
    tries_list = []

    # Build param space
    space = [
        Integer(int(min_npts), int(max_npts), name="n_points"),
        Real(alpha_min, alpha_max, name="alpha_"),
        Real(beta_min, beta_max, name="beta_"),
        Categorical(freq_choices, name="freq_"),
        Categorical(lb_choices, name="lb_"),
        Categorical(ewm_bool_choices, name="do_ewm_"),
        Real(ewm_alpha_min, ewm_alpha_max, name="ewm_alpha_")
    ]

    # UI for progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    start_time = time.time()

    def on_step(res):
        done = len(res.x_iters)
        pct = int(done * 100 / n_calls)
        elapsed = time.time() - start_time
        progress_text.text(f"Progress: {pct}% complete. Elapsed: {elapsed:.1f}s")
        progress_bar.progress(pct)

    def objective(x):
        """
        x => [n_points, alpha_, beta_, freq_, lb_, do_ewm_, ewm_alpha_]
        We'll pass them to run_one_combo => if use_direct_solver=True => direct approach ignoring n_points,
        else param approach uses n_points.
        """
        n_points_  = x[0]
        alpha_     = x[1]
        beta_      = x[2]
        freq_      = x[3]
        lb_        = x[4]
        do_ewm_    = x[5]
        ewm_alpha_ = x[6]

        # minor clamp
        if do_ewm_ and ewm_alpha_ <= 0:
            ewm_alpha_ = 1e-6
        elif do_ewm_ and ewm_alpha_ > 1:
            ewm_alpha_ = 1.0

        # call run_one_combo => pass use_direct_solver
        result_dict = run_one_combo(
            df_prices=df_prices,
            df_instruments=df_instruments,
            asset_cls_list=asset_cls_list,
            sec_type_list=sec_type_list,
            class_sum_constraints=class_sum_constraints,
            subtype_constraints=subtype_constraints,
            daily_rf=daily_rf,
            combo=(n_points_, alpha_, beta_, freq_, lb_),
            transaction_cost_value=transaction_cost_value,
            transaction_cost_type=transaction_cost_type,
            trade_buffer_pct=trade_buffer_pct,
            use_michaud=False,
            n_boot=10,
            do_shrink_means=True,
            do_shrink_cov=True,
            reg_cov=False,
            do_ledoitwolf=False,
            do_ewm=do_ewm_,
            ewm_alpha=ewm_alpha_,
            use_direct_solver=use_direct_solver  # <--- key param
        )

        sr_val = result_dict["Sharpe Ratio"]
        tries_list.append({
            "n_points": n_points_,
            "alpha": alpha_,
            "beta": beta_,
            "rebal_freq": freq_,
            "lookback_m": lb_,
            "do_ewm": do_ewm_,
            "ewm_alpha": ewm_alpha_,
            "Sharpe Ratio": sr_val,
            "Annual Ret": result_dict["Annual Ret"],
            "Annual Vol": result_dict["Annual Vol"]
        })

        return -sr_val

    # Wait for user
    if not st.button("Run Bayesian Optimization"):
        return pd.DataFrame()

    from skopt import gp_minimize

    with st.spinner("Running Bayesian..."):
        res = gp_minimize(
            objective,
            space,
            n_calls=n_calls,
            random_state=42,
            callback=[on_step]
        )

    df_out = pd.DataFrame(tries_list)
    if not df_out.empty:
        best_idx = df_out["Sharpe Ratio"].idxmax()
        best_row = df_out.loc[best_idx].copy()
        st.write("**Best Found**:", dict(best_row))
        st.dataframe(df_out)

        import json
        best_json = df_out.loc[[best_idx]].to_json(orient="records", indent=2)
        st.download_button("Download Best Param (JSON)", best_json, "best_bayes.json", "application/json")

    return df_out