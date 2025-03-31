"""
File: modules/backtesting/rolling_bayesian.py

Implements a rolling Bayesian optimization approach with an optional
Online(HMM) approach that automatically aggregates *all* columns in df_prices.
"""

import streamlit as st
import pandas as pd
import numpy as np
import time

# scikit-optimize
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import (RBF, Matern, RationalQuadratic)

# Your helper that runs a single combination
from modules.backtesting.rolling_gridsearch import run_one_combo

# The aggregator-based HMM approach
from modules.regime_detection.hmm_aggregator import AggregatedHMMRegimeManager


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
    Bayesian Optimization extended with an optional "Online(HMM)" approach:
      - If user picks "Online(HMM)", we run HMM aggregator on *all* columns 
        in df_prices to detect regime, then proceed with normal Bayesian steps.
      - If user picks "Manual", skip HMM and do a standard approach.

    Returns a DataFrame of tried hyperparam combos and their Sharpe.
    """

    st.write("## Bayesian Optimization")

    # -----------------------------------------------------
    # 0) Approach => "Manual" or "Online(HMM)"
    # -----------------------------------------------------
    approach_choice = st.radio(
        "Bayesian Approach:",
        ["Manual", "Online(HMM)"],
        index=0
    )

    current_regime = None

    if approach_choice == "Online(HMM)":
        st.write("### HMM-Based Regime Detection (Aggregate All Instruments)")

        # 1) Let user pick # states, rolling vol window, eq vs old weighting
        hmm_n_states = st.number_input("Number of hidden states", 2, 5, 2, step=1)
        hmm_window   = st.number_input("Rolling window for HMM vol", 5, 252, 20, step=5)
        eq_weighting = st.checkbox("Use equal weighting across all instruments?", value=True)

        # Subset last X days
        hmm_lookback_days = st.number_input("HMM lookback (days)", 30, 2000, 252, step=10)
        if len(df_prices) > hmm_lookback_days:
            df_hmm = df_prices.iloc[-hmm_lookback_days:].copy()
        else:
            df_hmm = df_prices.copy()

        # 2) Button to run aggregator detection
        if st.button("Run Aggregated HMM"):
            with st.spinner("Fitting Aggregated HMM..."):
                agg_manager = AggregatedHMMRegimeManager(
                    n_states=hmm_n_states,
                    window_vol=hmm_window,
                    random_state=42,
                    use_equal_weights=eq_weighting
                )
                states, current_regime = agg_manager.fit_predict(
                    df_prices=df_hmm, 
                    df_instruments=df_instruments
                )

            if states.size > 0:
                st.success(f"Done. Current regime => State {current_regime}")
                if st.checkbox("Show HMM state stats?"):
                    agg_manager.print_state_stats()
            else:
                st.warning("No data for aggregator => states are empty.")

        st.write("---")


    # -----------------------------------------------------
    # 1) Choose solver approach => param vs direct
    # -----------------------------------------------------
    solver_choice = st.radio(
        "Solver Approach for Bayesian? (cvxpy-based):",
        ["Parametric (cvxpy)", "Direct (cvxpy)"],
        index=0
    )
    use_direct_solver = (solver_choice == "Direct (cvxpy)")

    n_calls = st.number_input("Number of Bayesian evaluations (n_calls)", 5, 500, 20, step=5)

    # If param => let user specify n_points or range
    if not use_direct_solver:
        st.write("### Efficient Frontier Points Option")
        points_option = st.radio("Frontier n_points approach:", ["Range", "Fixed"], index=0)
        if points_option == "Fixed":
            fixed_points = st.number_input("Frontier n_points", 1, 999, 50, step=1)
            n_points_space = Categorical([fixed_points], name="n_points")
        else:
            c1, c2 = st.columns(2)
            with c1:
                min_npts = st.number_input("Min n_points", 1, 999, 25, step=5)
            with c2:
                max_npts = st.number_input("Max n_points", 1, 999, 50, step=5)
            n_points_space = Integer(int(min_npts), int(max_npts), name="n_points")
    else:
        n_points_space = None

    # -----------------------------------------------------
    # 2) The rest of param ranges: alpha, beta, etc.
    # -----------------------------------------------------
    st.write("### Hyperparameter Ranges")

    alpha_min = st.slider("Alpha min (mean shrink)", 0.0, 1.0, 0.0, 0.05)
    alpha_max = st.slider("Alpha max (mean shrink)", 0.0, 1.0, 1.0, 0.05)

    beta_min = st.slider("Beta min (cov shrink)", 0.0, 1.0, 0.0, 0.05)
    beta_max = st.slider("Beta max (cov shrink)", 0.0, 1.0, 1.0, 0.05)

    freq_choices = st.multiselect("Possible Rebal Frequencies (months)", [1, 3, 6], default=[1,3,6])
    if not freq_choices:
        freq_choices = [1]

    lb_choices = st.multiselect("Possible Lookback Windows (months)", [3,6,12], default=[3,6,12])
    if not lb_choices:
        lb_choices = [3]

    st.write("### EWM Covariance")
    ewm_bool_choices = st.multiselect("Use EWM Cov?", [False, True], default=[False,True])
    if not ewm_bool_choices:
        ewm_bool_choices = [False]

    ewm_alpha_min = st.slider("EWM alpha min", 0.0,1.0,0.0,0.05)
    ewm_alpha_max = st.slider("EWM alpha max", 0.0,1.0,1.0,0.05)

    if use_direct_solver:
        space = [
            Real(alpha_min, alpha_max, name="alpha_"),
            Real(beta_min,  beta_max,  name="beta_"),
            Categorical(freq_choices,  name="freq_"),
            Categorical(lb_choices,    name="lb_"),
            Categorical(ewm_bool_choices, name="do_ewm_"),
            Real(ewm_alpha_min, ewm_alpha_max, name="ewm_alpha_")
        ]
    else:
        # param => include n_points
        space = [
            n_points_space,
            Real(alpha_min, alpha_max, name="alpha_"),
            Real(beta_min,  beta_max,  name="beta_"),
            Categorical(freq_choices,  name="freq_"),
            Categorical(lb_choices,    name="lb_"),
            Categorical(ewm_bool_choices, name="do_ewm_"),
            Real(ewm_alpha_min, ewm_alpha_max, name="ewm_alpha_")
        ]

    tries_list = []

    # -----------------------------------------------------
    # 3) The objective function
    # -----------------------------------------------------
    def objective(x):
        """
        If direct => x => [alpha_, beta_, freq_, lb_, do_ewm_, ewm_alpha_].
        If param => x => [n_points_, alpha_, beta_, freq_, lb_, do_ewm_, ewm_alpha_].
        """
        try:
            if use_direct_solver:
                alpha_     = x[0]
                beta_      = x[1]
                freq_      = x[2]
                lb_        = x[3]
                do_ewm_    = x[4]
                ewm_alpha_ = x[5]
                n_points_  = None
            else:
                n_points_  = x[0]
                alpha_     = x[1]
                beta_      = x[2]
                freq_      = x[3]
                lb_        = x[4]
                do_ewm_    = x[5]
                ewm_alpha_ = x[6]

            # clamp if needed
            if do_ewm_ and (ewm_alpha_ <= 0):
                ewm_alpha_ = 1e-6
            elif do_ewm_ and (ewm_alpha_ > 1):
                ewm_alpha_ = 1.0

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
                use_direct_solver=use_direct_solver
            )

            sr_val = result_dict["Sharpe Ratio"]
            tries_list.append({
                "n_points": n_points_ if n_points_ is not None else "direct",
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

            if np.isnan(sr_val) or np.isinf(sr_val):
                return 1e9
            return -sr_val

        except Exception as e:
            st.error(f"Error in objective => {x}: {e}")
            tries_list.append({"params": x, "error": str(e)})
            return 1e9

    # For progress
    progress_bar = st.progress(0)
    progress_text = st.empty()
    start_time = time.time()

    def on_step(res):
        done = len(res.x_iters)
        pct = int(done * 100 / n_calls)
        elapsed = time.time() - start_time
        progress_text.text(f"Progress: {pct}% complete. Elapsed: {elapsed:.1f}s")
        progress_bar.progress(pct)

    # -----------------------------------------------------
    # 4) GP config
    # -----------------------------------------------------
    st.write("### Gaussian Process Settings")

    kernel_choice = st.selectbox("Select GP Kernel", ["Matern","RBF","RationalQuadratic"], index=0)
    length_scale_init = st.slider("Kernel length_scale", 0.1,10.0,1.0,0.1)

    # Matern => user picks nu
    matern_nu = st.selectbox("Matern Nu", [0.5, 1.5, 2.5], index=2)

    alpha_val = st.number_input("GP alpha (noise)", min_value=1e-9, max_value=1.0, value=0.01,
                                step=0.01, format="%.6f")
    normalize_y = st.checkbox("Normalize GP outputs (Sharpe)?", value=True)

    if kernel_choice == "Matern":
        chosen_kernel = Matern(length_scale=length_scale_init, nu=matern_nu)
    elif kernel_choice == "RationalQuadratic":
        chosen_kernel = RationalQuadratic(length_scale=length_scale_init, alpha=1.0)
    else:
        chosen_kernel = RBF(length_scale=length_scale_init)

    gp_model = GaussianProcessRegressor(
        kernel=chosen_kernel,
        alpha=alpha_val,
        normalize_y=normalize_y,
        random_state=42
    )

    # -----------------------------------------------------
    # 5) Run Bayesian if user wants
    # -----------------------------------------------------
    if not st.button("Run Bayesian Optimization"):
        return pd.DataFrame()

    with st.spinner("Running Bayesian optimization..."):
        res = gp_minimize(
            func=objective,
            dimensions=space,
            base_estimator=gp_model,
            n_calls=n_calls,
            random_state=42,
            callback=[on_step]
        )

    df_out = pd.DataFrame(tries_list)
    if not df_out.empty:
        best_ = df_out.sort_values("Sharpe Ratio", ascending=False).iloc[0]
        st.write("**Best Found** =>", dict(best_))
        st.dataframe(df_out)

    return df_out