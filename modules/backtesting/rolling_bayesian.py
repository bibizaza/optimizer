# File: rolling_bayesian.py

import streamlit as st
import pandas as pd
import numpy as np
import time

# scikit-optimize
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import (
    RBF, Matern, RationalQuadratic
)

# Hypothetical helper that executes a single combination of hyperparams
# for param vs direct approach, etc. Make sure you have the correct import:
# from modules.backtesting.rolling_gridsearch import run_one_combo
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
    Bayesian Optimization with a user-configurable Gaussian Process (GP) as the surrogate.
    """

    st.write("## Bayesian Optimization")

    # -------------------------------------------
    # 1) Choose solver approach: param or direct
    # -------------------------------------------
    solver_choice = st.radio(
        "Solver Approach for Bayesian?",
        ["Parametric (cvxpy)", "Direct (cvxpy)"],
        index=0
    )
    use_direct_solver = (solver_choice == "Direct (cvxpy)")

    # # of Bayesian evaluations
    n_calls = st.number_input("Number of Bayesian evaluations (n_calls)", 5, 500, 20, step=5)

    # If param => let user specify frontier n_points or range
    if not use_direct_solver:
        st.write("### Efficient Frontier Points Option")
        points_option = st.radio("Select how to pick n_points:", ["Range", "Fixed"], index=0)
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

    # -------------------------------------------
    # 2) The rest of param ranges: alpha, beta, etc.
    # -------------------------------------------
    st.write("### Hyperparameter Ranges")

    alpha_min = st.slider("Alpha min (mean shrink)", 0.0, 1.0, 0.0, 0.05)
    alpha_max = st.slider("Alpha max (mean shrink)", 0.0, 1.0, 1.0, 0.05)

    beta_min = st.slider("Beta min (cov shrink)", 0.0, 1.0, 0.0, 0.05)
    beta_max = st.slider("Beta max (cov shrink)", 0.0, 1.0, 1.0, 0.05)

    freq_choices = st.multiselect("Possible Rebal Frequencies (months)", [1, 3, 6], default=[1, 3, 6])
    if not freq_choices:
        freq_choices = [1]

    lb_choices = st.multiselect("Possible Lookback Windows (months)", [3, 6, 12], default=[3, 6, 12])
    if not lb_choices:
        lb_choices = [3]

    st.write("### EWM Covariance")
    ewm_bool_choices = st.multiselect("Use EWM Cov?", [False, True], default=[False, True])
    if not ewm_bool_choices:
        ewm_bool_choices = [False]

    ewm_alpha_min = st.slider("EWM alpha min", 0.0, 1.0, 0.0, 0.05)
    ewm_alpha_max = st.slider("EWM alpha max", 0.0, 1.0, 1.0, 0.05)

    # The base dimension list
    space_common = [
        Real(alpha_min, alpha_max, name="alpha_"),
        Real(beta_min, beta_max,   name="beta_"),
        Categorical(freq_choices,  name="freq_"),
        Categorical(lb_choices,    name="lb_"),
        Categorical(ewm_bool_choices, name="do_ewm_"),
        Real(ewm_alpha_min, ewm_alpha_max, name="ewm_alpha_")
    ]
    if use_direct_solver:
        # Direct => no n_points in dimension
        space = space_common
    else:
        space = [n_points_space] + space_common

    # We'll store partial results here
    tries_list = []

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

    # -------------------------------------------
    # 3) Define the objective function
    # -------------------------------------------
    def objective(x):
        """
        If direct => x is [alpha_, beta_, freq_, lb_, do_ewm_, ewm_alpha_].
        If param => x is [n_points_, alpha_, beta_, freq_, lb_, do_ewm_, ewm_alpha_].
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

            # clamp ewm_alpha if do_ewm_ is True
            if do_ewm_:
                if ewm_alpha_ <= 0: ewm_alpha_ = 1e-6
                if ewm_alpha_ > 1:  ewm_alpha_ = 1.0

            # Run a single combo of hyperparams
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
            return -sr_val  # we minimize negative SR => maximize SR
        except Exception as e:
            st.error(f"Error in objective with params {x}: {e}")
            tries_list.append({"params": x, "error": str(e)})
            return 1e9

    # -------------------------------------------
    # 4) Expose GP configuration
    # -------------------------------------------
    st.write("### Gaussian Process Settings")

    kernel_choice = st.selectbox("Select GP Kernel", ["Matern", "RBF", "RationalQuadratic"], index=0)
    length_scale_init = st.slider("Kernel length_scale", 0.1, 10.0, 1.0, 0.1)

    # If Matern => let user pick nu
    default_nu = 2.5
    matern_nu = st.selectbox("Matern Nu (smoothness)", [0.5, 1.5, 2.5], index=2)

    alpha_val = st.number_input("GP alpha (noise)", min_value=1e-9, max_value=1.0, value=0.01, step=0.01, format="%.6f")
    normalize_y = st.checkbox("Normalize GP outputs (Sharpe)?", value=True)

    # Build the chosen kernel
    if kernel_choice == "Matern":
        chosen_kernel = Matern(length_scale=length_scale_init, nu=matern_nu)
    elif kernel_choice == "RationalQuadratic":
        chosen_kernel = RationalQuadratic(length_scale=length_scale_init, alpha=1.0)
    else:  # "RBF"
        from skopt.learning.gaussian_process.kernels import RBF
        chosen_kernel = RBF(length_scale=length_scale_init)

    # Create the custom GP
    gp_model = GaussianProcessRegressor(
        kernel=chosen_kernel,
        alpha=alpha_val,
        normalize_y=normalize_y,
        random_state=42
    )

    # -------------------------------------------
    # 5) Actually run the Bayesian search
    # -------------------------------------------
    if not st.button("Run Bayesian Optimization"):
        return pd.DataFrame()

    with st.spinner("Running Bayesian optimization..."):
        res = gp_minimize(
            func=objective,
            dimensions=space,
            base_estimator=gp_model,  # <--- the GP
            n_calls=n_calls,
            random_state=42,
            callback=[on_step]
        )

    # Prepare final results
    df_out = pd.DataFrame(tries_list)
    if not df_out.empty:
        # Show best combo
        best_ = df_out.sort_values("Sharpe Ratio", ascending=False).iloc[0]
        st.write("**Best Found** =>", dict(best_))
        st.dataframe(df_out)

    return df_out
