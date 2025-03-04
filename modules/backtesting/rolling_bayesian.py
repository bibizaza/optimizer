# File: modules/backtesting/rolling_bayesian.py

import streamlit as st
import pandas as pd
import numpy as np
import time

# scikit-optimize
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical

# Import run_one_combo using dot-separated module path
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
    
    Offers two options for specifying efficient frontier points:
      1) Range: A lower and upper bound (Integer space).
      2) Fixed: A single, fixed value (Categorical space).
      
    A try/except block in the objective function catches any error (such as NaNs/Infs)
    arising from invalid parameter combinations, and returns a penalty value.
    """
    st.write("## Bayesian Optimization")

    # Option to use direct solver vs. parametric approach
    use_direct_solver = st.checkbox("Use Direct Solver for Bayesian?", value=False)

    # Number of Bayesian evaluations
    n_calls = st.number_input("Number of Bayesian evaluations (n_calls)", 5, 500, 20, step=5)

    st.write("### Efficient Frontier Points Option")
    # Choose between a range or a fixed number of points.
    points_option = st.radio("Select efficient frontier points option:", ["Range", "Fixed"], index=0)

    if points_option == "Fixed":
        fixed_points = st.number_input("n_points", 1, 999, 50, step=1)
        # Use a Categorical space to hold a constant value.
        n_points_space = Categorical([fixed_points], name="n_points")
    else:
        c1, c2 = st.columns(2)
        with c1:
            min_npts = st.number_input("Min n_points", 1, 999, 25, step=5)
        with c2:
            max_npts = st.number_input("Max n_points", 1, 999, 50, step=5)
        n_points_space = Integer(int(min_npts), int(max_npts), name="n_points")

    st.write("### Parameter Ranges")
    alpha_min = st.slider("Alpha min (mean shrink)", 0.0, 1.0, 0.0, 0.05)
    alpha_max = st.slider("Alpha max (mean shrink)", 0.0, 1.0, 1.0, 0.05)

    beta_min = st.slider("Beta min (cov shrink)", 0.0, 1.0, 0.0, 0.05)
    beta_max = st.slider("Beta max (cov shrink)", 0.0, 1.0, 1.0, 0.05)

    # Possible rebalancing frequencies (months)
    freq_choices = st.multiselect("Possible Rebal Frequencies (months)", [1, 3, 6], default=[1, 3, 6])
    if not freq_choices:
        freq_choices = [1]

    # Possible lookback windows (months)
    lb_choices = st.multiselect("Possible Lookback Windows (months)", [3, 6, 12], default=[3, 6, 12])
    if not lb_choices:
        lb_choices = [3]

    st.write("### EWM Covariance")
    ewm_bool_choices = st.multiselect("Use EWM Cov?", [False, True], default=[False, True])
    if not ewm_bool_choices:
        ewm_bool_choices = [False]
    ewm_alpha_min = st.slider("EWM alpha min", 0.0, 1.0, 0.0, 0.05)
    ewm_alpha_max = st.slider("EWM alpha max", 0.0, 1.0, 1.0, 0.05)

    # List to store all tries
    tries_list = []

    # Build the parameter space
    space = [
        n_points_space,
        Real(alpha_min, alpha_max, name="alpha_"),
        Real(beta_min, beta_max, name="beta_"),
        Categorical(freq_choices, name="freq_"),
        Categorical(lb_choices, name="lb_"),
        Categorical(ewm_bool_choices, name="do_ewm_"),
        Real(ewm_alpha_min, ewm_alpha_max, name="ewm_alpha_")
    ]

    # UI for progress reporting
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
        The objective function to be minimized.
        x: [n_points, alpha_, beta_, freq_, lb_, do_ewm_, ewm_alpha_]
        
        Wraps the call to run_one_combo in a try/except block. If any error occurs
        (for example, due to invalid parameter combinations producing NaNs/Infs),
        a penalty value is returned.
        """
        try:
            n_points_  = x[0]
            alpha_     = x[1]
            beta_      = x[2]
            freq_      = x[3]
            lb_        = x[4]
            do_ewm_    = x[5]
            ewm_alpha_ = x[6]

            # Clamp ewm_alpha_ if needed
            if do_ewm_ and ewm_alpha_ <= 0:
                ewm_alpha_ = 1e-6
            elif do_ewm_ and ewm_alpha_ > 1:
                ewm_alpha_ = 1.0

            # Call run_one_combo; note that if use_direct_solver=True, n_points_ is ignored internally.
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
                use_direct_solver=use_direct_solver  # key parameter for solver selection
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

            # If the Sharpe Ratio is NaN or infinite, return a penalty.
            if np.isnan(sr_val) or np.isinf(sr_val):
                return 1e6

            # We minimize the negative Sharpe Ratio.
            return -sr_val

        except Exception as e:
            st.error(f"Error in objective with parameters {x}: {e}")
            # Optionally, record the failed combination.
            tries_list.append({
                "n_points": x[0],
                "alpha": x[1],
                "beta": x[2],
                "rebal_freq": x[3],
                "lookback_m": x[4],
                "do_ewm": x[5],
                "ewm_alpha": x[6],
                "Sharpe Ratio": np.nan,
                "error": str(e)
            })
            # Return a large penalty to discourage this combination.
            return 1e6

    # Wait for the user to start optimization
    if not st.button("Run Bayesian Optimization"):
        return pd.DataFrame()

    with st.spinner("Running Bayesian Optimization..."):
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