"""
File: rolling_bayesian.py

This module provides a rolling Bayesian optimization approach
with an optional "Online(HMM)" aggregator-based regime detection
that uses all instruments in df_prices to build a single
aggregated price series for the HMM.

Requires:
  - 'hmmlearn' for the HMM logic
  - 'scikit-optimize' for Bayesian optimization
  - 'cvxpy', etc., for the actual solver approach

Flow:
  - The user picks "Manual" (skip HMM) or "Online(HMM)" in Streamlit.
  - If "Online(HMM)", the user can run a minimal aggregator-based HMM
    that aggregates all instruments (by either equal weights or
    df_instruments["Weight_Old"]) and detects a regime from
    daily returns + rolling volatility features.
  - After that, we proceed with normal Bayesian hyperparameter search:
    param (frontier-based) or direct approach (cvxpy),
    plus a user-configured Gaussian Process as the surrogate model.
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

# The function that runs a single combination of hyperparams:
from modules.backtesting.rolling_gridsearch import run_one_combo

###############################################################################
# 1) Minimal aggregator-based HMM
###############################################################################
try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False

class AggregatedHMMRegimeManager:
    """
    Builds an aggregated price series from all columns in df_prices (weighted
    either equally or by df_instruments["Weight_Old"]). Then uses daily returns
    + rolling volatility (over 'window_vol' days) as a 2D feature. Finally, a
    GaussianHMM is fit on that feature to detect hidden states.

    Example usage:
        agg_manager = AggregatedHMMRegimeManager(n_states=3, window_vol=20, ...)
        states, current_state = agg_manager.fit_predict(df_prices, df_instruments)
    """
    def __init__(
        self,
        n_states: int = 2,
        window_vol: int = 20,
        random_state: int = 42,
        use_equal_weights: bool = True
    ):
        if not HMMLEARN_AVAILABLE:
            raise ImportError("Requires 'hmmlearn'. Install with 'pip install hmmlearn'.")

        self.n_states = n_states
        self.window_vol = window_vol
        self.random_state = random_state
        self.use_equal_weights = use_equal_weights

        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            random_state=random_state
        )
        self.fitted = False

    def _build_aggregate_series(
        self,
        df_prices: pd.DataFrame,
        df_instruments: pd.DataFrame
    ) -> pd.Series:
        """
        Aggregates all columns in df_prices into one series:
          price(t) = sum_i [ weight_i * price_i(t) ]
        If 'use_equal_weights' is False and df_instruments["Weight_Old"] is found,
        we use that to get each instrument's weight. Otherwise, we do equal weighting.
        """
        n_instruments = df_prices.shape[1]

        # 1) Determine weighting
        if (not self.use_equal_weights) and ("Weight_Old" in df_instruments.columns):
            # read old weights
            w_map = {}
            for _, row in df_instruments.iterrows():
                tkr = row["#ID"]
                w_  = row["Weight_Old"]
                w_map[tkr] = w_
            total_w = sum(w_map.values())
            if total_w < 1e-12:
                # fallback to equal
                weights = np.ones(n_instruments) / n_instruments
            else:
                w_list = []
                for c in df_prices.columns:
                    w_list.append(w_map.get(c, 0.0))
                weights = np.array(w_list) / total_w
        else:
            # default => equal weighting
            weights = np.ones(n_instruments) / n_instruments

        # 2) Build aggregator
        arr_prices = df_prices.values
        arr_prices = np.nan_to_num(arr_prices, nan=0.0)
        aggregated = arr_prices.dot(weights)
        sr_agg = pd.Series(aggregated, index=df_prices.index, name="AggPrice")

        # forward/back fill if needed
        sr_agg = sr_agg.replace(0.0, np.nan).fillna(method="ffill").fillna(method="bfill")
        return sr_agg

    def _prepare_features(self, sr_agg: pd.Series) -> np.ndarray:
        """
        Returns shape (T,2) => daily_ret, rolling_vol.
        """
        ret = sr_agg.pct_change().fillna(0.0)
        vol = ret.rolling(self.window_vol).std().fillna(0.0)
        X   = np.column_stack([ret.values, vol.values])
        return X

    def fit_predict(self, df_prices, df_instruments):
        """
        Aggregates => build features => fit HMM => predict => return states, last state
        """
        sr_agg = self._build_aggregate_series(df_prices, df_instruments)
        X = self._prepare_features(sr_agg)

        self.model.fit(X)
        self.fitted = True

        if len(X) < 1:
            return np.array([]), None

        states = self.model.predict(X)
        current_state = states[-1]
        return states, current_state

    def print_state_stats(self):
        """
        Print out means & covariance for interpretability.
        """
        if not self.fitted:
            print("HMM not fitted yet.")
            return
        for i in range(self.n_states):
            mu = self.model.means_[i]    # shape (2,)
            cov= self.model.covars_[i]   # shape (2,2)
            print(f"State {i}: mean_ret={mu[0]:.4f}, mean_vol={mu[1]:.4f}\n Cov:\n{cov}\n")


###############################################################################
# 2) Rolling Bayesian with optional HMM
###############################################################################

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
    A rolling / "one-shot" Bayesian optimization with Streamlit UI:
      - "Manual" => skip aggregator HMM
      - "Online(HMM)" => aggregator-based HMM => fit & detect regime

    Then we do param or direct approach:
      - param => includes n_points
      - direct => skip n_points

    Finally, we define a GP model => run gp_minimize => present best combo.

    Returns: DataFrame of tried combos & Sharpe.
    """

    st.write("## Bayesian Optimization")

    # 0) Approach => "Manual" or "Online(HMM)"
    approach_choice = st.radio(
        "Bayesian Approach:",
        ["Manual", "Online(HMM)"],
        index=0
    )

    current_regime = None
    if approach_choice == "Online(HMM)":
        st.write("### Aggregator-based HMM: Detect Regime")

        hmm_n_states = st.number_input("Number of states in HMM", 2, 5, 2, step=1)
        hmm_window   = st.number_input("Rolling window for vol", 5, 252, 20, step=5)
        eq_weighting = st.checkbox("Use equal weighting for aggregator?", value=True)

        lookback_days= st.number_input("HMM lookback (days)", 30, 2000, 252, step=10)
        if len(df_prices) > lookback_days:
            df_hmm = df_prices.iloc[-lookback_days:].copy()
        else:
            df_hmm = df_prices.copy()

        if st.button("Run Aggregated HMM"):
            with st.spinner("Fitting aggregator-based HMM..."):
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
                st.success(f"HMM done. Current regime => State {current_regime}")
                if st.checkbox("Show HMM state stats?"):
                    agg_manager.print_state_stats()
            else:
                st.warning("No data => aggregator returned empty states.")
        st.write("---")


    # 1) solver => param or direct
    solver_choice = st.radio(
        "Solver Approach (cvxpy-based)?",
        ["Parametric (cvxpy)", "Direct (cvxpy)"],
        index=0
    )
    use_direct_solver = (solver_choice == "Direct (cvxpy)")

    # number of calls
    n_calls = st.number_input("Number of Bayesian calls (n_calls)", 5, 500, 20, step=5)

    # If param => define n_points
    n_points_space = None
    if not use_direct_solver:
        st.write("### Frontier n_points Approach")
        points_opt = st.radio("n_points => fixed or range?", ["Fixed","Range"], index=0)
        if points_opt == "Fixed":
            fixed_np = st.number_input("Frontier n_points", 1, 999, 50, step=1)
            n_points_space = Categorical([fixed_np], name="n_points")
        else:
            c1, c2 = st.columns(2)
            with c1:
                min_np = st.number_input("Min n_points", 1, 999, 25, step=5)
            with c2:
                max_np = st.number_input("Max n_points", 1, 999, 50, step=5)
            n_points_space = Integer(int(min_np), int(max_np), name="n_points")

    # 2) The rest: alpha, beta, freq, lb, ewm, ewm_alpha
    alpha_min = st.slider("Alpha min (mean shrink)", 0.0, 1.0, 0.0, 0.05)
    alpha_max = st.slider("Alpha max (mean shrink)", 0.0, 1.0, 1.0, 0.05)

    beta_min  = st.slider("Beta min (cov shrink)", 0.0,1.0,0.0, 0.05)
    beta_max  = st.slider("Beta max (cov shrink)", 0.0,1.0,1.0, 0.05)

    freq_ch   = st.multiselect("Possible Rebal Frequencies", [1,3,6], [1,3,6])
    if not freq_ch: freq_ch=[1]

    lb_ch     = st.multiselect("Lookback Windows (months)", [3,6,12], [3,6,12])
    if not lb_ch: lb_ch=[3]

    ewm_bool  = st.multiselect("Use EWM Cov?", [False,True], [False,True])
    if not ewm_bool: ewm_bool=[False]

    ewm_alpha_min = st.slider("EWM alpha min", 0.0,1.0,0.0,0.05)
    ewm_alpha_max = st.slider("EWM alpha max", 0.0,1.0,1.0,0.05)

    # Build dimension
    if use_direct_solver:
        dims = [
            Real(alpha_min, alpha_max, name="alpha_"),
            Real(beta_min,  beta_max,  name="beta_"),
            Categorical(freq_ch, name="freq_"),
            Categorical(lb_ch,   name="lb_"),
            Categorical(ewm_bool,name="do_ewm_"),
            Real(ewm_alpha_min, ewm_alpha_max, name="ewm_alpha_")
        ]
    else:
        dims = [
            n_points_space,
            Real(alpha_min, alpha_max, name="alpha_"),
            Real(beta_min,  beta_max,  name="beta_"),
            Categorical(freq_ch,  name="freq_"),
            Categorical(lb_ch,    name="lb_"),
            Categorical(ewm_bool, name="do_ewm_"),
            Real(ewm_alpha_min, ewm_alpha_max, name="ewm_alpha_")
        ]

    tries_list = []

    # 3) objective function
    def objective(x):
        """
        direct => [alpha_, beta_, freq_, lb_, do_ewm_, ewm_alpha_]
        param  => [n_points_, alpha_, beta_, freq_, lb_, do_ewm_, ewm_alpha_]
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

            # clamp
            if do_ewm_ and ewm_alpha_<=0:
                ewm_alpha_ = 1e-6
            elif do_ewm_ and ewm_alpha_>1:
                ewm_alpha_ = 1.0

            res_dict = run_one_combo(
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
            sr_val = res_dict["Sharpe Ratio"]
            tries_list.append({
                "n_points": (n_points_ if n_points_ is not None else "direct"),
                "alpha": alpha_,
                "beta":  beta_,
                "rebal_freq": freq_,
                "lookback_m": lb_,
                "do_ewm": do_ewm_,
                "ewm_alpha": ewm_alpha_,
                "Sharpe Ratio": sr_val,
                "Annual Ret": res_dict["Annual Ret"],
                "Annual Vol": res_dict["Annual Vol"]
            })
            if np.isnan(sr_val) or np.isinf(sr_val):
                return 1e9
            return -sr_val

        except Exception as e:
            st.error(f"Error in objective => {x}: {e}")
            tries_list.append({"params": x, "error": str(e)})
            return 1e9

    # progress
    progress_bar = st.progress(0)
    progress_text= st.empty()
    start_time   = time.time()

    def on_step(res):
        done = len(res.x_iters)
        pct = int(done * 100 / n_calls)
        elapsed = time.time() - start_time
        progress_text.text(f"Progress: {pct}% done. Elapsed={elapsed:.1f}s")
        progress_bar.progress(pct)

    # 4) GP config
    st.write("### Gaussian Process Settings")
    kernel_choice = st.selectbox("GP Kernel", ["Matern","RBF","RationalQuadratic"], index=0)
    length_scale_init = st.slider("Kernel length_scale", 0.1,10.0,1.0,0.1)
    matern_nu = st.selectbox("Matern Nu", [0.5,1.5,2.5], index=2)
    alpha_val = st.number_input("GP alpha(noise)", 1e-9, 1.0, 0.01, step=0.01)
    normalize_y= st.checkbox("Normalize GP outputs(Sharpe)?", value=True)

    if kernel_choice=="Matern":
        chosen_kernel = Matern(length_scale=length_scale_init, nu=matern_nu)
    elif kernel_choice=="RationalQuadratic":
        chosen_kernel = RationalQuadratic(length_scale=length_scale_init, alpha=1.0)
    else:
        chosen_kernel = RBF(length_scale=length_scale_init)

    gp_model = GaussianProcessRegressor(
        kernel=chosen_kernel,
        alpha=alpha_val,
        normalize_y=normalize_y,
        random_state=42
    )

    # 5) Button => run gp_minimize
    if not st.button("Run Bayesian Optimization"):
        return pd.DataFrame()

    with st.spinner("Running gp_minimize..."):
        res = gp_minimize(
            objective,
            dimensions=dims,
            base_estimator=gp_model,
            n_calls=n_calls,
            random_state=42,
            callback=[on_step]
        )

    # done => build df
    df_out = pd.DataFrame(tries_list)
    if not df_out.empty:
        best_ = df_out.sort_values("Sharpe Ratio", ascending=False).iloc[0]
        st.write("**Best Found** =>", dict(best_))
        st.dataframe(df_out)

    return df_out