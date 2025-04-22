# File: rolling_bayesian.py

import streamlit as st
import pandas as pd
import numpy as np
import time

# scikit-optimize
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
from skopt.learning import GaussianProcessRegressor
from skopt.learning.gaussian_process.kernels import (Matern, RBF, RationalQuadratic)

# Rolling monthly backtests (they call compute_extended_metrics at the end)
from modules.backtesting.rolling_monthly import (
    rolling_backtest_monthly_param_sharpe,
    rolling_backtest_monthly_direct_sharpe
)

# Param / Direct solver => returns some summary not used for final line stats
from modules.optimization.cvxpy_parametric import parametric_max_sharpe_aclass_subtype
from modules.optimization.cvxpy_direct import direct_max_sharpe_aclass_subtype

###############################################################################
# 1) The single-run helper => run_one_combo
###############################################################################
def run_one_combo(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    asset_cls_list: list[str],
    sec_type_list: list[str],
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float,

    use_direct_solver: bool,
    n_points: int|None,  # used if param approach

    cov_estimator: str,  # "sample","ewma","dcc_garch"
    ewm_alpha: float|None,
    dcc_dist: str|None,  # "normal" or "t"
    shrinkage: str,      # "none","diagonal","ledoitwolf"
    diag_beta: float|None,

    # NOW includes "shrink_to_zero" as well:
    mean_tech: str,      # "none","grand_mean","shrink_to_zero"
    mean_alpha: float|None,

    rebal_freq: int,
    lookback_m: int,

    transaction_cost_value: float,
    transaction_cost_type: str,
    trade_buffer_pct: float
) -> dict:
    """
    1) If param => define param_sharpe_fn => rolling_backtest_monthly_param_sharpe
    2) If direct => define direct_sharpe_fn => rolling_backtest_monthly_direct_sharpe
    Then read the final extended metrics => "CAGR","Volatility","Sharpe" => rename them.
    """

    col_tickers = df_prices.columns.tolist()

    # -----------------------------------------------------------
    # Mean shrink logic => handle "none", "grand_mean", "shrink_to_zero"
    # -----------------------------------------------------------
    if mean_tech == "none":
        alpha_mean_shr = 0.0
    else:
        # either "grand_mean" or "shrink_to_zero" => use mean_alpha if present
        alpha_mean_shr = mean_alpha if mean_alpha else 0.0

    # Diagonal beta if shrinkage="diagonal"
    diag_shrink_b = diag_beta if (shrinkage == "diagonal") else 0.0

    # =========================================
    # ============== PARAMETRIC ==============
    # =========================================
    if not use_direct_solver:
        def param_sharpe_fn(sub_ret: pd.DataFrame):
            w_opt, summary = parametric_max_sharpe_aclass_subtype(
                df_returns=sub_ret,
                tickers=col_tickers,
                asset_classes=asset_cls_list,
                security_types=sec_type_list,
                class_constraints=class_sum_constraints,
                subtype_constraints=subtype_constraints,
                daily_rf=daily_rf,
                no_short=True,
                n_points=n_points if n_points else 15,

                # Cov + shrink
                cov_estimator=cov_estimator,
                ewm_alpha=ewm_alpha if ewm_alpha else 0.06,
                garch_dist=dcc_dist if dcc_dist else "normal",
                shrinkage=shrinkage,
                diag_shrink_beta=diag_shrink_b,

                # NEW: pass method name & alpha
                mean_tech=mean_tech,
                alpha_mean_shrink=alpha_mean_shr,
            )
            return w_opt, summary

        sr_line, final_w, _, _, _, ext_metrics = rolling_backtest_monthly_param_sharpe(
            df_prices=df_prices,
            df_instruments=df_instruments,
            param_sharpe_fn=param_sharpe_fn,
            start_date=df_prices.index[0],
            end_date=df_prices.index[-1],
            months_interval=rebal_freq,
            window_days=lookback_m * 21,
            transaction_cost_value=transaction_cost_value,
            transaction_cost_type=transaction_cost_type,
            trade_buffer_pct=trade_buffer_pct,
            daily_rf=daily_rf
        )

    # ========================================
    # ================ DIRECT ================
    # ========================================
    else:
        def direct_sharpe_fn(sub_ret: pd.DataFrame):
            w_opt, summary = direct_max_sharpe_aclass_subtype(
                df_returns=sub_ret,
                tickers=col_tickers,
                asset_classes=asset_cls_list,
                security_types=sec_type_list,
                class_constraints=class_sum_constraints,
                subtype_constraints=subtype_constraints,
                daily_rf=daily_rf,
                no_short=True,

                cov_estimator=cov_estimator,
                ewm_alpha=ewm_alpha if ewm_alpha else 0.06,
                garch_dist=dcc_dist if dcc_dist else "normal",
                shrinkage=shrinkage,
                diag_shrink_beta=diag_shrink_b,

                # NEW: pass method name & alpha
                mean_tech=mean_tech,
                alpha_mean_shrink=alpha_mean_shr,
            )
            return w_opt, summary

        sr_line, final_w, _, _, _, ext_metrics = rolling_backtest_monthly_direct_sharpe(
            df_prices=df_prices,
            df_instruments=df_instruments,
            direct_sharpe_fn=direct_sharpe_fn,
            start_date=df_prices.index[0],
            end_date=df_prices.index[-1],
            months_interval=rebal_freq,
            window_days=lookback_m * 21,
            transaction_cost_value=transaction_cost_value,
            transaction_cost_type=transaction_cost_type,
            trade_buffer_pct=trade_buffer_pct,
            daily_rf=daily_rf
        )

    # If no ext_metrics => fallback
    if not ext_metrics:
        return {
            "Sharpe Ratio": 0.0,
            "Annual Ret":   0.0,
            "Annual Vol":   0.0
        }

    # Now read the keys from compute_extended_metrics(...) 
    # e.g. "Sharpe","CAGR","Volatility"
    sharpe_val = ext_metrics.get("Sharpe", 0.0)
    ann_ret    = ext_metrics.get("CAGR",   0.0)
    ann_vol    = ext_metrics.get("Volatility", 0.0)

    return {
        "Sharpe Ratio": sharpe_val,
        "Annual Ret":   ann_ret,
        "Annual Vol":   ann_vol
    }

###############################################################################
# 2) The main Bayesian function
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
    We define dimension search space => pass combos to run_one_combo => each 
    returns "Sharpe Ratio","Annual Ret","Annual Vol" from final line-based approach 
    (via ext_metrics in rolling_backtest).
    """

    st.title("Bayesian Optimization (Line-based final metrics)")

    # 1) Solver approach
    solver_choice = st.selectbox(
        "Solver Approach", 
        ["Parametric (Markowitz)","Direct (Markowitz)"],
        index=0
    )
    use_direct_solver = (solver_choice=="Direct (Markowitz)")

    # 2) n_calls
    n_calls = st.number_input("Number of Bayesian evaluations", 5, 500, 20, step=5)

    # If param => dimension for n_points
    n_points_dim = None
    if not use_direct_solver:
        st.subheader("Param => n_points")
        param_mode = st.radio("Frontier Points Setting", ["Fixed","Range"], index=0)
        if param_mode=="Fixed":
            fixed_npoints = st.number_input("n_points(frontier)",1,999,15,step=1)
            n_points_dim = Categorical([fixed_npoints], name="n_points_")
        else:
            col1, col2 = st.columns(2)
            with col1:
                min_npt = st.number_input("Min n_points",1,999,10,step=1)
            with col2:
                max_npt = st.number_input("Max n_points",1,999,25,step=1)
            n_points_dim = Integer(int(min_npt), int(max_npt), name="n_points_")

    # Cov estimator => multi
    st.subheader("Cov Estimator (multi)")
    cov_est_choices = st.multiselect(
        "Pick from sample, ewma, dcc_garch, MCD",
        ["sample","ewma","dcc_garch","mcd"],
        default=["sample"]
    )
    if not cov_est_choices:
        cov_est_choices=["sample"]
    cov_est_dim = Categorical(cov_est_choices, name="cov_est_")

    # If "ewma" => dimension for ewm_alpha
    ewm_alpha_dim = None
    if "ewma" in cov_est_choices:
        st.write("EWM alpha range")
        colA, colB= st.columns(2)
        with colA:
            ewm_min= st.slider("ewm alpha min",0.0,1.0,0.05,0.01)
        with colB:
            ewm_max= st.slider("ewm alpha max",0.0,1.0,0.3,0.01)
        ewm_alpha_dim= Real(ewm_min, ewm_max, name="ewm_alpha_")

    # If "dcc_garch" => dimension for distribution
    dcc_dist_dim= None
    if "dcc_garch" in cov_est_choices:
        st.write("DCC GARCH => normal|t")
        dlist= st.multiselect("Pick dist(s)", ["normal","t"], ["normal"])
        if not dlist:
            dlist=["normal"]
        dcc_dist_dim= Categorical(dlist, name="dcc_dist_")

    # Cov Improvements => multi
    st.subheader("Cov Improvements")
    shrink_choices = st.multiselect(
        "Pick from none, diagonal, ledoitwolf, oas",
        ["none", "diagonal", "ledoitwolf","oas"],
        ["none"]
    )
    if not shrink_choices:
        shrink_choices = ["none"]
    shrink_dim = Categorical(shrink_choices, name="shrinkage_")

    diag_dim = None
    if "diagonal" in shrink_choices:
        st.write("Diagonal shrink beta range")
        dd1, dd2 = st.columns(2)
        with dd1:
            diag_min = st.slider("diag beta min", 0.0, 1.0, 0.0, 0.05)
        with dd2:
            diag_max = st.slider("diag beta max", 0.0, 1.0, 0.5, 0.05)
        diag_dim = Real(diag_min, diag_max, name="diag_beta_")

    # Mean Improvements => multi
    st.subheader("Mean Improvements")
    mean_choices = st.multiselect(
        "Pick from none, grand_mean, shrink_to_zero",
        ["none", "grand_mean", "shrink_to_zero"],
        ["none"]
    )
    mean_dim = Categorical(mean_choices, name="mean_tech_")

    mean_alpha_dim = None
    # We show alpha slider if user picks "grand_mean" OR "shrink_to_zero":
    if "grand_mean" in mean_choices or "shrink_to_zero" in mean_choices:
        st.write("Mean alpha range")
        mm1, mm2 = st.columns(2)
        with mm1:
            alpha_min = st.slider("mean alpha min", 0.0, 1.0, 0.0, 0.05)
        with mm2:
            alpha_max = st.slider("mean alpha max", 0.0, 1.0, 0.3, 0.05)
        mean_alpha_dim = Real(alpha_min, alpha_max, name="mean_alpha_")

    # Rebalance freq
    st.subheader("Rebalance freq (months)")
    freq_list= st.multiselect("Possible freq(s)", [1,3,6],[1,3])
    if not freq_list:
        freq_list=[1]
    freq_dim= Categorical(freq_list, name="freq_")

    # Lookback
    st.subheader("Lookback (months)")
    lb_list= st.multiselect("Possible lookbacks", [3,6,12],[3,6])
    if not lb_list:
        lb_list=[3]
    lb_dim= Categorical(lb_list, name="lookback_")

    # Build dimension
    dims=[]
    if (not use_direct_solver) and (n_points_dim is not None):
        dims.append(n_points_dim)
    dims.append(cov_est_dim)
    if ewm_alpha_dim is not None:
        dims.append(ewm_alpha_dim)
    if dcc_dist_dim is not None:
        dims.append(dcc_dist_dim)
    dims.append(shrink_dim)
    if diag_dim is not None:
        dims.append(diag_dim)
    dims.append(mean_dim)
    if mean_alpha_dim is not None:
        dims.append(mean_alpha_dim)
    dims.append(freq_dim)
    dims.append(lb_dim)

    # Gaussian Process config
    st.subheader("Gaussian Process Settings")
    kernel_choice= st.selectbox("Kernel",["Matern","RBF","RationalQuadratic"],0)
    length_scale_init= st.slider("length_scale",0.1,10.0,1.0,0.1)
    alpha_gp= st.number_input("GP alpha (noise)",1e-9,1.0,0.01,0.01)
    normalize_y= st.checkbox("Normalize Y?", True)

    if kernel_choice=="Matern":
        matern_nu= st.selectbox("Matern nu",[0.5,1.5,2.5],2)
        chosen_kernel= Matern(length_scale=length_scale_init, nu=matern_nu)
    elif kernel_choice=="RBF":
        chosen_kernel= RBF(length_scale=length_scale_init)
    else:
        chosen_kernel= RationalQuadratic(length_scale=length_scale_init, alpha=1.0)

    gp_model= GaussianProcessRegressor(
        kernel= chosen_kernel,
        alpha= alpha_gp,
        normalize_y= normalize_y,
        random_state=42
    )

    tries_list=[]
    start_time= time.time()
    progress_bar= st.progress(0)
    progress_txt= st.empty()

    def on_step(res):
        done= len(res.x_iters)
        pct= int(done*100/n_calls)
        elapsed= time.time()- start_time
        progress_txt.text(f"Progress: {pct}%, Elapsed: {elapsed:.1f}s")
        progress_bar.progress(pct)

    def objective_func(x):
        idx=0
        if not use_direct_solver:
            n_points_= x[idx]
            idx+=1
        else:
            n_points_= None

        cov_est_= x[idx]; idx+=1
        ewm_alpha_= None
        if ewm_alpha_dim is not None:
            ewm_alpha_= x[idx]
            idx+=1

        dcc_dist_= None
        if dcc_dist_dim is not None:
            dcc_dist_= x[idx]
            idx+=1

        shrink_= x[idx]; idx+=1
        diag_beta_= None
        if diag_dim is not None:
            diag_beta_= x[idx]
            idx+=1

        mean_tech_= x[idx]; idx+=1
        mean_alpha_= None
        if mean_alpha_dim is not None:
            mean_alpha_= x[idx]
            idx+=1

        freq_= x[idx]; idx+=1
        lb_= x[idx]; idx+=1

        # Now call run_one_combo => returns "Sharpe Ratio","Annual Ret","Annual Vol"
        result= run_one_combo(
            df_prices=df_prices,
            df_instruments=df_instruments,
            asset_cls_list=asset_cls_list,
            sec_type_list=sec_type_list,
            class_sum_constraints=class_sum_constraints,
            subtype_constraints=subtype_constraints,
            daily_rf= daily_rf,

            use_direct_solver= use_direct_solver,
            n_points= n_points_,

            cov_estimator= cov_est_,
            ewm_alpha= ewm_alpha_,
            dcc_dist= dcc_dist_,
            shrinkage= shrink_,
            diag_beta= diag_beta_,

            mean_tech= mean_tech_,
            mean_alpha= mean_alpha_,

            rebal_freq= freq_,
            lookback_m= lb_,

            transaction_cost_value= transaction_cost_value,
            transaction_cost_type= transaction_cost_type,
            trade_buffer_pct= trade_buffer_pct
        )

        sr_val= result["Sharpe Ratio"]
        tries_list.append({
            "n_points": n_points_,
            "cov_estimator": cov_est_,
            "ewm_alpha": ewm_alpha_,
            "dcc_dist": dcc_dist_,
            "shrinkage": shrink_,
            "diag_beta": diag_beta_,
            "mean_tech": mean_tech_,
            "mean_alpha": mean_alpha_,
            "rebal_freq": freq_,
            "lookback": lb_,
            "Sharpe Ratio": sr_val,
            "Annual Ret": result["Annual Ret"],
            "Annual Vol": result["Annual Vol"]
        })
        if np.isnan(sr_val) or np.isinf(sr_val):
            return 1e6
        return -sr_val

    if st.button("Run Bayesian Optimization"):
        with st.spinner("Running Bayesian..."):
            res= gp_minimize(
                func= objective_func,
                dimensions= dims,
                base_estimator= gp_model,
                n_calls= n_calls,
                random_state=42,
                callback=[on_step]
            )
        df_res= pd.DataFrame(tries_list)
        if not df_res.empty:
            best_= df_res.sort_values("Sharpe Ratio", ascending=False).iloc[0]
            st.write("**Best Found** =>", dict(best_))
            st.dataframe(df_res)
            return df_res
        else:
            st.warning("No results recorded.")
            return pd.DataFrame()
    else:
        return pd.DataFrame()
