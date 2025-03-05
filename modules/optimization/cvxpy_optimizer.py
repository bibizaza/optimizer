# File: modules/optimization/cvxpy_optimizer.py

import numpy as np
import cvxpy as cp
import pandas as pd

# Local utilities for covariance shrinking
from modules.optimization.utils.cov_shrink import (
    nearest_pd, shrink_cov_diagonal,
    compute_ewm_cov, ledoitwolf_cov
)

# Local utility for mean shrinking
from modules.optimization.utils.mean_shrink import shrink_mean_to_grand_mean


###############################################################################
# 1) Parametric Approach (Frontier scanning => n_points)
###############################################################################
def parametric_max_sharpe_aclass_subtype(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    security_types: list[str],
    class_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float = 0.0,
    no_short: bool = True,
    n_points: int = 15,
    regularize_cov: bool = False,
    shrink_means: bool = False,
    alpha: float = 0.3,
    shrink_cov: bool = False,
    beta: float = 0.2,
    use_ledoitwolf: bool = False,
    do_ewm: bool = False,
    ewm_alpha: float = 0.06
):
    """
    We build many candidate target returns in [targ_min, targ_max].
    For each target, we solve a QP that minimizes vol s.t. return >= target.
    We pick the ex-post best Sharpe. If 'no_short' is True => w>=0.
    Class & subtype constraints are enforced.
    """

    n = len(tickers)
    if df_returns.shape[1] != n:
        raise ValueError("df_returns shape mismatch vs # tickers.")
    if len(asset_classes) != n:
        raise ValueError("asset_classes length mismatch.")
    if len(security_types) != n:
        raise ValueError("security_types length mismatch.")

    # 1) Clean returns => remove Inf/NaN
    df_ret_clean = df_returns.replace([np.inf, -np.inf], np.nan)
    df_ret_clean = df_ret_clean.dropna(axis=1, how='all')
    df_ret_clean = df_ret_clean.dropna(axis=0, how='all').fillna(0.0)
    if df_ret_clean.shape[1] < 1 or df_ret_clean.shape[0] < 2:
        # fallback => eq weights
        best_w = np.ones(n)/n
        summary = {
            "Annual Return (%)": 0.0,
            "Annual Vol (%)": 0.0,
            "Sharpe Ratio": 0.0
        }
        return best_w, summary

    # 2) Covariance
    if do_ewm:
        cov_raw = compute_ewm_cov(df_ret_clean, alpha=ewm_alpha)
    else:
        if use_ledoitwolf:
            cov_raw = ledoitwolf_cov(df_ret_clean)
        else:
            cov_raw = df_ret_clean.cov().values
            if regularize_cov:
                cov_raw = nearest_pd(cov_raw)
            if shrink_cov and beta>0:
                cov_raw = shrink_cov_diagonal(cov_raw, beta)

    # remove leftover Inf/NaN
    cov_raw = np.nan_to_num(cov_raw, nan=0.0, posinf=0.0, neginf=0.0)

    # SHIFT to ensure we are strictly PSD
    SHIFT_EPS = 1e-8
    cov_fixed = cov_raw + SHIFT_EPS*np.eye(n)

    # !!! KEY FIX => use cp.psd_wrap(...) to skip ARPACK PSD check !!!
    P = cp.psd_wrap(cov_fixed)

    # 3) Means
    mean_ret = df_ret_clean.mean().values
    if shrink_means and alpha>0:
        mean_ret = shrink_mean_to_grand_mean(mean_ret, alpha)
    ann_rf = daily_rf * 252

    best_sharpe = -np.inf
    best_w = np.ones(n)/n

    asset_ann_ret = mean_ret*252
    targ_min = max(0.0, asset_ann_ret.min())
    targ_max = asset_ann_ret.max()
    candidate_targets = np.linspace(targ_min, targ_max, n_points)

    for targ in candidate_targets:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, P))  # uses psd_wrap
        cons = [cp.sum(w)==1]
        if no_short:
            cons.append(w>=0)

        # return >= targ
        cons.append( (mean_ret @ w)*252 >= targ )

        # class constraints
        unique_cls = set(asset_classes)
        for cl_ in unique_cls:
            idxs = [i for i,a_ in enumerate(asset_classes) if a_== cl_]
            cdict = class_constraints.get(cl_,{})
            min_class= cdict.get("min_class_weight",0.0)
            max_class= cdict.get("max_class_weight",1.0)
            cons.append(cp.sum(w[idxs])>= min_class)
            cons.append(cp.sum(w[idxs])<= max_class)

            # subtype
            for i_ in idxs:
                stp= security_types[i_]
                if (cl_, stp) in subtype_constraints:
                    stvals= subtype_constraints[(cl_, stp)]
                    mini= stvals.get("min_instrument",0.0)
                    maxi= stvals.get("max_instrument",1.0)
                    cons.append(w[i_]>= mini)
                    cons.append(w[i_]<= maxi)

        prob = cp.Problem(objective, cons)

        solved = False
        for solver_ in [cp.ECOS]:
            try:
                prob.solve(solver=solver_, verbose=False)
                if prob.status in ["optimal","optimal_inaccurate"] and w.value is not None:
                    solved= True
                    break
            except (cp.error.SolverError, cp.error.DCPError):
                pass

        if not solved or w.value is None:
            continue

        w_val = w.value
        vol_ann = float(np.sqrt(w_val.T @ cov_fixed @ w_val)* np.sqrt(252))
        ret_ann = float(mean_ret@ w_val*252)
        if vol_ann<1e-12:
            sr_ = -np.inf
        else:
            sr_ = (ret_ann - ann_rf)/ vol_ann

        if sr_> best_sharpe:
            best_sharpe = sr_
            best_w = w_val.copy()

    final_ret = float(mean_ret@ best_w * 252)
    final_vol = float(np.sqrt(best_w.T@ cov_fixed@ best_w)* np.sqrt(252))
    summary= {
        "Annual Return (%)": round(final_ret,2),
        "Annual Vol (%)": round(final_vol,2),
        "Sharpe Ratio": round(best_sharpe,4)
    }
    return best_w, summary


###############################################################################
# 2) Direct Approach => single solve => no n_points scanning
###############################################################################
def direct_max_sharpe_aclass_subtype(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    security_types: list[str],
    class_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float = 0.0,
    no_short: bool = True,
    regularize_cov: bool = False,
    shrink_means: bool = False,
    alpha: float = 0.3,
    shrink_cov: bool = False,
    beta: float = 0.2,
    use_ledoitwolf: bool = False,
    do_ewm: bool = False,
    ewm_alpha: float = 0.06
):
    """
    Single solve => maximize mean_annual@ w - ann_rf => ex post sharpe is ret/vol
    If no_short => w>=0. Also enforce class/subtype constraints.
    """
    n = len(tickers)
    if df_returns.shape[1] != n:
        raise ValueError("df_returns shape mismatch vs # tickers.")
    if len(asset_classes) != n:
        raise ValueError("asset_classes length mismatch.")
    if len(security_types) != n:
        raise ValueError("security_types length mismatch.")

    # Clean returns
    df_ret_clean = df_returns.replace([np.inf, -np.inf], np.nan)
    df_ret_clean = df_ret_clean.dropna(axis=1, how='all')
    df_ret_clean = df_ret_clean.dropna(axis=0, how='all').fillna(0.0)

    if df_ret_clean.shape[1]<1 or df_ret_clean.shape[0]<2:
        best_w= np.ones(n)/n
        summary= {"Annual Return (%)":0.0,"Annual Vol (%)":0.0,"Sharpe Ratio":0.0}
        return best_w, summary

    # Cov
    if do_ewm:
        cov_raw = compute_ewm_cov(df_ret_clean, alpha=ewm_alpha)
    else:
        if use_ledoitwolf:
            cov_raw = ledoitwolf_cov(df_ret_clean)
        else:
            cov_raw = df_ret_clean.cov().values
            if regularize_cov:
                cov_raw = nearest_pd(cov_raw)
            if shrink_cov and beta>0:
                cov_raw = shrink_cov_diagonal(cov_raw, beta)

    cov_raw = np.nan_to_num(cov_raw, nan=0.0, posinf=0.0, neginf=0.0)
    SHIFT_EPS=1e-8
    cov_fixed= cov_raw + SHIFT_EPS*np.eye(n)

    mean_daily= df_ret_clean.mean().values
    if shrink_means and alpha>0:
        mean_daily= shrink_mean_to_grand_mean(mean_daily, alpha)
    mean_annual= mean_daily*252
    ann_rf= daily_rf*252

    w= cp.Variable(n)
    cons= [cp.sum(w)==1]
    if no_short:
        cons.append(w>=0)

    # class constraints
    unique_cls= set(asset_classes)
    for cl_ in unique_cls:
        idxs= [i for i,a_ in enumerate(asset_classes) if a_==cl_]
        cdict= class_constraints.get(cl_,{})
        min_class= cdict.get("min_class_weight",0.0)
        max_class= cdict.get("max_class_weight",1.0)
        cons.append(cp.sum(w[idxs])>= min_class)
        cons.append(cp.sum(w[idxs])<= max_class)

        for i_ in idxs:
            stp= security_types[i_]
            if (cl_, stp) in subtype_constraints:
                stvals= subtype_constraints[(cl_, stp)]
                mini= stvals.get("min_instrument",0.0)
                maxi= stvals.get("max_instrument",1.0)
                cons.append(w[i_]>= mini)
                cons.append(w[i_]<= maxi)

    obj= cp.Maximize(mean_annual@ w - ann_rf)
    prob= cp.Problem(obj, cons)

    solved= False
    best_w= np.zeros(n)
    for solver_ in [cp.ECOS]:
        try:
            prob.solve(solver=solver_, verbose=False)
            if prob.status in ["optimal","optimal_inaccurate"] and w.value is not None:
                solved= True
                break
        except (cp.error.SolverError, cp.error.DCPError) as e:
            pass

    if not solved or w.value is None:
        summary= {"Annual Return (%)":0.0, "Annual Vol (%)":0.0, "Sharpe Ratio":0.0}
        return best_w, summary

    w_val= w.value
    ret_ann= float(mean_annual@ w_val)
    var_daily= float(w_val.T@ cov_fixed@ w_val)
    vol_ann= np.sqrt(var_daily)* np.sqrt(252)
    sr_= 0.0
    if vol_ann>1e-12:
        sr_= (ret_ann- ann_rf)/ vol_ann

    best_w= w_val.copy()
    summary= {
        "Annual Return (%)": round(ret_ann,2),
        "Annual Vol (%)": round(vol_ann,2),
        "Sharpe Ratio": round(sr_,4)
    }
    return best_w, summary
