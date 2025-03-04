# File: modules/optimization/cvxpy_optimizer.py

import numpy as np
import cvxpy as cp
import pandas as pd

from modules.optimization.utils.cov_shrink import (
    nearest_pd, shrink_cov_diagonal,
    compute_ewm_cov, ledoitwolf_cov
)
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
    1) We build many candidate target returns in [targ_min, targ_max].
    2) For each, we solve a QP that minimizes vol subject to return >= target.
    3) We pick the solution with best ex-post Sharpe.
    4) We skip scs or reorder solver preference => prefer ECOS, skip SCS if memory fails.
    """
    if security_types is None:
        security_types = ["Unknown"] * df_returns.shape[1]

    n = len(tickers)
    if df_returns.shape[1] != n:
        raise ValueError("df_returns shape mismatch with number of tickers.")
    if len(asset_classes) != n:
        raise ValueError("asset_classes length mismatch.")
    if len(security_types) != n:
        raise ValueError("security_types length mismatch.")

    # Build covariance
    if do_ewm:
        cov_raw = compute_ewm_cov(df_returns, alpha=ewm_alpha)
    else:
        if use_ledoitwolf:
            cov_raw = ledoitwolf_cov(df_returns)
        else:
            cov_raw = df_returns.cov().values
            if regularize_cov:
                cov_raw = nearest_pd(cov_raw)
            if shrink_cov and beta > 0:
                cov_raw = shrink_cov_diagonal(cov_raw, beta)

    SHIFT_EPS = 1e-8
    cov_fixed = cov_raw + SHIFT_EPS * np.eye(n)
    cov_expr = cp.psd_wrap(cov_fixed)

    # Mean returns
    mean_ret = df_returns.mean().values
    if shrink_means and alpha > 0:
        mean_ret = shrink_mean_to_grand_mean(mean_ret, alpha)

    ann_rf = daily_rf * 252
    best_sharpe = -np.inf
    best_w = np.ones(n) / n

    # Build candidate target returns in [targ_min, targ_max]
    asset_ann_ret = mean_ret * 252
    targ_min = max(0.0, asset_ann_ret.min())
    targ_max = asset_ann_ret.max()
    candidate_targets = np.linspace(targ_min, targ_max, n_points)

    for targ in candidate_targets:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov_expr))
        constraints = [cp.sum(w) == 1]
        if no_short:
            constraints.append(w >= 0)

        # Return constraint => (mean_ret @ w)*252 >= targ
        constraints.append((mean_ret @ w)*252 >= targ)

        # Class-level constraints
        unique_cls = set(asset_classes)
        for cl in unique_cls:
            idxs = [i for i,a_ in enumerate(asset_classes) if a_ == cl]
            cdict = class_constraints.get(cl, {})
            min_class = cdict.get("min_class_weight", 0.0)
            max_class = cdict.get("max_class_weight", 1.0)
            constraints.append(cp.sum(w[idxs]) >= min_class)
            constraints.append(cp.sum(w[idxs]) <= max_class)

            # Subtype constraints
            for i_ in idxs:
                stp = security_types[i_]
                if (cl, stp) in subtype_constraints:
                    stvals = subtype_constraints[(cl, stp)]
                    min_inst = stvals.get("min_instrument", 0.0)
                    max_inst = stvals.get("max_instrument", 1.0)
                    constraints.append(w[i_] >= min_inst)
                    constraints.append(w[i_] <= max_inst)

        prob = cp.Problem(objective, constraints)

        # Instead of [cp.SCS, cp.ECOS], let's prefer just ECOS or reorder
        solved = False
        for solver_ in [cp.ECOS]:
            try:
                prob.solve(solver=solver_, verbose=False)
                if prob.status in ["optimal","optimal_inaccurate"] and w.value is not None:
                    solved = True
                    break
            except (cp.error.SolverError, cp.error.DCPError) as e:
                pass

        if not solved:
            # skip this target if solver fails
            continue

        w_val = w.value
        vol_ann = np.sqrt(w_val.T @ cov_fixed @ w_val) * np.sqrt(252)
        ret_ann = (mean_ret @ w_val)*252
        sr_ = (ret_ann - ann_rf)/(vol_ann+1e-12) if vol_ann>1e-12 else -np.inf
        if sr_ > best_sharpe:
            best_sharpe = sr_
            best_w = w_val.copy()

    final_ret = (mean_ret @ best_w)*252
    final_vol = np.sqrt(best_w.T @ cov_fixed @ best_w)* np.sqrt(252)
    summary = {
        "Annual Return (%)": round(final_ret, 2),
        "Annual Vol (%)": round(final_vol, 2),
        "Sharpe Ratio": round(best_sharpe, 4)
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
    We do a single solve => Maximize mean_annual@w - ann_rf => ex post we get Sharpe = ...
    Reordered solver preference => prefer ECOS, skip SCS if memory errors arise
    """
    if security_types is None:
        security_types = ["Unknown"] * df_returns.shape[1]

    n = len(tickers)
    if df_returns.shape[1] != n:
        raise ValueError("df_returns shape mismatch with number of tickers.")
    if len(asset_classes) != n:
        raise ValueError("asset_classes length mismatch.")
    if len(security_types) != n:
        raise ValueError("security_types length mismatch.")

    # Cov
    if do_ewm:
        cov_raw = compute_ewm_cov(df_returns, alpha=ewm_alpha)
    else:
        if use_ledoitwolf:
            cov_raw = ledoitwolf_cov(df_returns)
        else:
            cov_raw = df_returns.cov().values
            if regularize_cov:
                cov_raw = nearest_pd(cov_raw)
            if shrink_cov and beta>0:
                cov_raw = shrink_cov_diagonal(cov_raw, beta)

    SHIFT_EPS = 1e-8
    cov_fixed = cov_raw + SHIFT_EPS*np.eye(n)

    mean_daily = df_returns.mean().values
    if shrink_means and alpha>0:
        mean_daily = shrink_mean_to_grand_mean(mean_daily, alpha)
    mean_annual = mean_daily* 252
    ann_rf = daily_rf* 252

    w= cp.Variable(n)
    constr= [cp.sum(w)==1]
    if no_short:
        constr.append(w>=0)

    # Class constraints
    unique_cls= set(asset_classes)
    for cl_ in unique_cls:
        idxs= [i for i,a_ in enumerate(asset_classes) if a_== cl_]
        cdict= class_constraints.get(cl_, {})
        min_class= cdict.get("min_class_weight",0.0)
        max_class= cdict.get("max_class_weight",1.0)
        constr.append(cp.sum(w[idxs])>= min_class)
        constr.append(cp.sum(w[idxs])<= max_class)

        # subType constraints
        for i_ in idxs:
            stp= security_types[i_]
            if (cl_, stp) in subtype_constraints:
                stvals= subtype_constraints[(cl_, stp)]
                mini= stvals.get("min_instrument",0.0)
                maxi= stvals.get("max_instrument",1.0)
                constr.append(w[i_]>= mini)
                constr.append(w[i_]<= maxi)

    # Maximize => mean_annual@ w - ann_rf
    obj= cp.Maximize(mean_annual@ w - ann_rf)
    prob= cp.Problem(obj, constr)

    solved= False
    best_w= np.zeros(n)
    # Only try ECOS first => skip SCS
    for solver_ in [cp.ECOS]:
        try:
            prob.solve(solver=solver_, verbose=False)
            if prob.status in ["optimal","optimal_inaccurate"] and w.value is not None:
                solved= True
                break
        except (cp.error.SolverError, cp.error.DCPError) as e:
            pass

    if not solved:
        # Return zeros
        summary= {
            "Annual Return (%)": 0.0,
            "Annual Vol (%)": 0.0,
            "Sharpe Ratio": 0.0
        }
        return best_w, summary

    w_val= w.value
    ret_ann= mean_annual@ w_val
    var_daily= w_val.T@ cov_fixed@ w_val
    vol_ann= np.sqrt(var_daily)* np.sqrt(252)
    sr_= -np.inf
    if vol_ann>1e-12:
        sr_= (ret_ann- ann_rf)/ vol_ann

    best_w= w_val.copy()
    summary= {
        "Annual Return (%)": round(ret_ann,2),
        "Annual Vol (%)": round(vol_ann,2),
        "Sharpe Ratio": round(sr_,4)
    }
    return best_w, summary