# File: modules/optimization/cvxpy_direct.py

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
    Single-step approach => directly maximize the portfolio's (mean_annual - rf).
    Ex-post Sharpe is then (ret / vol). If no_short => w >= 0. Class & subtype
    constraints are enforced.

    Returns:
      best_w   : np.array of final weights
      summary  : dict with {'Annual Return (%)', 'Annual Vol (%)', 'Sharpe Ratio'}
    """

    n = len(tickers)
    if df_returns.shape[1] != n:
        raise ValueError("df_returns shape mismatch vs # tickers.")
    if len(asset_classes) != n:
        raise ValueError("asset_classes length mismatch.")
    if len(security_types) != n:
        raise ValueError("security_types length mismatch.")

    # 1) Clean returns
    df_ret_clean = df_returns.replace([np.inf, -np.inf], np.nan)
    df_ret_clean = df_ret_clean.dropna(axis=1, how='all')
    df_ret_clean = df_ret_clean.dropna(axis=0, how='all').fillna(0.0)
    if df_ret_clean.shape[1] < 1 or df_ret_clean.shape[0] < 2:
        best_w = np.ones(n) / max(n,1)
        return best_w, {
            "Annual Return (%)": 0.0,
            "Annual Vol (%)": 0.0,
            "Sharpe Ratio": 0.0
        }

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
            if shrink_cov and beta > 0:
                cov_raw = shrink_cov_diagonal(cov_raw, beta)

    cov_raw = np.nan_to_num(cov_raw, nan=0.0, posinf=0.0, neginf=0.0)
    SHIFT_EPS = 1e-8
    cov_fixed = cov_raw + SHIFT_EPS * np.eye(n)

    # 3) Means
    mean_daily = df_ret_clean.mean().values
    if shrink_means and alpha > 0:
        mean_daily = shrink_mean_to_grand_mean(mean_daily, alpha)
    mean_annual = mean_daily * 252
    ann_rf = daily_rf * 252

    # 4) Build CVXPY problem => maximize (mean_annual @ w - ann_rf)
    w = cp.Variable(n)
    cons = [cp.sum(w) == 1]
    if no_short:
        cons.append(w >= 0)

    # Class constraints
    unique_cls = set(asset_classes)
    for cl_ in unique_cls:
        idxs = [i for i, a_ in enumerate(asset_classes) if a_ == cl_]
        cdict = class_constraints.get(cl_, {})
        min_class = cdict.get("min_class_weight", 0.0)
        max_class = cdict.get("max_class_weight", 1.0)
        cons.append(cp.sum(w[idxs]) >= min_class)
        cons.append(cp.sum(w[idxs]) <= max_class)

        for i_ in idxs:
            stp = security_types[i_]
            if (cl_, stp) in subtype_constraints:
                stvals = subtype_constraints[(cl_, stp)]
                mini = stvals.get("min_instrument", 0.0)
                maxi = stvals.get("max_instrument", 1.0)
                cons.append(w[i_] >= mini)
                cons.append(w[i_] <= maxi)

    obj = cp.Maximize(mean_annual @ w - ann_rf)
    prob = cp.Problem(obj, cons)

    solved = False
    best_w = np.zeros(n)
    for solver_ in [cp.ECOS]:
        try:
            prob.solve(solver=solver_, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                solved = True
                break
        except (cp.error.SolverError, cp.error.DCPError):
            pass

    if not solved or w.value is None:
        # fallback => zero or eq weight
        summary = {"Annual Return (%)": 0.0, "Annual Vol (%)": 0.0, "Sharpe Ratio": 0.0}
        return best_w, summary

    w_val = w.value
    ret_ann = float(mean_annual @ w_val)
    var_daily = float(w_val.T @ cov_fixed @ w_val)
    vol_ann = float(np.sqrt(var_daily) * np.sqrt(252))
    sr_ = 0.0
    if vol_ann > 1e-12:
        sr_ = (ret_ann - ann_rf) / vol_ann

    best_w = w_val.copy()
    summary = {
        "Annual Return (%)": round(ret_ann, 2),
        "Annual Vol (%)": round(vol_ann, 2),
        "Sharpe Ratio": round(sr_, 4)
    }
    return best_w, summary