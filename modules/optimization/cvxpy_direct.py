# File: modules/optimization/cvxpy_direct.py

import numpy as np
import cvxpy as cp
import pandas as pd

from modules.optimization.utils.cov_utils import build_covariance_matrix
from modules.optimization.utils.mean_shrink import (
    shrink_mean_to_grand_mean,
    shrink_mean_to_zero  # <-- newly imported
)

def direct_max_sharpe_aclass_subtype(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    security_types: list[str],
    class_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float = 0.0,
    no_short: bool = True,

    # --- Covariance Estimator & Shrinkage ---
    cov_estimator: str = "sample",        # "sample", "ewma", or "dcc_garch"
    shrinkage: str = "none",              # "none", "diagonal", "ledoitwolf"
    ewm_alpha: float = 0.06,
    diag_shrink_beta: float = 0.2,
    regularize_cov: bool = False,
    nearest_pd_epsilon: float = 1e-6,

    # DCC-GARCH params
    garch_p: int = 1,
    garch_q: int = 1,
    garch_dist: str = "normal",
    dcc_alpha: float = 0.05,
    dcc_beta: float = 0.90,

    # NEW: mean_tech, alpha => replace old "shrink_means" bool
    mean_tech: str = "none",               # "none", "shrink_to_grand_mean", "shrink_to_zero"
    alpha_mean_shrink: float = 0.0,
):
    """
    Single-step approach => maximize portfolio's expected return minus rf,
    i.e. Maximize(mean_annual@w - ann_rf). We measure ex-post Sharpe as
    (ret - rf) / vol.
    
    The covariance is built by calling build_covariance_matrix(...).
    We currently do NOT integrate that covariance into the objective 
    (we said we'll do that later). For now, we only do performance screening.

    Returns:
      best_w   : np.array of final weights
      summary  : dict with {"Annual Return (%)","Annual Vol (%)","Sharpe Ratio"}
    """

    n = len(tickers)
    if df_returns.shape[1] != n:
        raise ValueError("df_returns shape mismatch vs # tickers.")
    if len(asset_classes) != n:
        raise ValueError("asset_classes length mismatch.")
    if len(security_types) != n:
        raise ValueError("security_types length mismatch.")

    # 1) Clean returns => remove Inf/NaN
    df_ret_clean = df_returns.replace([np.inf, -np.inf], np.nan).dropna(how='all', axis=1)
    df_ret_clean = df_ret_clean.dropna(how='all', axis=0).fillna(0.0)
    if df_ret_clean.shape[1] < 1 or df_ret_clean.shape[0] < 2:
        # fallback => eq weights
        best_w = np.ones(n) / max(n, 1)
        return best_w, {
            "Annual Return (%)": 0.0,
            "Annual Vol (%)": 0.0,
            "Sharpe Ratio": 0.0
        }

    # 2) Build Covariance (we do not use it in the objective yet, 
    #    but do it for consistent usage)
    cov_raw = build_covariance_matrix(
        df_returns=df_ret_clean,
        estimator=cov_estimator,
        shrinkage=shrinkage,
        ewm_alpha=ewm_alpha,
        diag_shrink_beta=diag_shrink_beta,
        regularize_cov=regularize_cov,
        nearest_pd_epsilon=nearest_pd_epsilon,
        garch_p=garch_p,
        garch_q=garch_q,
        garch_dist=garch_dist,
        dcc_alpha=dcc_alpha,
        dcc_beta=dcc_beta
    )
    SHIFT_EPS = 1e-8
    cov_fixed = cov_raw + SHIFT_EPS * np.eye(n)

    # 3) Means
    mean_daily = df_ret_clean.mean().values
    if mean_tech == "shrink_to_grand_mean" and alpha_mean_shrink > 0:
        mean_daily = shrink_mean_to_grand_mean(mean_daily, alpha_mean_shrink)
    elif mean_tech == "shrink_to_zero" and alpha_mean_shrink > 0:
        mean_daily = shrink_mean_to_zero(mean_daily, alpha_mean_shrink)

    mean_annual = mean_daily * 252
    ann_rf = daily_rf * 252

    # 4) Single-step CVX: maximize (mean_annual@w - ann_rf)
    w = cp.Variable(n)
    objective = cp.Maximize(mean_annual @ w - ann_rf)
    cons = [cp.sum(w) == 1]
    if no_short:
        cons.append(w >= 0)

    # Class constraints
    unique_cls = set(asset_classes)
    for cls_ in unique_cls:
        idxs = [i for i, a_ in enumerate(asset_classes) if a_ == cls_]
        cdict = class_constraints.get(cls_, {})
        min_cls = cdict.get("min_class_weight", 0.0)
        max_cls = cdict.get("max_class_weight", 1.0)
        cons.append(cp.sum(w[idxs]) >= min_cls)
        cons.append(cp.sum(w[idxs]) <= max_cls)

        # Subtype constraints
        for i_ in idxs:
            stp = security_types[i_]
            if (cls_, stp) in subtype_constraints:
                stvals = subtype_constraints[(cls_, stp)]
                mini = stvals.get("min_instrument", 0.0)
                maxi = stvals.get("max_instrument", 1.0)
                cons.append(w[i_] >= mini)
                cons.append(w[i_] <= maxi)

    prob = cp.Problem(objective, cons)
    solved = False
    best_w = np.zeros(n)

    # 5) Solve
    for solver_ in [cp.ECOS, cp.OSQP]:
        try:
            prob.solve(solver=solver_, verbose=False)
            if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                solved = True
                break
        except (cp.error.SolverError, cp.error.DCPError):
            pass

    if not solved or w.value is None:
        # fallback => zero weights
        summary = {"Annual Return (%)": 0.0, "Annual Vol (%)": 0.0, "Sharpe Ratio": 0.0}
        return best_w, summary

    # 6) Evaluate ex-post metrics
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
