# modules/optimization/cvxpy_optimizer.py

import numpy as np
import cvxpy as cp
import pandas as pd

from modules.optimization.utils.cov_shrink import (
    nearest_pd, shrink_cov_diagonal,
    compute_ewm_cov, ledoitwolf_cov
)
from modules.optimization.utils.mean_shrink import shrink_mean_to_grand_mean

def parametric_max_sharpe(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    class_constraints: dict,
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
    ewm_alpha: float = 0.06,
    security_types: list[str] = None
):
    """
    A two-layer constraint approach:
      Layer 1) For each asset class => sum(w_class) in [min_class_weight, max_class_weight].
      Layer 2) If class_constraints[cl].get("security_types"), for each stype => sum(w_of_that_stype_in_that_class).

    No references to min_instrument_weight or max_instrument_weight.
    """
    n = len(tickers)
    if df_returns.shape[1] != n:
        raise ValueError("df_returns shape mismatch.")
    if len(asset_classes) != n:
        raise ValueError("asset_classes mismatch.")
    if security_types is None:
        # If you have no #Security_Type column, fill with "Unknown"
        security_types = ["Unknown"]*n

    # 1) Build covariance
    if do_ewm:
        cov_raw = compute_ewm_cov(df_returns, ewm_alpha)
    else:
        if use_ledoitwolf:
            cov_raw = ledoitwolf_cov(df_returns)
        else:
            cov_raw = df_returns.cov().values
            if regularize_cov:
                cov_raw = nearest_pd(cov_raw)
            if shrink_cov:
                cov_raw = shrink_cov_diagonal(cov_raw, beta)

    SHIFT_EPS = 1e-8
    cov_fixed = cov_raw + SHIFT_EPS*np.eye(n)
    cov_expr = cp.psd_wrap(cov_fixed)

    # 2) possibly shrink means
    mean_ret = df_returns.mean().values
    if shrink_means and alpha>0:
        mean_ret = shrink_mean_to_grand_mean(mean_ret, alpha)

    ann_rf = daily_rf*252
    best_sharpe = -np.inf
    best_weights = np.ones(n)/n

    # param approach => loop over candidate return targets
    asset_ann_ret = mean_ret*252
    target_min = max(0.0, asset_ann_ret.min())
    target_max = asset_ann_ret.max()
    candidate_targets = np.linspace(target_min, target_max, n_points)

    for targ in candidate_targets:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov_expr))
        constraints = [cp.sum(w)==1]
        if no_short:
            constraints.append(w>=0)

        # target return
        constraints.append( (mean_ret @ w)*252 >= targ )

        # for each class => sum(w_of_that_class) in [min_class_weight, max_class_weight]
        unique_classes = set(asset_classes)
        for cl_ in unique_classes:
            idxs = [i for i,a_ in enumerate(asset_classes) if a_==cl_]
            cc = class_constraints.get(cl_, {})
            if "min_class_weight" in cc:
                constraints.append(cp.sum(w[idxs]) >= cc["min_class_weight"])
            if "max_class_weight" in cc:
                constraints.append(cp.sum(w[idxs]) <= cc["max_class_weight"])

            # second layer => security_types dict
            if "security_types" in cc:
                st_dict = cc["security_types"]
                # example => { "Stock": {"min_weight":0.01, "max_weight":0.02}, ...}
                for stype_name, stype_values in st_dict.items():
                    # gather instruments in idxs with security_types[i]==stype_name
                    stype_idxs = [i for i in idxs if security_types[i]==stype_name]
                    if not stype_idxs:
                        continue
                    if "min_weight" in stype_values:
                        constraints.append(cp.sum(w[stype_idxs]) >= stype_values["min_weight"])
                    if "max_weight" in stype_values:
                        constraints.append(cp.sum(w[stype_idxs]) <= stype_values["max_weight"])

        # solve
        prob = cp.Problem(objective, constraints)
        success=False
        for solver_ in [cp.SCS, cp.ECOS]:
            try:
                prob.solve(solver=solver_, verbose=False)
                if prob.status in ["optimal","optimal_inaccurate"] and w.value is not None:
                    success=True
                    break
            except (cp.error.SolverError, cp.error.DCPError):
                pass
        if not success:
            continue

        w_val = w.value
        vol_ann = np.sqrt(w_val.T @ cov_fixed @ w_val)*np.sqrt(252)
        ret_ann = (mean_ret @ w_val)*252
        if vol_ann>1e-12:
            sharpe = (ret_ann - ann_rf)/vol_ann
        else:
            sharpe = -np.inf
        if sharpe>best_sharpe:
            best_sharpe = sharpe
            best_weights = w_val.copy()

    final_ret = (mean_ret @ best_weights)*252
    final_vol = np.sqrt(best_weights.T @ cov_fixed @ best_weights)*np.sqrt(252)
    summary = {
      "Annual Return (%)": round(final_ret*100,2),
      "Annual Vol (%)":    round(final_vol*100,2),
      "Sharpe Ratio":      round(best_sharpe,4)
    }
    return best_weights, summary