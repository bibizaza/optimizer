import numpy as np
import cvxpy as cp
import pandas as pd

from modules.optimization.utils.cov_shrink import (
    nearest_pd, shrink_cov_diagonal,
    compute_ewm_cov, ledoitwolf_cov
)
from modules.optimization.utils.mean_shrink import shrink_mean_to_grand_mean

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
    Computes an optimal portfolio (maximizing Sharpe) subject to:
      1) For each asset class, the sum of weights must lie within [min_class_weight, max_class_weight].
      2) For each instrument with a sub‑type constraint, its weight must lie within [min_instrument, max_instrument].
      
    Parameters:
      (See docstring for detailed explanation.)
      
    Returns:
      - best_w: the optimal weight vector (numpy array)
      - summary: a dict with "Annual Return (%)", "Annual Vol (%)", and "Sharpe Ratio"
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

    # 1) Build covariance matrix
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

    # 2) Process mean returns
    mean_ret = df_returns.mean().values
    if shrink_means and alpha > 0:
        mean_ret = shrink_mean_to_grand_mean(mean_ret, alpha)

    ann_rf = daily_rf * 252
    best_sharpe = -np.inf
    best_w = np.ones(n) / n

    # 3) Scan candidate target returns
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

        # Target return constraint
        constraints.append((mean_ret @ w) * 252 >= targ)

        # Layer 1: Enforce class-sum constraints
        unique_cls = set(asset_classes)
        for cl in unique_cls:
            idxs = [i for i, a in enumerate(asset_classes) if a == cl]
            cdict = class_constraints.get(cl, {})
            min_class = cdict.get("min_class_weight", 0.0)
            max_class = cdict.get("max_class_weight", 1.0)
            constraints.append(cp.sum(w[idxs]) >= min_class)
            constraints.append(cp.sum(w[idxs]) <= max_class)

            # Layer 2: Enforce individual (per-instrument) bounds for those with subtype constraints.
            # (If no constraint is provided for a given (class, security_type) pair, default is [0,1].)
            for i in idxs:
                stp = security_types[i]
                # Look up per-instrument bounds in subtype_constraints.
                if (cl, stp) in subtype_constraints:
                    stvals = subtype_constraints[(cl, stp)]
                    min_inst = stvals.get("min_instrument", 0.0)
                    max_inst = stvals.get("max_instrument", 1.0)
                    constraints.append(w[i] >= min_inst)
                    constraints.append(w[i] <= max_inst)

        # Solve the optimization problem.
        prob = cp.Problem(objective, constraints)
        solved = False
        for solver in [cp.SCS, cp.ECOS]:
            try:
                prob.solve(solver=solver, verbose=False)
                if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                    solved = True
                    break
            except (cp.error.SolverError, cp.error.DCPError):
                pass
        if not solved:
            continue

        w_val = w.value
        vol_ann = np.sqrt(w_val.T @ cov_fixed @ w_val) * np.sqrt(252)
        ret_ann = (mean_ret @ w_val) * 252
        sr = (ret_ann - ann_rf) / vol_ann if vol_ann > 1e-12 else -np.inf
        if sr > best_sharpe:
            best_sharpe = sr
            best_w = w_val.copy()

    final_ret = (mean_ret @ best_w) * 252
    final_vol = np.sqrt(best_w.T @ cov_fixed @ best_w) * np.sqrt(252)
    summary = {
        "Annual Return (%)": round(final_ret * 100, 2),
        "Annual Vol (%)": round(final_vol * 100, 2),
        "Sharpe Ratio": round(best_sharpe, 4)
    }
    return best_w, summary
