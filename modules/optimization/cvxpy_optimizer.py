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
    ewm_alpha: float = 0.06
):
    """
    A parametric frontier => max Sharpe approach.

    **Now** enforces min_instrument_weight if present in class_constraints.
    Also uses psd_wrap and a small SHIFT_EPS on the covariance to avoid ARPACK errors.

    class_constraints[class_name] can include:
      - "min_class_weight"
      - "max_class_weight"
      - "min_instrument_weight"
      - "max_instrument_weight"
    """
    n = len(tickers)
    if df_returns.shape[1] != n:
        raise ValueError("df_returns shape mismatch.")
    if len(asset_classes) != n:
        raise ValueError("asset_classes mismatch.")

    # 1) Build covariance
    if do_ewm:
        cov_raw = compute_ewm_cov(df_returns, alpha=ewm_alpha)
    else:
        if use_ledoitwolf:
            cov_raw = ledoitwolf_cov(df_returns)
        else:
            cov_raw = df_returns.cov().values
            if regularize_cov:
                cov_raw = nearest_pd(cov_raw)
            if shrink_cov:
                cov_raw = shrink_cov_diagonal(cov_raw, beta)

    # SHIFT => ensure strictly positive diagonal
    SHIFT_EPS = 1e-8
    cov_fixed = cov_raw + SHIFT_EPS * np.eye(len(cov_raw))

    # wrap => skip ARPACK checks
    cov_expr = cp.psd_wrap(cov_fixed)

    # 2) Possibly shrink means
    mean_ret = df_returns.mean().values
    if shrink_means and alpha > 0:
        mean_ret = shrink_mean_to_grand_mean(mean_ret, alpha)

    ann_rf = daily_rf * 252
    best_sharpe = -np.inf
    best_weights = np.ones(n) / n

    # 3) parametric approach => n_points
    asset_ann_ret = mean_ret * 252
    target_min = max(0.0, asset_ann_ret.min())
    target_max = asset_ann_ret.max()
    candidate_targets = np.linspace(target_min, target_max, n_points)

    for targ in candidate_targets:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, cov_expr))
        constraints = [cp.sum(w) == 1]

        # no_short => w >= 0
        if no_short:
            constraints.append(w >= 0)

        # class constraints
        # now we handle min_instrument_weight if present
        unique_classes = set(asset_classes)
        for cl_ in unique_classes:
            idxs = [i for i,a in enumerate(asset_classes) if a == cl_]
            cc = class_constraints.get(cl_, {})
            if "min_class_weight" in cc:
                constraints.append(cp.sum(w[idxs]) >= cc["min_class_weight"])
            if "max_class_weight" in cc:
                constraints.append(cp.sum(w[idxs]) <= cc["max_class_weight"])
            # new: if min_instrument_weight => for each instrument in idxs
            if "min_instrument_weight" in cc:
                min_iw = cc["min_instrument_weight"]
                for i_ in idxs:
                    constraints.append(w[i_] >= min_iw)
            if "max_instrument_weight" in cc:
                max_iw = cc["max_instrument_weight"]
                for i_ in idxs:
                    constraints.append(w[i_] <= max_iw)

        # target return constraint
        constraints.append((mean_ret @ w)*252 >= targ)

        prob = cp.Problem(objective, constraints)

        # Solve => fallback from SCS => ECOS
        success = False
        for solver_ in [cp.SCS, cp.ECOS]:
            try:
                prob.solve(solver=solver_, verbose=False)
                if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
                    success = True
                    break
            except (cp.error.SolverError, cp.error.DCPError):
                continue  # try next solver

        if not success:
            continue

        # Evaluate Sharpe
        w_val = w.value
        vol_ann = np.sqrt(w_val.T @ cov_fixed @ w_val) * np.sqrt(252)
        ret_ann = (mean_ret @ w_val) * 252
        if vol_ann > 1e-12:
            sharpe = (ret_ann - ann_rf) / vol_ann
        else:
            sharpe = -np.inf

        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = w_val.copy()

    final_ret = (mean_ret @ best_weights) * 252
    final_vol = np.sqrt(best_weights.T @ cov_fixed @ best_weights) * np.sqrt(252)
    summary = {
        "Annual Return (%)": round(final_ret*100, 2),
        "Annual Vol (%)":    round(final_vol*100, 2),
        "Sharpe Ratio":      round(best_sharpe, 4)
    }
    return best_weights, summary
