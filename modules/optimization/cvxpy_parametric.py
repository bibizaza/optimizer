import numpy as np
import pandas as pd
import cvxpy as cp

# Import our dynamic backend. ``xp`` is either CuPy or NumPy and
# ``pd_xp`` is either cuDF or Pandas. ``GPU_AVAILABLE`` indicates
# whether the GPU path is active.
from modules.optimization.utils.backend import xp, pd_xp, GPU_AVAILABLE  # noqa: F401

# Import our covariance builder and mean shrinkage functions. These are
# implemented to operate on the selected backend transparently.
from modules.optimization.utils.cov_utils import (
    build_covariance_matrix,
)

from modules.optimization.utils.mean_shrink import (
    shrink_mean_to_grand_mean,
    shrink_mean_to_zero,
)

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

    # === Covariance Estimator & Shrinkage ===
    cov_estimator: str = "sample",         # "sample", "ewma", "dcc_garch"
    shrinkage: str = "none",               # "none", "diagonal", "ledoitwolf"
    ewm_alpha: float = 0.06,
    diag_shrink_beta: float = 0.2,
    regularize_cov: bool = False,
    nearest_pd_epsilon: float = 1e-6,

    # (DCC-GARCH params if using "dcc_garch")
    garch_p: int = 1,
    garch_q: int = 1,
    garch_dist: str = "normal",
    dcc_alpha: float = 0.05,
    dcc_beta: float = 0.90,

    # === Mean shrink?
    mean_tech: str = "none", 
    alpha_mean_shrink: float = 0.0,
):
    """
    Parametric approach for Maximum Sharpe:
      1) We sample multiple target returns in [targ_min, targ_max].
      2) For each target, we do a min-variance optimization subject to
         w >= 0 (if no_short) and portfolio return >= target, plus
         any class/subtype constraints.
      3) We pick ex-post which solution yields the highest Sharpe ratio.

    Args:
      df_returns          : DataFrame of daily returns (T x N)
      tickers             : list of ticker names (length N)
      asset_classes       : list of asset_class for each ticker
      security_types      : list of security_types for each ticker
      class_constraints   : dict with e.g. {cls_name: {"min_class_weight":..., "max_class_weight":...}}
      subtype_constraints : dict with e.g. {(cls_name, subtype_name): {"min_instrument":..., "max_instrument":...}}
      daily_rf            : daily risk-free rate (decimal, e.g. 0.0001 for ~2.5% annual if 252 days)
      no_short            : if True, enforce w >= 0
      n_points            : number of points in the frontier to sample
      cov_estimator       : "sample" | "ewma" | "dcc_garch"
      shrinkage           : "none" | "diagonal" | "ledoitwolf"
      ewm_alpha           : used if cov_estimator=="ewma"
      diag_shrink_beta    : used if shrinkage=="diagonal"
      regularize_cov      : if True => nearest_pd
      nearest_pd_epsilon  : small eps to ensure PSD
      garch_*             : DCC-GARCH hyperparams if using "dcc_garch"
      shrink_means        : if True => shrink to grand mean
      alpha_mean_shrink   : fraction for mean shrink

    Returns:
      best_w : np.array of final weights that yield highest Sharpe
      summary: dict with {
                 "Annual Return (%)",
                 "Annual Vol (%)",
                 "Sharpe Ratio"
               }
    """
    n = len(tickers)
    if df_returns.shape[1] != n:
        raise ValueError("df_returns shape mismatch vs # tickers.")
    if len(asset_classes) != n:
        raise ValueError("asset_classes length mismatch.")
    if len(security_types) != n:
        raise ValueError("security_types length mismatch.")

    # 1) Clean returns => remove inf/NaN
    # We perform data cleaning on the CPU using pandas to ensure
    # compatibility with cuDF, then transfer to the GPU if available.
    df_ret_clean = df_returns.replace([np.inf, -np.inf], np.nan).dropna(how='all', axis=1)
    df_ret_clean = df_ret_clean.dropna(how='all', axis=0).fillna(0.0)
    if df_ret_clean.shape[1] < 1 or df_ret_clean.shape[0] < 2:
        # fallback => equal weights if not enough data
        best_w = np.ones(n) / max(n, 1)
        return best_w, {
            "Annual Return (%)": 0.0,
            "Annual Vol (%)": 0.0,
            "Sharpe Ratio": 0.0
        }

    # 2) Convert cleaned returns to the appropriate backend. When
    # running on the GPU, we convert the pandas DataFrame to cuDF;
    # otherwise we keep the pandas DataFrame. ``pd_xp.from_pandas``
    # handles the conversion internally.
    if GPU_AVAILABLE:
        df_ret_dev = pd_xp.from_pandas(df_ret_clean)
    else:
        df_ret_dev = df_ret_clean

    # 3) Build the covariance matrix on the selected backend. The
    # resulting matrix is an xp.ndarray (CuPy or NumPy) depending
    # on ``GPU_AVAILABLE``.
    cov_raw = build_covariance_matrix(
        df_returns=df_ret_dev,
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
        dcc_beta=dcc_beta,
    )

    # SHIFT to ensure well‑conditioned matrices. Use ``xp.eye`` so that
    # the shift happens on the appropriate device.
    SHIFT_EPS = 1e-8
    cov_shifted = cov_raw + SHIFT_EPS * xp.eye(n)

    # 4) Compute expected returns on the device
    mean_ret_dev = df_ret_dev.mean().values
    # Apply mean shrinkage if requested. The shrinkage functions
    # operate on either CuPy or NumPy arrays transparently.
    if mean_tech == "shrink_to_grand_mean" and alpha_mean_shrink > 0:
        mean_ret_dev = shrink_mean_to_grand_mean(mean_ret_dev, alpha_mean_shrink)
    elif mean_tech == "shrink_to_zero" and alpha_mean_shrink > 0:
        mean_ret_dev = shrink_mean_to_zero(mean_ret_dev, alpha_mean_shrink)

    # Compute annualised risk‑free rate
    ann_rf = daily_rf * 252

    # Prepare to search over target returns. Convert device arrays
    # back to CPU for operations that must run on the CPU (linspace).
    # xp.asnumpy returns a NumPy array when on the GPU; on the CPU it
    # simply returns the original array.
    mean_ret_cpu = xp.asnumpy(mean_ret_dev) if GPU_AVAILABLE else mean_ret_dev
    asset_ann_ret = mean_ret_cpu * 252
    targ_min = max(0.0, asset_ann_ret.min())
    targ_max = asset_ann_ret.max()
    candidate_targets = np.linspace(targ_min, targ_max, n_points)

    # Transfer covariance matrix to the CPU for CVXPY. cvxpy does not
    # support CuPy arrays; therefore we convert here. ``cov_cpu`` is
    # always a NumPy array at this point.
    cov_cpu = xp.asnumpy(cov_shifted) if GPU_AVAILABLE else cov_shifted

    best_sharpe = -np.inf
    # Placeholder for the best weights on the CPU
    best_w_cpu = np.ones(n) / max(n, 1)

    # Wrap covariance matrix for cvxpy; this remains constant across
    # the loop. ``cp.psd_wrap`` ensures numerical stability.
    P = cp.psd_wrap(cov_cpu)

    for targ in candidate_targets:
        w = cp.Variable(n)
        objective = cp.Minimize(cp.quad_form(w, P))
        cons = [cp.sum(w) == 1]
        if no_short:
            cons.append(w >= 0)
        # Portfolio return constraint. ``mean_ret_cpu`` is a 1‑D
        # NumPy array at this point.
        cons.append((mean_ret_cpu @ w) * 252 >= targ)

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
                    subd = subtype_constraints[(cls_, stp)]
                    mini = subd.get("min_instrument", 0.0)
                    maxi = subd.get("max_instrument", 1.0)
                    cons.append(w[i_] >= mini)
                    cons.append(w[i_] <= maxi)

        prob = cp.Problem(objective, cons)
        solved = False
        for solver_ in [cp.ECOS, cp.OSQP]:
            try:
                prob.solve(solver=solver_, verbose=False)
                if prob.status in ("optimal", "optimal_inaccurate") and w.value is not None:
                    solved = True
                    break
            except (cp.error.SolverError, cp.error.DCPError):
                pass

        if not solved or w.value is None:
            continue

        # Extract weights on the CPU
        w_val_cpu = w.value
        # Compute annualised volatility and return on the CPU
        vol_ann = float(np.sqrt(w_val_cpu.T @ cov_cpu @ w_val_cpu) * np.sqrt(252))
        ret_ann = float(mean_ret_cpu @ w_val_cpu * 252)
        if vol_ann < 1e-12:
            sr_ = -np.inf
        else:
            sr_ = (ret_ann - ann_rf) / vol_ann

        if sr_ > best_sharpe:
            best_sharpe = sr_
            best_w_cpu = w_val_cpu.copy()

    final_ret = float(mean_ret_cpu @ best_w_cpu * 252)
    final_vol = float(np.sqrt(best_w_cpu.T @ cov_cpu @ best_w_cpu) * np.sqrt(252))
    summary = {
        "Annual Return (%)": round(final_ret, 2),
        "Annual Vol (%)": round(final_vol, 2),
        "Sharpe Ratio": round(best_sharpe, 4),
    }

    return best_w_cpu, summary
