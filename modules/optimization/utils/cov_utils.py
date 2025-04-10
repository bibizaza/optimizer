# File: modules/optimization/utils/cov_utils.py

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf
from sklearn.covariance import MinCovDet

###############################################################################
# 1) Basic PSD & Shrink Helpers
###############################################################################
def nearest_pd(cov: np.ndarray, epsilon: float = 1e-6) -> np.ndarray:
    """
    Force cov to be positive semidefinite by clipping negative eigenvalues.
    """
    vals, vecs = np.linalg.eigh(cov)
    vals_clipped = np.clip(vals, epsilon, None)
    cov_fixed = (vecs * vals_clipped) @ vecs.T
    cov_fixed = 0.5 * (cov_fixed + cov_fixed.T)
    return cov_fixed

def shrink_cov_diagonal(cov_mat: np.ndarray, beta: float = 0.2) -> np.ndarray:
    """
    Simple diagonal shrink:
      cov_shrunk = (1 - beta)*cov_mat + beta * diag_mean * I
    """
    diag_mean = np.mean(np.diag(cov_mat))
    n = cov_mat.shape[0]
    I_ = np.eye(n)
    cov_shrunk = (1 - beta)*cov_mat + beta*diag_mean*I_
    return cov_shrunk

def ledoitwolf_cov(df_returns: pd.DataFrame) -> np.ndarray:
    """
    Fit a LedoitWolf model to df_returns and return the shrunk covariance.
    """
    lw_model = LedoitWolf().fit(df_returns.values)
    return lw_model.covariance_

###############################################################################
# 2) EWMA Covariance
###############################################################################
def compute_ewm_cov(df_returns: pd.DataFrame, alpha: float = 0.06) -> np.ndarray:
    """
    Compute an Exponential Weighted Moving Cov using pandas ewm().cov().
    We'll pick the final NxN block at the last date.
    """
    df_ewm_cov = df_returns.ewm(alpha=alpha, adjust=False).cov()
    last_date = df_returns.index[-1]
    final_cov_block = df_ewm_cov.xs(last_date, level=0)
    return final_cov_block.values

###############################################################################
# 3) Manual DCC-GARCH (arch≥7 compatible)
###############################################################################
from arch.univariate import arch_model

def _fit_garch_one_asset(series, p=1, q=1, dist="normal"):
    """
    Helper: fit GARCH(p,q) => final cond_vol, standardized residuals
    Returns cond_vol (array) & eps (array)
    """
    am = arch_model(series, mean="Constant", vol="GARCH", p=p, q=q, dist=dist)
    res = am.fit(disp="off")
    sigma = res.conditional_volatility
    resid = res.resid
    eps = resid / sigma.replace(0.0, np.nan)
    eps = eps.fillna(0.0)
    return sigma.values, eps.values

def _dcc_recursive(eps_matrix, alpha=0.05, beta=0.90):
    """
    Minimal DCC(1,1) recursion. Returns a list of correlation matrices over time.
    R_t_list[-1] is final correlation.
    """
    T, N = eps_matrix.shape
    R_bar = np.corrcoef(eps_matrix, rowvar=False)
    R_bar = np.nan_to_num(R_bar, nan=0.0)

    R_t_list = []
    Q_t = R_bar.copy()

    for t in range(T):
        if t > 0:
            e_lag = eps_matrix[t-1, :].reshape(-1,1)
            Q_t = (1 - alpha - beta)*R_bar + alpha*(e_lag @ e_lag.T) + beta*Q_t
        # correlation from Q_t
        diag_q = np.sqrt(np.diag(Q_t))
        diag_q[diag_q<=1e-12] = 1e-12
        D_inv = np.diag(1.0/diag_q)
        R_ = D_inv @ Q_t @ D_inv
        R_ = 0.5*(R_ + R_.T)
        R_t_list.append(R_)
    return R_t_list

def compute_dcc_garch_cov(
    df_returns: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    dcc_alpha: float = 0.05,
    dcc_beta: float = 0.90
) -> np.ndarray:
    """
    1) Fit univariate GARCH => cond vol & eps for each asset
    2) DCC(1,1) recursion => final correlation
    3) Cov = diag(final_vols) * R_final * diag(final_vols)
    """
    T, N = df_returns.shape
    cond_vols = np.zeros((T, N))
    eps_matrix = np.zeros((T, N))

    # Fit GARCH for each asset
    for i, col in enumerate(df_returns.columns):
        ser = df_returns[col].dropna()
        sig_i, eps_i = _fit_garch_one_asset(ser, p=p, q=q, dist=dist)
        # reindex if needed
        sig_s = pd.Series(sig_i, index=ser.index).reindex(df_returns.index, fill_value=0.0).values
        eps_s = pd.Series(eps_i, index=ser.index).reindex(df_returns.index, fill_value=0.0).values
        cond_vols[:, i] = sig_s
        eps_matrix[:, i] = eps_s

    # DCC recursion
    R_list = _dcc_recursive(eps_matrix, alpha=dcc_alpha, beta=dcc_beta)
    R_final = R_list[-1]

    # final-day vol => NxN covariance
    final_vols = cond_vols[-1, :]
    D = np.diag(final_vols)
    cov_final = D @ R_final @ D
    return cov_final

###############################################################################
# 4) Manual DCC-GARCH (arch≥7 compatible)
###############################################################################

def compute_mcd_cov(df_returns: pd.DataFrame) -> np.ndarray:
    """
    Attempt to compute a robust covariance using MCD.
    If it fails (e.g. too many missing data => not enough rows),
    fallback to sample covariance and emit a warning/log.
    """
    try:
        mcd_model = MinCovDet().fit(df_returns.values)
        cov_mcd = mcd_model.covariance_
        cov_mcd = np.nan_to_num(cov_mcd, nan=0.0, posinf=0.0, neginf=0.0)
        return cov_mcd
    except ValueError as e:
        # For example, "kth out of bounds" => not enough rows
        print(f"[WARNING] MCD failed: {e}. Falling back to sample covariance.")
        # fallback => sample
        cov_fallback = df_returns.cov().values
        cov_fallback = np.nan_to_num(cov_fallback, nan=0.0, posinf=0.0, neginf=0.0)
        return cov_fallback
    
###############################################################################
# 5) Master function: build_covariance_matrix
###############################################################################
def build_covariance_matrix(
    df_returns: pd.DataFrame,
    estimator: str = "sample",   # "sample", "ewma", "dcc_garch", "mcd"
    shrinkage: str = "none",     # "none", "diagonal", "ledoitwolf"
    ewm_alpha: float = 0.06,
    diag_shrink_beta: float = 0.2,
    regularize_cov: bool = False,
    nearest_pd_epsilon: float = 1e-6,

    # DCC-GARCH params
    garch_p: int = 1,
    garch_q: int = 1,
    garch_dist: str = "normal",
    dcc_alpha: float = 0.05,
    dcc_beta: float = 0.90
) -> np.ndarray:
    """
    Main entry point for building a covariance matrix given a df_returns DataFrame
    and user-chosen parameters.

    Possible estimator values:
      - "sample"
      - "ewma"
      - "dcc_garch"
      - "mcd"  (NEW: Minimum Covariance Determinant, robust to outliers)
    """
    if df_returns.shape[0] < 2:
        return np.eye(df_returns.shape[1])

    # 1) Base covariance
    if estimator == "sample":
        cov_raw = df_returns.cov().values

    elif estimator == "ewma":
        cov_raw = compute_ewm_cov(df_returns, alpha=ewm_alpha)

    elif estimator == "dcc_garch":
        cov_raw = compute_dcc_garch_cov(
            df_returns,
            p=garch_p,
            q=garch_q,
            dist=garch_dist,
            dcc_alpha=dcc_alpha,
            dcc_beta=dcc_beta
        )

    elif estimator == "mcd":
        # NEW: robust MCD approach
        cov_raw = compute_mcd_cov(df_returns)

    else:
        raise ValueError(f"Unknown estimator: {estimator}")

    # 2) Shrinkage
    if shrinkage == "ledoitwolf":
        lw_model = LedoitWolf().fit(df_returns.values)
        cov_shrunk = lw_model.covariance_
    elif shrinkage == "diagonal":
        cov_shrunk = shrink_cov_diagonal(cov_raw, beta=diag_shrink_beta)
    elif shrinkage == "none":
        cov_shrunk = cov_raw
    else:
        raise ValueError(f"Unknown shrinkage: {shrinkage}")

    # 3) (Optional) nearest_pd for regularization
    if regularize_cov:
        cov_final = nearest_pd(cov_shrunk, epsilon=nearest_pd_epsilon)
    else:
        cov_final = cov_shrunk

    cov_final = np.nan_to_num(cov_final, nan=0.0, posinf=0.0, neginf=0.0)
    return cov_final