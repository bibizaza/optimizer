# modules/optimization/utils/cov_shrink.py

import numpy as np
import pandas as pd
from sklearn.covariance import LedoitWolf

def nearest_pd(cov: np.ndarray, epsilon: float=1e-6) -> np.ndarray:
    """
    Force cov to be positive semidefinite by clipping negative eigenvalues.
    """
    vals, vecs = np.linalg.eigh(cov)
    vals_clipped = np.clip(vals, epsilon, None)
    cov_fixed = (vecs * vals_clipped) @ vecs.T
    cov_fixed = 0.5*(cov_fixed + cov_fixed.T)  # ensure symmetrical
    return cov_fixed

def shrink_cov_diagonal(cov_mat: np.ndarray, beta: float=0.2) -> np.ndarray:
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

def compute_ewm_cov(df_returns: pd.DataFrame, alpha: float=0.06) -> np.ndarray:
    """
    Compute an Exponential Weighted Moving (EWM) covariance using pandas ewm().cov().
    We'll pick the final NxN block at the last date.
    """
    # df_returns is time-indexed
    df_ewm_cov = df_returns.ewm(alpha=alpha, adjust=False).cov()
    # multi-index => pick last date
    last_date = df_returns.index[-1]
    final_cov_block = df_ewm_cov.xs(last_date, level=0)
    return final_cov_block.values
