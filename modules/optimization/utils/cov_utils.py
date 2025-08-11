"""
Covariance matrix estimators with optional GPU acceleration.

This module implements a handful of covariance estimators used in
portfolio construction. The estimators operate on either CPU or GPU
data structures depending on the backend selected at runtime. When
running on a GPU, arrays are represented by CuPy and tabular data by
cuDF; on the CPU, the equivalents are NumPy and Pandas.

Supported estimators
--------------------
* ``sample``: Unbiased sample covariance of demeaned returns.
* ``ewma``: Exponential weighted moving average covariance with
  parameter ``ewm_alpha``.
* ``mcd``: Minimum covariance determinant for robust estimation
  (requires scikit‑learn).

Optional shrinkage schemes can be applied to any estimator:

* ``none``: No shrinkage.
* ``diagonal``: Linear blend between the covariance and its diagonal
  matrix, with weight ``diag_shrink_beta``.
* ``ledoitwolf``: Ledoit–Wolf shrinkage (requires scikit‑learn).
* ``oas``: Oracle Approximating Shrinkage (requires scikit‑learn).

If ``regularize_cov`` is set to True, a small multiple of the identity
matrix is added to ensure positive definiteness. Additional
regularisation schemes can be added here as needed.

Note that GPU acceleration is only applied to the parts of the
computation that are amenable to parallelisation. Some estimators or
shrinkage methods may fall back to CPU operations when the necessary
libraries are not available on the GPU. The final covariance matrix
will always be returned as either a CuPy or NumPy array depending on
the active backend.
"""

from __future__ import annotations

import warnings
from typing import Optional

from .backend import xp, pd_xp, GPU_AVAILABLE  # noqa: F401

try:
    # scikit‑learn provides Ledoit–Wolf, OAS and MCD estimators
    from sklearn.covariance import LedoitWolf, OAS, MinCovDet  # type: ignore
except ImportError:
    LedoitWolf = None  # type: ignore
    OAS = None  # type: ignore
    MinCovDet = None  # type: ignore

def build_covariance_matrix(
    df_returns,
    *,
    estimator: str = "sample",
    shrinkage: str = "none",
    ewm_alpha: float = 0.06,
    diag_shrink_beta: float = 0.2,
    regularize_cov: bool = False,
    nearest_pd_epsilon: float = 1e-6,
    garch_p: int = 1,
    garch_q: int = 1,
    garch_dist: str = "normal",
    dcc_alpha: float = 0.05,
    dcc_beta: float = 0.90,
) -> "xp.ndarray":
    """Compute a covariance matrix from a returns DataFrame.

    Parameters
    ----------
    df_returns : pd.DataFrame or cudf.DataFrame
        Table of demeaned or raw returns. The number of rows should be
        the time dimension and the number of columns the asset count.
    estimator : {"sample", "ewma", "mcd"}, optional
        Which covariance estimator to use. Defaults to ``"sample"``.
    shrinkage : {"none", "diagonal", "ledoitwolf", "oas"}, optional
        Optional shrinkage method applied after computing the base
        covariance. Defaults to ``"none"``.
    ewm_alpha : float, optional
        Decay factor for the EWMA estimator. Only used when
        ``estimator="ewma"``. Should lie in (0,1]. Defaults to 0.06.
    diag_shrink_beta : float, optional
        Weight of the diagonal component for ``shrinkage="diagonal"``.
        The resulting covariance is ``(1-beta)*cov + beta*diag(cov)``. Defaults
        to 0.2.
    regularize_cov : bool, optional
        If True, adds ``nearest_pd_epsilon * I`` to the covariance to
        ensure it is positive semi‑definite. Defaults to False.
    nearest_pd_epsilon : float, optional
        Small constant added to the diagonal when ``regularize_cov`` is
        True. Defaults to 1e-6.
    garch_p, garch_q, garch_dist, dcc_alpha, dcc_beta : various
        Parameters for DCC‑GARCH models. Currently not implemented in
        this simplified implementation. Provided for API compatibility.

    Returns
    -------
    xp.ndarray
        Covariance matrix as either a CuPy or NumPy array, depending
        on the active backend.
    """
    # Convert input to the appropriate DataFrame type if necessary
    # (pandas vs cudf). In practise, callers should provide the correct
    # type, but we handle common cases for robustness.
    if GPU_AVAILABLE:
        # If a pandas DataFrame is passed on GPU mode, convert to cuDF
        import pandas as _pd  # type: ignore
        if isinstance(df_returns, _pd.DataFrame):
            df_returns = pd_xp.from_pandas(df_returns)
    else:
        # On CPU, ensure we have a pandas DataFrame
        import pandas as _pd  # type: ignore
        if not isinstance(df_returns, _pd.DataFrame):
            df_returns = df_returns.to_pandas()

    # Extract the matrix of returns. For cuDF this yields a CuPy array
    # when indexing with ``.values``, for pandas it yields a NumPy array.
    try:
        returns_mat = df_returns.values  # type: ignore
    except Exception:
        # Fallback: convert via to_numpy()
        returns_mat = df_returns.to_numpy()

    # Ensure we have a 2‑D array
    if returns_mat.ndim == 1:
        returns_mat = returns_mat.reshape(-1, 1)

    T, N = returns_mat.shape

    # Compute the base covariance matrix
    if estimator.lower() == "ewma":
        # Exponential weighted moving average covariance. Weights decay
        # backwards in time so that more recent observations receive a
        # larger weight. ``ewm_alpha`` is the smoothing factor; larger
        # values give more weight to recent data. See e.g. RiskMetrics.
        alpha = float(ewm_alpha)
        if alpha <= 0 or alpha > 1:
            raise ValueError("ewm_alpha must lie in (0,1].")
        # Compute weights: w_t = (1-alpha)^(T-1-t)
        idx = xp.arange(T - 1, -1, -1)
        # Use xp.power to support both NumPy and CuPy
        raw_weights = xp.power((1.0 - alpha), idx)
        norm_weights = raw_weights / xp.sum(raw_weights)
        # Demean the returns
        mean_vec = xp.sum(returns_mat * norm_weights[:, None], axis=0)
        demeaned = returns_mat - mean_vec
        # Apply square root of weights to each row
        sqrt_w = xp.sqrt(norm_weights)[:, None]
        scaled = demeaned * sqrt_w
        # The covariance is the inner product of scaled rows
        cov_mat = scaled.T @ scaled
        # By construction, this is already normalized because weights sum to 1

    elif estimator.lower() == "mcd":
        # Minimum covariance determinant for robust estimation. This
        # estimator is CPU only as scikit‑learn does not support CuPy.
        if MinCovDet is None:
            warnings.warn(
                "MinCovDet estimator unavailable; falling back to sample covariance.",
                RuntimeWarning,
            )
            estimator = "sample"
        else:
            # Ensure data is on CPU
            if GPU_AVAILABLE:
                data_cpu = xp.asnumpy(returns_mat)
            else:
                data_cpu = returns_mat
            mcd = MinCovDet().fit(data_cpu)
            cov_cpu = mcd.covariance_
            cov_mat = xp.asarray(cov_cpu) if GPU_AVAILABLE else cov_cpu

    # Default and fallback to sample covariance
    if estimator.lower() == "sample":
        # Demean along the time dimension
        mean_vec = xp.mean(returns_mat, axis=0)
        demeaned = returns_mat - mean_vec
        if T > 1:
            cov_mat = (demeaned.T @ demeaned) / (T - 1)
        else:
            cov_mat = xp.eye(N)

    # Apply shrinkage schemes
    shrinkage = shrinkage.lower()
    if shrinkage == "diagonal":
        # Blend the covariance with its diagonal. This guards against
        # instability by dampening off‑diagonal entries.
        beta = float(diag_shrink_beta)
        diag = xp.diag(xp.diag(cov_mat))
        cov_mat = (1.0 - beta) * cov_mat + beta * diag

    elif shrinkage == "ledoitwolf":
        if LedoitWolf is None:
            warnings.warn(
                "LedoitWolf shrinkage unavailable; falling back to diagonal shrinkage.",
                RuntimeWarning,
            )
            beta = float(diag_shrink_beta)
            diag = xp.diag(xp.diag(cov_mat))
            cov_mat = (1.0 - beta) * cov_mat + beta * diag
        else:
            # scikit‑learn expects CPU arrays. Convert if on GPU.
            if GPU_AVAILABLE:
                data_cpu = xp.asnumpy(returns_mat)
            else:
                data_cpu = returns_mat
            lw = LedoitWolf().fit(data_cpu)
            cov_cpu = lw.covariance_
            cov_mat = xp.asarray(cov_cpu) if GPU_AVAILABLE else cov_cpu

    elif shrinkage == "oas":
        if OAS is None:
            warnings.warn(
                "OAS shrinkage unavailable; falling back to diagonal shrinkage.",
                RuntimeWarning,
            )
            beta = float(diag_shrink_beta)
            diag = xp.diag(xp.diag(cov_mat))
            cov_mat = (1.0 - beta) * cov_mat + beta * diag
        else:
            if GPU_AVAILABLE:
                data_cpu = xp.asnumpy(returns_mat)
            else:
                data_cpu = returns_mat
            oas = OAS().fit(data_cpu)
            cov_cpu = oas.covariance_
            cov_mat = xp.asarray(cov_cpu) if GPU_AVAILABLE else cov_cpu

    # Optional regularisation: ensure the covariance is positive definite
    if regularize_cov:
        eps = float(nearest_pd_epsilon)
        cov_mat = cov_mat + eps * xp.eye(cov_mat.shape[0])

    return cov_mat


__all__ = ["build_covariance_matrix"]
