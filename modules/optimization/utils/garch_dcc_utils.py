# File: modules/optimization/utils/garch_dcc_utils.py

import numpy as np
import pandas as pd
from arch.univariate import arch_model

########################################################################
# 1) Univariate GARCH Fit for Each Asset
########################################################################

def fit_univariate_garch(
    df_returns_col: pd.Series,
    mean: str = 'Constant',
    vol: str = 'GARCH',
    p: int = 1,
    q: int = 1,
    dist: str = 'normal',
    disp: str = 'off'
):
    """
    Fit a univariate GARCH(p,q) model to a single asset's returns (Series).
    Returns:
      model_fit: the fitted arch_model result
      cond_vol : np.array of conditional volatilities for each time
      std_resid: np.array of standardized residuals
    """
    # 1) Build arch_model
    am = arch_model(
        df_returns_col,
        mean=mean,
        vol=vol,
        p=p,
        q=q,
        dist=dist
    )

    # 2) Fit
    res = am.fit(disp=disp)  # disp='off' to suppress iteration messages

    # 3) Extract final cond_vol, residuals, etc.
    cond_vol = res.conditional_volatility  # pd.Series, same index as input
    resid = res.resid
    # Standardized: eps = resid / sigma
    eps = resid / cond_vol.replace(0.0, np.nan)
    eps = eps.fillna(0.0).values  # convert to np array
    cond_vol = cond_vol.values

    return res, cond_vol, eps


########################################################################
# 2) Minimal DCC(1,1) Recursion
########################################################################

def dcc_recursive(
    eps_matrix: np.ndarray,
    alpha: float = 0.05,
    beta: float = 0.90
):
    """
    A manual DCC(1,1) recursion to compute NxN correlation for each time step.

    eps_matrix: shape (T, N) => standardized residuals
    alpha, beta: DCC parameters in (0,1).

    Returns:
      R_t_list: a list of NxN correlation matrices, for t=0..T-1
    """
    T, N = eps_matrix.shape
    # 1) sample correlation as "long-run" average correlation R_bar
    R_bar = np.corrcoef(eps_matrix, rowvar=False)
    R_bar = np.nan_to_num(R_bar, nan=0.0)

    # 2) We'll store a correlation matrix for each t
    R_t_list = []
    # 3) Initialize Q_0 with R_bar
    Q_t = R_bar.copy()

    for t in range(T):
        if t == 0:
            # correlation at t=0 is just from Q_t=R_bar
            pass
        else:
            # e_{t-1}: shape (N,) => we do e_{t-1} outer e_{t-1}
            e_lag = eps_matrix[t-1, :].reshape(-1,1)  # (N,1)
            Q_lag = Q_t
            # Q_t = (1-alpha-beta)*R_bar + alpha* e_{t-1} e_{t-1}' + beta* Q_{t-1}
            Q_t = (1 - alpha - beta)*R_bar + alpha*(e_lag @ e_lag.T) + beta*Q_lag

        # Then get correlation matrix R_t from Q_t
        diag_Q = np.sqrt(np.diag(Q_t))
        diag_Q[diag_Q <= 1e-12] = 1e-12  # avoid /0
        D_inv = np.diag(1.0 / diag_Q)
        R_ = D_inv @ Q_t @ D_inv
        # ensure symmetrical
        R_ = 0.5*(R_ + R_.T)

        R_t_list.append(R_)

    return R_t_list


########################################################################
# 3) Build DCC-GARCH Covariance (arch >=7.0 friendly)
########################################################################

def compute_dcc_garch_cov(
    df_returns: pd.DataFrame,
    # GARCH hyperparams:
    p: int = 1,
    q: int = 1,
    dist: str = "normal",
    # DCC recursion hyperparams:
    dcc_alpha: float = 0.05,
    dcc_beta: float = 0.90
) -> np.ndarray:
    """
    1) For each asset => univariate GARCH(p,q) => final cond_vol & standardized eps.
    2) Then run DCC(1,1) recursion on the entire eps_matrix => correlation for each t
       R_t_list => pick the final day correlation R_T
    3) Cov = D_T * R_T * D_T, where D_T = diag of final-day cond_vol.

    This approach does NOT rely on arch.covariance.DCC, which is removed in arch>=7.
    """
    T, N = df_returns.shape
    cond_vol_matrix = np.zeros((T, N))
    eps_matrix = np.zeros((T, N))

    # 1) Fit univariate GARCH for each asset
    for i, col in enumerate(df_returns.columns):
        series = df_returns[col].dropna()
        _, sigma_i, eps_i = fit_univariate_garch(series, p=p, q=q, dist=dist)

        # align to main index if needed
        sig_s = pd.Series(sigma_i, index=series.index).reindex(df_returns.index, fill_value=0.0).values
        eps_s = pd.Series(eps_i, index=series.index).reindex(df_returns.index, fill_value=0.0).values
        cond_vol_matrix[:, i] = sig_s
        eps_matrix[:, i] = eps_s

    # 2) DCC recursion => get correlation over time => final day correlation
    R_t_list = dcc_recursive(eps_matrix, alpha=dcc_alpha, beta=dcc_beta)
    R_final = R_t_list[-1]  # NxN correlation at day T-1 (0-based => T-1 is final)

    # 3) final-day covariance
    final_vols = cond_vol_matrix[-1, :]  # shape (N,)
    D = np.diag(final_vols)
    cov_final = D @ R_final @ D
    return cov_final
