# File: modules/garch_models/dcc_garch_estimation.py

import numpy as np
import pandas as pd

# arch library
from arch.univariate import arch_model
from arch.multivariate import ConstantMean, GARCH, DCC

def fit_univariate_garch(
    series: pd.Series,
    p: int = 1,
    q: int = 1,
    mean_model: str = "Zero",
    vol_model: str = "GARCH",
    dist: str = "normal"
):
    """
    Fit a univariate GARCH(p,q) model to a single asset return series using 'arch' library.
    Returns a dict containing the fitted model, the daily conditional vols,
    and standardized residuals.
    """
    # E.g. mean="Zero" means no AR terms
    if mean_model.lower() == "zero":
        am = arch_model(
            series.dropna(),
            p=p, q=q,
            mean="Zero", vol=vol_model, dist=dist
        )
    else:
        # or "constant", "ARX", etc.
        am = arch_model(
            series.dropna(),
            p=p, q=q,
            mean=mean_model, vol=vol_model, dist=dist
        )

    res = am.fit(disp="off")  # "off" => no console output
    cond_vol = res.conditional_volatility
    # Standardized residuals => e_t / sigma_t
    resid = res.resid / cond_vol

    return {
        "model": am,
        "fitted_res": res,
        "cond_vol": cond_vol,       # aligned with the series index
        "std_resid": resid
    }

def fit_dcc_garch(
    df_returns: pd.DataFrame,
    p: int = 1,
    q: int = 1,
    dcc_p: int = 1,
    dcc_q: int = 1,
    mean_model: str = "Zero",
    vol_model: str = "GARCH",
    dist: str = "normal"
):
    """
    1) Fit univariate GARCH(p,q) for each column in df_returns.
    2) Collect standardized residuals => pass to DCC.
    3) Fit DCC( dcc_p, dcc_q ).
    Returns:
      - A dictionary with univariate fits, the DCC object,
        plus daily correlation/cov estimates if needed.
    """
    # Fit univariate GARCH to each asset
    columns = df_returns.columns
    all_fits = {}
    # We'll store NxT array of std residuals
    std_resids = []

    for col in columns:
        res_dict = fit_univariate_garch(
            df_returns[col],
            p=p, q=q,
            mean_model=mean_model,
            vol_model=vol_model,
            dist=dist
        )
        all_fits[col] = res_dict
        std_resids.append(res_dict["std_resid"].dropna())

    # We must align these residuals in time => find common index intersection
    common_idx = set(std_resids[0].index)
    for r_ in std_resids[1:]:
        common_idx = common_idx.intersection(r_.index)
    common_idx = sorted(list(common_idx))

    # Build matrix NxT of standardized residuals
    # (Note: arch.multivariate expects shape T x N => each row = time, each col = asset)
    arr_list = []
    for i, col in enumerate(columns):
        sub = all_fits[col]["std_resid"].loc[common_idx]
        arr_list.append(sub.values)
    # shape NxT => transpose to T x N
    std_resid_matrix = np.array(arr_list)
    std_resid_matrix = std_resid_matrix.T  # shape => T x N

    # Now feed to arch.multivariate DCC
    # We first define a "ConstantMean" + "GARCH" model for each column's residual
    # but we're ignoring univariate GARCH inside the multivariate step because
    # we've already handled the GARCH for each asset. We'll do a "ConstantMean" on residuals=0?
    mc = ConstantMean(std_resid_matrix)
    mc.volatility = GARCH(p=1, q=1)  # minimal GARCH. Or you can put p=0 if purely DCC for residuals
    dcc = DCC(p=dcc_p, q=dcc_q)

    # Combine => big model
    mc.distribution = dcc

    # Fit
    res_dcc = mc.fit()
    # Now we can extract the time-varying correlations from res_dcc
    # E.g. res_dcc.forecast(horizon=1), or we can get res_dcc.covariance, etc.

    return {
        "univariate_fits": all_fits,   # dict of {col -> {model, fitted_res, cond_vol, std_resid}}
        "dcc_model": mc,
        "dcc_fit": res_dcc,
        "common_index": common_idx
    }

def get_dcc_time_varying_covariances(dcc_fit_result, horizon: int = 1):
    """
    Suppose we want to fetch the entire time series of dynamic covariance
    from the fitted DCC object. For instance, arch.multivariate returns
    an array of shape (T, N, N).
    Or you can forecast horizon steps ahead.

    Returns a dict or array of covariance matrices over time
    (one per day in the sample).
    """
    # If you've done "dcc_fit_result['dcc_fit'].forecasts", you can retrieve:
    # e.g. R_t => correlation, then multiply by each sigma?

    # We'll do a direct approach:
    dcc_res = dcc_fit_result["dcc_fit"]
    # arch's doc => after fitting, e.g. dcc_res.covariance.values, etc.
    # Actually for DCC => we might do:
    cov_series = dcc_res.covariance
    # shape => (T, N, N)

    # or forecast => e.g. dcc_res.forecast(horizon)
    # We'll skip the details. Just show we can gather:
    return cov_series

def dcc_garch_forecast_next_cov(df_returns: pd.DataFrame):
    """
    Example function: Fit univariate GARCH + DCC up to the last date,
    return the forecasted NxN covariance for the next day.
    This is the function you'll feed into your optimizer each rolling step.

    Very simplified pseudo-code. Tweak as needed.
    """
    # 1) Fit univariate GARCH + DCC
    fit_result = fit_dcc_garch(df_returns)

    # 2) Suppose we get the last day correlation from the DCC
    # Then multiply each asset's next-day stdev from univariate fit => NxN covariance
    # We'll do a naive approach => just use the final day in-sample correlation + final stdev.
    # In real code you'd do a separate forecast step for day t+1.

    dcc_res = fit_result["dcc_fit"]
    # Usually you'd do => dcc_res.forecast(horizon=1), or some method to get next step
    # But let's just say we want the "last day correlation" from the model:
    cov_series = dcc_res.covariance
    # shape => (T, N, N)
    # the final slice => cov_series.values[-1, :, :]

    last_cov = cov_series.values[-1]
    # That is the "time-varying covariance" on the last in-sample day

    return last_cov