# extended_metrics.py

import pandas as pd
import numpy as np

def zero_metrics_dict() -> dict:
    """
    Returns a dictionary of metric names mapped to 0.0 (or 0).
    Used when there's not enough data to compute metrics
    (e.g. daily returns is empty).
    """
    return {
        "Total Return":    0.0,
        "Annual Return":   0.0,
        "Annual Vol":      0.0,
        "Sharpe":          0.0,
        "MaxDD":           0.0,
        "TimeToRecovery":  0.0,
        "VaR_1M99":        0.0,
        "CVaR_1M99":       0.0,
        "Skew":            0.0,
        "Kurtosis":        0.0,
        "Sortino":         0.0,
        "Calmar":          0.0,
        "Omega":           0.0
    }

def compute_extended_metrics(series_abs: pd.Series, daily_rf: float = 0.0) -> dict:
    """
    Computes a variety of extended performance metrics from a
    timeseries of absolute portfolio values (series_abs).

    Safely short-circuits if daily returns is empty or nearly empty
    to avoid ZeroDivisionError or invalid exponent.
    """
    # Typically we assume 252 trading days per year
    freq = 252

    # Check if we have at least 2 data points in series_abs
    if len(series_abs) < 2:
        # Not enough to compute daily returns
        return zero_metrics_dict()

    # Compute daily returns => short-circuit if empty
    daily_ret = series_abs.pct_change().dropna()
    n_days = len(daily_ret)
    if n_days == 0:
        # daily_ret is empty => fallback
        return zero_metrics_dict()

    # 1) Total & Annual Returns
    total_ret = series_abs.iloc[-1] / series_abs.iloc[0] - 1
    ann_ret = (1 + total_ret)**(freq / n_days) - 1

    # 2) Annual Vol & Sharpe
    ann_vol = daily_ret.std() * np.sqrt(freq)
    ann_rf  = daily_rf * freq

    if ann_vol < 1e-12:
        sharpe = 0.0
    else:
        sharpe = (ann_ret - ann_rf) / ann_vol

    # 3) Max Drawdown & Time to Recovery
    run_max = series_abs.cummax()
    dd = series_abs / run_max - 1
    maxdd = dd.min() if not dd.empty else 0.0
    dd_idx = dd.idxmin() if not dd.empty else None
    if dd_idx is None:
        ttr = 0
    else:
        peak_val = run_max.loc[dd_idx]
        after_dd = series_abs.loc[dd_idx:]
        rec_idx = after_dd[after_dd >= peak_val].index
        if len(rec_idx) > 0:
            ttr = (rec_idx[0] - dd_idx).days
        else:
            ttr = (series_abs.index[-1] - dd_idx).days

    # 4) 1-month VaR & CVaR at 99%
    lookb = 21  # ~1 month
    df_1m = series_abs.pct_change(lookb).dropna()
    if len(df_1m) < 2:
        var_1m99 = 0.0
        cvar_1m99 = 0.0
    else:
        sorted_vals = np.sort(df_1m.values)
        idx_ = int(0.01 * len(sorted_vals))
        idx_ = max(idx_, 0)
        var_1m99 = sorted_vals[idx_]
        tail = sorted_vals[sorted_vals <= var_1m99]
        if len(tail) > 0:
            cvar_1m99 = tail.mean()
        else:
            cvar_1m99 = var_1m99

    # 5) Skew & Kurtosis
    skew_  = daily_ret.skew()
    kurt_  = daily_ret.kurt()

    # 6) Sortino => separate std of negative daily returns
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(freq) if len(downside) > 1 else 1e-12
    if downside_vol < 1e-12:
        sortino = 0.0
    else:
        sortino = (ann_ret - ann_rf) / downside_vol

    # 7) Calmar => annual return / drawdown
    if maxdd < 0:
        calmar = ann_ret / abs(maxdd)
    else:
        calmar = 0.0

    # 8) Omega => sum of positives / sum of negatives
    pos_sum = daily_ret[daily_ret > 0].sum()
    neg_sum = daily_ret[daily_ret < 0].sum()
    if abs(neg_sum) < 1e-12:
        omega = 999.0
    else:
        omega = pos_sum / abs(neg_sum)

    return {
        "Total Return":    total_ret,
        "Annual Return":   ann_ret,
        "Annual Vol":      ann_vol,
        "Sharpe":          sharpe,
        "MaxDD":           maxdd,
        "TimeToRecovery":  ttr,
        "VaR_1M99":        var_1m99,
        "CVaR_1M99":       cvar_1m99,
        "Skew":            skew_,
        "Kurtosis":        kurt_,
        "Sortino":         sortino,
        "Calmar":          calmar,
        "Omega":           omega
    }
