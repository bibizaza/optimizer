# File: modules/analytics/extended_metrics.py

import pandas as pd
import numpy as np

def compute_extended_metrics(series_abs: pd.Series, daily_rf: float = 0.0) -> dict:
    """
    Returns a dictionary of extended metrics from a daily absolute portfolio value series:
      - Total Return
      - Annual Return
      - Annual Vol
      - Sharpe
      - MaxDD
      - TimeToRecovery
      - VaR_1M99, CVaR_1M99
      - Skew, Kurtosis
      - Sortino, Calmar, Omega

    If 'series_abs' is too short (<2 points), returns zeros or placeholders.
    daily_rf is your daily risk-free rate (decimal).
    """
    if len(series_abs) < 2:
        return {
            "Total Return": 0.0,
            "Annual Return": 0.0,
            "Annual Vol": 0.0,
            "Sharpe": 0.0,
            "MaxDD": 0.0,
            "TimeToRecovery": 0,
            "VaR_1M99": 0.0,
            "CVaR_1M99": 0.0,
            "Skew": 0.0,
            "Kurtosis": 0.0,
            "Sortino": 0.0,
            "Calmar": 0.0,
            "Omega": 0.0
        }

    daily_ret = series_abs.pct_change().dropna()
    n_days = len(daily_ret)
    trading_days = 252

    # 1) Basic performance
    total_ret = series_abs.iloc[-1] / series_abs.iloc[0] - 1
    ann_ret = (1 + total_ret)**(trading_days / n_days) - 1
    ann_vol = daily_ret.std() * np.sqrt(trading_days)
    ann_rf = daily_rf * trading_days

    sharpe = 0.0
    if ann_vol > 1e-12:
        sharpe = (ann_ret - ann_rf) / ann_vol

    # 2) Max Drawdown / TimeToRecovery
    running_max = series_abs.cummax()
    dd_series = (series_abs / running_max) - 1.0
    max_dd = dd_series.min() if not dd_series.empty else 0.0
    dd_idx = dd_series.idxmin() if not dd_series.empty else None
    if dd_idx is None:
        ttr_days = 0
    else:
        peak_val = running_max.loc[dd_idx]
        after_dd = series_abs.loc[dd_idx:]
        recovered_idx = after_dd[after_dd >= peak_val].index
        if len(recovered_idx) > 0:
            ttr_days = (recovered_idx[0] - dd_idx).days
        else:
            ttr_days = (series_abs.index[-1] - dd_idx).days

    # 3) 1M VaR 99% / CVaR 99%
    lookback_m = 21
    df_1m = series_abs.pct_change(lookback_m).dropna()
    if len(df_1m) < 2:
        var_1m_99 = 0.0
        cvar_1m_99= 0.0
    else:
        sorted_r = np.sort(df_1m.values)
        idx_1pct = int(0.01 * len(sorted_r))
        idx_1pct = max(0, idx_1pct)
        var_1m_99 = sorted_r[idx_1pct]
        tail = sorted_r[sorted_r <= var_1m_99]
        cvar_1m_99 = tail.mean() if len(tail) > 0 else var_1m_99

    # 4) skew / kurtosis
    skew_ = daily_ret.skew()
    kurt_ = daily_ret.kurt()

    # 5) Sortino => downside std
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(trading_days) if len(downside) > 1 else 1e-12
    sortino = 0.0
    if downside_vol > 1e-12:
        sortino = (ann_ret - ann_rf) / downside_vol

    # 6) Calmar => ann_ret / abs(maxDD)
    calmar = 0.0
    if max_dd < 0:
        calmar = ann_ret / abs(max_dd)

    # 7) Omega => sum of positives / abs sum of negatives
    pos_sum = daily_ret[daily_ret > 0].sum()
    neg_sum = daily_ret[daily_ret < 0].sum()
    if abs(neg_sum) < 1e-12:
        omega = 999.0
    else:
        omega = pos_sum / abs(neg_sum)

    # Return everything
    return {
        "Total Return": total_ret,
        "Annual Return": ann_ret,
        "Annual Vol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": max_dd,
        "TimeToRecovery": ttr_days,
        "VaR_1M99": var_1m_99,
        "CVaR_1M99": cvar_1m_99,
        "Skew": skew_,
        "Kurtosis": kurt_,
        "Sortino": sortino,
        "Calmar": calmar,
        "Omega": omega
    }
