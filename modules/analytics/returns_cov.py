# modules/analytics/returns_cov.py

import pandas as pd
import numpy as np

def compute_performance_metrics(
    series_abs: pd.Series,
    daily_rf: float = 0.0,
    trading_days_per_year: int = 252,
    lookback_for_month: int = 21
) -> dict:
    if len(series_abs) < 2:
        return {
            "Total Return": 0.0,
            "Annualized Return": 0.0,
            "Annualized Volatility": 0.0,
            "Sharpe Ratio": 0.0,
            "Max Drawdown": 0.0,
            "Time to Recovery (days)": 0,
            "1M VaR 99%": 0.0,
            "1M CVaR 99%": 0.0
        }

    total_return = series_abs.iloc[-1]/ series_abs.iloc[0] -1
    daily_ret = series_abs.pct_change().dropna()
    n_days = len(daily_ret)
    ann_return = (1+ total_return)**(trading_days_per_year/ n_days) -1
    ann_vol = daily_ret.std()* np.sqrt(trading_days_per_year)

    annualized_rf = daily_rf* trading_days_per_year
    sharpe= 0.0
    if ann_vol>1e-12:
        sharpe= (ann_return- annualized_rf)/ ann_vol

    max_dd, ttr_days= _compute_max_dd_and_ttr(series_abs)

    df_1m= series_abs.pct_change(lookback_for_month).dropna()
    if len(df_1m)<2:
        var_1m_99= 0.0
        cvar_1m_99= 0.0
    else:
        sorted_ = np.sort(df_1m.values)
        idx_1pct= int(0.01* len(sorted_))
        idx_1pct= max(idx_1pct,0)
        var_1m_99= sorted_[idx_1pct]
        tail= sorted_[sorted_<= var_1m_99]
        cvar_1m_99= tail.mean() if len(tail)>0 else var_1m_99

    return {
        "Total Return": total_return,
        "Annualized Return": ann_return,
        "Annualized Volatility": ann_vol,
        "Sharpe Ratio": sharpe,
        "Max Drawdown": max_dd,
        "Time to Recovery (days)": ttr_days,
        "1M VaR 99%": var_1m_99,
        "1M CVaR 99%": cvar_1m_99
    }

def _compute_max_dd_and_ttr(series_abs: pd.Series)-> tuple[float,int]:
    running_max= series_abs.cummax()
    dd= (series_abs/running_max)-1.0
    max_dd= dd.min()
    dd_idx= dd.idxmin()
    peak_val= running_max.loc[dd_idx]
    after_dd= series_abs.loc[dd_idx:]
    recovered_idx= after_dd[after_dd>= peak_val].index
    if len(recovered_idx)>0:
        ttr_days= (recovered_idx[0]- dd_idx).days
    else:
        ttr_days= (series_abs.index[-1]- dd_idx).days
    return max_dd, ttr_days