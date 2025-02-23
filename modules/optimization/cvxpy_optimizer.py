# File: modules/optimization/cvxpy_optimizer.py

import numpy as np
import pandas as pd
import riskfolio as rf

def parametric_max_sharpe_no_constraints(
    df_returns: pd.DataFrame,
    daily_rf: float = 0.0,
    frequency: int = 252
):
    """
    Minimal function:
     - No constraints
     - No shrink
     - Standard historical approach in Riskfolio
     - Solve single Max Sharpe
    """
    n = df_returns.shape[1]
    # Create a Riskfolio portfolio object
    port = rf.Portfolio(returns=df_returns)

    # Estimate historical mu/cov
    port.assets_stats(method_mu='hist', method_cov='hist')
    # No constraints => skip A_ineq

    # Solve for max Sharpe
    risk_measure = 'MV'  # or 'CVaR' if you prefer
    rf_annual = daily_rf * frequency

    try:
        w_solutions = port.optimization(
            model='Classic',
            rm=risk_measure,
            obj='Sharpe',
            rf=rf_annual,
            l=0.0,
            hist=True,
            alpha=0.95,  # only used if CVaR
            weight_bounds=(0,1)
        )
    except Exception as e:
        # fallback => eq weight
        best_w = np.ones(n)/n
        summary = {
            "Annual Return (%)": 0.0,
            "Annual Vol (%)": 0.0,
            "Sharpe Ratio": 0.0
        }
        return best_w, summary

    if w_solutions is None:
        best_w = np.ones(n)/n
        summary = {
            "Annual Return (%)": 0.0,
            "Annual Vol (%)": 0.0,
            "Sharpe Ratio": 0.0
        }
        return best_w, summary

    best_w = w_solutions.values

    # Evaluate final performance
    perf = port.portfolio_performance(
        w=best_w,
        verbose=False,
        riskfreerate=rf_annual
    )
    ann_ret, ann_risk, ann_sharpe = perf
    summary = {
        "Annual Return (%)": round(ann_ret*100, 2),
        "Annual Vol (%)":    round(ann_risk*100, 2),
        "Sharpe Ratio":      round(ann_sharpe, 4)
    }

    return best_w, summary