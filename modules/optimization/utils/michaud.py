# modules/optimization/utils/michaud.py

import numpy as np
import random
import pandas as pd

def parametric_max_sharpe_michaud(
    df_returns: pd.DataFrame,
    tickers,
    asset_classes,
    class_constraints,
    daily_rf=0.0,
    no_short=True,
    n_points=15,
    regularize_cov=False,
    shrink_means=False,
    alpha=0.3,
    shrink_cov=False,
    beta=0.2,
    use_ledoitwolf=False,
    ewm_cov=False,
    ewm_alpha=0.06,
    n_boot=10
):
    """
    Michaud's resampling approach:
      1) n_boot bootstrap draws from df_returns
      2) each => parametric_max_sharpe(...) => w_i
      3) average w_i => w_mich
      4) measure final performance with same approach
    """
    # local import => breaks cycle => ensures fully loaded
    from modules.optimization.cvxpy_optimizer import (
        parametric_max_sharpe,
        _build_final_cov_and_means_for_summary
    )

    original_index = df_returns.index
    n_rows = len(original_index)
    all_weights = []

    for _ in range(n_boot):
        sampled_idx = [random.choice(original_index) for _ in range(n_rows)]
        df_boot = df_returns.loc[sampled_idx]

        w_i, sum_i = parametric_max_sharpe(
            df_returns= df_boot,
            tickers= tickers,
            asset_classes= asset_classes,
            class_constraints= class_constraints,
            daily_rf= daily_rf,
            no_short= no_short,
            n_points= n_points,
            regularize_cov= regularize_cov,
            shrink_means= shrink_means,
            alpha= alpha,
            shrink_cov= shrink_cov,
            beta= beta,
            use_ledoitwolf= use_ledoitwolf,
            ewm_cov= ewm_cov,
            ewm_alpha= ewm_alpha
        )
        all_weights.append(w_i)

    w_mich = np.mean(all_weights, axis=0)

    mean_ret, cov_final = _build_final_cov_and_means_for_summary(
        df_returns,
        shrink_means, alpha,
        regularize_cov,
        shrink_cov, beta,
        use_ledoitwolf,
        ewm_cov, ewm_alpha,
        daily_rf
    )

    ann_factor= 252
    ann_rf= daily_rf * ann_factor

    rd= mean_ret @ w_mich
    ann_ret= rd* ann_factor
    ann_vol= float(np.sqrt(w_mich.T@ cov_final@ w_mich)* np.sqrt(ann_factor))
    ann_sharpe= 0.0
    if ann_vol>1e-12:
        ann_sharpe= (ann_ret- ann_rf)/ ann_vol

    summary= {
        "Annual Return (%)": round(ann_ret*100,2),
        "Annual Vol (%)": round(ann_vol*100,2),
        "Sharpe Ratio": round(ann_sharpe,4)
    }
    return w_mich, summary
