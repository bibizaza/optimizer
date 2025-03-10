# File: modules/optimization/cvxpy_parametric_cvar.py

import numpy as np
import pandas as pd
import cvxpy as cp

def aggregate_returns(df_daily: pd.DataFrame, freq_choice: str) -> pd.DataFrame:
    """
    Aggregates (or resamples) daily returns to the chosen frequency:
      'daily'  -> no change,
      'weekly' -> resample('W').sum(),
      'monthly'-> resample('M').sum(),
      'annual' -> resample('Y').sum().

    This is a simplistic aggregator that sums up returns each period, 
    which is approximate if returns can be large. If you want 
    compounding (e.g., (1+r).prod()-1), adapt accordingly.
    """
    if freq_choice == "daily":
        return df_daily

    # Ensure the index is DateTime for resample to work
    if not isinstance(df_daily.index, pd.DatetimeIndex):
        raise ValueError("df_daily index must be a pd.DatetimeIndex for resample")

    if freq_choice == "weekly":
        return df_daily.resample('W').sum()
    elif freq_choice == "monthly":
        return df_daily.resample('M').sum()
    elif freq_choice == "annual":
        return df_daily.resample('Y').sum()
    else:
        # default to daily if unknown
        return df_daily

def parametric_min_cvar_aclass_subtype(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    security_types: list[str],
    class_constraints: dict,
    subtype_constraints: dict,
    cvar_alpha: float = 0.95,
    no_short: bool = True,
    n_points: int = 15,
    daily_rf: float = 0.0,
    freq_choice: str = "daily"
):
    """
    Parametric approach scanning multiple target returns in [targ_min, targ_max],
    each time solving "minimize CVaR subject to mean >= target".

    1) We first aggregate df_returns from daily to the user's chosen frequency 
       (weekly/monthly/etc.) => df_agg
    2) Then we do the usual param approach on that aggregated data.

    Returns
    -------
    best_w : np.ndarray
    summary : dict
        e.g. {"Annual Return (%)","CVaR (%)","Sharpe Ratio"}
    """
    # 1) Aggregate returns
    df_agg = aggregate_returns(df_returns, freq_choice)
    if df_agg.shape[0] < 2:
        fallback = np.ones(len(tickers)) / max(len(tickers), 1)
        return fallback, {
            "Annual Return (%)": 0.0,
            "CVaR (%)": 0.0,
            "Sharpe Ratio": 0.0
        }

    n = len(tickers)
    if df_agg.shape[1] != n:
        raise ValueError("df_agg shape mismatch vs # tickers.")
    if len(asset_classes) != n:
        raise ValueError("asset_classes length mismatch.")
    if len(security_types) != n:
        raise ValueError("security_types length mismatch.")

    # 2) Clean aggregator results
    df_clean = df_agg.replace([np.inf, -np.inf], np.nan).dropna(how='all', axis=1)
    df_clean = df_clean.dropna(how='all', axis=0).fillna(0.0)

    T_ = df_clean.shape[0]
    if T_ < 2:
        fallback = np.ones(n)/n
        return fallback, {"Annual Return (%)":0.0, "CVaR (%)":0.0, "Sharpe Ratio":0.0}

    ret_vals = df_clean.values  # shape (T_, n)
    mean_periodic = df_clean.mean().values
    # If freq_choice is monthly, we might do "annualize" by multiply by 12, etc.
    # For param scanning, we define:
    if freq_choice == "weekly":
        # ~52 periods per year
        ann_factor = 52
    elif freq_choice == "monthly":
        ann_factor = 12
    elif freq_choice == "annual":
        ann_factor = 1
    else:
        # daily by default ~252
        ann_factor = 252

    # We'll define param approach: scan target returns in [targ_min, targ_max]
    # in terms of annual scale
    asset_periodic_ret = mean_periodic * ann_factor
    targ_min = max(0.0, asset_periodic_ret.min())
    targ_max = asset_periodic_ret.max()
    candidate_targets = np.linspace(targ_min, targ_max, n_points)

    best_sharpe = -np.inf
    best_w = np.zeros(n)

    for targ in candidate_targets:
        w = cp.Variable(n)
        eta = cp.Variable()
        u = cp.Variable(T_, nonneg=True)

        # objective => minimize CVaR
        # CVaR = eta + 1 / ((1 - alpha)* T_) * sum(u)
        objective = cp.Minimize(eta + 1.0/((1 - cvar_alpha)* T_)* cp.sum(u))

        cons = [cp.sum(w)==1]
        if no_short:
            cons.append(w >= 0)

        # class constraints
        unique_cls = set(asset_classes)
        for cl_ in unique_cls:
            idxs= [i for i,a_ in enumerate(asset_classes) if a_==cl_]
            cdict= class_constraints.get(cl_, {})
            min_c= cdict.get("min_class_weight", 0.0)
            max_c= cdict.get("max_class_weight", 1.0)
            cons.append(cp.sum(w[idxs])>= min_c)
            cons.append(cp.sum(w[idxs])<= max_c)

            for i_ in idxs:
                stp= security_types[i_]
                if (cl_, stp) in subtype_constraints:
                    stvals= subtype_constraints[(cl_, stp)]
                    mini= stvals.get("min_instrument", 0.0)
                    maxi= stvals.get("max_instrument", 1.0)
                    cons.append(w[i_]>= mini)
                    cons.append(w[i_]<= maxi)

        # return >= targ
        # mean_periodic => average periodic ret
        cons.append((mean_periodic @ w)* ann_factor >= targ)

        # cvar constraints
        # losses[t] = - (ret_vals[t] @ w)
        # u[t] >= -(ret_vals[t]@w) - eta
        for t in range(T_):
            cons.append(u[t] >= -(ret_vals[t] @ w) - eta)

        prob = cp.Problem(objective, cons)
        solved= False
        for solver_ in [cp.ECOS, cp.SCS]:
            try:
                prob.solve(solver=solver_, verbose=False)
                if prob.status in ["optimal","optimal_inaccurate"] and w.value is not None:
                    solved = True
                    break
            except:
                pass
        if not solved or w.value is None:
            continue

        w_val = w.value
        # ex-post sharpe => we'll do it on the aggregated freq
        # aggregator => r_ptf[t] = ret_vals[t] @ w_val
        # We'll "annualize" the mean, then do Sharpe
        r_ptf = df_clean @ w_val
        ptf_mean_period = r_ptf.mean()
        ptf_std_period = r_ptf.std()

        if ptf_std_period < 1e-12:
            sr_ = -np.inf
        else:
            # periodic RF => daily_rf is daily, but if we are monthly => monthly_rf?
            # For simplicity, we just do (ptf_mean_period - daily_rf) ...
            # but a more precise approach: scale daily_rf up or re-estimate a periodic rf
            sr_ = (ptf_mean_period - daily_rf) / ptf_std_period

        if sr_> best_sharpe:
            best_sharpe = sr_
            best_w = w_val.copy()

    # final stats
    if np.allclose(best_w, 0.0):
        return best_w, {"Annual Return (%)": 0.0, "CVaR (%)": 0.0, "Sharpe Ratio": 0.0}

    # compute ex-post stats on the aggregator scale
    r_ptf = df_clean @ best_w
    mean_p = r_ptf.mean()
    std_p = r_ptf.std()
    # approximate annual ret
    ann_ret = float(mean_p * ann_factor)
    # approximate daily-based cvar or aggregator-based cvar => aggregator approach
    # aggregator-based cvar => we define losses[t] = -(r_ptf[t])
    # sort them, etc.
    losses = -r_ptf.values
    sorted_losses = np.sort(losses)
    T_agg = len(sorted_losses)
    idx_cvar = int(np.ceil(cvar_alpha * T_agg))
    if idx_cvar>=T_agg:
        idx_cvar= T_agg-1
    cvar_approx = sorted_losses[idx_cvar:].mean()

    # sharpe aggregator-based
    if std_p<1e-12:
        sr_final=0.0
    else:
        sr_final= (mean_p- daily_rf)/ std_p

    summary= {
        "Annual Return (%)": round(ann_ret*100,2),
        "CVaR (%)": round(cvar_approx*100,2),
        "Sharpe Ratio": round(sr_final,4)
    }
    return best_w, summary
