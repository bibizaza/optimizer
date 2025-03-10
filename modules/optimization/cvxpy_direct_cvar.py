# File: modules/optimization/cvxpy_direct_cvar.py

import numpy as np
import pandas as pd
import cvxpy as cp

def aggregate_returns(df_daily: pd.DataFrame, freq_choice: str) -> pd.DataFrame:
    """
    Same aggregator as in parametric file:
      daily => no change
      weekly => resample('W').sum()
      monthly => resample('M').sum()
      annual => resample('Y').sum()
    """
    if freq_choice == "daily":
        return df_daily

    if not isinstance(df_daily.index, pd.DatetimeIndex):
        raise ValueError("df_daily index must be a pd.DatetimeIndex for resample")

    if freq_choice == "weekly":
        return df_daily.resample('W').sum()
    elif freq_choice == "monthly":
        return df_daily.resample('M').sum()
    elif freq_choice == "annual":
        return df_daily.resample('Y').sum()
    else:
        return df_daily

def direct_max_return_cvar_constraint_aclass_subtype(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    security_types: list[str],
    class_constraints: dict,
    subtype_constraints: dict,
    cvar_alpha: float = 0.95,
    cvar_limit: float = 0.10,
    daily_rf: float = 0.0,
    no_short: bool = True,
    freq_choice: str = "daily"
):
    """
    Direct approach => Maximize (mean_annual@ w) subject to CVaR <= cvar_limit,
    but the aggregator approach means we first resample returns to user-chosen freq,
    so cvar_limit is at that freq.

    e.g. freq_choice='monthly' => we measure monthly cvar <= 0.10
    """
    # 1) aggregator
    df_agg = aggregate_returns(df_returns, freq_choice)
    if df_agg.shape[0] < 2:
        fallback = np.ones(len(tickers))/ max(len(tickers),1)
        return fallback, {"Annual Return (%)":0.0, "CVaR (%)":0.0, "Sharpe Ratio":0.0}

    n = len(tickers)
    if df_agg.shape[1] != n:
        raise ValueError("df_agg shape mismatch # tickers")
    if len(asset_classes) != n:
        raise ValueError("asset_classes length mismatch.")
    if len(security_types) != n:
        raise ValueError("security_types length mismatch.")

    # 2) Clean aggregator
    df_clean = df_agg.replace([np.inf, -np.inf], np.nan).dropna(how='all', axis=1)
    df_clean = df_clean.dropna(how='all', axis=0).fillna(0.0)

    T_ = df_clean.shape[0]
    if T_<2:
        best_w= np.ones(n)/ n
        return best_w, {
            "Annual Return (%)":0.0,
            "CVaR (%)":0.0,
            "Sharpe Ratio":0.0
        }

    ret_vals = df_clean.values  # (T_ x n)
    mean_periodic = df_clean.mean().values

    # figure out annualization factor
    if freq_choice=="weekly":
        ann_factor = 52
    elif freq_choice=="monthly":
        ann_factor = 12
    elif freq_choice=="annual":
        ann_factor = 1
    else:
        ann_factor = 252

    # define CVXPY variables
    w = cp.Variable(n)
    eta = cp.Variable()
    u = cp.Variable(T_, nonneg=True)

    # objective => maximize mean_periodic@ w * ann_factor
    objective = cp.Maximize(mean_periodic @ w * ann_factor)

    cons = [cp.sum(w)==1]
    if no_short:
        cons.append(w>=0)

    # class constraints
    unique_cls = set(asset_classes)
    for cl_ in unique_cls:
        idxs= [i for i,a_ in enumerate(asset_classes) if a_==cl_]
        cdict= class_constraints.get(cl_,{})
        min_c= cdict.get("min_class_weight",0.0)
        max_c= cdict.get("max_class_weight",1.0)
        cons.append(cp.sum(w[idxs])>= min_c)
        cons.append(cp.sum(w[idxs])<= max_c)

        for i_ in idxs:
            stp= security_types[i_]
            if (cl_, stp) in subtype_constraints:
                subdict= subtype_constraints[(cl_, stp)]
                mini= subdict.get("min_instrument",0.0)
                maxi= subdict.get("max_instrument",1.0)
                cons.append(w[i_]>=mini)
                cons.append(w[i_]<=maxi)

    # cvar constraints => cvar_expr <= cvar_limit
    # for each t => u[t] >= -(ret_vals[t] dot w) - eta
    for t in range(T_):
        cons.append( u[t] >= -(ret_vals[t] @ w) - eta )

    cvar_expr = eta + 1.0/((1 - cvar_alpha)* T_) * cp.sum(u)
    cons.append( cvar_expr <= cvar_limit )

    prob = cp.Problem(objective, cons)
    solved= False
    for solver_ in [cp.ECOS, cp.SCS, cp.CVXOPT]:
        try:
            prob.solve(solver=solver_, verbose=False)
            if prob.status in ["optimal","optimal_inaccurate"] and w.value is not None:
                solved= True
                break
        except:
            pass

    if (not solved) or (w.value is None):
        best_w= np.ones(n)/n
        return best_w, {
            "Annual Return (%)": 0.0,
            "CVaR (%)": 0.0,
            "Sharpe Ratio": 0.0
        }

    best_w= w.value

    # ex-post stats on aggregator scale
    r_ptf = df_clean @ best_w
    mean_p = r_ptf.mean()
    std_p = r_ptf.std()
    ann_ret = float(mean_p* ann_factor)

    # aggregator-based cvar
    losses= -r_ptf.values
    sorted_losses = np.sort(losses)
    T_agg= len(sorted_losses)
    idx_cvar= int(np.ceil(cvar_alpha* T_agg))
    if idx_cvar>= T_agg:
        idx_cvar= T_agg-1
    cvar_est= sorted_losses[idx_cvar:].mean()

    if std_p<1e-12:
        sr_ = 0.0
    else:
        sr_ = (mean_p- daily_rf)/ std_p

    summary= {
        "Annual Return (%)": round(ann_ret*100,2),
        "CVaR (%)": round(cvar_est*100,2),
        "Sharpe Ratio": round(sr_,4)
    }
    return best_w, summary
