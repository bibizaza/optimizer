# File: modules/optimization/cvxpy_parametric_cvar.py

import numpy as np
import pandas as pd
import cvxpy as cp

def aggregate_returns(df_daily: pd.DataFrame, freq_choice: str="daily") -> pd.DataFrame:
    """
    Resample daily returns to the chosen frequency:
      'daily'   => no resample
      'weekly'  => df_daily.resample('W').sum()
      'monthly' => df_daily.resample('M').sum()
      'annual'  => df_daily.resample('Y').sum()

    If freq_choice is unrecognized or 'daily', we just return df_daily as is.
    Index must be a DatetimeIndex for resample to work properly.
    """
    if freq_choice=="daily":
        return df_daily

    if not isinstance(df_daily.index, pd.DatetimeIndex):
        raise ValueError("df_daily must have a DatetimeIndex to resample for CVaR freq.")

    if freq_choice=="weekly":
        return df_daily.resample('W').sum()
    elif freq_choice=="monthly":
        return df_daily.resample('M').sum()
    elif freq_choice=="annual":
        return df_daily.resample('Y').sum()
    else:
        # fallback => treat as daily
        return df_daily


def dynamic_min_cvar_with_fallback(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    security_types: list[str],
    class_constraints: dict,
    subtype_constraints: dict,
    old_w: np.ndarray,
    cvar_alpha: float = 0.95,
    no_short: bool = True,
    daily_rf: float = 0.0,
    freq_choice: str = "daily",
    clamp_factor: float = 1.5,
    max_weight_each: float = 1.0,
    max_iter: int = 10
):
    """
    A parametric CVaR approach that:
      1) Aggregates returns to freq_choice (daily, weekly, monthly, annual).
      2) Binary-searches for the highest feasible target => minimize CVaR subject to mean >= target.
      3) If no feasible solution => returns (None, None), so rolling can fallback to old weights.

    :param df_returns: (T x n) daily returns. We'll internally resample to freq_choice.
    :param tickers: list of ticker names, length n
    :param asset_classes: parallel list of class strings, length n
    :param security_types: parallel list of security type strings, length n
    :param class_constraints: {class_name: {"min_class_weight":..., "max_class_weight":...}}
    :param subtype_constraints: {(class, sec_type): {"min_instrument":..., "max_instrument":...}}
    :param old_w: fallback weights if infeasible
    :param cvar_alpha: typical 0.95
    :param no_short: if True => w >= 0
    :param daily_rf: daily risk-free => used for ex-post Sharpe
    :param freq_choice: "daily","weekly","monthly","annual"
    :param clamp_factor: we clamp our highest scanning target to clamp_factor * eq_weight return
    :param max_weight_each: e.g. 1.0 => no single asset can exceed 100%
    :param max_iter: e.g. 10 => number of binary search steps

    Return: (best_w, summary) or (None,None) if no feasible solution.
    summary => {"Annual Return (%)", "CVaR (%)", "Sharpe Ratio"}
    """
    print("\nDEBUG => dynamic_min_cvar_with_fallback => freq_choice=", freq_choice)
    print("DEBUG => tickers:", tickers)

    # 1) aggregator => from daily to weekly/monthly/annual if chosen
    df_agg = aggregate_returns(df_returns, freq_choice)
    print("DEBUG => aggregator columns after resample =>", df_agg.columns.tolist())

    # 2) Reorder columns by tickers if present
    missing_cols = [tk for tk in tickers if tk not in df_agg.columns]
    if missing_cols:
        print("WARNING => aggregator missing these tickers:", missing_cols)
        # fallback => no feasible
        return None, None

    df_agg = df_agg[tickers]
    # remove inf, drop all-NaN rows, fill leftover
    df_agg = df_agg.replace([np.inf, -np.inf], np.nan).dropna(how="all").fillna(0.0)
    if df_agg.shape[0]<2:
        print("WARNING => aggregator <2 rows => fallback.")
        return None, None

    ret_vals = df_agg.values
    T_, n_ = ret_vals.shape
    if n_ != len(tickers):
        print("ERROR => aggregator columns != # of tickers => fallback.")
        return None, None

    print(f"DEBUG => final aggregator shape=({T_},{n_}), ret_vals OK for CVaR.")
    mean_periodic = df_agg.mean().values

    # 2b) figure out annual factor
    if freq_choice=="weekly":
        ann_factor= 52
    elif freq_choice=="monthly":
        ann_factor= 12
    elif freq_choice=="annual":
        ann_factor= 1
    else:
        ann_factor= 252  # daily

    # scanning range
    targ_min = max(0.0, (mean_periodic* ann_factor).min())
    raw_targ_max = (mean_periodic* ann_factor).max()

    eq_w = np.ones(n_)/ n_
    eq_ret_ann = float(mean_periodic @ eq_w)* ann_factor
    guess_high = min(raw_targ_max, clamp_factor* eq_ret_ann)
    if guess_high < targ_min:
        guess_high= targ_min

    low= targ_min
    high= guess_high
    best_sharpe= -np.inf
    best_w= None

    def solve_cvar_for_target(target_val: float):
        w= cp.Variable(n_)
        eta= cp.Variable()
        u= cp.Variable(T_, nonneg=True)

        objective= cp.Minimize(eta + 1.0/((1- cvar_alpha)* T_)* cp.sum(u))

        # constraints => sum(w)=1, w>=0 if no_short
        cons= [cp.sum(w)==1]
        if no_short:
            cons.append(w>=0)

        # class constraints
        unique_cls= set(asset_classes)
        for cl_ in unique_cls:
            idxs = [i for i,a_ in enumerate(asset_classes) if a_== cl_]
            cdict= class_constraints.get(cl_, {})
            min_c= cdict.get("min_class_weight",0.0)
            max_c= cdict.get("max_class_weight",1.0)
            if idxs:
                cons.append(cp.sum(w[idxs])>= min_c)
                cons.append(cp.sum(w[idxs])<= max_c)
                # subtype
                for i_ in idxs:
                    stp= security_types[i_]
                    if (cl_, stp) in subtype_constraints:
                        stvals= subtype_constraints[(cl_, stp)]
                        mini= stvals.get("min_instrument",0.0)
                        maxi= stvals.get("max_instrument",1.0)
                        cons.append(w[i_]>= mini)
                        cons.append(w[i_]<= maxi)

        if max_weight_each<1.0:
            cons.append(w<= max_weight_each)

        # mean >= target
        cons.append( (mean_periodic @ w)* ann_factor >= target_val )

        # CVaR constraints => for each t => u[t]>= -(ret_vals[t_] @ w)-eta
        for t_ in range(T_):
            cons.append( u[t_] >= -(ret_vals[t_] @ w) - eta )

        prob= cp.Problem(objective, cons)
        solved= False
        w_sol= None
        for solver_ in [cp.ECOS, cp.SCS]:
            try:
                prob.solve(solver=solver_, verbose=False)
                if prob.status in ["optimal","optimal_inaccurate"] and w.value is not None:
                    solved= True
                    w_sol= w.value
                    break
            except:
                pass
        return w_sol, solved

    # 3) binary search or scanning
    for _ in range(max_iter):
        mid= 0.5*(low+high)
        w_sol, feasible= solve_cvar_for_target(mid)
        if feasible and w_sol is not None:
            # push up
            low= mid
            # ex-post daily Sharpe
            r_ptf= ret_vals @ w_sol
            m_= r_ptf.mean()
            s_= r_ptf.std()
            if s_<1e-12:
                sr_= -np.inf
            else:
                sr_= (m_ - daily_rf)/ s_
            if sr_> best_sharpe:
                best_sharpe= sr_
                best_w= w_sol
        else:
            # not feasible => reduce high
            high= mid
        if (high- low)<1e-6:
            break

    if best_w is None:
        print("DEBUG => dynamic_min_cvar => no feasible => fallback.")
        return None, None

    # 4) final ex-post stats
    r_ptf= ret_vals @ best_w
    m_= r_ptf.mean()
    s_= r_ptf.std()
    ann_ret= float(m_* ann_factor)

    # cvar => sort negative of the returns
    losses= -r_ptf
    sorted_losses= np.sort(losses)
    idx_cvar= int(np.ceil(cvar_alpha* len(sorted_losses)))
    if idx_cvar>= len(sorted_losses):
        idx_cvar= len(sorted_losses)-1
    cvar_val= sorted_losses[idx_cvar:].mean()

    if s_<1e-12:
        sr_=0.0
    else:
        sr_= (m_ - daily_rf)/ s_

    summary= {
        "Annual Return (%)": round(ann_ret*100,2),
        "CVaR (%)": round(cvar_val*100,2),
        "Sharpe Ratio": round(sr_,4)
    }
    return best_w, summary