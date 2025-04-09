import numpy as np
import pandas as pd
import cvxpy as cp

def aggregate_returns(df_daily: pd.DataFrame, freq_choice: str="daily") -> pd.DataFrame:
    """
    Resample daily returns to the chosen frequency:
      'daily'   => no resample
      'weekly'  => resample('W').sum()
      'monthly' => resample('M').sum()
      'annual'  => resample('Y').sum()

    If freq_choice is unrecognized or 'daily', we just return df_daily as is.
    Index must be a DatetimeIndex for resample to work properly.
    """
    if freq_choice == "daily":
        return df_daily

    if not isinstance(df_daily.index, pd.DatetimeIndex):
        raise ValueError("df_daily must have a DatetimeIndex to resample for CVaR freq.")

    if freq_choice == "weekly":
        return df_daily.resample('W').sum()
    elif freq_choice == "monthly":
        return df_daily.resample('M').sum()
    elif freq_choice == "annual":
        return df_daily.resample('Y').sum()
    else:
        # fallback => daily
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
    Scenario-based parametric CVaR approach:
      1) Aggregates returns to freq_choice (daily/weekly/monthly/annual).
      2) Binary-search for highest feasible target => minimize CVaR subject to mean >= target.
      3) If no feasible => return (None, None) so rolling can fallback to old weights.

    Returns: (best_w, summary) or (None,None)
    summary => {"Annual Return (%)", "CVaR (%)", "Sharpe Ratio"}
    """
    # 1) Resample the daily returns to freq_choice => scenario-based
    df_agg = aggregate_returns(df_returns, freq_choice=freq_choice)
    df_agg = df_agg.replace([np.inf, -np.inf], np.nan).dropna(how='all').fillna(0.0)
    if df_agg.shape[0] < 2 or df_agg.shape[1] < 1:
        return None, None

    # Make sure columns match tickers, in order
    missing_cols = [tk for tk in tickers if tk not in df_agg.columns]
    if missing_cols:
        # Some tickers missing => fallback
        return None, None
    df_agg = df_agg[tickers]

    T_, n_ = df_agg.shape
    if n_ != len(tickers):
        return None, None

    ret_vals = df_agg.values  # shape (T_, n_)
    mean_periodic = df_agg.mean().values  # shape (n_,)

    # 2) Decide an annualization factor
    if freq_choice=="weekly":
        ann_factor = 52
    elif freq_choice=="monthly":
        ann_factor = 12
    elif freq_choice=="annual":
        ann_factor = 1
    else:
        ann_factor = 252  # daily

    # scanning range => mean_periodic * ann_factor
    eq_w = np.ones(n_)/n_
    eq_ret_ann = float(mean_periodic @ eq_w) * ann_factor

    targ_min = max(0.0, eq_ret_ann * 0.2)   # or you can use actual min of assets
    targ_max = clamp_factor * eq_ret_ann    # some upper guess

    # We'll do a binary search
    best_sharpe = -np.inf
    best_w = None
    low = targ_min
    high = targ_max

    def solve_cvar_for_target(targ: float):
        w = cp.Variable(n_)
        eta = cp.Variable()
        u = cp.Variable(T_, nonneg=True)

        objective = cp.Minimize( eta + (1.0/((1 - cvar_alpha)* T_))* cp.sum(u) )
        cons = [cp.sum(w) == 1.0]
        if no_short:
            cons.append(w >= 0)

        # Class & subtype constraints
        unique_cls = set(asset_classes)
        for cl_ in unique_cls:
            idxs = [i for i,a_ in enumerate(asset_classes) if a_==cl_]
            cdict = class_constraints.get(cl_, {})
            min_c = cdict.get("min_class_weight", 0.0)
            max_c = cdict.get("max_class_weight", 1.0)
            if idxs:
                cons.append( cp.sum(w[idxs]) >= min_c )
                cons.append( cp.sum(w[idxs]) <= max_c )
                # subtype
                for i_ in idxs:
                    stp = security_types[i_]
                    if (cl_, stp) in subtype_constraints:
                        subvals = subtype_constraints[(cl_, stp)]
                        mini = subvals.get("min_instrument", 0.0)
                        maxi = subvals.get("max_instrument", 1.0)
                        cons.append( w[i_] >= mini )
                        cons.append( w[i_] <= maxi )

        if max_weight_each < 1.0:
            cons.append(w <= max_weight_each)

        # mean >= targ
        cons.append( (mean_periodic @ w)* ann_factor >= targ )

        # cvar constraints => for each scenario t => u[t] >= -( ret_vals[t]@ w ) - eta
        for t_ in range(T_):
            cons.append( u[t_] >= -(ret_vals[t_] @ w) - eta )

        prob = cp.Problem(objective, cons)
        solved = False
        w_sol = None
        for solver_ in [cp.ECOS, cp.SCS]:
            try:
                prob.solve(solver=solver_, verbose=False)
                if (prob.status in ["optimal","optimal_inaccurate"]) and (w.value is not None):
                    solved = True
                    w_sol = w.value
                    break
            except:
                pass
        return w_sol, solved

    for _ in range(max_iter):
        mid = 0.5*(low + high)
        w_sol, feasible = solve_cvar_for_target(mid)
        if feasible and (w_sol is not None):
            # Update low => we push the target higher
            low = mid
            # Evaluate ex-post Sharpe on aggregator scale
            r_ptf = ret_vals @ w_sol  # shape (T_,)
            m_ = r_ptf.mean()
            s_ = r_ptf.std()
            if s_ < 1e-12:
                sr_ = -np.inf
            else:
                sr_ = (m_ - daily_rf) / s_
            if sr_ > best_sharpe:
                best_sharpe = sr_
                best_w = w_sol
        else:
            # Not feasible => lower high
            high = mid

        if (high - low) < 1e-8:
            break

    if best_w is None:
        # No feasible => fallback
        return None, None

    # Final summary => ex-post stats on aggregator scale
    r_ptf = ret_vals @ best_w
    m_ = r_ptf.mean()
    s_ = r_ptf.std()
    ann_ret = float(m_ * ann_factor)
    losses = -r_ptf
    sorted_losses = np.sort(losses)
    idx_cvar = int(np.ceil(cvar_alpha * len(sorted_losses)))
    if idx_cvar >= len(sorted_losses):
        idx_cvar = len(sorted_losses) - 1
    cvar_val = sorted_losses[idx_cvar:].mean()

    if s_ < 1e-12:
        sr_ = 0.0
    else:
        sr_ = (m_ - daily_rf) / s_

    summary = {
        "Annual Return (%)": round(ann_ret*100, 2),
        "CVaR (%)":          round(cvar_val*100, 2),
        "Sharpe Ratio":      round(sr_, 4)
    }
    return best_w, summary


###############################################################################
# 2) A simple `param_cvar_fn` for rolling
###############################################################################
def param_cvar_fn(sub_ret: pd.DataFrame, old_w: np.ndarray,
                  tickers: list[str], asset_classes: list[str],
                  security_types: list[str],
                  class_constraints: dict,
                  subtype_constraints: dict,
                  cvar_alpha: float=0.95,
                  no_short: bool=True,
                  daily_rf: float=0.0,
                  freq_choice: str="daily",
                  clamp_factor: float=1.5,
                  max_weight_each: float=1.0,
                  max_iter: int=10
                 ):
    """
    A thin wrapper to call dynamic_min_cvar_with_fallback(...) for scenario-based param CVaR.
    This is the function you pass as `param_cvar_fn` in rolling_backtest_monthly_param_cvar.

    sub_ret => daily returns snippet
    old_w => fallback from rolling
    cvar_alpha => e.g. 0.95
    freq_choice => 'daily','weekly','monthly','annual'
    etc.
    """
    # sub_ret shape => (T, n)
    tickers_ = sub_ret.columns.tolist()
    # be sure we pass the same order to dynamic_min_cvar
    # or you can rely on `tickers` if guaranteed matching columns
    if len(tickers_) != len(tickers):
        # or reorder sub_ret
        sub_ret = sub_ret[tickers]

    w_opt, summ = dynamic_min_cvar_with_fallback(
        df_returns=sub_ret,
        tickers=tickers,
        asset_classes=asset_classes,
        security_types=security_types,
        class_constraints=class_constraints,
        subtype_constraints=subtype_constraints,
        old_w=old_w,
        cvar_alpha=cvar_alpha,
        no_short=no_short,
        daily_rf=daily_rf,
        freq_choice=freq_choice,
        clamp_factor=clamp_factor,
        max_weight_each=max_weight_each,
        max_iter=max_iter
    )
    return w_opt, summ
