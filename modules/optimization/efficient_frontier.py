# modules/optimization/efficient_frontier.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import cvxpy as cp

########################################################################
# 1) Utility Functions for Mean/Cov Shrink
########################################################################

def shrink_mean_to_grand_mean(raw_means: np.ndarray, alpha: float) -> np.ndarray:
    """
    Shrinks each mean toward the grand mean by a factor alpha.
    alpha=0 => no shrink, alpha=1 => fully grand mean.
    """
    grand = np.mean(raw_means)
    return (1 - alpha) * raw_means + alpha * grand

def nearest_pd(cov: np.ndarray, epsilon=1e-12) -> np.ndarray:
    """
    Force a matrix to be positive semidefinite by clipping negative eigenvalues.
    """
    vals, vecs = np.linalg.eigh(cov)
    vals_clipped = np.clip(vals, epsilon, None)
    cov_fixed = (vecs * vals_clipped) @ vecs.T
    # symmetrize
    return 0.5 * (cov_fixed + cov_fixed.T)

def shrink_cov_diagonal(cov_: np.ndarray, beta: float) -> np.ndarray:
    """
    Shrink covariance toward diagonal by factor beta.
    beta=0 => original, beta=1 => fully diagonal.
    """
    diag_ = np.diag(np.diag(cov_))
    return (1 - beta) * cov_ + beta * diag_

def compute_ewm_cov(df_returns: pd.DataFrame, alpha=0.06) -> np.ndarray:
    """
    Compute exponentially weighted covariance from a returns DataFrame, alpha in (0,1].
    """
    arr = df_returns.values
    n = len(arr)
    weights = (1 - alpha) ** np.arange(n)[::-1]
    weights /= weights.sum()
    mean_w = np.average(arr, axis=0, weights=weights)
    arr_centered = arr - mean_w
    cov_ = np.einsum("ti,tj,t->ij", arr_centered, arr_centered, weights)
    # symmetrize
    return 0.5 * (cov_ + cov_.T)

def ledoitwolf_cov(df_returns: pd.DataFrame) -> np.ndarray:
    """
    Stub for a real LedoitWolf method. Here we just return the sample cov
    for simplicity, but you can replace with actual LedoitWolf from sklearn.
    """
    return df_returns.cov().values

########################################################################
# 2) Cached Build Mean/Cov
########################################################################

@st.cache_data(show_spinner=False)
def build_advanced_mean_cov(
    df_returns: pd.DataFrame,
    do_shrink_means: bool,
    alpha: float,
    do_shrink_cov: bool,
    beta: float,
    use_ledoitwolf: bool,
    do_ewm: bool,
    ewm_alpha: float,
    regularize_cov: bool
):
    """
    Build daily means and covariance with optional shrinkage or EWM.
    This is cached by Streamlit to avoid recomputing if arguments/data are unchanged.
    """
    raw_means = df_returns.mean().values
    if do_shrink_means and alpha > 0:
        raw_means = shrink_mean_to_grand_mean(raw_means, alpha)

    if do_ewm:
        cov_ = compute_ewm_cov(df_returns, alpha=ewm_alpha)
    else:
        if use_ledoitwolf:
            cov_ = ledoitwolf_cov(df_returns)
        else:
            cov_ = df_returns.cov().values
            if regularize_cov:
                cov_ = nearest_pd(cov_)
            if do_shrink_cov and beta > 0:
                cov_ = shrink_cov_diagonal(cov_, beta)

    return raw_means, cov_

########################################################################
# 3) CVXPY-based Frontier
########################################################################

def compute_efficient_frontier_cvxpy(
    df_returns: pd.DataFrame,
    n_points: int = 50,
    clamp_min_return: float = 0.0,
    remove_dominated: bool = True,
    do_shrink_means: bool = False,
    alpha: float = 0.3,
    do_shrink_cov: bool = False,
    beta: float = 0.2,
    use_ledoitwolf: bool = False,
    do_ewm: bool = False,
    ewm_alpha: float = 0.06,
    regularize_cov: bool = False,
    final_w: np.ndarray = None,
    class_constraints: dict = None,
    col_tickers: list = None,
    df_instruments: pd.DataFrame = None
):
    """
    Build an efficient frontier using CVXPY on df_returns (daily returns).
    - no short => w>=0, sum(w)=1
    - optional class constraints
    - optionally include final_w in target set

    Returns two arrays: (fvol, fret) => frontier vol & ret, both annualized.
    """
    # 1) build daily means + cov (with caching)
    mean_ret_d, cov_ = build_advanced_mean_cov(
        df_returns,
        do_shrink_means, alpha,
        do_shrink_cov, beta,
        use_ledoitwolf, do_ewm, ewm_alpha,
        regularize_cov
    )

    def vol_daily(w):
        return np.sqrt(w.T @ cov_ @ w)
    def ret_daily(w):
        return mean_ret_d @ w

    ret_min_d = mean_ret_d.min()
    ret_max_d = mean_ret_d.max()

    candidate_targets = np.linspace(ret_min_d, ret_max_d, n_points)

    # If we want to ensure final_w's return is in the candidate set:
    if final_w is not None:
        final_ret_d = ret_daily(final_w)
        candidate_targets = np.append(candidate_targets, final_ret_d)
        candidate_targets = np.unique(candidate_targets)
        candidate_targets.sort()

    n_assets = df_returns.shape[1]
    tickers = col_tickers if col_tickers else [f"A{i}" for i in range(n_assets)]
    class_constraints = class_constraints or {}

    frontier_points = []

    for targ_d in candidate_targets:
        w = cp.Variable(n_assets)
        objective = cp.Minimize(cp.quad_form(w, cov_))
        constraints = [cp.sum(w) == 1, w >= 0]

        # If we have class constraints
        for cl, cc in class_constraints.items():
            idxs = []
            for i, tk in enumerate(tickers):
                if df_instruments is not None:
                    dfc = df_instruments[df_instruments["#ID"] == tk]
                    if not dfc.empty:
                        cl_ = dfc["#Asset"].iloc[0]
                        if cl_ == cl:
                            idxs.append(i)
            if len(idxs) > 0:
                if "min_class_weight" in cc:
                    constraints.append(cp.sum(w[idxs]) >= cc["min_class_weight"])
                if "max_class_weight" in cc:
                    constraints.append(cp.sum(w[idxs]) <= cc["max_class_weight"])
                if "max_instrument_weight" in cc:
                    for i_ in idxs:
                        constraints.append(w[i_] <= cc["max_instrument_weight"])

        # target daily return >= targ_d
        constraints.append(ret_daily(w) >= targ_d)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)
        if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
            w_ = w.value
            vd = vol_daily(w_)
            rd = ret_daily(w_)
            vol_ann = vd * np.sqrt(252)
            ret_ann = rd * 252
            if ret_ann >= clamp_min_return:
                frontier_points.append((vol_ann, ret_ann))

    # sort by ascending vol
    frontier_points.sort(key=lambda x: x[0])

    # remove dominated if asked
    if remove_dominated:
        cleaned = []
        best_ret = -9999999
        for (v, r) in frontier_points:
            if r > best_ret:
                cleaned.append((v, r))
                best_ret = r
        frontier_points = cleaned

    if not frontier_points:
        return np.array([]), np.array([])
    fvol = np.array([p[0] for p in frontier_points])
    fret = np.array([p[1] for p in frontier_points])
    return fvol, fret

########################################################################
# 4) A Helper to Specifically Use a 12-Month Window
########################################################################

def compute_efficient_frontier_12m(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    do_shrink_means: bool = False,
    alpha: float = 0.3,
    do_shrink_cov: bool = False,
    beta: float = 0.2,
    use_ledoitwolf: bool = False,
    do_ewm: bool = False,
    ewm_alpha: float = 0.06,
    regularize_cov: bool = False,
    class_constraints: dict = None,
    col_tickers: list = None,
    n_points: int = 50,
    clamp_min_return: float = 0.0,
    remove_dominated: bool = True,
    final_w: np.ndarray = None
):
    """
    1) Subset the last ~12 months (252 trading days) from df_prices
    2) Convert to daily returns
    3) Call compute_efficient_frontier_cvxpy on that ~12-month data
    4) Return (fvol, fret) for the 12-month-based annualized frontier.
    """
    if len(df_prices) < 253:
        # If you have less than 253 rows total, just use all
        df_12m = df_prices.copy()
    else:
        last_day = df_prices.index[-1]
        # ~252 days from the end
        one_year_ago = df_prices.index[-252]
        # If you prefer to do a real date range:
        # one_year_ago = last_day - pd.Timedelta(days=365)
        df_12m = df_prices.loc[one_year_ago:last_day].copy()

    df_returns_12m = df_12m.pct_change().fillna(0.0)

    fvol, fret = compute_efficient_frontier_cvxpy(
        df_returns=df_returns_12m,
        n_points=n_points,
        clamp_min_return=clamp_min_return,
        remove_dominated=remove_dominated,
        do_shrink_means=do_shrink_means,
        alpha=alpha,
        do_shrink_cov=do_shrink_cov,
        beta=beta,
        use_ledoitwolf=use_ledoitwolf,
        do_ewm=do_ewm,
        ewm_alpha=ewm_alpha,
        regularize_cov=regularize_cov,
        final_w=final_w,
        class_constraints=class_constraints,
        col_tickers=col_tickers,
        df_instruments=df_instruments
    )
    return fvol, fret

########################################################################
# 5) Plotting Helpers
########################################################################

def interpolate_frontier_for_vol(front_vol: np.ndarray, front_ret: np.ndarray, old_vol: float):
    """
    Interpolate the frontier to find the ret at 'old_vol' for comparison.
    If old_vol is outside the frontier range, clamp to edges.
    Return (vol, ret).
    """
    if len(front_vol) == 0:
        return None, None
    if old_vol <= front_vol[0]:
        return old_vol, front_ret[0]
    if old_vol >= front_vol[-1]:
        return old_vol, front_ret[-1]

    for i in range(len(front_vol) - 1):
        v0 = front_vol[i]
        v1 = front_vol[i+1]
        if v0 <= old_vol <= v1:
            r0 = front_ret[i]
            r1 = front_ret[i+1]
            if (v1 - v0) == 0:
                return old_vol, r0
            ratio = (old_vol - v0) / (v1 - v0)
            matched_ret = r0 + ratio * (r1 - r0)
            return old_vol, matched_ret
    return old_vol, front_ret[-1]

def plot_frontier_comparison(front_vol: np.ndarray,
                             front_ret: np.ndarray,
                             old_vol: float, old_ret: float,
                             new_vol: float, new_ret: float,
                             same_vol: float, same_ret: float,
                             title: str = "Constrained Frontier: Old vs New",
                             arrow_shift: float = 40.0):
    """
    Plotly chart comparing frontier with old vs new portfolio, plus a point
    on the frontier at old vol. All returns are in annual % (front_ret, old_ret, new_ret).
    """
    import plotly.express as px
    def fmt2(x):
        return float(f"{x*100:.2f}")

    fvol_pct = [fmt2(v) for v in front_vol]
    fret_pct = [fmt2(r) for r in front_ret]
    old_vol_pct = fmt2(old_vol)
    old_ret_pct = fmt2(old_ret)
    new_vol_pct = fmt2(new_vol)
    new_ret_pct = fmt2(new_ret)
    same_vol_pct = fmt2(same_vol)
    same_ret_pct = fmt2(same_ret)

    df_front = pd.DataFrame({"Vol(%)": fvol_pct, "Ret(%)": fret_pct})
    fig = px.scatter(df_front, x="Vol(%)", y="Ret(%)", title=title,
                     labels={"Vol(%)": "Annual Vol (%)", "Ret(%)": "Annual Return (%)"})
    fig.update_traces(mode="markers+lines")

    # Old vs new vs sameVol
    fig.add_scatter(x=[old_vol_pct], y=[old_ret_pct],
                    mode="markers", name="Old (Real)",
                    marker=dict(color="red", size=10))
    fig.add_scatter(x=[new_vol_pct], y=[new_ret_pct],
                    mode="markers", name="New (Optimized)",
                    marker=dict(color="green", size=10))
    fig.add_scatter(x=[same_vol_pct], y=[same_ret_pct],
                    mode="markers", name="Frontier (Same Vol)",
                    marker=dict(color="blue", size=12, symbol="diamond"))

    # optional lines
    ylo = min(old_ret_pct, same_ret_pct)
    yhi = max(old_ret_pct, same_ret_pct)
    left_x = min(fvol_pct + [old_vol_pct, new_vol_pct, same_vol_pct])
    fig.add_shape(type="line", xref="x", yref="y",
                  x0=old_vol_pct, x1=old_vol_pct,
                  y0=ylo, y1=yhi,
                  line=dict(color="lightgrey", dash="dash", width=2),
                  layer="below")
    fig.add_shape(type="line", xref="x", yref="y",
                  x0=left_x, x1=same_vol_pct,
                  y0=same_ret_pct, y1=same_ret_pct,
                  line=dict(color="lightgrey", dash="dash", width=2),
                  layer="below")

    fig.add_annotation(xref="x", yref="y",
                       x=left_x, y=same_ret_pct,
                       text=f"{same_ret_pct:.2f}%",
                       showarrow=True, arrowhead=2,
                       arrowcolor="lightgrey", ax=arrow_shift, ay=0,
                       font=dict(color="lightgrey"))
    return fig