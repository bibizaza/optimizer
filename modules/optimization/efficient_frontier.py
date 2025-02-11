# modules/optimization/efficient_frontier.py

import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import cvxpy as cp

########################################
# Advanced Mean/Covariance Utilities
########################################

def shrink_mean_to_grand_mean(raw_means: np.ndarray, alpha: float) -> np.ndarray:
    """
    Shrinks each mean toward the grand mean by a factor alpha.
    raw_means: 1D array of individual asset means
    alpha: shrink intensity (0 => no shrink, 1 => fully grand mean)
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
    Shrink covariance toward its diagonal by factor beta.
    If beta=1 => fully diagonal, if beta=0 => original.
    """
    diag_ = np.diag(np.diag(cov_))
    return (1 - beta) * cov_ + beta * diag_

def compute_ewm_cov(df_returns: pd.DataFrame, alpha=0.06) -> np.ndarray:
    """
    Compute exponentially weighted covariance from a returns DataFrame.
    alpha => smoothing factor (0<alpha<1).
    """
    arr = df_returns.values
    n_rows = len(arr)
    weights = (1 - alpha) ** np.arange(n_rows)[::-1]
    weights /= weights.sum()
    mean_w = np.average(arr, axis=0, weights=weights)
    arr_center = arr - mean_w
    cov_ = np.einsum('ti,tj,t->ij', arr_center, arr_center, weights)
    # symmetrize
    return 0.5 * (cov_ + cov_.T)

def ledoitwolf_cov(df_returns: pd.DataFrame) -> np.ndarray:
    """
    Use the sample covariance from the DataFrame as a stand-in
    for a LedoitWolf approach or optionally import sklearn's LedoitWolf.
    Here it's returning .cov() for simplicity.
    """
    return df_returns.cov().values

########################################
# Cached Build Mean/Cov
########################################

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
    Because it is decorated with @st.cache_data, repeated identical calls
    (same DataFrame content + same boolean flags) will be cached.
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

########################################
# CVXPY-based Efficient Frontier
########################################

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
    Build an efficient frontier using CVXPY. We assume:
    - no short selling => weights >= 0
    - sum(weights) = 1
    - optional class constraints for min/max class weights, max instrument weight

    We do a grid of target returns => each solved with a min-vol objective,
    then possibly remove dominated points => final frontier.

    final_w => optional final portfolio to ensure it's included in the frontier
               (only for plotting convenience if you like).
    """
    # 1) build daily means + cov with caching
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
        return np.sum(mean_ret_d * w)

    ret_min_d = mean_ret_d.min()
    ret_max_d = mean_ret_d.max()

    candidate_targets = np.linspace(ret_min_d, ret_max_d, n_points)
    if final_w is not None:
        # ensure final portfolio's return is considered => so we incorporate it in frontier
        final_ret_d = ret_daily(final_w)
        candidate_targets = np.append(candidate_targets, final_ret_d)
        candidate_targets = np.unique(candidate_targets)
        candidate_targets.sort()

    n_assets = df_returns.shape[1]
    tickers = col_tickers if col_tickers else [f"A{i}" for i in range(n_assets)]
    class_constraints = class_constraints or {}
    frontier_points = []

    for targ_d in candidate_targets:
        # define cvxpy variables
        w = cp.Variable(n_assets)
        # objective => min vol => min quad_form(w, cov)
        objective = cp.Minimize(cp.quad_form(w, cov_))

        # constraints => sum(weights)=1, w>=0
        constraints = [cp.sum(w) == 1, w >= 0]

        # class constraints if any
        for cl, cc in class_constraints.items():
            # find which indices => that class
            idxs = []
            for i, tk in enumerate(tickers):
                dfc = df_instruments[df_instruments["#ID"] == tk]
                if not dfc.empty:
                    cl_ = dfc["#Asset"].iloc[0]
                    if cl_ == cl:
                        idxs.append(i)
            if len(idxs) == 0:
                continue
            if "min_class_weight" in cc:
                constraints.append(cp.sum(w[idxs]) >= cc["min_class_weight"])
            if "max_class_weight" in cc:
                constraints.append(cp.sum(w[idxs]) <= cc["max_class_weight"])
            if "max_instrument_weight" in cc:
                for i_ in idxs:
                    constraints.append(w[i_] <= cc["max_instrument_weight"])

        # target return constraint
        constraints.append((mean_ret_d @ w) >= targ_d)

        prob = cp.Problem(objective, constraints)
        prob.solve(solver=cp.SCS, verbose=False)

        if prob.status in ["optimal", "optimal_inaccurate"] and w.value is not None:
            w_ = w.value
            vd = np.sqrt(w_.T @ cov_ @ w_)
            rd = np.sum(mean_ret_d * w_)
            vol_ann = vd * np.sqrt(252)
            ret_ann = rd * 252
            if ret_ann >= clamp_min_return:
                frontier_points.append((vol_ann, ret_ann))

    # sort points by ascending vol
    frontier_points.sort(key=lambda x: x[0])

    # remove dominated if asked
    if remove_dominated:
        cleaned = []
        best_ret = -np.inf
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

########################################
# Frontier Plotting Helpers
########################################

def interpolate_frontier_for_vol(front_vol: np.ndarray, front_ret: np.ndarray, old_vol: float):
    """
    Interpolate the frontier to find the return at 'old_vol' for comparison.
    If old_vol is outside the frontier range, we clamp to the edges.
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

def plot_frontier_comparison(front_vol: np.ndarray, front_ret: np.ndarray,
                             old_vol: float, old_ret: float,
                             new_vol: float, new_ret: float,
                             same_vol: float, same_ret: float,
                             title: str = "Constrained Frontier: Old vs New",
                             arrow_shift: float = 40.0):
    """
    Plotly chart comparing the frontier with the old vs new portfolio, plus a point
    on the frontier that has the same vol as the old.
    """
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

    # draw lines
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
