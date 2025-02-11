import pandas as pd
import numpy as np
from scipy.optimize import minimize
import plotly.express as px

########################################
# Advanced Mean/Covariance Utilities
########################################

def shrink_mean_to_grand_mean(raw_means: np.ndarray, alpha: float) -> np.ndarray:
    grand = np.mean(raw_means)
    return (1 - alpha) * raw_means + alpha * grand

def nearest_pd(cov: np.ndarray, epsilon=1e-12) -> np.ndarray:
    vals, vecs = np.linalg.eigh(cov)
    vals_clipped = np.clip(vals, epsilon, None)
    cov_fixed = (vecs * vals_clipped) @ vecs.T
    return 0.5 * (cov_fixed + cov_fixed.T)

def shrink_cov_diagonal(cov_: np.ndarray, beta: float) -> np.ndarray:
    diag_ = np.diag(np.diag(cov_))
    return (1 - beta) * cov_ + beta * diag_

def compute_ewm_cov(df_returns: pd.DataFrame, alpha=0.06) -> np.ndarray:
    arr = df_returns.values
    n_rows = len(arr)
    weights = (1 - alpha) ** np.arange(n_rows)[::-1]
    weights /= weights.sum()
    mean_w = np.average(arr, axis=0, weights=weights)
    arr_center = arr - mean_w
    cov_ = np.einsum('ti,tj,t->ij', arr_center, arr_center, weights)
    return 0.5 * (cov_ + cov_.T)

def ledoitwolf_cov(df_returns: pd.DataFrame) -> np.ndarray:
    return df_returns.cov().values

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
# Efficient Frontier Functions
########################################

def compute_efficient_frontier(
    df_returns: pd.DataFrame,
    cons: list,           # list of constraints (each a dict)
    bnds: list,           # list of bounds (tuples)
    n_points: int = 50,
    clamp_min_return: float = 0.0,
    remove_dominated: bool = True,
    # advanced options
    do_shrink_means: bool = False,
    alpha: float = 0.3,
    do_shrink_cov: bool = False,
    beta: float = 0.2,
    use_ledoitwolf: bool = False,
    do_ewm: bool = False,
    ewm_alpha: float = 0.06,
    regularize_cov: bool = False,
    # NEW: force final portfolio onto the frontier if provided
    final_w: np.ndarray = None
):
    # Compute advanced daily means and covariance
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
        final_ret_d = ret_daily(final_w)
        candidate_targets = np.append(candidate_targets, final_ret_d)
        candidate_targets = np.unique(candidate_targets)
        candidate_targets.sort()

    frontier_points = []
    w0 = np.ones(len(mean_ret_d)) / len(mean_ret_d)
    for targ_d in candidate_targets:
        eq_con = {"type": "eq", "fun": lambda w, td=targ_d: ret_daily(w) - td}
        all_cons = cons + [eq_con]
        res = minimize(vol_daily, w0, method="SLSQP", bounds=bnds, constraints=all_cons)
        if res.success and res.x is not None:
            w_ = res.x
            vd = vol_daily(w_)
            rd = ret_daily(w_)
            vol_ann = vd * np.sqrt(252)
            ret_ann = rd * 252
            if ret_ann >= clamp_min_return:
                frontier_points.append((vol_ann, ret_ann))
    frontier_points.sort(key=lambda x: x[0])
    if not frontier_points:
        return np.array([]), np.array([])
    cleaned = []
    best_ret = -np.inf
    for (v, r) in frontier_points:
        if remove_dominated:
            if r > best_ret:
                cleaned.append((v, r))
                best_ret = r
        else:
            cleaned.append((v, r))
    if not cleaned:
        return np.array([]), np.array([])
    fvol = np.array([p[0] for p in cleaned])
    fret = np.array([p[1] for p in cleaned])
    return fvol, fret

def interpolate_frontier_for_vol(front_vol: np.ndarray, front_ret: np.ndarray, old_vol: float):
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
    # Convert to percentage with 2 decimals.
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
    fig.add_scatter(x=[old_vol_pct], y=[old_ret_pct],
                    mode="markers", name="Old (Real)",
                    marker=dict(color="red", size=10))
    fig.add_scatter(x=[new_vol_pct], y=[new_ret_pct],
                    mode="markers", name="New (Optimized)",
                    marker=dict(color="green", size=10))
    fig.add_scatter(x=[same_vol_pct], y=[same_ret_pct],
                    mode="markers", name="Frontier (Same Vol)",
                    marker=dict(color="blue", size=12, symbol="diamond"))
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