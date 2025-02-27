import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dateutil.relativedelta import relativedelta
import riskfolio as rf
from collections import defaultdict

#####################################################
# 1) Utility: nearest_pd
#####################################################
def nearest_pd(mat: np.ndarray, eps_init=1e-12, tries=5)-> np.ndarray:
    """
    Force a covariance matrix to be positive semidefinite by repeated attempts
    at clipping negative eigenvalues, expanding epsilon if needed.
    """
    mat_ = 0.5 * (mat + mat.T)  # ensure symmetry
    eps = eps_init
    for _ in range(tries):
        try:
            vals, vecs = np.linalg.eigh(mat_)
            vals_clipped = np.clip(vals, eps, None)
            return (vecs * vals_clipped) @ vecs.T
        except np.linalg.LinAlgError:
            eps *= 10
    # fallback => diagonal
    diag_ = np.mean(np.diag(mat_))
    return diag_ * np.eye(mat_.shape[0])

#####################################################
# 2) Extended Metrics & Formatting
#####################################################
def compute_extended_metrics(series_abs: pd.Series, daily_rf: float = 0.0) -> dict:
    freq = 252
    if len(series_abs) < 2:
        return dict.fromkeys([
            "Total Return","Annual Return","Annual Vol","Sharpe","MaxDD","TimeToRecovery",
            "VaR_1M99","CVaR_1M99","Skew","Kurtosis","Sortino","Calmar","Omega"
        ], 0.0)

    daily_ret = series_abs.pct_change().dropna()
    n_days = len(daily_ret)
    if n_days == 0:
        return dict.fromkeys([
            "Total Return","Annual Return","Annual Vol","Sharpe","MaxDD","TimeToRecovery",
            "VaR_1M99","CVaR_1M99","Skew","Kurtosis","Sortino","Calmar","Omega"
        ], 0.0)

    total_ret = series_abs.iloc[-1] / series_abs.iloc[0] - 1
    ann_ret = (1 + total_ret)**(freq / n_days) - 1
    ann_vol = daily_ret.std() * np.sqrt(freq)
    ann_rf = daily_rf * freq
    sharpe = (ann_ret - ann_rf) / ann_vol if ann_vol > 1e-12 else 0.0

    run_max = series_abs.cummax()
    dd = series_abs / run_max - 1
    maxdd = dd.min() if not dd.empty else 0.0
    dd_idx = dd.idxmin() if not dd.empty else None
    if dd_idx is None:
        ttr = 0
    else:
        peak_val = run_max.loc[dd_idx]
        after_dd = series_abs.loc[dd_idx:]
        rec = after_dd[after_dd >= peak_val].index
        ttr = (rec[0] - dd_idx).days if len(rec) > 0 else (series_abs.index[-1] - dd_idx).days

    lookb = 21  # ~1 month
    df_1m = series_abs.pct_change(lookb).dropna()
    if len(df_1m) < 2:
        var_1m99 = 0.0
        cvar_1m99 = 0.0
    else:
        sorted_vals = np.sort(df_1m.values)
        idx = int(0.01 * len(sorted_vals))
        idx = max(idx, 0)
        var_1m99 = sorted_vals[idx]
        tail = sorted_vals[sorted_vals <= var_1m99]
        cvar_1m99 = tail.mean() if len(tail) > 0 else var_1m99

    skew_ = daily_ret.skew()
    kurt_ = daily_ret.kurt()
    downside = daily_ret[daily_ret < 0]
    downside_vol = downside.std() * np.sqrt(freq) if len(downside) > 1 else 1e-12
    sortino = (ann_ret - ann_rf) / downside_vol if downside_vol > 1e-12 else 0.0
    calmar = ann_ret / abs(maxdd) if maxdd < 0 else 0.0
    pos_sum = daily_ret[daily_ret > 0].sum()
    neg_sum = daily_ret[daily_ret < 0].sum()
    omega = pos_sum / abs(neg_sum) if abs(neg_sum) > 1e-12 else 999.0

    return {
        "Total Return": total_ret,
        "Annual Return": ann_ret,
        "Annual Vol": ann_vol,
        "Sharpe": sharpe,
        "MaxDD": maxdd,
        "TimeToRecovery": ttr,
        "VaR_1M99": var_1m99,
        "CVaR_1M99": cvar_1m99,
        "Skew": skew_,
        "Kurtosis": kurt_,
        "Sortino": sortino,
        "Calmar": calmar,
        "Omega": omega
    }

def format_ext_table(oldm: dict, newm: dict) -> pd.DataFrame:
    keys = [
        "Total Return","Annual Return","Annual Vol","Sharpe","MaxDD","TimeToRecovery",
        "VaR_1M99","CVaR_1M99","Skew","Kurtosis","Sortino","Calmar","Omega"
    ]
    rows = []
    for k in keys:
        rows.append((k, oldm.get(k, 0.0), newm.get(k, 0.0)))
    df = pd.DataFrame(rows, columns=["Metric", "Old", "New"])
    df.set_index("Metric", inplace=True)
    return df

#####################################################
# 3) Basic Data & Old Portfolio Tools
#####################################################
def parse_excel(file, streamlit_sheet="streamlit", histo_sheet="Histo_Price"):
    df_instruments = pd.read_excel(file, sheet_name=streamlit_sheet, header=0)
    df_prices_raw = pd.read_excel(file, sheet_name=histo_sheet, header=0)
    if df_prices_raw.columns[0] != "Date":
        df_prices_raw.rename(columns={df_prices_raw.columns[0]: "Date"}, inplace=True)
    df_prices_raw["Date"] = pd.to_datetime(df_prices_raw["Date"], errors="coerce")
    df_prices_raw.dropna(subset=["Date"], inplace=True)
    df_prices_raw.set_index("Date", inplace=True)
    df_prices_raw.sort_index(inplace=True)
    df_prices_raw = df_prices_raw.apply(pd.to_numeric, errors="coerce")
    return df_instruments, df_prices_raw

def clean_df_prices(df_prices: pd.DataFrame, min_coverage=0.8) -> pd.DataFrame:
    df = df_prices.copy()
    coverage = df.notna().sum(axis=1)
    thresh = df.shape[1] * min_coverage
    df = df[coverage >= thresh]
    df = df.ffill().bfill()
    return df

def build_old_portfolio_line(df_instruments: pd.DataFrame, df_prices: pd.DataFrame) -> pd.Series:
    df = df_prices.ffill().bfill().sort_index()
    ticker_qty = {r["#ID"]: r["#Quantity"] for _, r in df_instruments.iterrows()}
    cols = df.columns
    old_shares = np.array([ticker_qty.get(c, 0.0) for c in cols])
    vals = [np.sum(old_shares * row.values) for _, row in df.iterrows()]
    sr = pd.Series(vals, index=df.index, name="Old_Ptf")
    if len(sr) > 1 and sr.iloc[0] > 1e-12:
        sr = sr / sr.iloc[0]
    return sr

#####################################################
# 4) Trade Buffer & Transaction Cost
#####################################################
def compute_tx_cost(port_val: float, old_w: np.ndarray, new_w: np.ndarray, tx_val: float, tx_type: str) -> float:
    turnover = np.sum(np.abs(new_w - old_w))
    if tx_type == "percentage":
        return port_val * turnover * tx_val
    else:
        traded = (np.abs(new_w - old_w) > 1e-12).sum()
        return traded * tx_val

def apply_trade_buffer(old_w: np.ndarray, new_w: np.ndarray, thr: float) -> np.ndarray:
    diffs = new_w - old_w
    updated = new_w.copy()
    for i in range(len(updated)):
        if abs(float(diffs[i])) < thr:
            updated[i] = old_w[i]
    total = np.sum(updated)
    if total <= 0:
        return old_w if np.sum(old_w) > 1e-12 else np.ones(len(updated)) / len(updated)
    return updated / total

#####################################################
# 5) Riskfolio-based Max Sharpe (with NaN checks)
#####################################################
def param_max_sharpe_aclass_subtype(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    security_types: list[str],
    daily_rf: float = 0.0,
    frequency: int = 252,
    class_constraints: dict = None,
    subtype_constraints: dict = None,
    keep_current: bool = False,
    old_class_alloc: dict = None,
    buffer_pct: float = 0.0,
    do_ewm: bool = False,
    ewm_alpha: float = 0.06,
    do_shrink_means: bool = False,
    alpha_mean: float = 0.3,
    do_shrink_cov: bool = False,
    beta_cov: float = 0.2
) -> np.ndarray:
    if class_constraints is None:
        class_constraints = {}
    if subtype_constraints is None:
        subtype_constraints = {}

    # 1) Clean the returns (remove inf, fill NaN with 0, drop all-NaN col/row)
    df_ret = df_returns.replace([np.inf, -np.inf], np.nan)
    df_ret = df_ret.dropna(how='all', axis=1)  # drop columns entirely NaN
    df_ret = df_ret.dropna(how='all', axis=0)  # drop rows entirely NaN
    df_ret = df_ret.fillna(0.0)

    n = df_ret.shape[1]
    if n < 1:
        # Not enough columns to optimize
        return np.array([])

    # 2) Build raw mean/cov
    if do_ewm:
        a_ = max(1e-12, min(ewm_alpha, 1.0))
        df_mu = df_ret.ewm(alpha=a_, adjust=False).mean()
        raw_mu = df_mu.iloc[-1].values
        df_cov = df_ret.ewm(alpha=a_, adjust=False).cov()
        last_d = df_ret.index[-1]
        try:
            raw_cov = df_cov.xs(last_d, level=0).values
        except KeyError:
            # If the ewm cov is empty or no data for last_d, fallback
            return np.ones(n)/n
    else:
        raw_mu = df_ret.mean().values
        raw_cov = df_ret.cov().values

    # 3) Convert any leftover NaN/inf => 0
    raw_mu = np.nan_to_num(raw_mu, nan=0.0, posinf=0.0, neginf=0.0)
    raw_cov = np.nan_to_num(raw_cov, nan=0.0, posinf=0.0, neginf=0.0)

    # 4) Shrink means if requested
    if do_shrink_means and alpha_mean > 1e-12:
        gm = np.mean(raw_mu)
        raw_mu = (1 - alpha_mean)* raw_mu + alpha_mean* gm

    # 5) nearest_pd + shrink covariance
    raw_cov = nearest_pd(raw_cov, 1e-12, 5)
    if do_shrink_cov and beta_cov>1e-12:
        diag_ = np.mean(np.diag(raw_cov))
        raw_cov = (1 - beta_cov)* raw_cov + beta_cov* diag_* np.eye(n)

    # Final check => if not finite, fallback
    if not np.isfinite(raw_cov).all():
        st.write("Cov matrix still has non-finite values => fallback to equal weights.")
        return np.ones(n)/n

    # 6) Build riskfolio Portfolio
    port = rf.Portfolio(returns=df_ret)
    port.mu = pd.Series(raw_mu, index=df_ret.columns)
    port.cov = pd.DataFrame(raw_cov, index=df_ret.columns, columns=df_ret.columns)

    # Prepare class->list-of-index
    from collections import defaultdict
    class2idx = defaultdict(list)
    for i, cl_ in enumerate(asset_classes):
        class2idx[cl_].append(i)

    A_ineq = []
    b_ineq = []

    def add_sum_le(idxs, limit):
        row = np.zeros(n)
        for ix in idxs:
            row[ix] = 1.0
        A_ineq.append(row)
        b_ineq.append(limit)

    def add_sum_ge(idxs, limit):
        row = np.zeros(n)
        for ix in idxs:
            row[ix] = -1.0
        A_ineq.append(row)
        b_ineq.append(-limit)

    # keep_current => old +/- buffer
    if keep_current and old_class_alloc is not None:
        for cl, oldw in old_class_alloc.items():
            idxs = class2idx[cl]
            if not idxs:
                continue
            lo = max(0.0, oldw - buffer_pct)
            hi = min(1.0, oldw + buffer_pct)
            add_sum_le(idxs, hi)
            add_sum_ge(idxs, lo)
    else:
        # custom class constraints
        for cl, cdict in class_constraints.items():
            idxs = class2idx[cl]
            if not idxs:
                continue
            mn = cdict.get("min_class_weight", 0.0)
            mx = cdict.get("max_class_weight", 1.0)
            add_sum_le(idxs, mx)
            add_sum_ge(idxs, mn)

    # Subtype => per ticker
    for i, (ac, stp) in enumerate(zip(asset_classes, security_types)):
        if (ac, stp) in subtype_constraints:
            stvals = subtype_constraints[(ac, stp)]
            mi = stvals.get("min_instrument", 0.0)
            mx = stvals.get("max_instrument", 1.0)
            if mx < 0.9999999:
                row_up = np.zeros(n)
                row_up[i] = 1.0
                A_ineq.append(row_up)
                b_ineq.append(mx)
            if mi > 1e-12:
                row_dn = np.zeros(n)
                row_dn[i] = -1.0
                A_ineq.append(row_dn)
                b_ineq.append(-mi)

    if A_ineq:
        A_ = np.array(A_ineq)
        b_ = np.array(b_ineq)
        if b_.ndim == 1:
            b_ = b_.reshape(-1,1)
        port.ainequality = A_
        port.binequality = b_

    # 7) Solve for Max Sharpe
    try:
        w_sol = port.optimization(
            model='Classic',
            rm='MV',
            obj='Sharpe',
            rf=daily_rf* frequency,
            hist=True
        )
    except Exception as e:
        st.write("Solver exception:", e)
        # fallback => equal weights
        return np.ones(n)/n

    if w_sol is None:
        return np.ones(n)/n

    return w_sol.values.flatten()

#####################################################
# 6) SHIFT-based Rolling
#####################################################
def rolling_shifted_backtest(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    user_start: pd.Timestamp,
    lookback_m: int,
    rebal_m: int,
    daily_rf: float,
    class_sum_constraints: dict,
    subtype_constraints: dict,
    keep_current: bool,
    buffer_pct: float,
    trade_buffer_pct: float,
    tx_cost_value: float,
    tx_cost_type: str,
    do_ewm: bool,
    ewm_alpha: float,
    do_shrink_means: bool,
    alpha_mean: float,
    do_shrink_cov: bool,
    beta_cov: float
) -> (pd.Series, pd.DataFrame):
    """
    Rolling SHIFT-based backtest with rebalancing. 
    Uses param_max_sharpe_aclass_subtype(...) each rebalance.
    """
    dfp = df_prices.sort_index().ffill().bfill()
    all_dt = dfp.index
    if len(all_dt) < 21:
        return pd.Series([], dtype=float), pd.DataFrame([])
    lb_days = lookback_m * 21

    # SHIFT
    if user_start < all_dt[0]:
        user_start = all_dt[0]
    if user_start > all_dt[-1]:
        return pd.Series([], dtype=float), pd.DataFrame([])
    idx_0 = all_dt.get_indexer([user_start], method='bfill')[0]
    if idx_0 < lb_days:
        idx_0 = lb_days
    if idx_0 >= len(all_dt):
        return pd.Series([], dtype=float), pd.DataFrame([])
    SHIFT_day = all_dt[idx_0]

    # monthly rebal
    def last_day_of_month(d):
        nm = d + relativedelta(months=1)
        return nm.replace(day=1) - pd.Timedelta(days=1)
    rebal_cands = []
    if SHIFT_day < all_dt[-1]:
        c = last_day_of_month(SHIFT_day)
        while c <= all_dt[-1]:
            rebal_cands.append(c)
            c = last_day_of_month(c + relativedelta(months=rebal_m))

    def shift_valid(d):
        arr = all_dt[all_dt >= d]
        return arr[0] if len(arr) > 0 else all_dt[-1]
    rebal_dates = sorted({shift_valid(x) for x in rebal_cands})

    # old class allocation if keep_current
    old_class_alloc = {}
    if keep_current:
        df_instruments["Value"] = df_instruments["#Quantity"] * df_instruments["#Last_Price"]
        tv = df_instruments["Value"].sum()
        if tv <= 0:
            tv = 1.0
        df_instruments["Weight_Old"] = df_instruments["Value"] / tv
        grp = df_instruments.groupby("#Asset")["Weight_Old"].sum()
        old_class_alloc = grp.to_dict()

    # build col->(class, stype)
    tk_map_cls = {}
    tk_map_stp = {}
    for _, r_ in df_instruments.iterrows():
        tid = str(r_["#ID"])
        acl = str(r_["#Asset"])
        stp = str(r_.get("#Security_Type", "Unknown"))
        if pd.isna(stp):
            stp = "Unknown"
        tk_map_cls[tid] = acl
        tk_map_stp[tid] = stp
    col_list = dfp.columns.tolist()
    asset_cls_list = [tk_map_cls.get(tk, "Unknown") for tk in col_list]
    sec_type_list  = [tk_map_stp.get(tk, "Unknown") for tk in col_list]
    nA = len(col_list)

    SHIFT_px = dfp.iloc[idx_0].values
    valid = SHIFT_px > 0
    eq_ct = valid.sum()
    eq_w = 1.0 / eq_ct if eq_ct > 0 else 0.0
    shares = np.zeros(nA)
    for i in range(nA):
        if valid[i]:
            shares[i] = (1.0 * eq_w) / SHIFT_px[i]

    daily_vals = []
    rebal_log = []
    df_ret = dfp.pct_change().fillna(0.0)

    for d in range(idx_0, len(all_dt)):
        day = all_dt[d]
        px_ = dfp.iloc[d].values
        curr_val = np.sum(shares * px_)
        daily_vals.append(curr_val)

        if day in rebal_dates and d >= lb_days:
            sub_ret = df_ret.iloc[d - lb_days:d]
            old_w = (shares * px_) / curr_val if curr_val > 1e-12 else np.zeros(nA)
            # solver
            w_raw = param_max_sharpe_aclass_subtype(
                df_returns=sub_ret,
                tickers=col_list,
                asset_classes=asset_cls_list,
                security_types=sec_type_list,
                daily_rf=daily_rf,
                frequency=252,
                class_constraints=class_sum_constraints,
                subtype_constraints=subtype_constraints,
                keep_current=keep_current,
                old_class_alloc=old_class_alloc,
                buffer_pct=buffer_pct,
                do_ewm=do_ewm,
                ewm_alpha=ewm_alpha,
                do_shrink_means=do_shrink_means,
                alpha_mean=alpha_mean,
                do_shrink_cov=do_shrink_cov,
                beta_cov=beta_cov
            )
            # trade buffer
            if trade_buffer_pct > 1e-12:
                w_final = apply_trade_buffer(old_w, w_raw, trade_buffer_pct)
            else:
                w_final = w_raw

            # tx cost
            cost = compute_tx_cost(curr_val, old_w, w_final, tx_cost_value, tx_cost_type)
            new_val = curr_val - cost
            if new_val < 0:
                new_val = 0.0
            new_shares = np.zeros(nA)
            for i in range(nA):
                if w_final[i] > 1e-12 and px_[i] > 1e-12:
                    new_shares[i] = (new_val * w_final[i]) / px_[i]
            shares = new_shares

            rebal_log.append({
                "Date": day,
                "OldW": old_w.copy(),
                "NewW": w_final.copy(),
                "TxCost": cost,
                "PortValBefore": curr_val,
                "PortValAfter": new_val
            })

    sr_line = pd.Series(daily_vals, index=all_dt[idx_0:], name="New_Ptf")
    if len(sr_line) > 1 and sr_line.iloc[0] > 1e-12:
        sr_line = sr_line / sr_line.iloc[0]
    return sr_line, pd.DataFrame(rebal_log)

#####################################################
# 7) run_one_combo (Expanded for Bayesian)
#####################################################
def run_one_combo(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    asset_cls_list: list[str],
    sec_type_list: list[str],
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float,
    combo: tuple,
    transaction_cost_value: float,
    transaction_cost_type: str,
    trade_buffer_pct: float,
    do_shrink_means: bool = True,
    do_shrink_cov: bool = True,
    do_ewm: bool = False,
    ewm_alpha: float = 0.06
) -> dict:
    """
    combo => typically (alpha_mean, beta_cov, rebal_freq, lookback_m)
    We'll map them to the actual arguments that rolling_shifted_backtest requires.
    """
    alpha_mean, beta_cov, rebal_freq, lookback_m = combo

    sr_line, _ = rolling_shifted_backtest(
        df_prices=df_prices,
        df_instruments=df_instruments,
        user_start=df_prices.index[0],  # or user-chosen start
        lookback_m=lookback_m,
        rebal_m=rebal_freq,
        daily_rf=daily_rf,
        class_sum_constraints=class_sum_constraints,
        subtype_constraints=subtype_constraints,
        keep_current=False,
        buffer_pct=0.0,
        trade_buffer_pct=trade_buffer_pct,
        tx_cost_value=transaction_cost_value,
        tx_cost_type=transaction_cost_type,
        do_ewm=do_ewm,
        ewm_alpha=ewm_alpha,
        do_shrink_means=do_shrink_means,
        alpha_mean=alpha_mean,
        do_shrink_cov=do_shrink_cov,
        beta_cov=beta_cov
    )
    if len(sr_line) < 2:
        return {"Sharpe Ratio": 0.0, "Annual Ret": 0.0, "Annual Vol": 0.0}
    metrics_ = compute_extended_metrics(sr_line, daily_rf)
    return {
        "Sharpe Ratio": metrics_["Sharpe"],
        "Annual Ret":   metrics_["Annual Return"],
        "Annual Vol":   metrics_["Annual Vol"]
    }

#####################################################
# 8) Bayesian Optimization
#####################################################
from skopt import gp_minimize
from skopt.space import Real, Categorical
import time

def rolling_bayesian_optimization(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float,
    transaction_cost_value: float,
    transaction_cost_type: str,
    trade_buffer_pct: float
) -> pd.DataFrame:
    st.subheader("Bayesian Optimization Example (Integrated)")

    n_calls = st.number_input("Number of Bayesian evaluations (n_calls)", 5, 500, 15, step=5)

    # Let user choose possible rebal frequencies
    freq_choices = st.multiselect("Possible Rebal Frequencies (months)", [1, 3, 6], default=[1, 3, 6])
    if not freq_choices:
        freq_choices = [1]
    # Let user choose possible lookback windows
    lb_choices = st.multiselect("Possible Lookback Windows (months)", [3, 6, 12], default=[3, 6, 12])
    if not lb_choices:
        lb_choices = [3]

    alpha_min = st.slider("alpha_mean min",0.0,1.0,0.0,0.05)
    alpha_max = st.slider("alpha_mean max",0.0,1.0,1.0,0.05)
    beta_min  = st.slider("beta_cov min",0.0,1.0,0.0,0.05)
    beta_max  = st.slider("beta_cov max",0.0,1.0,1.0,0.05)

    st.write("### EWM Covariance Ranges")
    ewm_bool_choices = st.multiselect("do_ewm possible values", [False, True], default=[False, True])
    if not ewm_bool_choices:
        ewm_bool_choices = [False]
    ewm_alpha_min = st.slider("EWM alpha min",0.0,1.0,0.0,0.05)
    ewm_alpha_max = st.slider("EWM alpha max",0.0,1.0,1.0,0.05)

    # We'll need asset_cls_list, sec_type_list
    tk_map_cls = {}
    tk_map_stp = {}
    for _, r_ in df_instruments.iterrows():
        tid = str(r_["#ID"])
        acl = str(r_["#Asset"])
        stp = str(r_.get("#Security_Type", "Unknown"))
        if pd.isna(stp):
            stp = "Unknown"
        tk_map_cls[tid] = acl
        tk_map_stp[tid] = stp
    col_list = df_prices.columns.tolist()
    asset_cls_list = [tk_map_cls.get(tk, "Unknown") for tk in col_list]
    sec_type_list  = [tk_map_stp.get(tk, "Unknown") for tk in col_list]

    tries_list = []
    space = [
        Real(alpha_min, alpha_max, name="alpha_"),
        Real(beta_min,  beta_max,  name="beta_"),
        Categorical(freq_choices, name="freq_"),
        Categorical(lb_choices,   name="lb_"),
        Categorical(ewm_bool_choices, name="do_ewm_"),
        Real(ewm_alpha_min, ewm_alpha_max, name="ewm_alpha_")
    ]

    progress_bar = st.progress(0)
    progress_text = st.empty()
    start_time = time.time()

    def on_step(res):
        done = len(res.x_iters)
        pct = int(done * 100 / n_calls)
        elapsed = time.time() - start_time
        progress_text.text(f"Progress: {pct}% complete. Elapsed: {elapsed:.1f}s")
        progress_bar.progress(pct)

    def objective(x):
        alpha_ = x[0]
        beta_  = x[1]
        freq_  = x[2]
        lb_    = x[3]
        do_ewm_    = x[4]
        ewm_alpha_ = x[5]

        # Quick fix if do_ewm is True
        if do_ewm_:
            ewm_alpha_ = max(1e-12, min(ewm_alpha_, 1.0))

        combo = (alpha_, beta_, freq_, lb_)
        result = run_one_combo(
            df_prices=df_prices,
            df_instruments=df_instruments,
            asset_cls_list=asset_cls_list,
            sec_type_list=sec_type_list,
            class_sum_constraints=class_sum_constraints,
            subtype_constraints=subtype_constraints,
            daily_rf=daily_rf,
            combo=combo,
            transaction_cost_value=transaction_cost_value,
            transaction_cost_type=transaction_cost_type,
            trade_buffer_pct=trade_buffer_pct,
            do_shrink_means=True,
            do_shrink_cov=True,
            do_ewm=do_ewm_,
            ewm_alpha=ewm_alpha_
        )
        tries_list.append({
            "alpha_mean": alpha_,
            "beta_cov":   beta_,
            "rebal_freq": freq_,
            "lookback_m": lb_,
            "do_ewm":     do_ewm_,
            "ewm_alpha":  ewm_alpha_,
            "Sharpe Ratio": result["Sharpe Ratio"],
            "Annual Ret":   result["Annual Ret"],
            "Annual Vol":   result["Annual Vol"]
        })
        return -result["Sharpe Ratio"]

    if st.button("Run Bayesian Optimization"):
        with st.spinner("Running Bayesian optimization..."):
            res = gp_minimize(
                objective,
                space,
                n_calls=n_calls,
                random_state=42,
                callback=[on_step]
            )
        df_out = pd.DataFrame(tries_list)
        st.dataframe(df_out)
        if not df_out.empty:
            best_ = df_out.sort_values("Sharpe Ratio", ascending=False).iloc[0]
            st.write("**Best Found** =>", dict(best_))
        return df_out
    else:
        return pd.DataFrame()

#####################################################
# Main Application
#####################################################
def main():
    st.title("Shift-based Rolling + Riskfolio + Bayesian Integration (with NaN checks)")

    # 1) Upload and parse
    xfile = st.file_uploader("Upload Excel", type=["xlsx"])
    if not xfile:
        st.stop()

    try:
        df_instruments, df_prices = parse_excel(xfile)
    except Exception as e:
        st.error(f"Failed parse => {e}")
        st.stop()

    coverage = st.slider("Coverage fraction =>",0.0,1.0,0.8,0.05)
    df_clean = clean_df_prices(df_prices, coverage)
    st.write(f"Data => shape={df_clean.shape}, from {df_clean.index.min()} to {df_clean.index.max()}")

    # Build "Old" portfolio line
    df_instruments["Value"] = df_instruments["#Quantity"]* df_instruments["#Last_Price"]
    totv = df_instruments["Value"].sum()
    if totv <= 0: 
        totv = 1.0
    df_instruments["Weight_Old"] = df_instruments["Value"]/ totv
    old_line = build_old_portfolio_line(df_instruments, df_clean)

    # 2) Constraints
    c_mode = st.selectbox("Constraints =>",["keep_current","custom"], index=0)
    buffer_pct = 0.0
    class_sum_constraints={}
    subtype_constraints={}

    if c_mode=="keep_current":
        bf= st.number_input("Buffer % =>",0.0,100.0,5.0,1.0)
        buffer_pct = bf/100.0
    else:
        st.write("Min/Max per class -> in %")
        for ccl in df_instruments["#Asset"].dropna().unique():
            c1,c2= st.columns(2)
            with c1:
                mnv= st.number_input(f"Min% for {ccl}",0.0,100.0,0.0,1.0)/100.0
            with c2:
                mxv= st.number_input(f"Max% for {ccl}",0.0,100.0,100.0,1.0)/100.0
            class_sum_constraints[ccl] = {
                "min_class_weight": mnv,
                "max_class_weight": mxv
            }

    st.write("Subtype constraints => e.g. (Equity, 'ETF')")
    df_instruments.fillna({"#Asset":"Unknown","#Security_Type":"Unknown"}, inplace=True)
    grp_ = df_instruments.groupby(["#Asset","#Security_Type"]).size().index.tolist()
    for (acl, stp) in grp_:
        cA,cB= st.columns(2)
        with cA:
            mi_= st.number_input(f"Min {acl}-{stp}", 0.0,100.0,0.0,1.0)/100.0
        with cB:
            mx_= st.number_input(f"Max {acl}-{stp}", 0.0,100.0,10.0,1.0)/100.0
        subtype_constraints[(acl, stp)] = {
            "min_instrument": mi_,
            "max_instrument": mx_
        }

    # 3) Transaction cost, daily RF
    drf= st.number_input("DailyRf(%) =>",0.0,10.0,0.0,0.1)/100.0
    tr_buf = st.number_input("Trade Buffer(%) =>",0.0,20.0,1.0,1.0)/100.0
    tx_type= st.selectbox("TxCost =>",["percentage","ticket_fee"], index=0)
    if tx_type=="percentage":
        c_val= st.number_input("Cost(%) =>",0.0,10.0,0.1,0.1)/100.0
    else:
        c_val= st.number_input("Ticket =>",0.0,1e9,10.0,10.0)

    # 4) SHIFT-based Rolling vs Bayesian
    approach= st.radio("Approach =>",["Manual Rolling","Bayesian"], index=0)

    if approach=="Manual Rolling":
        user_sd= st.date_input("SHIFT Start =>",value=df_clean.index.min().date())
        user_start= pd.Timestamp(user_sd)
        lookb_= st.selectbox("Lookback(m)",[3,6,12], index=0)
        rebal_= st.selectbox("Rebalance(m)",[1,3,6], index=0)
        do_ewm= st.checkbox("Use EWMA?",False)
        ewm_a= st.slider("EWMA alpha =>",0.0,1.0,0.06,0.01)
        do_smu= st.checkbox("Shrink Means?",False)
        alpha_m= st.slider("Alpha mean =>",0.0,1.0,0.3,0.01)
        do_sco= st.checkbox("Shrink Cov?",False)
        beta_c= st.slider("Beta cov =>",0.0,1.0,0.2,0.01)

        if st.button("Run Rolling"):
            sr_line, df_reb= rolling_shifted_backtest(
                df_prices=df_clean,
                df_instruments=df_instruments,
                user_start=user_start,
                lookback_m=lookb_,
                rebal_m=rebal_,
                daily_rf=drf,
                class_sum_constraints=class_sum_constraints,
                subtype_constraints=subtype_constraints,
                keep_current=(c_mode=="keep_current"),
                buffer_pct= buffer_pct,
                trade_buffer_pct= tr_buf,
                tx_cost_value= c_val,
                tx_cost_type= tx_type,
                do_ewm= do_ewm,
                ewm_alpha= ewm_a,
                do_shrink_means= do_smu,
                alpha_mean= alpha_m,
                do_shrink_cov= do_sco,
                beta_cov= beta_c
            )
            if len(sr_line)<2:
                st.warning("No backtest results.")
                return

            SHIFTd= sr_line.index[0]
            old_slice= old_line.loc[old_line.index >= SHIFTd].copy()
            if len(old_slice)<2:
                st.warning("No overlap SHIFT vs Old")
                return
            if old_slice.iloc[0]>1e-12:
                old_slice= old_slice/ old_slice.iloc[0]
            idx_u= old_slice.index.union(sr_line.index)
            old_u= old_slice.reindex(idx_u, method='ffill')
            new_u= sr_line.reindex(idx_u, method='ffill')
            df_cmp= pd.DataFrame({"Old": old_u,"New": new_u}, index= idx_u)
            lo_= df_cmp.min().min()
            hi_= df_cmp.max().max()
            lbound= lo_*0.99 if lo_>0 else lo_*1.01
            fig= px.line(df_cmp, x=df_cmp.index, y=df_cmp.columns)
            fig.update_yaxes(range=[lbound, hi_*1.01])
            st.plotly_chart(fig, use_container_width=True)

            ext_old= compute_extended_metrics(df_cmp["Old"], drf)
            ext_new= compute_extended_metrics(df_cmp["New"], drf)
            df_met= format_ext_table(ext_old, ext_new)
            st.write("### Extended Metrics")
            st.dataframe(df_met)
            st.write("### Rebalance Log")
            st.dataframe(df_reb)

    else:
        # Run the integrated Bayesian approach
        df_bayes = rolling_bayesian_optimization(
            df_prices=df_clean,
            df_instruments=df_instruments,
            class_sum_constraints=class_sum_constraints,
            subtype_constraints=subtype_constraints,
            daily_rf=drf,
            transaction_cost_value=c_val,
            transaction_cost_type=tx_type,
            trade_buffer_pct=tr_buf
        )
        if df_bayes.empty:
            st.write("No result from Bayesian.")
        else:
            st.write("Bayesian results =>", df_bayes)

if __name__=="__main__":
    main()
