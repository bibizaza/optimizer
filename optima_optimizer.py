import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from dateutil.relativedelta import relativedelta
import riskfolio as rf
from collections import defaultdict

##############################################################################
# 1) Utility: Nearest PD
##############################################################################
def nearest_pd(mat: np.ndarray, eps_init=1e-12, tries=5)-> np.ndarray:
    """
    Force a covariance matrix to be positive semi-definite by repeated attempts
    at clipping negative eigenvalues, expanding epsilon if needed.
    """
    mat_ = 0.5*(mat + mat.T)
    eps = eps_init
    for _ in range(tries):
        try:
            vals, vecs = np.linalg.eigh(mat_)
            vals_clipped = np.clip(vals, eps, None)
            return (vecs * vals_clipped) @ vecs.T
        except np.linalg.LinAlgError:
            eps *= 10
    # fallback => diagonal if repeated fails
    diag_ = np.mean(np.diag(mat_))
    return diag_ * np.eye(mat_.shape[0])

##############################################################################
# 2) Extended Metrics
##############################################################################
def compute_extended_metrics(series_abs: pd.Series, daily_rf: float=0.0)-> dict:
    """
    Returns a dict of extended performance/risk metrics:
      - "Total Return", "Annual Return", "Annual Vol", "Sharpe", "MaxDD", "TimeToRecovery",
        "VaR_1M99", "CVaR_1M99", "Skew", "Kurtosis", "Sortino", "Calmar", "Omega"
    If insufficient data, returns zeros.
    """
    if len(series_abs) < 2:
        zkeys = [
            "Total Return","Annual Return","Annual Vol","Sharpe","MaxDD","TimeToRecovery",
            "VaR_1M99","CVaR_1M99","Skew","Kurtosis","Sortino","Calmar","Omega"
        ]
        return dict.fromkeys(zkeys,0.0)

    freq = 252
    daily_ret = series_abs.pct_change().dropna()
    total_ret = series_abs.iloc[-1]/ series_abs.iloc[0] -1
    n_days = len(daily_ret)
    ann_ret = (1+ total_ret)**(freq/n_days) -1
    ann_vol = daily_ret.std()* np.sqrt(freq)
    ann_rf = daily_rf* freq
    sharpe = 0.0
    if ann_vol>1e-12:
        sharpe = (ann_ret - ann_rf)/ ann_vol

    run_max = series_abs.cummax()
    dd_ = series_abs/ run_max -1
    maxdd = dd_.min() if not dd_.empty else 0.0
    dd_idx = dd_.idxmin() if not dd_.empty else None
    if dd_idx is None:
        ttr = 0
    else:
        peakv = run_max.loc[dd_idx]
        after_dd = series_abs.loc[dd_idx:]
        rec_ = after_dd[ after_dd >= peakv ].index
        if len(rec_)>0:
            ttr = (rec_[0] - dd_idx).days
        else:
            ttr = (series_abs.index[-1] - dd_idx).days

    # 1M VaR/CVaR
    lookb= 21
    df_1m= series_abs.pct_change(lookb).dropna()
    if len(df_1m)<2:
        var_1m99= 0.0
        cvar_1m99= 0.0
    else:
        sorted_ = np.sort(df_1m.values)
        idx_ = int(0.01* len(sorted_))
        idx_ = max(idx_,0)
        var_1m99= sorted_[idx_]
        tail= sorted_[ sorted_ <= var_1m99]
        if len(tail)>0:
            cvar_1m99= tail.mean()
        else:
            cvar_1m99= var_1m99

    skew_ = daily_ret.skew()
    kurt_ = daily_ret.kurt()
    # sortino => downside
    neg_ = daily_ret[daily_ret<0]
    neg_vol= neg_.std()* np.sqrt(freq) if len(neg_)>1 else 1e-12
    sortino= 0.0
    if neg_vol>1e-12:
        sortino= (ann_ret - ann_rf)/ neg_vol

    calmar= 0.0
    if maxdd<0:
        calmar= ann_ret/ abs(maxdd)

    pos_sum= daily_ret[daily_ret>0].sum()
    neg_sum= daily_ret[daily_ret<0].sum()
    if abs(neg_sum)<1e-12:
        omega= 999.0
    else:
        omega= pos_sum/ abs(neg_sum)

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

def format_ext_table(oldm: dict, newm: dict)-> pd.DataFrame:
    keys= [
        "Total Return","Annual Return","Annual Vol","Sharpe","MaxDD","TimeToRecovery",
        "VaR_1M99","CVaR_1M99","Skew","Kurtosis","Sortino","Calmar","Omega"
    ]
    rows=[]
    for k_ in keys:
        vo= oldm.get(k_,0.0)
        vn= newm.get(k_,0.0)
        rows.append( (k_, vo, vn) )
    df_ = pd.DataFrame(rows, columns=["Metric","Old","New"])
    df_.set_index("Metric", inplace=True)
    return df_

##############################################################################
# 3) Trade Buffer + TxCost
##############################################################################
def compute_tx_cost(port_val, old_w, new_w, tx_val, tx_type):
    """
    turnover = sum(|new - old|)
    if percentage => cost= port_val* turnover* tx_val
    else => cost= (#instruments traded)* tx_val
    """
    turnover= np.sum(np.abs(new_w - old_w))
    if tx_type=="percentage":
        return port_val* turnover* tx_val
    else:
        traded= (np.abs(new_w - old_w)>1e-12).sum()
        return traded* tx_val

def apply_trade_buffer(old_w, new_w, thr):
    """
    Revert small changes if |new - old| < thr. Then re-normalize if sum>0. 
    Prints top 3 diffs for debugging.
    """
    diffs= new_w - old_w
    st.write("DEBUG trade_buffer => old vs new => top 3 diffs:", diffs[:3])
    upd= new_w.copy()
    for i in range(len(upd)):
        if abs(diffs[i])< thr:
            upd[i]= old_w[i]
    s_= np.sum(upd)
    if s_<=0:
        st.write("DEBUG => sum <=0 => fallback eq or old")
        if np.sum(old_w)>1e-12:
            return old_w
        else:
            return np.ones(len(upd))/ len(upd)
    else:
        return upd/ s_

##############################################################################
# 4) Riskfolio solver => param_max_sharpe_aclass_subtype
##############################################################################
def param_max_sharpe_aclass_subtype(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    security_types: list[str],
    daily_rf: float=0.0,
    frequency: int=252,
    class_constraints: dict = None,
    subtype_constraints: dict= None,
    keep_current: bool=False,
    old_class_alloc: dict= None,
    buffer_pct: float=0.0,
    do_ewm: bool= False,
    ewm_alpha: float=0.06,
    do_shrink_means: bool= False,
    alpha_mean: float= 0.3,
    do_shrink_cov: bool= False,
    beta_cov: float= 0.2
)-> np.ndarray:
    """
    - No short
    - Summation=1
    - Possibly keep_current => old +/- buffer
    - Possibly shrink means/cov or do ewm
    - Subtype constraints => min_instrument, max_instrument => for each ticker
    Debug prints included.
    """
    if class_constraints is None:
        class_constraints={}
    if subtype_constraints is None:
        subtype_constraints={}

    n= df_returns.shape[1]
    if n<1:
        st.write("DEBUG solver => n=0 => fallback eq")
        return np.ones(0)

    # Step A: Build raw mean/cov
    df_returns= df_returns.copy().fillna(0.0)
    if do_ewm:
        st.write("DEBUG solver => ewm alpha=", ewm_alpha)
        alpha_ = max(1e-12, min(ewm_alpha, 1.0))
        df_mu= df_returns.ewm(alpha= alpha_, adjust=False).mean()
        raw_mu= df_mu.iloc[-1].values
        df_cov= df_returns.ewm(alpha= alpha_, adjust=False).cov()
        lastd= df_returns.index[-1]
        raw_cov= df_cov.xs(lastd, level=0).values
    else:
        raw_mu= df_returns.mean().values
        raw_cov= df_returns.cov().values

    if do_shrink_means and alpha_mean>1e-12:
        st.write("DEBUG solver => shrink means => alpha=", alpha_mean)
        gm= np.mean(raw_mu)
        raw_mu= (1- alpha_mean)* raw_mu + alpha_mean* gm

    raw_cov= nearest_pd(raw_cov,1e-12,5)
    if do_shrink_cov and beta_cov>1e-12:
        st.write("DEBUG solver => shrink cov => beta=", beta_cov)
        diag_ = np.mean(np.diag(raw_cov))
        raw_cov= (1- beta_cov)* raw_cov + beta_cov* diag_* np.eye(n)

    st.write("DEBUG solver => final mean[:3]=", raw_mu[:3], " cov shape=", raw_cov.shape)

    # Step B: Create portfolio
    port= rf.Portfolio(returns= df_returns)
    coln= df_returns.columns
    port.mu= pd.Series(raw_mu, index= coln)
    port.cov= pd.DataFrame(raw_cov, index= coln, columns= coln)

    # Step C: A/b constraints
    A_ineq=[]
    b_ineq=[]
    from collections import defaultdict
    class2idx= defaultdict(list)
    for i, cl_ in enumerate(asset_classes):
        class2idx[cl_].append(i)

    def add_sum_le(idxs, limit):
        row= np.zeros(n)
        for ix in idxs: row[ix]=1.0
        A_ineq.append(row)
        b_ineq.append(limit)

    def add_sum_ge(idxs, limit):
        row= np.zeros(n)
        for ix in idxs: row[ix]= -1.0
        A_ineq.append(row)
        b_ineq.append(-limit)

    if keep_current and old_class_alloc is not None:
        st.write("DEBUG solver => keep_current => old_class_alloc +/-", buffer_pct)
        for cl, oldw in old_class_alloc.items():
            idxs= class2idx[cl]
            st.write(f"DEBUG solver => class={cl}, oldw={oldw:.3f}, idxs={idxs}")
            if len(idxs)==0: continue
            low_= max(0.0, oldw- buffer_pct)
            hi_= min(1.0, oldw+ buffer_pct)
            add_sum_le(idxs, hi_)
            add_sum_ge(idxs, low_)
    else:
        st.write("DEBUG solver => custom =>", class_constraints)
        for cl, cdict in class_constraints.items():
            idxs= class2idx[cl]
            st.write(f"DEBUG solver => class={cl}, idxs={idxs}, cdict={cdict}")
            if len(idxs)==0: continue
            mn= cdict.get("min_class_weight",0.0)
            mx= cdict.get("max_class_weight",1.0)
            add_sum_le(idxs, mx)
            add_sum_ge(idxs, mn)

    # Subtype constraints => each ticker i => (class_i, stype_i) => maybe min_instrument, max_instrument
    for i, (cl_, stp_) in enumerate(zip(asset_classes, security_types)):
        if (cl_, stp_) in subtype_constraints:
            subD= subtype_constraints[(cl_, stp_)]
            mn_sub= subD.get("min_instrument",0.0)
            mx_sub= subD.get("max_instrument",1.0)
            if mx_sub< 1.0- 1e-12:
                row= np.zeros(n)
                row[i]= 1.0
                A_ineq.append(row)
                b_ineq.append(mx_sub)
            if mn_sub>1e-12:
                row2= np.zeros(n)
                row2[i]= -1.0
                A_ineq.append(row2)
                b_ineq.append(-mn_sub)

    if A_ineq:
        A_ = np.array(A_ineq)
        b_ = np.array(b_ineq)
        st.write("DEBUG solver => A_.shape=", A_.shape, " b_.shape=", b_.shape, " before reshape")
        st.write("DEBUG solver => A_ =>\n", A_)
        st.write("DEBUG solver => b_ =>\n", b_)
        if b_.ndim==1:
            b_= b_.reshape(-1,1)
        port.ainequality= A_
        port.binequality= b_

    # Step D: Solve
    try:
        w_sol= port.optimization(
            model='Classic',
            rm='MV',
            obj='Sharpe',
            rf= daily_rf* frequency,
            hist=True
        )
    except Exception as e:
        st.write("DEBUG solver => exception =>", e)
        st.write("DEBUG solver => fallback eq")
        return np.ones(n)/ n

    if w_sol is None:
        st.write("DEBUG solver => no solution => fallback eq")
        return np.ones(n)/ n

    w_arr= w_sol.values.flatten()
    st.write("DEBUG solver => final w[:5]=", w_arr[:5])
    return w_arr

##############################################################################
# 5) SHIFT-based Rolling + Rebal
##############################################################################
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
)-> (pd.Series, pd.DataFrame):
    dfp= df_prices.sort_index().ffill().bfill()
    all_dt= dfp.index
    if len(all_dt)<21:
        return pd.Series([], dtype=float), pd.DataFrame([])

    # SHIFT
    lb_days= lookback_m* 21
    # find SHIFT loc
    if user_start< all_dt[0]:
        user_start= all_dt[0]
    if user_start> all_dt[-1]:
        return pd.Series([],dtype=float), pd.DataFrame([])

    idx_0= all_dt.get_indexer([user_start], method='bfill')[0]
    if idx_0< lb_days:
        idx_0= lb_days
    if idx_0>= len(all_dt):
        return pd.Series([],dtype=float), pd.DataFrame([])

    SHIFT_day= all_dt[idx_0]

    # build monthly rebal days
    def last_day_of_month(dd):
        nm= dd+ relativedelta(months=1)
        return nm.replace(day=1)- pd.Timedelta(days=1)
    rebal_cands= []
    if SHIFT_day< all_dt[-1]:
        c_ = last_day_of_month(SHIFT_day)
        while c_<= all_dt[-1]:
            rebal_cands.append(c_)
            c_= last_day_of_month(c_+ relativedelta(months= rebal_m))

    def shift_valid(d):
        arr= all_dt[all_dt>= d]
        if len(arr)>0: return arr[0]
        return all_dt[-1]
    rebal_dates= sorted({shift_valid(x) for x in rebal_cands})

    # build old_class_alloc if keep_current
    old_class_alloc={}
    if keep_current:
        df_instruments["Value"]= df_instruments["#Quantity"]* df_instruments["#Last_Price"]
        tv= df_instruments["Value"].sum()
        if tv<=0: tv=1.0
        df_instruments["Weight_Old"]= df_instruments["Value"]/ tv
        grp_= df_instruments.groupby("#Asset")["Weight_Old"].sum()
        old_class_alloc= grp_.to_dict()

    # Ticker-> class, stype
    tk_map_cls= {}
    tk_map_stp= {}
    for _, r_ in df_instruments.iterrows():
        tid= str(r_["#ID"])
        acl= str(r_["#Asset"])
        stp= str(r_.get("#Security_Type","Unknown"))
        tk_map_cls[tid]= acl
        tk_map_stp[tid]= stp

    col_list= dfp.columns.tolist()
    asset_cls_list= [ tk_map_cls.get(tk,"Unknown") for tk in col_list ]
    sec_type_list=  [ tk_map_stp.get(tk,"Unknown") for tk in col_list ]

    nA= len(col_list)
    SHIFT_px= dfp.iloc[idx_0].values
    valid= SHIFT_px>0
    eq_ct= valid.sum()
    eq_w= 1.0/ eq_ct if eq_ct>0 else 0.0
    shares= np.zeros(nA)
    port_val0=1.0
    for i in range(nA):
        if valid[i]:
            shares[i]= (port_val0* eq_w)/ SHIFT_px[i]

    daily_vals= []
    rebal_log=[]
    df_ret= dfp.pct_change().fillna(0.0)

    for d_ in range(idx_0, len(all_dt)):
        day= all_dt[d_]
        px_= dfp.iloc[d_].values
        curr_val= np.sum(shares* px_)
        daily_vals.append(curr_val)

        if day in rebal_dates and d_>= lb_days:
            st.write(f"[DEBUG rolling] Rebalance @ {day}, sub_ret => {all_dt[d_-lb_days]} to {all_dt[d_-1]}")
            sub_ret= df_ret.iloc[d_ - lb_days : d_]
            # build old_w
            if curr_val<=1e-12:
                old_w= np.zeros(nA)
            else:
                old_w= (shares* px_)/ curr_val

            # call riskfolio solver
            w_raw= param_max_sharpe_aclass_subtype(
                df_returns=sub_ret,
                tickers= col_list,
                asset_classes= asset_cls_list,
                security_types= sec_type_list,
                daily_rf= daily_rf,
                frequency=252,
                class_constraints= class_sum_constraints,
                subtype_constraints= subtype_constraints,
                keep_current= keep_current,
                old_class_alloc= old_class_alloc,
                buffer_pct= buffer_pct,
                do_ewm= do_ewm,
                ewm_alpha= ewm_alpha,
                do_shrink_means= do_shrink_means,
                alpha_mean= alpha_mean,
                do_shrink_cov= do_shrink_cov,
                beta_cov= beta_cov
            )

            # optional trade buffer
            if trade_buffer_pct>1e-12:
                w_final= apply_trade_buffer(old_w, w_raw, trade_buffer_pct)
            else:
                w_final= w_raw

            # transaction cost
            cost= compute_tx_cost(curr_val, old_w, w_final, tx_cost_value, tx_cost_type)
            new_val= curr_val - cost
            if new_val<0:
                new_val= 0.0

            new_shares= np.zeros(nA)
            for i in range(nA):
                if w_final[i]>1e-12 and px_[i]>1e-12:
                    new_shares[i]= (new_val* w_final[i])/ px_[i]
            shares= new_shares

            rebal_log.append({
                "Date": day,
                "OldW": old_w.copy(),
                "NewW": w_final.copy(),
                "TxCost": cost,
                "PortValBefore": curr_val,
                "PortValAfter": new_val
            })

    sr_line= pd.Series(daily_vals, index= all_dt[idx_0:], name="New_Ptf")
    if len(sr_line)>1 and sr_line.iloc[0]>1e-12:
        sr_line= sr_line/ sr_line.iloc[0]

    return sr_line, pd.DataFrame(rebal_log)

##############################################################################
# main
##############################################################################
def main():
    st.title("SHIFT-based Rolling w/ Riskfolio (Aclass+Subtype), Buffers & TxCost")

    xfile= st.file_uploader("Upload Excel with 'streamlit' & 'Histo_Price'", type=["xlsx"])
    if not xfile:
        st.stop()

    # parse
    df_instruments= pd.DataFrame()
    df_prices= pd.DataFrame()
    try:
        df_i= pd.read_excel(xfile, sheet_name="streamlit", header=0)
        df_p= pd.read_excel(xfile, sheet_name="Histo_Price", header=0)
        if df_p.columns[0]!="Date":
            df_p.rename(columns={df_p.columns[0]:"Date"}, inplace=True)
        df_p["Date"]= pd.to_datetime(df_p["Date"], errors="coerce")
        df_p.dropna(subset=["Date"], inplace=True)
        df_p.set_index("Date", inplace=True)
        df_p= df_p.sort_index().apply(pd.to_numeric, errors="coerce")
        df_instruments= df_i.copy()
        df_prices= df_p.copy()
    except Exception as e:
        st.error(f"Error parse => {e}")
        st.stop()

    coverage= st.slider("Coverage fraction =>",0.0,1.0,0.8,0.05)
    # clean
    df_prices= df_prices.ffill().bfill()
    coverage_cnt= df_prices.notna().sum(axis=1)
    min_req= df_prices.shape[1]* coverage
    df_prices= df_prices[coverage_cnt>= min_req].ffill().bfill().sort_index()

    st.write(f"Cleaned => shape={df_prices.shape}, from {df_prices.index.min()} to {df_prices.index.max()}")

    # Build old line
    df_instruments["Value"]= df_instruments["#Quantity"]* df_instruments["#Last_Price"]
    totv= df_instruments["Value"].sum()
    if totv<=0:
        totv=1.0
    df_instruments["Weight_Old"]= df_instruments["Value"]/ totv
    # old line
    col_list= df_prices.columns
    qty_map= {}
    for _, rr in df_instruments.iterrows():
        tid= rr["#ID"]
        q_ = rr["#Quantity"]
        qty_map[tid]= q_
    old_sh= np.array([qty_map.get(c,0.0) for c in col_list])
    oldvals= (df_prices* old_sh).sum(axis=1)
    old_line= oldvals.copy()
    if len(old_line)>1 and old_line.iloc[0]>1e-12:
        old_line= old_line/ old_line.iloc[0]
    old_line.name= "Old_Ptf"

    # constraints input
    mode_= st.selectbox("Constraint Mode =>",["custom","keep_current"], index=1)
    buffer_= 0.0
    class_sum_constraints={}
    if mode_=="keep_current":
        bf_ = st.number_input("Buffer(%) =>",0.0,100.0,5.0,1.0)
        buffer_= bf_/100.0
    else:
        st.info("Specify min/max for classes => e.g. 0-100 => no real constraint.")
        cl_uniq= df_instruments["#Asset"].dropna().unique()
        for ccl in cl_uniq:
            c1,c2= st.columns(2)
            with c1:
                mn_= st.number_input(f"Min% for {ccl}",0.0,100.0,0.0,1.0)/100.0
            with c2:
                mx_= st.number_input(f"Max% for {ccl}",0.0,100.0,100.0,1.0)/100.0
            class_sum_constraints[ccl]= {
                "min_class_weight": mn_,
                "max_class_weight": mx_
            }

    # subtype
    st.write("### Subtype constraints => e.g. (Equity, 'ETF') => min_instrument, max_instrument")
    df_instruments.fillna({"#Asset":"Unknown","#Security_Type":"Unknown"}, inplace=True)
    st_list= df_instruments.groupby(["#Asset","#Security_Type"]).size().index.tolist()
    subtype_constraints={}
    for (acl, stp) in st_list:
        cA, cB= st.columns(2)
        with cA:
            mini= st.number_input(f"Min% for {acl}-{stp}",0.0,100.0,0.0,1.0)/100.0
        with cB:
            maxi= st.number_input(f"Max% for {acl}-{stp}",0.0,100.0,100.0,1.0)/100.0
        subtype_constraints[(acl, stp)] = {
            "min_instrument": mini,
            "max_instrument": maxi
        }

    # daily_rf
    drf= st.number_input("DailyRf(%) =>",0.0,10.0,0.0,0.1)/100.0
    # trade buffer, cost
    tr_buf= st.number_input("Trade Buffer(%) =>",0.0,20.0,1.0,0.5)/100.0
    tx_type= st.selectbox("TxCost =>",["percentage","ticket_fee"], index=0)
    if tx_type=="percentage":
        c_ = st.number_input("Cost(%) =>",0.0,10.0,0.1,0.1)/100.0
    else:
        c_ = st.number_input("TicketFee =>",0.0,1e9,10.0,10.0)

    # SHIFT param
    user_start_date= st.date_input("SHIFT Start =>", value=df_prices.index.min().date())
    user_ts= pd.Timestamp(user_start_date)
    lookb_m= st.selectbox("Lookback(m)",[3,6,12], index=0)
    rebal_f= st.selectbox("Rebalance freq(m)", [1,3,6], index=0)

    # ewm / shrink
    do_ewm= st.checkbox("Use EWM?",False)
    ewm_al= st.slider("EWM alpha =>",0.0,1.0,0.06,0.01)
    do_smu= st.checkbox("Shrink Means?",False)
    alpha_m= st.slider("Alpha (means) =>",0.0,1.0,0.3,0.05)
    do_sco= st.checkbox("Shrink Cov?",False)
    beta_c= st.slider("Beta (cov) =>",0.0,1.0,0.2,0.05)

    if st.button("Run Rolling SHIFT"):
        sr_line, df_reb= rolling_shifted_backtest(
            df_prices= df_prices,
            df_instruments= df_instruments,
            user_start= user_ts,
            lookback_m= lookb_m,
            rebal_m= rebal_f,
            daily_rf= drf,
            class_sum_constraints= class_sum_constraints,
            subtype_constraints= subtype_constraints,
            keep_current= (mode_=="keep_current"),
            buffer_pct= buffer_,
            trade_buffer_pct= tr_buf,
            tx_cost_value= c_,
            tx_cost_type= tx_type,
            do_ewm= do_ewm,
            ewm_alpha= ewm_al,
            do_shrink_means= do_smu,
            alpha_mean= alpha_m,
            do_shrink_cov= do_sco,
            beta_cov= beta_c
        )
        if sr_line.empty or len(sr_line)<2:
            st.warning("No new line => maybe SHIFT or coverage is too large.")
            return

        # unify w/ old
        SHIFTd= sr_line.index[0]
        old_slice= old_line.loc[SHIFTd:].copy()
        if len(old_slice)<1:
            st.warning("No overlap SHIFT vs old => can't compare.")
            return
        if old_slice.iloc[0]>1e-12:
            old_slice= old_slice/ old_slice.iloc[0]

        idx_all= old_slice.index.union(sr_line.index)
        old_u= old_slice.reindex(idx_all, method='ffill')
        new_u= sr_line.reindex(idx_all, method='ffill')
        df_cmp= pd.DataFrame({"Old": old_u, "New": new_u}, index= idx_all)

        # chart
        mmin= df_cmp.min().min()
        mmax= df_cmp.max().max()
        low_= mmin*0.99 if mmin>0 else mmin*1.01
        high_= mmax*1.01
        fig= px.line(df_cmp, x=df_cmp.index, y=df_cmp.columns)
        fig.update_yaxes(range=[low_, high_])
        st.plotly_chart(fig, use_container_width=True)

        # metrics
        ext_old= compute_extended_metrics(df_cmp["Old"], drf)
        ext_new= compute_extended_metrics(df_cmp["New"], drf)
        df_met= format_ext_table(ext_old, ext_new)

        def fm(k,v):
            if k in ["Total Return","Annual Return","Annual Vol","MaxDD","VaR_1M99","CVaR_1M99"]:
                return f"{v*100:.2f}%"
            elif k=="TimeToRecovery":
                return f"{v:.0f}"
            else:
                return f"{v:.3f}"

        for rr in df_met.index:
            df_met.loc[rr,"Old"]= fm(rr, df_met.loc[rr,"Old"])
            df_met.loc[rr,"New"]= fm(rr, df_met.loc[rr,"New"])

        st.write("### Extended Metrics => SHIFT at", SHIFTd)
        st.dataframe(df_met)

        st.write("### Rebalance Log")
        st.dataframe(df_reb)


if __name__=="__main__":
    main()
