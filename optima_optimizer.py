import streamlit as st
import numpy as np
import pandas as pd
import riskfolio as rf
import plotly.express as px
import time
from dateutil.relativedelta import relativedelta
from collections import defaultdict

# scikit-optimize
from skopt import gp_minimize
from skopt.space import Real, Categorical

########################################
# 1) Cov / Mean Shrink Utils
########################################
def nearest_pd(matrix: np.ndarray, eps=1e-12):
    """Force symmetric PSD by clipping negative eigenvalues."""
    mat= 0.5*(matrix + matrix.T)
    vals, vecs= np.linalg.eigh(mat)
    vals_clipped= np.clip(vals, eps, None)
    return vecs @ np.diag(vals_clipped) @ vecs.T

def shrink_cov_diagonal(cov_mat: np.ndarray, beta:float=0.2)-> np.ndarray:
    diag_m= np.mean(np.diag(cov_mat))
    n= cov_mat.shape[0]
    return (1-beta)* cov_mat + beta* diag_m* np.eye(n)

def compute_ewm_cov(df_ret: pd.DataFrame, alpha=0.06)-> np.ndarray:
    df_c= df_ret.ewm(alpha=alpha, adjust=False).cov()
    last_ = df_ret.index[-1]
    final_= df_c.xs(last_, level=0)
    return final_.values

def compute_ewm_mean(df_ret: pd.DataFrame, alpha=0.06)-> np.ndarray:
    df_m= df_ret.ewm(alpha=alpha, adjust=False).mean()
    return df_m.iloc[-1].values

def shrink_mean_to_grand_mean(raw_mu: np.ndarray, alpha=0.3)-> np.ndarray:
    gm= np.mean(raw_mu)
    return (1-alpha)* raw_mu + alpha* gm

########################################
# 2) parse_excel, clean_df_prices, build_old_portfolio_line
########################################
def parse_excel(file, streamlit_sheet="streamlit", histo_sheet="Histo_Price"):
    df_instruments= pd.read_excel(file, sheet_name= streamlit_sheet, header=0)
    df_prices_raw= pd.read_excel(file, sheet_name= histo_sheet, header=0)
    if df_prices_raw.columns[0] != "Date":
        df_prices_raw.rename(columns={ df_prices_raw.columns[0]:"Date" }, inplace=True)
    df_prices_raw["Date"]= pd.to_datetime(df_prices_raw["Date"], errors="coerce")
    df_prices_raw.dropna(subset=["Date"], inplace=True)
    df_prices_raw.set_index("Date", inplace=True)
    df_prices_raw.sort_index(inplace=True)
    df_prices_raw= df_prices_raw.apply(pd.to_numeric, errors="coerce")
    return df_instruments, df_prices_raw

def clean_df_prices(df_prices: pd.DataFrame, coverage_thr=0.8)-> pd.DataFrame:
    dfp= df_prices.copy()
    coverage= dfp.notna().sum(axis=1)
    n_cols= dfp.shape[1]
    threshold= coverage_thr* n_cols
    dfp= dfp[ coverage>=threshold ].ffill().bfill()
    return dfp

def build_old_portfolio_line(df_instruments, df_prices):
    dfp= df_prices.sort_index().ffill().bfill()
    ticker_qty={}
    for _, row_ in df_instruments.iterrows():
        tkr= row_["#ID"]
        qty= row_["#Quantity"]
        ticker_qty[tkr]= qty
    col_list= dfp.columns
    old_shares= np.array([ticker_qty.get(c,0.0) for c in col_list])
    vals= [ np.sum(old_shares* rrow.values) for _, rrow in dfp.iterrows() ]
    sr= pd.Series(vals, index= dfp.index, name="Old_Ptf")
    if len(sr)>1 and sr.iloc[0]>0:
        sr= sr/ sr.iloc[0]
    return sr

########################################
# 3) The Riskfolio param solver
########################################
def param_max_sharpe_aclass_subtype(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    security_types: list[str],
    class_constraints: dict,
    subtype_constraints: dict,
    daily_rf=0.0,
    frequency=252,
    old_class_alloc=None,
    keep_current=False,
    buffer_pct=0.0,
    # ewm
    use_ewm=False,
    ewm_alpha=0.06,
    # shrink means
    do_shrink_means=False,
    alpha_shr=0.3,
    # shrink cov
    do_shrink_cov=False,
    beta_shr=0.2
)-> np.ndarray:
    """
    Single pass => direct max Sharpe with class+subtype constraints.
    """
    n= len(tickers)
    if df_returns.shape[1]!= n:
        # fallback eq
        return np.ones(n)/n

    # 1) Build mu / cov
    if use_ewm:
        raw_mu_daily= compute_ewm_mean(df_returns, alpha= ewm_alpha)
        raw_cov= compute_ewm_cov(df_returns, alpha= ewm_alpha)
    else:
        raw_mu_daily= df_returns.mean().values
        raw_cov= df_returns.cov().values

    if do_shrink_means and alpha_shr>0:
        raw_mu_daily= shrink_mean_to_grand_mean(raw_mu_daily, alpha= alpha_shr)

    raw_cov= nearest_pd(raw_cov, 1e-12)
    if do_shrink_cov and beta_shr>0:
        raw_cov= shrink_cov_diagonal(raw_cov, beta= beta_shr)

    port= rf.Portfolio(returns= df_returns)
    # override stats
    port.mu= pd.Series(raw_mu_daily, index=df_returns.columns)
    port.cov= pd.DataFrame(raw_cov, index=df_returns.columns, columns=df_returns.columns)

    # constraints => A_ineq, b_ineq
    from collections import defaultdict
    class2idx= defaultdict(list)
    for i, c_ in enumerate(asset_classes):
        class2idx[c_].append(i)

    A_=[]
    b_=[]
    def add_sum_le(rows, limit):
        row= np.zeros(n)
        row[rows]=1.0
        A_.append(row)
        b_.append(limit)
    def add_sum_ge(rows, limit):
        row= np.zeros(n)
        row[rows]= -1.0
        A_.append(row)
        b_.append(-limit)

    if keep_current and old_class_alloc:
        for cl, oldw in old_class_alloc.items():
            idxs= class2idx[cl]
            if idxs:
                lo= max(0.0, oldw- buffer_pct)
                hi= min(1.0, oldw+ buffer_pct)
                add_sum_le(idxs, hi)
                add_sum_ge(idxs, lo)
    else:
        # custom class constraints
        for cl, cdict in class_constraints.items():
            idxs= class2idx[cl]
            if idxs:
                mn= cdict.get("min_class_weight", 0.0)
                mx= cdict.get("max_class_weight", 1.0)
                add_sum_le(idxs, mx)
                add_sum_ge(idxs, mn)

    # subtype
    subtype_map={}
    for (acl, stp), cdict in subtype_constraints.items():
        subtype_map[(acl, stp)] = (
            cdict.get("min_instrument",0.0),
            cdict.get("max_instrument",1.0)
        )
    for i, (ac_, st_) in enumerate(zip(asset_classes, security_types)):
        if (ac_, st_) in subtype_map:
            mn_i, mx_i= subtype_map[(ac_, st_)]
            if mx_i<1.0:
                r_= np.zeros(n)
                r_[i]=1.0
                A_.append(r_)
                b_.append(mx_i)
            if mn_i>0:
                r_= np.zeros(n)
                r_[i]= -1.0
                A_.append(r_)
                b_.append(-mn_i)

    if A_:
        A_arr= np.array(A_)
        b_arr= np.array(b_)
        if b_arr.ndim==1:
            b_arr= b_arr.reshape(-1,1)
        port.ainequality= A_arr
        port.binequality= b_arr

    risk_measure='MV'
    rf_annual= daily_rf* frequency
    try:
        w_sol= port.optimization(
            model='Classic',
            rm=risk_measure,
            obj='Sharpe',
            rf= rf_annual
        )
        if w_sol is None:
            return np.ones(n)/n
        return w_sol.values
    except:
        return np.ones(n)/n

########################################
# 4) SHIFTED_START rolling
########################################
def rolling_shifted_backtest(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    user_start: pd.Timestamp,
    end_date: pd.Timestamp,
    lookback_days: int,
    months_interval: int,
    daily_rf: float=0.0,
    class_sum_constraints: dict=None,
    subtype_constraints: dict=None,
    keep_current=False,
    buffer_pct=0.0,
    use_ewm=False,
    ewm_alpha=0.06,
    do_shrink_means=False,
    alpha_shr=0.3,
    do_shrink_cov=False,
    beta_shr=0.2,
    tx_cost_val=0.0,
    tx_cost_type="percentage"
)-> dict:
    """
    SHIFTED_START rolling. Return a dict with Sharpe, Ret, Vol, final_weights
    """
    if class_sum_constraints is None:
        class_sum_constraints={}
    if subtype_constraints is None:
        subtype_constraints={}

    dfp= df_prices.ffill().bfill().sort_index()
    if end_date:
        dfp= dfp.loc[:end_date]
    all_dates= dfp.index
    if len(all_dates)< lookback_days:
        return {"Sharpe Ratio":0.0,"Annual Ret":0.0,"Annual Vol":0.0,"final_weights": np.zeros(dfp.shape[1])}

    start_loc= all_dates.get_indexer([user_start], method='bfill')[0]
    SHIFT_loc= max(start_loc, lookback_days)
    if SHIFT_loc>= len(all_dates):
        return {"Sharpe Ratio":0.0,"Annual Ret":0.0,"Annual Vol":0.0,"final_weights": np.zeros(dfp.shape[1])}

    SHIFT_DAY= all_dates[SHIFT_loc]

    def last_day_of_month(d):
        nm= d+ relativedelta(months=1)
        return nm.replace(day=1)- pd.Timedelta(days=1)
    rebal_dates=[]
    if SHIFT_DAY< all_dates[-1]:
        c= last_day_of_month(SHIFT_DAY)
        while c<= all_dates[-1]:
            rebal_dates.append(c)
            c= last_day_of_month(c+ relativedelta(months= months_interval))
    def shift_to_valid(d):
        fut= all_dates[all_dates>= d]
        if len(fut)>0:
            return fut[0]
        return all_dates[-1]
    rebal_dates= sorted({ shift_to_valid(x) for x in rebal_dates})

    # Build asset_class & stype
    col_list= dfp.columns.tolist()
    cls_map={}
    stp_map={}
    for _, row_ in df_instruments.iterrows():
        t_= row_["#ID"]
        ac_= row_["#Asset"]
        s_= row_.get("#Security_Type","Unknown")
        if pd.isna(s_):
            s_="Unknown"
        cls_map[t_]= ac_
        stp_map[t_]= s_
    asset_cls_list= [ cls_map[tk] for tk in col_list ]
    sec_type_list=  [ stp_map[tk] for tk in col_list ]

    old_class_alloc={}
    if keep_current:
        df_instruments["Value"]= df_instruments["#Quantity"]* df_instruments["#Last_Price"]
        tot_val= df_instruments["Value"].sum()
        if tot_val<=0: tot_val=1.0
        df_instruments["Weight_Old"]= df_instruments["Value"]/ tot_val
        c_sums= df_instruments.groupby("#Asset")["Weight_Old"].sum()
        old_class_alloc= c_sums.to_dict()

    SHIFT_px= dfp.iloc[SHIFT_loc].values
    valid_mask= SHIFT_px>0
    eq_cnt= valid_mask.sum()
    eq_w= 1.0/ eq_cnt if eq_cnt>0 else 0.0
    nA= dfp.shape[1]
    shares= np.zeros(nA)
    roll_val= 1.0
    for i in range(nA):
        if valid_mask[i]:
            shares[i]= (roll_val* eq_w)/ SHIFT_px[i]

    daily_vals=[]
    df_ret= dfp.pct_change().fillna(0.0)
    day_idx= SHIFT_loc

    last_w= np.zeros(nA)  # track final weights after last rebal
    for idx_ in range(day_idx, len(all_dates)):
        day= all_dates[idx_]
        px_= dfp.loc[day].values
        v_= np.sum(shares* px_)
        daily_vals.append(v_)

        if day in rebal_dates and idx_>= lookback_days:
            start_= idx_ - lookback_days
            sub_ret= df_ret.iloc[start_: idx_]
            old_val= v_
            # old weights
            old_w= np.zeros(nA)
            if old_val>1e-12:
                for i in range(nA):
                    if px_[i]>1e-12:
                        old_w[i]= (shares[i]* px_[i])/ old_val

            # call solver
            w_opt= param_max_sharpe_aclass_subtype(
                df_returns= sub_ret,
                tickers= col_list,
                asset_classes= asset_cls_list,
                security_types= sec_type_list,
                class_constraints= class_sum_constraints,
                subtype_constraints= subtype_constraints,
                daily_rf= daily_rf,
                frequency=252,
                old_class_alloc= old_class_alloc,
                keep_current= keep_current,
                buffer_pct= buffer_pct,
                use_ewm= use_ewm,
                ewm_alpha= ewm_alpha,
                do_shrink_means= do_shrink_means,
                alpha_shr= alpha_shr,
                do_shrink_cov= do_shrink_cov,
                beta_shr= beta_shr
            )
            # cost
            turnover= np.sum(np.abs(w_opt - old_w))
            cost_=0.0
            if tx_cost_type=="percentage":
                cost_= old_val* turnover* tx_cost_val
            else:
                traded= (np.abs(w_opt- old_w)>1e-12).sum()
                cost_= traded* tx_cost_val

            new_val= old_val- cost_
            if new_val<0:
                new_val=0.0

            new_sh= np.zeros(nA)
            if new_val>1e-12:
                for i in range(nA):
                    if w_opt[i]>1e-12 and px_[i]>1e-12:
                        new_sh[i]= (new_val* w_opt[i])/ px_[i]
            shares= new_sh
            last_w= w_opt

    sr_series= pd.Series(daily_vals, index= all_dates[day_idx:], name="New_Ptf")
    if sr_series.iloc[0]<=0:
        sr_series.iloc[0]=1.0
    sr_norm= sr_series/ sr_series.iloc[0]

    dr= sr_norm.pct_change().dropna()
    if len(dr)<3:
        return {"Sharpe Ratio":0.0,"Annual Ret":0.0,"Annual Vol":0.0,"final_weights": last_w}
    m_= dr.mean()
    s_= dr.std()
    ann_ret= (1+m_)**252 -1
    ann_vol= s_* np.sqrt(252)
    ann_rf= daily_rf*252
    shp=0.0
    if ann_vol>1e-12:
        shp= (ann_ret- ann_rf)/ ann_vol
    return {
        "Sharpe Ratio": shp,
        "Annual Ret": ann_ret,
        "Annual Vol": ann_vol,
        "final_weights": last_w
    }

########################################
# 5) run_one_combo => used by bayesian
########################################
def run_one_combo(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float,
    combo: tuple,  # (alpha_, beta_, freq_, lb_)
    transaction_cost_value: float,
    transaction_cost_type: str,
    trade_buffer_pct: float,
    do_shrink_means=True,
    do_shrink_cov=True,
    use_ewm=False,
    ewm_alpha=0.06
)-> dict:
    alpha_= combo[0]
    beta_=  combo[1]
    freq_=  combo[2]
    lb_=    combo[3]
    window_days= lb_*21

    perf= rolling_shifted_backtest(
        df_prices= df_prices,
        df_instruments= df_instruments,
        user_start= df_prices.index[0],
        end_date= df_prices.index[-1],
        lookback_days= window_days,
        months_interval= freq_,
        daily_rf= daily_rf,
        class_sum_constraints= class_sum_constraints,
        subtype_constraints= subtype_constraints,
        keep_current= False,
        buffer_pct= 0.0,
        use_ewm= use_ewm,
        ewm_alpha= ewm_alpha,
        do_shrink_means= do_shrink_means,
        alpha_shr= alpha_,
        do_shrink_cov= do_shrink_cov,
        beta_shr= beta_,
        tx_cost_val= transaction_cost_value,
        tx_cost_type= transaction_cost_type
    )
    return {
        "Sharpe Ratio": perf["Sharpe Ratio"],
        "Annual Ret": perf["Annual Ret"],
        "Annual Vol": perf["Annual Vol"],
        "final_weights": perf["final_weights"]
    }

########################################
# 6) rolling_bayesian_optimization
########################################
def rolling_bayesian_optimization(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float,
    transaction_cost_value: float,
    transaction_cost_type: str,
    trade_buffer_pct: float
)-> pd.DataFrame:
    st.subheader("Bayesian => let's tune alpha, beta, freq, lb, EWM, etc.")

    n_calls= st.number_input("Number of Bayesian evaluations",5,500,20,step=5)
    st.write("### Parameter Ranges")
    alpha_min= st.slider("Alpha(min) for mean shrink",0.0,1.0,0.0,0.05)
    alpha_max= st.slider("Alpha(max) for mean shrink",0.0,1.0,1.0,0.05)
    beta_min=  st.slider("Beta(min) for cov shrink",0.0,1.0,0.0,0.05)
    beta_max=  st.slider("Beta(max) for cov shrink",0.0,1.0,1.0,0.05)

    freq_choices= st.multiselect("Possible rebal freq(months)", [1,3,6],[1,3,6])
    if not freq_choices: freq_choices=[1]
    lb_choices= st.multiselect("Possible lookback months", [3,6,12],[3,6,12])
    if not lb_choices: lb_choices=[3]

    st.write("### EWM Cov")
    ewm_bool= st.multiselect("Use EWM?", [False,True], default=[False,True])
    if not ewm_bool: ewm_bool=[False]
    ewm_alpha_min= st.slider("EWM alpha(min)",0.0,1.0,0.0,0.05)
    ewm_alpha_max= st.slider("EWM alpha(max)",0.0,1.0,1.0,0.05)

    from skopt import gp_minimize
    from skopt.space import Real, Categorical

    space= [
        Real(alpha_min, alpha_max, name="alpha_"),
        Real(beta_min, beta_max, name="beta_"),
        Categorical(freq_choices, name="freq_"),
        Categorical(lb_choices,   name="lb_"),
        Categorical(ewm_bool,     name="do_ewm_"),
        Real(ewm_alpha_min, ewm_alpha_max, name="ewm_alpha_")
    ]
    tries_list=[]

    progress_bar= st.progress(0)
    progress_text= st.empty()
    start_t= time.time()

    def on_step(res):
        done= len(res.x_iters)
        pct= int(done*100/ n_calls)
        elapsed= time.time()- start_t
        progress_text.text(f"Bayes => {pct}% done, elapsed={elapsed:.1f}s")
        progress_bar.progress(pct)

    def objective(x):
        alpha_= x[0]
        beta_=  x[1]
        freq_=  x[2]
        lb_=    x[3]
        do_ewm_= x[4]
        ewm_al_= x[5]
        if do_ewm_ and ewm_al_<1e-9:
            ewm_al_=1e-9

        combo= (alpha_, beta_, freq_, lb_)
        result= run_one_combo(
            df_prices= df_prices,
            df_instruments= df_instruments,
            class_sum_constraints= class_sum_constraints,
            subtype_constraints= subtype_constraints,
            daily_rf= daily_rf,
            combo= combo,
            transaction_cost_value= transaction_cost_value,
            transaction_cost_type= transaction_cost_type,
            trade_buffer_pct= trade_buffer_pct,
            do_shrink_means=True,
            do_shrink_cov=True,
            use_ewm= do_ewm_,
            ewm_alpha= ewm_al_
        )
        sr_= result["Sharpe Ratio"]
        tries_list.append({
            "alpha": alpha_,"beta": beta_,"rebal_freq":freq_,"lookback_m":lb_,
            "do_ewm": do_ewm_,"ewm_alpha": ewm_al_,
            "Sharpe Ratio": sr_,"Annual Ret": result["Annual Ret"],"Annual Vol": result["Annual Vol"]
        })
        return -sr_

    if not st.button("Run Bayesian"):
        return pd.DataFrame()

    with st.spinner("Running Bayesian..."):
        res= gp_minimize(objective, space, n_calls=n_calls, random_state=42, callback=[on_step])

    df_out= pd.DataFrame(tries_list)
    if df_out.empty:
        return df_out
    best_idx= df_out["Sharpe Ratio"].idxmax()
    best_row= df_out.loc[best_idx]
    st.write("**Best Found** =>", dict(best_row))
    st.dataframe(df_out)
    return df_out

########################################
# 7) main => two approaches: (A) Single Rolling, (B) Bayesian
########################################
def main():
    st.title("SHIFTED_START Rolling => Manual Single Rolling vs. Bayesian")

    # 1) Load data
    excel_file= st.file_uploader("Upload Excel(.xlsx)", type=["xlsx"])
    if not excel_file:
        st.stop()
    df_instruments, df_prices= parse_excel(excel_file)
    if df_prices.empty:
        st.error("No valid data in Excel.")
        st.stop()

    coverage= st.slider("Coverage fraction =>", 0.0,1.0,0.8,0.05)
    df_clean= clean_df_prices(df_prices, coverage)
    st.write("Data => shape=", df_clean.shape, "from=", df_clean.index.min(),"to=", df_clean.index.max())

    # constraints
    st.write("## Constraints => Class mode (custom or keep_current?), Subtype, etc.")
    approach= st.radio("Constraint approach =>", ["custom","keep_current"], index=0)
    buffer_pct=0.0
    class_sum_constraints={}
    subtype_constraints={}

    unique_cls= df_instruments["#Asset"].unique().tolist()
    if approach=="custom":
        st.write("Min/Max per class =>")
        for cl_ in unique_cls:
            c1, c2= st.columns(2)
            with c1:
                mn_= st.number_input(f"Min(%) for {cl_}",0.0,100.0,0.0,5.0)/100.0
            with c2:
                mx_= st.number_input(f"Max(%) for {cl_}",0.0,100.0,100.0,5.0)/100.0
            class_sum_constraints[cl_]= {"min_class_weight": mn_,"max_class_weight": mx_}
    else:
        st.write("keep_current => add buffer% around old class w => e.g. 5 => 5%")
        b_in= st.number_input("Buffer(%) =>",0.0,50.0,5.0,1.0)
        buffer_pct= b_in/100.0

    st.subheader("Subtype constraints => per (class, #Security_Type)")
    df_instruments["SecType"]= df_instruments["#Security_Type"].fillna("Unknown")
    stp_grp= df_instruments.groupby(["#Asset","SecType"]).size().index.tolist()
    for (ac, stp) in stp_grp:
        c1, c2= st.columns(2)
        with c1:
            mn_i= st.number_input(f"Min(%) for {ac}-{stp}",0.0,100.0,0.0,1.0)/100.0
        with c2:
            mx_i= st.number_input(f"Max(%) for {ac}-{stp}",0.0,100.0,100.0,1.0)/100.0
        subtype_constraints[(ac,stp)] = {"min_instrument": mn_i,"max_instrument": mx_i}

    st.write("---")
    # transaction cost
    st.write("### Transaction Cost")
    c_in= st.number_input("Tx cost(%) => e.g. 0.1 => 0.1%",0.0,100.0,0.1,0.1)
    tx_cost_val= c_in/100.0
    tx_cost_type= st.selectbox("Tx cost type =>",["percentage","ticket_fee"], index=0)

    st.write("## Choose Approach => Single Rolling or Bayesian")
    which_ap= st.radio("Approach =>",["Single Rolling","Bayesian"], index=0)

    if which_ap=="Single Rolling":
        # do SHIFTED_START with user-chosen param
        st.write("## Single Rolling SHIFTED_START => choose param")
        rebal_freq= st.selectbox("Rebalance freq(months)", [1,3,6], index=0)
        lookback_m= st.selectbox("Lookback window (months)", [3,6,12], index=0)
        w_days= lookback_m*21

        do_ewm= st.checkbox("Use EWM Cov?", False)
        ewm_a= st.slider("EWM alpha =>",0.0,1.0,0.06,0.01)
        do_shr_m= st.checkbox("Shrink Means?", True)
        alpha_sh= st.slider("Alpha for mean shrink =>",0.0,1.0,0.3,0.05)
        do_shr_c= st.checkbox("Shrink Cov => diagonal?", True)
        beta_sh= st.slider("Beta for cov =>",0.0,1.0,0.2,0.05)

        daily_rf= st.number_input("Daily RF(%) =>",0.0,10.0,0.0,0.1)/100.0

        if st.button("Run Single Rolling"):
            # do SHIFTED
            perf= rolling_shifted_backtest(
                df_prices= df_clean,
                df_instruments= df_instruments,
                user_start= df_clean.index[0],
                end_date= df_clean.index[-1],
                lookback_days= w_days,
                months_interval= rebal_freq,
                daily_rf= daily_rf,
                class_sum_constraints= class_sum_constraints,
                subtype_constraints= subtype_constraints,
                keep_current= (approach=="keep_current"),
                buffer_pct= buffer_pct,
                use_ewm= do_ewm,
                ewm_alpha= ewm_a,
                do_shrink_means= do_shr_m,
                alpha_shr= alpha_sh,
                do_shrink_cov= do_shr_c,
                beta_shr= beta_sh,
                tx_cost_val= tx_cost_val,
                tx_cost_type= tx_cost_type
            )
            sr_= perf["Sharpe Ratio"]
            ret_= perf["Annual Ret"]
            vol_= perf["Annual Vol"]
            st.write(f"**Sharpe** => {sr_:.2f},  Annual Ret={ret_*100:.2f}%, Vol={vol_*100:.2f}%")
            st.write("Final w =>", perf["final_weights"])

            # plot old vs new
            old_line= build_old_portfolio_line(df_instruments, df_clean)
            SHIFT_loc= w_days
            if SHIFT_loc>= len(df_clean.index):
                st.warning("Not enough data to SHIFT.")
                return
            shift_day= df_clean.index[SHIFT_loc]
            # build new_line from the rolling => we have daily_vals but we only stored them in 'perf'?
            # We'll adapt => for demonstration, let's just show final. Or we can store daily series if we want.

            st.info("If you want a daily plot, you'd store the daily series similarly to how you do it in the code. Omitted here for brevity.")
    else:
        st.write("## Bayesian => will tune alpha,beta, rebal freq, lookback, EWM alpha.")
        daily_rf= st.number_input("Daily RF(%) =>",0.0,10.0,0.0,0.1)/100.0
        df_bayes= rolling_bayesian_optimization(
            df_prices= df_clean,
            df_instruments= df_instruments,
            class_sum_constraints= class_sum_constraints,
            subtype_constraints= subtype_constraints,
            daily_rf= daily_rf,
            transaction_cost_value= tx_cost_val,
            transaction_cost_type= tx_cost_type,
            trade_buffer_pct= 0.0
        )
        if not df_bayes.empty:
            st.write("**Bayes done** => best combos above.")


if __name__=="__main__":
    main()
