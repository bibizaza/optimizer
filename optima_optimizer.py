import streamlit as st
import pandas as pd
import numpy as np
import cvxpy as cp
import io
import time
import traceback
from dateutil.relativedelta import relativedelta

# scikit-optimize for Bayesian
from skopt import gp_minimize
from skopt.space import Integer, Real, Categorical
import concurrent.futures

##############################################################################
# 1) Parsing, Cleaning, Old Portfolio
##############################################################################
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
    df_prices = df_prices.copy()
    coverage = df_prices.notna().sum(axis=1)
    n_cols   = df_prices.shape[1]
    thr      = n_cols * min_coverage
    df_prices= df_prices[coverage >= thr]
    df_prices= df_prices.sort_index().fillna(method="ffill").fillna(method="bfill")
    return df_prices

def build_old_portfolio_line(df_instruments: pd.DataFrame, df_prices: pd.DataFrame) -> pd.Series:
    df_prices = df_prices.sort_index().fillna(method="ffill").fillna(method="bfill")
    ticker_qty= {}
    for _, row in df_instruments.iterrows():
        tk = row["#ID"]
        qty= row["#Quantity"]
        ticker_qty[tk] = qty
    col_list= df_prices.columns
    old_shares= np.array([ticker_qty.get(c,0.0) for c in col_list])
    vals= [np.sum(old_shares* r.values) for _, r in df_prices.iterrows()]
    sr= pd.Series(vals, index=df_prices.index)
    if len(sr)>0 and sr.iloc[0]<=0:
        sr.iloc[0]=1.0
    if len(sr)>0:
        sr= sr/sr.iloc[0]
    sr.name="Old_Ptf"
    return sr

##############################################################################
# 2) UI for constraints => custom vs keep_current, plus sub-type constraints
##############################################################################
def get_main_constraints(df_instruments: pd.DataFrame, df_prices: pd.DataFrame) -> dict:
    earliest = df_prices.index.min()
    latest   = df_prices.index.max()
    earliest_valid = earliest + pd.Timedelta(days=365)

    user_start_date= st.date_input(
        "Start Date",
        min_value=earliest.date(),
        value=min(earliest_valid, latest).date(),
        max_value=latest.date()
    )
    user_start= pd.Timestamp(user_start_date)
    if user_start< earliest_valid:
        st.error("Start date must be >= 1 year from earliest.")
        st.stop()

    constraint_mode= st.selectbox("Constraint Mode", ["custom","keep_current"], index=0)
    st.write(f"Selected: {constraint_mode}")

    have_sec_type= ("#Security_Type" in df_instruments.columns)
    all_classes  = df_instruments["#Asset"].unique()

    class_sum_constraints= {}
    subtype_constraints  = {}
    buffer_pct= 0.0

    if constraint_mode=="custom":
        st.info("**Custom** => min/max for each asset class, plus sub-type constraints for each security_type.")
        for cl in all_classes:
            with st.expander(f"Asset Class: {cl}", expanded=False):
                c1,c2= st.columns(2)
                with c1:
                    mn_cls= st.number_input(f"{cl} MIN class sum(%)",0.0,100.0,0.0, step=5.0)
                with c2:
                    mx_cls= st.number_input(f"{cl} MAX class sum(%)",0.0,100.0,100.0, step=5.0)
                class_sum_constraints[cl]= {
                    "min_class_weight": mn_cls/100.0,
                    "max_class_weight": mx_cls/100.0
                }

                df_cl= df_instruments[df_instruments["#Asset"]== cl]
                if not have_sec_type:
                    st.caption("(No #Security_Type => skipping sub-type constraints for this class.)")
                else:
                    stypes= df_cl["#Security_Type"].dropna().unique()
                    if len(stypes)==0:
                        st.caption("(No sub-types => skipping.)")
                    else:
                        st.write("**Define min_instrument, max_instrument for each sub-type**")
                        for stp in stypes:
                            cA,cB= st.columns(2)
                            with cA:
                                min_sub= st.number_input(f"[{cl}-{stp}] min_instrument(%)",0.0,100.0,0.0, step=1.0, key=f"{cl}_{stp}_min")
                            with cB:
                                max_sub= st.number_input(f"[{cl}-{stp}] max_instrument(%)",0.0,100.0,100.0, step=5.0, key=f"{cl}_{stp}_max")
                            subtype_constraints[(cl, stp)] = {
                                "min_instrument": min_sub/100.0,
                                "max_instrument": max_sub/100.0
                            }

    else:
        st.info("**Keep Current** => class sum = old ± buffer, but also sub-type constraints at instrument level.")
        buff_in= st.number_input("Buffer(%) around old class weight",0.0,100.0,5.0)
        buffer_pct= buff_in/100.0
        for cl in all_classes:
            # we'll override min/max later
            class_sum_constraints[cl] = {"min_class_weight":0.0, "max_class_weight":1.0}
            with st.expander(f"[{cl}] Security-Type constraints", expanded=False):
                df_cl= df_instruments[df_instruments["#Asset"]== cl]
                if not have_sec_type:
                    st.caption("(No #Security_Type => skipping sub-type.)")
                else:
                    stypes= df_cl["#Security_Type"].dropna().unique()
                    if len(stypes)==0:
                        st.caption("(No sub-types => skipping.)")
                    else:
                        for stp in stypes:
                            cA,cB= st.columns(2)
                            with cA:
                                min_sub= st.number_input(f"[{cl}-{stp}] min_instrument(%)",0.0,100.0,0.0, step=1.0, key=f"{cl}_{stp}_min")
                            with cB:
                                max_sub= st.number_input(f"[{cl}_{stp}] max_instrument(%)",0.0,100.0,100.0, step=5.0, key=f"{cl}_{stp}_max")
                            subtype_constraints[(cl, stp)] = {
                                "min_instrument": min_sub/100.0,
                                "max_instrument": max_sub/100.0
                            }

    daily_rf= st.number_input("Daily RF(%)",0.0,100.0,0.0)/100.0

    cost_type= st.selectbox("Transaction Cost Type",["percentage","ticket_fee"], index=0)
    if cost_type=="percentage":
        cost_val= st.number_input("Cost(%)",0.0,100.0,1.0, step=0.5)/100.0
        transaction_cost_value= cost_val
    else:
        transaction_cost_value= st.number_input("Ticket Fee",0.0,1e9,10.0, step=10.0)

    tb_in= st.number_input("Trade Buffer(%)",0.0,50.0,1.0)
    trade_buffer_pct= tb_in/100.0

    return {
        "user_start": pd.Timestamp(user_start),
        "constraint_mode": constraint_mode,
        "buffer_pct": buffer_pct,
        "class_sum_constraints": class_sum_constraints,
        "subtype_constraints": subtype_constraints,
        "daily_rf": daily_rf,
        "cost_type": cost_type,
        "transaction_cost_value": transaction_cost_value,
        "trade_buffer_pct": trade_buffer_pct
    }

##############################################################################
# 3) parametric_max_sharpe_aclass_subtype => the solver
##############################################################################
def parametric_max_sharpe_aclass_subtype(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    security_types: list[str],
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float=0.0,
    no_short: bool=True,
    n_points: int=15,
    regularize_cov: bool=False,
    shrink_means: bool=False,
    alpha: float=0.3,
    shrink_cov: bool=False,
    beta: float=0.2,
    use_ledoitwolf: bool=False,
    do_ewm: bool=False,
    ewm_alpha: float=0.06
):
    import cvxpy as cp

    # build covariance (simplified)
    cov_raw= df_returns.cov().values
    SHIFT=1e-8
    cov_raw += SHIFT*np.eye(len(cov_raw))
    cov_expr= cp.psd_wrap(cov_raw)

    mean_ret= df_returns.mean().values
    ann_rf  = daily_rf*252
    best_sharpe= -np.inf
    best_w= np.zeros(len(tickers))

    ann_means= mean_ret* 252
    targ_min = max(0.0, ann_means.min())
    targ_max = ann_means.max()
    candidate_targets= np.linspace(targ_min, targ_max, n_points)

    for targ in candidate_targets:
        w= cp.Variable(len(tickers))
        objective= cp.Minimize(cp.quad_form(w, cov_expr))
        constraints= [cp.sum(w)==1]
        if no_short:
            constraints.append(w>=0)

        # target return => (mean_ret@ w)*252 >= targ
        constraints.append((mean_ret@ w)*252 >= targ)

        # class sum
        for cl_ in class_sum_constraints:
            cdict= class_sum_constraints[cl_]
            mn= cdict.get("min_class_weight",0.0)
            mx= cdict.get("max_class_weight",1.0)
            idxs= [i for i,a_ in enumerate(asset_classes) if a_== cl_]
            constraints.append(cp.sum(w[idxs])>= mn)
            constraints.append(cp.sum(w[idxs])<= mx)

        # sub-type => each instrument => w[i] in [min_instrument, max_instrument]
        for i in range(len(tickers)):
            cl = asset_classes[i]
            stp= security_types[i]
            stvals= subtype_constraints.get((cl, stp), {})
            min_i= stvals.get("min_instrument",0.0)
            max_i= stvals.get("max_instrument",1.0)
            constraints.append(w[i]>= min_i)
            constraints.append(w[i]<= max_i)

        prob= cp.Problem(objective, constraints)
        solved=False
        for solver_ in [cp.SCS, cp.ECOS]:
            try:
                prob.solve(solver=solver_, verbose=False)
                if prob.status in ["optimal","optimal_inaccurate"] and w.value is not None:
                    solved=True
                    break
            except:
                pass
        if not solved:
            continue

        w_val= w.value
        vol_ann= np.sqrt(w_val.T@ cov_raw@ w_val)* np.sqrt(252)
        ret_ann= (mean_ret@ w_val)* 252
        if vol_ann>1e-12:
            sr= (ret_ann- ann_rf)/ vol_ann
        else:
            sr= -np.inf
        if sr> best_sharpe:
            best_sharpe= sr
            best_w= w_val.copy()

    final_ret= (mean_ret@ best_w)*252
    final_vol= np.sqrt(best_w.T@ cov_raw@ best_w)* np.sqrt(252)
    summary= {
        "Annual Return (%)": round(final_ret*100,2),
        "Annual Vol (%)":    round(final_vol*100,2),
        "Sharpe Ratio":      round(best_sharpe,4)
    }
    return best_w, summary

##############################################################################
# 4) Rolling monthly with debug => returns 5 items
##############################################################################
def last_day_of_month(date: pd.Timestamp)-> pd.Timestamp:
    return (date+ relativedelta(months=1)).replace(day=1)- pd.Timedelta(days=1)

def shift_to_valid_day(date: pd.Timestamp, valid_idx: pd.Index)-> pd.Timestamp:
    if date in valid_idx:
        return date
    aft= valid_idx[valid_idx>= date]
    if len(aft)>0:
        return aft[0]
    return valid_idx[-1]

def build_monthly_rebal_dates(start_date, end_date, months_interval, df_prices):
    rebal=[]
    curr= last_day_of_month(start_date)
    while curr<= end_date:
        rebal.append(curr)
        curr= last_day_of_month(curr+ relativedelta(months=months_interval))
    valid_idx= df_prices.index
    final=[]
    for d in rebal:
        d_shifted= shift_to_valid_day(d, valid_idx)
        if d_shifted<= end_date:
            final.append(d_shifted)
    return sorted(list(set(final)))

def compute_transaction_cost(curr_val, old_w, new_w, tx_cost_val, tx_cost_type):
    turnover= np.sum(np.abs(new_w- old_w))
    if tx_cost_type=="percentage":
        return curr_val* turnover* tx_cost_val
    else:
        inst_traded= (np.abs(new_w- old_w)>1e-9).sum()
        return inst_traded* tx_cost_val

def apply_trade_buffer(old_w, new_w, buffer_thr):
    w_adj= new_w.copy()
    diffs= w_adj- old_w
    for i in range(len(old_w)):
        if abs(diffs[i])< buffer_thr:
            w_adj[i]= old_w[i]
    s= np.sum(w_adj)
    if s<=1e-12:
        if np.sum(old_w)>1e-12:
            return old_w
        else:
            return np.ones_like(old_w)/ len(old_w)
    return w_adj/s

def rolling_backtest_monthly_param_sharpe(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    param_sharpe_fn,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    months_interval=1,
    window_days=252,
    transaction_cost_value=0.0,
    transaction_cost_type="percentage",
    trade_buffer_pct=0.0
)-> tuple[pd.Series, np.ndarray, np.ndarray, pd.Timestamp, pd.DataFrame]:
    """
    Return => (sr_norm, last_w_final, old_w_last_final, final_rebal_date, df_rebal_history)
    """
    df_prices= df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices)<2:
        empty_line= pd.Series([1.0], index=df_prices.index[:1], name="Rolling_Ptf")
        empty_df= pd.DataFrame(columns=["Date","OldWeights","NewWeights","TxCost","PortValBefore","PortValAfter"])
        return empty_line, np.zeros(df_prices.shape[1]), np.zeros(df_prices.shape[1]), None, empty_df

    rebal_dates= build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1], months_interval, df_prices)
    dates= df_prices.index
    n_days= len(dates)
    n_assets= df_prices.shape[1]

    # init
    rolling_val= 1.0
    shares= np.zeros(n_assets)
    p0= df_prices.iloc[0].fillna(0.0).values
    valid_mask= (p0>0)
    if valid_mask.sum()>0:
        eq_w= 1.0/ valid_mask.sum()
        for i in range(n_assets):
            if valid_mask[i]:
                shares[i]= (rolling_val* eq_w)/ p0[i]

    daily_vals= [np.sum(shares* p0)]
    df_returns= df_prices.pct_change().fillna(0.0)

    last_w_final= np.zeros(n_assets)
    final_old_w_last= np.zeros(n_assets)
    final_rebal_date= None
    rebal_events= []

    for d in range(1,n_days):
        day= dates[d]
        prices_today= df_prices.loc[day].fillna(0.0).values
        rolling_val= np.sum(shares* prices_today)
        daily_vals.append(rolling_val)

        if day in rebal_dates and d>0:
            sum_price_shares= np.sum(shares* prices_today)
            if sum_price_shares<=1e-12:
                old_w= np.zeros(n_assets)
            else:
                old_w= (shares* prices_today)/ sum_price_shares
            final_old_w_last= old_w.copy()
            final_rebal_date= day

            # trailing window
            start_idx= max(0,d- window_days)
            sub_ret= df_returns.iloc[start_idx:d]

            w_opt,_= param_sharpe_fn(sub_ret)

            if trade_buffer_pct>0:
                w_adj= apply_trade_buffer(old_w, w_opt, trade_buffer_pct)
            else:
                w_adj= w_opt

            cost= compute_transaction_cost(rolling_val, old_w, w_adj, transaction_cost_value, transaction_cost_type)
            old_val= rolling_val
            rolling_val-= cost
            if rolling_val<0:
                rolling_val=0.0

            money_alloc= rolling_val* w_adj
            shares= np.zeros(n_assets)
            for i in range(n_assets):
                if w_adj[i]>0 and prices_today[i]>0:
                    shares[i]= money_alloc[i]/ prices_today[i]
            last_w_final= w_adj.copy()

            rebal_events.append({
                "Date": day,
                "OldWeights": old_w.copy(),
                "NewWeights": w_adj.copy(),
                "TxCost": cost,
                "PortValBefore": old_val,
                "PortValAfter": rolling_val
            })

    sr= pd.Series(daily_vals, index=dates, name="Rolling_Ptf")
    if sr.iloc[0]<=0:
        sr.iloc[0]=1.0
    sr_norm= sr/sr.iloc[0]
    sr_norm.name="Rolling_Ptf"
    df_rebal= pd.DataFrame(rebal_events)
    return sr_norm, last_w_final, final_old_w_last, final_rebal_date, df_rebal

##############################################################################
# 5) run_one_combo(...) => single param => returns Sharpe, etc.
##############################################################################
def run_one_combo(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    asset_classes: list[str],
    security_types: list[str],
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float,
    combo: tuple,
    transaction_cost_value: float,
    transaction_cost_type: str,
    trade_buffer_pct: float,
    do_shrink_means: bool=True,
    do_shrink_cov: bool=True,
    reg_cov: bool=False,
    do_ledoitwolf: bool=False,
    do_ewm: bool=False,
    ewm_alpha: float=0.06
) -> dict:
    """
    combo => (n_points, alpha, beta, freq_m, lb_m)
    We run rolling => measure Sharpe => return dict
    """
    n_points, alpha_val, beta_val, freq_m, lb_m= combo

    def param_sharpe_fn(sub_ret: pd.DataFrame):
        # Calls parametric solver with sub-type constraints
        w_opt, summ= parametric_max_sharpe_aclass_subtype(
            df_returns=sub_ret,
            tickers= df_prices.columns.tolist(),
            asset_classes= asset_classes,
            security_types= security_types,
            class_sum_constraints= class_sum_constraints,
            subtype_constraints= subtype_constraints,
            daily_rf= daily_rf,
            no_short=True,
            n_points=n_points,
            regularize_cov=reg_cov,
            shrink_means=do_shrink_means,
            alpha=alpha_val,
            shrink_cov=do_shrink_cov,
            beta=beta_val,
            use_ledoitwolf=do_ledoitwolf,
            do_ewm=do_ewm,
            ewm_alpha= ewm_alpha
        )
        return w_opt, summ

    window_days= lb_m*21
    sr_line, final_w, _, _ , _ = rolling_backtest_monthly_param_sharpe(
        df_prices= df_prices,
        df_instruments= df_instruments,
        param_sharpe_fn= param_sharpe_fn,
        start_date= df_prices.index[0],
        end_date= df_prices.index[-1],
        months_interval= freq_m,
        window_days= window_days,
        transaction_cost_value= transaction_cost_value,
        transaction_cost_type= transaction_cost_type,
        trade_buffer_pct= trade_buffer_pct
    )

    if len(sr_line)>1:
        from modules.analytics.returns_cov import compute_performance_metrics
        perf= compute_performance_metrics(sr_line* sr_line.iloc[0], daily_rf=daily_rf)
        sr= perf["Sharpe Ratio"]
        ann_ret= perf["Annualized Return"]
        ann_vol= perf["Annualized Volatility"]
    else:
        sr= 0.0
        ann_ret= 0.0
        ann_vol= 0.0

    return {
        "n_points": n_points,
        "alpha": alpha_val,
        "beta": beta_val,
        "rebal_freq": freq_m,
        "lookback_m": lb_m,
        "Sharpe Ratio": sr,
        "Annual Ret": ann_ret,
        "Annual Vol": ann_vol
    }

##############################################################################
# 6) rolling_grid_search(...) => parallel grid
##############################################################################
def rolling_grid_search(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    asset_classes: list[str],
    security_types: list[str],
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float,
    frontier_points_list: list[float],
    alpha_list: list[float],
    beta_list: list[float],
    rebal_freq_list: list[int],
    lookback_list: list[int],
    transaction_cost_value: float,
    transaction_cost_type: str,
    trade_buffer_pct: float,
    do_shrink_means: bool=True,
    do_shrink_cov: bool=True,
    reg_cov: bool=False,
    do_ledoitwolf: bool=False,
    do_ewm: bool=False,
    ewm_alpha: float=0.06,
    max_workers: int=4
) -> pd.DataFrame:

    combos=[]
    for npts in frontier_points_list:
        for a_ in alpha_list:
            for b_ in beta_list:
                for freq_ in rebal_freq_list:
                    for lb_ in lookback_list:
                        combos.append((npts,a_,b_, freq_, lb_))
    total= len(combos)
    st.write(f"Total combos: {total}")
    results=[]
    progress_bar= st.progress(0)
    progress_txt= st.empty()
    start_time= time.time()

    def do_run_one_combo(c_):
        return run_one_combo(
            df_prices, df_instruments,
            asset_classes, security_types,
            class_sum_constraints, subtype_constraints,
            daily_rf, c_,
            transaction_cost_value, transaction_cost_type, trade_buffer_pct,
            do_shrink_means, do_shrink_cov, reg_cov, do_ledoitwolf, do_ewm, ewm_alpha
        )

    completed=0
    from concurrent.futures import ProcessPoolExecutor, as_completed
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_map= {}
        for c_ in combos:
            fut= executor.submit(do_run_one_combo, c_)
            future_map[fut]= c_

        for fut in as_completed(future_map):
            combo_= future_map[fut]
            try:
                rdict= fut.result()
            except Exception as exc:
                st.error(f"Error for combo {combo_}: {exc}")
                st.text(traceback.format_exc())
                rdict= {
                    "n_points": combo_[0],
                    "alpha":    combo_[1],
                    "beta":     combo_[2],
                    "rebal_freq": combo_[3],
                    "lookback_m": combo_[4],
                    "Sharpe Ratio": 0.0,
                    "Annual Ret": 0.0,
                    "Annual Vol": 0.0
                }
            results.append(rdict)
            completed+=1
            pct= int(completed*100/ total)
            elapsed= time.time()- start_time
            progress_bar.progress(pct)
            progress_txt.text(f"Progress: {pct}% => {completed}/{total} combos, Elapsed: {elapsed:.1f}s")

    progress_txt.text(f"Done in {time.time()-start_time:.1f}s.")
    df_out= pd.DataFrame(results)
    return df_out

##############################################################################
# 7) Bayesian => run_bayesian_inplace
##############################################################################
def run_bayesian_inplace(
    df_prices, df_instruments,
    asset_classes, security_types,
    class_sum_constraints,
    subtype_constraints,
    daily_rf,
    transaction_cost_value,
    transaction_cost_type,
    trade_buffer_pct
):
    st.write("### Bayesian with sub-type constraints")
    n_calls= st.number_input("Number of Bayesian calls", 5,200,20, step=5)

    c1,c2= st.columns(2)
    with c1:
        min_npts= st.number_input("Min n_points",1,999,5,step=5)
    with c2:
        max_npts= st.number_input("Max n_points",1,999,100,step=5)

    alpha_min= st.slider("Alpha min",0.0,1.0,0.0,0.05)
    alpha_max= st.slider("Alpha max",0.0,1.0,1.0,0.05)
    beta_min = st.slider("Beta min",0.0,1.0,0.0,0.05)
    beta_max = st.slider("Beta max",0.0,1.0,1.0,0.05)

    freq_list= st.multiselect("Possible rebal freq (months)", [1,3,6,12], [1,3])
    if not freq_list:
        freq_list=[1]
    lb_list= st.multiselect("Possible lookback (months)", [3,6,12], [3,6])
    if not lb_list:
        lb_list=[3]

    st.write("**Press 'Run Bayesian' to start**")
    tries=[]
    from skopt.space import Integer,Real,Categorical
    space= [
        Integer(int(min_npts), int(max_npts), name="n_points"),
        Real(alpha_min, alpha_max, name="alpha_"),
        Real(beta_min, beta_max,  name="beta_"),
        Categorical(freq_list, name="freq_"),
        Categorical(lb_list,   name="lb_")
    ]

    import time
    progress_bar= st.progress(0)
    progress_txt= st.empty()
    start_t= time.time()

    def on_step(res):
        done= len(res.x_iters)
        pct= int(done*100/ n_calls)
        elapsed= time.time()- start_t
        progress_txt.text(f"Progress: {pct}% ( {done}/{n_calls} ) Elapsed: {elapsed:.1f}s")
        progress_bar.progress(min(pct,100))

    def objective(x):
        (npts, alpha_val, beta_val, freq_val, lb_val)= x
        combo= (npts, alpha_val, beta_val, freq_val, lb_val)
        out= run_one_combo(
            df_prices, df_instruments,
            asset_classes, security_types,
            class_sum_constraints, subtype_constraints,
            daily_rf, combo,
            transaction_cost_value, transaction_cost_type, trade_buffer_pct
        )
        tries.append({
            "n_points":npts,
            "alpha": alpha_val,
            "beta": beta_val,
            "freq": freq_val,
            "lb": lb_val,
            "Sharpe Ratio": out["Sharpe Ratio"],
            "Annual Ret":   out["Annual Ret"],
            "Annual Vol":   out["Annual Vol"]
        })
        return -out["Sharpe Ratio"]

    if st.button("Run Bayesian"):
        from skopt import gp_minimize
        with st.spinner("Running Bayesian..."):
            res= gp_minimize(
                objective,
                space,
                n_calls=n_calls,
                random_state=42,
                callback=[on_step]
            )
        df_out= pd.DataFrame(tries)
        best_idx= df_out["Sharpe Ratio"].idxmax()
        best_row= df_out.loc[best_idx]
        st.write("**Best Found** =>", dict(best_row))
        st.dataframe(df_out)
    else:
        st.info("Click to run Bayesian approach.")


##############################################################################
# 8) Main Streamlit => merges everything
##############################################################################
def show_excel_to_parquet_converter():
    st.subheader("Excel => Parquet Converter")
    excel_file= st.file_uploader("Upload Excel (.xlsx)",type=["xlsx"])
    if not excel_file:
        st.stop()
    df_instruments, df_prices_raw= parse_excel(excel_file)
    st.write("**Raw Instruments**", df_instruments.head())
    st.write("**Raw Prices**", df_prices_raw.head())

    coverage= st.slider("Min coverage fraction",0.0,1.0,0.8,0.05)
    df_prices_clean= clean_df_prices(df_prices_raw, coverage)
    st.write("**Cleaned** =>", df_prices_clean.head())

    buffer_instruments= io.BytesIO()
    df_instruments.to_parquet(buffer_instruments,index=False)
    buffer_instruments.seek(0)

    buffer_prices= io.BytesIO()
    df_prices_clean.to_parquet(buffer_prices,index=True)
    buffer_prices.seek(0)

    st.download_button("Download instruments.parquet",
        data= buffer_instruments,
        file_name="instruments.parquet"
    )
    st.download_button("Download prices.parquet",
        data= buffer_prices,
        file_name="prices.parquet"
    )
    st.info("Use these next time for faster data loading.")


def main():
    st.title("Optima with Security-Type Constraints + Debug Rebalance + Grid/Bayesian")

    # 1) approach_data
    approach_data= st.radio("Data Source Approach",
        ["One-time Convert Excel->Parquet","Use Excel for Analysis","Use Parquet for Analysis"]
    )
    if approach_data=="One-time Convert Excel->Parquet":
        show_excel_to_parquet_converter()
        st.stop()

    if approach_data=="Use Excel for Analysis":
        excel_file= st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
        if not excel_file:
            st.stop()
        df_instruments, df_prices= parse_excel(excel_file)
    else:
        st.info("Upload instruments.parquet & prices.parquet")
        fi= st.file_uploader("instruments.parquet", type=["parquet"])
        fp= st.file_uploader("prices.parquet", type=["parquet"])
        if not fi or not fp:
            st.stop()
        df_instruments= pd.read_parquet(fi)
        df_prices= pd.read_parquet(fp)

    coverage= st.slider("Min coverage fraction",0.0,1.0,0.8,0.05)
    df_prices_clean= clean_df_prices(df_prices, coverage)
    st.write(f"**Clean data** => shape={df_prices_clean.shape}, from {df_prices_clean.index.min()} to {df_prices_clean.index.max()}")

    # 2) constraints
    main_constr= get_main_constraints(df_instruments, df_prices_clean)
    user_start= main_constr["user_start"]
    constraint_mode= main_constr["constraint_mode"]
    buffer_pct= main_constr["buffer_pct"]
    class_sum_constraints= main_constr["class_sum_constraints"]
    subtype_constraints= main_constr["subtype_constraints"]
    daily_rf= main_constr["daily_rf"]
    cost_type= main_constr["cost_type"]
    transaction_cost_value= main_constr["transaction_cost_value"]
    trade_buffer_pct= main_constr["trade_buffer_pct"]

    # Build Weight_Old
    df_instruments["Value"]= df_instruments["#Quantity"]* df_instruments["#Last_Price"]
    tot_val= df_instruments["Value"].sum()
    if tot_val<=0:
        tot_val=1.0
        df_instruments.loc[df_instruments.index[0],"Value"]=1.0
    df_instruments["Weight_Old"]= df_instruments["Value"]/ tot_val

    # If keep_current => override class sums
    if constraint_mode=="keep_current":
        class_old_w= df_instruments.groupby("#Asset")["Weight_Old"].sum()
        for cl_ in df_instruments["#Asset"].unique():
            oldw= class_old_w.get(cl_,0.0)
            mn= max(0.0, oldw- buffer_pct)
            mx= min(1.0, oldw+ buffer_pct)
            class_sum_constraints[cl_]= {
                "min_class_weight": mn,
                "max_class_weight": mx
            }

    # subset from user_start
    df_sub= df_prices_clean.loc[pd.Timestamp(user_start):]
    if len(df_sub)<2:
        st.error("Not enough data from the selected start date.")
        st.stop()

    col_tickers= df_sub.columns.tolist()
    have_sec_type= ("#Security_Type" in df_instruments.columns)
    asset_cls_list= []
    sec_type_list= []
    for tk in col_tickers:
        row_ = df_instruments[df_instruments["#ID"]== tk]
        if not row_.empty:
            asset_cls_list.append(row_["#Asset"].iloc[0])
            if have_sec_type:
                stp= row_["#Security_Type"].iloc[0]
                if pd.isna(stp):
                    stp= "Unknown"
                sec_type_list.append(stp)
            else:
                sec_type_list.append("Unknown")
        else:
            asset_cls_list.append("Unknown")
            sec_type_list.append("Unknown")

    # 3) approach => manual single rolling, grid, bayesian
    approach= st.radio("Analysis Approach", ["Manual Single Rolling","Grid Search","Bayesian Optimization"], index=0)

    if approach=="Manual Single Rolling":
        rebal_freq= st.selectbox("Rebalance Frequency (months)", [1,3,6], index=0)
        lookback_m= st.selectbox("Lookback Window (months)", [3,6,12], index=0)
        window_days= lookback_m*21

        reg_cov= st.checkbox("Regularize Cov?",False)
        do_ledoitwolf= st.checkbox("Use LedoitWolf Cov?",False)
        do_ewm= st.checkbox("Use EWM Cov?",False)
        ewm_alpha= st.slider("EWM alpha",0.0,1.0,0.06,0.01)

        st.write("**Mean & Cov Shrink**")
        do_shrink_means= st.checkbox("Shrink Means?",True)
        alpha_shrink= st.slider("Alpha(for means)",0.0,1.0,0.3,0.05)
        do_shrink_cov= st.checkbox("Shrink Cov(diagonal)?",True)
        beta_shrink= st.slider("Beta(for cov)",0.0,1.0,0.2,0.05)

        # no michaud for simplicity
        n_points_man= st.number_input("Frontier #points",5,100,15,step=5)

        if st.button("Run Rolling (Manual)"):
            def param_sharpe_fn(sub_ret: pd.DataFrame):
                w_opt, summary= parametric_max_sharpe_aclass_subtype(
                    df_returns=sub_ret,
                    tickers= col_tickers,
                    asset_classes= asset_cls_list,
                    security_types= sec_type_list,
                    class_sum_constraints= class_sum_constraints,
                    subtype_constraints= subtype_constraints,
                    daily_rf= daily_rf,
                    no_short=True,
                    n_points=n_points_man,
                    regularize_cov= reg_cov,
                    shrink_means= do_shrink_means,
                    alpha= alpha_shrink,
                    shrink_cov= do_shrink_cov,
                    beta= beta_shrink,
                    use_ledoitwolf= do_ledoitwolf,
                    do_ewm= do_ewm,
                    ewm_alpha= ewm_alpha
                )
                return w_opt, summary

            sr_line, final_w, old_w_last, final_rebal_date, df_rebal = rolling_backtest_monthly_param_sharpe(
                df_prices= df_sub,
                df_instruments= df_instruments,
                param_sharpe_fn= param_sharpe_fn,
                start_date= df_sub.index[0],
                end_date= df_sub.index[-1],
                months_interval= rebal_freq,
                window_days= window_days,
                transaction_cost_value= transaction_cost_value,
                transaction_cost_type= cost_type,
                trade_buffer_pct= trade_buffer_pct
            )
            st.write("### Rebalance Debug Table")
            st.dataframe(df_rebal)

            # performance
            from modules.analytics.returns_cov import compute_performance_metrics
            old_line= build_old_portfolio_line(df_instruments, df_sub)
            idx_all= old_line.index.union(sr_line.index)
            old_line_u= old_line.reindex(idx_all, method="ffill")
            new_line_u= sr_line.reindex(idx_all, method="ffill")
            old0= old_line_u.iloc[0]
            new0= new_line_u.iloc[0]
            df_cum= pd.DataFrame({
                "Old(%)": (old_line_u/old0 -1)*100,
                "New(%)": (new_line_u/new0 -1)*100
            }, index= idx_all)
            st.line_chart(df_cum)

            perf_old= compute_performance_metrics(old_line_u* old0, daily_rf= daily_rf)
            perf_new= compute_performance_metrics(new_line_u* new0, daily_rf= daily_rf)
            df_perf= pd.DataFrame({"Old": perf_old,"New":perf_new})
            st.write("**Performance** =>")
            st.dataframe(df_perf)

            # show final day weight diffs
            from modules.analytics.weight_display import display_instrument_weight_diff, display_class_weight_diff
            display_instrument_weight_diff(df_instruments, col_tickers, final_w)
            display_class_weight_diff(df_instruments, col_tickers, asset_cls_list, final_w)

    elif approach=="Grid Search":
        st.subheader("Grid Search (Parallel) => security-type constraints")
        frontier_points_list= st.multiselect("Frontier Points(n_points)", [5,10,15,20,30], [5,10,15])
        alpha_list= st.multiselect("Alpha values", [0,0.1,0.2,0.3,0.4,0.5], [0,0.1,0.3])
        beta_list= st.multiselect("Beta values",  [0,0.1,0.2,0.3,0.4,0.5], [0.1,0.2])
        rebal_freq_list= st.multiselect("Rebalance freq (months)", [1,3,6], [1,3])
        lookback_list= st.multiselect("Lookback (months)", [3,6,12], [3,6])
        max_workers= st.number_input("Max Workers",1,64,4, step=1)

        if st.button("Run Grid Search"):
            if (not frontier_points_list or not alpha_list or not beta_list or not rebal_freq_list or not lookback_list):
                st.error("Please select at least one value in each parameter.")
            else:
                from modules.analytics.returns_cov import compute_performance_metrics

                df_gs= rolling_grid_search(
                    df_prices= df_sub,
                    df_instruments= df_instruments,
                    asset_classes= asset_cls_list,
                    security_types= sec_type_list,
                    class_sum_constraints= class_sum_constraints,
                    subtype_constraints= subtype_constraints,
                    daily_rf= daily_rf,
                    frontier_points_list= frontier_points_list,
                    alpha_list= alpha_list,
                    beta_list= beta_list,
                    rebal_freq_list= rebal_freq_list,
                    lookback_list= lookback_list,
                    transaction_cost_value= transaction_cost_value,
                    transaction_cost_type= cost_type,
                    trade_buffer_pct= trade_buffer_pct,
                    do_shrink_means=True,
                    do_shrink_cov=True,
                    reg_cov=False,
                    do_ledoitwolf=False,
                    do_ewm=False,
                    ewm_alpha=0.06,
                    max_workers=max_workers
                )
                st.dataframe(df_gs)
                best_= df_gs.sort_values("Sharpe Ratio", ascending=False).head(5)
                st.write("**Top 5** =>")
                st.dataframe(best_)

    else:
        # Bayesian approach
        st.subheader("Bayesian => security-type constraints")
        run_bayesian_inplace(
            df_prices=df_sub,
            df_instruments=df_instruments,
            asset_classes= asset_cls_list,
            security_types= sec_type_list,
            class_sum_constraints= class_sum_constraints,
            subtype_constraints= subtype_constraints,
            daily_rf= daily_rf,
            transaction_cost_value= transaction_cost_value,
            transaction_cost_type= cost_type,
            trade_buffer_pct= trade_buffer_pct
        )

if __name__=="__main__":
    main()