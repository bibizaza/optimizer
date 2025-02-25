import streamlit as st
import pandas as pd
import numpy as np
import riskfolio as rf
from dateutil.relativedelta import relativedelta
import plotly.express as px
from collections import defaultdict

#######################################
# 1) Utility: nearest_pd => PSD
#######################################
def nearest_pd(cov, epsilon=1e-12):
    """
    Force a symmetric matrix to be PSD by clipping negative eigenvalues.
    """
    mat = 0.5*(cov + cov.T)
    vals, vecs = np.linalg.eigh(mat)
    vals_clipped = np.clip(vals, epsilon, None)
    return (vecs @ np.diag(vals_clipped) @ vecs.T)

#######################################
# 2) Diagonal Cov Shrink => beta
#######################################
def shrink_cov_diagonal(cov_mat: np.ndarray, beta: float=0.2) -> np.ndarray:
    """
    Simple diagonal shrink:
      cov_shrunk = (1 - beta)*cov_mat + beta*(diag_mean)*I
    """
    diag_mean = np.mean(np.diag(cov_mat))
    n = cov_mat.shape[0]
    I_ = np.eye(n)
    return (1 - beta)*cov_mat + beta*diag_mean*I_

#######################################
# 3) EWM Cov => compute_ewm_cov
#######################################
def compute_ewm_cov(df_returns: pd.DataFrame, alpha: float=0.06) -> np.ndarray:
    """
    Compute EWM covariance using df.ewm(alpha=...).cov() and pick the final NxN slice.
    """
    df_ewm = df_returns.ewm(alpha=alpha, adjust=False).cov()
    last_date = df_returns.index[-1]
    final_block = df_ewm.xs(last_date, level=0)  # NxN
    return final_block.values

#######################################
# 4) EWM Mean => optional
#######################################
def compute_ewm_mean(df_returns: pd.DataFrame, alpha: float=0.06) -> np.ndarray:
    """
    We'll do a rolling EWM for the daily returns, pick the last row as the 'most recent' means.
    """
    df_ewm = df_returns.ewm(alpha=alpha, adjust=False).mean()
    return df_ewm.iloc[-1].values  # shape (N,)

#######################################
# 5) shrink_mean => alpha
#######################################
def shrink_mean_to_grand_mean(raw_means: np.ndarray, alpha: float=0.3) -> np.ndarray:
    grand_mean = np.mean(raw_means)
    return (1 - alpha)*raw_means + alpha*grand_mean

#######################################
# 6) parse + clean + old ptf
#######################################
def parse_excel(file, streamlit_sheet="streamlit", histo_sheet="Histo_Price"):
    df_instruments = pd.read_excel(file, sheet_name=streamlit_sheet, header=0)
    df_prices_raw  = pd.read_excel(file, sheet_name=histo_sheet, header=0)
    if df_prices_raw.columns[0] != "Date":
        df_prices_raw.rename(columns={df_prices_raw.columns[0]: "Date"}, inplace=True)
    df_prices_raw["Date"] = pd.to_datetime(df_prices_raw["Date"], errors="coerce")
    df_prices_raw.dropna(subset=["Date"], inplace=True)
    df_prices_raw.set_index("Date", inplace=True)
    df_prices_raw.sort_index(inplace=True)
    df_prices_raw = df_prices_raw.apply(pd.to_numeric, errors="coerce")
    return df_instruments, df_prices_raw

def clean_df_prices(df_prices: pd.DataFrame, min_coverage=0.8):
    df_prices = df_prices.copy()
    coverage = df_prices.notna().sum(axis=1)
    n_cols   = df_prices.shape[1]
    thresh   = n_cols* min_coverage
    df_prices= df_prices[coverage>= thresh].sort_index().ffill().bfill()
    return df_prices

def build_old_portfolio_line(df_instruments, df_prices):
    df_prices= df_prices.sort_index().ffill().bfill()
    ticker_qty={}
    for _, row in df_instruments.iterrows():
        tk= row["#ID"]
        qty= row["#Quantity"]
        ticker_qty[tk]= qty
    col_list= df_prices.columns
    old_shares= np.array([ticker_qty.get(c,0.0) for c in col_list])
    vals=[]
    for _, rowv in df_prices.iterrows():
        vals.append(np.sum(old_shares* rowv.values))
    sr= pd.Series(vals, index=df_prices.index, name="Old_Ptf")
    if len(sr)>1 and sr.iloc[0]!=0:
        sr= sr/sr.iloc[0]
    return sr

#######################################
# 7) The solver => ewm / hist for means & cov
#    => optional mean shrink + diag cov
#    => build class+subtype constraints
#######################################
def parametric_max_sharpe_aclass_subtype(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    security_types: list[str],
    class_constraints: dict,
    subtype_constraints: dict,
    old_class_alloc: dict=None,
    buffer_pct: float=0.0,
    daily_rf: float=0.0,
    frequency: int=252,
    use_keep_current: bool=False,
    debug: bool=True,
    # EWM
    use_ewm: bool=False,
    ewm_alpha: float=0.06,
    # mean shrink
    shrink_means: bool=False,
    alpha_shrink: float=0.3,
    # diag cov shrink
    shrink_cov: bool=False,
    beta_shrink: float=0.2
):
    n = len(tickers)
    if df_returns.shape[1] != n:
        if debug:
            st.write("[DEBUG solver] mismatch => fallback eq weighting")
        return np.ones(n)/n

    # We'll do the data approach manually:
    # 1) If use_ewm => ewm mean, ewm cov. Otherwise => hist mean, hist cov
    if use_ewm:
        # daily means = last row of ewm
        raw_mu_daily = compute_ewm_mean(df_returns, alpha=ewm_alpha)  # shape(n,)
        raw_cov = compute_ewm_cov(df_returns, alpha=ewm_alpha)
        if debug:
            st.write(f"[DEBUG solver] EWM approach => alpha={ewm_alpha}, shapeCov={raw_cov.shape}")
    else:
        # hist
        raw_mu_daily = df_returns.mean(axis=0).values  # shape(n,)
        raw_cov      = df_returns.cov().values
        if debug:
            st.write("[DEBUG solver] Hist approach => shapeCov=", raw_cov.shape)

    # optionally shrink mean
    if shrink_means and alpha_shrink>0:
        raw_mu_daily = shrink_mean_to_grand_mean(raw_mu_daily, alpha=alpha_shrink)
        if debug:
            st.write(f"[DEBUG solver] mean shrink => alpha={alpha_shrink}")

    # fix cov => nearest_pd => optional diag shrink
    raw_cov= nearest_pd(raw_cov,1e-12)
    if shrink_cov and beta_shrink>0:
        raw_cov= shrink_cov_diagonal(raw_cov, beta=beta_shrink)
        if debug:
            st.write(f"[DEBUG solver] diag cov shrink => beta={beta_shrink}")

    # We'll create a minimal rf.Portfolio just to rely on Riskfolio to do 'optimization'
    # Then override port.mu & port.cov
    port = rf.Portfolio(returns=df_returns)
    # initialize
    port.assets_stats(method_mu='hist', method_cov='hist')  # or any
    # now override
    port.mu = pd.Series(raw_mu_daily, index=df_returns.columns)
    port.cov= pd.DataFrame(raw_cov, index=df_returns.columns, columns=df_returns.columns)

    # 2) Build constraints => class sum, then per-instrument subtype
    from collections import defaultdict
    class2idx= defaultdict(list)
    for i, cl in enumerate(asset_classes):
        class2idx[cl].append(i)

    A_ineq=[]
    b_ineq=[]

    def add_sum_le(idxs, lim):
        row= np.zeros(n)
        for ix in idxs:
            row[ix]=1.0
        A_ineq.append(row)
        b_ineq.append(lim)

    def add_sum_ge(idxs, lim):
        row= np.zeros(n)
        for ix in idxs:
            row[ix]= -1.0
        A_ineq.append(row)
        b_ineq.append(-lim)

    # class sum
    if use_keep_current and old_class_alloc:
        if debug:
            st.write("[DEBUG solver] keep_current => old ±", buffer_pct)
        for cl, oldw in old_class_alloc.items():
            idxs= class2idx[cl]
            if idxs:
                low_= max(0.0, oldw - buffer_pct)
                high_=min(1.0, oldw + buffer_pct)
                if debug:
                    st.write(f"[DEBUG solver] class={cl}, oldw={oldw:.3f}, idxs={idxs}, range=({low_:.3f},{high_:.3f})")
                add_sum_le(idxs, high_)
                add_sum_ge(idxs, low_)
    else:
        if debug:
            st.write("[DEBUG solver] custom class constraints =>", class_constraints)
        for cl, cdict in class_constraints.items():
            idxs= class2idx[cl]
            if idxs:
                mn= cdict.get("min_class_weight",0.0)
                mx= cdict.get("max_class_weight",1.0)
                if debug:
                    st.write(f"[DEBUG solver] class={cl}, idxs={idxs}, range=({mn},{mx})")
                add_sum_le(idxs, mx)
                add_sum_ge(idxs, mn)

    # subtype => each single instrument => w[i] in [min_i, max_i]
    # build map
    subtype_map={}
    for (acl, stp), cdict in subtype_constraints.items():
        subtype_map[(acl, stp)] = (
            cdict.get("min_instrument",0.0),
            cdict.get("max_instrument",1.0)
        )

    for i, (acl, stp) in enumerate(zip(asset_classes, security_types)):
        if (acl, stp) in subtype_map:
            mn_i, mx_i= subtype_map[(acl, stp)]
            if mx_i<1.0:
                row_le= np.zeros(n)
                row_le[i]=1.0
                A_ineq.append(row_le)
                b_ineq.append(mx_i)
            if mn_i>0:
                row_ge= np.zeros(n)
                row_ge[i]= -1.0
                A_ineq.append(row_ge)
                b_ineq.append(-mn_i)

    if A_ineq:
        A_= np.array(A_ineq)
        b_= np.array(b_ineq)
        if b_.ndim==1:
            b_= b_.reshape(-1,1)
        port.ainequality= A_
        port.binequality= b_
        if debug:
            st.write(f"[DEBUG solver] final A_ shape={A_.shape}, b_ shape={b_.shape}")
    else:
        if debug:
            st.write("[DEBUG solver] no linear constraints => sum(w)=1 in MV, no short => 0..1")

    # no short => 0..1
    port.lowerlng=0.0
    port.upperlng=1.0

    # 3) Solve => max Sharpe (MV)
    rf_annual= daily_rf * frequency
    try:
        w_solutions= port.optimization(
            model='Classic',
            rm='MV',
            obj='Sharpe',
            rf=rf_annual,
            hist=True
        )
        if w_solutions is None:
            if debug:
                st.write("[DEBUG solver] w_solutions=None => fallback eq weight")
            return np.ones(n)/n
        if debug:
            st.write("[DEBUG solver] final =>", w_solutions.to_dict())
        return w_solutions.values
    except Exception as e:
        if debug:
            st.write("[DEBUG solver] solver exception =>", e)
        return np.ones(n)/n

#######################################
# SHIFTED_START Rolling
#######################################
def rolling_shifted_backtest_aclass_subtype(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    daily_rf: float=0.0,
    user_start: pd.Timestamp=None,
    end_date: pd.Timestamp=None,
    window_days: int=126,
    months_interval: int=1,
    class_sum_constraints: dict=None,
    subtype_constraints: dict=None,
    use_keep_current: bool=False,
    buffer_pct: float=0.0,
    debug: bool=True,
    # EWM
    use_ewm: bool=False,
    ewm_alpha: float=0.06,
    # Mean shrink
    shrink_means: bool=False,
    alpha_shrink: float=0.3,
    # Cov diag shrink
    shrink_cov: bool=False,
    beta_shrink: float=0.2
):
    if class_sum_constraints is None:
        class_sum_constraints={}
    if subtype_constraints is None:
        subtype_constraints={}

    df_prices= df_prices.sort_index().ffill().bfill()
    if end_date:
        df_prices= df_prices.loc[:end_date]
    all_dates= df_prices.index
    if len(all_dates)< window_days:
        sr= pd.Series([1.0], index= all_dates[:1], name="New_Ptf")
        return sr, None

    if user_start is None:
        user_start= all_dates[0]
    start_loc= all_dates.get_indexer([user_start], method='bfill')[0]
    SHIFTED_START_loc= max(start_loc, window_days)
    if SHIFTED_START_loc>= len(all_dates):
        sr= pd.Series([1.0], index= all_dates[:1], name="New_Ptf")
        return sr, None
    SHIFTED_START= all_dates[SHIFTED_START_loc]

    def last_day_of_month(d):
        nm= d+ relativedelta(months=1)
        return nm.replace(day=1) - pd.Timedelta(days=1)

    rebal_dates=[]
    if SHIFTED_START< all_dates[-1]:
        c= last_day_of_month(SHIFTED_START)
        while c<= all_dates[-1]:
            rebal_dates.append(c)
            c= last_day_of_month(c+ relativedelta(months=months_interval))

    def shift_to_valid(d):
        fut= all_dates[all_dates>= d]
        if len(fut)>0:
            return fut[0]
        return all_dates[-1]
    rebal_dates= sorted({shift_to_valid(x) for x in rebal_dates})

    if debug:
        st.write(f"[DEBUG rolling] SHIFTED_START={SHIFTED_START}, SHIFTED_START_loc={SHIFTED_START_loc}")
        st.write("[DEBUG rolling] rebal_dates =>", rebal_dates)

    col_tickers= df_prices.columns.tolist()
    class_map={}
    stype_map={}
    for _, row_ in df_instruments.iterrows():
        tk= row_["#ID"]
        acl= row_["#Asset"]
        st_ = row_.get("#Security_Type","Unknown")
        if pd.isna(st_):
            st_="Unknown"
        class_map[tk]=acl
        stype_map[tk]=st_

    asset_cls_list= [class_map[tk] for tk in col_tickers]
    security_type_list= [stype_map[tk] for tk in col_tickers]

    old_class_alloc={}
    if use_keep_current:
        df_instruments["Value"]= df_instruments["#Quantity"]* df_instruments["#Last_Price"]
        tot_val= df_instruments["Value"].sum()
        if tot_val<=0:
            tot_val=1.0
        df_instruments["Weight_Old"]= df_instruments["Value"]/ tot_val
        class_sums= df_instruments.groupby("#Asset")["Weight_Old"].sum()
        old_class_alloc= class_sums.to_dict()
        if debug:
            st.write("[DEBUG rolling] old_class_alloc =>", old_class_alloc)

    daily_dates= all_dates[SHIFTED_START_loc:]
    n_assets= df_prices.shape[1]
    shares= np.zeros(n_assets)

    SHIFTED_START_price= df_prices.iloc[SHIFTED_START_loc].values
    valid_mask= (SHIFTED_START_price>0)
    eq_w=1.0/ valid_mask.sum() if valid_mask.sum()>0 else 1.0
    tot_val=1.0
    for i in range(n_assets):
        if valid_mask[i]:
            shares[i]= (tot_val* eq_w)/ SHIFTED_START_price[i]

    daily_vals=[tot_val]
    df_returns= df_prices.pct_change().fillna(0.0)

    for d_idx in range(1, len(daily_dates)):
        day= daily_dates[d_idx]
        prices_today= df_prices.loc[day].values
        rolling_val= np.sum(shares* prices_today)
        daily_vals.append(rolling_val)

        if day in rebal_dates and d_idx>= window_days:
            sub_ret= df_returns.iloc[d_idx-window_days: d_idx]
            if debug:
                st.write(f"[DEBUG rolling] Rebalance @ {day}, sub_ret => {sub_ret.index[0]} -> {sub_ret.index[-1]}, shape={sub_ret.shape}")
            w_opt= parametric_max_sharpe_aclass_subtype(
                df_returns= sub_ret,
                tickers= col_tickers,
                asset_classes= asset_cls_list,
                security_types= security_type_list,
                class_constraints= class_sum_constraints,
                subtype_constraints= subtype_constraints,
                old_class_alloc= old_class_alloc,
                buffer_pct= buffer_pct,
                daily_rf= daily_rf,
                frequency= 252,
                use_keep_current= use_keep_current,
                debug= debug,
                use_ewm= st.session_state.get("use_ewm",False),   # We'll override below
                ewm_alpha= st.session_state.get("ewm_alpha",0.06),
                shrink_means= st.session_state.get("shrink_means",False),
                alpha_shrink= st.session_state.get("alpha_shr",0.3),
                shrink_cov= st.session_state.get("shrink_cov",False),
                beta_shrink= st.session_state.get("beta_cov",0.2)
            )
            # Actually let's just pass as function params or do the same approach
            # We'll do a simpler approach => pass them as function arguments
            # for clarity in the final code.

            w_opt= parametric_max_sharpe_aclass_subtype(
                df_returns= sub_ret,
                tickers= col_tickers,
                asset_classes= asset_cls_list,
                security_types= security_type_list,
                class_constraints= class_sum_constraints,
                subtype_constraints= subtype_constraints,
                old_class_alloc= old_class_alloc,
                buffer_pct= buffer_pct,
                daily_rf= daily_rf,
                frequency=252,
                use_keep_current= use_keep_current,
                debug= debug,
                use_ewm= use_ewm,
                ewm_alpha= ewm_alpha,
                shrink_means= shrink_means,
                alpha_shrink= alpha_shrink,
                shrink_cov= shrink_cov,
                beta_shrink= beta_shrink
            )

            if debug:
                st.write("[DEBUG rolling] final w_opt =>", w_opt)

            if rolling_val<0:
                rolling_val=0.0

            new_shares= np.zeros(n_assets)
            for i in range(n_assets):
                if w_opt[i]>1e-15 and prices_today[i]>0:
                    new_shares[i]= (rolling_val*w_opt[i])/ prices_today[i]
            shares= new_shares

    sr= pd.Series(daily_vals, index=daily_dates, name="New_Ptf")
    if sr.iloc[0]<=0:
        sr.iloc[0]=1.0
    sr_norm= sr/sr.iloc[0]
    return sr_norm, shares


#######################################
# compute_perf_stats
#######################################
def compute_perf_stats(series: pd.Series, daily_rf: float=0.0, freq=252):
    if len(series)<2:
        return 0.0, 0.0, 0.0
    dr= series.pct_change().dropna()
    mu_= dr.mean()
    sd_= dr.std()
    ann_ret= (1+mu_)**freq -1
    ann_vol= sd_* np.sqrt(freq)
    ann_rf= daily_rf*freq
    shp=0.0
    if ann_vol>1e-12:
        shp= (ann_ret - ann_rf)/ ann_vol
    return ann_ret, ann_vol, shp

#######################################
# main
#######################################
def main():
    st.title("SHIFTED_START Rolling + EWM Cov + Diag Cov + Mean Shrink + Class + Subtype")

    excel_file= st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    if not excel_file:
        st.stop()

    df_instruments, df_prices= parse_excel(excel_file)
    if df_prices.empty:
        st.error("No data.")
        st.stop()

    coverage= st.slider("Coverage fraction",0.0,1.0,0.8,0.05)
    df_clean= clean_df_prices(df_prices, coverage)
    st.write(f"Clean => shape={df_clean.shape}, from {df_clean.index.min()} to {df_clean.index.max()}")

    earliest= df_clean.index.min()
    latest= df_clean.index.max()
    user_start_date= st.date_input("User Start SHIFTED_START", value=earliest.date(),
                                   min_value=earliest.date(), max_value=latest.date())
    user_start= pd.Timestamp(user_start_date)

    lookback_m= st.selectbox("Lookback (months)", [3,6,12], index=0)
    window_days= lookback_m*21
    rebal_freq= st.selectbox("Rebalance freq (months)", [1,3,6], index=0)
    daily_rf= st.number_input("Daily RF(%)",0.0,10.0,0.0,0.1)/100.0

    # Class constraints
    constraint_mode= st.selectbox("Class Sum Constraints =>", ["custom","keep_current"], index=0)
    buffer_pct= 0.0
    class_sum_constraints={}
    all_classes= df_instruments["#Asset"].unique().tolist()
    if constraint_mode=="custom":
        st.write("**Custom class sum** => min/max sum for each asset class.")
        for cl in all_classes:
            c1, c2= st.columns(2)
            with c1:
                mn_cls= st.number_input(f"Min class sum for {cl}(%)",0.0,100.0,0.0,step=5.0)/100.0
            with c2:
                mx_cls= st.number_input(f"Max class sum for {cl}(%)",0.0,100.0,100.0,step=5.0)/100.0
            class_sum_constraints[cl]={
                "min_class_weight": mn_cls,
                "max_class_weight": mx_cls
            }
    else:
        st.write("**Keep Current** => old sum ± buffer% => each class.")
        buff_= st.number_input("Buffer(%) around old sum weight",0.0,50.0,5.0,1.0)
        buffer_pct= buff_/100.0

    st.write("---")
    st.write("**Subtype constraints** => per instrument => w[i] in [min_i, max_i].")
    df_instruments["SecType"] = df_instruments["#Security_Type"].fillna("Unknown")
    pairs= df_instruments.groupby(["#Asset","SecType"]).size().index.tolist()
    subtype_constraints={}
    for (acl, stp) in pairs:
        stp_label= f"{acl}-{stp}"
        c1, c2= st.columns(2)
        with c1:
            mn_i= st.number_input(f"Min instr(%) for {stp_label}",0.0,100.0,0.0,step=1.0)/100.0
        with c2:
            mx_i= st.number_input(f"Max instr(%) for {stp_label}",0.0,100.0,10.0,step=1.0)/100.0
        subtype_constraints[(acl, stp)] = {
            "min_instrument": mn_i,
            "max_instrument": mx_i
        }

    st.write("---")
    # EWM
    do_ewm= st.checkbox("Use EWM Cov?", value=False)
    ewm_al= 0.06
    if do_ewm:
        ewm_al= st.slider("EWM alpha",0.0,1.0,0.06,0.01)
    # mean shrink
    do_shrink= st.checkbox("Shrink Means?", value=False)
    alpha_shr= 0.3
    if do_shrink:
        alpha_shr= st.slider("Alpha (mean shrink)",0.0,1.0,0.3,0.05)
    # diag cov shrink
    do_cov= st.checkbox("Diag Cov Shrink?", value=False)
    beta_cov=0.2
    if do_cov:
        beta_cov= st.slider("Beta (cov diag shrink)",0.0,1.0,0.2,0.05)

    if st.button("Run SHIFTED_START Rolling"):
        sr_line, final_shares= rolling_shifted_backtest_aclass_subtype(
            df_prices= df_clean,
            df_instruments= df_instruments,
            daily_rf= daily_rf,
            user_start= user_start,
            end_date= latest,
            window_days= window_days,
            months_interval= rebal_freq,
            class_sum_constraints= class_sum_constraints,
            subtype_constraints= subtype_constraints,
            use_keep_current=(constraint_mode=="keep_current"),
            buffer_pct= buffer_pct,
            debug=True,
            use_ewm= do_ewm,
            ewm_alpha= ewm_al,
            shrink_means= do_shrink,
            alpha_shrink= alpha_shr,
            shrink_cov= do_cov,
            beta_shrink= beta_cov
        )
        if len(sr_line)<=1:
            st.warning("No SHIFTED_START => coverage or data length is insufficient, or constraints infeasible.")
            return

        SHIFTED_START_date= sr_line.index[0]
        old_line= build_old_portfolio_line(df_instruments, df_clean)
        if SHIFTED_START_date< old_line.index[0]:
            st.error("SHIFTED_START < old data => no overlap.")
            return
        if SHIFTED_START_date not in old_line.index:
            old_line= old_line.reindex(old_line.index.union([SHIFTED_START_date]).sort_values(), method='ffill')
        old_line_shifted= old_line.loc[SHIFTED_START_date:].copy()
        base_val= old_line_shifted.iloc[0]
        if base_val<=0:
            base_val=1.0
        old_line_shifted= old_line_shifted/base_val

        idx_union= old_line_shifted.index.union(sr_line.index)
        old_line_u= old_line_shifted.reindex(idx_union, method='ffill')
        new_line_u= sr_line.reindex(idx_union, method='ffill')

        def compute_stats(s: pd.Series, daily_rf=0.0, freq=252):
            if len(s)<2:
                return 0.0, 0.0, 0.0
            dr_= s.pct_change().dropna()
            mu_= dr_.mean()
            sd_= dr_.std()
            ann_r= (1+mu_)**freq -1
            ann_v= sd_* np.sqrt(freq)
            ann_rf_= daily_rf*freq
            shp_=0
            if ann_v>1e-12:
                shp_= (ann_r - ann_rf_)/ ann_v
            return ann_r, ann_v, shp_

        ann_ret_old, ann_vol_old, ann_shp_old= compute_stats(old_line_u, daily_rf)
        ann_ret_new, ann_vol_new, ann_shp_new= compute_stats(new_line_u, daily_rf)

        df_plot= pd.DataFrame({"Old":old_line_u,"New":new_line_u}, index= idx_union)
        y_min= df_plot.min().min()
        y_max= df_plot.max().max()
        fig= px.line(df_plot, x=df_plot.index, y=df_plot.columns,
                     title="SHIFTED_START => EWM Cov? + Diag Cov? + Mean Shrink? => Max Sharpe(MV)")
        below= y_min*0.99 if y_min>0 else y_min*1.01
        fig.update_yaxes(range=[below, y_max*1.01])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Performance Stats (SHIFTED_START onward)")
        st.write(f"**Old** => Return={ann_ret_old*100:.2f}%, Vol={ann_vol_old*100:.2f}%, Sharpe={ann_shp_old:.2f}")
        st.write(f"**New** => Return={ann_ret_new*100:.2f}%, Vol={ann_vol_new*100:.2f}%, Sharpe={ann_shp_new:.2f}")
        st.write("Final Old =>", old_line_u.iloc[-1])
        st.write("Final New =>", new_line_u.iloc[-1])


if __name__=="__main__":
    main()
