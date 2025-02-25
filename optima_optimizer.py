import streamlit as st
import pandas as pd
import numpy as np
import riskfolio as rf
from dateutil.relativedelta import relativedelta
from collections import defaultdict
import plotly.express as px

#############################################
# Force Covariance to PSD
#############################################
def nearest_pd(cov, epsilon=1e-12):
    """
    Force a symmetric matrix to be PSD by clipping negative eigenvalues.
    """
    mat = 0.5*(cov + cov.T)
    vals, vecs = np.linalg.eigh(mat)
    vals_clipped = np.clip(vals, epsilon, None)
    return (vecs @ np.diag(vals_clipped) @ vecs.T)

#############################################
# parse_excel + clean_df_prices + build_old_portfolio_line
#############################################
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

def clean_df_prices(df_prices: pd.DataFrame, min_coverage=0.8) -> pd.DataFrame:
    df_prices = df_prices.copy()
    coverage  = df_prices.notna().sum(axis=1)
    n_cols    = df_prices.shape[1]
    threshold = n_cols * min_coverage
    df_prices = df_prices[coverage>=threshold]
    df_prices = df_prices.sort_index().ffill().bfill()
    return df_prices

def build_old_portfolio_line(df_instruments, df_prices):
    df_prices = df_prices.sort_index().ffill().bfill()
    ticker_qty={}
    for _, row in df_instruments.iterrows():
        tk  = row["#ID"]
        qty = row["#Quantity"]
        ticker_qty[tk] = qty
    col_list= df_prices.columns
    old_shares= np.array([ticker_qty.get(c,0.0) for c in col_list])
    vals=[]
    for _, rowv in df_prices.iterrows():
        vals.append(np.sum(old_shares* rowv.values))
    sr= pd.Series(vals, index=df_prices.index, name="Old_Ptf")
    # normalize to start=1 if possible
    if len(sr)>1 and sr.iloc[0]!=0:
        sr= sr/sr.iloc[0]
    return sr

#############################################
# parametric_max_sharpe_aclass_subtype
#   - class constraints => sum in [min,max]
#   - subtype constraints => *each instrument* in [min_inst, max_inst]
#############################################
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
    debug: bool=True
):
    n = len(tickers)
    if df_returns.shape[1]!=n:
        if debug:
            st.write("[DEBUG solver] mismatch => fallback eq")
        return np.ones(n)/n

    # 1) Create portfolio
    port = rf.Portfolio(returns=df_returns)
    port.assets_stats(method_mu='hist', method_cov='hist')

    # 2) Force PSD
    cov_arr= port.cov.values
    cov_psd= nearest_pd(cov_arr,1e-12)
    port.cov= pd.DataFrame(cov_psd, index=port.cov.index, columns=port.cov.columns)

    from collections import defaultdict
    class2idx= defaultdict(list)
    for i, cl in enumerate(asset_classes):
        class2idx[cl].append(i)

    A_ineq=[]
    b_ineq=[]

    def add_sum_le(idxs, limit):
        row= np.zeros(n)
        for ix in idxs:
            row[ix]=1.0
        A_ineq.append(row)
        b_ineq.append(limit)

    def add_sum_ge(idxs, limit):
        row= np.zeros(n)
        for ix in idxs:
            row[ix]=-1.0
        A_ineq.append(row)
        b_ineq.append(-limit)

    # 3) Class constraints => sum
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
        # custom => each class => sum in [min,max]
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

    # 4) Subtype => *per-instrument*
    if debug:
        st.write("[DEBUG solver] Subtype constraints =>", subtype_constraints)
    # We build a map from (class, stype) to (min_inst, max_inst).
    # Then for each instrument i that belongs => w[i] in [min_inst, max_inst].
    # i.e. "each instrument must be in that range"
    # We'll create a single row per instrument i => w[i]<= max_instrument, w[i]>= min_instrument
    subtype_map= {}
    for (acl, stp), cdict in subtype_constraints.items():
        # just store for easy lookup
        subtype_map[(acl, stp)] = (cdict.get("min_instrument",0.0), cdict.get("max_instrument",1.0))

    # Now loop over each instrument i => interpret "subtype constraint => each single instrument in that subtype"
    for i, (cl, stp) in enumerate(zip(asset_classes, security_types)):
        if (cl, stp) in subtype_map:
            (mn_inst, mx_inst)= subtype_map[(cl, stp)]
            # w[i] <= mx_inst
            if mx_inst< 1.0:
                row_le= np.zeros(n)
                row_le[i]= 1.0
                A_ineq.append(row_le)
                b_ineq.append(mx_inst)
            # w[i] >= mn_inst
            if mn_inst> 0.0:
                row_ge= np.zeros(n)
                row_ge[i]= -1.0
                A_ineq.append(row_ge)
                b_ineq.append(-mn_inst)

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
            st.write("[DEBUG solver] no linear constraints => sum(w)=1 from MV, lowerlng=0 => no short")

    # 5) ensure no short
    port.lowerlng=0.0
    port.upperlng=1.0

    # 6) solve => max Sharpe (MV)
    rf_annual= daily_rf* frequency
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
                st.write("[DEBUG solver] w_solutions is None => fallback eq")
            return np.ones(n)/n
        if debug:
            st.write("[DEBUG solver] final =>", w_solutions.to_dict())
        return w_solutions.values
    except Exception as e:
        if debug:
            st.write("[DEBUG solver] solver exception =>", e)
        return np.ones(n)/n


#############################################
# SHIFTED_START rolling
#############################################
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
    debug: bool=True
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
        nm= d + relativedelta(months=1)
        return nm.replace(day=1)- pd.Timedelta(days=1)

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

    # gather each ticker => class + stype
    col_tickers= df_prices.columns.tolist()
    class_map={}
    stype_map={}
    for _, row_ in df_instruments.iterrows():
        tk= row_["#ID"]
        acl= row_["#Asset"]
        st_= row_.get("#Security_Type","Unknown")
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
    eq_w= 1.0/ valid_mask.sum() if valid_mask.sum()>0 else 1.0
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
                st.write(f"[DEBUG rolling] Rebalance @ {day} => sub_ret from {sub_ret.index[0]} to {sub_ret.index[-1]}, shape={sub_ret.shape}")
            w_opt= parametric_max_sharpe_aclass_subtype(
                df_returns=sub_ret,
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
                debug= debug
            )
            if debug:
                st.write("[DEBUG rolling] final w_opt =>", w_opt)

            # no transaction cost => cost=0
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


#############################################
# compute_perf_stats
#############################################
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
        shp= (ann_ret-ann_rf)/ann_vol
    return ann_ret, ann_vol, shp

#############################################
# main
#############################################
def main():
    st.title("SHIFTED_START Rolling with Class Sum Constraints & Subtype *Per-Instrument* Constraints")

    # 1) load data
    excel_file= st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    if not excel_file:
        st.stop()

    df_instruments, df_prices= parse_excel(excel_file)
    if df_prices.empty:
        st.error("No valid data.")
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
    rebal_freq= st.selectbox("Rebalance freq (months)",[1,3,6], index=0)
    daily_rf= st.number_input("Daily RF(%)",0.0,10.0,0.0,0.1)/100.0

    constraint_mode= st.selectbox("Constraint Mode for Classes => sum", ["custom","keep_current"], index=0)
    buffer_pct= 0.0
    class_sum_constraints={}
    all_classes= df_instruments["#Asset"].unique().tolist()
    if constraint_mode=="custom":
        st.write("**Custom class sum constraints** => specify min/max sum for each asset class.")
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
        st.write("**Keep current** => old class weight ± buffer% => sum constraints.")
        buff_= st.number_input("Buffer(%) around old class weight", 0.0,50.0,5.0,1.0)
        buffer_pct= buff_/100.0

    st.write("---")
    st.write("**Subtype Per-Instrument** => each instrument in that subtype must be in [min_instrument, max_instrument].")
    # gather (class, stype)
    df_instruments["SecType"] = df_instruments["#Security_Type"].fillna("Unknown")
    pairs= df_instruments.groupby(["#Asset","SecType"]).size().index.tolist()
    subtype_constraints={}
    for (acl,stp) in pairs:
        # user says: for each instrument in that pair => w[i] in [min_instrument, max_instrument]
        stp_label= f"{acl}-{stp}"
        c1, c2= st.columns(2)
        with c1:
            mn_i= st.number_input(f"Min instrument(%) for {stp_label}", 0.0,100.0,0.0,step=1.0)/100.0
        with c2:
            mx_i= st.number_input(f"Max instrument(%) for {stp_label}", 0.0,100.0,10.0,step=1.0)/100.0
        # We'll interpret that as "each instrument in that subtype => w[i] in [mn_i, mx_i]"
        subtype_constraints[(acl,stp)] = {
            "min_instrument": mn_i,
            "max_instrument": mx_i
        }

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
            debug=True
        )
        if len(sr_line)<=1:
            st.warning("No SHIFTED_START => possibly coverage or data length is insufficient.")
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
                return 0,0,0
            ret_= s.pct_change().dropna()
            mu_= ret_.mean()
            sd_= ret_.std()
            ann_r= (1+mu_)**freq -1
            ann_v= sd_* np.sqrt(freq)
            ann_rf_= daily_rf*freq
            shp_=0
            if ann_v>1e-12:
                shp_= (ann_r-ann_rf_)/ann_v
            return ann_r, ann_v, shp_

        ann_ret_old, ann_vol_old, ann_shp_old= compute_stats(old_line_u, daily_rf)
        ann_ret_new, ann_vol_new, ann_shp_new= compute_stats(new_line_u, daily_rf)

        df_plot= pd.DataFrame({"Old":old_line_u, "New":new_line_u}, index=idx_union)
        y_min= df_plot.min().min()
        y_max= df_plot.max().max()
        fig= px.line(df_plot, x=df_plot.index, y=df_plot.columns,
                     title="SHIFTED_START Rolling => class sum + subtype per-instrument constraints => Max Sharpe (MV)")
        below= y_min*0.99 if y_min>0 else y_min*1.01
        fig.update_yaxes(range=[below, y_max*1.01])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Performance Stats (SHIFTED_START onward)")
        st.write(f"**Old** => Ret={ann_ret_old*100:.2f}%, Vol={ann_vol_old*100:.2f}%, Sharpe={ann_shp_old:.2f}")
        st.write(f"**New** => Ret={ann_ret_new*100:.2f}%, Vol={ann_vol_new*100:.2f}%, Sharpe={ann_shp_new:.2f}")
        st.write("Final Old =>", old_line_u.iloc[-1])
        st.write("Final New =>", new_line_u.iloc[-1])


if __name__=="__main__":
    main()
