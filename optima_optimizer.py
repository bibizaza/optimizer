import streamlit as st
import pandas as pd
import numpy as np
import riskfolio as rf
from dateutil.relativedelta import relativedelta
import plotly.express as px

############################
# Utility to force PD
############################
def nearest_pd(a, epsilon=1e-8):
    """Force a symmetric matrix to be positive-definite by clipping negative eigenvalues."""
    a = 0.5*(a + a.T)
    vals, vecs = np.linalg.eigh(a)
    vals = np.clip(vals, epsilon, None)
    return (vecs @ np.diag(vals) @ vecs.T)

############################
# parse_excel, clean_df_prices, build_old_portfolio_line
############################
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
    coverage = df_prices.notna().sum(axis=1)
    n_cols = df_prices.shape[1]
    threshold = n_cols * min_coverage
    df_prices = df_prices[coverage >= threshold]
    df_prices = df_prices.sort_index().ffill().bfill()
    return df_prices

def build_old_portfolio_line(df_instruments, df_prices):
    df_prices = df_prices.sort_index().ffill().bfill()
    ticker_qty = {}
    for _, row in df_instruments.iterrows():
        tk = row["#ID"]
        qty = row["#Quantity"]
        ticker_qty[tk] = qty
    col_list = df_prices.columns
    old_shares = np.array([ticker_qty.get(c, 0.0) for c in col_list])
    vals = [np.sum(old_shares * rowv.values) for _, rowv in df_prices.iterrows()]
    sr = pd.Series(vals, index=df_prices.index)
    sr.name = "Old_Ptf"
    return sr

############################
# compute_perf_stats
############################
def compute_perf_stats(series: pd.Series, daily_rf: float=0.0, freq=252):
    if len(series)<2:
        return 0.0, 0.0, 0.0
    daily_ret = series.pct_change().fillna(0.0)
    avg_daily = daily_ret.mean()
    vol_daily = daily_ret.std()
    ann_ret   = (1+ avg_daily)**(freq)-1
    ann_vol   = vol_daily*np.sqrt(freq)
    ann_rf    = daily_rf*freq
    sharpe    = 0.0
    if ann_vol>1e-12:
        sharpe = (ann_ret - ann_rf)/ ann_vol
    return ann_ret, ann_vol, sharpe

############################
# Key: Debug-enabled parametric_max_sharpe_classconstraints
############################
def parametric_max_sharpe_classconstraints(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    class_sum_constraints: dict,
    old_class_alloc: dict=None,
    buffer_pct: float=0.0,
    daily_rf: float=0.0,
    frequency: int=252,
    use_keep_current: bool=False,
    debug: bool=True
):
    """
    - Force PD covariance
    - Skip classes w/ no tickers
    - Single-row shape fix
    - Debug prints around port.ainequality / port.binequality
    """
    n = len(tickers)
    if df_returns.shape[1] != n:
        raise ValueError(f"[DEBUG] mismatch => df_returns has {df_returns.shape[1]} cols, tickers={n}.")

    # Create portfolio
    port = rf.Portfolio(returns=df_returns)
    port.assets_stats(method_mu='hist', method_cov='hist')

    # Force PD on port.cov
    # We do nearest_pd on the matrix
    cov_pd = nearest_pd(port.cov.values)
    port.cov = pd.DataFrame(cov_pd, index=port.cov.index, columns=port.cov.columns)

    from collections import defaultdict
    class2idx = defaultdict(list)
    for i, ac in enumerate(asset_classes):
        class2idx[ac].append(i)

    A_ineq = []
    b_ineq = []

    def add_sum_le(idxs, limit):
        if len(idxs)==0:
            if debug:
                print(f"[DEBUG] skip add_sum_le => no idxs => limit={limit}")
            return
        row = np.zeros(n)
        for ix in idxs:
            row[ix]=1.0
        A_ineq.append(row)
        b_ineq.append(limit)

    def add_sum_ge(idxs, limit):
        if len(idxs)==0:
            if debug:
                print(f"[DEBUG] skip add_sum_ge => no idxs => limit={limit}")
            return
        row = np.zeros(n)
        for ix in idxs:
            row[ix]= -1.0
        A_ineq.append(row)
        b_ineq.append(-limit)

    if use_keep_current and old_class_alloc is not None:
        if debug:
            print("[DEBUG] Using keep_current => old_class_alloc + buffer =", buffer_pct)
        for cl, oldw in old_class_alloc.items():
            idxs= class2idx[cl]
            if debug:
                print(f"   -> class='{cl}', oldw={oldw:.3f}, idxs={idxs}")
            if len(idxs)==0:
                print("[DEBUG] skip => no final tickers for class",cl)
                continue
            low_ = max(0.0, oldw - buffer_pct)
            high_= min(1.0, oldw + buffer_pct)
            add_sum_le(idxs, high_)
            add_sum_ge(idxs, low_)
    else:
        if debug:
            print("[DEBUG] Using custom constraints =>", class_sum_constraints)
        for cl, cdict in class_sum_constraints.items():
            idxs= class2idx[cl]
            if debug:
                print(f"   -> class='{cl}', idxs={idxs}, cdict={cdict}")
            if len(idxs)==0:
                print("[DEBUG] skip => no final tickers for class",cl)
                continue
            mn= cdict.get("min_class_weight",0.0)
            mx= cdict.get("max_class_weight",1.0)
            add_sum_le(idxs, mx)
            add_sum_ge(idxs, mn)

    if debug:
        print("[DEBUG] Done building constraint rows => len(A_ineq)=", len(A_ineq))

    if len(A_ineq)>0:
        A = np.array(A_ineq)
        b = np.array(b_ineq)
        if debug:
            print("[DEBUG] A shape before fix =>", A.shape, "b shape =>", b.shape)
            print("[DEBUG] A =>", A)
            print("[DEBUG] b =>", b)
        # Single-row fix
        if A.ndim==1:
            A= A.reshape(1, -1)
        if b.ndim==0:
            b= b.reshape(1,)
        if A.shape[0]!= b.shape[0]:
            raise ValueError(f"[DEBUG] row mismatch => A.shape[0]={A.shape[0]}, b.shape[0]={b.shape[0]}")
        if A.shape[1]!= n:
            raise ValueError(f"[DEBUG] col mismatch => A.shape[1]={A.shape[1]}, n={n}")

        if debug:
            print("[DEBUG] A shape after fix =>", A.shape, "b shape =>", b.shape)

        port.ainequality = A

        # We'll do a try-except around port.binequality to catch the IndexError
        try:
            # We can also do some shape prints right before:
            a_stored = port.ainequality
            if debug and a_stored is not None:
                print("[DEBUG] Just set port.ainequality => shape =", a_stored.shape)
            print("[DEBUG] About to set port.binequality => b shape=", b.shape)
            port.binequality = b
        except IndexError as e:
            # debug info
            print("[DEBUG] Caught IndexError at port.binequality = b step!")
            a_stored = getattr(port, 'ainequality', None)
            if a_stored is not None:
                print("[DEBUG] port.ainequality shape=", a_stored.shape)
                print("[DEBUG] port.ainequality =>", a_stored)
            print("[DEBUG] b shape =>", b.shape)
            print("[DEBUG] b =>", b)
            raise e  # re-raise
    else:
        if debug:
            print("[DEBUG] No constraints => skip port.ainequality/binequality")

    risk_measure='MV'
    rf_annual= daily_rf* frequency
    if debug:
        print("[DEBUG] Attempting optimization => risk_measure=MV, rf_annual=", rf_annual)
    try:
        w_solutions= port.optimization(
            model='Classic',
            rm= risk_measure,
            obj= 'Sharpe',
            rf= rf_annual,
            l= 0.0,
            hist= True,
            alpha= 0.95,
            weight_bounds= (0,1)
        )
    except Exception as e:
        if debug:
            print("[DEBUG] optimization exception =>", e)
        return np.ones(n)/n

    if w_solutions is None:
        if debug:
            print("[DEBUG] w_solutions is None => fallback eq weights.")
        return np.ones(n)/n

    return w_solutions.values

############################################
# SHIFTED_START rolling
############################################
def rolling_backtest_shifted_start_class(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    daily_rf: float=0.0,
    user_start: pd.Timestamp=None,
    end_date: pd.Timestamp=None,
    window_days: int=126,
    months_interval: int=1,
    class_sum_constraints: dict=None,
    use_keep_current: bool=False,
    buffer_pct: float=0.0
):
    if class_sum_constraints is None:
        class_sum_constraints= {}

    df_prices= df_prices.sort_index().ffill().bfill()
    if end_date:
        df_prices= df_prices.loc[:end_date]
    all_dates= df_prices.index
    if len(all_dates)< window_days:
        return pd.Series([1.0], index= all_dates[:1], name="New_Ptf"), None

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
    rebal_dates= []
    if SHIFTED_START< all_dates[-1]:
        c= last_day_of_month(SHIFTED_START)
        while c<= all_dates[-1]:
            rebal_dates.append(c)
            c= last_day_of_month(c+ relativedelta(months= months_interval))
    def shift_to_valid(d):
        if d in all_dates:
            return d
        fut= all_dates[all_dates>= d]
        if len(fut)>0:
            return fut[0]
        return all_dates[-1]
    rebal_dates= sorted({shift_to_valid(x) for x in rebal_dates})

    col_tickers= df_prices.columns.tolist()
    asset_cls_map= {}
    for _, row in df_instruments.iterrows():
        tk= row["#ID"]
        asset_cls_map[tk]= row["#Asset"]
    col_classes= [asset_cls_map.get(tk,"Unknown") for tk in col_tickers]

    old_class_alloc={}
    if use_keep_current:
        df_instruments["Value"]= df_instruments["#Quantity"]* df_instruments["#Last_Price"]
        tot_val= df_instruments["Value"].sum()
        if tot_val<=0:
            tot_val=1.0
        df_instruments["Weight_Old"]= df_instruments["Value"]/ tot_val
        class_old= df_instruments.groupby("#Asset")["Weight_Old"].sum()
        old_class_alloc= class_old.to_dict()

    daily_dates= all_dates[SHIFTED_START_loc:]
    n_assets= df_prices.shape[1]
    shares= np.zeros(n_assets)
    SHIFTED_START_price= df_prices.iloc[SHIFTED_START_loc].values
    eq_mask= (SHIFTED_START_price>0)
    eq_w= 1.0/ eq_mask.sum() if eq_mask.sum()>0 else 1.0
    tot_val= 1.0
    for i in range(n_assets):
        if eq_mask[i]:
            shares[i]= (tot_val* eq_w)/ SHIFTED_START_price[i]

    daily_vals= [tot_val]
    df_returns= df_prices.pct_change().fillna(0.0)

    for idx in range(1, len(daily_dates)):
        day= daily_dates[idx]
        day_loc= all_dates.get_loc(day)
        rolling_val= np.sum(shares* df_prices.iloc[day_loc].values)
        daily_vals.append(rolling_val)

        if (day in rebal_dates) and (day_loc>= window_days):
            sub_ret= df_returns.iloc[day_loc- window_days: day_loc]
            w_opt= parametric_max_sharpe_classconstraints(
                df_returns= sub_ret,
                tickers= col_tickers,
                asset_classes= col_classes,
                class_sum_constraints= class_sum_constraints,
                old_class_alloc= old_class_alloc,
                buffer_pct= buffer_pct,
                daily_rf= daily_rf,
                frequency= 252,
                use_keep_current= use_keep_current,
                debug=True  # <--- debugging ON
            )
            new_val= rolling_val
            new_shares= np.zeros(n_assets)
            prices_today= df_prices.iloc[day_loc].values
            for j in range(n_assets):
                if w_opt[j]>1e-15 and prices_today[j]>0:
                    new_shares[j]= (new_val*w_opt[j])/ prices_today[j]
            shares= new_shares

    sr= pd.Series(daily_vals, index= daily_dates, name="New_Ptf")
    if sr.iloc[0]<=0:
        sr.iloc[0]=1.0
    sr_norm= sr/sr.iloc[0]
    return sr_norm, shares

############################################
# main
############################################
def main():
    st.title("SHIFTED_START + Class Constraints + Force PD Cov + Full Debug")

    excel_file= st.file_uploader("Upload Excel (.xlsx)", type=["xlsx"])
    if not excel_file:
        st.stop()

    df_instruments, df_prices= parse_excel(excel_file)
    if df_prices.empty:
        st.error("No data in Excel.")
        st.stop()

    coverage= st.slider("Coverage fraction",0.0,1.0,0.8,0.05)
    df_clean= clean_df_prices(df_prices, coverage)
    st.write(f"Shape => {df_clean.shape}, from {df_clean.index.min()} to {df_clean.index.max()}")

    approach= st.selectbox("Constraint Mode",["custom","keep_current"], index=0)
    buffer_pct= 0.0
    if approach=="keep_current":
        buff_in= st.number_input("Buffer(%) around old class weight",0.0,100.0,5.0,1.0)
        buffer_pct= buff_in/100.0

    # gather classes
    asset_classes= df_instruments["#Asset"].unique().tolist()
    st.write("Asset Classes =>", asset_classes)

    class_sum_constraints={}
    if approach=="custom":
        st.write("Min/Max constraints for each class:")
        for cl in asset_classes:
            c1, c2= st.columns(2)
            with c1:
                mn_val= st.number_input(f"Min weight for {cl}",0.0,1.0,0.0,0.05)
            with c2:
                mx_val= st.number_input(f"Max weight for {cl}",0.0,1.0,1.0,0.05)
            class_sum_constraints[cl]= {
                "min_class_weight": mn_val,
                "max_class_weight": mx_val
            }
    else:
        st.write("We skip user-defined => we'll do keep_current => old ± buffer.")
        class_sum_constraints= {}

    earliest= df_clean.index.min()
    latest= df_clean.index.max()
    user_start_date= st.date_input("User Start Date", value= earliest.date(),
                                   min_value= earliest.date(), max_value= latest.date())
    user_start= pd.Timestamp(user_start_date)
    lookback_m= st.selectbox("Lookback( months )",[3,6,12], index=0)
    window_days= lookback_m*21
    rebal_freq= st.selectbox("Rebalance freq (months)", [1,3,6], index=0)
    daily_rf= st.number_input("Daily RF(%)",0.0,10.0,0.0,0.1)/100.0

    if st.button("Run SHIFTED_START w/ Debug"):
        sr_line, final_shares= rolling_backtest_shifted_start_class(
            df_prices= df_clean,
            df_instruments= df_instruments,
            daily_rf= daily_rf,
            user_start= user_start,
            end_date= latest,
            window_days= window_days,
            months_interval= rebal_freq,
            class_sum_constraints= class_sum_constraints,
            use_keep_current= (approach=="keep_current"),
            buffer_pct= buffer_pct
        )
        if len(sr_line)<2:
            st.warning("No SHIFTED_START => not enough data or coverage.")
            return
        SHIFTED_START_date= sr_line.index[0]

        # old line
        old_line= build_old_portfolio_line(df_instruments, df_clean)
        if SHIFTED_START_date< old_line.index[0]:
            st.error("Shifted start < old data => no overlap.")
            return
        if SHIFTED_START_date not in old_line.index:
            old_line= old_line.reindex(old_line.index.union([SHIFTED_START_date]).sort_values(), method='ffill')

        old_line_shifted= old_line.loc[SHIFTED_START_date:].copy()
        basev= old_line_shifted.iloc[0]
        if basev<=0:
            basev=1.0
        old_line_shifted= old_line_shifted/basev

        idx_union= old_line_shifted.index.union(sr_line.index)
        old_line_u= old_line_shifted.reindex(idx_union, method='ffill')
        new_line_u= sr_line.reindex(idx_union, method='ffill')

        def do_stats(line):
            return compute_perf_stats(line, daily_rf=daily_rf)
        old_ret, old_vol, old_sharpe= do_stats(old_line_u)
        new_ret, new_vol, new_sharpe= do_stats(new_line_u)

        df_plot= pd.DataFrame({"Old": old_line_u, "New": new_line_u}, index= idx_union)
        y_min= df_plot.min().min()
        y_max= df_plot.max().max()
        fig= px.line(df_plot, x=df_plot.index, y=df_plot.columns)
        below= y_min*0.99 if y_min>0 else y_min*1.01
        fig.update_yaxes(range=[below, y_max*1.01])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("SHIFTED_START Perf Stats")
        st.write(f"**Old** => Ret={old_ret*100:.2f}%, Vol={old_vol*100:.2f}%, Sharpe={old_sharpe:.2f}")
        st.write(f"**New** => Ret={new_ret*100:.2f}%, Vol={new_vol*100:.2f}%, Sharpe={new_sharpe:.2f}")
        st.write("Final Old =>", old_line_u.iloc[-1])
        st.write("Final New =>", new_line_u.iloc[-1])

        st.info("Check your terminal for [DEBUG] prints. If IndexError occurs, the try-except block around port.binequality will show shapes.")
        

if __name__=="__main__":
    main()