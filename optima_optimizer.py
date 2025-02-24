import streamlit as st
import pandas as pd
import numpy as np
import riskfolio as rf
from dateutil.relativedelta import relativedelta
import plotly.express as px

#############################################
# 1) Data Loading + Old Portfolio
#############################################
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
    n_cols = df_prices.shape[1]
    threshold = n_cols * min_coverage
    df_prices = df_prices[coverage >= threshold]
    df_prices = df_prices.sort_index().fillna(method="ffill").fillna(method="bfill")
    return df_prices

def build_old_portfolio_line(df_instruments, df_prices):
    df_prices = df_prices.sort_index().fillna(method="ffill").fillna(method="bfill")
    ticker_qty = {}
    for _, row in df_instruments.iterrows():
        tkr = row["#ID"]
        qty = row["#Quantity"]
        ticker_qty[tkr] = qty
    col_list = df_prices.columns
    old_shares = np.array([ticker_qty.get(c,0.0) for c in col_list])
    vals = [np.sum(old_shares * rowv.values) for _, rowv in df_prices.iterrows()]
    sr = pd.Series(vals, index=df_prices.index)
    sr.name = "Old_Ptf"
    return sr

#############################################
# 2) The bulletproof Riskfolio solver w/ fix
#############################################
def parametric_max_sharpe_classconstraints(
    df_returns: pd.DataFrame,
    tickers: list[str],
    asset_classes: list[str],
    class_sum_constraints: dict,
    old_class_alloc: dict = None,
    buffer_pct: float = 0.0,
    daily_rf: float= 0.0,
    frequency: int = 252,
    use_keep_current: bool=False
):
    """
    Solve a single max Sharpe with Riskfolio, building constraints per asset class.
    If use_keep_current=True => we ignore class_sum_constraints min/max
      and do old_class_alloc ± buffer.

    Key fix: we reshape b => (m,1) before assigning port.binequality to avoid
    IndexError: tuple index out of range
    """
    n = len(tickers)
    if df_returns.shape[1] != n:
        raise ValueError("df_returns columns != len(tickers).")

    # 1) Basic portfolio
    port = rf.Portfolio(returns=df_returns)
    port.assets_stats(method_mu='hist', method_cov='hist')

    # 2) Build A_ineq, b_ineq
    from collections import defaultdict
    class2idx = defaultdict(list)
    for i, ac in enumerate(asset_classes):
        class2idx[ac].append(i)

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

    if use_keep_current and old_class_alloc is not None:
        # build constraints from old_class_alloc ± buffer
        for cl, oldw in old_class_alloc.items():
            idxs = class2idx[cl]
            if not idxs:
                continue
            low_ = max(0.0, oldw - buffer_pct)
            high_= min(1.0, oldw + buffer_pct)
            add_sum_le(idxs, high_)
            add_sum_ge(idxs, low_)
    else:
        # use class_sum_constraints
        for cl, cdict in class_sum_constraints.items():
            idxs = class2idx[cl]
            if not idxs:
                continue
            mn = cdict.get("min_class_weight", 0.0)
            mx = cdict.get("max_class_weight", 1.0)
            add_sum_le(idxs, mx)
            add_sum_ge(idxs, mn)

    if len(A_ineq) > 0:
        A = np.array(A_ineq)  # shape (m, n)
        b = np.array(b_ineq)  # shape (m,)

        # Force A to be 2D => typically it's already shape(m,n)
        # if A.ndim==1:
        #     A = A.reshape(1, -1)

        # We must reshape b => (m,1) so Riskfolio doesn't error
        if b.ndim == 1:
            b = b.reshape(-1, 1)  # shape (m,1)

        port.ainequality = A
        port.binequality = b  # now shape (m,1)

    # 3) Solve for max Sharpe
    risk_measure='MV'
    rf_annual= daily_rf * frequency

    try:
        w_solutions= port.optimization(
            model='Classic',
            rm=risk_measure,
            obj='Sharpe',
            rf=rf_annual,
            l=0.0,
            hist=True,
            alpha=0.95,
            weight_bounds=(0,1)
        )
    except:
        return np.ones(n)/n

    if w_solutions is None:
        return np.ones(n)/n

    return w_solutions.values

#############################################
# 3) SHIFTED_START rolling
#############################################
def rolling_backtest_shifted_start_class(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    daily_rf: float=0.0,
    user_start: pd.Timestamp=None,
    end_date: pd.Timestamp=None,
    window_days: int=126,
    months_interval: int=1,
    class_sum_constraints: dict = None,
    use_keep_current: bool=False,
    buffer_pct: float=0.0
):
    if class_sum_constraints is None:
        class_sum_constraints = {}

    df_prices = df_prices.sort_index().fillna(method="ffill").fillna(method="bfill")
    if end_date:
        df_prices = df_prices.loc[:end_date]
    all_dates= df_prices.index
    if len(all_dates) < window_days:
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
        return nm.replace(day=1) - pd.Timedelta(days=1)

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
    # build a map => ticker => #Asset
    asset_cls_map= {}
    for _, row_ in df_instruments.iterrows():
        tk= row_["#ID"]
        asset_cls_map[tk]= row_["#Asset"]
    col_classes= [asset_cls_map.get(tk,"Unknown") for tk in col_tickers]

    old_class_alloc= {}
    if use_keep_current:
        df_instruments["Value"]= df_instruments["#Quantity"]* df_instruments["#Last_Price"]
        tot_val= df_instruments["Value"].sum()
        if tot_val<=0:
            tot_val=1.0
        df_instruments["Weight_Old"]= df_instruments["Value"]/ tot_val
        class_sums= df_instruments.groupby("#Asset")["Weight_Old"].sum()
        old_class_alloc= class_sums.to_dict()

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
        prices_today= df_prices.loc[day].values
        rolling_val= np.sum(shares* prices_today)
        daily_vals.append(rolling_val)

        if day in rebal_dates and idx>= window_days:
            start_idx= idx - window_days
            sub_ret= df_returns.iloc[start_idx: idx]
            w_opt= parametric_max_sharpe_classconstraints(
                df_returns= sub_ret,
                tickers= col_tickers,
                asset_classes= col_classes,
                class_sum_constraints= class_sum_constraints,
                old_class_alloc= old_class_alloc,
                buffer_pct= buffer_pct,
                daily_rf= daily_rf,
                frequency= 252,
                use_keep_current= use_keep_current
            )
            cost= 0.0
            old_val= rolling_val
            rolling_val-= cost
            if rolling_val<0:
                rolling_val=0.0

            new_shares= np.zeros(n_assets)
            for i in range(n_assets):
                if w_opt[i]>1e-15 and prices_today[i]>0:
                    new_shares[i]= (rolling_val* w_opt[i])/ prices_today[i]
            shares= new_shares

    sr= pd.Series(daily_vals, index=daily_dates, name="New_Ptf")
    if sr.iloc[0]<=0:
        sr.iloc[0]=1.0
    sr_norm= sr/sr.iloc[0]
    return sr_norm, shares

#############################################
# Performance stats
#############################################
def compute_perf_stats(series: pd.Series, daily_rf: float=0.0, freq=252):
    if len(series)<2:
        return 0.0, 0.0, 0.0
    daily_ret= series.pct_change().fillna(0.0)
    avg_daily= daily_ret.mean()
    vol_daily= daily_ret.std()
    ann_ret= (1+ avg_daily)**(freq)-1
    ann_vol= vol_daily* np.sqrt(freq)
    ann_rf= daily_rf* freq
    sharpe= 0.0
    if ann_vol> 1e-12:
        sharpe= (ann_ret- ann_rf)/ ann_vol
    return ann_ret, ann_vol, sharpe

#############################################
# Streamlit main
#############################################
def main():
    st.title("Rolling + SHIFTED_START + Class Constraints (Custom vs KeepCurrent) + b-Reshape Fix")

    # 1) Load
    excel_file= st.file_uploader("Upload Excel (.xlsx) with 'streamlit' & 'Histo_Price'", type=["xlsx"])
    if not excel_file:
        st.stop()
    df_instruments, df_prices= parse_excel(excel_file)
    if df_prices.empty:
        st.error("No data.")
        st.stop()

    # 2) coverage
    coverage= st.slider("Coverage fraction", 0.0,1.0,0.8,0.05)
    df_clean= clean_df_prices(df_prices, coverage)
    st.write("Data shape:", df_clean.shape, df_clean.index.min(), df_clean.index.max())

    # 3) user picks approach
    constraint_mode= st.selectbox("Constraint Mode", ["custom","keep_current"], index=0)
    buffer_pct= 0.0
    if constraint_mode=="keep_current":
        buff_in= st.number_input("Buffer (%) around old class weight",0.0,100.0,5.0,1.0)
        buffer_pct= buff_in/ 100.0

    st.write("---")
    # read the classes from the DF
    asset_classes= df_instruments["#Asset"].unique().tolist()
    st.write("Asset Classes found:", asset_classes)

    class_sum_constraints= {}
    if constraint_mode=="custom":
        st.write("## Provide min/max for each asset class.")
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
        st.write("We'll skip user inputs for custom. We'll do keep_current in the solver.")
        class_sum_constraints= {}

    user_start_date= st.date_input("User Start", value=df_clean.index.min().date(),
                                   min_value=df_clean.index.min().date(),
                                   max_value=df_clean.index.max().date())
    user_start= pd.Timestamp(user_start_date)

    lookback_m= st.selectbox("Lookback (months)", [3,6,12], index=0)
    window_days= lookback_m*21
    rebal_freq= st.selectbox("Rebalance freq (months)", [1,3,6], index=0)
    daily_rf= st.number_input("Daily RF(%)",0.0,10.0,0.0,0.1)/100.0

    if st.button("Run SHIFTED_START with Constraints"):
        sr_line, final_shares= rolling_backtest_shifted_start_class(
            df_prices= df_clean,
            df_instruments= df_instruments,
            daily_rf= daily_rf,
            user_start= user_start,
            end_date= df_clean.index.max(),
            window_days= window_days,
            months_interval= rebal_freq,
            class_sum_constraints= class_sum_constraints,
            use_keep_current= (constraint_mode=="keep_current"),
            buffer_pct= buffer_pct
        )
        if len(sr_line)<=1:
            st.warning("Not enough data or coverage for SHIFTED_START => can't run backtest.")
            return
        SHIFTED_START_date= sr_line.index[0]

        # build old line, shift & rebase
        old_line= build_old_portfolio_line(df_instruments, df_clean)
        if SHIFTED_START_date< old_line.index[0]:
            st.error("SHIFTED_START < old_line start => no overlap.")
            return
        if SHIFTED_START_date not in old_line.index:
            old_line= old_line.reindex(
                old_line.index.union([SHIFTED_START_date]).sort_values(),
                method='ffill'
            )
        old_line_shifted= old_line.loc[SHIFTED_START_date:].copy()
        base_val= old_line_shifted.iloc[0]
        if base_val<=0:
            base_val=1.0
        old_line_shifted= old_line_shifted/ base_val

        idx_union= old_line_shifted.index.union(sr_line.index)
        old_line_u= old_line_shifted.reindex(idx_union, method='ffill')
        new_line_u= sr_line.reindex(idx_union, method='ffill')

        # compute stats
        def compute_stats(line):
            ann_ret, ann_vol, ann_sharpe= compute_perf_stats(line, daily_rf=daily_rf)
            return ann_ret, ann_vol, ann_sharpe

        old_ret, old_vol, old_sharpe= compute_stats(old_line_u)
        new_ret, new_vol, new_sharpe= compute_stats(new_line_u)

        df_plot= pd.DataFrame({
            "Old": old_line_u,
            "New": new_line_u
        }, index= idx_union)
        y_min= df_plot.min().min()
        y_max= df_plot.max().max()
        fig= px.line(df_plot, x=df_plot.index, y=df_plot.columns)
        buffer_below= y_min*0.99 if y_min>0 else y_min*1.01
        fig.update_yaxes(range=[buffer_below, y_max*1.01])
        st.plotly_chart(fig, use_container_width=True)

        st.subheader("Performance Stats (from SHIFTED_START onward)")
        st.write(f"**Old**: AnnRet={old_ret*100:.2f}%, AnnVol={old_vol*100:.2f}%, Sharpe={old_sharpe:.2f}")
        st.write(f"**New**: AnnRet={new_ret*100:.2f}%, AnnVol={new_vol*100:.2f}%, Sharpe={new_sharpe:.2f}")

        st.write("Final New:", new_line_u.iloc[-1])
        st.write("Final Old:", old_line_u.iloc[-1])


if __name__=="__main__":
    main()
