import streamlit as st
import pandas as pd
import numpy as np
import riskfolio as rf
from dateutil.relativedelta import relativedelta

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

def param_no_constraints(df_returns: pd.DataFrame, daily_rf: float=0.0, frequency:int=252):
    """
    No constraints, no shrink. Single max Sharpe with Riskfolio.
    """
    n = df_returns.shape[1]
    port = rf.Portfolio(returns=df_returns)
    port.assets_stats(method_mu='hist', method_cov='hist')

    risk_measure = 'MV'
    rf_annual = daily_rf*frequency

    try:
        w_solutions = port.optimization(
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

def rolling_backtest_shifted_start(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    daily_rf: float = 0.0,
    user_start: pd.Timestamp = None,
    end_date: pd.Timestamp = None,
    window_days: int = 126,
    months_interval: int = 1
):
    """
    1) Slices df_prices up to end_date,
    2) Fills missing,
    3) Finds SHIFTED_START (the day we have >= window_days behind user_start),
    4) On SHIFTED_START => first trailing-window optimization,
    5) Then monthly rebal from SHIFTED_START forward.
    """
    df_prices = df_prices.sort_index()
    if end_date:
        df_prices = df_prices.loc[:end_date]
    df_prices = df_prices.fillna(method="ffill").fillna(method="bfill")
    all_dates = df_prices.index
    if len(all_dates) < window_days:
        return pd.Series([1.0], index=all_dates[:1], name="NoConstr"), None

    if user_start is None:
        user_start = all_dates[0]

    # find user_start_loc
    start_loc = all_dates.get_indexer([user_start], method='bfill')[0]
    SHIFTED_START_loc = max(start_loc, window_days)
    if SHIFTED_START_loc>=len(all_dates):
        sr = pd.Series([1.0], index=all_dates[:1], name="NoConstr")
        return sr, None

    SHIFTED_START = all_dates[SHIFTED_START_loc]

    # build rebal dates from SHIFTED_START
    def last_day_of_month(d):
        nm = d + relativedelta(months=1)
        return nm.replace(day=1) - pd.Timedelta(days=1)

    rebal_dates = []
    if SHIFTED_START<all_dates[-1]:
        curr = last_day_of_month(SHIFTED_START)
        while curr<= all_dates[-1]:
            rebal_dates.append(curr)
            curr = last_day_of_month(curr + relativedelta(months=months_interval))
    valid_idx = all_dates
    def shift_to_valid(d):
        if d in valid_idx:
            return d
        fut_ = valid_idx[valid_idx>=d]
        if len(fut_)>0:
            return fut_[0]
        return valid_idx[-1]
    rebal_dates = sorted(list({shift_to_valid(d) for d in rebal_dates}))

    # We'll hold eq weight at SHIFTED_START, then rebal if SHIFTED_START in rebal_dates
    daily_dates = all_dates[SHIFTED_START_loc:]
    n_assets = df_prices.shape[1]
    shares = np.zeros(n_assets)

    # eq init
    SHIFTED_START_price = df_prices.iloc[SHIFTED_START_loc].values
    eq_mask = (SHIFTED_START_price>0)
    eq_w = 1.0/ eq_mask.sum() if eq_mask.sum()>0 else 1.0
    tot_val = 1.0
    for i in range(n_assets):
        if eq_mask[i]:
            shares[i] = (tot_val*eq_w)/ SHIFTED_START_price[i]

    daily_vals = [tot_val]

    df_returns = df_prices.pct_change().fillna(0.0)
    for idx in range(1, len(daily_dates)):
        day = daily_dates[idx]
        day_loc = all_dates.get_loc(day)
        rolling_val = np.sum(shares*df_prices.iloc[day_loc].values)
        daily_vals.append(rolling_val)

        # rebal if day in rebal_dates AND day_loc>=window_days
        if day in rebal_dates and day_loc>=window_days:
            sub_ret = df_returns.iloc[day_loc - window_days : day_loc]
            w_opt= param_no_constraints(sub_ret, daily_rf=daily_rf)
            new_val= rolling_val
            new_shares= np.zeros(n_assets)
            prices_today= df_prices.iloc[day_loc].values
            for j in range(n_assets):
                if w_opt[j]>1e-15 and prices_today[j]>0:
                    new_shares[j] = (new_val*w_opt[j])/ prices_today[j]
            shares= new_shares

    sr = pd.Series(daily_vals, index=daily_dates, name="NoConstr_New")
    # rebase so SHIFTED_START => 1.0
    if sr.iloc[0]<=0:
        sr.iloc[0]=1.0
    sr_norm = sr/sr.iloc[0]
    return sr_norm, shares

def main():
    st.title("Rolling Backtest - Shift Start For Both Old & New at SHIFTED_START")

    excel_file = st.file_uploader("Upload Excel (.xlsx) with 'streamlit' & 'Histo_Price'", type=["xlsx"])
    if not excel_file:
        st.stop()

    df_instruments, df_prices = parse_excel(excel_file)
    if df_prices.empty:
        st.error("No data.")
        st.stop()

    coverage = st.slider("Coverage fraction", 0.0,1.0,0.8,0.05)
    df_clean= clean_df_prices(df_prices, coverage)
    st.write("Data shape:", df_clean.shape, df_clean.index.min(), df_clean.index.max())

    earliest= df_clean.index.min()
    latest  = df_clean.index.max()
    user_start_date= st.date_input("User Start", value=earliest.date(), min_value=earliest.date(), max_value=latest.date())
    user_start= pd.Timestamp(user_start_date)

    lookback_m= st.selectbox("Lookback (months)", [3,6,12], index=0)
    window_days= lookback_m*21
    rebal_freq= st.selectbox("Rebalance freq (months)", [1,3,6], index=0)
    daily_rf= st.number_input("Daily RF(%)",0.0,10.0,0.0,0.1)/100.0

    if st.button("Run Both Old & New Starting SHIFTED_START"):
        sr_line, final_shares= rolling_backtest_shifted_start(
            df_prices=df_clean,
            df_instruments=df_instruments,
            daily_rf=daily_rf,
            user_start=user_start,
            end_date=latest,
            window_days=window_days,
            months_interval=rebal_freq
        )
        SHIFTED_START_date = sr_line.index[0] if len(sr_line)>0 else None
        if SHIFTED_START_date is None:
            st.warning("No SHIFTED_START => check data & coverage.")
            return

        # build old line but also clip & rebase at SHIFTED_START_date
        old_line = build_old_portfolio_line(df_instruments, df_clean)
        if SHIFTED_START_date< old_line.index[0]:
            st.error("Shifted start is before old data => no overlap.")
            return

        # reindex & ffill so SHIFTED_START_date is in old_line
        if SHIFTED_START_date not in old_line.index:
            # add SHIFTED_START_date with ffill
            old_line= old_line.reindex(
                old_line.index.union([SHIFTED_START_date]).sort_values(),
                method='ffill'
            )

        # Now slice from SHIFTED_START_date forward
        old_line_shifted = old_line.loc[SHIFTED_START_date:].copy()
        # rebase at SHIFTED_START_date => old_line_shifted.iloc[0]
        base_val= old_line_shifted.iloc[0]
        if base_val<=0:
            base_val=1.0
        old_line_shifted= old_line_shifted/base_val

        # unify with sr_line
        # sr_line already 1.0 at SHIFTED_START
        idx_union= old_line_shifted.index.union(sr_line.index)
        old_line_u= old_line_shifted.reindex(idx_union, method='ffill')
        new_line_u= sr_line.reindex(idx_union, method='ffill')

        df_plot= pd.DataFrame({
            "Old": old_line_u,
            "New": new_line_u
        }, index= idx_union)
        st.line_chart(df_plot)
        st.write("Final New:", new_line_u.iloc[-1])
        st.write("Final Old:", old_line_u.iloc[-1])

if __name__=="__main__":
    main()