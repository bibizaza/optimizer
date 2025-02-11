# modules/backtesting/rolling_monthly.py

import pandas as pd
import numpy as np

def build_monthly_rebal_dates(
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    months_interval: int=1,
    df_prices: pd.DataFrame=None
)-> list[pd.Timestamp]:
    """
    Build a list of rebal dates from start_date => end_date 
    at month-end intervals. If months_interval=1 => monthly, if=3 => quarterly.
    We shift to next valid day if the exact date is not in df_prices index.
    """
    from dateutil.relativedelta import relativedelta

    rebal_dates= []
    current= start_date
    # find the last day of that month
    current= last_day_of_month(current)

    while current<= end_date:
        rebal_dates.append(current)
        current= current+ relativedelta(months=months_interval)
        # shift to last day of that new month
        current= last_day_of_month(current)

    # shift to valid day if not in df_prices
    if df_prices is not None:
        valid_idx= df_prices.index
        final_dates= []
        for d_ in rebal_dates:
            d_shifted= shift_to_valid_day(d_, valid_idx)
            if d_shifted<= end_date:
                final_dates.append(d_shifted)
        return sorted(list(set(final_dates)))
    else:
        return rebal_dates

def last_day_of_month(date: pd.Timestamp)-> pd.Timestamp:
    """
    Return the last day of the month for 'date'
    """
    from dateutil.relativedelta import relativedelta
    next_month= date+ relativedelta(months=1)
    first_day_next= pd.Timestamp(year= next_month.year, month= next_month.month, day=1)
    # one day before that
    return first_day_next- pd.Timedelta(days=1)

def shift_to_valid_day(date: pd.Timestamp, valid_idx: pd.Index)-> pd.Timestamp:
    """
    If 'date' not in valid_idx, shift to the next day in valid_idx after 'date'.
    If none, return last available date. 
    """
    if date in valid_idx:
        return date
    after= valid_idx[valid_idx>= date]
    if len(after)>0:
        return after[0]
    return valid_idx[-1]

def rolling_backtest_monthly(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    optimize_fn,  # returns (weights, summary)
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    months_interval: int=1,  # 1 => monthly, 3 => quarterly
    transaction_cost_value: float=0.001,
    transaction_cost_type: str="percentage",
    missing_instrument_mode: str="partial-liveliness",
    window_days: int=252,
    debug: bool=True
)-> pd.Series:
    """
    Date-based monthly rebal:
      1) invests on start_date => partial-liveliness or equal-dist
      2) each rebal in rebal_dates => if there's <1 year data => partial slice
      3) partial-liveliness => skip price=0 instruments that day
         equal-dist => invests fraction even if price=0 => worthless
      4) cost => percentage or ticket fee
    Returns => 'Rolling_Ptf' as daily series from start_date => end_date => day0=1.0
    """
    df_prices= df_prices.sort_index()
    # slice
    df_prices= df_prices.loc[start_date: end_date].copy()
    if len(df_prices)<2:
        if debug:
            print("[DEBUG Rolling monthly] => not enough data => single point 1.0")
        return pd.Series([1.0], index= df_prices.index[:1], name="Rolling_Ptf")

    rebal_dates= build_monthly_rebal_dates(start_date, end_date, months_interval, df_prices)
    if debug:
        print("[DEBUG monthly rebal =>]", rebal_dates)

    dates= df_prices.index
    n_days= len(dates)
    n_assets= df_prices.shape[1]
    rolling_val= 1.0
    shares= np.zeros(n_assets)

    # day0 invests
    day0= dates[0]
    day0_prices= df_prices.iloc[0].fillna(0.0).values
    if missing_instrument_mode=="partial-liveliness":
        valid_mask= (day0_prices>0)
        sum_valid= np.sum(valid_mask)
        if sum_valid>0:
            eq_w= 1.0/sum_valid
            for i in range(n_assets):
                if valid_mask[i]:
                    money_i= rolling_val* eq_w
                    shares[i]= money_i/day0_prices[i]
    else:
        # "equal-dist"
        eq_w= 1.0/ n_assets
        for i in range(n_assets):
            if day0_prices[i]>0:
                money_i= rolling_val* eq_w
                shares[i]= money_i/day0_prices[i]

    daily_vals= []
    daily_vals.append(np.sum(shares* day0_prices))

    df_returns= df_prices.pct_change().fillna(0.0)

    for d_i in range(1, n_days):
        day= dates[d_i]
        prices_today= df_prices.loc[day].fillna(0.0).values
        rolling_val= np.sum(shares* prices_today)
        daily_vals.append(rolling_val)

        # check if day in rebal_dates
        if day in rebal_dates and d_i>0:
            # we do a 1-year slice => [d_i-window_days, d_i)
            start_i= max(0, d_i- window_days)
            sub_returns= df_returns.iloc[start_i:d_i]
            new_w, _info= optimize_fn(sub_returns)

            # cost => measure turnover
            sum_price_shares= np.sum(shares* prices_today)
            if sum_price_shares<=0:
                old_w= np.zeros(n_assets)
            else:
                old_w= (shares* prices_today)/ sum_price_shares

            if missing_instrument_mode=="partial-liveliness":
                live_mask= (prices_today>0)
                s_live= np.sum(new_w[live_mask])
                if s_live>0:
                    new_w_rescaled= new_w.copy()
                    new_w_rescaled[~live_mask]=0.0
                    new_w_rescaled[live_mask]= new_w_rescaled[live_mask]/ s_live
                else:
                    # skip
                    continue
                turnover= np.sum(np.abs(new_w_rescaled - old_w))
                cost= compute_transaction_cost(rolling_val, turnover, transaction_cost_value, transaction_cost_type, old_w, new_w_rescaled)
                rolling_val-= cost

                money_alloc= rolling_val* new_w_rescaled
                shares= np.zeros(n_assets)
                for i_ in range(n_assets):
                    if new_w_rescaled[i_]>0 and prices_today[i_]>0:
                        shares[i_]= money_alloc[i_]/ prices_today[i_]
            else:
                # equal-dist
                turnover= np.sum(np.abs(new_w- old_w))
                cost= compute_transaction_cost(rolling_val, turnover, transaction_cost_value, transaction_cost_type, old_w, new_w)
                rolling_val-= cost

                money_alloc= rolling_val* new_w
                shares= np.zeros(n_assets)
                for i_ in range(n_assets):
                    if new_w[i_]>0 and prices_today[i_]>0:
                        shares[i_]= money_alloc[i_]/ prices_today[i_]

    sr_abs= pd.Series(daily_vals, index= dates)
    if sr_abs.iloc[0]<=0:
        sr_abs.iloc[0]=1.0
    sr_norm= sr_abs/ sr_abs.iloc[0]
    sr_norm.name= "Rolling_Ptf"
    return sr_norm


def compute_transaction_cost(
    rolling_val: float,
    turnover: float,
    transaction_cost_value: float,
    transaction_cost_type: str,
    old_w: np.ndarray,
    new_w: np.ndarray
)-> float:
    if transaction_cost_type=="percentage":
        cost= rolling_val* turnover* transaction_cost_value
    else:
        # ticket_fee => instruments_traded => sum(1 if abs(new_w[i]- old_w[i])>1e-9)
        instruments_traded= 0
        for i in range(len(new_w)):
            if abs(new_w[i]- old_w[i])>1e-9:
                instruments_traded+=1
        cost= instruments_traded* transaction_cost_value
    return cost
