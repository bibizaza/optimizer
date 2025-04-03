# File: modules/backtesting/rolling_monthly.py

import streamlit as st
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from modules.analytics.extended_metrics import compute_extended_metrics

###############################################################################
# Helper Functions
###############################################################################
def last_day_of_month(date: pd.Timestamp) -> pd.Timestamp:
    nm = date + relativedelta(months=1)
    return nm.replace(day=1) - pd.Timedelta(days=1)

def shift_to_valid_day(date: pd.Timestamp, valid_idx: pd.Index) -> pd.Timestamp:
    if date in valid_idx:
        return date
    after = valid_idx[valid_idx >= date]
    if len(after) > 0:
        return after[0]
    return valid_idx[-1]

def build_monthly_rebal_dates(start_date, end_date, months_interval, df_prices):
    rebal_dates = []
    current = last_day_of_month(start_date)
    while current <= end_date:
        rebal_dates.append(current)
        current = last_day_of_month(current + relativedelta(months=months_interval))

    valid_idx = df_prices.index
    final_dates= []
    for d in rebal_dates:
        d_shifted= shift_to_valid_day(d, valid_idx)
        if d_shifted <= end_date:
            final_dates.append(d_shifted)

    return sorted(list(set(final_dates)))

def compute_transaction_cost(curr_val, old_w, new_w, tx_cost_value, tx_cost_type):
    turnover = np.sum(np.abs(new_w - old_w))
    if tx_cost_type == "percentage":
        return curr_val * turnover * tx_cost_value
    else:
        inst_traded = (np.abs(new_w - old_w) > 1e-9).sum()
        return inst_traded * tx_cost_value


###############################################################################
# Rolling Markowitz - Param (Sharpe)
###############################################################################
def rolling_backtest_monthly_param_sharpe(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    param_sharpe_fn,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    months_interval: int=1,
    window_days: int=252,
    transaction_cost_value: float=0.0,
    transaction_cost_type: str="percentage",
    trade_buffer_pct: float=0.0,
    daily_rf: float=0.0
):
    """
    Returns 7 items:
      (sr_norm, last_w_final, final_shares, final_old_w_last,
       final_rebal_date, df_rebal, ext_metrics)
    """
    df_prices = df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices) < 2:
        sr_ = pd.Series([1.0], index=df_prices.index[:1], name="Rolling_Ptf")
        empty_df= pd.DataFrame()
        return sr_, np.zeros(df_prices.shape[1]), np.zeros(df_prices.shape[1]), \
               np.zeros(df_prices.shape[1]), None, empty_df, {}

    rebal_dates= build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1],
                                           months_interval, df_prices)
    dates= df_prices.index
    n_days= len(dates)
    n_assets= df_prices.shape[1]

    rolling_val=1.0
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

    for d in range(1, n_days):
        day= dates[d]
        px_today= df_prices.loc[day].fillna(0.0).values
        rolling_val= np.sum(shares* px_today)
        daily_vals.append(rolling_val)

        if day in rebal_dates:
            sum_ps= np.sum(shares* px_today)
            if sum_ps<=1e-12:
                old_w= np.zeros(n_assets)
            else:
                old_w= (shares* px_today)/ sum_ps

            final_old_w_last= old_w.copy()
            final_rebal_date= day

            start_idx= max(0, d-window_days)
            sub_ret= df_returns.iloc[start_idx:d]
            if len(sub_ret)<2 or sub_ret.shape[1]<1:
                rebal_events.append({
                    "Date": day,
                    "OldWeights": old_w.copy(),
                    "NewWeights": old_w.copy(),
                    "TxCost": 0.0,
                    "PortValBefore": rolling_val,
                    "PortValAfter": rolling_val
                })
                continue

            w_opt, _= param_sharpe_fn(sub_ret)
            cost= compute_transaction_cost(rolling_val, old_w, w_opt,
                                           transaction_cost_value, transaction_cost_type)
            old_val= rolling_val
            rolling_val-= cost
            if rolling_val<0:
                rolling_val=0

            new_alloc= rolling_val* w_opt
            shares= np.zeros(n_assets)
            for i in range(n_assets):
                if w_opt[i]>1e-9 and px_today[i]>0:
                    shares[i]= new_alloc[i]/ px_today[i]

            last_w_final= w_opt.copy()
            rebal_events.append({
                "Date": day,
                "OldWeights": old_w.copy(),
                "NewWeights": w_opt.copy(),
                "TxCost": cost,
                "PortValBefore": old_val,
                "PortValAfter": rolling_val
            })

    sr_= pd.Series(daily_vals, index=dates, name="Rolling_Ptf")
    if sr_.iloc[0]<=0:
        sr_.iloc[0]=1.0
    sr_norm= sr_/ sr_.iloc[0]
    sr_norm.name= "Rolling_Ptf"

    df_rebal= pd.DataFrame(rebal_events)
    ext_metrics= compute_extended_metrics(sr_norm, daily_rf=daily_rf)
    final_shares= shares.copy()

    return sr_norm, last_w_final, final_shares, final_old_w_last, final_rebal_date, df_rebal, ext_metrics


###############################################################################
# Rolling Markowitz - Direct (Sharpe)
###############################################################################
def rolling_backtest_monthly_direct_sharpe(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    direct_sharpe_fn,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    months_interval: int=1,
    window_days: int=252,
    transaction_cost_value: float=0.0,
    transaction_cost_type: str="percentage",
    trade_buffer_pct: float=0.0,
    daily_rf: float=0.0
):
    """
    Returns (sr_norm, last_w_final, final_shares, final_old_w_last,
             final_rebal_date, df_rebal, ext_metrics)
    """
    df_prices= df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices)<2:
        sr_= pd.Series([1.0], index=df_prices.index[:1], name="Rolling_Ptf")
        empty_df= pd.DataFrame()
        return sr_, np.zeros(df_prices.shape[1]), np.zeros(df_prices.shape[1]), \
               np.zeros(df_prices.shape[1]), None, empty_df, {}

    rebal_dates= build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1],
                                           months_interval, df_prices)
    dates= df_prices.index
    n_days= len(dates)
    n_assets= df_prices.shape[1]

    rolling_val=1.0
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
        px_today= df_prices.loc[day].fillna(0.0).values
        rolling_val= np.sum(shares* px_today)
        daily_vals.append(rolling_val)

        if day in rebal_dates:
            sum_ps= np.sum(shares* px_today)
            if sum_ps<=1e-12:
                old_w= np.zeros(n_assets)
            else:
                old_w= (shares* px_today)/ sum_ps
            final_old_w_last= old_w.copy()
            final_rebal_date= day

            start_idx= max(0, d-window_days)
            sub_ret= df_returns.iloc[start_idx:d]
            if len(sub_ret)<2 or sub_ret.shape[1]<1:
                rebal_events.append({
                    "Date": day,
                    "OldWeights": old_w.copy(),
                    "NewWeights": old_w.copy(),
                    "TxCost": 0.0,
                    "PortValBefore": rolling_val,
                    "PortValAfter": rolling_val
                })
                continue

            w_opt, _= direct_sharpe_fn(sub_ret)
            cost= compute_transaction_cost(rolling_val, old_w, w_opt,
                                           transaction_cost_value, transaction_cost_type)
            old_val= rolling_val
            rolling_val-= cost
            if rolling_val<0:
                rolling_val=0

            new_alloc= rolling_val* w_opt
            shares= np.zeros(n_assets)
            for i in range(n_assets):
                if w_opt[i]>1e-9 and px_today[i]>0:
                    shares[i]= new_alloc[i]/ px_today[i]

            last_w_final= w_opt.copy()
            rebal_events.append({
                "Date": day,
                "OldWeights": old_w.copy(),
                "NewWeights": w_opt.copy(),
                "TxCost": cost,
                "PortValBefore": old_val,
                "PortValAfter": rolling_val
            })

    sr_= pd.Series(daily_vals, index=dates, name="Rolling_Ptf")
    if sr_.iloc[0]<=0:
        sr_.iloc[0]=1.0
    sr_norm= sr_/ sr_.iloc[0]
    sr_norm.name= "Rolling_Ptf"

    df_rebal= pd.DataFrame(rebal_events)
    ext_metrics= compute_extended_metrics(sr_norm, daily_rf=daily_rf)
    final_shares= shares.copy()
    return sr_norm, last_w_final, final_shares, final_old_w_last, final_rebal_date, df_rebal, ext_metrics


###############################################################################
# Param-based Rolling (CVaR)
###############################################################################
def rolling_backtest_monthly_param_cvar(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    param_cvar_fn,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    months_interval: int=1,
    window_days: int=252,
    transaction_cost_value: float=0.0,
    transaction_cost_type: str="percentage",
    trade_buffer_pct: float=0.0,
    daily_rf: float=0.0
):
    """
    Returns 7 items => sr_norm, last_w_final, final_shares, final_old_w_last,
                       final_rebal_date, df_rebal, ext_metrics
    """
    df_prices= df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices)<2:
        sr_= pd.Series([1.0], index=df_prices.index[:1], name="Rolling_Ptf")
        empty_df= pd.DataFrame()
        return sr_, np.zeros(df_prices.shape[1]), np.zeros(df_prices.shape[1]), \
               np.zeros(df_prices.shape[1]), None, empty_df, {}

    rebal_dates= build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1],
                                           months_interval, df_prices)
    dates= df_prices.index
    n_days= len(dates)
    n_assets= df_prices.shape[1]

    rolling_val=1.0
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
        px_today= df_prices.loc[day].fillna(0.0).values
        rolling_val= np.sum(shares* px_today)
        daily_vals.append(rolling_val)

        if day in rebal_dates:
            sum_ps= np.sum(shares* px_today)
            if sum_ps<=1e-12:
                old_w= np.zeros(n_assets)
            else:
                old_w= (shares* px_today)/ sum_ps
            final_old_w_last= old_w.copy()
            final_rebal_date= day

            start_idx= max(0, d-window_days)
            sub_ret= df_returns.iloc[start_idx:d]
            if len(sub_ret)<2 or sub_ret.shape[1]<1:
                rebal_events.append({
                    "Date": day,
                    "OldWeights": old_w.copy(),
                    "NewWeights": old_w.copy(),
                    "TxCost": 0.0,
                    "PortValBefore": rolling_val,
                    "PortValAfter": rolling_val
                })
                continue

            w_opt, summary_ = param_cvar_fn(sub_ret, old_w=old_w)
            if w_opt is None:
                w_opt= old_w.copy()

            cost= compute_transaction_cost(rolling_val, old_w, w_opt,
                                           transaction_cost_value, transaction_cost_type)
            old_val= rolling_val
            rolling_val-= cost
            if rolling_val<0:
                rolling_val=0

            new_alloc= rolling_val* w_opt
            shares= np.zeros(n_assets)
            for i in range(n_assets):
                if w_opt[i]>1e-9 and px_today[i]>0:
                    shares[i]= new_alloc[i]/ px_today[i]

            last_w_final= w_opt.copy()
            rebal_events.append({
                "Date": day,
                "OldWeights": old_w.copy(),
                "NewWeights": w_opt.copy(),
                "TxCost": cost,
                "PortValBefore": old_val,
                "PortValAfter": rolling_val
            })

    sr_= pd.Series(daily_vals, index=dates, name="Rolling_Ptf")
    if sr_.iloc[0]<=0:
        sr_.iloc[0]=1.0
    sr_norm= sr_/ sr_.iloc[0]
    sr_norm.name= "Rolling_Ptf"

    df_rebal= pd.DataFrame(rebal_events)
    ext_metrics= compute_extended_metrics(sr_norm, daily_rf=daily_rf)
    final_shares= shares.copy()

    return sr_norm, last_w_final, final_shares, final_old_w_last, final_rebal_date, df_rebal, ext_metrics


###############################################################################
# Direct Rolling (CVaR)
###############################################################################
def rolling_backtest_monthly_direct_cvar(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    direct_cvar_fn,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    months_interval: int=1,
    window_days: int=252,
    transaction_cost_value: float=0.0,
    transaction_cost_type: str="percentage",
    trade_buffer_pct: float=0.0,
    daily_rf: float=0.0
):
    """
    Return 7 items => sr_norm, last_w_final, final_shares, final_old_w_last,
                      final_rebal_date, df_rebal, ext_metrics
    """
    df_prices= df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices)<2:
        sr_= pd.Series([1.0], index=df_prices.index[:1], name="Rolling_Ptf")
        empty_df= pd.DataFrame()
        return sr_, np.zeros(df_prices.shape[1]), np.zeros(df_prices.shape[1]), \
               np.zeros(df_prices.shape[1]), None, empty_df, {}

    rebal_dates= build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1],
                                           months_interval, df_prices)
    dates= df_prices.index
    n_days= len(dates)
    n_assets= df_prices.shape[1]

    rolling_val=1.0
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
        px_day= df_prices.loc[day].fillna(0.0).values
        rolling_val= np.sum(shares* px_day)
        daily_vals.append(rolling_val)

        if day in rebal_dates:
            sum_ps= np.sum(shares* px_day)
            if sum_ps<=1e-12:
                old_w= np.zeros(n_assets)
            else:
                old_w= (shares* px_day)/ sum_ps
            final_old_w_last= old_w.copy()
            final_rebal_date= day

            start_idx= max(0, d-window_days)
            sub_ret= df_returns.iloc[start_idx:d]
            if len(sub_ret)<2 or sub_ret.shape[1]<1:
                rebal_events.append({
                    "Date": day,
                    "OldWeights": old_w.copy(),
                    "NewWeights": old_w.copy(),
                    "TxCost": 0.0,
                    "PortValBefore": rolling_val,
                    "PortValAfter": rolling_val
                })
                continue

            w_opt, _= direct_cvar_fn(sub_ret)
            cost= compute_transaction_cost(rolling_val, old_w, w_opt,
                                           transaction_cost_value, transaction_cost_type)
            old_val= rolling_val
            rolling_val-= cost
            if rolling_val<0:
                rolling_val=0

            new_alloc= rolling_val* w_opt
            shares= np.zeros(n_assets)
            for i in range(n_assets):
                if w_opt[i]>1e-9 and px_day[i]>0:
                    shares[i]= new_alloc[i]/ px_day[i]

            last_w_final= w_opt.copy()
            rebal_events.append({
                "Date": day,
                "OldWeights": old_w.copy(),
                "NewWeights": w_opt.copy(),
                "TxCost": cost,
                "PortValBefore": old_val,
                "PortValAfter": rolling_val
            })

    sr_= pd.Series(daily_vals, index=dates, name="Rolling_Ptf")
    if sr_.iloc[0]<=0:
        sr_.iloc[0]=1.0
    sr_norm= sr_/ sr_.iloc[0]
    sr_norm.name= "Rolling_Ptf"

    df_rebal= pd.DataFrame(rebal_events)
    ext_metrics= compute_extended_metrics(sr_norm, daily_rf=daily_rf)

    final_shares= shares.copy()
    return sr_norm, last_w_final, final_shares, final_old_w_last, final_rebal_date, df_rebal, ext_metrics


###############################################################################
# COST IMPACT
###############################################################################
def compute_cost_impact(df_rebal: pd.DataFrame, final_portfolio_value: float) -> dict:
    """
    Summarize total Tx cost vs final portfolio value => fraction of final wealth
    """
    if df_rebal is None or df_rebal.empty or "TxCost" not in df_rebal.columns:
        return {
            "Total Cost": 0.0,
            "Cost as % of Final Value": 0.0,
            "Avg Cost per Rebalance": 0.0,
            "Number of Rebalances": 0
        }
    total_cost = df_rebal["TxCost"].sum()
    n_rebals = len(df_rebal)
    avg_cost = total_cost / n_rebals if n_rebals>0 else 0.0
    cost_pct = 0.0
    if final_portfolio_value>1e-12:
        cost_pct= total_cost / final_portfolio_value

    return {
        "Total Cost": total_cost,
        "Cost as % of Final Value": cost_pct,
        "Avg Cost per Rebalance": avg_cost,
        "Number of Rebalances": n_rebals
    }


###############################################################################
# Old Portfolio => Return sr_norm & final_shares
###############################################################################
def backtest_buy_and_hold_that_drifts(
    df_prices: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    ticker_qty: dict
):
    """
    Return 2 items => (sr_norm, final_shares).
    """
    df_prices = df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices) < 2:
        sr_ = pd.Series([1.0], index=df_prices.index[:1], name="BuyHoldDrift")
        return sr_, np.array([])

    col_list= df_prices.columns
    shares= np.array([ticker_qty.get(c,0.0) for c in col_list], dtype=float)

    port_vals= []
    for idx in df_prices.index:
        px= df_prices.loc[idx].fillna(0.0).values
        val_= np.sum(shares* px)
        port_vals.append(val_)

    sr_= pd.Series(port_vals, index=df_prices.index, name="BuyHoldDrift")
    if sr_.iloc[0]<=0:
        sr_.iloc[0]=1.0
    sr_norm= sr_/ sr_.iloc[0]
    sr_norm.name= "BuyHoldDrift"

    return sr_norm, shares


def backtest_strategic_rebalanced(
    df_prices: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    strategic_weights: np.ndarray,
    months_interval: int = 12,
    transaction_cost_value: float = 0.0,
    transaction_cost_type: str = "percentage",
    daily_rf: float = 0.0
):
    """
    Return 2 items => (sr_norm, final_shares).
    """
    df_prices = df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices) < 2:
        sr_ = pd.Series([1.0], index=df_prices.index[:1], name="StrategicPtf")
        return sr_, np.array([])

    sw_sum = strategic_weights.sum()
    if abs(sw_sum - 1.0) > 1e-6:
        strategic_weights= strategic_weights/ sw_sum

    rebal_dates = build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1],
                                            months_interval, df_prices)
    dates= df_prices.index
    n_days= len(dates)
    n_assets= df_prices.shape[1]

    rolling_val=1.0
    shares= np.zeros(n_assets)
    p0= df_prices.iloc[0].fillna(0.0).values
    valid_mask= (p0>0)
    tmp_sw= strategic_weights.copy()
    tmp_sw[~valid_mask]= 0.0
    s2= tmp_sw.sum()
    if s2>1e-12:
        tmp_sw= tmp_sw/ s2

    # initial buy
    for i in range(n_assets):
        if p0[i]>0:
            shares[i]= (rolling_val* tmp_sw[i])/ p0[i]

    daily_vals= [np.sum(shares* p0)]

    for d in range(1,n_days):
        day= dates[d]
        px_day= df_prices.loc[day].fillna(0.0).values
        rolling_val= np.sum(shares* px_day)
        daily_vals.append(rolling_val)

        if day in rebal_dates:
            sum_ps= np.sum(shares* px_day)
            if sum_ps<=1e-12:
                old_w= np.zeros(n_assets)
            else:
                old_w= (shares* px_day)/ sum_ps

            cost= compute_transaction_cost(rolling_val, old_w, tmp_sw,
                                           transaction_cost_value, transaction_cost_type)
            rolling_val-= cost
            if rolling_val<0:
                rolling_val=0

            new_alloc= rolling_val* tmp_sw
            shares= np.zeros(n_assets)
            for i in range(n_assets):
                if px_day[i]>0:
                    shares[i]= new_alloc[i]/ px_day[i]

    sr_= pd.Series(daily_vals, index=dates, name="StrategicPtf")
    if sr_.iloc[0]<=0:
        sr_.iloc[0]=1.0
    sr_norm= sr_/ sr_.iloc[0]
    sr_norm.name= "StrategicPtf"

    final_shares= shares.copy()
    return sr_norm, final_shares


###############################################################################
# OPTIONAL: Grid Search (Placeholder)
###############################################################################
def rolling_grid_search(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    # plus many other params...
):
    pass