import streamlit as st
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

# Extended metrics
from modules.analytics.extended_metrics import compute_extended_metrics

###############################################################################
# Helper Functions
###############################################################################
def last_day_of_month(date: pd.Timestamp) -> pd.Timestamp:
    nm = date + relativedelta(months=1)
    return nm.replace(day=1) - pd.Timedelta(days=1)

def shift_to_valid_day(date: pd.Timestamp, valid_idx: pd.Index) -> pd.Timestamp:
    """
    Given a date and a sorted pd.Index of valid trading days,
    shift the date to the next valid day if the exact date is not present.
    """
    if date in valid_idx:
        return date
    after = valid_idx[valid_idx >= date]
    if len(after) > 0:
        return after[0]
    return valid_idx[-1]

def build_monthly_rebal_dates(start_date, end_date, months_interval, df_prices):
    """
    Build a list of monthly rebalancing dates, properly shifted to
    valid trading dates within df_prices.index.
    """
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
    """
    Compute transaction cost given the old weights, new weights, current portfolio value,
    the transaction cost value, and the cost type.
    - 'percentage' type: cost = (turnover) * (curr_val) * (tx_cost_value)
    - 'per_instrument' type: cost = (# instruments traded) * (tx_cost_value)
    """
    turnover = np.sum(np.abs(new_w - old_w))
    if tx_cost_type == "percentage":
        return curr_val * turnover * tx_cost_value
    else:
        inst_traded = (np.abs(new_w - old_w) > 1e-9).sum()
        return inst_traded * tx_cost_value


###############################################################################
# 1) Param-based Rolling (Markowitz - Sharpe)
###############################################################################
def rolling_backtest_monthly_param_sharpe(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    param_sharpe_fn,   # (sub_ret) => (w_opt, summary)
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    months_interval: int = 1,
    window_days: int = 252,
    transaction_cost_value: float = 0.0,
    transaction_cost_type: str = "percentage",
    trade_buffer_pct: float = 0.0,
    daily_rf: float = 0.0
):
    """
    Rolling backtest for parametric Markowitz (Sharpe maximization).
    Each rebal date, we call param_sharpe_fn on the trailing window of returns.
    """
    df_prices = df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices) < 2:
        empty_line = pd.Series([1.0], index=df_prices.index[:1], name="Rolling_Ptf")
        empty_df = pd.DataFrame(columns=["Date","OldWeights","NewWeights","TxCost","PortValBefore","PortValAfter"])
        return empty_line, np.zeros(df_prices.shape[1]), np.zeros(df_prices.shape[1]), None, empty_df, {}

    rebal_dates = build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1],
                                            months_interval, df_prices)
    dates = df_prices.index
    n_days = len(dates)
    n_assets = df_prices.shape[1]

    # Start with 1.0 capital in an equal-weighted approach (among valid assets).
    rolling_val = 1.0
    shares = np.zeros(n_assets)
    p0 = df_prices.iloc[0].fillna(0.0).values
    valid_mask = (p0 > 0)
    if valid_mask.sum() > 0:
        eq_w = 1.0 / valid_mask.sum()
        for i in range(n_assets):
            if valid_mask[i]:
                shares[i] = (rolling_val * eq_w) / p0[i]

    daily_vals = [np.sum(shares * p0)]
    df_returns = df_prices.pct_change().fillna(0.0)

    last_w_final = np.zeros(n_assets)
    final_old_w_last = np.zeros(n_assets)
    final_rebal_date = None
    rebal_events = []

    for d in range(1, n_days):
        day = dates[d]
        prices_today = df_prices.loc[day].fillna(0.0).values
        rolling_val = np.sum(shares * prices_today)
        daily_vals.append(rolling_val)

        if day in rebal_dates and d > 0:
            sum_price_shares = np.sum(shares * prices_today)
            if sum_price_shares <= 1e-12:
                old_w = np.zeros(n_assets)
            else:
                old_w = (shares * prices_today) / sum_price_shares
            final_old_w_last = old_w.copy()
            final_rebal_date = day

            start_idx = max(0, d - window_days)
            sub_ret = df_returns.iloc[start_idx:d]

            if sub_ret.shape[0] < 2 or sub_ret.shape[1] < 1:
                rebal_events.append({
                    "Date": day,
                    "OldWeights": old_w.copy(),
                    "NewWeights": old_w.copy(),
                    "TxCost": 0.0,
                    "PortValBefore": rolling_val,
                    "PortValAfter": rolling_val
                })
                continue

            # call param_sharpe_fn
            w_opt, _ = param_sharpe_fn(sub_ret)

            cost = compute_transaction_cost(rolling_val, old_w, w_opt,
                                            transaction_cost_value, transaction_cost_type)
            old_val = rolling_val
            rolling_val -= cost
            if rolling_val < 0:
                rolling_val = 0.0

            new_alloc = rolling_val * w_opt
            shares = np.zeros(n_assets)
            for i in range(n_assets):
                if w_opt[i] > 1e-15 and prices_today[i] > 0:
                    shares[i] = new_alloc[i] / prices_today[i]
            last_w_final = w_opt.copy()

            rebal_events.append({
                "Date": day,
                "OldWeights": old_w.copy(),
                "NewWeights": w_opt.copy(),
                "TxCost": cost,
                "PortValBefore": old_val,
                "PortValAfter": rolling_val
            })

    sr = pd.Series(daily_vals, index=dates, name="Rolling_Ptf")
    if sr.iloc[0] <= 0:
        sr.iloc[0] = 1.0
    sr_norm = sr / sr.iloc[0]
    sr_norm.name = "Rolling_Ptf"

    df_rebal = pd.DataFrame(rebal_events)
    ext_metrics = compute_extended_metrics(sr_norm, daily_rf=daily_rf)
    return sr_norm, last_w_final, final_old_w_last, final_rebal_date, df_rebal, ext_metrics


###############################################################################
# 2) Direct Rolling (Markowitz - Sharpe)
###############################################################################
def rolling_backtest_monthly_direct_sharpe(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    direct_sharpe_fn,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    months_interval: int = 1,
    window_days: int = 252,
    transaction_cost_value: float = 0.0,
    transaction_cost_type: str = "percentage",
    trade_buffer_pct: float = 0.0,
    daily_rf: float = 0.0
):
    """
    Rolling backtest for direct Markowitz (Sharpe) approach.
    Each rebal date, we call direct_sharpe_fn on the trailing window of returns.
    """
    df_prices = df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices) < 2:
        empty_line = pd.Series([1.0], index=df_prices.index[:1], name="Rolling_Ptf")
        empty_df = pd.DataFrame(columns=["Date","OldWeights","NewWeights","TxCost","PortValBefore","PortValAfter"])
        return empty_line, np.zeros(df_prices.shape[1]), np.zeros(df_prices.shape[1]), None, empty_df, {}

    rebal_dates = build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1],
                                            months_interval, df_prices)
    dates = df_prices.index
    n_days = len(dates)
    n_assets = df_prices.shape[1]

    rolling_val = 1.0
    shares = np.zeros(n_assets)
    p0 = df_prices.iloc[0].fillna(0.0).values
    valid_mask = (p0 > 0)
    if valid_mask.sum() > 0:
        eq_w = 1.0 / valid_mask.sum()
        for i in range(n_assets):
            if valid_mask[i]:
                shares[i] = (rolling_val * eq_w) / p0[i]

    daily_vals = [np.sum(shares * p0)]
    df_returns = df_prices.pct_change().fillna(0.0)

    last_w_final = np.zeros(n_assets)
    final_old_w_last = np.zeros(n_assets)
    final_rebal_date = None
    rebal_events = []

    for d in range(1, n_days):
        day = dates[d]
        prices_today = df_prices.loc[day].fillna(0.0).values
        rolling_val = np.sum(shares * prices_today)
        daily_vals.append(rolling_val)

        if day in rebal_dates and d > 0:
            sum_price_shares = np.sum(shares * prices_today)
            if sum_price_shares <= 1e-12:
                old_w = np.zeros(n_assets)
            else:
                old_w = (shares * prices_today) / sum_price_shares
            final_old_w_last = old_w.copy()
            final_rebal_date = day

            start_idx = max(0, d - window_days)
            sub_ret = df_returns.iloc[start_idx:d]

            if sub_ret.shape[0] < 2 or sub_ret.shape[1] < 1:
                rebal_events.append({
                    "Date": day,
                    "OldWeights": old_w.copy(),
                    "NewWeights": old_w.copy(),
                    "TxCost": 0.0,
                    "PortValBefore": rolling_val,
                    "PortValAfter": rolling_val
                })
                continue

            w_opt, _ = direct_sharpe_fn(sub_ret)

            cost = compute_transaction_cost(rolling_val, old_w, w_opt,
                                            transaction_cost_value, transaction_cost_type)
            old_val = rolling_val
            rolling_val -= cost
            if rolling_val < 0:
                rolling_val = 0.0

            new_alloc = rolling_val * w_opt
            shares = np.zeros(n_assets)
            for i in range(n_assets):
                if w_opt[i] > 1e-15 and prices_today[i] > 0:
                    shares[i] = new_alloc[i] / prices_today[i]

            last_w_final = w_opt.copy()
            rebal_events.append({
                "Date": day,
                "OldWeights": old_w.copy(),
                "NewWeights": w_opt.copy(),
                "TxCost": cost,
                "PortValBefore": old_val,
                "PortValAfter": rolling_val
            })

    sr = pd.Series(daily_vals, index=dates, name="Rolling_Ptf")
    if sr.iloc[0] <= 0:
        sr.iloc[0] = 1.0
    sr_norm = sr / sr.iloc[0]
    sr_norm.name = "Rolling_Ptf"

    df_rebal = pd.DataFrame(rebal_events)
    ext_metrics = compute_extended_metrics(sr_norm, daily_rf=daily_rf)
    return sr_norm, last_w_final, final_old_w_last, final_rebal_date, df_rebal, ext_metrics


###############################################################################
# 3) Param-based Rolling (CVaR) with Debug & Fallback
###############################################################################
def rolling_backtest_monthly_param_cvar(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    param_cvar_fn,    # (sub_ret, old_w=...) => (w_opt, summary) or (None, None)
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    months_interval: int = 1,
    window_days: int = 252,
    transaction_cost_value: float = 0.0,
    transaction_cost_type: str = "percentage",
    trade_buffer_pct: float = 0.0,
    daily_rf: float = 0.0
):
    """
    If param_cvar_fn returns (None, None) => fallback to old weights => keep_current.
    We add debug prints to see final w_opt or fallback usage.
    """
    df_prices = df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices) < 2:
        empty_line = pd.Series([1.0], index=df_prices.index[:1], name="Rolling_Ptf")
        empty_df = pd.DataFrame(columns=["Date","OldWeights","NewWeights","TxCost","PortValBefore","PortValAfter"])
        return empty_line, np.zeros(df_prices.shape[1]), np.zeros(df_prices.shape[1]), None, empty_df, {}

    rebal_dates = build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1],
                                            months_interval, df_prices)
    dates = df_prices.index
    n_days = len(dates)
    n_assets = df_prices.shape[1]

    rolling_val = 1.0
    shares = np.zeros(n_assets)
    p0 = df_prices.iloc[0].fillna(0.0).values
    valid_mask = (p0 > 0)
    if valid_mask.sum() > 0:
        eq_w = 1.0 / valid_mask.sum()
        for i in range(n_assets):
            if valid_mask[i]:
                shares[i] = (rolling_val * eq_w) / p0[i]

    daily_vals = [np.sum(shares * p0)]
    df_returns = df_prices.pct_change().fillna(0.0)

    last_w_final = np.zeros(n_assets)
    final_old_w_last = np.zeros(n_assets)
    final_rebal_date = None
    rebal_events = []

    for d in range(1, n_days):
        day = dates[d]
        prices_today = df_prices.loc[day].fillna(0.0).values
        rolling_val = np.sum(shares * prices_today)
        daily_vals.append(rolling_val)

        if day in rebal_dates and d > 0:
            sum_price_shares = np.sum(shares * prices_today)
            if sum_price_shares <= 1e-12:
                old_w = np.zeros(n_assets)
            else:
                old_w = (shares * prices_today) / sum_price_shares
            final_old_w_last = old_w.copy()
            final_rebal_date = day

            start_idx = max(0, d - window_days)
            sub_ret = df_returns.iloc[start_idx:d]
            if sub_ret.shape[0] < 2 or sub_ret.shape[1] < 1:
                rebal_events.append({
                    "Date": day,
                    "OldWeights": old_w.copy(),
                    "NewWeights": old_w.copy(),
                    "TxCost": 0.0,
                    "PortValBefore": rolling_val,
                    "PortValAfter": rolling_val
                })
                continue

            print(f"\n[rolling_param_cvar] Rebalance on {day}, old Cash weight ~ {old_w}")

            w_opt, summary = param_cvar_fn(sub_ret, old_w=old_w)
            if w_opt is None:
                print("DEBUG => param_cvar_fn returned None => fallback to old weights (keep_current).")
                w_opt = old_w.copy()
            else:
                print("DEBUG => param_cvar_fn found feasible solution => w_opt sum:", w_opt.sum())

            cost = compute_transaction_cost(rolling_val, old_w, w_opt,
                                            transaction_cost_value, transaction_cost_type)
            old_val = rolling_val
            rolling_val -= cost
            if rolling_val < 0:
                rolling_val = 0.0

            new_alloc = rolling_val * w_opt
            shares = np.zeros(n_assets)
            for i in range(n_assets):
                if w_opt[i] > 1e-15 and prices_today[i] > 0:
                    shares[i] = new_alloc[i] / prices_today[i]

            last_w_final = w_opt.copy()

            rebal_events.append({
                "Date": day,
                "OldWeights": old_w.copy(),
                "NewWeights": w_opt.copy(),
                "TxCost": cost,
                "PortValBefore": old_val,
                "PortValAfter": rolling_val
            })

    sr = pd.Series(daily_vals, index=dates, name="Rolling_Ptf")
    if sr.iloc[0] <= 0:
        sr.iloc[0] = 1.0
    sr_norm = sr / sr.iloc[0]
    sr_norm.name = "Rolling_Ptf"

    df_rebal = pd.DataFrame(rebal_events)
    ext_metrics = compute_extended_metrics(sr_norm, daily_rf=daily_rf)
    return sr_norm, last_w_final, final_old_w_last, final_rebal_date, df_rebal, ext_metrics


###############################################################################
# 4) Direct Rolling (CVaR)
###############################################################################
def rolling_backtest_monthly_direct_cvar(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    direct_cvar_fn,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    months_interval: int = 1,
    window_days: int = 252,
    transaction_cost_value: float = 0.0,
    transaction_cost_type: str = "percentage",
    trade_buffer_pct: float = 0.0,
    daily_rf: float = 0.0
):
    """
    Rolling backtest for direct CVaR approach.
    Each rebal date, we call direct_cvar_fn on the trailing window of returns.
    """
    df_prices = df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices) < 2:
        empty_line = pd.Series([1.0], index=df_prices.index[:1], name="Rolling_Ptf")
        empty_df = pd.DataFrame(columns=["Date","OldWeights","NewWeights","TxCost","PortValBefore","PortValAfter"])
        return empty_line, np.zeros(df_prices.shape[1]), np.zeros(df_prices.shape[1]), None, empty_df, {}

    rebal_dates = build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1],
                                            months_interval, df_prices)
    dates = df_prices.index
    n_days = len(dates)
    n_assets = df_prices.shape[1]

    rolling_val = 1.0
    shares = np.zeros(n_assets)
    p0 = df_prices.iloc[0].fillna(0.0).values
    valid_mask = (p0 > 0)
    if valid_mask.sum() > 0:
        eq_w = 1.0 / valid_mask.sum()
        for i in range(n_assets):
            if valid_mask[i]:
                shares[i] = (rolling_val * eq_w) / p0[i]

    daily_vals = [np.sum(shares * p0)]
    df_returns = df_prices.pct_change().fillna(0.0)

    last_w_final = np.zeros(n_assets)
    final_old_w_last = np.zeros(n_assets)
    final_rebal_date = None
    rebal_events = []

    for d in range(1, n_days):
        day = dates[d]
        prices_today = df_prices.loc[day].fillna(0.0).values
        rolling_val = np.sum(shares * prices_today)
        daily_vals.append(rolling_val)

        if day in rebal_dates and d > 0:
            sum_price_shares = np.sum(shares * prices_today)
            if sum_price_shares <= 1e-12:
                old_w = np.zeros(n_assets)
            else:
                old_w = (shares * prices_today) / sum_price_shares
            final_old_w_last = old_w.copy()
            final_rebal_date = day

            start_idx = max(0, d - window_days)
            sub_ret = df_returns.iloc[start_idx:d]
            if sub_ret.shape[0] < 2 or sub_ret.shape[1] < 1:
                rebal_events.append({
                    "Date": day,
                    "OldWeights": old_w.copy(),
                    "NewWeights": old_w.copy(),
                    "TxCost": 0.0,
                    "PortValBefore": rolling_val,
                    "PortValAfter": rolling_val
                })
                continue

            w_opt, _ = direct_cvar_fn(sub_ret)

            cost = compute_transaction_cost(rolling_val, old_w, w_opt,
                                            transaction_cost_value, transaction_cost_type)
            old_val = rolling_val
            rolling_val -= cost
            if rolling_val < 0:
                rolling_val = 0.0

            new_alloc = rolling_val * w_opt
            shares = np.zeros(n_assets)
            for i in range(n_assets):
                if w_opt[i] > 1e-15 and prices_today[i] > 0:
                    shares[i] = new_alloc[i] / prices_today[i]

            last_w_final = w_opt.copy()
            rebal_events.append({
                "Date": day,
                "OldWeights": old_w.copy(),
                "NewWeights": w_opt.copy(),
                "TxCost": cost,
                "PortValBefore": old_val,
                "PortValAfter": rolling_val
            })

    sr = pd.Series(daily_vals, index=dates, name="Rolling_Ptf")
    if sr.iloc[0] <= 0:
        sr.iloc[0] = 1.0
    sr_norm = sr / sr.iloc[0]
    sr_norm.name = "Rolling_Ptf"

    df_rebal = pd.DataFrame(rebal_events)
    ext_metrics = compute_extended_metrics(sr_norm, daily_rf=daily_rf)
    return sr_norm, last_w_final, final_old_w_last, final_rebal_date, df_rebal, ext_metrics


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
    avg_cost = total_cost / n_rebals if n_rebals > 0 else 0.0
    cost_pct = 0.0
    if final_portfolio_value > 1e-12:
        cost_pct = total_cost / final_portfolio_value

    return {
        "Total Cost": total_cost,
        "Cost as % of Final Value": cost_pct,
        "Avg Cost per Rebalance": avg_cost,
        "Number of Rebalances": n_rebals
    }


###############################################################################
# 5) Old Portfolio Backtests
###############################################################################
def backtest_buy_and_hold_that_drifts(
    df_prices: pd.DataFrame,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    ticker_qty: dict
):
    """
    Simulates a buy-and-hold portfolio (no rebalancing) that starts with EXACTLY
    'ticker_qty' shares for each instrument on day0. Then it simply holds them,
    letting weights drift as prices move.

    *Equivalent to your old "build_old_portfolio_line" approach*
    If the #Quantity are the same as "today," we interpret that as having bought
    them at the start_date and never rebalanced.

    The returned Series is normalized to 1.0 on the first day of df_prices.
    """
    df_prices = df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices) < 2:
        return pd.Series([1.0], index=df_prices.index[:1], name="BuyHoldDrift")

    col_list = df_prices.columns
    # Build an array of shares that is the same length as df_prices.columns
    shares = np.array([ticker_qty.get(c, 0.0) for c in col_list], dtype=float)

    # Calculate daily portfolio value = sum(shares_i * price_i)
    port_values = []
    for idx in df_prices.index:
        prices_row = df_prices.loc[idx].fillna(0.0).values
        val_day = np.sum(shares * prices_row)
        port_values.append(val_day)

    sr = pd.Series(port_values, index=df_prices.index, name="BuyHoldDrift")
    # Normalize to 1.0 at day0
    if sr.iloc[0] <= 0:
        sr.iloc[0] = 1.0
    sr_norm = sr / sr.iloc[0]
    sr_norm.name = "BuyHoldDrift"
    return sr_norm


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
    Simulates a "strategically rebalanced" portfolio. We assume the user wants to
    maintain a fixed weight vector 'strategic_weights' throughout the entire backtest,
    rebalancing every 'months_interval' months.

    - strategic_weights should sum to 1. If not, we normalize it.
    - We start with 1.0 capital, buy shares according to strategic_weights,
      and then on each rebal date, we realign to that same weighting.

    Returns a pd.Series of portfolio values (normalized to 1.0 at the first day).
    """
    df_prices = df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices) < 2:
        return pd.Series([1.0], index=df_prices.index[:1], name="StrategicPtf")

    sw_sum = strategic_weights.sum()
    if abs(sw_sum - 1.0) > 1e-6:
        strategic_weights = strategic_weights / sw_sum

    rebal_dates = build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1],
                                            months_interval, df_prices)
    dates = df_prices.index
    n_days = len(dates)
    n_assets = df_prices.shape[1]

    rolling_val = 1.0
    shares = np.zeros(n_assets)
    p0 = df_prices.iloc[0].fillna(0.0).values
    valid_mask = (p0 > 0)
    # If some assets have zero price, we can't buy them, so we set that weight to 0
    # and re-normalize.
    tmp_sw = strategic_weights.copy()
    tmp_sw[~valid_mask] = 0.0
    s2 = tmp_sw.sum()
    if s2 > 1e-12:
        tmp_sw = tmp_sw / s2

    # initial buy
    for i in range(n_assets):
        if p0[i] > 0:
            shares[i] = (rolling_val * tmp_sw[i]) / p0[i]

    daily_vals = [np.sum(shares * p0)]

    for d in range(1, n_days):
        day = dates[d]
        prices_today = df_prices.loc[day].fillna(0.0).values
        rolling_val = np.sum(shares * prices_today)
        daily_vals.append(rolling_val)

        # Rebalance if day is in rebal_dates
        if day in rebal_dates and d > 0:
            # old weights
            sum_price_shares = np.sum(shares * prices_today)
            if sum_price_shares <= 1e-12:
                old_w = np.zeros(n_assets)
            else:
                old_w = (shares * prices_today) / sum_price_shares

            cost = compute_transaction_cost(rolling_val, old_w, tmp_sw,
                                            transaction_cost_value, transaction_cost_type)
            rolling_val -= cost
            if rolling_val < 0:
                rolling_val = 0.0

            # new shares
            new_alloc = rolling_val * tmp_sw
            shares = np.zeros(n_assets)
            for i in range(n_assets):
                if prices_today[i] > 0:
                    shares[i] = new_alloc[i] / prices_today[i]

    sr = pd.Series(daily_vals, index=dates, name="StrategicPtf")
    if sr.iloc[0] <= 0:
        sr.iloc[0] = 1.0
    sr_norm = sr / sr.iloc[0]
    sr_norm.name = "StrategicPtf"

    return sr_norm


###############################################################################
# OPTIONAL: Grid Search (Placeholder)
###############################################################################
def rolling_grid_search(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    # plus many other params...
):
    """
    Placeholder if you want a parallel or serial approach to multiple param combos.
    """
    pass
