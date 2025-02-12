import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
from modules.backtesting.trade_buffer import apply_trade_buffer

def last_day_of_month(date: pd.Timestamp) -> pd.Timestamp:
    next_m = date + relativedelta(months=1)
    return next_m.replace(day=1) - pd.Timedelta(days=1)

def shift_to_valid_day(date: pd.Timestamp, valid_idx: pd.Index) -> pd.Timestamp:
    if date in valid_idx:
        return date
    after = valid_idx[valid_idx >= date]
    if len(after) > 0:
        return after[0]
    return valid_idx[-1]

def build_monthly_rebal_dates(start_date: pd.Timestamp,
                              end_date: pd.Timestamp,
                              months_interval: int,
                              df_prices: pd.DataFrame) -> list[pd.Timestamp]:
    rebal_dates = []
    current = last_day_of_month(start_date)
    while current <= end_date:
        rebal_dates.append(current)
        current = last_day_of_month(current + relativedelta(months=months_interval))
    valid_idx = df_prices.index
    final_dates = []
    for d in rebal_dates:
        d_shifted = shift_to_valid_day(d, valid_idx)
        if d_shifted <= end_date:
            final_dates.append(d_shifted)
    return sorted(list(set(final_dates)))

def compute_transaction_cost(curr_val: float,
                             old_w: np.ndarray,
                             new_w: np.ndarray,
                             tx_cost_value: float,
                             tx_cost_type: str) -> float:
    turnover = np.sum(np.abs(new_w - old_w))
    if tx_cost_type == "percentage":
        return curr_val * turnover * tx_cost_value
    else:
        inst_traded = (np.abs(new_w - old_w) > 1e-9).sum()
        return inst_traded * tx_cost_value

def rolling_backtest_monthly_param_sharpe(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    param_sharpe_fn,
    start_date: pd.Timestamp,
    end_date: pd.Timestamp,
    months_interval: int = 1,
    window_days: int = 252,
    transaction_cost_value: float = 0.0,
    transaction_cost_type: str = "percentage",
    trade_buffer_pct: float = 0.0
) -> tuple[pd.Series, np.ndarray, np.ndarray, pd.Timestamp, pd.DataFrame]:
    """
    Performs a monthly rolling backtest.
    Returns:
      - sr_norm: normalized portfolio value time series
      - last_w_final: final weight vector at last rebalance
      - final_old_w_last_rebal: weights immediately before the last rebalance
      - final_rebal_date: date of the last rebalance
      - df_rebal: DataFrame logging each rebalance event
    """
    df_prices = df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices) < 2:
        empty_line = pd.Series([1.0], index=df_prices.index[:1], name="Rolling_Ptf")
        empty_df = pd.DataFrame(columns=["Date", "OldWeights", "NewWeights", "TxCost", "PortValBefore", "PortValAfter"])
        return empty_line, np.zeros(df_prices.shape[1]), np.zeros(df_prices.shape[1]), None, empty_df

    rebal_dates = build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1], months_interval, df_prices)
    dates = df_prices.index
    n_days = len(dates)
    n_assets = df_prices.shape[1]

    # Initialize portfolio: equal weights for instruments with nonzero initial prices.
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
            w_opt, _ = param_sharpe_fn(sub_ret)

            if trade_buffer_pct > 0:
                w_adj = apply_trade_buffer(old_w, w_opt, trade_buffer_pct)
            else:
                w_adj = w_opt

            cost = compute_transaction_cost(rolling_val, old_w, w_adj, transaction_cost_value, transaction_cost_type)
            old_val = rolling_val
            rolling_val -= cost
            if rolling_val < 0:
                rolling_val = 0.0

            money_alloc = rolling_val * w_adj
            shares = np.zeros(n_assets)
            for i in range(n_assets):
                if w_adj[i] > 0 and prices_today[i] > 0:
                    shares[i] = money_alloc[i] / prices_today[i]
            last_w_final = w_adj.copy()

            rebal_events.append({
                "Date": day,
                "OldWeights": old_w.copy(),
                "NewWeights": w_adj.copy(),
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
    return sr_norm, last_w_final, final_old_w_last, final_rebal_date, df_rebal
