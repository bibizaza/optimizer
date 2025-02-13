# modules/backtesting/rolling_monthly.py

import streamlit as st
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import traceback

# Optional: If you use a trade buffer function
from modules.backtesting.trade_buffer import apply_trade_buffer

def last_day_of_month(date: pd.Timestamp) -> pd.Timestamp:
    next_m = date + relativedelta(months=1)
    return next_m.replace(day=1) - pd.Timedelta(days=1)

def shift_to_valid_day(date: pd.Timestamp, valid_idx: pd.Index) -> pd.Timestamp:
    """Shift 'date' forward to the next valid day in 'valid_idx' if not found."""
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
    """Return a list of valid rebalancing dates at 'months_interval' months apart."""
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
    """Calculate transaction cost. If 'percentage', cost = currentVal * turnover * costVal."""
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
    Rolls from start_date to end_date in monthly intervals:
      - param_sharpe_fn(sub_ret) => returns (weights, summary).
      - We apply transaction costs, build a time series of portfolio values.

    Returns a 5-tuple:
      1) sr_norm: a pd.Series of the portfolio's normalized daily values
      2) last_w_final: the final chosen weights from the last rebalance
      3) final_old_w_last: weights before the last rebalance
      4) final_rebal_date: the last rebalance date
      5) df_rebal: DataFrame logging each rebalance event

    If you only want 4 items, you can ignore one of them, e.g.:
      sr_line, final_w, _, final_rebal_date, _ = rolling_backtest_monthly_param_sharpe(...)
    """
    df_prices = df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices) < 2:
        empty_line = pd.Series([1.0], index=df_prices.index[:1], name="Rolling_Ptf")
        empty_df = pd.DataFrame(columns=["Date","OldWeights","NewWeights","TxCost","PortValBefore","PortValAfter"])
        return empty_line, np.zeros(df_prices.shape[1]), np.zeros(df_prices.shape[1]), None, empty_df

    rebal_dates = build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1],
                                            months_interval, df_prices)
    dates = df_prices.index
    n_days = len(dates)
    n_assets = df_prices.shape[1]

    # Initialize portfolio (equal weights for nonzero initial instruments)
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

            # Build the sub-DataFrame for returns for the lookback window
            start_idx = max(0, d - window_days)
            sub_ret = df_returns.iloc[start_idx:d]

            # Solve for new weights
            w_opt, _ = param_sharpe_fn(sub_ret)

            # If we have a trade buffer
            if trade_buffer_pct > 0:
                if "apply_trade_buffer" in globals():
                    w_adj = apply_trade_buffer(old_w, w_opt, trade_buffer_pct)
                else:
                    w_adj = w_opt
            else:
                w_adj = w_opt

            # Compute cost
            cost = compute_transaction_cost(rolling_val, old_w, w_adj, transaction_cost_value, transaction_cost_type)
            old_val = rolling_val
            rolling_val -= cost
            if rolling_val < 0:
                rolling_val = 0.0

            # Reallocate
            money_alloc = rolling_val * w_adj
            shares = np.zeros(n_assets)
            for i in range(n_assets):
                if w_adj[i] > 1e-15 and prices_today[i] > 0:
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


def rolling_grid_search(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    asset_cls_list: list[str],
    sec_type_list: list[str],
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float,
    frontier_points_list: list[float],
    alpha_list: list[float],
    beta_list: list[float],
    rebal_freq_list: list[int],
    lookback_list: list[int],
    transaction_cost_value: float = 0.0,
    transaction_cost_type: str = "percentage",
    trade_buffer_pct: float = 0.0,
    use_michaud: bool = False,
    n_boot: int = 10,
    do_shrink_means: bool = True,
    do_shrink_cov: bool = True,
    reg_cov: bool = False,
    do_ledoitwolf: bool = False,
    do_ewm: bool = False,
    ewm_alpha: float = 0.06,
    max_workers: int = 4
) -> pd.DataFrame:
    """
    Parallel grid search that calls run_one_combo(...) in child processes.
    """
    import concurrent.futures
    import time

    from modules.backtesting.rolling_gridsearch import run_one_combo

    combos = []
    for n_pts in frontier_points_list:
        for alpha_ in alpha_list:
            for beta_ in beta_list:
                for freq_ in rebal_freq_list:
                    for lb_ in lookback_list:
                        combos.append((n_pts, alpha_, beta_, freq_, lb_))

    total = len(combos)
    df_out = []

    start_time = time.time()
    progress_text = st.empty()
    progress_bar = st.progress(0)
    completed = 0

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_combo = {}
        for combo in combos:
            future = executor.submit(
                run_one_combo,
                df_prices,
                df_instruments,
                asset_cls_list,
                sec_type_list,
                class_sum_constraints,
                subtype_constraints,
                daily_rf,
                combo,
                transaction_cost_value,
                transaction_cost_type,
                trade_buffer_pct,
                use_michaud,
                n_boot,
                do_shrink_means,
                do_shrink_cov,
                reg_cov,
                do_ledoitwolf,
                do_ewm,
                ewm_alpha
            )
            future_to_combo[future] = combo

        for future in concurrent.futures.as_completed(future_to_combo):
            combo = future_to_combo[future]
            try:
                result_dict = future.result()
            except Exception as exc:
                st.error(f"Error for combo {combo}: {exc}")
                st.text(traceback.format_exc())
                n_pts, alpha_, beta_, freq_, lb_ = combo
                result_dict = {
                    "n_points": n_pts,
                    "alpha_mean": alpha_,
                    "beta_cov": beta_,
                    "rebal_freq": freq_,
                    "lookback_m": lb_,
                    "Sharpe Ratio": 0.0,
                    "Annual Ret": 0.0,
                    "Annual Vol": 0.0,
                    "final_weights": None
                }

            df_out.append(result_dict)
            completed += 1
            pct = int(completed * 100 / total)
            elapsed = time.time() - start_time
            progress_text.text(f"Grid Search: {pct}% done, Elapsed: {elapsed:.1f}s")
            progress_bar.progress(pct)

    total_elapsed = time.time() - start_time
    progress_text.text(f"Grid search done in {total_elapsed:.1f}s.")
    return pd.DataFrame(df_out)
