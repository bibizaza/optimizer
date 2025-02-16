# File: modules/backtesting/rolling_monthly.py

import streamlit as st
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta
import traceback

# If you have a trade buffer function, you might import it:
# from modules.backtesting.trade_buffer import apply_trade_buffer

def last_day_of_month(date: pd.Timestamp) -> pd.Timestamp:
    """
    Returns the last day of the month for 'date'.
    """
    next_m = date + relativedelta(months=1)
    return next_m.replace(day=1) - pd.Timedelta(days=1)

def shift_to_valid_day(date: pd.Timestamp, valid_idx: pd.Index) -> pd.Timestamp:
    """
    Shift 'date' forward to the next valid day in 'valid_idx' if 'date' not in the index.
    If none found, returns the last valid day.
    """
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
    """
    Return a sorted list of valid rebalancing dates at 'months_interval' intervals
    between 'start_date' and 'end_date', each shifted to a trading day in df_prices.
    """
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
    final_dates = sorted(list(set(final_dates)))
    return final_dates

def compute_transaction_cost(curr_val: float,
                             old_w: np.ndarray,
                             new_w: np.ndarray,
                             tx_cost_value: float,
                             tx_cost_type: str) -> float:
    """
    Calculates transaction cost when going from old_w to new_w,
    given the current portfolio value 'curr_val'.

    If 'percentage', cost = curr_val * turnover * tx_cost_value
    Else cost = (# instruments changed) * tx_cost_value
    """
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
    Performs a monthly rolling backtest:
      - param_sharpe_fn(sub_ret) => (w_opt, summary)
      - applies transaction costs
      - returns (sr_norm, last_w_final, final_old_w_last, final_rebal_date, df_rebal)
    
    sr_norm: pd.Series of daily portfolio values, normalized to 1 at start.
    last_w_final: final chosen weights from last rebal
    final_old_w_last: old weights before the last rebal
    final_rebal_date: date of last rebal
    df_rebal: details each rebal event (TxCost, etc.)
    """
    # 1) subset df_prices
    df_prices = df_prices.sort_index().loc[start_date:end_date]
    if len(df_prices) < 2:
        empty_line = pd.Series([1.0], index=df_prices.index[:1], name="Rolling_Ptf")
        empty_df = pd.DataFrame(columns=["Date","OldWeights","NewWeights","TxCost","PortValBefore","PortValAfter"])
        return empty_line, np.zeros(df_prices.shape[1]), np.zeros(df_prices.shape[1]), None, empty_df

    # 2) build monthly rebal dates
    rebal_dates = build_monthly_rebal_dates(df_prices.index[0], df_prices.index[-1],
                                            months_interval, df_prices)
    dates = df_prices.index
    n_days = len(dates)
    n_assets = df_prices.shape[1]

    # 3) init portfolio => equal weighting among nonzero instruments at day 0
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

    # 4) loop daily
    for d in range(1, n_days):
        day = dates[d]
        prices_today = df_prices.loc[day].fillna(0.0).values
        rolling_val = np.sum(shares * prices_today)
        daily_vals.append(rolling_val)

        # if day is a rebal date => do param_sharpe_fn
        if day in rebal_dates and d > 0:
            sum_price_shares = np.sum(shares * prices_today)
            if sum_price_shares <= 1e-12:
                old_w = np.zeros(n_assets)
            else:
                old_w = (shares * prices_today) / sum_price_shares
            final_old_w_last = old_w.copy()
            final_rebal_date = day

            # sub-ret for last 'window_days'
            start_idx = max(0, d - window_days)
            sub_ret = df_returns.iloc[start_idx:d]

            # param_sharpe_fn => (w_opt, summary)
            w_opt, _ = param_sharpe_fn(sub_ret)

            # optional trade buffer
            if trade_buffer_pct > 0:
                if "apply_trade_buffer" in globals():
                    w_adj = apply_trade_buffer(old_w, w_opt, trade_buffer_pct)
                else:
                    w_adj = w_opt
            else:
                w_adj = w_opt

            # cost
            cost = compute_transaction_cost(rolling_val, old_w, w_adj,
                                            transaction_cost_value, transaction_cost_type)
            old_val = rolling_val
            rolling_val -= cost
            if rolling_val < 0:
                rolling_val = 0.0

            # reallocate => new shares
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


########################################################################
# New function summarizing transaction cost impact
########################################################################
def compute_cost_impact(df_rebal: pd.DataFrame, final_portfolio_value: float) -> dict:
    """
    Summarizes the total transaction cost from df_rebal["TxCost"]
    vs. the final portfolio value => see how big cost is as fraction of final wealth.

    If df_rebal is empty or missing 'TxCost', returns zeros.

    Returns
    -------
    dict
        {
          "Total Cost": float,
          "Cost as % of Final Value": float,
          "Avg Cost per Rebalance": float,
          "Number of Rebalances": int
        }
    """
    if df_rebal is None or "TxCost" not in df_rebal.columns or len(df_rebal) == 0:
        return {
            "Total Cost": 0.0,
            "Cost as % of Final Value": 0.0,
            "Avg Cost per Rebalance": 0.0,
            "Number of Rebalances": 0
        }

    total_cost = df_rebal["TxCost"].sum()
    n_rebals = len(df_rebal)
    avg_cost = total_cost / n_rebals if n_rebals > 0 else 0.0

    cost_pct_final = 0.0
    if final_portfolio_value > 0:
        cost_pct_final = total_cost / final_portfolio_value

    return {
        "Total Cost": total_cost,
        "Cost as % of Final Value": cost_pct_final,
        "Avg Cost per Rebalance": avg_cost,
        "Number of Rebalances": n_rebals
    }

########################################################################
# rolling_grid_search logic (optional)
########################################################################
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
    Parallel grid search with combos of (n_points, alpha, beta, rebal_freq, lookback).
    Calls run_one_combo(...) from rolling_gridsearch for each combo,
    collects Sharpe Ratio, etc. for each combo, and returns a DataFrame.
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