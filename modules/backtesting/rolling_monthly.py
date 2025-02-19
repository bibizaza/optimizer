# File: modules/backtesting/rolling_monthly.py

import streamlit as st
import pandas as pd
import numpy as np
from dateutil.relativedelta import relativedelta

# If you have a trade buffer function
# from modules.backtesting.trade_buffer import apply_trade_buffer

# Import extended metrics
from modules.analytics.extended_metrics import compute_extended_metrics


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
    trade_buffer_pct: float = 0.0,
    daily_rf: float = 0.0
) -> tuple[
    pd.Series,    # sr_norm
    np.ndarray,   # last_w_final
    np.ndarray,   # final_old_w_last
    pd.Timestamp, # final_rebal_date
    pd.DataFrame, # df_rebal
    dict          # ext_metrics
]:
    """
    Rolls monthly between start_date and end_date:
     - param_sharpe_fn(sub_ret) => returns (w_opt, summary)
     - applies transaction cost
     - returns 6 items: (sr_norm, last_w_final, final_old_w_last, final_rebal_date, df_rebal, ext_metrics)

    ext_metrics => from compute_extended_metrics
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
            w_opt, _ = param_sharpe_fn(sub_ret)

            # optional trade buffer
            if trade_buffer_pct > 0:
                if "apply_trade_buffer" in globals():
                    w_adj = apply_trade_buffer(old_w, w_opt, trade_buffer_pct)
                else:
                    w_adj = w_opt
            else:
                w_adj = w_opt

            cost = compute_transaction_cost(rolling_val, old_w, w_adj,
                                            transaction_cost_value, transaction_cost_type)
            old_val = rolling_val
            rolling_val -= cost
            if rolling_val < 0:
                rolling_val = 0.0

            money_alloc = rolling_val * w_adj
            shares = np.zeros(n_assets)
            for i in range(n_assets):
                if w_adj[i] > 1e-15 and prices_today[i] > 0:
                    shares[i] = money_alloc[i] / prices_today[i]

            last_w_final = w_adj.copy()

            rebal_events.append({
                "Date": day,
                "OldWeights": old_w.copy(),
                "NewWeights": w_opt.copy(),  # or w_adj if you prefer
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

    # *** Extended Metrics ***
    ext_metrics = compute_extended_metrics(sr_norm, daily_rf=daily_rf)

    return sr_norm, last_w_final, final_old_w_last, final_rebal_date, df_rebal, ext_metrics


def compute_cost_impact(df_rebal: pd.DataFrame, final_portfolio_value: float) -> dict:
    """
    Summarizes the total transaction cost from df_rebal["TxCost"]
    vs. the final portfolio value => fraction of final wealth.
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
    if final_portfolio_value > 1e-12:
        cost_pct_final = total_cost / final_portfolio_value

    return {
        "Total Cost": total_cost,
        "Cost as % of Final Value": cost_pct_final,
        "Avg Cost per Rebalance": avg_cost,
        "Number of Rebalances": n_rebals
    }


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
    A parallel grid search example, returning a DataFrame of combos.
    If you want extended metrics in each run, you can adapt. 
    For now, we keep it minimal or you can fill in.
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
