# modules/backtesting/rolling_gridsearch.py

import pandas as pd
import numpy as np
import time
import streamlit as st
import concurrent.futures
import traceback

from modules.analytics.returns_cov import compute_performance_metrics
from modules.backtesting.rolling_monthly import rolling_backtest_monthly_param_sharpe
from modules.optimization.cvxpy_optimizer import parametric_max_sharpe
from modules.optimization.utils.michaud import parametric_max_sharpe_michaud

def run_one_combo(
    df_prices, df_instruments, asset_classes, class_constraints,
    daily_rf, combo,
    transaction_cost_value, transaction_cost_type, trade_buffer_pct,
    use_michaud, n_boot, do_shrink_means, do_shrink_cov, reg_cov,
    do_ledoitwolf, do_ewm, ewm_alpha
):
    """
    Top-level function => can be pickled.
    combo => (n_pts, alpha_, beta_, freq_, lb_)
    Runs a single param set, returning performance metrics & final weights.
    """
    n_pts, alpha_, beta_, freq_, lb_ = combo
    col_tickers = df_prices.columns.tolist()

    def param_sharpe_fn(sub_ret: pd.DataFrame):
        if use_michaud:
            w_opt, _ = parametric_max_sharpe_michaud(
                df_returns=sub_ret,
                tickers=col_tickers,
                asset_classes=asset_classes,
                class_constraints=class_constraints,
                daily_rf=daily_rf,
                no_short=True,
                n_points=n_pts,
                regularize_cov=reg_cov,
                shrink_means=do_shrink_means,
                alpha=alpha_,
                shrink_cov=do_shrink_cov,
                beta=beta_,
                use_ledoitwolf=do_ledoitwolf,
                do_ewm=do_ewm,
                ewm_alpha=ewm_alpha,
                n_boot=n_boot
            )
        else:
            w_opt, _ = parametric_max_sharpe(
                df_returns=sub_ret,
                tickers=col_tickers,
                asset_classes=asset_classes,
                class_constraints=class_constraints,
                daily_rf=daily_rf,
                no_short=True,
                n_points=n_pts,
                regularize_cov=reg_cov,
                shrink_means=do_shrink_means,
                alpha=alpha_,
                shrink_cov=do_shrink_cov,
                beta=beta_,
                use_ledoitwolf=do_ledoitwolf,
                do_ewm=do_ewm,
                ewm_alpha=ewm_alpha
            )
        return w_opt, {}

    # Convert months to days
    window_days = lb_ * 21

    sr_line, final_w, _, _ = rolling_backtest_monthly_param_sharpe(
        df_prices=df_prices,
        df_instruments=df_instruments,
        param_sharpe_fn=param_sharpe_fn,
        start_date=df_prices.index[0],
        end_date=df_prices.index[-1],
        months_interval=freq_,
        window_days=window_days,
        transaction_cost_value=transaction_cost_value,
        transaction_cost_type=transaction_cost_type,
        trade_buffer_pct=trade_buffer_pct
    )

    if len(sr_line) > 1:
        perf = compute_performance_metrics(sr_line, daily_rf=daily_rf)
        sr = perf["Sharpe Ratio"]
        ann_ret = perf["Annualized Return"]
        ann_vol = perf["Annualized Volatility"]
    else:
        sr = 0.0
        ann_ret = 0.0
        ann_vol = 0.0

    return {
        "n_points": n_pts,
        "alpha_mean": alpha_,
        "beta_cov": beta_,
        "rebal_freq": freq_,
        "lookback_m": lb_,
        "Sharpe Ratio": sr,
        "Annual Ret": ann_ret,
        "Annual Vol": ann_vol,
        "final_weights": final_w
    }

def rolling_grid_search(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    asset_classes: list[str],
    class_constraints: dict,
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
    Parallel grid search with concurrent.futures.
    Each combo => run_one_combo => child process => returns performance.
    """
    combos = []
    for n_pts in frontier_points_list:
        for alpha_ in alpha_list:
            for beta_ in beta_list:
                for freq_ in rebal_freq_list:
                    for lb_ in lookback_list:
                        combos.append((n_pts, alpha_, beta_, freq_, lb_))
    total = len(combos)
    df_out = []

    progress_text = st.empty()
    progress_bar = st.progress(0)
    start_time = time.time()
    completed = 0

    from concurrent.futures import ProcessPoolExecutor, as_completed

    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        future_to_combo = {}
        for combo in combos:
            future = executor.submit(
                run_one_combo,
                df_prices, df_instruments, asset_classes, class_constraints,
                daily_rf, combo,
                transaction_cost_value, transaction_cost_type, trade_buffer_pct,
                use_michaud, n_boot, do_shrink_means, do_shrink_cov, reg_cov,
                do_ledoitwolf, do_ewm, ewm_alpha
            )
            future_to_combo[future] = combo

        for future in as_completed(future_to_combo):
            combo = future_to_combo[future]
            try:
                result_dict = future.result()
            except Exception as exc:
                # Debugging lines: Show error & traceback in Streamlit
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
            percent_complete = int(completed * 100 / total)
            elapsed = time.time() - start_time
            progress_text.text(
                f"Progress: {percent_complete}% complete. "
                f"Elapsed: {elapsed:.1f}s"
            )
            progress_bar.progress(percent_complete)

    total_elapsed = time.time() - start_time
    progress_text.text(f"Grid search done in {total_elapsed:.1f}s.")
    return pd.DataFrame(df_out)
