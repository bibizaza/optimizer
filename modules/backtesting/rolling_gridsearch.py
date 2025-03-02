# File: modules/backtesting/rolling_gridsearch.py

import pandas as pd
import numpy as np
import traceback
import streamlit as st

from modules.backtesting.rolling_monthly import (
    rolling_backtest_monthly_param_sharpe,
    rolling_backtest_monthly_direct_sharpe
)
# If you compute extra stats:
from modules.analytics.returns_cov import compute_performance_metrics

# Param solver & direct solver:
from modules.optimization.cvxpy_optimizer import (
    parametric_max_sharpe_aclass_subtype,
    direct_max_sharpe_aclass_subtype
)

###############################################################################
# run_one_combo(...) => the single worker
###############################################################################
def run_one_combo(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    asset_cls_list: list[str],
    sec_type_list: list[str],
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float,
    combo: tuple,
    transaction_cost_value: float,
    transaction_cost_type: str,
    trade_buffer_pct: float,
    use_michaud: bool,
    n_boot: int,
    do_shrink_means: bool,
    do_shrink_cov: bool,
    reg_cov: bool,
    do_ledoitwolf: bool,
    do_ewm: bool,
    ewm_alpha: float,
    use_direct_solver: bool = False  # <--- NEW param for direct vs param
) -> dict:
    """
    This function tries a single combo => (n_points, alpha, beta, freq, lb).
    We run monthly rolling => either param or direct approach, depending on
    'use_direct_solver'. Then we return final Sharpe, final weights, etc.

    No param lines removed—just added direct logic in 'if use_direct_solver'.
    """
    # Unpack combo
    n_pts, alpha_, beta_, freq_, lb_ = combo
    col_tickers = df_prices.columns.tolist()

    # Decide param approach vs direct approach
    if not use_direct_solver:
        # Param approach => define param_sharpe_fn => pass to rolling_backtest_monthly_param_sharpe
        def param_sharpe_fn(sub_ret: pd.DataFrame):
            # We do not remove your param constraints => scanning target returns internally
            w_opt, _ = parametric_max_sharpe_aclass_subtype(
                df_returns=sub_ret,
                tickers=col_tickers,
                asset_classes=asset_cls_list,
                security_types=sec_type_list,
                class_constraints=class_sum_constraints,
                subtype_constraints=subtype_constraints,
                daily_rf=daily_rf,
                no_short=True,
                n_points=n_pts,  # <--- param approach uses n_pts
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

        sr_line, final_w, _, _, _, ext_metrics = rolling_backtest_monthly_param_sharpe(
            df_prices=df_prices,
            df_instruments=df_instruments,
            param_sharpe_fn=param_sharpe_fn,
            start_date=df_prices.index[0],
            end_date=df_prices.index[-1],
            months_interval=freq_,
            window_days=lb_ * 21,
            transaction_cost_value=transaction_cost_value,
            transaction_cost_type=transaction_cost_type,
            trade_buffer_pct=trade_buffer_pct,
            daily_rf=daily_rf
        )

    else:
        # Direct approach => ignore n_pts => define direct_sharpe_fn => pass to rolling_backtest_monthly_direct_sharpe
        def direct_sharpe_fn(sub_ret: pd.DataFrame):
            w_opt, _ = direct_max_sharpe_aclass_subtype(
                df_returns=sub_ret,
                tickers=col_tickers,
                asset_classes=asset_cls_list,
                security_types=sec_type_list,
                class_constraints=class_sum_constraints,
                subtype_constraints=subtype_constraints,
                daily_rf=daily_rf,
                no_short=True,
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

        sr_line, final_w, _, _, _, ext_metrics = rolling_backtest_monthly_direct_sharpe(
            df_prices=df_prices,
            df_instruments=df_instruments,
            direct_sharpe_fn=direct_sharpe_fn,
            start_date=df_prices.index[0],
            end_date=df_prices.index[-1],
            months_interval=freq_,
            window_days=lb_ * 21,
            transaction_cost_value=transaction_cost_value,
            transaction_cost_type=transaction_cost_type,
            trade_buffer_pct=trade_buffer_pct,
            daily_rf=daily_rf
        )

    # If we got no results => meltdown
    if len(sr_line) < 2 or (not ext_metrics):
        return {
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

    # Extract final stats
    sharpe_val = ext_metrics.get("Sharpe", 0.0)
    ann_ret = ext_metrics.get("Annual Return", 0.0)
    ann_vol = ext_metrics.get("Annual Vol", 0.0)

    return {
        "n_points": n_pts,
        "alpha_mean": alpha_,
        "beta_cov": beta_,
        "rebal_freq": freq_,
        "lookback_m": lb_,
        "Sharpe Ratio": sharpe_val,
        "Annual Ret": ann_ret,
        "Annual Vol": ann_vol,
        "final_weights": final_w
    }


###############################################################################
# rolling_grid_search(...) => parallel or single-thread
###############################################################################
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
    max_workers: int = 4,
    use_direct_solver: bool = False  # <-- optional param if you want direct approach
) -> pd.DataFrame:
    """
    A standard grid search. We'll build combos => call run_one_combo(...) for each.

    If 'use_direct_solver' is True => run_one_combo calls direct approach,
    else param approach => 'n_points' is used. 
    """
    import concurrent.futures
    import time

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

    # We'll do parallel
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
                ewm_alpha,
                use_direct_solver  # pass it down
            )
            future_to_combo[future] = combo

        for future in concurrent.futures.as_completed(future_to_combo):
            combo_ = future_to_combo[future]
            try:
                result_dict = future.result()
            except Exception as exc:
                st.error(f"Error for combo {combo_}: {exc}")
                st.text(traceback.format_exc())
                # fallback
                n_pts, alpha_, beta_, freq_, lb_ = combo_
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


###############################################################################
# Additional lines if your original code has advanced logging, debug prints, etc.
###############################################################################
# ... Place your extra code here ...