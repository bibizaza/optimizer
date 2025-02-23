# File: modules/backtesting/rolling_gridsearch.py

import pandas as pd
import numpy as np
import streamlit as st
import traceback
import concurrent.futures

from modules.backtesting.rolling_monthly import rolling_backtest_monthly_param_sharpe
from modules.analytics.returns_cov import compute_performance_metrics
from modules.optimization.cvxpy_optimizer import parametric_max_sharpe_aclass_subtype
# If you have a Michaud solver, import it here if needed
# from modules.optimization.utils.michaud import parametric_max_sharpe_michaud

def run_one_combo(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    asset_cls_list: list[str],
    sec_type_list: list[str],
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float,
    combo: tuple,  # e.g. (alpha_, beta_, freq_, lb_)
    transaction_cost_value: float,
    transaction_cost_type: str,
    trade_buffer_pct: float,
    use_michaud: bool,
    do_shrink_means: bool,
    do_shrink_cov: bool,
    reg_cov: bool,
    do_ledoitwolf: bool,
    do_ewm: bool,
    ewm_alpha: float,
    n_boot: int = 10
) -> dict:
    """
    Single-combo worker:
     combo = (alpha_, beta_, freq_, lb_).
    1) Builds a param_sharpe_fn that calls parametric_max_sharpe_aclass_subtype
       (or Michaud, if use_michaud=True).
    2) Runs rolling_backtest_monthly_param_sharpe.
    3) Returns performance metrics in a dict.
    """

    alpha_, beta_, freq_, lb_ = combo
    col_tickers = df_prices.columns.tolist()

    def param_sharpe_fn(sub_ret: pd.DataFrame):
        """Build and call the chosen optimizer (Riskfolio or Michaud)."""
        if use_michaud:
            from modules.optimization.utils.michaud import parametric_max_sharpe_michaud
            w_opt, _ = parametric_max_sharpe_michaud(
                df_returns=sub_ret,
                tickers=col_tickers,
                asset_classes=asset_cls_list,
                # security_types if needed
                class_constraints=class_sum_constraints,
                daily_rf=daily_rf,
                no_short=True,
                # The old 'n_points' is removed
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
            w_opt, _ = parametric_max_sharpe_aclass_subtype(
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

    # Convert months to days
    window_days = lb_ * 21

    # Rolling backtest with this param_sharpe_fn
    sr_line, final_w, _, _, _, _ = rolling_backtest_monthly_param_sharpe(
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

    # Compute performance metrics if enough data
    if len(sr_line) > 1:
        perf = compute_performance_metrics(sr_line, daily_rf=daily_rf)
        sr_val = perf["Sharpe Ratio"]
        ann_ret = perf["Annualized Return"]
        ann_vol = perf["Annualized Volatility"]
    else:
        sr_val = 0.0
        ann_ret = 0.0
        ann_vol = 0.0

    return {
        "alpha_mean": alpha_,
        "beta_cov": beta_,
        "rebal_freq": freq_,
        "lookback_m": lb_,
        "Sharpe Ratio": sr_val,
        "Annual Ret": ann_ret,
        "Annual Vol": ann_vol,
        "final_weights": final_w
    }


def rolling_grid_search(
    df_prices: pd.DataFrame,
    df_instruments: pd.DataFrame,
    asset_cls_list: list[str],
    sec_type_list: list[str],
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float,
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
    A parallel grid search enumerating alpha_, beta_, freq_, lb_ combos.
    For each combo:
      - run_one_combo(...) => rolling backtest => compute metrics => store in df.
    """

    combos = []
    for alpha_ in alpha_list:
        for beta_ in beta_list:
            for freq_ in rebal_freq_list:
                for lb_ in lookback_list:
                    combos.append((alpha_, beta_, freq_, lb_))

    total = len(combos)
    df_out = []

    st.write(f"Total combos: {total}")
    progress_text = st.empty()
    progress_bar = st.progress(0)
    completed = 0

    def update_progress():
        pct = int(completed * 100 / total)
        progress_text.text(f"Grid Search: {pct}% done")
        progress_bar.progress(pct)

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
                do_shrink_means,
                do_shrink_cov,
                reg_cov,
                do_ledoitwolf,
                do_ewm,
                ewm_alpha,
                n_boot
            )
            future_to_combo[future] = combo

        for future in concurrent.futures.as_completed(future_to_combo):
            combo = future_to_combo[future]
            try:
                result_dict = future.result()
            except Exception as exc:
                st.error(f"Error for combo {combo}: {exc}")
                st.text(traceback.format_exc())
                alpha_, beta_, freq_, lb_ = combo
                result_dict = {
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
            update_progress()

    return pd.DataFrame(df_out)