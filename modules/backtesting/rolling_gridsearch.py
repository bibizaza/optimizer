# modules/backtesting/rolling_gridsearch.py

import pandas as pd
import numpy as np
from modules.backtesting.rolling_monthly import rolling_backtest_monthly_param_sharpe
from modules.analytics.returns_cov import compute_performance_metrics

# If you have Michaud in a separate file:
# from modules.optimization.utils.michaud import parametric_max_sharpe_michaud
from modules.optimization.cvxpy_optimizer import parametric_max_sharpe_aclass_subtype

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
    ewm_alpha: float
) -> dict:
    """
    This is the single-combo worker called by rolling_grid_search.
    combo => (n_pts, alpha_, beta_, freq_, lb_).
    Returns a dict with performance metrics & final weights.
    """
    n_pts, alpha_, beta_, freq_, lb_ = combo
    col_tickers = df_prices.columns.tolist()

    def param_sharpe_fn(sub_ret: pd.DataFrame):
        # If you want to handle Michaud vs standard:
        if use_michaud:
            from modules.optimization.utils.michaud import parametric_max_sharpe_michaud
            w_opt, _ = parametric_max_sharpe_michaud(
                df_returns=sub_ret,
                tickers=col_tickers,
                asset_classes=asset_cls_list,
                # Possibly add security_types if your michaud variant requires it
                class_constraints=class_sum_constraints,
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
            # Standard approach with security-type constraints
            w_opt, _ = parametric_max_sharpe_aclass_subtype(
                df_returns=sub_ret,
                tickers=col_tickers,
                asset_classes=asset_cls_list,
                security_types=sec_type_list,
                class_constraints=class_sum_constraints,
                subtype_constraints=subtype_constraints,
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

    # NOTE: rolling_backtest_monthly_param_sharpe returns 5 items:
    # (sr_norm, last_w_final, final_old_w_last, final_rebal_date, df_rebal)
    sr_line, final_w, _, _, _ = rolling_backtest_monthly_param_sharpe(
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

    # Now compute performance
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
        "n_points": n_pts,
        "alpha_mean": alpha_,
        "beta_cov": beta_,
        "rebal_freq": freq_,
        "lookback_m": lb_,
        "Sharpe Ratio": sr_val,
        "Annual Ret": ann_ret,
        "Annual Vol": ann_vol,
        "final_weights": final_w
    }
