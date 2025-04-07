# modules/analytics/risk_contributions.py

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px

def compute_empirical_var_cvar(returns: np.ndarray, alpha: float) -> tuple[float,float]:
    """
    Compute empirical (historical) VaR and CVaR for a 1D array of returns.

    returns: shape (N_days,)
    alpha: e.g. 0.95

    returns: (var_value, cvar_value)
       var_value < 0 => e.g. -0.03 means a 3% loss VaR at 95% level
       cvar_value < 0 => average of worst (1-alpha) tail
    """
    if len(returns) < 2:
        return (0.0, 0.0)

    sorted_rets = np.sort(returns)  # ascending
    n = len(sorted_rets)
    tail_size = int((1.0 - alpha) * n)  # e.g. 5% tail if alpha=0.95
    tail_size = max(min(tail_size, n-1), 0)

    var_ = sorted_rets[tail_size]
    tail_values = sorted_rets[:tail_size+1]  # everything below or at var_
    if len(tail_values) > 0:
        cvar_ = tail_values.mean()
    else:
        cvar_ = var_
    return (var_, cvar_)

def compute_portfolio_returns(df_returns: pd.DataFrame, weights: np.ndarray) -> np.ndarray:
    """
    df_returns: shape (N_days, n_assets)
    weights: shape (n_assets,)

    returns a 1D array of length N_days = daily portfolio returns
    """
    return df_returns.values @ weights

def marginal_risk_contributions(
    df_returns: pd.DataFrame,
    weights: np.ndarray,
    alpha: float,
    risk_measure: str = "cvar",
    bump_size: float = 1e-4,
    do_renormalize: bool = True
) -> tuple[np.ndarray, float]:
    """
    Estimate each asset's marginal contribution to VaR or CVaR by
    bumping the weight of each asset slightly and measuring the
    difference in portfolio risk.

    - risk_measure in {"var", "cvar"}
    - bump_size default=1e-4 => 0.01% weight shift
    - do_renormalize => if True, after bump we scale all weights so sum=1.0

    returns: (marginal_contribs, portfolio_risk)
      marginal_contribs => shape (n_assets,)
      portfolio_risk => float (the baseline portfolio VaR or CVaR)
    """
    base_pret = compute_portfolio_returns(df_returns, weights)
    base_var, base_cvar = compute_empirical_var_cvar(base_pret, alpha)
    base_risk = base_cvar if risk_measure=="cvar" else base_var

    n_assets = len(weights)
    mrc = np.zeros(n_assets)

    for i in range(n_assets):
        w_bumped = weights.copy()
        w_bumped[i] += bump_size
        if do_renormalize:
            s_ = w_bumped.sum()
            if s_ > 1e-15:
                w_bumped /= s_

        new_pret = compute_portfolio_returns(df_returns, w_bumped)
        new_var, new_cvar = compute_empirical_var_cvar(new_pret, alpha)
        new_risk = new_cvar if risk_measure=="cvar" else new_var

        # difference
        diff = (new_risk - base_risk) / bump_size
        mrc[i] = diff

    return mrc, base_risk

def build_contribution_table(
    df_returns: pd.DataFrame,
    col_tickers: list[str],
    weights: np.ndarray,
    alpha: float = 0.95,
    measure_type: str = "cvar"
) -> pd.DataFrame:
    """
    Build a DataFrame with columns:
      Ticker, Weight, Marginal_XX, Component_XX, Share_XX
    where XX in {VaR, CVaR}, depending on measure_type.

    measure_type => "var" or "cvar"
    """

    # 1) get marginal contributions
    mrc, port_risk = marginal_risk_contributions(
        df_returns=df_returns,
        weights=weights,
        alpha=alpha,
        risk_measure=measure_type
    )

    # 2) component = weight[i] * mrc[i]
    comp = weights * mrc

    # 3) Because VaR or CVaR is typically negative => port_risk < 0
    # we get the share by comp / port_risk. If port_risk=0 => fallback
    share = np.zeros_like(comp)
    if abs(port_risk) > 1e-15:
        share = comp / port_risk  # fraction

    # 4) Build DataFrame
    label = measure_type.upper()
    df_out = pd.DataFrame({
        "Ticker": col_tickers,
        "Weight": weights,
        f"Marginal_{label}": mrc,
        f"Component_{label}": comp,
        f"Share_{label}(%)": share * 100.0
    })
    # 5) Sort by highest to lowest component
    df_out.sort_values(by=f"Component_{label}", ascending=True, inplace=True)
    # Because it might be negative => if you want "largest magnitude" you might do ascending=False
    # but let's assume we want largest negative at bottom => set ascending=True if negative

    # 6) Return
    return df_out, port_risk

def display_risk_contributions(
    df_returns: pd.DataFrame,
    col_tickers: list[str],
    weights: np.ndarray,
    alpha: float = 0.95,
    show_var: bool = True,
    show_cvar: bool = True
):
    """
    High-level function to build & display risk contributions for
    VaR and/or CVaR at confidence alpha.

    1) Build & display VaR table
    2) Build & display CVaR table
    3) Plot bar charts for each

    The user can choose whether to show VaR, CVaR, or both.
    """
    st.subheader(f"Risk Contributions (alpha={alpha*100:.1f}%)")

    if show_var:
        df_var, port_var = build_contribution_table(
            df_returns, col_tickers, weights, alpha, measure_type="var"
        )
        st.write(f"**Portfolio VaR** = {port_var:.4f} (typically a negative # => e.g. -0.03 => 3% loss)")
        st.dataframe(
            df_var.style.format({
                "Weight": "{:.2%}",
                "Marginal_VAR": "{:.4f}",
                "Component_VAR": "{:.4f}",
                "Share_VAR(%)": "{:.2f}"
            })
        )
        # Plot bar => component
        fig_var = px.bar(
            df_var,
            x="Ticker",
            y="Component_VAR",
            hover_data=["Weight", "Marginal_VAR", "Share_VAR(%)"],
            title="VaR Component Contributions",
            labels={"Component_VAR": "VaR Contribution"}
        )
        fig_var.add_hrect(y0=0, y1=0, line_width=1, line_dash="dot", line_color="black")
        st.plotly_chart(fig_var)

    if show_cvar:
        df_cvar, port_cvar = build_contribution_table(
            df_returns, col_tickers, weights, alpha, measure_type="cvar"
        )
        st.write(f"**Portfolio CVaR** = {port_cvar:.4f} (negative => e.g. -0.05 => 5% tail loss)")
        st.dataframe(
            df_cvar.style.format({
                "Weight": "{:.2%}",
                "Marginal_CVAR": "{:.4f}",
                "Component_CVAR": "{:.4f}",
                "Share_CVAR(%)": "{:.2f}"
            })
        )
        fig_cvar = px.bar(
            df_cvar,
            x="Ticker",
            y="Component_CVAR",
            hover_data=["Weight", "Marginal_CVAR", "Share_CVAR(%)"],
            title="CVaR Component Contributions",
            labels={"Component_CVAR": "CVaR Contribution"}
        )
        fig_cvar.add_hrect(y0=0, y1=0, line_width=1, line_dash="dot", line_color="black")
        st.plotly_chart(fig_cvar)
