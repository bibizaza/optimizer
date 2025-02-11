# modules/backtesting/single_period_cumret.py

import pandas as pd
import numpy as np

def single_period_cumret_compare(
    df_instruments: pd.DataFrame,
    df_prices: pd.DataFrame,
    new_weights: np.ndarray,
    transaction_cost_value: float = 0.0,
    transaction_cost_type: str = "percentage",  # or "ticket_fee"
    missing_instrument_mode: str = "partial-liveliness",  # or "equal_distribution"
    debug: bool = False
) -> pd.DataFrame:
    """
    Single-Period backtest => "Old_Ptf" and "New_Ptf".

    - Fill missing data with forward+back fill.
    - "Old_Ptf": no rebal, #Quantity * daily price.
    - "New_Ptf": invests day0, skipping or not skipping instruments with day0 price<=0 
      depending on missing_instrument_mode.
    - Transaction cost can be "percentage" or "ticket_fee".
    """

    df_prices = df_prices.sort_index()
    df_prices = df_prices.fillna(method="ffill").fillna(method="bfill")

    if debug:
        print("[DEBUG single_period_cumret_compare] => shape=", df_prices.shape)

    df_instruments["Value"] = df_instruments["#Quantity"] * df_instruments["#Last_Price"]
    old_val_current = df_instruments["Value"].sum()

    # build old quantity map
    old_quantities= {}
    for col in df_prices.columns:
        row= df_instruments[df_instruments["#ID"]== col]
        if not row.empty:
            old_quantities[col]= row["#Quantity"].iloc[0]
        else:
            old_quantities[col]= 0.0

    # daily old
    old_vals= []
    for _, rowp in df_prices.iterrows():
        val=0.0
        for tkr in df_prices.columns:
            val += old_quantities[tkr]* rowp[tkr]
        old_vals.append(val)
    series_old= pd.Series(old_vals, index=df_prices.index, name="Old_Ptf")

    # new => day0 invests
    day0_prices= df_prices.iloc[0].values
    valid_mask= (day0_prices>0)
    if missing_instrument_mode=="partial-liveliness":
        sum_valid= np.sum(new_weights[valid_mask])
        if sum_valid<=0:
            raise ValueError("[DEBUG single] => no valid instrument day0 => can't invest new.")
        new_w_rescaled= new_weights.copy()
        new_w_rescaled[~valid_mask]=0.0
        new_w_rescaled[valid_mask]= new_w_rescaled[valid_mask]/ sum_valid
    else:
        # "equal_distribution" => do NOT re-scale => that fraction invests in a 0-price instrument => worthless
        # we keep new_weights as is
        new_w_rescaled= new_weights.copy()

    # transaction cost => depends on transaction_cost_type
    if transaction_cost_type=="percentage":
        # turnover ~ sum(|new_w_rescaled|)=1 if sum(new_w)=1
        turnover= np.sum(np.abs(new_w_rescaled)) # ~1
        cost= old_val_current* turnover* transaction_cost_value
    else:
        # "ticket_fee" => number_of_instruments_traded => count how many new_w_rescaled>0
        instruments_traded= (new_w_rescaled>0).sum()
        cost= instruments_traded* transaction_cost_value

    effective_cap= old_val_current- cost
    if effective_cap<=0:
        raise ValueError("[DEBUG single] => transaction cost kills all capital => can't invest")

    # allocate shares
    new_shares= []
    for i,w in enumerate(new_w_rescaled):
        price0= day0_prices[i]
        if w>0 and price0>0:
            money_alloc= w* effective_cap
            s= money_alloc/ price0
        else:
            s=0.0
        new_shares.append(s)
    new_shares= np.array(new_shares)

    # daily new => sum( new_shares * daily price)
    new_vals= []
    for _,rowp in df_prices.iterrows():
        val_new= np.sum(new_shares* rowp.values)
        new_vals.append(val_new)
    series_new= pd.Series(new_vals, index=df_prices.index, name="New_Ptf")

    df_compare= pd.concat([series_old, series_new], axis=1)
    if debug:
        print("[DEBUG single_period_cumret_compare => HEAD]", df_compare.head(5))
    return df_compare


def compute_performance_metrics(series_values: pd.Series) -> dict:
    series_values= series_values.dropna()
    if len(series_values)<2:
        return {
            "Total Return": 0.0,
            "Annualized Return":0.0,
            "Annualized Vol":0.0,
            "Sharpe":0.0,
            "Max Drawdown":0.0,
            "1M VaR 99%":0.0
        }

    daily_ret= series_values.pct_change().dropna()
    n_days= len(daily_ret)
    total_return= (series_values.iloc[-1]/ series_values.iloc[0]) -1
    annual_return= (1+ total_return)**(252/n_days) -1
    annual_vol= daily_ret.std()* np.sqrt(252)
    sharpe= 0.0
    if annual_vol>0:
        sharpe= annual_return/ annual_vol
    run_max= series_values.cummax()
    dd_series= (series_values/run_max)-1
    max_dd= dd_series.min()

    var_daily= daily_ret.quantile(0.01)
    var_month= var_daily* np.sqrt(21)
    if var_month<0:
        var_month= -var_month

    return {
        "Total Return": total_return,
        "Annualized Return": annual_return,
        "Annualized Vol": annual_vol,
        "Sharpe": sharpe,
        "Max Drawdown": max_dd,
        "1M VaR 99%": var_month
    }
