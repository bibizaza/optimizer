# modules/data_loading/cleaner.py

import pandas as pd

def clean_df_prices(df_prices: pd.DataFrame, min_coverage: float=0.8) -> pd.DataFrame:
    """
    1) For each date (row), check how many columns are non-NaN.
       If fraction < min_coverage => we drop that row entirely.
    2) After dropping, we forward-fill partial columns to fill isolated NaNs.

    min_coverage=0.8 => we require at least 80% non-NaN columns that day.
    """
    df_prices = df_prices.copy()
    # step1 => drop rows with coverage < min_coverage
    n_cols= df_prices.shape[1]
    coverage_count= df_prices.notna().sum(axis=1)  # how many non-NaN on each row
    threshold= min_coverage* n_cols
    row_mask= (coverage_count>= threshold)
    df_prices= df_prices[row_mask]

    # step2 => forward fill => backward fill
    df_prices= df_prices.sort_index()
    df_prices= df_prices.fillna(method="ffill").fillna(method="bfill")
    return df_prices
