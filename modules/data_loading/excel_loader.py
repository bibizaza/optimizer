# modules/data_loading/excel_loader.py

import pandas as pd

def parse_excel(
    file_path: str,
    streamlit_sheet: str = "streamlit",
    histo_sheet: str = "Histo_Price"
) -> (pd.DataFrame, pd.DataFrame):
    """
    Reads the Excel file and returns:
      df_instruments (from 'streamlit') + df_prices (from 'Histo_Price').
    Now includes debugging checks for df_prices.
    """

    # 1) Read 'streamlit'
    df_instruments = pd.read_excel(file_path, sheet_name=streamlit_sheet, header=0)

    # 2) Read 'Histo_Price'
    df_prices_raw = pd.read_excel(file_path, sheet_name=histo_sheet, header=0)

    # Ensure first column is "Date"
    if df_prices_raw.columns[0] != "Date":
        df_prices_raw.rename(columns={df_prices_raw.columns[0]: "Date"}, inplace=True)

    # Convert 'Date' to datetime, drop invalid
    df_prices_raw["Date"] = pd.to_datetime(df_prices_raw["Date"], errors="coerce")
    df_prices_raw.dropna(subset=["Date"], inplace=True)
    df_prices_raw.set_index("Date", inplace=True)
    df_prices_raw.sort_index(inplace=True)

    # Debug: Print a summary of columns & dtypes
    print("\n[DEBUG] df_prices_raw columns:", df_prices_raw.columns)
    print("[DEBUG] df_prices_raw dtypes:\n", df_prices_raw.dtypes)
    print("[DEBUG] First few rows of df_prices_raw:\n", df_prices_raw.head())

    # 3) Force numeric conversion for all columns
    #    Any non-numeric entries become NaN
    df_prices_raw = df_prices_raw.apply(pd.to_numeric, errors="coerce")

    # If any column is entirely NaN, we might drop it
    for col in df_prices_raw.columns:
        if df_prices_raw[col].isna().all():
            print(f"[DEBUG] Dropping column '{col}' entirely NaN after numeric conversion.")
            df_prices_raw.drop(columns=[col], inplace=True)

    # Debug: check for negative or zero prices
    # If your data can have 0 or negative validly, skip or handle differently
    zero_or_neg = (df_prices_raw <= 0).any().any()
    if zero_or_neg:
        print("[WARNING] Some prices are zero or negative, which will break log returns!")
        # Optionally, you could remove them or handle in a custom way here.

    return df_instruments, df_prices_raw
