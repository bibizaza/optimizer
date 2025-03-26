# modules/excel_exports/export_backtest_to_excel.py

import io
import pandas as pd

def export_backtest_results_to_excel(
    sr_line_new: pd.Series,
    sr_line_old: pd.Series,
    df_rebal: pd.DataFrame,
    ext_metrics_new: dict,
    ext_metrics_old: dict,
    final_w_array=None,
    old_w_array=None,
    tickers=None
) -> bytes:
    """
    Creates an Excel file in-memory with multiple sheets:
      1) The new portfolio time series (sr_line_new).
      2) The old portfolio time series (sr_line_old).
      3) Rebalancing details (df_rebal).
      4) Extended metrics for old & new.
      5) Weights sheet with [Ticker | OldWeight | NewWeight], if arrays + tickers given.
    """

    output = io.BytesIO()

    with pd.ExcelWriter(output, engine="xlsxwriter") as writer:
        # 1) New portfolio line
        if sr_line_new is not None:
            df_new = sr_line_new.to_frame("New_Ptf_Value")
            df_new.to_excel(writer, sheet_name="NewPortfolio", index=True)

        # 2) Old portfolio line
        if sr_line_old is not None:
            df_old = sr_line_old.to_frame("Old_Ptf_Value")
            df_old.to_excel(writer, sheet_name="OldPortfolio", index=True)

        # 3) Rebalancing details
        if df_rebal is not None and not df_rebal.empty:
            df_rebal.to_excel(writer, sheet_name="RebalLog", index=False)

        # 4) Extended metrics (old vs new)
        if ext_metrics_new and ext_metrics_old:
            df_metrics_new = pd.DataFrame(list(ext_metrics_new.items()), columns=["Metric","Value_New"])
            df_metrics_old = pd.DataFrame(list(ext_metrics_old.items()), columns=["Metric","Value_Old"])
            df_metrics_merged = pd.merge(df_metrics_old, df_metrics_new, on="Metric", how="outer")
            df_metrics_merged.to_excel(writer, sheet_name="Metrics", index=False)

        # 5) If user passed final/old arrays => make a Weights sheet
        if (final_w_array is not None) and (old_w_array is not None) and (tickers is not None):
            # convert them to Series => combine
            s_old = pd.Series(old_w_array, index=tickers, name="OldWeight")
            s_new = pd.Series(final_w_array, index=tickers, name="NewWeight")
            df_w = pd.concat([s_old, s_new], axis=1)
            df_w.to_excel(writer, sheet_name="Weights", index=True)

    output.seek(0)
    return output.getvalue()
