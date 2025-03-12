# File: modules/ui/hyperparam_ui.py

import streamlit as st
import pandas as pd
from modules.backtesting.rolling_monthly import rolling_grid_search
from modules.backtesting.rolling_bayesian import rolling_bayesian_optimization

def show_hyperparam_ui(
    df_sub: pd.DataFrame,
    df_instruments: pd.DataFrame,
    asset_cls_list: list[str],
    sec_type_list: list[str],
    class_sum_constraints: dict,
    subtype_constraints: dict,
    daily_rf: float,
    transaction_cost_value: float,
    transaction_cost_type: str,
    trade_buffer_pct: float
):
    """
    Encapsulates all the Streamlit widgets & logic for
    Grid Search & Bayesian hyperparameter approaches.
    Called from optima_optimizer.py or elsewhere.
    """

    hyper_choice = st.radio("Hyperparameter Method:", ["Grid Search","Bayesian"], index=0)

    if hyper_choice == "Grid Search":
        st.subheader("Grid => param or direct Markowitz => pass use_direct if needed.")

        frontier_points_list= st.multiselect(
            "Frontier Points (n_points)", [5,10,15,20,30], default=[5,10,15]
        )
        alpha_list= st.multiselect("Alpha (means)", [0,0.1,0.2,0.3,0.4,0.5], [0,0.1,0.3])
        beta_list= st.multiselect("Beta (cov)", [0,0.1,0.2,0.3,0.4,0.5], [0.1,0.2])
        rebal_freq_list= st.multiselect("Rebalance freq (months)", [1,3,6], [1,3])
        lookback_list= st.multiselect("Lookback (months)", [3,6,12], [3,6])
        max_workers= st.number_input("Max Workers",1,64,4,step=1)
        use_direct_gs= st.checkbox("Use Direct Solver in Grid Search?", value=False)

        if st.button("Run Grid Search"):
            if (not frontier_points_list 
                or not alpha_list 
                or not beta_list
                or not rebal_freq_list 
                or not lookback_list):
                st.error("Select at least one param in each list.")
            else:
                # call rolling_grid_search
                df_gs= rolling_grid_search(
                    df_prices= df_sub,
                    df_instruments= df_instruments,
                    asset_cls_list= asset_cls_list,
                    sec_type_list= sec_type_list,
                    class_sum_constraints= class_sum_constraints,
                    subtype_constraints= subtype_constraints,
                    daily_rf= daily_rf,
                    frontier_points_list= frontier_points_list,
                    alpha_list= alpha_list,
                    beta_list= beta_list,
                    rebal_freq_list= rebal_freq_list,
                    lookback_list= lookback_list,
                    transaction_cost_value= transaction_cost_value,
                    transaction_cost_type= transaction_cost_type,
                    trade_buffer_pct= trade_buffer_pct,
                    use_michaud=False,
                    n_boot=10,
                    do_shrink_means=True,
                    do_shrink_cov=True,
                    reg_cov=False,
                    do_ledoitwolf=False,
                    do_ewm=False,
                    ewm_alpha=0.06,
                    max_workers= max_workers,
                    use_direct_solver= use_direct_gs
                )
                st.dataframe(df_gs)
                if "Sharpe Ratio" in df_gs.columns:
                    best_= df_gs.sort_values("Sharpe Ratio", ascending=False).head(5)
                    st.write("**Top 5 combos by Sharpe**")
                    st.dataframe(best_)

    else:
        # "Bayesian"
        st.subheader("Bayesian => param or direct Markowitz => ignoring n_points for direct.")
        df_bayes= rolling_bayesian_optimization(
            df_prices= df_sub,
            df_instruments= df_instruments,
            asset_cls_list= asset_cls_list,
            sec_type_list= sec_type_list,
            class_sum_constraints= class_sum_constraints,
            subtype_constraints= subtype_constraints,
            daily_rf= daily_rf,
            transaction_cost_value= transaction_cost_value,
            transaction_cost_type= transaction_cost_type,
            trade_buffer_pct= trade_buffer_pct
        )
        if not df_bayes.empty:
            st.write("Bayesian Search Results:")
            st.dataframe(df_bayes)