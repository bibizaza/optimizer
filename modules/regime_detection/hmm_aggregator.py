"""
File: aggregator_hmm.py

Provides 'AggregatedHMMRegimeManager', which:
  1) Builds an aggregated daily return & rolling volatility 
     from all instruments in df_prices (optionally weighting them).
  2) Uses an HMM on that 2D feature ( [daily_ret, rolling_vol] ) 
     to detect a hidden regime.

Requires: 'hmmlearn' library.

Usage:
  aggregator = AggregatedHMMRegimeManager(
      n_states=3,
      window_vol=20,
      use_equal_weights=False,
      random_state=42
  )
  states, current_state = aggregator.fit_predict(df_sub, df_instruments)
"""

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    # raise or warn as needed

class AggregatedHMMRegimeManager:
    def __init__(
        self,
        n_states: int = 2,
        window_vol: int = 20,
        random_state: int = 42,
        use_equal_weights: bool = True
    ):
        """
        Parameters
        ----------
        n_states : int
            Number of hidden states in the HMM.
        window_vol : int
            Rolling window for daily volatility calculation.
        random_state : int
            Random seed for the HMM.
        use_equal_weights : bool
            If True, we equally weight instruments for the aggregated return.
            If False, we use each instrument's old weight in df_instruments["Weight_Old"] 
            (if that column is missing, we fallback to equal).
        """
        if not HMMLEARN_AVAILABLE:
            raise ImportError("Requires 'hmmlearn'. Install via 'pip install hmmlearn'.")
        self.n_states = n_states
        self.window_vol = window_vol
        self.random_state = random_state
        self.use_equal_weights = use_equal_weights

        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            random_state=random_state
        )
        self.fitted = False

    def _build_aggregate_series(
        self,
        df_prices: pd.DataFrame,
        df_instruments: pd.DataFrame
    ) -> pd.Series:
        """
        Return a single aggregated price series from df_prices, 
        using either equal weighting or df_instruments["Weight_Old"] as weights.
        Then we can compute daily returns & rolling vol on that single series.
        """
        n_instruments = df_prices.shape[1]

        # 1) If we don't have "Weight_Old" or we want equal => do equal weighting
        if (not self.use_equal_weights) and ("Weight_Old" in df_instruments.columns):
            # build a map {ticker => old_weight}
            w_map = {}
            for _, row in df_instruments.iterrows():
                tkr = row["#ID"]
                w_  = row["Weight_Old"]
                w_map[tkr] = w_
            # Ensure sum of weights is 1
            # if sum < 1 => leftover is unallocated => treat as 0 or adjust
            # if sum > 1 => we can re-normalize or just keep
            # For clarity, let's re-normalize:
            total_w = sum(w_map.values())
            if total_w < 1e-12:
                # fallback to equal
                weights = np.ones(n_instruments) / n_instruments
            else:
                weights = []
                for c in df_prices.columns:
                    w_ = w_map.get(c, 0.0)
                    weights.append(w_)
                weights = np.array(weights) / total_w
        else:
            # equal weighting
            weights = np.ones(n_instruments) / n_instruments

        # 2) build aggregated price as the sum_{i} of weight_i * price_i
        arr_prices = df_prices.values
        # be careful with NaN => fill or forward/back fill
        arr_prices = np.nan_to_num(arr_prices, nan=0.0)

        # aggregated = sum across columns (axis=1) of w_i * col_i
        # shape => (n_days, n_instruments) * (n_instruments,) => (n_days,)
        aggregated = arr_prices.dot(weights)

        sr_agg = pd.Series(aggregated, index=df_prices.index, name="AggregatePrice")
        # If we want, we can forward fill if there's any zero or missing
        sr_agg = sr_agg.replace(0.0, np.nan).fillna(method="ffill").fillna(method="bfill")
        return sr_agg

    def _prepare_features(self, sr_agg: pd.Series) -> np.ndarray:
        """
        For the aggregator series sr_agg, compute:
          daily_ret, rolling_vol
        Return shape (T,2).
        """
        ret = sr_agg.pct_change().fillna(0.0)
        vol = ret.rolling(self.window_vol).std().fillna(0.0)
        X = np.column_stack([ret.values, vol.values])
        return X

    def fit_predict(
        self,
        df_prices: pd.DataFrame,
        df_instruments: pd.DataFrame
    ):
        """
        1) Build aggregator
        2) Fit HMM
        3) Predict states, return states array & the last state
        """
        sr_agg = self._build_aggregate_series(df_prices, df_instruments)
        X = self._prepare_features(sr_agg)

        self.model.fit(X)
        self.fitted = True

        states = self.model.predict(X)
        current_state = states[-1]
        return states, current_state

    def update_fit(self, df_prices, df_instruments):
        """
        Re-fit or partial update if you want. 
        Currently just re-fit from scratch.
        """
        sr_agg = self._build_aggregate_series(df_prices, df_instruments)
        X = self._prepare_features(sr_agg)
        self.model.fit(X)
        self.fitted = True

    def predict_states(self, df_prices, df_instruments):
        """
        If model not fitted => fit first. Otherwise just predict.
        """
        if not self.fitted:
            return self.fit_predict(df_prices, df_instruments)
        sr_agg = self._build_aggregate_series(df_prices, df_instruments)
        X = self._prepare_features(sr_agg)
        states = self.model.predict(X)
        return states, states[-1]

    def print_state_stats(self):
        """
        Print means & cov of each state. 
        """
        if not self.fitted:
            print("HMM not fitted yet.")
            return
        means = self.model.means_
        covars= self.model.covars_
        for i in range(self.n_states):
            mu_ret, mu_vol = means[i]
            print(f"State {i} => mean_ret={mu_ret:.4f}, mean_vol={mu_vol:.4f}")
            print("   Cov:\n", covars[i])