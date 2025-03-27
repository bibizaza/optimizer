"""
File: hmm_regime.py

Minimal HMM-based regime detection using hmmlearn.
Focus: uses daily returns & rolling volatility as features.

Usage:
------
from modules.regime_detection.hmm_regime import HMMRegimeDetector

hmm_detector = HMMRegimeDetector(n_states=2, window_vol=20, random_state=42)

# df_prices is a DataFrame with at least one column, e.g. "Close" or "S&P500"
# or any reference series you consider "market" for regime detection.
# Suppose you want to detect regime from the 'SP500' column.

states, current_state = hmm_detector.fit_predict(df_prices["SP500"])

print("Last predicted regime state:", current_state)

# If you'd like to see the means for each hidden state (for interpretation):
hmm_detector.print_state_stats()
"""

import numpy as np
import pandas as pd

try:
    from hmmlearn.hmm import GaussianHMM
    HMMLEARN_AVAILABLE = True
except ImportError:
    HMMLEARN_AVAILABLE = False
    # If hmmlearn is not installed, you might raise an error or warn.

class HMMRegimeDetector:
    """
    A minimal class for detecting market regimes using an HMM
    with daily returns & rolling volatility as features.
    """

    def __init__(self, n_states=2, window_vol=20, random_state=42):
        """
        Parameters
        ----------
        n_states : int
            Number of hidden states in the HMM (e.g., 2 for bull/bear or
            low/high-vol).
        window_vol : int
            Rolling window (in days) to compute volatility.
        random_state : int
            Seed for reproducible results.
        """
        if not HMMLEARN_AVAILABLE:
            raise ImportError("The 'hmmlearn' library is required for HMMRegimeDetector. "
                              "Please install it via 'pip install hmmlearn'.")

        self.n_states = n_states
        self.window_vol = window_vol
        self.random_state = random_state
        self.model = GaussianHMM(
            n_components=n_states,
            covariance_type="full",
            random_state=random_state
        )
        self.fitted = False

    def _prepare_features(self, series_prices: pd.Series) -> np.ndarray:
        """
        Build the (ret, vol) feature array from a single price series.

        Returns
        -------
        X : np.ndarray of shape (T, 2)
            Each row = [daily_return, rolling_volatility].
        """
        # Daily returns
        ret = series_prices.pct_change().fillna(0.0)

        # Rolling volatility
        vol = ret.rolling(self.window_vol).std().fillna(0.0)

        # Convert to np array
        # In practice, you might want to drop the first 'window_vol-1' rows
        # to avoid artificially low vol. We'll keep them for simplicity.
        X = np.column_stack([ret.values, vol.values])

        return X

    def fit_predict(self, series_prices: pd.Series):
        """
        Fit the HMM on the given price series and predict the regime
        for each time step.

        Parameters
        ----------
        series_prices : pd.Series
            A timeseries of prices. Index is optional but recommended.

        Returns
        -------
        states : np.ndarray of shape (T,)
            The predicted hidden state label for each time point.
        current_state : int
            The last predicted state (i.e., the "current" regime).
        """
        X = self._prepare_features(series_prices)

        # Fit the HMM
        self.model.fit(X)
        self.fitted = True

        # Predict states
        states = self.model.predict(X)
        # The last element => current regime
        current_state = states[-1]

        return states, current_state

    def update_fit(self, series_prices: pd.Series):
        """
        If you'd like to update the fit on new data. 
        (In the simplest approach, you just re-fit from scratch.)
        """
        X = self._prepare_features(series_prices)
        self.model.fit(X)
        self.fitted = True

    def predict_states(self, series_prices: pd.Series):
        """
        Predict states on a new price series (assuming model is already fitted).
        If not fitted, will fit first. 
        """
        if not self.fitted:
            return self.fit_predict(series_prices)

        X = self._prepare_features(series_prices)
        states = self.model.predict(X)
        return states, states[-1]

    def print_state_stats(self):
        """
        Prints the means/covariances for each hidden state for interpretability.
        e.g. If state 0 => (ret ~0.001, vol ~0.02), you can guess it might be
        a 'calm' regime, etc.
        """
        if not self.fitted:
            print("HMM not fitted yet.")
            return

        means = self.model.means_  # shape (n_states, 2)
        covars = self.model.covars_  # shape (n_states, 2, 2)
        for i in range(self.n_states):
            mu_ret, mu_vol = means[i]
            print(f"State {i} => mean_ret={mu_ret:.4f}, mean_vol={mu_vol:.4f}")
            print(f"   Covariance:\n{covars[i]}")
