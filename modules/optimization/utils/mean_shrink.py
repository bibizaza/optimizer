# modules/optimization/utils/mean_shrink.py

import numpy as np

def shrink_mean_to_grand_mean(raw_means: np.ndarray, alpha: float=0.3) -> np.ndarray:
    """
    Shrink each asset's mean toward the cross-sectional grand mean.
    raw_means: shape (N,) - daily mean returns for each asset
    alpha: fraction in [0,1]. 0 => no shrink, 1 => all means = grand_mean
    """
    grand_mean = np.mean(raw_means)
    return (1 - alpha)*raw_means + alpha*grand_mean
