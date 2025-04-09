# File: modules/optimization/utils/mean_shrink.py

import numpy as np

def shrink_mean_to_grand_mean(mean_vector: np.ndarray, alpha: float) -> np.ndarray:
    """
    Already existing in your code:
      mean_vector => shape (N,)
      alpha => blend fraction
      final_means = (1 - alpha)*mean_vector + alpha*grand
    """
    grand = np.mean(mean_vector)
    return (1 - alpha)*mean_vector + alpha*grand

def shrink_mean_to_zero(mean_vector: np.ndarray, alpha: float) -> np.ndarray:
    """
    NEW function => shrink each asset's mean to 0 by fraction alpha:
      final = (1 - alpha)*mean_vector + alpha*0
             = (1 - alpha)*mean_vector
    """
    # alpha=0 => no change
    # alpha=1 => all means => 0
    return (1.0 - alpha)*mean_vector
