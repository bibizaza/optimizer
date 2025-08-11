"""
Functions to perform mean shrinkage on return vectors.

Portfolio optimization algorithms often benefit from regularising
expected return estimates. These helper functions implement two
simple shrinkage schemes: shrinking towards the grand mean of the
vector and shrinking towards zero. They are written to operate on
both NumPy and CuPy arrays via the backend abstraction.

Usage
-----
>>> from modules.optimization.utils.mean_shrink import (
...     shrink_mean_to_grand_mean, shrink_mean_to_zero
... )
>>> import numpy as np
>>> x = np.array([0.05, 0.07, 0.03])
>>> shrink_mean_to_grand_mean(x, 0.5)
array([0.05 , 0.05 , 0.05 ])
"""

from .backend import xp

def shrink_mean_to_grand_mean(mean_vec, alpha):
    """Shrink a mean vector towards its grand mean.

    Parameters
    ----------
    mean_vec : array-like
        Vector of mean estimates. Can be a NumPy or CuPy array.
    alpha : float
        Shrinkage intensity in the interval [0, 1]. An ``alpha`` of 0
        returns the original mean vector; an ``alpha`` of 1 replaces
        every element with the grand mean.

    Returns
    -------
    array-like
        The shrunk mean vector, with the same type (NumPy or CuPy) as
        the input.
    """
    if alpha <= 0:
        return mean_vec
    # Compute the grand mean along the first axis
    grand_mean = xp.mean(mean_vec)
    return (1.0 - alpha) * mean_vec + alpha * grand_mean


def shrink_mean_to_zero(mean_vec, alpha):
    """Shrink a mean vector towards zero.

    Parameters
    ----------
    mean_vec : array-like
        Vector of mean estimates. Can be a NumPy or CuPy array.
    alpha : float
        Shrinkage intensity in the interval [0, 1]. An ``alpha`` of 0
        returns the original mean vector; an ``alpha`` of 1 returns a
        zero vector.

    Returns
    -------
    array-like
        The shrunk mean vector, with the same type as the input.
    """
    if alpha <= 0:
        return mean_vec
    return (1.0 - alpha) * mean_vec


__all__ = ["shrink_mean_to_grand_mean", "shrink_mean_to_zero"]
