"""
A module that handles the trade buffer logic while respecting per-instrument minimum weights.
"""

import numpy as np

def apply_trade_buffer(
    old_w: np.ndarray,
    new_w: np.ndarray,
    buffer_thr: float,
    min_instrument_weights: np.ndarray = None
) -> np.ndarray:
    """
    Applies a "trade buffer" to reduce small changes in weights, but also ensures that
    each instrument respects any 'min_instrument_weight'.

    1) If the absolute change (new_w[i] - old_w[i]) is below buffer_thr, revert to old_w[i].
    2) If 'min_instrument_weights' is provided, ensure updated[i] >= min_instrument_weights[i].
    3) Re-normalize the updated weights so sum(updated) = 1 if possible. If sum(...) <= 0, revert.

    Parameters
    ----------
    old_w : np.ndarray
        Weight vector before rebalancing.
    new_w : np.ndarray
        Newly optimized weight vector from the solver.
    buffer_thr : float
        Threshold for ignoring small changes. Example: 0.01 => changes below 1% are reverted.
    min_instrument_weights : np.ndarray, optional
        An array of min weights for each instrument. If not provided, defaults to 0.

    Returns
    -------
    updated : np.ndarray
        The final, post-buffer (and min-weight-enforced) weight vector, re-normalized.
    """
    updated = new_w.copy()
    diffs = updated - old_w

    # 1) Revert small changes
    for i in range(len(updated)):
        if abs(diffs[i]) < buffer_thr:
            updated[i] = old_w[i]

    # 2) Enforce min instrument weights
    if min_instrument_weights is not None:
        for i in range(len(updated)):
            if updated[i] < min_instrument_weights[i]:
                updated[i] = min_instrument_weights[i]

    # 3) Re-normalize
    total_w = np.sum(updated)
    if total_w <= 0:
        # fallback if sum is zero or negative => revert to old or equal
        old_sum = np.sum(old_w)
        if old_sum > 1e-12:
            updated = old_w
        else:
            updated = np.ones(len(updated)) / len(updated)
    else:
        updated = updated / total_w

    return updated


def build_min_instrument_weights_array(
    asset_classes: list[str],
    class_constraints: dict
) -> np.ndarray:
    """
    Utility function to build an array of min_instrument_weight for each instrument
    given a list of asset_classes and a constraints dictionary.

    Example:
        asset_classes = ["Equity", "Equity", "Bond", "Commodity"]
        class_constraints = {
            "Equity": {"min_instrument_weight": 0.01},
            "Bond": {"min_instrument_weight": 0.02},
            "Commodity": {}
        }
        => returns array [0.01, 0.01, 0.02, 0.0]

    If no "min_instrument_weight" is specified for a class, defaults to 0.0.
    """
    n = len(asset_classes)
    min_w = np.zeros(n)
    for i, cl_ in enumerate(asset_classes):
        cc = class_constraints.get(cl_, {})
        min_inst = cc.get("min_instrument_weight", 0.0)
        min_w[i] = min_inst
    return min_w
