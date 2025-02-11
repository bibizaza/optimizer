# modules/optimization/constraints.py

import numpy as np
import pandas as pd

def build_asset_class_constraints_mode(
    df_instruments: pd.DataFrame,
    asset_classes: list,
    constraint_mode: str,
    buffer_pct: float,
    user_custom_constraints: dict
) -> dict:
    """
    Returns a class_constraints dict for each asset class based on the chosen mode:
      - "custom": uses user_custom_constraints exactly for min_class_weight, max_class_weight, max_instrument_weight
      - "keep_current": derives min/max from old class weight ± buffer_pct,
                        but uses user_custom_constraints to retrieve max_instrument_weight

    Parameters
    ----------
    df_instruments : pd.DataFrame
        Must have a "Weight_Old" column for each instrument (#Asset, #ID, #Quantity, #Last_Price => Value => Weight_Old).
    asset_classes : list
        The asset class for each instrument, in the same order as the final MPT weights.
        (We only need it to ensure we handle all classes that appear in the old portfolio.)
    constraint_mode : str
        Either "custom" or "keep_current".
    buffer_pct : float
        e.g., 0.05 means ±5%. Only used if constraint_mode=="keep_current".
    user_custom_constraints : dict
        The user’s dictionary keyed by asset class, e.g.:
          {
            "Equity": {
              "min_class_weight": 0.2,
              "max_class_weight": 0.5,
              "max_instrument_weight": 0.3
            },
            ...
          }
        For "custom" mode, we read min/max from here.
        For "keep_current" mode, we only read "max_instrument_weight" from here
        (the min/max are derived from old weights ± buffer).

    Returns
    -------
    class_constraints : dict
        A dict keyed by asset class name, each containing:
          {
            "min_class_weight": float,
            "max_class_weight": float,
            "max_instrument_weight": float
          }
        that can be passed to build_constraints(...).
    """

    # If "custom", return user_custom_constraints directly (for min/max).
    if constraint_mode == "custom":
        return user_custom_constraints

    # If "keep_current", we compute old class weight from df_instruments,
    # then do ± buffer_pct, but still pull max_instrument_weight from user_custom_constraints.
    class_constraints = {}

    # Summation of old portfolio's weight by asset
    class_old_w = df_instruments.groupby("#Asset")["Weight_Old"].sum()  # e.g. {"Equity": 0.40, "Bond":0.30,...}

    # For each class that appears in asset_classes or in df_instruments
    classes_in_data = df_instruments["#Asset"].unique()

    # We'll define constraints for each relevant class
    for cl in classes_in_data:
        old_w = class_old_w.get(cl, 0.0)  # if missing, assume 0
        min_cl = max(0.0, old_w - buffer_pct)
        max_cl = min(1.0, old_w + buffer_pct)

        # read user’s max_instrument_weight if present
        if cl in user_custom_constraints:
            max_inst = user_custom_constraints[cl].get("max_instrument_weight", 1.0)
        else:
            max_inst = 1.0

        class_constraints[cl] = {
            "min_class_weight": min_cl,
            "max_class_weight": max_cl,
            "max_instrument_weight": max_inst
        }

    return class_constraints


def build_constraints(
    weights: np.ndarray,
    asset_classes: list,
    class_constraints: dict
):
    """
    Build SLSQP constraints for:
      1) sum(weights) = 1
      2) no short selling => weight[i] >= 0 (via bounds)
      3) For each asset class:
         - min_class_weight <= sum(w_class) <= max_class_weight
         - Each instrument in that class must not exceed 'max_instrument_weight'

    Parameters
    ----------
    weights : np.ndarray
        e.g. np.ones(n)/n for initial guess
    asset_classes : list of str
        The asset class for each instrument, same order as weights
    class_constraints : dict
        e.g. {
          "Equity": {
            "min_class_weight": 0.2,
            "max_class_weight": 0.5,
            "max_instrument_weight": 0.3
          },
          ...
        }

    Returns
    -------
    cons : list of dict
        A list of constraints usable by scipy.optimize
    bounds : list of tuple
        Bounds for each weight, e.g. (0,1) for no short selling
    """
    n = len(weights)
    if len(asset_classes) != n:
        raise ValueError("asset_classes length != number of weights")

    # 1) sum(weights)=1
    cons = [
        {"type":"eq","fun": lambda w: np.sum(w)-1.0}
    ]

    # 2) no short => [0,1]
    bounds = [(0.0,1.0) for _ in range(n)]

    # 3) class constraints
    unique_classes = list(set(asset_classes))
    for cl in unique_classes:
        data = class_constraints.get(cl, {})
        min_cl = data.get("min_class_weight", 0.0)
        max_cl = data.get("max_class_weight", 1.0)
        max_inst= data.get("max_instrument_weight", 1.0)

        # indices for that class
        idxs= [i for i,a in enumerate(asset_classes) if a==cl]

        def class_min(w, idxs=idxs, min_cl=min_cl):
            return np.sum(w[idxs]) - min_cl
        def class_max(w, idxs=idxs, max_cl=max_cl):
            return max_cl - np.sum(w[idxs])

        cons.append({"type":"ineq","fun": class_min})
        cons.append({"type":"ineq","fun": class_max})

        # max_instrument_weight => w[i]<= max_inst
        for i in idxs:
            def inst_max_weight(w, i=i, max_inst=max_inst):
                return max_inst- w[i]
            cons.append({"type":"ineq","fun": inst_max_weight})

    return cons, bounds


def build_constraints_from_mode(
    df_instruments: pd.DataFrame,
    weights: np.ndarray,
    asset_classes: list,
    constraint_mode: str,
    buffer_pct: float,
    user_custom_constraints: dict
):
    """
    High-level function that:
      1) builds a 'class_constraints' dict using build_asset_class_constraints_mode
      2) calls build_constraints(...) to get (cons,bounds)

    Steps:
      - df_instruments must have "Weight_Old" for each instrument
      - user_custom_constraints used either in "custom" mode or for "max_instrument_weight" in "keep_current"
    """
    # 1) derive the final class_constraints
    class_constraints = build_asset_class_constraints_mode(
        df_instruments=df_instruments,
        asset_classes=asset_classes,
        constraint_mode=constraint_mode,
        buffer_pct=buffer_pct,
        user_custom_constraints=user_custom_constraints
    )

    # 2) build the final constraints & bounds
    cons, bnds = build_constraints(
        weights=weights,
        asset_classes=asset_classes,
        class_constraints=class_constraints
    )
    return cons, bnds
