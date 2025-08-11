"""
Dynamic backend selection for numerical computing.

This module attempts to import GPU‑accelerated libraries (CuPy and
cuDF) to accelerate heavy numerical computations on machines equipped
with an NVIDIA GPU. If those imports fail (for example, on systems
without a CUDA‑capable GPU or without RAPIDS installed), it falls
back to standard NumPy and Pandas. The selected backend is exposed via
the module attributes ``xp`` and ``pd_xp``.

Environment
-----------
If you wish to force the CPU path even when a GPU is present, set
the environment variable ``FORCE_CPU_ONLY`` to ``"1"`` before
importing this module.

Example
-------
>>> from modules.optimization.utils.backend import xp, pd_xp, GPU_AVAILABLE
>>> arr = xp.ones((3, 3))
>>> df = pd_xp.DataFrame(arr)
"""

import os

# The main numerical array module (NumPy or CuPy)
xp = None  # type: ignore
# The tabular data module (Pandas or cuDF)
pd_xp = None  # type: ignore
# Flag indicating whether a GPU is available and CuPy/cuDF were loaded
GPU_AVAILABLE: bool = False

# Honour a user‑set environment variable to skip GPU usage. This can be
# convenient for debugging on GPU‑equipped machines.
_force_cpu = os.environ.get("FORCE_CPU_ONLY", "0") == "1"

if not _force_cpu:
    try:
        # Attempt to import the CUDA libraries. If either import fails
        # we will fall back to the CPU path.
        import cupy as _cupy  # type: ignore
        import cudf as _cudf  # type: ignore

        # Check that at least one CUDA device is present. Without a
        # device, CuPy will still import but raise errors on use.
        if _cupy.cuda.runtime.getDeviceCount() > 0:
            GPU_AVAILABLE = True
            xp = _cupy  # use CuPy for array math
            pd_xp = _cudf  # use cuDF for DataFrames
            # Informational message for developers; remove print for production
            print("✅ GPU detected. Using CuPy and cuDF for acceleration.")
        else:
            raise RuntimeError("CuPy imported but no CUDA devices found.")

    except Exception:
        # Any exception here means we cannot use the GPU libraries, so we
        # fall back to NumPy/Pandas.
        GPU_AVAILABLE = False
        import numpy as _numpy  # type: ignore
        import pandas as _pandas  # type: ignore
        xp = _numpy
        pd_xp = _pandas
        print("⚠️ GPU not available or RAPIDS not installed. Falling back to NumPy and Pandas.")
else:
    # Forced CPU mode; use standard NumPy/Pandas regardless of GPU
    GPU_AVAILABLE = False
    import numpy as _numpy  # type: ignore
    import pandas as _pandas  # type: ignore
    xp = _numpy
    pd_xp = _pandas
    print("CPU mode forced by environment variable. Using NumPy and Pandas.")

__all__ = ["xp", "pd_xp", "GPU_AVAILABLE"]
