"""Numba CPU twins of the Triton 2D raster kernels; they consume the same primitive records."""

from __future__ import annotations

import importlib.util
import os

_STATE: dict[str, bool] = {}


def available() -> bool:
    """Return whether numba imports; the answer is cached."""
    if "ok" not in _STATE:
        ok = importlib.util.find_spec("numba") is not None
        if ok:
            try:
                import numba

                if "NUMBA_NUM_THREADS" not in os.environ and hasattr(os, "sched_getaffinity"):
                    numba.set_num_threads(max(1, min(numba.config.NUMBA_NUM_THREADS, len(os.sched_getaffinity(0)))))
            except Exception:  # noqa: BLE001 - an unusable numba means the fallback paths draw
                ok = False
        _STATE["ok"] = ok
    return _STATE["ok"]
