"""Numba CPU twins of the Triton raster kernels and of the 3-D analytic ray caster."""

from __future__ import annotations

import importlib.util
import os
from typing import Any

_STATE: dict[str, bool] = {}


def available() -> bool:
    """Return whether numba imports; the answer is cached."""
    if "ok" not in _STATE:
        ok = importlib.util.find_spec("numba") is not None
        if ok:
            try:
                from ..._core.cpus import numba_threads

                numba_threads()
            except Exception:  # noqa: BLE001 - an unusable numba means the fallback paths draw
                ok = False
        _STATE["ok"] = ok
    return _STATE["ok"]


def ray_module(backend: str, device) -> Any:
    """Return the Numba ray module for a CPU 3-D render, or None to cast with Torch.

    ``backend`` is ``"auto"`` (Numba when it imports), ``"numba"`` (required) or ``"torch"``.
    """
    if backend not in ("auto", "numba", "torch"):
        raise ValueError(f"3-D CPU ray backend must be 'auto', 'numba' or 'torch', got {backend!r}")
    if backend == "torch" or str(device) != "cpu":
        return None
    if not available():
        if backend == "numba":
            raise ValueError("backend='numba' needs numba, which does not import here")
        return None
    from . import rays3d as module

    return module
