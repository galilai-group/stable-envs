"""Re-export shim: ``StyleTensor`` now lives in ``stable_envs/_style.py``."""
from __future__ import annotations

from .._style import StyleTensor, rows_are_equal  # noqa: F401

__all__ = ["StyleTensor", "rows_are_equal"]
