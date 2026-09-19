"""Moved to tests/packaging/test_distribution.py. Kept so Oscar notes still resolve."""

from __future__ import annotations

import importlib.util
from pathlib import Path

_CANONICAL = (
    Path(__file__).resolve().parents[3] / "tests" / "packaging" / "test_distribution.py"
)
_SPEC = importlib.util.spec_from_file_location(
    "_stable_envs_packaging_distribution", _CANONICAL
)
assert _SPEC is not None and _SPEC.loader is not None
_MOD = importlib.util.module_from_spec(_SPEC)
_SPEC.loader.exec_module(_MOD)
globals().update(
    {name: getattr(_MOD, name) for name in dir(_MOD) if name.startswith("test_")}
)
