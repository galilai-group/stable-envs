"""Moved to scripts/ci/smoke_installed.py. Kept so Oscar notes still resolve."""

from __future__ import annotations

import runpy
import sys
from pathlib import Path

_CANONICAL = Path(__file__).resolve().parents[3] / "scripts" / "ci" / "smoke_installed.py"
sys.argv[0] = str(_CANONICAL)
runpy.run_path(str(_CANONICAL), run_name="__main__")
