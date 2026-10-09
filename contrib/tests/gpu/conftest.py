"""GPU tests for the contrib families.

``tests/gpu`` goes on ``sys.path`` so these files import ``_shared`` (and, for the cases split out of a
``tests/gpu`` file, that file) exactly as the files in ``tests/gpu`` do. The per-test style reset is
the one ``tests/gpu/conftest.py`` applies there.
"""

from __future__ import annotations

import sys
from pathlib import Path

_TESTS_GPU = str(Path(__file__).resolve().parents[3] / "tests" / "gpu")
if _TESTS_GPU not in sys.path:
    sys.path.insert(0, _TESTS_GPU)

from tests.gpu.conftest import _default_appearance_for_shared_envs  # noqa: E402,F401
