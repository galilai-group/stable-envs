"""Root conftest: put the repository on ``sys.path``, then load the test plugins.

The path insert lets a bare ``pytest`` import ``stable_envs`` without installing it. ``pytest_plugins``
must live in the root conftest, and after the path insert, because the plugins are imported from
``tests._runner``.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

pytest_plugins = ("tests._runner.safety_plugin", "tests._runner.shard")
