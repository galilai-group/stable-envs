"""Root conftest: make the repo importable, then load the safety plugin.

Two lines of real content, and both have to be here rather than anywhere else.

``sys.path`` first. The package is not installed in the environment the GPU node runs (verified:
``import stable_envs`` from outside the repo raises ``ModuleNotFoundError``), and ``tests/`` has no
``__init__.py``, so pytest inserts ``tests/gpu`` on the path and not the root. Every test module starts
with ``import stable_envs``, so without this a bare ``pytest`` from the repo root fails at collection
and the only way to run the suite is to remember ``PYTHONPATH=.`` — which is exactly the kind of
friction that ends with a suite nobody runs.

``pytest_plugins`` second, and it must be in the ROOT conftest: pytest refuses ``pytest_plugins`` in a
non-top-level conftest, and putting ``-p tests._runner.safety_plugin`` in ``addopts`` instead
would try to import the package before the line above has run. So the order is not incidental.

See ``tests/_runner/safety_plugin.py`` for what the plugin does and why each part of it exists.
"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = str(Path(__file__).resolve().parent)
if _ROOT not in sys.path:
    sys.path.insert(0, _ROOT)

pytest_plugins = ("tests._runner.safety_plugin",)
