"""Tests for the contrib families: ``python -m pytest contrib/tests`` from the repository root.

The root ``conftest.py`` (repository on ``sys.path``, safety plugin) applies here as it does to
``tests/``. This file registers every contrib class before any test module is collected, so modules
that read ``stable_envs.registry.REGISTRY`` at import time see them, and stamps the same directory
markers ``tests/conftest.py`` stamps (``unit/``, ``regression/``, ``gpu/``), so ``-m "not gpu"``
selects the same kind of test here as there.
"""

from __future__ import annotations

from pathlib import Path

import pytest

import contrib

contrib.load()

_MARKERS = {"unit": pytest.mark.unit, "regression": pytest.mark.regression, "gpu": pytest.mark.gpu}


def pytest_collection_modifyitems(config, items):
    del config
    here = Path(__file__).resolve().parent
    for item in items:
        path = Path(getattr(item, "path", item.fspath)).resolve()
        try:
            bucket = path.relative_to(here).parts[0]
        except (ValueError, IndexError):
            continue
        mark = _MARKERS.get(bucket)
        if mark is not None:
            item.add_marker(mark)
