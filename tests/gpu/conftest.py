"""GPU suite setup: reset each shared env to its default appearance before every test.

The ``env`` fixtures here are module-scoped, so without the reset one test's style would leak into the
next. States are always passed explicitly, so they need no reset; tests that change ``size`` restore it.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_appearance_for_shared_envs(request):
    """Reset the ``env`` fixture's style, only for tests that already request ``env``."""
    if "env" in request.fixturenames:
        env = request.getfixturevalue("env")
        env.set_style(None)
    yield
