"""Shared setup for the GPU suite: put every shared env back at its default appearance per test.

The ``env`` fixtures in this directory are MODULE-scoped, and they have to be: rebuilding one per test
means an OGBench plate bake per test, which is minutes. The price of sharing is that anything a test
writes onto the env is the initial condition of the next one, and the suite's own claim is that a frame
is a function of ``(state, style, size)`` — a claim it cannot make honestly while its results depend on
which test ran first.

``test_reproducibility.py`` already resets style for its own module. This does the same thing for every
module in the directory, so the property is structural rather than a thing each file remembers. It is
what makes the shuffled-order runs meaningful: with pytest-randomly installed, ``pytest tests/gpu -k
PushT -p randomly --randomly-seed=N`` and the deterministic order give the same result, and that is
only true because no test inherits an appearance from another.

State needs no equivalent reset: every check in this suite passes the states it renders explicitly, so
a leftover ``_state`` cannot reach a later frame. Size does need one and is handled where it happens —
``check_resolution_consistency`` and ``test_render_size_does_not_leak_into_later_calls`` each restore
the env's size before returning, because for the highway family ``size`` IS carried on the env (see
``HIGHWAY_SIZE_LEAK``) and a leaked one silently rewrites every later frame.
"""
from __future__ import annotations

import pytest


@pytest.fixture(autouse=True)
def _default_appearance_for_shared_envs(request):
    """Reset the shared env's style before each test, without forcing the fixture on tests that
    do not use one — ``request.getfixturevalue`` here would build every env for
    ``test_registry_is_not_empty``, which needs none of them."""
    if "env" in request.fixturenames:
        env = request.getfixturevalue("env")
        env.set_style(None)
    yield
