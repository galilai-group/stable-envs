"""GPU suite setup: reset each shared env to its default appearance before every test.

The ``env`` fixtures here are module-scoped, so without the reset one test's style would leak into the
next. States are always passed explicitly, so they need no reset; tests that change ``size`` restore it.
"""

from __future__ import annotations

import pytest


@pytest.fixture(autouse=True, scope="module")
def _release_gl_after_module():
    """Release every GL context a module leaves behind and require created == released afterward."""
    from stable_envs._core.mujoco_gl import release_all_gl
    from tests._runner import resource_guard

    yield
    release_all_gl()
    counts = resource_guard.take()
    sites = "\n".join("  " + " <- ".join(frames[::-1][:4]) for frames in resource_guard.live_sites())
    assert counts.context_live == 0, f"{counts.context_live} EGL contexts are still live after the module:\n{sites}"


@pytest.fixture(autouse=True)
def _default_appearance_for_shared_envs(request):
    """Reset the ``env`` fixture's style, only for tests that already request ``env``."""
    if "env" in request.fixturenames:
        env = request.getfixturevalue("env")
        env.set_style(None)
    yield
