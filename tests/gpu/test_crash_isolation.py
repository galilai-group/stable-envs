"""The GPU half of the safety net: the guard attached to the REAL driver classes, and real crashes.

``tests/integration/test_safety_net.py`` proves the supervisor and the guard's policy on generated CPU-only
modules, in about a second. What it cannot prove is the part that only exists on a GPU node: that the
counters are wired to the classes MuJoCo and ogbench actually go through, and that a crash with LIVE
EGL contexts held by a module-scoped fixture behaves the same as a crash in an empty process. That is
what this file is for.

The crashing tests are marked ``selftest`` and are SKIPPED unless ``SE_SELFTEST_CRASH=1``, because they
abort the interpreter on purpose and running them by accident would do to a real run exactly what the
supervisor exists to prevent. They are kept in the repo rather than run once by hand and deleted: a
safety net that is never exercised is a safety net nobody can trust, and the way to exercise this one
is to run

    SE_SELFTEST_CRASH=1 python -m stable_envs.testing.supervisor -- tests/gpu -m selftest -v

and confirm the summary reports the crashes by name and still reports every other test.
"""

from __future__ import annotations

import ctypes
import os

import numpy as np
import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

import stable_envs as se
from stable_envs.testing import resource_guard


def test_the_resource_guard_is_attached_to_the_classes_the_driver_goes_through():
    """The counters must be on the real ``mujoco.egl.GLContext``, not on a copy or an alias of it.

    Not a formality. ``mujoco.GLContext``, ``mujoco.gl_context.GLContext`` and ``mujoco.egl.GLContext``
    are three names for one class under ``MUJOCO_GL=egl``, and ogbench, gymnasium and this repo each
    reach it through a different one. A guard patched onto the wrong name would count zero for an
    entire run and report a clean bill of health for a suite that was leaking contexts — which is
    strictly worse than no guard, because it would be believed.
    """
    mujoco = pytest.importorskip("mujoco")
    resource_guard.install()
    assert getattr(mujoco.egl.GLContext.__init__, "_se_guarded", False), (
        "EGL context creation is not instrumented; every number the safety summary prints would be "
        "zero regardless of what the run did"
    )
    assert getattr(mujoco.egl.GLContext.free, "_se_guarded", False), (
        "context RELEASE is not instrumented — the one event that is never legitimate"
    )
    assert getattr(mujoco.Renderer.__init__, "_se_guarded", False)
    assert mujoco.GLContext is mujoco.egl.GLContext, (
        "MUJOCO_GL is not egl on this host, so the instrumented class is not the one in use; the "
        "guard's numbers describe a backend nothing is rendering through"
    )


def test_building_an_env_does_not_release_anything():
    """The steady-state claim, measured on a real env rather than asserted in a docstring.

    Constructing an env, rendering with it and dropping it must create a bounded number of contexts and
    release none — including when the env goes out of scope, because ``GLContext.__del__`` calls
    ``free`` and a reference dropped at the end of a function is exactly how incident 1 and incident 2
    released contexts while other envs were still using theirs.
    """
    cells = sorted(n for n, c in se.registry.REGISTRY.items() if "ogbench" in c.__module__)
    if not cells:
        pytest.skip("no EGL-backed env registered")
    before = resource_guard.take()
    env = se.make(
        "PointmazeMediumNavigate" if "PointmazeMediumNavigate" in cells else cells[0], num_envs=2, device="cuda"
    )
    env.render(env.sample_states(2, rng=np.random.default_rng(0)))
    del env
    import gc

    gc.collect()
    d = resource_guard.delta(before)
    assert d["context_freed"] == 0, (
        f"building and dropping one env released {d['context_freed']} EGL context(s) — that is the "
        f"defect that corrupts every other live renderer in the process:\n"
        f"{resource_guard.format_sites([s for s in d['sites'] if s[0].endswith('freed')])}"
    )
    assert d["renderer_closed"] == 0


# ------------------------------------------------------------------ opt-in: real crashes


@pytest.mark.selftest
def test_selftest_segfault_with_live_egl_contexts():
    """SIGSEGV while a real env and its EGL contexts are live. Must cost exactly this test.

    The empty-process version of this is already covered on CPU; what is being checked here is that
    the breadcrumb survives a death that happens with a module-scoped MuJoCo fixture in play and a GPU
    context open, which is the situation every real incident happened in.
    """
    env = se.make("PushT", num_envs=4, device="cuda")
    env.render(env.sample_states(4, rng=np.random.default_rng(0)))
    ctypes.string_at(0)  # genuine null dereference in libc


@pytest.mark.selftest
def test_selftest_abort_with_live_egl_contexts():
    """SIGABRT — the signal a MuJoCo fatal error raises, and the one that ended four of these runs."""
    env = se.make("PushT", num_envs=4, device="cuda")
    env.render(env.sample_states(4, rng=np.random.default_rng(0)))
    os.abort()


@pytest.mark.selftest
def test_selftest_releasing_a_real_egl_context_is_caught_as_a_failure():
    """EXPECTED TO FAIL with ``EGL_CONTEXT_RELEASED``. That failure is the passing result.

    A real context, created and freed through the real driver. If this test ever comes back green the
    guard has stopped seeing the thing it was written to see, and every "0 released" in a summary since
    then meant nothing.
    """
    mujoco = pytest.importorskip("mujoco")
    ctx = mujoco.egl.GLContext(64, 64)
    ctx.free()
