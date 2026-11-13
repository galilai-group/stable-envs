"""GPU checks for the test safety net: the resource guard on the real MuJoCo classes, and real crashes.

``tests/integration/test_safety_net.py`` covers the supervisor and guard policy on CPU. This file
checks that the guard counters are attached to the classes MuJoCo and ogbench use, and that a crash
with live EGL contexts is isolated to one test.

The crashing tests are marked ``selftest`` and skipped unless ``SE_SELFTEST_CRASH=1``, since they abort
the interpreter on purpose. Run them with

    SE_SELFTEST_CRASH=1 python -m tests._runner.supervisor -- tests/gpu -m selftest -v

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
from tests._runner import resource_guard


def test_the_resource_guard_is_attached_to_the_classes_the_driver_goes_through():
    """The guard instruments the real ``mujoco.egl.GLContext`` and ``mujoco.Renderer``.

    Under ``MUJOCO_GL=egl``, ``mujoco.GLContext``, ``mujoco.gl_context.GLContext`` and
    ``mujoco.egl.GLContext`` are one class reached through different names.
    """
    mujoco = pytest.importorskip("mujoco")
    resource_guard.install()
    assert getattr(mujoco.egl.GLContext.__init__, "_se_guarded", False), "EGL context creation is not instrumented"
    assert getattr(mujoco.egl.GLContext.free, "_se_guarded", False), "EGL context release is not instrumented"
    assert getattr(mujoco.Renderer.__init__, "_se_guarded", False)
    assert mujoco.GLContext is mujoco.egl.GLContext, (
        "MUJOCO_GL is not egl on this host, so the instrumented class is not the one in use"
    )


def test_building_an_env_does_not_release_anything():
    """Building, rendering and dropping an EGL-backed env releases no context or renderer.

    ``GLContext.__del__`` calls ``free``, so dropping the last reference must not reach it.
    """
    cells = sorted(n for n, c in se.registry.REGISTRY.items() if "ogbench" in c.__module__)
    if not cells:
        pytest.skip("no EGL-backed env registered")
    before = resource_guard.take()
    env = se.make(
        "PointMazeMediumNavigate" if "PointMazeMediumNavigate" in cells else cells[0], num_envs=2, device="cuda"
    )
    env._render(env.sample_states(2, rng=np.random.default_rng(0)))
    del env
    import gc

    gc.collect()
    d = resource_guard.delta(before)
    assert d["context_freed"] == 0, (
        f"building and dropping one env released {d['context_freed']} EGL context(s), which corrupts "
        f"other live renderers in the process:\n"
        f"{resource_guard.format_sites([s for s in d['sites'] if s[0].endswith('freed')])}"
    )
    assert d["renderer_closed"] == 0


# ------------------------------------------------------------------ opt-in: real crashes


@pytest.mark.selftest
def test_selftest_segfault_with_live_egl_contexts():
    """SIGSEGV while a real env and its GPU contexts are live; the supervisor must lose only this test."""
    env = se.make("PushT", num_envs=4, device="cuda")
    env._render(env.sample_states(4, rng=np.random.default_rng(0)))
    ctypes.string_at(0)  # genuine null dereference in libc


@pytest.mark.selftest
def test_selftest_abort_with_live_egl_contexts():
    """SIGABRT (what a MuJoCo fatal error raises) while a real env is live."""
    env = se.make("PushT", num_envs=4, device="cuda")
    env._render(env.sample_states(4, rng=np.random.default_rng(0)))
    os.abort()


@pytest.mark.selftest
def test_selftest_releasing_a_real_egl_context_is_caught_as_a_failure():
    """Expected to fail with ``EGL_CONTEXT_RELEASED``: frees a real EGL context through the driver.

    A pass means the guard does not detect context release.
    """
    mujoco = pytest.importorskip("mujoco")
    ctx = mujoco.egl.GLContext(64, 64)
    ctx.free()
