"""The classic-control fused kernel against its op-list description and the per-primitive path.

Three implementations draw the same picture:

    gfxdraw/__init__.py   per primitive, batched torch      the specification
    gfxdraw/classic.py    render_classic_torch              the same rules, per pixel, in torch
    gfxdraw/classic.py    render_classic (Triton)           the same again, in one kernel

The first two are compared on CPU (a difference is a wrong op list: geometry, colour, order or
window); the last two on CUDA (a difference is a Triton coding fault). ``test_fast_paths.py`` checks
``env.render`` end to end with the kernel on and off.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import stable_envs as se
from stable_envs.render.gfxdraw import classic as FC
from tests.gpu._shared import assert_frames_identical, make_or_skip
from tests.helpers.gfxdraw_reference import render_classic_torch

CLASSIC = ("Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum")

#: A fixed seed per state family (``hash(kind)`` is salted per interpreter).
SEEDS = {"corpus": 0, "reachable": 101, "off-surface": 202, "far-angles": 303}

#: State families: the frozen corpus, reachable samples, and states past the reachable range.
STATE_CLASSES = {
    "corpus": None,
    "reachable": lambda env, n, rng: np.asarray(env.sample_states(n, rng), np.float32),
    "off-surface": lambda env, n, rng: _wide(env, n, rng, 3.5),
    "far-angles": lambda env, n, rng: _wide(env, n, rng, 12.0),
}


def _wide(env, n, rng, k):
    """States drawn from ``k`` times the reachable range."""
    name = type(env).__name__
    if name == "CartPole":
        return np.stack([rng.uniform(-2.4 * k, 2.4 * k, n), rng.uniform(-0.21 * k, 0.21 * k, n)], 1).astype(np.float32)
    if name == "Pendulum":
        return np.stack([rng.uniform(-np.pi * k, np.pi * k, n), rng.uniform(-2.0 * k, 2.0 * k, n)], 1).astype(
            np.float32
        )
    if name == "Acrobot":
        return rng.uniform(-np.pi * k, np.pi * k, (n, 2)).astype(np.float32)
    lo, hi = env.MINP, env.MAXP
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * k
    return rng.uniform(mid - half, mid + half, (n, 1)).astype(np.float32)


def _states(env, kind, n, seed):
    if kind == "corpus":
        from pathlib import Path

        npz = Path(__file__).resolve().parents[2] / "tests/fixtures/states/fidelity" / f"{type(env).__name__}.npz"
        if not npz.is_file():
            pytest.skip(f"no frozen corpus for {type(env).__name__} in this checkout")
        pool = np.load(npz)["states"]
        return np.asarray(pool[np.arange(n) % len(pool)], np.float32)
    return STATE_CLASSES[kind](env, n, np.random.default_rng(seed))


def per_row_style(env, groups: int = 4, seed: int = 3):
    """A style with several distinct rows and both antialias settings in one batch.

    Primitive colours, the antialias switch and MountainCar's baked terrain plate must all flow
    through the op list per row.
    """
    from stable_envs.testing import conformance as C

    style = C.uniform_style(env, env.B, groups=groups, seed=seed)
    if style is None:
        return None
    style = np.asarray(style, np.float32).copy()
    o, _w = env.resolve_style(None, None, batch=1).layout["rendering.antialias"]
    style[:, o] = (np.arange(env.B) % 2).astype(np.float32)
    return style


def _ops(env, s, sty):
    """The env's op list at the geometry its render draws, per-lane physics lengths included."""
    geo = env._varied_geometry(sty, own=False)
    return env._ops(s, sty, *(() if geo is None else (geo,)))


def _described(env, states, style=None):
    """``render_classic_torch`` over the env's own op list: the middle implementation."""
    s = torch.as_tensor(np.asarray(states, np.float32), device=env.device)
    s = s.reshape(-1, env.state_dim).to(torch.float64)
    sty = env.resolve_style(style, None, batch=s.shape[0])
    return render_classic_torch(_ops(env, s, sty).finish())


def _per_primitive(env, states, style=None):
    """``gfxdraw.py``'s path, reached with the kernel switched off."""
    was = env.use_fused_render
    try:
        env.use_fused_render = False
        return env._render(states, style=style).clone()
    finally:
        env.use_fused_render = was


# ------------------------------------------------------------------ joint 1: the description
@pytest.mark.parametrize("name", CLASSIC)
@pytest.mark.parametrize("kind", sorted(STATE_CLASSES))
def test_the_described_frame_is_the_drawn_frame(name, kind):
    """CPU: ``render_classic_torch`` over the op list equals the per-primitive render."""
    env = make_or_skip(name, num_envs=8, device="cpu")
    if not getattr(env, "fused_capable", False):
        pytest.skip(f"{name} has no fused description")
    states = _states(env, kind, env.B, seed=SEEDS[kind])
    assert_frames_identical(
        _described(env, states),
        _per_primitive(env, states),
        what=f"{name} [{kind}] described vs drawn",
        why="render_classic_torch is the per-pixel form of gfxdraw.py's rules; a difference here is "
        "a mis-described frame (geometry, colour, order or window in Ops), not a kernel bug",
    )


@pytest.mark.parametrize("name", CLASSIC)
def test_the_description_carries_per_row_appearance(name):
    """CPU: the op-list description matches the per-primitive render under a per-row style."""
    env = make_or_skip(name, num_envs=8, device="cpu")
    if not getattr(env, "fused_capable", False):
        pytest.skip(f"{name} has no fused description")
    states = _states(env, "corpus", env.B, seed=0)
    style = per_row_style(env)
    if style is None:
        pytest.skip(f"{name} declares no variation")
    assert_frames_identical(
        _described(env, states, style),
        _per_primitive(env, states, style),
        what=f"{name} per-row style, mixed antialias",
        why="every appearance knob must flow through the op list per row",
    )


# ------------------------------------------------------------------ joint 2: the kernel
@pytest.mark.gpu
@pytest.mark.parametrize("name", CLASSIC)
@pytest.mark.parametrize("kind", sorted(STATE_CLASSES))
@pytest.mark.parametrize("style_kind", ["default", "per-row"])
def test_the_kernel_is_the_description(name, kind, style_kind):
    """CUDA: the Triton kernel equals ``render_classic_torch`` on the same op list.

    The ``per-row`` case drives the kernel's per-environment colour, ``aa`` flag and plate loads.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    if not FC.HAVE_TRITON:
        pytest.skip("triton is unavailable")
    env = make_or_skip(name, num_envs=8, device="cuda")
    if not getattr(env, "fused_capable", False):
        pytest.skip(f"{name} has no fused description")
    states = _states(env, kind, env.B, seed=SEEDS[kind])
    style = None
    if style_kind == "per-row":
        style = per_row_style(env)
        if style is None:
            pytest.skip(f"{name} declares no variation")
    s = torch.as_tensor(states, device=env.device).reshape(-1, env.state_dim).to(torch.float64)
    sty = env.resolve_style(style, None, batch=s.shape[0])
    ops = _ops(env, s, sty).finish()

    if style_kind == "per-row":
        # Check the batch really is mixed: both aa settings, distinct colours, and (MountainCar) one plate
        # row per environment, the kernel's PLATE_B = B specialisation.
        assert len(set(ops.aa.tolist())) == 2, f"{name}: the batch does not hold both aa settings"
        cols = ops.meta_t[..., FC.CM_R : FC.CM_B + 1]
        assert int((cols != cols[:1]).any()), f"{name}: every environment drew the same colours"
        if ops.plate is not None:
            assert int(ops.plate.shape[1]) == env.B, (
                f"{name}: the baked plate has {ops.plate.shape[1]} row(s) for {env.B} environments, "
                f"so the kernel's PLATE_B = B path is not being exercised"
            )

    assert_frames_identical(
        FC.render_classic(ops),
        render_classic_torch(ops),
        what=f"{name} [{kind}, {style_kind}] kernel vs description",
        why="the two evaluate the same per-pixel expression; a difference is a Triton coding fault "
        "(a shape, an integer width, a branch)",
    )


@pytest.mark.gpu
@pytest.mark.parametrize("name", CLASSIC)
def test_env_render_agrees_under_a_per_row_style(name):
    """CUDA: ``env._render(states, style=...)`` is identical with the kernel on and off under a per-row style.

    Unlike the op-list comparison, this also covers style resolution on both paths.
    """
    if not (torch.cuda.is_available() and FC.HAVE_TRITON):
        pytest.skip("needs a CUDA device with triton")
    env = make_or_skip(name, num_envs=8, device="cuda")
    if not getattr(env, "fused_capable", False):
        pytest.skip(f"{name} has no fused description")
    states = _states(env, "corpus", env.B, seed=0)
    style = per_row_style(env)
    if style is None:
        pytest.skip(f"{name} declares no variation")
    was = env.use_fused_render
    try:
        env.use_fused_render = True
        fast = env._render(states, style=style).clone()
        env.use_fused_render = False
        slow = env._render(states, style=style).clone()
    finally:
        env.use_fused_render = was
    assert_frames_identical(
        fast,
        slow,
        what=f"{name} per-row style, kernel on vs off",
        why="every appearance knob must reach the kernel as data",
    )


@pytest.mark.gpu
@pytest.mark.parametrize("name", CLASSIC)
def test_the_kernel_is_batch_and_chunk_invariant(name):
    """CUDA: the fused render is identical whole, in chunks and permuted."""
    if not (torch.cuda.is_available() and FC.HAVE_TRITON):
        pytest.skip("needs a CUDA device with triton")
    env = make_or_skip(name, num_envs=12, device="cuda")
    if not getattr(env, "fused_capable", False):
        pytest.skip(f"{name} has no fused description")
    states = _states(env, "corpus", 12, seed=0)
    whole = env._render(states).clone()
    pieces = []
    for i, k in ((0, 4), (4, 8)):
        with se.make(name, num_envs=k, device="cuda") as chunk:
            pieces.append(chunk._render(states[i : i + k]).clone())
    chunks = torch.cat(pieces)
    assert_frames_identical(
        chunks, whole, what=f"{name} chunked vs whole (fused)", why="the fused kernel must be batch-size invariant"
    )
    perm = np.array([7, 0, 11, 3, 5, 1, 9, 2, 10, 4, 8, 6])
    shuffled = env._render(states[perm]).clone()
    assert_frames_identical(
        shuffled, whole[perm], what=f"{name} permuted (fused)", why="the fused kernel must be order-independent"
    )


@pytest.mark.gpu
@pytest.mark.parametrize("name", CLASSIC)
def test_the_cpu_path_is_untouched_by_the_kernel(name):
    """A ``device="cpu"`` env never selects the fused kernel."""
    cpu = make_or_skip(name, num_envs=4, device="cpu")
    assert cpu._use_fused() is False, f"{name} on device='cpu' selected the fused kernel"
