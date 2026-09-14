"""The classic-control fused kernel, against the two things it has to be.

There are three implementations of one picture in this repository now, and the chain between them
is what makes the fast one admissible:

    gfxdraw.py            per primitive, batched torch      the executable specification
    gfxdraw_classic.py    render_classic_torch              the same rules, per pixel, in torch
    gfxdraw_classic.py    render_classic (Triton)           the same again, in one kernel

``test_fast_paths.py`` asserts the end-to-end joint (``env.render`` with the kernel on and off) on
the frozen corpus and on states chosen to be awkward. This file asserts the two joints separately,
because when they disagree the two failures want different files opened: a break at the first joint
is a mis-DESCRIBED frame (the op list has the wrong geometry, colour or order) and a break at the
second is a mis-CODED kernel (a Triton shape, an integer width, a branch). It also runs the first
joint on the CPU, where Triton does not exist, so a description error is caught on a laptop.

The per-primitive cases below are not a random sweep. Each one is a rule from ``gfxdraw.py``'s
docstring that has its own failure mode: the scanline fill's two crossings, the disc's paint-once
guards, the Wu fringe's clipping and its two-pixel pair, the axis-aligned short-circuit that gets no
anti-aliasing at all, the two-pass ``aacircle`` fringe, SDL's blit rule against SDL_gfx's pixel
writer, and painter's order between primitives that overlap.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import stable_envs as se
from stable_envs.render import gfxdraw_classic as FC
from tests.gpu._shared import assert_frames_identical, make_or_skip

CLASSIC = ("Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum")

#: A fixed seed per state family. NOT ``hash(kind)``: string hashing is salted per interpreter, so
#: that would draw different states every run -- a failure nobody could reproduce and a pass that is
#: not evidence. ``tests/gpu/test_conformance.py`` documents the same trap for its parity draw.
SEEDS = {"corpus": 0, "reachable": 101, "off-surface": 202, "far-angles": 303}

#: State families, each chosen for a rule it stresses. See the module docstring.
STATE_CLASSES = {
    "corpus": None,
    "reachable": lambda env, n, rng: np.asarray(env.sample_states(n, rng), np.float32),
    "off-surface": lambda env, n, rng: _wide(env, n, rng, 3.5),
    "far-angles": lambda env, n, rng: _wide(env, n, rng, 12.0),
}


def _wide(env, n, rng, k):
    """States scaled ``k`` times past the reachable range, so bounds are tested rather than assumed."""
    name = type(env).__name__
    if name == "CartPole":
        return np.stack([rng.uniform(-2.4 * k, 2.4 * k, n),
                         rng.uniform(-0.21 * k, 0.21 * k, n)], 1).astype(np.float32)
    if name == "Pendulum":
        return np.stack([rng.uniform(-np.pi * k, np.pi * k, n),
                         rng.uniform(-2.0 * k, 2.0 * k, n)], 1).astype(np.float32)
    if name == "Acrobot":
        return rng.uniform(-np.pi * k, np.pi * k, (n, 2)).astype(np.float32)
    lo, hi = env.MINP, env.MAXP
    mid, half = (lo + hi) / 2, (hi - lo) / 2 * k
    return rng.uniform(mid - half, mid + half, (n, 1)).astype(np.float32)


def _states(env, kind, n, seed):
    if kind == "corpus":
        from pathlib import Path
        npz = (Path(__file__).resolve().parents[2] / "examples/results/step4/states"
               / f"{type(env).__name__}.npz")
        if not npz.is_file():
            pytest.skip(f"no frozen corpus for {type(env).__name__} in this checkout")
        pool = np.load(npz)["states"]
        return np.asarray(pool[np.arange(n) % len(pool)], np.float32)
    return STATE_CLASSES[kind](env, n, np.random.default_rng(seed))


def per_row_style(env, groups: int = 4, seed: int = 3):
    """A style with several DISTINCT rows and both antialias settings inside one batch.

    Everything a style can change has to flow through the op list as data -- the colour of each
    primitive, the anti-alias switch, and for MountainCar the baked terrain plate, which is keyed on
    the two leaves that can move it. A kernel that cached any of them on anything but static
    geometry passes the default-style test and fails here, and that is the entire reason a batch is
    allowed to hold 4,096 appearances.
    """
    from stable_envs.testing import conformance as C

    style = C.uniform_style(env, env.B, groups=groups, seed=seed)
    if style is None:
        return None
    style = np.asarray(style, np.float32).copy()
    o, _w = env.resolve_style(None, None, batch=1).layout["rendering.antialias"]
    style[:, o] = (np.arange(env.B) % 2).astype(np.float32)
    return style


def _described(env, states, style=None):
    """``render_classic_torch`` over the env's own op list: the middle implementation."""
    s = torch.as_tensor(np.asarray(states, np.float32), device=env.device)
    s = s.reshape(-1, env.state_dim).to(torch.float64)
    sty = env.resolve_style(style, None, batch=s.shape[0])
    return FC.render_classic_torch(env._ops(s, sty).finish())


def _per_primitive(env, states, style=None):
    """``gfxdraw.py``'s path, reached with the kernel switched off."""
    was = env.use_fused_render
    try:
        env.use_fused_render = False
        return env.render(states, style=style).clone()
    finally:
        env.use_fused_render = was


# ------------------------------------------------------------------ joint 1: the description
@pytest.mark.parametrize("name", CLASSIC)
@pytest.mark.parametrize("kind", sorted(STATE_CLASSES))
def test_the_described_frame_is_the_drawn_frame(name, kind):
    """CPU. The op list must describe exactly what the per-primitive path draws.

    Not marked ``gpu``: the description is device-independent and a mistake in it — a window that
    clips a primitive, an order that swaps two of them, a colour taken from the wrong style path —
    is the kind of error that should cost a laptop second, not a GPU job.
    """
    env = make_or_skip(name, batch=8, device="cpu")
    if not getattr(env, "fused_capable", False):
        pytest.skip(f"{name} has no fused description")
    states = _states(env, kind, env.B, seed=SEEDS[kind])
    assert_frames_identical(
        _described(env, states), _per_primitive(env, states),
        what=f"{name} [{kind}] described vs drawn",
        why="render_classic_torch is the per-pixel form of gfxdraw.py's rules; a difference here is "
            "a mis-described frame (geometry, colour, order or window in Ops), not a kernel bug")


@pytest.mark.parametrize("name", CLASSIC)
def test_the_description_carries_per_row_appearance(name):
    """CPU. Colours, the anti-alias switch and MountainCar's baked terrain are all per ENVIRONMENT.

    A kernel that cached any of them on anything but static geometry would pass the default-style
    test and fail here, which is the entire reason a batch is allowed to hold 4,096 appearances.
    """
    env = make_or_skip(name, batch=8, device="cpu")
    if not getattr(env, "fused_capable", False):
        pytest.skip(f"{name} has no fused description")
    states = _states(env, "corpus", env.B, seed=0)
    style = per_row_style(env)
    if style is None:
        pytest.skip(f"{name} declares no variation")
    assert_frames_identical(
        _described(env, states, style), _per_primitive(env, states, style),
        what=f"{name} per-row style, mixed antialias",
        why="every appearance knob must flow through the op list as data; a cached stencil or a "
            "batch-wide colour would show up only here")


# ------------------------------------------------------------------ joint 2: the kernel
@pytest.mark.gpu
@pytest.mark.parametrize("name", CLASSIC)
@pytest.mark.parametrize("kind", sorted(STATE_CLASSES))
@pytest.mark.parametrize("style_kind", ["default", "per-row"])
def test_the_kernel_is_the_description(name, kind, style_kind):
    """CUDA. The Triton kernel against the torch form of the same closed expression.

    The ``per-row`` half is the guard for every mixed-row path the kernel has, and it is separate
    from the CPU guard above because those are different code: on CUDA the per-environment colour,
    the per-environment ``aa`` flag and MountainCar's per-environment plate are all loads inside the
    kernel, and nothing else in this suite drives them. Until this existed they were covered only
    by the corpus-style digest in the Step 4 replay, which is one style built one way.
    """
    if not torch.cuda.is_available():
        pytest.skip("needs a CUDA device")
    if not FC.HAVE_TRITON:
        pytest.skip("triton is unavailable")
    env = make_or_skip(name, batch=8, device="cuda")
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
    ops = env._ops(s, sty).finish()

    if style_kind == "per-row":
        # The mixed-row paths must actually be mixed, or this case is the default one with more
        # steps. Both antialias settings present; at least two distinct colours for some primitive;
        # and for MountainCar a plate with one row PER ENVIRONMENT, which is the PLATE_B = B
        # specialisation of the kernel and is otherwise never reached.
        assert len(set(ops.aa.tolist())) == 2, f"{name}: the batch does not hold both aa settings"
        cols = ops.meta_t[..., FC.CM_R:FC.CM_B + 1]
        assert int((cols != cols[:1]).any()), f"{name}: every environment drew the same colours"
        if ops.plate is not None:
            assert int(ops.plate.shape[1]) == env.B, (
                f"{name}: the baked plate has {ops.plate.shape[1]} row(s) for {env.B} environments, "
                f"so the kernel's PLATE_B = B path is not being exercised")

    assert_frames_identical(
        FC.render_classic(ops), FC.render_classic_torch(ops),
        what=f"{name} [{kind}, {style_kind}] kernel vs description",
        why="the two evaluate the same per-pixel expression; a difference is a Triton coding fault "
            "(a shape, an integer width, a branch), not a rule")


@pytest.mark.gpu
@pytest.mark.parametrize("name", CLASSIC)
def test_env_render_agrees_under_a_per_row_style(name):
    """CUDA, end to end: ``env.render(states, style=...)`` with the kernel on and off.

    The description guard above compares two evaluations of one op list; this one goes through
    ``render`` twice, so a per-row style that were resolved differently on the two paths -- rather
    than evaluated differently -- would show up here and nowhere else.
    """
    if not (torch.cuda.is_available() and FC.HAVE_TRITON):
        pytest.skip("needs a CUDA device with triton")
    env = make_or_skip(name, batch=8, device="cuda")
    if not getattr(env, "fused_capable", False):
        pytest.skip(f"{name} has no fused description")
    states = _states(env, "corpus", env.B, seed=0)
    style = per_row_style(env)
    if style is None:
        pytest.skip(f"{name} declares no variation")
    was = env.use_fused_render
    try:
        env.use_fused_render = True
        fast = env.render(states, style=style).clone()
        env.use_fused_render = False
        slow = env.render(states, style=style).clone()
    finally:
        env.use_fused_render = was
    assert_frames_identical(fast, slow, what=f"{name} per-row style, kernel on vs off",
                            why="every appearance knob must reach the kernel as data")


@pytest.mark.gpu
@pytest.mark.parametrize("name", CLASSIC)
def test_the_kernel_is_batch_and_chunk_invariant(name):
    """CUDA. One state list rendered whole, in chunks, and out of order must give one answer.

    A per-pixel kernel has no batch-dependent state to get wrong, which is exactly why a failure
    here would mean something shared crept in — a cache keyed on ``B``, an op list built from an
    aggregate over the batch.
    """
    if not (torch.cuda.is_available() and FC.HAVE_TRITON):
        pytest.skip("needs a CUDA device with triton")
    env = make_or_skip(name, batch=12, device="cuda")
    if not getattr(env, "fused_capable", False):
        pytest.skip(f"{name} has no fused description")
    states = _states(env, "corpus", 12, seed=0)
    whole = env.render(states).clone()
    chunks = torch.cat([se.make(name, batch=k, device="cuda").render(states[i:i + k])
                        for i, k in ((0, 4), (4, 8))])
    assert_frames_identical(chunks, whole, what=f"{name} chunked vs whole (fused)",
                            why="the fused kernel must be batch-size invariant")
    perm = np.array([7, 0, 11, 3, 5, 1, 9, 2, 10, 4, 8, 6])
    shuffled = env.render(states[perm]).clone()
    assert_frames_identical(shuffled, whole[perm], what=f"{name} permuted (fused)",
                            why="the fused kernel must be order-independent")


@pytest.mark.gpu
@pytest.mark.parametrize("name", CLASSIC)
def test_the_cpu_path_is_untouched_by_the_kernel(name):
    """A ``device="cpu"`` render must be the per-primitive one, kernel or no kernel.

    Triton has no CPU backend here, so this is true by construction — and it is asserted anyway,
    because "the fused path is CUDA-only" is a claim about a dispatch predicate and predicates rot.
    """
    cpu = make_or_skip(name, batch=4, device="cpu")
    assert cpu._use_fused() is False, (
        f"{name} on device='cpu' selected the fused kernel; the CPU path must stay the "
        f"per-primitive one, whose pixels this ticket does not change")
