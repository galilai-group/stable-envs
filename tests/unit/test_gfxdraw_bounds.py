"""Classic-control ``WIN_*`` windows, per-edge ``KMAX_*`` and disc stencils are bounds on every state.

Checked on states far outside the reachable set: each is rendered once as the code does it and once with
the bounds loosened (window = whole surface, generous ``kmax``, disc via ``fill_circle_mask``). Any pixel
difference is a bound that truncated.
"""

from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import stable_envs as se
from stable_envs.envs.classic_control import envs as CC

CLASSIC = ("Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum")


def _extreme(name, n, rng):
    """States far outside ``sample_states``: |x| up to 9, angles 24 turns wide, cars off the track."""
    if name == "CartPole":
        return np.stack([rng.uniform(-9.0, 9.0, n), rng.uniform(-24 * np.pi, 24 * np.pi, n)], 1).astype(np.float32)
    if name == "Pendulum":
        return np.stack([rng.uniform(-24 * np.pi, 24 * np.pi, n), rng.uniform(-2.0, 2.0, n)], 1).astype(np.float32)
    if name == "Acrobot":
        return rng.uniform(-24 * np.pi, 24 * np.pi, (n, 2)).astype(np.float32)
    return rng.uniform(-6.0, 5.0, (n, 1)).astype(np.float32)


def _states(name, n, seed):
    return _extreme(name, n, np.random.default_rng(seed))


def _render(env, states):
    env.use_fused_render = False
    return env._render(states, backend="port").clone()


@pytest.fixture
def unbounded(monkeypatch):
    """Force every primitive to the loosest form: whole surface, generous ``kmax``, no disc stencil.

    Patched at the module-level ``_poly``/``_circ`` helpers every ``_draw`` goes through.
    """
    KMAX = 1024

    def poly(cv, verts, color, bidx, kmax, win=None):
        from stable_envs.render import gfxdraw as G

        G.polygon(cv.use(color), verts, None, None, bidx, KMAX, True, window=None)

    def circ(cv, cx, cy, r, color, bidx, gfx):
        from stable_envs.render import gfxdraw as G

        G.circle(cv.use(color), cx, cy, r, None, None, bidx, gfx.hw(r), gfx.stencil(r), True, window=None, disc=None)

    monkeypatch.setattr(CC, "_poly", poly)
    monkeypatch.setattr(CC, "_circ", circ)
    return KMAX


@pytest.mark.parametrize("name", CLASSIC)
def test_the_windows_and_kmax_are_bounds_on_extreme_states(name, unbounded, monkeypatch):
    """The same states drawn bounded and unbounded give identical frames."""
    if name not in se.registry.REGISTRY:
        pytest.skip(f"{name} is not registered in this interpreter")
    states = _states(name, 24, seed=20260913)

    env = se.make(name, num_envs=len(states), device="cpu")
    loose = _render(env, states)
    monkeypatch.undo()  # put the real _poly/_circ back
    env2 = se.make(name, num_envs=len(states), device="cpu")
    tight = _render(env2, states)

    d = (tight.to(torch.int16) - loose.to(torch.int16)).abs()
    n = int((d.amax(-1) > 0).sum())
    assert n == 0, (
        f"{name}: the bounded primitives differ from the unbounded ones on {n:,d} of "
        f"{d[..., 0].numel():,d} pixels (max |diff| {int(d.max())}): a WIN_* window, KMAX_* or disc "
        f"stencil is too tight"
    )


#: Envs whose extreme states reach the surface border, and why. Pendulum's 114 px rod about the center
#: of a 500 px canvas never does.
LEAVES_SURFACE = {
    "CartPole": "|x| > 2.5 drives the cart off a 600 px surface, engaging clipLine",
    "Acrobot": "both links extended reach 228 px from a centre 250 px from the border",
    "MountainCar": "positions outside the track put the car past either end",
    "MountainCarContinuous": "as MountainCar",
}


@pytest.mark.parametrize("name", CLASSIC)
def test_the_extreme_states_actually_exercise_something(name):
    """The extreme states change the frame and, where possible, reach the border."""
    if name not in se.registry.REGISTRY:
        pytest.skip(f"{name} is not registered in this interpreter")
    states = _states(name, 24, seed=20260913)
    env = se.make(name, num_envs=len(states), device="cpu")
    frames = _render(env, states)
    H, W = env.native
    # Many frames show only static furniture because the moving primitive is off the surface.
    distinct = len({f.numpy().tobytes() for f in frames})
    assert distinct >= 3, f"{name}: only {distinct} distinct frames from {len(states)} extreme states"
    if name not in LEAVES_SURFACE:
        return
    bg = frames[:, 0, 0, :].to(torch.int16)
    ink = (frames.to(torch.int16) - bg.view(-1, 1, 1, 3)).abs().amax(-1) > 0
    touches_edge = ink[:, 0, :].any(1) | ink[:, H - 1, :].any(1) | ink[:, :, 0].any(1) | ink[:, :, W - 1].any(1)
    assert bool(touches_edge.any()), (
        f"{name}: no extreme state drew anything on the border, although {LEAVES_SURFACE[name]}"
    )


def test_a_deliberately_short_kmax_is_caught(monkeypatch):
    """Shrinking one ``kmax`` changes pixels, so the comparison is sensitive to the bound."""
    if "CartPole" not in se.registry.REGISTRY:
        pytest.skip("CartPole is not registered in this interpreter")
    states = _states("CartPole", 8, seed=1)
    good = _render(se.make("CartPole", num_envs=len(states), device="cpu"), states)
    monkeypatch.setattr(CC.CartPole, "KMAX_POLE", (8, 8, 8, 8))
    bad = _render(se.make("CartPole", num_envs=len(states), device="cpu"), states)
    assert not torch.equal(good, bad), "shrinking CartPole's pole kmax to 8 changed no pixels"
