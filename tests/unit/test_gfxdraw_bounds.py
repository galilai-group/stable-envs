"""The classic-control windows and per-edge ``kmax`` are BOUNDS, and a bound needs adversarial input.

Step 6 made three claims about `stable_envs/envs/classic_control.py` that the frozen corpus cannot
check, because the corpus is 64 states chosen for coverage of the reachable set and a bound is only
interesting outside it:

1. **Every ``WIN_*`` window is a superset of its primitive's footprint for EVERY state**, not for the
   states a rollout visits. The argument is that the extent of a rotation or of a sine is a constant
   even when its position is not — so it has to hold for a cart driven off the surface, for an angle
   two dozen turns from zero, and for a MountainCar position well outside the track.
2. **Each per-edge ``KMAX_*`` bounds that edge's step count.** Too small and the Wu pass silently
   drops fringe pixels; the first version of the patch had Acrobot's and Pendulum's tuples rotated
   by one, which cost exactly that and was caught by the digest replay rather than by reasoning.
3. **A disc drawn as its own offset stencil paints the same pixels as the windowed predicate.**

Each is checked here by rendering the same states twice: once as the code does it, and once with the
bound deliberately loosened — the window forced to the whole surface, ``kmax`` forced to a generous
constant, the disc forced back through ``fill_circle_mask``. If a bound were too tight the two would
differ, and the difference would be the pixels it truncated.

CPU, seconds, no GPU: these are properties of the geometry, not of a device.
"""
from __future__ import annotations

import numpy as np
import pytest

torch = pytest.importorskip("torch")

import stable_envs as se
from stable_envs.envs import classic_control as CC

CLASSIC = ("Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum")

#: Far outside anything ``sample_states`` draws. CartPole's cart leaves the 600 px surface at
#: |x| > 2.5 and these reach 9; the angles are 24 turns wide; MountainCar's positions put the car
#: off both ends of the track. Fixed, not random-seeded per run: a bound that fails intermittently
#: is a bound that fails.
def _extreme(name, n, rng):
    if name == "CartPole":
        return np.stack([rng.uniform(-9.0, 9.0, n), rng.uniform(-24 * np.pi, 24 * np.pi, n)],
                        1).astype(np.float32)
    if name == "Pendulum":
        return np.stack([rng.uniform(-24 * np.pi, 24 * np.pi, n), rng.uniform(-2.0, 2.0, n)],
                        1).astype(np.float32)
    if name == "Acrobot":
        return rng.uniform(-24 * np.pi, 24 * np.pi, (n, 2)).astype(np.float32)
    return rng.uniform(-6.0, 5.0, (n, 1)).astype(np.float32)


def _states(name, n, seed):
    return _extreme(name, n, np.random.default_rng(seed))


def _render(env, states):
    return env.render(states).clone()


@pytest.fixture
def unbounded(monkeypatch):
    """Force every primitive to the loosest form: whole surface, generous ``kmax``, no disc stencil.

    Patched at the two module-level helpers ``_draw`` goes through, so no env's ``_draw`` has to be
    duplicated here and a new env picks the check up for free.
    """
    KMAX = 1024

    def poly(cv, verts, color, bidx, kmax, win=None):
        from stable_envs.render import gfxdraw as G
        G.polygon(cv.use(color), verts, None, None, bidx, KMAX, True, window=None)

    def circ(cv, cx, cy, r, color, bidx, gfx):
        from stable_envs.render import gfxdraw as G
        G.circle(cv.use(color), cx, cy, r, None, None, bidx, gfx.hw(r), gfx.stencil(r), True,
                 window=None, disc=None)

    monkeypatch.setattr(CC, "_poly", poly)
    monkeypatch.setattr(CC, "_circ", circ)
    return KMAX


@pytest.mark.parametrize("name", CLASSIC)
def test_the_windows_and_kmax_are_bounds_on_extreme_states(name, unbounded, monkeypatch):
    """One state list, drawn bounded and unbounded. A too-tight bound shows up as lost pixels."""
    if name not in se.REGISTRY:
        pytest.skip(f"{name} is not registered in this interpreter")
    states = _states(name, 24, seed=20260913)

    env = se.make(name, batch=len(states), device="cpu")
    loose = _render(env, states)
    monkeypatch.undo()                      # put the real _poly/_circ back
    env2 = se.make(name, batch=len(states), device="cpu")
    tight = _render(env2, states)

    d = (tight.to(torch.int16) - loose.to(torch.int16)).abs()
    n = int((d.amax(-1) > 0).sum())
    assert n == 0, (
        f"{name}: the bounded primitives differ from the unbounded ones on {n:,d} of "
        f"{d[..., 0].numel():,d} pixels over states outside the reachable set (max |diff| "
        f"{int(d.max())}). Either a WIN_* window is not a superset of its primitive's footprint for "
        f"every state, or a per-edge KMAX_* is under that edge's step count, or the disc stencil "
        f"paints a different set than fill_circle_mask.")


#: Which envs an extreme state can push OFF the surface, and what that exercises. Pendulum is not
#: one of them and cannot be: its rod pivots about the centre of a 500 px canvas and reaches 114 px,
#: so no angle takes it near a border. Its extreme axis is the ANGLE itself, whose only effect on
#: the frame is which integer each vertex truncates to — so the guard there is that wildly wrapped
#: angles still move the picture, not that they leave it.
LEAVES_SURFACE = {
    "CartPole": "|x| > 2.5 drives the cart off a 600 px surface, engaging clipLine",
    "Acrobot": "both links extended reach 228 px from a centre 250 px from the border",
    "MountainCar": "positions outside the track put the car past either end",
    "MountainCarContinuous": "as MountainCar",
}


@pytest.mark.parametrize("name", CLASSIC)
def test_the_extreme_states_actually_exercise_something(name):
    """The guard on the guard: states that changed nothing would make the bound check vacuous."""
    if name not in se.REGISTRY:
        pytest.skip(f"{name} is not registered in this interpreter")
    states = _states(name, 24, seed=20260913)
    env = se.make(name, batch=len(states), device="cpu")
    frames = _render(env, states)
    H, W = env.native
    # Not "most frames differ": at these ranges the moving primitive is off the surface for most
    # states, which is the case under test — many frames collapse onto the same static furniture by
    # design. What must hold is that the set contains both kinds.
    distinct = len({f.numpy().tobytes() for f in frames})
    assert distinct >= 3, (
        f"{name}: {distinct} distinct frames from {len(states)} extreme states; they are not "
        f"moving the picture, so the bound check above is comparing one scene with itself")
    if name not in LEAVES_SURFACE:
        return
    bg = frames[:, 0, 0, :].to(torch.int16)
    ink = (frames.to(torch.int16) - bg.view(-1, 1, 1, 3)).abs().amax(-1) > 0
    touches_edge = (ink[:, 0, :].any(1) | ink[:, H - 1, :].any(1)
                    | ink[:, :, 0].any(1) | ink[:, :, W - 1].any(1))
    assert bool(touches_edge.any()), (
        f"{name}: no extreme state drew anything on the border, although {LEAVES_SURFACE[name]}. "
        f"SDL_gfx's clipping and the window clamps are therefore not being exercised.")


def test_a_deliberately_short_kmax_is_caught(monkeypatch):
    """The check has to be able to fail. Shrink one bound and watch the pixels move.

    Without this, a window/kmax test that silently stopped comparing anything would read as a pass
    forever — which is the failure mode the Acrobot tuple rotation slipped through on the first try.
    """
    if "CartPole" not in se.REGISTRY:
        pytest.skip("CartPole is not registered in this interpreter")
    states = _states("CartPole", 8, seed=1)
    good = _render(se.make("CartPole", batch=len(states), device="cpu"), states)
    monkeypatch.setattr(CC.CartPole, "KMAX_POLE", (8, 8, 8, 8))
    bad = _render(se.make("CartPole", batch=len(states), device="cpu"), states)
    assert not torch.equal(good, bad), (
        "shrinking CartPole's pole kmax to 8 changed no pixels, so the comparison in this file is "
        "not sensitive to the bound it claims to check")
