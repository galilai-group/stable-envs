"""Batched tendon kinematics is bit-identical to a scalar per-polyline reference.

A one-ulp change in a capsule pose enters the ray quadratic as ``disc = b*b - a*c`` and flips a
hit/miss decision on silhouette rays, so every comparison here is exact (``==``), not ``allclose``.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

try:
    import stable_envs.envs.dmcontrol as D
except ImportError as exc:  # dm_control / mujoco absent: legitimately skippable
    pytest.skip(
        f"stable_envs.envs.dmcontrol needs dm_control ({exc}); this file is about its tendon "
        f"arithmetic and has nothing to assert without it",
        allow_module_level=True,
    )
except RuntimeError as exc:
    # Narrow on purpose: only a missing GL context skips; any other import failure must fail the file.
    if "GL" not in str(exc):
        raise
    pytest.skip(f"no usable offscreen GL context here ({exc})", allow_module_level=True)


def _catenary_reference(x0, x1, gravity, length: float, npoints: int) -> np.ndarray:
    """``catenary`` with a scalar loop over the interior points.

    Only the final loop differs from the shipped function; the solve above it is the module's own.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    d = x0 - x1
    if math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) > length:
        return np.stack([x0, x1])
    up = np.asarray(gravity, dtype=np.float64) * -1.0
    n = math.sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2])
    up = np.array([1.0, 0.0, 0.0]) if n < D._MJMINVAL else up * (1.0 / n)
    x01 = x1 - x0
    ac = x01 - up * (up[0] * x01[0] + up[1] * x01[1] + up[2] * x01[2])
    n = math.sqrt(ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2])
    ac = np.zeros(3) if n < D._MJMINVAL else ac * (1.0 / n)
    h = x01[0] * ac[0] + x01[1] * ac[1] + x01[2] * ac[2]
    v = x01[0] * up[0] + x01[1] * up[1] + x01[2] * up[2]
    if length > 100 * h:
        d_up = -0.5 * (math.sqrt(length * length - h * h) - v)
        return np.stack([x0, x0 + up * d_up + ac * (h * d_up / (2 * d_up - v)), x1])
    bh = D._solve_catenary(v, h, length) * h
    h_off = -0.5 * (math.log((length + v) / (length - v)) * bh - h)
    v_off = -D._cosh_sinh(h_off / bh)[0] * bh
    pts = np.empty((npoints, 3))
    pts[0], pts[npoints - 1] = x0, x1
    for i in range(1, npoints - 1):
        hor = i * h / npoints
        pts[i] = x0 + ac * hor + up * (bh * D._cosh_sinh((hor - h_off) / bh)[0] + v_off)
    return pts


def _chain_reference(polys, nseg):
    """``_chain_capsules`` applied per polyline."""
    out = [D._chain_capsules(p, nseg) for p in polys]
    return (np.stack([o[0] for o in out]), np.stack([o[1] for o in out]), np.stack([o[2] for o in out]))


#: A sweep that reaches all three branches of ``mjv_catenary``: slack (the catenary proper),
#: near-vertical (``length > 100*h``, three points) and taut (``dist > length``, two points).
#: ball_in_cup's own numbers -- a 0.3 m string, a cup a few centimetres across.
GRAVITY = np.array([0.0, 0.0, -9.81])


def _sweep():
    rng = np.random.default_rng(20260913)
    cases = []
    for _ in range(64):
        x0 = rng.uniform(-0.2, 0.2, 3)
        for scale in (0.02, 0.15, 0.3, 0.9):
            x1 = x0 + rng.uniform(-1.0, 1.0, 3) * scale
            cases.append((x0, x1, 0.3))
    cases.append((np.zeros(3), np.array([0.0, 0.0, -0.29]), 0.3))  # near-vertical
    cases.append((np.zeros(3), np.array([0.4, 0.0, 0.0]), 0.3))  # taut
    return cases


def test_catenary_interior_points_are_bit_identical():
    """The vectorised interior points equal the scalar loop's bit for bit."""
    seen = set()
    for x0, x1, length in _sweep():
        got = D.catenary(x0, x1, GRAVITY, length, 29)
        want = _catenary_reference(x0, x1, GRAVITY, length, 29)
        seen.add(len(got))
        assert got.shape == want.shape, f"branch changed: {got.shape} vs {want.shape}"
        assert np.array_equal(got.view(np.uint64), want.view(np.uint64)), (
            f"catenary moved by {np.abs(got - want).max():.3e} on x0={x0} x1={x1}; a one-ulp change "
            f"in a capsule pose flips silhouette pixels in the ray-caster"
        )
    assert seen >= {2, 3, 29}, f"the sweep did not reach every branch, only lengths {sorted(seen)}"


def test_chain_capsules_batch_matches_the_per_polyline_version():
    """The batched capsule chain equals the per-polyline chain, in float64 and after the float32 cast."""
    nseg = 28
    for npts in (2, 3, 29):  # K = 1, 2, 28, all divisors of 28
        polys = [p for p in (D.catenary(a, b, GRAVITY, L, 29) for a, b, L in _sweep()) if len(p) == npts]
        if len(polys) < 2:  # synthesise a second so the batch axis is real
            polys = polys * 2
        polys = polys[:16]
        stacked = np.stack(polys)
        gp, gm, gh = D._chain_capsules_batch(stacked, nseg)
        wp, wm, wh = _chain_reference(polys, nseg)
        for name, g, w in (("pos", gp, wp), ("mat", gm, wm), ("half", gh, wh)):
            assert g.shape == w.shape, f"{name}: {g.shape} vs {w.shape}"
            assert np.array_equal(g.view(np.uint64), w.view(np.uint64)), (
                f"{name} differs by {np.abs(g - w).max():.3e} at npts={npts}"
            )
            gf, wf = g.astype(np.float32), w.astype(np.float32)
            assert np.array_equal(gf.view(np.uint32), wf.view(np.uint32)), (
                f"{name} differs after the float32 cast the renderer applies"
            )


def test_batched_fk_matches_per_env_fk_on_frozen_ball_in_cup_states():
    """End to end on the frozen corpus, with the scalar reference callables patched in."""
    from pathlib import Path

    import torch

    import stable_envs as se

    npz = Path(__file__).resolve().parents[2] / "tests/fixtures/states/fidelity/DMCBallInCupCatch.npz"
    if not npz.is_file():
        pytest.skip("frozen DMCBallInCupCatch corpus not present")
    try:
        env = se.make("DMCBallInCupCatch", num_envs=64, device="cpu")
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"DMCBallInCupCatch is not constructible here: {type(exc).__name__}: {exc}")
    states = torch.as_tensor(np.asarray(np.load(npz)["states"], np.float32))
    sc = env._scene(env.size)
    assert sc.tendon is not None, "this test is about the tendon path and this scene has none"

    got = env._fk(states, sc)

    # Patched where they are looked up: ``suite`` calls the batch chain, ``_tendon`` calls ``catenary``.
    # The call counts guard against patching a name the code path never reads.
    calls = {"chain": 0, "cat": 0}

    def _chain_patched(pts, nseg):
        calls["chain"] += 1
        return _chain_reference(list(pts), nseg)

    def _cat_patched(*a, **kw):
        calls["cat"] += 1
        return _catenary_reference(*a, **kw)

    real_batch, real_cat = D.suite._chain_capsules_batch, D._tendon.catenary
    try:
        D.suite._chain_capsules_batch = _chain_patched
        D._tendon.catenary = _cat_patched
        want = env._fk(states, sc)
    finally:
        D.suite._chain_capsules_batch, D._tendon.catenary = real_batch, real_cat
    assert calls["chain"] and calls["cat"], f"the reference implementations were never called ({calls})"

    for name, g, w in zip(("xpos", "xmat", "size", "rgb"), got, want):
        g, w = g.numpy(), w.numpy()
        assert np.array_equal(g.view(np.uint32), w.view(np.uint32)), (
            f"_fk's {name} differs by {np.abs(g - w).max():.3e} over 64 frozen states"
        )
