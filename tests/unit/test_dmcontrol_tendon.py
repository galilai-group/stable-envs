"""The batched tendon FK must be the IDENTITY, not an approximation of it.

``DMControlSuite._fk`` used to build ``ball_in_cup``'s 28 tendon capsules once per environment: at
batch 256 on a 3090 that was 512 ``numpy.cross`` and 768 ``numpy.linalg.norm`` dispatches on
``(28,3)`` arrays, 46% of ``_fk``, and ``_fk`` was 90.7% of the render while the card sat 5.3%
utilised. The fix gives that arithmetic a batch axis and vectorises ``catenary``'s interior-point
loop.

Both are claims of BIT-IDENTITY, and a ray-caster is where that matters: a one-ulp change in a
capsule pose enters the ray quadratic as ``disc = b*b - a*c``, which on a silhouette ray amplifies it
by ~1e3-1e4, and ``disc > 0`` is a HIT/MISS decision that flips a whole pixel (see
``stable_envs/render/raycast3d.py``'s note on ``_mv``). So the pre-optimisation expressions are
written out below and the new ones are held to them exactly — ``==``, not ``allclose``.
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
    # A machine with no usable offscreen GL raises RuntimeError out of mujoco's gl_context at import
    # (``invalid value for environment variable MUJOCO_GL: egl`` on a laptop), which importorskip does
    # not catch -- so the whole file used to ERROR at collection there.
    #
    # NARROW ON PURPOSE. Catching every exception would mean that a dmcontrol.py broken by an edit --
    # a SyntaxError, a NameError, a bad module-level constant -- SKIPS the bit-identity tests instead
    # of failing them, and these are the tests standing between a one-ulp change in a capsule pose and
    # a silently different picture. Anything that is not a missing GL context must reach the runner.
    if "MUJOCO_GL" not in str(exc) and "GL" not in str(exc):
        raise
    pytest.skip(f"no usable offscreen GL context here ({exc})", allow_module_level=True)


def _catenary_reference(x0, x1, gravity, length: float, npoints: int) -> np.ndarray:
    """``dmcontrol.catenary`` exactly as it stood before the interior points were vectorised.

    Only the final loop differs from the shipped function; everything above it is called from the
    module so that this stays a test of the change and not a second implementation of MuJoCo's solve.
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
    for i in range(1, npoints - 1):  # THE LOOP THE CHANGE REPLACED
        hor = i * h / npoints
        pts[i] = x0 + ac * hor + up * (bh * D._cosh_sinh((hor - h_off) / bh)[0] + v_off)
    return pts


def _chain_reference(polys, nseg):
    """``_chain_capsules`` per polyline, i.e. the loop ``_chain_capsules_batch`` replaced."""
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
    """The vectorised interior points must equal the scalar loop's, pattern for pattern."""
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
    """Same arithmetic with one more leading axis — asserted, not argued."""
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


def test_fk_is_unchanged_on_the_frozen_ball_in_cup_states():
    """End to end on the frozen Step-4 corpus: the batched ``_fk`` must equal the per-env one.

    The reference is built by putting the pre-optimisation callables back, so this compares the two
    code paths through the same MuJoCo, on the states Step 4 certified.
    """
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

    # The two callables are substituted WHERE THEY ARE LOOKED UP, which since the package
    # split is the module that calls each: ``suite`` calls ``_chain_capsules_batch`` and
    # ``_tendon._tendon_polyline`` calls ``catenary``. Patching the package attribute instead would
    # leave the real implementations running on both sides and this test would pass vacuously, so
    # the substitution is COUNTED and the count is asserted.
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
    assert calls["chain"] and calls["cat"], (
        f"the reference implementations were never called ({calls}), so the comparison below would "
        f"be the batched path against itself"
    )

    for name, g, w in zip(("xpos", "xmat", "size", "rgb"), got, want):
        g, w = g.numpy(), w.numpy()
        assert np.array_equal(g.view(np.uint32), w.view(np.uint32)), (
            f"_fk's {name} differs by {np.abs(g - w).max():.3e} over 64 frozen states — the batched "
            f"tendon build is not the identity it claims to be"
        )
