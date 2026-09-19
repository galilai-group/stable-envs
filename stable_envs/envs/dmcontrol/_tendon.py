"""MuJoCo's spatial tendon, reproduced: ``mjv_catenary`` and the capsule chain it becomes.

``ball_in_cup`` is the only shipped cell that draws one, and it is here rather than in
``ball_in_cup.py`` because this is a MUJOCO fact and not a domain declaration: any DM Control domain
with a spatial tendon needs exactly this, and the function it ports (``mjv_catenary``) is exported
from ``libmujoco`` and absent from the Python bindings, which is why it exists at all. The port is
bit-exact — 0 ULP against the C implementation over the sweep in
``tests/unit/test_dmcontrol_tendon.py``.

Patching for a test: each callable is looked up in the module that CALLS it, so
``tests/unit/test_dmcontrol_tendon.py`` substitutes ``_chain_capsules_batch`` on
``stable_envs.envs.dmcontrol.suite`` (which calls it) and ``catenary`` on this module (which
``_tendon_polyline`` calls it from). Patching the package attribute reaches neither.
"""

from __future__ import annotations

import math

import numpy as np

_MJMINVAL = 1e-15
_CAT_TOL = 1e-9


def _cosh_sinh(x: float) -> tuple[float, float]:
    e = math.exp(x)
    return 0.5 * (e + 1.0 / e), 0.5 * (e - 1.0 / e)


def _cat_residual(b: float, intercept: float, want_grad: bool):
    a = 0.5 / b
    ch, sh = _cosh_sinh(a)
    res = 1.0 / math.sqrt(2 * b * sh - 1.0) - intercept
    grad = (a * ch - sh) * math.pow(2 * b * sh - 1.0, -1.5) if want_grad else None
    return res, grad


def _solve_catenary(v: float, h: float, length: float) -> float:
    """MuJoCo's own solve: Newton on the change-of-variables form, 50 steps, tol 1e-9, with a
    backtracking halving loop of at most 10 that accepts the first step reducing ``|res|``."""
    intercept = 1.0 / math.sqrt(math.sqrt(length * length - v * v) / h - 1.0)
    b = intercept / math.sqrt(24.0)
    for _ in range(50):
        res, grad = _cat_residual(b, intercept, True)
        if abs(res) < _CAT_TOL:
            break
        step = -res / grad
        for _ in range(10):
            if abs(_cat_residual(b + step, intercept, False)[0]) < abs(res):
                break
            step *= 0.5
        b += step
    return b


def catenary(x0, x1, gravity, length: float, npoints: int) -> np.ndarray:
    """MuJoCo's ``mjv_catenary``, in numpy. ``(N,3)`` float64 with ``N <= npoints``.

    A spatial tendon with slack is not drawn as a straight line: since MuJoCo 3.2 it hangs, and
    ``dm_control``'s ``ball_in_cup`` string is 28 capsules along a catenary rather than the one
    connector the XML reads like. That is the "mover kind the ray-caster does not have" that
    ``docs/research/dmcontrol.md`` S8 recorded, and it is not an approximation problem: ``mjv_catenary``
    is exported from ``libmujoco`` but is NOT in the Python bindings, so it had to be reproduced.

    It is reproduced EXACTLY. Validated against the C symbol through ctypes on 440 inputs spanning all
    three branches and on 64 real ``ball_in_cup`` states: max absolute error 0.0, max ULP difference 0
    (``scratch/dmcontrol/probe_catenary.py``). Two things buy that and neither is stylistic —
    ``math.*`` is a direct call into the same libm MuJoCo's ``mju_*`` macros expand to, where a numpy
    ufunc is not, and the Newton loop below is MuJoCo's iteration count, tolerance, seed and
    backtracking rather than an equivalent solve.

    Three branches, all of which ``ball_in_cup`` reaches during a normal rollout:

      * ``dist > length`` -- taut, two points. (Reached from here only defensively: the CALLER decides
        slackness the way ``mjv_isCatenary`` does, from ``ten_length`` against the limit, and takes the
        wrap points rather than this branch when the tendon is taut, because MuJoCo does.)
      * ``length > 100 * h`` -- near-vertical, three points and a hanging-bead approximation rather
        than a catenary. Fires in 3 of 64 sampled states.
      * otherwise ``npoints`` points, spaced linearly in the HORIZONTAL coordinate and divided by
        ``npoints`` rather than ``npoints - 1``, so the last interior point sits at ``(N-2)/N`` of the
        span and the final capsule comes out about twice as long as the others. That reads like an
        off-by-one and it is what MuJoCo draws, which is the only thing that matters here.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    d = x0 - x1
    if math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) > length:
        return np.stack([x0, x1])
    up = np.asarray(gravity, dtype=np.float64) * -1.0
    n = math.sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2])
    up = np.array([1.0, 0.0, 0.0]) if n < _MJMINVAL else up * (1.0 / n)
    x01 = x1 - x0
    ac = x01 - up * (up[0] * x01[0] + up[1] * x01[1] + up[2] * x01[2])
    n = math.sqrt(ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2])
    ac = np.zeros(3) if n < _MJMINVAL else ac * (1.0 / n)
    h = x01[0] * ac[0] + x01[1] * ac[1] + x01[2] * ac[2]  # horizontal extent
    v = x01[0] * up[0] + x01[1] * up[1] + x01[2] * up[2]  # vertical rise
    if length > 100 * h:
        d_up = -0.5 * (math.sqrt(length * length - h * h) - v)
        return np.stack([x0, x0 + up * d_up + ac * (h * d_up / (2 * d_up - v)), x1])
    bh = _solve_catenary(v, h, length) * h
    h_off = -0.5 * (math.log((length + v) / (length - v)) * bh - h)
    v_off = -_cosh_sinh(h_off / bh)[0] * bh
    pts = np.empty((npoints, 3))
    pts[0], pts[npoints - 1] = x0, x1
    # The interior points, written as ONE numpy expression rather than as a Python loop of (3,)
    # temporaries. ``_cosh_sinh`` stays SCALAR and stays ``math.exp``: the 0-ULP validation against
    # libmujoco's ``mjv_catenary`` (scratch/dmcontrol/probe_catenary.py, 440 inputs) depends on
    # calling the same libm that MuJoCo's ``mju_*`` macros expand to, and ``np.exp`` is a different
    # implementation. What is vectorised is only the assembly, which is elementwise in the same
    # association order -- ``(x0 + ac*hor) + up*w``, componentwise -- so it is bit-identical.
    # Measured on DMCBallInCupCatch at batch 256: 24.5 ms -> ~8 ms of a 88 ms ``_fk``.
    if npoints > 2:
        hor = np.arange(1, npoints - 1, dtype=np.float64) * h / npoints
        w = np.fromiter((bh * _cosh_sinh((hh - h_off) / bh)[0] + v_off for hh in hor), np.float64, npoints - 2)
        pts[1 : npoints - 1] = x0 + ac * hor[:, None] + up * w[:, None]
    return pts


def _chain_capsules(pts: np.ndarray, nseg: int):
    """A polyline of ``K`` segments -> exactly ``nseg`` capsule poses. ``(pos, mat, half_length)``.

    The chain has to be a FIXED number of primitives, because the mover set is fixed at bake time and
    a batch holds states from all three branches at once — 28 segments when the string hangs, 2 when it
    is near-vertical, 1 when it is taut. Each real segment is therefore split into ``nseg // K``
    COLLINEAR pieces, which is exact rather than an approximation: the union of collinear capsules
    covering a segment is that segment's capsule, and the surface normal agrees too — a point on an
    interior piece's cap sits exactly on the shared axis's perpendicular, so its normal is the same
    radial vector the full capsule's cylinder gives there. Nothing is resampled and no piece is
    degenerate.

    Only the third column of the rotation is meaningful: a capsule is a surface of revolution about
    its own axis, so any orthonormal completion draws the same solid.
    """
    K = len(pts) - 1
    if nseg % K:
        raise RuntimeError(
            f"a tendon drawn as {K} segments cannot be split into {nseg} pieces "
            f"without resampling it; {nseg} must be a multiple of every branch's "
            f"segment count"
        )
    per = nseg // K
    p0 = np.asarray(pts[:-1], np.float64).reshape(K, 1, 3)
    p1 = np.asarray(pts[1:], np.float64).reshape(K, 1, 3)
    t = (np.arange(per, dtype=np.float64) / per).reshape(1, per, 1)
    lo = (p0 + (p1 - p0) * t).reshape(nseg, 3)
    hi = (p0 + (p1 - p0) * (t + 1.0 / per)).reshape(nseg, 3)
    axis = hi - lo
    half = np.linalg.norm(axis, axis=1) * 0.5
    z = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), _MJMINVAL)
    # an arbitrary orthonormal completion, taken from the world axis least aligned with z
    ref = np.zeros_like(z)
    ref[np.arange(len(z)), np.argmin(np.abs(z), axis=1)] = 1.0
    x = np.cross(ref, z)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), _MJMINVAL)
    y = np.cross(z, x)
    return 0.5 * (lo + hi), np.stack([x, y, z], axis=2), half


def _chain_capsules_batch(pts: np.ndarray, nseg: int):
    """``_chain_capsules`` with a BATCH axis. ``pts`` ``(M, K+1, 3)`` -> ``(M,nseg,3)``,
    ``(M,nseg,3,3)``, ``(M,nseg)``.

    THE WHOLE POINT IS THAT IT IS THE SAME ARITHMETIC. Every line below is the single-polyline
    version with one more leading axis: the products and differences are elementwise, ``norm`` and
    ``argmin`` reduce the SAME trailing axis of length 3, and ``cross`` is component arithmetic. A
    leading axis cannot change a per-row result, and ``tests/unit/test_dmcontrol_tendon.py`` asserts
    that against ``_chain_capsules`` on the frozen ball_in_cup states rather than leaving it as an
    argument.

    Why it exists: ``_fk`` called ``_chain_capsules`` ONCE PER ENVIRONMENT, and at batch 256 that was
    512 ``numpy.cross`` dispatches and 768 ``numpy.linalg.norm`` dispatches on ``(28,3)`` arrays --
    46% of ``_fk``, which was itself 90.7% of a DMCBallInCupCatch render on a 3090 (the card was 5.3%
    utilised). The work is trivially batched because the polylines differ only in their VALUES; they
    are grouped by segment count, of which ball_in_cup's tendon has exactly three possible values
    (1 taut, 2 near-vertical, 28 catenary).
    """
    M, npts, _ = pts.shape
    K = npts - 1
    if nseg % K:
        raise RuntimeError(
            f"a tendon drawn as {K} segments cannot be split into {nseg} pieces "
            f"without resampling it; {nseg} must be a multiple of every branch's "
            f"segment count"
        )
    per = nseg // K
    p0 = np.asarray(pts[:, :-1], np.float64).reshape(M, K, 1, 3)
    p1 = np.asarray(pts[:, 1:], np.float64).reshape(M, K, 1, 3)
    t = (np.arange(per, dtype=np.float64) / per).reshape(1, 1, per, 1)
    lo = (p0 + (p1 - p0) * t).reshape(M, nseg, 3)
    hi = (p0 + (p1 - p0) * (t + 1.0 / per)).reshape(M, nseg, 3)
    axis = hi - lo
    half = np.linalg.norm(axis, axis=-1) * 0.5
    z = axis / np.maximum(np.linalg.norm(axis, axis=-1, keepdims=True), _MJMINVAL)
    ref = np.zeros_like(z)
    np.put_along_axis(ref, np.argmin(np.abs(z), axis=-1)[..., None], 1.0, axis=-1)
    x = np.cross(ref, z)
    x /= np.maximum(np.linalg.norm(x, axis=-1, keepdims=True), _MJMINVAL)
    y = np.cross(z, x)
    return 0.5 * (lo + hi), np.stack([x, y, z], axis=-1), half


class _Tendon:
    """The one spatial tendon a scene draws, and everything a frame needs about it."""

    __slots__ = ("id", "radius", "nseg", "ncat", "grey", "constraint", "upper", "gravity")


def _find_tendon(m) -> _Tendon | None:
    """The scene's drawn spatial tendon, or ``None``. Refuses anything it cannot draw exactly.

    ``ntendon > 0`` is not the question: ``point_mass`` and ``fish`` both carry two tendons and draw
    neither, because they are FIXED tendons (joint-space couplings with no spatial path). A tendon
    reaches the frame only if it wraps SITES or wrapping objects.
    """
    import mujoco

    spatial = []
    for t in range(int(m.ntendon)):
        a = int(m.tendon_adr[t])
        n = int(m.tendon_num[t])
        kinds = [int(m.wrap_type[a + k]) for k in range(n)]
        if any(k != int(mujoco.mjtWrap.mjWRAP_JOINT) for k in kinds):
            spatial.append((t, kinds))
    if not spatial:
        return None
    if len(spatial) > 1:
        raise RuntimeError(f"this scene draws {len(spatial)} spatial tendons; the mover set here carries one")
    t, kinds = spatial[0]
    site = int(mujoco.mjtWrap.mjWRAP_SITE)
    if kinds != [site, site] or not int(m.tendon_limited[t]):
        # ``mjv_isCatenary`` needs exactly two sites and a length constraint (or a spring with a
        # deadband); anything else takes MuJoCo's wrapping path, which computes a different polyline.
        raise RuntimeError(
            f"tendon {t} wraps {kinds} and limited={int(m.tendon_limited[t])}; the "
            f"catenary path here covers a two-site, length-limited tendon only"
        )
    td = _Tendon()
    td.id = t
    td.radius = float(m.tendon_width[t])
    td.upper = float(m.tendon_range[t][1])
    td.ncat = min(int(m.vis.quality.numslices) + 1, 100)
    td.nseg = td.ncat - 1
    td.grey = np.asarray(m.tendon_rgba[t], np.float64)[:3].copy()
    td.constraint = np.asarray(m.vis.rgba.constraint, np.float64)[:3].copy()
    td.gravity = np.asarray(m.opt.gravity, np.float64).copy()
    return td


def _tendon_frame(m, d, td: _Tendon):
    """``(pos, mat, half_length, rgb)`` for this state's chain. ``mj_forward`` must have run.

    Which branch is a question about the CONSTRAINT, not about the geometry: ``mjv_isCatenary`` asks
    whether the length limit is satisfied, and only then hands the two endpoints to ``mjv_catenary``
    with the limit as the target length. Taut, MuJoCo takes a completely different path — one straight
    capsule between the wrap points, recoloured by the limit's IMPEDANCE:

        rgb = (1 - imp) * tendon_rgba + imp * vis.rgba.constraint

    which is why the string turns red when the ball pulls the cup. ``imp`` is column 2 of
    ``efc_KBIP`` on the tendon-limit row, and that is measured rather than guessed: over 88 taut
    states the four columns reproduce MuJoCo's own scene colour to 1384, 52, **2.9e-08** and 23
    respectively (``scratch/dmcontrol/probe_tendon_colour.py``). The 2.9e-08 is the float32 round-trip
    through ``mjvGeom.rgba``.

    A random policy leaves ``ball_in_cup`` taut in 35-44% of the states it visits, so this is not a
    branch a renderer can decline: skipping it would paint a third of the dataset's strings grey.
    """
    pts, rgb = _tendon_polyline(m, d, td)
    pos, mat, half = _chain_capsules(pts, td.nseg)
    return pos, mat, half, rgb


def _tendon_polyline(m, d, td: _Tendon):
    """``(pts, rgb)``: the chain's POLYLINE and its colour, for one state. ``mj_forward`` must have run.

    Split out of ``_tendon_frame`` so that the part which must run per environment (it reads ``d``,
    which holds one state at a time) is separated from the part which must NOT (``_chain_capsules``,
    which is pure array arithmetic on the polyline and batches over the whole call). ``_tendon_frame``
    is unchanged in behaviour and is what the single-polyline tests compare against.

    The polyline has two, three or ``ncat`` points depending on the branch, which is why ``_fk``
    groups the batch by point count before batching the chain.
    """
    import mujoco

    adr, num = int(d.ten_wrapadr[td.id]), int(d.ten_wrapnum[td.id])
    w = np.asarray(d.wrap_xpos, np.float64).reshape(-1, 3)[adr : adr + num]
    if float(d.ten_length[td.id]) < td.upper:
        return catenary(w[0], w[1], td.gravity, td.upper, td.ncat), td.grey
    # A COPY, not ``w[:2]``. ``np.asarray(d.wrap_xpos, np.float64)`` returns the MjData buffer itself
    # when it is already float64, so the slice is a live view; the caller now keeps polylines until
    # the whole batch has been walked, and the next environment's ``mj_forward`` would rewrite this
    # one under it. The slack branch above already returns a fresh array.
    pts = np.array(w[:2])
    imp = 0.0
    if int(d.nefc):
        et = np.asarray(d.efc_type, np.int64)
        ei = np.asarray(d.efc_id, np.int64)
        hit = np.flatnonzero((et == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)) & (ei == td.id))
        if len(hit):
            imp = float(np.asarray(d.efc_KBIP)[hit[0], 2])
    return pts, (1.0 - imp) * td.grey + imp * td.constraint
