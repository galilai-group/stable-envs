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
    """Solve MuJoCo's transformed catenary equation with matching Newton/backtracking steps."""
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
    """Reproduce mjv_catenary points in float64.

    Return two points when taut, three for the near-vertical approximation, or
    npoints otherwise. Interior spacing divides by npoints, not npoints-1, to
    match MuJoCo. Keep scalar libm calls and arithmetic order for numeric parity.
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
    # Keep scalar math.exp calls for libm parity; vectorize only point assembly in the same order.
    if npoints > 2:
        hor = np.arange(1, npoints - 1, dtype=np.float64) * h / npoints
        w = np.fromiter((bh * _cosh_sinh((hh - h_off) / bh)[0] + v_off for hh in hor), np.float64, npoints - 2)
        pts[1 : npoints - 1] = x0 + ac * hor[:, None] + up * w[:, None]
    return pts


def _chain_capsules(pts: np.ndarray, nseg: int):
    """Split each polyline segment into collinear capsules; return poses and half-lengths.

    Require nseg to be divisible by the input segment count. Only the rotation
    axis matters for an untextured capsule; its orthonormal completion is arbitrary.
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
    # Complete the capsule axis using the least-aligned world axis.
    ref = np.zeros_like(z)
    ref[np.arange(len(z)), np.argmin(np.abs(z), axis=1)] = 1.0
    x = np.cross(ref, z)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), _MJMINVAL)
    y = np.cross(z, x)
    return 0.5 * (lo + hi), np.stack([x, y, z], axis=2), half


def _chain_capsules_batch(pts: np.ndarray, nseg: int):
    """Convert polylines (M, K+1, 3) to fixed-size capsule chains with the scalar arithmetic.

    Return positions (M, nseg, 3), rotations (M, nseg, 3, 3), and half-lengths
    (M, nseg). Require nseg to be divisible by K.
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
    """Geometry and constraint parameters for one spatial tendon."""

    __slots__ = ("id", "radius", "nseg", "ncat", "grey", "constraint", "upper", "gravity")


def _find_tendon(m) -> _Tendon | None:
    """Find one two-site, length-limited spatial tendon; return None for joint-only tendons.

    Reject multiple spatial tendons and unsupported wrapping configurations.
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
        # Other wrapping paths need a different polyline construction.
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
    """Return capsule positions, rotations, half-lengths, and RGB after mj_forward."""
    pts, rgb = _tendon_polyline(m, d, td)
    pos, mat, half = _chain_capsules(pts, td.nseg)
    return pos, mat, half, rgb


def _tendon_polyline(m, d, td: _Tendon):
    """Return owned polyline points and RGB after mj_forward.

    Slack tendons use a catenary; taut tendons use wrap endpoints and interpolate
    constraint color by efc_KBIP impedance. Point count varies by branch, so callers
    group polylines before batching capsule construction.
    """
    import mujoco

    adr, num = int(d.ten_wrapadr[td.id]), int(d.ten_wrapnum[td.id])
    w = np.asarray(d.wrap_xpos, np.float64).reshape(-1, 3)[adr : adr + num]
    if float(d.ten_length[td.id]) < td.upper:
        return catenary(w[0], w[1], td.gravity, td.upper, td.ncat), td.grey
    # Copy the live wrap buffer: the next row's mj_forward would overwrite a retained view.
    pts = np.array(w[:2])
    imp = 0.0
    if int(d.nefc):
        et = np.asarray(d.efc_type, np.int64)
        ei = np.asarray(d.efc_id, np.int64)
        hit = np.flatnonzero((et == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON)) & (ei == td.id))
        if len(hit):
            imp = float(np.asarray(d.efc_KBIP)[hit[0], 2])
    return pts, (1.0 - imp) * td.grey + imp * td.constraint
