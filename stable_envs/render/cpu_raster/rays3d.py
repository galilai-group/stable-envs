"""Numba twins of the ``raycast3d`` intersectors that test each ray packet against its overlapping primitives.

Every intersection repeats the float32 operation order of the Torch intersectors, so a hit is the same
float as there. Culling only skips primitives whose inflated bounding sphere cannot meet a packet's rays.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from ..primitives import BOX, CAPSULE, CYLINDER, ELLIPSOID, SPHERE

F32 = np.float32
INF = F32(1e30)
TINY = F32(1e-12)
CLAMP = F32(1e6)
ZERO = F32(0.0)
ONE = F32(1.0)
#: Rays per packet of a ray list; a packet shares one primitive candidate list.
PACKET = 16
#: Pixels along each side of a camera-ray tile.
TILE = 8
# Bound slack: relative, absolute and distance-scaled (float32 cancellation at long range).
_REL, _ABS, _NOISE, _ANGLE = 1.01, 1e-6, 16.0 * 1.1920929e-7, 1e-5


@njit(cache=True, error_model="numpy", inline="always")
def _mv(r00, r01, r02, r10, r11, r12, r20, r21, r22, x, y, z):
    return (
        r00 * x + r01 * y + r02 * z,
        r10 * x + r11 * y + r12 * z,
        r20 * x + r21 * y + r22 * z,
    )


@njit(cache=True, error_model="numpy", inline="always")
def _mtv(r00, r01, r02, r10, r11, r12, r20, r21, r22, x, y, z):
    return (
        r00 * x + r10 * y + r20 * z,
        r01 * x + r11 * y + r21 * z,
        r02 * x + r12 * y + r22 * z,
    )


@njit(cache=True, error_model="numpy", inline="always")
def _unit(x, y, z):
    n = math.sqrt(x * x + y * y + z * z)
    n = max(n, TINY)
    return x / n, y / n, z / n


@njit(cache=True, error_model="numpy")
def hit(kind, s0, s1, s2, px, py, pz, r00, r01, r02, r10, r11, r12, r20, r21, r22, ox, oy, oz, dx, dy, dz, normal):
    """Return (t, nx, ny, nz) of one ray and one primitive of size s, position p and rotation r; t is INF on a miss."""
    nx = ZERO
    ny = ZERO
    nz = ZERO
    if kind == SPHERE:
        cx = ox - px
        cy = oy - py
        cz = oz - pz
        bq = cx * dx + cy * dy + cz * dz
        c = (cx * cx + cy * cy + cz * cz) - s0 * s0
        disc = bq * bq - c
        sq = math.sqrt(max(disc, ZERO))
        t = -bq - sq
        if not t > ZERO:
            t = -bq + sq
        if not (disc > ZERO and t > ZERO):
            t = INF
        if normal:
            tc = min(t, CLAMP)
            nx = (cx + tc * dx) / s0
            ny = (cy + tc * dy) / s0
            nz = (cz + tc * dz) / s0
        return t, nx, ny, nz
    ox, oy, oz = _mtv(r00, r01, r02, r10, r11, r12, r20, r21, r22, ox - px, oy - py, oz - pz)
    dx, dy, dz = _mtv(r00, r01, r02, r10, r11, r12, r20, r21, r22, dx, dy, dz)
    if kind == CAPSULE or kind == CYLINDER:
        a = dx * dx + dy * dy
        bq = ox * dx + oy * dy
        c = (ox * ox + oy * oy) - s0 * s0
        disc = bq * bq - a * c
        sq = math.sqrt(max(disc, ZERO))
        best = INF
        side = False
        for sgn in range(2):
            t = (-bq - sq if sgn == 0 else -bq + sq) / max(a, TINY)
            z = oz + t * dz
            if disc > ZERO and a > TINY and t > ZERO and abs(z) <= s1 and t < best:
                best = t
                side = True
        if kind == CAPSULE:
            for cap in range(2):
                zc = -s1 if cap == 0 else s1
                qx = ox - ZERO
                qy = oy - ZERO
                qz = oz - zc
                bb = qx * dx + qy * dy + qz * dz
                cc = (qx * qx + qy * qy + qz * qz) - s0 * s0
                dd = bb * bb - cc
                s2q = math.sqrt(max(dd, ZERO))
                for sgn in range(2):
                    t = -bb - s2q if sgn == 0 else -bb + s2q
                    if dd > ZERO and t > ZERO and t < best:
                        best = t
            if normal:
                tc = min(best, CLAMP)
                px = ox + tc * dx
                py = oy + tc * dy
                pz = oz + tc * dz
                zc = max(min(pz, s1), -s1)
                lx, ly, lz = _unit(px - ZERO, py - ZERO, pz - zc)
                nx, ny, nz = _mv(r00, r01, r02, r10, r11, r12, r20, r21, r22, lx, ly, lz)
            return best, nx, ny, nz
        capz = ZERO
        flat = abs(dz) > TINY
        den = dz if flat else TINY
        for cap in range(2):
            zc = -s1 if cap == 0 else s1
            t = (zc - oz) / den
            px = ox + t * dx
            py = oy + t * dy
            if t > ZERO and px * px + py * py <= s0 * s0 and flat and t < best:
                best = t
                side = False
                capz = -ONE if cap == 0 else ONE
        if normal:
            tc = min(best, CLAMP)
            if side:
                lx = (ox + tc * dx) / s0
                ly = (oy + tc * dy) / s0
                lz = ZERO / s0
            else:
                lx = ZERO
                ly = ZERO
                lz = capz
            nx, ny, nz = _mv(r00, r01, r02, r10, r11, r12, r20, r21, r22, lx, ly, lz)
        return best, nx, ny, nz
    if kind == BOX:
        ix = ONE / (TINY if abs(dx) < TINY else dx)
        iy = ONE / (TINY if abs(dy) < TINY else dy)
        iz = ONE / (TINY if abs(dz) < TINY else dz)
        ax = (-s0 - ox) * ix
        bx = (s0 - ox) * ix
        ay = (-s1 - oy) * iy
        by = (s1 - oy) * iy
        az = (-s2 - oz) * iz
        bz = (s2 - oz) * iz
        tmin = max(max(min(ax, bx), min(ay, by)), min(az, bz))
        tmax = min(min(max(ax, bx), max(ay, by)), max(az, bz))
        ok = tmax >= max(tmin, ZERO)
        t = tmin if tmin > ZERO else tmax
        if not (ok and t > ZERO):
            t = INF
        if normal:
            tc = min(t, CLAMP)
            px = ox + tc * dx
            py = oy + tc * dy
            pz = oz + tc * dz
            rx = abs(px) / max(s0, TINY)
            ry = abs(py) / max(s1, TINY)
            rz = abs(pz) / max(s2, TINY)
            lx = ZERO
            ly = ZERO
            lz = ZERO
            if ry > rx and ry >= rz:
                ly = ONE if py > ZERO else (-ONE if py < ZERO else ZERO)
            elif rz > rx and rz > ry:
                lz = ONE if pz > ZERO else (-ONE if pz < ZERO else ZERO)
            else:
                lx = ONE if px > ZERO else (-ONE if px < ZERO else ZERO)
            nx, ny, nz = _mv(r00, r01, r02, r10, r11, r12, r20, r21, r22, lx, ly, lz)
        return t, nx, ny, nz
    if kind == ELLIPSOID:
        ix = ONE / max(s0, TINY)
        iy = ONE / max(s1, TINY)
        iz = ONE / max(s2, TINY)
        ux = ox * ix
        uy = oy * iy
        uz = oz * iz
        vx = dx * ix
        vy = dy * iy
        vz = dz * iz
        a = vx * vx + vy * vy + vz * vz
        bq = ux * vx + uy * vy + uz * vz
        c = ux * ux + uy * uy + uz * uz - ONE
        disc = bq * bq - a * c
        sq = math.sqrt(max(disc, ZERO))
        a_safe = max(a, TINY)
        t = (-bq - sq) / a_safe
        if not t > ZERO:
            t = (-bq + sq) / a_safe
        if not (disc > ZERO and t > ZERO and a > TINY and s0 > ZERO and s1 > ZERO and s2 > ZERO):
            t = INF
        if normal:
            tc = min(t, CLAMP)
            lx, ly, lz = _unit((ox + tc * dx) * ix * ix, (oy + tc * dy) * iy * iy, (oz + tc * dz) * iz * iz)
            nx, ny, nz = _mv(r00, r01, r02, r10, r11, r12, r20, r21, r22, lx, ly, lz)
        return t, nx, ny, nz
    return INF, nx, ny, nz


@njit(cache=True, error_model="numpy", inline="always")
def _hit(kinds, S, P, R, b, k, ox, oy, oz, dx, dy, dz, normal):
    """``hit`` of primitive k of row b; scalar arguments keep array reference counting out of the hot call."""
    return hit(
        kinds[k],
        S[b, k, 0],
        S[b, k, 1],
        S[b, k, 2],
        P[b, k, 0],
        P[b, k, 1],
        P[b, k, 2],
        R[b, k, 0, 0],
        R[b, k, 0, 1],
        R[b, k, 0, 2],
        R[b, k, 1, 0],
        R[b, k, 1, 1],
        R[b, k, 1, 2],
        R[b, k, 2, 0],
        R[b, k, 2, 1],
        R[b, k, 2, 2],
        ox,
        oy,
        oz,
        dx,
        dy,
        dz,
        normal,
    )


@njit(cache=True, error_model="numpy", inline="always")
def _radius(kind, S, b, k):
    s0 = float(S[b, k, 0])
    s1 = float(S[b, k, 1])
    s2 = float(S[b, k, 2])
    if kind == SPHERE:
        return s0
    if kind == CAPSULE or kind == CYLINDER:
        return s0 + s1
    if kind == BOX:
        return math.sqrt(s0 * s0 + s1 * s1 + s2 * s2)
    return max(max(s0, s1), s2)


@njit(cache=True, error_model="numpy", inline="always")
def _overlaps(cx, cy, cz, spread, ax, ay, az, angle, kind, S, P, b, k):
    """Whether rays from within ``spread`` of c, inside the cone (axis, angle), can meet primitive k."""
    vx = float(P[b, k, 0]) - cx
    vy = float(P[b, k, 1]) - cy
    vz = float(P[b, k, 2]) - cz
    dist2 = vx * vx + vy * vy + vz * vz
    r = _radius(kind, S, b, k) * _REL + _ABS
    r = math.sqrt(r * r + _NOISE * dist2) + spread
    if dist2 <= r * r:
        return True
    dist = math.sqrt(dist2)
    cos_phi = min(1.0, max(-1.0, (vx * ax + vy * ay + vz * az) / dist))
    return math.acos(cos_phi) <= angle + math.asin(min(1.0, r / dist)) + _ANGLE


@njit(cache=True, error_model="numpy")
def _packet_bounds(o, d, a, e):
    """Return the bounding sphere (centre, radius) of origins and the direction cone (axis, half-angle)."""
    cx = 0.0
    cy = 0.0
    cz = 0.0
    ax = 0.0
    ay = 0.0
    az = 0.0
    for q in range(a, e):
        cx += o[q, 0]
        cy += o[q, 1]
        cz += o[q, 2]
        x = float(d[q, 0])
        y = float(d[q, 1])
        z = float(d[q, 2])
        n = math.sqrt(x * x + y * y + z * z)
        if n > 0:
            ax += x / n
            ay += y / n
            az += z / n
    m = e - a
    cx /= m
    cy /= m
    cz /= m
    spread = 0.0
    n = math.sqrt(ax * ax + ay * ay + az * az)
    wide = n < 1e-9
    if not wide:
        ax /= n
        ay /= n
        az /= n
    # The widest ray has the smallest cosine to the axis; one arccosine then gives the half-angle.
    lowest = 1.0
    for q in range(a, e):
        x = o[q, 0] - cx
        y = o[q, 1] - cy
        z = o[q, 2] - cz
        spread = max(spread, math.sqrt(x * x + y * y + z * z))
        if not wide:
            x = float(d[q, 0])
            y = float(d[q, 1])
            z = float(d[q, 2])
            dn = math.sqrt(x * x + y * y + z * z)
            if dn > 0:
                lowest = min(lowest, (x * ax + y * ay + z * az) / dn)
    angle = math.pi if wide else math.acos(min(1.0, max(-1.0, lowest)))
    return cx, cy, cz, spread * (1.0 + 1e-6) + 1e-7, ax, ay, az, angle


@njit(cache=True, error_model="numpy")
def packet_bounds(o, d, starts):
    """Per-packet origin spheres (P, 4) and direction cones (P, 4) of rays split at ``starts``."""
    n = len(starts) - 1
    sphere = np.empty((n, 4))
    cone = np.empty((n, 4))
    for p in range(n):
        cx, cy, cz, sp, ax, ay, az, ang = _packet_bounds(o, d, starts[p], starts[p + 1])
        sphere[p, 0] = cx
        sphere[p, 1] = cy
        sphere[p, 2] = cz
        sphere[p, 3] = sp
        cone[p, 0] = ax
        cone[p, 1] = ay
        cone[p, 2] = az
        cone[p, 3] = ang
    return sphere, cone


@njit(cache=True, error_model="numpy")
def packets(env, size):
    """Split rays sorted by row into runs of at most ``size`` rays of one row; return run starts."""
    n = len(env)
    out = np.empty(n + 1, np.int64)
    m = 0
    run = 0
    for q in range(n):
        if q == 0 or env[q] != env[q - 1] or run == size:
            out[m] = q
            m += 1
            run = 0
        run += 1
    out[m] = n
    return out[: m + 1]


@njit(cache=True, error_model="numpy")
def tile_cones(dirs, H, W, tile):
    """Direction cones (T, 4) of the camera-ray tiles of an H x W grid, row-major tiles."""
    tx = (W + tile - 1) // tile
    ty = (H + tile - 1) // tile
    out = np.empty((tx * ty, 4))
    idx = np.empty(tile * tile, np.int64)
    for t in range(tx * ty):
        x0 = (t % tx) * tile
        y0 = (t // tx) * tile
        m = 0
        for y in range(y0, min(y0 + tile, H)):
            for x in range(x0, min(x0 + tile, W)):
                idx[m] = y * W + x
                m += 1
        sub = dirs[idx[:m]]
        _, _, _, _, ax, ay, az, ang = _packet_bounds(sub, sub, 0, m)
        out[t, 0] = ax
        out[t, 1] = ay
        out[t, 2] = az
        out[t, 3] = ang
    return out


@njit(cache=True, error_model="numpy", inline="always")
def _candidates(kinds, S, P, b, cx, cy, cz, spread, ax, ay, az, angle, cand):
    nc = 0
    for k in range(len(kinds)):
        if _overlaps(cx, cy, cz, spread, ax, ay, az, angle, kinds[k], S, P, b, k):
            cand[nc] = k
            nc += 1
    return nc


@njit(cache=True, error_model="numpy", inline="always")
def _near(kind, S, P, b, k, ox, oy, oz, dx, dy, dz):
    """Whether one ray can meet primitive k's inflated bounding sphere; float64, so it never rejects a hit."""
    vx = float(P[b, k, 0]) - float(ox)
    vy = float(P[b, k, 1]) - float(oy)
    vz = float(P[b, k, 2]) - float(oz)
    ex = float(dx)
    ey = float(dy)
    ez = float(dz)
    d2 = ex * ex + ey * ey + ez * ez
    if d2 == 0.0:
        return True
    r = _radius(kind, S, b, k) * _REL + _ABS
    r2 = r * r + _NOISE * (vx * vx + vy * vy + vz * vz)
    cx = vy * ez - vz * ey
    cy = vz * ex - vx * ez
    cz = vx * ey - vy * ex
    if cx * cx + cy * cy + cz * cz > r2 * d2:
        return False
    return vx * ex + vy * ey + vz * ez + math.sqrt(r2 * d2) >= 0.0


@njit(cache=True, error_model="numpy", inline="always")
def _nearest(kinds, S, P, R, b, cand, nc, ox, oy, oz, dx, dy, dz):
    """Nearest hit among the candidates; only the nearest primitive computes its normal."""
    best = INF
    bi = -1
    for c in range(nc):
        k = cand[c]
        if not _near(kinds[k], S, P, b, k, ox, oy, oz, dx, dy, dz):
            continue
        t = _hit(kinds, S, P, R, b, k, ox, oy, oz, dx, dy, dz, False)[0]
        if t < best:
            best = t
            bi = k
    if bi < 0:
        return best, ZERO, ZERO, ZERO, 0
    _, nx, ny, nz = _hit(kinds, S, P, R, b, bi, ox, oy, oz, dx, dy, dz, True)
    return best, nx, ny, nz, bi


@njit(cache=True, error_model="numpy", inline="always")
def _blocked(kinds, S, P, R, b, cand, nc, ox, oy, oz, dx, dy, dz, lim, skip):
    for c in range(nc):
        k = cand[c]
        if k == skip or not _near(kinds[k], S, P, b, k, ox, oy, oz, dx, dy, dz):
            continue
        t = _hit(kinds, S, P, R, b, k, ox, oy, oz, dx, dy, dz, False)[0]
        if t > ZERO and t < lim:
            return True
    return False


@njit(parallel=True, cache=True, error_model="numpy")
def cast_rays(kinds, S, P, R, env, o, d, starts, t_out, n_out, i_out):
    """Nearest hits of a ray list sorted by row ``env``, packets split at ``starts``."""
    for p in prange(len(starts) - 1):
        a = starts[p]
        e = starts[p + 1]
        b = env[a]
        cx, cy, cz, sp, ax, ay, az, ang = _packet_bounds(o, d, a, e)
        cand = np.empty(len(kinds), np.int64)
        nc = _candidates(kinds, S, P, b, cx, cy, cz, sp, ax, ay, az, ang, cand)
        for q in range(a, e):
            t, nx, ny, nz, i = _nearest(
                kinds, S, P, R, b, cand, nc, o[q, 0], o[q, 1], o[q, 2], d[q, 0], d[q, 1], d[q, 2]
            )
            t_out[q] = t
            n_out[q, 0] = nx
            n_out[q, 1] = ny
            n_out[q, 2] = nz
            i_out[q] = i


@njit(parallel=True, cache=True, error_model="numpy")
def blocked_rays(kinds, S, P, R, env, o, d, lim, skip, starts, out):
    """Whether a primitive other than ``skip`` meets each ray with 0 < t < lim; rays sorted by row."""
    for p in prange(len(starts) - 1):
        a = starts[p]
        e = starts[p + 1]
        b = env[a]
        cx, cy, cz, sp, ax, ay, az, ang = _packet_bounds(o, d, a, e)
        cand = np.empty(len(kinds), np.int64)
        nc = _candidates(kinds, S, P, b, cx, cy, cz, sp, ax, ay, az, ang, cand)
        for q in range(a, e):
            out[q] = _blocked(
                kinds, S, P, R, b, cand, nc, o[q, 0], o[q, 1], o[q, 2], d[q, 0], d[q, 1], d[q, 2], lim[q], skip[q]
            )
