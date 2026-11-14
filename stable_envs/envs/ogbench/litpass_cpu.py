"""Numba CPU twin of the maze light passes (lit plates and composite shader) in the float32 order of the Torch path."""

from __future__ import annotations

import contextlib

import numba
import numpy as np
from numba import njit, prange

from ..._core.cpus import numba_threads
from ...render.cpu_raster.rays3d import _hit

F32 = np.float32
_F0 = F32(0.0)
_F1 = F32(1.0)
_F255 = F32(255.0)
_TINY = F32(1e-12)


@contextlib.contextmanager
def frame_threads(frames):
    """Run the kernels of a pass over ``frames`` frames on at most one Numba thread per frame."""
    before = numba_threads()
    numba.set_num_threads(max(1, min(before, frames)))
    try:
        yield
    finally:
        numba.set_num_threads(before)


@njit(cache=True, error_model="numpy", inline="always")
def _pixel(rect, r, j, W):
    """Flat plate index of window pixel ``j`` of row ``r``; the window is (x0, y0, width) in row-major order."""
    a = j // rect[r, 2]
    return (rect[r, 1] + a) * W + rect[r, 0] + (j - a * rect[r, 2])


@njit(cache=True, error_model="numpy", inline="always")
def _dot(x, y, z):
    return x * x + y * y + z * z


@njit(cache=True, error_model="numpy", inline="always")
def _view_and_spot(eye, lp, norms, q, x, y, z):
    """Unit view vector, unit vector to the spotlight and its distance, from the norms in ``norms[:2]``."""
    n = max(norms[0, q], _TINY)
    vx, vy, vz = (eye[0] - x) / n, (eye[1] - y) / n, (eye[2] - z) / n
    dist = norms[1, q]
    n = max(dist, _TINY)
    return vx, vy, vz, (lp[0] - x) / n, (lp[1] - y) / n, (lp[2] - z) / n, dist


@njit(parallel=True, cache=True, error_model="numpy")
def squares(rect, rowq, start, W, hit, eye, lpos, norms):
    """Squared lengths of the view vector and of the vector to the spotlight, at each row's ``start``."""
    for q in prange(len(rowq)):
        r = rowq[q]
        lp = lpos[r]
        p = _pixel(rect, r, q - start[r], W)
        x, y, z = hit[p, 0], hit[p, 1], hit[p, 2]
        norms[0, q] = _dot(eye[0] - x, eye[1] - y, eye[2] - z)
        norms[1, q] = _dot(lp[0] - x, lp[1] - y, lp[2] - z)


@njit(parallel=True, cache=True, error_model="numpy")
def halves(rect, rowq, start, W, hit, shin, eye, lpos, D, to_light, norms, powers, exps):
    """Squared lengths of the half vectors, the spot cone base of ``powers[0]`` and the shininess exponents."""
    for q in prange(len(rowq)):
        r = rowq[q]
        lp = lpos[r]
        p = _pixel(rect, r, q - start[r], W)
        vx, vy, vz, lx, ly, lz, _ = _view_and_spot(eye, lp, norms, q, hit[p, 0], hit[p, 1], hit[p, 2])
        powers[0, q] = max(-(lx * D[0] + ly * D[1] + lz * D[2]), _F0)
        norms[2, q] = _dot(lx + vx, ly + vy, lz + vz)
        for k in range(to_light.shape[0]):
            norms[3 + k, q] = _dot(to_light[k, 0] + vx, to_light[k, 1] + vy, to_light[k, 2] + vz)
        exps[q] = shin[p]


@njit(cache=True, error_model="numpy", inline="always")
def _half(nx, ny, nz, hx, hy, hz, length):
    n = max(length, _TINY)
    return max(nx * (hx / n) + ny * (hy / n) + nz * (hz / n), _F0)


@njit(parallel=True, cache=True, error_model="numpy")
def specular_bases(rect, rowq, start, W, hit, normal, eye, lpos, to_light, norms, powers):
    """clamp(dot(n, unit half vector), min=0) of the spotlight and of each shadowing light."""
    for q in prange(len(rowq)):
        r = rowq[q]
        lp = lpos[r]
        p = _pixel(rect, r, q - start[r], W)
        nx, ny, nz = normal[p, 0], normal[p, 1], normal[p, 2]
        vx, vy, vz, lx, ly, lz, _ = _view_and_spot(eye, lp, norms, q, hit[p, 0], hit[p, 1], hit[p, 2])
        powers[1, q] = _half(nx, ny, nz, lx + vx, ly + vy, lz + vz, norms[2, q])
        for k in range(to_light.shape[0]):
            tx, ty, tz = to_light[k, 0], to_light[k, 1], to_light[k, 2]
            powers[2 + k, q] = _half(nx, ny, nz, tx + vx, ty + vy, tz + vz, norms[3 + k, q])


@njit(cache=True, error_model="numpy", inline="always")
def _lit(amb, dif, spec, c, ndl, ms, hl, gain, tex):
    """One light's byte-rounded channel of ``_light_pass``."""
    lit = amb * c + dif * c * ndl + spec * (ms * hl)
    col = _F0 + lit * gain
    return np.rint(min(max(col, _F0), _F1) * tex * _F255)


@njit(cache=True, error_model="numpy", inline="always")
def _blocked(kinds, S, P, R, b, cand, nc, ox, oy, oz, dx, dy, dz, lim, skip=-1):
    for c in range(nc):
        k = cand[c]
        if k == skip:
            continue
        t = _hit(kinds, S, P, R, b, k, ox, oy, oz, dx, dy, dz, False)[0]
        if t > _F0 and t < lim:
            return True
    return False


@njit(parallel=True, cache=True, error_model="numpy")
def prepare(rect, count, plate, hit, valid, lpos, wall_lo, wall_hi, slack, out, cand, ncand):
    """Copy the plate to each row, and list the walls whose box meets the box of the row's shadow rays.

    The box holds the window's points and the light; a culled wall cannot block a ray of the row.
    """
    W = plate.shape[1]
    for r in prange(lpos.shape[0]):
        out[r] = plate
        lp = lpos[r]
        lo0, lo1, lo2 = np.float64(lp[0]), np.float64(lp[1]), np.float64(lp[2])
        hi0, hi1, hi2 = lo0, lo1, lo2
        for j in range(count[r]):
            p = _pixel(rect, r, j, W)
            if valid[p]:
                lo0 = min(lo0, np.float64(hit[p, 0]))
                lo1 = min(lo1, np.float64(hit[p, 1]))
                lo2 = min(lo2, np.float64(hit[p, 2]))
                hi0 = max(hi0, np.float64(hit[p, 0]))
                hi1 = max(hi1, np.float64(hit[p, 1]))
                hi2 = max(hi2, np.float64(hit[p, 2]))
        nc = 0
        for k in range(len(wall_lo)):
            if (
                wall_lo[k, 0] <= hi0 + slack
                and wall_lo[k, 1] <= hi1 + slack
                and wall_lo[k, 2] <= hi2 + slack
                and wall_hi[k, 0] >= lo0 - slack
                and wall_hi[k, 1] >= lo1 - slack
                and wall_hi[k, 2] >= lo2 - slack
            ):
                cand[r, nc] = k
                nc += 1
        ncand[r] = nc


@njit(parallel=True, cache=True, error_model="numpy")
def lit_plates(
    rect,
    rowq,
    start,
    plate,
    hit,
    normal,
    mat,
    tex,
    ms,
    valid,
    eye,
    lpos,
    spot,
    spot_col,
    dir_light,
    dir_col,
    norms,
    powers,
    wall_kind,
    wall_size,
    wall_pos,
    wall_rot,
    cand,
    ncand,
    agent_kind,
    agent_size,
    agent_pos,
    agent_rot,
    out,
):
    """Per-row plates: static pixels gain the tracking light and lose the agent's shadow.

    ``spot`` is (D xyz, s xyz, u xyz, cos_cut, tan_half, k0, k1, k2, eps, castshadow, slack); ``dir_light`` rows are
    (to_light xyz, pos xyz, fwd xyz, near). ``norms`` and ``powers`` hold the pass's square roots and powers.
    """
    W = plate.shape[1]
    D0, D1, D2 = spot[0], spot[1], spot[2]
    cos_cut, tan_half = spot[9], spot[10]
    k0, k1, k2, eps = spot[11], spot[12], spot[13], spot[14]
    cast = spot[15] != _F0
    agent = np.arange(len(agent_kind))
    for q in prange(len(rowq)):
        r = rowq[q]
        p = _pixel(rect, r, q - start[r], W)
        if not valid[p]:
            continue
        lp = lpos[r]
        x, y, z = hit[p, 0], hit[p, 1], hit[p, 2]
        nx, ny, nz = normal[p, 0], normal[p, 1], normal[p, 2]
        vx, vy, vz, lx, ly, lz, dist = _view_and_spot(eye, lp, norms, q, x, y, z)
        sd = -(lx * D0 + ly * D1 + lz * D2)
        gain = powers[0, q] if sd >= cos_cut else _F0
        gain = gain / (k0 + k1 * dist + k2 * dist * dist)
        ndl = max(nx * lx + ny * ly + nz * lz, _F0)
        hl = powers[1, q] if ndl > _F0 else _F0
        a0 = _lit(spot_col[0, 0], spot_col[1, 0], spot_col[2, 0], mat[p, 0], ndl, ms[p], hl, gain, tex[p, 0])
        a1 = _lit(spot_col[0, 1], spot_col[1, 1], spot_col[2, 1], mat[p, 1], ndl, ms[p], hl, gain, tex[p, 1])
        a2 = _lit(spot_col[0, 2], spot_col[1, 2], spot_col[2, 2], mat[p, 2], ndl, ms[p], hl, gain, tex[p, 2])
        ox = x + eps * nx
        oy = y + eps * ny
        oz = z + eps * nz
        if cast:
            rx, ry, rz = x - lp[0], y - lp[1], z - lp[2]
            depth = rx * D0 + ry * D1 + rz * D2
            reach = tan_half * depth
            inside = (
                depth > _F0
                and abs(rx * spot[3] + ry * spot[4] + rz * spot[5]) <= reach
                and abs(rx * spot[6] + ry * spot[7] + rz * spot[8]) <= reach
            )
            if inside:
                lim = dist - eps
                if _blocked(
                    wall_kind, wall_size, wall_pos, wall_rot, 0, cand[r], ncand[r], ox, oy, oz, lx, ly, lz, lim
                ):
                    a0, a1, a2 = _F0, _F0, _F0
                elif _blocked(
                    agent_kind, agent_size, agent_pos, agent_rot, r, agent, len(agent), ox, oy, oz, lx, ly, lz, lim
                ):
                    a0, a1, a2 = _F0, _F0, _F0
        for k in range(dir_light.shape[0]):
            tx, ty, tz = dir_light[k, 0], dir_light[k, 1], dir_light[k, 2]
            lim = (x - dir_light[k, 3]) * dir_light[k, 6] + (y - dir_light[k, 4]) * dir_light[k, 7]
            lim = lim + (z - dir_light[k, 5]) * dir_light[k, 8] - dir_light[k, 9]
            if _blocked(
                agent_kind, agent_size, agent_pos, agent_rot, r, agent, len(agent), ox, oy, oz, tx, ty, tz, lim
            ):
                ndl = max(nx * tx + ny * ty + nz * tz, _F0)
                hl = powers[2 + k, q] if ndl > _F0 else _F0
                c = dir_col[k]
                a0 = a0 - _lit(c[0, 0], c[1, 0], c[2, 0], mat[p, 0], ndl, ms[p], hl, _F1, tex[p, 0])
                a1 = a1 - _lit(c[0, 1], c[1, 1], c[2, 1], mat[p, 1], ndl, ms[p], hl, _F1, tex[p, 1])
                a2 = a2 - _lit(c[0, 2], c[1, 2], c[2, 2], mat[p, 2], ndl, ms[p], hl, _F1, tex[p, 2])
        yy = p // W
        xx = p - yy * W
        out[r, yy, xx, 0] = np.uint8(min(max(F32(plate[yy, xx, 0]) + a0, _F0), _F255))
        out[r, yy, xx, 1] = np.uint8(min(max(F32(plate[yy, xx, 1]) + a1, _F0), _F255))
        out[r, yy, xx, 2] = np.uint8(min(max(F32(plate[yy, xx, 2]) + a2, _F0), _F255))


@njit(parallel=True, cache=True, error_model="numpy")
def sample_squares(X, v, rows, lpos, to_light, norms):
    """Squared lengths of the half vectors of the directional lights and of the vectors to the spotlight."""
    nl = to_light.shape[0]
    for i in prange(X.shape[0]):
        for k in range(nl):
            norms[k, i] = _dot(to_light[k, 0] + v[i, 0], to_light[k, 1] + v[i, 1], to_light[k, 2] + v[i, 2])
        lp = lpos[rows[i]]
        norms[nl, i] = _dot(lp[0] - X[i, 0], lp[1] - X[i, 1], lp[2] - X[i, 2])


@njit(cache=True, error_model="numpy", inline="always")
def _spot_unit(X, rows, lpos, norms, nl, i):
    lp = lpos[rows[i]]
    n = max(norms[nl, i], _TINY)
    return (lp[0] - X[i, 0]) / n, (lp[1] - X[i, 1]) / n, (lp[2] - X[i, 2]) / n


@njit(parallel=True, cache=True, error_model="numpy")
def sample_halves(X, n, v, rows, lpos, D, to_light, norms, powers, ih, prim_shin, exps):
    """Bases of the directional lights' powers, the spot cone base and the spot half vector's squared length."""
    nl = to_light.shape[0]
    for i in prange(X.shape[0]):
        for k in range(nl):
            tx, ty, tz = to_light[k, 0], to_light[k, 1], to_light[k, 2]
            powers[1 + k, i] = _half(n[i, 0], n[i, 1], n[i, 2], tx + v[i, 0], ty + v[i, 1], tz + v[i, 2], norms[k, i])
        lx, ly, lz = _spot_unit(X, rows, lpos, norms, nl, i)
        powers[0, i] = max(-(lx * D[0] + ly * D[1] + lz * D[2]), _F0)
        norms[nl + 1, i] = _dot(lx + v[i, 0], ly + v[i, 1], lz + v[i, 2])
        exps[i] = prim_shin[ih[i]]


@njit(parallel=True, cache=True, error_model="numpy")
def sample_spot_bases(X, n, v, rows, lpos, norms, powers):
    nl = norms.shape[0] - 2
    for i in prange(X.shape[0]):
        lx, ly, lz = _spot_unit(X, rows, lpos, norms, nl, i)
        powers[1 + nl, i] = _half(
            n[i, 0], n[i, 1], n[i, 2], lx + v[i, 0], ly + v[i, 1], lz + v[i, 2], norms[nl + 1, i]
        )


@njit(parallel=True, cache=True, error_model="numpy")
def shade_samples(
    X,
    n,
    v,
    rows,
    ih,
    mat,
    lpos,
    prim_ms,
    base_col,
    nb,
    dir_light,
    dir_col,
    spot,
    spot_col,
    norms,
    powers,
    wall_kind,
    wall_size,
    wall_pos,
    wall_rot,
    cand,
    agent_kind,
    agent_size,
    agent_pos,
    agent_rot,
    out,
):
    """``shader`` colours in [0, 255] of each sample.

    The first ``nb`` rows of ``dir_light`` are the lights without shadows, drawn in one pass. Each other
    directional light and the spotlight add a pass of their own.
    """
    nl = dir_light.shape[0]
    D0, D1, D2 = spot[0], spot[1], spot[2]
    cos_cut, tan_half = spot[9], spot[10]
    k0, k1, k2, eps = spot[11], spot[12], spot[13], spot[14]
    cast = spot[15] != _F0
    agent = np.arange(len(agent_kind))
    for i in prange(X.shape[0]):
        b = rows[i]
        skip = ih[i]
        x, y, z = X[i, 0], X[i, 1], X[i, 2]
        nx, ny, nz = n[i, 0], n[i, 1], n[i, 2]
        ms = prim_ms[skip]
        ox = x + eps * nx
        oy = y + eps * ny
        oz = z + eps * nz
        t0, t1, t2 = _F0, _F0, _F0
        if nb:
            s0, s1, s2 = _F0, _F0, _F0
            for k in range(nb):
                ndl = max(nx * dir_light[k, 0] + ny * dir_light[k, 1] + nz * dir_light[k, 2], _F0)
                hl = powers[1 + k, i] if ndl > _F0 else _F0
                c = base_col[k]
                s0 = s0 + (c[0, 0] * mat[i, 0] + c[1, 0] * mat[i, 0] * ndl + c[2, 0] * (ms * hl)) * _F1
                s1 = s1 + (c[0, 1] * mat[i, 1] + c[1, 1] * mat[i, 1] * ndl + c[2, 1] * (ms * hl)) * _F1
                s2 = s2 + (c[0, 2] * mat[i, 2] + c[1, 2] * mat[i, 2] * ndl + c[2, 2] * (ms * hl)) * _F1
            t0 = np.rint(min(max(s0, _F0), _F1) * _F1 * _F255)
            t1 = np.rint(min(max(s1, _F0), _F1) * _F1 * _F255)
            t2 = np.rint(min(max(s2, _F0), _F1) * _F1 * _F255)
        for k in range(nb, nl):
            tx, ty, tz = dir_light[k, 0], dir_light[k, 1], dir_light[k, 2]
            lim = (x - dir_light[k, 3]) * dir_light[k, 6] + (y - dir_light[k, 4]) * dir_light[k, 7]
            lim = lim + (z - dir_light[k, 5]) * dir_light[k, 8] - dir_light[k, 9]
            if not _blocked(
                agent_kind, agent_size, agent_pos, agent_rot, b, agent, len(agent), ox, oy, oz, tx, ty, tz, lim, skip
            ):
                ndl = max(nx * tx + ny * ty + nz * tz, _F0)
                hl = powers[1 + k, i] if ndl > _F0 else _F0
                c = dir_col[k - nb]
                t0 = t0 + _lit(c[0, 0], c[1, 0], c[2, 0], mat[i, 0], ndl, ms, hl, _F1, _F1)
                t1 = t1 + _lit(c[0, 1], c[1, 1], c[2, 1], mat[i, 1], ndl, ms, hl, _F1, _F1)
                t2 = t2 + _lit(c[0, 2], c[1, 2], c[2, 2], mat[i, 2], ndl, ms, hl, _F1, _F1)
        lp = lpos[b]
        dist = norms[nl, i]
        lx, ly, lz = _spot_unit(X, rows, lpos, norms, nl, i)
        dark = False
        if cast:
            rx, ry, rz = x - lp[0], y - lp[1], z - lp[2]
            depth = rx * D0 + ry * D1 + rz * D2
            reach = tan_half * depth
            if (
                depth > _F0
                and abs(rx * spot[3] + ry * spot[4] + rz * spot[5]) <= reach
                and abs(rx * spot[6] + ry * spot[7] + rz * spot[8]) <= reach
            ):
                lim = dist - eps
                dark = _blocked(
                    wall_kind, wall_size, wall_pos, wall_rot, 0, cand, len(cand), ox, oy, oz, lx, ly, lz, lim
                )
                if not dark:
                    dark = _blocked(
                        agent_kind,
                        agent_size,
                        agent_pos,
                        agent_rot,
                        b,
                        agent,
                        len(agent),
                        ox,
                        oy,
                        oz,
                        lx,
                        ly,
                        lz,
                        lim,
                        skip,
                    )
        if not dark:
            sd = -(lx * D0 + ly * D1 + lz * D2)
            gain = powers[0, i] if sd >= cos_cut else _F0
            gain = gain / (k0 + k1 * dist + k2 * dist * dist)
            ndl = max(nx * lx + ny * ly + nz * lz, _F0)
            hl = powers[1 + nl, i] if ndl > _F0 else _F0
            t0 = t0 + _lit(spot_col[0, 0], spot_col[1, 0], spot_col[2, 0], mat[i, 0], ndl, ms, hl, gain, _F1)
            t1 = t1 + _lit(spot_col[0, 1], spot_col[1, 1], spot_col[2, 1], mat[i, 1], ndl, ms, hl, gain, _F1)
            t2 = t2 + _lit(spot_col[0, 2], spot_col[1, 2], spot_col[2, 2], mat[i, 2], ndl, ms, hl, gain, _F1)
        out[i, 0] = min(t0, _F255)
        out[i, 1] = min(t1, _F255)
        out[i, 2] = min(t2, _F255)
