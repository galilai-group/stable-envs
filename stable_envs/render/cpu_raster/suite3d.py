"""Numba twin of the fixed-camera DM Control compositor: one tile of one frame per task.

The static plate supplies each pixel's color. The kernel casts the movers, their shadows on the plate,
their mirror images in the floor, and shades the covered pixels in the float32 operation order of
``dmcontrol._shading._shade``.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .rays3d import INF, ONE, ZERO, _blocked, _candidates, _nearest

F32 = np.float32
DIRECTIONAL = 1
CLAMP = F32(1e6)
TINY = F32(1e-12)
TINY2 = F32(1e-24)


@njit(cache=True, error_model="numpy", inline="always")
def _dot(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


@njit(cache=True, error_model="numpy", inline="always")
def _pow_scalar(x, e):
    """``tensor ** e`` for a Python float ``e``, with Torch's special exponents."""
    if e == 0.0:
        return ONE
    if e == 1.0:
        return x
    if e == 2.0:
        return x * x
    if e == 3.0:
        return x * x * x
    if e == 0.5:
        return math.sqrt(x)
    if e == -0.5:
        return ONE / math.sqrt(x)
    if e == -1.0:
        return ONE / x
    if e == -2.0:
        return ONE / (x * x)
    return math.pow(x, F32(e))


@njit(cache=True, error_model="numpy")
def _shade(px, py, pz, nx, ny, nz, vx, vy, vz, b, k, unlit, mat_rgb, spec, shin, tex, lf, li, scale, out):
    """``_shading._shade`` of one point times the primitive's texture color and 255.

    ``li`` rows are (kind, castshadow, spot) and ``lf`` rows are (ambient, diffuse, specular, pos x3,
    toward-light x3, axis x3, attenuation x3, cos cutoff, exponent).
    """
    m0 = mat_rgb[b, k, 0]
    m1 = mat_rgb[b, k, 1]
    m2 = mat_rgb[b, k, 2]
    o0 = ZERO
    o1 = ZERO
    o2 = ZERO
    hl = ZERO
    exponent = F32(128.0) * shin[k]
    for i in range(li.shape[0]):
        if li[i, 0] == DIRECTIONAL:
            lx = lf[i, 6]
            ly = lf[i, 7]
            lz = lf[i, 8]
            att = ONE
        else:
            wx = lf[i, 3] - px
            wy = lf[i, 4] - py
            wz = lf[i, 5] - pz
            dist = math.sqrt(max(_dot(wx, wy, wz, wx, wy, wz), TINY2))
            lx = wx / dist
            ly = wy / dist
            lz = wz / dist
            att = ONE / max(lf[i, 12] + lf[i, 13] * dist + lf[i, 14] * dist * dist, TINY)
            if li[i, 2]:
                se = _dot(lf[i, 9], lf[i, 10], lf[i, 11], -lx, -ly, -lz)
                att = att * _pow_scalar(max(se, ZERO), float(lf[i, 16])) if se >= lf[i, 15] else ZERO
        lit = att
        if unlit and li[i, 1]:
            lit = att * ZERO
        ndl = max(_dot(nx, ny, nz, lx, ly, lz), ZERO)
        hx = lx + vx
        hy = ly + vy
        hz = lz + vz
        hn = max(math.sqrt(hx * hx + hy * hy + hz * hz), TINY)
        hx = hx / hn
        hy = hy / hn
        hz = hz / hn
        ndh = max(_dot(nx, ny, nz, hx, hy, hz), ZERO)
        a = att * lf[i, 0]
        o0 = o0 + m0 * a
        o1 = o1 + m1 * a
        o2 = o2 + m2 * a
        d = lit * ndl * lf[i, 1]
        o0 = o0 + m0 * d
        o1 = o1 + m1 * d
        o2 = o2 + m2 * d
        hl = hl + lit * lf[i, 2] * math.pow(ndh, exponent)
    s = hl * spec[k] * scale
    out[0] = min(max(o0 + s, ZERO), ONE) * tex[k, 0] * F32(255.0)
    out[1] = min(max(o1 + s, ZERO), ONE) * tex[k, 1] * F32(255.0)
    out[2] = min(max(o2 + s, ZERO), ONE) * tex[k, 2] * F32(255.0)


@njit(cache=True)
def tile_rays(index, res, tile):
    """Group the pixels with ``index >= 0`` by camera-ray tile: run starts (T+1) and their pixels."""
    tx = (res + tile - 1) // tile
    T = tx * ((res + tile - 1) // tile)
    count = np.zeros(T + 1, np.int64)
    for j in range(res * res):
        if index[j] >= 0:
            count[(j // res // tile) * tx + (j % res) // tile + 1] += 1
    starts = np.cumsum(count)
    fill = starts[:-1].copy()
    pixels = np.empty(starts[-1], np.int64)
    for j in range(res * res):
        if index[j] >= 0:
            t = (j // res // tile) * tx + (j % res) // tile
            pixels[fill[t]] = j
            fill[t] += 1
    return starts, pixels


@njit(parallel=True, cache=True, error_model="numpy")
def render(
    kinds,
    S,
    P,
    R,
    origin,
    dirs,
    res,
    tile,
    cones,
    static_t,
    plate,
    shadow_of,
    dark,
    so,
    sd,
    slim,
    reach,
    shadow_starts,
    shadow_pixels,
    shadow_sphere,
    shadow_cone,
    mirror_of,
    ro,
    rd,
    surf,
    mirror_starts,
    mirror_pixels,
    mirror_sphere,
    mirror_cone,
    reflectance,
    unreflected,
    mat_rgb,
    spec,
    shin,
    tex,
    lf,
    li,
    shadow_light,
    self_shadow,
    eps,
    scale,
    frame,
    out,
    quantize,
):
    """Draw frames: ``frame`` (B, N, 3) float32 before rounding, or ``out`` (B, N, 3) uint8 when ``quantize``."""
    B = P.shape[0]
    K = len(kinds)
    tx = (res + tile - 1) // tile
    T = cones.shape[0]
    ox = origin[0]
    oy = origin[1]
    oz = origin[2]
    for job in prange(B * T):
        b = job // T
        tl = job % T
        x0 = (tl % tx) * tile
        y0 = (tl // tx) * tile
        tw = min(tile, res - x0)
        th = min(tile, res - y0)
        m = tw * th
        base = np.empty((m, 3), np.float32)
        for y in range(th):
            for x in range(tw):
                j = (y0 + y) * res + x0 + x
                for c in range(3):
                    base[y * tw + x, c] = plate[j, c]
        cand = np.empty(K, np.int64)
        every = np.arange(K)
        col = np.empty(3, np.float32)
        # Mover shadows on the static surfaces: the plate's shadowed color where a mover blocks the light.
        a = shadow_starts[tl]
        e = shadow_starts[tl + 1]
        if shadow_light >= 0 and e > a:
            nc = _candidates(
                kinds,
                S,
                P,
                b,
                shadow_sphere[tl, 0],
                shadow_sphere[tl, 1],
                shadow_sphere[tl, 2],
                shadow_sphere[tl, 3],
                shadow_cone[tl, 0],
                shadow_cone[tl, 1],
                shadow_cone[tl, 2],
                shadow_cone[tl, 3],
                cand,
            )
            if nc:
                for r in range(a, e):
                    j = shadow_pixels[r]
                    q = shadow_of[j]
                    hit = _blocked(
                        kinds,
                        S,
                        P,
                        R,
                        b,
                        cand,
                        nc,
                        so[q, 0],
                        so[q, 1],
                        so[q, 2],
                        sd[q, 0],
                        sd[q, 1],
                        sd[q, 2],
                        slim[q],
                        -1,
                    )
                    if hit and reach[q]:
                        loc = (j // res - y0) * tw + j % res - x0
                        for c in range(3):
                            base[loc, c] = dark[q, c]
        # Mirror images: reflectance times the mover's color plus the rest times the unreflected surface.
        a = mirror_starts[tl]
        e = mirror_starts[tl + 1]
        if reflectance > ZERO and e > a:
            nc = _candidates(
                kinds,
                S,
                P,
                b,
                mirror_sphere[tl, 0],
                mirror_sphere[tl, 1],
                mirror_sphere[tl, 2],
                mirror_sphere[tl, 3],
                mirror_cone[tl, 0],
                mirror_cone[tl, 1],
                mirror_cone[tl, 2],
                mirror_cone[tl, 3],
                cand,
            )
            if nc:
                for r in range(a, e):
                    j = mirror_pixels[r]
                    q = mirror_of[j]
                    t, nx, ny, nz, k = _nearest(
                        kinds, S, P, R, b, cand, nc, ro[q, 0], ro[q, 1], ro[q, 2], rd[q, 0], rd[q, 1], rd[q, 2]
                    )
                    if not t < INF:
                        continue
                    tc = min(t, CLAMP)
                    _shade(
                        ro[q, 0] + tc * rd[q, 0],
                        ro[q, 1] + tc * rd[q, 1],
                        ro[q, 2] + tc * rd[q, 2],
                        nx,
                        ny,
                        nz,
                        -rd[q, 0],
                        -rd[q, 1],
                        -rd[q, 2],
                        b,
                        k,
                        False,
                        mat_rgb,
                        spec,
                        shin,
                        tex,
                        lf,
                        li,
                        scale,
                        col,
                    )
                    loc = (j // res - y0) * tw + j % res - x0
                    for c in range(3):
                        base[loc, c] = reflectance * col[c] + unreflected * surf[q, c]
        # Movers in front of the static surface, shaded with their own shadow test.
        nc = _candidates(
            kinds,
            S,
            P,
            b,
            float(ox),
            float(oy),
            float(oz),
            0.0,
            cones[tl, 0],
            cones[tl, 1],
            cones[tl, 2],
            cones[tl, 3],
            cand,
        )
        for y in range(th):
            for x in range(tw):
                j = (y0 + y) * res + x0 + x
                loc = y * tw + x
                if nc:
                    dx = dirs[j, 0]
                    dy = dirs[j, 1]
                    dz = dirs[j, 2]
                    t, nx, ny, nz, k = _nearest(kinds, S, P, R, b, cand, nc, ox, oy, oz, dx, dy, dz)
                    if t < static_t[j]:
                        tc = min(t, CLAMP)
                        px = ox + tc * dx
                        py = oy + tc * dy
                        pz = oz + tc * dz
                        unlit = False
                        if shadow_light >= 0 and self_shadow:
                            i = shadow_light
                            if li[i, 0] == DIRECTIONAL:
                                ex = lf[i, 6]
                                ey = lf[i, 7]
                                ez = lf[i, 8]
                                lim = INF
                            else:
                                wx = lf[i, 3] - px
                                wy = lf[i, 4] - py
                                wz = lf[i, 5] - pz
                                lim = math.sqrt(max(_dot(wx, wy, wz, wx, wy, wz), TINY2))
                                ex = wx / lim
                                ey = wy / lim
                                ez = wz / lim
                            unlit = _blocked(
                                kinds,
                                S,
                                P,
                                R,
                                b,
                                every,
                                K,
                                px + ex * eps,
                                py + ey * eps,
                                pz + ez * eps,
                                ex,
                                ey,
                                ez,
                                lim,
                                k,
                            )
                        _shade(
                            px,
                            py,
                            pz,
                            nx,
                            ny,
                            nz,
                            -dx,
                            -dy,
                            -dz,
                            b,
                            k,
                            unlit,
                            mat_rgb,
                            spec,
                            shin,
                            tex,
                            lf,
                            li,
                            scale,
                            col,
                        )
                        for c in range(3):
                            base[loc, c] = col[c]
                for c in range(3):
                    v = base[loc, c]
                    if quantize:
                        out[b, j, c] = np.uint8(min(max(np.rint(v), ZERO), F32(255.0)))
                    else:
                        frame[b, j, c] = v
