"""Numba twin of ``cpu3d``: one floor plane, analytic movers, MJWarp shading, one tile of one frame per task.

Each pixel repeats the float32 operation order of the Torch renderer. Shadow and mirror rays of a tile
test only the movers whose bounds reach the tile's rays.
"""

from __future__ import annotations

import math

import numpy as np
from numba import njit, prange

from .rays3d import INF, ONE, ZERO, _blocked, _candidates, _nearest, _packet_bounds

F32 = np.float32
MINVAL = F32(1e-15)
ORIGIN_EPS = F32(1e-4)
MAX_T_EPS = F32(1e-3)
FAR = F32(1e8)
HALF = F32(0.5)
SPOT, DIRECTIONAL = 0, 1


@njit(cache=True, error_model="numpy", inline="always")
def _dot(ax, ay, az, bx, by, bz):
    return ax * bx + ay * by + az * bz


@njit(cache=True, error_model="numpy", inline="always")
def _normalize(x, y, z):
    n = math.sqrt(x * x + y * y + z * z)
    s = n if n > ZERO else ONE
    return x / s, y / s, z / s, n


@njit(cache=True, error_model="numpy", inline="always")
def _pow_scalar(x, e):
    """``torch.pow(tensor, e)`` for a Python float ``e``, with Torch's special exponents."""
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


@njit(cache=True, error_model="numpy", inline="always")
def _light(kind, lp, ld, cut, expo, px, py, pz):
    """Direction toward the light, its distance, the spot attenuation and the cone test at a point."""
    if kind == DIRECTIONAL:
        lx, ly, lz, _ = _normalize(-ld[0], -ld[1], -ld[2])
        return lx, ly, lz, FAR, ONE, True
    lx, ly, lz, dist = _normalize(lp[0] - px, lp[1] - py, lp[2] - pz)
    ax, ay, az, _ = _normalize(ld[0], ld[1], ld[2])
    cosine = (-lx) * ax + (-ly) * ay + (-lz) * az
    return lx, ly, lz, dist, _pow_scalar(max(cosine, ZERO), expo), cosine >= cut


@njit(cache=True, error_model="numpy", inline="always")
def _specular(nx, ny, nz, lx, ly, lz, vx, vy, vz, spec, shin, weight):
    hx, hy, hz, _ = _normalize(lx + vx, ly + vy, lz + vz)
    ndoth = max(_dot(nx, ny, nz, hx, hy, hz), ZERO)
    value = spec * math.pow(ndoth, shin) * weight
    return value if (spec > ZERO and shin > ZERO) else ZERO


@njit(cache=True, error_model="numpy", inline="always")
def _search(g, x):
    """``torch.searchsorted(g, x)``: the count of entries below ``x``."""
    lo = 0
    hi = len(g)
    while lo < hi:
        mid = (lo + hi) // 2
        if g[mid] < x:
            lo = mid + 1
        else:
            hi = mid
    return lo


@njit(cache=True, error_model="numpy")
def _trilinear(texels, sizes, offsets, u, v, lod, out):
    count = sizes.shape[0]
    lod = min(max(lod, ZERO), F32(count - 1))
    low = int(math.floor(lod))
    frac = lod - F32(low)
    C = texels.shape[1]
    for c in range(C):
        out[c] = ZERO
    for side in range(2):
        level = low if side == 0 else min(low + 1, count - 1)
        w = sizes[level, 0]
        h = sizes[level, 1]
        off = offsets[level]
        x = u * F32(w) - HALF
        y = v * F32(h) - HALF
        ix = int(math.floor(x))
        iy = int(math.floor(y))
        fx = x - F32(ix)
        fy = y - F32(iy)
        x0 = ix % w
        x1 = (ix + 1) % w
        y0 = iy % h
        y1 = (iy + 1) % h
        for c in range(C):
            top = texels[off + y0 * w + x0, c] * (ONE - fx) + texels[off + y0 * w + x1, c] * fx
            bottom = texels[off + y1 * w + x0, c] * (ONE - fx) + texels[off + y1 * w + x1, c] * fx
            value = top * (ONE - fy) + bottom * fy
            if side == 0:
                out[c] = value * (ONE - frac)
            else:
                out[c] = out[c] + value * frac


@njit(cache=True, error_model="numpy")
def _sample(texels, sizes, offsets, tex_h, tex_w, u, v, g0, g1, g2, g3, out):
    a = g0 * F32(tex_w)
    b = g1 * F32(tex_h)
    c = g2 * F32(tex_w)
    d = g3 * F32(tex_h)
    rho2 = max(a * a + b * b, c * c + d * d)
    lod = HALF * math.log2(max(rho2, ONE))
    _trilinear(texels, sizes, offsets, u, v, lod, out)


@njit(cache=True, error_model="numpy")
def _vertex(
    vtx,
    slot,
    b,
    lights_i,
    lights_cut,
    lights_exp,
    lrel,
    ldir,
    head_active,
    cam_back,
    sp_p,
    sh_p,
    enable_specular,
    plane,
    pm,
    grid_x,
    grid_y,
    cam,
    vval,
):
    """Light one floor grid vertex: per light its diffuse factor and specular term, then the headlight's."""
    gx_n = len(grid_x)
    gxv = grid_x[vtx % gx_n]
    gyv = grid_y[vtx // gx_n]
    px = ((plane[0] + gxv * pm[b, 0, 0]) + gyv * pm[b, 0, 1]) - cam[b, 0]
    py = ((plane[1] + gxv * pm[b, 1, 0]) + gyv * pm[b, 1, 1]) - cam[b, 1]
    pz = ((plane[2] + gxv * pm[b, 2, 0]) + gyv * pm[b, 2, 1]) - cam[b, 2]
    vx, vy, vz, _ = _normalize(-px, -py, -pz)
    pnx = pm[b, 0, 2]
    pny = pm[b, 1, 2]
    pnz = pm[b, 2, 2]
    nl = lights_i.shape[0]
    for li in range(nl):
        vval[slot, 2 * li] = ZERO
        vval[slot, 2 * li + 1] = ZERO
        if not lights_i[li, 1]:
            continue
        lx, ly, lz, _, att, cone = _light(
            lights_i[li, 0], lrel[b, li], ldir[b, li], lights_cut[li], lights_exp[li], px, py, pz
        )
        vdot = max(_dot(pnx, pny, pnz, lx, ly, lz), ZERO)
        vw = att * (ONE if (cone and vdot > ZERO) else ZERO)
        vval[slot, 2 * li] = vdot * vw
        if enable_specular:
            vval[slot, 2 * li + 1] = _specular(pnx, pny, pnz, lx, ly, lz, vx, vy, vz, sp_p, sh_p, vw)
    vval[slot, 2 * nl] = ZERO
    if head_active and enable_specular:
        lx = cam_back[0]
        ly = cam_back[1]
        lz = cam_back[2]
        vw = ONE if _dot(pnx, pny, pnz, lx, ly, lz) > ZERO else ZERO
        vval[slot, 2 * nl] = _specular(pnx, pny, pnz, lx, ly, lz, vx, vy, vz, sp_p, sh_p, vw)


@njit(cache=True, error_model="numpy")
def _shade(
    b,
    base,
    p,
    n,
    view,
    pid,
    vis,
    mat_idx,
    lights_i,
    lights_cut,
    lights_exp,
    lights_amb,
    lights_spec,
    lrel,
    ldir,
    ldif,
    head,
    head_active,
    cam_back,
    mspec,
    mshin,
    memis,
    enable_specular,
    plane,
    pm,
    grid_x,
    grid_y,
    cam,
    vkey,
    vval,
    out,
):
    """``cpu3d._shade_hits`` for one hit; ``vis`` holds each light's shadow visibility.

    The floor is lit at its grid triangle's vertices and interpolated, as MuJoCo tessellates it.
    ``vkey`` and ``vval`` cache lit vertices of this row; ``vkey[-1]`` counts the cached vertices.
    """
    sp_m = mspec[mat_idx] if enable_specular else F32(0.5)
    sh_m = mshin[mat_idx] if enable_specular else F32(64.0)
    em = memis[mat_idx]
    r0 = base[0] * em
    r1 = base[1] * em
    r2 = base[2] * em
    nl = lights_i.shape[0]
    if head_active:
        r0 = r0 + base[0] * head[0, 0]
        r1 = r1 + base[1] * head[0, 1]
        r2 = r2 + base[2] * head[0, 2]
    elif nl == 0:
        fb = F32(0.3)
        r0 = r0 + base[0] * fb
        r1 = r1 + base[1] * fb
        r2 = r2 + base[2] * fb
    for li in range(nl):
        if lights_i[li, 1]:
            r0 = r0 + base[0] * lights_amb[li, 0]
            r1 = r1 + base[1] * lights_amb[li, 1]
            r2 = r2 + base[2] * lights_amb[li, 2]
    on_plane = pid == 0
    s_a = 0
    s_b = 0
    s_c = 0
    w0 = ZERO
    w1 = ZERO
    w2 = ZERO
    if on_plane:
        lx = (p[0] + cam[b, 0]) - plane[0]
        ly = (p[1] + cam[b, 1]) - plane[1]
        lz = (p[2] + cam[b, 2]) - plane[2]
        x = _dot(lx, ly, lz, pm[b, 0, 0], pm[b, 1, 0], pm[b, 2, 0])
        y = _dot(lx, ly, lz, pm[b, 0, 1], pm[b, 1, 1], pm[b, 2, 1])
        gx_n = len(grid_x)
        ix = min(max(_search(grid_x, x), 1), gx_n - 1)
        iy = min(max(_search(grid_y, y), 1), len(grid_y) - 1)
        fx = min(max((x - grid_x[ix - 1]) / (grid_x[ix] - grid_x[ix - 1]), ZERO), ONE)
        fy = min(max((y - grid_y[iy - 1]) / (grid_y[iy] - grid_y[iy - 1]), ZERO), ONE)
        first = (iy - 1) * gx_n + ix - 1
        last = iy * gx_n + ix
        other = first + (1 if fx >= fy else gx_n)
        w0 = ONE - max(fx, fy)
        w1 = min(fx, fy)
        w2 = abs(fx - fy)
        size = len(vkey) - 1
        if vkey[size] > size - 3:
            vkey[size] = 0
        for k in range(3):
            vtx = first if k == 0 else (last if k == 1 else other)
            slot = -1
            for q in range(vkey[size]):
                if vkey[q] == vtx:
                    slot = q
                    break
            if slot < 0:
                slot = vkey[size]
                vkey[size] += 1
                vkey[slot] = vtx
                _vertex(
                    vtx,
                    slot,
                    b,
                    lights_i,
                    lights_cut,
                    lights_exp,
                    lrel,
                    ldir,
                    head_active,
                    cam_back,
                    mspec[mat_idx],
                    mshin[mat_idx],
                    enable_specular,
                    plane,
                    pm,
                    grid_x,
                    grid_y,
                    cam,
                    vval,
                )
            if k == 0:
                s_a = slot
            elif k == 1:
                s_b = slot
            else:
                s_c = slot
    for li in range(nl):
        if not lights_i[li, 1]:
            continue
        if on_plane:
            g = 2 * li
            d0 = ((ldif[b, li, 0] * vval[s_a, g]) * w0 + (ldif[b, li, 0] * vval[s_b, g]) * w1) + (
                ldif[b, li, 0] * vval[s_c, g]
            ) * w2
            d1 = ((ldif[b, li, 1] * vval[s_a, g]) * w0 + (ldif[b, li, 1] * vval[s_b, g]) * w1) + (
                ldif[b, li, 1] * vval[s_c, g]
            ) * w2
            d2 = ((ldif[b, li, 2] * vval[s_a, g]) * w0 + (ldif[b, li, 2] * vval[s_b, g]) * w1) + (
                ldif[b, li, 2] * vval[s_c, g]
            ) * w2
            d0 = d0 * vis[li]
            d1 = d1 * vis[li]
            d2 = d2 * vis[li]
            sp = ((vval[s_a, g + 1] * w0 + vval[s_b, g + 1] * w1) + vval[s_c, g + 1] * w2) * vis[li]
        else:
            lx, ly, lz, _, att, cone = _light(
                lights_i[li, 0], lrel[b, li], ldir[b, li], lights_cut[li], lights_exp[li], p[0], p[1], p[2]
            )
            ndotl = max(_dot(n[0], n[1], n[2], lx, ly, lz), ZERO)
            live = cone and ndotl > ZERO
            wgt = att * vis[li] * (ONE if live else ZERO)
            f = ndotl * wgt
            d0 = ldif[b, li, 0] * f
            d1 = ldif[b, li, 1] * f
            d2 = ldif[b, li, 2] * f
            sp = ZERO
            if enable_specular:
                sp = _specular(n[0], n[1], n[2], lx, ly, lz, view[0], view[1], view[2], sp_m, sh_m, wgt)
        r0 = r0 + base[0] * d0
        r1 = r1 + base[1] * d1
        r2 = r2 + base[2] * d2
        if enable_specular:
            r0 = r0 + lights_spec[li, 0] * sp
            r1 = r1 + lights_spec[li, 1] * sp
            r2 = r2 + lights_spec[li, 2] * sp
    if head_active:
        lx = cam_back[0]
        ly = cam_back[1]
        lz = cam_back[2]
        ndotl = max(_dot(n[0], n[1], n[2], lx, ly, lz), ZERO)
        wgt = ONE if ndotl > ZERO else ZERO
        f = ndotl * wgt
        r0 = r0 + base[0] * (head[1, 0] * f)
        r1 = r1 + base[1] * (head[1, 1] * f)
        r2 = r2 + base[2] * (head[1, 2] * f)
        if enable_specular:
            if on_plane:
                h = 2 * nl
                sp = (vval[s_a, h] * w0 + vval[s_b, h] * w1) + vval[s_c, h] * w2
            else:
                sp = _specular(n[0], n[1], n[2], lx, ly, lz, view[0], view[1], view[2], sp_m, sh_m, wgt)
            r0 = r0 + head[2, 0] * sp
            r1 = r1 + head[2, 1] * sp
            r2 = r2 + head[2, 2] * sp
    out[0] = min(max(r0, ZERO), ONE)
    out[1] = min(max(r1, ZERO), ONE)
    out[2] = min(max(r2, ZERO), ONE)


@njit(cache=True, error_model="numpy", inline="always")
def _plane_blocked(ox, oy, oz, dx, dy, dz, lim, b, cam, plane, pm, sx, sy):
    """The floor's own occlusion of a shadow ray: ``ray_plane`` from the shadow origin, front face only."""
    wx = ox + cam[b, 0]
    wy = oy + cam[b, 1]
    wz = oz + cam[b, 2]
    lz = _dot(dx, dy, dz, pm[b, 0, 2], pm[b, 1, 2], pm[b, 2, 2])
    front = lz <= -MINVAL
    tp = _dot(plane[0] - wx, plane[1] - wy, plane[2] - wz, pm[b, 0, 2], pm[b, 1, 2], pm[b, 2, 2]) / (
        lz if front else -ONE
    )
    ok = front and tp >= ZERO and tp < lim
    if ok and (sx > ZERO or sy > ZERO):
        hx = (wx + dx * tp) - plane[0]
        hy = (wy + dy * tp) - plane[1]
        hz = (wz + dz * tp) - plane[2]
        if sx > ZERO:
            ok = ok and abs(_dot(hx, hy, hz, pm[b, 0, 0], pm[b, 1, 0], pm[b, 2, 0])) <= sx
        if sy > ZERO:
            ok = ok and abs(_dot(hx, hy, hz, pm[b, 0, 1], pm[b, 1, 1], pm[b, 2, 1])) <= sy
    return ok


@njit(parallel=True, cache=True, error_model="numpy")
def render(
    kinds,
    S,
    P,
    R,
    dirs,
    res,
    tile,
    cones,
    cam,
    plane,
    pm,
    plane_size,
    grid_x,
    grid_y,
    tex,
    tex_sizes,
    tex_offsets,
    tex_hw,
    tex_affine,
    uv_grad,
    uv_tilt,
    pal_tex,
    pal_default,
    pal_rest,
    pal_rows,
    pal_moved,
    mat_rgb,
    own_rgb,
    prim_mat,
    nmat,
    tex00,
    mspec,
    mshin,
    memis,
    enable_specular,
    lights_i,
    lights_cut,
    lights_exp,
    lights_amb,
    lights_spec,
    lrel,
    ldir,
    ldif,
    head,
    head_active,
    cam_rot,
    shadows,
    reflectance,
    sky,
    quant_round,
    out,
):
    """Draw (B, res*res, 3) uint8 frames. Mover poses ``P`` are relative to each row's camera ``cam``."""
    B = P.shape[0]
    K = len(kinds)
    tx = (res + tile - 1) // tile
    T = cones.shape[0]
    nl = lights_i.shape[0]
    sx = plane_size[0]
    sy = plane_size[1]
    has_tex = tex.shape[0] > 0
    cam_back = cam_rot[:, 2].copy()
    for job in prange(B * T):
        b = job // T
        tl = job % T
        x0 = (tl % tx) * tile
        y0 = (tl // tx) * tile
        m = 0
        jj = np.empty(tile * tile, np.int64)
        for y in range(y0, min(y0 + tile, res)):
            for x in range(x0, min(x0 + tile, res)):
                jj[m] = y * res + x
                m += 1
        tb = np.empty(m, np.float32)
        tpl = np.empty(m, np.float32)
        pid = np.empty(m, np.int64)
        nrm = np.empty((m, 3), np.float32)
        cand = np.empty(K, np.int64)
        nc = _candidates(
            kinds, S, P, b, 0.0, 0.0, 0.0, 0.0, cones[tl, 0], cones[tl, 1], cones[tl, 2], cones[tl, 3], cand
        )
        # Relative plane origin and the plane frame of this row.
        qx = -(plane[0] - cam[b, 0])
        qy = -(plane[1] - cam[b, 1])
        qz = -(plane[2] - cam[b, 2])
        lo0 = _dot(qx, qy, qz, pm[b, 0, 0], pm[b, 1, 0], pm[b, 2, 0])
        lo1 = _dot(qx, qy, qz, pm[b, 0, 1], pm[b, 1, 1], pm[b, 2, 1])
        lo2 = _dot(qx, qy, qz, pm[b, 0, 2], pm[b, 1, 2], pm[b, 2, 2])
        for i in range(m):
            j = jj[i]
            dx = dirs[j, 0]
            dy = dirs[j, 1]
            dz = dirs[j, 2]
            lv0 = _dot(dx, dy, dz, pm[b, 0, 0], pm[b, 1, 0], pm[b, 2, 0])
            lv1 = _dot(dx, dy, dz, pm[b, 0, 1], pm[b, 1, 1], pm[b, 2, 1])
            lvz = _dot(dx, dy, dz, pm[b, 0, 2], pm[b, 1, 2], pm[b, 2, 2])
            front = lvz <= -MINVAL
            t = -lo2 / (lvz if front else -ONE)
            ok = front and t >= ZERO
            if sx > ZERO:
                ok = ok and abs(lo0 + t * lv0) <= sx
            if sy > ZERO:
                ok = ok and abs(lo1 + t * lv1) <= sy
            tpl[i] = t
            best = t if ok else INF
            pi = 0 if ok else -1
            nx = pm[b, 0, 2]
            ny = pm[b, 1, 2]
            nz = pm[b, 2, 2]
            if nc:
                tm, mx, my, mz, im = _nearest(kinds, S, P, R, b, cand, nc, ZERO, ZERO, ZERO, dx, dy, dz)
                if _dot(mx, my, mz, dx, dy, dz) > ZERO:
                    tm = INF
                if tm < best:
                    best = tm
                    pi = im + 1
                    nx = mx
                    ny = my
                    nz = mz
            tb[i] = best
            pid[i] = pi
            nrm[i, 0] = nx
            nrm[i, 1] = ny
            nrm[i, 2] = nz

        base = np.empty((m, 3), np.float32)
        pt = np.empty((m, 3), np.float32)
        texel = np.empty(3, np.float32)
        weights = np.empty(pal_tex.shape[1], np.float32)
        for i in range(m):
            if pid[i] < 0:
                continue
            j = jj[i]
            t = tb[i]
            for c in range(3):
                pt[i, c] = dirs[j, c] * t
            k = pid[i]
            mi = prim_mat[k]
            for c in range(3):
                v = mat_rgb[b, mi, c] if mi < nmat else own_rgb[k, c]
                base[i, c] = v * (tex00[mi, c] if k > 0 else ONE)
            if k == 0 and has_tex:
                rx = (pt[i, 0] + cam[b, 0]) - plane[0]
                ry = (pt[i, 1] + cam[b, 1]) - plane[1]
                rz = (pt[i, 2] + cam[b, 2]) - plane[2]
                lx = _dot(rx, ry, rz, pm[b, 0, 0], pm[b, 1, 0], pm[b, 2, 0])
                ly = _dot(rx, ry, rz, pm[b, 0, 1], pm[b, 1, 1], pm[b, 2, 1])
                u = HALF * lx * tex_affine[0] + tex_affine[2]
                v = F32(-0.5) * ly * tex_affine[1] + tex_affine[3]
                u = u - F32(math.floor(u))
                v = v - F32(math.floor(v))
                height = _dot(
                    plane[0] - cam[b, 0],
                    plane[1] - cam[b, 1],
                    plane[2] - cam[b, 2],
                    pm[b, 0, 2],
                    pm[b, 1, 2],
                    pm[b, 2, 2],
                )
                if uv_tilt:
                    g0, g1, g2, g3 = _uv_gradient(dirs, j, cam_rot, pm, b, tex_affine)
                else:
                    g0 = uv_grad[j, 0]
                    g1 = uv_grad[j, 1]
                    g2 = uv_grad[j, 2]
                    g3 = uv_grad[j, 3]
                g0 = g0 * height
                g1 = g1 * height
                g2 = g2 * height
                g3 = g3 * height
                _sample(tex, tex_sizes, tex_offsets, tex_hw[0], tex_hw[1], u, v, g0, g1, g2, g3, texel)
                if pal_moved[b]:
                    _sample(pal_tex, tex_sizes, tex_offsets, tex_hw[0], tex_hw[1], u, v, g0, g1, g2, g3, weights)
                    npal = pal_default.shape[0]
                    for c in range(3):
                        acc = weights[0] * pal_rows[b, 0, c]
                        for q in range(1, npal):
                            acc = acc + weights[q] * pal_rows[b, q, c]
                        texel[c] = acc + weights[npal] * pal_rest[c]
                for c in range(3):
                    base[i, c] = base[i, c] * texel[c]

        # Shadow visibility per light: each light's rays from this tile test only the movers they can reach.
        vis = np.ones((m, max(nl, 1)), np.float32)
        if shadows and K:
            so = np.empty((m, 3), np.float32)
            sd = np.empty((m, 3), np.float32)
            slim = np.empty(m, np.float32)
            idx = np.empty(m, np.int64)
            for li in range(nl):
                if not (lights_i[li, 1] and lights_i[li, 2]):
                    continue
                q = 0
                for i in range(m):
                    if pid[i] < 0:
                        continue
                    lx, ly, lz, dist, _, cone = _light(
                        lights_i[li, 0],
                        lrel[b, li],
                        ldir[b, li],
                        lights_cut[li],
                        lights_exp[li],
                        pt[i, 0],
                        pt[i, 1],
                        pt[i, 2],
                    )
                    ndotl = max(_dot(nrm[i, 0], nrm[i, 1], nrm[i, 2], lx, ly, lz), ZERO)
                    if not (cone and ndotl > ZERO):
                        continue
                    for c in range(3):
                        so[q, c] = pt[i, c] + nrm[i, c] * ORIGIN_EPS
                    sd[q, 0] = lx
                    sd[q, 1] = ly
                    sd[q, 2] = lz
                    slim[q] = dist - MAX_T_EPS
                    idx[q] = i
                    q += 1
                if q == 0:
                    continue
                cx, cy, cz, sp, ax, ay, az, ang = _packet_bounds(so, sd, 0, q)
                nsc = _candidates(kinds, S, P, b, cx, cy, cz, sp, ax, ay, az, ang, cand)
                for r in range(q):
                    blocked = _blocked(
                        kinds,
                        S,
                        P,
                        R,
                        b,
                        cand,
                        nsc,
                        so[r, 0],
                        so[r, 1],
                        so[r, 2],
                        sd[r, 0],
                        sd[r, 1],
                        sd[r, 2],
                        slim[r],
                        -1,
                    )
                    if not blocked:
                        blocked = _plane_blocked(
                            so[r, 0],
                            so[r, 1],
                            so[r, 2],
                            sd[r, 0],
                            sd[r, 1],
                            sd[r, 2],
                            slim[r],
                            b,
                            cam,
                            plane,
                            pm,
                            sx,
                            sy,
                        )
                    if blocked:
                        vis[idx[r], li] = ZERO

        rgb = np.empty((m, 3), np.float32)
        view = np.empty(3, np.float32)
        vkey = np.zeros(33, np.int64)
        vval = np.empty((32, 2 * nl + 1), np.float32)
        for i in range(m):
            j = jj[i]
            if pid[i] < 0:
                for c in range(3):
                    rgb[i, c] = sky[j, c]
                continue
            for c in range(3):
                view[c] = -dirs[j, c]
            _shade(
                b,
                base[i],
                pt[i],
                nrm[i],
                view,
                pid[i],
                vis[i],
                prim_mat[pid[i]],
                lights_i,
                lights_cut,
                lights_exp,
                lights_amb,
                lights_spec,
                lrel,
                ldir,
                ldif,
                head,
                head_active,
                cam_back,
                mspec,
                mshin,
                memis,
                enable_specular,
                plane,
                pm,
                grid_x,
                grid_y,
                cam,
                vkey,
                vval,
                rgb[i],
            )

        # Mirror rays from floor pixels, cast as one packet per tile.
        if reflectance > ZERO and K:
            ro = np.empty((m, 3), np.float32)
            rd = np.empty((m, 3), np.float32)
            idx = np.empty(m, np.int64)
            q = 0
            pnx = pm[b, 0, 2]
            pny = pm[b, 1, 2]
            pnz = pm[b, 2, 2]
            for i in range(m):
                if pid[i] != 0:
                    continue
                j = jj[i]
                dx = dirs[j, 0]
                dy = dirs[j, 1]
                dz = dirs[j, 2]
                two = F32(2.0) * _dot(dx, dy, dz, pnx, pny, pnz)
                rd[q, 0] = dx - two * pnx
                rd[q, 1] = dy - two * pny
                rd[q, 2] = dz - two * pnz
                ro[q, 0] = dx * tpl[i] + pnx * ORIGIN_EPS
                ro[q, 1] = dy * tpl[i] + pny * ORIGIN_EPS
                ro[q, 2] = dz * tpl[i] + pnz * ORIGIN_EPS
                idx[q] = i
                q += 1
            if q:
                cx, cy, cz, sp, ax, ay, az, ang = _packet_bounds(ro, rd, 0, q)
                nrc = _candidates(kinds, S, P, b, cx, cy, cz, sp, ax, ay, az, ang, cand)
                if nrc:
                    ones = np.ones(max(nl, 1), np.float32)
                    rbase = np.empty(3, np.float32)
                    rp = np.empty(3, np.float32)
                    rn = np.empty(3, np.float32)
                    rv = np.empty(3, np.float32)
                    refl = np.empty(3, np.float32)
                    for r in range(q):
                        t, mx, my, mz, im = _nearest(
                            kinds, S, P, R, b, cand, nrc, ro[r, 0], ro[r, 1], ro[r, 2], rd[r, 0], rd[r, 1], rd[r, 2]
                        )
                        if not (t < INF and _dot(mx, my, mz, rd[r, 0], rd[r, 1], rd[r, 2]) <= ZERO):
                            continue
                        k = im + 1
                        mi = prim_mat[k]
                        for c in range(3):
                            v = mat_rgb[b, mi, c] if mi < nmat else own_rgb[k, c]
                            rbase[c] = v * tex00[mi, c] * reflectance
                            rp[c] = ro[r, c] + t * rd[r, c]
                            rv[c] = -rd[r, c]
                        rn[0] = mx
                        rn[1] = my
                        rn[2] = mz
                        _shade(
                            b,
                            rbase,
                            rp,
                            rn,
                            rv,
                            k,
                            ones,
                            mi,
                            lights_i,
                            lights_cut,
                            lights_exp,
                            lights_amb,
                            lights_spec,
                            lrel,
                            ldir,
                            ldif,
                            head,
                            head_active,
                            cam_back,
                            mspec,
                            mshin,
                            memis,
                            enable_specular,
                            plane,
                            pm,
                            grid_x,
                            grid_y,
                            cam,
                            vkey,
                            vval,
                            refl,
                        )
                        i = idx[r]
                        for c in range(3):
                            rgb[i, c] = min(max(refl[c] + rgb[i, c], ZERO), ONE)

        for i in range(m):
            j = jj[i]
            for c in range(3):
                v = rgb[i, c] * F32(255.0)
                if quant_round:
                    v = v + HALF
                v = F32(math.floor(v))
                out[b, j, c] = np.uint8(min(max(v, ZERO), F32(255.0)))


@njit(cache=True, error_model="numpy")
def _uv_gradient(dirs, j, rot, pm, b, tex_affine):
    """``Scene._plane_uv_gradients`` for one ray and this row's plane frame."""
    s = -_dot(dirs[j, 0], dirs[j, 1], dirs[j, 2], rot[0, 2], rot[1, 2], rot[2, 2])
    rx = dirs[j, 0] / s
    ry = dirs[j, 1] / s
    rz = dirs[j, 2] / s
    den = _dot(rx, ry, rz, pm[b, 0, 2], pm[b, 1, 2], pm[b, 2, 2])
    if not abs(den) > MINVAL:
        den = MINVAL
    out = np.empty(4, np.float32)
    for side in range(2):
        px = tex_affine[4 + side] * rot[0, side]
        py = tex_affine[4 + side] * rot[1, side]
        pz = tex_affine[4 + side] * rot[2, side]
        f = _dot(px, py, pz, pm[b, 0, 2], pm[b, 1, 2], pm[b, 2, 2]) / den
        hx = (px - rx * f) / den
        hy = (py - ry * f) / den
        hz = (pz - rz * f) / den
        out[2 * side] = tex_affine[6] * _dot(hx, hy, hz, pm[b, 0, 0], pm[b, 1, 0], pm[b, 2, 0])
        out[2 * side + 1] = tex_affine[7] * _dot(hx, hy, hz, pm[b, 0, 1], pm[b, 1, 1], pm[b, 2, 1])
    return out[0], out[1], out[2], out[3]
