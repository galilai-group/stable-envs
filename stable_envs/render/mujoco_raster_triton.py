"""Triton kernels of the CUDA backend of ``render.mujoco_raster``.

``vertices`` lights every corner per frame with the CPU backend's formula (``mujoco_raster._shade``): the same
light rows, power table and clamps, in float32. ``opaque`` resolves one multisample pass of the opaque raster
against the background and the shadow maps. ``shadow`` draws the moving casters into a window of the shadow map.
``blend`` lists the translucent fragments of each sample and blends them in MuJoCo's order.
"""

from __future__ import annotations

import torch
import triton
import triton.language as tl

EMPTY = (1 << 63) - 1
#: Bits of a packed fragment: geom rank (15), triangle index in its geom (16), translucent triangle (20).
TRI_BITS, LOCAL_BITS, RANK_BITS = 20, 16, 15
DEPTH_BITS = float(2**24 - 1)
OFFSET_FACTOR, OFFSET_UNITS = -1.5, -4.0
TRANSPARENT_ALPHA = 0.995
# Kernels read module values only as constexpr.
_DEPTH_BITS = tl.constexpr(DEPTH_BITS)
_OFFSET_FACTOR = tl.constexpr(OFFSET_FACTOR)
_OFFSET_UNITS = tl.constexpr(OFFSET_UNITS)
_TRANSPARENT_ALPHA = tl.constexpr(TRANSPARENT_ALPHA)


@triton.jit
def _vertices(
    VPos,
    VNor,
    VGeom,
    Xf,
    Rgba,
    Mat,
    Lut,
    Cam,
    PV,
    Lights,
    Clip,
    Attr,
    Trans,
    V,
    G,
    L,
    NLUT,
    ambient,
    BLOCK: tl.constexpr,
    LW: tl.constexpr,
):
    c = tl.program_id(1)
    v = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = v < V
    g = tl.load(VGeom + v, mask=m, other=0)
    lx_ = tl.load(VPos + v * 3 + 0, mask=m, other=0.0)
    ly_ = tl.load(VPos + v * 3 + 1, mask=m, other=0.0)
    lz_ = tl.load(VPos + v * 3 + 2, mask=m, other=0.0)
    nx_ = tl.load(VNor + v * 3 + 0, mask=m, other=0.0)
    ny_ = tl.load(VNor + v * 3 + 1, mask=m, other=1.0)
    nz_ = tl.load(VNor + v * 3 + 2, mask=m, other=0.0)
    xf = (c * G + g).to(tl.int64) * 12
    m0 = tl.load(Xf + xf + 0, mask=m, other=1.0)
    m1 = tl.load(Xf + xf + 1, mask=m, other=0.0)
    m2 = tl.load(Xf + xf + 2, mask=m, other=0.0)
    m3 = tl.load(Xf + xf + 3, mask=m, other=0.0)
    m4 = tl.load(Xf + xf + 4, mask=m, other=1.0)
    m5 = tl.load(Xf + xf + 5, mask=m, other=0.0)
    m6 = tl.load(Xf + xf + 6, mask=m, other=0.0)
    m7 = tl.load(Xf + xf + 7, mask=m, other=0.0)
    m8 = tl.load(Xf + xf + 8, mask=m, other=1.0)
    px = m0 * lx_ + m1 * ly_ + m2 * lz_ + tl.load(Xf + xf + 9, mask=m, other=0.0)
    py = m3 * lx_ + m4 * ly_ + m5 * lz_ + tl.load(Xf + xf + 10, mask=m, other=0.0)
    pz = m6 * lx_ + m7 * ly_ + m8 * lz_ + tl.load(Xf + xf + 11, mask=m, other=0.0)
    nx = m0 * nx_ + m1 * ny_ + m2 * nz_
    ny = m3 * nx_ + m4 * ny_ + m5 * nz_
    nz = m6 * nx_ + m7 * ny_ + m8 * nz_
    nn = tl.sqrt(nx * nx + ny * ny + nz * nz)
    nn = tl.where(nn > 0, nn, 1.0)
    nx = nx / nn
    ny = ny / nn
    nz = nz / nn
    cb = c.to(tl.int64) * 12
    vx = tl.load(Cam + cb + 0) - px
    vy = tl.load(Cam + cb + 1) - py
    vz = tl.load(Cam + cb + 2) - pz
    vn = tl.sqrt(vx * vx + vy * vy + vz * vz)
    vn = tl.where(vn > 0, vn, 1.0)
    vx = vx / vn
    vy = vy / vn
    vz = vz / vn
    cg = (c * G + g).to(tl.int64) * 4
    r = tl.load(Rgba + cg + 0, mask=m, other=0.0)
    gg = tl.load(Rgba + cg + 1, mask=m, other=0.0)
    bb = tl.load(Rgba + cg + 2, mask=m, other=0.0)
    spec = tl.load(Mat + g * 6 + 0, mask=m, other=0.0)
    emis = tl.load(Mat + g * 6 + 2, mask=m, other=0.0)
    lrow = tl.load(Mat + g * 6 + 3, mask=m, other=0.0).to(tl.int64)
    # First pass (lights of role 0 and emission), shadow light (role 1) and every light (translucent geoms).
    f0 = (emis + ambient) * r
    f1 = (emis + ambient) * gg
    f2 = (emis + ambient) * bb
    s0 = ambient * r
    s1 = ambient * gg
    s2 = ambient * bb
    a0 = f0
    a1 = f1
    a2 = f2
    for k in range(L):
        row = Lights + (c * L + k).to(tl.int64) * LW
        directional = tl.load(row + 0) == 1.0
        role = tl.load(row + 1)
        qx = tl.load(row + 2)
        qy = tl.load(row + 3)
        qz = tl.load(row + 4)
        rx = qx - px
        ry = qy - py
        rz = qz - pz
        dist = tl.sqrt(rx * rx + ry * ry + rz * rz)
        dd = tl.where(dist > 0, dist, 1.0)
        lx = tl.where(directional, qx, rx / dd)
        ly = tl.where(directional, qy, ry / dd)
        lz = tl.where(directional, qz, rz / dd)
        att = 1.0 / (tl.load(row + 14) + tl.load(row + 15) * dist + tl.load(row + 16) * dist * dist)
        cutoff = tl.load(row + 20)
        sd = -(lx * tl.load(row + 17) + ly * tl.load(row + 18) + lz * tl.load(row + 19))
        expo = tl.load(row + 21)
        cone = tl.where(sd > 0, tl.exp2(expo * tl.log2(tl.maximum(sd, 1e-30))), tl.where(expo == 0.0, 1.0, 0.0))
        cone = tl.where(sd < cutoff, 0.0, cone)
        att = tl.where(cutoff > -1.5, att * cone, att)
        att = tl.where(directional, 1.0, att)
        t0 = tl.load(row + 5) * r
        t1 = tl.load(row + 6) * gg
        t2 = tl.load(row + 7) * bb
        ndl = nx * lx + ny * ly + nz * lz
        lit = ndl > 0
        t0 += tl.where(lit, ndl * tl.load(row + 8) * r, 0.0)
        t1 += tl.where(lit, ndl * tl.load(row + 9) * gg, 0.0)
        t2 += tl.where(lit, ndl * tl.load(row + 10) * bb, 0.0)
        hx = lx + vx
        hy = ly + vy
        hz = lz + vz
        hn = tl.sqrt(hx * hx + hy * hy + hz * hz)
        ndh = (nx * hx + ny * hy + nz * hz) / tl.where(hn > 0, hn, 1.0)
        shine = lit & (spec > 0) & (hn > 0) & (ndh > 0)
        tt = tl.minimum(ndh, 1.0) * (NLUT - 1)
        i = tl.minimum(tl.maximum(tt, 0.0).to(tl.int32), NLUT - 2)
        lut = Lut + lrow * NLUT + i
        lo = tl.load(lut, mask=m & shine, other=0.0)
        hi = tl.load(lut + 1, mask=m & shine, other=0.0)
        f = (lo + (tt - i) * (hi - lo)) * spec
        t0 += tl.where(shine, f * tl.load(row + 11), 0.0)
        t1 += tl.where(shine, f * tl.load(row + 12), 0.0)
        t2 += tl.where(shine, f * tl.load(row + 13), 0.0)
        first = role == 0.0
        second = role == 1.0
        f0 += tl.where(first, att * t0, 0.0)
        f1 += tl.where(first, att * t1, 0.0)
        f2 += tl.where(first, att * t2, 0.0)
        s0 += tl.where(second, att * t0, 0.0)
        s1 += tl.where(second, att * t1, 0.0)
        s2 += tl.where(second, att * t2, 0.0)
        a0 += att * t0
        a1 += att * t1
        a2 += att * t2
    out = (c.to(tl.int64) * V + v) * 9
    tl.store(Attr + out + 0, tl.minimum(tl.maximum(f0, 0.0), 1.0), mask=m)
    tl.store(Attr + out + 1, tl.minimum(tl.maximum(f1, 0.0), 1.0), mask=m)
    tl.store(Attr + out + 2, tl.minimum(tl.maximum(f2, 0.0), 1.0), mask=m)
    tl.store(Attr + out + 3, tl.minimum(tl.maximum(s0, 0.0), 1.0), mask=m)
    tl.store(Attr + out + 4, tl.minimum(tl.maximum(s1, 0.0), 1.0), mask=m)
    tl.store(Attr + out + 5, tl.minimum(tl.maximum(s2, 0.0), 1.0), mask=m)
    tl.store(Attr + out + 6, px, mask=m)
    tl.store(Attr + out + 7, py, mask=m)
    tl.store(Attr + out + 8, pz, mask=m)
    tr = (c.to(tl.int64) * V + v) * 3
    tl.store(Trans + tr + 0, tl.minimum(tl.maximum(a0, 0.0), 1.0), mask=m)
    tl.store(Trans + tr + 1, tl.minimum(tl.maximum(a1, 0.0), 1.0), mask=m)
    tl.store(Trans + tr + 2, tl.minimum(tl.maximum(a2, 0.0), 1.0), mask=m)
    pb = c.to(tl.int64) * 16
    cl = (c.to(tl.int64) * V + v) * 4
    for row4 in range(4):
        value = (
            tl.load(PV + pb + row4 * 4 + 0) * px
            + tl.load(PV + pb + row4 * 4 + 1) * py
            + tl.load(PV + pb + row4 * 4 + 2) * pz
            + tl.load(PV + pb + row4 * 4 + 3)
        )
        tl.store(Clip + cl + row4, value, mask=m)


def vertices(vpos, vnor, vgeom, transform, rgba, material, lut, cams, pv, lights, ambient):
    """Clip positions (C, V, 4), opaque attributes (C, V, 9) and translucent colours (C, V, 3) of every corner.

    The opaque attributes are the first-pass colour, the shadow light's colour and the world position.
    ``transform`` (C, G, 12) holds each geom's rotation (row major) and position.
    """
    C, G = rgba.shape[:2]
    V = len(vpos)
    clip = torch.empty((C, V, 4), device=vpos.device)
    attr = torch.empty((C, V, 9), device=vpos.device)
    trans = torch.empty((C, V, 3), device=vpos.device)
    block = 128
    with torch.cuda.device(vpos.device):
        _vertices[(triton.cdiv(V, block), C)](
            vpos,
            vnor,
            vgeom,
            transform,
            rgba,
            material,
            lut,
            cams,
            pv,
            lights,
            clip,
            attr,
            trans,
            V,
            G,
            lights.shape[1],
            lut.shape[1],
            float(ambient),
            block,
            lights.shape[2],
        )
    return clip, attr, trans


@triton.jit
def _reached(Static, Dyn, Look, c, x, y, z, m, N, i0, j0, ni, nj):
    """Whether the shadow light reaches world points (x, y, z): no caster in their shadow-map texel is nearer."""
    s = tl.load(Look + 0) * x + tl.load(Look + 1) * y + tl.load(Look + 2) * z + tl.load(Look + 3)
    t = tl.load(Look + 4) * x + tl.load(Look + 5) * y + tl.load(Look + 6) * z + tl.load(Look + 7)
    r = tl.load(Look + 8) * x + tl.load(Look + 9) * y + tl.load(Look + 10) * z + tl.load(Look + 11)
    r = tl.minimum(tl.maximum(r, 0.0), 1.0)
    si = tl.minimum(tl.maximum(tl.floor(s), 0.0), N - 1.0).to(tl.int32)
    ti = tl.minimum(tl.maximum(tl.floor(t), 0.0), N - 1.0).to(tl.int32)
    top = tl.load(Static + ti.to(tl.int64) * N + si, mask=m, other=0.0)
    di = si - i0
    dj = ti - j0
    inside = m & (di >= 0) & (di < ni) & (dj >= 0) & (dj < nj)
    dyn = tl.load(Dyn + (c.to(tl.int64) * nj + dj) * ni + di, mask=inside, other=0.0)
    return (r >= top) & (r >= dyn)


@triton.jit
def _opaque(
    Rast,
    RastDb,
    Tri,
    Attr,
    OGeom,
    TexMod,
    Index,
    First,
    Second,
    Sky,
    Cover,
    Depth,
    XY,
    Static,
    Dyn,
    Look,
    Base,
    Zbuf,
    V,
    k,
    sx,
    sy,
    N,
    i0,
    j0,
    ni,
    nj,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
    SHADOW: tl.constexpr,
):
    c = tl.program_id(1)
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = p < S * S
    u_ = tl.load(Index + c)
    pix = c.to(tl.int64) * S * S + p
    bpix = u_.to(tl.int64) * S * S + p
    cover = tl.load(Cover + (u_ * 4 + k).to(tl.int64) * S * S + p, mask=m, other=0) != 0
    depth = tl.load(Depth + (u_ * 4 + k).to(tl.int64) * S * S + p, mask=m, other=1.0)
    u = tl.load(Rast + pix * 4 + 0, mask=m, other=0.0)
    v = tl.load(Rast + pix * 4 + 1, mask=m, other=0.0)
    z = tl.load(Rast + pix * 4 + 2, mask=m, other=1.0)
    tid = tl.load(Rast + pix * 4 + 3, mask=m, other=0.0).to(tl.int32)
    hit = m & (tid > 0)
    # NVIDIA OpenGL shades at the pixel centre when it lies in the triangle, else at the sample.
    du_dx = tl.load(RastDb + pix * 4 + 0, mask=hit, other=0.0)
    du_dy = tl.load(RastDb + pix * 4 + 1, mask=hit, other=0.0)
    dv_dx = tl.load(RastDb + pix * 4 + 2, mask=hit, other=0.0)
    dv_dy = tl.load(RastDb + pix * 4 + 3, mask=hit, other=0.0)
    uc = u + du_dx * (0.5 - sx) + du_dy * (0.5 - sy)
    vc = v + dv_dx * (0.5 - sx) + dv_dy * (0.5 - sy)
    centre = (uc >= 0) & (vc >= 0) & (uc + vc <= 1)
    u = tl.where(centre, uc, u)
    v = tl.where(centre, vc, v)
    w = 1.0 - u - v
    t = tl.maximum(tid - 1, 0)
    a0 = (c.to(tl.int64) * V + tl.load(Tri + t * 3 + 0, mask=hit, other=0)) * 9
    a1 = (c.to(tl.int64) * V + tl.load(Tri + t * 3 + 1, mask=hit, other=0)) * 9
    a2 = (c.to(tl.int64) * V + tl.load(Tri + t * 3 + 2, mask=hit, other=0)) * 9
    px = (
        u * tl.load(Attr + a0 + 6, mask=hit, other=0.0)
        + v * tl.load(Attr + a1 + 6, mask=hit, other=0.0)
        + w * tl.load(Attr + a2 + 6, mask=hit, other=0.0)
    )
    py = (
        u * tl.load(Attr + a0 + 7, mask=hit, other=0.0)
        + v * tl.load(Attr + a1 + 7, mask=hit, other=0.0)
        + w * tl.load(Attr + a2 + 7, mask=hit, other=0.0)
    )
    pz = (
        u * tl.load(Attr + a0 + 8, mask=hit, other=0.0)
        + v * tl.load(Attr + a1 + 8, mask=hit, other=0.0)
        + w * tl.load(Attr + a2 + 8, mask=hit, other=0.0)
    )
    if SHADOW:
        lit = _reached(Static, Dyn, Look, c, px, py, pz, hit, N, i0, j0, ni, nj)
        fx = tl.load(XY + bpix * 2 + 0, mask=m, other=0.0)
        fy = tl.load(XY + bpix * 2 + 1, mask=m, other=0.0)
        floor_lit = _reached(Static, Dyn, Look, c, fx, fy, tl.zeros([BLOCK], tl.float32), m & cover, N, i0, j0, ni, nj)
    else:
        lit = hit & (tid < 0)
        floor_lit = m & (p < 0)
    geom = tl.load(OGeom + t, mask=hit, other=0)
    for ch in range(3):
        tex = tl.load(TexMod + geom * 3 + ch, mask=hit, other=1.0)
        c1 = (
            u * tl.load(Attr + a0 + ch, mask=hit, other=0.0)
            + v * tl.load(Attr + a1 + ch, mask=hit, other=0.0)
            + w * tl.load(Attr + a2 + ch, mask=hit, other=0.0)
        )
        c2 = (
            u * tl.load(Attr + a0 + 3 + ch, mask=hit, other=0.0)
            + v * tl.load(Attr + a1 + 3 + ch, mask=hit, other=0.0)
            + w * tl.load(Attr + a2 + 3 + ch, mask=hit, other=0.0)
        )
        first = tl.floor(tl.minimum(tl.maximum(tl.minimum(tl.maximum(c1, 0.0), 1.0) * tex, 0.0), 1.0) * 255.0 + 0.5)
        add = tl.floor(tl.minimum(tl.maximum(tl.minimum(tl.maximum(c2, 0.0), 1.0) * tex, 0.0), 1.0) * 255.0 + 0.5)
        obj = tl.where(lit, tl.minimum(first + add, 255.0), first)
        bf = tl.load(First + bpix * 3 + ch, mask=m, other=0).to(tl.float32)
        bs = tl.load(Second + bpix * 3 + ch, mask=m, other=0).to(tl.float32)
        sky = tl.load(Sky + bpix * 3 + ch, mask=m, other=0).to(tl.float32)
        back = tl.where(cover, tl.where(floor_lit, bs, bf), sky)
        tl.store(Base + ((c * 4 + k).to(tl.int64) * S * S + p) * 3 + ch, tl.where(hit, obj, back).to(tl.uint8), mask=m)
    tl.store(Zbuf + (c * 4 + k).to(tl.int64) * S * S + p, tl.where(hit, z, depth), mask=m)


def opaque(rast, rast_db, tri, attr, geom, texmod, index, background, shadow, base, zbuf, k, sample):
    """Write sample ``k`` of ``base`` (C, 4, S, S, 3) and ``zbuf`` from the opaque raster and the background.

    ``shadow`` is None without a shadow light, else (static map, moving-caster window, lookup matrix, window).
    """
    C, S = rast.shape[0], rast.shape[1]
    V = attr.shape[1]
    block = 256
    if shadow is None:
        static = dyn = torch.zeros(1, device=rast.device)
        look = torch.zeros(12, device=rast.device)
        i0 = j0 = i1 = j1 = 0
        n = 1
    else:
        static, dyn, look, ((i0, j0), (i1, j1)) = shadow
        n = static.shape[0]
    with torch.cuda.device(rast.device):
        _opaque[(triton.cdiv(S * S, block), C)](
            rast,
            rast_db,
            tri,
            attr,
            geom,
            texmod,
            index,
            background["first"],
            background["second"],
            background["sky"],
            background["cover"],
            background["depth"],
            background["xy"],
            static,
            dyn,
            look,
            base,
            zbuf,
            V,
            k,
            float(sample[0]),
            float(sample[1]),
            int(n),
            int(i0),
            int(j0),
            int(i1 - i0),
            int(j1 - j0),
            S,
            block,
            shadow is not None,
        )


@triton.jit
def _light_xyz(CPos, Tri, t, j, Xf, xf, Smat, i0, j0):
    """Shadow-map window x, y (less the window origin) and light depth of corner ``j`` of triangle ``t``."""
    i = tl.load(Tri + t * 3 + j)
    lx = tl.load(CPos + i * 3 + 0)
    ly = tl.load(CPos + i * 3 + 1)
    lz = tl.load(CPos + i * 3 + 2)
    wx = tl.load(Xf + xf + 0) * lx + tl.load(Xf + xf + 1) * ly + tl.load(Xf + xf + 2) * lz + tl.load(Xf + xf + 9)
    wy = tl.load(Xf + xf + 3) * lx + tl.load(Xf + xf + 4) * ly + tl.load(Xf + xf + 5) * lz + tl.load(Xf + xf + 10)
    wz = tl.load(Xf + xf + 6) * lx + tl.load(Xf + xf + 7) * ly + tl.load(Xf + xf + 8) * lz + tl.load(Xf + xf + 11)
    x = tl.load(Smat + 0) * wx + tl.load(Smat + 1) * wy + tl.load(Smat + 2) * wz + tl.load(Smat + 3) - i0
    y = tl.load(Smat + 4) * wx + tl.load(Smat + 5) * wy + tl.load(Smat + 6) * wz + tl.load(Smat + 7) - j0
    z = tl.load(Smat + 8) * wx + tl.load(Smat + 9) * wy + tl.load(Smat + 10) * wz + tl.load(Smat + 11)
    return x, y, z


@triton.jit
def _shadow(CPos, Tri, TGeom, Xf, Smat, Dyn, G, i0, j0, ni, nj, BLOCK: tl.constexpr):
    t = tl.program_id(0)
    c = tl.program_id(1)
    g = tl.load(TGeom + t)
    xf = (c * G + g).to(tl.int64) * 12
    x0, y0, z0 = _light_xyz(CPos, Tri, t, 0, Xf, xf, Smat, i0, j0)
    x1, y1, z1 = _light_xyz(CPos, Tri, t, 1, Xf, xf, Smat, i0, j0)
    x2, y2, z2 = _light_xyz(CPos, Tri, t, 2, Xf, xf, Smat, i0, j0)
    det = (x1 - x0) * (y2 - y0) - (x2 - x0) * (y1 - y0)
    # As in the CPU backend, only caster triangles that face the light draw the moving shadow map.
    valid = det > 0
    dd = tl.where(valid, det, 1.0)
    dzdx = ((z1 - z0) * (y2 - y0) - (z2 - z0) * (y1 - y0)) / dd
    dzdy = ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) / dd
    off = _OFFSET_FACTOR * tl.maximum(tl.abs(dzdx), tl.abs(dzdy)) + _OFFSET_UNITS / _DEPTH_BITS
    lox = tl.maximum(tl.ceil(tl.minimum(x0, tl.minimum(x1, x2)) - 0.5), 0.0).to(tl.int32)
    hix = tl.minimum(tl.floor(tl.maximum(x0, tl.maximum(x1, x2)) - 0.5), ni - 1.0).to(tl.int32)
    loy = tl.maximum(tl.ceil(tl.minimum(y0, tl.minimum(y1, y2)) - 0.5), 0.0).to(tl.int32)
    hiy = tl.minimum(tl.floor(tl.maximum(y0, tl.maximum(y1, y2)) - 0.5), nj - 1.0).to(tl.int32)
    width = hix - lox + 1
    height = hiy - loy + 1
    n = tl.where(valid & (width > 0) & (height > 0), width * height, 0)
    for start in range(0, n, BLOCK):
        i = start + tl.arange(0, BLOCK)
        m = i < n
        px = lox + i % width
        py = loy + i // width
        cx = px.to(tl.float32) + 0.5
        cy = py.to(tl.float32) + 0.5
        e0 = (x1 - cx) * (y2 - cy) - (x2 - cx) * (y1 - cy)
        e1 = (x2 - cx) * (y0 - cy) - (x0 - cx) * (y2 - cy)
        e2 = (x0 - cx) * (y1 - cy) - (x1 - cx) * (y0 - cy)
        inside = m & (e0 >= 0) & (e1 >= 0) & (e2 >= 0)
        z = tl.minimum(tl.maximum(z0 + dzdx * (cx - x0) + dzdy * (cy - y0) + off, 0.0), 1.0)
        bits = z.to(tl.int32, bitcast=True)
        tl.atomic_max(Dyn + (c.to(tl.int64) * nj + py) * ni + px, bits, mask=inside, sem="relaxed")


def shadow(cpos, tri, tgeom, transform, smat, window):
    """Light depth (C, nj, ni) of the caster triangles ``tri`` over a window of the shadow map.

    The depth has MuJoCo's polygon offset; a texel that no caster covers holds 0.
    """
    (i0, j0), (i1, j1) = window
    C, G = transform.shape[:2]
    ni, nj = int(i1 - i0), int(j1 - j0)
    bits = torch.zeros((C, nj, ni), dtype=torch.int32, device=transform.device)
    if len(tri) and ni > 0 and nj > 0:
        with torch.cuda.device(transform.device):
            _shadow[(len(tri), C)](cpos, tri, tgeom, transform, smat, bits, G, float(i0), float(j0), ni, nj, 64)
    return bits.view(torch.float32)


@triton.jit
def _append(
    px,
    py,
    m,
    x0,
    y0,
    z0,
    x1,
    y1,
    z1,
    x2,
    y2,
    z2,
    inv,
    packed,
    c,
    Zbuf,
    Count,
    Frag,
    sx,
    sy,
    k,
    S: tl.constexpr,
    K: tl.constexpr,
):
    xs = px.to(tl.float32) + sx
    ys = py.to(tl.float32) + sy
    e0 = (x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)
    e1 = (x0 - x2) * (ys - y2) - (y0 - y2) * (xs - x2)
    e2 = (x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0)
    inside = m & (e0 >= 0) & (e1 >= 0) & (e2 >= 0)
    z = (e0 * z0 + e1 * z1 + e2 * z2) * inv
    off = ((c * 4 + k) * S + py).to(tl.int64) * S + px
    zb = tl.load(Zbuf + off, mask=inside, other=-2.0)
    ok = inside & (z <= zb) & (z >= -1.0)
    slot = tl.atomic_add(Count + off, 1, mask=ok, sem="relaxed")
    tl.store(Frag + off * K + slot, packed, mask=ok & (slot < K))


@triton.jit
def _corner(Clip, Tri, c, t, j, V, S: tl.constexpr):
    """Window x, y (rows up), NDC depth and clip w of corner ``j`` of triangle ``t``."""
    i = tl.load(Tri + t * 3 + j)
    o = (c.to(tl.int64) * V + i) * 4
    x = tl.load(Clip + o + 0)
    y = tl.load(Clip + o + 1)
    z = tl.load(Clip + o + 2)
    w = tl.load(Clip + o + 3)
    return (x / w * 0.5 + 0.5) * S, (y / w * 0.5 + 0.5) * S, z / w, w


@triton.jit
def _fragments(
    Clip, Tri, TGeom, Alpha, Zbuf, Count, Frag, Samples, V, G, S: tl.constexpr, K: tl.constexpr, BLOCK: tl.constexpr
):
    t = tl.program_id(0)
    c = tl.program_id(1)
    geom = tl.load(TGeom + t)
    alpha = tl.load(Alpha + c.to(tl.int64) * G + geom)
    x0, y0, z0, w0 = _corner(Clip, Tri, c, t, 0, V, S)
    x1, y1, z1, w1 = _corner(Clip, Tri, c, t, 1, V, S)
    x2, y2, z2, w2 = _corner(Clip, Tri, c, t, 2, V, S)
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    # Back faces are culled; a triangle crossing the camera plane is not drawn; opaque geoms of this frame are not
    # translucent fragments.
    valid = (area > 0) & (w0 > 0) & (w1 > 0) & (w2 > 0) & (alpha < _TRANSPARENT_ALPHA)
    lox = tl.maximum(tl.ceil(tl.minimum(x0, tl.minimum(x1, x2)) - 0.875), 0.0).to(tl.int32)
    hix = tl.minimum(tl.floor(tl.maximum(x0, tl.maximum(x1, x2)) - 0.125), S - 1.0).to(tl.int32)
    loy = tl.maximum(tl.ceil(tl.minimum(y0, tl.minimum(y1, y2)) - 0.875), 0.0).to(tl.int32)
    hiy = tl.minimum(tl.floor(tl.maximum(y0, tl.maximum(y1, y2)) - 0.125), S - 1.0).to(tl.int32)
    width = hix - lox + 1
    height = hiy - loy + 1
    n = tl.where(valid & (width > 0) & (height > 0), width * height, 0)
    inv = 1.0 / tl.where(area != 0, area, 1.0)
    packed = t
    for start in range(0, n, BLOCK):
        i = start + tl.arange(0, BLOCK)
        m = i < n
        px = lox + i % width
        py = loy + i // width
        for k in range(4):
            sx = tl.load(Samples + 2 * k)
            sy = tl.load(Samples + 2 * k + 1)
            _append(px, py, m, x0, y0, z0, x1, y1, z1, x2, y2, z2, inv, packed, c, Zbuf, Count, Frag, sx, sy, k, S, K)


@triton.jit
def _blend_list(
    r,
    g,
    b,
    entries,
    m,
    px,
    py,
    c,
    Clip,
    Tri,
    Col,
    Alpha,
    TGeom,
    TexMod,
    V,
    G,
    sx,
    sy,
    S: tl.constexpr,
    K: tl.constexpr,
    EMPTY_: tl.constexpr,
):
    cols = tl.arange(0, K)
    for j in range(K):
        e = tl.sum(tl.where(cols[None, :] == j, entries, 0), 1)
        ok = m & (e != EMPTY_)
        t = (e & ((1 << 20) - 1)).to(tl.int32)
        geom = tl.load(TGeom + t, mask=ok, other=0)
        a = tl.load(Alpha + c * G + geom, mask=ok, other=0.0)
        a = tl.minimum(tl.maximum(a, 0.0), 1.0)
        i0 = tl.load(Tri + t * 3 + 0, mask=ok, other=0)
        i1 = tl.load(Tri + t * 3 + 1, mask=ok, other=0)
        i2 = tl.load(Tri + t * 3 + 2, mask=ok, other=0)
        o0 = (c.to(tl.int64) * V + i0) * 4
        o1 = (c.to(tl.int64) * V + i1) * 4
        o2 = (c.to(tl.int64) * V + i2) * 4
        w0 = tl.load(Clip + o0 + 3, mask=ok, other=1.0)
        w1 = tl.load(Clip + o1 + 3, mask=ok, other=1.0)
        w2 = tl.load(Clip + o2 + 3, mask=ok, other=1.0)
        x0 = (tl.load(Clip + o0 + 0, mask=ok, other=0.0) / w0 * 0.5 + 0.5) * S
        y0 = (tl.load(Clip + o0 + 1, mask=ok, other=0.0) / w0 * 0.5 + 0.5) * S
        x1 = (tl.load(Clip + o1 + 0, mask=ok, other=1.0) / w1 * 0.5 + 0.5) * S
        y1 = (tl.load(Clip + o1 + 1, mask=ok, other=0.0) / w1 * 0.5 + 0.5) * S
        x2 = (tl.load(Clip + o2 + 0, mask=ok, other=0.0) / w2 * 0.5 + 0.5) * S
        y2 = (tl.load(Clip + o2 + 1, mask=ok, other=1.0) / w2 * 0.5 + 0.5) * S
        area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
        inv = 1.0 / tl.where(area != 0, area, 1.0)
        # OpenGL shades at the pixel centre when it lies in the triangle, else at the sample.
        xc = px.to(tl.float32) + 0.5
        yc = py.to(tl.float32) + 0.5
        e0 = (x2 - x1) * (yc - y1) - (y2 - y1) * (xc - x1)
        e1 = (x0 - x2) * (yc - y2) - (y0 - y2) * (xc - x2)
        e2 = (x1 - x0) * (yc - y0) - (y1 - y0) * (xc - x0)
        centre = (e0 >= 0) & (e1 >= 0) & (e2 >= 0)
        xs = tl.where(centre, xc, px.to(tl.float32) + sx)
        ys = tl.where(centre, yc, py.to(tl.float32) + sy)
        b0 = ((x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)) * inv / w0
        b1 = ((x0 - x2) * (ys - y2) - (y0 - y2) * (xs - x2)) * inv / w1
        b2 = ((x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0)) * inv / w2
        total = b0 + b1 + b2
        total = tl.where(total != 0, total, 1.0)
        b0 = b0 / total
        b1 = b1 / total
        b2 = b2 / total
        q0 = (c.to(tl.int64) * V + i0) * 3
        q1 = (c.to(tl.int64) * V + i1) * 3
        q2 = (c.to(tl.int64) * V + i2) * 3
        cr = (
            b0 * tl.load(Col + q0 + 0, mask=ok, other=0.0)
            + b1 * tl.load(Col + q1 + 0, mask=ok, other=0.0)
            + b2 * tl.load(Col + q2 + 0, mask=ok, other=0.0)
        )
        cg = (
            b0 * tl.load(Col + q0 + 1, mask=ok, other=0.0)
            + b1 * tl.load(Col + q1 + 1, mask=ok, other=0.0)
            + b2 * tl.load(Col + q2 + 1, mask=ok, other=0.0)
        )
        cc = (
            b0 * tl.load(Col + q0 + 2, mask=ok, other=0.0)
            + b1 * tl.load(Col + q1 + 2, mask=ok, other=0.0)
            + b2 * tl.load(Col + q2 + 2, mask=ok, other=0.0)
        )
        cr = tl.minimum(tl.maximum(cr, 0.0), 1.0) * tl.load(TexMod + geom * 3 + 0, mask=ok, other=1.0)
        cg = tl.minimum(tl.maximum(cg, 0.0), 1.0) * tl.load(TexMod + geom * 3 + 1, mask=ok, other=1.0)
        cc = tl.minimum(tl.maximum(cc, 0.0), 1.0) * tl.load(TexMod + geom * 3 + 2, mask=ok, other=1.0)
        r = tl.where(ok, tl.floor(cr * a * 255.0 + r * (1.0 - a) + 0.5), r)
        g = tl.where(ok, tl.floor(cg * a * 255.0 + g * (1.0 - a) + 0.5), g)
        b = tl.where(ok, tl.floor(cc * a * 255.0 + b * (1.0 - a) + 0.5), b)
    return r, g, b


@triton.jit
def _resolve(
    Base,
    Count,
    Frag,
    Clip,
    Tri,
    Col,
    Alpha,
    TGeom,
    TLocal,
    Rank,
    TexMod,
    Out,
    Samples,
    V,
    G,
    S: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
    EMPTY_: tl.constexpr,
    MAXR: tl.constexpr,
):
    c = tl.program_id(1)
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = p < S * S
    px = p % S
    py = p // S
    acc_r = tl.zeros([BLOCK], tl.float32)
    acc_g = tl.zeros([BLOCK], tl.float32)
    acc_b = tl.zeros([BLOCK], tl.float32)
    cols = tl.arange(0, K)
    for k in range(4):
        sx = tl.load(Samples + 2 * k)
        sy = tl.load(Samples + 2 * k + 1)
        off = (c * 4 + k).to(tl.int64) * S * S + p
        r = tl.load(Base + off * 3 + 0, mask=m, other=0).to(tl.float32)
        g = tl.load(Base + off * 3 + 1, mask=m, other=0).to(tl.float32)
        b = tl.load(Base + off * 3 + 2, mask=m, other=0).to(tl.float32)
        n = tl.minimum(tl.load(Count + off, mask=m, other=0), K)
        if tl.max(n, 0) > 0:
            kept = m[:, None] & (cols[None, :] < n[:, None])
            tri = tl.load(Frag + off[:, None] * K + cols[None, :], mask=kept, other=0)
            geom = tl.load(TGeom + tri, mask=kept, other=0)
            rank_ = tl.load(Rank + c * G + geom, mask=kept, other=0).to(tl.int64)
            local = tl.load(TLocal + tri, mask=kept, other=0).to(tl.int64)
            entries = tl.where(kept, (((rank_ << 16) | local) << 20) | tri.to(tl.int64), EMPTY_)
            forward = tl.sort(entries, 1)
            rank = forward >> 36
            rest = forward & ((1 << 36) - 1)
            backward = tl.sort(tl.where(forward == EMPTY_, EMPTY_, ((MAXR - rank) << 36) | rest), 1)
            for order in range(2):
                listed = tl.where(order == 0, forward, backward)
                r, g, b = _blend_list(
                    r, g, b, listed, m, px, py, c, Clip, Tri, Col, Alpha, TGeom, TexMod, V, G, sx, sy, S, K, EMPTY_
                )
        acc_r += r
        acc_g += g
        acc_b += b
    # Image rows run top down.
    o = ((c.to(tl.int64) * S + (S - 1 - py)) * S + px) * 3
    tl.store(Out + o + 0, tl.floor((acc_r + 2.0) * 0.25).to(tl.uint8), mask=m)
    tl.store(Out + o + 1, tl.floor((acc_g + 2.0) * 0.25).to(tl.uint8), mask=m)
    tl.store(Out + o + 2, tl.floor((acc_b + 2.0) * 0.25).to(tl.uint8), mask=m)


def blend(clip, tri, color, rank, tgeom, tlocal, alpha, texmod, base, zbuf, layers, samples):
    """Blend translucent triangles over the samples ``base`` (C, 4, S, S, 3) and resolve them.

    ``clip`` (C, V, 4) and ``color`` (C, V, 3) are the corners' clip positions and lit colours; ``tri`` (T, 3)
    the candidate translucent triangles. A triangle whose geom is opaque in a frame is left out of that frame.
    Each sample keeps ``layers`` fragments, which blend in rank order of their geom and then in reverse rank
    order, each geom in triangle order, as MuJoCo draws translucent geoms. Returns (C, S, S, 3) uint8 frames with
    top-down rows.
    """
    C, V, _ = clip.shape
    T = len(tri)
    S = base.shape[2]
    G = rank.shape[1]
    if T >= 1 << TRI_BITS or G >= 1 << RANK_BITS or int(tlocal.max()) >= 1 << LOCAL_BITS:
        raise ValueError("translucent geometry exceeds the fragment packing")
    count = torch.zeros((C, 4, S, S), dtype=torch.int32, device=base.device)
    frag = torch.empty((C, 4, S, S, layers), dtype=torch.int32, device=base.device)
    positions = torch.tensor([v for xy in samples for v in xy], dtype=torch.float32, device=base.device)
    with torch.cuda.device(base.device):
        _fragments[(T, C)](clip, tri, tgeom, alpha, zbuf, count, frag, positions, V, G, S, layers, 32)
        out = torch.empty((C, S, S, 3), dtype=torch.uint8, device=base.device)
        block = 64
        _resolve[(triton.cdiv(S * S, block), C)](
            base,
            count,
            frag,
            clip,
            tri,
            color,
            alpha,
            tgeom,
            tlocal,
            rank,
            texmod,
            out,
            positions,
            V,
            G,
            S,
            layers,
            block,
            EMPTY,
            (1 << RANK_BITS) - 1,
        )
    return out
