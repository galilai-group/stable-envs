"""Triton kernels of the Cube/Scene CUDA rasterizer: transparent fragment lists and their ordered blend."""

from __future__ import annotations

import torch
import triton
import triton.language as tl

EMPTY = (1 << 63) - 1
#: Bits of a packed fragment: geom rank (15), triangle index in its geom (16), transparent triangle (20).
TRI_BITS, LOCAL_BITS, RANK_BITS = 20, 16, 15


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
    Clip,
    Tri,
    Rank,
    TGeom,
    TLocal,
    Zbuf,
    Count,
    Frag,
    Samples,
    T,
    V,
    G,
    S: tl.constexpr,
    K: tl.constexpr,
    BLOCK: tl.constexpr,
):
    t = tl.program_id(0)
    c = tl.program_id(1)
    x0, y0, z0, w0 = _corner(Clip, Tri, c, t, 0, V, S)
    x1, y1, z1, w1 = _corner(Clip, Tri, c, t, 1, V, S)
    x2, y2, z2, w2 = _corner(Clip, Tri, c, t, 2, V, S)
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    # Back faces are culled; a triangle crossing the camera plane is not drawn.
    valid = (area > 0) & (w0 > 0) & (w1 > 0) & (w2 > 0)
    lox = tl.maximum(tl.ceil(tl.minimum(x0, tl.minimum(x1, x2)) - 0.875), 0.0).to(tl.int32)
    hix = tl.minimum(tl.floor(tl.maximum(x0, tl.maximum(x1, x2)) - 0.125), S - 1.0).to(tl.int32)
    loy = tl.maximum(tl.ceil(tl.minimum(y0, tl.minimum(y1, y2)) - 0.875), 0.0).to(tl.int32)
    hiy = tl.minimum(tl.floor(tl.maximum(y0, tl.maximum(y1, y2)) - 0.125), S - 1.0).to(tl.int32)
    width = hix - lox + 1
    height = hiy - loy + 1
    n = tl.where(valid & (width > 0) & (height > 0), width * height, 0)
    inv = 1.0 / area
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
        inv = 1.0 / area
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
        cr = tl.minimum(tl.maximum(cr, 0.0), 1.0)
        cg = tl.minimum(tl.maximum(cg, 0.0), 1.0)
        cc = tl.minimum(tl.maximum(cc, 0.0), 1.0)
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
        n = tl.load(Count + off, mask=m, other=0)
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
                    r, g, b, listed, m, px, py, c, Clip, Tri, Col, Alpha, TGeom, V, G, sx, sy, S, K, EMPTY_
                )
        acc_r += r
        acc_g += g
        acc_b += b
    # Image rows run top down.
    o = ((c.to(tl.int64) * S + (S - 1 - py)) * S + px) * 3
    tl.store(Out + o + 0, tl.floor((acc_r + 2.0) * 0.25).to(tl.uint8), mask=m)
    tl.store(Out + o + 1, tl.floor((acc_g + 2.0) * 0.25).to(tl.uint8), mask=m)
    tl.store(Out + o + 2, tl.floor((acc_b + 2.0) * 0.25).to(tl.uint8), mask=m)


def blend(clip, tri, color, rank, tgeom, tlocal, alpha, base, zbuf, layers, samples):
    """Blend transparent triangles over the samples ``base`` (C, 4, S, S, 3) and resolve them.

    ``clip`` (C, V, 4) and ``color`` (C, V, 3) are the vertices' clip positions and lit colours; ``tri`` (T, 3)
    the transparent triangles. Each sample keeps ``layers`` fragments, which blend in rank order of their geom
    and then in reverse rank order, each geom in triangle order, as MuJoCo draws transparent geoms. Returns
    (C, S, S, 3) uint8 frames with top-down rows.
    """
    C, V, _ = clip.shape
    T = len(tri)
    S = base.shape[2]
    G = rank.shape[1]
    if T >= 1 << TRI_BITS or G >= 1 << RANK_BITS or int(tlocal.max()) >= 1 << LOCAL_BITS:
        raise ValueError("Cube/Scene transparent geometry exceeds the fragment packing")
    count = torch.zeros((C, 4, S, S), dtype=torch.int32, device=base.device)
    frag = torch.empty((C, 4, S, S, layers), dtype=torch.int32, device=base.device)
    positions = torch.tensor([v for xy in samples for v in xy], dtype=torch.float32, device=base.device)
    with torch.cuda.device(base.device):
        _fragments[(T, C)](clip, tri, rank, tgeom, tlocal, zbuf, count, frag, positions, T, V, G, S, layers, 32)
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


#: Shadow map texels hold caster heights plus this, as float bits, so that an integer maximum orders them.
HEIGHT_BIAS = 16.0


@triton.jit
def _shadow(
    Pos,
    Tri,
    Height,
    T,
    V,
    i0,
    j0,
    ni,
    nj,
    origin_x,
    origin_y,
    texel,
    BLOCK: tl.constexpr,
    BIAS: tl.constexpr,
    STRIDE: tl.constexpr,
    FIRST: tl.constexpr,
):
    t = tl.program_id(0)
    c = tl.program_id(1)
    a = tl.load(Tri + t * 3 + 0)
    b = tl.load(Tri + t * 3 + 1)
    d = tl.load(Tri + t * 3 + 2)
    pa = (c.to(tl.int64) * V + a) * STRIDE + FIRST
    pb = (c.to(tl.int64) * V + b) * STRIDE + FIRST
    pd = (c.to(tl.int64) * V + d) * STRIDE + FIRST
    x0 = (tl.load(Pos + pa + 0) - origin_x) / texel - i0
    y0 = (tl.load(Pos + pa + 1) - origin_y) / texel - j0
    z0 = tl.load(Pos + pa + 2)
    x1 = (tl.load(Pos + pb + 0) - origin_x) / texel - i0
    y1 = (tl.load(Pos + pb + 1) - origin_y) / texel - j0
    z1 = tl.load(Pos + pb + 2)
    x2 = (tl.load(Pos + pd + 0) - origin_x) / texel - i0
    y2 = (tl.load(Pos + pd + 1) - origin_y) / texel - j0
    z2 = tl.load(Pos + pd + 2)
    area = (x1 - x0) * (y2 - y0) - (y1 - y0) * (x2 - x0)
    # Height gradient per texel; MuJoCo's polygon offset is 1.5 times its largest component.
    gx = ((z1 - z0) * (y2 - y0) - (z2 - z0) * (y1 - y0)) / area
    gy = ((x1 - x0) * (z2 - z0) - (x2 - x0) * (z1 - z0)) / area
    offset = 1.5 * tl.maximum(tl.abs(gx), tl.abs(gy))
    lox = tl.maximum(tl.ceil(tl.minimum(x0, tl.minimum(x1, x2)) - 0.5), 0.0).to(tl.int32)
    hix = tl.minimum(tl.floor(tl.maximum(x0, tl.maximum(x1, x2)) - 0.5), ni - 1.0).to(tl.int32)
    loy = tl.maximum(tl.ceil(tl.minimum(y0, tl.minimum(y1, y2)) - 0.5), 0.0).to(tl.int32)
    hiy = tl.minimum(tl.floor(tl.maximum(y0, tl.maximum(y1, y2)) - 0.5), nj - 1.0).to(tl.int32)
    width = hix - lox + 1
    height = hiy - loy + 1
    n = tl.where((tl.abs(area) > 1e-12) & (width > 0) & (height > 0), width * height, 0)
    sign = tl.where(area > 0, 1.0, -1.0)
    inv = 1.0 / area
    for start in range(0, n, BLOCK):
        i = start + tl.arange(0, BLOCK)
        m = i < n
        px = lox + i % width
        py = loy + i // width
        xs = px.to(tl.float32) + 0.5
        ys = py.to(tl.float32) + 0.5
        e0 = (x2 - x1) * (ys - y1) - (y2 - y1) * (xs - x1)
        e1 = (x0 - x2) * (ys - y2) - (y0 - y2) * (xs - x2)
        e2 = (x1 - x0) * (ys - y0) - (y1 - y0) * (xs - x0)
        inside = m & (e0 * sign >= 0) & (e1 * sign >= 0) & (e2 * sign >= 0)
        z = (e0 * z0 + e1 * z1 + e2 * z2) * inv - offset + BIAS
        bits = tl.maximum(z, 0.0).to(tl.int32, bitcast=True)
        tl.atomic_max(Height + (c.to(tl.int64) * nj + py) * ni + px, bits, mask=inside, sem="relaxed")


def shadow_heights(attr, tri, window, origin, texel):
    """Top-down heights (C, nj, ni) of the triangles ``tri`` over a window of texels.

    ``attr`` (C, V, 9) holds the world vertices in its last three columns. Each texel holds the highest caster
    less MuJoCo's polygon offset, plus ``HEIGHT_BIAS``; texels without a caster hold 0.
    """
    (i0, j0), (i1, j1) = window
    C, V, _ = attr.shape
    ni, nj = int(i1 - i0), int(j1 - j0)
    bits = torch.zeros((C, nj, ni), dtype=torch.int32, device=attr.device)
    with torch.cuda.device(attr.device):
        _shadow[(len(tri), C)](
            attr,
            tri,
            bits,
            len(tri),
            V,
            float(i0),
            float(j0),
            ni,
            nj,
            float(origin[0]),
            float(origin[1]),
            float(texel),
            64,
            HEIGHT_BIAS,
            9,
            6,
        )
    return bits.view(torch.float32)


#: Fields of a row of the light table that ``vertices`` reads.
LIGHT_FIELDS = (
    "point",
    "shadow",
    "global",
    "x",
    "y",
    "z",
    "ambient",
    "diffuse",
    "specular",
    "att0",
    "att1",
    "att2",
    "cone",
    "cos",
    "exponent",
    "sx",
    "sy",
    "sz",
)


@triton.jit
def _specular_term(nx, ny, nz, lx, ly, lz, vx, vy, vz, ndl, shin):
    hx = lx + vx
    hy = ly + vy
    hz = lz + vz
    hn = tl.sqrt(hx * hx + hy * hy + hz * hz)
    ndh = tl.maximum((nx * hx + ny * hy + nz * hz) / tl.maximum(hn, 1e-12), 0.0)
    power = tl.where(ndh > 0, tl.exp2(shin * tl.log2(tl.maximum(ndh, 1e-30))), 0.0)
    return tl.where(ndl > 0, power, 0.0)


@triton.jit
def _vertices(
    VPos,
    VNor,
    VGeom,
    Xf,
    Rot,
    Rgba,
    Mat,
    Cam,
    Intensity,
    Lights,
    Clip,
    Attr,
    Trans,
    V,
    G,
    L,
    head_amb,
    head_dif,
    head_spe,
    BLOCK: tl.constexpr,
    NF: tl.constexpr,
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
    rt = (c * G + g).to(tl.int64) * 9
    px = tl.load(Xf + xf + 0) * lx_ + tl.load(Xf + xf + 1) * ly_ + tl.load(Xf + xf + 2) * lz_ + tl.load(Xf + xf + 9)
    py = tl.load(Xf + xf + 3) * lx_ + tl.load(Xf + xf + 4) * ly_ + tl.load(Xf + xf + 5) * lz_ + tl.load(Xf + xf + 10)
    pz = tl.load(Xf + xf + 6) * lx_ + tl.load(Xf + xf + 7) * ly_ + tl.load(Xf + xf + 8) * lz_ + tl.load(Xf + xf + 11)
    nx = tl.load(Rot + rt + 0) * nx_ + tl.load(Rot + rt + 1) * ny_ + tl.load(Rot + rt + 2) * nz_
    ny = tl.load(Rot + rt + 3) * nx_ + tl.load(Rot + rt + 4) * ny_ + tl.load(Rot + rt + 5) * nz_
    nz = tl.load(Rot + rt + 6) * nx_ + tl.load(Rot + rt + 7) * ny_ + tl.load(Rot + rt + 8) * nz_
    nn = tl.maximum(tl.sqrt(nx * nx + ny * ny + nz * nz), 1e-12)
    nx = nx / nn
    ny = ny / nn
    nz = nz / nn
    cb = c.to(tl.int64) * 22
    vx = tl.load(Cam + cb + 16) - px
    vy = tl.load(Cam + cb + 17) - py
    vz = tl.load(Cam + cb + 18) - pz
    vn = tl.maximum(tl.sqrt(vx * vx + vy * vy + vz * vz), 1e-12)
    vx = vx / vn
    vy = vy / vn
    vz = vz / vn
    shin = tl.load(Mat + g * 3 + 1, mask=m, other=0.0)
    hx = -tl.load(Cam + cb + 19)
    hy = -tl.load(Cam + cb + 20)
    hz = -tl.load(Cam + cb + 21)
    ndl = nx * hx + ny * hy + nz * hz
    d1 = head_amb + head_dif * tl.maximum(ndl, 0.0)
    s1 = head_spe * _specular_term(nx, ny, nz, hx, hy, hz, vx, vy, vz, ndl, shin)
    d2 = tl.zeros([BLOCK], tl.float32)
    s2 = tl.zeros([BLOCK], tl.float32)
    inten = tl.load(Intensity + c)
    for k in range(L):
        row = Lights + k * NF
        point = tl.load(row + 0)
        shadow = tl.load(row + 1)
        glob = tl.load(row + 2)
        qx = tl.load(row + 3)
        qy = tl.load(row + 4)
        qz = tl.load(row + 5)
        # A point light's direction is towards its position; a directional light's is stored.
        rx = tl.where(point > 0, qx - px, qx)
        ry = tl.where(point > 0, qy - py, qy)
        rz = tl.where(point > 0, qz - pz, qz)
        dist = tl.sqrt(rx * rx + ry * ry + rz * rz)
        ldx = rx / tl.maximum(dist, 1e-12)
        ldy = ry / tl.maximum(dist, 1e-12)
        ldz = rz / tl.maximum(dist, 1e-12)
        att = tl.where(
            point > 0, 1.0 / (tl.load(row + 9) + tl.load(row + 10) * dist + tl.load(row + 11) * dist * dist), 1.0
        )
        cos = -(ldx * tl.load(row + 15) + ldy * tl.load(row + 16) + ldz * tl.load(row + 17))
        cone = tl.where(cos >= tl.load(row + 13), tl.exp2(tl.load(row + 14) * tl.log2(tl.maximum(cos, 1e-30))), 0.0)
        att = tl.where(tl.load(row + 12) > 0, att * cone, att)
        ndl = nx * ldx + ny * ldy + nz * ldz
        diffuse = tl.where(glob > 0, inten, tl.load(row + 7))
        d = (tl.load(row + 6) + diffuse * tl.maximum(ndl, 0.0)) * att
        sp = tl.load(row + 8) * _specular_term(nx, ny, nz, ldx, ldy, ldz, vx, vy, vz, ndl, shin) * att
        d1 += tl.where(shadow > 0, 0.0, d)
        s1 += tl.where(shadow > 0, 0.0, sp)
        d2 += tl.where(shadow > 0, d, 0.0)
        s2 += tl.where(shadow > 0, sp, 0.0)
    cg = (c * G + g).to(tl.int64) * 4
    spec = tl.load(Mat + g * 3 + 0, mask=m, other=0.0)
    emis = tl.load(Mat + g * 3 + 2, mask=m, other=0.0)
    out = (c.to(tl.int64) * V + v) * 9
    tr = (c.to(tl.int64) * V + v) * 3
    for ch in range(3):
        rgb = tl.load(Rgba + cg + ch, mask=m, other=0.0)
        tl.store(Attr + out + ch, tl.minimum(tl.maximum(rgb * (d1 + emis) + spec * s1, 0.0), 1.0), mask=m)
        tl.store(Attr + out + 3 + ch, tl.minimum(tl.maximum(rgb * d2 + spec * s2, 0.0), 1.0), mask=m)
        tl.store(Trans + tr + ch, tl.minimum(tl.maximum(rgb * (d1 + d2 + emis) + spec * (s1 + s2), 0.0), 1.0), mask=m)
    tl.store(Attr + out + 6, px, mask=m)
    tl.store(Attr + out + 7, py, mask=m)
    tl.store(Attr + out + 8, pz, mask=m)
    cl = (c.to(tl.int64) * V + v) * 4
    for row in range(4):
        value = (
            tl.load(Cam + cb + row * 4 + 0) * px
            + tl.load(Cam + cb + row * 4 + 1) * py
            + tl.load(Cam + cb + row * 4 + 2) * pz
            + tl.load(Cam + cb + row * 4 + 3)
        )
        tl.store(Clip + cl + row, value, mask=m)


def vertices(vpos, vnor, vgeom, transform, rotation, rgba, material, camera, intensity, lights, head):
    """Clip positions (C, V, 4), opaque attributes (C, V, 9) and transparent colours (C, V, 3) of every vertex.

    The opaque attributes are the lit colour without the shadow light, the shadow light's colour and the world
    position; colours are clamped per vertex as OpenGL clamps them.
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
            rotation,
            rgba,
            material,
            camera,
            intensity,
            lights,
            clip,
            attr,
            trans,
            V,
            G,
            len(lights),
            float(head[0]),
            float(head[1]),
            float(head[2]),
            block,
            len(LIGHT_FIELDS),
        )
    return clip, attr, trans


@triton.jit
def _reached(Height, c, x, y, z, m, origin_x, origin_y, texel, i0, j0, ni, nj, bias, HEIGHT: tl.constexpr = 16.0):
    """Whether the shadow light reaches points (x, y, z): no offset caster in their texel lies above them."""
    iu = tl.floor((x - origin_x) / texel).to(tl.int32) - i0
    iv = tl.floor((y - origin_y) / texel).to(tl.int32) - j0
    inside = m & (iu >= 0) & (iu < ni) & (iv >= 0) & (iv < nj)
    top = tl.load(Height + (c.to(tl.int64) * nj + iv) * ni + iu, mask=inside, other=0.0)
    return ~(top > z + bias + HEIGHT)


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
    Height,
    Base,
    Zbuf,
    V,
    k,
    sx,
    sy,
    origin_x,
    origin_y,
    texel,
    i0,
    j0,
    ni,
    nj,
    bias,
    S: tl.constexpr,
    BLOCK: tl.constexpr,
):
    c = tl.program_id(1)
    p = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    m = p < S * S
    u_ = tl.load(Index + c)
    pix = c.to(tl.int64) * S * S + p
    bpix = u_.to(tl.int64) * S * S + p
    cover = tl.load(Cover + (u_ * 4 + k).to(tl.int64) * S * S + p, mask=m, other=0) != 0
    depth = tl.load(Depth + (u_ * 4 + k).to(tl.int64) * S * S + p, mask=m, other=1.0)
    fx = tl.load(XY + bpix * 2 + 0, mask=m, other=0.0)
    fy = tl.load(XY + bpix * 2 + 1, mask=m, other=0.0)
    floor_lit = _reached(
        Height, c, fx, fy, tl.zeros([BLOCK], tl.float32), m & cover, origin_x, origin_y, texel, i0, j0, ni, nj, bias
    )
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
    lit = _reached(Height, c, px, py, pz, hit, origin_x, origin_y, texel, i0, j0, ni, nj, bias)
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
        first = tl.floor(tl.minimum(tl.maximum(c1 * tex, 0.0), 1.0) * 255.0 + 0.5)
        second = tl.floor(tl.minimum(first + tl.minimum(tl.maximum(c2 * tex, 0.0), 1.0) * 255.0, 255.0) + 0.5)
        obj = tl.where(lit, second, first)
        bf = tl.load(First + bpix * 3 + ch, mask=m, other=0).to(tl.float32)
        bs = tl.load(Second + bpix * 3 + ch, mask=m, other=0).to(tl.float32)
        sky = tl.load(Sky + bpix * 3 + ch, mask=m, other=0).to(tl.float32)
        back = tl.where(cover, tl.where(floor_lit, bs, bf), sky)
        tl.store(Base + ((c * 4 + k).to(tl.int64) * S * S + p) * 3 + ch, tl.where(hit, obj, back).to(tl.uint8), mask=m)
    tl.store(Zbuf + (c * 4 + k).to(tl.int64) * S * S + p, tl.where(hit, z, depth), mask=m)


def opaque(
    rast,
    rast_db,
    tri,
    attr,
    geom,
    texmod,
    index,
    background,
    heights,
    window,
    origin,
    texel,
    bias,
    base,
    zbuf,
    k,
    sample,
):
    """Write sample ``k`` of ``base`` (C, 4, S, S, 3) and ``zbuf`` from the opaque raster and the background."""
    C, S = rast.shape[0], rast.shape[1]
    V = attr.shape[1]
    (i0, j0), (i1, j1) = window
    block = 256
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
            heights,
            base,
            zbuf,
            V,
            k,
            float(sample[0]),
            float(sample[1]),
            float(origin[0]),
            float(origin[1]),
            float(texel),
            int(i0),
            int(j0),
            int(i1 - i0),
            int(j1 - j0),
            float(bias),
            S,
            block,
        )
