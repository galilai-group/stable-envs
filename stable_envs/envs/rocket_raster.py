"""Rocket-local floating-point CUDA coverage for TinyRenderer-style pixels.

Nvdiffrast's quantized coverage can omit whole thin fin faces. This pass clips
triangles against the camera near plane, rasterizes integer pixel samples, and
returns the original face's perspective weights for the existing shading path.
Float32 coverage is measured against the source's double-barycentric renderer;
it is not a promise of bitwise source equivalence. No gradients are provided.
The two depth/face passes make visibility independent of GPU execution order.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _clip_triangles(Pos, Faces, Out, V: tl.constexpr, F: tl.constexpr):
    face, world = tl.program_id(0), tl.program_id(1)
    c = tl.arange(0, 4)
    i0, i1, i2 = tl.load(Faces + face * 3), tl.load(Faces + face * 3 + 1), tl.load(Faces + face * 3 + 2)
    a = tl.load(Pos + (world * V + i0) * 4 + c)
    b = tl.load(Pos + (world * V + i1) * 4 + c)
    d = tl.load(Pos + (world * V + i2) * 4 + c)
    wa, wb, wd = (
        tl.sum(tl.where(c == 3, a, 0), 0),
        tl.sum(tl.where(c == 3, b, 0), 0),
        tl.sum(tl.where(c == 3, d, 0), 0),
    )
    za, zb, zd = (
        tl.sum(tl.where(c == 2, a, 0), 0),
        tl.sum(tl.where(c == 2, b, 0), 0),
        tl.sum(tl.where(c == 2, d, 0), 0),
    )
    mixed = (wa < 0) | (wb < 0) | (wd < 0)
    ia, ib, ic = (wa > 0) & (za > -wa), (wb > 0) & (zb > -wb), (wd > 0) & (zd > -wd)
    count = ia.to(tl.int32) + ib.to(tl.int32) + ic.to(tl.int32)
    zero = tl.full((4,), 0, tl.float32)
    u0, u1, u2, u3, u4, u5 = zero, zero, zero, zero, zero, zero
    if not mixed:
        u0, u1, u2 = a, b, d
    elif (count == 1) | (count == 2):
        choose_a = tl.where(count == 1, ia, ~ia)
        choose_b = tl.where(count == 1, ib, ~ib)
        aa = tl.where(choose_a, a, tl.where(choose_b, b, d))
        bb = tl.where(choose_a, b, tl.where(choose_b, d, a))
        cc = tl.where(choose_a, d, tl.where(choose_b, a, b))
        da = tl.sum(tl.where((c == 2) | (c == 3), aa, 0), 0)
        db = tl.sum(tl.where((c == 2) | (c == 3), bb, 0), 0)
        dc = tl.sum(tl.where((c == 2) | (c == 3), cc, 0), 0)
        ab = (aa * db - bb * da) * (1 / (db - da))
        ac = (aa * dc - cc * da) * (1 / (dc - da))
        if count == 1:
            u0, u1, u2 = aa, ab, ac
        else:
            u0, u1, u2 = ab, bb, cc
            u3, u4, u5 = ab, cc, ac
    base = (world * F + face) * 24 + c
    tl.store(Out + base, u0)
    tl.store(Out + base + 4, u1)
    tl.store(Out + base + 8, u2)
    tl.store(Out + base + 12, u3)
    tl.store(Out + base + 16, u4)
    tl.store(Out + base + 20, u5)


@tr.jit
def _vertex(Pos, Faces, world, face, corner, V: tl.constexpr, R: tl.constexpr):
    idx = tl.load(Faces + face * 3 + corner)
    offset = (world * V + idx) * 4
    x, y = tl.load(Pos + offset), tl.load(Pos + offset + 1)
    z, w = tl.load(Pos + offset + 2), tl.load(Pos + offset + 3)
    # Preserve source float32 viewport arithmetic, then floating-point barycentrics.
    x = ((x * (R / 2)) + (w * (R / 2))) / w
    y = ((y * (R / 2)) + (w * (R / 2))) / w
    return x.to(tl.float32), y.to(tl.float32), z.to(tl.float32), w.to(tl.float32)


@tr.jit
def _weights(x, y, x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2):
    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    inv_det = 1.0 / det
    a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) * inv_det
    b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) * inv_det
    c = 1 - a - b
    p, q, r = a / w0, b / w1, c / w2
    total = p + q + r
    inv_total = 1.0 / total
    p, q, r = p * inv_total, q * inv_total, r * inv_total
    depth = p * z0 + q * z1 + r * z2
    ndc = a * (z0 / w0) + b * (z1 / w1) + c * (z2 / w2)
    valid = (
        (tl.abs(det) > 1e-12)
        & (a >= 0)
        & (b >= 0)
        & (c >= 0)
        & (w0 > 0)
        & (w1 > 0)
        & (w2 > 0)
        & (ndc >= -1)
        & (ndc <= 1)
    )
    return p, q, ndc, depth, valid


@tr.jit
def _triangles(
    Pos,
    Faces,
    Depth,
    Ids,
    V: tl.constexpr,
    R: tl.constexpr,
    IDS: tl.constexpr,
    BLOCK: tl.constexpr,
    STRIPES: tl.constexpr,
):
    face, world = tl.program_id(0), tl.program_id(1)
    x0, y0, z0, w0 = _vertex(Pos, Faces, world, face, 0, V, R)
    x1, y1, z1, w1 = _vertex(Pos, Faces, world, face, 1, V, R)
    x2, y2, z2, w2 = _vertex(Pos, Faces, world, face, 2, V, R)
    lo_x = tl.maximum(0, tl.ceil(tl.minimum(x0, tl.minimum(x1, x2)))).to(tl.int32)
    hi_x = tl.minimum(R - 1, tl.floor(tl.maximum(x0, tl.maximum(x1, x2)))).to(tl.int32)
    lo_y = tl.maximum(0, tl.ceil(tl.minimum(y0, tl.minimum(y1, y2)))).to(tl.int32)
    hi_y = tl.minimum(R - 1, tl.floor(tl.maximum(y0, tl.maximum(y1, y2)))).to(tl.int32)
    width = tl.maximum(0, hi_x - lo_x + 1)
    count = width * tl.maximum(0, hi_y - lo_y + 1)
    if (w0 > 0) & (w1 > 0) & (w2 > 0):
        for start in range(tl.program_id(2) * BLOCK, count, STRIPES * BLOCK):
            i = start + tl.arange(0, BLOCK)
            x, y = lo_x + i % width, lo_y + i // width
            p, q, ndc, depth, valid = _weights(
                x.to(tl.float32), y.to(tl.float32), x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2
            )
            mask = (i < count) & valid
            bits = depth.to(tl.float64).to(tl.int64, bitcast=True)
            key = tl.where(bits < 0, (~bits) ^ (-9223372036854775807 - 1), bits)
            offset = (world * R + y) * R + x
            if IDS:
                winning = tl.load(Depth + offset, mask=mask, other=9223372036854775807)
                tl.atomic_max(Ids + offset, face // 2 + 1, mask=mask & (key == winning), sem="relaxed")
            else:
                tl.atomic_min(Depth + offset, key, mask=mask, sem="relaxed")


@tr.jit
def _resolve(Pos, Faces, Ids, Out, V: tl.constexpr, R: tl.constexpr, BLOCK: tl.constexpr):
    world = tl.program_id(1)
    pixel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    hit = tl.load(Ids + world * R * R + pixel, mask=pixel < R * R, other=0)
    face = tl.maximum(hit - 1, 0)
    x0, y0, z0, w0 = _vertex(Pos, Faces, world, face, 0, V, R)
    x1, y1, z1, w1 = _vertex(Pos, Faces, world, face, 1, V, R)
    x2, y2, z2, w2 = _vertex(Pos, Faces, world, face, 2, V, R)
    p, q, ndc, depth, valid = _weights(
        (pixel % R).to(tl.float32), (pixel // R).to(tl.float32), x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2
    )
    base = (world * R * R + pixel) * 4
    tl.store(Out + base, tl.where(hit > 0, p, 0), mask=pixel < R * R)
    tl.store(Out + base + 1, tl.where(hit > 0, q, 0), mask=pixel < R * R)
    tl.store(Out + base + 2, tl.where(hit > 0, ndc, 0), mask=pixel < R * R)
    tl.store(Out + base + 3, hit, mask=pixel < R * R)


def rasterize(pos, faces, resolution):
    """Return float32 (B, H, W, 4) barycentrics/depth/face IDs on CUDA."""
    b, v, _ = pos.shape
    r = resolution
    depth = torch.full((b, r, r), torch.iinfo(torch.int64).max, device=pos.device, dtype=torch.int64)
    ids = torch.zeros((b, r, r), device=pos.device, dtype=torch.int32)
    out = torch.empty((b, r, r, 4), device=pos.device, dtype=torch.float32)
    f = len(faces)
    clipped = torch.empty((b, f * 6, 4), device=pos.device, dtype=pos.dtype)
    clip_faces = torch.arange(f * 6, device=pos.device, dtype=faces.dtype).reshape(f * 2, 3)
    _clip_triangles[(f, b)](pos, faces, clipped, v, f, enable_fp_fusion=False, num_warps=4)
    stripes = 4 if b < 4 and r >= 256 else 1
    for index in (False, True):
        _triangles[(f * 2, b, stripes)](
            clipped, clip_faces, depth, ids, f * 6, r, index, 128, stripes, enable_fp_fusion=False
        )
    _resolve[(tr.cdiv(r * r, 128), b)](pos, faces, ids, out, v, r, 128, enable_fp_fusion=False)
    return out
