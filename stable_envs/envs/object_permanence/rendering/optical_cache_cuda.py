"""Resolve opaque movers against immutable, sampled static optical paths.

Static transport already includes all alpha/Snell/Fresnel decisions. A moving
opaque surface can terminate a path earlier, never create another continuation.
This restriction makes reuse exact, rather than a flattened-background guess.
"""

import triton as tr
import triton.language as tl


@tr.jit(do_not_specialize=["Count"])
def composite(
    Ray,
    Travelled,
    StaticT,
    StaticFace,
    DynamicT,
    DynamicFace,
    DynamicNormal,
    FaceGeometry,
    Semantics,
    Labels,
    Colors,
    Axis,
    Alive,
    RGB,
    Depth,
    IDs,
    Paths,
    Count,
    Pixels: tl.constexpr,
    Samples: tl.constexpr,
    FIRST: tl.constexpr,
    COLLECT: tl.constexpr,
    BLOCK: tl.constexpr,
):
    q = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    valid = q < Count
    ray = tl.load(Ray + q, valid, 0)
    active = valid & tl.load(Alive + ray, valid, False)
    source = q // Samples if FIRST else q
    t = tl.load(DynamicT + source, active, float("inf")).to(tl.float64)
    face = tl.load(DynamicFace + source, active, -1)
    st = tl.load(StaticT + q, active, float("inf")).to(tl.float64)
    sf = tl.load(StaticFace + q, active, -1)
    label = tl.load(Semantics + face, active & (face >= 0), 32768)
    sl = tl.load(Semantics + sf, active & (sf >= 0), 32768)
    moving = active & (face >= 0) & ((t < st) | ((t == st) & ((label < sl) | ((label == sl) & (face < sf)))))
    winner = tl.where(moving, face, sf)
    distance = tl.where(moving, t, st)
    hit = active & (winner >= 0)
    geometry = tl.load(FaceGeometry + winner, hit, 0)
    if COLLECT:
        path = tl.load(Travelled + q, hit, 0).to(tl.float64) + distance
        tl.atomic_min(Paths + geometry * Pixels + ray // Samples, path, hit, sem="relaxed")
    if FIRST:
        depth = distance * tl.load(Axis + ray // Samples, hit, 0).to(tl.float64)
        semantic = tl.load(Labels + geometry, hit, 0)
        tl.store(Depth + ray, tl.where(hit, depth, 0), active)
        tl.store(IDs + ray, tl.where(hit, semantic, 0), active)
    nx = tl.load(DynamicNormal + 3 * source, moving, 0).to(tl.float64)
    ny = tl.load(DynamicNormal + 3 * source + 1, moving, 0).to(tl.float64)
    nz = tl.load(DynamicNormal + 3 * source + 2, moving, 0).to(tl.float64)
    strength = 0.35 + 0.65 * tl.minimum(1.0, tl.abs(nx * 0.25 + ny * (-0.35) + nz * 0.9027735043))
    for channel in tl.static_range(3):
        color = tl.load(Colors + 3 * geometry + channel, moving, 0).to(tl.float64)
        tl.store(RGB + 3 * ray + channel, color * strength, moving)
    tl.store(Alive + ray, False, moving)
