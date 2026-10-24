"""Unquantized CUDA pixel-center coverage for nvdiff's interpolation pipeline.

Recheck every triangle's conservative pixel bounds, including triangles omitted
entirely by quantized coverage. Float64 barycentrics and two visibility passes
avoid neighbour-mask heuristics and execution-order-dependent winners. No
gradients or derivative buffers. Callers handle near/far-clipped triangles with
their normal rasterizer; this pass requires all vertices inside depth planes.
"""

import torch
import triton as tr
import triton.language as tl


@tr.jit
def _vertex(Pos, Faces, world, face, corner, V: tl.constexpr, H: tl.constexpr, W: tl.constexpr):
    index = tl.load(Faces + face * 3 + corner)
    base = (world * V + index) * 4
    x = tl.load(Pos + base).to(tl.float64)
    y = tl.load(Pos + base + 1).to(tl.float64)
    z = tl.load(Pos + base + 2).to(tl.float64)
    w = tl.load(Pos + base + 3).to(tl.float64)
    return (x / w + 1) * (W / 2), (y / w + 1) * (H / 2), z, w


@tr.jit
def _weights(x, y, x0, y0, z0, w0, x1, y1, z1, w1, x2, y2, z2, w2):
    det = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
    a = ((y1 - y2) * (x - x2) + (x2 - x1) * (y - y2)) / det
    b = ((y2 - y0) * (x - x2) + (x0 - x2) * (y - y2)) / det
    c = 1 - a - b
    p, q, r = a / w0, b / w1, c / w2
    total = p + q + r
    z = a * (z0 / w0) + b * (z1 / w1) + c * (z2 / w2)
    valid = (det != 0) & (a >= 0) & (b >= 0) & (c >= 0) & (z >= -1) & (z <= 1)
    return p / total, q / total, z, valid


@tr.jit
def _triangles(
    Pos,
    Faces,
    Depth,
    Ids,
    Active,
    INDEXED: tl.constexpr,
    V: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    IDS: tl.constexpr,
    STRIPES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    face, world = tl.program_id(0), tl.program_id(1)
    if INDEXED:
        face = tl.load(Active + face)
    x0, y0, z0, w0 = _vertex(Pos, Faces, world, face, 0, V, H, W)
    x1, y1, z1, w1 = _vertex(Pos, Faces, world, face, 1, V, H, W)
    x2, y2, z2, w2 = _vertex(Pos, Faces, world, face, 2, V, H, W)
    # Expand only the enumeration box by one pixel; the actual coverage test
    # below uses the original unexpanded triangle with no silhouette epsilon.
    lx = tl.minimum(W, tl.maximum(0, tl.floor(tl.minimum(x0, tl.minimum(x1, x2)) - 0.5) - 1)).to(tl.int32)
    hx = tl.maximum(-1, tl.minimum(W - 1, tl.ceil(tl.maximum(x0, tl.maximum(x1, x2)) - 0.5) + 1)).to(tl.int32)
    ly = tl.minimum(H, tl.maximum(0, tl.floor(tl.minimum(y0, tl.minimum(y1, y2)) - 0.5) - 1)).to(tl.int32)
    hy = tl.maximum(-1, tl.minimum(H - 1, tl.ceil(tl.maximum(y0, tl.maximum(y1, y2)) - 0.5) + 1)).to(tl.int32)
    width = tl.maximum(0, hx - lx + 1)
    count = width * tl.maximum(0, hy - ly + 1)
    for start in range(tl.program_id(2) * BLOCK, count, STRIPES * BLOCK):
        index = start + tl.arange(0, BLOCK)
        x, y = lx + index % width, ly + index // width
        p, q, z, valid = _weights(
            x.to(tl.float64) + 0.5,
            y.to(tl.float64) + 0.5,
            x0,
            y0,
            z0,
            w0,
            x1,
            y1,
            z1,
            w1,
            x2,
            y2,
            z2,
            w2,
        )
        mask = (index < count) & valid
        bits = z.to(tl.int64, bitcast=True)
        key = tl.where(bits < 0, (~bits) ^ (-9223372036854775807 - 1), bits)
        offset = (world * H + y) * W + x
        if IDS:
            nearest = tl.load(Depth + offset, mask=mask, other=9223372036854775807)
            # Equal-depth ties deliberately choose the largest source face ID.
            tl.atomic_max(Ids + offset, face + 1, mask=mask & (nearest == key), sem="relaxed")
        else:
            tl.atomic_min(Depth + offset, key, mask=mask, sem="relaxed")


@tr.jit
def _resolve(Pos, Faces, Ids, Out, V: tl.constexpr, H: tl.constexpr, W: tl.constexpr, BLOCK: tl.constexpr):
    world = tl.program_id(1)
    pixel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    hit = tl.load(Ids + world * H * W + pixel, mask=pixel < H * W, other=0)
    face = tl.maximum(hit - 1, 0)
    x0, y0, z0, w0 = _vertex(Pos, Faces, world, face, 0, V, H, W)
    x1, y1, z1, w1 = _vertex(Pos, Faces, world, face, 1, V, H, W)
    x2, y2, z2, w2 = _vertex(Pos, Faces, world, face, 2, V, H, W)
    p, q, z, _ = _weights(
        (pixel % W).to(tl.float64) + 0.5,
        (pixel // W).to(tl.float64) + 0.5,
        x0,
        y0,
        z0,
        w0,
        x1,
        y1,
        z1,
        w1,
        x2,
        y2,
        z2,
        w2,
    )
    base = (world * H * W + pixel) * 4
    tl.store(Out + base, tl.where(hit > 0, p, 0), mask=pixel < H * W)
    tl.store(Out + base + 1, tl.where(hit > 0, q, 0), mask=pixel < H * W)
    tl.store(Out + base + 2, tl.where(hit > 0, z, 0), mask=pixel < H * W)
    tl.store(Out + base + 3, hit, mask=pixel < H * W)


def _visibility(pos, faces, depth, ids, resolution, active=None, background=None):
    """Update visibility in place, retaining original face IDs for depth ties."""
    h, w = resolution
    b, v, _ = pos.shape
    count = len(faces) if active is None else len(active)
    if not count:
        return
    stripes = 8 if b < 4 and h * w > 4096 else 1
    with torch.cuda.device(pos.device):
        for index in (False, True):
            if index and background is not None:
                # A closer mover invalidates the previous static face winner.
                ids.masked_fill_(depth != background, 0)
            _triangles[(count, b, stripes)](
                pos,
                faces,
                depth,
                ids,
                faces if active is None else active,
                active is not None,
                v,
                h,
                w,
                index,
                stripes,
                32,
                num_warps=1,
                enable_fp_fusion=False,
            )


class StaticCoverage:
    """Cache fixed triangles' pixel visibility, independent of scene semantics.

    Bind to one immutable face tensor and an explicit boolean face partition.
    Seeds store float64 ordered depth keys and original face IDs, so dynamic
    triangles retain exactly the full pass's visibility and equal-depth rule.
    Geometry/camera/resolution changes rebuild the seed. If a supposed static
    vertex differs within the batch, use the full pass instead. This cache does
    not assume that independently shaded RGB layers can simply be overlaid.
    """

    def __init__(self, faces, static_faces):
        if (
            static_faces.dtype != torch.bool
            or static_faces.device != faces.device
            or static_faces.shape != (len(faces),)
        ):
            raise ValueError("static_faces must be a boolean mask matching CUDA faces")
        self.faces = faces
        self.static = static_faces.nonzero().flatten().to(torch.int32).contiguous()
        self.dynamic = (~static_faces).nonzero().flatten().to(torch.int32).contiguous()
        self.vertices = faces[self.static.long()].flatten().unique().long()
        self.builds = 0
        self._positions = self._depth = self._ids = self._resolution = None

    def seed(self, pos, faces, resolution):
        """Return owned batch seeds, or None when the partition cannot be reused."""
        if faces is not self.faces:
            raise ValueError("static coverage belongs to a different face tensor")
        if not len(self.static):
            return None
        positions = pos[:, self.vertices]
        if not torch.equal(positions, positions[:1].expand_as(positions)):
            return None
        if (
            self._resolution != resolution
            or self._positions is None
            or self._positions.dtype != positions.dtype
            or not torch.equal(positions[:1], self._positions)
        ):
            h, w = resolution
            self._depth = torch.full((1, h, w), torch.iinfo(torch.int64).max, dtype=torch.int64, device=pos.device)
            self._ids = torch.zeros((1, h, w), dtype=torch.int32, device=pos.device)
            _visibility(pos[:1], faces, self._depth, self._ids, resolution, self.static)
            self._positions = positions[:1].clone()
            self._resolution = resolution
            self.builds += 1
        return (
            self._depth.expand(len(pos), -1, -1).clone(),
            self._ids.expand(len(pos), -1, -1).clone(),
            self.dynamic,
            self._depth,
        )


def rasterize(pos, faces, resolution, *, cache=None):
    """Return float32 (B,H,W,4), or None when depth clipping needs fallback.

    Input is contiguous CUDA float32/float64 clip vertices and int32 triangle indices.
    The fallback decision synchronizes once; this is not a CUDA-graph API.
    Optional StaticCoverage reuses fixed visibility, without quantizing depth.
    """
    if torch.is_grad_enabled() and pos.requires_grad:
        raise ValueError("precise coverage does not provide gradients")
    if (
        not pos.is_cuda
        or pos.dtype not in (torch.float32, torch.float64)
        or not pos.is_contiguous()
        or pos.ndim != 3
        or pos.shape[-1] != 4
        or not pos.shape[0]
        or not pos.shape[1]
    ):
        raise ValueError("coverage requires contiguous CUDA float32/float64 positions")
    if (
        faces.dtype != torch.int32
        or faces.device != pos.device
        or not faces.is_contiguous()
        or faces.ndim != 2
        or faces.shape[-1] != 3
        or not len(faces)
    ):
        raise ValueError("coverage requires contiguous int32 CUDA triangle indices")
    h, w = (resolution, resolution) if isinstance(resolution, int) else resolution
    clip_w, clip_z = pos[..., 3], pos[..., 2]
    if not ((clip_w > 0) & (clip_z > -clip_w) & (clip_z < clip_w) & torch.isfinite(pos).all(-1)).all():
        return None
    b, v, _ = pos.shape
    seed = None if cache is None else cache.seed(pos, faces, (h, w))
    if seed is None:
        depth = torch.full((b, h, w), torch.iinfo(torch.int64).max, dtype=torch.int64, device=pos.device)
        ids = torch.zeros((b, h, w), dtype=torch.int32, device=pos.device)
        active = background = None
    else:
        depth, ids, active, background = seed
    _visibility(pos, faces, depth, ids, (h, w), active, background)
    out = torch.empty((b, h, w, 4), dtype=torch.float32, device=pos.device)
    with torch.cuda.device(pos.device):
        _resolve[(tr.cdiv(h * w, 128), b)](pos, faces, ids, out, v, h, w, 128, enable_fp_fusion=False)
    return out
