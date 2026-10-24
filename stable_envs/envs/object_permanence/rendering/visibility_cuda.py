"""CUDA implementation of opaque-pixel-rays-v1; no oracle or Blender imports."""

import torch
import triton as tr
import triton.language as tl

from ....render.nvdiff_coverage import _vertex
from .viewports import PixelCenterRenderer
from .visibility import local_planes


@tr.jit
def _intersection(
    Vertices, Faces, Planes, Rays, Origin, face, world, x, y, V: tl.constexpr, F: tl.constexpr, W: tl.constexpr
):
    i0, i1, i2 = tl.load(Faces + face * 3), tl.load(Faces + face * 3 + 1), tl.load(Faces + face * 3 + 2)
    ax = tl.load(Vertices + (world * V + i0) * 3)
    ay = tl.load(Vertices + (world * V + i0) * 3 + 1)
    az = tl.load(Vertices + (world * V + i0) * 3 + 2)
    ex = tl.load(Vertices + (world * V + i1) * 3) - ax
    ey = tl.load(Vertices + (world * V + i1) * 3 + 1) - ay
    ez = tl.load(Vertices + (world * V + i1) * 3 + 2) - az
    fx = tl.load(Vertices + (world * V + i2) * 3) - ax
    fy = tl.load(Vertices + (world * V + i2) * 3 + 1) - ay
    fz = tl.load(Vertices + (world * V + i2) * 3 + 2) - az
    dx = tl.load(Rays + (y * W + x) * 3)
    dy = tl.load(Rays + (y * W + x) * 3 + 1)
    dz = tl.load(Rays + (y * W + x) * 3 + 2)
    px, py, pz = dy * fz - dz * fy, dz * fx - dx * fz, dx * fy - dy * fx
    det = ex * px + ey * py + ez * pz
    ox, oy, oz = tl.load(Origin) - ax, tl.load(Origin + 1) - ay, tl.load(Origin + 2) - az
    u = (ox * px + oy * py + oz * pz) / det
    qx, qy, qz = oy * ez - oz * ey, oz * ex - ox * ez, ox * ey - oy * ex
    v = (dx * qx + dy * qy + dz * qz) / det
    base = (world * F + face) * 4
    nx, ny, nz, offset = (
        tl.load(Planes + base),
        tl.load(Planes + base + 1),
        tl.load(Planes + base + 2),
        tl.load(Planes + base + 3),
    )
    denominator = nx * dx + ny * dy + nz * dz
    t = -(nx * tl.load(Origin) + ny * tl.load(Origin + 1) + nz * tl.load(Origin + 2) + offset) / denominator
    return 1 - u - v, u, t, (det != 0) & (denominator != 0) & (u >= 0) & (v >= 0) & (u + v <= 1)


@tr.jit
def _hits(
    Clip,
    Vertices,
    Faces,
    Planes,
    Rays,
    Origin,
    Labels,
    Depth,
    Winner,
    V: tl.constexpr,
    F: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    NEAR: tl.constexpr,
    FAR: tl.constexpr,
    IDS: tl.constexpr,
    STRIPES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    face, world = tl.program_id(0), tl.program_id(1)
    x0, y0, z0, w0 = _vertex(Clip, Faces, world, face, 0, V, H, W)
    x1, y1, z1, w1 = _vertex(Clip, Faces, world, face, 1, V, H, W)
    x2, y2, z2, w2 = _vertex(Clip, Faces, world, face, 2, V, H, W)
    lx = tl.minimum(W, tl.maximum(0, tl.floor(tl.minimum(x0, tl.minimum(x1, x2)) - 0.5) - 1)).to(tl.int32)
    hx = tl.maximum(-1, tl.minimum(W - 1, tl.ceil(tl.maximum(x0, tl.maximum(x1, x2)) - 0.5) + 1)).to(tl.int32)
    ly = tl.minimum(H, tl.maximum(0, tl.floor(tl.minimum(y0, tl.minimum(y1, y2)) - 0.5) - 1)).to(tl.int32)
    hy = tl.maximum(-1, tl.minimum(H - 1, tl.ceil(tl.maximum(y0, tl.maximum(y1, y2)) - 0.5) + 1)).to(tl.int32)
    width = tl.maximum(0, hx - lx + 1)
    count = width * tl.maximum(0, hy - ly + 1)
    for start in range(tl.program_id(2) * BLOCK, count, STRIPES * BLOCK):
        i = start + tl.arange(0, BLOCK)
        x, y = lx + i % width, ly + i // width
        p, q, t, hit = _intersection(
            Vertices, Faces, Planes, Rays, Origin, face, world, tl.minimum(x, W - 1), tl.minimum(y, H - 1), V, F, W
        )
        valid = (i < count) & hit & (t >= NEAR) & (t <= FAR)
        index = (world * H + y) * W + x
        key = t.to(tl.int64, bitcast=True)  # positive binary64 depth is monotonically ordered
        if IDS:
            nearest = tl.load(Depth + index, valid, other=9223372036854775807)
            label = tl.load(Labels + face).to(tl.int64)
            tl.atomic_min(Winner + index, label * (F + 1) + face, valid & (key == nearest), sem="relaxed")
        else:
            tl.atomic_min(Depth + index, key, valid, sem="relaxed")


@tr.jit
def _output(
    Vertices,
    Faces,
    Planes,
    Rays,
    Origin,
    Winner,
    Out,
    V: tl.constexpr,
    F: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    BLOCK: tl.constexpr,
):
    world = tl.program_id(1)
    pixel = tl.program_id(0) * BLOCK + tl.arange(0, BLOCK)
    selected = tl.load(Winner + world * H * W + pixel, pixel < H * W, other=9223372036854775807)
    hit = selected != 9223372036854775807
    face = tl.where(hit, selected % (F + 1), 0).to(tl.int32)
    # Different faces per pixel require scalar corner gathers here.
    a, b, t, _ = _intersection(
        Vertices, Faces, Planes, Rays, Origin, face, world, pixel % W, tl.minimum(pixel // W, H - 1), V, F, W
    )
    base = (world * H * W + pixel) * 4
    tl.store(Out + base, tl.where(hit, a, 0), pixel < H * W)
    tl.store(Out + base + 1, tl.where(hit, b, 0), pixel < H * W)
    tl.store(Out + base + 2, tl.where(hit, t, 0), pixel < H * W)
    tl.store(Out + base + 3, tl.where(hit, face + 1, 0), pixel < H * W)


class ContractRenderer(PixelCenterRenderer):
    """Use canonical rays/semantic IDs while reusing nvdiff interpolation/shading.

    Full coverage is intentional for the first correctness implementation. Static
    transforms/assets still stay resident. This path never falls back to a
    different visibility convention and rejects depth-plane crossing explicitly.
    """

    def __init__(self, *args, camera_metadata, semantic_ids, **kwargs):
        if kwargs.get("supersample", 1) != 1:
            raise ValueError("visibility contract requires one pixel-center sample")
        self.metadata = camera_metadata
        super().__init__(*args, **kwargs)
        scene = self.scene
        self.planes = torch.tensor(
            local_planes(scene.vert.cpu().numpy(), scene.face.cpu().numpy()), device=self.device
        )
        self.labels = semantic_ids[scene.fgeom].to(torch.int32).contiguous()
        world, p = (
            torch.tensor(camera_metadata["world"], dtype=torch.float64, device=self.device),
            torch.tensor(camera_metadata["projection"], dtype=torch.float64, device=self.device),
        )
        h, w = self.res if not isinstance(self.res, int) else (self.res, self.res)
        # nvdiff coverage is bottom-up; the public image is flipped by BatchRenderer.
        y, x = torch.meshgrid(
            torch.arange(h, device=self.device, dtype=torch.float64),
            torch.arange(w, device=self.device, dtype=torch.float64),
            indexing="ij",
        )
        local = torch.stack(
            (
                (2 * (x + 0.5) / w - 1 + p[0, 2]) / p[0, 0],
                (2 * (y + 0.5) / h - 1 + p[1, 2]) / p[1, 1],
                -torch.ones_like(x),
            ),
            -1,
        )
        self.rays = (local @ world[:3, :3].T).contiguous()
        self.origin = world[:3, 3].contiguous()

    def _clip(self, xpos, xmat, view, proj):
        result = super()._clip(xpos.float(), xmat.float(), view, proj)
        rotation, position = xmat.double(), xpos.double()
        vg = self.scene.vgeom
        self.vertices = (
            torch.einsum("bvij,vj->bvi", rotation[:, vg], self.scene.vert.double()) + position[:, vg]
        ).contiguous()
        normals = torch.einsum("fi,bfij->bfj", self.planes[:, :3], torch.linalg.inv(rotation)[:, self.scene.fgeom])
        offset = self.planes[None, :, 3] - (normals * position[:, self.scene.fgeom]).sum(-1)
        planes = torch.cat((normals, offset[..., None]), -1)
        scale = normals.gather(-1, normals.abs().argmax(-1, keepdim=True))
        self.world_planes = torch.where(scale != 0, planes / scale, 0).contiguous()
        homogeneous = torch.cat((self.vertices, torch.ones_like(self.vertices[..., :1])), -1)
        clip = (
            homogeneous
            @ torch.as_tensor(view, dtype=torch.float64, device=self.device).transpose(-1, -2)
            @ torch.as_tensor(proj, dtype=torch.float64, device=self.device).T
        )
        return clip.contiguous(), result[1], result[2]

    def _rasterize(self, positions, resolution):
        h, w = resolution
        b, v, _ = positions.shape
        if not (
            torch.isfinite(positions).all(-1) & (positions[..., 3] > 0) & (positions[..., 2].abs() < positions[..., 3])
        ).all():
            raise ValueError("visibility contract does not support depth-plane crossing")
        count = len(self.scene.face)
        depth = torch.full((b, h, w), torch.iinfo(torch.int64).max, device=self.device, dtype=torch.int64)
        winner = torch.full_like(depth, torch.iinfo(torch.int64).max)
        stripes = 8 if b < 4 else 1
        args = (
            positions,
            self.vertices,
            self.scene.face,
            self.world_planes,
            self.rays,
            self.origin,
            self.labels,
            depth,
            winner,
            v,
            count,
            h,
            w,
            self.metadata["near"],
            self.metadata["far"],
        )
        with torch.cuda.device(self.device):
            for ids in (False, True):
                _hits[(count, b, stripes)](*args, ids, stripes, 32, num_warps=1, enable_fp_fusion=False)
        with torch.cuda.device(self.device):
            out = torch.empty((b, h, w, 4), device=self.device, dtype=torch.float32)
            _output[(tr.cdiv(h * w, 128), b)](
                self.vertices,
                self.scene.face,
                self.world_planes,
                self.rays,
                self.origin,
                winner,
                out,
                v,
                count,
                h,
                w,
                128,
                enable_fp_fusion=False,
            )
        self.coverage_depth = out[..., 2].flip(1)
        return out, None

    def render_many(self, *args, **kwargs):
        images = super().render_many(*args, **kwargs)
        if "depth" in images:
            images["depth"].copy_(self.coverage_depth)
        return images

    def _release(self):
        for name in ("planes", "labels", "rays", "origin", "vertices", "world_planes", "coverage_depth"):
            if hasattr(self, name):
                setattr(self, name, None)
        super()._release()
