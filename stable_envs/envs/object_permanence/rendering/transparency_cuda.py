"""Exact ordered alpha events with full drawing and reusable static-layer seeds."""

import numpy as np
import torch
import triton as tr
import triton.language as tl

from ....render import nvdiff as ND
from ....render.nvdiff_coverage import _vertex
from .buffers import outputs
from .transparency import CONTRACT as ALPHA_CONTRACT
from .transparency_cache import StaticAlphaCoverage, merge
from .visibility import CONTRACT_V2
from .visibility_cuda import ContractRenderer, _intersection, _output


@tr.jit
def _peel_hits(
    Clip,
    Vertices,
    Faces,
    Planes,
    Rays,
    Origin,
    Labels,
    Alpha,
    PreviousDepth,
    PreviousWinner,
    Depth,
    Winner,
    Active,
    INDEXED: tl.constexpr,
    FIRST: tl.constexpr,
    IDS: tl.constexpr,
    V: tl.constexpr,
    F: tl.constexpr,
    H: tl.constexpr,
    W: tl.constexpr,
    RAY_GRID: tl.constexpr,
    ORIGIN_GRID: tl.constexpr,
    NEAR: tl.constexpr,
    FAR: tl.constexpr,
    STRIPES: tl.constexpr,
    BLOCK: tl.constexpr,
):
    face, world = tl.program_id(0), tl.program_id(1)
    if INDEXED:
        face = tl.load(Active + face)
    alpha = tl.load(Alpha + face)
    if alpha > 0:
        x0, y0, z0, w0 = _vertex(Clip, Faces, world, face, 0, V, H, W)
        x1, y1, z1, w1 = _vertex(Clip, Faces, world, face, 1, V, H, W)
        x2, y2, z2, w2 = _vertex(Clip, Faces, world, face, 2, V, H, W)
        lx = tl.minimum(W, tl.maximum(0, tl.floor(tl.minimum(x0, tl.minimum(x1, x2)) - 0.5) - 1)).to(tl.int32)
        hx = tl.maximum(-1, tl.minimum(W - 1, tl.ceil(tl.maximum(x0, tl.maximum(x1, x2)) - 0.5) + 1)).to(tl.int32)
        ly = tl.minimum(H, tl.maximum(0, tl.floor(tl.minimum(y0, tl.minimum(y1, y2)) - 0.5) - 1)).to(tl.int32)
        hy = tl.maximum(-1, tl.minimum(H - 1, tl.ceil(tl.maximum(y0, tl.maximum(y1, y2)) - 0.5) + 1)).to(tl.int32)
        width = tl.maximum(0, hx - lx + 1)
        count = width * tl.maximum(0, hy - ly + 1)
        label = tl.load(Labels + face).to(tl.int64)
        for start in range(tl.program_id(2) * BLOCK, count, STRIPES * BLOCK):
            i = start + tl.arange(0, BLOCK)
            x, y = lx + i % width, ly + i // width
            a, b, t, hit = _intersection(
                Vertices,
                Faces,
                Planes,
                Rays,
                Origin,
                face,
                world,
                tl.minimum(x, W - 1),
                tl.minimum(y, H - 1),
                V,
                F,
                W,
                RAY_GRID,
                ORIGIN_GRID,
            )
            valid = (i < count) & hit & (t >= NEAR) & (t <= FAR)
            pixel = (world * H + y) * W + x
            key = t.to(tl.int64, bitcast=True)
            if not FIRST:
                prev = tl.load(PreviousDepth + pixel, valid, other=9223372036854775807)
                winner = tl.load(PreviousWinner + pixel, valid, other=9223372036854775807)
                previous_label = winner // (F + 1)
                previous_alpha = tl.load(Alpha + winner % (F + 1), valid & (winner != 9223372036854775807), other=1)
                valid = (
                    valid
                    & (prev != 9223372036854775807)
                    & (previous_alpha < 1)
                    & ((key > prev) | ((key == prev) & (label > previous_label)))
                )
            if IDS:
                nearest = tl.load(Depth + pixel, valid, other=9223372036854775807)
                tl.atomic_min(Winner + pixel, label * (F + 1) + face, valid & (key == nearest), sem="relaxed")
            else:
                tl.atomic_min(Depth + pixel, key, valid, sem="relaxed")


class AlphaRenderer(ContractRenderer):
    """Straight-ray alpha surfaces, exact event ordering and explicit capacity.

    Existing RGB/depth/IDs retain their shapes. Optional masks/weights are
    [B,G,H,W]; layer_depth/layer_ids are [B,H,W,L], in front-to-back order.
    Masks describe overlapping geometric support, not a perceptual threshold.
    """

    def __init__(self, *args, max_layers=16, **kwargs):
        if type(max_layers) is not int or max_layers < 1:
            raise ValueError("max_layers must be a positive integer")
        self.max_layers = max_layers
        if kwargs.get("shading") != "tiny" or kwargs.get("textures", True) or kwargs.get("skybox", True):
            raise ValueError("alpha surfaces require tiny shading without textures or skybox")
        # Diagnostic control for reproducing the original full-capacity baseline.
        # It changes work performed, never event visibility or image semantics.
        self.shade_empty_layers = False
        # Reuse V2 camera/ray mathematics; material visibility is a separate policy.
        kwargs["visibility_contract"] = CONTRACT_V2
        super().__init__(*args, **kwargs)
        self.visibility_contract = ALPHA_CONTRACT
        self._alpha_resolution = (self.res, self.res) if isinstance(self.res, int) else self.res
        self._alpha_version = -1
        self._refresh_alpha()
        self._layer_cache = (
            StaticAlphaCoverage(self.scene.face, self._static_geoms[self.scene.fgeom])
            if self._static_geoms is not None
            else None
        )

    def _refresh_alpha(self):
        if self._alpha_version == self.scene.alpha._version:
            return
        values = self.scene.alpha.detach().cpu().numpy()
        if not np.isfinite(values).all() or (values < 0).any() or (values > 1).any():
            raise ValueError("alpha must be finite and in [0,1]")
        self.face_alpha = self.scene.alpha[self.scene.fgeom].contiguous()
        self._alpha_version = self.scene.alpha._version

    def _peel(self, positions, active=None, *, capacity=None):
        b, v, _ = positions.shape
        h, w = self._alpha_resolution
        capacity = self.max_layers if capacity is None else capacity
        count = len(self.scene.face) if active is None else len(active)
        opaque_only = bool(count) and bool(
            (self.face_alpha if active is None else self.face_alpha[active.long()]).eq(1).all()
        )
        allocation = capacity if self.shade_empty_layers or (count and not opaque_only) else 1
        depth = torch.full((allocation, b, h, w), torch.iinfo(torch.int64).max, dtype=torch.int64, device=self.device)
        winner = torch.full_like(depth, torch.iinfo(torch.int64).max)
        if not count:
            return depth[:1], winner[:1]
        previous_d, previous_w = depth[0], winner[0]
        stripes = 8 if b < 4 else 1
        used = 0
        for layer in range(capacity + 1):
            d = depth[layer] if layer < capacity else torch.full_like(depth[0], torch.iinfo(torch.int64).max)
            win = winner[layer] if layer < capacity else torch.full_like(winner[0], torch.iinfo(torch.int64).max)
            for ids in (False, True):
                _peel_hits[(count, b, stripes)](
                    positions,
                    self.vertices,
                    self.scene.face,
                    self.world_planes,
                    self.rays,
                    self.origin,
                    self.labels,
                    self.face_alpha,
                    previous_d,
                    previous_w,
                    d,
                    win,
                    active,
                    active is not None,
                    layer == 0,
                    ids,
                    v,
                    len(self.scene.face),
                    h,
                    w,
                    self.rays.ndim == 3,
                    self.origin.ndim == 3,
                    self.metadata["near"],
                    self.metadata["far"],
                    stripes,
                    128,
                    enable_fp_fusion=False,
                )
            if not (win != torch.iinfo(torch.int64).max).any():
                break
            if layer == capacity:
                raise ValueError(
                    "transparent layer capacity exceeded; increase max_layers, no layers were silently dropped"
                )
            previous_d, previous_w = d, win
            used = layer + 1
            if opaque_only:
                break
        retained = capacity if self.shade_empty_layers else max(1, used)
        return depth[:retained], winner[:retained]

    def _layers(self, positions):
        valid = torch.isfinite(positions).all(-1) & (positions[..., 3] > 0)
        if not self.orthographic:
            valid &= positions[..., 2].abs() < positions[..., 3]
        active = self.scene.alpha[self.scene.vgeom] > 0
        if not (valid | ~active[None]).all():
            raise ValueError("visibility contract does not support depth-plane crossing")
        seed = (
            None
            if self._layer_cache is None
            else self._layer_cache.seed(
                positions,
                self.vertices,
                self.world_planes,
                self.rays,
                self.origin,
                self.labels,
                self.face_alpha,
                self._alpha_resolution,
                self.metadata["near"],
                self.metadata["far"],
                self.max_layers,
                self._peel,
            )
        )
        if seed is None:
            return self._peel(positions)
        sd, sw, active = seed
        dynamic = self._peel(positions, active)
        return merge((sd, sw), dynamic, self.face_alpha, self.max_layers, len(self.scene.face))

    def render_many(self, xpos, xmat, cam, *, modes=("rgb", "depth", "ids"), out=None, mat_rgb=None):
        import nvdiffrast.torch as dr

        self._check_open("render")
        modes = tuple(modes)
        allowed = {"rgb", "depth", "ids", "masks", "weights", "layer_depth", "layer_ids"}
        if not modes or len(set(modes)) != len(modes) or set(modes) - allowed:
            raise ValueError("invalid alpha rendering modes")
        h, w = self._alpha_resolution
        b = len(xpos)
        if out is not None:
            outputs(
                modes,
                out,
                batch=b,
                height=h,
                width=w,
                objects=self.scene.G,
                device=self.scene.vert.device,
                allowed=allowed,
                layers=self.max_layers,
            )
        self._refresh_alpha()
        scene = self.scene
        view, projection = cam.matrices(aspect=w / h)
        clip, normals, _ = self._clip(xpos, xmat, view, projection)
        depths, winners = self._layers(clip)
        hit = winners != torch.iinfo(torch.int64).max
        faces = torch.where(hit, winners % (len(scene.face) + 1), 0).long()
        gids = scene.fgeom[faces]
        alpha = torch.where(hit, scene.alpha[gids], 0)
        images = {}
        if "depth" in modes:
            images["depth"] = torch.where(hit[0], depths[0].view(torch.float64).float(), 0).flip(1)
        if "ids" in modes:
            images["ids"] = torch.where(hit[0], gids[0] + 1, 0).to(torch.int16).flip(1)
        if "layer_depth" in modes:
            value = torch.zeros((self.max_layers, b, h, w), dtype=torch.float32, device=self.device)
            value[: len(depths)] = torch.where(hit, depths.view(torch.float64).float(), 0)
            images["layer_depth"] = value.permute(1, 2, 3, 0).flip(1).contiguous()
        if "layer_ids" in modes:
            value = torch.zeros((self.max_layers, b, h, w), dtype=torch.int16, device=self.device)
            value[: len(depths)] = torch.where(hit, gids + 1, 0).to(torch.int16)
            images["layer_ids"] = value.permute(1, 2, 3, 0).flip(1).contiguous()
        if "masks" in modes:
            images["masks"] = torch.zeros((b, scene.G, h, w), dtype=torch.uint8, device=self.device)
        if "weights" in modes:
            images["weights"] = torch.zeros((b, scene.G, h, w), dtype=torch.float32, device=self.device)
        rgb = torch.zeros((b, h, w, 3), device=self.device) if "rgb" in modes else None
        transmittance = torch.ones((b, h, w), device=self.device)
        tiny = self.tiny
        assert tiny is not None
        for layer in range(len(depths)):
            contribution = transmittance * alpha[layer]
            if "masks" in modes:
                images["masks"].scatter_reduce_(
                    1,
                    gids[layer, None].permute(1, 0, 2, 3),
                    hit[layer, None].permute(1, 0, 2, 3).to(torch.uint8),
                    reduce="amax",
                    include_self=True,
                )
            if "weights" in modes:
                images["weights"].scatter_add_(1, gids[layer, None].permute(1, 0, 2, 3), contribution[:, None])
            if rgb is not None:
                raster = torch.empty((b, h, w, 4), dtype=torch.float32, device=self.device)
                _output[(tr.cdiv(h * w, 128), b)](
                    self.vertices,
                    scene.face,
                    self.world_planes,
                    self.rays,
                    self.origin,
                    self.rays.ndim == 3,
                    self.origin.ndim == 3,
                    winners[layer],
                    raster,
                    scene.V,
                    len(scene.face),
                    h,
                    w,
                    128,
                    enable_fp_fusion=False,
                )
                n, _ = dr.interpolate(normals, raster, scene.face)
                n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                base = ND.gather_material(scene.mat_rgb if mat_rgb is None else mat_rgb, gids[layer])
                color = ND.shade_tiny(n, base, tiny).clamp(0, 1)
                rgb += contribution[..., None] * color
            transmittance *= 1 - alpha[layer]
        for mode in ("masks", "weights"):
            if mode in images:
                images[mode] = images[mode].flip(2)
        if "masks" in images:
            images["masks"] = images["masks"].bool()
        if rgb is not None:
            rgb += transmittance[..., None] * torch.as_tensor(self.background, dtype=torch.float32, device=self.device)
            images["rgb"] = ND.QUANT(rgb, self.quant).flip(1)
        if out is not None:
            for mode in modes:
                out[mode].copy_(images[mode])
            return out
        return images

    def _release(self):
        if hasattr(self, "_layer_cache") and self._layer_cache is not None:
            self._layer_cache.clear()
            self._layer_cache = None
        if hasattr(self, "face_alpha"):
            del self.face_alpha
        super()._release()
