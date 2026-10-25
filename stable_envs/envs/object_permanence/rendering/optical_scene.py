"""Experimental sampled dielectric replay, with reusable static transport paths.

The opt-in recording policy defines eight deterministic surface-event samples at
one pixel centre. This is ideal geometric optics, not Blender rough BSDF parity.
The static cache stores complete ray segments, not just the first depth image;
opaque movers can intercept any reflected/refracted segment without rebuilding it.
"""

from __future__ import annotations

import numpy as np
import torch

from ....render._resources import Closeable
from .optics import CONTRACT, MATERIAL, POLICY, closed_volume, validate_material
from .optics_cuda import TriangleQuery
from .transforms import row_matrix3
from .visibility import local_planes, validate_camera


def random_event(ray, event):
    """Counter RNG: independent of batch order, compaction and cache state."""
    h = (ray * 747796405 + event * 2891336453) & 0xFFFFFFFF
    h = ((h ^ (h >> 16)) * 2246822519) & 0xFFFFFFFF
    h = ((h ^ (h >> 13)) * 3266489917) & 0xFFFFFFFF
    h = h ^ (h >> 16)
    return (h.double() + 0.5) / 4294967296


def dielectric(d, n, n1, n2, choice):
    """Snell + exact unpolarized Fresnel; zero refraction under TIR."""
    n = torch.where(((d * n).sum(-1) > 0)[:, None], -n, n)
    c = (-(d * n).sum(-1)).clamp(0, 1)
    eta = n1 / n2
    k = 1 - eta * eta * (1 - c * c)
    ct = k.clamp_min(0).sqrt()
    rs = ((n1 * c - n2 * ct) / (n1 * c + n2 * ct)).square()
    rp = ((n1 * ct - n2 * c) / (n1 * ct + n2 * c)).square()
    f = torch.where(n1 == n2, 0.0, torch.where(k <= 0, 1.0, (rs + rp) / 2))
    reflect = choice < f
    reflected = d + 2 * c[:, None] * n
    refracted = eta[:, None] * d + (eta * c - ct)[:, None] * n
    direction = torch.where(reflect[:, None], reflected, refracted)
    return direction / direction.norm(dim=-1, keepdim=True), reflect


class OpticalSceneRenderer(Closeable):
    """CUDA optical replay with exact cached/full sampled-path equivalence.

    ``depth`` is primary camera-axis depth; ``ids`` is its first surface.
    ``masks`` is geometric support visited by at least one deterministic sample.
    ``path_depth`` is the shortest cumulative geometric ray distance to an object
    across those samples, zero when absent. It is not camera-axis depth after a
    bend. Only moving opaque objects currently permit static transport caching;
    dynamic dielectrics safely fall back to complete tracing.
    """

    resource_kind = "object_permanence.OpticalSceneRenderer"

    def __init__(
        self, trajectory, *, size=180, device="cuda", chunk_size=4, cache_static=True, visibility_contract=None
    ):
        if visibility_contract != CONTRACT:
            raise ValueError("optical replay requires its explicit sampled-ray contract")
        m = trajectory.manifest
        if m.get("material_convention") != MATERIAL or m.get("optical_policy") != POLICY:
            raise ValueError("unsupported optical recording policy")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("optical replay requires CUDA")
        self.device = torch.device("cuda", torch.cuda.current_device()) if self.device.index is None else self.device
        if type(size) is not int or size <= 0 or type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("size and chunk_size must be positive integers")
        if type(cache_static) is not bool:
            raise ValueError("cache_static must be boolean")
        camera = m["camera"]
        validate_camera(camera)
        if camera.get("projection_type") != "perspective" or size * camera["width"] % camera["height"]:
            raise ValueError("optical v1 requires a fixed perspective camera and matching aspect ratio")
        self.height, self.width = size, size * camera["width"] // camera["height"]
        self.trajectory, self.chunk_size, self.cache_static = trajectory, chunk_size, cache_static
        self.objects = m["objects"]
        self.count = len(self.objects)
        self.samples = POLICY["samples"]
        if self.count > 62:
            raise ValueError("optical v1 supports at most 62 volume membership bits")
        self.base = torch.tensor([o["transform"] for o in self.objects], device=self.device, dtype=torch.float64)
        self.nodes = torch.tensor([o["node"] for o in self.objects], device=self.device)
        self.static = self.nodes < 0
        self.material = torch.tensor(
            [[o["rgba"][3], o["optics"]["transmission"], o["optics"]["ior"]] for o in self.objects],
            device=self.device,
            dtype=torch.float64,
        )
        self.colors = torch.tensor([o["rgba"][:3] for o in self.objects], device=self.device, dtype=torch.float64)
        indices = sorted({o["optics"]["ior"] for o in self.objects if o["optics"]["transmission"] > 0})
        if len(indices) != 1:
            raise ValueError("optical v1 requires a common dielectric IOR; overlaps are treated as its union")
        self.ior = indices[0]
        meshes = []
        for o in self.objects:
            validate_material(o["optics"])
            mesh = trajectory.mesh(o["asset"])
            tri = mesh["vertices"][mesh["triangles"]].astype(np.float64)
            if o["optics"]["transmission"]:
                closed_volume(tri)
            meshes.append(tri)
        self.query = TriangleQuery(meshes, device=self.device, semantic_ids=[o["id"] for o in self.objects])
        self.local_planes = torch.tensor(
            np.concatenate([local_planes(t.reshape(-1, 3), np.arange(t.size // 3).reshape(-1, 3)) for t in meshes]),
            device=self.device,
        )
        self.labels = torch.tensor([o["id"] for o in self.objects], device=self.device, dtype=torch.int16)
        self.background = torch.tensor([0.96, 0.97, 0.98], device=self.device, dtype=torch.float64)
        world = torch.tensor(camera["world"], device=self.device, dtype=torch.float64)
        p = np.asarray(camera["projection"])
        yy, xx = np.mgrid[: self.height, : self.width]
        local = torch.tensor(
            np.stack(
                (
                    (2 * (xx + 0.5) / self.width - 1 + p[0, 2]) / p[0, 0],
                    (1 - 2 * (yy + 0.5) / self.height + p[1, 2]) / p[1, 1],
                    -np.ones_like(xx),
                ),
                -1,
            ).reshape(-1, 3),
            device=self.device,
            dtype=torch.float64,
        )
        rays = row_matrix3(local, world[:3, :3].T)
        self.axis_factor = 1 / rays.norm(dim=-1)
        rays = rays / rays.norm(dim=-1, keepdim=True)
        self.origins = world[:3, 3].expand(len(rays) * self.samples, 3).contiguous()
        self.directions = rays.repeat_interleave(self.samples, 0).contiguous()
        self.view_forward = -world[:3, 2]
        self.camera = camera
        self._segments = None
        self._seed: tuple[torch.Tensor, torch.Tensor, torch.Tensor] | None = None
        self.fuse_cache = True
        self.use_ray_limits = True
        self.builds = 0
        self.fallbacks = 0
        self._cache_possible = all(
            o["node"] < 0 or (o["rgba"][3] == 1 and o["optics"]["transmission"] == 0) for o in self.objects
        )
        self._opened()

    def _transforms(self, exact):
        base = self.base.clone()
        moving = self.nodes >= 0
        base[moving] = exact[self.nodes[moving]]
        positions = base[:, :3, 3].contiguous()
        inverse = torch.linalg.inv(base[:, :3, :3]).contiguous()
        normals = row_matrix3(self.local_planes[:, :3], inverse[self.query.labels])
        offset = self.local_planes[:, 3] - (normals * positions[self.query.labels]).sum(-1)
        planes = torch.cat((normals, offset[:, None]), -1)
        scale = normals.gather(-1, normals.abs().argmax(-1, keepdim=True))
        self._planes = (planes / scale).contiguous()
        return positions, inverse

    def _shade(self, normal, label):
        # Deliberately simple symmetric diffuse shading; no BSDF-fidelity claim.
        light = torch.tensor([0.25, -0.35, 0.9027735043], device=self.device, dtype=torch.float64)
        return self.colors[label] * (0.35 + 0.65 * (normal * light).sum(-1).abs().clamp(0, 1))[:, None]

    def _walk(self, positions, inverse, selected, *, save=False, segments=None, collect=True):
        total = len(self.origins)
        pixels = self.height * self.width
        ray = torch.arange(total, device=self.device)
        origin, direction = self.origins, self.directions
        membership = torch.zeros(total, device=self.device, dtype=torch.int64)
        travelled = torch.zeros(total, device=self.device, dtype=torch.float64)
        rgb = torch.zeros(total, 3, device=self.device, dtype=torch.float64)
        depth = torch.zeros(total, device=self.device, dtype=torch.float64)
        ids = torch.zeros(total, device=self.device, dtype=torch.int16)
        paths = torch.full(
            (self.count * pixels if collect else 0,), float("inf"), device=self.device, dtype=torch.float64
        )
        saved = []
        for event in range(POLICY["max_events"]):
            if not len(ray):
                break
            if segments is None:
                t, face, normal = self.query(origin, direction, positions, inverse, selected, self._planes)
            else:
                entry = segments[event]
                # Static paths carry global sample IDs, so dynamic termination is
                # just a lookup/gate; no cached history is advanced in place.
                index = torch.searchsorted(entry[0], ray)
                st, sf, sn = (entry[k][index] for k in (3, 4, 5))
                t, face, normal = self.query(
                    origin, direction, positions, inverse, (~self.static) & (self.material[:, 0] > 0), self._planes
                )
                rank = self.query.semantics
                left = rank[face.clamp_min(0)]
                right = rank[sf.clamp_min(0)]
                closer = (t < st) | ((t == st) & ((left < right) | ((left == right) & (face < sf))))
                t, face, normal = (
                    torch.where(closer, t, st),
                    torch.where(closer, face, sf),
                    torch.where(closer[:, None], normal, sn),
                )
            if save:
                saved.append((ray.clone(), origin, direction, t, face, normal, travelled))
            hit = face >= 0
            rgb[ray[~hit]] = self.background
            ray, origin, direction, membership, travelled, t, face, normal = (
                x[hit] for x in (ray, origin, direction, membership, travelled, t, face, normal)
            )
            if not len(ray):
                break
            label = self.query.labels[face]
            alpha, tau, _ = self.material[label].unbind(-1)
            visible = alpha > 0
            distance = travelled + t
            where = label[visible] * pixels + ray[visible] // self.samples
            if collect:
                paths.scatter_reduce_(0, where, distance[visible], reduce="amin", include_self=True)
            if event == 0:
                depth[ray] = t * self.axis_factor[ray // self.samples]
                ids[ray] = self.labels[label]
                if torch.any((depth[ray] < self.camera["near"]) | (depth[ray] > self.camera["far"])):
                    raise ValueError("optical v1 primary geometry crosses camera clipping interval")
            choice = random_event(ray, 2 * event)
            null = choice < (1 - alpha)
            absorb = (~null) & (choice < (1 - alpha + alpha * (1 - tau)))
            rgb[ray[absorb]] = self._shade(normal[absorb], label[absorb])
            keep = ~absorb
            ray, origin, direction, membership, travelled, t, normal, label, null = (
                x[keep] for x in (ray, origin, direction, membership, travelled, t, normal, label, null)
            )
            if not len(ray):
                break
            dielectric_event = self.material[label, 1] > 0
            next_membership = membership ^ torch.where(dielectric_event, torch.ones_like(label) << label, 0)
            n1 = torch.where(membership != 0, torch.full_like(travelled, self.ior), torch.ones_like(travelled))
            n2 = torch.where(next_membership != 0, torch.full_like(travelled, self.ior), torch.ones_like(travelled))
            bent, reflect = dielectric(direction, normal, n1, n2, random_event(ray, 2 * event + 1))
            newdirection = torch.where(null[:, None], direction, bent)
            crossing = null | ~reflect
            membership = torch.where(crossing, next_membership, membership)
            origin = (origin + t[:, None] * direction).contiguous()
            direction = newdirection.contiguous()
            travelled = travelled + t
        else:
            if len(ray):
                raise RuntimeError("optical path capacity exhausted; no live path may be truncated")
        if save:
            self._seed = (rgb, depth, ids)
        result = {
            "rgb": (rgb.reshape(pixels, self.samples, 3).mean(1).clamp(0, 1) * 255)
            .round()
            .to(torch.uint8)
            .reshape(self.height, self.width, 3),
            "depth": depth.reshape(pixels, self.samples)[:, 0].float().reshape(self.height, self.width),
            "ids": ids.reshape(pixels, self.samples)[:, 0].reshape(self.height, self.width),
        }
        if collect:
            paths = paths.reshape(self.count, self.height, self.width)
            result.update(masks=paths.isfinite(), path_depth=torch.where(paths.isfinite(), paths, 0).float())
        return result, saved

    def _cached(self, positions, inverse, *, collect):
        import triton

        from .optical_cache_cuda import composite

        assert self._seed is not None and self._segments is not None
        rgb, depth, ids = (value.clone() for value in self._seed)
        alive = torch.ones(len(rgb), device=self.device, dtype=torch.bool)
        pixels = self.height * self.width
        paths = torch.full(
            (self.count * pixels if collect else 0,), float("inf"), device=self.device, dtype=torch.float64
        )
        selected = (~self.static) & (self.material[:, 0] > 0)
        for event, entry in enumerate(self._segments):
            ray, origin, direction, st, sf, _, travelled = entry
            if event == 0:
                origin, direction = origin[:: self.samples].contiguous(), direction[:: self.samples].contiguous()
            limit = (st[:: self.samples].contiguous() if event == 0 else st) if self.use_ray_limits else None
            t, face, normal = self.query(origin, direction, positions, inverse, selected, self._planes, limit=limit)
            composite[(triton.cdiv(len(ray), 128),)](
                ray,
                travelled,
                st,
                sf,
                t,
                face,
                normal,
                self.query.labels,
                self.query.semantics,
                self.labels,
                self.colors,
                self.axis_factor,
                alive,
                rgb,
                depth,
                ids,
                paths,
                len(ray),
                pixels,
                self.samples,
                event == 0,
                collect,
                128,
                enable_fp_fusion=False,
            )
        if torch.any((ids > 0) & ((depth < self.camera["near"]) | (depth > self.camera["far"]))):
            raise ValueError("optical v1 primary geometry crosses camera clipping interval")
        result = {
            "rgb": (rgb.reshape(pixels, self.samples, 3).mean(1).clamp(0, 1) * 255)
            .round()
            .to(torch.uint8)
            .reshape(self.height, self.width, 3),
            "depth": depth.reshape(pixels, self.samples)[:, 0].float().reshape(self.height, self.width),
            "ids": ids.reshape(pixels, self.samples)[:, 0].reshape(self.height, self.width),
        }
        if collect:
            paths = paths.reshape(self.count, self.height, self.width)
            result.update(masks=paths.isfinite(), path_depth=torch.where(paths.isfinite(), paths, 0).float())
        return result

    def render(self, frames, *, mode="rgb", out=None):
        return self.render_many(frames, modes=(mode,), out=None if out is None else {mode: out})[mode]

    def render_many(self, frames, *, modes=("rgb", "depth", "ids"), out=None):
        """Include owned CPU pose selection and CUDA upload in the public call."""
        self._check_open("render")
        modes = tuple(modes)
        if not modes or len(set(modes)) != len(modes) or set(modes) - {"rgb", "depth", "ids", "masks", "path_depth"}:
            raise ValueError("invalid optical output modes")
        exact = torch.as_tensor(self.trajectory.transforms(frames), device=self.device)
        cached = self.cache_static and self._cache_possible
        if self.cache_static and not self._cache_possible:
            self.fallbacks += 1
        if cached and self._segments is None:
            p, inv = self._transforms(exact[0])
            _, self._segments = self._walk(p, inv, self.static & (self.material[:, 0] > 0), save=True, collect=False)
            self.builds += 1
        shapes = {k: (len(exact), self.height, self.width) + ((3,) if k == "rgb" else ()) for k in modes}
        for k in ("masks", "path_depth"):
            if k in modes:
                shapes[k] = (len(exact), self.count, self.height, self.width)
        types = {
            "rgb": torch.uint8,
            "depth": torch.float32,
            "ids": torch.int16,
            "masks": torch.bool,
            "path_depth": torch.float32,
        }
        if out is not None:
            if set(out) != set(modes):
                raise ValueError("out must contain exactly the requested modes")
            for k in modes:
                value = out[k]
                if (
                    not torch.is_tensor(value)
                    or tuple(value.shape) != shapes[k]
                    or value.dtype != types[k]
                    or value.device != self.device
                    or not value.is_contiguous()
                ):
                    raise ValueError("out must match shape, dtype, device and be contiguous")
        result = (
            out if out is not None else {k: torch.empty(shapes[k], dtype=types[k], device=self.device) for k in modes}
        )
        selected = self.material[:, 0] > 0
        collect = bool(set(modes) & {"masks", "path_depth"})
        for index, matrix in enumerate(exact):
            p, inv = self._transforms(matrix)
            if cached and self.fuse_cache:
                row = self._cached(p, inv, collect=collect)
            else:
                row, _ = self._walk(p, inv, selected, collect=collect, segments=self._segments if cached else None)
            for k in modes:
                result[k][index].copy_(row[k])
        return result

    def _release(self):
        self._segments = None
        self._seed = None
        del self.query
        for name, value in list(vars(self).items()):
            if torch.is_tensor(value):
                setattr(self, name, None)
