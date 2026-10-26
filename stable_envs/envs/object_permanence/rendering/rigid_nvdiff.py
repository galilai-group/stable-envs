"""Full-frame multi-object rendering of exported rigid scene recordings."""

from __future__ import annotations

from typing import Any

import numpy as np
import torch

from ....render import nvdiff as ND
from ....render._resources import Closeable
from .transforms import rotations
from .viewports import CropCamera, PixelCenterRenderer


class RigidSceneRenderer(Closeable):
    """Borrow a RigidTrajectory and render arbitrary sampled frames on CUDA.

    All objects share one depth buffer. Legacy fixed-camera coverage reuses static
    visibility, while movers compete using original face IDs. No crop or
    single-mover assumption. Geometry IDs map to manifest semantic IDs. The
    trajectory remains caller-owned and must stay open while rendering.
    Optional geometry_precision="float64" preserves projection arithmetic, not
    extra precision absent from recorded meshes, poses or shading attributes.
    Opt-in visibility_contract="stable-envs/opaque-pixel-rays-v1" uses float64
    quaternion/visibility arithmetic and semantic-ID ties, with optional static
    visibility seeds. cache_static=False keeps complete drawing for comparison.
    Explicit opaque-pixel-rays-v2 additionally accepts fixed orthographic
    cameras, with per-pixel ray origins and camera-axis interval clipping.
    V3 optional relative scales affect vertices and inverse-transposed normals;
    scale-only nodes remain dynamic and reuse the same static visibility seed.
    """

    resource_kind = "object_permanence.RigidSceneRenderer"

    def __init__(
        self,
        trajectory,
        *,
        size=180,
        device="cuda",
        chunk_size=4,
        coverage_mode="refined",
        cache_static=True,
        geometry_precision="float32",
        visibility_contract=None,
        max_layers=16,
    ):
        self.renderer: Any
        self._optical = None
        if geometry_precision not in ("float32", "float64"):
            raise ValueError("geometry_precision must be float32 or float64")
        if trajectory.manifest.get("schema") == "stable-envs/object-permanence-rigid-v5":
            if coverage_mode != "refined":
                raise ValueError("optical rays do not support raster coverage overrides")
            from .optical_scene import OpticalSceneRenderer

            self._optical = OpticalSceneRenderer(
                trajectory,
                size=size,
                device=device,
                chunk_size=chunk_size,
                cache_static=cache_static,
                visibility_contract=visibility_contract,
            )
            self.renderer = self._optical
            self.device = self._optical.device
            self.height, self.width = self._optical.height, self._optical.width
            self.visibility_contract = visibility_contract
            self.chunk_size = chunk_size
            self._opened()
            return
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("rigid scene rendering requires CUDA")
        if type(size) is not int or size <= 0 or type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("size and chunk_size must be positive integers")
        camera_type = trajectory.manifest["camera"].get("projection_type", "perspective")
        from .transparency import CONTRACT as ALPHA_CONTRACT

        self._transparent = trajectory.manifest.get("material_convention") is not None
        self.max_layers = max_layers
        if self._transparent and visibility_contract != ALPHA_CONTRACT:
            raise ValueError("transparent recordings require their explicit visibility contract")
        if visibility_contract == ALPHA_CONTRACT and not self._transparent:
            raise ValueError("transparent visibility requires a versioned material recording")
        self._scaled = trajectory.has_scales
        if self._scaled and visibility_contract != trajectory.manifest.get("visibility_contract"):
            raise ValueError("scaled recordings require their explicit visibility contract")
        if camera_type == "orthographic" and visibility_contract not in (
            "stable-envs/opaque-pixel-rays-v2",
            ALPHA_CONTRACT,
        ):
            raise ValueError("orthographic recordings require explicit opaque-pixel-rays-v2")
        renderer_class: type = PixelCenterRenderer
        renderer_options: dict = {}
        if visibility_contract is not None:
            from .visibility import CONTRACTS, validate_camera
            from .visibility_cuda import ContractRenderer

            if visibility_contract not in (*CONTRACTS, ALPHA_CONTRACT) or coverage_mode != "refined":
                raise ValueError("unsupported visibility contract or coverage mode")
            validate_camera(
                trajectory.manifest["camera"],
                contract="stable-envs/opaque-pixel-rays-v2" if self._transparent else visibility_contract,
            )
            geometry_precision = "float64"
            renderer_class = ContractRenderer
            if self._transparent:
                from .transparency_cuda import AlphaRenderer

                renderer_class = AlphaRenderer
                renderer_options["max_layers"] = max_layers
            renderer_options.update(
                {
                    "camera_metadata": trajectory.manifest["camera"],
                    "semantic_ids": torch.tensor(
                        [o["id"] for o in trajectory.manifest["objects"]], device=self.device
                    ),
                    "visibility_contract": visibility_contract,
                    "affine_transforms": self._scaled,
                }
            )
        camera_dtype = torch.float64 if geometry_precision == "float64" else torch.float32
        self.visibility_contract = visibility_contract
        manifest = trajectory.manifest
        camera = manifest["camera"]
        if size * camera["width"] % camera["height"]:
            raise ValueError("size must preserve the recorded camera aspect ratio")
        self.height, self.width = size, size * camera["width"] // camera["height"]
        if type(cache_static) is not bool:
            raise ValueError("cache_static must be boolean")
        self.trajectory, self.chunk_size = trajectory, chunk_size
        self.cache_static = cache_static
        objects = manifest["objects"]
        nodes = np.asarray([obj["node"] for obj in objects])
        self._moving = torch.as_tensor(np.flatnonzero(nodes >= 0), device=self.device)
        self._nodes = torch.as_tensor(nodes[nodes >= 0], device=self.device)
        base = torch.tensor([obj["pose"] for obj in objects], dtype=torch.float32, device=self.device)
        self._positions = base[None, :, :3].contiguous()
        self._rotations = rotations(base[:, 3:].double() if visibility_contract is not None else base[:, 3:])[None]
        self._node_count = len(manifest["dynamic_nodes"])
        world = np.asarray(camera["world"], np.float64)
        projection = np.asarray(camera["projection"], np.float64)
        self.camera = CropCamera(
            world[:3, 3],
            world[:3, :3],
            90.0 if camera_type == "orthographic" else float(np.rad2deg(2 * np.arctan(1 / projection[1, 1]))),
            camera["near"],
            camera["far"],
        )
        self.camera.view = torch.as_tensor(np.linalg.inv(world)[None], dtype=camera_dtype, device=self.device)
        self.camera.projection = torch.as_tensor(projection, dtype=camera_dtype, device=self.device)
        self._ids = torch.tensor([0, *[o["id"] for o in manifest["objects"]]], dtype=torch.int16, device=self.device)
        scene = ND.Scene.from_triangles(**trajectory.triangles(), device=self.device)
        try:
            self.renderer = renderer_class(
                scene=scene,
                res=(self.height, self.width),
                device=self.device,
                textures=False,
                mipmap=False,
                skybox=False,
                shading="tiny",
                coverage_mode=coverage_mode,
                geometry_precision=geometry_precision,
                static_geoms=torch.as_tensor(nodes < 0, device=self.device) if cache_static else None,
                **renderer_options,
            )
        except BaseException:
            scene.close()
            raise
        self.renderer.owns_scene = True
        self.renderer.background = np.array([0.96, 0.97, 0.98])
        self._opened()

    def render(self, frames, *, mode="rgb", out=None):
        """Return one CUDA field from zero-based clip frame indices."""
        buffers = None if out is None else {mode: out}
        return self.render_many(frames, modes=(mode,), out=buffers)[mode]

    def render_many(self, frames, *, modes=("rgb", "depth", "ids"), out=None):
        """Upload selected CPU poses once; share coverage across requested fields."""
        self._check_open("render")
        if self._optical is not None:
            self._optical.chunk_size = self.chunk_size
            return self._optical.render_many(frames, modes=modes, out=out)
        modes = tuple(modes)
        allowed = {"rgb", "depth", "ids"}
        if self._transparent:
            allowed |= {"masks", "weights", "layer_depth", "layer_ids"}
        if not modes or len(set(modes)) != len(modes) or set(modes) - allowed:
            raise ValueError("modes must be unique rgb/depth/ids")
        poses = self.trajectory.poses(frames) if self.cache_static else self.trajectory.object_poses(frames)
        scales = None
        if self._scaled:
            selected = self.trajectory.scales(frames) if self.cache_static else self.trajectory.object_scales(frames)
            scales = torch.as_tensor(selected, device=self.device)
        shapes = {m: (len(poses), self.height, self.width) + ((3,) if m == "rgb" else ()) for m in modes}
        dtypes = {"rgb": torch.uint8, "depth": torch.float32, "ids": torch.int16}
        for mode in ("masks", "weights"):
            if mode in modes:
                shapes[mode] = (len(poses), len(self.trajectory.manifest["objects"]), self.height, self.width)
        for mode in ("layer_depth", "layer_ids"):
            if mode in modes:
                shapes[mode] = (len(poses), self.height, self.width, self.max_layers)
        dtypes.update(masks=torch.bool, weights=torch.float32, layer_depth=torch.float32, layer_ids=torch.int16)
        if out is not None:
            if set(out) != set(modes):
                raise ValueError("out must contain exactly the requested modes")
            for m in modes:
                value = out[m]
                if (
                    not torch.is_tensor(value)
                    or tuple(value.shape) != shapes[m]
                    or value.dtype != dtypes[m]
                    or value.device != self._ids.device
                    or not value.is_contiguous()
                ):
                    raise ValueError("out must match shape, dtype, CUDA device and be contiguous")
        result = (
            {m: torch.empty(shapes[m], dtype=dtypes[m], device=self.device) for m in modes} if out is None else out
        )
        rows = torch.as_tensor(poses, device=self.device)
        for start in range(0, len(rows), self.chunk_size):
            stop = min(start + self.chunk_size, len(rows))
            part = rows[start:stop]
            if self.cache_static:
                positions_gpu = self._positions.expand(len(part), -1, -1).clone()
                rotations_gpu = self._rotations.expand(len(part), -1, -1, -1).clone()
                if self._node_count:
                    quaternions = part[..., 3:].reshape(-1, 4)
                    node_rotations = rotations(
                        quaternions.double() if self.visibility_contract else quaternions
                    ).reshape(len(part), self._node_count, 3, 3)
                    if scales is not None:
                        node_rotations = node_rotations * scales[start:stop, :, None, :]
                    positions_gpu[:, self._moving] = part[:, self._nodes, :3]
                    rotations_gpu[:, self._moving] = node_rotations[:, self._nodes]
            else:
                positions_gpu = part[..., :3].contiguous()
                quaternions = part[..., 3:].reshape(-1, 4)
                rotations_gpu = rotations(quaternions.double() if self.visibility_contract else quaternions).reshape(
                    len(part), part.shape[1], 3, 3
                )
                if scales is not None:
                    rotations_gpu = rotations_gpu * scales[start:stop, :, None, :]
            buffers = {m: result[m][start:stop] for m in modes}
            self.renderer.render_many(positions_gpu, rotations_gpu, self.camera, modes=modes, out=buffers)
            if "ids" in modes:
                buffers["ids"].copy_(self._ids[buffers["ids"].long()])
            if "layer_ids" in modes:
                buffers["layer_ids"].copy_(self._ids[buffers["layer_ids"].long()])
        return result

    def _release(self):
        if self._optical is not None:
            self._optical.close()
            return
        self.renderer.close()
        del self._ids, self._positions, self._rotations, self._moving, self._nodes, self.camera, self.trajectory
