"""Full-frame multi-object rendering of exported rigid scene recordings."""

from __future__ import annotations

import numpy as np
import torch

from ....render import nvdiff as ND
from ....render._resources import Closeable
from .transforms import rotations
from .viewports import CropCamera, PixelCenterRenderer


class RigidSceneRenderer(Closeable):
    """Borrow a RigidTrajectory and render arbitrary sampled frames on CUDA.

    All objects share one depth buffer. Fixed-camera recordings reuse static
    visibility, while all movers compete using original face IDs. No crop or
    single-mover assumption. Geometry IDs map to manifest semantic IDs. The
    trajectory remains caller-owned and must stay open while rendering.
    Existing G34Renderer/G34Replay interfaces and their optimized path are separate.
    Optional geometry_precision="float64" preserves projection arithmetic, not
    extra precision absent from recorded meshes, poses or shading attributes.
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
    ):
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("rigid scene rendering requires CUDA")
        if type(size) is not int or size <= 0 or type(chunk_size) is not int or chunk_size <= 0:
            raise ValueError("size and chunk_size must be positive integers")
        if geometry_precision not in ("float32", "float64"):
            raise ValueError("geometry_precision must be float32 or float64")
        camera_dtype = torch.float64 if geometry_precision == "float64" else torch.float32
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
        self._rotations = rotations(base[:, 3:])[None]
        self._node_count = len(manifest["dynamic_nodes"])
        world = np.asarray(camera["world"], np.float64)
        projection = np.asarray(camera["projection"], np.float64)
        self.camera = CropCamera(
            world[:3, 3],
            world[:3, :3],
            float(np.rad2deg(2 * np.arctan(1 / projection[1, 1]))),
            camera["near"],
            camera["far"],
        )
        self.camera.view = torch.as_tensor(np.linalg.inv(world)[None], dtype=camera_dtype, device=self.device)
        self.camera.projection = torch.as_tensor(projection, dtype=camera_dtype, device=self.device)
        self._ids = torch.tensor([0, *[o["id"] for o in manifest["objects"]]], dtype=torch.int16, device=self.device)
        scene = ND.Scene.from_triangles(**trajectory.triangles(), device=self.device)
        try:
            self.renderer = PixelCenterRenderer(
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
        modes = tuple(modes)
        if not modes or len(set(modes)) != len(modes) or any(m not in ("rgb", "depth", "ids") for m in modes):
            raise ValueError("modes must be unique rgb/depth/ids")
        poses = self.trajectory.poses(frames) if self.cache_static else self.trajectory.object_poses(frames)
        shapes = {m: (len(poses), self.height, self.width) + ((3,) if m == "rgb" else ()) for m in modes}
        dtypes = {"rgb": torch.uint8, "depth": torch.float32, "ids": torch.int16}
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
                    node_rotations = rotations(part[..., 3:].reshape(-1, 4)).reshape(len(part), self._node_count, 3, 3)
                    positions_gpu[:, self._moving] = part[:, self._nodes, :3]
                    rotations_gpu[:, self._moving] = node_rotations[:, self._nodes]
            else:
                positions_gpu = part[..., :3].contiguous()
                rotations_gpu = rotations(part[..., 3:].reshape(-1, 4)).reshape(len(part), part.shape[1], 3, 3)
            buffers = {m: result[m][start:stop] for m in modes}
            self.renderer.render_many(positions_gpu, rotations_gpu, self.camera, modes=modes, out=buffers)
            if "ids" in modes:
                buffers["ids"].copy_(self._ids[buffers["ids"].long()])
        return result

    def _release(self):
        self.renderer.close()
        del self._ids, self._positions, self._rotations, self._moving, self._nodes, self.camera, self.trajectory
