"""Engine-free G34 triangles and batched CUDA rasterization.

Read the packaged declarative scene with the standard library, not MuJoCo.
All five objects share the depth buffer. An nvdiff-only static bake and cropped
dynamic viewports avoid engine calls and bound high-precision coverage work.
"""

from __future__ import annotations

from importlib.resources import files
from xml.etree import ElementTree

import numpy as np
import torch

from ....render import nvdiff as ND
from ....render._resources import Closeable
from ..scenes.g34 import OBJECT_NAMES
from .viewports import CropCamera, PixelCenterRenderer, crop_spheres


def scene_arrays():
    """Return local triangles, fixed transforms, colors and the source camera.

    The XML is only an asset description here. G34 uses axis-aligned boxes and
    one 48-by-24 UV sphere; unsupported geometry must fail rather than disappear.
    Geom order is the frozen oracle's semantic ID order, minus one.
    """
    root = ElementTree.fromstring(files(__package__.rsplit(".", 1)[0]).joinpath("assets/g34/scene.xml").read_text())
    geoms = {g.attrib["name"]: g for g in root.findall(".//geom") if "name" in g.attrib}
    vertices, normals, faces, owners, colors, positions = [], [], [], [], [], []
    offset = 0
    for index, name in enumerate(OBJECT_NAMES):
        geom = geoms[name]
        size = np.fromstring(geom.attrib["size"], sep=" ")
        if geom.attrib["type"] == "box":
            v, n, f = ND._box(size)
        elif geom.attrib["type"] == "sphere":
            v, n, f = ND._sphere(np.repeat(size[0], 3), nslice=48, nstack=24)
        else:
            raise ValueError(f"Unsupported G34 geometry: {geom.attrib['type']}")
        vertices.append(v)
        normals.append(n)
        faces.append(f + offset)
        owners.append(np.full(len(v), index, np.int64))
        colors.append(np.fromstring(geom.attrib["rgba"], sep=" "))
        positions.append(np.fromstring(geom.attrib.get("pos", "0 0 0"), sep=" "))
        offset += len(v)
    camera = root.find(".//camera")
    axes = np.fromstring(camera.attrib["xyaxes"], sep=" ").reshape(2, 3)
    right = axes[0] / np.linalg.norm(axes[0])
    up = axes[1] - np.dot(axes[1], right) * right
    up /= np.linalg.norm(up)
    cam = ND.Camera(
        np.fromstring(camera.attrib["pos"], sep=" "),
        np.stack([right, up, np.cross(right, up)], -1),
        float(camera.attrib["fovy"]),
        0.01,
        100.0,
    )
    return (*map(np.concatenate, (vertices, normals, faces, owners)), np.array(colors), np.array(positions), cam)


def rotations(quaternions):
    """Convert wxyz quaternions to matrices on their device, without an engine."""
    q = quaternions / torch.linalg.vector_norm(quaternions, dim=-1, keepdim=True)
    w, x, y, z = q.unbind(-1)
    return torch.stack(
        (
            1 - 2 * (y * y + z * z),
            2 * (x * y - w * z),
            2 * (x * z + w * y),
            2 * (x * y + w * z),
            1 - 2 * (x * x + z * z),
            2 * (y * z - w * x),
            2 * (x * z - w * y),
            2 * (y * z + w * x),
            1 - 2 * (x * x + y * y),
        ),
        -1,
    ).reshape(-1, 3, 3)


class G34Renderer(Closeable):
    """Own one reusable CUDA triangle scene/context; input rows are xyz+wxyz.

    Rows may reside on CPU: conversion/upload is part of each render call.
    RGB, camera-axis depth and one-based semantic IDs use identical coverage.
    Colors use simple fixed lighting, not a Blender or MuJoCo pixel contract.
    """

    resource_kind = "object_permanence.G34Renderer"

    def __init__(
        self, *, size=90, device="cuda", window_size=None, coverage_mode="scaled", background_coverage="refined"
    ):
        if not isinstance(size, int) or isinstance(size, bool) or size < 9 or size % 9:
            raise ValueError("size must be a positive multiple of 9")
        if window_size is not None and (
            not isinstance(window_size, int) or isinstance(window_size, bool) or not 1 <= window_size <= size
        ):
            raise ValueError("window_size must be a positive integer no larger than image height")
        self.device = torch.device(device)
        if self.device.type != "cuda" or not torch.cuda.is_available():
            raise ValueError("G34 nvdiff rendering requires CUDA")
        if background_coverage not in ("scaled", "refined", "native"):
            raise ValueError("background_coverage must be scaled, refined or native")
        self.background_coverage = background_coverage
        self.height, self.width = size, size * 16 // 9
        v, n, f, vg, rgba, positions, self.camera = scene_arrays()
        scene = ND.Scene.from_triangles(v, n, f, vg, rgba, device=self.device)
        try:
            self.renderer = PixelCenterRenderer(
                scene=scene,
                res=(self.height, self.width),
                device=self.device,
                shading="tiny",
                coverage_mode=coverage_mode,
                textures=False,
                mipmap=False,
                skybox=False,
            )
        except BaseException:
            scene.close()
            raise
        self.renderer.owns_scene = True
        self.renderer.background = np.array([0.96, 0.97, 0.98])
        self.positions = torch.as_tensor(positions, dtype=torch.float32, device=self.device)
        self.identity = torch.eye(3, device=self.device).expand(len(OBJECT_NAMES), 3, 3)
        # Derive the motion bound from the mesh, not a second hardcoded radius.
        self.radius = np.linalg.norm(v[vg == len(OBJECT_NAMES) - 1], axis=-1).max()
        view, projection = self.camera.matrices(self.width / self.height)
        self._view = torch.as_tensor(view, dtype=torch.float32, device=self.device)
        self._full_camera = CropCamera(
            self.camera.pos, self.camera.mat, self.camera.fovy, self.camera.znear, self.camera.zfar
        )
        self._full_camera.view = self._view
        self._full_camera.projection = torch.as_tensor(projection, dtype=torch.float32, device=self.device)
        self._plate = None
        # One fixed viewport contains the mesh's projected sphere plus margin.
        # Bounds are checked for every pose; no clipping or extra tiled draws.
        self.patch = max(8, int(np.ceil(12 * size / 180))) if window_size is None else window_size
        self.chunk = 128
        self._iy = torch.arange(self.patch, device=self.device)[None, :, None]
        self._ix = torch.arange(self.patch, device=self.device)[None, None, :]
        self._opened()

    def _bake(self, modes):
        """Build missing static fields from one shared coverage pass."""
        if self._plate is None:
            self._plate = {}
        missing = tuple(mode for mode in modes if mode not in self._plate)
        if not missing:
            return
        positions = self.positions[None].clone()
        positions[:, -1, 0] = 10000
        self.renderer.height, self.renderer.width = self.height, self.width
        coverage = self.renderer.coverage_mode
        try:
            self.renderer.coverage_mode = self.background_coverage
            self._plate.update(
                self.renderer.render_many(positions, self.identity[None], self._full_camera, modes=missing)
            )
        finally:
            self.renderer.coverage_mode = coverage

    def _crop(self, host_rows):
        """Project the ball center and build exact sub-frusta on the CPU.

        Nominal replay owns CPU poses. GPU-supplied arbitrary poses require a
        host copy here; this path is not advertised as device-only rendering.
        Reject off-screen/near-camera poses instead of silently clipping them.
        """
        proj, x, y = crop_spheres(
            self.camera,
            host_rows[:, None, :3].double().numpy(),
            np.array([self.radius]),
            image_size=(self.height, self.width),
            window_size=self.patch,
        )
        cam = CropCamera(self.camera.pos, self.camera.mat, self.camera.fovy, self.camera.znear, self.camera.zfar)
        cam.view = self._view
        cam.projection = proj
        return cam, x, y

    def render(self, rows, *, mode="rgb", out=None):
        """Retain the single-mode Tensor interface."""
        buffers = None if out is None else {mode: out}
        return self.render_many(rows, modes=(mode,), out=buffers)[mode]

    @torch.no_grad()
    def render_many(self, rows, *, modes=("rgb", "depth", "ids"), out=None):
        """Upload once and share coverage between requested outputs per chunk."""
        self._check_open("render")
        rows = torch.as_tensor(rows)
        if rows.ndim != 2 or rows.shape[1] != 7 or not len(rows):
            raise ValueError("G34 render poses must have nonempty shape (B, 7)")
        if not torch.isfinite(rows).all() or (rows[:, 3:].square().sum(-1) == 0).any():
            raise ValueError("G34 poses must be finite with nonzero quaternions")
        modes = tuple(modes)
        if not modes or len(set(modes)) != len(modes) or any(m not in ("rgb", "depth", "ids") for m in modes):
            raise ValueError("modes must be unique rgb/depth/ids")
        shapes = {m: (len(rows), self.height, self.width) + ((3,) if m == "rgb" else ()) for m in modes}
        if out is not None:
            if set(out) != set(modes):
                raise ValueError("out must contain exactly the requested modes")
            for mode in modes:
                value = out[mode]
                dtype = {"rgb": torch.uint8, "depth": torch.float32, "ids": torch.int16}[mode]
                if (
                    not torch.is_tensor(value)
                    or tuple(value.shape) != shapes[mode]
                    or value.dtype != dtype
                    or value.device != self.positions.device
                    or not value.is_contiguous()
                ):
                    raise ValueError("out must match shape, dtype, CUDA device and be contiguous")
        host = rows.detach().to(device="cpu", dtype=torch.float32)
        # Validate every crop before allocating or writing caller output.
        cameras, xs, ys = self._crop(host)
        self._bake(modes)
        result = {m: self._plate[m].expand(shapes[m]).clone() for m in modes} if out is None else out
        if out is not None:
            for mode in modes:
                out[mode].copy_(self._plate[mode].expand(shapes[mode]))
        rows = host.to(self.device)
        # Upload each batch's projections and origins once; chunks are GPU views.
        projection = torch.as_tensor(cameras.projection, dtype=torch.float32, device=self.device)
        origins = torch.as_tensor(np.stack([xs, ys]), device=self.device)
        self.renderer.height = self.renderer.width = self.patch
        for start in range(0, len(rows), self.chunk):
            stop = min(start + self.chunk, len(rows))
            cam = CropCamera(self.camera.pos, self.camera.mat, self.camera.fovy, self.camera.znear, self.camera.zfar)
            cam.view = self._view
            cam.projection = projection[start:stop]
            x, y = origins[:, start:stop]
            xpos = self.positions.expand(stop - start, -1, -1).clone()
            xmat = self.identity.expand(stop - start, -1, -1, -1).clone()
            xpos[:, -1] = rows[start:stop, :3]
            xmat[:, -1] = rotations(rows[start:stop, 3:])
            images = self.renderer.render_many(xpos, xmat, cam, modes=modes)
            b = torch.arange(start, stop, device=self.device)[:, None, None]
            iy = y[:, None, None] + self._iy
            ix = x[:, None, None] + self._ix
            for mode in modes:
                result[mode][b, iy, ix] = images[mode]
        return result

    def _release(self):
        self.renderer.close()
        self.positions = self.identity = None
        self._view = self._full_camera = self._ix = self._iy = None
        self._plate = None
