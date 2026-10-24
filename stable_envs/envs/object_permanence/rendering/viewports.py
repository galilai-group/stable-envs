"""Reusable point-sampled viewports for fixed-camera triangle replays.

Asset import and task semantics stay with each scene. These helpers know only
camera geometry, conservative world-space bounds, and coverage precision.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from ....render import nvdiff as ND

if TYPE_CHECKING:
    import torch


class PixelCenterRenderer(ND.BatchRenderer):
    """Choose coverage precision while retaining original pixel centers.

    nvdiffrast CUDA uses four subpixel bits. Odd scale 17 gives approximately
    eight bits at source resolution. Only coverage is enlarged; interpolation
    and shading use the selected centers. Refined mode instead uses a float64
    triangle recheck at source resolution, falling back to scaled coverage for
    depth clipping or missing Triton. Native mode is an explicit experiment.
    Optional CUDA bool static_geoms[G] enables refined visibility caching with
    automatic clip-position/resolution invalidation. Topology stays immutable.
    Optional float64 geometry retains matrix/vertex projection precision before
    refined coverage. Scaled fallback casts to float32 and sets last_fallback.
    Derivative buffers/mipmapping are unsupported by these sampling paths.
    """

    def __init__(
        self,
        *args,
        coverage_scale=17,
        coverage_mode="scaled",
        static_geoms=None,
        geometry_precision="float32",
        **kwargs,
    ):
        if (
            not isinstance(coverage_scale, int)
            or isinstance(coverage_scale, bool)
            or coverage_scale < 1
            or coverage_scale % 2 != 1
        ):
            raise ValueError("coverage_scale must be a positive odd integer")
        if kwargs.get("mipmap", True):
            raise ValueError("point-sampled coverage requires mipmap=False")
        if coverage_mode not in ("scaled", "refined", "native"):
            raise ValueError("coverage_mode must be scaled, refined or native")
        if tuple(kwargs.get("sample_offset", (0, 0))) != (0, 0):
            raise ValueError("pixel-center coverage requires zero sample_offset")
        if geometry_precision not in ("float32", "float64"):
            raise ValueError("geometry_precision must be float32 or float64")
        if geometry_precision == "float64" and coverage_mode != "refined":
            raise ValueError("float64 geometry requires refined coverage")
        self.geometry_precision = geometry_precision
        self.coverage_scale = coverage_scale
        self.coverage_mode = coverage_mode
        self.last_fallback = False
        self._static_geoms = static_geoms
        self._static_coverage = None
        super().__init__(*args, **kwargs)

    def _clip(self, xpos, xmat, view, proj):
        if self.geometry_precision == "float32":
            return super()._clip(xpos, xmat, view, proj)
        import torch

        view = torch.as_tensor(view, dtype=torch.float64, device=self.device)
        proj = torch.as_tensor(proj, dtype=torch.float64, device=self.device)
        return self.scene.clip(xpos, xmat, view, proj, coverage_dtype=torch.float64)

    def _rasterize(self, positions, resolution):
        self.last_fallback = False
        if self.coverage_mode == "native":
            return super()._rasterize(positions, resolution)
        if self.coverage_mode == "refined":
            try:
                from ....render.nvdiff_coverage import StaticCoverage, rasterize
            except ModuleNotFoundError as error:
                if error.name is None or error.name.split(".")[0] != "triton":
                    raise
                raster = None
            else:
                if self._static_geoms is not None and self._static_coverage is None:
                    self._static_coverage = StaticCoverage(self.scene.face, self._static_geoms[self.scene.fgeom])
                raster = rasterize(positions, self.scene.face, resolution, cache=self._static_coverage)
            if raster is not None:
                return raster, None
            self.last_fallback = True
        h, w = (resolution, resolution) if isinstance(resolution, int) else resolution
        scale = self.coverage_scale
        raster, _ = super()._rasterize(positions.float(), (h * scale, w * scale))
        center = scale // 2
        return raster[:, center::scale, center::scale].contiguous(), None

    def _release(self):
        self._static_coverage = self._static_geoms = None
        super()._release()


class CropCamera(ND.Camera):
    """An off-axis view with optional pre-uploaded view/projection matrices.

    Position/orientation remain ordinary Camera metadata. Supplying CUDA matrices
    lets BatchRenderer reuse them without a new host upload for every chunk.
    """

    view: np.ndarray | torch.Tensor | None
    projection: np.ndarray | torch.Tensor

    def matrices(self, aspect=1.0):
        view = getattr(self, "view", None)
        if view is None:
            view, _ = super().matrices(aspect)
        return view, self.projection


def crop_spheres(camera, centers, radii, *, image_size, window_size, margin=0.5):
    """Return CPU float64 crop projections and int64 top-left x/y offsets.

    ``centers`` is (B,D,3) world coordinates and ``radii`` is (D,): D dynamic
    objects share one window per frame. Bounds use exact perspective extrema of
    enclosing spheres, including off-axis centers. Camera is fixed/unbatched.
    Window size is fixed across batches to preserve sample/batch invariance.

    Reject bounds crossing the near plane, image edge, or window margin. Callers
    must choose a larger window or full-frame drawing for unsupported scenes;
    this helper never silently clips or tiles a moving object.
    """
    centers = np.asarray(centers, np.float64)
    radii = np.asarray(radii, np.float64)
    if (
        centers.ndim != 3
        or centers.shape[-1] != 3
        or not centers.shape[0]
        or not centers.shape[1]
        or radii.shape != (centers.shape[1],)
        or not np.isfinite(centers).all()
        or not np.isfinite(radii).all()
        or np.any(radii <= 0)
    ):
        raise ValueError("bounds require finite centers (B,D,3) and positive radii (D,)")
    h, w = image_size
    if (
        not isinstance(window_size, int)
        or isinstance(window_size, bool)
        or window_size < 1
        or not np.isfinite(margin)
        or margin < 0
    ):
        raise ValueError("window size and margin must be valid")
    view, projection = camera.matrices(w / h)
    if view.shape != (1, 4, 4) or projection.shape != (4, 4):
        raise ValueError("crop bounds require one fixed perspective camera")
    points = centers @ view[0, :3, :3].T + view[0, :3, 3]
    d, r = -points[..., 2], radii[None]
    if np.any(d - r <= camera.znear) or np.any(d + r >= camera.zfar):
        raise ValueError("pose exceeds viewport depth bounds")
    denominator = d * d - r * r
    # Tangent rays to a sphere: extrema of x/depth and y/depth. A simple
    # radius/depth estimate misses the extra footprint of off-axis objects.
    mid = points[..., :2] * d[..., None] / denominator[..., None]
    span = r[..., None] * np.sqrt(points[..., :2] ** 2 + denominator[..., None]) / denominator[..., None]
    scale = np.array([projection[0, 0] * w / 2, -projection[1, 1] * h / 2])
    a = (mid - span) * scale + np.array([w / 2, h / 2])
    b = (mid + span) * scale + np.array([w / 2, h / 2])
    lo = np.minimum(a, b).min(axis=1)
    hi = np.maximum(a, b).max(axis=1)
    # Choose the nearest integer origin, avoiding the nearly-one-pixel loss on
    # one side caused by always flooring. Keep at least half a pixel of margin.
    origins = np.floor((lo + hi - window_size) / 2 + 0.5).astype(np.int64)
    if (
        np.any(lo - origins < margin)
        or np.any(origins + window_size - hi < margin)
        or np.any(origins < 0)
        or np.any(origins + window_size > np.array([w, h]))
    ):
        raise ValueError("pose exceeds viewport window bounds")
    x, y = origins.T
    proj = np.broadcast_to(projection, (len(centers), 4, 4)).copy()
    proj[:, 0] = projection[0] * w / window_size + projection[3] * ((w - 2 * x - window_size) / window_size)[:, None]
    proj[:, 1] = projection[1] * h / window_size + projection[3] * ((2 * y + window_size - h) / window_size)[:, None]
    return proj, x, y
