"""Reusable point-sampled viewports for fixed-camera triangle replays.

Asset import and task semantics stay with each scene. These helpers know only
camera geometry and coverage precision.
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
