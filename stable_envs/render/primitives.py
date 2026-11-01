"""MuJoCo primitive geometry codes, unrolled vector helpers, and conservative bounding radii."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

# mjtGeom codes, read directly from an MjModel.
PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX = 0, 2, 3, 4, 5, 6

# Supported analytic intersections, shared by backend validation.
PRIMITIVE_KINDS = (PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX)

GEOM_NAMES = {
    0: "mjGEOM_PLANE",
    1: "mjGEOM_HFIELD",
    2: "mjGEOM_SPHERE",
    3: "mjGEOM_CAPSULE",
    4: "mjGEOM_ELLIPSOID",
    5: "mjGEOM_CYLINDER",
    6: "mjGEOM_BOX",
    7: "mjGEOM_MESH",
    8: "mjGEOM_SDF",
}

# A finite miss sentinel avoids NaNs in later arithmetic.
INF = 1e30


# Keep vector arithmetic unrolled: batch-dependent BLAS reductions can move silhouette hit tests.


def dot3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Dot product of broadcastable (..., 3) tensors with fixed arithmetic order."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def mv3(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Multiply (..., 3, 3) matrices by broadcastable (..., 3) column vectors."""
    return torch.stack(
        [
            M[..., 0, 0] * v[..., 0] + M[..., 0, 1] * v[..., 1] + M[..., 0, 2] * v[..., 2],
            M[..., 1, 0] * v[..., 0] + M[..., 1, 1] * v[..., 1] + M[..., 1, 2] * v[..., 2],
            M[..., 2, 0] * v[..., 0] + M[..., 2, 1] * v[..., 1] + M[..., 2, 2] * v[..., 2],
        ],
        -1,
    )


def mtv3(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Multiply by the transposed matrix, converting world vectors to local coordinates."""
    return torch.stack(
        [
            M[..., 0, 0] * v[..., 0] + M[..., 1, 0] * v[..., 1] + M[..., 2, 0] * v[..., 2],
            M[..., 0, 1] * v[..., 0] + M[..., 1, 1] * v[..., 1] + M[..., 2, 1] * v[..., 2],
            M[..., 0, 2] * v[..., 0] + M[..., 1, 2] * v[..., 1] + M[..., 2, 2] * v[..., 2],
        ],
        -1,
    )


# NumPy and torch bounds must remain conservative under the same geometry law.


def bound_radius(kind, size) -> np.ndarray:
    """Return conservative radii for MuJoCo kinds (P,) and sizes (P, 3).

    Planes and unknown kinds use the finite miss sentinel INF, not IEEE infinity.
    """
    kinds = np.asarray(kind)
    s = np.asarray(size, np.float64)
    out = np.zeros(len(kinds))
    for i, k in enumerate(kinds):
        if k == SPHERE:
            out[i] = s[i, 0]
        elif k in (CAPSULE, CYLINDER):
            out[i] = s[i, 0] + s[i, 1]
        elif k == BOX:
            out[i] = float(np.linalg.norm(s[i]))
        elif k == ELLIPSOID:
            out[i] = float(s[i].max())
        else:
            out[i] = INF
    return out


def bound_radius_tensor(kind: int, size: torch.Tensor) -> torch.Tensor | None:
    """Return radii for one kind and sizes (..., 3), or None for unbounded/unknown kinds."""
    if kind == SPHERE:
        return size[..., 0]
    if kind in (CAPSULE, CYLINDER):
        return size[..., 0] + size[..., 1]
    if kind == BOX:
        return torch.sqrt(size[..., 0] ** 2 + size[..., 1] ** 2 + size[..., 2] ** 2)
    if kind == ELLIPSOID:
        return torch.maximum(torch.maximum(size[..., 0], size[..., 1]), size[..., 2])
    return None  # PLANE and other unbounded kinds are always tested


@dataclass
class Primitives:
    """Parallel geometry and material arrays shared across frames; poses are supplied separately.

    Treat sizes as immutable once radii have been cached.
    """

    kind: np.ndarray  # (P,) mjtGeom
    size: np.ndarray  # (P,3)
    mat_rgb: np.ndarray  # (P,3) in [0,1]
    tex_rgb: np.ndarray  # (P,3) in [0,1]; ones when untextured
    spec: np.ndarray  # (P,) material specular

    def __len__(self) -> int:
        return len(self.kind)

    def subset(self, ids: Sequence[int]) -> Primitives:
        """Return the primitives at ``ids``; each subset is cached, so its radii are cached too."""
        key = tuple(int(j) for j in ids)
        cache = self.__dict__.setdefault("_subset_cache", {})
        if key not in cache:
            take = np.asarray(key, np.intp)
            cache[key] = Primitives(
                kind=np.asarray(self.kind)[take],
                size=np.asarray(self.size)[take],
                mat_rgb=np.asarray(self.mat_rgb)[take],
                tex_rgb=np.asarray(self.tex_rgb)[take],
                spec=np.asarray(self.spec)[take],
            )
        return cache[key]

    def to(self, device, dtype=torch.float32) -> dict:
        """Copy numeric arrays to device tensors; keep kinds as host integers for dispatch."""
        t = lambda x, d=dtype: torch.tensor(np.asarray(x), device=device, dtype=d)  # noqa: E731
        return dict(
            kind=[int(k) for k in self.kind],
            size=t(self.size),
            mat=t(self.mat_rgb),
            tex=t(self.tex_rgb),
            spec=t(self.spec),
        )

    def radii(self, device, dtype, scale: float) -> torch.Tensor | None:
        """Cache scaled radii by device, dtype, and scale; return None for nonfinite values."""
        cache = getattr(self, "_radii_cache", None)
        if cache is None:
            cache = self._radii_cache = {}
        key = (str(device), dtype, float(scale))
        if key not in cache:
            r = self.bound_radius() * float(scale)
            cache[key] = None if not np.isfinite(r).all() else torch.tensor(r, device=device, dtype=dtype)
        return cache[key]

    def bound_radius(self) -> np.ndarray:
        """Return conservative host radii using the shared primitive bounds."""
        return bound_radius(self.kind, self.size)
