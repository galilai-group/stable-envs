"""The vocabulary the two 3-D backends share: what a primitive IS, before either one draws it.

``raycast3d`` casts analytic primitives with a batched torch loop and a fused Triton kernel;
``cpu3d`` draws MuJoCo's own shading law on the CPU and borrows those intersectors for its movers
and its shadow rays. Both therefore have to agree on three things, and until Step 9 each carried
its own copy of all three:

  * **the kind codes.** ``PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX = 0, 2, 3, 4, 5, 6``
    were spelled out twice, so a model whose geom MuJoCo renumbered would have had to be caught
    twice. They are mjtGeom's own numbers, which is what lets a scene be built straight from an
    ``MjModel`` without a translation table.
  * **the elementwise 3-vector arithmetic.** ``dot3``/``mv3``/``mtv3`` are written out as scalar
    multiplies and adds rather than as ``einsum``/``bmm``, and that spelling is load-bearing rather
    than stylistic: see the long note in ``raycast3d`` above ``_mv``. An elementwise kernel computes
    each output from its own inputs, so its result cannot depend on how many other elements were
    launched beside it, and that is what makes both backends batch-size reproducible. One copy of
    the expression is one place for that property to live.
  * **the bounding radius.** A radius that contains a primitive about its own origin is used for
    three different jobs -- ``raycast3d``'s per-ray bound, its screen-space tile, and ``cpu3d``'s
    mover window and light footprint -- and the three must not drift apart, because each one is a
    claim that something OUTSIDE the radius cannot be hit. ``cpu3d`` used to reach it by building a
    throwaway ``Primitives`` with zeroed colours purely to call the method on it.

Nothing here draws, shades or launches anything: this module imports ``numpy`` and ``torch`` and
nothing else in the package, so it sits below both backends and cannot introduce a cycle. Following
the no-abstraction-without-demand rule (plan §16), every name below has at least two consumers --
the two backends, or a backend and its tests -- and the module deliberately has no base classes,
no registry and no dispatch: it is a vocabulary, not a framework.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

#: mjtGeom codes, so a scene can be built straight from an MjModel without a translation table.
PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX = 0, 2, 3, 4, 5, 6

#: Every kind the analytic intersectors implement, in mjtGeom order. ``raycast3d`` rejects anything
#: outside this set (in both its torch and its Triton path) and ``cpu3d`` derives its mover set from
#: it, so a kind added to one backend cannot silently be missing from the other's error message.
PRIMITIVE_KINDS = (PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX)

#: mjtGeom names, for error messages that name the geom a model actually carries rather than a code.
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

#: The ray parameter of a miss. Large enough to lose every comparison, finite so that arithmetic on
#: it (a clamp, a ``min``) stays a number rather than becoming a NaN.
INF = 1e30


# --------------------------------------------------------------------------- 3-vector arithmetic
# UNROLLED ON PURPOSE. ``torch.einsum("...ij,...j->...i", M, v)`` contracts j by lowering to ``bmm``,
# so a bundle of shape (B, N, 3) becomes B*N independent 3x3-by-3x1 gemms and cuBLAS chooses the
# kernel from that batch COUNT -- which makes ``M @ v`` for one fixed (M, v) come back one ulp apart
# at B=1 and at B=8. One ulp does not stay one ulp in a ray-caster: it enters the quadratic as
# ``disc = b*b - a*c``, which amplifies it by ~1e3-1e4 on a silhouette ray, and ``disc > 0`` is a
# HIT/MISS decision. The full measurement is in ``raycast3d``'s "frame changes" note; this is where
# the expressions live so that both backends inherit the property rather than re-deriving it.


def dot3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """``(a*b).sum(-1)`` for 3-vectors, written out for the reason above."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def mv3(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``M @ v`` for 3-vectors. ``M`` (...,3,3), ``v`` (...,3), broadcasting against each other."""
    return torch.stack(
        [
            M[..., 0, 0] * v[..., 0] + M[..., 0, 1] * v[..., 1] + M[..., 0, 2] * v[..., 2],
            M[..., 1, 0] * v[..., 0] + M[..., 1, 1] * v[..., 1] + M[..., 1, 2] * v[..., 2],
            M[..., 2, 0] * v[..., 0] + M[..., 2, 1] * v[..., 1] + M[..., 2, 2] * v[..., 2],
        ],
        -1,
    )


def mtv3(M: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """``M.T @ v``: world -> geom frame. Same argument as ``mv3``; the transpose is read off the index."""
    return torch.stack(
        [
            M[..., 0, 0] * v[..., 0] + M[..., 1, 0] * v[..., 1] + M[..., 2, 0] * v[..., 2],
            M[..., 0, 1] * v[..., 0] + M[..., 1, 1] * v[..., 1] + M[..., 2, 1] * v[..., 2],
            M[..., 0, 2] * v[..., 0] + M[..., 1, 2] * v[..., 1] + M[..., 2, 2] * v[..., 2],
        ],
        -1,
    )


# --------------------------------------------------------------------------- bounding radii
# ONE GEOMETRIC LAW, TWO SPELLINGS, AND THEY MUST NOT DRIFT. A bounding radius is only ever used to
# assert that something outside it cannot be hit -- a ray skipped, a screen window sized, a shadow
# footprint closed -- so a radius that is too small in one spelling and right in the other is a
# silently wrong picture rather than a slow one. The numpy form answers for a whole set of
# primitives whose sizes are fixed at bake time; the torch form answers per kind for sizes that a
# per-environment ``size_scale`` may have scaled on the device. Same table of cases, deliberately
# adjacent, so a kind cannot be added to one and forgotten in the other.


def bound_radius(kind, size) -> np.ndarray:
    """``(P,)`` numpy: a radius containing each primitive about its own origin. Conservative.

    ``kind`` (P,) mjtGeom codes, ``size`` (P,3) MuJoCo geom sizes. An unbounded kind (the plane, and
    any future one) reports ``INF``, which is what callers test with ``np.isfinite`` before they use
    the result to bound anything.
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
            out[i] = float(s[i].max())  # the largest semi-axis contains it
        else:
            out[i] = INF
    return out


def bound_radius_tensor(kind: int, size: torch.Tensor) -> torch.Tensor | None:
    """The same law for ONE kind over a device tensor of sizes, or ``None`` if it is unbounded.

    ``size`` is (...,3) and the result is (...); ``None`` rather than an infinite tensor because the
    caller's response to an unbounded kind is to skip the bound entirely rather than to compare
    against infinity. Computed from the SIZES THE CALLER PASSED rather than from a baked table, so a
    per-environment ``size_scale`` is already in it and cannot drift out of sync with the geometry
    being tested.
    """
    if kind == SPHERE:
        return size[..., 0]
    if kind in (CAPSULE, CYLINDER):
        return size[..., 0] + size[..., 1]
    if kind == BOX:
        return torch.sqrt(size[..., 0] ** 2 + size[..., 1] ** 2 + size[..., 2] ** 2)
    if kind == ELLIPSOID:
        return torch.maximum(torch.maximum(size[..., 0], size[..., 1]), size[..., 2])
    return None  # PLANE, and any future unbounded kind: always tested


# --------------------------------------------------------------------------- the scene description


@dataclass
class Primitives:
    """A set of geoms sharing one draw. ``kind``/``size`` are static; poses come in per frame.

    This is the whole scene description the 3-D backends take: no hierarchy, no per-geom objects and
    no materials system -- five parallel arrays, because everything downstream of it is a batched
    tensor program that wants columns rather than records.
    """

    kind: np.ndarray  # (P,) mjtGeom
    size: np.ndarray  # (P,3)
    mat_rgb: np.ndarray  # (P,3) in [0,1]
    tex_rgb: np.ndarray  # (P,3) in [0,1]; ones when untextured
    spec: np.ndarray  # (P,) material specular

    def __len__(self) -> int:
        return len(self.kind)

    def to(self, device, dtype=torch.float32) -> dict:
        """The device-side form: kinds stay host-side ints (the cast loop branches on them)."""
        t = lambda x, d=dtype: torch.tensor(np.asarray(x), device=device, dtype=d)  # noqa: E731
        return dict(
            kind=[int(k) for k in self.kind],
            size=t(self.size),
            mat=t(self.mat_rgb),
            tex=t(self.tex_rgb),
            spec=t(self.spec),
        )

    def radii(self, device, dtype, scale: float) -> torch.Tensor | None:
        """``bound_radius() * scale`` as a device tensor, or None if any primitive is unbounded.

        Memoised on the instance: a ``Primitives`` is built once per baked scene and its sizes do not
        change, so rebuilding this per chunk was one blocking host-to-device copy per window per chunk.
        """
        cache = getattr(self, "_radii_cache", None)
        if cache is None:
            cache = self._radii_cache = {}
        key = (str(device), dtype, float(scale))
        if key not in cache:
            r = self.bound_radius() * float(scale)
            cache[key] = None if not np.isfinite(r).all() else torch.tensor(r, device=device, dtype=dtype)
        return cache[key]

    def bound_radius(self) -> np.ndarray:
        """(P,) a radius that contains the primitive about its own origin. Conservative, for tiling."""
        return bound_radius(self.kind, self.size)
