"""Batched analytic ray-cast for 3-D scenes made of MuJoCo's primitive geoms.

The 2-D path (``raster2d``) draws a ``Scene2D`` of polygons and discs. That is the wrong IR for a MuJoCo
scene: the picture is a perspective projection of boxes, spheres, capsules and cylinders, and flattening
it to polygons would mean pre-projecting geometry that the camera is entitled to move. So this module
keeps the scene in 3-D and answers the only question a renderer has to answer per pixel — which surface
does this ray hit, and what colour is it there.

Every primitive here is a closed-form intersection (a quadratic, or a slab test), which is why an
analytic caster can be exact rather than approximate: there is no tessellation to match. Measured against
MuJoCo's own segmentation buffer on the OGBench mazes, the geom-id map produced by this code agrees on
**100.000% of interior pixels with zero mismatches**, across pointmaze/antmaze/humanoidmaze and across
both the fixed top-down camera and the agent-tracking ``back`` camera.

Two conventions were measured rather than read off the source, and both are invisible by eye:

  * **The camera is the AVERAGE of the two ``mjvGLCamera`` eyes.** ``mjv_updateScene`` fills a left and a
    right eye separated by ``vis.global_.ipd`` (0.068 m). Rendering mono from ``camera[0]`` is a 34 mm
    lateral offset that misplaces 0.7% of pixels; the average reproduces the reference exactly (scored:
    0.9928 / 0.9952 / **1.0000** for eye0 / eye1 / average).
  * **MuJoCo's depth buffer is perpendicular distance**, ``t * (d . forward)``, not radial ``t``. Scored
    at 0.001-0.012 m against zfar 1700-3300 m for the perpendicular convention.

Shading is the fixed-function composition, and the composition matters more than the constants:

    primary = mat_rgb * (A + sum_l D_l * max(0, n.l_l))  +  sum_l S_l * spec * max(0, n.h_l)^p
    final   = primary * tex_rgb

The specular term is WHITE and ADDITIVE, and the texture modulates the whole primary colour. That is not
cosmetic: a textured surface (the maze floor, whose material colour is (1,1,1) and whose colour comes
entirely from a texture) then sees the specular as a pure MULTIPLIER, while an untextured surface (walls,
agent) sees it added. Both behaviours were measured — the channel spread of ``obs/base`` is 3e-4 on the
floor and 6.8e-2 on the agent — and a multiplicative model fitted on the floor overshoots wall tops by
6.5/255 and mangles the agent's highlight. With the composition right, wall tops go from 13/255 worst to
2/255 and vertical wall faces become bit-exact.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from . import _resources as _RES
from .primitives import (  # noqa: F401
    BOX,
    CAPSULE,
    CYLINDER,
    ELLIPSOID,
    GEOM_NAMES,
    INF,
    PLANE,
    PRIMITIVE_KINDS,
    SPHERE,
    Primitives,
)
from .primitives import bound_radius_tensor as _bound_radius
from .primitives import dot3 as _dot
from .primitives import mtv3 as _mtv
from .primitives import mv3 as _mv

#: What the cast functions take and give back. ``Kinds`` is host-side on purpose: the torch loop
#: branches on each code in Python and the Triton path turns it into a device tensor once, memoised,
#: so a tensor here would be a host-to-device copy on every pass. ``Cast`` is (t, normal, index):
#: the ray parameter, the world-space unit normal at the hit, and the winning primitive's index.
Kinds = Sequence[int] | np.ndarray
Cast = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # reported by the renderer, never silent
    HAVE_TRITON = False

# The mjtGeom kind codes, ``INF``, the ``Primitives`` scene description, the unrolled 3-vector
# arithmetic and the bounding-radius law live in ``primitives`` because ``cpu3d`` needs exactly the
# same ones and used to carry its own copies. They are re-exported here under the names they have
# always had, so ``R3.SPHERE``, ``R3.INF``, ``R3.Primitives``, ``R3._dot``, ``R3._mv``, ``R3._mtv``
# and ``R3._bound_radius`` are unchanged for every caller and every test.


@dataclass(frozen=True)
class Camera:
    """A pinhole, in the exact form ``mjvGLCamera`` reports it.

    ``near``/``bottom``/``top``/``center`` are the frustum at the near plane; ``mjr_render`` derives
    left/right from ``center +- 0.5*(top-bottom)*aspect``. Storing the frustum rather than a field of
    view keeps this faithful to an off-centre or stereo camera instead of assuming a symmetric one.
    """

    pos: tuple
    forward: tuple
    up: tuple
    near: float
    top: float
    bottom: float
    center: float = 0.0

    @staticmethod
    def from_mjv(glcam_a, glcam_b=None) -> Camera:
        """From one ``mjvGLCamera``, or the AVERAGE of two (which is what mono rendering uses)."""

        def g(c):
            return (
                np.asarray(c.pos, np.float64),
                np.asarray(c.forward, np.float64),
                np.asarray(c.up, np.float64),
                float(c.frustum_near),
                float(c.frustum_top),
                float(c.frustum_bottom),
                float(c.frustum_center),
            )

        a = g(glcam_a)
        v = a if glcam_b is None else tuple((x + y) / 2 for x, y in zip(a, g(glcam_b)))
        return Camera(tuple(v[0]), tuple(v[1]), tuple(v[2]), v[3], v[4], v[5], v[6])

    def basis(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        f = np.asarray(self.forward, np.float64)
        f = f / np.linalg.norm(f)
        r = np.cross(f, np.asarray(self.up, np.float64))
        r = r / np.linalg.norm(r)
        u = np.cross(r, f)
        return r, u, f

    def rays(self, H: int, W: int, device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
        """``(o (3,), d (H*W,3))`` unit ray directions through pixel CENTRES, row 0 at the top."""
        r, u, f = self.basis()
        half_h = 0.5 * (self.top - self.bottom)
        half_w = half_h * (W / H)
        mid_y = 0.5 * (self.top + self.bottom)
        xs = self.center + (2 * (np.arange(W) + 0.5) / W - 1) * half_w
        ys = mid_y + (1 - 2 * (np.arange(H) + 0.5) / H) * half_h
        X, Y = np.meshgrid(xs, ys)
        d = X[..., None] * r + Y[..., None] * u + self.near * f
        d = d / np.linalg.norm(d, axis=-1, keepdims=True)
        return (
            torch.tensor(self.pos, device=device, dtype=dtype),
            torch.tensor(d.reshape(-1, 3), device=device, dtype=dtype),
        )

    def project(self, pts: torch.Tensor, aspect: float = 1.0) -> torch.Tensor:
        """World ``(...,3)`` -> normalised device coords in [-1,1]^2. Screen-space tiling only.

        ``aspect`` is W/H: the frustum reports its vertical extent and ``mjr_render`` widens it
        horizontally, so x must be divided by ``half_h*aspect`` and y by ``half_h``.
        """
        R, origin = _proj_tensors(self, pts.device, pts.dtype)
        rel = pts - origin
        # ``rel @ R.T`` would be a gemm whose M dimension is the batch, and the tile window it feeds is
        # an INTEGER pixel range: one ulp on the wrong side of a floor() moves the whole cast window and
        # every ray in it. Same elementwise form as the intersectors, for the same reason.
        loc = _mv(R, rel)
        z = loc[..., 2].clamp_min(1e-6)
        half_h = 0.5 * (self.top - self.bottom)
        mid_y = 0.5 * (self.top + self.bottom)
        x = (loc[..., 0] / z * self.near - self.center) / (half_h * aspect)
        y = (loc[..., 1] / z * self.near - mid_y) / half_h
        return torch.stack([x, y], -1)


# --------------------------------------------------------------------------- constant-tensor caches
#
# ``torch.tensor(x, device="cuda")`` is not a cheap constructor. It allocates PAGEABLE host memory and
# copies from it, and a pageable host-to-device copy cannot be queued -- the driver has to stage it, so
# the call blocks until the stream drains. Measured with cProfile on DMCFingerTurnEasy at 256 px batch
# 256: 248 such calls, 37 ms of a 121 ms render, i.e. 150 us EACH for a nine-element matrix. Twenty-seven
# of the thirty-one per chunk are CONSTANTS -- a camera basis, eight corner sign patterns, a floor pose,
# a set of bounding radii -- rebuilt because they were written where they are used rather than where
# they are computed. Memoising them is not an optimisation of the arithmetic; it removes a stall.

_Memo = _RES.BoundedCache

_CACHE_MAX = 32
_PROJ_CACHE = _Memo(_CACHE_MAX, name="raycast3d.proj")
_CORNER_CACHE = _Memo(_CACHE_MAX, name="raycast3d.corner")


def _proj_tensors(cam: Camera, device, dtype):
    """``(R, origin)`` for ``Camera.project``. Keyed on the camera, which is a frozen dataclass."""
    key = (cam, str(device), dtype)
    hit = _PROJ_CACHE.get(key)
    if hit is None:
        r, u, f = cam.basis()
        hit = _PROJ_CACHE.put(
            key,
            (
                torch.tensor(np.stack([r, u, f]), device=device, dtype=dtype),
                torch.tensor(cam.pos, device=device, dtype=dtype),
            ),
        )
    return hit


def _corner_masks(device):
    """The eight ``(1,1,3)`` bool patterns that pick hi/lo per axis to enumerate an AABB's corners."""
    key = str(device)
    hit = _CORNER_CACHE.get(key)
    if hit is None:
        hit = _CORNER_CACHE.put(
            key, [torch.tensor([bool(m & 1), bool(m & 2), bool(m & 4)], device=device).view(1, 1, 3) for m in range(8)]
        )
    return hit


@dataclass
class Lights:
    """Directional lights, plus the material/exponent constants of the shading law.

    ``dirs`` point TOWARDS the light. All OGBench maze lights are grey, so the per-light weights are
    scalars; ``spec_scale`` is the material specular (0.5 for every material in these scenes) folded in
    by the caller if it varies per primitive.
    """

    dirs: np.ndarray  # (L,3), unit, toward the light
    ambient: np.ndarray  # (L,)
    diffuse: np.ndarray  # (L,)
    specular: np.ndarray  # (L,)
    exponent: float

    def to(self, device, dtype=torch.float32):
        t = lambda x: torch.tensor(np.asarray(x, np.float64), device=device, dtype=dtype)  # noqa: E731
        return (t(self.dirs), t(self.ambient), t(self.diffuse), t(self.specular))


# --------------------------------------------------------------------------- frame changes
# Every primitive except the sphere works in its own frame, so each one starts by rotating the ray into
# that frame and ends by rotating the normal back out. Both are 3x3 matrix-vector products, and how they
# are SPELLED decides whether the renderer is batch-size reproducible: ``_mv``/``_mtv``/``_dot`` are
# ``primitives.mv3``/``mtv3``/``dot3``, written out as nine multiplies and six adds rather than as an
# ``einsum``, and that module's note carries the measurement (a cuBLAS kernel chosen from the batch
# COUNT moves ``M @ v`` by an ulp, the ray quadratic amplifies it by ~1e3-1e4, and ``disc > 0`` is a
# HIT/MISS decision that then flips 8-44 pixels a frame). They are imported rather than defined here
# because ``cpu3d`` needs the same property from the same expression.


# --------------------------------------------------------------------------- intersections
# Each returns (t, normal) with t = INF where the ray misses. Rays are (..., 3); the primitive pose is
# broadcast against them. Normals are world-space and unit.


def _sphere(o, d, pos, r):
    oc = o - pos
    b = _dot(oc, d)
    c = _dot(oc, oc) - r * r
    disc = b * b - c
    sq = torch.sqrt(torch.clamp(disc, min=0))
    t = -b - sq
    t = torch.where(t > 0, t, -b + sq)
    hit = (disc > 0) & (t > 0)
    t = torch.where(hit, t, torch.full_like(t, INF))
    n = (oc + t.unsqueeze(-1).clamp(max=1e6) * d) / r.unsqueeze(-1)
    return t, n


def _ellipsoid(o, d, pos, R, size):
    """Ray / ellipsoid with semi-axes ``size`` (...,3), closed form, in the geom frame.

    Scale the ray into the frame where the ellipsoid is the unit sphere -- ``ol' = ol / size``,
    ``dl' = dl / size`` componentwise -- and it is the sphere quadratic with a leading coefficient:
    ``|dl'|^2 t^2 + 2 (ol'.dl') t + |ol'|^2 - 1 = 0``, nearer positive root first, ``disc > 0`` and
    ``t > 0`` as ``_sphere``. The normal is the gradient of ``sum (x_i / s_i)^2`` at the hit,
    ``p_l / size^2``, normalised and rotated back. With three equal semi-axes it is the sphere.
    A zero semi-axis is a miss rather than a division: MuJoCo compiles none, and the fuzz feeds
    them. Same association order as ``_sphere`` and the same ``_mtv``/``_mv`` frame change as
    ``_capsule``, for the reason ``_mv`` gives; the fused twin in ``_tri_isect`` is line for line.
    """
    ol = _mtv(R, o - pos)
    dl = _mtv(R, d)
    inv = 1.0 / torch.clamp(size, min=1e-12)
    olx = ol[..., 0] * inv[..., 0]
    oly = ol[..., 1] * inv[..., 1]
    olz = ol[..., 2] * inv[..., 2]
    dlx = dl[..., 0] * inv[..., 0]
    dly = dl[..., 1] * inv[..., 1]
    dlz = dl[..., 2] * inv[..., 2]
    a = dlx * dlx + dly * dly + dlz * dlz
    b = olx * dlx + oly * dly + olz * dlz
    c = olx * olx + oly * oly + olz * olz - 1.0
    disc = b * b - a * c
    sq = torch.sqrt(torch.clamp(disc, min=0))
    a_safe = torch.clamp(a, min=1e-12)
    t = (-b - sq) / a_safe
    t = torch.where(t > 0, t, (-b + sq) / a_safe)
    hit = (disc > 0) & (t > 0) & (a > 1e-12) & (size[..., 0] > 0) & (size[..., 1] > 0) & (size[..., 2] > 0)
    t = torch.where(hit, t, torch.full_like(t, INF))
    pl = ol + t.unsqueeze(-1).clamp(max=1e6) * dl
    nl = pl * inv * inv
    nl = nl / torch.clamp(torch.sqrt(_dot(nl, nl)).unsqueeze(-1), min=1e-12)
    return t, _mv(R, nl)


def _capsule(o, d, pos, R, r, hh):
    """Z-aligned capsule in the geom frame: a cylinder of half-length ``hh`` capped by two spheres."""
    ol = _mtv(R, o - pos)
    dl = _mtv(R, d)
    a = dl[..., 0] ** 2 + dl[..., 1] ** 2
    b = ol[..., 0] * dl[..., 0] + ol[..., 1] * dl[..., 1]
    c = ol[..., 0] ** 2 + ol[..., 1] ** 2 - r * r
    disc = b * b - a * c
    sq = torch.sqrt(torch.clamp(disc, min=0))
    best = torch.full_like(a, INF)
    for sgn in (-1.0, 1.0):
        t = (-b + sgn * sq) / torch.clamp(a, min=1e-12)
        z = ol[..., 2] + t * dl[..., 2]
        ok = (disc > 0) & (a > 1e-12) & (t > 0) & (z.abs() <= hh)
        best = torch.where(ok & (t < best), t, best)
    for sz in (-1.0, 1.0):
        zc = sz * hh
        oc = ol - torch.stack(
            [torch.zeros_like(ol[..., 0]), torch.zeros_like(ol[..., 0]), torch.zeros_like(ol[..., 0]) + zc], -1
        )
        bb = _dot(oc, dl)
        cc = _dot(oc, oc) - r * r
        dd = bb * bb - cc
        s2 = torch.sqrt(torch.clamp(dd, min=0))
        for sgn in (-1.0, 1.0):
            t = -bb + sgn * s2
            ok = (dd > 0) & (t > 0)
            best = torch.where(ok & (t < best), t, best)
    pl = ol + best.unsqueeze(-1).clamp(max=1e6) * dl
    zc = torch.maximum(torch.minimum(pl[..., 2], hh), -hh)
    nl = pl - torch.stack([torch.zeros_like(zc), torch.zeros_like(zc), zc], -1)
    nl = nl / torch.clamp(torch.sqrt(_dot(nl, nl)).unsqueeze(-1), min=1e-12)
    return best, _mv(R, nl)


# capsule normals use the clamped-axis form, which is the same expression on the barrel and on both
# caps: a point beyond an end measures to that end's centre, which is exactly the hemisphere normal.


def _cylinder(o, d, pos, R, r, hh):
    ol = _mtv(R, o - pos)
    dl = _mtv(R, d)
    a = dl[..., 0] ** 2 + dl[..., 1] ** 2
    b = ol[..., 0] * dl[..., 0] + ol[..., 1] * dl[..., 1]
    c = ol[..., 0] ** 2 + ol[..., 1] ** 2 - r * r
    disc = b * b - a * c
    sq = torch.sqrt(torch.clamp(disc, min=0))
    best = torch.full_like(a, INF)
    side = torch.zeros_like(a, dtype=torch.bool)
    for sgn in (-1.0, 1.0):
        t = (-b + sgn * sq) / torch.clamp(a, min=1e-12)
        z = ol[..., 2] + t * dl[..., 2]
        ok = (disc > 0) & (a > 1e-12) & (t > 0) & (z.abs() <= hh)
        upd = ok & (t < best)
        best = torch.where(upd, t, best)
        side = side | upd
    capz = torch.zeros_like(a)
    for sgn in (-1.0, 1.0):
        zc = sgn * hh
        t = (zc - ol[..., 2]) / torch.where(dl[..., 2].abs() > 1e-12, dl[..., 2], torch.full_like(dl[..., 2], 1e-12))
        px = ol[..., 0] + t * dl[..., 0]
        py = ol[..., 1] + t * dl[..., 1]
        ok = (t > 0) & (px * px + py * py <= r * r) & (dl[..., 2].abs() > 1e-12)
        upd = ok & (t < best)
        best = torch.where(upd, t, best)
        side = side & ~upd
        capz = torch.where(upd, torch.full_like(capz, sgn), capz)
    pl = ol + best.unsqueeze(-1).clamp(max=1e6) * dl
    nl_side = torch.stack([pl[..., 0], pl[..., 1], torch.zeros_like(pl[..., 0])], -1) / r.unsqueeze(-1)
    nl_cap = torch.stack([torch.zeros_like(capz), torch.zeros_like(capz), capz], -1)
    nl = torch.where(side.unsqueeze(-1), nl_side, nl_cap)
    return best, _mv(R, nl)


def _box(o, d, pos, R, size):
    ol = _mtv(R, o - pos)
    dl = _mtv(R, d)
    inv = 1.0 / torch.where(dl.abs() < 1e-12, torch.full_like(dl, 1e-12), dl)
    t1 = (-size - ol) * inv
    t2 = (size - ol) * inv
    tmin = torch.minimum(t1, t2).max(-1).values
    tmax = torch.maximum(t1, t2).min(-1).values
    hit = tmax >= torch.clamp(tmin, min=0.0)
    t = torch.where(tmin > 0, tmin, tmax)
    t = torch.where(hit & (t > 0), t, torch.full_like(t, INF))
    pl = ol + t.unsqueeze(-1).clamp(max=1e6) * dl
    rel = pl.abs() / torch.clamp(size, min=1e-12)
    ax = rel.argmax(-1, keepdim=True)
    nl = torch.zeros_like(pl).scatter_(-1, ax, torch.sign(pl.gather(-1, ax)))
    return t, _mv(R, nl)


def _plane(o, d, pos, R, size):
    n = R[..., :, 2]
    den = _dot(d, n)
    num = _dot(pos - o, n)
    t = torch.where(
        den.abs() > 1e-12,
        num / torch.where(den.abs() > 1e-12, den, torch.full_like(den, 1.0)),
        torch.full_like(den, INF),
    )
    p = o + t.unsqueeze(-1).clamp(max=1e6) * d
    loc = _mtv(R, p - pos)
    ok = (t > 0) & (loc[..., 0].abs() <= size[..., 0]) & (loc[..., 1].abs() <= size[..., 1])
    return torch.where(ok, t, torch.full_like(t, INF)), n.expand_as(p)


def intersect(
    kind: int, size: torch.Tensor, pos: torch.Tensor, R: torch.Tensor, o: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """One primitive against a ray bundle. ``pos`` (...,3), ``R`` (...,3,3), ``o``/``d`` (...,3)."""
    if kind == SPHERE:
        return _sphere(o, d, pos, size[..., 0:1].squeeze(-1))
    if kind == CAPSULE:
        return _capsule(o, d, pos, R, size[..., 0:1].squeeze(-1), size[..., 1:2].squeeze(-1))
    if kind == CYLINDER:
        return _cylinder(o, d, pos, R, size[..., 0:1].squeeze(-1), size[..., 1:2].squeeze(-1))
    if kind == BOX:
        return _box(o, d, pos, R, size)
    if kind == ELLIPSOID:
        return _ellipsoid(o, d, pos, R, size)
    if kind == PLANE:
        return _plane(o, d, pos, R, size)
    raise ValueError(f"raycast3d has no intersector for mjtGeom {kind}; scenes here are primitive-only")


# --------------------------------------------------------------------------- batched primitive casts
#
# ``intersect`` evaluates ONE primitive against a ray bundle, so a caller with K primitives writes a
# Python loop and pays K passes of ~15-50 elementwise kernels over the WHOLE ray grid. That is the
# shape of the cost, and it is memory traffic rather than arithmetic: at 256 px, ss=2 and a 32-frame
# chunk one ``(B,N)`` intermediate is 33 MB, ``_capsule`` holds around forty of them, and the answer
# the caller keeps is one float and one 3-vector per ray. Measured at 256 px / batch 1024, the four
# ray passes of a DM Control frame are 80-92% of the render and the total scales linearly with the
# primitive count: 834 ms at P=1 (point_mass) through 7,550 ms at P=7 (finger).
#
# Two reorganisations of the SAME tensor program were tried and neither moves it, which is what
# points at the program rather than at its schedule:
#
#   * **Give the primitives their own axis.** Intermediates become ``(B,K,N,3)`` -- 5.6 GB per tensor
#     at batch 256, 256 px, K=28 -- so it OOMs long before it helps.
#   * **Batch over primitives and TILE over rays** under a fixed element budget, which fits in memory
#     and was measured at exactly zero speed-up: the traffic per element is unchanged and that is what
#     the passes are bound by. It was reverted, and this loop is what it replaced.
#
# So the loop below is the REFERENCE, kept for the reason ``raster2d.render_torch`` is kept: it is
# short enough to read as a definition of what a cast is, it runs where Triton does not, and the tests
# diff the fused kernel against it. The fused kernel is the one that removes the traffic, by keeping
# the whole primitive loop in registers and writing only the answer.


def _cast_nearest_torch_unbounded(
    kinds: Kinds, sizes: torch.Tensor, pos: torch.Tensor, R: torch.Tensor, o: torch.Tensor, d: torch.Tensor
) -> Cast:
    """``cast_nearest_torch`` with every primitive tested against every ray. THE DEFINITION.

    Kept verbatim as the thing ``PRIMITIVE_BOUNDS`` has to be bit-identical to, for the reason
    ``raster2d.render_torch`` is kept: a fast path is admissible only against a slow one somebody can
    read, and ``tests/gpu/test_fast_paths.py`` diffs the two on the envs themselves.
    """
    B, N = o.shape[0], o.shape[1]
    best = torch.full((B, N), INF, device=o.device, dtype=o.dtype)
    nrm = torch.zeros_like(o)
    idx = torch.zeros((B, N), device=o.device, dtype=torch.long)
    for k, kind in enumerate(int(x) for x in kinds):
        t, n = intersect(kind, sizes[:, k : k + 1], pos[:, k : k + 1], R[:, k : k + 1], o, d)
        upd = t < best
        best = torch.where(upd, t, best)
        nrm = torch.where(upd.unsqueeze(-1), n.expand(B, N, 3), nrm)
        idx = torch.where(upd, torch.full_like(idx, k), idx)
    return best, nrm, idx


#: Test a primitive only against the rays whose line passes within its own bounding sphere, instead of
#: against every ray in the bundle. ``False`` restores ``_cast_nearest_torch_unbounded`` /
#: ``_any_blocking_torch_unbounded`` verbatim, which is what the differential tests diff against.
#:
#: WHY THIS AND NOT A SCREEN BOX. The cast is handed ``o``/``d`` and nothing else -- no camera, no
#: pixel indices, and for the reflection pass the origins are per-ray floor points rather than one
#: pinhole -- so a screen-space box is not computable here without threading a camera through five
#: call sites. The ray/sphere distance test needs none of that and costs ~8 elementwise ops against
#: the ~200 an intersector costs.
#:
#: WHAT THE SKIP CLAIMS, STATED HONESTLY. A ray outside the inflated bound misses the primitive
#: **within the intersector's own rounding budget** -- ``k = 16`` ulp, the ``_BOUND_SLACK_NOISE``
#: term below -- not in exact arithmetic. The distinction is not pedantry: ``_capsule`` in float32 at
#: 100 m reports hits OUTSIDE the primitive's true bound (measured, below), so "provably misses" is
#: a claim about geometry that this intersector does not honour. What is actually established is
#: that over the 21 live configs the two paths produce bit-identical frames on both devices, and
#: that the differential fuzz finds no disagreement on adversarial scenes. Baseline for every such
#: replay in this workstream: ``89fa5ca``.
#:
#: THIS IS THE CPU PATH ONLY. ``_launch``'s fused Triton kernel keeps every primitive in registers and
#: has nothing to gain from skipping one; it is untouched, which is also why the GPU digests cannot
#: move. ``nonzero`` synchronises on CUDA, so the torch fallback on a CUDA device pays K syncs per
#: call -- that path exists for the tests that diff the two backends, not for throughput.
#: DEFAULT ON, and it was off for the whole of RAYCAST-5 until a sweep said it could come back.
#:
#: RAYCAST-4 shipped this bound on for every kind. It gained 3.08x on ``ball_in_cup`` and cost
#: ``PointmazeLargeNavigate`` 8 % and ``PointmazeMediumNavigate`` 25 % -- a regression on a row with a
#: frozen CPU certification -- so it was turned off and left off while RAYCAST-5 found out
#: why. The repair is ``_BOUNDED_KINDS`` below plus the call-level short circuit in the two casts.
#:
#: TURNED BACK ON against a measured bar, not a hoped-for one (job 6354150, four INTERLEAVED legs,
#: off/on/off/on, one node, 15 repeats): every row clears 0.97x at every certifiable batch, the maze
#: rows return to 1.00-1.06x, and the one sub-bar reading -- cartpole at 0.969 -- is at B=1, which the
#: frozen spec forbids certifying on, inside that row's own 5.8 % off-vs-off noise and an 11.7 % worst
#: noise floor. The bar is finer than one thread on a shared node can resolve, and that is recorded
#: rather than smoothed over: see the RAYCAST-5 section of tests/fixtures/regression/join/step6/raycast3d/findings.md.
PRIMITIVE_BOUNDS: bool = True

#: The bound is valid only for UNIT ray directions, and that is a fact about the INTERSECTORS rather
#: than about the bound. ``_sphere`` solves ``t^2 - 2bt + c = 0`` with the leading coefficient dropped,
#: which is the unit-ray quadratic; hand it ``|d| = 2`` and the ``t`` it returns is not the geometric
#: ray parameter, so a geometrically exact bound and the intersector disagree about which rays hit.
#: Every call site in this repo passes normalised directions -- ``cached_rays`` normalises, the
#: reflection of a unit vector about a unit normal is unit, and the shadow rays are normalised at
#: construction -- so this is an invariant, and it is CHECKED rather than assumed: a bundle that
#: violates it takes the unbounded path, the way ``_packed`` routes an exotic layout to the torch loop.
#: The differential fuzz in tests/unit/test_raycast3d_primitive_bounds.py found this by feeding it
#: ``d`` scaled by 0-3x; without the check, 23 of 400 shadow-ray answers moved.
#:
#: THE TOLERANCE IS NOT A COMFORT MARGIN, AND 1e-4 WAS WRONG. Independent review of 3b9e81c found
#: that inside a 1e-4 window the identity this module claims is FALSE, not merely strained: the
#: cap-sphere test drops the leading ``|d|^2``, so at ``|d|^2 = 1 + 9e-5`` and 100 m the reviewer's
#: probe lost 113,008 of 126,184 unbounded hits. They are garbage hits either way -- an intersector
#: handed a non-unit ray is not computing a ray parameter -- but the check is what decides whether a
#: bundle is allowed through, and a bundle it admits must get the SAME garbage from both paths.
#:
#: So the window is sized to the real call sites rather than to a round number: the worst measured
#: ``||d|^2 - 1||`` across DM Control and the Mujoco* families -- camera, reflection and shadow
#: passes -- is **2.4e-7**. 1e-6 leaves ~4x headroom over that and sits at or below
#: ``_BOUND_SLACK_NOISE`` (1.9e-6), which is the term the bound already carries for the
#: intersector's own resolution, so nothing can slip between the two. 1e-4 was ~400x looser than
#: anything that has ever been passed to this function.
_UNIT_D_TOL = 1e-6


def _unit_rays(dd) -> bool:
    return bool(torch.all((dd - 1.0).abs() <= _UNIT_D_TOL))


#: The bound is inflated before it is used to EXCLUDE a ray, by a relative term, an absolute floor,
#: and -- the one that matters -- a term proportional to the DISTANCE to the primitive.
#:
#: THE SCALE TERM IS NOT DEFENSIVE PADDING, IT IS THE FIX FOR A MEASURED FAILURE. The first version of
#: this bound computed the squared perpendicular distance as ``|oc|^2 |d|^2 - (oc.d)^2``. On
#: ``HumanoidmazeGiantNavigate`` -- the largest maze, so the camera is ~100 m from a 5 cm capsule --
#: those two terms are each ~1e4 and differ by ~2.5e-3, which is at float32's resolution there: the
#: subtraction cancels away exactly the quantity it is meant to produce. One ray was pruned that the
#: primitive actually hit, and one pixel of 3,211,264 moved by 22/255 (job 6348207). Twenty of the
#: twenty-one live configs were unaffected, which is what a precision failure that needs a long lever
#: arm looks like.
#:
#: ``|oc x d|`` is the same geometry without the cancellation: its components are differences of
#: products of order ``|oc||d|`` but the RESULT is of order ``r|d|``, so the absolute error is
#: ~``eps*|oc||d|`` rather than ~``eps*|oc|^2|d|^2``. An earlier draft of this comment described a
#: separate ``8*eps*|oc|`` inflation covering that; **no such term exists in the code and none is
#: needed** -- the quadrature term below is larger and subsumes it. The comment is corrected rather
#: than the code, because the code was right and the description was not.
#: AND THE TERM THAT DOES EXIST: the bound has to absorb the INTERSECTOR's own resolution, not its own.
#: ``_capsule`` forms ``c = olx^2 + oly^2 - r^2`` in float32. At ``|ol| = 100`` m the first two terms
#: are ~1e4, where float32's resolution is 1.2e-3 -- and a 2 cm capsule's ``r^2`` is 4e-4, so **the
#: radius is entirely lost** and ``disc > 0`` decides hit-or-miss on rounding noise. Measured: at
#: ``|ol| = 100`` m the shipped intersector reports a hit for a ray whose true perpendicular distance
#: (float64) is 0.0277 m from a capsule whose own bounding radius is 0.0221 m -- **a hit outside the
#: primitive's own bound**. That is a property of the CPU cast as shipped, not of this bound, and it
#: is why ``HumanoidmazeGiantNavigate`` -- the largest maze, so the longest lever arm -- was the one
#: configuration of twenty-one whose pixel moved.
#:
#: A bound that is geometrically exact is therefore NOT conservative against that intersector. The
#: uncertainty in the effective radius is ``|ol| * sqrt(eps)`` (0.0345 m at 100 m), so it is added in
#: QUADRATURE -- invisible when ``r >> |ol| sqrt(eps)``, which is every DM Control cell, and dominant
#: only where the intersector has already stopped being able to tell. It costs almost no pruning:
#: at 100 m the effective radius becomes 0.14 m, which is well under a pixel on a giant maze.
_BOUND_SLACK_REL, _BOUND_SLACK_ABS = 1.01, 1e-6
_BOUND_SLACK_NOISE = 16.0 * 1.1920929e-7  # (|oc| * sqrt(this))^2, added to r^2


#: WHICH KINDS THE BOUND IS APPLIED TO, and this table is a measurement rather than a preference.
#:
#: For one primitive over ``N`` rays the two paths cost
#:
#:     unbounded:  cost_intersect(kind) * N
#:     bounded:    cost_bound * N  +  cost_intersect(kind) * N * p        (p = fraction kept)
#:
#: so the bound pays iff ``cost_bound < cost_intersect(kind) * (1 - p)``. ``p`` is a property of the
#: scene and is not knowable here -- but ``p >= 0``, so the condition that must hold for EVERY scene,
#: batch and resolution is simply ``cost_bound < cost_intersect(kind)``. A kind whose intersector is
#: cheaper than the bound's own mask + ``nonzero`` + gather + scatter can never pay: there is nothing
#: left to save even if the bound prunes perfectly. That ratio is a fact about the intersector's
#: arithmetic, not about the eight rows anybody happened to sweep.
#:
#: THIS IS WHY POINTMAZE REGRESSED, and the census is measured rather than assumed
#: (``experiments/r5_mover_kinds.json``): all four Pointmaze configs cast **exactly one primitive, a
#: SPHERE**, over a 200,704-ray bundle, so the bound spent an O(N) ``nonzero`` + gather + scatter to
#: narrow the cheapest intersector in the file. Under this table they take the unbounded body and the
#: bound becomes a no-op for them. ``ball_in_cup`` is the opposite -- 33 CAPSULES and 3 spheres, so
#: 33 of its 36 primitives stay bounded -- and DM Control generally keeps most of its own: cartpole
#: 2/2, finger_spin 7/7, reacher 2/4, acrobot 3/4. The maze families keep 12/13 (antmaze) and 16/19
#: (humanoidmaze).
#:
#: MEASURED by the frozen measurement record, THREE independent runs in one job (6356311,
#: node1843, Xeon 8268, 1 thread) over B in {1,16,64} x N in {7056, 50176, 200704}. The bound is timed
#: at its worst case -- N-1 rays kept, so the gather and the scatter run at essentially full width --
#: in the ``cast_nearest`` shape, which is the branch that decides this table: it gathers ``best`` and
#: scatters ``best``/``nrm``/``idx``, and pricing the cheaper ``any_blocking`` shape instead (as the
#: first version did) understates the bound by about 2x. ``ns_intersect / ns_bound``:
#:
#:     kind       worst over widths, per run   best     verdict
#:     SPHERE     0.27  0.28  0.28             0.46     never reaches 1: can NEVER pay
#:     BOX        1.13  1.11  1.13                      above 1 in every run
#:     CYLINDER   1.24  1.23  1.24                      above 1 in every run
#:     CAPSULE    1.48  1.48  1.47                      above 1 in every run
#:
#: THE RULE IS THE NECESSARY CONDITION, NOT A MARGIN, and that is a correction. The first version used
#: ``1 + 3*worst_rsd`` -- three times the spread of an INDIVIDUAL timing -- which is not the statistic
#: the decision consumes. Run on a busier node it moved from 1.23 to 1.75 and would have un-bounded
#: every kind: the three runs above, whose ratios agree to within 1-3 %, carry three DIFFERENT
#: predicates in their own ``predicate`` fields for exactly that reason. A threshold that a node's
#: load average can flip is not a derived constant. What stands instead is ``ratio > 1``, which is the
#: condition derived above and needs no constant at all; the spread of the ratio across runs is 1-3 %,
#: which is the error bar that matters and is nowhere near the decision boundary.
#:
#: The separation survives the re-pricing: the dearest unbounded kind reaches 0.46 at its best, the
#: cheapest bounded kind never falls below 1.11, and the line at 1.0 sits inside that 2.4x gap. (The
#: earlier note claimed a ">2x gap between 0.83 and 1.84" about where a THRESHOLD sat; that framing
#: is retired along with the threshold.) Numbers in
#: ``tests/fixtures/regression/join/step6/raycast3d/experiments/r6_kind_cost.json``, which supersedes the
#: ``r5_kind_cost.json`` of the first attempt;
#: ``test_the_bounded_kinds_are_the_ones_the_committed_MEASUREMENT_chose`` keeps this set, that file
#: and the rule behind it from drifting apart.
_BOUNDED_KINDS = frozenset({CAPSULE, CYLINDER, BOX})


def _kind_is_bounded(kind: int) -> bool:
    """Is the bound worth its own cost for this primitive kind? See ``_BOUNDED_KINDS``."""
    return kind in _BOUNDED_KINDS


# ``_bound_radius`` is ``primitives.bound_radius_tensor``, imported at the top: the radius the
# ray bound uses is the same geometric law as the one ``Primitives.bound_radius`` bakes for the
# screen tile and ``cpu3d`` uses for its mover window, and the three must not drift apart.


def _rays_near(
    pos_k: torch.Tensor, rad: torch.Tensor, o: torch.Tensor, d: torch.Tensor, dd: torch.Tensor
) -> torch.Tensor:
    """``(B,N)`` bool: could this ray hit a sphere of radius ``rad`` about ``pos_k`` at ``t > 0``?

    ``|oc x d|^2 <= r^2 |d|^2`` is the perpendicular-distance test with ``|d|^2`` multiplied through,
    so a degenerate ``d = 0`` reads ``0 <= 0`` and INCLUDES the ray rather than producing a NaN that
    would silently exclude it. The cross product rather than ``|oc|^2|d|^2 - (oc.d)^2`` because the
    latter cancels catastrophically when the primitive is far away, and ``_BOUND_SLACK_NOISE`` on top
    because the INTERSECTOR cancels there too -- both measured, both at ``_BOUND_SLACK_REL`` above.
    The second term drops rays whose whole bound lies behind the origin, where ``intersect``'s own
    ``t > 0`` would reject them anyway.
    """
    if o.device.type == "cpu":
        # Separate components avoid packed (B,N,3) offset/cross temporaries and
        # strided reads in the following dot products. Keep the original
        # arithmetic order; CUDA retains its existing expression below.
        x, y, z = (pos_k[..., j] - o[..., j] for j in range(3))
        dx, dy, dz = (d[..., j] for j in range(3))
        cx, cy, cz = y * dz - z * dy, z * dx - x * dz, x * dy - y * dx
        r0 = rad * _BOUND_SLACK_REL + _BOUND_SLACK_ABS
        r2 = r0 * r0 + _BOUND_SLACK_NOISE * (x * x + y * y + z * z)
        return ((cx * cx + cy * cy + cz * cz) <= r2 * dd) & (
            x * dx + y * dy + z * dz + torch.sqrt(r2 * dd) >= 0
        )
    oc = pos_k - o
    cr = torch.cross(oc.expand_as(d), d, dim=-1)
    r0 = rad * _BOUND_SLACK_REL + _BOUND_SLACK_ABS
    r2 = r0 * r0 + _BOUND_SLACK_NOISE * _dot(oc, oc)
    return (_dot(cr, cr) <= r2 * dd) & (_dot(oc, d) + torch.sqrt(r2 * dd) >= 0)


def cast_nearest_torch(
    kinds: Kinds, sizes: torch.Tensor, pos: torch.Tensor, R: torch.Tensor, o: torch.Tensor, d: torch.Tensor
) -> Cast:
    """Nearest primitive hit over a ray bundle: ``(t, normal, index)``. The readable definition.

    ``kinds`` (K,) host-side ints; ``sizes`` (B,K,S), ``pos`` (B,K,3), ``R`` (B,K,3,3); ``o``/``d``
    (B,N,3). Returns ``t`` (B,N), world-space unit ``normal`` (B,N,3) and the winning primitive's index
    (B,N), with ``t = INF`` where nothing is hit.

    The nearest-hit reduction is a per-primitive ``t < best`` in primitive ORDER, not a ``torch.min``
    over a primitive axis: ``min`` does not promise the lowest index on a tie and this does, which is
    what makes two primitives touching at a pixel resolve the same way in every implementation of it.

    WHAT THE BOUND BUYS, AND WHY IT IS THE IDENTITY. Each primitive used to be evaluated against every
    ray in the bundle: measured on the certification host, ``DMCReacherHard`` issued 80,884
    ray-primitive tests for a 7,056-pixel frame -- **11.5 tests per pixel for a four-primitive scene**
    -- because the four ray passes each run over a window that is 64-95% of the frame. The window is a
    rectangle around ALL the movers; a single capsule covers a small part of it.

    A ray whose line passes further from a primitive's origin than that primitive's bounding radius
    cannot intersect it, so ``intersect`` would return ``INF`` there and ``t < best`` would be False.
    Skipping it therefore cannot change ``best``, ``nrm`` or ``idx`` -- not approximately, at all. The
    set is the UNION over the batch of the rays near the primitive in ANY row, so the argument holds
    row by row even though the primitive moves per row. Unbounded kinds (the plane) keep the full
    bundle.
    """
    if not PRIMITIVE_BOUNDS:
        return _cast_nearest_torch_unbounded(kinds, sizes, pos, R, o, d)
    # NOTHING IN THIS CALL IS BOUNDED -> there is no bound to take, so take the definition. This is
    # the SAME table as below, read once at the call instead of once per primitive; it adds no
    # constant and no second predicate. Without it a scene of unbounded kinds still pays the
    # preamble -- ``_dot(d, d)`` over the whole bundle plus a full reduction in ``_unit_rays`` -- for
    # a loop that then bounds nothing. On the Pointmaze rows (one SPHERE, 200,704 rays, ss=2) that
    # preamble alone measured 0.91-0.98x with the ray-primitive test COUNT identical in both legs,
    # which is what identified it: identical counts mean identical work in the loop, so the whole
    # difference was arithmetic done before it.
    if not any(_kind_is_bounded(int(x)) for x in kinds):
        return _cast_nearest_torch_unbounded(kinds, sizes, pos, R, o, d)
    B, N = o.shape[0], o.shape[1]
    best = torch.full((B, N), INF, device=o.device, dtype=o.dtype)
    # (B,N,3) explicitly rather than ``zeros_like(o)``: ``o`` arrives expanded from one origin, and
    # this one is written into by index below rather than rebuilt by ``where``. Same values.
    nrm = torch.zeros((B, N, 3), device=o.device, dtype=o.dtype)
    idx = torch.zeros((B, N), device=o.device, dtype=torch.long)
    dd = _dot(d, d)
    if not _unit_rays(dd):
        return _cast_nearest_torch_unbounded(kinds, sizes, pos, R, o, d)
    for k, kind in enumerate(int(x) for x in kinds):
        js = None
        rad = _bound_radius(kind, sizes[:, k : k + 1]) if _kind_is_bounded(kind) else None
        if rad is not None:
            js = torch.nonzero(_rays_near(pos[:, k : k + 1], rad, o, d, dd).any(0), as_tuple=False).reshape(-1)
            if js.numel() == 0:
                continue
            if js.numel() == N:  # nothing pruned; the gather would be pure overhead
                js = None
        oo, ddir = (o, d) if js is None else (o[:, js], d[:, js])
        t, n = intersect(kind, sizes[:, k : k + 1], pos[:, k : k + 1], R[:, k : k + 1], oo, ddir)
        if js is None:
            upd = t < best
            best = torch.where(upd, t, best)
            nrm = torch.where(upd.unsqueeze(-1), n.expand(B, N, 3), nrm)
            idx = torch.where(upd, torch.full_like(idx, k), idx)
        else:
            bs = best[:, js]
            upd = t < bs
            best[:, js] = torch.where(upd, t, bs)
            nrm[:, js] = torch.where(upd.unsqueeze(-1), n.expand(B, js.numel(), 3), nrm[:, js])
            idx[:, js] = torch.where(upd, k, idx[:, js])
    return best, nrm, idx


def _any_blocking_torch_unbounded(
    kinds: Kinds,
    sizes: torch.Tensor,
    pos: torch.Tensor,
    R: torch.Tensor,
    o: torch.Tensor,
    d: torch.Tensor,
    lim: torch.Tensor,
    skip: torch.Tensor | None = None,
) -> torch.Tensor:
    """``any_blocking_torch`` with every primitive tested against every ray. THE DEFINITION."""
    B, N = o.shape[0], o.shape[1]
    blocked = torch.zeros((B, N), device=o.device, dtype=torch.bool)
    for k, kind in enumerate(int(x) for x in kinds):
        t, _ = intersect(kind, sizes[:, k : k + 1], pos[:, k : k + 1], R[:, k : k + 1], o, d)
        hit = (t > 0) & (t < lim)
        if skip is not None:
            hit = hit & (skip != k)
        blocked |= hit
    return blocked


def any_blocking_torch(
    kinds: Kinds,
    sizes: torch.Tensor,
    pos: torch.Tensor,
    R: torch.Tensor,
    o: torch.Tensor,
    d: torch.Tensor,
    lim: torch.Tensor,
    skip: torch.Tensor | None = None,
) -> torch.Tensor:
    """Does any primitive block each ray before ``lim``? ``(B,N)`` bool. Shadow-ray form of the above.

    Same bound and the same argument: a ray outside a primitive's bounding sphere gets ``t = INF``,
    so ``hit`` is False and ``blocked |= hit`` leaves it alone.
    """
    if not PRIMITIVE_BOUNDS:
        return _any_blocking_torch_unbounded(kinds, sizes, pos, R, o, d, lim, skip)
    # NOTHING IN THIS CALL IS BOUNDED -> there is no bound to take, so take the definition. This is
    # the SAME table as below, read once at the call instead of once per primitive; it adds no
    # constant and no second predicate. Without it a scene of unbounded kinds still pays the
    # preamble -- ``_dot(d, d)`` over the whole bundle plus a full reduction in ``_unit_rays`` -- for
    # a loop that then bounds nothing. On the Pointmaze rows (one SPHERE, 200,704 rays, ss=2) that
    # preamble alone measured 0.91-0.98x with the ray-primitive test COUNT identical in both legs,
    # which is what identified it: identical counts mean identical work in the loop, so the whole
    # difference was arithmetic done before it.
    if not any(_kind_is_bounded(int(x)) for x in kinds):
        return _any_blocking_torch_unbounded(kinds, sizes, pos, R, o, d, lim, skip)
    B, N = o.shape[0], o.shape[1]
    blocked = torch.zeros((B, N), device=o.device, dtype=torch.bool)
    dd = _dot(d, d)
    if not _unit_rays(dd):
        return _any_blocking_torch_unbounded(kinds, sizes, pos, R, o, d, lim, skip)
    for k, kind in enumerate(int(x) for x in kinds):
        js = None
        rad = _bound_radius(kind, sizes[:, k : k + 1]) if _kind_is_bounded(kind) else None
        if rad is not None:
            js = torch.nonzero(_rays_near(pos[:, k : k + 1], rad, o, d, dd).any(0), as_tuple=False).reshape(-1)
            if js.numel() == 0:
                continue
            if js.numel() == N:
                js = None
        if js is None:
            t, _ = intersect(kind, sizes[:, k : k + 1], pos[:, k : k + 1], R[:, k : k + 1], o, d)
            hit = (t > 0) & (t < lim)
            if skip is not None:
                hit = hit & (skip != k)
            blocked |= hit
        else:
            t, _ = intersect(kind, sizes[:, k : k + 1], pos[:, k : k + 1], R[:, k : k + 1], o[:, js], d[:, js])
            hit = (t > 0) & (t < lim[:, js])
            if skip is not None:
                hit = hit & (skip[:, js] != k)
            blocked[:, js] |= hit
    return blocked


# --------------------------------------------------------------------------- the fused cast
#
# One kernel, the primitive loop inside it, nothing materialized but the answer. This is the same
# argument ``raster2d`` makes for the 2-D path and it transfers exactly: every quantity between two
# primitives is dead the moment the nearest-hit test has looked at it, so it never needs to reach
# memory at all. The loop is a RUNTIME loop rather than ``tl.static_range`` for the reason recorded in
# ``raster2d``: unrolling collapses past a few dozen primitives and refuses to compile past ~340.
#
# WHAT MAKES IT ADMISSIBLE HERE IS NOT THE SPEED, IT IS THAT IT IS BIT-IDENTICAL. Triton's DEFAULT
# fp32 lowering is not: measured over 4.2 M random inputs against the torch expression it replaces,
#
#     a*b + c                            836,950 / 4,194,304 float32 patterns differ
#     ax*bx + ay*by + az*bz  (``_dot``) 1,492,420 / 4,194,304
#     a / b                             1,249,701 / 4,194,304
#     sqrt(x)                             709,903 / 4,194,304
#
# and one ulp is not a rounding detail in a ray-caster -- the module docstring above measures 1.2e-7
# in ``dl`` amplifying to 2.4e-4 in ``disc = b*b - a*c``, and ``disc > 0`` is a HIT/MISS decision that
# flips whole pixels. Three knobs take every one of those four counts to exactly ZERO, and all three
# are load-bearing:
#
#   * ``enable_fp_fusion=False`` at launch. LLVM contracts ``a*b + c`` into one ``fma.rn.f32``, which
#     rounds once where torch -- two kernels, two roundings -- rounds twice. This is Triton's
#     ``-ffp-contract=off`` and it is the reason the multiply-adds below can be written normally.
#   * ``tl.math.div_rn`` for every division. Plain ``/`` lowers to the non-IEEE ``div.full.f32``.
#   * ``tl.sqrt_rn`` for every root. ``tl.sqrt`` lowers to ``sqrt.approx.f32``.
#
# the frozen measurement record and ``triton_exact2.py`` are those measurements; ``tests/gpu``
# diffs the kernel against ``cast_nearest_torch`` on the envs themselves.

if HAVE_TRITON:

    @triton.jit
    def _tri_isect(
        kind,
        s0,
        s1,
        s2,
        px,
        py,
        pz,
        r00,
        r01,
        r02,
        r10,
        r11,
        r12,
        r20,
        r21,
        r22,
        ox,
        oy,
        oz,
        dx,
        dy,
        dz,
        INF: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """One primitive against a block of rays: ``(t, nx, ny, nz)``, ``t = INF`` on a miss.

        Line for line the same expressions as ``_sphere``/``_capsule``/``_cylinder``/``_box``/``_plane``
        above, in the same association order, with ``div_rn``/``sqrt_rn`` in place of ``/``/``sqrt``.
        Every branch assigns all four outputs because Triton merges a conditional's live values by
        name and a variable the ``else`` leg does not write has no type to merge.
        """
        zero = tl.zeros([BLOCK], dtype=tl.float32)
        never = tl.arange(0, BLOCK) < 0  # an all-False mask
        t = zero + INF
        nx = zero
        ny = zero
        nz = zero

        if kind == 2:  # SPHERE
            r = s0
            ocx = ox - px
            ocy = oy - py
            ocz = oz - pz
            b = ocx * dx + ocy * dy + ocz * dz
            c = ocx * ocx + ocy * ocy + ocz * ocz - r * r
            disc = b * b - c
            sq = tl.sqrt_rn(tl.maximum(disc, 0.0))
            tt = -b - sq
            tt = tl.where(tt > 0, tt, -b + sq)
            hit = (disc > 0) & (tt > 0)
            t = tl.where(hit, tt, INF)
            tc = tl.minimum(t, 1e6)
            nx = tl.math.div_rn(ocx + tc * dx, r)
            ny = tl.math.div_rn(ocy + tc * dy, r)
            nz = tl.math.div_rn(ocz + tc * dz, r)

        elif kind == 3 or kind == 5:  # CAPSULE (3) and CYLINDER (5)
            r = s0
            hh = s1
            ax = ox - px
            ay = oy - py
            az = oz - pz
            olx = r00 * ax + r10 * ay + r20 * az
            oly = r01 * ax + r11 * ay + r21 * az
            olz = r02 * ax + r12 * ay + r22 * az
            dlx = r00 * dx + r10 * dy + r20 * dz
            dly = r01 * dx + r11 * dy + r21 * dz
            dlz = r02 * dx + r12 * dy + r22 * dz
            a = dlx * dlx + dly * dly
            b = olx * dlx + oly * dly
            c = olx * olx + oly * oly - r * r
            disc = b * b - a * c
            sq = tl.sqrt_rn(tl.maximum(disc, 0.0))
            best = zero + INF
            side = never
            # barrel, both roots, in the loop order the torch version walks
            t0 = tl.math.div_rn(-b + -1.0 * sq, tl.maximum(a, 1e-12))
            z0 = olz + t0 * dlz
            ok0 = (disc > 0) & (a > 1e-12) & (t0 > 0) & (tl.abs(z0) <= hh)
            u0 = ok0 & (t0 < best)
            best = tl.where(u0, t0, best)
            side = side | u0
            t1 = tl.math.div_rn(-b + 1.0 * sq, tl.maximum(a, 1e-12))
            z1 = olz + t1 * dlz
            ok1 = (disc > 0) & (a > 1e-12) & (t1 > 0) & (tl.abs(z1) <= hh)
            u1 = ok1 & (t1 < best)
            best = tl.where(u1, t1, best)
            side = side | u1

            capz = zero
            if kind == 3:  # spherical caps
                for sz in tl.static_range(2):
                    zc = (2.0 * sz - 1.0) * hh
                    ocz = olz - zc
                    bb = olx * dlx + oly * dly + ocz * dlz
                    cc = olx * olx + oly * oly + ocz * ocz - r * r
                    dd = bb * bb - cc
                    s2r = tl.sqrt_rn(tl.maximum(dd, 0.0))
                    ta = -bb + -1.0 * s2r
                    best = tl.where((dd > 0) & (ta > 0) & (ta < best), ta, best)
                    tb = -bb + 1.0 * s2r
                    best = tl.where((dd > 0) & (tb > 0) & (tb < best), tb, best)
            else:  # flat caps
                for sz in tl.static_range(2):
                    sgn = 2.0 * sz - 1.0
                    zc = sgn * hh
                    den = tl.where(tl.abs(dlz) > 1e-12, dlz, 1e-12)
                    tc2 = tl.math.div_rn(zc - olz, den)
                    qx = olx + tc2 * dlx
                    qy = oly + tc2 * dly
                    okc = (tc2 > 0) & (qx * qx + qy * qy <= r * r) & (tl.abs(dlz) > 1e-12)
                    upd = okc & (tc2 < best)
                    best = tl.where(upd, tc2, best)
                    side = side & ~upd
                    capz = tl.where(upd, sgn, capz)

            bc = tl.minimum(best, 1e6)
            plx = olx + bc * dlx
            ply = oly + bc * dly
            plz = olz + bc * dlz
            if kind == 3:
                zcl = tl.maximum(tl.minimum(plz, hh), -hh)
                mx = plx
                my = ply
                mz = plz - zcl
                den = tl.maximum(tl.sqrt_rn(mx * mx + my * my + mz * mz), 1e-12)
                nlx = tl.math.div_rn(mx, den)
                nly = tl.math.div_rn(my, den)
                nlz = tl.math.div_rn(mz, den)
            else:
                # ``nl_side``'s third component is ``zeros / r``, i.e. exactly +0.0 for the positive
                # radii a geom can have -- not ``0 * pl_z / r``, whose sign follows ``pl_z``
                nlx = tl.where(side, tl.math.div_rn(plx, r), 0.0)
                nly = tl.where(side, tl.math.div_rn(ply, r), 0.0)
                nlz = tl.where(side, 0.0, capz)
            t = best
            nx = r00 * nlx + r01 * nly + r02 * nlz
            ny = r10 * nlx + r11 * nly + r12 * nlz
            nz = r20 * nlx + r21 * nly + r22 * nlz

        elif kind == 4:  # ELLIPSOID
            # ``_ellipsoid`` line for line: the ray in the geom frame, scaled to the unit sphere
            ax = ox - px
            ay = oy - py
            az = oz - pz
            olx0 = r00 * ax + r10 * ay + r20 * az
            oly0 = r01 * ax + r11 * ay + r21 * az
            olz0 = r02 * ax + r12 * ay + r22 * az
            dlx0 = r00 * dx + r10 * dy + r20 * dz
            dly0 = r01 * dx + r11 * dy + r21 * dz
            dlz0 = r02 * dx + r12 * dy + r22 * dz
            ivx = tl.math.div_rn(1.0, tl.maximum(s0, 1e-12))
            ivy = tl.math.div_rn(1.0, tl.maximum(s1, 1e-12))
            ivz = tl.math.div_rn(1.0, tl.maximum(s2, 1e-12))
            olx = olx0 * ivx
            oly = oly0 * ivy
            olz = olz0 * ivz
            dlx = dlx0 * ivx
            dly = dly0 * ivy
            dlz = dlz0 * ivz
            a = dlx * dlx + dly * dly + dlz * dlz
            b = olx * dlx + oly * dly + olz * dlz
            c = olx * olx + oly * oly + olz * olz - 1.0
            disc = b * b - a * c
            sq = tl.sqrt_rn(tl.maximum(disc, 0.0))
            a_safe = tl.maximum(a, 1e-12)
            tt = tl.math.div_rn(-b - sq, a_safe)
            tt = tl.where(tt > 0, tt, tl.math.div_rn(-b + sq, a_safe))
            hit = (disc > 0) & (tt > 0) & (a > 1e-12) & (s0 > 0) & (s1 > 0) & (s2 > 0)
            t = tl.where(hit, tt, INF)
            tc = tl.minimum(t, 1e6)
            plx = olx0 + tc * dlx0
            ply = oly0 + tc * dly0
            plz = olz0 + tc * dlz0
            gx = plx * ivx * ivx
            gy = ply * ivy * ivy
            gz = plz * ivz * ivz
            den = tl.maximum(tl.sqrt_rn(gx * gx + gy * gy + gz * gz), 1e-12)
            nlx = tl.math.div_rn(gx, den)
            nly = tl.math.div_rn(gy, den)
            nlz = tl.math.div_rn(gz, den)
            nx = r00 * nlx + r01 * nly + r02 * nlz
            ny = r10 * nlx + r11 * nly + r12 * nlz
            nz = r20 * nlx + r21 * nly + r22 * nlz

        elif kind == 6:  # BOX
            ax = ox - px
            ay = oy - py
            az = oz - pz
            olx = r00 * ax + r10 * ay + r20 * az
            oly = r01 * ax + r11 * ay + r21 * az
            olz = r02 * ax + r12 * ay + r22 * az
            dlx = r00 * dx + r10 * dy + r20 * dz
            dly = r01 * dx + r11 * dy + r21 * dz
            dlz = r02 * dx + r12 * dy + r22 * dz
            ivx = tl.math.div_rn(1.0, tl.where(tl.abs(dlx) < 1e-12, 1e-12, dlx))
            ivy = tl.math.div_rn(1.0, tl.where(tl.abs(dly) < 1e-12, 1e-12, dly))
            ivz = tl.math.div_rn(1.0, tl.where(tl.abs(dlz) < 1e-12, 1e-12, dlz))
            ax1 = (-s0 - olx) * ivx
            ax2 = (s0 - olx) * ivx
            ay1 = (-s1 - oly) * ivy
            ay2 = (s1 - oly) * ivy
            az1 = (-s2 - olz) * ivz
            az2 = (s2 - olz) * ivz
            tmin = tl.maximum(tl.maximum(tl.minimum(ax1, ax2), tl.minimum(ay1, ay2)), tl.minimum(az1, az2))
            tmax = tl.minimum(tl.minimum(tl.maximum(ax1, ax2), tl.maximum(ay1, ay2)), tl.maximum(az1, az2))
            hit = tmax >= tl.maximum(tmin, 0.0)
            tt = tl.where(tmin > 0, tmin, tmax)
            t = tl.where(hit & (tt > 0), tt, INF)
            tc = tl.minimum(t, 1e6)
            plx = olx + tc * dlx
            ply = oly + tc * dly
            plz = olz + tc * dlz
            e0 = tl.math.div_rn(tl.abs(plx), tl.maximum(s0, 1e-12))
            e1 = tl.math.div_rn(tl.abs(ply), tl.maximum(s1, 1e-12))
            e2 = tl.math.div_rn(tl.abs(plz), tl.maximum(s2, 1e-12))
            # argmax with the FIRST maximum winning, which is what ``Tensor.argmax`` promises and what
            # decides the normal on an exact corner hit
            m = e0
            c1 = e1 > m
            m = tl.where(c1, e1, m)
            c2 = e2 > m
            sel = tl.where(c2, plz, tl.where(c1, ply, plx))
            sg = tl.where(sel > 0, 1.0, tl.where(sel < 0, -1.0, sel))
            nlx = tl.where((~c1) & (~c2), sg, 0.0)
            nly = tl.where(c1 & (~c2), sg, 0.0)
            nlz = tl.where(c2, sg, 0.0)
            nx = r00 * nlx + r01 * nly + r02 * nlz
            ny = r10 * nlx + r11 * nly + r12 * nlz
            nz = r20 * nlx + r21 * nly + r22 * nlz

        else:  # PLANE
            nx = zero + r02
            ny = zero + r12
            nz = zero + r22
            den = dx * r02 + dy * r12 + dz * r22
            num = (px - ox) * r02 + (py - oy) * r12 + (pz - oz) * r22
            fin = tl.abs(den) > 1e-12
            tt = tl.where(fin, tl.math.div_rn(num, tl.where(fin, den, 1.0)), INF)
            tc = tl.minimum(tt, 1e6)
            qx = ox + tc * dx - px
            qy = oy + tc * dy - py
            qz = oz + tc * dz - pz
            lx = r00 * qx + r10 * qy + r20 * qz
            ly = r01 * qx + r11 * qy + r21 * qz
            t = tl.where((tt > 0) & (tl.abs(lx) <= s0) & (tl.abs(ly) <= s1), tt, INF)

        return t, nx, ny, nz

    @triton.jit
    def _tri_cast(
        kind_ptr,
        sz_ptr,
        po_ptr,
        rm_ptr,
        o_ptr,
        d_ptr,
        lim_ptr,
        skip_ptr,
        t_out,
        n_out,
        i_out,
        b_out,
        K,
        N,
        szb,
        szk,
        pob,
        pok,
        rmb,
        rmk,
        ob,
        on,
        db,
        dn,
        lb,
        ln,
        kb,
        kn,
        MODE: tl.constexpr,
        HAS_SKIP: tl.constexpr,
        INF: tl.constexpr,
        BLOCK: tl.constexpr,
    ):
        """``MODE=0`` nearest hit (t / normal / index); ``MODE=1`` "is anything blocking before lim".

        One program owns ``BLOCK`` rays of one batch element, so every primitive quantity is a SCALAR
        load hoisted out of the ray vector -- which is the whole point, and why the grid is
        ``(B, ceil(N/BLOCK))`` rather than a flat one that would straddle two frames.
        """
        b = tl.program_id(0)
        i = tl.program_id(1) * BLOCK + tl.arange(0, BLOCK)
        m = i < N
        i64 = i.to(tl.int64)
        b64 = b.to(tl.int64)
        oo = b64 * ob + i64 * on
        do = b64 * db + i64 * dn
        ox = tl.load(o_ptr + oo + 0, mask=m, other=0.0)
        oy = tl.load(o_ptr + oo + 1, mask=m, other=0.0)
        oz = tl.load(o_ptr + oo + 2, mask=m, other=0.0)
        dx = tl.load(d_ptr + do + 0, mask=m, other=0.0)
        dy = tl.load(d_ptr + do + 1, mask=m, other=0.0)
        dz = tl.load(d_ptr + do + 2, mask=m, other=0.0)

        best = tl.zeros([BLOCK], dtype=tl.float32) + INF
        bnx = tl.zeros([BLOCK], dtype=tl.float32)
        bny = tl.zeros([BLOCK], dtype=tl.float32)
        bnz = tl.zeros([BLOCK], dtype=tl.float32)
        bidx = tl.zeros([BLOCK], dtype=tl.int64)
        blocked = tl.arange(0, BLOCK) < 0
        if MODE == 1:
            lim = tl.load(lim_ptr + b64 * lb + i64 * ln, mask=m, other=0.0)
            skip = tl.zeros([BLOCK], dtype=tl.int64)
            if HAS_SKIP:
                skip = tl.load(skip_ptr + b64 * kb + i64 * kn, mask=m, other=0)

        for k in range(K):
            k64 = k.to(tl.int64)
            so = b64 * szb + k64 * szk
            po = b64 * pob + k64 * pok
            ro = b64 * rmb + k64 * rmk
            t, nx, ny, nz = _tri_isect(
                tl.load(kind_ptr + k),
                tl.load(sz_ptr + so + 0),
                tl.load(sz_ptr + so + 1),
                tl.load(sz_ptr + so + 2),
                tl.load(po_ptr + po + 0),
                tl.load(po_ptr + po + 1),
                tl.load(po_ptr + po + 2),
                tl.load(rm_ptr + ro + 0),
                tl.load(rm_ptr + ro + 1),
                tl.load(rm_ptr + ro + 2),
                tl.load(rm_ptr + ro + 3),
                tl.load(rm_ptr + ro + 4),
                tl.load(rm_ptr + ro + 5),
                tl.load(rm_ptr + ro + 6),
                tl.load(rm_ptr + ro + 7),
                tl.load(rm_ptr + ro + 8),
                ox,
                oy,
                oz,
                dx,
                dy,
                dz,
                INF=INF,
                BLOCK=BLOCK,
            )
            if MODE == 0:
                upd = t < best
                best = tl.where(upd, t, best)
                bnx = tl.where(upd, nx, bnx)
                bny = tl.where(upd, ny, bny)
                bnz = tl.where(upd, nz, bnz)
                bidx = tl.where(upd, k64, bidx)
            else:
                hit = (t > 0) & (t < lim)
                if HAS_SKIP:
                    hit = hit & (skip != k64)
                blocked = blocked | hit

        flat = b64 * N + i64  # int64 throughout: B*N passes 2^31 at 256 px ss=2
        if MODE == 0:
            po = flat * 3
            tl.store(t_out + flat, best, mask=m)
            tl.store(n_out + po + 0, bnx, mask=m)
            tl.store(n_out + po + 1, bny, mask=m)
            tl.store(n_out + po + 2, bnz, mask=m)
            tl.store(i_out + flat, bidx, mask=m)
        else:
            tl.store(b_out + flat, blocked.to(tl.uint8), mask=m)


#: Rays per program. 256 keeps the per-primitive scalar loads amortised over a warp-multiple vector
#: without spilling the ~30 live registers an intersector holds.
CAST_BLOCK = 256

#: (device, kinds) -> the (K,) int32 the kernel branches on. Cast kinds arrive as a numpy array or a
#: Python list and are constant for the life of a scene, so rebuilding the tensor per call would be a
#: host-to-device copy on the critical path of every pass.
_KIND_CACHE = _Memo(64, name="raycast3d.kind")


def _kind_tensor(kinds, device) -> torch.Tensor:
    key = (str(device), tuple(int(k) for k in kinds))
    hit = _KIND_CACHE.get(key)
    if hit is None:
        # The kernel's last branch IS the plane, so an unrecognised code would be drawn as one
        # instead of raising the way ``intersect`` does. Same error, at the same place, either path.
        bad = sorted({k for k in key[1] if k not in PRIMITIVE_KINDS})
        if bad:
            raise ValueError(f"raycast3d has no intersector for mjtGeom {bad}; scenes here are primitive-only")
        hit = _KIND_CACHE.put(key, torch.tensor(key[1], device=device, dtype=torch.int32))
    return hit


def _packed(*ts) -> bool:
    """Is every tensor's trailing axis (and, for ``R``, its 3x3 block) laid out unit-stride?

    The kernel walks the leading two axes by explicit stride and the rest by ``+1``, so a transposed
    or otherwise exotic layout would be read scrambled. That is a silent wrong picture, so it is a
    CHECK that routes to the torch loop rather than an assumption — no caller in this repo produces
    one, which is exactly why it would go unnoticed if one started to.
    """
    for t in ts:
        if t is None:
            continue
        if t.stride(-1) != 1:
            return False
        if t.dim() == 4 and t.stride(-2) != t.shape[-1]:
            return False
    return True


#: Force every cast onto one backend, for the tests that diff the two. ``None`` means ``auto``. It is
#: a module-level switch rather than a parameter threaded through five renderers because the property
#: under test is a property of the whole render, and a test that had to pass ``backend=`` down every
#: call site would be testing the plumbing it just wrote.
FORCE_BACKEND: str | None = None

#: Shade only the rays that actually landed on a mover -- as the union over the batch -- rather than
#: every ray of the screen window. ``False`` restores the expression this replaced, verbatim, so
#: ``tests/gpu/test_fast_paths.py`` can diff the two on the envs themselves. Same reason
#: ``FORCE_BACKEND`` is a module-level switch: the property under test is a property of the whole
#: composite, not of a helper, and threading a ``compact=`` argument down five call sites would be
#: testing the plumbing rather than the claim. ``DMControlSuite`` carries the identical switch under
#: the identical name for its own path.
COVER_COMPACTION: bool = True

#: ``composite`` sizes its window from the BOUND rather than from the caller's ``tile`` ceiling, and
#: runs its tail over the covered output pixels rather than the whole window. ``False`` restores
#: ``_composite_window``, the expression this replaced, verbatim -- same reason ``COVER_COMPACTION``
#: and ``PRIMITIVE_BOUNDS`` are module-level switches: the property under test is a property of the
#: whole composite, and a ``compact=`` argument threaded down the call sites would be testing the
#: plumbing rather than the claim.
COMPOSITE_COMPACT_MERGE: bool = True


def _use_triton(o) -> bool:
    return HAVE_TRITON and o.is_cuda and o.dtype == torch.float32


def _launch(kinds, sizes, pos, R, o, d, lim, skip, mode):
    """Shared launch for both modes. Strides are passed rather than assumed: the callers hand in
    ``o.expand(B,N,3)`` (stride 0 on two axes) and ``sizes.expand(B,P,3)``, and making those
    contiguous would materialize exactly the (B,N,3) tensors the kernel exists to avoid."""
    B, N = o.shape[0], o.shape[1]
    K = len(kinds)
    dev = o.device
    kt = _kind_tensor(kinds, dev)
    t_out = n_out = i_out = b_out = None
    if mode == 0:
        t_out = torch.empty((B, N), device=dev, dtype=torch.float32)
        n_out = torch.empty((B, N, 3), device=dev, dtype=torch.float32)
        i_out = torch.empty((B, N), device=dev, dtype=torch.int64)
        dummy = t_out
    else:
        b_out = torch.empty((B, N), device=dev, dtype=torch.uint8)
        dummy = b_out
    lim_t = lim if lim is not None else dummy
    skip_t = skip if skip is not None else i_out if i_out is not None else dummy
    ls = lim.stride() if lim is not None else (0, 0)
    ks = skip.stride() if skip is not None else (0, 0)
    grid = (B, triton.cdiv(N, CAST_BLOCK))
    with torch.cuda.device(dev):
        _tri_cast[grid](
            kt,
            sizes,
            pos,
            R,
            o,
            d,
            lim_t,
            skip_t,
            t_out if mode == 0 else dummy,
            n_out if mode == 0 else dummy,
            i_out if mode == 0 else dummy,
            b_out if mode == 1 else dummy,
            K,
            N,
            sizes.stride(0),
            sizes.stride(1),
            pos.stride(0),
            pos.stride(1),
            R.stride(0),
            R.stride(1),
            o.stride(0),
            o.stride(1),
            d.stride(0),
            d.stride(1),
            ls[0],
            ls[1],
            ks[0],
            ks[1],
            MODE=mode,
            HAS_SKIP=skip is not None,
            INF=INF,
            BLOCK=CAST_BLOCK,
            enable_fp_fusion=False,
        )
    return (t_out, n_out, i_out) if mode == 0 else b_out.view(torch.bool)


def cast_nearest(
    kinds: Kinds,
    sizes: torch.Tensor,
    pos: torch.Tensor,
    R: torch.Tensor,
    o: torch.Tensor,
    d: torch.Tensor,
    backend: str = "auto",
) -> Cast:
    """Nearest primitive hit over a ray bundle: ``(t, normal, index)``.

    ``auto`` fuses on CUDA and falls back to ``cast_nearest_torch`` everywhere else. The two are
    bit-identical, which ``tests/gpu/test_fast_paths.py`` asserts on the envs rather than on a
    synthetic scene.
    """
    backend = FORCE_BACKEND or backend
    if backend == "torch" or (backend == "auto" and not (_use_triton(o) and _packed(sizes, pos, R, o, d))):
        return cast_nearest_torch(kinds, sizes, pos, R, o, d)
    return _launch(kinds, sizes, pos, R, o, d, None, None, 0)


def any_blocking(
    kinds: Kinds,
    sizes: torch.Tensor,
    pos: torch.Tensor,
    R: torch.Tensor,
    o: torch.Tensor,
    d: torch.Tensor,
    lim: torch.Tensor,
    skip: torch.Tensor | None = None,
    backend: str = "auto",
) -> torch.Tensor:
    """Does any primitive block each ray before ``lim``? ``(B,N)`` bool. Shadow-ray form of the above."""
    backend = FORCE_BACKEND or backend
    if backend == "torch" or (backend == "auto" and not (_use_triton(o) and _packed(sizes, pos, R, o, d, lim, skip))):
        return any_blocking_torch(kinds, sizes, pos, R, o, d, lim, skip)
    return _launch(kinds, sizes, pos, R, o, d, lim.expand(o.shape[0], o.shape[1]), skip, 1)


#: ``(camera, Rh, Rw, device)`` -> the ray bundle through it. Small, because a bundle is
#: ``Rh*Rw*3`` float32 -- 3.1 MB at 512x512 -- and no caller here has more than a couple of live
#: (camera, resolution) pairs at once.
_RAY_CACHE_MAX = 8
_RAY_CACHE = _Memo(_RAY_CACHE_MAX, name="raycast3d.rays")


def cached_rays(cam: Camera, H: int, W: int, device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """``cam.rays(H, W, device)``, memoised on the camera itself.

    ``Camera`` is a frozen dataclass of tuples and floats, so it is its own cache key and two scenes
    that look through the same camera share one bundle. That matters because ``rays`` is NUMPY work
    in float64 — a meshgrid and half a dozen ``(H,W,3)`` temporaries, then a host-to-device copy —
    and it is a pure function of (camera, resolution): at 256 px and supersample 2 it was being
    rebuilt once per ``composite`` call and once per DM Control CHUNK, i.e. 32 times per 1024 frames,
    to produce the same 3.1 MB every time.
    """
    key = (cam, int(H), int(W), str(device), dtype)
    hit = _RAY_CACHE.get(key)
    if hit is None:
        hit = _RAY_CACHE.put(key, cam.rays(H, W, device, dtype))
    return hit


# --------------------------------------------------------------------------- shading


def shade(
    n: torch.Tensor,
    v: torch.Tensor,
    ldir: torch.Tensor,
    lamb: torch.Tensor,
    ldif: torch.Tensor,
    lspec: torch.Tensor,
    exponent: float,
    mat: torch.Tensor,
    tex: torch.Tensor,
    spec: torch.Tensor,
) -> torch.Tensor:
    """``(...,3)`` colour in [0,1]. See the module docstring for why specular is additive and white."""
    amb = lamb.sum()
    prim = mat * amb
    highlight = torch.zeros(n.shape[:-1], device=n.device, dtype=n.dtype)
    for i in range(ldir.shape[0]):
        l = ldir[i]
        ndl = torch.clamp(_dot(n, l), min=0)
        prim = prim + mat * (ldif[i] * ndl).unsqueeze(-1)
        h = l + v
        h = h / torch.clamp(torch.sqrt(_dot(h, h)).unsqueeze(-1), min=1e-12)
        highlight = highlight + lspec[i] * torch.clamp(_dot(n, h), min=0) ** exponent
    prim = prim + (highlight * spec).unsqueeze(-1)
    return torch.clamp(prim * tex, 0, 1)


# --------------------------------------------------------------------------- the renderer


@torch.no_grad()
def static_depth(cam: Camera, prims: Primitives, xpos, xmat, H: int, W: int, device) -> torch.Tensor:
    """``(H*W,)`` perpendicular-distance-free ray parameter ``t`` of the nearest STATIC surface.

    Kept as ``t`` rather than converted to a depth so the agent composite is a single comparison in the
    same units. MuJoCo's own depth buffer would do, but it carries up to 12 mm of float error at these
    zfar values, which is enough to flip the test where the agent touches the floor.
    """
    o, d = cached_rays(cam, H, W, device)
    P = prims.to(device)
    best = torch.full((H * W,), INF, device=device)
    for i, k in enumerate(P["kind"]):
        pos = torch.tensor(xpos[i], device=device, dtype=torch.float32)
        R = torch.tensor(xmat[i].reshape(3, 3), device=device, dtype=torch.float32)
        t, _ = intersect(
            k, P["size"][i].expand(H * W, 3), pos.expand(H * W, 3), R.expand(H * W, 3, 3), o.expand(H * W, 3), d
        )
        best = torch.minimum(best, t)
    return best


@torch.no_grad()
def composite(
    plate: torch.Tensor,
    plate_t: torch.Tensor,
    cam: Camera,
    prims: Primitives,
    xpos: torch.Tensor,
    xmat: torch.Tensor,
    lights: Lights,
    H: int,
    W: int,
    supersample: int = 1,
    out: torch.Tensor | None = None,
    tile: int | None = None,
    mat_rgb: torch.Tensor | None = None,
    size_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """``(B,H,W,3)`` uint8: the moving primitives drawn over a baked static plate.

    ``plate`` is ``(H,W,3)`` uint8 and ``plate_t`` the matching ``(H*W,)`` ray parameter, both constant
    for a given (scene, variation, size). Only the primitives in ``prims`` are cast per frame, which is
    the whole performance argument: the agent covers 20-150 pixels of 50176, so casting the full frame
    spends ~99.9% of its rays on geometry that cannot move.

    ``tile`` restricts the cast to a ``tile x tile`` screen-space window around the primitives. The window
    is derived from a conservative world-space bound, and the renderer falls back to the full frame
    whenever the bound does not fit — so the output is identical either way, which is asserted in the
    conformance suite rather than assumed.

    ``mat_rgb`` ``(B,P,3)`` and ``size_scale`` ``(B,)`` override the primitive colour and size PER
    ENVIRONMENT. That is what makes batched appearance randomisation real rather than nominal: without
    it a batch of 4096 environments would all have to share one agent colour, and a caller sampling a
    style per environment would silently get the first row's appearance on every frame.
    ``COMPOSITE_COMPACT_MERGE`` chooses the window from the bound rather than from ``tile`` and runs
    the tail in the compact cover index space; ``_composite_window`` below is the expression it
    replaces, kept verbatim.
    """

    if not COMPOSITE_COMPACT_MERGE:
        return _composite_window(
            plate, plate_t, cam, prims, xpos, xmat, lights, H, W, supersample, out, tile, mat_rgb, size_scale
        )

    device = plate.device
    B = xpos.shape[0]
    ss = int(supersample)
    Rh, Rw = H * ss, W * ss
    o, d = cached_rays(cam, Rh, Rw, device)
    P = prims.to(device)
    ldir, lamb, ldif, lspec = lights.to(device)

    if out is None:
        out = torch.empty((B, H, W, 3), dtype=torch.uint8, device=device)
    else:
        out = out.view(B, H, W, 3)

    # THE WINDOW IS SIZED TO THE BOUND, NOT TO THE CALLER'S MAXIMUM, and at 224 px that is most of
    # this function's cost. ``tile`` is a ceiling -- ogbench passes 32 and maze2d 48 -- while the
    # span the bound actually reports is 9-28 output pixels depending on the body, and EVERY term
    # below except the frame write and the shade is proportional to ``(tile*ss)^2``. Measured on the
    # certification node at 224 px, ss=2, batch 64, PointmazeLargeNavigate: the window-sized terms
    # (the cast, the two box averages, the per-environment ray gather) are 72% of the composite
    # while 4.7% of those samples can be covered, and its span is 14x13 -- so nine tenths of that
    # 72% is spent on samples the bound has already proved empty.
    #
    # Narrowing the window cannot change a value for the same reason the tiled path is allowed to
    # exist at all: a ray outside the span cannot reach a mover, so ``intersect`` returns ``INF``,
    # ``cov`` is False, ``col`` is +0.0 and the composite writes back the plate byte that
    # ``out.copy_`` has already put there. That is the argument this function's docstring already
    # makes against the full frame, applied one rung further in.
    org = None
    if tile is not None and tile < H and tile < W:
        bs = 1.0 if size_scale is None else float(size_scale.max())
        fit = _fit_tile(cam, prims, xpos, H, W, tile, device, bound_scale=bs)
        if fit is not None:
            org, tile = fit
    if org is None:
        N = Rh * Rw
        oo = o.expand(B, N, 3)
        dd = d.expand(B, N, 3)
        bt = (plate_t if ss == 1 else _upsample_t(plate_t, H, W, ss)).expand(B, N)
        obase = None
    else:
        # the window is chosen in OUTPUT pixels and then expanded, so a supersampled tile still lands
        # on whole output pixels and the box average is the same one the full-frame path takes
        idx = _expand_tile(org, tile, ss, Rw, device)
        N = idx.shape[1]
        oo = o.expand(B, N, 3)
        dd = d[idx.reshape(-1)].view(B, N, 3)
        # The occlusion threshold is read at the window's OUTPUT pixels and then blown up, rather
        # than blown up over the whole frame and then read at the window. ``_upsample_t`` is
        # nearest-neighbour, so ``base_t[(y*ss+a)*Rw + (x*ss+b)] == plate_t[y*W + x]`` exactly -- a
        # pure data movement either way -- but the shipped spelling builds a ``(H*ss*W*ss,)``
        # intermediate per call, which is the one term left in the tiled path that does not shrink
        # with the window.
        obase = _expand_tile(org, tile, 1, W, device)
        bt = plate_t[obase].view(B, tile, 1, tile, 1).expand(B, tile, ss, tile, ss).reshape(B, N)

    nprim = len(P["kind"])
    sz = P["size"].view(1, nprim, 3)
    if size_scale is not None:
        sz = sz * size_scale.view(B, 1, 1)
    t, n, i_hit = cast_nearest(P["kind"], sz.expand(B, nprim, 3), xpos, xmat, oo, dd)
    cov = t < bt

    # THE SHADED SET IS PER ROW, NOT THE UNION OVER THE BATCH. ``COVER_COMPACTION`` shades
    # ``nonzero(cov.any(0))``, which is sound but far looser than it reads: the window origin tracks
    # the movers' projected bound, and the agent's position INSIDE that window still drifts by
    # several pixels across a batch, so at 224 px, ss=2, batch 64 the union is 785 of 4,096 samples
    # while each row covers 193. Flattening ``(B, N)`` before the ``nonzero`` selects exactly the
    # covered (row, sample) pairs -- 4.1x fewer -- and the ``where`` that discarded the union's
    # uncovered members disappears with them, because every selected element is covered by
    # construction.
    #
    # ``shade`` is elementwise in every argument it takes (its only reduction is ``_dot``, unrolled
    # over the three components for exactly this reason, and ``lamb.sum()`` over the lights, which
    # does not depend on the ray shape), so shading a gathered subset returns the values the
    # superset held at those positions. ``col`` is therefore the same tensor either way: +0.0
    # everywhere no row covers, the shaded colour everywhere one does.
    col = torch.zeros((B, N, 3), device=device, dtype=n.dtype)
    flat = torch.nonzero(cov.reshape(-1), as_tuple=False).reshape(-1)
    if flat.numel():
        rows = flat.div(N, rounding_mode="floor")
        cols = flat - rows * N
        ih = i_hit[rows, cols]
        mat = P["mat"][ih] if mat_rgb is None else mat_rgb[rows, ih]
        c = (
            shade(
                n[rows, cols],
                -dd[rows, cols],
                ldir,
                lamb,
                ldif,
                lspec,
                lights.exponent,
                mat,
                P["tex"][ih],
                P["spec"][ih],
            )
            * 255.0
        )
        col.view(-1, 3).index_copy_(0, flat, c)

    covf = cov.to(col.dtype)
    if org is None:
        col = col.view(B, H, ss, W, ss, 3).sum((2, 4)) / (ss * ss)
        covf = covf.view(B, H, ss, W, ss).sum((2, 4)) / (ss * ss)
        base = plate.to(col.dtype).unsqueeze(0)
        out.copy_(torch.clamp(torch.round(base * (1 - covf.unsqueeze(-1)) + col), 0, 255).to(torch.uint8))
        return out

    # Tiled: box-average inside the window, write the plate everywhere and the COVERED pixels over it.
    #
    # The two box averages keep the spelling they had -- a four-element float sum has an association
    # order and a compacted reduction is not obliged to reproduce it, and this function's contract is
    # bit-exactness, not approximate agreement. What is compacted is everything after them, which is
    # elementwise and therefore free of that question.
    #
    # A pixel the cover does not touch is ALREADY CORRECT in ``out``: ``covf`` is 0 there, so
    # ``merged`` is ``patch * 1.0 + 0.0`` -- a plate byte promoted to float, which ``round`` returns
    # and ``clamp(0,255)`` cannot move -- and the scatter would write back the byte the broadcast
    # already wrote. So only the union of covered OUTPUT pixels needs gathering, blending, rounding
    # and scattering at all; the rest of the window is a plate copy of a plate copy. And the gather
    # takes the plate as the uint8 it is and promotes what it took, rather than promoting all 50,176
    # pixels of the frame to float to use 1,024 of them.
    col = col.view(B, tile, ss, tile, ss, 3).sum((2, 4)) / (ss * ss)
    covf = covf.view(B, tile, ss, tile, ss).sum((2, 4)) / (ss * ss)
    out.copy_(plate.view(1, H, W, 3).expand(B, H, W, 3))
    covf = covf.reshape(B, -1)
    ps = torch.nonzero((covf > 0).any(0), as_tuple=False).reshape(-1)
    if ps.numel():
        oidx = obase[:, ps].unsqueeze(-1).expand(B, ps.numel(), 3)
        patch = torch.gather(plate.reshape(1, H * W, 3).expand(B, H * W, 3), 1, oidx).to(col.dtype)
        merged = patch * (1 - covf[:, ps].unsqueeze(-1)) + col.reshape(B, -1, 3)[:, ps]
        out.view(B, H * W, 3).scatter_(1, oidx, torch.clamp(torch.round(merged), 0, 255).to(torch.uint8))
    return out


def _fit_tile(
    cam: Camera, prims: Primitives, xpos: torch.Tensor, H: int, W: int, tile: int, device, bound_scale: float = 1.0
):
    """``((x0, y0), tile)`` -- the window the BOUND asks for, or None if it does not fit ``tile``.

    ``_tile_origin`` answers "does the caller's tile hold the movers"; this answers "what does the
    bound actually need", which is the same question one rung finer. The span is projected once (one
    device-to-host synchronisation, inside ``screen_span``) and the tile is then the largest extent
    over the batch, capped by the caller's ceiling -- so a config whose movers already fill ``tile``
    gets ``tile`` and behaves exactly as before, and a 13-pixel sphere in a 32-pixel window gets 13.

    The clamp is safe at any tile: with ``t >= w`` the window ``[clamp(x0,0,W-t), +t)`` contains
    ``[x0, x0+w) & [0, W)`` at both edges, which is the property ``_tile_origin`` already relies on.

    Rungs are not rounded to a ladder. ``_expand_tile``'s memo is keyed on ``(tile, ss, width,
    device)`` and holds 16 entries, so a process alternating between several configs re-derives a
    ``(t*ss)^2`` integer grid now and then; that is microseconds against a frame, and it buys the
    tightest window the bound permits rather than the nearest power of two -- which matters, because
    the two Pointmaze rows this was measured for report spans of 14 and 21 and a power-of-two ladder
    would have rounded the second one straight back to 32.
    """
    span = screen_span(cam, prims, xpos, H, W, device, bound_scale)
    if span is None:
        return None
    x0, y0, w, h = span
    need = max(int(w), int(h), 1)
    if need > tile:
        return None
    return (x0.clamp(0, W - need), y0.clamp(0, H - need)), need


@torch.no_grad()
def _composite_window(
    plate: torch.Tensor,
    plate_t: torch.Tensor,
    cam: Camera,
    prims: Primitives,
    xpos: torch.Tensor,
    xmat: torch.Tensor,
    lights: Lights,
    H: int,
    W: int,
    supersample: int = 1,
    out: torch.Tensor | None = None,
    tile: int | None = None,
    mat_rgb: torch.Tensor | None = None,
    size_scale: torch.Tensor | None = None,
) -> torch.Tensor:
    """The pre-RAYCAST-6 ``composite``: the caller's ``tile`` verbatim, the union cover
    compaction, and the whole ``tile x tile`` window gathered, blended and scattered.

    Kept expression for expression so ``COMPOSITE_COMPACT_MERGE`` has something to be diffed
    against on the environments themselves, which is what ``tests/gpu/test_fast_paths.py``
    does with it. See ``composite`` for what the arguments mean.
    """
    device = plate.device
    B = xpos.shape[0]
    ss = int(supersample)
    Rh, Rw = H * ss, W * ss
    o, d = cached_rays(cam, Rh, Rw, device)
    P = prims.to(device)
    ldir, lamb, ldif, lspec = lights.to(device)

    if out is None:
        out = torch.empty((B, H, W, 3), dtype=torch.uint8, device=device)
    else:
        out = out.view(B, H, W, 3)

    base_t = plate_t if ss == 1 else _upsample_t(plate_t, H, W, ss)
    org = None
    if tile is not None and tile < H and tile < W:
        bs = 1.0 if size_scale is None else float(size_scale.max())
        org = _tile_origin(cam, prims, xpos, H, W, tile, device, bound_scale=bs)
    if org is None:
        N = Rh * Rw
        oo = o.expand(B, N, 3)
        dd = d.expand(B, N, 3)
        bt = base_t.expand(B, N)
    else:
        # the window is chosen in OUTPUT pixels and then expanded, so a supersampled tile still lands
        # on whole output pixels and the box average is the same one the full-frame path takes
        idx = _expand_tile(org, tile, ss, Rw, device)
        N = idx.shape[1]
        oo = o.expand(B, N, 3)
        dd = d[idx.reshape(-1)].view(B, N, 3)
        bt = base_t[idx]

    # One cast for all K primitives, then ONE shade of the winner -- not K shades selected by a
    # ``where``, which is what this loop used to be. The two agree bit for bit and the argument is
    # that both are the same function of the same winner: ``shade`` is elementwise in the normal and
    # in the material constants, so evaluating it at the nearest primitive's (n, mat, tex, spec) is
    # the value the selected branch carried. What changes is that the K-1 discarded shades are never
    # computed, and on OGBench's ant that is 12 of 13.
    nprim = len(P["kind"])
    sz = P["size"].view(1, nprim, 3)
    if size_scale is not None:
        sz = sz * size_scale.view(B, 1, 1)
    t, n, i_hit = cast_nearest(P["kind"], sz.expand(B, nprim, 3), xpos, xmat, oo, dd)
    cov = t < bt

    # A WINDOW IS NOT A COVERAGE MASK, and on this path the difference is most of the cost.
    #
    # ``tile`` is a rectangle around the movers' projected bound. The agent inside it is a sphere or
    # a small articulated body: measured on the CPU certification node at 224 px, ss=2, batch 256,
    # ``shade`` and the four material gathers below are 47% of a PointmazeLargeNavigate render
    # (115.5 ms + 44.3 ms of 337.0 ms) while the cast that decides coverage is 19%. Every one of
    # ``shade``'s ~45 elementwise passes was running over the whole window and the result was then
    # thrown away by the ``where``.
    #
    # RESTRICTING THE SET CANNOT CHANGE A VALUE. ``col`` is ``where(cov, c, 0)`` and every ray's
    # colour is a function of that ray alone, so at a ray position where NO row of the batch is
    # covered the value is ``+0.0`` for every row whatever the shader would have said -- which is
    # exactly what ``torch.zeros`` holds there. The union is over ray POSITIONS rather than pixels
    # (position j is a different pixel in each row once the window moves per environment), which is
    # as sound for the same reason. ``DMControlSuite._render_group`` makes this argument already and
    # calls it ``COVER_COMPACTION``; ``composite`` never had it.
    #
    # Everything downstream sees the identical ``(B, N, 3)`` tensor it saw before, so the box average
    # and the composite are untouched and stay bit-for-bit what they were.
    if COVER_COMPACTION:
        col = torch.zeros((B, N, 3), device=device, dtype=n.dtype)
        js = torch.nonzero(cov.any(0), as_tuple=False).reshape(-1)
        if js.numel():
            ih = i_hit[:, js]
            mat = P["mat"][ih] if mat_rgb is None else mat_rgb[torch.arange(B, device=device).unsqueeze(1), ih]
            c = (
                shade(n[:, js], -dd[:, js], ldir, lamb, ldif, lspec, lights.exponent, mat, P["tex"][ih], P["spec"][ih])
                * 255.0
            )
            col.index_copy_(1, js, torch.where(cov[:, js].unsqueeze(-1), c, torch.zeros_like(c)))
    else:  # the expression above is the identity of this one
        mat = P["mat"][i_hit] if mat_rgb is None else mat_rgb[torch.arange(B, device=device).unsqueeze(1), i_hit]
        c = shade(n, -dd, ldir, lamb, ldif, lspec, lights.exponent, mat, P["tex"][i_hit], P["spec"][i_hit]) * 255.0
        col = torch.where(cov.unsqueeze(-1), c, torch.zeros_like(c))

    covf = cov.to(col.dtype)
    if org is None:
        col = col.view(B, H, ss, W, ss, 3).sum((2, 4)) / (ss * ss)
        covf = covf.view(B, H, ss, W, ss).sum((2, 4)) / (ss * ss)
        base = plate.to(col.dtype).unsqueeze(0)
        out.copy_(torch.clamp(torch.round(base * (1 - covf.unsqueeze(-1)) + col), 0, 255).to(torch.uint8))
        return out

    # Tiled: box-average inside the window, write the plate everywhere and the window over it.
    #
    # The plate is broadcast as UINT8 straight into the destination rather than promoted to a
    # (B,H*W,3) float canvas that is then cloned, rounded, clamped and cast back. Those four passes
    # were the whole remaining cost of an OGBench frame once the cast was fused -- 805 MB of float32
    # per pass at batch 1024 and 256 px, against a 201 MB uint8 output -- and every one of them is
    # the IDENTITY outside the window: a plate byte promoted to float is an integer, ``round`` returns
    # it and ``clamp(0,255)`` cannot move a value that came out of a uint8. So the two paths write the
    # same bytes by construction, and only the tile*tile pixels the movers can reach are computed.
    col = col.view(B, tile, ss, tile, ss, 3).sum((2, 4)) / (ss * ss)
    covf = covf.view(B, tile, ss, tile, ss).sum((2, 4)) / (ss * ss)
    oidx = _expand_tile(org, tile, 1, W, device).unsqueeze(-1).expand(B, tile * tile, 3)
    out.copy_(plate.view(1, H, W, 3).expand(B, H, W, 3))
    patch = torch.gather(plate.reshape(1, H * W, 3).to(col.dtype).expand(B, H * W, 3), 1, oidx)
    merged = patch * (1 - covf.reshape(B, -1, 1)) + col.reshape(B, -1, 3)
    out.view(B, H * W, 3).scatter_(1, oidx, torch.clamp(torch.round(merged), 0, 255).to(torch.uint8))
    return out


def _upsample_t(plate_t, H, W, ss):
    """Nearest-neighbour blow-up of the static ray parameter to the supersample grid.

    The static ``t`` is only used as an occlusion threshold, and it is smooth except across silhouettes,
    where the agent is either fully in front or fully behind by far more than the interpolation error.
    """
    return plate_t.view(H, W).repeat_interleave(ss, 0).repeat_interleave(ss, 1).reshape(-1)


def screen_span(cam: Camera, prims: Primitives, xpos: torch.Tensor, H: int, W: int, device, bound_scale: float = 1.0):
    """``(x0, y0, w, h)``: the window origin per batch element (device) and its worst extent (host).

    The bound is deliberately conservative — the world-space AABB of each primitive's bounding sphere,
    all 8 corners projected — because a window that clips a mover would silently drop pixels, and the
    speed-up is worthless if the two paths can disagree.

    ONE projection and ONE device-to-host transfer, and that is the whole reason this is a separate
    function. A caller choosing a tile size walks a ladder of candidate sizes, and asking
    ``_tile_origin`` for each rung re-projected the corners and synchronised two or three times per
    rung: measured at 4.7-6.4 ms per 32-frame chunk on the DM Control and gym MuJoCo cells, which is
    37% of a MujocoReacher render and more than its ray passes cost. The extent does not depend on
    the tile size, so it is computed once and the ladder is walked on the host.
    """
    radt = prims.radii(device, xpos.dtype, bound_scale)
    if radt is None:  # host-side: a PLANE has no finite bound
        return None
    lo = xpos - radt.view(1, -1, 1)
    hi = xpos + radt.view(1, -1, 1)
    corners = [torch.where(mask, hi, lo) for mask in _corner_masks(device)]
    pts = torch.cat(corners, 1)  # (B, 8P, 3)
    ndc = cam.project(pts, aspect=W / H)
    px = (ndc[..., 0] + 1) * 0.5 * W
    py = (1 - ndc[..., 1]) * 0.5 * H
    x0 = px.amin(1).floor().long() - 1
    x1 = px.amax(1).ceil().long() + 1
    y0 = py.amin(1).floor().long() - 1
    y1 = py.amax(1).ceil().long() + 1
    probe = torch.stack(
        [(x1 - x0).max(), (y1 - y0).max(), torch.isfinite(ndc).all().long()]
    ).cpu()  # the one synchronisation
    if not int(probe[2]):
        return None
    return x0, y0, int(probe[0]), int(probe[1])


def _tile_origin(
    cam: Camera, prims: Primitives, xpos: torch.Tensor, H: int, W: int, tile: int, device, bound_scale: float = 1.0
):
    """``(x0, y0)`` per batch element in OUTPUT pixels, or None if the primitives do not fit ``tile``.

    When it does not fit, the caller falls back to the full frame and the output is identical either
    way — which ``tests/gpu/test_fast_paths.py`` asserts rather than assumes.
    """
    span = screen_span(cam, prims, xpos, H, W, device, bound_scale)
    if span is None:
        return None
    x0, y0, w, h = span
    if w > tile or h > tile:
        return None
    return x0.clamp(0, W - tile), y0.clamp(0, H - tile)


#: ``(tile, ss, width, device)`` -> the ``(tile*ss)^2`` flat offsets of a window at the origin.
_TILE_CACHE_MAX = 16
_TILE_CACHE = _Memo(_TILE_CACHE_MAX, name="raycast3d.tile")


def _expand_tile(org, tile: int, ss: int, Wpix: int, device) -> torch.Tensor:
    """``(B, (tile*ss)^2)`` flat indices into a ``ss``-times supersampled frame of width ``Wpix``.

    ``(y0*ss + a)*Wpix + (x0*ss + b)`` is ``(y0*ss*Wpix + x0*ss) + (a*Wpix + b)``: a per-environment
    SCALAR plus a grid that depends only on (tile, ss, width). The grid was being rebuilt per call as
    two ``(B, T, T)`` int64 expands and a multiply-add -- 8 MB of index traffic per call at 224 px,
    batch 256, twice per composite -- to produce the same 4,096 numbers every time. Integer
    arithmetic, so the memoised form is exactly the expression it replaces.
    """
    x0, y0 = org
    B = x0.shape[0]
    key = (int(tile), int(ss), int(Wpix), str(device))
    base = _TILE_CACHE.get(key)
    if base is None:
        T = tile * ss
        ar = torch.arange(T, device=device)
        base = _TILE_CACHE.put(key, (ar.view(T, 1) * Wpix + ar.view(1, T)).reshape(1, -1))
    return (y0.view(B, 1) * ss * Wpix + x0.view(B, 1) * ss) + base
