"""Batched analytic rendering of MuJoCo primitive geometry.

Torch and Triton share intersection and shading conventions. Moving primitives
are composited over a fixed RGB/depth plate; compact windows avoid casting the
static background again. Float operation order is part of the pixel contract.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import torch

from .. import tuning
from . import _resources as _RES
from .primitives import (
    BOX,
    CAPSULE,
    CYLINDER,
    ELLIPSOID,
    INF,
    PLANE,
    PRIMITIVE_KINDS,
    SPHERE,
    Primitives,
    bound_radius_tensor,
    dot3,
    mtv3,
    mv3,
)

Kinds = Sequence[int] | np.ndarray
Cast = tuple[torch.Tensor, torch.Tensor, torch.Tensor]

try:
    import triton
    import triton.language as tl

    HAVE_TRITON = True
except ImportError:  # reported by the renderer, never silent
    HAVE_TRITON = False


@dataclass(frozen=True)
class Camera:
    """Pinhole camera with the near-plane frustum reported by ``mjvGLCamera``.

    Keeping the frustum supports off-centre and stereo cameras.
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
        """Build a mono camera from one eye or the average of the two stereo eyes."""

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
        """Return origin (3,) and unit directions (H*W, 3) through pixel centres, top row first."""
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
        """Project world points (..., 3) to normalised device coordinates (..., 2).

        ``aspect`` is W/H. The result is used to choose conservative screen windows.
        """
        R, origin = _proj_tensors(self, pts.device, pts.dtype)
        rel = pts - origin
        # Keep projection independent of batch-dependent BLAS rounding at pixel boundaries.
        loc = mv3(R, rel)
        z = loc[..., 2].clamp_min(1e-6)
        half_h = 0.5 * (self.top - self.bottom)
        mid_y = 0.5 * (self.top + self.bottom)
        x = (loc[..., 0] / z * self.near - self.center) / (half_h * aspect)
        y = (loc[..., 1] / z * self.near - mid_y) / half_h
        return torch.stack([x, y], -1)


_Memo = _RES.BoundedCache

_CACHE_MAX = 32
# Cache device constants to avoid repeated pageable host-to-device copies.
_PROJ_CACHE = _Memo(_CACHE_MAX, name="raycast3d.proj")
_CORNER_CACHE = _Memo(_CACHE_MAX, name="raycast3d.corner")


def _proj_tensors(cam: Camera, device, dtype):
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
    key = str(device)
    hit = _CORNER_CACHE.get(key)
    if hit is None:
        hit = _CORNER_CACHE.put(
            key, [torch.tensor([bool(m & 1), bool(m & 2), bool(m & 4)], device=device).view(1, 1, 3) for m in range(8)]
        )
    return hit


@dataclass
class Lights:
    """Directional lights with unit directions toward each light and scalar intensity weights."""

    dirs: np.ndarray  # (L,3), unit, toward the light
    ambient: np.ndarray  # (L,)
    diffuse: np.ndarray  # (L,)
    specular: np.ndarray  # (L,)
    exponent: float

    def to(self, device, dtype=torch.float32):
        t = lambda x: torch.tensor(np.asarray(x, np.float64), device=device, dtype=dtype)  # noqa: E731
        return (t(self.dirs), t(self.ambient), t(self.diffuse), t(self.specular))


def _sphere(o, d, pos, r, normal=True):
    oc = o - pos
    b = dot3(oc, d)
    c = dot3(oc, oc) - r * r
    disc = b * b - c
    sq = torch.sqrt(torch.clamp(disc, min=0))
    t = -b - sq
    t = torch.where(t > 0, t, -b + sq)
    hit = (disc > 0) & (t > 0)
    t = torch.where(hit, t, torch.full_like(t, INF))
    if not normal:
        return t, None
    n = (oc + t.unsqueeze(-1).clamp(max=1e6) * d) / r.unsqueeze(-1)
    return t, n


def _ellipsoid(o, d, pos, R, size, normal=True):
    """Intersect an ellipsoid by scaling local rays to a unit sphere.

    Normals are normalised gradients in the local frame, rotated back to world space.
    A nonpositive semi-axis produces a miss.
    """
    ol = mtv3(R, o - pos)
    dl = mtv3(R, d)
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
    if not normal:
        return t, None
    pl = ol + t.unsqueeze(-1).clamp(max=1e6) * dl
    nl = pl * inv * inv
    nl = nl / torch.clamp(torch.sqrt(dot3(nl, nl)).unsqueeze(-1), min=1e-12)
    return t, mv3(R, nl)


def _capsule(o, d, pos, R, r, hh, normal=True):
    """Intersect a local Z-axis capsule with radius r and cylinder half-length hh."""
    ol = mtv3(R, o - pos)
    dl = mtv3(R, d)
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
        bb = dot3(oc, dl)
        cc = dot3(oc, oc) - r * r
        dd = bb * bb - cc
        s2 = torch.sqrt(torch.clamp(dd, min=0))
        for sgn in (-1.0, 1.0):
            t = -bb + sgn * s2
            ok = (dd > 0) & (t > 0)
            best = torch.where(ok & (t < best), t, best)
    if not normal:
        return best, None
    pl = ol + best.unsqueeze(-1).clamp(max=1e6) * dl
    zc = torch.maximum(torch.minimum(pl[..., 2], hh), -hh)
    nl = pl - torch.stack([torch.zeros_like(zc), torch.zeros_like(zc), zc], -1)
    nl = nl / torch.clamp(torch.sqrt(dot3(nl, nl)).unsqueeze(-1), min=1e-12)
    return best, mv3(R, nl)


def _cylinder(o, d, pos, R, r, hh, normal=True):
    ol = mtv3(R, o - pos)
    dl = mtv3(R, d)
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
    if not normal:
        return best, None
    pl = ol + best.unsqueeze(-1).clamp(max=1e6) * dl
    nl_side = torch.stack([pl[..., 0], pl[..., 1], torch.zeros_like(pl[..., 0])], -1) / r.unsqueeze(-1)
    nl_cap = torch.stack([torch.zeros_like(capz), torch.zeros_like(capz), capz], -1)
    nl = torch.where(side.unsqueeze(-1), nl_side, nl_cap)
    return best, mv3(R, nl)


def _box(o, d, pos, R, size, normal=True):
    ol = mtv3(R, o - pos)
    dl = mtv3(R, d)
    inv = 1.0 / torch.where(dl.abs() < 1e-12, torch.full_like(dl, 1e-12), dl)
    t1 = (-size - ol) * inv
    t2 = (size - ol) * inv
    tmin = torch.minimum(t1, t2).max(-1).values
    tmax = torch.maximum(t1, t2).min(-1).values
    hit = tmax >= torch.clamp(tmin, min=0.0)
    t = torch.where(tmin > 0, tmin, tmax)
    t = torch.where(hit & (t > 0), t, torch.full_like(t, INF))
    if not normal:
        return t, None
    pl = ol + t.unsqueeze(-1).clamp(max=1e6) * dl
    rel = pl.abs() / torch.clamp(size, min=1e-12)
    ax = rel.argmax(-1, keepdim=True)
    nl = torch.zeros_like(pl).scatter_(-1, ax, torch.sign(pl.gather(-1, ax)))
    return t, mv3(R, nl)


def _plane(o, d, pos, R, size):
    n = R[..., :, 2]
    den = dot3(d, n)
    num = dot3(pos - o, n)
    t = torch.where(
        den.abs() > 1e-12,
        num / torch.where(den.abs() > 1e-12, den, torch.full_like(den, 1.0)),
        torch.full_like(den, INF),
    )
    p = o + t.unsqueeze(-1).clamp(max=1e6) * d
    loc = mtv3(R, p - pos)
    ok = (t > 0) & (loc[..., 0].abs() <= size[..., 0]) & (loc[..., 1].abs() <= size[..., 1])
    return torch.where(ok, t, torch.full_like(t, INF)), n.expand_as(p)


def intersect(
    kind: int, size: torch.Tensor, pos: torch.Tensor, R: torch.Tensor, o: torch.Tensor, d: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return ray parameters and world-space normals, with t=INF on misses.

    Poses (..., 3) and rotations (..., 3, 3) broadcast against rays (..., 3).
    Unsupported MuJoCo geometry kinds raise ValueError.
    """
    return _intersect(kind, size, pos, R, o, d, normal=True)


def _intersect_distance(kind, size, pos, R, o, d):
    """Return hit distances without constructing normals for occlusion rays."""
    return _intersect(kind, size, pos, R, o, d, normal=False)[0]


def _intersect(kind, size, pos, R, o, d, *, normal):
    if kind == SPHERE:
        return _sphere(o, d, pos, size[..., 0:1].squeeze(-1), normal=normal)
    if kind == CAPSULE:
        return _capsule(o, d, pos, R, size[..., 0:1].squeeze(-1), size[..., 1:2].squeeze(-1), normal=normal)
    if kind == CYLINDER:
        return _cylinder(o, d, pos, R, size[..., 0:1].squeeze(-1), size[..., 1:2].squeeze(-1), normal=normal)
    if kind == BOX:
        return _box(o, d, pos, R, size, normal=normal)
    if kind == ELLIPSOID:
        return _ellipsoid(o, d, pos, R, size, normal=normal)
    if kind == PLANE:
        return _plane(o, d, pos, R, size)
    raise ValueError(f"raycast3d has no intersector for mjtGeom {kind}; scenes here are primitive-only")


def _cast_nearest_torch_unbounded(
    kinds: Kinds, sizes: torch.Tensor, pos: torch.Tensor, R: torch.Tensor, o: torch.Tensor, d: torch.Tensor
) -> Cast:
    B, N = o.shape[0], o.shape[1]
    best = torch.full((B, N), INF, device=o.device, dtype=o.dtype)
    nrm = torch.zeros_like(o)
    idx = torch.zeros((B, N), device=o.device, dtype=torch.long)
    for k, kind in enumerate(int(x) for x in kinds):
        t, n = intersect(kind, sizes[:, k : k + 1], pos[:, k : k + 1], R[:, k : k + 1], o, d)
        # Strict comparison in primitive order preserves the lowest index on ties.
        upd = t < best
        best = torch.where(upd, t, best)
        nrm = torch.where(upd.unsqueeze(-1), n.expand(B, N, 3), nrm)
        idx = torch.where(upd, torch.full_like(idx, k), idx)
    return best, nrm, idx


# Sphere/capsule quadratics assume unit rays; bounds must not prune their non-unit results.
_UNIT_D_TOL = 1e-6


def _unit_rays(dd) -> bool:
    return bool(torch.all((dd - 1.0).abs() <= _UNIT_D_TOL))


# Distant float32 quadratics can report hits outside exact geometric bounds.
# Inflate r² by a distance-scaled noise budget to preserve those hit decisions.
_BOUND_SLACK_REL, _BOUND_SLACK_ABS = 1.01, 1e-6
_BOUND_SLACK_NOISE = 16.0 * 1.1920929e-7  # (|oc| * sqrt(this))^2, added to r^2


# Bounds pay for costlier intersections; spheres and ellipsoids use the full ray set.
_BOUNDED_KINDS = frozenset({CAPSULE, CYLINDER, BOX})


def _kind_is_bounded(kind: int) -> bool:
    return kind in _BOUNDED_KINDS


def _rays_near(
    pos_k: torch.Tensor, rad: torch.Tensor, o: torch.Tensor, d: torch.Tensor, dd: torch.Tensor
) -> torch.Tensor:
    """Return a (B, N) mask of rays within an inflated forward bounding sphere."""
    # Cross products avoid cancellation in |oc|²|d|² - (oc·d)² at long distances.
    if o.device.type == "cpu":
        x, y, z = (pos_k[..., j] - o[..., j] for j in range(3))
        dx, dy, dz = (d[..., j] for j in range(3))
        # CPU torch.cross contracts multiply/subtract; addcmul preserves that tangency rounding.
        cx = torch.addcmul(-z * dy, y, dz)
        cy = torch.addcmul(-x * dz, z, dx)
        cz = torch.addcmul(-y * dx, x, dy)
        r0 = rad * _BOUND_SLACK_REL + _BOUND_SLACK_ABS
        r2 = r0 * r0 + _BOUND_SLACK_NOISE * (x * x + y * y + z * z)
        return ((cx * cx + cy * cy + cz * cz) <= r2 * dd) & (x * dx + y * dy + z * dz + torch.sqrt(r2 * dd) >= 0)
    oc = pos_k - o
    cr = torch.cross(oc.expand_as(d), d, dim=-1)
    r0 = rad * _BOUND_SLACK_REL + _BOUND_SLACK_ABS
    r2 = r0 * r0 + _BOUND_SLACK_NOISE * dot3(oc, oc)
    return (dot3(cr, cr) <= r2 * dd) & (dot3(oc, d) + torch.sqrt(r2 * dd) >= 0)


def cast_nearest_torch(
    kinds: Kinds,
    sizes: torch.Tensor,
    pos: torch.Tensor,
    R: torch.Tensor,
    o: torch.Tensor,
    d: torch.Tensor,
    *,
    bounds: bool = True,
) -> Cast:
    """Return nearest ray parameters (B, N), normals (B, N, 3), and indices (B, N).

    ``kinds`` is a host sequence of K codes. Sizes and positions are (B, K, 3),
    rotations (B, K, 3, 3), and ray origins/directions (B, N, 3). Misses have t=INF.
    With ``bounds=False``, every ray is tested against every primitive.
    """
    if not bounds:
        return _cast_nearest_torch_unbounded(kinds, sizes, pos, R, o, d)
    # Avoid bound setup and reductions when none of the primitive kinds can benefit.
    if not any(_kind_is_bounded(int(x)) for x in kinds):
        return _cast_nearest_torch_unbounded(kinds, sizes, pos, R, o, d)
    B, N = o.shape[0], o.shape[1]
    best = torch.full((B, N), INF, device=o.device, dtype=o.dtype)
    nrm = torch.zeros((B, N, 3), device=o.device, dtype=o.dtype)
    idx = torch.zeros((B, N), device=o.device, dtype=torch.long)
    dd = dot3(d, d)
    if not _unit_rays(dd):
        return _cast_nearest_torch_unbounded(kinds, sizes, pos, R, o, d)
    for k, kind in enumerate(int(x) for x in kinds):
        js = None
        rad = bound_radius_tensor(kind, sizes[:, k : k + 1]) if _kind_is_bounded(kind) else None
        if rad is not None:
            js = torch.nonzero(_rays_near(pos[:, k : k + 1], rad, o, d, dd).any(0), as_tuple=False).reshape(-1)
            if js.numel() == 0:
                continue
            if js.numel() == N:  # nothing pruned; skip the gather
                js = None
        oo, ddir = (o, d) if js is None else (o[:, js], d[:, js])
        t, n = intersect(kind, sizes[:, k : k + 1], pos[:, k : k + 1], R[:, k : k + 1], oo, ddir)
        if js is None:
            # Strict comparison in primitive order preserves the lowest index on ties.
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
    B, N = o.shape[0], o.shape[1]
    blocked = torch.zeros((B, N), device=o.device, dtype=torch.bool)
    for k, kind in enumerate(int(x) for x in kinds):
        t = _intersect_distance(kind, sizes[:, k : k + 1], pos[:, k : k + 1], R[:, k : k + 1], o, d)
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
    *,
    bounds: bool = True,
) -> torch.Tensor:
    """Return a (B, N) mask of hits with 0 < t < lim, excluding each ray's skip index.

    Inputs follow ``cast_nearest_torch``; lim and skip have shape (B, N).
    With ``bounds=False``, every ray is tested against every primitive.
    """
    if not bounds:
        return _any_blocking_torch_unbounded(kinds, sizes, pos, R, o, d, lim, skip)
    # Avoid bound setup and reductions when none of the primitive kinds can benefit.
    if not any(_kind_is_bounded(int(x)) for x in kinds):
        return _any_blocking_torch_unbounded(kinds, sizes, pos, R, o, d, lim, skip)
    B, N = o.shape[0], o.shape[1]
    blocked = torch.zeros((B, N), device=o.device, dtype=torch.bool)
    dd = dot3(d, d)
    if not _unit_rays(dd):
        return _any_blocking_torch_unbounded(kinds, sizes, pos, R, o, d, lim, skip)
    for k, kind in enumerate(int(x) for x in kinds):
        js = None
        rad = bound_radius_tensor(kind, sizes[:, k : k + 1]) if _kind_is_bounded(kind) else None
        if rad is not None:
            js = torch.nonzero(_rays_near(pos[:, k : k + 1], rad, o, d, dd).any(0), as_tuple=False).reshape(-1)
            if js.numel() == 0:
                continue
            if js.numel() == N:
                js = None
        if js is None:
            t = _intersect_distance(kind, sizes[:, k : k + 1], pos[:, k : k + 1], R[:, k : k + 1], o, d)
            hit = (t > 0) & (t < lim)
            if skip is not None:
                hit = hit & (skip != k)
            blocked |= hit
        else:
            t = _intersect_distance(kind, sizes[:, k : k + 1], pos[:, k : k + 1], R[:, k : k + 1], o[:, js], d[:, js])
            hit = (t > 0) & (t < lim[:, js])
            if skip is not None:
                hit = hit & (skip[:, js] != k)
            blocked[:, js] |= hit
    return blocked


# Exact cast parity needs unfused multiply/add, div_rn, and sqrt_rn together.
# Keep the primitive loop dynamic to limit compilation size.
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
        SPHERE: tl.constexpr = SPHERE,
        CAPSULE: tl.constexpr = CAPSULE,
        CYLINDER: tl.constexpr = CYLINDER,
        ELLIPSOID: tl.constexpr = ELLIPSOID,
        BOX: tl.constexpr = BOX,
    ):
        """Intersect a ray block with a sphere, capsule, cylinder, ellipsoid, box, or plane.

        Return (t, nx, ny, nz), with t=INF on misses. Each branch assigns every output.
        """
        zero = tl.zeros([BLOCK], dtype=tl.float32)
        never = tl.arange(0, BLOCK) < 0  # all-False mask
        t = zero + INF
        nx = zero
        ny = zero
        nz = zero

        if kind == SPHERE:
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

        elif kind == CAPSULE or kind == CYLINDER:
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
            if kind == CAPSULE:  # spherical caps
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
            if kind == CAPSULE:
                zcl = tl.maximum(tl.minimum(plz, hh), -hh)
                mx = plx
                my = ply
                mz = plz - zcl
                den = tl.maximum(tl.sqrt_rn(mx * mx + my * my + mz * mz), 1e-12)
                nlx = tl.math.div_rn(mx, den)
                nly = tl.math.div_rn(my, den)
                nlz = tl.math.div_rn(mz, den)
            else:
                # The side normal uses +0.0, regardless of the sign of the local hit height.
                nlx = tl.where(side, tl.math.div_rn(plx, r), 0.0)
                nly = tl.where(side, tl.math.div_rn(ply, r), 0.0)
                nlz = tl.where(side, 0.0, capz)
            t = best
            nx = r00 * nlx + r01 * nly + r02 * nlz
            ny = r10 * nlx + r11 * nly + r12 * nlz
            nz = r20 * nlx + r21 * nly + r22 * nlz

        elif kind == ELLIPSOID:
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

        elif kind == BOX:
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
            # The first maximum wins, matching Torch argmax on exact box corners.
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
                # Strict comparison in primitive order preserves the lowest index on ties.
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


_KIND_CACHE = _Memo(64, name="raycast3d.kind")


def _kind_tensor(kinds, device) -> torch.Tensor:
    key = (str(device), tuple(int(k) for k in kinds))
    hit = _KIND_CACHE.get(key)
    if hit is None:
        # The final kernel branch is a plane, so reject unknown geometry before launch.
        bad = sorted({k for k in key[1] if k not in PRIMITIVE_KINDS})
        if bad:
            raise ValueError(f"raycast3d has no intersector for mjtGeom {bad}; scenes here are primitive-only")
        hit = _KIND_CACHE.put(key, torch.tensor(key[1], device=device, dtype=torch.int32))
    return hit


def _packed(*ts) -> bool:
    """Check the trailing axes that the kernel indexes without explicit strides."""
    for t in ts:
        if t is None:
            continue
        if t.stride(-1) != 1:
            return False
        if t.dim() == 4 and t.stride(-2) != t.shape[-1]:
            return False
    return True


def _use_triton(backend, o, *inputs, skip=None) -> bool:
    if backend not in ("auto", "torch", "triton"):
        raise ValueError(f"unknown raycast backend {backend!r}; expected 'auto', 'torch', or 'triton'")
    if backend == "torch":
        return False
    reason = None
    if not HAVE_TRITON:
        reason = "Triton is not installed"
    elif not o.is_cuda:
        reason = "rays must be on CUDA"
    elif any(t.device != o.device or t.dtype != torch.float32 for t in (o, *inputs)):
        reason = "floating-point inputs must be float32 on the same CUDA device"
    elif skip is not None and (skip.device != o.device or skip.dtype not in (torch.int32, torch.int64)):
        reason = "skip indices must be int32 or int64 on the ray device"
    elif not _packed(o, *inputs, skip):
        reason = "tensor trailing axes must be packed, including each 3x3 rotation"
    if reason is not None and backend == "triton":
        raise ValueError(f"raycast backend='triton': {reason}")
    return reason is None


def _launch(kinds, sizes, pos, R, o, d, lim, skip, mode):
    # Leading strides allow expanded inputs without materialising each ray origin or size.
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
    block = tuning.get("raycast_block", device=dev)
    grid = (B, triton.cdiv(N, block))
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
            BLOCK=block,
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
    """Return (ray parameter, world normal, primitive index); see ``cast_nearest_torch``.

    ``auto`` uses Triton for packed float32 inputs on CUDA, otherwise Torch.
    An unknown backend or unsupported explicit ``triton`` request raises ValueError.
    """
    if not _use_triton(backend, o, sizes, pos, R, d):
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
    """Return a (B, N) mask of hits before lim, excluding each ray's optional skip index.

    Backend selection and errors follow ``cast_nearest``.
    """
    if not _use_triton(backend, o, sizes, pos, R, d, lim, skip=skip):
        return any_blocking_torch(kinds, sizes, pos, R, o, d, lim, skip)
    return _launch(kinds, sizes, pos, R, o, d, lim.expand(o.shape[0], o.shape[1]), skip, 1)


_RAY_CACHE_MAX = 8
_RAY_CACHE = _Memo(_RAY_CACHE_MAX, name="raycast3d.rays")


def cached_rays(cam: Camera, H: int, W: int, device, dtype=torch.float32) -> tuple[torch.Tensor, torch.Tensor]:
    """Return cached camera rays, keyed by camera, resolution, device, and dtype."""
    key = (cam, int(H), int(W), str(device), dtype)
    hit = _RAY_CACHE.get(key)
    if hit is None:
        hit = _RAY_CACHE.put(key, cam.rays(H, W, device, dtype))
    return hit


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
    """Return RGB (..., 3) in [0, 1].

    White specular is added to the material-lit colour before texture modulation;
    texturing only the diffuse term changes highlights on textured surfaces.
    """
    amb = lamb.sum()
    prim = mat * amb
    highlight = torch.zeros(n.shape[:-1], device=n.device, dtype=n.dtype)
    for i in range(ldir.shape[0]):
        l = ldir[i]
        ndl = torch.clamp(dot3(n, l), min=0)
        prim = prim + mat * (ldif[i] * ndl).unsqueeze(-1)
        h = l + v
        h = h / torch.clamp(torch.sqrt(dot3(h, h)).unsqueeze(-1), min=1e-12)
        highlight = highlight + lspec[i] * torch.clamp(dot3(n, h), min=0) ** exponent
    prim = prim + (highlight * spec).unsqueeze(-1)
    return torch.clamp(prim * tex, 0, 1)


@torch.no_grad()
def static_depth(cam: Camera, prims: Primitives, xpos, xmat, H: int, W: int, device) -> torch.Tensor:
    """Return nearest static-surface ray parameters (H*W,), in radial units.

    MuJoCo depth is perpendicular distance, t * dot(direction, forward).
    Keeping radial t lets the mover composite compare hits without conversion.
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
    shader=None,
    cast: Sequence[int] | None = None,
) -> torch.Tensor:
    """Draw moving primitives over a static plate; return (B, H, W, 3) uint8.

    ``plate`` is (H, W, 3) or per-environment (B, H, W, 3) uint8 and ``plate_t`` is
    (H*W,) radial ray depth. Poses are (B, P, 3) positions and (B, P, 3, 3) rotations.
    ``tile`` caps the conservative screen window; an oversized bound falls back to the
    full frame. ``mat_rgb`` (B, P, 3) and ``size_scale`` (B,) override appearance per
    environment. ``shader(rows, hit, normal, view, prim, mat)`` replaces ``lights``
    and returns covered-sample colour in [0, 255].
    A supplied ``out`` buffer is filled and returned as a (B, H, W, 3) view.
    ``cast`` limits the cast to these primitive indices; hits keep their index in ``prims``.
    The other primitives must not cover any sample that these can cover.
    """
    device = plate.device
    ids = None if cast is None else [int(j) for j in cast]
    bound_prims = prims if ids is None else prims.subset(ids)
    bound_pos = xpos if ids is None else xpos[:, ids]
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

    # Rays outside a conservative mover bound leave the static plate unchanged.
    org = None
    if tile is not None and tile < H and tile < W:
        bs = 1.0 if size_scale is None else float(size_scale.max())
        fit = _fit_tile(cam, bound_prims, bound_pos, H, W, tile, device, bound_scale=bs)
        if fit is not None:
            org, tile = fit
    if org is None:
        N = Rh * Rw
        oo = o.expand(B, N, 3)
        dd = d.expand(B, N, 3)
        bt = (plate_t if ss == 1 else _upsample_t(plate_t, H, W, ss)).expand(B, N)
        obase = None
    else:
        # Align tiles to output pixels so supersample averaging uses the same groups.
        idx = _expand_tile(org, tile, ss, Rw, device)
        N = idx.shape[1]
        oo = o.expand(B, N, 3)
        dd = d[idx.reshape(-1)].view(B, N, 3)
        # Read depth only at tile pixels, then repeat; avoid a full supersampled depth buffer.
        obase = _expand_tile(org, tile, 1, W, device)
        bt = plate_t[obase].view(B, tile, 1, tile, 1).expand(B, tile, ss, tile, ss).reshape(B, N)

    nprim = len(P["kind"])
    sz = P["size"].view(1, nprim, 3)
    if size_scale is not None:
        sz = sz * size_scale.view(B, 1, 1)
    if ids is None:
        t, n, i_hit = cast_nearest(P["kind"], sz.expand(B, nprim, 3), xpos, xmat, oo, dd)
    else:
        sel = torch.as_tensor(ids, device=device)
        kinds = [P["kind"][j] for j in ids]
        t, n, i_hit = cast_nearest(kinds, sz.expand(B, nprim, 3)[:, sel], xpos[:, sel], xmat[:, sel], oo, dd)
        i_hit = sel[i_hit]
    cov = t < bt

    col = torch.zeros((B, N, 3), device=device, dtype=n.dtype)
    # Shade covered (row, sample) pairs; uncaptured samples retain exact +0.0.
    flat = torch.nonzero(cov.reshape(-1), as_tuple=False).reshape(-1)
    if flat.numel():
        rows = flat.div(N, rounding_mode="floor")
        cols = flat - rows * N
        ih = i_hit[rows, cols]
        mat = P["mat"][ih] if mat_rgb is None else mat_rgb[rows, ih]
        if shader is not None:
            ray = dd[rows, cols]
            hit = oo[rows, cols] + t[rows, cols].unsqueeze(-1) * ray
            c = shader(rows, hit, n[rows, cols], -ray, ih, mat)
        else:
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
        base = plate.to(col.dtype).view(-1, H, W, 3)
        out.copy_(torch.clamp(torch.round(base * (1 - covf.unsqueeze(-1)) + col), 0, 255).to(torch.uint8))
        return out

    col = col.view(B, tile, ss, tile, ss, 3).sum((2, 4)) / (ss * ss)
    covf = covf.view(B, tile, ss, tile, ss).sum((2, 4)) / (ss * ss)
    out.copy_(plate.view(-1, H, W, 3).expand(B, H, W, 3))
    covf = covf.reshape(B, -1)
    # Uncovered output pixels already contain the plate byte; blend only the covered union.
    ps = torch.nonzero((covf > 0).any(0), as_tuple=False).reshape(-1)
    if ps.numel():
        oidx = obase[:, ps].unsqueeze(-1).expand(B, ps.numel(), 3)
        patch = torch.gather(plate.reshape(-1, H * W, 3).expand(B, H * W, 3), 1, oidx).to(col.dtype)
        merged = patch * (1 - covf[:, ps].unsqueeze(-1)) + col.reshape(B, -1, 3)[:, ps]
        out.view(B, H * W, 3).scatter_(1, oidx, torch.clamp(torch.round(merged), 0, 255).to(torch.uint8))
    return out


def _fit_tile(
    cam: Camera, prims: Primitives, xpos: torch.Tensor, H: int, W: int, tile: int, device, bound_scale: float = 1.0
):
    """Return the tightest square window within tile, or None if the bound does not fit."""
    span = screen_span(cam, prims, xpos, H, W, device, bound_scale)
    if span is None:
        return None
    x0, y0, w, h = span
    need = max(int(w), int(h), 1)
    if need > tile:
        return None
    return (x0.clamp(0, W - need), y0.clamp(0, H - need)), need


def _upsample_t(plate_t, H, W, ss):
    """Repeat static depth onto the supersample grid for mover occlusion tests."""
    return plate_t.view(H, W).repeat_interleave(ss, 0).repeat_interleave(ss, 1).reshape(-1)


def screen_bounds(
    cam: Camera, prims: Primitives, xpos: torch.Tensor, H: int, W: int, device, bound_scale: float = 1.0
):
    """Return per-row pixel bounds (x0, x1, y0, y1) and the projected corners, or None if unbounded.

    The bounds hold every sample that the primitives can cover, with one pixel of margin.
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
    return (x0, x1, y0, y1), ndc


def screen_span(cam: Camera, prims: Primitives, xpos: torch.Tensor, H: int, W: int, device, bound_scale: float = 1.0):
    """Return (x0, y0, width, height) for projected primitive bounds, or None if unbounded.

    Origins are per-row device tensors; extents are host integers covering the batch.
    Projects all eight bounding-box corners and synchronises once.
    """
    bounds = screen_bounds(cam, prims, xpos, H, W, device, bound_scale)
    if bounds is None:
        return None
    (x0, x1, y0, y1), ndc = bounds
    probe = torch.stack(
        [(x1 - x0).max(), (y1 - y0).max(), torch.isfinite(ndc).all().long()]
    ).cpu()  # the only synchronization
    if not int(probe[2]):
        return None
    return x0, y0, int(probe[0]), int(probe[1])


_TILE_CACHE_MAX = 16
_TILE_CACHE = _Memo(_TILE_CACHE_MAX, name="raycast3d.tile")


def _expand_tile(org, tile: int, ss: int, Wpix: int, device) -> torch.Tensor:
    """Return flat (B, (tile*ss)^2) sample indices into an image of width Wpix."""
    x0, y0 = org
    B = x0.shape[0]
    key = (int(tile), int(ss), int(Wpix), str(device))
    base = _TILE_CACHE.get(key)
    if base is None:
        T = tile * ss
        ar = torch.arange(T, device=device)
        base = _TILE_CACHE.put(key, (ar.view(T, 1) * Wpix + ar.view(1, T)).reshape(1, -1))
    return (y0.view(B, 1) * ss * Wpix + x0.view(B, 1) * ss) + base
