"""nvdiffrast as a MuJoCo batch rasterizer, with the two operations MJWarp's raytracer does not have.

``stable_envs/render/mjwarp.py`` is the 3-D backend this repo recommends and it is parity on
flat-material scenes. Two families are blocked on it, and both blockers are the same shape — a texture
that MJWarp cannot sample the way MuJoCo does:

1. **Minification.** ``mujoco_warp`` builds every ``wp.Texture2D`` at ``num_mip_levels=1`` and samples
   with no level of detail, so the far tenth of Gymnasium's locomotion checker floor comes out at full
   black-to-white contrast (luminance spread 92) where MuJoCo's mipmap has averaged it to one grey
   (spread 1.5-3.4), and 0.0017-0.0069 % of those pixels are bit-exact. That floor carries
   99.45-100.00 % of the interior error on all six domains
   (``docs/research/mujoco_locomotion.md`` Section 6).
2. **Texgen for primitives.** ``sample_texture`` derives UVs for ``PLANE`` and ``MESH`` only, so a
   textured ``BOX``/``CYLINDER``/``SPHERE`` samples texel (0,0) and comes out one flat colour. A
   RoboCasa kitchen has 178-401 of them (``docs/research/robocasa.md`` Section 7).

nvdiffrast addresses both **by construction** rather than by a fix: ``rasterize()`` returns image-space
barycentric derivatives, ``interpolate(..., diff_attrs=)`` turns them into ``d(u,v)/d(x,y)``, and
``texture(..., filter_mode="linear-mipmap-linear")`` does prefiltered sampling off an internally built
pyramid; and the UVs are ours to supply, so there is no geom type it cannot texture. What it does not
give you is a renderer — it knows nothing about MuJoCo, so everything between an ``MjModel`` and a
triangle in clip space is this module.

**What it is worth, measured** (``docs/research/nvdiffrast.md``). Switching ``filter_mode`` alone, same
pipeline either way, on the far tenth of Gymnasium's locomotion floor: **0.0000-0.0001 of pixels
bit-exact without the pyramid — MJWarp's regime exactly — and 0.66-0.81 with it**, luminance spread 95
falling to 3-9 against a reference of 0-3. And on RoboCasa, textured boxes go from 0.2-18 % of pixels
within 8 with texture modulation off to **57-99.9 %** with these UVs, reaching **0.976 of interior
pixels bit-exact** on the best of three kitchens. On OGBench, the gate, the whole frame is
**0.9984-0.9997 interior-exact**.

It is still not parity on locomotion (0.54-0.86 interior on the official scene) and it is **3-5x
slower than MJWarp at 224 px and 8-30x at 64 px**. Prefer ``mjwarp.BatchRenderer`` unless one of the
two operations above is what you need. The two numbers a caller needs before choosing this:

* **Geometry is per world.** nvdiffrast's instanced mode wants the full ``(B, V, 4)`` clip-space vertex
  buffer, so device memory grows with batch x scene vertex count, where MJWarp shares one BVH across
  worlds and stores per-geom transforms. That is what caps the usable batch, and it is a property of
  the API, not of this file.
* **Mipmapping needs power-of-two textures.** nvdiffrast asserts it. MuJoCo's own textures frequently
  are not (Gymnasium's checker is 100x100, OGBench's grid 300x300), and ``prepare_texture`` records
  what it had to do about that, because resampling the base level is a change to the picture and has
  to be declared rather than absorbed.

Conventions taken from measurement, each of which is silent when wrong:

* **The camera is the mono camera**, i.e. the average of ``mjv_updateScene``'s two ``mjvGLCamera``
  eyes; ``camera[0]`` alone is a 34 mm lateral offset that misplaces 0.7 % of pixels
  (``docs/research/ogbench.md`` Section 2). Building the frustum from ``cam_xpos``/``cam_xmat``/``fovy``
  reproduces the average exactly, because the two eyes' ``ipd`` offsets cancel.
* **nvdiffrast returns row 0 at the BOTTOM** (OpenGL) and MuJoCo returns it at the top, so the frame is
  flipped on the way out. Getting this wrong survives every symmetric test.
* **Round, do not truncate, on the way to uint8.** This is the same rule ``mjwarp.gl_rounding()``
  applies to MJWarp, where it was worth interior 0.791 -> 0.9988 on Fetch. nvdiffrast hands back floats
  and leaves the quantisation to the caller, so here it is simply done right; ``QUANT`` names it so a
  probe can measure the alternative instead of assuming.
* **Shading is the fixed-function composition, and the composition matters more than the constants:**
  ``primary = mat_rgb * (A + sum_l D_l max(0, n.l)) + sum_l S_l * spec * max(0, n.h)^p`` and then
  ``final = primary * tex_rgb``. The specular term is WHITE and ADDITIVE and the texture modulates the
  whole primary colour; a multiplicative fit on a textured floor extrapolates onto untextured geometry
  and overshoots wall tops by 6.5/255 (``docs/research/ogbench.md``). The half-vector is the
  **local-viewer** one, ``h = normalize(l + v)``, which scores RMS 0.0187 against 0.0569 for the
  infinite-viewer form.

What this module does NOT do, in the order those things cost pixels: **no shadows** (worth interior
0.427 -> 0.753 on HalfCheetah's near floor and nothing at all on three of the six domains), **no
transparency** (partially transparent geoms are depth-tested rather than blended in a second pass,
which is what caps RoboCasa; zero-alpha geoms ARE dropped), **no reflections** (measured worth:
nothing, on every scene here), and **no heightfields or SDF geoms** (skipped).

This module is deliberately env-agnostic and imports neither ``mujoco`` nor ``nvdiffrast`` at module
scope; both are looked up inside the functions that need them, so importing ``stable_envs`` on a machine
without either still works.

**Four generic extensions (Step 7), every one of them OFF by default so that every existing caller
renders byte-for-byte what it rendered before:**

* ``Scene.from_triangles`` — a scene from a triangle soup and a per-geom appearance table, with no
  ``MjModel``. The ``MjModel`` constructor is unchanged and builds the same members.
* ``BatchRenderer.render(..., mat_rgb=)`` — a PER-ROW ``(B, G, 3)`` material colour, one gather instead
  of one ``Scene`` per distinct colour (``gather_material``).
* ``BatchRenderer(shading="tiny", tiny=TinyShading(...))`` — Bullet's TinyRenderer fixed-function law
  beside MuJoCo's (``shade_tiny``): one world-fixed directional light, the specular term MULTIPLIED into
  the material colour with an infinite-viewer reflection whose exponent is a parameter, coefficients
  ``ambient + diffuse*max(0, n.l) + specular*max(0, r_z)^e``. Measured on real frames before it was
  written (``scratch/step7/rocket-landing/tiny_law_probe.py``, ``tiny_law_probe2.py``): the light is
  world-fixed (not head-lamp), the exponent is 2, ``r_z`` is the WORLD z of the reflection, and the
  output is TRUNCATED to uint8 (``quant="trunc"``).
* ``BatchRenderer(tex_filter="nearest")`` and ``sample_offset=(ox, oy)`` — unfiltered texel lookup, and
  a sub-pixel shift of the whole frame for renderers that sample pixel CORNERS rather than centres
  (TinyRenderer tests integer pixel coordinates, bottom-up; nvdiffrast tests ``i + 0.5``; ``(0.5, 0.5)``
  makes the two agree, measured 25/25 columns and 13/13 rows in ``tiny_law_probe.py``).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from . import _resources

#: mjtGeom codes, so a scene can be built from an MjModel with no translation table.
PLANE, HFIELD, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = 0, 1, 2, 3, 4, 5, 6, 7

#: Groups ``mujoco.Renderer`` shows at its default ``MjvOption``. Matching them on both sides is the
#: difference between comparing two renders of one scene and comparing two different scenes.
DEFAULT_GROUPS = (0, 1, 2)

#: MuJoCo's ``mjVisual.quality`` defaults. A sphere is 28 longitude slices by 16 latitude stacks, and
#: the silhouette of the tessellation is what the reference draws -- an analytic sphere is a DIFFERENT
#: shape, by up to half a pixel at the limb.
NUMSLICES, NUMSTACKS = 28, 16

#: Ray-direction grids a ``BatchRenderer`` keeps. One grid is ``R*R*n*4`` bytes, the working set is a
#: single camera orientation per call site, and the bound is the one the uncapped dict already used —
#: what changed is that the entry over the bound is the LEAST RECENTLY USED one rather than all of
#: them. See ``BatchRenderer._ray_dirs``.
RAY_CACHE_MAX = 8


# --------------------------------------------------------------------------- quantisation

def QUANT(x, mode: str = "round"):
    """float in [0,1] -> uint8. ``round`` is what the GL pipeline MuJoCo renders through does.

    Exposed as a function with a mode rather than inlined so that the alternative can be MEASURED. The
    same one-level rule was the single largest fidelity term on the MJWarp backend (Fetch interior
    0.791 -> 0.9988, ``mjwarp.gl_rounding``), and the only reason to believe it here is a measurement
    on this backend.
    """
    import torch
    y = torch.clamp(x, 0.0, 1.0) * 255.0
    if mode == "round":
        y = torch.round(y)
    elif mode == "trunc":
        y = torch.floor(y)
    else:
        raise ValueError(f"unknown quantisation mode {mode!r}")
    return y.to(torch.uint8)


# --------------------------------------------------------------------------- generic pieces
# Pure torch, no nvdiffrast, no mujoco: each of these is unit-tested on the CPU against numbers that were
# MEASURED on the renderer they emulate (``tests/unit/test_nvdiff_triangles.py``).


@dataclass
class TinyShading:
    """Bullet TinyRenderer's fixed-function law, as ``TinyRendererVisualShapeConverter::render`` applies it.

    ``out = base * (ambient + diffuse * max(0, n.l) + specular * max(0, r_z) ** exponent) * color`` with
    ``r = 2 (n.l) n - l`` the reflection of the light about the normal and ``r_z`` its WORLD z component
    (an infinite viewer looking down -z, whatever the camera does). ``dir`` points TOWARD the light and is
    normalised here. The defaults are Bullet's own for a z-up world: direction ``(-50, 30, 100)``,
    coefficients 0.6 / 0.35 / 0.05, white light, exponent 2 -- each reproduces the default frame with
    0 differing pixels when passed explicitly (``examples/results/step7/rocket-landing/phase0/
    phase0_light_sweep.json``; the exponent from ``tiny_law2.json``: 190/255 = 0.8639 ** 2).
    """

    dir: tuple = (-50.0, 30.0, 100.0)
    ambient: float = 0.6
    diffuse: float = 0.35
    specular: float = 0.05
    exponent: float = 2.0
    color: tuple = (1.0, 1.0, 1.0)

    def unit_dir(self) -> np.ndarray:
        d = np.asarray(self.dir, np.float64)
        return d / np.linalg.norm(d)


def shade_tiny(n, base, law: TinyShading):
    """``n`` unit WORLD normals ``(..., 3)``, ``base`` material colour ``(..., 3)`` -> unclamped colour."""
    import torch
    l = torch.as_tensor(law.unit_dir(), dtype=base.dtype, device=base.device)
    ndl = (n * l).sum(-1, keepdim=True)
    r = 2.0 * ndl * n - l
    rz = r[..., 2:3].clamp_min(0.0)
    term = law.ambient + law.diffuse * ndl.clamp_min(0.0) + law.specular * torch.pow(rz, law.exponent)
    col = torch.as_tensor(np.asarray(law.color, np.float32), dtype=base.dtype, device=base.device)
    return base * term * col


def gather_material(mat, gid):
    """``mat (G, 3)`` shared, or ``(B, G, 3)`` per row; ``gid (B, H, W)`` -> ``(B, H, W, 3)``.

    The per-row form is what makes a batch of B appearances ONE render: colour is otherwise a per-geom
    constant baked into the ``Scene`` (``mujoco_locomotion.py`` says why its visual tier refuses to offer
    the knob), and a different ``Scene`` per colour is a different rasterization per colour.
    """
    import torch
    if mat.dim() == 2:
        return mat[gid]
    b = torch.arange(gid.shape[0], device=gid.device).view(-1, 1, 1)
    return mat[b, gid]


def _mm4(a, b):
    """``a @ b`` for trailing ``(4, 4)`` matrices as four UNROLLED elementwise products summed in a fixed
    order. No matmul and no reduction kernel: each output element is computed by the same four
    multiplies and three adds whatever the leading (batch) shape, so the result cannot depend on how
    many other rows share the launch."""
    return (a[..., :, 0:1] * b[..., 0:1, :] + a[..., :, 1:2] * b[..., 1:2, :]
            + a[..., :, 2:3] * b[..., 2:3, :] + a[..., :, 3:4] * b[..., 3:4, :])


def mvp_rows(proj, view, M):
    """``(B, G, 4, 4)`` model-view-projection per row: ``proj @ view @ M``, BATCH-INVARIANT.

    ``proj (4,4)``, ``view (B,4,4)`` or ``(4,4)``, ``M (B,G,4,4)``. Computed in float64 with ``_mm4`` and
    cast once to ``M.dtype``, so the value is the correctly rounded product and is bitwise the same for
    a row whether it is rendered alone or in a batch of 4096.

    Why not ``(proj @ view)[:, None] @ M``: that is one cuBLAS batched GEMM over ``B*G`` 4x4 products,
    and cuBLAS picks its kernel -- hence its FMA and summation order -- by the batch count. Measured on
    an RTX 3090 (torch 2.14.0+cu126, TF32 off, ``PFRocketLanding``, 13 geoms): identical results for
    B <= 24 and a different rounding for B >= 31, 46 of a row's 208 entries off by up to 1.5e-5 at
    100-500 m coordinates, which moved 1282 clip-space vertex coordinates and flipped ~10 silhouette,
    internal-edge and truncation pixels per frame between a state rendered alone and the same state in
    a batch (``examples/results/step7/rocket-landing/tier/probe_batch/``, jobs 6407151/6407256). The
    sparse per-geom stage and the rasterizer were exact on identical input; the GEMM alone reproduced
    the split. Elementwise products have no such kernel choice.
    """
    import torch
    p = proj.to(torch.float64)
    v = view.to(torch.float64)
    vp = _mm4(p, v)                                                   # (B,4,4) or (4,4)
    vp = vp[:, None] if vp.ndim == 3 else vp[None, None]
    return _mm4(vp, M.to(torch.float64)).to(M.dtype)


def offset_clip(pos, R: int, offset) -> "torch.Tensor":
    """Shift clip-space vertices ``(B, V, 4)`` by ``offset`` PIXELS at resolution ``R`` (x right, y UP).

    nvdiffrast tests coverage at pixel centres ``(i + 0.5, j + 0.5)`` in a bottom-up frame; a rasterizer
    that tests integer corners ``(i, j)`` sees the same picture only if the geometry is moved by half a
    pixel first, which is what ``(0.5, 0.5)`` does. The shift is applied in clip space, scaled by ``w``,
    so it is exactly a screen-space translation after the divide.
    """
    ox, oy = float(offset[0]), float(offset[1])
    if ox == 0.0 and oy == 0.0:
        return pos
    out = pos.clone()
    out[..., 0] = out[..., 0] + ox * (2.0 / R) * out[..., 3]
    out[..., 1] = out[..., 1] + oy * (2.0 / R) * out[..., 3]
    return out


# --------------------------------------------------------------------------- tessellation
# MuJoCo draws primitives as triangle meshes, so matching it means matching the tessellation, not
# out-doing it with an analytic surface. Each builder returns (vert, norm, face) in the geom's own
# frame, already scaled by geom_size.


def _sphere(radii, nslice: int = NUMSLICES, nstack: int = NUMSTACKS):
    """Lat-long sphere, ``radii`` a 3-vector so this covers ELLIPSOID as well as SPHERE.

    Stacks run pole to pole and slices around; the seam is closed by reusing column 0, so the mesh is
    watertight and the limb is a ``nslice``-gon rather than a circle -- which is what the reference
    draws.
    """
    th = np.linspace(0.0, 2.0 * np.pi, nslice + 1)                 # closes on itself
    ph = np.linspace(0.0, np.pi, nstack + 1)
    T, P = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(P) * np.cos(T), np.sin(P) * np.sin(T), np.cos(P)], -1)
    v = n * np.asarray(radii, np.float64)
    nrm = n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)
    if not np.allclose(radii, radii[0]):                            # ellipsoid: normal is not the point
        nrm = v / np.asarray(radii, np.float64) ** 2
        nrm = nrm / np.linalg.norm(nrm, axis=-1, keepdims=True).clip(1e-12)
    idx = np.arange((nslice + 1) * (nstack + 1)).reshape(nslice + 1, nstack + 1)
    f = []
    for i in range(nslice):
        for j in range(nstack):
            a, b, c, d = idx[i, j], idx[i + 1, j], idx[i + 1, j + 1], idx[i, j + 1]
            f.append([a, b, c])
            f.append([a, c, d])
    return (v.reshape(-1, 3), nrm.reshape(-1, 3), np.asarray(f, np.int64))


def _capsule(r: float, hl: float, nslice: int = NUMSLICES, nstack: int = NUMSTACKS):
    """Cylinder of half-length ``hl`` about z, capped by two hemispheres of radius ``r``."""
    th = np.linspace(0.0, 2.0 * np.pi, nslice + 1)
    ct, st = np.cos(th), np.sin(th)
    V, N, F = [], [], []
    off = 0

    for z in (-hl, hl):                                             # barrel
        V.append(np.stack([r * ct, r * st, np.full(nslice + 1, z)], -1))
        N.append(np.stack([ct, st, np.zeros(nslice + 1)], -1))
    for i in range(nslice):
        a, b = i, i + 1
        c, d = (nslice + 1) + i + 1, (nslice + 1) + i
        F += [[a, b, c], [a, c, d]]
    off = 2 * (nslice + 1)

    half = max(nstack // 2, 1)
    for sign, zc in ((1.0, hl), (-1.0, -hl)):                       # hemispherical caps
        ang = np.linspace(0.0, np.pi / 2.0, half + 1)
        base = off
        for a in ang:
            rr, zz = r * np.cos(a), sign * r * np.sin(a)
            V.append(np.stack([rr * ct, rr * st, np.full(nslice + 1, zc + zz)], -1))
            N.append(np.stack([np.cos(a) * ct, np.cos(a) * st,
                               np.full(nslice + 1, sign * np.sin(a))], -1))
        for k in range(half):
            for i in range(nslice):
                a0 = base + k * (nslice + 1) + i
                b0 = a0 + 1
                c0 = base + (k + 1) * (nslice + 1) + i + 1
                d0 = c0 - 1
                F += [[a0, b0, c0], [a0, c0, d0]]
        off += (half + 1) * (nslice + 1)
    return np.concatenate(V), np.concatenate(N), np.asarray(F, np.int64)


def _cylinder(r: float, hl: float, nslice: int = NUMSLICES):
    th = np.linspace(0.0, 2.0 * np.pi, nslice + 1)
    ct, st = np.cos(th), np.sin(th)
    ring = lambda z: np.stack([r * ct, r * st, np.full(nslice + 1, z)], -1)  # noqa: E731
    side_n = np.stack([ct, st, np.zeros(nslice + 1)], -1)
    V = [ring(-hl), ring(hl), ring(-hl), ring(hl),
         np.array([[0.0, 0.0, -hl], [0.0, 0.0, hl]])]
    N = [side_n, side_n,
         np.tile([0.0, 0.0, -1.0], (nslice + 1, 1)), np.tile([0.0, 0.0, 1.0], (nslice + 1, 1)),
         np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]])]
    n1 = nslice + 1
    F = []
    for i in range(nslice):
        F += [[i, i + 1, n1 + i + 1], [i, n1 + i + 1, n1 + i]]
        F += [[4 * n1, 2 * n1 + i + 1, 2 * n1 + i]]                 # bottom cap
        F += [[4 * n1 + 1, 3 * n1 + i, 3 * n1 + i + 1]]             # top cap
    return np.concatenate(V), np.concatenate(N), np.asarray(F, np.int64)


def _box(half):
    """Six quads with per-face normals. Vertices are NOT shared between faces: a shared corner would

    average three normals into something no face has, and a box's whole appearance is its flat faces.
    """
    s = np.asarray(half, np.float64)
    V, N, F = [], [], []
    for ax in range(3):
        u, w = (ax + 1) % 3, (ax + 2) % 3
        for sgn in (1.0, -1.0):
            quad = np.zeros((4, 3))
            for k, (p, q) in enumerate([(-1, -1), (1, -1), (1, 1), (-1, 1)]):
                quad[k, ax] = sgn * s[ax]
                quad[k, u] = p * s[u]
                quad[k, w] = q * s[w]
            nrm = np.zeros(3)
            nrm[ax] = sgn
            base = 4 * len(F) // 2
            V.append(quad)
            N.append(np.tile(nrm, (4, 1)))
            # wind so the face is front-facing seen from +normal
            tri = [[base, base + 1, base + 2], [base, base + 2, base + 3]]
            F += tri if sgn > 0 else [t[::-1] for t in tri]
    return np.concatenate(V), np.concatenate(N), np.asarray(F, np.int64)


def _plane(size, extent: float):
    """A MuJoCo plane is infinite where ``size`` is 0; the renderer draws a large finite quad."""
    hx = size[0] if size[0] > 0 else 20.0 * extent
    hy = size[1] if size[1] > 0 else 20.0 * extent
    v = np.array([[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0]])
    n = np.tile([0.0, 0.0, 1.0], (4, 1))
    return v, n, np.array([[0, 1, 2], [0, 2, 3]], np.int64)


# --------------------------------------------------------------------------- appearance resolution

def effective_rgba(m, g: int) -> np.ndarray:
    """``(4,)`` the colour MuJoCo draws geom ``g`` with, resolved MuJoCo's way rather than MJWarp's.

    A geom's own ``rgba`` wins unless it is the compiler default ``(0.5, 0.5, 0.5, 1)``, in which case
    its material supplies it. Reading the material first is the rule MJWarp follows and it is wrong by
    13-19/255 on the Gymnasium locomotion floor, whose ``MatPlane`` has no ``rgba`` while the floor
    geom is ``0.8 0.9 0.8``.
    """
    rgba = np.asarray(m.geom_rgba[g], np.float64).copy()
    mid = int(m.geom_matid[g])
    if mid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
        rgba = np.asarray(m.mat_rgba[mid], np.float64).copy()
    return rgba


def _tex_of(m, mid: int) -> int:
    """The RGB texture id of material ``mid``, across MuJoCo versions that made ``mat_texid`` a vector."""
    if mid < 0:
        return -1
    t = m.mat_texid[mid]
    if np.ndim(t) == 0:
        return int(t)
    import mujoco
    role = int(getattr(mujoco.mjtTextureRole, "mjTEXROLE_RGB", 1))
    return int(t[role]) if role < len(t) else int(np.max(t))


def _is_cube(m, g: int) -> bool:
    """Does geom ``g`` wear a CUBE or SKYBOX texture? Decides whether texgen makes a uv or a direction."""
    t = _tex_of(m, int(m.geom_matid[g]))
    return t >= 0 and int(m.tex_type[t]) != 0


def texture_array(m, tex: int) -> np.ndarray:
    """``(h, w, 3)`` uint8 for texture ``tex``, from whichever attribute this MuJoCo calls it."""
    buf = getattr(m, "tex_data", None)
    if buf is None:
        buf = m.tex_rgb
    w, h = int(m.tex_width[tex]), int(m.tex_height[tex])
    nc = int(m.tex_nchannel[tex]) if hasattr(m, "tex_nchannel") else 3
    a0 = int(m.tex_adr[tex])
    a = np.asarray(buf[a0:a0 + w * h * nc], np.uint8).reshape(h, w, nc)
    return np.repeat(a, 3, -1) if nc == 1 else a[..., :3]


# --------------------------------------------------------------------------- texgen
# MuJoCo generates texture coordinates for primitives with OBJECT_LINEAR texgen: an affine map from
# the geom's own frame to (s, t, r). Everything below is an affine map, and the CONSTANTS in it are
# calibrated in scratch/nvdiff/probe_texgen.py against MuJoCo's own frames rather than read off a
# source file -- every constant in this repo that was read rather than measured turned out wrong.
#
# `texuniform` is the one qualitative switch: false means the repeat count spans the object, true means
# it is per unit of length.


def half_extent(m, g: int, drawn_plane: float = 0.0) -> np.ndarray:
    """``(3,)`` the per-axis half size ``texuniform="false"`` divides by, per geom type.

    It is NOT ``geom_size`` for every type, and the exception was measured (``probe_texgen3.py``): a
    capsule and a cylinder use their RADIUS on both texture axes rather than their half-length, so a
    long thin capsule is not stretched along its own axis. Everything else -- box, ellipsoid, sphere,
    plane -- uses the per-axis size, verified against plane sizes 0.5 to 32 with the scale recovered to
    within 0.05 % and independent of ``stat.extent`` and of where the plane sits
    (``probe_plane.py``).
    """
    s = np.asarray(m.geom_size[g], np.float64)
    t = int(m.geom_type[g])
    if t == PLANE:
        hx = s[0] if s[0] > 0 else drawn_plane
        hy = s[1] if s[1] > 0 else drawn_plane
        return np.array([max(hx, 1e-12), max(hy, 1e-12), 1.0])
    if t == SPHERE:
        return np.array([s[0], s[0], s[0]])
    if t in (CAPSULE, CYLINDER):
        return np.array([s[0], s[0], max(s[1], 1e-12)])
    return np.where(s > 0, s, 1.0)                                  # BOX, ELLIPSOID, MESH


#: The texgen offset. One number for every geom type: measured 0.4966-0.4999 on planes, boxes,
#: spheres, cylinders and capsules across sizes and repeats.
TEXGEN_OFFSET = 0.5


def texgen(m, g: int, drawn_plane: float = 0.0, cube: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """``(A (3,3), b (3,))`` mapping a geom-local position to texture coordinates ``A @ p + b``.

    MuJoCo generates texture coordinates for every primitive by ONE object-linear projection along the
    geom's local z: ``u`` is a function of local x alone and ``v`` of local y alone, on every face of
    every type. That is not what a reading of the code suggests (a per-face unwrap, or a cube map), and
    it is not what MJWarp implements (it derives no UVs for anything but planes and meshes); it is what
    a texture painted with its own texel index shows when you read the frame back --
    ``scratch/nvdiff/probe_texgen{2,3}.py``. It is also independent of ``numquads``, which was the
    tessellation hypothesis and is refuted (slope identical at 1, 2, 4 and 8).

        u = +0.5 * repeat_x * x * scale_x + offset
        v = -0.5 * repeat_y * y * scale_y + offset

    with ``scale = 1`` when ``texuniform`` (repeat per unit length) and ``1/half_extent`` otherwise
    (repeat across the object), and ``offset`` 0.5 except on a plane, where it is 0.25.

    Measured slopes against the law: box (0.4,0.3,0.2) repeat 0.5 -> 0.6241 / -0.8350 (law 0.625 /
    -0.8333); sphere r=0.4 -> 0.6209 / -0.6273; capsule r=0.3 -> 0.8289 / -0.8358; planes of half-size
    0.5 to 32 recover ``scale = size`` to within 0.05 %; OGBench's real 16 m floor, measured through
    its own env rather than a toy, gives 0.031256 against the law's 0.031250 with offset 0.49988.

    One warning about how these were obtained, because it cost an afternoon: a slope read off a
    WRAPPED texture coordinate comes back at exactly half its true value whenever the unwrap misses a
    period, and half a slope is a plausible-looking number. The synthetic planes first read as
    ``0.5*repeat/(2*size)`` with offset 0.25 for precisely that reason, and the real floor -- where
    ``texrepeat`` is small enough that u never leaves [0,1] -- says ``0.5*repeat/size`` with offset
    0.5. Measure texgen with a repeat that cannot wrap.
    """
    mid = int(m.geom_matid[g])
    rep = np.asarray(m.mat_texrepeat[mid], np.float64) if mid >= 0 else np.ones(2)
    uniform = bool(m.mat_texuniform[mid]) if mid >= 0 else False
    h = half_extent(m, g, drawn_plane)
    scale = np.ones(3) if uniform else 1.0 / h
    A = np.zeros((3, 3))
    A[0, 0] = 0.5 * rep[0] * scale[0]
    A[1, 1] = -0.5 * rep[1] * scale[1]
    if cube:
        # A cube map is indexed by a DIRECTION, so the offset that centres a 2D map would tilt every
        # lookup toward one corner. The direction is the local position; only its orientation is read,
        # so the repeat and the scale drop out too.
        return np.eye(3), np.zeros(3)
    return A, np.array([TEXGEN_OFFSET, TEXGEN_OFFSET, 0.0])


# --------------------------------------------------------------------------- scene

@dataclass
class Lights:
    """The lights as ``mjv_updateScene`` RESOLVES them, plus the headlight it attaches to the camera.

    Read off ``mjvScene.lights`` rather than off the XML: a model light inherits defaults, OGBench
    deletes one light's ambient in pixels mode, and the headlight is not in the model at all. ``dirs``
    point FROM the light.
    """

    pos: np.ndarray                 # (L,3)
    dir: np.ndarray                 # (L,3) unit, direction the light travels
    directional: np.ndarray         # (L,) bool
    ambient: np.ndarray             # (L,3)
    diffuse: np.ndarray             # (L,3)
    specular: np.ndarray            # (L,3)
    attenuation: np.ndarray         # (L,3)
    cutoff: np.ndarray              # (L,)
    exponent: np.ndarray            # (L,)
    headlight: np.ndarray           # (L,) bool -- pos/dir follow the camera, per frame

    @staticmethod
    def from_scene(scn) -> "Lights":
        n = int(scn.nlight)
        g = lambda a: np.asarray(a, np.float64)                      # noqa: E731

        def flag(i, name, fallback):
            """``mjvLight`` gained a ``type`` enum where it used to carry booleans; read either."""
            li = scn.lights[i]
            if hasattr(li, name):
                return bool(getattr(li, name))
            return bool(fallback(li))

        return Lights(
            pos=g([scn.lights[i].pos for i in range(n)]).reshape(n, 3),
            dir=g([scn.lights[i].dir for i in range(n)]).reshape(n, 3),
            directional=np.array([flag(i, "directional", lambda li: int(li.type) == 1)
                                  for i in range(n)]),
            ambient=g([scn.lights[i].ambient for i in range(n)]).reshape(n, 3),
            diffuse=g([scn.lights[i].diffuse for i in range(n)]).reshape(n, 3),
            specular=g([scn.lights[i].specular for i in range(n)]).reshape(n, 3),
            attenuation=g([scn.lights[i].attenuation for i in range(n)]).reshape(n, 3),
            cutoff=g([scn.lights[i].cutoff for i in range(n)]).reshape(n),
            exponent=g([scn.lights[i].exponent for i in range(n)]).reshape(n),
            headlight=np.array([flag(i, "headlight", lambda li: False) for i in range(n)]),
        )


@dataclass
class Camera:
    """A pinhole in MuJoCo's terms. ``mat`` columns are right / up / BACKWARD (cameras look down -z).

    ``pos`` and ``mat`` may carry a leading batch dimension, which is what a tracking camera needs.
    """

    pos: np.ndarray
    mat: np.ndarray
    fovy: float
    znear: float
    zfar: float

    @staticmethod
    def from_model(m, d, cam: int) -> "Camera":
        znear = float(m.vis.map.znear) * float(m.stat.extent)
        zfar = float(m.vis.map.zfar) * float(m.stat.extent)
        return Camera(np.asarray(d.cam_xpos[cam], np.float64).reshape(3),
                      np.asarray(d.cam_xmat[cam], np.float64).reshape(3, 3),
                      float(m.cam_fovy[cam]), znear, zfar)

    @staticmethod
    def from_glcam(m, scn) -> "Camera":
        """The MONO camera: the average of ``mjv_updateScene``'s two eyes.

        This is the one a free ``MjvCamera`` (OGBench's top-down view) only exists as. The eyes'
        ``ipd`` offsets cancel in the average, which is why this is a camera and not a compromise.
        """
        a, b = scn.camera[0], scn.camera[1]
        pos = 0.5 * (np.asarray(a.pos, np.float64) + np.asarray(b.pos, np.float64))
        fwd = 0.5 * (np.asarray(a.forward, np.float64) + np.asarray(b.forward, np.float64))
        up = 0.5 * (np.asarray(a.up, np.float64) + np.asarray(b.up, np.float64))
        fwd = fwd / np.linalg.norm(fwd)
        right = np.cross(fwd, up)
        right = right / np.linalg.norm(right)
        up = np.cross(right, fwd)
        near = 0.5 * (float(a.frustum_near) + float(b.frustum_near))
        top = 0.5 * (float(a.frustum_top) + float(b.frustum_top))
        fovy = 2.0 * math.degrees(math.atan2(top, near))
        return Camera(pos, np.stack([right, up, -fwd], -1), fovy, near,
                      float(m.vis.map.zfar) * float(m.stat.extent))

    def matrices(self, aspect: float = 1.0):
        """``(view (...,4,4), proj (4,4))``. OpenGL clip convention, which is what nvdiffrast wants."""
        pos = np.atleast_2d(np.asarray(self.pos, np.float64))
        mat = np.asarray(self.mat, np.float64).reshape(-1, 3, 3)
        n = max(len(pos), len(mat))
        pos = np.broadcast_to(pos, (n, 3))
        mat = np.broadcast_to(mat, (n, 3, 3))
        view = np.zeros((n, 4, 4))
        view[:, :3, :3] = np.transpose(mat, (0, 2, 1))
        view[:, :3, 3] = -np.einsum("bji,bj->bi", mat, pos)
        view[:, 3, 3] = 1.0
        f = 1.0 / math.tan(math.radians(self.fovy) / 2.0)
        zn, zf = self.znear, self.zfar
        proj = np.zeros((4, 4))
        proj[0, 0] = f / aspect
        proj[1, 1] = f
        proj[2, 2] = (zf + zn) / (zn - zf)
        proj[2, 3] = 2.0 * zf * zn / (zn - zf)
        proj[3, 2] = -1.0
        return view, proj


class Scene(_resources.Closeable):
    """Everything static: one triangle soup, per-geom appearance, and the texture pyramids.

    Built once per ``MjModel``. A frame then needs only ``geom_xpos``/``geom_xmat``, which is what makes
    the render a pure function of a compact state.

    **Ownership.** A ``Scene`` owns device memory and nothing else — the vertex/normal/uv/index
    tensors, the sparse transform matrices, the per-geom appearance vectors and the texture pyramids.
    No context, no EGL, nothing shared. ``close()`` drops them; the ``MjModel`` it was built from is
    the caller's and is not referenced after construction. A ``Scene`` handed to a ``BatchRenderer``
    with ``scene=`` stays the CALLER's to close (see ``BatchRenderer``); one the renderer builds
    itself is closed with it.
    """

    resource_kind = "nvdiff.Scene"

    def __init__(self, m, groups=DEFAULT_GROUPS, device="cuda", nslice: int = NUMSLICES,
                 nstack: int = NUMSTACKS, skip_alpha_zero: bool = True, tex_policy: str = "pot"):
        import torch
        self.groups = tuple(groups)
        self.tex_policy = tex_policy
        self.device = device
        V, N, U, GI, F, FG = [], [], [], [], [], []
        self.geoms: list[int] = []
        nv = 0
        extent = float(m.stat.extent)
        for g in range(m.ngeom):
            if int(m.geom_group[g]) not in self.groups:
                continue
            rgba = effective_rgba(m, g)
            if skip_alpha_zero and rgba[3] <= 0.0:
                continue
            t = int(m.geom_type[g])
            sz = np.asarray(m.geom_size[g], np.float64)
            if t == PLANE:
                v, n, f = _plane(sz, extent)
            elif t == SPHERE:
                v, n, f = _sphere([sz[0]] * 3, nslice, nstack)
            elif t == ELLIPSOID:
                v, n, f = _sphere(sz[:3], nslice, nstack)
            elif t == CAPSULE:
                v, n, f = _capsule(float(sz[0]), float(sz[1]), nslice, nstack)
            elif t == CYLINDER:
                v, n, f = _cylinder(float(sz[0]), float(sz[1]), nslice)
            elif t == BOX:
                v, n, f = _box(sz[:3])
            elif t == MESH:
                v, n, f, uv_mesh = self._mesh(m, g)
            else:
                continue                                            # hfield / sdf: not handled
            if t == MESH and uv_mesh is not None:
                # A mesh carries its own texcoords; generating them would overwrite the asset's UV
                # layout with a projection, which is exactly the artefact this module exists to avoid.
                uv = np.concatenate([uv_mesh, np.zeros((len(uv_mesh), 1))], -1)
            else:
                A, b = texgen(m, g, drawn_plane=20.0 * extent, cube=_is_cube(m, g))
                uv = v @ A.T + b
            V.append(v)
            N.append(n)
            U.append(uv)
            GI.append(np.full(len(v), len(self.geoms), np.int64))
            F.append(f + nv)
            FG.append(np.full(len(f), len(self.geoms), np.int64))
            nv += len(v)
            self.geoms.append(g)

        dev = torch.device(device)
        self._assemble(np.concatenate(V), np.concatenate(N), np.concatenate(U), np.concatenate(GI),
                       np.concatenate(F), np.concatenate(FG), dev)
        self._appearance(m, dev)
        self._sparse(dev)
        self._opened()

    def _release(self) -> None:
        """Drop every device tensor this scene holds. Idempotent; frees nothing shared.

        Listed by name rather than swept with ``vars()`` so that a member added later is a deliberate
        decision about its lifetime instead of silently retained (or silently freed).
        """
        for name in ("vert", "norm", "uv", "vgeom", "face", "fgeom", "mjgeom", "S3", "S4", "S3p",
                     "mat_rgb", "alpha", "spec", "shininess", "emission", "texid"):
            if hasattr(self, name):
                setattr(self, name, None)
        self.textures = {}

    def _assemble(self, vert, norm, uv, vgeom, face, fgeom, dev) -> None:
        import torch
        self.vert = torch.as_tensor(vert, dtype=torch.float32, device=dev)
        self.norm = torch.as_tensor(norm, dtype=torch.float32, device=dev)
        self.uv = torch.as_tensor(uv, dtype=torch.float32, device=dev)
        self.vgeom = torch.as_tensor(vgeom, dtype=torch.int64, device=dev)
        self.face = torch.as_tensor(face, dtype=torch.int32, device=dev)
        self.fgeom = torch.as_tensor(fgeom, dtype=torch.int64, device=dev)
        self.V, self.G = int(self.vert.shape[0]), len(self.geoms)
        self.mjgeom = torch.as_tensor(np.asarray(self.geoms, np.int64), device=dev)

    @classmethod
    def from_triangles(cls, vert, norm, face, vert_geom, rgba, *, uv=None, specular=None, shininess=None,
                       emission=None, textures=None, tex_of_geom=None, device="cuda",
                       tex_policy: str = "pot") -> "Scene":
        """A scene from a triangle soup, with no ``MjModel``: the same members, the same render.

        ``vert (V,3)`` in each geom's OWN frame, ``norm (V,3)``, ``face (F,3)`` indices, ``vert_geom (V,)``
        the geom each vertex belongs to (``0..G-1``, every vertex of a triangle in the same geom),
        ``rgba (G,4)``; optional ``uv (V,2|3)``, per-geom ``specular``/``shininess``/``emission`` ``(G,)``,
        ``textures {tex_id: HxWx3|4 array}`` and ``tex_of_geom (G,)`` (``-1`` for none). Geoms are drawn
        in the order given and ``render`` expects ``xpos``/``xmat`` in that order.

        This is the entry a scene that is not a MuJoCo model uses -- a URDF's primitives, an OBJ -- and
        the reason it is a constructor on this class rather than a new one is that everything after
        assembly (the sparse transform, the appearance gather, the texture pyramids) is shared and must
        stay shared, so a fix to one path is a fix to both.
        """
        import torch
        self = cls.__new__(cls)
        self.groups = ()
        self.tex_policy = tex_policy
        self.device = device
        vert = np.asarray(vert, np.float64).reshape(-1, 3)
        norm = np.asarray(norm, np.float64).reshape(-1, 3)
        face = np.asarray(face, np.int64).reshape(-1, 3)
        vgeom = np.asarray(vert_geom, np.int64).reshape(-1)
        rgba = np.asarray(rgba, np.float64).reshape(-1, 4)
        G = len(rgba)
        if len(vgeom) != len(vert) or len(norm) != len(vert):
            raise ValueError(f"vert {vert.shape}, norm {norm.shape} and vert_geom {vgeom.shape} must agree")
        if vgeom.min() < 0 or vgeom.max() >= G:
            raise ValueError(f"vert_geom must index rgba's {G} geoms")
        fgeom = vgeom[face[:, 0]]
        if not (np.all(vgeom[face[:, 1]] == fgeom) and np.all(vgeom[face[:, 2]] == fgeom)):
            raise ValueError("every triangle must lie in one geom")
        if uv is None:
            uv3 = np.zeros((len(vert), 3))
        else:
            uv = np.asarray(uv, np.float64).reshape(len(vert), -1)
            uv3 = np.concatenate([uv, np.zeros((len(vert), 3 - uv.shape[1]))], -1) if uv.shape[1] < 3 else uv
        self.geoms = list(range(G))
        dev = torch.device(device)
        self._assemble(vert, norm, uv3, vgeom, face, fgeom, dev)
        z = np.zeros(G)
        self.mat_rgb = torch.as_tensor(rgba[:, :3], dtype=torch.float32, device=dev)
        self.alpha = torch.as_tensor(rgba[:, 3], dtype=torch.float32, device=dev)
        self.spec = torch.as_tensor(z if specular is None else np.asarray(specular, np.float64),
                                    dtype=torch.float32, device=dev)
        self.shininess = torch.as_tensor(z if shininess is None else np.asarray(shininess, np.float64),
                                         dtype=torch.float32, device=dev)
        self.emission = torch.as_tensor(z if emission is None else np.asarray(emission, np.float64),
                                        dtype=torch.float32, device=dev)
        texid = np.full(G, -1, np.int64) if tex_of_geom is None else np.asarray(tex_of_geom, np.int64)
        self.texid = torch.as_tensor(texid, device=dev)
        self.texid_np = texid
        self.texcube_np = np.zeros(G, bool)
        self.textures = {}
        for t in sorted({int(x) for x in texid if x >= 0}):
            if textures is None or t not in textures:
                raise ValueError(f"tex_of_geom names texture {t} but textures has no such entry")
            self.textures[t] = prepare_texture_array(np.asarray(textures[t]), dev, tex_policy)
        self._sparse(dev)
        self._opened()
        return self

    # -- geometry ----------------------------------------------------------

    @staticmethod
    def _mesh(m, g: int):
        """A MuJoCo mesh, DE-INDEXED to one vertex per triangle corner.

        MuJoCo indexes position, normal and texcoord with three SEPARATE per-face index arrays
        (``mesh_face``, ``mesh_facenormal``, ``mesh_facetexcoord``), which is how a hard edge and a UV
        seam are expressed. There is no way to carry that on shared vertices, and both cheap
        alternatives are wrong pictures: reusing the position indices for normals smooths edges MuJoCo
        draws sharp, and recomputing area-weighted vertex normals is a different surface. De-indexing
        costs 3 vertices per triangle and is exact.
        """
        did = int(m.geom_dataid[g])
        va, vn = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
        fa, fn = int(m.mesh_faceadr[did]), int(m.mesh_facenum[did])
        vert = np.asarray(m.mesh_vert[va:va + vn], np.float64).reshape(-1, 3)
        face = np.asarray(m.mesh_face[fa:fa + fn], np.int64).reshape(-1, 3)
        na = int(m.mesh_normaladr[did]) if hasattr(m, "mesh_normaladr") else va
        nn = int(m.mesh_normalnum[did]) if hasattr(m, "mesh_normalnum") else vn
        nrm = np.asarray(m.mesh_normal[na:na + nn], np.float64).reshape(-1, 3)
        fnrm = (np.asarray(m.mesh_facenormal[fa:fa + fn], np.int64).reshape(-1, 3)
                if hasattr(m, "mesh_facenormal") else face)

        v = vert[face.reshape(-1)]
        n = nrm[np.clip(fnrm.reshape(-1), 0, len(nrm) - 1)] if len(nrm) else np.zeros_like(v)
        if not len(nrm):
            e1 = vert[face[:, 1]] - vert[face[:, 0]]
            e2 = vert[face[:, 2]] - vert[face[:, 0]]
            n = np.repeat(np.cross(e1, e2), 3, 0)
            n = n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)

        uv = None
        ta = int(m.mesh_texcoordadr[did]) if hasattr(m, "mesh_texcoordadr") else -1
        if ta >= 0 and hasattr(m, "mesh_texcoord") and len(m.mesh_texcoord):
            tn = int(m.mesh_texcoordnum[did]) if hasattr(m, "mesh_texcoordnum") else 0
            tc = np.asarray(m.mesh_texcoord[ta:ta + tn], np.float64).reshape(-1, 2)
            ftc = (np.asarray(m.mesh_facetexcoord[fa:fa + fn], np.int64).reshape(-1, 3)
                   if hasattr(m, "mesh_facetexcoord") else face)
            if len(tc):
                uv = tc[np.clip(ftc.reshape(-1), 0, len(tc) - 1)]
        f = np.arange(3 * len(face), dtype=np.int64).reshape(-1, 3)
        return v, n, f, uv

    def _sparse(self, dev):
        """``S4``/``S3``: the per-vertex gather of its geom's transform, folded into a sparse matmul.

        ``xmat[b, vgeom] @ vert`` as a gather materialises a ``(B, V, 3, 3)`` tensor, which is nine
        floats per vertex per world and is the dominant allocation at any interesting batch size. As a
        sparse product it never exists: ``S4`` is ``(V, 4G)`` holding each vertex's homogeneous local
        position in its own geom's four columns, so one ``spmm`` against the stacked per-geom matrices
        produces every world's clip-space vertex directly.
        """
        import torch
        v4 = torch.cat([self.vert, torch.ones(self.V, 1, device=dev)], -1)
        rows = torch.arange(self.V, device=dev).repeat_interleave(4)
        cols = (self.vgeom[:, None] * 4 + torch.arange(4, device=dev)[None]).reshape(-1)
        self.S4 = torch.sparse_coo_tensor(torch.stack([rows, cols]), v4.reshape(-1),
                                          (self.V, 4 * self.G)).coalesce().to_sparse_csr()
        rows3 = torch.arange(self.V, device=dev).repeat_interleave(3)
        cols3 = (self.vgeom[:, None] * 3 + torch.arange(3, device=dev)[None]).reshape(-1)
        self.S3 = torch.sparse_coo_tensor(torch.stack([rows3, cols3]), self.norm.reshape(-1),
                                          (self.V, 3 * self.G)).coalesce().to_sparse_csr()
        self.S3p = torch.sparse_coo_tensor(torch.stack([rows3, cols3]), self.vert.reshape(-1),
                                           (self.V, 3 * self.G)).coalesce().to_sparse_csr()

    # -- appearance --------------------------------------------------------

    def _appearance(self, m, dev):
        import torch
        G = self.G
        rgb = np.zeros((G, 3))
        alpha = np.ones(G)
        spec = np.zeros(G)
        shin = np.zeros(G)
        emis = np.zeros(G)
        texid = np.full(G, -1, np.int64)
        texcube = np.zeros(G, bool)
        for i, g in enumerate(self.geoms):
            rgba = effective_rgba(m, g)
            rgb[i] = rgba[:3]
            alpha[i] = rgba[3]
            mid = int(m.geom_matid[g])
            if mid >= 0:
                spec[i] = float(m.mat_specular[mid])
                shin[i] = float(m.mat_shininess[mid])
                emis[i] = float(m.mat_emission[mid])
                t = _tex_of(m, mid)
                if t >= 0:
                    texid[i] = t
                    texcube[i] = int(m.tex_type[t]) != 0            # 0 == mjTEXTURE_2D
        self.mat_rgb = torch.as_tensor(rgb, dtype=torch.float32, device=dev)
        self.alpha = torch.as_tensor(alpha, dtype=torch.float32, device=dev)
        self.spec = torch.as_tensor(spec, dtype=torch.float32, device=dev)
        self.shininess = torch.as_tensor(shin, dtype=torch.float32, device=dev)
        self.emission = torch.as_tensor(emis, dtype=torch.float32, device=dev)
        self.texid = torch.as_tensor(texid, device=dev)
        self.texcube_np = texcube
        self.texid_np = texid

        self.textures: dict[int, dict] = {}
        for t in sorted({int(x) for x in texid if x >= 0}):
            self.textures[t] = prepare_texture(m, t, dev, self.tex_policy)

    # -- transforms --------------------------------------------------------

    def clip(self, xpos, xmat, view, proj):
        """``(B, V, 4)`` clip-space vertices, ``(B, V, 3)`` world normals and world positions."""
        import torch
        B, G = xpos.shape[0], self.G
        M = torch.zeros(B, G, 4, 4, device=xpos.device, dtype=xpos.dtype)
        M[:, :, :3, :3] = xmat
        M[:, :, :3, 3] = xpos
        M[:, :, 3, 3] = 1.0
        mvp = mvp_rows(proj, view, M)                                 # batch-invariant, see mvp_rows
        pos = torch.sparse.mm(self.S4, mvp.permute(1, 3, 0, 2).reshape(4 * G, 4 * B))
        pos = pos.reshape(self.V, B, 4).permute(1, 0, 2).contiguous()
        nrm = torch.sparse.mm(self.S3, xmat.permute(1, 3, 0, 2).reshape(3 * G, 3 * B))
        nrm = nrm.reshape(self.V, B, 3).permute(1, 0, 2).contiguous()
        wp = torch.sparse.mm(self.S3p, xmat.permute(1, 3, 0, 2).reshape(3 * G, 3 * B))
        wp = wp.reshape(self.V, B, 3).permute(1, 0, 2) + xpos[:, self.vgeom]
        return pos, nrm, wp.contiguous()


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def prepare_texture(m, t: int, dev, policy: str = "pot") -> dict:
    """A texture ready for ``dr.texture``, plus what had to be done to it to make mipmapping legal.

    nvdiffrast asserts power-of-two dimensions for any mipmapped filter mode. MuJoCo's textures often
    are not -- Gymnasium's locomotion checker is 100x100, OGBench's grid 300x300 -- so the base level
    is resampled, and ``resampled`` records it so the residual it causes is declared rather than
    absorbed into "the renderer disagrees a bit".

    ``mjTEXTURE_CUBE`` / ``mjTEXTURE_SKYBOX`` arrive as a vertical strip of six faces in OpenGL cube
    order (+X, -X, +Y, -Y, +Z, -Z) and are split into the ``(6, h, w, 3)`` nvdiffrast wants.
    """
    a = texture_array(m, t).astype(np.float32) / 255.0
    kind = int(m.tex_type[t])
    return prepare_texture_array(a, dev, policy, cube=kind != 0)


def prepare_texture_array(a: np.ndarray, dev, policy: str = "pot", cube: bool = False) -> dict:
    """``prepare_texture`` for an array: float ``[0,1]`` ``HxWx3`` (a uint8 ``HxWx3|4`` is converted).

    The MuJoCo path calls this after reading the model's texel block; a scene built with
    ``Scene.from_triangles`` calls it directly. One function so the power-of-two policy and the cube-strip
    handling cannot drift between the two.
    """
    import torch
    a = np.asarray(a)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    a = np.ascontiguousarray(a, dtype=np.float32)
    if cube:
        # MuJoCo stores a cube/skybox texture as a VERTICAL STRIP of six square faces -- except when
        # the model gave it one image for all six sides, in which case it stores that image once. A
        # blind ``reshape(6, h//6, w, 3)`` is right for the first and raises on the second, which is
        # how a RoboCasa kitchen (one 2048x2048 cube texture) took the whole benchmark down.
        h, w = a.shape[0], a.shape[1]
        if h == 6 * w:
            a = a.reshape(6, w, w, 3)
        elif h == w:
            a = np.repeat(a[None], 6, 0)
        else:                                                       # not a layout we can name
            cube = False
    hw = a.shape[-3:-1]
    ok = is_pow2(hw[0]) and is_pow2(hw[1])
    resampled = None
    # A cube map is sampled with a plain linear filter (a skybox is at infinity and a cube-mapped
    # primitive is small), so it never needs a pyramid and must NOT be resampled for one: on
    # Gymnasium's locomotion skybox, resampling 100x100 to 128x128 costs interior exactness 1.0000 ->
    # 0.8329 and mean 0.04 -> 0.42 over 53 % of the frame, for nothing.
    if not ok and policy == "pot" and not cube:
        a, resampled = _to_pow2(a)
        ok = True
    tex = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    return {"tex": tex, "cube": cube, "resampled": resampled, "mipmap_ok": ok,
            "shape": tuple(a.shape)}


def _to_pow2(a: np.ndarray):
    """Nearest-neighbour resample to power-of-two dims. Returns ``(array, note)``.

    Needed because nvdiffrast refuses to build a pyramid over an odd extent ("cannot downsample an odd
    extent greater than 1"), which rules out 300x300 (OGBench's grid), 100x100 (Gymnasium's checker)
    and 127x762 (MuJoCo's ``texgeom``) -- and MuJoCo itself does NOT resample them: a 300x300 texture
    painted with its own texel index shows 297 of its 300 rows in a magnified frame.

    So this is a DECLARED change to the picture, not a neutral implementation detail. What it costs
    depends entirely on the texture: on a 2x2 checker the nearest resample 100 -> 128 reproduces the
    pattern exactly (every source block maps to a whole destination block), and on a per-texel gradient
    it discards 44 rows. ``resampled`` carries the note so a caller can report which case it is in.
    """
    hw = a.shape[-3:-1]
    tgt = tuple(1 << int(round(math.log2(max(s, 1)))) for s in hw)
    iy = np.minimum((np.arange(tgt[0]) + 0.5) * hw[0] / tgt[0], hw[0] - 1).astype(np.int64)
    ix = np.minimum((np.arange(tgt[1]) + 0.5) * hw[1] / tgt[1], hw[1] - 1).astype(np.int64)
    out = a[..., iy, :, :][..., :, ix, :]
    return np.ascontiguousarray(out), f"{hw[0]}x{hw[1]} -> {tgt[0]}x{tgt[1]} (nearest)"


# --------------------------------------------------------------------------- the renderer

class BatchRenderer(_resources.Closeable):
    """``(B, ngeom, ...)`` geom frames -> ``(B, H, W, 3)`` uint8 on device.

    The frames, not the qpos, are the input: producing ``geom_xpos``/``geom_xmat`` from a state is
    MuJoCo's or MJWarp's job and costs 0.6 % of a frame (``docs/research/backend3d.md``), so keeping it
    outside means this class can be driven by either and measured without one of them in the way.
    ``render_qpos`` is the convenience wrapper that runs MuJoCo's own CPU kinematics.

    **Ownership, and the one case where it is conditional.** This renderer always owns its
    ``nvdiffrast.torch.RasterizeCudaContext`` — a CUDA rasterizer context, which is NOT an EGL/GL
    context and is not subject to the never-release rule that governs those (``_resources`` states
    that rule; nvdiffrast's CUDA backend is exactly why this backend was chosen over the OpenGL one).
    It also owns the skybox texture it prepares and the ray-direction cache.

    The ``Scene`` is the conditional part. Built here (``m=`` given, no ``scene=``) it is owned here
    and ``close()`` closes it. Handed in (``scene=``) it belongs to the caller — ``rocket_landing.py``
    builds one ``Scene.from_triangles`` and could hand it to several renderers — so ``close()`` drops
    the reference and leaves the scene alive. ``owns_scene`` says which, so a caller never has to
    guess::

        with BatchRenderer(m, res=240) as br:      # br.owns_scene is True; the scene closes with it
            ...
        scene = Scene.from_triangles(...)
        with BatchRenderer(None, scene=scene) as br:   # br.owns_scene is False; scene survives
            ...
        scene.close()

    ``owns_scene`` is a plain attribute and is meant to be set. A caller that builds a scene purely
    to hand it to one renderer and keeps no other reference to it — ``rocket_landing.py`` is exactly
    that — writes ``br.owns_scene = True`` after construction, and the scene is then closed with the
    renderer instead of surviving, unreferenced, until the collector notices. The default is the
    conservative one because the damage runs the other way: a renderer that closed a scene its caller
    was still using would be a use-after-free in whatever renders next.

    No ``__del__``, for the reason ``_resources.Closeable`` records. An unclosed renderer holds its
    context until the process ends and is countable with ``_resources.live("nvdiff.BatchRenderer")``.
    """

    resource_kind = "nvdiff.BatchRenderer"

    def __init__(self, m=None, res: int = 224, groups=DEFAULT_GROUPS, device: str = "cuda",
                 textures: bool = True, mipmap: bool = True, skybox: bool = True,
                 quant: str = "round", nslice: int = NUMSLICES, nstack: int = NUMSTACKS,
                 supersample: int = 1, tex_policy: str = "pot", aniso: int = 1, *,
                 scene: "Scene | None" = None, shading: str = "mujoco", tiny: "TinyShading | None" = None,
                 tex_filter: str = "linear", sample_offset=(0.0, 0.0)):
        """``m`` an ``MjModel``, or ``scene=`` a prebuilt ``Scene`` (``Scene.from_triangles``) and no model.

        The keyword-only options are the Step-7 extensions and their defaults select the paths every
        earlier caller used: ``shading="mujoco"`` (``_shade``), ``tex_filter="linear"`` (the mipmapped
        pyramid when ``mipmap`` is on), ``sample_offset=(0, 0)`` (pixel centres). ``shading="tiny"`` needs
        ``tiny=TinyShading(...)`` (or takes its defaults) and lights set with ``set_lights`` are ignored.
        """
        import torch

        import nvdiffrast.torch as dr
        if scene is None and m is None:
            raise ValueError("BatchRenderer needs an MjModel (m=) or a prebuilt Scene (scene=)")
        if shading not in ("mujoco", "tiny"):
            raise ValueError(f"shading must be 'mujoco' or 'tiny', got {shading!r}")
        if tex_filter not in ("linear", "nearest"):
            raise ValueError(f"tex_filter must be 'linear' or 'nearest', got {tex_filter!r}")
        self.m = m
        self.res = int(res)
        self.ss = int(supersample)
        self.device = torch.device(device)
        self.use_tex = bool(textures)
        self.mipmap = bool(mipmap)
        self.use_skybox = bool(skybox)
        self.quant = quant
        self.tex_policy = tex_policy
        self.aniso = int(aniso)
        self.shading = shading
        self.tiny = tiny if tiny is not None else (TinyShading() if shading == "tiny" else None)
        self.tex_filter = tex_filter
        self.sample_offset = (float(sample_offset[0]), float(sample_offset[1]))
        #: True when this renderer BUILT its scene and therefore closes it; False when a caller handed
        #: one in with ``scene=`` and kept ownership of it.
        self.owns_scene = scene is None
        self.scene = scene if scene is not None else Scene(m, groups, device=device, nslice=nslice,
                                                           nstack=nstack, tex_policy=tex_policy)
        self.ctx = dr.RasterizeCudaContext(device=self.device)
        self.lights: Lights | None = None
        self.background = np.array([0.0, 0.0, 0.0])
        self.skybox_tex = None
        #: ``_ray_dirs``'s grid, which is a constant of (resolution, camera ORIENTATION, fovy) and not
        #: of the camera's position or of the state. See ``_ray_dirs`` for what leaving it uncached
        #: costs. Bounded and LRU (``_resources.BoundedCache``): it was a dict that emptied itself
        #: wholesale at eight entries, which discards the grid the CURRENT camera is using as readily
        #: as the coldest one. The values are identical either way — a miss rebuilds the same array
        #: from the same expression — so this changes cost, never a pixel.
        self._ray_cache = _resources.BoundedCache(RAY_CACHE_MAX, name="nvdiff.ray_dirs")
        #: OPTIONAL SHADOW TERM, and the only extension point in the shading law:
        #: ``shadow(light_index, p, n) -> (B,H,W,1) in [0,1]`` or ``None``, where ``p`` and ``n`` are
        #: the world position and unit normal of the shaded point. It multiplies that light's DIFFUSE
        #: and SPECULAR and never its ambient, which is what a shadow does — a surface the light does
        #: not reach still sees the scene's fill.
        #:
        #: It is a hook rather than an implementation because a shadow map is a property of the LIGHT
        #: and the scene, not of this rasterizer: a directional light over a locomotion floor needs
        #: one orthographic pass, a spotlight in a kitchen needs a perspective one, and a scene with
        #: no shadow-casting light needs none at all. ``None`` is the default and costs nothing.
        #: ``stable_envs/envs/mujoco_locomotion.py`` (``_ShadowPass``) is the worked example.
        self.shadow = None
        self._opened()

    def _release(self) -> None:
        """Drop the rasterizer context, the ray grids, the skybox, and the scene IF this one owns it.

        The context goes last: nvdiffrast frees its device-side state when the last reference to it
        goes, and the cached grids and the scene's tensors are ordinary torch allocations that should
        be back in the caching allocator before that happens. The shadow hook is cleared because it is
        a closure a caller installed, and it typically captures a second renderer.
        """
        self._ray_cache.clear()
        self.skybox_tex = None
        self.shadow = None
        self.lights = None
        scene, self.scene = self.scene, None
        if self.owns_scene and scene is not None:
            scene.close()
        self.ctx = None

    # -- setup -------------------------------------------------------------

    def set_lights(self, lights: Lights) -> None:
        self.lights = lights

    def set_skybox(self, m) -> None:
        """Attach the model's skybox texture, if it has one."""
        import mujoco
        if m is None:
            return
        for t in range(m.ntex):
            if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
                self.skybox_tex = prepare_texture(m, t, self.device, self.tex_policy)
                return

    def resolve_from_mujoco(self, m, d, camera=None, option=None):
        """Read the RESOLVED lights and camera out of an ``mjvScene``, the way MuJoCo sees them.

        Lights inherit defaults, envs delete attributes at load time, and the headlight is not in the
        model at all -- so the only trustworthy source for what is actually lighting the scene is the
        scene MuJoCo builds. This costs one ``mjv_updateScene`` at construction and nothing per frame.
        """
        import mujoco
        opt = option if option is not None else mujoco.MjvOption()
        scn = mujoco.MjvScene(m, maxgeom=max(m.ngeom * 2, 1000))
        cam = camera if camera is not None else mujoco.MjvCamera()
        mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        self.lights = Lights.from_scene(scn)
        self.set_skybox(m)
        return Camera.from_glcam(m, scn), scn

    # -- the frame ---------------------------------------------------------

    def render(self, xpos, xmat, cam: Camera, out=None, mat_rgb=None):
        """``xpos (B,G,3)``, ``xmat (B,G,3,3)`` on device -> ``(B,H,W,3)`` uint8 on device.

        ``G`` is the number of geoms this scene DRAWS (``scene.geoms`` maps them back to model ids), so
        a caller holding full ``geom_xpos`` should index it with ``scene.mjgeom`` first. ``mat_rgb``, if
        given, is a PER-ROW ``(B, G, 3)`` material colour replacing the scene's ``mat_rgb`` for this call.
        """
        import torch

        import nvdiffrast.torch as dr
        self._check_open("render")
        S = self.scene
        R = self.res * self.ss
        B = int(xpos.shape[0])
        view_np, proj_np = cam.matrices(aspect=1.0)
        view = torch.as_tensor(view_np, dtype=torch.float32, device=self.device)
        proj = torch.as_tensor(proj_np, dtype=torch.float32, device=self.device)
        campos = torch.as_tensor(np.atleast_2d(np.asarray(cam.pos, np.float64)),
                                 dtype=torch.float32, device=self.device)
        camfwd = torch.as_tensor(-np.asarray(cam.mat, np.float64).reshape(-1, 3, 3)[:, :, 2],
                                 dtype=torch.float32, device=self.device)

        pos, nrm, wpos = S.clip(xpos, xmat, view, proj)
        pos = offset_clip(pos, R, self.sample_offset)                # identity at (0, 0)
        rast, db = dr.rasterize(self.ctx, pos, S.face, resolution=[R, R], grad_db=self.mipmap)

        attr = torch.cat([nrm, wpos], -1)                            # (B,V,6)
        interp, _ = dr.interpolate(attr, rast, S.face)
        n = interp[..., :3]
        p = interp[..., 3:6]
        n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)

        tid = rast[..., 3].long()                                    # 0 = miss, else triangle+1
        hit = tid > 0
        gid = torch.where(hit, S.fgeom[(tid - 1).clamp_min(0)], torch.zeros_like(tid))

        base = gather_material(S.mat_rgb if mat_rgb is None else mat_rgb, gid)
        if self.shading == "tiny":
            col = shade_tiny(n, base, self.tiny)
        else:
            col = self._shade(n, p, base, S.spec[gid], S.shininess[gid], S.emission[gid],
                              campos, camfwd)
        if self.use_tex:
            col = col * self._texture(S, rast, db, gid)
        # CLAMP AFTER THE TEXTURE, never before. The primary colour routinely exceeds 1 -- on a floor
        # lit head-on it is A + D + specular ~ 1.75 -- and MuJoCo multiplies the texture into that and
        # clamps the product. Clamping the primary first caps the floor at its raw texel value and
        # costs ~65 % of an OGBench frame while leaving every flat interior pixel exact, which reads
        # like a texture bug and is a composition bug.
        col = col.clamp(0.0, 1.0)

        img = torch.where(hit[..., None], col, self._background(cam, R, B, campos, camfwd))
        if self.ss > 1:
            img = img.view(B, self.res, self.ss, self.res, self.ss, 3).mean((2, 4))
        img = QUANT(img, self.quant).flip(1)                          # nvdiffrast row 0 is the BOTTOM
        if out is not None:
            out.view(B, self.res, self.res, 3).copy_(img)
            return out
        return img

    def render_qpos(self, qpos, cam: Camera | None = None, camera_id: int = -1, out=None):
        """``(B,nq)`` -> frames, running MuJoCo's own CPU kinematics. Convenience, not the fast path."""
        import mujoco
        import torch
        q = np.asarray(qpos.cpu() if hasattr(qpos, "cpu") else qpos, np.float64)
        d = mujoco.MjData(self.m)
        xp, xm = [], []
        cp, cm = [], []
        for row in q:
            d.qpos[:] = row
            mujoco.mj_forward(self.m, d)
            xp.append(np.asarray(d.geom_xpos, np.float64)[self.scene.geoms])
            xm.append(np.asarray(d.geom_xmat, np.float64).reshape(-1, 3, 3)[self.scene.geoms])
            if camera_id >= 0:
                # .copy(): ``d.cam_xpos[i]`` is a VIEW into MjData, which the next iteration
                # overwrites, so appending it gives every state the LAST state's camera. On a
                # ``track`` camera that is a plausible-looking picture of the right scene from the
                # wrong place -- it cost frame mean 34.8/255 against 3.1 here before it was found.
                cp.append(np.array(d.cam_xpos[camera_id], np.float64, copy=True))
                cm.append(np.array(d.cam_xmat[camera_id], np.float64, copy=True).reshape(3, 3))
        xpos = torch.as_tensor(np.stack(xp), dtype=torch.float32, device=self.device)
        xmat = torch.as_tensor(np.stack(xm), dtype=torch.float32, device=self.device)
        if camera_id >= 0:
            cam = Camera(np.stack(cp), np.stack(cm), float(self.m.cam_fovy[camera_id]),
                         float(self.m.vis.map.znear) * float(self.m.stat.extent),
                         float(self.m.vis.map.zfar) * float(self.m.stat.extent))
        if cam is None:
            raise ValueError("render_qpos needs either a Camera or a camera_id >= 0")
        return self.render(xpos, xmat, cam, out=out)

    # -- shading -----------------------------------------------------------

    def _shade(self, n, p, base, spec, shin, emis, campos, camfwd):
        """``primary = rgb*(A + sum D*max(0,n.l)) + sum S*spec*max(0,n.h)^p``, specular WHITE and ADDED.

        The composition is the part that was measured and the part that a plausible alternative gets
        wrong: fitting a multiplicative law on a textured floor (where the material colour is white and
        all the colour is texture, so an added white specular READS as a multiplier) and extrapolating
        it onto untextured geometry overshoots wall tops by 6.5/255. ``docs/research/ogbench.md``.
        """
        import torch
        L = self.lights
        v = campos.view(-1, 1, 1, 3) - p
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        prim = torch.zeros_like(base)
        high = torch.zeros(base.shape[:-1] + (3,), device=base.device, dtype=base.dtype)
        if L is None:
            return base
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=base.device)  # noqa: E731
        exp = (128.0 * shin).unsqueeze(-1)
        for i in range(len(L.pos)):
            amb, dif, sp = t(L.ambient[i]), t(L.diffuse[i]), t(L.specular[i])
            if L.headlight[i]:
                ldir = -camfwd.view(-1, 1, 1, 3)                     # toward the light
                ldir = ldir.expand_as(v)
                att = torch.ones(base.shape[:-1] + (1,), device=base.device)
            elif L.directional[i]:
                d = t(L.dir[i]).view(1, 1, 1, 3)
                ldir = (-d / d.norm().clamp_min(1e-9)).expand_as(v)
                att = torch.ones(base.shape[:-1] + (1,), device=base.device)
            else:
                rel = t(L.pos[i]).view(1, 1, 1, 3) - p
                dist = rel.norm(dim=-1, keepdim=True).clamp_min(1e-9)
                ldir = rel / dist
                a = L.attenuation[i]
                att = 1.0 / (float(a[0]) + float(a[1]) * dist + float(a[2]) * dist * dist).clamp_min(1e-9)
            if self.shadow is not None:
                # A shadow enters exactly where a light's own attenuation does, which is why it is
                # applied here and not to the finished colour: ``att`` scales the diffuse and the
                # specular of THIS light and leaves the ambient term below untouched.
                lit = self.shadow(i, p, n)
                if lit is not None:
                    att = att * lit
            ndl = (n * ldir).sum(-1, keepdim=True).clamp_min(0.0)
            prim = prim + base * amb + base * dif * ndl * att
            h = ldir + v
            h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            ndh = (n * h).sum(-1, keepdim=True).clamp_min(0.0)
            high = high + sp * spec.unsqueeze(-1) * torch.pow(ndh, exp) * att
        return prim + high + base * emis.unsqueeze(-1)                # UNCLAMPED: see render()

    # -- texturing ---------------------------------------------------------

    def _texture(self, S, rast, db, gid):
        """``(B,H,W,3)`` texture modulation, one prefiltered ``dr.texture`` pass per distinct texture.

        This is the operation the whole module exists for. ``dr.interpolate(..., diff_attrs=)`` turns
        ``rasterize``'s barycentric image-space derivatives into ``d(u,v)/d(x,y)``, which is exactly the
        footprint a minification filter needs and exactly what a raytracer that casts one ray per pixel
        does not have.

        One pass per texture rather than one atlas: an atlas bleeds across tile boundaries at every mip
        level, which is a wrong picture rather than a slow one. The cost is linear in the number of
        DISTINCT textures actually worn by visible geometry, and it is reported per scene in
        ``docs/research/nvdiffrast.md`` rather than hidden.
        """
        import torch

        import nvdiffrast.torch as dr
        B = rast.shape[0]
        out = torch.ones(rast.shape[:3] + (3,), device=rast.device)
        if not S.textures:
            return out
        uv3, uv3_da = dr.interpolate(S.uv.expand(B, -1, -1).contiguous(), rast, S.face,
                                     rast_db=db, diff_attrs="all")
        for t, spec in S.textures.items():
            sel = (S.texid[gid] == t)
            if not bool(sel.any()):
                continue
            if spec["cube"]:
                # MuJoCo maps a direction into cube space by rotating 90 degrees about X
                # (S=x, T=z, R=-y), the same convention its skybox uses.
                d = torch.stack([uv3[..., 0], uv3[..., 2], -uv3[..., 1]], -1).contiguous()
                c = dr.texture(spec["tex"][None], d, boundary_mode="cube", filter_mode="linear")
            else:
                uv = uv3[..., :2].contiguous()
                uv_da = uv3_da[..., :4].contiguous()
                if self.tex_filter == "nearest":
                    c = dr.texture(spec["tex"][None], uv, filter_mode="nearest", boundary_mode="wrap")
                elif self.mipmap and spec["mipmap_ok"]:
                    c = self._sample(spec["tex"], uv, uv_da)
                else:
                    c = dr.texture(spec["tex"][None], uv, filter_mode="linear",
                                   boundary_mode="wrap")
            out = torch.where(sel[..., None], c, out)
        return out


    def _sample(self, tex, uv, uv_da):
        """Trilinear, optionally with an ANISOTROPIC approximation of GL's minification filter.

        An isotropic mipmap picks its level from the LONGER footprint axis, so a floor seen at grazing
        incidence -- the near half of every locomotion frame -- is blurred across the axis that was
        already sharp. GL is not doing that: MuJoCo's own near band has luminance spread 95.11 while an
        isotropic trilinear sample gives 94.32 and an unfiltered one 95.34, i.e. the reference sits
        BETWEEN the two, which is the signature of anisotropic filtering and not of either extreme.

        nvdiffrast has no anisotropic mode, so this is the standard reconstruction: take ``aniso``
        samples along the major footprint axis, each at the level of detail the MINOR axis asks for,
        and average them. ``aniso=1`` is plain trilinear and is the default, because the cost is linear
        in the tap count and whether it pays is a measurement per scene, not a general truth.
        """
        import torch

        import nvdiffrast.torch as dr
        n = int(self.aniso)
        if n <= 1:
            return dr.texture(tex[None], uv, uv_da, filter_mode="linear-mipmap-linear",
                              boundary_mode="wrap")
        ax = uv_da[..., 0:2]                                         # d(u,v)/dx
        ay = uv_da[..., 2:4]                                         # d(u,v)/dy
        lx = ax.norm(dim=-1, keepdim=True)
        ly = ay.norm(dim=-1, keepdim=True)
        major = torch.where(lx >= ly, ax, ay)
        lmaj = torch.maximum(lx, ly).clamp_min(1e-12)
        lmin = torch.minimum(lx, ly).clamp_min(1e-12)
        ratio = (lmin / lmaj).clamp(1.0 / n, 1.0)                    # capped like GL's max anisotropy
        da = uv_da * ratio
        out = 0.0
        for k in range(n):
            t = (k + 0.5) / n - 0.5
            c = dr.texture(tex[None], (uv + t * major).contiguous(), da.contiguous(),
                           filter_mode="linear-mipmap-linear", boundary_mode="wrap")
            out = out + c
        return out / n

    # -- background --------------------------------------------------------

    def _background(self, cam, R, B, campos, camfwd):
        """The skybox, sampled as a cube map along each pixel's ray, or a flat colour without one."""
        import torch

        import nvdiffrast.torch as dr
        if not (self.use_skybox and self.skybox_tex is not None):
            return torch.as_tensor(self.background, dtype=torch.float32,
                                   device=self.device).view(1, 1, 1, 3)
        d = self._ray_dirs(cam, R, B)
        # MuJoCo maps a world direction into cube space by rotating 90 degrees about X: S=x, T=z, R=-y.
        d = torch.stack([d[..., 0], d[..., 2], -d[..., 1]], -1)
        c = dr.texture(self.skybox_tex["tex"][None], d, boundary_mode="cube", filter_mode="linear")
        return c

    def _ray_dirs(self, cam, R, B):
        """One unit direction per output pixel, CACHED on what it actually depends on.

        This grid is built in numpy, in float64, on the host, and it is a function of the resolution,
        the camera's ORIENTATION and its fovy alone — not of its position and not of the state. Left
        uncached it is a ``(n, R, R, 3)`` host array plus a host-to-device copy on EVERY frame, which
        at 240 px is 3.6 ms and is invisible next to a large batch and is most of a small one:
        MEASURED on Gymnasium's HalfCheetah at 240 px, 6.08 ms a frame at batch 1 of which 3.73 ms was
        this, and 8.2 ms at batch 64 of which 3.6 ms was this. Caching it takes batch 1 from 158 to
        ~370 f/s and batch 64 from 6.8 k to ~11 k, and changes no pixel: the entry is the same array
        the uncached path built, and nothing writes through the returned tensor.

        The key is the orientation's BYTES rather than the camera object's ``id``, which is what the
        vestigial ``key = (R, B, id(cam))`` line this replaces would have used: a ``Camera`` is
        rebuilt per call by every caller in this repo, and CPython reuses the id of a freed object, so
        an id-keyed cache is both always-miss and occasionally wrong.
        """
        import torch
        mat = np.asarray(cam.mat, np.float64).reshape(-1, 3, 3)
        key = (int(R), mat.tobytes(), float(cam.fovy))
        t = self._ray_cache.get(key)
        if t is None:
            right, up, back = mat[:, :, 0], mat[:, :, 1], mat[:, :, 2]
            fwd = -back
            f = math.tan(math.radians(cam.fovy) / 2.0)
            ys = (2.0 * (np.arange(R) + 0.5) / R - 1.0) * f          # row 0 at the BOTTOM
            xs = (2.0 * (np.arange(R) + 0.5) / R - 1.0) * f
            X, Y = np.meshgrid(xs, ys)
            d = (X[None, ..., None] * right[:, None, None] + Y[None, ..., None] * up[:, None, None]
                 + fwd[:, None, None])
            d = d / np.linalg.norm(d, axis=-1, keepdims=True)
            # The bound and the eviction are the cache's (``RAY_CACHE_MAX``, least-recently-used);
            # a grid is R*R*n*4 bytes.
            t = self._ray_cache.put(key, torch.as_tensor(d, dtype=torch.float32, device=self.device))
        return t.expand(B, R, R, 3) if t.shape[0] == 1 else t
