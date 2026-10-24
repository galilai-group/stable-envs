"""Batched CUDA triangle rendering, with optional MuJoCo geometry import."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from . import _resources

if TYPE_CHECKING:
    import torch

#: mjtGeom codes, read directly from an MjModel.
PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX, MESH = 0, 2, 3, 4, 5, 6, 7

#: Match the default visible geom groups in MuJoCo MjvOption.
DEFAULT_GROUPS = (0, 1, 2)

#: Default MuJoCo sphere tessellation; changing it moves silhouette edges.
NUMSLICES, NUMSTACKS = 28, 16

#: Maximum ray-direction grids retained per renderer.
RAY_CACHE_MAX = 8


def QUANT(x, mode: str = "round"):
    """Quantize normalized colors to uint8 using the requested rounding mode."""
    import torch

    y = torch.clamp(x, 0.0, 1.0) * 255.0
    if mode == "round":
        y = torch.round(y)
    elif mode == "trunc":
        y = torch.floor(y)
    else:
        raise ValueError(f"unknown quantisation mode {mode!r}")
    return y.to(torch.uint8)


@dataclass
class TinyShading:
    """Bullet TinyRenderer lighting parameters for a z-up world.

    ``dir`` points toward the light. Specular uses the reflected direction's world-z
    component, independent of camera orientation.
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
    """Shade unit world normals and RGB material colors, returning unclamped RGB."""
    import torch

    l = torch.as_tensor(law.unit_dir(), dtype=base.dtype, device=base.device)
    ndl = (n * l).sum(-1, keepdim=True)
    r = 2.0 * ndl * n - l
    rz = r[..., 2:3].clamp_min(0.0)
    term = law.ambient + law.diffuse * ndl.clamp_min(0.0) + law.specular * torch.pow(rz, law.exponent)
    col = torch.as_tensor(np.asarray(law.color, np.float32), dtype=base.dtype, device=base.device)
    return base * term * col


def gather_material(mat, gid):
    """Gather (B, H, W, 3) colors from shared (G, 3) or per-row (B, G, 3) materials."""
    import torch

    if mat.dim() == 2:
        return mat[gid]
    b = torch.arange(gid.shape[0], device=gid.device).view(-1, 1, 1)
    return mat[b, gid]


def _mm4(a, b):
    """Multiply trailing 4x4 matrices with a fixed order of elementwise operations.

    Avoid batch-size-dependent GEMM kernels that can change rounding and silhouette
    pixels when the same state is rendered in a different batch.
    """
    return (
        a[..., :, 0:1] * b[..., 0:1, :]
        + a[..., :, 1:2] * b[..., 1:2, :]
        + a[..., :, 2:3] * b[..., 2:3, :]
        + a[..., :, 3:4] * b[..., 3:4, :]
    )


def mvp_rows(proj, view, M):
    """Compose projection, view, and model matrices with batch-invariant arithmetic.

    Accept proj/view (B, 4, 4) or (4, 4), and M (B, G, 4, 4).
    Compute in float64 with fixed operation order, then cast once to M.dtype.
    """
    import torch

    p = proj.to(torch.float64)
    v = view.to(torch.float64)
    vp = _mm4(p, v)  # (B,4,4) or (4,4)
    vp = vp[:, None] if vp.ndim == 3 else vp[None, None]
    return _mm4(vp, M.to(torch.float64)).to(M.dtype)


def offset_clip(pos, R: int | tuple[int, int], offset) -> torch.Tensor:
    """Translate (B, V, 4) clip vertices by a pixel offset at resolution R.

    Positive offsets move right and up. Scale the shift by clip w so the translation
    is uniform after perspective division; half-pixel offsets align corner sampling.
    """
    ox, oy = float(offset[0]), float(offset[1])
    if ox == 0.0 and oy == 0.0:
        return pos
    h, w = (R, R) if isinstance(R, int) else R
    out = pos.clone()
    out[..., 0] = out[..., 0] + ox * (2.0 / w) * out[..., 3]
    out[..., 1] = out[..., 1] + oy * (2.0 / h) * out[..., 3]
    return out


# Builders return local-frame vertices, normals, and faces scaled by geom size.


def _sphere(radii, nslice: int = NUMSLICES, nstack: int = NUMSTACKS):
    """Build a watertight latitude/longitude sphere or ellipsoid from three radii."""
    th = np.linspace(0.0, 2.0 * np.pi, nslice + 1)  # closes on itself
    ph = np.linspace(0.0, np.pi, nstack + 1)
    T, P = np.meshgrid(th, ph, indexing="ij")
    n = np.stack([np.sin(P) * np.cos(T), np.sin(P) * np.sin(T), np.cos(P)], -1)
    v = n * np.asarray(radii, np.float64)
    nrm = n / np.linalg.norm(n, axis=-1, keepdims=True).clip(1e-12)
    if not np.allclose(radii, radii[0]):  # ellipsoid: normal is not the point
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
    """Build a z-axis cylinder of half-length hl with hemispherical caps of radius r."""
    th = np.linspace(0.0, 2.0 * np.pi, nslice + 1)
    ct, st = np.cos(th), np.sin(th)
    V, N, F = [], [], []
    off = 0

    for z in (-hl, hl):  # barrel
        V.append(np.stack([r * ct, r * st, np.full(nslice + 1, z)], -1))
        N.append(np.stack([ct, st, np.zeros(nslice + 1)], -1))
    for i in range(nslice):
        a, b = i, i + 1
        c, d = (nslice + 1) + i + 1, (nslice + 1) + i
        F += [[a, b, c], [a, c, d]]
    off = 2 * (nslice + 1)

    half = max(nstack // 2, 1)
    for sign, zc in ((1.0, hl), (-1.0, -hl)):  # hemispherical caps
        ang = np.linspace(0.0, np.pi / 2.0, half + 1)
        base = off
        for a in ang:
            rr, zz = r * np.cos(a), sign * r * np.sin(a)
            V.append(np.stack([rr * ct, rr * st, np.full(nslice + 1, zc + zz)], -1))
            N.append(np.stack([np.cos(a) * ct, np.cos(a) * st, np.full(nslice + 1, sign * np.sin(a))], -1))
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
    V = [ring(-hl), ring(hl), ring(-hl), ring(hl), np.array([[0.0, 0.0, -hl], [0.0, 0.0, hl]])]
    N = [
        side_n,
        side_n,
        np.tile([0.0, 0.0, -1.0], (nslice + 1, 1)),
        np.tile([0.0, 0.0, 1.0], (nslice + 1, 1)),
        np.array([[0.0, 0.0, -1.0], [0.0, 0.0, 1.0]]),
    ]
    n1 = nslice + 1
    F = []
    for i in range(nslice):
        F += [[i, i + 1, n1 + i + 1], [i, n1 + i + 1, n1 + i]]
        F += [[4 * n1, 2 * n1 + i + 1, 2 * n1 + i]]  # bottom cap
        F += [[4 * n1 + 1, 3 * n1 + i, 3 * n1 + i + 1]]  # top cap
    return np.concatenate(V), np.concatenate(N), np.asarray(F, np.int64)


def _box(half):
    """Build six faces with separate vertices so normals stay flat across box edges."""
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
            # Wind triangles so each face is front-facing from its outward normal.
            tri = [[base, base + 1, base + 2], [base, base + 2, base + 3]]
            F += tri if sgn > 0 else [t[::-1] for t in tri]
    return np.concatenate(V), np.concatenate(N), np.asarray(F, np.int64)


def _plane(size, extent: float):
    """Build a finite plane quad, replacing infinite zero-size axes with a drawing extent."""
    hx = size[0] if size[0] > 0 else 20.0 * extent
    hy = size[1] if size[1] > 0 else 20.0 * extent
    v = np.array([[-hx, -hy, 0.0], [hx, -hy, 0.0], [hx, hy, 0.0], [-hx, hy, 0.0]])
    n = np.tile([0.0, 0.0, 1.0], (4, 1))
    return v, n, np.array([[0, 1, 2], [0, 2, 3]], np.int64)


def effective_rgba(m, g: int) -> np.ndarray:
    """Resolve geom RGBA, using its material only when geom RGBA matches the compiler default."""
    rgba = np.asarray(m.geom_rgba[g], np.float64).copy()
    mid = int(m.geom_matid[g])
    if mid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
        rgba = np.asarray(m.mat_rgba[mid], np.float64).copy()
    return rgba


def _tex_of(m, mid: int) -> int:
    """Resolve the RGB texture ID for scalar or role-indexed MuJoCo material layouts."""
    if mid < 0:
        return -1
    t = m.mat_texid[mid]
    if np.ndim(t) == 0:
        return int(t)
    import mujoco

    role = int(getattr(mujoco.mjtTextureRole, "mjTEXROLE_RGB", 1))
    return int(t[role]) if role < len(t) else int(np.max(t))


def _is_cube(m, g: int) -> bool:
    t = _tex_of(m, int(m.geom_matid[g]))
    return t >= 0 and int(m.tex_type[t]) != 0


def texture_array(m, tex: int) -> np.ndarray:
    """Read texture ``tex`` of a compiled model as an (H, W, 3) uint8 array."""
    w, h = int(m.tex_width[tex]), int(m.tex_height[tex])
    nc = int(m.tex_nchannel[tex]) if hasattr(m, "tex_nchannel") else 3
    a0 = int(m.tex_adr[tex])
    buf = getattr(m, "tex_data", None)
    if buf is None:
        buf = m.tex_rgb
    a = np.asarray(buf[a0 : a0 + w * h * nc], np.uint8).reshape(h, w, nc)
    return np.repeat(a, 3, -1) if nc == 1 else np.ascontiguousarray(a[..., :3])


def half_extent(m, g: int, drawn_plane: float = 0.0) -> np.ndarray:
    """Return per-axis texgen half-extents for a geom.

    Capsules and cylinders use radius on both texture axes; planes use drawn_plane
    for infinite axes. These are not uniformly equal to geom_size.
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
    return np.where(s > 0, s, 1.0)  # BOX, ELLIPSOID, MESH


#: Center 2-D texture coordinates for every primitive type.
TEXGEN_OFFSET = 0.5


def texgen(m, g: int, drawn_plane: float = 0.0, cube: bool = False) -> tuple[np.ndarray, np.ndarray]:
    """Return (A, b) for geom-local texture coordinates A @ position + b.

    For 2-D maps, project local x/y with signed half-repeat scales and offset 0.5.
    ``texuniform`` uses world-length repeats; otherwise divide by half-extents.
    Cube maps use the unshifted local direction, ignoring repeat and scale.
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
        # Cube lookup uses a direction; offsetting it would rotate the sampled direction.
        return np.eye(3), np.zeros(3)
    return A, np.array([TEXGEN_OFFSET, TEXGEN_OFFSET, 0.0])


@dataclass
class Lights:
    """Resolved mjvScene lights, including the camera headlight.

    Directions point away from the lights. Read resolved scene values so inherited
    model defaults and runtime environment edits are included.
    """

    pos: np.ndarray  # (L,3)
    dir: np.ndarray  # (L,3) unit, direction the light travels
    directional: np.ndarray  # (L,) bool
    ambient: np.ndarray  # (L,3)
    diffuse: np.ndarray  # (L,3)
    specular: np.ndarray  # (L,3)
    attenuation: np.ndarray  # (L,3)
    cutoff: np.ndarray  # (L,)
    exponent: np.ndarray  # (L,)
    headlight: np.ndarray  # (L,) bool -- pos/dir follow the camera, per frame

    @staticmethod
    def from_scene(scn) -> Lights:
        n = int(scn.nlight)
        g = lambda a: np.asarray(a, np.float64)  # noqa: E731

        def flag(i, name, fallback):
            """Read a boolean light field, or derive it from the light-type enum when absent."""
            li = scn.lights[i]
            if hasattr(li, name):
                return bool(getattr(li, name))
            return bool(fallback(li))

        return Lights(
            pos=g([scn.lights[i].pos for i in range(n)]).reshape(n, 3),
            dir=g([scn.lights[i].dir for i in range(n)]).reshape(n, 3),
            directional=np.array([flag(i, "directional", lambda li: int(li.type) == 1) for i in range(n)]),
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
    """Pinhole camera with matrix columns right, up, and backward.

    Position and orientation may carry a leading batch dimension; the camera looks
    along local negative z.
    """

    pos: np.ndarray
    mat: np.ndarray
    fovy: float
    znear: float
    zfar: float

    @staticmethod
    def from_glcam(m, scn) -> Camera:
        """Build a mono camera by averaging the two resolved eyes, canceling their IPD offsets."""
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
        return Camera(pos, np.stack([right, up, -fwd], -1), fovy, near, float(m.vis.map.zfar) * float(m.stat.extent))

    def matrices(self, aspect: float = 1.0):
        """Return OpenGL view (..., 4, 4) and projection (4, 4) matrices."""
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
    """Static triangle geometry, appearance, and textures on one device.

    Owns its device tensors but no rasterizer or EGL context. A renderer given this
    Scene borrows it unless ownership is explicitly transferred. Close the Scene
    only after all borrowers are finished.
    """

    resource_kind = "nvdiff.Scene"

    def __init__(
        self,
        m,
        groups=DEFAULT_GROUPS,
        device="cuda",
        nslice: int = NUMSLICES,
        nstack: int = NUMSTACKS,
        skip_alpha_zero: bool = True,
        tex_policy: str = "pot",
    ):
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
                continue  # hfield / sdf: not handled
            if t == MESH and uv_mesh is not None:
                # Preserve explicit mesh UVs instead of replacing the asset's layout with generated coordinates.
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

        if not self.geoms:
            raise ValueError("nvdiff.Scene has no visible geometry after applying filters")
        dev = torch.device(device)
        self._assemble(
            np.concatenate(V),
            np.concatenate(N),
            np.concatenate(U),
            np.concatenate(GI),
            np.concatenate(F),
            np.concatenate(FG),
            dev,
        )
        self._appearance(m, dev)
        self._sparse(dev)
        self._opened()

    def _release(self) -> None:
        """Drop owned geometry, transform, appearance, and texture tensors."""
        for name in (
            "vert",
            "norm",
            "uv",
            "vgeom",
            "face",
            "fgeom",
            "mjgeom",
            "S3",
            "S4",
            "S4_64",
            "S3p",
            "mat_rgb",
            "alpha",
            "spec",
            "shininess",
            "emission",
            "texid",
        ):
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
    def from_triangles(
        cls,
        vert,
        norm,
        face,
        vert_geom,
        rgba,
        *,
        uv=None,
        specular=None,
        shininess=None,
        emission=None,
        textures=None,
        tex_of_geom=None,
        device="cuda",
        tex_policy: str = "pot",
    ) -> Scene:
        """Build a Scene from triangles expressed in each geom's local frame.

        Inputs are vert/norm (V, 3), face (F, 3), vert_geom (V,), and rgba (G, 4).
        All three vertices of a face must belong to the same geom. Optional UVs have
        shape (V, 2|3); material scalars have shape (G,). Textures map IDs to image
        arrays, and tex_of_geom uses -1 for untextured geoms.

        Render transforms must use the same geom order as these arrays.
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
        self.spec = torch.as_tensor(
            z if specular is None else np.asarray(specular, np.float64), dtype=torch.float32, device=dev
        )
        self.shininess = torch.as_tensor(
            z if shininess is None else np.asarray(shininess, np.float64), dtype=torch.float32, device=dev
        )
        self.emission = torch.as_tensor(
            z if emission is None else np.asarray(emission, np.float64), dtype=torch.float32, device=dev
        )
        texid = np.full(G, -1, np.int64) if tex_of_geom is None else np.asarray(tex_of_geom, np.int64)
        self.texid = torch.as_tensor(texid, device=dev)
        self.textures = {}
        for t in sorted({int(x) for x in texid if x >= 0}):
            if textures is None or t not in textures:
                raise ValueError(f"tex_of_geom names texture {t} but textures has no such entry")
            self.textures[t] = prepare_texture_array(np.asarray(textures[t]), dev, tex_policy)
        self._sparse(dev)
        self._opened()
        return self

    @staticmethod
    def _mesh(m, g: int):
        """Expand each triangle corner using separate position, normal, and UV indices.

        MuJoCo meshes index these attributes independently. Sharing position indices
        for all attributes would smooth hard edges and erase UV seams.
        """
        did = int(m.geom_dataid[g])
        va, vn = int(m.mesh_vertadr[did]), int(m.mesh_vertnum[did])
        fa, fn = int(m.mesh_faceadr[did]), int(m.mesh_facenum[did])
        vert = np.asarray(m.mesh_vert[va : va + vn], np.float64).reshape(-1, 3)
        face = np.asarray(m.mesh_face[fa : fa + fn], np.int64).reshape(-1, 3)
        na = int(m.mesh_normaladr[did]) if hasattr(m, "mesh_normaladr") else va
        nn = int(m.mesh_normalnum[did]) if hasattr(m, "mesh_normalnum") else vn
        nrm = np.asarray(m.mesh_normal[na : na + nn], np.float64).reshape(-1, 3)
        fnrm = (
            np.asarray(m.mesh_facenormal[fa : fa + fn], np.int64).reshape(-1, 3)
            if hasattr(m, "mesh_facenormal")
            else face
        )

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
            tc = np.asarray(m.mesh_texcoord[ta : ta + tn], np.float64).reshape(-1, 2)
            ftc = (
                np.asarray(m.mesh_facetexcoord[fa : fa + fn], np.int64).reshape(-1, 3)
                if hasattr(m, "mesh_facetexcoord")
                else face
            )
            if len(tc):
                uv = tc[np.clip(ftc.reshape(-1), 0, len(tc) - 1)]
        f = np.arange(3 * len(face), dtype=np.int64).reshape(-1, 3)
        return v, n, f, uv

    def _sparse(self, dev):
        """Build sparse transforms without allocating a (B, V, 3, 3) gathered matrix.

        S4 stores homogeneous local vertices in their geom's four columns; S3 carries
        the corresponding normal transform.
        """
        import torch

        v4 = torch.cat([self.vert, torch.ones(self.V, 1, device=dev)], -1)
        rows = torch.arange(self.V, device=dev).repeat_interleave(4)
        cols = (self.vgeom[:, None] * 4 + torch.arange(4, device=dev)[None]).reshape(-1)
        self.S4 = (
            torch.sparse_coo_tensor(torch.stack([rows, cols]), v4.reshape(-1), (self.V, 4 * self.G))
            .coalesce()
            .to_sparse_csr()
        )
        rows3 = torch.arange(self.V, device=dev).repeat_interleave(3)
        cols3 = (self.vgeom[:, None] * 3 + torch.arange(3, device=dev)[None]).reshape(-1)
        self.S3 = (
            torch.sparse_coo_tensor(torch.stack([rows3, cols3]), self.norm.reshape(-1), (self.V, 3 * self.G))
            .coalesce()
            .to_sparse_csr()
        )
        self.S3p = (
            torch.sparse_coo_tensor(torch.stack([rows3, cols3]), self.vert.reshape(-1), (self.V, 3 * self.G))
            .coalesce()
            .to_sparse_csr()
        )

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
                    texcube[i] = int(m.tex_type[t]) != 0  # 0 == mjTEXTURE_2D
        self.mat_rgb = torch.as_tensor(rgb, dtype=torch.float32, device=dev)
        self.alpha = torch.as_tensor(alpha, dtype=torch.float32, device=dev)
        self.spec = torch.as_tensor(spec, dtype=torch.float32, device=dev)
        self.shininess = torch.as_tensor(shin, dtype=torch.float32, device=dev)
        self.emission = torch.as_tensor(emis, dtype=torch.float32, device=dev)
        self.texid = torch.as_tensor(texid, device=dev)

        self.textures: dict[int, dict] = {}
        for t in sorted({int(x) for x in texid if x >= 0}):
            self.textures[t] = prepare_texture(m, t, dev, self.tex_policy)

    def clip(self, xpos, xmat, view, proj, *, coverage_dtype=None):
        """Return clip vertices, world normals and positions.

        Optional float64 coverage keeps camera/model composition and sparse vertex
        projection in double precision. Normal/position shading attributes retain
        the original input precision. Default arithmetic is unchanged.
        """
        import torch

        self._check_open("clip")
        B, G = xpos.shape[0], self.G
        M = torch.zeros(B, G, 4, 4, device=xpos.device, dtype=xpos.dtype)
        M[:, :, :3, :3] = xmat
        M[:, :, :3, 3] = xpos
        M[:, :, 3, 3] = 1.0
        if coverage_dtype not in (None, torch.float64):
            raise ValueError("coverage_dtype must be None or torch.float64")
        matrix = M if coverage_dtype is None else M.to(coverage_dtype)
        sparse = self.S4
        if coverage_dtype is not None:
            if getattr(self, "S4_64", None) is None:
                self.S4_64 = self.S4.to(torch.float64)
            sparse = self.S4_64
        mvp = mvp_rows(proj, view, matrix)
        pos = torch.sparse.mm(sparse, mvp.permute(1, 3, 0, 2).reshape(4 * G, 4 * B))
        pos = pos.reshape(self.V, B, 4).permute(1, 0, 2).contiguous()
        nrm = torch.sparse.mm(self.S3, xmat.permute(1, 3, 0, 2).reshape(3 * G, 3 * B))
        nrm = nrm.reshape(self.V, B, 3).permute(1, 0, 2).contiguous()
        wp = torch.sparse.mm(self.S3p, xmat.permute(1, 3, 0, 2).reshape(3 * G, 3 * B))
        wp = wp.reshape(self.V, B, 3).permute(1, 0, 2) + xpos[:, self.vgeom]
        return pos, nrm, wp.contiguous()


def is_pow2(n: int) -> bool:
    return n > 0 and (n & (n - 1)) == 0


def prepare_texture(m, t: int, dev, policy: str = "pot") -> dict:
    """Prepare a MuJoCo texture for sampling, recording any power-of-two resampling.

    Cube maps retain their dimensions and use six faces in OpenGL order.
    """
    a = texture_array(m, t).astype(np.float32) / 255.0
    kind = int(m.tex_type[t])
    return prepare_texture_array(a, dev, policy, cube=kind != 0)


def prepare_texture_array(a: np.ndarray, dev, policy: str = "pot", cube: bool = False) -> dict:
    """Prepare an RGB image and return its device tensor and sampling metadata.

    Accept normalized floats or uint8 RGB/RGBA. Cube input may be a six-face vertical
    strip or one square image reused on every face. ``policy="pot"`` resamples
    non-power-of-two 2-D images; ``resampled`` records the resulting approximation.
    """
    import torch

    a = np.asarray(a)
    if a.dtype == np.uint8:
        a = a.astype(np.float32) / 255.0
    if a.ndim == 3 and a.shape[-1] == 4:
        a = a[..., :3]
    a = np.ascontiguousarray(a, dtype=np.float32)
    if cube:
        # Cube textures store either six vertically stacked faces or one image reused on all sides.
        h, w = a.shape[0], a.shape[1]
        if h == 6 * w:
            a = a.reshape(6, w, w, 3)
        elif h == w:
            a = np.repeat(a[None], 6, 0)
        else:  # unrecognized layout: sample as a 2-D image
            cube = False
    hw = a.shape[-3:-1]
    ok = is_pow2(hw[0]) and is_pow2(hw[1])
    resampled = None
    # Cube sampling uses a linear filter without mipmaps, so power-of-two resampling is unnecessary.
    if not ok and policy == "pot" and not cube:
        a, resampled = _to_pow2(a)
        ok = True
    tex = torch.as_tensor(np.ascontiguousarray(a), dtype=torch.float32, device=dev)
    return {"tex": tex, "cube": cube, "resampled": resampled, "mipmap_ok": ok, "shape": tuple(a.shape)}


def _to_pow2(a: np.ndarray):
    """Nearest-resample to power-of-two dimensions and return the image plus a change record."""
    hw = a.shape[-3:-1]
    tgt = tuple(1 << int(round(math.log2(max(s, 1)))) for s in hw)
    iy = np.minimum((np.arange(tgt[0]) + 0.5) * hw[0] / tgt[0], hw[0] - 1).astype(np.int64)
    ix = np.minimum((np.arange(tgt[1]) + 0.5) * hw[1] / tgt[1], hw[1] - 1).astype(np.int64)
    out = a[..., iy, :, :][..., :, ix, :]
    return np.ascontiguousarray(out), f"{hw[0]}x{hw[1]} -> {tgt[0]}x{tgt[1]} (nearest)"


def close_cached_renderer(c) -> None:
    """Close the renderer held by a family cache entry before eviction."""
    br = getattr(c, "br", None)
    if br is not None:
        br.close()


class BatchRenderer(_resources.Closeable):
    """Render geom transforms to (B, H, W, 3) uint8 frames with nvdiffrast CUDA.

    Owns the rasterizer context, skybox textures, and ray cache. Scenes built from
    ``m`` are owned; ``scene=`` is borrowed by default. Set ``owns_scene=True`` only
    when transferring sole responsibility for closing that Scene.

    Close explicitly or use a context manager. This CUDA context is not an EGL/GL
    context and can be released normally. Kinematics is supplied by the caller;
    ``render_qpos`` provides a convenience path through CPU MuJoCo.
    """

    resource_kind = "nvdiff.BatchRenderer"

    def __init__(
        self,
        m=None,
        res: int | tuple[int, int] = 224,
        groups=DEFAULT_GROUPS,
        device: str = "cuda",
        textures: bool = True,
        mipmap: bool = True,
        skybox: bool = True,
        quant: str = "round",
        nslice: int = NUMSLICES,
        nstack: int = NUMSTACKS,
        supersample: int = 1,
        tex_policy: str = "pot",
        aniso: int = 1,
        *,
        scene: Scene | None = None,
        shading: str = "mujoco",
        tiny: TinyShading | None = None,
        tex_filter: str = "linear",
        sample_offset=(0.0, 0.0),
    ):
        """Use an MjModel or a prebuilt Scene to configure the renderer.

        ``res`` is an integer for square output or (height, width).
        ``shading="tiny"`` uses TinyShading parameters and ignores scene lights.
        ``sample_offset`` shifts pixel coverage; ``tex_filter`` selects linear or
        nearest sampling. Texture mipmapping and anisotropy are configured separately.
        """
        import nvdiffrast.torch as dr
        import torch

        if scene is None and m is None:
            raise ValueError("BatchRenderer needs an MjModel (m=) or a prebuilt Scene (scene=)")
        if shading not in ("mujoco", "tiny"):
            raise ValueError(f"shading must be 'mujoco' or 'tiny', got {shading!r}")
        if tex_filter not in ("linear", "nearest"):
            raise ValueError(f"tex_filter must be 'linear' or 'nearest', got {tex_filter!r}")
        self.m = m
        self.res = int(res) if isinstance(res, int) else tuple(res)
        self.height, self.width = (self.res, self.res) if isinstance(self.res, int) else self.res
        if self.height < 1 or self.width < 1:
            raise ValueError("res must be positive: int or (height, width)")
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
        #: Borrow supplied scenes; close only scenes this renderer owns.
        self.owns_scene = scene is None
        self.scene = (
            scene
            if scene is not None
            else Scene(m, groups, device=device, nslice=nslice, nstack=nstack, tex_policy=tex_policy)
        )
        self.ctx = dr.RasterizeCudaContext(device=self.device)
        self.lights: Lights | None = None
        self.background = np.array([0.0, 0.0, 0.0])
        self.skybox_tex = None
        #: Rays depend on camera orientation and fovy, not camera position.
        self._ray_cache = _resources.BoundedCache(RAY_CACHE_MAX, name="nvdiff.ray_dirs")
        #: Optional shadow(i, world_position, unit_normal) returns (B,H,W,1) in [0,1] or None.
        #: It attenuates only that light's diffuse/specular terms, leaving ambient intact.
        self.shadow = None
        self._opened()

    def _release(self) -> None:
        """Drop caches, hooks, and any owned Scene before releasing the CUDA rasterizer context."""
        self._ray_cache.clear()
        self.skybox_tex = None
        self.shadow = None
        self.lights = None
        scene, self.scene = self.scene, None
        if self.owns_scene and scene is not None:
            scene.close()
        self.ctx = None

    def set_lights(self, lights: Lights) -> None:
        self._check_open("set_lights")
        self.lights = lights

    def set_skybox(self, m) -> None:
        """Prepare the model's skybox texture when present."""
        self._check_open("set_skybox")
        import mujoco

        if m is None:
            return
        for t in range(m.ntex):
            if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX):
                self.skybox_tex = prepare_texture(m, t, self.device, self.tex_policy)
                return

    def resolve_from_mujoco(self, m, d, camera=None, option=None):
        """Resolve lights and camera through mjv_updateScene, including runtime defaults and the headlight."""
        self._check_open("resolve_from_mujoco")
        import mujoco

        opt = option if option is not None else mujoco.MjvOption()
        scn = mujoco.MjvScene(m, maxgeom=max(m.ngeom * 2, 1000))
        cam = camera if camera is not None else mujoco.MjvCamera()
        mujoco.mjv_updateScene(m, d, opt, None, cam, mujoco.mjtCatBit.mjCAT_ALL, scn)
        self.lights = Lights.from_scene(scn)
        self.set_skybox(m)
        return Camera.from_glcam(m, scn), scn

    def _rasterize(self, positions, resolution):
        """Rasterize coverage; families may override this with another GPU rasterizer."""
        import nvdiffrast.torch as dr

        positions = offset_clip(positions, resolution, self.sample_offset)
        return dr.rasterize(
            self.ctx,
            positions,
            self.scene.face,
            resolution=[resolution, resolution] if isinstance(resolution, int) else list(resolution),
            grad_db=self.mipmap,
        )

    def _clip(self, xpos, xmat, view, proj):
        """Transform geometry; subclasses may retain higher camera precision."""
        import torch

        view = torch.as_tensor(view, dtype=torch.float32, device=self.device)
        proj = torch.as_tensor(proj, dtype=torch.float32, device=self.device)
        return self.scene.clip(xpos, xmat, view, proj)

    def render(self, xpos, xmat, cam: Camera, out=None, mat_rgb=None, *, mode="rgb"):
        """Render one mode; preserve the tensor/out contract of the original API."""
        image = self.render_many(xpos, xmat, cam, modes=(mode,), mat_rgb=mat_rgb)[mode]
        if out is not None:
            if mode == "rgb":
                out.view(image.shape).copy_(image)
            else:
                out.copy_(image)
            return out
        return image

    def render_many(self, xpos, xmat, cam: Camera, *, modes=("rgb", "depth", "ids"), out=None, mat_rgb=None):
        """Generate selected outputs from one transform/rasterization pass.

        Return a dict of CUDA tensors: rgb uint8 (B,H,W,3), positive camera-axis
        depth float32 (B,H,W), and one-based geom IDs int16 (B,H,W). Misses have
        zero depth/ID. Modes must be unique and nonempty; diagnostics require
        supersample=1. Optional out must contain matching buffers for all modes;
        validate the entire mapping before writing any caller buffer.

        Shares the rasterizer hook, textures, lighting and row convention with
        render(), including family-specific coverage implementations.
        """
        import nvdiffrast.torch as dr
        import torch

        self._check_open("render")
        modes = tuple(modes)
        if (
            not modes
            or len(set(modes)) != len(modes)
            or any(mode not in ("rgb", "depth", "ids") for mode in modes)
            or (self.ss != 1 and any(mode != "rgb" for mode in modes))
        ):
            raise ValueError("modes must be unique rgb/depth/ids; diagnostics require supersample=1")
        S = self.scene
        H, W = self.height, self.width
        R = H * self.ss if H == W else (H * self.ss, W * self.ss)
        B = int(xpos.shape[0])
        if out is not None:
            if set(out) != set(modes):
                raise ValueError("out must contain exactly the requested modes")
            for mode in modes:
                shape = (B, H, W, 3) if mode == "rgb" else (B, H, W)
                dtype = {"rgb": torch.uint8, "depth": torch.float32, "ids": torch.int16}[mode]
                value = out[mode]
                if (
                    not torch.is_tensor(value)
                    or tuple(value.shape) != shape
                    or value.dtype != dtype
                    or value.device != self.scene.vert.device
                    or not value.is_contiguous()
                ):
                    raise ValueError(f"out[{mode!r}] must match shape, dtype and CUDA device, and be contiguous")
        view_np, proj_np = cam.matrices(aspect=W / H)
        campos = torch.as_tensor(
            np.atleast_2d(np.asarray(cam.pos, np.float64)), dtype=torch.float32, device=self.device
        )
        camfwd = torch.as_tensor(
            -np.asarray(cam.mat, np.float64).reshape(-1, 3, 3)[:, :, 2], dtype=torch.float32, device=self.device
        )
        pos, nrm, wpos = self._clip(xpos, xmat, view_np, proj_np)
        rast, db = self._rasterize(pos, R)
        tid = rast[..., 3].long()
        hit = tid > 0
        gid = torch.where(hit, S.fgeom[(tid - 1).clamp_min(0)], torch.zeros_like(tid))
        images = {}
        if "ids" in modes:
            images["ids"] = torch.where(hit, gid + 1, 0).to(torch.int16).flip(1)
        if "rgb" in modes:
            attr = torch.cat([nrm, wpos], -1)
            interp, _ = dr.interpolate(attr, rast, S.face)
            n, p = interp[..., :3], interp[..., 3:6]
            n = n / n.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            base = gather_material(S.mat_rgb if mat_rgb is None else mat_rgb, gid)
            if self.shading == "tiny":
                col = shade_tiny(n, base, self.tiny)
            else:
                col = self._shade(n, p, base, S.spec[gid], S.shininess[gid], S.emission[gid], campos, camfwd)
            if self.use_tex:
                col = col * self._texture(S, rast, db, gid)
            col = col.clamp(0.0, 1.0)
            img = torch.where(hit[..., None], col, self._background(cam, R, B, campos, camfwd))
            if self.ss > 1:
                img = img.view(B, H, self.ss, W, self.ss, 3).mean((2, 4))
            images["rgb"] = QUANT(img, self.quant).flip(1)
        elif "depth" in modes:
            p, _ = dr.interpolate(wpos, rast, S.face)
        if "depth" in modes:
            z = ((p - campos[:, None, None]) * camfwd[:, None, None]).sum(-1)
            images["depth"] = torch.where(hit, z, 0).flip(1)
        if out is not None:
            for mode in modes:
                out[mode].copy_(images[mode])
            return out
        return {mode: images[mode] for mode in modes}

    def _shade(self, n, p, base, spec, shin, emis, campos, camfwd):
        """Return unclamped material lighting with additive white specular.

        Texture modulation and clamping happen later in render.
        """
        import torch

        L = self.lights
        if L is None:
            return base
        v = campos.view(-1, 1, 1, 3) - p
        v = v / v.norm(dim=-1, keepdim=True).clamp_min(1e-9)
        prim = torch.zeros_like(base)
        high = torch.zeros(base.shape[:-1] + (3,), device=base.device, dtype=base.dtype)
        t = lambda x: torch.as_tensor(np.asarray(x, np.float32), device=base.device)  # noqa: E731
        exp = (128.0 * shin).unsqueeze(-1)
        for i in range(len(L.pos)):
            amb, dif, sp = t(L.ambient[i]), t(L.diffuse[i]), t(L.specular[i])
            if L.headlight[i]:
                ldir = -camfwd.view(-1, 1, 1, 3)  # toward the light
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
                # Apply shadow attenuation per light before diffuse/specular accumulation; keep ambient unchanged.
                lit = self.shadow(i, p, n)
                if lit is not None:
                    att = att * lit
            ndl = (n * ldir).sum(-1, keepdim=True).clamp_min(0.0)
            prim = prim + base * amb + base * dif * ndl * att
            h = ldir + v
            h = h / h.norm(dim=-1, keepdim=True).clamp_min(1e-9)
            ndh = (n * h).sum(-1, keepdim=True).clamp_min(0.0)
            high = high + sp * spec.unsqueeze(-1) * torch.pow(ndh, exp) * att
        return prim + high + base * emis.unsqueeze(-1)

    def _texture(self, S, rast, db, gid):
        """Sample texture modulation using interpolated UV derivatives for mip selection.

        Sample each distinct texture separately so mip levels cannot bleed across atlas
        boundaries. Cube maps use directional lookup instead of 2-D UVs.
        """
        import nvdiffrast.torch as dr
        import torch

        B = rast.shape[0]
        out = torch.ones(rast.shape[:3] + (3,), device=rast.device)
        if not S.textures:
            return out
        uv3, uv3_da = dr.interpolate(S.uv.expand(B, -1, -1).contiguous(), rast, S.face, rast_db=db, diff_attrs="all")
        for t, spec in S.textures.items():
            sel = S.texid[gid] == t
            if not bool(sel.any()):
                continue
            if spec["cube"]:
                # MuJoCo rotates directions 90 degrees about X into cube space (S=x, T=z, R=-y).
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
                    c = dr.texture(spec["tex"][None], uv, filter_mode="linear", boundary_mode="wrap")
            out = torch.where(sel[..., None], c, out)
        return out

    def _sample(self, tex, uv, uv_da):
        """Sample with configured filtering and optional anisotropic taps.

        For anisotropy, average taps along the major footprint axis at the reduced LOD.
        Tap count increases sampling work; aniso=1 uses the ordinary filtered path.
        """
        import nvdiffrast.torch as dr
        import torch

        n = int(self.aniso)
        if n <= 1:
            return dr.texture(tex[None], uv, uv_da, filter_mode="linear-mipmap-linear", boundary_mode="wrap")
        ax = uv_da[..., [0, 2]]  # (du/dX, dv/dX)
        ay = uv_da[..., [1, 3]]  # (du/dY, dv/dY)
        lx = ax.norm(dim=-1, keepdim=True)
        ly = ay.norm(dim=-1, keepdim=True)
        major = torch.where(lx >= ly, ax, ay)
        lmaj = torch.maximum(lx, ly).clamp_min(1e-12)
        lmin = torch.minimum(lx, ly).clamp_min(1e-12)
        ratio = (lmin / lmaj).clamp(1.0 / n, 1.0)  # capped like GL's max anisotropy
        da = uv_da * ratio
        out = 0.0
        for k in range(n):
            t = (k + 0.5) / n - 0.5
            c = dr.texture(
                tex[None],
                (uv + t * major).contiguous(),
                da.contiguous(),
                filter_mode="linear-mipmap-linear",
                boundary_mode="wrap",
            )
            out = out + c
        return out / n

    def _background(self, cam, R, B, campos, camfwd):
        """Sample the skybox along pixel rays, or broadcast the flat background."""
        import nvdiffrast.torch as dr
        import torch

        if not (self.use_skybox and self.skybox_tex is not None):
            return torch.as_tensor(self.background, dtype=torch.float32, device=self.device).view(1, 1, 1, 3)
        d = self._ray_dirs(cam, R, B)
        # MuJoCo maps a world direction into cube space by rotating 90 degrees about X: S=x, T=z, R=-y.
        d = torch.stack([d[..., 0], d[..., 2], -d[..., 1]], -1)
        c = dr.texture(self.skybox_tex["tex"][None], d, boundary_mode="cube", filter_mode="linear")
        return c

    def _ray_dirs(self, cam, R, B):
        """Return cached world-space pixel rays for camera orientation, fovy, and resolution.

        Key by orientation values, not camera identity or position. Shared orientations
        broadcast across the batch; callers must not mutate the returned view.
        """
        self._check_open("_ray_dirs")
        import torch

        mat = np.asarray(cam.mat, np.float64).reshape(-1, 3, 3)
        H, W = (R, R) if isinstance(R, int) else R
        key = (H, W, mat.tobytes(), float(cam.fovy))
        t = self._ray_cache.get(key)
        if t is None:
            right, up, back = mat[:, :, 0], mat[:, :, 1], mat[:, :, 2]
            fwd = -back
            f = math.tan(math.radians(cam.fovy) / 2.0)
            ys = (2.0 * (np.arange(H) + 0.5) / H - 1.0) * f  # row 0 at the bottom
            xs = (2.0 * (np.arange(W) + 0.5) / W - 1.0) * f * (W / H)
            X, Y = np.meshgrid(xs, ys)
            d = X[None, ..., None] * right[:, None, None] + Y[None, ..., None] * up[:, None, None] + fwd[:, None, None]
            d = d / np.linalg.norm(d, axis=-1, keepdims=True)
            t = self._ray_cache.put(key, torch.as_tensor(d, dtype=torch.float32, device=self.device))
        return t.expand(B, H, W, 3) if t.shape[0] == 1 else t
