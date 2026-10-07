from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import raycast3d as R3
from .primitives import (
    GEOM_NAMES,
    INF,
    PLANE,
    PRIMITIVE_KINDS,
    bound_radius,
)
from .primitives import dot3 as _dot
from .texture_mip import mip_pyramid, plane_grid, plane_texgen, sample_plane_filtered, sample_skybox_filtered

# Movers share raycast3d primitive support; the floor is intersected separately.
MOVER_KINDS = tuple(k for k in PRIMITIVE_KINDS if k != PLANE)
# Supported MuJoCo light types; point and image lights are rejected.
LIGHT_SPOT, LIGHT_DIRECTIONAL = 0, 1
#: MuJoCo's material defaults, which the megakernel uses for a geom without a material.
DEFAULT_MAT_SPECULAR, DEFAULT_MAT_SHININESS, MAX_SHININESS = 0.5, 0.5, 128.0
# Blocked direct lighting contributes nothing; an unlit scene retains the fallback ambient.
SHADOW_VISIBLE = 0.0
NO_LIGHT_AMBIENT_FALLBACK = 0.3
#: The shadow-ray offsets the megakernel uses.
SHADOW_ORIGIN_EPS, SHADOW_MAX_T_EPS = 1.0e-4, 1.0e-3
#: ``MJ_MINVAL``, the front-face threshold of ``ray_plane``.
MJ_MINVAL = 1e-15
SKY_FALLBACK = (0.1, 0.1, 0.2)
FOOTPRINT_MARGIN = 0.05  # World-space padding for the projected shadow bound.
WINDOW_NEAR = 1e-3  # Use the full frame when a mover approaches the camera plane.
# MJWarp packs floats by truncation; rounding is an explicit alternate convention.
QUANT = "trunc"

# Oracle switches bypass spatial bounds without changing the expected pixels.
FORCE_FULL_FRAME: bool = False
SHADOW_TEST_ALL_HITS: bool = False

# Limit intermediate ray buffers to a small working set.
CPU_CHUNK = 16
# CUDA texture weights use eight fractional bits; zero selects floating-point weights.
TEX_FRAC_BITS = 8


def texture_array(m, tex: int) -> np.ndarray:
    """``(h, w, 3)`` uint8 for texture ``tex`` of a compiled model."""
    w, h = int(m.tex_width[tex]), int(m.tex_height[tex])
    nc = int(m.tex_nchannel[tex]) if hasattr(m, "tex_nchannel") else 3
    a0 = int(m.tex_adr[tex])
    buf = getattr(m, "tex_data", None)
    if buf is None:
        buf = m.tex_rgb
    a = np.asarray(buf[a0 : a0 + w * h * nc], np.uint8).reshape(h, w, nc)
    return np.repeat(a, 3, -1) if nc == 1 else np.ascontiguousarray(a[..., :3])


def bilinear(tex: torch.Tensor, u: torch.Tensor, v: torch.Tensor, frac_bits: int | None = None) -> torch.Tensor:
    """Sample float32 RGB texels at normalized UV coordinates, clamping at borders.

    Centers lie at (i + 0.5) / width. Weights truncate to eight fractional bits
    by default, matching CUDA texture filtering. Callers handle UV wrapping.
    """
    if frac_bits is None:
        frac_bits = TEX_FRAC_BITS
    h, w = int(tex.shape[0]), int(tex.shape[1])
    x = u * w - 0.5
    y = v * h - 0.5
    x0 = torch.floor(x)
    y0 = torch.floor(y)
    fx = x - x0
    fy = y - y0
    if frac_bits:
        q = float(1 << frac_bits)
        fx = torch.floor(fx * q) / q
        fy = torch.floor(fy * q) / q
    x0 = x0.to(torch.int64)
    y0 = y0.to(torch.int64)
    x1 = (x0 + 1).clamp_(0, w - 1)
    y1 = (y0 + 1).clamp_(0, h - 1)
    x0 = x0.clamp_(0, w - 1)
    y0 = y0.clamp_(0, h - 1)
    flat = tex.reshape(-1, 3)
    c00 = flat[y0 * w + x0]
    c10 = flat[y0 * w + x1]
    c01 = flat[y1 * w + x0]
    c11 = flat[y1 * w + x1]
    fx = fx.unsqueeze(-1)
    fy = fy.unsqueeze(-1)
    # the unit's own association: lerp along x on each row, then along y
    top = c00 + (c10 - c00) * fx
    bot = c01 + (c11 - c01) * fx
    return top + (bot - top) * fy


def sample_skybox(sky: torch.Tensor, face_w: int, d: torch.Tensor) -> torch.Tensor:
    """``sample_skybox`` of ``mujoco_warp``: cube-map lookup of world directions ``d (...,3)``.

    MuJoCo rotates a world direction into cube space by 90 degrees about X (S=x, T=z, R=-y); the
    six faces are stacked vertically in OpenGL order +X,-X,+Y,-Y,+Z,-Z; ``t`` is clamped half a
    texel inside its face so the linear filter never bleeds into the next one.
    """
    rx, ry, rz = d[..., 0], d[..., 2], -d[..., 1]
    arx, ary, arz = rx.abs(), ry.abs(), rz.abs()
    # the same three-way branch, in the same order, as the kernel
    mx = (arx >= ary) & (arx >= arz)
    my = ~mx & (ary >= arz)
    mz = ~mx & ~my
    px, nx = mx & (rx > 0), mx & ~(rx > 0)
    py, ny = my & (ry > 0), my & ~(ry > 0)
    pz, nz = mz & (rz > 0), mz & ~(rz > 0)
    ma = torch.where(mx, arx, torch.where(my, ary, arz))
    face = (
        torch.where(nx, 1, 0)
        + torch.where(py, 2, 0)
        + torch.where(ny, 3, 0)
        + torch.where(pz, 4, 0)
        + torch.where(nz, 5, 0)
    )
    sc = torch.where(px, -rz, torch.where(nx, rz, torch.where(py, rx, torch.where(ny, rx, torch.where(pz, rx, -rx)))))
    tc = torch.where(px | nx, -ry, torch.where(py, rz, torch.where(ny, -rz, -ry)))
    ma_safe = torch.where(ma == 0, torch.ones_like(ma), ma)
    s = (sc / ma_safe + 1.0) * 0.5
    t = (tc / ma_safe + 1.0) * 0.5
    t_min = 0.5 / float(face_w)
    t = t.clamp(t_min, 1.0 - t_min)
    v = (face.to(t.dtype) + t) * (1.0 / 6.0)
    return bilinear(sky, s, v)


@dataclass
class Light:
    kind: int  # LIGHT_SPOT or LIGHT_DIRECTIONAL
    active: bool  # ``light_active``: the megakernel skips an inactive light entirely
    castshadow: bool
    cutoff_deg: float
    exponent: float
    ambient: np.ndarray  # (3,)
    diffuse: np.ndarray  # (3,) the model's own; scaled per env by the caller
    specular: np.ndarray  # (3,)
    attenuation_default: bool  # (1, 0, 0) -> no distance term, as the kernel folds it


class Scene:
    """Static data for a MuJoCo camera, floor, and analytic moving primitives.

    Accepts fixed/trackcom perspective cameras. Shadow-casting spotlights require
    a horizontal upward-facing floor and must remain above it in every frame.
    Render poses come from MuJoCo kinematics; materials may vary per frame.
    """

    def __init__(
        self,
        m,
        camera: int,
        res: int,
        movers: list[int],
        groups=(0, 1, 2),
        enable_specular: bool = True,
        use_shadows: bool = True,
        device: str | torch.device = "cpu",
    ):
        import mujoco

        self.device = torch.device(device)
        self.res = int(res)
        self.enable_specular = bool(enable_specular)
        self.use_shadows = bool(use_shadows)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        mode = int(m.cam_mode[camera])
        if mode not in (0, 2):  # fixed, trackcom: orientation is constant
            raise ValueError(
                f"cpu3d: camera {camera} has mode {mode}; only a camera whose "
                f"orientation is a constant of the model (fixed/trackcom) is drawn here"
            )
        if int(m.cam_projection[camera]) != 0:
            raise ValueError(
                f"cpu3d: camera {camera} has cam_projection {int(m.cam_projection[camera])}; "
                f"only the perspective projection (0) is drawn"
            )
        if float(np.abs(np.asarray(m.cam_sensorsize[camera])).max()) != 0.0:
            raise ValueError(
                f"cpu3d: camera {camera} has a non-zero cam_sensorsize (intrinsics); the "
                f"fovy frustum is the only one implemented"
            )
        self.xmat = np.asarray(d.cam_xmat[camera], np.float64).reshape(3, 3)
        self.fovy = float(m.cam_fovy[camera])
        self.dirs = self._rays(self.xmat, self.fovy, self.res)  # (N,3) float32 world

        # the plane: exactly one, static (world body), textured or not
        planes = [
            g
            for g in range(m.ngeom)
            if int(m.geom_type[g]) == PLANE and int(m.geom_group[g]) in groups and int(m.geom_bodyid[g]) == 0
        ]
        if len(planes) != 1:
            raise ValueError(f"cpu3d: expected one static plane in groups {groups}, found {planes}")
        g = planes[0]
        self.plane_pos = np.asarray(d.geom_xpos[g], np.float64).copy()
        self.plane_mat = np.asarray(d.geom_xmat[g], np.float64).reshape(3, 3).copy()
        self.plane_size = np.asarray(m.geom_size[g], np.float64).copy()
        self.plane_matid = int(m.geom_matid[g])
        self.plane_reflectance = float(m.mat_reflectance[self.plane_matid]) if self.plane_matid >= 0 else 0.0
        self.plane_tex = None
        self.plane_texscale = np.ones(2)
        self.plane_texoffset = np.full(2, -0.5)
        if self.plane_matid >= 0:
            t = m.mat_texid[self.plane_matid]
            tex = int(t[1]) if np.ndim(t) else int(t)  # the RGB role, as MJWarp
            if tex >= 0:
                self.plane_tex = torch.as_tensor(texture_array(m, tex).astype(np.float32) / 255.0, device=self.device)
                self.plane_texscale, self.plane_texoffset = plane_texgen(
                    self.plane_size,
                    m.mat_texrepeat[self.plane_matid],
                    bool(m.mat_texuniform[self.plane_matid]),
                    tuple(int(v) for v in mujoco.__version__.split(".")[:2]),
                )
        repeat = m.mat_texrepeat[self.plane_matid] if self.plane_matid >= 0 else (0, 0)
        self.plane_grid = tuple(
            torch.as_tensor(
                plane_grid(
                    float(size),
                    float(self.plane_size[2]),
                    float(m.vis.map.zfar * m.stat.extent),
                    float(rep),
                    int(mujoco.mjMAXPLANEGRID),
                ),
                dtype=torch.float32,
                device=self.device,
            )
            for size, rep in zip(self.plane_size[:2], repeat, strict=True)
        )
        self.plane_mips = mip_pyramid(self.plane_tex) if self.plane_tex is not None else None
        self.plane_uv_gradients = self._plane_uv_gradients() if self.plane_mips is not None else None

        # Camera orientation is fixed, so filtered sky pixels can be cached.
        skybox = [t for t in range(m.ntex) if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)]
        if skybox:
            tex = torch.as_tensor(texture_array(m, skybox[0]).astype(np.float32) / 255.0, device=self.device)
            self.sky = sample_skybox_filtered(
                tex,
                int(m.tex_width[skybox[0]]),
                torch.as_tensor(self.dirs, device=self.device).reshape(self.res, self.res, 3),
            ).reshape(-1, 3)
        else:
            self.sky = torch.tensor(SKY_FALLBACK, device=self.device).expand(self.dirs.shape[0], 3)

        # movers: the geoms whose pose the state decides
        self.movers = [int(g) for g in movers]
        kinds = [int(m.geom_type[g]) for g in self.movers]
        bad = sorted({k for k in kinds if k not in MOVER_KINDS})
        if bad:
            names = [f"{GEOM_NAMES.get(k, k)} (mjtGeom {k})" for k in bad]
            raise ValueError(
                f"cpu3d draws the analytic primitives {[GEOM_NAMES[k] for k in MOVER_KINDS]} and this "
                f"model's movers include {names}; there is no CPU intersector for them"
            )
        self.kinds = kinds
        self.sizes = torch.as_tensor(
            np.asarray([m.geom_size[g] for g in self.movers], np.float64), dtype=torch.float32, device=self.device
        ).reshape(-1, 3)
        self.mover_matid = np.asarray([int(m.geom_matid[g]) for g in self.movers], np.int64)
        self.mover_rgba = np.asarray([m.geom_rgba[g] for g in self.movers], np.float64).reshape(-1, 4)
        # The same conservative radius bounds mover pixels and possible shadows.
        self.radii = torch.as_tensor(
            bound_radius(kinds, self.sizes.cpu().numpy()), dtype=torch.float32, device=self.device
        )

        # materials: specular / shininess per material id, the kernel's defaults for none
        nmat = int(m.nmat)
        spec = np.full(nmat + 1, DEFAULT_MAT_SPECULAR)
        shin = np.full(nmat + 1, DEFAULT_MAT_SHININESS * MAX_SHININESS)
        emis = np.zeros(nmat + 1)
        for i in range(nmat):
            spec[i] = float(m.mat_specular[i])
            shin[i] = float(m.mat_shininess[i]) * MAX_SHININESS
            emis[i] = float(m.mat_emission[i])
        # MJWarp samples texel (0, 0) on non-plane primitives without generated UVs.
        tex00 = np.ones((nmat + 1, 3))
        for i in range(nmat):
            t = m.mat_texid[i]
            tex = int(t[1]) if np.ndim(t) else int(t)
            if tex >= 0:
                tex00[i] = texture_array(m, tex)[0, 0].astype(np.float64) / 255.0
        self.mat_tex00 = torch.as_tensor(tex00, dtype=torch.float32, device=self.device)
        self.mat_spec = torch.as_tensor(spec, dtype=torch.float32, device=self.device)
        self.mat_shin = torch.as_tensor(shin, dtype=torch.float32, device=self.device)
        self.mat_emis = torch.as_tensor(emis, dtype=torch.float32, device=self.device)
        self.nmat = nmat
        # per-primitive material index; -1 (no material) maps to the defaults slot ``nmat``
        mid = np.concatenate([[self.plane_matid], self.mover_matid])
        self.prim_mat = torch.as_tensor(np.where(mid >= 0, mid, nmat), dtype=torch.int64, device=self.device)
        # a geom with no material carries its own rgba; the caller's ``mat_rgb`` table gets a row
        # per primitive appended for those so the gather is uniform
        self.prim_own_rgb = torch.as_tensor(
            np.concatenate([[np.asarray(m.geom_rgba[g], np.float64)[:3]], self.mover_rgba[:, :3]]),
            dtype=torch.float32,
            device=self.device,
        )

        # lights, in model order; the headlight from vis
        self.lights = [
            Light(
                kind=int(m.light_type[i])
                if hasattr(m, "light_type")
                else (LIGHT_DIRECTIONAL if bool(m.light_directional[i]) else LIGHT_SPOT),
                active=bool(m.light_active[i]) if hasattr(m, "light_active") else True,
                castshadow=bool(m.light_castshadow[i]),
                cutoff_deg=float(m.light_cutoff[i]),
                exponent=float(m.light_exponent[i]),
                ambient=np.asarray(m.light_ambient[i], np.float64),
                diffuse=np.asarray(m.light_diffuse[i], np.float64),
                specular=np.asarray(m.light_specular[i], np.float64),
                attenuation_default=bool(np.allclose(np.asarray(m.light_attenuation[i], np.float32), [1.0, 0.0, 0.0])),
            )
            for i in range(int(m.nlight))
        ]
        for L in self.lights:
            if L.kind not in (LIGHT_SPOT, LIGHT_DIRECTIONAL):
                raise ValueError(
                    f"cpu3d: light type {L.kind} (mjtLightType: 2 POINT, 3 IMAGE) is not "
                    f"implemented; SPOT (0) and DIRECTIONAL (1) are"
                )
            if L.castshadow and L.kind == LIGHT_DIRECTIONAL and self.use_shadows:
                raise ValueError(
                    "cpu3d: a shadow-casting DIRECTIONAL light is not implemented (the "
                    "megakernel casts its shadow rays with max_t = 1e8; the footprint "
                    "bound here requires a positional spotlight)"
                )
            if not L.attenuation_default:
                raise ValueError(
                    "cpu3d: non-default light attenuation is not implemented "
                    "(the megakernel folds the default at compile time; add the "
                    "1/(k0 + k1 d + k2 d^2) term before drawing such a model)"
                )
        hl = m.vis.headlight
        self.headlight_active = bool(hl.active)
        self.headlight = (
            np.asarray(hl.ambient, np.float64),
            np.asarray(hl.diffuse, np.float64),
            np.asarray(hl.specular, np.float64),
        )
        self.plane_z_up = bool(np.allclose(self.plane_mat[:, 2], [0, 0, 1], rtol=0, atol=1e-12))
        if (
            self.use_shadows
            and any(light.active and light.castshadow for light in self.lights)
            and not self.plane_z_up
        ):
            raise ValueError(
                "cpu3d: shadow bounds require a horizontal upward-facing plane; use_shadows=False disables them"
            )

    def _plane_uv_gradients(self) -> torch.Tensor:
        """Cache UV derivatives per unit signed camera-to-plane distance."""
        rays = torch.as_tensor(self.dirs, device=self.device)
        rotation = torch.as_tensor(self.xmat, dtype=torch.float32, device=self.device)
        plane = torch.as_tensor(self.plane_mat, dtype=torch.float32, device=self.device)
        rays = rays / -_dot(rays, rotation[:, 2])[:, None]
        denominator = _dot(rays, plane[:, 2])
        denominator = torch.where(denominator.abs() > MJ_MINVAL, denominator, torch.full_like(denominator, MJ_MINVAL))
        pixel = 2 * math.tan(math.radians(self.fovy) * 0.5) / self.res
        gradients = []
        for derivative in (pixel * rotation[:, 0], -pixel * rotation[:, 1]):
            hit_derivative = (
                derivative - rays * (_dot(derivative, plane[:, 2]) / denominator)[:, None]
            ) / denominator[:, None]
            gradients.extend(
                (
                    0.5 * float(self.plane_texscale[0]) * _dot(hit_derivative, plane[:, 0]),
                    -0.5 * float(self.plane_texscale[1]) * _dot(hit_derivative, plane[:, 1]),
                )
            )
        return torch.stack(gradients, dim=-1)

    @staticmethod
    def _rays(xmat: np.ndarray, fovy: float, res: int) -> np.ndarray:
        """Construct float32 pixel-center rays in MJWarp's arithmetic order.

        Normalize in camera space before rotating; the near distance cancels.
        """
        W = H = int(res)
        f = np.float32(math.tan(0.5 * math.radians(fovy)))
        u = (np.arange(W, dtype=np.float32) + np.float32(0.5)) / np.float32(W)
        v = (np.arange(H, dtype=np.float32) + np.float32(0.5)) / np.float32(H)
        x = (-f + (2.0 * f) * u).astype(np.float32)  # left + (right-left)*u
        y = (f + (-2.0 * f) * v).astype(np.float32)  # top + (bottom-top)*v
        X, Y = np.meshgrid(x, y)
        dloc = np.stack([X, Y, np.full_like(X, -1.0, dtype=np.float32)], -1).reshape(-1, 3)
        nrm = np.sqrt((dloc * dloc).sum(-1, keepdims=True)).astype(np.float32)
        dloc = (dloc / nrm).astype(np.float32)
        R = xmat.astype(np.float32)
        d = np.stack(
            [
                R[0, 0] * dloc[:, 0] + R[0, 1] * dloc[:, 1] + R[0, 2] * dloc[:, 2],
                R[1, 0] * dloc[:, 0] + R[1, 1] * dloc[:, 1] + R[1, 2] * dloc[:, 2],
                R[2, 0] * dloc[:, 0] + R[2, 1] * dloc[:, 1] + R[2, 2] * dloc[:, 2],
            ],
            -1,
        )
        return d.astype(np.float32)

    def mat_table(self, mat_rgb: torch.Tensor) -> torch.Tensor:
        """Extend (B,nmat,3) colours with a defaults slot and each primitive's own RGB."""
        B = mat_rgb.shape[0]
        own = self.prim_own_rgb.unsqueeze(0).expand(B, -1, -1)
        pad = torch.zeros((B, 1, 3), dtype=mat_rgb.dtype, device=mat_rgb.device)
        return torch.cat([mat_rgb, pad, own], 1)


# ``_dot`` is ``primitives.dot3``, imported at the top: the same unrolled expression the
# intersectors use, for the same reproducibility reason, from one definition instead of two.


def _normalize(v: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(v / |v|, |v|)``, leaving a zero vector alone rather than dividing by zero."""
    n = torch.sqrt(_dot(v, v))
    return v / torch.where(n > 0, n, torch.ones_like(n)).unsqueeze(-1), n


# Distinguish no visible movers from None, which requests the full frame.
WINDOW_EMPTY = "empty"


def _mover_window(scene: Scene, xpos_rel: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, int, int] | str | None:
    """Bound camera-relative (B,K,3) movers with per-row pixel windows.

    Return (x0, y0, common_width, common_height), None for the full frame, or
    WINDOW_EMPTY when every mover is offscreen. Pad each bound by one pixel.
    """
    B, K = xpos_rel.shape[:2]
    W = H = scene.res
    R = torch.as_tensor(scene.xmat, dtype=torch.float32, device=xpos_rel.device)
    # camera coordinates: columns of xmat are right / up / back
    loc = torch.stack([_dot(xpos_rel, R[:, 0]), _dot(xpos_rel, R[:, 1]), _dot(xpos_rel, R[:, 2])], -1)
    depth = -loc[..., 2]  # forward distance
    r = scene.radii.view(1, K)
    f = math.tan(0.5 * math.radians(scene.fovy))
    near = depth - r
    if bool((near <= WINDOW_NEAR).any()):
        return None
    inv = 1.0 / near
    px = (loc[..., 0] * inv / f + 1.0) * 0.5 * W - 0.5
    py = (-loc[..., 1] * inv / f + 1.0) * 0.5 * H - 0.5
    pr = r * inv / f * 0.5 * W
    x0 = (px - pr).floor().amin(1).long() - 1
    x1 = (px + pr).ceil().amax(1).long() + 2
    y0 = (py - pr).floor().amin(1).long() - 1
    y1 = (py + pr).ceil().amax(1).long() + 2
    x0 = x0.clamp(0, W)
    y0 = y0.clamp(0, H)
    x1 = x1.clamp(0, W)
    y1 = y1.clamp(0, H)
    w = int((x1 - x0).max())
    h = int((y1 - y0).max())
    if w <= 0 or h <= 0:
        return WINDOW_EMPTY
    if w * h >= W * H:
        return None
    x0 = torch.minimum(x0, torch.full_like(x0, W - w))
    y0 = torch.minimum(y0, torch.full_like(y0, H - h))
    return x0, y0, w, h


def _window_index(x0: torch.Tensor, y0: torch.Tensor, w: int, h: int, W: int, device) -> torch.Tensor:
    """``(B, h*w)`` flat pixel indices of each env's window."""
    ar_y = torch.arange(h, device=device).view(1, h, 1)
    ar_x = torch.arange(w, device=device).view(1, 1, w)
    return ((y0.view(-1, 1, 1) + ar_y) * W + (x0.view(-1, 1, 1) + ar_x)).reshape(x0.shape[0], -1)


def _light_footprint(scene: Scene, xpos_rel: torch.Tensor, light_rel: torch.Tensor, cam: torch.Tensor):
    """Return (xy_min, xy_max, unbounded) shadow bounds for a spotlight.

    Project bounding-box corners onto the horizontal floor. A corner at or above
    the light sets that row's boolean unbounded mask: test every floor pixel.
    """
    B, K = xpos_rel.shape[:2]
    r = scene.radii.view(1, K, 1)
    c = xpos_rel + cam.view(B, 1, 3)  # world
    signs = torch.tensor(
        [[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
        dtype=torch.float32,
        device=xpos_rel.device,
    ).view(1, 1, 8, 3)
    q = c.unsqueeze(2) + r.unsqueeze(2) * signs  # (B,K,8,3)
    L = (light_rel + cam).view(B, 1, 1, 3)
    zp = float(scene.plane_pos[2])
    dz = q[..., 2] - L[..., 2]
    below = dz < -1e-6
    s = (zp - L[..., 2]) / torch.where(below, dz, torch.full_like(dz, -1.0))
    hit = L + s.unsqueeze(-1) * (q - L)
    xy = hit[..., :2].reshape(B, -1, 2)
    unbounded = ~below.reshape(B, -1).all(1)
    lo = xy.amin(1) - FOOTPRINT_MARGIN
    hi = xy.amax(1) + FOOTPRINT_MARGIN
    return lo, hi, unbounded


@torch.no_grad()
def render(
    scene: Scene,
    cam_pos: torch.Tensor,
    xpos: torch.Tensor,
    xmat: torch.Tensor,
    light_pos: torch.Tensor,
    light_dir: torch.Tensor,
    mat_rgb: torch.Tensor,
    light_diffuse: torch.Tensor,
    out: torch.Tensor | None = None,
    *,
    chunk_size: int = CPU_CHUNK,
) -> torch.Tensor:
    """``(B,H,W,3)`` uint8.

    ``cam_pos (B,3)``, ``xpos (B,K,3)``, ``xmat (B,K,3,3)`` world poses of the movers,
    ``light_pos``/``light_dir (B,L,3)``, ``mat_rgb (B,nmat,3)``, ``light_diffuse (B,L,3)`` — all
    float32 on the scene's device, all as MuJoCo's own ``mj_camlight`` produces them.
    """
    if isinstance(chunk_size, bool) or not isinstance(chunk_size, int) or chunk_size <= 0:
        raise ValueError("chunk_size must be a positive integer")
    B = int(cam_pos.shape[0])
    W = H = scene.res
    if out is None:
        out = torch.empty((B, H, W, 3), dtype=torch.uint8, device=scene.device)
    for lo in range(0, B, chunk_size):
        hi = min(B, lo + chunk_size)
        out[lo:hi] = _render_chunk(
            scene,
            cam_pos[lo:hi],
            xpos[lo:hi],
            xmat[lo:hi],
            light_pos[lo:hi],
            light_dir[lo:hi],
            mat_rgb[lo:hi],
            light_diffuse[lo:hi],
        ).view(hi - lo, H, W, 3)
    return out


def _render_chunk(
    scene: Scene,
    cam: torch.Tensor,
    xpos: torch.Tensor,
    xmat: torch.Tensor,
    light_pos: torch.Tensor,
    light_dir: torch.Tensor,
    mat_rgb: torch.Tensor,
    light_diffuse: torch.Tensor,
) -> torch.Tensor:
    """One ``CPU_CHUNK``-sized group of frames: ``(B, N, 3)`` uint8, flat in the pixel axis."""
    dev = scene.device
    B = int(cam.shape[0])
    K = len(scene.kinds)
    W = H = scene.res
    N = W * H
    d = torch.as_tensor(scene.dirs, device=dev)  # (N,3)
    dB = d.unsqueeze(0).expand(B, N, 3)

    # ---- everything relative to the camera: the origin is 0 for every env
    xpos_rel = xpos - cam.view(B, 1, 3)
    if K:
        inside = (torch.sqrt(_dot(xpos_rel, xpos_rel)) <= scene.radii.view(1, K)).any()
        if bool(inside):
            raise ValueError(
                "cpu3d: the camera is inside a mover's bounding sphere in this batch; the "
                "megakernel's backface cull would then change which hit is returned and "
                "this backend does not reproduce that case"
            )
    pn = torch.as_tensor(scene.plane_mat[:, 2], dtype=torch.float32, device=dev)
    ppos_w = torch.as_tensor(scene.plane_pos, dtype=torch.float32, device=dev)
    for li, L in enumerate(scene.lights):
        if L.active and L.castshadow and scene.use_shadows:
            above = _dot(light_pos[:, li] - ppos_w, pn) > 0
            if not bool(above.all()):
                raise ValueError(
                    "cpu3d: a shadow-casting light is on or below the plane in this batch; "
                    "the footprint bound on shadow rays assumes the light above the plane"
                )
    ppos = ppos_w.view(1, 3) - cam  # (B,3)
    pm = torch.as_tensor(scene.plane_mat, dtype=torch.float32, device=dev)

    # ---- the plane, exactly ``ray_plane``: local frame, front-face only, bounded when size > 0
    lo_p = torch.stack([_dot(-ppos, pm[:, 0]), _dot(-ppos, pm[:, 1]), _dot(-ppos, pm[:, 2])], -1)  # (B,3) lpnt
    lv = torch.stack([_dot(d, pm[:, 0]), _dot(d, pm[:, 1]), _dot(d, pm[:, 2])], -1)  # (N,3) lvec
    lvz = lv[:, 2].unsqueeze(0)  # (1,N)
    front = lvz <= -MJ_MINVAL
    t_plane = -lo_p[:, 2].unsqueeze(1) / torch.where(front, lvz, torch.full_like(lvz, -1.0))  # (B,N)
    ok = front & (t_plane >= 0)
    p0 = lo_p[:, 0].unsqueeze(1) + t_plane * lv[:, 0].unsqueeze(0)
    p1 = lo_p[:, 1].unsqueeze(1) + t_plane * lv[:, 1].unsqueeze(0)
    sx, sy = float(scene.plane_size[0]), float(scene.plane_size[1])
    if sx > 0:
        ok = ok & (p0.abs() <= sx)
    if sy > 0:
        ok = ok & (p1.abs() <= sy)
    t_best = torch.where(ok, t_plane, torch.full_like(t_plane, INF))
    prim = torch.where(
        ok, torch.zeros_like(t_plane, dtype=torch.int64), torch.full_like(t_plane, -1, dtype=torch.int64)
    )  # 0 = plane, 1.. = movers
    n_plane = pm[:, 2]
    nrm = n_plane.view(1, 1, 3).expand(B, N, 3).clone()

    # ---- the movers, cast only inside their window, through the shared intersectors
    win = WINDOW_EMPTY if not K else (None if FORCE_FULL_FRAME else _mover_window(scene, xpos_rel))
    if win != WINDOW_EMPTY:
        if win is None:
            idx = None
            oo = torch.zeros((B, N, 3), device=dev)
            dd = dB
        else:
            x0, y0, w, h = win
            idx = _window_index(x0, y0, w, h, W, dev)  # (B, Nw)
            dd = d[idx]  # (B, Nw, 3)
            oo = torch.zeros(dd.shape, device=dev)
        tm, nm, im = R3.cast_nearest(scene.kinds, scene.sizes.unsqueeze(0).expand(B, K, 3), xpos_rel, xmat, oo, dd)
        # Match primary-ray backface culling. Cameras inside mover bounds are refused.
        tm = torch.where(_dot(nm, dd) > 0, torch.full_like(tm, INF), tm)
        if idx is None:
            upd = tm < t_best
            t_best = torch.where(upd, tm, t_best)
            prim = torch.where(upd, im + 1, prim)
            nrm = torch.where(upd.unsqueeze(-1), nm, nrm)
        else:
            tb = torch.gather(t_best, 1, idx)
            upd = tm < tb
            t_best.scatter_(1, idx, torch.where(upd, tm, tb))
            prim.scatter_(1, idx, torch.where(upd, im + 1, torch.gather(prim, 1, idx)))
            i3 = idx.unsqueeze(-1).expand(-1, -1, 3)
            nrm.scatter_(1, i3, torch.where(upd.unsqueeze(-1), nm, torch.gather(nrm, 1, i3)))

    # ---- shade only the rays that hit something, compacted across the chunk
    hit = prim >= 0
    flat = torch.nonzero(hit.reshape(-1), as_tuple=False).reshape(-1)
    env = flat // N
    ray = flat % N
    sky = scene.sky  # (N,3) in [0,1]
    rgb = sky.unsqueeze(0).expand(B, N, 3).clone()
    if flat.numel():
        t = t_best.reshape(-1)[flat]
        n = nrm.reshape(-1, 3)[flat]
        pid = prim.reshape(-1)[flat]
        dsel = d[ray]
        p = dsel * t.unsqueeze(-1)  # camera-relative hit
        view = -dsel
        table = scene.mat_table(mat_rgb)  # (B, nmat+1+P, 3)
        mat_idx = scene.prim_mat[pid]  # material slot
        own = torch.where(mat_idx == scene.nmat, scene.nmat + 1 + pid, mat_idx)
        base = table[env, own]
        # textured non-plane geoms: the megakernel's texel (0, 0); ones for the plane and untextured
        base = base * torch.where((pid > 0).unsqueeze(-1), scene.mat_tex00[mat_idx], torch.ones_like(base))
        # Plane UVs follow the source texgen law; the projected footprint selects mip levels.
        if scene.plane_tex is not None:
            on_plane = pid == 0
            if bool(on_plane.any()):
                j = torch.nonzero(on_plane, as_tuple=False).reshape(-1)
                pw = p[j] + cam[env[j]]  # world hit point
                rel = pw - ppos_w
                lx = _dot(rel, pm[:, 0])
                ly = _dot(rel, pm[:, 1])
                rep = scene.plane_texscale
                u = 0.5 * lx * float(rep[0]) + float(scene.plane_texoffset[0])
                v = -0.5 * ly * float(rep[1]) + float(scene.plane_texoffset[1])
                u = u - torch.floor(u)
                v = v - torch.floor(v)
                base = base.clone()
                height = _dot(ppos_w - cam[env[j]], pn)
                gradient = scene.plane_uv_gradients[ray[j]] * height[:, None]
                base[j] = base[j] * sample_plane_filtered(scene.plane_mips, u, v, *gradient.unbind(-1))
        rgb.view(-1, 3)[flat] = _shade_hits(
            scene, base, p, n, view, pid, env, cam, xpos_rel, xmat, light_pos, light_dir, light_diffuse
        )
    if scene.plane_reflectance > 0 and K:
        _reflect(scene, rgb, prim, t_plane, cam, xpos_rel, xmat, light_pos, light_dir, mat_rgb, light_diffuse)
    # ``pack_rgba_to_uint32``: C ``int()`` truncates; under ``gl_rounding`` it is int(x*255 + 0.5)
    q = rgb * 255.0
    if QUANT == "round":
        q = q + 0.5
    return torch.floor(q).clamp_(0, 255).to(torch.uint8)


def _plane_vertices(scene, p, owners, cam):
    """Return unique visible grid vertices and each floor hit's triangle weights."""
    matrix = torch.as_tensor(scene.plane_mat, dtype=torch.float32, device=p.device)
    origin = torch.as_tensor(scene.plane_pos, dtype=torch.float32, device=p.device)
    local = p + cam[owners] - origin
    x, y = _dot(local, matrix[:, 0]), _dot(local, matrix[:, 1])
    gx, gy = scene.plane_grid
    ix = torch.searchsorted(gx, x.contiguous()).clamp(1, len(gx) - 1)
    iy = torch.searchsorted(gy, y.contiguous()).clamp(1, len(gy) - 1)
    fx = ((x - gx[ix - 1]) / (gx[ix] - gx[ix - 1])).clamp(0, 1)
    fy = ((y - gy[iy - 1]) / (gy[iy] - gy[iy - 1])).clamp(0, 1)
    first = (iy - 1) * len(gx) + ix - 1
    last = iy * len(gx) + ix
    other = first + torch.where(fx >= fy, 1, len(gx))
    keys = torch.stack((first, last, other), -1) + owners[:, None] * (len(gx) * len(gy))
    unique, inverse = torch.unique(keys, return_inverse=True)
    vertex_env = unique // (len(gx) * len(gy))
    vertex = unique % (len(gx) * len(gy))
    world = origin + gx[vertex % len(gx), None] * matrix[:, 0] + gy[vertex // len(gx), None] * matrix[:, 1]
    point = world - cam[vertex_env]
    weights = torch.stack((1 - torch.maximum(fx, fy), torch.minimum(fx, fy), (fx - fy).abs()), -1)
    return vertex_env, point, inverse, weights


def _interpolate_vertices(values, inverse, weights):
    return (values[inverse] * weights[..., None]).sum(1)


def _light_vectors(light, position, direction, point):
    if light.kind == LIGHT_DIRECTIONAL:
        light_dir, _ = _normalize(-direction)
        distance = torch.full_like(point[:, 0], 1e8)
        attenuation = torch.ones_like(distance)
        cone = torch.ones_like(distance, dtype=torch.bool)
    else:
        light_dir, distance = _normalize(position - point)
        axis, _ = _normalize(direction)
        cosine = _dot(-light_dir, axis)
        cone = cosine >= math.cos(math.radians(light.cutoff_deg))
        attenuation = torch.pow(cosine.clamp_min(0.0), light.exponent)
    return light_dir, distance, attenuation, cone


def _shade_hits(
    scene, base, p, n, view, pid, env, cam, xpos_rel, xmat, light_pos, light_dir, light_diffuse, *, shadows=True
):
    """Shade compact primary or reflected hits using the same material and lights."""
    dev = p.device
    t = p[:, 0]
    K = len(scene.kinds)
    use_shadows = scene.use_shadows and shadows
    mat_idx = scene.prim_mat[pid]
    pn = torch.as_tensor(scene.plane_mat[:, 2], dtype=torch.float32, device=dev)
    ppos_w = torch.as_tensor(scene.plane_pos, dtype=torch.float32, device=dev)
    pm = torch.as_tensor(scene.plane_mat, dtype=torch.float32, device=dev)
    sx, sy = float(scene.plane_size[0]), float(scene.plane_size[1])
    lrel = light_pos - cam[:, None]
    plane_rows = (pid == 0).nonzero(as_tuple=True)[0]
    if plane_rows.numel():
        vertex_env, vertex_point, inverse, weights = _plane_vertices(scene, p[plane_rows], env[plane_rows], cam)
        vertex_view, _ = _normalize(-vertex_point)
        vertex_normal = pn.expand_as(vertex_point)
    mspec = scene.mat_spec[mat_idx] if scene.enable_specular else torch.full_like(t, DEFAULT_MAT_SPECULAR)
    mshin = (
        scene.mat_shin[mat_idx] if scene.enable_specular else torch.full_like(t, DEFAULT_MAT_SHININESS * MAX_SHININESS)
    )
    memis = scene.mat_emis[mat_idx]
    result = base * memis.unsqueeze(-1)
    hl_amb, hl_dif, hl_spec = (torch.as_tensor(x, dtype=torch.float32, device=dev) for x in scene.headlight)
    if scene.headlight_active:
        result = result + base * hl_amb
    elif not scene.lights:
        result = result + base * NO_LIGHT_AMBIENT_FALLBACK
    for li, L in enumerate(scene.lights):
        if L.active:
            result = result + base * torch.as_tensor(L.ambient, dtype=torch.float32, device=dev)

    # shadow candidates: mover hits, and plane hits inside the movers' light footprint
    if use_shadows and K and any(L.castshadow for L in scene.lights):
        cand = pid > 0
        for li, L in enumerate(scene.lights):
            if not L.active or not L.castshadow or L.kind == LIGHT_DIRECTIONAL:
                continue
            lo_xy, hi_xy, unb = _light_footprint(scene, xpos_rel, lrel[:, li], cam)
            pw = (p + cam[env])[:, :2]
            inside = ((pw >= lo_xy[env]) & (pw <= hi_xy[env])).all(-1) | unb[env]
            cand = cand | ((pid == 0) & inside)
        if SHADOW_TEST_ALL_HITS:
            cand = torch.ones_like(pid, dtype=torch.bool)
    else:
        cand = torch.zeros_like(pid, dtype=torch.bool)

    for li, L in enumerate(scene.lights):
        if not L.active:
            continue
        Ldir, dist, att, cone = _light_vectors(L, lrel[env, li], light_dir[env, li], p)
        ndotl = _dot(n, Ldir).clamp_min(0.0)
        live = cone & (ndotl > 0)
        visible = torch.ones_like(t)
        if L.castshadow and use_shadows and K:
            q = torch.nonzero(live & cand, as_tuple=False).reshape(-1)
            if q.numel():
                o_s = p[q] + n[q] * SHADOW_ORIGIN_EPS
                d_s = Ldir[q]
                lim = dist[q] - SHADOW_MAX_T_EPS
                blocked = _any_blocking_flat(scene, xpos_rel, xmat, env[q], o_s, d_s, lim)
                # the plane occludes too (the megakernel shadows against its whole BVH):
                # ``ray_plane`` from the shadow origin, front face only, inside the bounds
                o_w = o_s + cam[env[q]]
                lz = _dot(d_s, pn)
                fr = lz <= -MJ_MINVAL
                tp = _dot(ppos_w - o_w, pn) / torch.where(fr, lz, torch.full_like(lz, -1.0))
                okp = fr & (tp >= 0) & (tp < lim)
                if sx > 0 or sy > 0:
                    hp = o_w + d_s * tp.unsqueeze(-1) - ppos_w
                    if sx > 0:
                        okp = okp & (_dot(hp, pm[:, 0]).abs() <= sx)
                    if sy > 0:
                        okp = okp & (_dot(hp, pm[:, 1]).abs() <= sy)
                blocked = blocked | okp
                vis_q = torch.where(blocked, torch.full_like(lim, SHADOW_VISIBLE), torch.ones_like(lim))
                visible = visible.clone()
                visible[q] = vis_q
        wgt = att * visible * live.to(t.dtype)
        dif = light_diffuse[env, li] * (ndotl * wgt).unsqueeze(-1)
        if plane_rows.numel():
            vertex_light, _, vertex_att, vertex_cone = _light_vectors(
                L, lrel[vertex_env, li], light_dir[vertex_env, li], vertex_point
            )
            vertex_dot = _dot(vertex_normal, vertex_light).clamp_min(0)
            vertex_weight = vertex_att * (vertex_cone & (vertex_dot > 0)).to(t.dtype)
            vertex_diffuse = light_diffuse[vertex_env, li] * (vertex_dot * vertex_weight)[:, None]
            dif[plane_rows] = _interpolate_vertices(vertex_diffuse, inverse, weights) * visible[plane_rows, None]
        result = result + base * dif
        if scene.enable_specular:
            sp = _specular(n, Ldir, view, mspec, mshin, wgt)
            if plane_rows.numel():
                plane_mat = scene.prim_mat[0]
                vertex_spec = _specular(
                    vertex_normal,
                    vertex_light,
                    vertex_view,
                    scene.mat_spec[plane_mat],
                    scene.mat_shin[plane_mat],
                    vertex_weight,
                )
                sp[plane_rows] = (
                    _interpolate_vertices(vertex_spec[:, None], inverse, weights)[:, 0] * visible[plane_rows]
                )
            result = result + torch.as_tensor(L.specular, dtype=torch.float32, device=dev) * sp.unsqueeze(-1)
    if scene.headlight_active:
        R = torch.as_tensor(scene.xmat, dtype=torch.float32, device=dev)
        Ldir = R[:, 2].view(1, 3).expand_as(n)  # -cam_fwd: toward the camera
        ndotl = _dot(n, Ldir).clamp_min(0.0)
        live = ndotl > 0
        wgt = live.to(t.dtype)
        result = result + base * (hl_dif * (ndotl * wgt).unsqueeze(-1))
        if scene.enable_specular:
            sp = _specular(n, Ldir, view, mspec, mshin, wgt)
            if plane_rows.numel():
                plane_mat = scene.prim_mat[0]
                vertex_light = R[:, 2].expand_as(vertex_point)
                vertex_weight = (_dot(vertex_normal, vertex_light) > 0).to(t.dtype)
                vertex_spec = _specular(
                    vertex_normal,
                    vertex_light,
                    vertex_view,
                    scene.mat_spec[plane_mat],
                    scene.mat_shin[plane_mat],
                    vertex_weight,
                )
                sp[plane_rows] = _interpolate_vertices(vertex_spec[:, None], inverse, weights)[:, 0]
            result = result + hl_spec * sp.unsqueeze(-1)
    return result.clamp(0.0, 1.0)


def _reflect(scene, rgb, prim, plane_t, cam, xpos_rel, xmat, light_pos, light_dir, mat_rgb, light_diffuse):
    """Add MuJoCo-style unshadowed reflections within mirrored mover bounds."""
    B, N = prim.shape
    dev = cam.device
    normal = torch.as_tensor(scene.plane_mat[:, 2], dtype=torch.float32, device=dev)
    floor = torch.as_tensor(scene.plane_pos, dtype=torch.float32, device=dev)
    height = _dot(xpos_rel + cam[:, None] - floor, normal)
    mirrored = xpos_rel - 2 * height[..., None] * normal
    win = None if FORCE_FULL_FRAME else _mover_window(scene, mirrored)
    if win == WINDOW_EMPTY:
        return
    idx = torch.arange(N, device=dev).expand(B, -1) if win is None else _window_index(*win, scene.res, dev)
    on_plane = prim.gather(1, idx) == 0
    if not bool(on_plane.any()):
        return
    rays = torch.as_tensor(scene.dirs, device=dev)[idx]
    origins = rays * plane_t.gather(1, idx)[..., None]
    directions = rays - 2 * _dot(rays, normal)[..., None] * normal
    origins = origins + normal * SHADOW_ORIGIN_EPS
    distance, normals, movers = R3.cast_nearest(
        scene.kinds, scene.sizes[None].expand(B, -1, -1), xpos_rel, xmat, origins, directions
    )
    seen = on_plane & (distance < INF) & (_dot(normals, directions) <= 0)
    env, pixel = seen.nonzero(as_tuple=True)
    if not env.numel():
        return
    pid = movers[env, pixel] + 1
    material = scene.prim_mat[pid]
    own = torch.where(material == scene.nmat, scene.nmat + 1 + pid, material)
    base = scene.mat_table(mat_rgb)[env, own] * scene.mat_tex00[material] * scene.plane_reflectance
    direction = directions[env, pixel]
    point = origins[env, pixel] + distance[env, pixel, None] * direction
    reflected = _shade_hits(
        scene,
        base,
        point,
        normals[env, pixel],
        -direction,
        pid,
        env,
        cam,
        xpos_rel,
        xmat,
        light_pos,
        light_dir,
        light_diffuse,
        shadows=False,
    )
    dst = idx[env, pixel]
    # MuJoCo scales reflected geom RGB before shading, then adds the floor (GL_ONE, GL_ONE).
    rgb[env, dst] = (reflected + rgb[env, dst]).clamp(0, 1)


def _specular(n, light, view, specular, shininess, weight):
    half, _ = _normalize(light + view)
    ndoth = _dot(n, half).clamp_min(0.0)
    value = specular * torch.pow(ndoth, shininess) * weight
    return torch.where((specular > 0) & (shininess > 0), value, torch.zeros_like(value))


def _any_blocking_flat(
    scene: Scene,
    xpos_rel: torch.Tensor,
    xmat: torch.Tensor,
    env: torch.Tensor,
    o: torch.Tensor,
    d: torch.Tensor,
    lim: torch.Tensor,
) -> torch.Tensor:
    """Test compacted shadow rays, returning (Q,) boolean occlusion.

    Group by environment, pad with zero-length rays, then restore input order.
    """
    B, K = xpos_rel.shape[:2]
    Q = env.numel()
    order = torch.argsort(env, stable=True)
    env_s = env[order]
    counts = torch.bincount(env_s, minlength=B)
    M = int(counts.max())
    starts = torch.cumsum(counts, 0) - counts
    slot = torch.arange(Q, device=env.device) - starts[env_s]
    oo = torch.zeros((B, M, 3), device=o.device)
    dd = torch.zeros((B, M, 3), device=o.device)
    dd[..., 2] = 1.0
    ll = torch.zeros((B, M), device=o.device)
    oo[env_s, slot] = o[order]
    dd[env_s, slot] = d[order]
    ll[env_s, slot] = lim[order]
    blocked = R3.any_blocking(scene.kinds, scene.sizes.unsqueeze(0).expand(B, K, 3), xpos_rel, xmat, oo, dd, ll)
    out = torch.empty(Q, dtype=torch.bool, device=o.device)
    out[order] = blocked[env_s, slot]
    return out
