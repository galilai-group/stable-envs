"""A CPU-native 3-D backend for MuJoCo scenes that MJWarp draws on the GPU: same renderer semantics,
different implementation.

MJWarp (``stable_envs/render/mjwarp.py``) draws the DM Control body-tracking cells — camera and
spotlight ride the torso, so no static plate exists in world space — with one megakernel per pixel:
a ray through the pixel centre, the nearest primitive by BVH, one bilinear texture tap on a plane,
a cube-mapped skybox on a miss, and MuJoCo's fixed-function composition (headlight at the camera,
per-light diffuse/specular with a spot cone, a 0.3 shadow factor from an any-hit ray). It runs on
the GPU only, and ``device="cpu"`` on those cells used to be answered by the GPU behind the caller's
back (Step 3 labelled the rows ``DEVICE_NOT_HONOURED``).

This module is the CPU answer, and it is the SAME picture computed a different way (the Step 6
policy: backend implementation may differ by device, renderer semantics may not). Every law below
is MJWarp's, transcribed from ``mujoco_warp/_src/render.py`` 3.11/3.12 — ``compute_ray``,
``sample_skybox``, ``sample_texture`` (PLANE branch), ``ray_plane``, the megakernel's shading order,
``pack_rgba_to_uint32`` under ``gl_rounding`` — and not MuJoCo's, because the contract these cells
are certified under is the visual-tier declaration measured on MJWarp's frames; matching MuJoCo's
mipmapped floor would be a *different* picture from the one the declaration describes.

What makes it a CPU design rather than a port (measured in
``examples/results/step6/cpu-3d/design.md``):

* **The camera translates and never rotates** (``mode="trackcom"``), so the ray-direction grid is a
  constant per (orientation, resolution) and the SKYBOX is a constant image in camera space —
  99.4–99.7 % of skybox pixels are bit-identical across the frozen corpus in MJWarp's own frames. It
  is baked once through MJWarp's own sampler and copied per frame.
* **The floor is one plane with one texture tap**, computed analytically per pixel; no BVH.
* **The movers are 6–19 capsules and spheres covering 2.5–7 % of the pixels**, cast only inside
  their projected window through ``raycast3d.cast_nearest`` (the shared intersectors, bit-identical
  to the fused CUDA kernel), never over the full frame.
* **Shadow rays run only where a shadow can land**: on mover hits, and on floor hits inside the
  movers' light-space footprint. Two PRECONDITIONS make that bound exact, and both are checked
  rather than assumed: the shadow casters are point/spot lights (a directional caster is refused at
  construction), and the light sits strictly above the plane in every frame (checked per chunk;
  then no upward shadow ray can hit the plane's front face, so the plane occludes nothing outside
  the tested set -- inside it the plane IS tested, as the megakernel tests its whole BVH). A third
  precondition, the camera outside every mover's bounding sphere (checked per chunk), makes the
  megakernel's primary-ray backface cull a no-op, and it is applied anyway.

Everything is elementwise per ray or per (ray, primitive), and the per-ray reductions are written
unrolled for the reason ``raycast3d._mv`` gives, so the frame is batch-size, order and chunk
invariant by construction. Bit-exactness against MJWarp is NOT claimed and is not achievable by a
different renderer (CUDA's 9-bit texture-filter weights, ``fast_math``, ``powf``); what is claimed is
the same picture to within a level on the overwhelming majority of pixels, and it is measured on the
frozen states rather than assumed (``examples/results/step6/cpu-3d/``).

The module knows nothing about environments: it takes a compiled ``MjModel`` plus a camera id to build
a ``Scene``, and per-frame poses (from MuJoCo's own ``mj_kinematics``/``mj_comPos``/``mj_camlight``)
to render. It imports neither ``warp`` nor CUDA anything.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch

from . import raycast3d as R3

#: mjtGeom codes this backend draws as movers (the ones ``raycast3d`` intersects) and the plane.
#: ELLIPSOID joined in Step 7 (``raycast3d._ellipsoid``, for DM Control's quadruped torso); nothing
#: else here is kind-specific -- the window and the shadow footprint read ``bound_radius``.
PLANE, SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX = 0, 2, 3, 4, 5, 6
MOVER_KINDS = (SPHERE, CAPSULE, ELLIPSOID, CYLINDER, BOX)
GEOM_NAMES = {0: "mjGEOM_PLANE", 1: "mjGEOM_HFIELD", 2: "mjGEOM_SPHERE", 3: "mjGEOM_CAPSULE",
              4: "mjGEOM_ELLIPSOID", 5: "mjGEOM_CYLINDER", 6: "mjGEOM_BOX", 7: "mjGEOM_MESH",
              8: "mjGEOM_SDF"}
#: mjtLightType: 0 SPOT, 1 DIRECTIONAL, 2 POINT, 3 IMAGE. The megakernel treats every non-directional
#: light as positional and applies the spot cone when the model has any spot light; this backend
#: implements SPOT and DIRECTIONAL and refuses POINT/IMAGE at ``Scene`` construction.
LIGHT_SPOT, LIGHT_DIRECTIONAL, LIGHT_POINT, LIGHT_IMAGE = 0, 1, 2, 3
#: MuJoCo's material defaults, which the megakernel uses for a geom without a material.
DEFAULT_MAT_SPECULAR, DEFAULT_MAT_SHININESS, MAX_SHININESS = 0.5, 0.5, 128.0
#: ``NO_LIGHT_AMBIENT_FALLBACK``: what MJWarp multiplies a shadowed light's contribution by. Not 0.
SHADOW_VISIBLE = 0.3
#: The shadow-ray offsets the megakernel uses.
SHADOW_ORIGIN_EPS, SHADOW_MAX_T_EPS = 1.0e-4, 1.0e-3
#: ``MJ_MINVAL``, the front-face threshold of ``ray_plane``.
MJ_MINVAL = 1e-15
#: HOW A FLOAT BECOMES A BYTE. ``mujoco_warp``'s ``pack_rgba_to_uint32`` TRUNCATES (C ``int()``) and
#: ``stable_envs.render.mjwarp.gl_rounding`` patches it to round -- but only for kernels generated
#: inside its block. ``DMControlTrackCom``'s context (``dmcontrol._tc_context``) captures its CUDA
#: graph OUTSIDE that block, so the frames the four trackcom cells are certified on -- the frozen
#: Step-4 digests and the declared visual-tier numbers -- are the truncating kernel's. The CPU
#: backend follows the frames as certified: ``"trunc"`` by default, ``"round"`` available. Measured
#: on the frozen states (examples/results/step6/cpu-3d/m1_local*): rounding leaves 83-91 % of floor
#: and body pixels exactly one level ABOVE MJWarp's; truncating removes that offset.
QUANT = "trunc"

#: TEST SWITCHES. ``FORCE_FULL_FRAME`` casts the movers over every ray instead of their projected
#: window; ``SHADOW_TEST_ALL_HITS`` sends a shadow ray from every lit hit instead of only those inside
#: the movers' light-space footprint. Both must give the identical frame -- the window and the
#: footprint are conservative bounds, not approximations -- and ``tests/unit/test_cpu3d.py`` asserts
#: it byte for byte. Module-level for the reason ``raycast3d.FORCE_BACKEND`` is.
FORCE_FULL_FRAME: bool = False
SHADOW_TEST_ALL_HITS: bool = False

#: Frames per compacted pass. Sixteen 84 px frames keep every (B,N) intermediate inside L2 on the
#: certification core, where an elementwise op costs 0.9 ns/element; at 64 frames it is 2.2 ns.
CPU_CHUNK = 16
#: CUDA's texture unit computes bilinear weights in 9-bit fixed point (8 fractional bits). MJWarp
#: samples through that unit; emulating the quantisation is the difference between "within a level"
#: and "mostly exact" on the floor and skybox, and it is measured rather than assumed (M1a). 0 = float.
TEX_FRAC_BITS = 8


# --------------------------------------------------------------------------- textures

def texture_array(m, tex: int) -> np.ndarray:
    """``(h, w, 3)`` uint8 for texture ``tex`` of a compiled model."""
    w, h = int(m.tex_width[tex]), int(m.tex_height[tex])
    nc = int(m.tex_nchannel[tex]) if hasattr(m, "tex_nchannel") else 3
    a0 = int(m.tex_adr[tex])
    buf = getattr(m, "tex_data", None)
    if buf is None:
        buf = m.tex_rgb
    a = np.asarray(buf[a0:a0 + w * h * nc], np.uint8).reshape(h, w, nc)
    return np.repeat(a, 3, -1) if nc == 1 else np.ascontiguousarray(a[..., :3])


def bilinear(tex: torch.Tensor, u: torch.Tensor, v: torch.Tensor, frac_bits: int | None = None
             ) -> torch.Tensor:
    """Sample ``tex (h,w,3)`` float32 at normalised ``(u, v)`` the way a CUDA texture unit does.

    Texel centres sit at ``(i + 0.5) / w``; the weights are the fractional parts of ``u*w - 0.5``,
    truncated to ``frac_bits`` bits (CUDA: 9-bit fixed point, 8 fractional). Coordinates are
    clamped at the border, which is the unit's default address mode; the callers below wrap ``u``
    and ``v`` into [0, 1) themselves, as MJWarp does, so clamping only decides the half-texel at
    the seam.
    """
    if frac_bits is None:                      # bound at CALL time, so the module switch reaches here
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
    zero = torch.zeros_like(rx)
    face = torch.zeros_like(rx, dtype=torch.int64)
    sc = zero.clone()
    tc = zero.clone()
    ma = torch.ones_like(rx)
    # the same three-way branch, in the same order, as the kernel
    mx = (arx >= ary) & (arx >= arz)
    my = ~mx & (ary >= arz)
    mz = ~mx & ~my
    px, nx = mx & (rx > 0), mx & ~(rx > 0)
    py, ny = my & (ry > 0), my & ~(ry > 0)
    pz, nz = mz & (rz > 0), mz & ~(rz > 0)
    ma = torch.where(mx, arx, torch.where(my, ary, arz))
    face = (torch.where(px, 0, 0) + torch.where(nx, 1, 0) + torch.where(py, 2, 0)
            + torch.where(ny, 3, 0) + torch.where(pz, 4, 0) + torch.where(nz, 5, 0))
    sc = torch.where(px, -rz, torch.where(nx, rz, torch.where(py, rx, torch.where(
        ny, rx, torch.where(pz, rx, -rx)))))
    tc = torch.where(px | nx, -ry, torch.where(py, rz, torch.where(ny, -rz, -ry)))
    ma_safe = torch.where(ma == 0, torch.ones_like(ma), ma)
    s = (sc / ma_safe + 1.0) * 0.5
    t = (tc / ma_safe + 1.0) * 0.5
    t_min = 0.5 / float(face_w)
    t = t.clamp(t_min, 1.0 - t_min)
    v = (face.to(t.dtype) + t) * (1.0 / 6.0)
    return bilinear(sky, s, v)


# --------------------------------------------------------------------------- the scene

@dataclass
class Light:
    kind: int                    # LIGHT_SPOT or LIGHT_DIRECTIONAL
    active: bool                 # ``light_active``: the megakernel skips an inactive light entirely
    castshadow: bool
    cutoff_deg: float
    exponent: float
    ambient: np.ndarray          # (3,)
    diffuse: np.ndarray          # (3,) the model's own; scaled per env by the caller
    specular: np.ndarray         # (3,)
    attenuation_default: bool    # (1, 0, 0) -> no distance term, as the kernel folds it


class Scene:
    """Everything a frame needs that the state cannot change: rays, skybox plate, plane, movers.

    Built from a compiled ``MjModel`` for one camera at one resolution. The camera's ORIENTATION is
    read from ``cam_xmat`` at the model's initial pose and assumed constant, which is what a
    ``fixed`` or ``trackcom`` camera guarantees; the builder asserts the mode. Materials are
    resolved per env at render time (``mat_rgb``), so an appearance knob costs nothing here.
    """

    def __init__(self, m, camera: int, res: int, movers: list[int], groups=(0, 1, 2),
                 enable_specular: bool = True, use_shadows: bool = True,
                 device: str | torch.device = "cpu"):
        import mujoco
        self.device = torch.device(device)
        self.res = int(res)
        self.camera = int(camera)
        self.enable_specular = bool(enable_specular)
        self.use_shadows = bool(use_shadows)
        d = mujoco.MjData(m)
        mujoco.mj_forward(m, d)
        mode = int(m.cam_mode[camera])
        if mode not in (0, 2):                         # fixed, trackcom: orientation is constant
            raise ValueError(f"cpu3d: camera {camera} has mode {mode}; only a camera whose "
                             f"orientation is a constant of the model (fixed/trackcom) is drawn here")
        if int(m.cam_projection[camera]) != 0:
            raise ValueError(f"cpu3d: camera {camera} has cam_projection {int(m.cam_projection[camera])}; "
                             f"only the perspective projection (0) is drawn")
        if float(np.abs(np.asarray(m.cam_sensorsize[camera])).max()) != 0.0:
            raise ValueError(f"cpu3d: camera {camera} has a non-zero cam_sensorsize (intrinsics); the "
                             f"fovy frustum is the only one implemented")
        self.xmat = np.asarray(d.cam_xmat[camera], np.float64).reshape(3, 3)
        self.fovy = float(m.cam_fovy[camera])
        self.dirs = self._rays(self.xmat, self.fovy, self.res)              # (N,3) float32 world

        # the plane: exactly one, static (world body), textured or not
        planes = [g for g in range(m.ngeom) if int(m.geom_type[g]) == PLANE
                  and int(m.geom_group[g]) in groups and int(m.geom_bodyid[g]) == 0]
        if len(planes) != 1:
            raise ValueError(f"cpu3d: expected one static plane in groups {groups}, found {planes}")
        g = planes[0]
        self.plane_pos = np.asarray(d.geom_xpos[g], np.float64).copy()
        self.plane_mat = np.asarray(d.geom_xmat[g], np.float64).reshape(3, 3).copy()
        self.plane_size = np.asarray(m.geom_size[g], np.float64).copy()
        self.plane_matid = int(m.geom_matid[g])
        self.plane_geom = g
        self.plane_tex = None
        self.plane_texrepeat = np.ones(2)
        if self.plane_matid >= 0:
            t = m.mat_texid[self.plane_matid]
            tex = int(t[1]) if np.ndim(t) else int(t)                     # the RGB role, as MJWarp
            if tex >= 0:
                self.plane_tex = torch.as_tensor(texture_array(m, tex).astype(np.float32) / 255.0,
                                                 device=self.device)
                self.plane_texrepeat = np.asarray(m.mat_texrepeat[self.plane_matid], np.float64)

        # the skybox plate, through MJWarp's own sampler, once
        self.sky = None
        skybox = [t for t in range(m.ntex) if int(m.tex_type[t]) == int(mujoco.mjtTexture.mjTEXTURE_SKYBOX)]
        if skybox:
            tex = torch.as_tensor(texture_array(m, skybox[0]).astype(np.float32) / 255.0,
                                  device=self.device)
            self.sky = sample_skybox(tex, int(m.tex_width[skybox[0]]),
                                     torch.as_tensor(self.dirs, device=self.device))   # (N,3) [0,1]
        else:
            self.sky = torch.tensor([0.1, 0.1, 0.2], device=self.device).expand(self.dirs.shape[0], 3)

        # movers: the geoms whose pose the state decides
        self.movers = [int(g) for g in movers]
        kinds = [int(m.geom_type[g]) for g in self.movers]
        bad = sorted({k for k in kinds if k not in MOVER_KINDS})
        if bad:
            names = [f"{GEOM_NAMES.get(k, k)} (mjtGeom {k})" for k in bad]
            raise ValueError(
                f"cpu3d draws the analytic primitives {[GEOM_NAMES[k] for k in MOVER_KINDS]} and this "
                f"model's movers include {names}; there is no CPU intersector for them")
        self.kinds = kinds
        self.sizes = torch.as_tensor(np.asarray([m.geom_size[g] for g in self.movers], np.float64),
                                     dtype=torch.float32, device=self.device).reshape(-1, 3)
        self.mover_matid = np.asarray([int(m.geom_matid[g]) for g in self.movers], np.int64)
        self.mover_rgba = np.asarray([m.geom_rgba[g] for g in self.movers], np.float64).reshape(-1, 4)
        self.radii = torch.as_tensor(R3.Primitives(
            kind=np.asarray(kinds), size=self.sizes.cpu().numpy(), mat_rgb=np.zeros((len(kinds), 3)),
            tex_rgb=np.ones((len(kinds), 3)), spec=np.zeros(len(kinds))).bound_radius(),
            dtype=torch.float32, device=self.device)

        # materials: specular / shininess per material id, the kernel's defaults for none
        nmat = int(m.nmat)
        spec = np.full(nmat + 1, DEFAULT_MAT_SPECULAR)
        shin = np.full(nmat + 1, DEFAULT_MAT_SHININESS * MAX_SHININESS)
        emis = np.zeros(nmat + 1)
        for i in range(nmat):
            spec[i] = float(m.mat_specular[i])
            shin[i] = float(m.mat_shininess[i]) * MAX_SHININESS
            emis[i] = float(m.mat_emission[i])
        # a textured NON-plane geom: ``sample_texture`` derives no UV for it and samples (0, 0), i.e.
        # the corner texel of its material's RGB texture modulates the base colour. Reproduced per
        # material as a constant colour (ones when untextured).
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
        self.prim_mat = torch.as_tensor(np.where(mid >= 0, mid, nmat), dtype=torch.int64,
                                        device=self.device)
        # a geom with no material carries its own rgba; the caller's ``mat_rgb`` table gets a row
        # per primitive appended for those so the gather is uniform
        self.prim_own_rgb = torch.as_tensor(
            np.concatenate([[np.asarray(m.geom_rgba[g], np.float64)[:3]], self.mover_rgba[:, :3]]),
            dtype=torch.float32, device=self.device)

        # lights, in model order; the headlight from vis
        self.lights = [Light(kind=int(m.light_type[i]) if hasattr(m, "light_type")
                             else (LIGHT_DIRECTIONAL if bool(m.light_directional[i]) else LIGHT_SPOT),
                             active=bool(m.light_active[i]) if hasattr(m, "light_active") else True,
                             castshadow=bool(m.light_castshadow[i]),
                             cutoff_deg=float(m.light_cutoff[i]), exponent=float(m.light_exponent[i]),
                             ambient=np.asarray(m.light_ambient[i], np.float64),
                             diffuse=np.asarray(m.light_diffuse[i], np.float64),
                             specular=np.asarray(m.light_specular[i], np.float64),
                             attenuation_default=bool(np.allclose(
                                 np.asarray(m.light_attenuation[i], np.float32), [1.0, 0.0, 0.0])))
                       for i in range(int(m.nlight))]
        for L in self.lights:
            if L.kind not in (LIGHT_SPOT, LIGHT_DIRECTIONAL):
                raise ValueError(f"cpu3d: light type {L.kind} (mjtLightType: 2 POINT, 3 IMAGE) is not "
                                 f"implemented; SPOT (0) and DIRECTIONAL (1) are")
            if L.castshadow and L.kind == LIGHT_DIRECTIONAL and self.use_shadows:
                raise ValueError("cpu3d: a shadow-casting DIRECTIONAL light is not implemented (the "
                                 "megakernel casts its shadow rays with max_t = 1e8; the footprint "
                                 "bound here is written for a point light)")
            if not L.attenuation_default:
                raise ValueError("cpu3d: non-default light attenuation is not implemented "
                                 "(the megakernel folds the default at compile time; add the "
                                 "1/(k0 + k1 d + k2 d^2) term before drawing such a model)")
        hl = m.vis.headlight
        self.headlight_active = bool(hl.active)
        self.headlight = (np.asarray(hl.ambient, np.float64), np.asarray(hl.diffuse, np.float64),
                          np.asarray(hl.specular, np.float64))
        self.plane_z_up = float(self.plane_mat[2, 2]) > 0.999

    @staticmethod
    def _rays(xmat: np.ndarray, fovy: float, res: int) -> np.ndarray:
        """``compute_ray`` of ``mujoco_warp`` for every pixel, then rotated into the world.

        Float32 throughout, in the kernel's own order: a normalised camera-frame direction, then
        ``cam_xmat @ dir``. ``znear`` cancels in the direction and is left at 1.
        """
        W = H = int(res)
        f = np.float32(math.tan(0.5 * math.radians(fovy)))
        u = (np.arange(W, dtype=np.float32) + np.float32(0.5)) / np.float32(W)
        v = (np.arange(H, dtype=np.float32) + np.float32(0.5)) / np.float32(H)
        x = (-f + (2.0 * f) * u).astype(np.float32)                        # left + (right-left)*u
        y = (f + (-2.0 * f) * v).astype(np.float32)                        # top + (bottom-top)*v
        X, Y = np.meshgrid(x, y)
        dloc = np.stack([X, Y, np.full_like(X, -1.0, dtype=np.float32)], -1).reshape(-1, 3)
        nrm = np.sqrt((dloc * dloc).sum(-1, keepdims=True)).astype(np.float32)
        dloc = (dloc / nrm).astype(np.float32)
        R = xmat.astype(np.float32)
        d = np.stack([R[0, 0] * dloc[:, 0] + R[0, 1] * dloc[:, 1] + R[0, 2] * dloc[:, 2],
                      R[1, 0] * dloc[:, 0] + R[1, 1] * dloc[:, 1] + R[1, 2] * dloc[:, 2],
                      R[2, 0] * dloc[:, 0] + R[2, 1] * dloc[:, 1] + R[2, 2] * dloc[:, 2]], -1)
        return d.astype(np.float32)

    def mat_table(self, mat_rgb: torch.Tensor) -> torch.Tensor:
        """``(B, nmat, 3)`` per-env material colours -> ``(B, nmat + 1 + P, 3)`` primitive base
        colours: material rows, then the defaults slot (unused for a geom with a material), then one
        row per primitive carrying its own rgba, which is what a material-less geom is coloured by."""
        B = mat_rgb.shape[0]
        own = self.prim_own_rgb.unsqueeze(0).expand(B, -1, -1)
        pad = torch.zeros((B, 1, 3), dtype=mat_rgb.dtype, device=mat_rgb.device)
        return torch.cat([mat_rgb, pad, own], 1)


# --------------------------------------------------------------------------- helpers

def _dot(a, b):
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _normalize(v):
    n = torch.sqrt(_dot(v, v))
    return v / torch.where(n > 0, n, torch.ones_like(n)).unsqueeze(-1), n


def _mover_window(scene: Scene, xpos_rel: torch.Tensor):
    """Per-env pixel rectangle enclosing every mover's bounding sphere, and its common size.

    ``xpos_rel`` is ``(B,K,3)`` mover positions RELATIVE TO THE CAMERA. Projected through the
    same frustum the rays use, padded by one pixel; a sphere that reaches the camera plane makes
    the window the whole frame. Returns ``(x0 (B,), y0 (B,), w, h)`` with ``w, h`` the batch-wide
    maximum so one gather serves the chunk, or ``None`` for the full frame.
    """
    B, K = xpos_rel.shape[:2]
    W = H = scene.res
    R = torch.as_tensor(scene.xmat, dtype=torch.float32, device=xpos_rel.device)
    # camera coordinates: columns of xmat are right / up / back
    loc = torch.stack([_dot(xpos_rel, R[:, 0]), _dot(xpos_rel, R[:, 1]), _dot(xpos_rel, R[:, 2])], -1)
    depth = -loc[..., 2]                                                   # forward distance
    r = scene.radii.view(1, K)
    f = math.tan(0.5 * math.radians(scene.fovy))
    near = depth - r
    if bool((near <= 1e-3).any()):
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
        return "empty"
    if w * h >= W * H:
        return None
    x0 = torch.minimum(x0, torch.full_like(x0, W - w))
    y0 = torch.minimum(y0, torch.full_like(y0, H - h))
    return x0, y0, w, h


def _window_index(x0, y0, w, h, W, device):
    """``(B, h*w)`` flat pixel indices of each env's window."""
    ar_y = torch.arange(h, device=device).view(1, h, 1)
    ar_x = torch.arange(w, device=device).view(1, 1, w)
    return ((y0.view(-1, 1, 1) + ar_y) * W + (x0.view(-1, 1, 1) + ar_x)).reshape(x0.shape[0], -1)


def _light_footprint(scene: Scene, xpos_rel: torch.Tensor, light_rel: torch.Tensor,
                     cam: torch.Tensor):
    """World-xy AABB, per env, of where the movers can shadow the plane from a point light.

    Every mover's bounding-sphere box is projected from the light onto the plane; a corner above
    the light makes the footprint unbounded (``None`` for that env means test every floor pixel).
    Conservative by construction, so restricting shadow rays to it cannot change a pixel.
    """
    B, K = xpos_rel.shape[:2]
    r = scene.radii.view(1, K, 1)
    c = xpos_rel + cam.view(B, 1, 3)                                       # world
    signs = torch.tensor([[sx, sy, sz] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)],
                         dtype=torch.float32, device=xpos_rel.device).view(1, 1, 8, 3)
    q = c.unsqueeze(2) + r.unsqueeze(2) * signs                            # (B,K,8,3)
    L = (light_rel + cam).view(B, 1, 1, 3)
    zp = float(scene.plane_pos[2])
    dz = q[..., 2] - L[..., 2]
    below = dz < -1e-6
    s = (zp - L[..., 2]) / torch.where(below, dz, torch.full_like(dz, -1.0))
    hit = L + s.unsqueeze(-1) * (q - L)
    xy = hit[..., :2].reshape(B, -1, 2)
    unbounded = ~below.reshape(B, -1).all(1)
    lo = xy.amin(1) - 0.05
    hi = xy.amax(1) + 0.05
    return lo, hi, unbounded


# --------------------------------------------------------------------------- the frame

@torch.no_grad()
def render(scene: Scene, cam_pos: torch.Tensor, xpos: torch.Tensor, xmat: torch.Tensor,
           light_pos: torch.Tensor, light_dir: torch.Tensor, mat_rgb: torch.Tensor,
           light_diffuse: torch.Tensor, out: torch.Tensor | None = None) -> torch.Tensor:
    """``(B,H,W,3)`` uint8.

    ``cam_pos (B,3)``, ``xpos (B,K,3)``, ``xmat (B,K,3,3)`` world poses of the movers,
    ``light_pos``/``light_dir (B,L,3)``, ``mat_rgb (B,nmat,3)``, ``light_diffuse (B,L,3)`` — all
    float32 on the scene's device, all as MuJoCo's own ``mj_camlight`` produces them.
    """
    B = int(cam_pos.shape[0])
    W = H = scene.res
    N = W * H
    if out is None:
        out = torch.empty((B, H, W, 3), dtype=torch.uint8, device=scene.device)
    for lo in range(0, B, CPU_CHUNK):
        hi = min(B, lo + CPU_CHUNK)
        out[lo:hi] = _render_chunk(scene, cam_pos[lo:hi], xpos[lo:hi], xmat[lo:hi],
                                   light_pos[lo:hi], light_dir[lo:hi], mat_rgb[lo:hi],
                                   light_diffuse[lo:hi]).view(hi - lo, H, W, 3)
    return out


def _render_chunk(scene, cam, xpos, xmat, light_pos, light_dir, mat_rgb, light_diffuse):
    dev = scene.device
    B = int(cam.shape[0])
    K = len(scene.kinds)
    W = H = scene.res
    N = W * H
    d = torch.as_tensor(scene.dirs, device=dev)                            # (N,3)
    dB = d.unsqueeze(0).expand(B, N, 3)

    # ---- everything relative to the camera: the origin is 0 for every env
    xpos_rel = xpos - cam.view(B, 1, 3)
    lrel = light_pos - cam.view(B, 1, 3)                                    # (B,L,3)
    # PRECONDITIONS (see the module docstring), checked on every chunk rather than assumed:
    if K:
        inside = (torch.sqrt(_dot(xpos_rel, xpos_rel)) <= scene.radii.view(1, K)).any()
        if bool(inside):
            raise ValueError("cpu3d: the camera is inside a mover's bounding sphere in this batch; the "
                             "megakernel's backface cull would then change which hit is returned and "
                             "this backend does not reproduce that case")
    pn = torch.as_tensor(scene.plane_mat[:, 2], dtype=torch.float32, device=dev)
    ppos_w = torch.as_tensor(scene.plane_pos, dtype=torch.float32, device=dev)
    for li, L in enumerate(scene.lights):
        if L.active and L.castshadow and scene.use_shadows:
            above = _dot(light_pos[:, li] - ppos_w, pn) > 0
            if not bool(above.all()):
                raise ValueError("cpu3d: a shadow-casting light is on or below the plane in this batch; "
                                 "the footprint bound on shadow rays assumes the light above the plane")
    ppos = torch.as_tensor(scene.plane_pos, dtype=torch.float32, device=dev).view(1, 3) - cam  # (B,3)
    pm = torch.as_tensor(scene.plane_mat, dtype=torch.float32, device=dev)

    # ---- the plane, exactly ``ray_plane``: local frame, front-face only, bounded when size > 0
    lo_p = torch.stack([_dot(-ppos, pm[:, 0]), _dot(-ppos, pm[:, 1]), _dot(-ppos, pm[:, 2])], -1)  # (B,3) lpnt
    lv = torch.stack([_dot(d, pm[:, 0]), _dot(d, pm[:, 1]), _dot(d, pm[:, 2])], -1)             # (N,3) lvec
    lvz = lv[:, 2].unsqueeze(0)                                            # (1,N)
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
    INF = R3.INF
    t_best = torch.where(ok, t_plane, torch.full_like(t_plane, INF))
    prim = torch.where(ok, torch.zeros_like(t_plane, dtype=torch.int64),
                       torch.full_like(t_plane, -1, dtype=torch.int64))   # 0 = plane, 1.. = movers
    n_plane = pm[:, 2]
    nrm = n_plane.view(1, 1, 3).expand(B, N, 3).clone()

    # ---- the movers, cast only inside their window, through the shared intersectors
    win = "empty" if not K else (None if FORCE_FULL_FRAME else _mover_window(scene, xpos_rel))
    if win != "empty":
        if win is None:
            idx = None
            oo = torch.zeros((B, N, 3), device=dev)
            dd = dB
        else:
            x0, y0, w, h = win
            idx = _window_index(x0, y0, w, h, W, dev)                       # (B, Nw)
            dd = d[idx]                                                     # (B, Nw, 3)
            oo = torch.zeros(dd.shape, device=dev)
        tm, nm, im = R3.cast_nearest(scene.kinds, scene.sizes.unsqueeze(0).expand(B, K, 3),
                                     xpos_rel, xmat, oo, dd)
        # the megakernel's backface cull on primary rays (``enable_backface_culling``): a hit whose
        # normal faces away from the ray is dropped. With the camera outside every mover (checked
        # above) no nearest hit is back-facing, so this is exact rather than an approximation.
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
    sky = scene.sky                                                          # (N,3) in [0,1]
    rgb = sky.unsqueeze(0).expand(B, N, 3).clone()
    if flat.numel():
        t = t_best.reshape(-1)[flat]
        n = nrm.reshape(-1, 3)[flat]
        pid = prim.reshape(-1)[flat]
        dsel = d[ray]
        p = dsel * t.unsqueeze(-1)                                           # camera-relative hit
        view = -dsel
        table = scene.mat_table(mat_rgb)                                     # (B, nmat+1+P, 3)
        mat_idx = scene.prim_mat[pid]                                        # material slot
        own = torch.where(mat_idx == scene.nmat, scene.nmat + 1 + pid, mat_idx)
        base = table[env, own]
        # textured non-plane geoms: the megakernel's texel (0, 0); ones for the plane and untextured
        base = base * torch.where((pid > 0).unsqueeze(-1), scene.mat_tex00[mat_idx],
                                  torch.ones_like(base))
        # texture: the plane's one bilinear tap under MJWarp's PLANE texgen law
        if scene.plane_tex is not None:
            on_plane = pid == 0
            if bool(on_plane.any()):
                j = torch.nonzero(on_plane, as_tuple=False).reshape(-1)
                pw = p[j] + cam[env[j]]                                      # world hit point
                rel = pw - torch.as_tensor(scene.plane_pos, dtype=torch.float32, device=dev)
                lx = _dot(rel, pm[:, 0])
                ly = _dot(rel, pm[:, 1])
                rep = scene.plane_texrepeat
                u = 0.5 * lx * float(rep[0]) - 0.5
                v = -0.5 * ly * float(rep[1]) - 0.5
                u = u - torch.floor(u)
                v = v - torch.floor(v)
                base = base.clone()
                base[j] = base[j] * bilinear(scene.plane_tex, u, v)
        mspec = scene.mat_spec[mat_idx] if scene.enable_specular else torch.full_like(t, DEFAULT_MAT_SPECULAR)
        mshin = scene.mat_shin[mat_idx] if scene.enable_specular else torch.full_like(t, DEFAULT_MAT_SHININESS * MAX_SHININESS)
        memis = scene.mat_emis[mat_idx]
        result = base * memis.unsqueeze(-1)
        hl_amb, hl_dif, hl_spec = (torch.as_tensor(x, dtype=torch.float32, device=dev) for x in scene.headlight)
        if scene.headlight_active:
            result = result + base * hl_amb
        elif not scene.lights:
            result = result + base * SHADOW_VISIBLE            # NO_LIGHT_AMBIENT_FALLBACK, nlight == 0
        for li, L in enumerate(scene.lights):
            if L.active:
                result = result + base * torch.as_tensor(L.ambient, dtype=torch.float32, device=dev)

        # shadow candidates: mover hits, and plane hits inside the movers' light footprint
        if scene.use_shadows and K and any(L.castshadow for L in scene.lights):
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
            if L.kind == LIGHT_DIRECTIONAL:
                Lv = -light_dir[env, li]
                Ldir, _ = _normalize(Lv)
                dist = torch.full_like(t, 1e8)
                att = torch.ones_like(t)
                cone = torch.ones_like(t, dtype=torch.bool)
            else:
                Ldir, dist = _normalize(lrel[env, li] - p)
                att = torch.ones_like(t)
                ldir_n, _ = _normalize(light_dir[env, li])
                cos_theta = _dot(-Ldir, ldir_n)
                cos_cut = math.cos(math.radians(L.cutoff_deg))
                cone = cos_theta >= cos_cut
                att = torch.pow(cos_theta.clamp_min(0.0), L.exponent)
            ndotl = _dot(n, Ldir).clamp_min(0.0)
            live = cone & (ndotl > 0)
            visible = torch.ones_like(t)
            if L.castshadow and scene.use_shadows and K:
                q = torch.nonzero(live & cand, as_tuple=False).reshape(-1)
                if q.numel():
                    o_s = p[q] + n[q] * SHADOW_ORIGIN_EPS
                    d_s = Ldir[q]
                    lim = (dist[q] - SHADOW_MAX_T_EPS)
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
            result = result + base * dif
            if scene.enable_specular:
                Hh, _ = _normalize(Ldir + view)
                ndoth = _dot(n, Hh).clamp_min(0.0)
                sp = (mspec * torch.pow(ndoth, mshin) * wgt)
                sp = torch.where((mspec > 0) & (mshin > 0), sp, torch.zeros_like(sp))
                result = result + torch.as_tensor(L.specular, dtype=torch.float32, device=dev) * sp.unsqueeze(-1)
        if scene.headlight_active:
            R = torch.as_tensor(scene.xmat, dtype=torch.float32, device=dev)
            Ldir = R[:, 2].view(1, 3).expand_as(n)                          # -cam_fwd: toward the camera
            ndotl = _dot(n, Ldir).clamp_min(0.0)
            live = ndotl > 0
            wgt = live.to(t.dtype)
            result = result + base * (hl_dif * (ndotl * wgt).unsqueeze(-1))
            if scene.enable_specular:
                Hh, _ = _normalize(Ldir + view)
                ndoth = _dot(n, Hh).clamp_min(0.0)
                sp = mspec * torch.pow(ndoth, mshin) * wgt
                sp = torch.where((mspec > 0) & (mshin > 0), sp, torch.zeros_like(sp))
                result = result + hl_spec * sp.unsqueeze(-1)
        rgb.view(-1, 3)[flat] = result.clamp(0.0, 1.0)
    # ``pack_rgba_to_uint32``: C ``int()`` truncates; under ``gl_rounding`` it is int(x*255 + 0.5)
    q = rgb * 255.0
    if QUANT == "round":
        q = q + 0.5
    return torch.floor(q).clamp_(0, 255).to(torch.uint8)


def _any_blocking_flat(scene, xpos_rel, xmat, env, o, d, lim):
    """Shadow rays for a compacted set of hit points: ``(Q,)`` bool via ``raycast3d.any_blocking``.

    The shared intersectors take ``(B, N, ...)`` bundles; the candidates are regrouped per env by
    sorting, cast as one ragged-padded bundle, and scattered back. Padding rays get ``lim = 0`` so
    they can never register a hit.
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
    blocked = R3.any_blocking(scene.kinds, scene.sizes.unsqueeze(0).expand(B, K, 3), xpos_rel, xmat,
                              oo, dd, ll)
    out = torch.empty(Q, dtype=torch.bool, device=o.device)
    out[order] = blocked[env_s, slot]
    return out
