"""MJWarp model preparation, texture uploads, and batched CUDA rendering.

Import sets WARP_CACHE_PATH to ~/.cache/warp only when the caller has not set it.
Optional CUDA dependencies are loaded when a render context is requested.
"""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass, field

import numpy as np

from ..._core.warp_cuda import require_warp_cuda
from .. import _resources

os.environ.setdefault("WARP_CACHE_PATH", os.path.expanduser("~/.cache/warp"))

#: Default visible groups in MuJoCo MjvOption.
DEFAULT_GROUPS = (0, 1, 2)
#: Group that ``prepare_model`` moves invisible geometry into, so MJWarp does not draw it opaque.
HIDDEN_GROUP = 5
#: mjtGeom codes whose UVs MJWarp can derive. Everything else samples texel (0,0) when textured.
TEXTURED_OK = (0, 7)  # PLANE, MESH
#: The MJWarp sampler accepts 2-D maps; cube and skybox maps require a different path.
TEXTURE_SAMPLER_OK = (0,)  # mjTEXTURE_2D
#: OpenGL accepts spot cutoffs in [0, 90], plus the non-spot sentinel 180.
GL_SPOT_CUTOFF_MAX = 90.0
#: cos(180 degrees) = -1 disables the shader's hard cone test.
GL_SPOT_CUTOFF_OFF = 180.0


def effective_alpha(m, kind: str = "geom") -> np.ndarray:
    """Resolve geom or site alpha using MuJoCo's material precedence.

    The element's RGBA wins unless it matches the compiler default; then its
    material supplies the alpha. Return one value per element.
    """
    rgba = np.asarray(m.geom_rgba if kind == "geom" else m.site_rgba, np.float64)
    matid = np.asarray(m.geom_matid if kind == "geom" else m.site_matid, np.int64)
    out = rgba[:, 3].copy()
    default = np.all(np.isclose(rgba, [0.5, 0.5, 0.5, 1.0]), axis=1)
    has_mat = matid >= 0
    take = default & has_mat
    if take.any():
        out[take] = np.asarray(m.mat_rgba, np.float64)[matid[take], 3]
    return out


@dataclass
class Prepared:
    """Model edits and saved arrays needed to undo them."""

    hidden_geoms: list[int]
    hidden_sites: list[int]
    groups: tuple[int, ...]
    restore: dict
    neutralised_spots: list[int] = field(default_factory=list)

    def summary(self) -> str:
        """Describe the hidden elements and adjusted spot lights in one line."""
        out = (
            f"{len(self.hidden_geoms)} invisible geoms and {len(self.hidden_sites)} invisible "
            f"sites moved out of groups {self.groups}"
        )
        if self.neutralised_spots:
            out += (
                f"; {len(self.neutralised_spots)} spot light(s) with an out-of-range cutoff "
                f"given MuJoCo's semantics (no angular attenuation)"
            )
        return out


def neutralise_illegal_spots(m) -> tuple[list[int], dict]:
    """Disable angular attenuation for invalid spot cutoffs, in place.

    OpenGL rejects cutoffs outside [0, 90] except the non-spot sentinel 180.
    Set both exponent=0 and cutoff=180: changing only the exponent leaves a hard
    cone in MJWarp. Keep the light type unchanged; MuJoCo's classic renderer
    handles point lights differently. Return affected light IDs and original arrays.
    """
    hit, restore = (
        [],
        {"light_exponent": np.array(m.light_exponent, copy=True), "light_cutoff": np.array(m.light_cutoff, copy=True)},
    )
    for i in range(int(m.nlight)):
        if hasattr(m, "light_type"):
            is_spot = int(m.light_type[i]) == 0
        else:  # MuJoCo builds without light_type expose only the directional flag
            is_spot = not int(m.light_directional[i])
        if not is_spot:
            continue
        cutoff = float(m.light_cutoff[i])
        if not (0.0 <= cutoff <= GL_SPOT_CUTOFF_MAX):
            m.light_exponent[i] = 0.0
            m.light_cutoff[i] = GL_SPOT_CUTOFF_OFF
            hit.append(i)
    return hit, restore


def prepare_model(m, groups=DEFAULT_GROUPS, alpha_eps: float = 0.0, spot_semantics: bool = True) -> Prepared:
    """Prepare a model for MJWarp, in place, and return restoration data.

    Hide geoms and sites whose effective alpha is at most ``alpha_eps`` by moving
    them from enabled groups to HIDDEN_GROUP. Optionally normalize spot cutoffs.
    The caller must exclude HIDDEN_GROUP from rendering.
    """
    ga = effective_alpha(m, "geom")
    sa = effective_alpha(m, "site")
    restore = {"geom_group": np.array(m.geom_group, copy=True), "site_group": np.array(m.site_group, copy=True)}
    gh = [i for i in range(m.ngeom) if ga[i] <= alpha_eps and int(m.geom_group[i]) in groups]
    sh = [i for i in range(m.nsite) if sa[i] <= alpha_eps and int(m.site_group[i]) in groups]
    if gh:
        m.geom_group[gh] = HIDDEN_GROUP
    if sh:
        m.site_group[sh] = HIDDEN_GROUP
    spots = []
    if spot_semantics:
        spots, spot_restore = neutralise_illegal_spots(m)
        restore.update(spot_restore)
    return Prepared(gh, sh, tuple(groups), restore, spots)


def restore_model(m, p: Prepared) -> None:
    m.geom_group[:] = p.restore["geom_group"]
    m.site_group[:] = p.restore["site_group"]
    if "light_exponent" in p.restore:
        m.light_exponent[:] = p.restore["light_exponent"]
    if "light_cutoff" in p.restore:
        m.light_cutoff[:] = p.restore["light_cutoff"]


def _textured_geoms(m, groups):
    for g in range(m.ngeom):
        mid = int(m.geom_matid[g])
        if int(m.geom_group[g]) not in groups or mid < 0:
            continue
        ids = [int(x) for x in np.atleast_1d(m.mat_texid[mid]) if int(x) >= 0]
        if not ids:
            continue
        bad_kind = any(int(m.tex_type[t]) not in TEXTURE_SAMPLER_OK for t in ids)
        missing_uv = False
        if int(m.geom_type[g]) == 7 and hasattr(m, "mesh_texcoordnum"):
            mi = int(m.geom_dataid[g])
            missing_uv = mi >= 0 and int(m.mesh_texcoordnum[mi]) == 0
        yield g, mid, bad_kind or missing_uv


def texture_gaps(m, groups=DEFAULT_GROUPS) -> list[tuple[int, str, int]]:
    """List visible textured geoms with unsupported UVs or texture kinds.

    Return (geom ID, name, geom type) tuples. Missing mesh UVs or non-2-D maps
    cannot be sampled safely; unsupported primitive UVs instead produce a flat texel.
    """
    import mujoco

    return [
        (g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}", int(m.geom_type[g]))
        for g, _, unsafe in _textured_geoms(m, groups)
        if unsafe or int(m.geom_type[g]) not in TEXTURED_OK
    ]


def texture_mean_rgb(m, tex: int) -> np.ndarray:
    """Return the mean RGB of a texture as a three-element array in [0, 1]."""
    w, h = int(m.tex_width[tex]), int(m.tex_height[tex])
    nc = int(m.tex_nchannel[tex]) if hasattr(m, "tex_nchannel") else 3
    adr = int(m.tex_adr[tex])
    a = np.asarray(m.tex_data[adr : adr + w * h * nc], np.float64).reshape(h, w, nc)
    return (a[..., :3].mean((0, 1)) if nc >= 3 else np.repeat(a.mean((0, 1)), 3)) / 255.0


def unsamplable_texture_materials(m, groups=DEFAULT_GROUPS) -> dict[int, list[int]]:
    """Map material IDs to visible geoms with non-2-D textures or missing mesh UVs.

    This excludes primitives that can be sampled but use incorrect UVs.
    """
    out: dict[int, list[int]] = {}
    for g, mid, unsafe in _textured_geoms(m, groups):
        if unsafe:
            out.setdefault(mid, []).append(g)
    return out


def bake_unsamplable_textures(m, groups=DEFAULT_GROUPS) -> dict:
    """Replace unsupported texture materials with a mean RGB color, in place.

    Prefer the RGB texture role for the mean; otherwise use the first texture.
    Remove every texture role on each selected material, including roles that are
    individually supported. Return restoration arrays and records of all removals.

    This changes the image. Callers must declare and measure the approximation;
    it is not an exact texture conversion.
    """
    import mujoco

    rgb_role = int(getattr(mujoco.mjtTextureRole, "mjTEXROLE_RGB", 1))
    restore = {"mat_texid": np.array(m.mat_texid, copy=True), "mat_rgba": np.array(m.mat_rgba, copy=True)}
    targets = unsamplable_texture_materials(m, groups)
    baked = []
    for mid, geoms in sorted(targets.items()):
        t = m.mat_texid[mid]
        ids = [int(x) for x in np.atleast_1d(t) if int(x) >= 0]
        tex = int(t[rgb_role]) if np.ndim(t) and int(t[rgb_role]) >= 0 else (ids[0] if ids else -1)
        if tex < 0:
            continue
        mean = texture_mean_rgb(m, tex)
        # Record every removed texture role, not just the one supplying the mean color.
        roles = enumerate(t) if np.ndim(t) else [(None, t)]
        removed = [
            {
                "texture": int(tid),
                "role": role,
                "name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, int(tid)),
                "tex_type": int(m.tex_type[int(tid)]),
                "unsamplable": bool(int(m.tex_type[int(tid)]) not in TEXTURE_SAMPLER_OK),
            }
            for role, tid in roles
            if int(tid) >= 0
        ]
        m.mat_rgba[mid, :3] = mean
        if np.ndim(t):
            m.mat_texid[mid, :] = -1
        else:
            m.mat_texid[mid] = -1
        baked.append(
            {
                "material": mid,
                "material_name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, mid),
                "texture": tex,
                "texture_name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_TEXTURE, tex),
                "tex_type": int(m.tex_type[tex]),
                "removed_textures": removed,
                "n_removed": len(removed),
                "mean_rgb": [round(float(x), 6) for x in mean],
                "geoms": geoms,
                "geom_names": [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) for g in geoms],
            }
        )
    return {
        "baked": baked,
        "n_materials": len(baked),
        "n_geoms": sum(len(b["geoms"]) for b in baked),
        "restore": restore,
    }


def bake_primitive_textures(m, groups=DEFAULT_GROUPS) -> dict:
    """Replace primitive-only 2-D texture materials with their mean color, in place.

    Skip materials also worn by planes or meshes so their valid texturing remains
    intact. Return baked, mixed, and skipped material IDs plus restoration arrays.
    Use only for a declared visual approximation, not an exact-parity path.
    """
    import mujoco

    rgb_role = int(getattr(mujoco.mjtTextureRole, "mjTEXROLE_RGB", 1))
    restore = {"mat_texid": np.array(m.mat_texid, copy=True), "mat_rgba": np.array(m.mat_rgba, copy=True)}
    wearers: dict[int, set[int]] = {}
    for g in range(m.ngeom):
        if int(m.geom_group[g]) in groups and int(m.geom_matid[g]) >= 0:
            wearers.setdefault(int(m.geom_matid[g]), set()).add(int(m.geom_type[g]))
    baked, mixed, skipped = [], [], []
    for mid, types in wearers.items():
        t = m.mat_texid[mid]
        tex = int(t[rgb_role]) if np.ndim(t) else int(t)
        if tex < 0:
            continue
        if any(ty in TEXTURED_OK for ty in types):
            mixed.append(mid)
            continue
        if int(m.tex_type[tex]) != int(mujoco.mjtTexture.mjTEXTURE_2D):
            skipped.append(mid)  # cube/skybox: not what a primitive samples
            continue
        m.mat_rgba[mid, :3] = texture_mean_rgb(m, tex)
        if np.ndim(t):
            m.mat_texid[mid, :] = -1
        else:
            m.mat_texid[mid] = -1
        baked.append(mid)
    return {"baked": baked, "mixed": mixed, "skipped": skipped, "restore": restore}


def restore_primitive_textures(m, b: dict) -> None:
    m.mat_texid[:] = b["restore"]["mat_texid"]
    m.mat_rgba[:] = b["restore"]["mat_rgba"]


def needs_com_pos(m) -> bool:
    """Return whether any camera or light tracks or targets a subtree center of mass."""
    import mujoco

    com = (int(mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM), int(mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM))
    return bool(
        np.isin(np.asarray(m.cam_mode, np.int64), com).any() or np.isin(np.asarray(m.light_mode, np.int64), com).any()
    )


# Warp texture views do not own their Torch storage; retain it if the context cannot hold a reference.
TEX_KEEP_REASON = "mjwarp tex_lod texture handed to wp.from_torch (no ownership transfer)"


def release_render_context(rc) -> None:
    """Drop texture tensor keepalives attached to an MJWarp render context.

    The caller must also drop the context to release its device arrays. No EGL
    context is owned here, and repeated calls are safe.
    """
    try:
        rc._se_tex_lod_keep = None
    except Exception:  # noqa: BLE001
        pass  # a C type with no settable attribute


def release_cached_ctx(c) -> None:
    """Drop texture keepalives from a cached family context before eviction."""
    rc = getattr(c, "rc", None)
    if rc is not None:
        release_render_context(rc)


def _patch_target(module, name):
    target = getattr(module, name, None)
    if not callable(target):
        raise RuntimeError(f"incompatible MJWarp: {module.__name__}.{name} must be callable")
    return target


def create_render_context(m, tex_lod: int = 1, **kw):
    """Create an MJWarp context with one upload of the model's texture blob.

    ``tex_lod > 1`` area-downsamples textures before upload to the sampler. This is
    a visual approximation, not automatic mip selection; it needs its own fidelity
    measurement. Downsampled tensor storage stays alive with the context.

    Temporarily patches a process-global MJWarp helper. Do not overlap this call
    with other threads constructing contexts; the original helper is restored on exit.
    """
    import mujoco_warp as mjw
    import warp as wp
    from mujoco_warp._src import render_util as ru

    original = _patch_target(ru, "create_warp_texture")
    _patch_target(ru, "_convert_texture_data")
    blob: dict = {}
    lod = int(tex_lod)
    keep: list = []

    def hoisted(model, tex_id: int):
        if "data" not in blob:
            blob["data"] = wp.array(model.tex_data, dtype=wp.uint8)
        w, h = int(model.tex_width[tex_id]), int(model.tex_height[tex_id])
        out = wp.zeros((h, w, 4), dtype=float)
        wp.launch(
            ru._convert_texture_data,
            dim=(w, h),
            inputs=[w, int(model.tex_adr[tex_id]), int(model.tex_nchannel[tex_id]), blob["data"]],
            outputs=[out],
        )
        if lod > 1:
            import torch.nn.functional as F

            t = wp.to_torch(out)
            t = (
                F.interpolate(t.permute(2, 0, 1)[None], size=(max(1, h // lod), max(1, w // lod)), mode="area")[0]
                .permute(1, 2, 0)
                .contiguous()
            )
            keep.append(t)
            out = wp.from_torch(t)
        return wp.Texture2D(out, filter_mode=wp.TextureFilterMode.LINEAR)

    ru.create_warp_texture = hoisted
    try:
        rc = mjw.create_render_context(m, **kw)
    finally:
        ru.create_warp_texture = original
    if keep:
        # Keep downsampled tensor storage alive for exactly as long as its context when supported.
        try:
            rc._se_tex_lod_keep = keep
        except AttributeError:  # pragma: no cover
            for t in keep:
                # Without a settable attribute, retain the storage for the process lifetime, even past the cap.
                _resources.retain_process_lifetime(t, TEX_KEEP_REASON, overflow="retain")
    return rc


@contextlib.contextmanager
def gl_rounding():
    """Temporarily round RGB channels before MJWarp packs them to uint8.

    Enclose graph capture as well as warmup: each render call generates a kernel
    using the current packer, so a capture outside this block retains truncation.
    This patches process-global MJWarp internals; concurrent rendering must be excluded.
    """
    import warp as wp
    from mujoco_warp._src import render as R

    original = _patch_target(R, "pack_rgba_to_uint32")

    @wp.func
    def pack_rgba_round(r: float, g: float, b: float, a: float) -> wp.uint32:
        return wp.uint32((int(a) << int(24)) | (int(r + 0.5) << int(16)) | (int(g + 0.5) << int(8)) | int(b + 0.5))

    R.pack_rgba_to_uint32 = pack_rgba_round
    try:
        yield
    finally:
        R.pack_rgba_to_uint32 = original


class BatchRenderer(_resources.Closeable):
    """Render ``(B, nq)`` joint positions to ``(B, H, W, 3)`` uint8 RGB on CUDA.

    Owns batch-sized MJWarp model/data, render buffers, and an optional captured
    graph. Close explicitly or use a context manager to release those resources.

    With ``prepare=True``, the caller's MuJoCo model is modified and left modified.
    Use ``restore_model`` with ``prepared`` when that model must be reused elsewhere.
    Texture gaps are reported on the instance; they are not automatically baked.
    """

    resource_kind = "mjwarp.BatchRenderer"

    def __init__(
        self,
        m,
        batch: int,
        res: int = 224,
        camera: int = 0,
        groups=DEFAULT_GROUPS,
        textures: bool = True,
        shadows: bool = True,
        skybox: bool = True,
        prepare: bool = True,
        capture: bool = True,
    ):
        import mujoco
        import mujoco_warp as mjw

        wp = require_warp_cuda("MJWarp batch rendering")
        self.prepared = prepare_model(m, groups) if prepare else None
        self.texture_gaps = texture_gaps(m, groups)
        self.needs_com = needs_com_pos(m)
        self.batch, self.res = int(batch), int(res)

        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        self.m = mjw.put_model(m)
        self.d = mjw.put_data(m, d0, nworld=int(batch))
        self.rc = create_render_context(
            m,
            nworld=int(batch),
            cam_res=(int(res), int(res)),
            render_rgb=True,
            cam_active=[i == int(camera) for i in range(m.ncam)],
            enabled_geom_groups=list(groups),
            use_textures=textures,
            use_shadows=shadows,
            render_skybox=skybox,
        )
        import torch

        self._bgra = wp.to_torch(self.rc.rgb_data).view(torch.uint8).reshape(int(batch), int(res), int(res), 4)
        self._wp, self._mjw = wp, mjw
        self.graph = None
        self._opened()
        if capture:
            self._capture()

    def _release(self) -> None:
        """Drop the graph before its referenced context and simulation buffers."""
        self.graph = None
        self._bgra = None
        rc, self.rc = getattr(self, "rc", None), None
        if rc is not None:
            release_render_context(rc)
        self.d = None
        self.m = None

    def _frame(self):
        self._mjw.kinematics(self.m, self.d)
        if self.needs_com:
            # Kinematics does not update subtree_com, which these cameras and lights read.
            self._mjw.com_pos(self.m, self.d)
        self._mjw.camlight(self.m, self.d)
        self._mjw.refit_bvh(self.m, self.d, self.rc)
        self._mjw.render(self.m, self.d, self.rc)

    def _capture(self):
        self._frame()
        self._wp.synchronize()
        with self._wp.ScopedCapture() as cap:
            self._frame()
        self.graph = cap.graph

    def render(self, qpos):
        """Render device-resident ``(B, nq)`` positions and return uint8 RGB frames."""
        self._check_open("render")
        self._wp.to_torch(self.d.qpos).copy_(qpos)
        if self.graph is not None:
            self._wp.capture_launch(self.graph)
        else:
            self._frame()
        return self._bgra[..., [2, 1, 0]]  # MJWarp packs bytes as BGRA.
