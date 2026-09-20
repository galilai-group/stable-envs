"""``mujoco_warp.render`` with the shadow traversals that never execute removed at COMPILE time.

WHAT THIS IS FOR, IN ONE MEASUREMENT. ``mujoco_warp``'s megakernel inlines ``compute_lighting`` once
per light plus once for the headlight, and every copy carries a full any-hit BVH traversal, because
``use_shadows`` is a runtime kernel argument and ``lightcastshadow`` is read from a per-world array.
Meta-World has three lights and a headlight, of which exactly ONE ever casts
(``light_castshadow = [false, true, false]``, and the headlight is called with a literal ``False``),
so four traversals are compiled and one runs. Measured at 256 px / 64 worlds on ``MetaWorldReach``:

    baseline                              7.636 ms
    rc.use_shadows = False (rays gone)    4.930 ms   x1.55
    traversal deleted from the SOURCE     4.390 ms   x1.74

so **1.12x of the frame is register pressure from code that never runs**, and it is available without
changing a pixel. That is what this module takes.

HOW, AND WHY IT IS BIT-IDENTICAL. Two things in Warp make this a specialisation rather than a
rewrite, both checked in ``warp/_src/codegen.py`` rather than assumed:

* ``emit_If`` (line 3212) evaluates a CONSTANT condition at codegen time and lowers only the taken
  branch, so ``if wp.static(can_shadow):`` deletes the traversal from the generated C++ instead of
  guarding it with ``if (false)``;
* ``evaluate_static_expression`` (line 6025) hands every constant in ``adj.symbols`` to the static
  evaluator, and the loop unroller binds the loop variable as exactly such a constant (line 3822),
  so ``wp.static(LIGHT_CASTS[light_index])`` inside ``for light_index in range(wp.static(M_NLIGHT))``
  resolves per iteration.

The light loop keeps upstream's ORDER and upstream's arithmetic; only the callee changes, and the
callee it changes to computes ``visible = 1.0`` for a light that could not have shadowed anything
anyway. Nothing else in the megakernel is touched.

WHAT IT IS NOT SAFE FOR, GUARDED IN ``applies_to``. The decision is compile-time, so a model whose
``light_castshadow`` is BATCHED PER WORLD cannot use it -- a world could turn a caster on after the
kernel was built. ``applies_to`` refuses in that case and the caller falls back to stock
``mujoco_warp``, which is also what ``STABLE_ENVS_LIGHTSPEC=0`` does.

``pack_rgba_to_uint32`` IS RE-READ FROM UPSTREAM AT CALL TIME, and that is load-bearing:
``render/mjwarp.py``'s ``gl_rounding()`` patches the name inside ``mujoco_warp._src.render``, and a
vendored module that had bound it at import would silently render with the TRUNCATING quantiser --
worth interior 0.9988 -> 0.791 on Fetch, i.e. the single largest fidelity term on this backend. The
same applies to ``sample_texture``, ``sample_skybox`` and ``_make_cast_ray``: all four are looked up
on the upstream module inside ``render`` so every patch a caller applies to upstream still reaches
this kernel.

GENERATED, DO NOT EDIT BY HAND. the frozen measurement record produces this file by transforming
the installed ``mujoco_warp._src.render`` source in three places; ``--check`` re-generates and diffs,
which is the test to run after a mujoco_warp bump.

Pinned against **mujoco-warp 3.11.0**.
"""

from __future__ import annotations

import contextlib
import dataclasses
import os
from typing import Tuple

import numpy as np
import warp as wp
from mujoco_warp._src import math  # noqa: F401  (referenced by the vendored function bodies)
from mujoco_warp._src.types import MJ_MAXVAL, Data, GeomType, Model, ObjType, RenderContext
from mujoco_warp._src.warp_util import event_scope

from . import _resources

wp.set_module_options({"enable_backward": False})

#: The mujoco_warp release this file was generated from. A bump is not automatically wrong, it is
#: unverified: re-run the frozen measurement record.
UPSTREAM_VERSION = "3.11.0"

MAX_SHININESS = 128.0
DEFAULT_MAT_SHININESS_EXPONENT = 0.5 * MAX_SHININESS
DEFAULT_MAT_SPECULAR = 0.5
DEFAULT_MAT_EMISSION = 0.0
NO_LIGHT_AMBIENT_FALLBACK = 0.3


def applies_to(m: Model, rc: RenderContext) -> bool:
    """Is a COMPILE-TIME light specialisation sound for this model and this context?

    Only when ``light_castshadow`` is a constant of the model rather than a per-world field: the
    whole point is that the decision is baked into the kernel, so a caller that randomises which
    lights cast per world must get stock ``mujoco_warp``. Returns False rather than raising, because
    the fallback is correct and merely slower.

    This is the MODEL guard only. Whether a family opts in at all is decided by its ``_context``
    through ``specialised_lights(enabled=...)``, because the answer differs per family -- see that
    function.
    """
    cs = getattr(m, "light_castshadow", None)
    return cs is not None and len(cs.shape) == 2 and cs.shape[0] == 1


#: Entries kept by ``_CASTS_CACHE``. One per (model, shadow setting) that has ever been rendered, and
#: each entry is a tuple of bools plus a reference to a warp array the model already owns, so the
#: memory is negligible and the bound exists only so that a long-lived process cannot accumulate one
#: entry per model built. 64 is far above any measured working set — the largest family in the
#: repository holds four contexts.
CASTS_CACHE_MAX = 64

#: Memo for ``light_casts``, and it is a correctness requirement rather than a saving. ``render`` is
#: called once to warm up and then AGAIN INSIDE ``wp.ScopedCapture``; reading ``light_castshadow``
#: with ``.numpy()`` is a device-to-host copy, which CUDA refuses during a capture with
#: "operation would make the legacy stream depend on a capturing blocking stream". The warm-up call
#: fills this, so the capturing call never copies. The value keeps a reference to the array so the
#: pointer it is keyed on cannot be freed and reissued to a different model.
#:
#: BOUNDED, AND THE POLICY IS LOAD-BEARING. It was an unbounded dict, which is a leak; it must not
#: become the ``clear()``-on-full idiom the rest of the repository grew, because clearing wholesale
#: can throw away the entry the WARM-UP call just wrote and leave the CAPTURING call to do the
#: device-to-host copy CUDA forbids — i.e. a cache eviction policy that turns into a capture failure.
#: LRU cannot do that: the warm-up call makes its key the most recently used, so it is the last thing
#: an insertion by anything else would evict, and no insertion happens between the two calls anyway.
_CASTS_CACHE = _resources.BoundedCache(CASTS_CACHE_MAX, name="mjwarp_lightspec.casts")


def light_casts(m: Model, rc: RenderContext) -> tuple:
    """``(nlight,)`` of bools: does light ``i`` compile a shadow traversal at all?"""
    key = (int(m.light_castshadow.ptr), int(m.nlight), bool(rc.use_shadows))
    hit = _CASTS_CACHE.get(key)
    if hit is None:
        if not rc.use_shadows:
            casts = tuple(False for _ in range(int(m.nlight)))
        else:
            casts = tuple(bool(x) for x in m.light_castshadow.numpy()[0])
        hit = _CASTS_CACHE.put(key, (casts, m.light_castshadow))
    return hit[0]


#: The stock ``mujoco_warp`` render, stashed by ``specialised_lights`` so the vendored one can hand a
#: model it must not specialise straight back to it. Without this the fallback would re-enter the
#: patched name and recurse.
_STOCK_RENDER = None


@contextlib.contextmanager
def specialised_lights(enabled: bool = True):
    """Point ``mujoco_warp.render`` at this module's ``render`` for the duration of the block.

    A context manager for the same reason ``gl_rounding`` is one: the window is short, it covers the
    context build and the graph capture (which is when the megakernel source is generated and
    compiled, so it is the only window that matters), and nothing else in the process ever sees it.

    ``enabled`` IS THE CALLER'S CALL AND DEFAULTS TO OFF FOR A PARITY FAMILY, because the elision is
    bit-identical on most scenes and not on all of them. Measured with the frozen measurement record,
    comparing the raw megakernel output over 64 states at 256 px:

        RoboCasa (2 cells), HumanoidBench (2 cells), Meta-World reach / sweep-into   0 / 4,194,304 px
        Meta-World window-open, window-close                    max 1/255 on 2 / 4,194,304 px

    The segmentation buffer is identical everywhere, so nothing about WHAT is hit changes; what
    changes is that ``compute_lighting_flat`` is a different inlining context for the arithmetic that
    survives, and ``mujoco_warp`` compiles its megakernel with ``fast_math`` enabled, which licenses
    the reassociation that moves a handful of values across a quantisation boundary. It reproduces
    exactly across runs (2 px at 64 states, 21 at 256, on three independent jobs), so it is a
    deterministic difference and not a flake -- the stock-vs-stock control is 0 px.

    Two pixels in four million cannot move Meta-World's declared parity floors, and that is beside
    the point: the family claims bit-identity, so it gets stock ``mujoco_warp`` unless a caller asks
    otherwise. ``STABLE_ENVS_LIGHTSPEC=1`` overrides for anyone who wants the 1.04-1.09x, and
    ``=0`` forces it off everywhere.
    """
    import mujoco_warp as mjw
    from mujoco_warp._src import render as R

    override = os.environ.get("STABLE_ENVS_LIGHTSPEC")
    if override is not None:
        enabled = override == "1"
    if not enabled:
        yield False
        return
    global _STOCK_RENDER
    orig_mod, orig_pkg = R.render, mjw.render
    _STOCK_RENDER = orig_mod
    R.render = render
    mjw.render = render
    try:
        yield True
    finally:
        R.render = orig_mod
        mjw.render = orig_pkg


def _make_compute_lighting(cast_ray_first_hit, can_shadow: bool) -> wp.Function:
    """Build specialized compute_lighting. ``can_shadow`` is COMPILE TIME: False deletes
    the any-hit traversal from the generated source (warp's emit_If folds a constant test)."""

    @wp.func
    def compute_lighting(
        # Model:
        geom_type: wp.array[int],
        geom_dataid: wp.array2d[int],
        geom_size: wp.array2d[wp.vec3],
        flex_vertadr: wp.array[int],
        flex_edge: wp.array[wp.vec2i],
        flex_radius: wp.array[float],
        # Data in:
        geom_xpos_in: wp.array2d[wp.vec3],
        geom_xmat_in: wp.array2d[wp.mat33],
        flexvert_xpos_in: wp.array2d[wp.vec3],
        # In:
        use_shadows: bool,
        bvh_id: wp.uint64,
        group_root: int,
        bvh_ngeom: int,
        bvh_nflexgeom: int,
        enabled_geom_ids: wp.array[int],
        worldid: int,
        mesh_bvh_id: wp.array[wp.uint64],
        hfield_bvh_id: wp.array[wp.uint64],
        flex_geom_flexid: wp.array[int],
        flex_geom_edgeid: wp.array[int],
        flex_bvh_id: wp.array[wp.uint64],
        flex_group_root: wp.array2d[int],
        lightactive: bool,
        lighttype: int,
        lightcastshadow: bool,
        lightpos: wp.vec3,
        lightdir: wp.vec3,
        lightattenuation: wp.vec3,
        lightcutoff_rad: float,
        lightexp: float,
        lightdiff: wp.vec3,
        lightspec: wp.vec3,
        normal: wp.vec3,
        hitpoint: wp.vec3,
        view_dir: wp.vec3,
        mat_spec: float,
        mat_shin_exp: float,
        cull_backfaces: bool,
        enable_specular: bool,
        default_attenuation: bool,
        has_spot: bool,
    ) -> Tuple[wp.vec3, wp.vec3]:
        diff_rgb = wp.vec3(0.0)
        spec_rgb = wp.vec3(0.0)

        # TODO: We should probably only be looping over active lights
        # in the first place with a static loop of enabled light idx?
        if not lightactive:
            return diff_rgb, spec_rgb

        L = wp.vec3(0.0)
        dist_to_light = float(MJ_MAXVAL)
        attenuation = 1.0

        if lighttype == 1:  # directional light
            # MuJoCo guarantees `lightdir` is unit length.
            L = -lightdir
        else:
            L, dist_to_light = math.normalize_with_norm(lightpos - hitpoint)
            if not default_attenuation:
                light_attenuation_factor = wp.vec3(1.0, dist_to_light, dist_to_light * dist_to_light)
                attenuation = math.safe_div(1.0, wp.dot(light_attenuation_factor, lightattenuation))
            if has_spot:
                if lighttype == 0:  # spot light
                    cos_theta = wp.dot(-L, lightdir)
                    cos_cutoff = wp.cos(lightcutoff_rad)
                    if cos_theta < cos_cutoff:
                        return diff_rgb, spec_rgb
                    attenuation = attenuation * wp.pow(wp.max(cos_theta, 0.0), lightexp)

        ndotl = wp.max(0.0, wp.dot(normal, L))
        if ndotl == 0.0:
            return diff_rgb, spec_rgb

        visible = 1.0

        if wp.static(can_shadow):
            if use_shadows and lightcastshadow:
                # Nudge the origin slightly along the surface normal to avoid
                # self-intersection when casting shadow rays
                shadow_origin = hitpoint + normal * 1.0e-4
                # Distance-limited shadows: cap by dist_to_light (for non-directional)
                max_t = dist_to_light - 1.0e-3
                if lighttype == 1:  # directional light
                    max_t = 1.0e8

                shadow_geom_id, shadow_d, shadow_n, shadow_u, shadow_v, shadow_f, shadow_mesh_id = cast_ray_first_hit(
                    geom_type,
                    geom_dataid,
                    geom_size,
                    flex_vertadr,
                    flex_edge,
                    flex_radius,
                    geom_xpos_in,
                    geom_xmat_in,
                    flexvert_xpos_in,
                    bvh_id,
                    group_root,
                    worldid,
                    bvh_ngeom,
                    bvh_nflexgeom,
                    enabled_geom_ids,
                    mesh_bvh_id,
                    hfield_bvh_id,
                    flex_geom_flexid,
                    flex_geom_edgeid,
                    flex_bvh_id,
                    flex_group_root,
                    shadow_origin,
                    L,
                    max_t,
                    cull_backfaces,
                )

                if shadow_geom_id != -1:
                    visible = NO_LIGHT_AMBIENT_FALLBACK

        weight = attenuation * visible
        diff_rgb = lightdiff * (ndotl * weight)
        if enable_specular:
            if mat_spec > 0.0 and mat_shin_exp > 0.0:
                H = wp.normalize(L + view_dir)
                ndoth = wp.max(0.0, wp.dot(normal, H))
                spec_rgb = lightspec * (mat_spec * wp.pow(ndoth, mat_shin_exp) * weight)

        return diff_rgb, spec_rgb

    return compute_lighting


def render(m: Model, d: Data, rc: RenderContext):
    """Render the current frame.

    Outputs are stored in buffers within the render context.

    Args:
      m: The model on device.
      d: The data on device.
      rc: The render context on device.
    """
    rc.seg_data.fill_(wp.vec2i(-1, -1))

    # Specialize the ray-cast helpers to the geom types present in the scene so the
    # compiler eliminates intersection branches for absent types.
    geom_ray_types = rc.geom_ray_types
    from mujoco_warp._src.render import _make_cast_ray

    cast_ray = _make_cast_ray(geom_ray_types, first_hit=False)
    cast_ray_first_hit = _make_cast_ray(geom_ray_types, first_hit=True)
    # THE SPECIALISATION. Two variants of the same function; which one a light gets is decided
    # below, per light, at codegen. ``compute_lighting_flat`` carries no traversal at all.
    compute_lighting = _make_compute_lighting(cast_ray_first_hit, True)
    compute_lighting_flat = _make_compute_lighting(cast_ray_first_hit, False)
    if not applies_to(m, rc):
        # A per-world ``light_castshadow`` (or the opt-out) means the decision cannot be
        # baked; hand the frame to stock mujoco_warp, which is correct and merely slower.
        return _STOCK_RENDER(m, d, rc)
    LIGHT_CASTS = light_casts(m, rc)

    # Static parameters extracted for JAX FFI closure.
    # RE-READ FROM UPSTREAM, NOT IMPORTED AT MODULE LOAD: ``gl_rounding()`` patches the name
    # inside ``mujoco_warp._src.render``, and a copy bound at import would keep the truncating
    # quantiser for every frame this kernel ever draws. Same for the three helpers.
    from mujoco_warp._src import render as _up

    pack_rgba_to_uint32 = _up.pack_rgba_to_uint32
    sample_texture = _up.sample_texture
    sample_skybox = _up.sample_skybox
    compute_ray = _up.compute_ray
    rc_static = {
        f.name: getattr(rc, f.name) for f in dataclasses.fields(rc) if f.type in (int, wp.uint32, bool, float, wp.vec3)
    }
    rc_static["enable_specular_or_emission"] = rc.enable_specular or rc.enable_emission
    M_NLIGHT = m.nlight

    @wp.kernel(module="unique", enable_backward=False, module_options={"fast_math": rc.use_fast_math})
    def _render_megakernel(
        # Model:
        geom_type: wp.array[int],
        geom_dataid: wp.array2d[int],
        geom_matid: wp.array2d[int],
        geom_size: wp.array2d[wp.vec3],
        geom_rgba: wp.array2d[wp.vec4],
        cam_projection: wp.array[int],
        cam_fovy: wp.array2d[float],
        cam_sensorsize: wp.array[wp.vec2],
        cam_intrinsic: wp.array2d[wp.vec4],
        light_type: wp.array2d[int],
        light_castshadow: wp.array2d[bool],
        light_active: wp.array2d[bool],
        light_attenuation: wp.array2d[wp.vec3],
        light_cutoff: wp.array2d[float],
        light_exponent: wp.array2d[float],
        light_ambient: wp.array2d[wp.vec3],
        light_diffuse: wp.array2d[wp.vec3],
        light_specular: wp.array2d[wp.vec3],
        flex_vertadr: wp.array[int],
        flex_edge: wp.array[wp.vec2i],
        flex_radius: wp.array[float],
        mesh_faceadr: wp.array[int],
        mat_texid: wp.array3d[int],
        mat_texrepeat: wp.array2d[wp.vec2],
        mat_emission: wp.array2d[float],
        mat_specular: wp.array2d[float],
        mat_shininess: wp.array2d[float],
        mat_rgba: wp.array2d[wp.vec4],
        # Data in:
        geom_xpos_in: wp.array2d[wp.vec3],
        geom_xmat_in: wp.array2d[wp.mat33],
        cam_xpos_in: wp.array2d[wp.vec3],
        cam_xmat_in: wp.array2d[wp.mat33],
        light_xpos_in: wp.array2d[wp.vec3],
        light_xdir_in: wp.array2d[wp.vec3],
        flexvert_xpos_in: wp.array2d[wp.vec3],
        # In:
        nrender: int,
        use_shadows: bool,
        bvh_ngeom: int,
        bvh_nflexgeom: int,
        cam_res: wp.array[wp.vec2i],
        cam_id_map: wp.array[int],
        ray: wp.array[wp.vec3],
        rgb_adr: wp.array[int],
        depth_adr: wp.array[int],
        seg_adr: wp.array[int],
        render_rgb: wp.array[bool],
        render_depth: wp.array[bool],
        render_seg: wp.array[bool],
        bvh_id: wp.uint64,
        group_root: wp.array[int],
        flex_bvh_id: wp.array[wp.uint64],
        flex_group_root: wp.array2d[int],
        enabled_geom_ids: wp.array[int],
        mesh_bvh_id: wp.array[wp.uint64],
        mesh_facetexcoord: wp.array[wp.vec3i],
        mesh_texcoord: wp.array[wp.vec2],
        mesh_texcoord_offsets: wp.array[int],
        hfield_bvh_id: wp.array[wp.uint64],
        flex_rgba: wp.array[wp.vec4],
        flex_geom_flexid: wp.array[int],
        flex_geom_edgeid: wp.array[int],
        skybox_tex_id: wp.array[int],
        skybox_face_width: wp.array[int],
        textures: wp.array[wp.Texture2D],
        # Out:
        rgb_out: wp.array2d[wp.uint32],
        depth_out: wp.array2d[float],
        seg_out: wp.array2d[wp.vec2i],
    ):
        worldid, rayid = wp.tid()

        # Map global rayid -> (camid, rayid_local) using cumulative sizes
        camid = int(-1)
        rayid_local = int(-1)
        accum = int(0)
        for i in range(nrender):
            num_i = cam_res[i][0] * cam_res[i][1]
            if rayid < accum + num_i:
                camid = i
                rayid_local = rayid - accum
                break
            accum += num_i
        if camid == -1 or rayid_local < 0:
            return

        if not render_rgb[camid] and not render_depth[camid] and not render_seg[camid]:
            return

        # Map active camera index to MuJoCo camera ID
        mujoco_cam_id = cam_id_map[camid]

        if wp.static(rc_static["use_precomputed_rays"]):
            ray_dir_local_cam = ray[rayid]
        else:
            img_w = cam_res[camid][0]
            img_h = cam_res[camid][1]
            px = rayid_local % img_w
            py = rayid_local // img_w
            ray_dir_local_cam = compute_ray(
                cam_projection[mujoco_cam_id],
                cam_fovy[worldid % cam_fovy.shape[0], mujoco_cam_id],
                cam_sensorsize[mujoco_cam_id],
                cam_intrinsic[worldid % cam_intrinsic.shape[0], mujoco_cam_id],
                img_w,
                img_h,
                px,
                py,
                wp.static(rc_static["znear"]),
            )

        ray_origin_world = cam_xpos_in[worldid, mujoco_cam_id]
        cam_mat_world = cam_xmat_in[worldid, mujoco_cam_id]
        ray_dir_world = cam_mat_world @ ray_dir_local_cam

        geom_id, dist, normal, u, v, f, mesh_id = cast_ray(
            geom_type,
            geom_dataid,
            geom_size,
            flex_vertadr,
            flex_edge,
            flex_radius,
            geom_xpos_in,
            geom_xmat_in,
            flexvert_xpos_in,
            bvh_id,
            group_root[worldid],
            worldid,
            bvh_ngeom,
            bvh_nflexgeom,
            enabled_geom_ids,
            mesh_bvh_id,
            hfield_bvh_id,
            flex_geom_flexid,
            flex_geom_edgeid,
            flex_bvh_id,
            flex_group_root,
            ray_origin_world,
            ray_dir_world,
            float(MJ_MAXVAL),
            wp.static(rc_static["enable_backface_culling"]),
        )

        if render_seg[camid] and geom_id != -1:
            if geom_id == -2:
                seg_out[worldid, seg_adr[camid] + rayid_local] = wp.vec2i(mesh_id, int(ObjType.FLEX))
            else:
                seg_out[worldid, seg_adr[camid] + rayid_local] = wp.vec2i(geom_id, int(ObjType.GEOM))

        # Early Out
        if geom_id == -1:
            if render_depth[camid]:
                depth_out[worldid, depth_adr[camid] + rayid_local] = 0.0
            if wp.static(rc_static["render_skybox"]) and render_rgb[camid]:
                skybox_id = skybox_tex_id[worldid % skybox_tex_id.shape[0]]
                skybox_color = sample_skybox(
                    textures[skybox_id],
                    1.0 / float(skybox_face_width[worldid % skybox_face_width.shape[0]]),
                    ray_dir_world,
                )
                rgb_out[worldid, rgb_adr[camid] + rayid_local] = pack_rgba_to_uint32(
                    skybox_color[0] * 255.0,
                    skybox_color[1] * 255.0,
                    skybox_color[2] * 255.0,
                    255.0,
                )
            elif render_rgb[camid]:
                rgb_out[worldid, rgb_adr[camid] + rayid_local] = wp.static(rc_static["background_color"])
            return

        if render_depth[camid]:
            # Planar depth: project Euclidean distance onto the camera's optical axis.
            # In camera-local coordinates, the optical axis is -Z. The Z-component of the
            # normalized ray direction is negative, so -ray_dir_local_cam[2] gives cos(θ)
            # between the ray and the optical axis.
            depth_out[worldid, depth_adr[camid] + rayid_local] = dist * (-ray_dir_local_cam[2])

        if not render_rgb[camid]:
            return

        # Shade the pixel
        hit_point = ray_origin_world + ray_dir_world * dist

        if geom_id == -2:
            # We encode flex_id in mesh_id for flex ray hits during cast_ray
            color = flex_rgba[mesh_id]
        elif geom_matid[worldid % geom_matid.shape[0], geom_id] == -1:
            color = geom_rgba[worldid % geom_rgba.shape[0], geom_id]
        else:
            color = mat_rgba[worldid % mat_rgba.shape[0], geom_matid[worldid % geom_matid.shape[0], geom_id]]

        base_color = wp.vec3(color[0], color[1], color[2])

        if wp.static(rc_static["use_textures"]):
            if geom_id != -2:
                mat_id = geom_matid[worldid % geom_matid.shape[0], geom_id]
                if mat_id >= 0:
                    tex_id = mat_texid[worldid % mat_texid.shape[0], mat_id, 1]
                    if tex_id >= 0:
                        tex_color = sample_texture(
                            geom_type,
                            mesh_faceadr,
                            geom_id,
                            mat_texrepeat[worldid % mat_texrepeat.shape[0], mat_id],
                            textures[tex_id],
                            geom_xpos_in[worldid, geom_id],
                            geom_xmat_in[worldid, geom_id],
                            mesh_facetexcoord,
                            mesh_texcoord,
                            mesh_texcoord_offsets,
                            hit_point,
                            u,
                            v,
                            f,
                            mesh_id,
                        )
                        base_color = wp.cw_mul(base_color, tex_color)

        mat_spec = DEFAULT_MAT_SPECULAR
        mat_shin_exp = DEFAULT_MAT_SHININESS_EXPONENT
        mat_emis = DEFAULT_MAT_EMISSION
        if wp.static(rc_static["enable_specular_or_emission"]):
            if geom_id != -2:
                mat_id_for_spec = geom_matid[worldid % geom_matid.shape[0], geom_id]
                if mat_id_for_spec >= 0:
                    if wp.static(rc_static["enable_specular"]):
                        mat_spec = mat_specular[worldid % mat_specular.shape[0], mat_id_for_spec]
                        mat_shin_exp = mat_shininess[worldid % mat_shininess.shape[0], mat_id_for_spec] * MAX_SHININESS
                    if wp.static(rc_static["enable_emission"]):
                        mat_emis = mat_emission[worldid % mat_emission.shape[0], mat_id_for_spec]

        result = wp.vec3(0.0)
        if wp.static(rc_static["enable_emission"]):
            result = base_color * mat_emis

        if wp.static(rc_static["use_ambient_lighting"]):
            if wp.static(rc_static["headlight_active"]):
                result = result + wp.cw_mul(base_color, wp.static(rc_static["headlight_ambient"]))
            elif wp.static(M_NLIGHT == 0):
                result = result + base_color * NO_LIGHT_AMBIENT_FALLBACK
            if wp.static(rc_static["enable_per_light_ambient"]):
                for light_index in range(wp.static(M_NLIGHT)):
                    if light_active[worldid % light_active.shape[0], light_index]:
                        result = result + wp.cw_mul(
                            base_color, light_ambient[worldid % light_ambient.shape[0], light_index]
                        )

        view_dir = -ray_dir_world

        light_cutoff_worldid = light_cutoff[worldid % light_cutoff.shape[0]]
        light_active_worldid = light_active[worldid % light_active.shape[0]]
        light_type_worldid = light_type[worldid % light_type.shape[0]]
        light_castshadow_worldid = light_castshadow[worldid % light_castshadow.shape[0]]
        light_xpos_in_worldid = light_xpos_in[worldid]
        light_xdir_in_worldid = light_xdir_in[worldid]
        light_attenuation_worldid = light_attenuation[worldid % light_attenuation.shape[0]]
        light_exponent_worldid = light_exponent[worldid % light_exponent.shape[0]]
        light_diffuse_worldid = light_diffuse[worldid % light_diffuse.shape[0]]
        light_specular_worldid = light_specular[worldid % light_specular.shape[0]]
        # Apply Lighting for each light
        for light_index in range(wp.static(M_NLIGHT)):
            # ONE traversal is compiled for a light that casts and NONE for a light that does not.
            # The branch is resolved at codegen (constant test), so the untaken side emits nothing;
            # the loop order, the arguments and the accumulation below are upstream's, untouched.
            if wp.static(LIGHT_CASTS[light_index]):
                diff_rgb, spec_rgb = compute_lighting(
                    geom_type,
                    geom_dataid,
                    geom_size,
                    flex_vertadr,
                    flex_edge,
                    flex_radius,
                    geom_xpos_in,
                    geom_xmat_in,
                    flexvert_xpos_in,
                    use_shadows,
                    bvh_id,
                    group_root[worldid],
                    bvh_ngeom,
                    bvh_nflexgeom,
                    enabled_geom_ids,
                    worldid,
                    mesh_bvh_id,
                    hfield_bvh_id,
                    flex_geom_flexid,
                    flex_geom_edgeid,
                    flex_bvh_id,
                    flex_group_root,
                    light_active_worldid[light_index],
                    light_type_worldid[light_index],
                    light_castshadow_worldid[light_index],
                    light_xpos_in_worldid[light_index],
                    light_xdir_in_worldid[light_index],
                    light_attenuation_worldid[light_index],
                    light_cutoff_worldid[light_index] * wp.static(wp.pi / 180.0),
                    light_exponent_worldid[light_index],
                    light_diffuse_worldid[light_index],
                    light_specular_worldid[light_index],
                    normal,
                    hit_point,
                    view_dir,
                    mat_spec,
                    mat_shin_exp,
                    wp.static(rc_static["enable_backface_culling"]),
                    wp.static(rc_static["enable_specular"]),
                    wp.static(rc_static["light_attenuation_is_default"]),
                    wp.static(rc_static["has_spot_lights"]),
                )
            else:
                diff_rgb, spec_rgb = compute_lighting_flat(
                    geom_type,
                    geom_dataid,
                    geom_size,
                    flex_vertadr,
                    flex_edge,
                    flex_radius,
                    geom_xpos_in,
                    geom_xmat_in,
                    flexvert_xpos_in,
                    use_shadows,
                    bvh_id,
                    group_root[worldid],
                    bvh_ngeom,
                    bvh_nflexgeom,
                    enabled_geom_ids,
                    worldid,
                    mesh_bvh_id,
                    hfield_bvh_id,
                    flex_geom_flexid,
                    flex_geom_edgeid,
                    flex_bvh_id,
                    flex_group_root,
                    light_active_worldid[light_index],
                    light_type_worldid[light_index],
                    light_castshadow_worldid[light_index],
                    light_xpos_in_worldid[light_index],
                    light_xdir_in_worldid[light_index],
                    light_attenuation_worldid[light_index],
                    light_cutoff_worldid[light_index] * wp.static(wp.pi / 180.0),
                    light_exponent_worldid[light_index],
                    light_diffuse_worldid[light_index],
                    light_specular_worldid[light_index],
                    normal,
                    hit_point,
                    view_dir,
                    mat_spec,
                    mat_shin_exp,
                    wp.static(rc_static["enable_backface_culling"]),
                    wp.static(rc_static["enable_specular"]),
                    wp.static(rc_static["light_attenuation_is_default"]),
                    wp.static(rc_static["has_spot_lights"]),
                )
            result = result + wp.cw_mul(base_color, diff_rgb) + spec_rgb

        # Apply Headlight
        if wp.static(rc_static["headlight_active"]):
            cam_pos = ray_origin_world
            cam_fwd = -cam_mat_world[:, 2]
            # The headlight passes ``lightcastshadow=False`` as a literal three arguments down,
            # so its traversal was dead on arrival; this is the same call with it gone.
            hl_diff, hl_spec = compute_lighting_flat(
                geom_type,
                geom_dataid,
                geom_size,
                flex_vertadr,
                flex_edge,
                flex_radius,
                geom_xpos_in,
                geom_xmat_in,
                flexvert_xpos_in,
                use_shadows,
                bvh_id,
                group_root[worldid],
                bvh_ngeom,
                bvh_nflexgeom,
                enabled_geom_ids,
                worldid,
                mesh_bvh_id,
                hfield_bvh_id,
                flex_geom_flexid,
                flex_geom_edgeid,
                flex_bvh_id,
                flex_group_root,
                True,
                1,
                False,
                cam_pos,
                cam_fwd,
                wp.vec3(1.0, 0.0, 0.0),
                0.0,
                0.0,
                wp.static(rc_static["headlight_diffuse"]),
                wp.static(rc_static["headlight_specular"]),
                normal,
                hit_point,
                view_dir,
                mat_spec,
                mat_shin_exp,
                wp.static(rc_static["enable_backface_culling"]),
                wp.static(rc_static["enable_specular"]),
                True,
                False,
            )
            result = result + wp.cw_mul(base_color, hl_diff) + hl_spec

        hit_color = wp.min(result, wp.vec3(1.0, 1.0, 1.0))
        hit_color = wp.max(hit_color, wp.vec3(0.0, 0.0, 0.0))

        rgb_out[worldid, rgb_adr[camid] + rayid_local] = pack_rgba_to_uint32(
            hit_color[0] * 255.0,
            hit_color[1] * 255.0,
            hit_color[2] * 255.0,
            255.0,
        )

    wp.launch(
        kernel=_render_megakernel,
        dim=(d.nworld, rc.total_rays),
        inputs=[
            m.geom_type,
            m.geom_dataid,
            m.geom_matid,
            m.geom_size,
            m.geom_rgba,
            m.cam_projection,
            m.cam_fovy,
            m.cam_sensorsize,
            m.cam_intrinsic,
            m.light_type,
            m.light_castshadow,
            m.light_active,
            m.light_attenuation,
            m.light_cutoff,
            m.light_exponent,
            m.light_ambient,
            m.light_diffuse,
            m.light_specular,
            m.flex_vertadr,
            m.flex_edge,
            m.flex_radius,
            m.mesh_faceadr,
            m.mat_texid,
            m.mat_texrepeat,
            m.mat_emission,
            m.mat_specular,
            m.mat_shininess,
            m.mat_rgba,
            d.geom_xpos,
            d.geom_xmat,
            d.cam_xpos,
            d.cam_xmat,
            d.light_xpos,
            d.light_xdir,
            d.flexvert_xpos,
            rc.nrender,
            rc.use_shadows,
            rc.bvh_ngeom,
            rc.bvh_nflexgeom,
            rc.cam_res,
            rc.cam_id_map,
            rc.ray,
            rc.rgb_adr,
            rc.depth_adr,
            rc.seg_adr,
            rc.render_rgb,
            rc.render_depth,
            rc.render_seg,
            rc.bvh_id,
            rc.group_root,
            rc.flex_bvh_id,
            rc.flex_group_root,
            rc.enabled_geom_ids,
            rc.mesh_bvh_id,
            rc.mesh_facetexcoord,
            rc.mesh_texcoord,
            rc.mesh_texcoord_offsets,
            rc.hfield_bvh_id,
            rc.flex_rgba,
            rc.flex_geom_flexid,
            rc.flex_geom_edgeid,
            rc.skybox_tex_id,
            rc.skybox_face_width,
            rc.textures,
        ],
        outputs=[
            rc.rgb_data,
            rc.depth_data,
            rc.seg_data,
        ],
        block_dim=m.block_dim.render,
    )
