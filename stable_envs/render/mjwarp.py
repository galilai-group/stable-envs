"""MJWarp's batch raytracer behind a ``render(qpos) -> (B,H,W,3) uint8`` surface, plus the model
pre-pass without which it silently draws a different scene.

``docs/research/backend3d.md`` measures MJWarp at 53k fps against MuJoCo EGL's 185, and at 98.9% of
interior pixels bit-identical on flat-material scenes. Getting there needs six things right, and
five of them are silent when wrong — a wrong answer with no error, no warning and a plausible-looking
number. They are all handled here so that no caller has to remember them:

1. **The buffer is BGRA.** ``pack_rgba_to_uint32`` writes ``(a<<24)|(r<<16)|(g<<8)|b``, so a
   little-endian byte view of ``rc.rgb_data`` is B, G, R, A. Reading it as RGB leaves geometry,
   camera, pose and shadows pixel-aligned while making every colour wrong, which reads exactly like
   "the renderer is ignoring materials". Measured cost: 4-6x in mean error, up to 40x in exact-pixel
   agreement — and it is INVISIBLE on grey scenes, where ``r == g == b`` makes the swap a no-op.
2. **MuJoCo renders shadows by default**, so ``use_shadows=False`` is a mismatch rather than a saving
   (antmaze 27.8 -> 30.4 dB with them on). The default here is on.
3. **Geom groups.** ``mujoco.Renderer`` shows groups 0-2; MJWarp shows what it is told. A model whose
   collision geoms sit in a visible group gets them painted over the visual mesh on one side only.
4. **MJWarp never reads alpha.** Its shader takes ``base_color = vec3(color[0], color[1], color[2])``
   and packs a constant 255, so a geom with ``rgba="0 1 0 0"`` — an invisible placement marker, a
   gripper axis, an inertia box — renders as opaque saturated colour. This is not a corner case: it
   is what made a RoboCasa kitchen score 7.6% exact and read as a texture-path failure, and 154 of
   that scene's 517 group-1 geoms were invisible markers covering ~40% of the frame. Moving them to a
   disabled group first took the mean error from 79.34 to 7.47 and within-8 agreement from 42.0% to
   82.8%. ``prepare_model`` does that, for geoms AND sites, from the EFFECTIVE alpha (a geom's own
   rgba, or its material's when the geom carries the compiler default).

5. **MJWarp TRUNCATES where OpenGL ROUNDS.** ``pack_rgba_to_uint32`` is
   ``int(a)<<24 | int(r)<<16 | int(g)<<8 | int(b)`` on values already scaled to [0,255], and C's
   ``int`` truncates; the GL pipeline MuJoCo renders through converts float to uint8 by rounding to
   nearest. So MJWarp is never ABOVE MuJoCo and is exactly one level below wherever the product's
   fractional part reaches 0.5. Measured on Fetch (flat materials, no shadows, no reflections, so
   nothing else is in play): the signed difference on flat interior pixels is ``{0: 86.8%, -1: 13.1%}``
   and **nothing else above 0.01%**, and correcting the rule takes interior agreement from **0.791 to
   0.9988** and whole-frame exactness from 0.718 to 0.910 — by far the largest single fidelity win
   available on this backend. ``gl_rounding()`` applies it; see that function for the caveat.

6. **``kinematics`` does not compute ``subtree_com``, so a com-tracking camera does not move.** The
   four-call frame loop (``kinematics``, ``camlight``, ``refit_bvh``, ``render``) is the one
   ``backend3d.md`` proposes and it is right for a fixed or body-mounted camera. A camera or light
   whose ``mode`` is ``TRACKCOM`` or ``TARGETBODYCOM`` reads ``d.subtree_com``, which ``kinematics``
   never writes — ``com_pos`` does. Left out, the camera stays wherever ``put_data``'s initial
   ``mj_forward`` put it, for every state, forever. Measured on Gymnasium's ``HalfCheetah-v5`` and
   ``Ant-v5``, whose ``track`` camera is ``TRACKCOM``: without ``com_pos`` the camera's spread over 8
   states is **0.0** against the reference's 0.930 and 1.420 m and it lands up to 1.75 m from where
   MuJoCo puts it; with it, ``cam_xpos`` agrees to **0.0**. ``BatchRenderer`` inserts the call when
   the model needs it and skips it otherwise, so the cost is paid only by scenes that have such a
   camera.

The one gap this module cannot close, recorded so it is not rediscovered: MJWarp's ``sample_texture``
derives UVs for ``PLANE`` and ``MESH`` only, so a textured BOX, CYLINDER, SPHERE or CAPSULE samples
texel (0,0) — MuJoCo paints 167 colours on a checkered box where MJWarp paints 2. ``texture_gaps``
returns the offending geoms so a caller can decide (re-express as a mesh with cube texcoords, which
recovers most of it: mean 50.30 -> 6.43) instead of shipping a scene that is quietly flat.

The second gap is **minification**. ``create_warp_texture`` builds a ``wp.Texture2D`` at
``num_mip_levels=1`` and the render kernel calls ``wp.texture_sample`` with no level of detail, so a
texel is chosen by bilinear filtering of the base level however many texels fall inside the pixel.
MuJoCo renders through GL, which mipmaps. On a texture seen at grazing incidence the two do not
approximate each other, they disagree completely: on Gymnasium's locomotion floor — a 100x100 checker
at ``texrepeat="60 60"`` on a 40 m plane — the far tenth of the reference has a luminance spread of
1.5-3.4 about one grey while MJWarp's has a spread of 92, and **0.0017-0.0069% of those pixels are
bit-exact** (``docs/research/mujoco_locomotion.md`` Section 6). Warp itself has the machinery
(``num_mip_levels``, ``mip_filter_mode``), but selecting a level needs ray differentials the render
kernel does not compute, so this is not a flag a caller can set. Supersampling does not substitute
for it: at 4x4 samples per pixel the far-field mean error falls from 100 to 8-14/255 and exactness
reaches only 0.6-1.4%, while the *near* field it blurs gets worse.

The third is a colour-resolution rule rather than a gap: **MJWarp reads a geom's colour from its
material whenever the geom has one, and MuJoCo prefers the geom's own ``rgba`` unless that is the
compiler default** ``(0.5, 0.5, 0.5, 1)``. ``effective_alpha`` already follows MuJoCo's rule for
alpha; the colour has no in-place fix, because one material is generally worn by geoms of several
colours. ``fetch.py`` and ``scratch/mujoco_locomotion/mjw_probe4_resolved.py`` both recompile with one
material per geom carrying the resolved colour. It is not cosmetic: uncorrected it moved the
locomotion floor's mean luminance by 13-19/255 and it was worth interior 0.016 -> 0.939 on a
flat-material control.

All of these are present in ``mujoco-warp`` 3.11.0, the latest release, and on ``main``.

This module is deliberately independent of any env: ``prepare_model`` and ``texture_gaps`` are model
surgery that any MuJoCo scene needs, and ``BatchRenderer`` is the integration surface from
``scratch/backend3d/mjwarp_surface.py``. It is imported lazily; ``mujoco_warp`` is not a dependency of
``stable_envs``.
"""
from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass

import numpy as np

os.environ.setdefault("WARP_CACHE_PATH",
                      os.path.expanduser("~/.cache/warp"))

#: Groups ``mujoco.Renderer`` shows at its default ``MjvOption``. Matching them on both sides is the
#: difference between comparing two renders of one scene and comparing two different scenes.
DEFAULT_GROUPS = (0, 1, 2)
#: Where ``prepare_model`` parks geometry MJWarp would otherwise draw opaque.
HIDDEN_GROUP = 5
#: mjtGeom codes whose UVs MJWarp can derive. Everything else samples texel (0,0) when textured.
TEXTURED_OK = (0, 7)                                    # PLANE, MESH


def effective_alpha(m, kind: str = "geom") -> np.ndarray:
    """``(n,)`` alpha actually used for each geom or site, following MuJoCo's own resolution order.

    An element's own ``rgba`` wins unless it is the compiler default ``(0.5, 0.5, 0.5, 1)``, in which
    case its material supplies the colour. Reading only ``geom_rgba`` would miss a material with
    ``rgba="... 0"``, and reading only the material would miss the far more common per-geom marker.
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
    """What ``prepare_model`` changed, so a caller can report it instead of trusting it."""

    hidden_geoms: list[int]
    hidden_sites: list[int]
    groups: tuple[int, ...]
    restore: dict

    def summary(self) -> str:
        return (f"{len(self.hidden_geoms)} invisible geoms and {len(self.hidden_sites)} invisible "
                f"sites moved out of groups {self.groups}")


def prepare_model(m, groups=DEFAULT_GROUPS, alpha_eps: float = 0.0) -> Prepared:
    """Move zero-alpha geometry out of the visible groups, IN PLACE. Returns what to restore.

    MJWarp draws every geom in an enabled group at full opacity because its shader never reads the
    alpha channel, so anything MuJoCo treats as invisible has to be removed from the model before it
    ever reaches ``create_render_context``. Doing it by group rather than by editing colours keeps the
    change reversible and keeps the two renderers describable by one sentence: "both were shown groups
    0-2, minus the geometry MuJoCo would have drawn with zero alpha".
    """
    ga = effective_alpha(m, "geom")
    sa = effective_alpha(m, "site")
    restore = {"geom_group": np.array(m.geom_group, copy=True),
               "site_group": np.array(m.site_group, copy=True)}
    gh = [i for i in range(m.ngeom) if ga[i] <= alpha_eps and int(m.geom_group[i]) in groups]
    sh = [i for i in range(m.nsite) if sa[i] <= alpha_eps and int(m.site_group[i]) in groups]
    if gh:
        m.geom_group[gh] = HIDDEN_GROUP
    if sh:
        m.site_group[sh] = HIDDEN_GROUP
    return Prepared(gh, sh, tuple(groups), restore)


def restore_model(m, p: Prepared) -> None:
    m.geom_group[:] = p.restore["geom_group"]
    m.site_group[:] = p.restore["site_group"]


def texture_gaps(m, groups=DEFAULT_GROUPS) -> list[tuple[int, str, int]]:
    """``[(geom id, name, mjtGeom), ...]`` for TEXTURED geoms whose UVs MJWarp cannot derive.

    Empty is the answer a caller wants: it means the scene's textures live on planes and meshes, which
    is the only case MJWarp's ``sample_texture`` handles. A non-empty list is not a warning to be
    logged and forgotten — those surfaces will render as one texel.
    """
    import mujoco
    out = []
    for g in range(m.ngeom):
        if int(m.geom_group[g]) not in groups or int(m.geom_type[g]) in TEXTURED_OK:
            continue
        mid = int(m.geom_matid[g])
        if mid < 0:
            continue
        t = m.mat_texid[mid]
        t = int(np.max(t)) if np.ndim(t) else int(t)
        if t >= 0:
            out.append((g, mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}",
                        int(m.geom_type[g])))
    return out


def texture_mean_rgb(m, tex: int) -> np.ndarray:
    """``(3,)`` in [0,1]: the mean colour of texture ``tex``, which is what a flat bake should use."""
    w, h = int(m.tex_width[tex]), int(m.tex_height[tex])
    nc = int(m.tex_nchannel[tex]) if hasattr(m, "tex_nchannel") else 3
    adr = int(m.tex_adr[tex])
    a = np.asarray(m.tex_data[adr:adr + w * h * nc], np.float64).reshape(h, w, nc)
    return (a[..., :3].mean((0, 1)) if nc >= 3 else np.repeat(a.mean((0, 1)), 3)) / 255.0


def bake_primitive_textures(m, groups=DEFAULT_GROUPS) -> dict:
    """Replace the texture on geoms MJWarp cannot UV with that texture's MEAN colour, IN PLACE.

    THIS IS A VISUAL-TIER TOOL AND MUST NOT BE USED ON A PARITY PATH. It changes the model so that a
    surface MuJoCo draws textured is drawn one flat colour, which is a deliberate approximation.

    The gap it closes is ``texture_gaps``: ``sample_texture`` derives UVs for ``PLANE`` and ``MESH``
    only, so a textured ``BOX``, ``CYLINDER``, ``SPHERE`` or ``CAPSULE`` samples texel **(0, 0)** and
    the whole surface comes out that one corner texel's colour. On a RoboCasa kitchen 178-401 of the
    ~350 visible geoms are in that class, and the corner texel is an arbitrary pixel of a photograph:
    a dark granite sink whose texture starts on a light grout seam renders as a light grey box, which
    is the single most visible artefact on this scene.

    The mean ought to be a much better flat colour than an arbitrary corner texel --
    ``docs/research/robocasa.md`` Section 7 bakes every textured material to its mean and finds
    MuJoCo's own frame moves by mean 7.99/255 with 94.0 % of pixels within 32, so the mean colour
    carries a surface's identity and the texture carries its detail.

    **ON ROBOCASA IT DOES NOT PAY, and that is measured rather than expected.** Mean |diff| against the
    official frame moves 6.60 -> 6.68 on kitchen L52 S03 and 32.50 -> 31.83 on L45 S33, with SSIM down
    on both (``docs/research/robocasa.md`` Section 17.6). Two reasons, both checkable from the return
    value: RoboCasa's box materials are often near-uniform (``textures/flat/light_gray.png`` and
    friends), so the corner texel is already about right; and only 38 of ~137 textured materials are
    worn EXCLUSIVELY by primitives, the other 99 being shared with meshes.

    A material worn by BOTH a primitive and a mesh is left alone and counted in ``mixed``. Flattening
    it would fix the box and break the mesh, which MJWarp textures correctly; fixing those needs a
    per-geom material clone, which needs a recompile, and is deliberately not done here. Returns the
    counts and the arrays to restore, so a caller can see how much of its scene the bake could reach
    before believing anything about the result.
    """
    import mujoco
    rgb_role = int(getattr(mujoco.mjtTextureRole, "mjTEXROLE_RGB", 1))
    restore = {"mat_texid": np.array(m.mat_texid, copy=True),
               "mat_rgba": np.array(m.mat_rgba, copy=True)}
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
            skipped.append(mid)                       # cube/skybox: not what a primitive samples
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
    """Does this model have a camera or light whose pose is derived from ``subtree_com``?

    ``mjCAMLIGHT_TRACKCOM`` positions the element at ``subtree_com[body] + poscom0`` and
    ``mjCAMLIGHT_TARGETBODYCOM`` aims it at one, and ``mujoco_warp.kinematics`` writes neither. A
    caller that answers yes must run ``com_pos`` between ``kinematics`` and ``camlight``; a caller
    that answers no must not, because the launch buys nothing. Gymnasium's six locomotion domains all
    answer yes -- their ``track`` camera is ``TRACKCOM`` -- and every MuJoCo scene shipped in this
    repo answers no.
    """
    import mujoco
    com = (int(mujoco.mjtCamLight.mjCAMLIGHT_TRACKCOM),
           int(mujoco.mjtCamLight.mjCAMLIGHT_TARGETBODYCOM))
    return bool(np.isin(np.asarray(m.cam_mode, np.int64), com).any()
                or np.isin(np.asarray(m.light_mode, np.int64), com).any())


#: Downsampled texture arrays whose owning render context would not hold them.
#: ``wp.from_torch`` does NOT take ownership, so a torch tensor that goes out of scope after being
#: handed to a live ``wp.Texture2D`` is freed under it and sampled as whatever the allocator put there
#: next. The tensors are normally attached to the context they belong to, so they are freed with it;
#: this is the fallback for a ``mujoco_warp`` build whose context object refuses an attribute, and it
#: leaks by design rather than by accident — a RoboCasa kitchen at LOD 2 is ~119 MB per context.
_TEX_KEEP: list = []


def create_render_context(m, tex_lod: int = 1, **kw):
    """``mjw.create_render_context`` without its per-texture re-upload of the whole texture blob.

    ``tex_lod`` box-filters every texture by that integer factor before it reaches the GPU, which is
    the mip level a GL pipeline would have picked at that minification anyway. It exists for the
    LEARNED-CORRECTOR path (``render/rendernet.py``) and is 1 — untouched — everywhere else. A
    RoboCasa kitchen ships 356 MB of texture data that ``mujoco_warp`` uploads at full resolution into
    float32 ``(h, w, 4)`` arrays, i.e. ~475 MB resident, sampled by a megakernel whose rays reach it
    in BVH order; the texture ABLATION says sampling is only 5.4 % of the frame, but that removes the
    arithmetic and leaves the working set, and this removes the working set. Measured on
    ``RoboCasaCounterToSink`` at 112 px: 2 is worth ~7 % of the render rate, and the pixels it costs
    are pixels the corrector is fitting anyway. IT IS NOT FREE AND MUST NOT BE TURNED ON FOR A PLAIN
    VISUAL-TIER RENDER — every declared number in ``envs/robocasa.py`` was measured at ``tex_lod=1``.

    ``mujoco_warp._src.render_util.create_warp_texture`` builds ONE texture and does
    ``wp.array(mjm.tex_data, dtype=wp.uint8)`` inside itself, but ``mjm.tex_data`` is MuJoCo's single
    concatenated blob for every texture in the model — so a context copies ``ntex`` times the whole
    blob to use one slice of it each time. It is quadratic in the wrong variable and invisible on
    small scenes.

    On a RoboCasa kitchen (109 textures, 356 MB of texture data) that is 109 x 356 MB = 38.8 GB of
    host-to-device traffic, and it is where the per-scene setup cost lives:

    ==============================  ===========  ==========
    stage                            upstream     hoisted
    ==============================  ===========  ==========
    texture upload                     7.600 s     0.106 s
    mesh BVH build                     1.231 s     1.138 s
    everything else                    0.149 s     0.092 s
    **create_render_context**        **8.980 s**  **1.336 s**
    transient device memory            2,338 MB      168 MB
    ==============================  ===========  ==========

    The frames are bit-identical (max abs diff 0 over the whole batch), which is the only reason this
    is a wrapper rather than a fork: nothing about the render changes, one array stops being rebuilt.
    Measured in ``scratch/robocasa/probe_fastsetup.py``; the same win applies to every textured
    MuJoCo scene, DM Control's plate bake included.

    Reported upstream-worthy as a one-line change; until it lands, this monkeypatches for the
    duration of the call and restores afterwards, so no other caller of ``mujoco_warp`` is affected.
    """
    import warp as wp

    import mujoco_warp as mjw
    from mujoco_warp._src import render_util as ru

    original = ru.create_warp_texture
    blob: dict = {}
    lod = int(tex_lod)
    keep: list = []

    def hoisted(model, tex_id: int):
        if "data" not in blob:
            blob["data"] = wp.array(model.tex_data, dtype=wp.uint8)
        w, h = int(model.tex_width[tex_id]), int(model.tex_height[tex_id])
        out = wp.zeros((h, w, 4), dtype=float)
        wp.launch(ru._convert_texture_data, dim=(w, h),
                  inputs=[w, int(model.tex_adr[tex_id]), int(model.tex_nchannel[tex_id]),
                          blob["data"]],
                  outputs=[out])
        if lod > 1:
            import torch.nn.functional as F
            t = wp.to_torch(out)
            t = F.interpolate(t.permute(2, 0, 1)[None], size=(max(1, h // lod), max(1, w // lod)),
                              mode="area")[0].permute(1, 2, 0).contiguous()
            keep.append(t)
            out = wp.from_torch(t)
        return wp.Texture2D(out, filter_mode=wp.TextureFilterMode.LINEAR)

    ru.create_warp_texture = hoisted
    try:
        rc = mjw.create_render_context(m, **kw)
    finally:
        ru.create_warp_texture = original
    if keep:
        # Tie the downsampled textures to the CONTEXT rather than to the module, so dropping a context
        # drops its 119 MB with it. A context cache that evicts (``envs/robocasa._CTX_CACHE``) would
        # otherwise free the context and keep the textures forever.
        try:
            rc._se_tex_lod_keep = keep
        except AttributeError:                                        # pragma: no cover
            _TEX_KEEP.extend(keep)
    return rc


@contextlib.contextmanager
def gl_rounding():
    """Make MJWarp quantise float->uint8 the way OpenGL does, for the duration of the block.

    ``pack_rgba_to_uint32`` receives colours already scaled to [0,255] and packs them with C's
    ``int()``, which truncates. MuJoCo's frames come out of a GL pipeline, which rounds to nearest.
    The two rules differ by one level on every pixel whose value has a fractional part >= 0.5, and
    that is not a rounding-error footnote — it is the dominant term:

    ======================================  ===========  ==========
    Fetch (480 px, 16 states, 4 scenes)      truncating    rounding
    ======================================  ===========  ==========
    interior pixels bit-exact                    0.791       0.9988
    all pixels bit-exact                         0.718       0.910
    mean |diff|                                   1.24         1.05
    ======================================  ===========  ==========

    and the signed difference on flat interior pixels is ``{0: 86.8%, -1: 13.1%}`` with every other
    value below 0.01%, which is what "the same float, quantised two ways" looks like and what no
    shading hypothesis predicts.

    Two things a caller must know:

    * **WRAP THE GRAPH CAPTURE, NOT ONLY THE CONTEXT AND THE WARM-UP FRAME.** ``mujoco_warp.render``
      does not compile a kernel that ``create_render_context`` generated: it DEFINES the megakernel
      inside itself, as ``@wp.kernel(module="unique")`` over a closure, so the source is generated
      afresh on every call from whatever ``pack_rgba_to_uint32`` is bound at that moment. A context
      built as ``with gl_rounding(): create_render_context(); frame()`` and then captured OUTSIDE the
      block generates a second, truncating kernel, and the graph every later render replays is that
      one — the patch reaches the warm-up frame and nothing else. Measured with
      ``scratch/perf/rounding.py``: the captured graph agreed with a TRUNCATING direct render on
      100.0% of channels and differed from a rounded one by exactly ``{0, -1}`` on 15.8% (Fetch),
      21.2% (Meta-World) and 63.8% (HumanoidBench) of them. Capturing inside the block costs nothing —
      the warm-up frame has already compiled and loaded that module, so the capture reuses it.
    * It patches ``mujoco_warp._src.render``, which is private. It is written as a context manager
      precisely so the window is short and so nothing else in the process ever sees it. Present in
      ``mujoco-warp`` 3.11.0, the latest release, and on ``main`` as of 2026-08-10.
    """
    import warp as wp

    from mujoco_warp._src import render as R

    @wp.func
    def pack_rgba_round(r: float, g: float, b: float, a: float) -> wp.uint32:
        return wp.uint32((int(a) << int(24)) | (int(r + 0.5) << int(16))
                         | (int(g + 0.5) << int(8)) | int(b + 0.5))

    original = R.pack_rgba_to_uint32
    R.pack_rgba_to_uint32 = pack_rgba_round
    try:
        yield
    finally:
        R.pack_rgba_to_uint32 = original


class BatchRenderer:
    """``qpos (B,nq)`` -> ``(B,H,W,3)`` uint8 on device, with the pre-pass applied and channels fixed.

    The model is prepared and left prepared for the life of the renderer: MJWarp reads the geometry
    once, at ``create_render_context``, so restoring the groups afterwards would not un-draw anything
    and would only make the object's behaviour depend on when the caller looked at the model. Call
    ``restore_model`` explicitly if the same ``MjModel`` is shared with a MuJoCo renderer.
    """

    def __init__(self, m, batch: int, res: int = 224, camera: int = 0, groups=DEFAULT_GROUPS,
                 textures: bool = True, shadows: bool = True, skybox: bool = True,
                 prepare: bool = True, capture: bool = True):
        import mujoco
        import warp as wp

        import mujoco_warp as mjw

        self.prepared = prepare_model(m, groups) if prepare else None
        self.texture_gaps = texture_gaps(m, groups)
        self.needs_com = needs_com_pos(m)
        self.m_mj, self.batch, self.res = m, int(batch), int(res)

        d0 = mujoco.MjData(m)
        mujoco.mj_forward(m, d0)
        self.m = mjw.put_model(m)
        self.d = mjw.put_data(m, d0, nworld=int(batch))
        self.rc = create_render_context(
            m, nworld=int(batch), cam_res=(int(res), int(res)), render_rgb=True,
            cam_active=[i == int(camera) for i in range(m.ncam)],
            enabled_geom_groups=list(groups), use_textures=textures, use_shadows=shadows,
            render_skybox=skybox)
        import torch
        self._bgra = wp.to_torch(self.rc.rgb_data).view(torch.uint8).reshape(
            int(batch), int(res), int(res), 4)
        self._wp, self._mjw = wp, mjw
        self.graph = None
        if capture:
            self._capture()

    def _frame(self):
        self._mjw.kinematics(self.m, self.d)
        if self.needs_com:
            # A TRACKCOM / TARGETBODYCOM camera or light reads ``d.subtree_com``, which
            # ``kinematics`` does not write. Omitted, the camera never leaves its ``put_data`` pose
            # and every frame is rendered from the wrong place with no error. Conditional because the
            # launch is pure cost on the scenes that do not need it.
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
        """``(B,nq)`` float32 on device -> ``(B,H,W,3)`` uint8 on device, RGB (channels swapped)."""
        self._wp.to_torch(self.d.qpos).copy_(qpos)
        if self.graph is not None:
            self._wp.capture_launch(self.graph)
        else:
            self._frame()
        return self._bgra[..., [2, 1, 0]]                # the swap; ~3% of frame time, never optional
