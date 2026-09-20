"""Reading a DM Control model: the constants, the caches, and the census of what an episode draws.

Everything here answers a question about a COMPILED ``MjModel`` and nothing here draws a pixel, which
is why it is one module and not part of either renderer: ``suite`` (the parity tier) and ``trackcom``
(the visual tier) disagree about how a frame is made and agree completely about what the model says.

The caches are module-level and deliberately so. One ``mujoco.Renderer`` per (domain, size) is a
POLICY, not an optimisation — releasing an EGL context corrupts every other live one — and a cache
that lived on a class or an instance would give a second env a second context. See ``_ENV_CACHE``.
"""

from __future__ import annotations

import numpy as np

from ..._mujoco_gl import configure_mujoco_gl

configure_mujoco_gl()

from ...render import _resources as _RES
from ...render import mjwarp as MW
from .._plate_texels import quantise as _quantise

#: ``mjvLight.type``: 1 is directional, 0 is a positional spot/point light. Every domain here has both
#: — the headlight, which is directional, sits at the camera and never casts a shadow, and the model
#: light, which is a 45-degree spotlight with exponent 10 that does.
LIGHT_DIRECTIONAL = 1

#: Effective specular weight, MEASURED by inverting the reference's own quantised colours rather than
#: taken from the material. It is the same kind of constant the OGBench study reports (there the
#: specular weight measured 0.25-0.26 against a nominal 0.30) and it is found the same way.
#:
#: Cartpole's cart is a BOX, so its front face has one normal and per-vertex and per-pixel shading
#: coincide on it by construction — which makes it the one surface in this family where a shading
#: constant can be read off directly. MuJoCo paints that face (227, 166, 104) in every one of 64
#: states. With the material colour (0.7, 0.5, 0.3) and ambient+diffuse 1.2 (headlight 0.4 + 0.8, the
#: spotlight contributing nothing at ``n.l = 0``), each channel constrains the additive white specular
#: term ``s`` to the interval that rounds to the observed level:
#:
#:     R  214.2 + 255 s -> 227   =>  s in [0.04824, 0.05216)
#:     G  153.0 + 255 s -> 166   =>  s in [0.04902, 0.05294)
#:     B   91.8 + 255 s -> 104   =>  s in [0.04588, 0.04980)
#:
#: whose intersection is ``s in [0.04902, 0.04980)`` — and the nominal value, material specular 0.5
#: times the headlight's 0.1 at ``n.h = 1``, is 0.05000, just OUTSIDE it. The midpoint gives the scale
#: below. Whether the missing 1.2% is the weight or a half-vector a hair under 1 is not separable from
#: one face at normal incidence, and it is applied as a weight.
#:
#: Worth 0.42-0.60 -> 1.00 of the interior-pixel metric on the two cartpole cells, and nothing
#: anywhere else: the term is only ever evaluated on mover pixels, and no other domain here has a
#: mover pixel that survives the interior mask.
SPECULAR_SCALE = 0.988

#: Geom/site groups ``mujoco.Renderer`` shows at its default ``MjvOption`` — measured [1,1,1,0,0,0]
#: for BOTH. A site in group 3 (acrobot's ``target``, finger's ellipsoids) is not drawn, and a
#: renderer that drew it would be adding geometry the reference does not have.
VISIBLE_GROUPS = (0, 1, 2)
#: A group nothing is in, used to hide the movers while the plate is baked.
HIDDEN_GROUP = 5

#: Model arrays a DM Control ``initialize_episode`` may DRAW per episode, and that the renderer reads.
#: The list is deliberately wider than what any task's source shows, because the point of the census in
#: ``_episode_slots`` is to catch a draw nobody transcribed rather than to confirm one that was.
EPISODE_ARRAYS = (
    "geom_pos",
    "geom_quat",
    "geom_size",
    "site_pos",
    "site_quat",
    "site_size",
    "body_pos",
    "body_quat",
    "light_pos",
    "light_dir",
)
#: Resets used to discover the draw. Six is what separates a per-episode draw from a task CONSTANT on
#: every domain here: ``finger-spin`` writes ``site_rgba`` and ``dof_damping`` at every
#: ``initialize_episode`` and writes the same values every time, so one reset cannot tell the two
#: apart and six can. More only costs reset time, and the count is fixed so the discovery is
#: reproducible across processes.
N_DRAWS = 6
#: Component labels, so a widened state dimension is named ``target_pos_x`` and not ``slot 3``.
_COMPONENT = {3: ("x", "y", "z"), 4: ("qw", "qx", "qy", "qz")}
_SCENE_CACHE_MAX = 96
_SCENE_CACHE = _RES.BoundedCache(_SCENE_CACHE_MAX, name="dmcontrol.scene")
#: ``(width, height, nchannel) -> (rgb1 mask, rgb2 mask)`` over the suite's builtin ``grid`` checker,
#: and ``{}`` until the first parity cell asks. See ``_grid_masks``.
_GRID_MASKS: dict = {}
#: ``(domain, task, size) -> texel slots for that cell's grid texture``, or ``None`` for a cell that
#: has no grid texture. Keyed per cell because the slots carry that model's ``tex_adr``.
_GRID_SLOTS: dict = {}
#: (domain, task, size) -> (dm_control env, MjModel, MjData, pristine snapshot, mujoco.Renderer,
#: episode slots). One Renderer per (domain, size), built once and NEVER closed.
_ENV_CACHE: dict = {}
#: (domain, task, res, worlds) -> a captured MJWarp context. Small, because a context owns
#: ``nworld * res^2 * 12`` bytes of device memory and holding several domains' at once buys nothing.
_TC_CACHE = _RES.BoundedCache(3, name="dmcontrol.tc", on_evict=MW.release_cached_ctx)
_TC_CACHE_MAX = 3
#: Worlds per MJWarp context, as in Meta-World and Fetch. Throughput saturates well below this (S7 of
#: the research note), so a bigger context only costs memory.
MJW_CHUNK = 64


def _check_dm_control_works() -> None:
    """Fail the MODULE import if dm_control cannot compile a model, not the first env construction.

    ``import dm_control`` succeeds under a version skew that makes every domain in the suite raise:
    ``dm_control`` builds its named-index layer from a generated table of every ``MjModel`` array
    field, ``mujoco`` 3.9.0 removed ``flex_bandwidth``, and 1.0.41's table still lists it — so
    ``Physics.__init__`` raises ``AttributeError`` for every model, including an empty one. That is a
    machine-level fact, so it belongs where a missing optional dependency belongs: at import, where
    ``stable_envs.envs.import_errors()`` reports it by name and the package loses these five envs
    rather than raising inside a test fixture thirty times.

    The probe is the smallest thing that exercises the failing path: compile a model with no bodies
    and build its index. Milliseconds, and no GL context.
    """
    from dm_control import mujoco as dm_mujoco

    try:
        dm_mujoco.Physics.from_xml_string("<mujoco/>")
    except AttributeError as e:
        import mujoco

        raise ImportError(
            f"dm_control is installed but cannot compile a model with mujoco "
            f"{mujoco.__version__}: {e}. This is a version skew, not a missing package — no "
            f"dm_control release works with mujoco 3.9. Consistent pairs: mujoco==3.11.0 with "
            f"dm_control==1.0.44, mujoco==3.10.0 with dm_control==1.0.43, mujoco==3.8.1 with "
            f"dm_control==1.0.41. See docs/research/dmcontrol.md."
        ) from e


_check_dm_control_works()


def _suite():
    from dm_control import suite

    return suite


# --------------------------------------------------------------------------- model reading


def _mat_rgb(m, matid: int, rgba) -> np.ndarray:
    """Base colour of a geom or site: its own rgba unless that is the compiler default and a material
    supplies one, which is the rule MuJoCo itself follows."""
    rgba = np.asarray(rgba, np.float64)
    if matid >= 0 and np.allclose(rgba, [0.5, 0.5, 0.5, 1.0]):
        return np.asarray(m.mat_rgba[matid], np.float64)[:3]
    return rgba[:3]


def _mat_spec_shin(m, matid: int) -> tuple[float, float]:
    if matid < 0:
        return 0.5, 0.5  # MuJoCo's material defaults
    return float(m.mat_specular[matid]), float(m.mat_shininess[matid])


def _grid_defaults() -> np.ndarray:
    """``[rgb1, rgb2, markrgb]`` -- the suite's own ``grid`` texture, READ from ``dm_control``'s
    ``common/materials.xml`` rather than transcribed here.

    ``markrgb`` is carried alongside the leaf's two colours because it is what makes the edge marks
    separable at all: the suite ships ``markrgb=".2 .3 .4"``, EQUAL to ``rgb2``, so in the shipped
    texture the marks are invisible to an equality mask -- but a recompile at a different ``rgb2``
    separates them, and ``_grid_masks`` uses exactly that to check its masks against whatever
    texture it is handed.

    It is read because it is simultaneously three things that must not be allowed to drift apart: the
    compiled texture's colours, this family's declared ``floor.color`` default, and
    ``stable_worldmodel``'s own ``init_value`` for the leaf (``[[0.1,0.2,0.3],[0.2,0.3,0.4]]``, in
    every one of its twelve DM Control wrappers). A constant here would make "our default IS swm's
    default" a coincidence that a ``dm_control`` release could quietly end.
    """
    import re

    from dm_control.suite import common

    xml = common.ASSETS["./common/materials.xml"].decode()
    mat = re.search(r'<texture[^>]*name="grid"[^>]*/>', xml)
    assert mat, "dm_control's common/materials.xml no longer declares a texture named 'grid'"
    out = []
    for attr in ("rgb1", "rgb2", "markrgb"):
        got = re.search(attr + r'="([^"]+)"', mat.group(0))
        assert got, f"the suite's grid texture no longer declares {attr}"
        out.append([float(x) for x in got.group(1).split()])
    return np.asarray(out, np.float64)


def _grid_masks(m, texid: int) -> tuple[np.ndarray, np.ndarray]:
    """Which texels of the compiled ``grid`` texture are rgb1 and which are rgb2.

    NOT DERIVED BY EQUALITY, and that is the whole point of this function. The suite declares
    ``markrgb=".2 .3 .4"``, which is EXACTLY its ``rgb2``, so in the compiled texture the 1,196
    edge-mark texels and the 44,402 rgb2 texels are byte-identical. ``stable_worldmodel`` honours
    ``floor.color`` by writing ``rgb1``/``rgb2`` and RECOMPILING: the recompile moves the rgb2 texels
    and leaves ``markrgb`` at its literal ``.2 .3 .4``. A mask taken by equality would move both, and
    the 1,196 marks are the frame's grid lines -- the most visible feature on the floor.

    So the masks are taken from a PROBE TEXTURE: the same builtin, the same size, the same ``mark``,
    compiled once with three mutually distinct colours, which separates the three sets positionally.
    The probe is a four-line MJCF and no part of any cell; the assertion below is what ties it to the
    real texture, by reconstructing that texture from the masks and the declared defaults and
    requiring the result to be byte-identical to what MuJoCo compiled.

    Cached on the texture's shape, because all twelve parity cells share one 300x300 checker.
    """
    import mujoco

    key = (
        int(m.tex_width[texid]),
        int(m.tex_height[texid]),
        int(m.tex_nchannel[texid]) if hasattr(m, "tex_nchannel") else 3,
    )
    hit = _GRID_MASKS.get(key)
    if hit is not None:
        return hit
    w, h, nch = key
    probe = mujoco.MjModel.from_xml_string(
        f'<mujoco><asset><texture name="grid" type="2d" builtin="checker" rgb1="1 0 0" '
        f'rgb2="0 1 0" width="{w}" height="{h}" mark="edge" markrgb="0 0 1"/>'
        f'<material name="grid" texture="grid"/></asset><worldbody>'
        f'<geom type="plane" size="1 1 1" material="grid"/></worldbody></mujoco>'
    )
    pt = mujoco.mj_name2id(probe, mujoco.mjtObj.mjOBJ_TEXTURE, "grid")
    pa = int(probe.tex_adr[pt])
    pn = w * h * (int(probe.tex_nchannel[pt]) if hasattr(probe, "tex_nchannel") else 3)
    pf = np.asarray(probe.tex_data[pa : pa + pn]).reshape(w * h, -1)[:, :3]
    m1 = np.all(pf == np.array([255, 0, 0], np.uint8)[None, :], axis=1)
    m2 = np.all(pf == np.array([0, 255, 0], np.uint8)[None, :], axis=1)
    mk = np.all(pf == np.array([0, 0, 255], np.uint8)[None, :], axis=1)
    assert (m1 | m2 | mk).all(), (
        "the probe checker compiled to a colour that is none of its three declared ones, so the "
        "masks do not describe MuJoCo's builtin and must not be used"
    )
    # The tie to the REAL texture: rebuild it from the masks plus the suite's declared colours
    # (markrgb == rgb2, so the marks take rgb2's default) and require MuJoCo's own bytes.
    d = _grid_defaults()
    adr = int(m.tex_adr[texid])
    n = w * h * nch
    real = np.asarray(m.tex_data[adr : adr + n]).reshape(-1, nch)[:, :3]
    rebuilt = np.empty_like(real)
    rebuilt[m1] = _quantise(d[0])
    rebuilt[m2] = _quantise(d[1])
    rebuilt[mk] = _quantise(d[2])
    assert np.array_equal(rebuilt, real), (
        f"the probe masks plus the suite's declared grid colours rebuild {int((rebuilt != real).any(1).sum())} "
        f"of {w * h} texels differently from the compiled texture; the probe no longer describes "
        f"this texture and a substitution through it would paint the wrong texels"
    )
    _GRID_MASKS[key] = (m1, m2)
    return _GRID_MASKS[key]


def _tex_rgb(m, matid: int) -> np.ndarray:
    """Flat texture colour, or ones. The movers in these domains are untextured; this is a guard
    rather than a texture sampler, and a textured mover would need a real one."""
    if matid < 0:
        return np.ones(3)
    tid = m.mat_texid[matid]
    tid = int(np.max(tid)) if np.ndim(tid) else int(tid)
    if tid < 0:
        return np.ones(3)
    attr = "tex_rgb" if hasattr(m, "tex_rgb") else "tex_data"
    a0 = int(m.tex_adr[tid])
    h, w = int(m.tex_height[tid]), int(m.tex_width[tid])
    reg = np.asarray(getattr(m, attr)[a0 : a0 + 3 * h * w], np.float64).reshape(-1, 3) / 255.0
    u = np.unique(reg, axis=0)
    return u[0] if len(u) == 1 else reg.mean(0)


def _effective_alpha(m, kind: str, rgba=None, mat_rgba=None) -> np.ndarray:
    """The alpha MuJoCo actually draws a geom or site with, resolved through its material.

    ``mujoco.Renderer`` selects geometry by GROUP, which is what ``VISIBLE_GROUPS`` covers, and then
    MuJoCo skips anything whose resolved alpha is zero. Both filters have to be applied or the
    difference is silent: ``finger``'s ``target`` and ``tip`` sites and ``fish``'s ``target`` geom are
    hidden that way rather than by group, so a renderer that reads groups alone paints markers the
    reference does not have — and paints them in the ray-cast pass, where they land on the metric.

    Same resolution ``stable_envs.render.mjwarp.effective_alpha`` performs, and the same rule
    ``_mat_rgb`` uses for colour: the element's own rgba unless that is the compiler default and a
    material supplies one.

    ``rgba``/``mat_rgba`` ask the question against a SNAPSHOT of those arrays instead of against the
    live model, and that is a correctness argument rather than a convenience. The ``MjModel`` in
    ``_bundle`` is shared by every instance of a class and is mutated by every bake; since
    ``visibility.*`` exists a bake can legitimately leave an element at alpha 0, so a caller deciding
    what this class DECLARES (rather than what this bake happens to draw) must ask the pristine
    snapshot or the declaration depends on what the previous env most recently rendered. ``_pristine``
    records what that mistake cost the last time it was made.
    """
    rgba = np.asarray(getattr(m, f"{kind}_rgba") if rgba is None else rgba, np.float64)
    mat = np.asarray(m.mat_rgba if mat_rgba is None else mat_rgba, np.float64)
    matid = np.asarray(getattr(m, f"{kind}_matid"), np.int64)
    a = rgba[:, 3].copy()
    default = np.all(np.isclose(rgba, [0.5, 0.5, 0.5, 1.0]), axis=1)
    for i in np.flatnonzero(default & (matid >= 0)):
        a[i] = float(mat[matid[i], 3])
    return a


def _row_width(m, arr: str) -> int:
    a = np.asarray(getattr(m, arr))
    return int(a.reshape(a.shape[0], -1).shape[1]) if a.ndim > 1 else 1


def _slot_element(m, arr: str, i: int) -> tuple[str, int]:
    """``(element kind, id)`` the flat index ``i`` of model array ``arr`` belongs to."""
    return arr.split("_")[0], int(i // _row_width(m, arr))


def _slot_name(m, arr: str, i: int) -> str:
    import mujoco

    kind, eid = _slot_element(m, arr, i)
    obj = {
        "geom": mujoco.mjtObj.mjOBJ_GEOM,
        "site": mujoco.mjtObj.mjOBJ_SITE,
        "body": mujoco.mjtObj.mjOBJ_BODY,
        "light": mujoco.mjtObj.mjOBJ_LIGHT,
    }[kind]
    name = mujoco.mj_id2name(m, obj, eid) or f"{kind}{eid}"
    w = _row_width(m, arr)
    comp = _COMPONENT.get(w, tuple(str(k) for k in range(w)))[int(i % w)]
    return f"{name}_{arr.split('_', 1)[1]}_{comp}"


def _episode_slots(env, m) -> tuple[tuple[str, int], ...]:
    """Which model entries the task DRAWS per episode, measured over ``N_DRAWS`` real episodes.

    ``docs/research/dmcontrol.md`` S8 recorded "reacher / finger / fish / swimmer move a target geom
    or site per EPISODE" as one fact, and read off the source it is four different ones. ``reacher``
    draws two floats (``geom_pos['target']`` x and y); ``finger-turn`` draws two (``site_pos``
    x and z) while ``finger-SPIN`` draws NONE — its ``initialize_episode`` writes ``site_rgba`` and
    ``dof_damping`` to the same constants every time; ``fish-upright`` likewise only hides its target.
    So the census is taken rather than transcribed, and it is taken by DIFFING episodes, because an
    edit that is applied at every reset and is the same every time is a property of the model, not of
    the episode, and putting it in the state would add a dimension nothing moves.

    The model is left on the LAST draw, which is deterministic (the task's ``RandomState`` is seeded by
    ``task_kwargs={"random": 0}`` and the count is fixed), so this doubles as "apply the task's
    constant model edits" — ``finger-spin``'s two zero-alpha sites reach the model only this way.

    WHAT THE GUARD BELOW REFUSES, AND WHY IT USED TO REFUSE MORE. A per-episode SIZE really is a
    different renderer: ``_prims`` reads each mover's ``geom_size`` ONCE at bake time, the whole batch
    shares it, and ``screen_span`` derives its conservative bound from ``sc.prims.size`` — so a size in
    the state would have to become a per-row tensor through every intersection routine and through the
    window projection. A per-episode light ``_dir`` is refused for the matching reason: ``_Lights`` is
    captured once per bake and ``_shade`` reads one direction for the whole batch.

    ORIENTATION IS NOT IN THAT CLASS, and the guard used to say it was. ``_fk`` writes the drawn slot
    into the model, runs ``mj_kinematics``, and reads ``d.geom_xmat``/``d.site_xmat`` into ``xm``, which
    is ``(B, P, 3, 3)`` — a PER-ROW rotation. ``_cast`` -> ``R3.cast_nearest(kind, sizes, xpos, xmat,
    ...)``, ``_occluded`` -> ``R3.any_blocking(..., xmat, ...)`` and ``_reflect`` all consume it per row,
    and ``screen_span`` never reads it at all (its bound is the bounding SPHERE, which no rotation
    moves). So a per-episode ``*_quat`` needs nothing from the renderer that a per-episode ``*_pos``
    does not already need.

    Measured rather than reasoned: ``manipulator-bring_ball`` draws ``body_quat['target_ball']`` qw and
    qy every episode, and with only the ``_quat`` term removed from the guard the shipped bake and
    mover pass render it at interior-pixel agreement 1.000000 — 0 bad of 4,154,763 over 3 draws of 64
    states at 224 px (``examples/results/step7/dmc-manipulator/``). The per-row claim is asserted directly
    in ``tests/unit/test_dmcontrol_episode_quat.py``, so a future change that makes ``xmat`` batch-shared
    fails there rather than silently drawing every environment at one draw's orientation.
    """
    have = [k for k in EPISODE_ARRAYS if hasattr(m, k)]
    draws = []
    for _ in range(N_DRAWS):
        env.reset()
        draws.append({k: np.array(getattr(m, k), copy=True).reshape(-1) for k in have})
    slots: list[tuple[str, int]] = []
    for k in have:
        stack = np.stack([dr[k] for dr in draws]).astype(np.float64)
        for i in np.flatnonzero(stack.max(0) - stack.min(0) > 0):
            slots.append((k, int(i)))
    bad = [s for s in slots if s[0].endswith(("_size", "_dir"))]
    if bad:
        # Not "unsupported for now": see the docstring. A size is shared by the batch at bake time and
        # a light direction is captured once per bake, so either in the state would render every
        # environment at one draw's value. A domain that starts to draw one must fail loudly.
        raise RuntimeError(
            f"{env.__class__.__name__}: the episode draws {[_slot_name(m, a, i) for a, i in bad]}, "
            f"which change a primitive's SIZE or a light's DIRECTION rather than a pose. The mover "
            f"primitives carry one size for the whole batch and the lights are captured once per "
            f"bake, so this domain needs a per-row size/light path before it can be honest. "
            f"(Orientation is NOT in this class: a per-episode *_quat goes through mj_kinematics "
            f"into the per-row xmat and is supported.)"
        )
    return tuple(slots)
