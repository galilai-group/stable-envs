"""DM Control suite — sixteen cells in two tiers, one state vector each.

Reference: ``dm_control.suite.load(domain, task).physics.render(h, w, camera_id=0)``, which is
``mujoco.Renderer`` at the default ``MjvOption`` (verified: 0/255 against a hand-built Renderer on
every domain probed). So the target is MuJoCo's own EGL raster of a lit 3-D scene, not a vector
drawing — ``docs/research/classic_control.md`` was right that this family needs a 3-D path, and this
is that path.

TWO TIERS, AND WHAT SPLITS THEM
-------------------------------

Camera 0 is a ``fixed`` model camera on eleven cells and a ``mode="trackcom"`` camera that rides the
body on five. That single fact decides the renderer and therefore the tier:

  * **PARITY** (``DMControlSuite``, eleven cells): the static half of the scene is a constant image,
    so it is baked through MuJoCo itself and only the movers are ray-cast. Interior-pixel agreement is
    exactly 1.000000 wherever the metric is defined.
  * **VISUAL** (``DMControlTrackCom``, five cells): with the camera on the body there is no static
    half to bake — hide every mover and the remaining frame still moves by 51-220/255 — so the whole
    scene goes to MJWarp every frame. Mean 4.4-8.1/255 against ``physics.render``, and the residual is
    two named texture mechanisms. These are NOT the official pixels; see ``DMControlTrackCom``.

WHAT THE STATE IS, AND WHY IT IS NOT ALWAYS ``qpos``
----------------------------------------------------

Measured, not asserted. Over 17 domains at 84 px (``scratch/dmcontrol/probe_load.py``): re-rendering
the same ``qpos`` changes 0 pixels, scaling ``qvel`` by 100 changes 0 pixels, reaching the same
``qpos`` after a different history changes 0 pixels, and EVERY ``qpos`` entry moves pixels in every
domain.

Within an episode that makes the state ``qpos``. Across episodes it does not, on four domains, because
``initialize_episode`` writes the episode's draw into the MODEL rather than into ``qpos``: ``reacher``
moves a target geom, ``finger-turn`` a target site, ``swimmer`` a target geom AND a target light.
Restoring ``qpos`` alone into a different episode leaves 171-219/255 of disagreement over 0.6-90% of
the frame; restoring ``qpos`` plus those 2-4 floats is **0/255** on every one
(``scratch/dmcontrol/probe_widen.py``). So the state is ``qpos`` plus the episode's draw, the draw is
DISCOVERED by diffing episodes rather than transcribed (``_episode_slots``), and ``state_names`` names
each entry. This is Meta-World's widening — ``docs/research/metaworld.md`` S1, where ``qpos`` plus 14
model arrays is bit-exact on 50/50 tasks — applied to a family that needs 2 to 4 floats of it.

One protocol rule from that same study was checked here and does NOT apply: several Meta-World tasks
write the model AFTER their last forward, so ``data.site_xpos`` is stale straight out of ``reset``. On
all thirteen DM Control domains probed the frame immediately after ``reset()`` is bit-identical to the
frame one ``mj_forward`` later (0/255, four resets each), because ``dm_control``'s ``Task.after_reset``
forwards after ``initialize_episode`` returns.

WHY THE PLATE + RAY-CAST DESIGN, AND WHAT MAKES DM CONTROL HARDER THAN OGBENCH
-----------------------------------------------------------------------------

The camera of these domains is a ``fixed`` model camera and the scene's static half never moves, so
floor, rails, walls, decorations and skybox are one constant image — the same bake OGBench uses. Two
things then differ, and both were measured rather than assumed
(``scratch/dmcontrol/probe_plate.py``, 8 states per domain at 84 px):

  * **the ground material has ``reflectance=.2``**, so a moving geom is MIRRORED in the floor;
  * **the model light is a shadow-casting SPOTLIGHT** (``cutoff=45``, ``exponent=10``,
    ``castshadow=1``, under ``<quality shadowsize="2048"/>``), so a moving geom DARKENS floor pixels
    it does not cover.

Neither can be baked, so the plate alone is not the frame. What the plate does cover is most of it:
the pixels that differ between the full frame and the plate are 0.15% (point_mass), 1.7% (acrobot),
2.9% (pendulum) and 3.0% (cartpole) of the frame, and of that the movers' OWN pixels are
0.14% / 1.2% / 1.3% / 2.0% — the rest is their shadows and reflections.

The shadow is handled without modelling MuJoCo's shadow map at all. A SECOND plate is baked with the
shadow-casting light's diffuse and specular set to zero, which is exactly the colour a static surface
has when that light does not reach it; a shadowed pixel then selects the second plate instead of the
first. That keeps every shadowed pixel as exact as the plate itself and leaves only the *shape* of
the shadow to the ray-caster, where a hard analytic edge meets a 2048-tap shadow map.

The reflection is handled the same way, from MuJoCo's own compositing rule
``frame = reflectance * mirror + (1 - reflectance) * surface``: a THIRD plate baked with
``mat_reflectance = 0`` gives ``surface`` directly, so a mover appearing in the mirror is
``reflectance * mover_colour + (1 - reflectance) * plate_norefl`` rather than a guess.

The tendon needs a third mechanism, and it is the one this family did not have. ``ball_in_cup`` draws
a spatial tendon — the string between ball and cup — and MuJoCo does not draw it as one connector: a
slack tendon HANGS, as a chain of 28 capsules along a catenary, and a taut one is a single capsule
recoloured by its limit constraint's impedance. That is a mover kind with no geom behind it, so it is
built per frame from the wrap points and the tendon's length (``catenary``, ``_tendon_frame``), and it
is exact rather than approximated: ``mjv_catenary`` is exported from ``libmujoco`` but absent from the
Python bindings, and the port here reproduces it to 0 ULP.

Three bakes per (domain, camera, size, appearance). A bake is one MuJoCo render, and the renderer is
cached per (domain, size) and NEVER closed — releasing an EGL context corrupts every other live one,
which is the rule ``stable_envs/testing/conformance.py`` carries.

WHAT PARITY IS, AND WHERE THE RESIDUAL LIVES
--------------------------------------------

``parity_interior_min`` is **not relaxed** on any of the eleven, and the evidence is a draw count
rather than one large evaluation: over **201 independent draws of 64 states** each (12,864 states per
cell) at the shipped ``default_supersample = 1``, interior-pixel agreement is exactly 1.0 wherever the
metric is defined — **zero disagreeing pixels out of up to 10.7 M per cell**. On four of them it is
UNDEFINED rather than 1.0, and that is a property of the scene: the interior mask keeps only pixels
whose 3x3 neighbourhood in the reference is one flat colour, and a DM Control frame often has none —
the checker floor sits under a spotlight gradient and the skybox is a speckled gradient.
``interior_frac`` measures 0.000000 on acrobot, point_mass and both reachers, so ``pixel_agreement``
returns None and the parity test falls back to all-pixel agreement; 0.0001-0.0052 on the two cartpoles
and pendulum; and 0.111-0.120 on ball_in_cup and the three finger cells, which do have flat regions.

That makes the whole-frame numbers the ones to read, and they need one qualification each way:
91.0-99.8% of pixels are bit-identical with mean error 0.05-1.09/255 (worst draw of the 201, per
cell), but the baked plate is most of the frame and is exact by construction. The number worth
defending is the part the ray-caster owns: **99.8-100.0% of every disagreeing pixel is inside the
mover pass**, and the plate's own pixels are 99.998-100.000% exact. What is left there is a one-level
rim on curved silhouettes (MuJoCo shades a tessellated sphere or capsule per vertex; an analytic
per-pixel evaluation is within a few levels of that, not equal to it) and a one-pixel band at the
shadow edge, where a 2048-tap shadow map meets one analytic occlusion ray.

Both of those last two are BETTER at ss=1 than they were at the ss=2 this family used to default to
(98.4-100.0% on the mover pass, plate 99.967-100.000%), and the reason is worth stating because it is
the same mechanism that makes the supersample droppable at all: a box filter over four samples MIXES
mover colour into pixels the segmentation buffer calls plate, so supersampling manufactures
disagreement on pixels that are otherwise exact by construction. At one ray per pixel a pixel is
plate or mover and nothing else.

One ``finger`` cell declares ``parity_all_exact_min`` — and nothing else. That floor is not the
renderer's: all-pixel exactness is interior exactness weighted by how much of the frame MOVES, and
finger's seven movers cover 9.4-11.7% of an 84 px frame against cartpole's 2.5-3.0%. See
``PARITY_NOTE``.

OPEN: THE INTERIOR CLAIM DOES NOT HOLD AT 256 PX, AND IT IS NOT THE RENDERER
----------------------------------------------------------------------------

Everything above is measured at ``default_size = 84``, which is the only size
``test_render_parity_with_official_renderer`` builds an env at. Rendered at 256 px the same cells
report interior disagreements — over 41 draws of 64 states: 137,164 of 6,435,029 on cartpole-balance,
155,852 of 6,506,486 on cartpole-swingup, 2,624 of 5,784,725 on acrobot, 2,739-4,144 of ~41 M on the
three finger cells, 400 of 7,894,559 on pendulum. **This is not caused by the supersample and it is
not a geometry error.** Three things say so and they should be read together:

* the counts are BIT-IDENTICAL at ss=1 and ss=2 on every cell that shows it, so nothing about
  coverage or sample positions is involved (the two reachers are the exception and there ss=1 is far
  BETTER — 76 against 1,634 — which points the same way);
* ``interior_mean_abs`` is 0.002-0.021/255, i.e. single pixels off by one level, not a shape landing
  in the wrong place;
* ``interior_frac`` itself rises with resolution (0.00014 -> 0.038 on cartpole-balance), so the mask
  is admitting a different population of pixels, not finding new errors in the same one.

The mechanism is the premise failure ``conformance.interior_mask`` documents in its own docstring:
"flat 3x3 in the reference" implies "no rounding ambiguity here" only for a renderer that fills
flat-coloured primitives, and it is false for a smooth per-pixel function whose quantisation contour
can sit arbitrarily far from any 3x3 non-flatness. DM Control's skybox and spotlit checker floor are
exactly that — slow ramps — and at 256 px they are three times slower per pixel, so a patch that was
non-flat at 84 px is flat at 256 px while the underlying float is still moving. It is the same class
of finding as ``MujocoPusher``'s one-ulp ``exp`` boundary in ``mujoco_locomotion.PARITY``: a real
disagreement, of one level, at a place where two implementations are entitled to round apart.

**It needs a declared floor with this mechanism named, not a silent pass**, and that is deliberately
NOT done here: these cells claim parity at their default size, the floor would have to be measured
per size, and widening the mask to accommodate a ramp would relax the strong claim for every
flat-shaded env in the registry — which is the trade ``interior_mask`` already refuses on the swm nav
family's behalf. Recorded so the next person to render this family at another resolution finds the
measurement rather than rediscovering it as a bug. Numbers in ``scratch/msaa/``.

MJWARP IS NOT USED FOR THE FIXED-CAMERA CELLS. MEASURED
--------------------------------------------------------

``docs/research/backend3d.md`` recommends MJWarp's batch raytracer, and it is the backend the VISUAL
tier uses. It is not at the parity bar on the fixed-camera cells (``scratch/dmcontrol/probe_render.py``
and ``probe_trackcom.py``, 84 px, every correction this repo knows about applied — BGRA channel swap,
the zero-alpha pre-pass, geom groups matched on both sides, and ``gl_rounding`` which these cells
deliberately do NOT take because it costs them mean 5.016 -> 5.076 and worse; see ``_tc_context``):
4.41/255 mean on
ball_in_cup, 4.65 on finger, 28.43 on reacher. Reading the buffer without the channel swap gives
**0.0% exact and MAE 30-62 on every domain**, and ``use_shadows=True`` changed the score by less than
1e-4 anywhere — MJWarp is not casting this suite's spotlight shadow.

The top-down domains are the worst by 6x and the reason is that same spotlight: ``reacher`` and
``fish`` look straight down at a floor that fills the frame, MuJoCo paints a strong radial falloff on
it and MJWarp renders it nearly flat, at mean 25-29/255 with 1-12% of pixels within 8. That is why
``fish`` is measured and not shipped even in the visual tier, while ``reacher`` ships at parity: the
plate holds that falloff exactly, because MuJoCo painted it.
"""
from __future__ import annotations

import math
import os

import numpy as np
import torch

# MuJoCo picks its GL backend from MUJOCO_GL and defaults to GLFW, which needs an X display; on a
# headless GPU node that aborts the interpreter rather than raising. Every machine this runs on
# renders offscreen. An explicit setting always wins.
os.environ.setdefault("MUJOCO_GL", "egl")

from ..core.env import StableEnv, register
from ..render import mjwarp as MW
from ..render import raycast3d as R3
from ..spaces import Box, Dict

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
EPISODE_ARRAYS = ("geom_pos", "geom_quat", "geom_size", "site_pos", "site_quat", "site_size",
                  "body_pos", "body_quat", "light_pos", "light_dir")
#: Resets used to discover the draw. Six is what separates a per-episode draw from a task CONSTANT on
#: every domain here: ``finger-spin`` writes ``site_rgba`` and ``dof_damping`` at every
#: ``initialize_episode`` and writes the same values every time, so one reset cannot tell the two
#: apart and six can. More only costs reset time, and the count is fixed so the discovery is
#: reproducible across processes.
N_DRAWS = 6
#: Component labels, so a widened state dimension is named ``target_pos_x`` and not ``slot 3``.
_COMPONENT = {3: ("x", "y", "z"), 4: ("qw", "qx", "qy", "qz")}

#: The domains that land in the PARITY tier: a ``fixed`` model camera (so there IS a static plate) and
#: primitive geoms only. A model the task DRAWS FROM per episode is no longer an exclusion — the drawn
#: entries enter the state and the elements they move become movers (``_episode_slots``), which is what
#: adds ``reacher`` and ``finger`` here.
#:
#: What is still excluded, and why, because "why is X missing" is the first question:
#:   * cheetah / walker / hopper / swimmer / fish / humanoid use a ``trackcom`` camera that rides the
#:     body, so the static scene moves with the state and there is no plate to bake. Five of them ship
#:     in the VISUAL tier below, through a backend that draws the whole scene every frame.
#:   * dog and quadruped carry ``<mesh>`` assets, which an analytic caster does not intersect.
#: ``ball_in_cup`` is here because the spatial tendon is now a mover kind the caster has: MuJoCo draws
#: it as a hanging CATENARY of 28 capsules, and ``catenary`` reproduces ``mjv_catenary`` bit-for-bit.
DOMAINS: dict[str, tuple[str, str, int]] = {
    "DMCCartpoleSwingup": ("cartpole", "swingup", 0),
    "DMCCartpoleBalance": ("cartpole", "balance", 0),
    "DMCPendulumSwingup": ("pendulum", "swingup", 0),
    "DMCAcrobotSwingup": ("acrobot", "swingup", 0),
    "DMCPointMassEasy": ("point_mass", "easy", 0),
    "DMCReacherEasy": ("reacher", "easy", 0),
    "DMCReacherHard": ("reacher", "hard", 0),
    "DMCFingerSpin": ("finger", "spin", 0),
    "DMCFingerTurnEasy": ("finger", "turn_easy", 0),
    "DMCFingerTurnHard": ("finger", "turn_hard", 0),
    "DMCBallInCupCatch": ("ball_in_cup", "catch", 0),
}

#: The domains whose camera 0 rides the body, drawn by MJWarp in the VISUAL tier. ``fish`` is measured
#: and deliberately absent — see ``DMControlTrackCom`` and ``docs/research/dmcontrol.md`` S9.
TRACKCOM_DOMAINS: dict[str, tuple[str, str, int]] = {
    "DMCCheetahRun": ("cheetah", "run", 0),
    "DMCWalkerWalk": ("walker", "walk", 0),
    "DMCHopperHop": ("hopper", "hop", 0),
    "DMCHumanoidStand": ("humanoid", "stand", 0),
    "DMCSwimmerSwimmer6": ("swimmer", "swimmer6", 0),
}

_SCENE_CACHE: dict = {}
_SCENE_CACHE_MAX = 96
#: (domain, task, size) -> (dm_control env, MjModel, MjData, pristine snapshot, mujoco.Renderer,
#: episode slots). One Renderer per (domain, size), built once and NEVER closed.
_ENV_CACHE: dict = {}
#: (domain, task, res, worlds) -> a captured MJWarp context. Small, because a context owns
#: ``nworld * res^2 * 12`` bytes of device memory and holding several domains' at once buys nothing.
_TC_CACHE: dict = {}
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
            f"dm_control==1.0.41. See docs/research/dmcontrol.md.") from e


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
        return 0.5, 0.5                                # MuJoCo's material defaults
    return float(m.mat_specular[matid]), float(m.mat_shininess[matid])


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
    reg = np.asarray(getattr(m, attr)[a0:a0 + 3 * h * w], np.float64).reshape(-1, 3) / 255.0
    u = np.unique(reg, axis=0)
    return u[0] if len(u) == 1 else reg.mean(0)


def _effective_alpha(m, kind: str) -> np.ndarray:
    """The alpha MuJoCo actually draws a geom or site with, resolved through its material.

    ``mujoco.Renderer`` selects geometry by GROUP, which is what ``VISIBLE_GROUPS`` covers, and then
    MuJoCo skips anything whose resolved alpha is zero. Both filters have to be applied or the
    difference is silent: ``finger``'s ``target`` and ``tip`` sites and ``fish``'s ``target`` geom are
    hidden that way rather than by group, so a renderer that reads groups alone paints markers the
    reference does not have — and paints them in the ray-cast pass, where they land on the metric.

    Same resolution ``stable_envs.render.mjwarp.effective_alpha`` performs, and the same rule
    ``_mat_rgb`` uses for colour: the element's own rgba unless that is the compiler default and a
    material supplies one.
    """
    rgba = np.asarray(getattr(m, f"{kind}_rgba"), np.float64)
    matid = np.asarray(getattr(m, f"{kind}_matid"), np.int64)
    a = rgba[:, 3].copy()
    default = np.all(np.isclose(rgba, [0.5, 0.5, 0.5, 1.0]), axis=1)
    for i in np.flatnonzero(default & (matid >= 0)):
        a[i] = float(m.mat_rgba[matid[i], 3])
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
    obj = {"geom": mujoco.mjtObj.mjOBJ_GEOM, "site": mujoco.mjtObj.mjOBJ_SITE,
           "body": mujoco.mjtObj.mjOBJ_BODY, "light": mujoco.mjtObj.mjOBJ_LIGHT}[kind]
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
    bad = [s for s in slots if s[0].endswith(("_size", "_quat", "_dir"))]
    if bad:
        # Not "unsupported for now": a per-episode SIZE is a different renderer. The mover
        # primitives' sizes are read once at bake time and shared by the whole batch, so a size in the
        # state would have to become a per-row tensor through every intersection routine. Nothing in
        # the shipped set draws one, and a domain that starts to must fail loudly rather than render
        # every environment at one draw's size.
        raise RuntimeError(
            f"{env.__class__.__name__}: the episode draws {[_slot_name(m, a, i) for a, i in bad]}, "
            f"which change a primitive's SIZE or ORIENTATION rather than its position. The mover "
            f"primitives carry one size and one orientation for the whole batch, so this domain needs "
            f"a per-row size/quat path before it can be honest.")
    return tuple(slots)


# --------------------------------------------------------------------------- spatial tendons

_MJMINVAL = 1e-15
_CAT_TOL = 1e-9


def _cosh_sinh(x: float) -> tuple[float, float]:
    e = math.exp(x)
    return 0.5 * (e + 1.0 / e), 0.5 * (e - 1.0 / e)


def _cat_residual(b: float, intercept: float, want_grad: bool):
    a = 0.5 / b
    ch, sh = _cosh_sinh(a)
    res = 1.0 / math.sqrt(2 * b * sh - 1.0) - intercept
    grad = (a * ch - sh) * math.pow(2 * b * sh - 1.0, -1.5) if want_grad else None
    return res, grad


def _solve_catenary(v: float, h: float, length: float) -> float:
    """MuJoCo's own solve: Newton on the change-of-variables form, 50 steps, tol 1e-9, with a
    backtracking halving loop of at most 10 that accepts the first step reducing ``|res|``."""
    intercept = 1.0 / math.sqrt(math.sqrt(length * length - v * v) / h - 1.0)
    b = intercept / math.sqrt(24.0)
    for _ in range(50):
        res, grad = _cat_residual(b, intercept, True)
        if abs(res) < _CAT_TOL:
            break
        step = -res / grad
        for _ in range(10):
            if abs(_cat_residual(b + step, intercept, False)[0]) < abs(res):
                break
            step *= 0.5
        b += step
    return b


def catenary(x0, x1, gravity, length: float, npoints: int) -> np.ndarray:
    """MuJoCo's ``mjv_catenary``, in numpy. ``(N,3)`` float64 with ``N <= npoints``.

    A spatial tendon with slack is not drawn as a straight line: since MuJoCo 3.2 it hangs, and
    ``dm_control``'s ``ball_in_cup`` string is 28 capsules along a catenary rather than the one
    connector the XML reads like. That is the "mover kind the ray-caster does not have" that
    ``docs/research/dmcontrol.md`` S8 recorded, and it is not an approximation problem: ``mjv_catenary``
    is exported from ``libmujoco`` but is NOT in the Python bindings, so it had to be reproduced.

    It is reproduced EXACTLY. Validated against the C symbol through ctypes on 440 inputs spanning all
    three branches and on 64 real ``ball_in_cup`` states: max absolute error 0.0, max ULP difference 0
    (``scratch/dmcontrol/probe_catenary.py``). Two things buy that and neither is stylistic —
    ``math.*`` is a direct call into the same libm MuJoCo's ``mju_*`` macros expand to, where a numpy
    ufunc is not, and the Newton loop below is MuJoCo's iteration count, tolerance, seed and
    backtracking rather than an equivalent solve.

    Three branches, all of which ``ball_in_cup`` reaches during a normal rollout:

      * ``dist > length`` -- taut, two points. (Reached from here only defensively: the CALLER decides
        slackness the way ``mjv_isCatenary`` does, from ``ten_length`` against the limit, and takes the
        wrap points rather than this branch when the tendon is taut, because MuJoCo does.)
      * ``length > 100 * h`` -- near-vertical, three points and a hanging-bead approximation rather
        than a catenary. Fires in 3 of 64 sampled states.
      * otherwise ``npoints`` points, spaced linearly in the HORIZONTAL coordinate and divided by
        ``npoints`` rather than ``npoints - 1``, so the last interior point sits at ``(N-2)/N`` of the
        span and the final capsule comes out about twice as long as the others. That reads like an
        off-by-one and it is what MuJoCo draws, which is the only thing that matters here.
    """
    x0 = np.asarray(x0, dtype=np.float64)
    x1 = np.asarray(x1, dtype=np.float64)
    d = x0 - x1
    if math.sqrt(d[0] * d[0] + d[1] * d[1] + d[2] * d[2]) > length:
        return np.stack([x0, x1])
    up = np.asarray(gravity, dtype=np.float64) * -1.0
    n = math.sqrt(up[0] * up[0] + up[1] * up[1] + up[2] * up[2])
    up = np.array([1.0, 0.0, 0.0]) if n < _MJMINVAL else up * (1.0 / n)
    x01 = x1 - x0
    ac = x01 - up * (up[0] * x01[0] + up[1] * x01[1] + up[2] * x01[2])
    n = math.sqrt(ac[0] * ac[0] + ac[1] * ac[1] + ac[2] * ac[2])
    ac = np.zeros(3) if n < _MJMINVAL else ac * (1.0 / n)
    h = x01[0] * ac[0] + x01[1] * ac[1] + x01[2] * ac[2]        # horizontal extent
    v = x01[0] * up[0] + x01[1] * up[1] + x01[2] * up[2]        # vertical rise
    if length > 100 * h:
        d_up = -0.5 * (math.sqrt(length * length - h * h) - v)
        return np.stack([x0, x0 + up * d_up + ac * (h * d_up / (2 * d_up - v)), x1])
    bh = _solve_catenary(v, h, length) * h
    h_off = -0.5 * (math.log((length + v) / (length - v)) * bh - h)
    v_off = -_cosh_sinh(h_off / bh)[0] * bh
    pts = np.empty((npoints, 3))
    pts[0], pts[npoints - 1] = x0, x1
    for i in range(1, npoints - 1):
        hor = i * h / npoints
        pts[i] = x0 + ac * hor + up * (bh * _cosh_sinh((hor - h_off) / bh)[0] + v_off)
    return pts


def _chain_capsules(pts: np.ndarray, nseg: int):
    """A polyline of ``K`` segments -> exactly ``nseg`` capsule poses. ``(pos, mat, half_length)``.

    The chain has to be a FIXED number of primitives, because the mover set is fixed at bake time and
    a batch holds states from all three branches at once — 28 segments when the string hangs, 2 when it
    is near-vertical, 1 when it is taut. Each real segment is therefore split into ``nseg // K``
    COLLINEAR pieces, which is exact rather than an approximation: the union of collinear capsules
    covering a segment is that segment's capsule, and the surface normal agrees too — a point on an
    interior piece's cap sits exactly on the shared axis's perpendicular, so its normal is the same
    radial vector the full capsule's cylinder gives there. Nothing is resampled and no piece is
    degenerate.

    Only the third column of the rotation is meaningful: a capsule is a surface of revolution about
    its own axis, so any orthonormal completion draws the same solid.
    """
    K = len(pts) - 1
    if nseg % K:
        raise RuntimeError(f"a tendon drawn as {K} segments cannot be split into {nseg} pieces "
                           f"without resampling it; {nseg} must be a multiple of every branch's "
                           f"segment count")
    per = nseg // K
    p0 = np.asarray(pts[:-1], np.float64).reshape(K, 1, 3)
    p1 = np.asarray(pts[1:], np.float64).reshape(K, 1, 3)
    t = (np.arange(per, dtype=np.float64) / per).reshape(1, per, 1)
    lo = (p0 + (p1 - p0) * t).reshape(nseg, 3)
    hi = (p0 + (p1 - p0) * (t + 1.0 / per)).reshape(nseg, 3)
    axis = hi - lo
    half = np.linalg.norm(axis, axis=1) * 0.5
    z = axis / np.maximum(np.linalg.norm(axis, axis=1, keepdims=True), _MJMINVAL)
    # an arbitrary orthonormal completion, taken from the world axis least aligned with z
    ref = np.zeros_like(z)
    ref[np.arange(len(z)), np.argmin(np.abs(z), axis=1)] = 1.0
    x = np.cross(ref, z)
    x /= np.maximum(np.linalg.norm(x, axis=1, keepdims=True), _MJMINVAL)
    y = np.cross(z, x)
    return 0.5 * (lo + hi), np.stack([x, y, z], axis=2), half


class _Tendon:
    """The one spatial tendon a scene draws, and everything a frame needs about it."""

    __slots__ = ("id", "radius", "nseg", "ncat", "grey", "constraint", "upper", "gravity")


def _find_tendon(m) -> "_Tendon | None":
    """The scene's drawn spatial tendon, or ``None``. Refuses anything it cannot draw exactly.

    ``ntendon > 0`` is not the question: ``point_mass`` and ``fish`` both carry two tendons and draw
    neither, because they are FIXED tendons (joint-space couplings with no spatial path). A tendon
    reaches the frame only if it wraps SITES or wrapping objects.
    """
    import mujoco
    spatial = []
    for t in range(int(m.ntendon)):
        a = int(m.tendon_adr[t])
        n = int(m.tendon_num[t])
        kinds = [int(m.wrap_type[a + k]) for k in range(n)]
        if any(k != int(mujoco.mjtWrap.mjWRAP_JOINT) for k in kinds):
            spatial.append((t, kinds))
    if not spatial:
        return None
    if len(spatial) > 1:
        raise RuntimeError(f"this scene draws {len(spatial)} spatial tendons; the mover set here "
                           f"carries one")
    t, kinds = spatial[0]
    site = int(mujoco.mjtWrap.mjWRAP_SITE)
    if kinds != [site, site] or not int(m.tendon_limited[t]):
        # ``mjv_isCatenary`` needs exactly two sites and a length constraint (or a spring with a
        # deadband); anything else takes MuJoCo's wrapping path, which computes a different polyline.
        raise RuntimeError(f"tendon {t} wraps {kinds} and limited={int(m.tendon_limited[t])}; the "
                           f"catenary path here covers a two-site, length-limited tendon only")
    td = _Tendon()
    td.id = t
    td.radius = float(m.tendon_width[t])
    td.upper = float(m.tendon_range[t][1])
    td.ncat = min(int(m.vis.quality.numslices) + 1, 100)
    td.nseg = td.ncat - 1
    td.grey = np.asarray(m.tendon_rgba[t], np.float64)[:3].copy()
    td.constraint = np.asarray(m.vis.rgba.constraint, np.float64)[:3].copy()
    td.gravity = np.asarray(m.opt.gravity, np.float64).copy()
    return td


def _tendon_frame(m, d, td: _Tendon):
    """``(pos, mat, half_length, rgb)`` for this state's chain. ``mj_forward`` must have run.

    Which branch is a question about the CONSTRAINT, not about the geometry: ``mjv_isCatenary`` asks
    whether the length limit is satisfied, and only then hands the two endpoints to ``mjv_catenary``
    with the limit as the target length. Taut, MuJoCo takes a completely different path — one straight
    capsule between the wrap points, recoloured by the limit's IMPEDANCE:

        rgb = (1 - imp) * tendon_rgba + imp * vis.rgba.constraint

    which is why the string turns red when the ball pulls the cup. ``imp`` is column 2 of
    ``efc_KBIP`` on the tendon-limit row, and that is measured rather than guessed: over 88 taut
    states the four columns reproduce MuJoCo's own scene colour to 1384, 52, **2.9e-08** and 23
    respectively (``scratch/dmcontrol/probe_tendon_colour.py``). The 2.9e-08 is the float32 round-trip
    through ``mjvGeom.rgba``.

    A random policy leaves ``ball_in_cup`` taut in 35-44% of the states it visits, so this is not a
    branch a renderer can decline: skipping it would paint a third of the dataset's strings grey.
    """
    import mujoco
    adr, num = int(d.ten_wrapadr[td.id]), int(d.ten_wrapnum[td.id])
    w = np.asarray(d.wrap_xpos, np.float64).reshape(-1, 3)[adr:adr + num]
    if float(d.ten_length[td.id]) < td.upper:
        pts = catenary(w[0], w[1], td.gravity, td.upper, td.ncat)
        rgb = td.grey
    else:
        pts = w[:2]
        imp = 0.0
        if int(d.nefc):
            et = np.asarray(d.efc_type, np.int64)
            ei = np.asarray(d.efc_id, np.int64)
            hit = np.flatnonzero((et == int(mujoco.mjtConstraint.mjCNSTR_LIMIT_TENDON))
                                 & (ei == td.id))
            if len(hit):
                imp = float(np.asarray(d.efc_KBIP)[hit[0], 2])
        rgb = (1.0 - imp) * td.grey + imp * td.constraint
    pos, mat, half = _chain_capsules(pts, td.nseg)
    return pos, mat, half, rgb


class _Lights:
    """The lights as ``mjvScene`` reports them, which is the only place the headlight exists.

    Nothing is folded together. The headlight is directional along the camera axis with
    ambient .4 / diffuse .8 / specular .1 and casts no shadow; the model light is a spotlight
    (cutoff 45 degrees, exponent 10) with ambient 0 / diffuse .7 / specular .3 that does. Summing them
    into one direction is legitimate for OGBench, where both point straight down, and would erase the
    spotlight falloff that IS the picture in point_mass.
    """

    __slots__ = ("kind", "pos", "dir", "ambient", "diffuse", "specular", "attenuation", "cutoff",
                 "exponent", "castshadow")

    def __init__(self, scene):
        n = int(scene.nlight)
        g = lambda f: np.stack([np.asarray(getattr(scene.lights[i], f), np.float64)  # noqa: E731
                                for i in range(n)])
        self.pos, self.dir = g("pos"), g("dir")
        self.ambient, self.diffuse, self.specular = g("ambient"), g("diffuse"), g("specular")
        self.attenuation = g("attenuation")
        self.cutoff = np.array([float(scene.lights[i].cutoff) for i in range(n)])
        self.exponent = np.array([float(scene.lights[i].exponent) for i in range(n)])
        self.castshadow = np.array([int(scene.lights[i].castshadow) for i in range(n)])
        self.kind = np.array([int(getattr(scene.lights[i], "type",
                                          getattr(scene.lights[i], "directional", 1)))
                              for i in range(n)])

    def shadow_light(self) -> int:
        idx = np.flatnonzero(self.castshadow > 0)
        return int(idx[0]) if len(idx) else -1

    def to(self, device) -> dict:
        t = lambda x: torch.tensor(np.asarray(x, np.float64), device=device,          # noqa: E731
                                   dtype=torch.float32)
        return dict(pos=t(self.pos), dir=t(self.dir), ambient=t(self.ambient),
                    diffuse=t(self.diffuse), specular=t(self.specular),
                    attenuation=t(self.attenuation), cutoff=[float(c) for c in self.cutoff],
                    exponent=[float(e) for e in self.exponent],
                    kind=[int(k) for k in self.kind],
                    castshadow=[bool(c) for c in self.castshadow])


class _Scene:
    """Everything a render needs that does not depend on the state."""

    __slots__ = ("plate", "plate_shadow", "plate_norefl", "reflectance", "static_t", "cam", "model",
                 "data", "movers", "size", "lights", "prims", "shadow_light", "floor", "eps",
                 "tendon", "shadow_candidates", "rays", "const")


# --------------------------------------------------------------------------- shading

def _norm(v):
    return v / torch.clamp(torch.sqrt((v * v).sum(-1, keepdim=True)), min=1e-12)


def _dot(a, b):
    """Unrolled for the reason ``raycast3d._dot`` is: a reduction whose kernel is chosen from the
    batch count sums its terms in a batch-dependent order, and one ulp there is a hit/miss flip."""
    return a[..., 0] * b[..., 0] + a[..., 1] * b[..., 1] + a[..., 2] * b[..., 2]


def _light_geometry(L: dict, i: int, p: torch.Tensor):
    """``(toward_light, attenuation)`` for light ``i`` at points ``p``.

    OpenGL fixed function, which is what ``mjr_render`` drives: a positional light attenuates by
    ``1/(k0 + k1 d + k2 d^2)`` and, inside its cutoff, by ``(spotdir . -l)^exponent``; a directional
    light does neither and has no position at all.
    """
    shape = (1,) * (p.dim() - 1) + (3,)
    if L["kind"][i] == LIGHT_DIRECTIONAL:
        l = -_norm(L["dir"][i].view(shape)).expand_as(p)
        return l, torch.ones(p.shape[:-1], device=p.device, dtype=p.dtype)
    v = L["pos"][i].view(shape) - p
    dist = torch.sqrt(torch.clamp(_dot(v, v), min=1e-24))
    l = v / dist.unsqueeze(-1)
    k = L["attenuation"][i]
    att = 1.0 / torch.clamp(k[0] + k[1] * dist + k[2] * dist * dist, min=1e-12)
    cut = L["cutoff"][i]
    if cut < 180.0:
        se = _dot(_norm(L["dir"][i].view(shape)).expand_as(l), -l)
        inside = se >= float(np.cos(np.deg2rad(cut)))
        att = torch.where(inside, att * torch.clamp(se, min=0) ** L["exponent"][i],
                          torch.zeros_like(att))
    return l, att


def _shade(p, n, v, L, mat, spec, shininess, unlit=None):
    """``(...,3)`` colour in [0,1] for surface points ``p`` with unit normal ``n``, viewer dir ``v``.

    The composition is the one measured in ``docs/research/ogbench.md``: the specular term is WHITE
    and ADDITIVE and the texture modulates the whole primary colour. What is new here is that the
    per-light weights cannot be summed in advance — the two lights point in different directions and
    one of them is a spotlight — and that ``unlit`` masks the diffuse and specular of the
    shadow-casting light only, because a shadow does not remove ambient.
    """
    out = torch.zeros_like(p)
    highlight = torch.zeros(p.shape[:-1], device=p.device, dtype=p.dtype)
    exponent = 128.0 * shininess
    for i in range(len(L["kind"])):
        l, att = _light_geometry(L, i, p)
        lit = att
        if unlit is not None and L["castshadow"][i]:
            lit = att * (~unlit).to(att.dtype)
        ndl = torch.clamp(_dot(n, l), min=0)
        h = _norm(l + v)
        ndh = torch.clamp(_dot(n, h), min=0)
        out = out + mat * (att * L["ambient"][i][0]).unsqueeze(-1)
        out = out + mat * (lit * ndl * L["diffuse"][i][0]).unsqueeze(-1)
        highlight = highlight + lit * L["specular"][i][0] * ndh ** exponent
    return torch.clamp(out + (highlight * spec * SPECULAR_SCALE).unsqueeze(-1), 0, 1)


def _upsample_rgb(img: torch.Tensor, ss: int) -> torch.Tensor:
    """Nearest-neighbour blow-up of a baked plate, chosen so the box average that follows is the
    IDENTITY on plate pixels: every sample inside an output pixel carries the same integer, so the
    mean is that integer and ``round`` returns it. The plate is exact by construction, not
    approximately preserved."""
    return img if ss == 1 else img.repeat_interleave(ss, 0).repeat_interleave(ss, 1)


# --------------------------------------------------------------------------- the env

class _DMCModel(StableEnv):
    """What every DM Control cell shares, whichever renderer draws it.

    The model plumbing, the widened state, the appearance knobs, the visited-state distribution and
    the bridge to ``dm_control``'s own renderer are properties of the DOMAIN. Which rasterizer draws
    the frame is not, and the two tiers below disagree about it entirely: ``DMControlSuite`` bakes a
    plate and ray-casts the movers (parity), ``DMControlTrackCom`` hands the whole scene to MJWarp
    every frame (visual). Everything that would otherwise be written twice, and drift, lives here.
    """

    DOMAIN: str = ""
    TASK: str = ""
    CAMERA: int = 0
    default_size = 84
    default_supersample = 1
    #: Whether the appearance space offers a field-of-view knob. It is a class flag rather than an
    #: always-on knob because whether it can be honoured is a property of the RENDERER: the ray-caster
    #: re-bakes per appearance and gets it for free, while MJWarp would need ``cam_fovy`` per world and
    #: ``put_model`` does not carry that array per world.
    CAMERA_KNOB = True

    def __init__(self, batch: int = 1, size: int | None = None, supersample: int | None = None,
                 device: str = "cuda", backend: str = "auto", seed: int = 0):
        m = self._model()
        self.nq = int(m.nq)
        self.state_dim = self.nq + len(self._slots())
        self.state_names = self._qpos_names(m) + self._slot_names(m)
        super().__init__(batch=batch, size=size,
                         supersample=self.default_supersample if supersample is None else supersample,
                         device=device, backend=backend, seed=seed)
        assert len(self.state_names) == self.state_dim

    # ---- model / reference plumbing --------------------------------------

    @classmethod
    def _bundle(cls, size: int):
        """``(dm_control env, MjModel, MjData, pristine, Renderer, slots)``, one per (domain, size).

        The model is put through ``N_DRAWS`` real episodes before anything is read off it, and that is
        load-bearing twice over. ``initialize_episode`` is where a DM Control task edits its model, and
        a model straight out of ``suite.load`` has had none of those edits applied — ``finger-spin``
        hides its ``target`` and ``tip`` sites by setting their alpha to zero there, so a renderer
        reading the freshly loaded model paints two markers the reference does not draw. Running the
        episodes also SEPARATES the constant edits from the per-episode draw, which is the distinction
        ``_episode_slots`` needs and which one reset cannot make.

        The draws are deterministic: ``task_kwargs={"random": 0}`` seeds the task's own
        ``RandomState`` and the count is fixed, so a fresh interpreter reaches the same canonical
        model. That matters for the same reason ``_pristine`` does — the cross-process bit-identity
        test is the only thing that can see a model that differs between two processes.
        """
        key = (cls.DOMAIN, cls.TASK, int(size))
        hit = _ENV_CACHE.get(key)
        if hit is not None:
            return hit
        import mujoco
        env = _suite().load(cls.DOMAIN, cls.TASK, task_kwargs={"random": 0})
        p = env.physics
        m = p.model.ptr if hasattr(p.model, "ptr") else p.model
        m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(size))
        m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(size))
        slots = _episode_slots(env, m)
        d = mujoco.MjData(m)
        pristine = dict(mat_rgba=np.array(m.mat_rgba, copy=True),
                        geom_rgba=np.array(m.geom_rgba, copy=True),
                        site_rgba=np.array(m.site_rgba, copy=True),
                        geom_group=np.array(m.geom_group, copy=True),
                        site_group=np.array(m.site_group, copy=True),
                        light_diffuse=np.array(m.light_diffuse, copy=True),
                        light_specular=np.array(m.light_specular, copy=True),
                        mat_reflectance=np.array(m.mat_reflectance, copy=True),
                        cam_fovy=np.array(m.cam_fovy, copy=True))
        for arr in {a for a, _ in slots}:
            pristine[arr] = np.array(getattr(m, arr), copy=True)
        # ONE Renderer, kept for the life of the process. Material colours, light colours and the
        # camera are read per frame, so no variation here needs a second GL context — only a TEXTURE
        # edit would, and none is offered. Closing it would corrupt every other live context.
        r = mujoco.Renderer(m, height=int(size), width=int(size))
        _ENV_CACHE[key] = (env, m, d, pristine, r, slots)
        return _ENV_CACHE[key]

    @classmethod
    def _slots(cls) -> tuple[tuple[str, int], ...]:
        """``((model array, flat index), ...)`` the episode draws — the widening of the state."""
        return cls._bundle(cls.default_size)[5]

    @classmethod
    def _slot_names(cls, m) -> tuple[str, ...]:
        return tuple(_slot_name(m, arr, i) for arr, i in cls._slots())

    @classmethod
    def _model(cls):
        return cls._bundle(cls.default_size)[1]

    @classmethod
    def _pristine(cls) -> dict:
        """The model AS COMPILED, which is the only thing an appearance default may be read from.

        The ``MjModel`` in ``_bundle`` is shared by every instance of this class and is mutated by every
        bake, and ``_apply_variation`` restores it from this snapshot at the START of a bake rather than
        at the end — so between bakes it carries the LAST appearance rendered, not the official one.
        Reading ``make_variation_space``'s ``init_value`` off the live model therefore made a new env's
        default appearance depend on what the previous env had most recently drawn.

        Measured, because it is invisible from inside one env: the default plate's sha256 changed after
        ``check_variations_do_something`` ran, and the only test that could see it was the cross-process
        one — a fresh interpreter has no previous bake, so the subprocess and the parent disagreed on
        what "default" meant while every same-process check passed.
        """
        return cls._bundle(cls.default_size)[3]

    @staticmethod
    def _qpos_names(m) -> tuple[str, ...]:
        """One label per ``qpos`` entry, read off the model's joint layout rather than transcribed —
        a state vector whose fields are mislabelled is worse than one with no labels."""
        import mujoco
        free = ("_x", "_y", "_z", "_qw", "_qx", "_qy", "_qz")
        ball = ("_qw", "_qx", "_qy", "_qz")
        out: list[str] = []
        for j in range(m.njnt):
            nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) or f"joint{j}"
            t = int(m.jnt_type[j])
            if t == int(mujoco.mjtJoint.mjJNT_FREE):
                out += [nm + s for s in free]
            elif t == int(mujoco.mjtJoint.mjJNT_BALL):
                out += [nm + s for s in ball]
            else:
                out.append(nm)
        return tuple(out)

    # ---- variation --------------------------------------------------------

    @classmethod
    def _used_materials(cls) -> list[str]:
        """Materials carried by a VISIBLE geom or site, in model order.

        Declared from the model rather than from a fixed list because
        ``check_variations_do_something`` moves every declared leaf and fails if the frame does not
        change: cartpole has no ``effector`` and acrobot no ``target``, so a shared knob set would
        advertise support that does not exist in three of the five domains.
        """
        import mujoco
        m, pr = cls._model(), cls._pristine()
        used = []
        for kind, arr, grp, n in (("geom", m.geom_matid, pr["geom_group"], m.ngeom),
                                  ("site", m.site_matid, pr["site_group"], m.nsite)):
            alpha = _effective_alpha(m, kind)
            for i in range(n):
                # alpha as well as group: ``finger-spin`` hides its ``target`` and ``tip`` sites by
                # setting their alpha to zero, and they are the ONLY wearers of the ``target``
                # material — so a group-only census declares a knob that cannot change any pixel and
                # ``check_variations_do_something`` fails on a knob that was never real.
                if int(grp[i]) not in VISIBLE_GROUPS or alpha[i] <= 0:
                    continue
                mid = int(arr[i])
                if mid >= 0 and mid not in used:
                    used.append(mid)
        return [mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i) or f"mat{i}" for i in used]

    def make_variation_space(self) -> Dict:
        """Appearance knobs whose DEFAULTS are read off the compiled model, so "the default style is
        the official appearance" is true by construction rather than by a transcribed constant.

        The colours are float ``Box`` in [0,1] rather than ``RGBBox`` for the reason the OGBench
        mazes are: MuJoCo's material colours are floats (0.7, 0.5, 0.3) that a uint8 channel cannot
        hold, and in a lit 3-D scene no material colour appears verbatim in any pixel anyway, because
        a pixel is the material colour times the lighting.
        """
        import mujoco
        m, pr = self._model(), self._pristine()
        mats = {}
        for name in self._used_materials():
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            # from the PRISTINE snapshot, never the live model — see _pristine for what that cost
            mats[name] = Box(low=0.0, high=1.0, shape=(3,),
                             init_value=[float(x) for x in pr["mat_rgba"][i][:3]])
        space = {
            "material": Dict(mats),
            "light": Dict({"diffuse_scale": Box(low=0.2, high=1.8, shape=(1,), init_value=[1.0])}),
        }
        if self.CAMERA_KNOB:
            space["camera"] = Dict({"fovy_scale": Box(low=0.55, high=1.8, shape=(1,),
                                                      init_value=[1.0])})
        return Dict(space)

    def _var(self, variation, path):
        if variation is not None and path in variation:
            return np.asarray(variation[path], np.float64).reshape(-1)
        node = self.variation_space
        for part in path.split("."):
            node = node[part]
        return np.asarray(node.value, np.float64).reshape(-1)

    # ---- states -----------------------------------------------------------

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """The domain's OWN visited distribution: its ``initialize_episode`` plus random actions.

        Not a convenient box, and not the reset distribution alone: cartpole-balance starts near
        upright and would never show the renderer a hanging pole, so a parity number measured there
        would be a statement about one pose. Every conformance number in the suite is measured on
        these states, and a parity test on states the env never reaches proves very little.

        The episode half of the state is read out of the model the episode wrote it into, so a
        sampled state is a pairing the domain genuinely produced rather than a ``qpos`` glued to a
        target drawn somewhere else.
        """
        rng = rng or self.rng
        seed = int(rng.integers(0, 2 ** 31 - 1))
        env = _suite().load(self.DOMAIN, self.TASK, task_kwargs={"random": seed})
        p = env.physics
        m = p.model.ptr if hasattr(p.model, "ptr") else p.model
        spec = env.action_spec()
        r2 = np.random.default_rng(seed)
        slots = self._slots()
        out = np.empty((int(batch), self.state_dim), np.float64)
        for i in range(int(batch)):
            env.reset()
            for _ in range(int(r2.integers(0, 40))):
                env.step(r2.uniform(spec.minimum, spec.maximum))
            out[i, :self.nq] = np.asarray(p.data.qpos, np.float64)
            for j, (arr, k) in enumerate(slots):
                out[i, self.nq + j] = np.asarray(getattr(m, arr)).reshape(-1)[k]
        return out.astype(np.float32)

    # ---- reference bridge -------------------------------------------------

    @classmethod
    def reference_env(cls, size: int = 84):
        """A FRESH official env, carrying the size it must render at.

        Never closed. ``test_state_is_sufficient`` drives one through two histories, and a
        dm_control ``Physics`` holds the EGL context its renderer created — releasing it corrupts
        rendering in every other live context in the process.
        """
        env = _suite().load(cls.DOMAIN, cls.TASK, task_kwargs={"random": 0})
        env.reset()
        env._se_size = int(size)
        return env

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Official frames for the given state, with the round-trip ASSERTED.

        ``qvel`` is set to zero and is provably irrelevant: scaling it by 100 changes zero pixels on
        every domain measured. The assertion is not decoration — without it, a state the reference did
        not actually reach produces a plausible parity number instead of a failure, and the mistake
        reads as a rendering discrepancy.
        """
        p = env.physics
        m = p.model.ptr if hasattr(p.model, "ptr") else p.model
        size = int(getattr(env, "_se_size", 0) or cls.default_size)
        slots = cls._slots()
        nq = int(m.nq)
        out = []
        for st in np.asarray(states, np.float64):
            # The episode half of the state goes into the reference's MODEL, which is where
            # ``initialize_episode`` put it. Written BEFORE the reset context so the ``mj_forward``
            # that closes the context sees it -- ``docs/research/metaworld.md`` S1.1 is the recorded
            # form of the opposite mistake, where a model edit after the last forward leaves
            # ``site_xpos`` holding the previous draw and no stored state can reproduce the frame.
            for (arr, k), v in zip(slots, st[nq:]):
                getattr(m, arr).reshape(-1)[k] = v
            with p.reset_context():
                p.data.qpos[:] = st[:nq]
                p.data.qvel[:] = 0
            got = np.concatenate([np.asarray(p.data.qpos, np.float64),
                                  [np.asarray(getattr(m, arr)).reshape(-1)[k] for arr, k in slots]])
            if np.abs(got - st).max() > 1e-9:
                raise AssertionError(
                    f"{cls.__name__}: the reference env did not reach the requested state "
                    f"(max {np.abs(got - st).max():.3e}); every parity number computed from it would "
                    f"be comparing two different scenes")
            out.append(p.render(height=size, width=size, camera_id=cls.CAMERA).copy())
        return np.stack(out)


# --------------------------------------------------------------------------- parity: plate + cast

class DMControlSuite(_DMCModel):
    """The fixed-camera DM Control domains: three baked plates and an analytic ray-cast.

    ``DOMAIN``/``TASK``/``CAMERA`` are set by the generated subclasses.
    """

    #: ONE ray per pixel, and that is a measurement rather than a saving. The reference is MSAA'd at
    #: MuJoCo's ``offsamples=4`` default, so a hard silhouette is the wrong answer at a mover's edge,
    #: and this was 2 for exactly that reason. What was never checked is whether the PARITY CLAIM
    #: needs it, and it does not:
    #:
    #: MSAA moves SILHOUETTE pixels. The strong assertion in
    #: ``test_render_parity_with_official_renderer`` is about INTERIOR pixels -- those whose 3x3
    #: neighbourhood in the official frame is one flat colour -- and averaging four samples inside a
    #: flat region returns the byte one sample returns. Measured over 201 seeded draws of 64 states on
    #: every cell of this family, the count of disagreeing interior pixels is IDENTICAL at ss=1 and
    #: ss=2; on ``DMCReacherEasy`` at 256 px ss=1 is strictly better (76 against 1,634). The two
    #: assertions supersampling does improve were never close to their bounds: the all-pixel floor is
    #: 0.90 and the worst cell here sits at 0.9103 at ss=1 (and 0.9053 at ss=2 -- ss=1 is the BETTER
    #: one on 11 of 15 cells, because an aliased edge is bit-identical to MuJoCo wherever MuJoCo's own
    #: MSAA happened to be fully covered, while a blended edge almost never is), and the mean ceiling
    #: is 4.0/255 against a worst of 1.085.
    #:
    #: WHAT THIS GIVES UP, stated because nothing asserts it: the worst single pixel roughly doubles
    #: (``max_abs`` 52 -> 134 on cartpole, 95 -> 179 on finger), and mean |diff| rises 1.5-2.3x while
    #: staying a quarter of its bound. Both are reported by ``pixel_agreement`` and neither is a
    #: threshold. In exchange the family renders 1.4-1.9x faster at 84 px and 2.8-3.4x at 256 px,
    #: which takes five of these cells over their own stored-frame throughput bar; none cleared it at
    #: ss=2. Every other conformance check -- batch-size invariance, chunked-render agreement, order
    #: independence, render purity, input-form invariance, determinism -- is BYTE-IDENTICAL at ss=1.
    #:
    #: ``supersample=2`` remains available per call and per instance for a caller who wants the
    #: smaller worst-pixel; it is the same kwarg it always was and it is not deprecated. See
    #: ``scratch/msaa/`` for the 615-draw survey, the throughput table and the conformance diff.
    default_supersample = 1
    #: Rays cast in one pass, so a large batch bounds its own memory. Every quantity here is
    #: (B, H*W*ss^2, 3) float32, and the arithmetic is per-ray, so chunking changes nothing about the
    #: answer -- which is asserted by ``test_frame_i_is_independent_of_batch_size`` rather than
    #: assumed. At 84 px and ss=2 this is ~580 environments a pass, at 256 px 64.
    #:
    #: Raised from 1<<23 once the casts were fused and the shadow and reflection passes were bounded,
    #: because what the chunk count costs is no longer memory, it is PYTHON: every chunk re-issues the
    #: whole render as a couple of hundred torch calls and synchronises twice to pick its screen
    #: windows, and these families run at 100% of one core. Swept at 256 px, batch 1024, f/s and peak
    #: device memory:
    #:
    #:     budget  envs/chunk  Cartpole    FingerTurn   BallInCup    Pusher       Reacher
    #:     1<<23   32          5,038/1.1G  2,330/2.5G   1,995/1.3G   4,156/1.8G   7,599/1.5G
    #:     1<<24   64          6,530/1.9G  2,502/4.8G   2,170/2.4G   4,839/3.3G   9,458/2.7G
    #:     1<<25   128         7,332/3.6G  2,500/9.4G   2,094/4.5G   5,256/6.2G  10,761/5.1G
    #:     1<<26   256         7,870/7.0G  2,419/18.5G  2,103/8.6G   5,165/12.2G 11,529/10.0G
    #:
    #: 1<<24 is where the trade turns: it is 1.09-1.31x over 1<<23 for half the memory of 1<<25, and
    #: the two envs that gain nothing past it (finger, ball_in_cup) are exactly the two whose mover
    #: window declines at 256 px, so their reflection pass runs over the FULL frame and a chunk holds
    #: a dozen (B, H*W*ss^2, 3) tensors at once. Going further buys 12% on three cells and doubles the
    #: memory on all of them.
    RAY_BUDGET = 1 << 24

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        sc = self._scene()
        assert sc.model.nq == self.nq, f"{type(self).__name__}: nq changed under us"

    @property
    def PLATE_PATHS(self) -> tuple[str, ...]:                          # noqa: N802
        """Style leaves that change the STATIC scene and therefore the three baked plates.

        EVERY leaf is one, and that is a property of the suite rather than laziness: DM Control
        shares a material between static and moving geoms (pendulum's ``decoration`` is on a moving
        base while point_mass's is on static walls), so there is no colour knob that can be applied
        to the movers alone. Declared here so the reproducibility suite can see, from the class, that
        this family re-bakes per appearance and cover it with one representative.
        """
        return tuple(p for p, _, _ in self.variation_space.flat_layout())

    # ---- scene bake -------------------------------------------------------

    def _plate_key(self, size, variation):
        return ((self.DOMAIN, self.TASK, int(self.CAMERA), int(size), str(self.device))
                + tuple(tuple(np.round(self._var(variation, p), 6)) for p in self.PLATE_PATHS))

    def _scene(self, size=None, variation=None) -> _Scene:
        size = int(size or self.size)
        key = self._plate_key(size, variation)
        hit = _SCENE_CACHE.get(key)
        if hit is not None:
            return hit
        if len(_SCENE_CACHE) >= _SCENE_CACHE_MAX:
            _SCENE_CACHE.clear()
        sc = self._bake(size, variation)
        _SCENE_CACHE[key] = sc
        return sc

    def _apply_variation(self, m, pr, variation):
        """Absolute assignment from the pristine snapshot, never an increment, so baking B after A
        gives the same model as baking B first. ``cam_fovy`` and the light scale are MULTIPLIERS and
        are exactly the two that would otherwise compound."""
        import mujoco
        m.mat_rgba[:] = pr["mat_rgba"]
        m.geom_rgba[:] = pr["geom_rgba"]
        m.site_rgba[:] = pr["site_rgba"]
        m.mat_reflectance[:] = pr["mat_reflectance"]
        # groups too, so a bake that raises part-way cannot leave the movers hidden for the next one
        m.geom_group[:] = pr["geom_group"]
        m.site_group[:] = pr["site_group"]
        # and the episode-drawn arrays, for the same reason one rung up: ``_fk`` writes a STATE ROW
        # into them, and a bake that ran after a render would otherwise be taken at whatever target
        # position the last environment of the last batch happened to hold.
        for arr in {a for a, _ in self._slots()}:
            getattr(m, arr)[:] = pr[arr]
        for name in self._used_materials():
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if i >= 0:
                m.mat_rgba[i, :3] = self._var(variation, f"material.{name}")
        m.cam_fovy[:] = pr["cam_fovy"] * float(self._var(variation, "camera.fovy_scale")[0])
        m.light_diffuse[:] = np.clip(
            pr["light_diffuse"] * float(self._var(variation, "light.diffuse_scale")[0]), 0, 1)
        m.light_specular[:] = pr["light_specular"]

    @staticmethod
    def _visible(m, kind: str) -> list[int]:
        """Drawn geoms or sites: in a visible GROUP and with a non-zero resolved ALPHA.

        Both conditions, because the two filters hide different things and either alone is silent.
        Group hides ``acrobot``'s ``target`` and ``finger``'s ellipsoids; alpha hides ``finger``'s
        ``target`` and ``tip`` sites, ``fish``'s ``target`` geom, and ``swimmer``'s six ``inertial``
        geoms and six sites — every one of them written by ``initialize_episode`` rather than by the
        XML, which is why ``_bundle`` runs the episodes before this is ever read.
        """
        grp = m.geom_group if kind == "geom" else m.site_group
        n = m.ngeom if kind == "geom" else m.nsite
        alpha = _effective_alpha(m, kind)
        return [i for i in range(n) if int(grp[i]) in VISIBLE_GROUPS and alpha[i] > 0]

    def _movers(self, m) -> tuple[list[int], list[int]]:
        """``(geom ids, site ids)`` whose pose depends on the STATE: off the world body, or drawn.

        Two sources, and the second is what lets ``reacher`` and ``finger-turn`` ship at all:

          * anything on a body other than the world's, whose pose is a function of ``qpos``. Sites are
            in here because they are DRAWN at the default ``MjvOption`` and several domains hang one
            off a moving body — hiding only the moving GEOMS left acrobot's plate varying by 28/255;
          * anything the EPISODE draws. ``reacher``'s target geom and ``finger-turn``'s target site
            sit on the WORLD body, so the first rule calls them static and they go into the plate,
            which is exactly why the plate was constant within an episode (0/255) and not across them
            (215/255 on reacher, 224/255 on finger-turn). Their position is in the state, so they are
            movers, and the plate is then 0/255 across episodes too.

        Treating them as movers rather than re-baking per draw is the difference between one bake per
        appearance and one bake per episode: a batch of 1024 states from 1024 episodes would otherwise
        cost 3072 MuJoCo renders.
        """
        ep_g, ep_s = set(), set()
        for arr, i in self._slots():
            kind, eid = _slot_element(m, arr, i)
            (ep_g if kind == "geom" else ep_s).add(eid)
        g = [i for i in self._visible(m, "geom")
             if int(m.geom_bodyid[i]) != 0 or i in ep_g]
        s = [i for i in self._visible(m, "site")
             if int(m.site_bodyid[i]) != 0 or i in ep_s]
        return g, s

    def _bake(self, size: int, variation) -> _Scene:
        import mujoco
        env, m, d, pr, r, _ = self._bundle(size)
        self._apply_variation(m, pr, variation)

        gmov, smov = self._movers(m)
        gstat = [i for i in self._visible(m, "geom") if i not in gmov]
        sstat = [i for i in self._visible(m, "site") if i not in smov]

        tendon = _find_tendon(m)

        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        opt = mujoco.MjvOption()                     # the default: exactly what dm_control renders at
        if tendon is not None:
            # The tendon is a MOVER and has to leave the plate with the rest of them, and it is the
            # one mover that cannot be hidden by a geom group because it is not a geom. Measured:
            # ball_in_cup's plate varies by 97/255 over 0.78% of the frame with every moving geom and
            # site hidden, and by 0/255 once mjVIS_TENDON goes off as well
            # (``scratch/dmcontrol/probe_widen.py``). That 97/255 is the whole of what
            # ``docs/research/dmcontrol.md`` S8 recorded as this domain's blocker.
            opt.flags[int(mujoco.mjtVisFlag.mjVIS_TENDON)] = 0

        def shot():
            r.update_scene(d, camera=self.CAMERA, scene_option=opt)
            return r.render().copy()

        m.geom_group[gmov] = HIDDEN_GROUP
        m.site_group[smov] = HIDDEN_GROUP
        plate = shot()
        r.update_scene(d, camera=self.CAMERA, scene_option=opt)
        cam = R3.Camera.from_mjv(r.scene.camera[0], r.scene.camera[1])
        lights = _Lights(r.scene)
        sl = lights.shadow_light()

        # plate 2: the shadow-casting light contributing only its ambient. This IS the colour of a
        # static surface the light does not reach, so nothing about MuJoCo's shadow MAP is modelled.
        if sl >= 0:
            head = 1 if int(m.vis.headlight.active) else 0
            mi = sl - head
            keep_d = np.array(m.light_diffuse, copy=True)
            keep_s = np.array(m.light_specular, copy=True)
            if 0 <= mi < m.nlight:
                m.light_diffuse[mi] = 0.0
                m.light_specular[mi] = 0.0
            plate_shadow = shot()
            m.light_diffuse[:] = keep_d
            m.light_specular[:] = keep_s
        else:
            plate_shadow = plate

        # plate 3: reflection removed, so MuJoCo's own compositing rule can be inverted
        keep_r = np.array(m.mat_reflectance, copy=True)
        m.mat_reflectance[:] = 0.0
        plate_norefl = shot()
        m.mat_reflectance[:] = keep_r

        m.geom_group[:] = pr["geom_group"]
        m.site_group[:] = pr["site_group"]

        sc = _Scene()
        sc.size = size
        sc.model, sc.data = m, d
        sc.cam, sc.lights, sc.shadow_light = cam, lights, sl
        sc.movers = (gmov, smov)
        sc.tendon = tendon
        sc.plate = torch.tensor(plate, device=self.device, dtype=torch.uint8)
        sc.plate_shadow = torch.tensor(plate_shadow, device=self.device, dtype=torch.uint8)
        sc.plate_norefl = torch.tensor(plate_norefl, device=self.device, dtype=torch.uint8)

        # the static depth is NOT cast here: it is needed at the supersampled grid, which depends on a
        # render-time argument, and `check_style_is_state` bakes one scene per sampled appearance --
        # so casting a full frame per bake would be paid 16 times for a buffer nothing had asked for.
        sprims, spos, smat = self._prims(m, d, gstat, sstat)
        sc.static_t = {}
        sc.shadow_candidates = {}
        sc.rays = {}
        sc.const = {}
        # the reflective plane, if any: MuJoCo only mirrors PLANES
        sc.reflectance, sc.floor = 0.0, None
        for i in gstat:
            mid = int(m.geom_matid[i])
            if int(m.geom_type[i]) == R3.PLANE and mid >= 0 and float(m.mat_reflectance[mid]) > 0:
                sc.reflectance = float(m.mat_reflectance[mid])
                sc.floor = (np.asarray(m.geom_size[i], np.float64),
                            np.asarray(d.geom_xpos[i], np.float64),
                            np.asarray(d.geom_xmat[i], np.float64).reshape(3, 3))
                break
        # a length scale for ray offsets, taken from the scene rather than assumed: cartpole is 8 m
        # across and point_mass 0.6, so one absolute epsilon cannot serve both
        sc.eps = float(max(1e-6, 1e-5 * np.abs(spos).max())) if len(spos) else 1e-6

        sc.prims, _, _ = self._prims(m, d, gmov, smov, tendon)
        return sc

    def _prims(self, m, d, geoms, sites, tendon: "_Tendon | None" = None):
        """``(Primitives, xpos (P,3), xmat (P,3,3))`` for a set of geoms AND sites, then the tendon.

        Sites carry their own type/size/rgba arrays, so they go through a parallel path rather than
        being pretended into geoms; the tendon's capsules go through a third, because they are not in
        the model at all — MuJoCo synthesises them per frame from the wrap points and the tendon's
        length. Their poses and half-lengths here are placeholders overwritten by ``_fk``; what is
        fixed at bake time is their COUNT, which is what lets the mover set stay a constant.
        """
        kind, size, rgb, tex, spec, shin, pos, mat = [], [], [], [], [], [], [], []
        for i in geoms:
            mid = int(m.geom_matid[i])
            s, sh = _mat_spec_shin(m, mid)
            kind.append(int(m.geom_type[i]))
            size.append(np.asarray(m.geom_size[i], np.float64))
            rgb.append(_mat_rgb(m, mid, m.geom_rgba[i]))
            tex.append(_tex_rgb(m, mid))
            spec.append(s)
            shin.append(sh)
            pos.append(np.asarray(d.geom_xpos[i], np.float64))
            mat.append(np.asarray(d.geom_xmat[i], np.float64).reshape(3, 3))
        for i in sites:
            mid = int(m.site_matid[i])
            s, sh = _mat_spec_shin(m, mid)
            kind.append(int(m.site_type[i]))
            size.append(np.asarray(m.site_size[i], np.float64))
            rgb.append(_mat_rgb(m, mid, m.site_rgba[i]))
            tex.append(_tex_rgb(m, mid))
            spec.append(s)
            shin.append(sh)
            pos.append(np.asarray(d.site_xpos[i], np.float64))
            mat.append(np.asarray(d.site_xmat[i], np.float64).reshape(3, 3))
        if tendon is not None:
            for _ in range(tendon.nseg):
                kind.append(R3.CAPSULE)
                size.append(np.array([tendon.radius, 0.0, 0.0]))
                rgb.append(tendon.grey.copy())
                tex.append(np.ones(3))
                # matid -1 on this tendon, so MuJoCo's material defaults apply, the same ones
                # ``_mat_spec_shin`` returns for an unmaterialled geom
                spec.append(0.5)
                shin.append(0.5)
                pos.append(np.zeros(3))
                mat.append(np.eye(3))
        if not kind:
            prims = R3.Primitives(kind=np.zeros(0, int), size=np.zeros((0, 3)),
                                  mat_rgb=np.zeros((0, 3)), tex_rgb=np.zeros((0, 3)),
                                  spec=np.zeros(0))
            prims.shininess = np.zeros(0)
            return prims, np.zeros((0, 3)), np.zeros((0, 3, 3))
        prims = R3.Primitives(kind=np.array(kind), size=np.stack(size), mat_rgb=np.stack(rgb),
                              tex_rgb=np.stack(tex), spec=np.array(spec))
        prims.shininess = np.array(shin)
        return prims, np.stack(pos), np.stack(mat)

    def _shadow_candidates(self, sc: _Scene, ss: int):
        """``(idx, plate_shadow_rows)``: the STATIC samples a mover's shadow could change, cached.

        Exact rather than approximate, and that is the whole reason it is allowed. A shadowed sample
        selects ``plate_shadow`` instead of ``plate``, so wherever the two plates carry the SAME
        colour the occlusion test cannot change the frame whatever it returns; and a ray that hits no
        static surface has nothing to shadow, which the ``st < INF`` term in the composite already
        said. Restricting the SET the test runs on therefore cannot change a VALUE — every ray's
        answer is computed from that ray alone.

        This is ``mujoco_locomotion``'s ``_shadow_candidates`` verbatim in intent; DM Control simply
        never had it, and its shadow pass was still starting from every sample in the frame.
        """
        key = (int(ss), bool(self.SHADOW_COMPACTION))
        hit = sc.shadow_candidates.get(key)
        if hit is not None:
            return hit
        a = _upsample_rgb(sc.plate, ss).reshape(-1, 3).to(torch.int16)
        b = _upsample_rgb(sc.plate_shadow, ss).reshape(-1, 3).to(torch.int16)
        if self.SHADOW_COMPACTION:
            cand = (a != b).any(-1) & (self._static_t(sc, ss).view(-1) < R3.INF)
            idx = torch.nonzero(cand, as_tuple=False).reshape(-1)
        else:
            idx = torch.arange(a.shape[0], device=a.device)     # the pass this compaction replaced
        rows = _upsample_rgb(sc.plate_shadow, ss).reshape(-1, 3).to(torch.float32)[idx]
        sc.shadow_candidates[key] = hit = (idx, rows.unsqueeze(0))
        return hit

    def _static_shadow_rays(self, sc: _Scene, ss: int):
        """``(idx, dark, o, d, lim, valid)`` for the shadow test on static surfaces, cached.

        THE SHADOW RAY IS A FUNCTION OF THE SCENE, NOT OF THE STATE. It starts at ``o + t*d`` for the
        STATIC ``t``, points at a light that does not move, and ends at that light — so its origin,
        direction and length are the same on every frame, and the only thing a state changes is
        whether a mover happens to be in the way. Recomputing them per chunk cost a dozen elementwise
        passes over a (B, samples, 3) tensor to rebuild numbers that were already known; what is left
        is the occlusion query itself, which is one fused kernel.
        """
        key = ("shadow", int(ss), bool(self.SHADOW_COMPACTION))
        hit = sc.const.get(key)
        if hit is not None:
            return hit
        cand, dark = self._shadow_candidates(sc, ss)
        o, d = self._per_chunk_constants(sc, ss)[:2]
        N = (sc.size * int(ss)) ** 2
        st = self._static_t(sc, ss).view(1, N)
        p = o.expand(1, N, 3)[:, cand] + st[:, cand].clamp(max=1e6).unsqueeze(-1) \
            * d.view(1, N, 3)[:, cand]
        L = sc.lights.to(self.device)
        sl = sc.shadow_light
        shape = (1,) * (p.dim() - 1) + (3,)
        if L["kind"][sl] == LIGHT_DIRECTIONAL:
            rd = (-_norm(L["dir"][sl].view(shape))).expand_as(p)
            lim = torch.full(p.shape[:-1], R3.INF, device=p.device)
        else:
            v = L["pos"][sl].view(shape) - p
            lim = torch.sqrt(torch.clamp(_dot(v, v), min=1e-24))
            rd = v / lim.unsqueeze(-1)
        hit = sc.const[key] = (cand, dark, p + rd * sc.eps, rd, lim, st[:, cand] < R3.INF)
        return hit

    def _floor_rays(self, sc: _Scene, ss: int):
        """``(idx, o, d, surf)``: the REFLECTED ray at every sample that shows the mirror. Cached.

        The same observation as ``_static_shadow_rays`` and a bigger payoff, because this prologue was
        the largest single term left in a DM Control chunk (37-53%, more than every ray cast put
        together). Which samples are the mirror, where the eye ray meets it, and which way it bounces
        are all fixed by the camera and the floor; only what the bounced ray HITS depends on the
        state. So the plane intersection, the ``|t_f - st|`` test, the hit point and the reflection
        are computed once per (scene, supersample) and the per-frame cost is one fused cast.

        This also retires the mirror WINDOW. A rectangle around the movers' reflected bound measured
        44-64% of the frame at 256 px — the reflections of a spread-out body are spread out — so it
        was bounding the pass by less than the floor mask does, at the cost of a projection and a
        synchronisation per chunk.
        """
        key = ("floor", int(ss))
        hit = sc.const.get(key)
        if hit is not None:
            return hit
        o, d = self._per_chunk_constants(sc, ss)[:2]
        N = (sc.size * int(ss)) ** 2
        st = self._static_t(sc, ss).view(1, N)
        oo, dd = o.expand(1, N, 3), d.view(1, N, 3)
        fsize, fpos, fmat = self._floor_tensors(sc)
        # WHICH samples are the mirror is answered by re-casting the plane and comparing ray
        # parameters, not by testing whether the hit point lies in the plane: at cartpole's 8 m scale
        # a float32 hit point carries ~1e-6 of slack and any absolute tolerance either misses the
        # floor or claims the rails as well.
        t_f, n = R3.intersect(R3.PLANE, fsize.view(1, 1, 3), fpos.view(1, 1, 3),
                              fmat.view(1, 1, 3, 3), oo, dd)
        on_plane = (t_f < R3.INF) & (st < R3.INF) & (torch.abs(t_f - st) <= 1e-4 * (1.0 + st))
        idx = torch.nonzero(on_plane.reshape(-1), as_tuple=False).reshape(-1)
        rd = dd - 2 * _dot(dd, n.expand_as(dd)).unsqueeze(-1) * n
        ro = oo + st.clamp(max=1e6).unsqueeze(-1) * dd + rd * sc.eps
        surf = sc.const.get(("norefl", int(ss)))
        if surf is None:
            surf = sc.const[("norefl", int(ss))] = \
                _upsample_rgb(sc.plate_norefl, ss).reshape(1, N, 3).to(torch.float32)
        hit = sc.const[key] = (idx, ro[:, idx].contiguous(), rd[:, idx].contiguous(),
                               surf[:, idx].contiguous())
        return hit

    def _static_t(self, sc: _Scene, ss: int) -> torch.Tensor:
        """The static ray parameter at the SUPERSAMPLED grid, cached per (scene, ss).

        Cast rather than upsampled from the output-resolution buffer. The shadow test starts from
        ``o + t*d`` at every sample, so a nearest-neighbour ``t`` would put the sub-samples around a
        silhouette on the wrong surface and scatter shadow pixels along every static edge.
        """
        hit = sc.static_t.get(int(ss))
        if hit is not None:
            return hit
        m, d = sc.model, sc.data
        gmov, smov = sc.movers
        gstat = [i for i in self._visible(m, "geom") if i not in gmov]
        sstat = [i for i in self._visible(m, "site") if i not in smov]
        import mujoco
        d.qpos[:] = m.qpos0
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        sprims, spos, smat = self._prims(m, d, gstat, sstat)
        R = sc.size * int(ss)
        t = R3.static_depth(sc.cam, sprims, spos, smat, R, R, self.device)
        sc.static_t[int(ss)] = t
        return t

    # ---- forward kinematics ----------------------------------------------

    def _fk(self, states: torch.Tensor, sc: _Scene):
        """``(B, state_dim)`` -> per-row mover poses, sizes and colours.

        ``(xpos (B,P,3), xmat (B,P,3,3), sizes (B,P,3), rgb (B,P,3))``. Three of the four are constant
        across the batch on every domain but ``ball_in_cup``, and they are still returned per row
        because the tendon's capsules change LENGTH and COLOUR with the state and there is no useful
        way to say that only sometimes.

        A pure function of the state, a couple of microseconds a call, and a Python loop over the batch
        — which reads like the bottleneck and measurably is not: 17 ms of cartpole's 232 ms at
        B=1024. What costs the rest is that three of the ray passes are FULL-FRAME. The movers can be
        tiled (they are 0.2-3.0% of the frame), but the shadow test starts from every STATIC pixel and
        the reflection test from every floor pixel, so those two scale with the image rather than with
        the movers. Bounding them to the movers' light-space and screen-space footprints is the next
        thing worth doing here; a batched GPU forward kinematics is not.

        A scene with a tendon runs ``mj_forward`` rather than ``mj_kinematics``, and pays for it: the
        colour of a taut tendon is its limit constraint's impedance, which only exists once the
        constraint set has been built. ``ball_in_cup`` is 4 dof and 6 geoms, so the extra work is
        microseconds; a domain where it were not could not use this path.
        """
        import mujoco
        gmov, smov = sc.movers
        td = sc.tendon
        nmov = len(gmov) + len(smov)
        P = nmov + (td.nseg if td is not None else 0)
        B = states.shape[0]
        st = states.detach().cpu().numpy().astype(np.float64)
        q, ep = st[:, :self.nq], st[:, self.nq:]
        xp = np.empty((B, P, 3))
        xm = np.empty((B, P, 3, 3))
        size = np.tile(np.asarray(sc.prims.size, np.float64), (B, 1, 1))
        rgb = np.tile(np.asarray(sc.prims.mat_rgb, np.float64), (B, 1, 1))
        m, d = sc.model, sc.data
        slots = self._slots()
        pr = self._pristine()
        try:
            for i in range(B):
                # The episode's draw goes into the MODEL and then through the same kinematics as the
                # joints, rather than being added to the mover's pose afterwards. That is not tidiness:
                # ``geom_pos`` is expressed in the parent BODY's frame, so a mover the episode draws on
                # a rotated body would land in the wrong place under any shortcut, and running it
                # through ``mj_kinematics`` is what makes the widened state mean exactly what it means
                # to ``dm_control``.
                for (arr, k), v in zip(slots, ep[i]):
                    getattr(m, arr).reshape(-1)[k] = v
                d.qpos[:] = q[i]
                if td is None:
                    mujoco.mj_kinematics(m, d)
                else:
                    d.qvel[:] = 0
                    mujoco.mj_forward(m, d)
                if gmov:
                    xp[i, :len(gmov)] = d.geom_xpos[gmov]
                    xm[i, :len(gmov)] = d.geom_xmat[gmov].reshape(-1, 3, 3)
                if smov:
                    xp[i, len(gmov):nmov] = d.site_xpos[smov]
                    xm[i, len(gmov):nmov] = d.site_xmat[smov].reshape(-1, 3, 3)
                if td is not None:
                    tp, tm, half, trgb = _tendon_frame(m, d, td)
                    xp[i, nmov:] = tp
                    xm[i, nmov:] = tm
                    size[i, nmov:, 1] = half
                    rgb[i, nmov:] = trgb
        finally:
            for arr in {a for a, _ in slots}:                 # never leave a state row in the model
                getattr(m, arr)[:] = pr[arr]
        t = lambda x: torch.tensor(x, device=self.device, dtype=torch.float32)   # noqa: E731
        return t(xp), t(xm), t(size), t(rgb)

    # ---- ray-cast helpers -------------------------------------------------

    @staticmethod
    def _cast(sc, o, d, xpos, xmat, sizes):
        """Nearest mover hit over a ray bundle: ``(t, normal, primitive index)``."""
        # One call, primitives batched and rays tiled. The Python loop this replaces cost 79% of a
        # BallInCupCatch render, whose tendon is 28 capsules re-tested by every pass.
        return R3.cast_nearest(sc.prims.kind, sizes, xpos, xmat, o, d)

    @staticmethod
    def _occluded(sc, p, L, sl, xpos, xmat, sizes, skip=None, eps: float = 1e-6):
        """Does any mover block the segment from ``p`` to light ``sl``? ``(B,N)`` bool.

        ``skip`` is the primitive each ray already sits on, excluded so a convex mover does not
        shadow its own front face. That is exact rather than an epsilon: a ray leaving a convex
        surface toward the light re-enters the same primitive only where ``n.l <= 0``, which the
        diffuse term already zeroes.
        """
        shape = (1,) * (p.dim() - 1) + (3,)
        if L["kind"][sl] == LIGHT_DIRECTIONAL:
            d = (-_norm(L["dir"][sl].view(shape))).expand_as(p)
            lim = torch.full(p.shape[:-1], R3.INF, device=p.device)
        else:
            v = L["pos"][sl].view(shape) - p
            lim = torch.sqrt(torch.clamp(_dot(v, v), min=1e-24))
            d = v / lim.unsqueeze(-1)
        o = p + d * eps
        return R3.any_blocking(sc.prims.kind, sizes, xpos, xmat, o, d, lim, skip=skip)

    # ---- render -----------------------------------------------------------

    @torch.no_grad()
    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        """``(...,state_dim)`` -> ``(...,H,W,3)`` uint8. Style may differ per environment.

        The batch is grouped by style, because every leaf changes the baked plates. Environments that
        share an appearance share one bake and are rendered in one call.
        """
        s = self._state if states is None else torch.as_tensor(states, device=self.device,
                                                               dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        B = s.shape[0]
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        sty = self.resolve_style(style, variation, batch=B)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)

        key = torch.cat([sty[p].reshape(B, -1) for p in self.PLATE_PATHS], 1).cpu().numpy()
        uniq, inv = np.unique(np.round(key.astype(np.float64), 6), axis=0, return_inverse=True)
        offs, o = [], 0
        for p in self.PLATE_PATHS:
            w = int(sty.layout[p][1])
            offs.append((o, w))
            o += w
        for gi in range(len(uniq)):
            sub = np.flatnonzero(inv == gi)
            over = {p: uniq[gi][a:a + w] for p, (a, w) in zip(self.PLATE_PATHS, offs)}
            sc = self._scene(size, over)
            idx = torch.as_tensor(sub, device=self.device, dtype=torch.long)
            step = max(1, int(self.RAY_BUDGET // max(1, size * size * ss * ss)))
            for a in range(0, len(sub), step):
                part = idx[a:a + step]
                buf[part] = self._render_group(sc, s[part], size, ss)
        return buf.view(*lead, size, size, 3) if lead else buf

    #: Screen windows tried, as a divisor of the output edge. Smallest first: a window that fits is
    #: always better than a larger one, and one that does not fit costs a projection and nothing else.
    #:
    #: The ladder runs past 2 on purpose. ``ball_in_cup``'s movers span 44x50 of an 84px frame -- 31% of
    #: it, and 1.03% actually covered -- so a half-frame window misses by eight pixels in y and the
    #: environment fell all the way back to full-frame casting. Measured with
    #: ``scratch/census/bbox_probe.py``, which also confirms the per-primitive bound is correct: 0 of 36
    #: primitives draw outside the box projected for them.
    MOVER_TILES = (8, 6, 4, 3, 2, 1.5, 1.25)

    #: Restrict the shadow-from-static pass to the samples where the two plates disagree. A class flag
    #: so ``tests/gpu/test_fast_paths.py`` can turn it off and compare against the pass it replaced;
    #: there is no reason for a caller to.
    SHADOW_COMPACTION = True
    #: Shade only the rays that landed on a mover / in the mirror, as the union over the chunk, when
    #: no screen window fits. Same reason for the flag: with it off the two paths must agree.
    COVER_COMPACTION = True

    def _mover_idx(self, sc: _Scene, xpos: torch.Tensor, size: int, ss: int):
        """Flat ray indices covering every pixel the movers can touch, or None for the full frame.

        The bound is ``screen_span``'s: the world-space AABB of each primitive's bounding sphere, all
        eight corners projected, padded a pixel each way. It is conservative on purpose -- a window that
        clipped a mover would drop pixels silently, and a faster renderer that can disagree with the
        slower one is worth nothing here.

        The span is projected ONCE and the ladder is then walked on the host. Asking for each rung
        separately re-projected the corners and synchronised per rung, which cost more than the ray
        passes the window exists to shrink.
        """
        span = R3.screen_span(sc.cam, sc.prims, xpos, size, size, xpos.device)
        if span is None:
            return None
        x0, y0, w, h = span
        for div in self.MOVER_TILES:
            tile = max(1, int(size / div))
            if tile >= size or w > tile or h > tile:
                continue
            org = (x0.clamp(0, size - tile), y0.clamp(0, size - tile))
            return R3._expand_tile(org, tile, ss, size * ss, xpos.device)
        return None

    def _per_chunk_constants(self, sc: _Scene, ss: int):
        """``(o, d, plate_f, tex, spec, shin, L)``: everything a chunk needs that no state changes.

        Cached per (scene, ss), because the batch is rendered in ``RAY_BUDGET``-sized chunks and every
        one of these was being rebuilt per chunk — at 256 px and ss=2 that is 32 rebuilds of a
        512x512 ray grid per 1024 frames. ``Camera.rays`` builds that grid in NUMPY, in float64, as
        half a dozen (512,512,3) temporaries, and then copies it to the device; it is a pure function
        of (camera, resolution) and it was the single largest term left in a DM Control render once
        the casts were fused. ``mujoco_locomotion`` has cached its rays all along and says so in
        ``_rays``; this family simply never did.
        """
        hit = sc.const.get(int(ss))
        if hit is None:
            dev = self.device
            R = sc.size * int(ss)
            o, d = sc.cam.rays(R, R, dev)
            hit = sc.const[int(ss)] = (
                o, d,
                _upsample_rgb(sc.plate, ss).reshape(1, -1, 3).to(torch.float32),
                torch.tensor(sc.prims.tex_rgb, device=dev, dtype=torch.float32),
                torch.tensor(sc.prims.spec, device=dev, dtype=torch.float32),
                torch.tensor(sc.prims.shininess, device=dev, dtype=torch.float32),
                sc.lights.to(dev))
        return hit

    def _render_group(self, sc: _Scene, s: torch.Tensor, size: int, ss: int) -> torch.Tensor:
        dev = self.device
        B = s.shape[0]
        R = size * ss
        N = R * R
        o, d, plate_f, tex_rgb, spec, shin, L = self._per_chunk_constants(sc, ss)
        oo = o.expand(B, N, 3)
        dd = d.view(1, N, 3).expand(B, N, 3)

        xpos, xmat, sizes, mat_rgb = self._fk(s, sc)
        P = xpos.shape[1]
        sl = sc.shadow_light
        st = self._static_t(sc, ss).view(1, N).expand(B, N)
        # ``mat_rgb`` is PER ROW, so the gather needs the batch index alongside the primitive index.
        # Only ball_in_cup's tendon actually varies with the state -- it is grey while the string
        # hangs and red while it is taut -- but a colour that is a tensor either way costs one gather
        # and removes the branch.
        bi = torch.arange(B, device=dev).unsqueeze(1)

        # The movers cover 0.2-3.3% of the frame (measured: 1.03% on ball_in_cup, 1.51% on cartpole,
        # 3.29% on reacher), so casting them against every ray spends ~99% of the pass on pixels they
        # cannot reach. Restrict the two mover passes to a conservative screen window; outside it
        # ``t_m`` is INF and ``cover`` is False, so the composite below is unchanged BY CONSTRUCTION
        # rather than by tolerance. ``_mover_idx`` returns None when no window fits and the full-frame
        # path runs, which is what keeps the two paths substitutable.
        widx = self._mover_idx(sc, xpos, size, ss)
        base = plate_f.expand(B, N, 3).clone()
        if widx is None:
            oo_w, dd_w = oo, dd
            t_w, n_w, i_w = self._cast(sc, oo_w, dd_w, xpos, xmat, sizes)
            cover_w = t_w < st
        else:
            w3 = widx.unsqueeze(-1).expand(-1, -1, 3)
            oo_w, dd_w = oo.gather(1, w3), dd.gather(1, w3)
            t_w, n_w, i_w = self._cast(sc, oo_w, dd_w, xpos, xmat, sizes)
            cover_w = t_w < st.gather(1, widx)

        # A WINDOW IS NOT A COVERAGE MASK, and the difference is most of the cost. The window is a
        # rectangle around the movers' projected bound; the movers cover 1.6-8.4% of the frame but
        # their rectangle is 25-100% of it, because a reacher arm or a finger spinner is long and
        # thin and diagonal. The cast is one fused kernel and does not care, but everything after it
        # -- the shadow ray and ~75 elementwise ops of shading -- is paid per ray. So narrow the
        # window down to the rays that ACTUALLY landed on a mover, as the union over the chunk.
        #
        # Restricting the SET cannot change a VALUE: every ray's colour is a function of that ray
        # alone, and outside the union ``cover`` is False for EVERY row, so the composite below takes
        # the plate there whatever the shader would have said. Note the union is over window
        # POSITIONS, not pixels -- position j is a different pixel in each row — which is exactly as
        # sound, for the same reason.
        if self.COVER_COMPACTION:
            js = torch.nonzero(cover_w.any(0), as_tuple=False).reshape(-1)
            widx = js.view(1, -1).expand(B, -1) if widx is None else widx[:, js]
            cover_w, t_w, n_w, i_w = cover_w[:, js], t_w[:, js], n_w[:, js], i_w[:, js]
            oo_w, dd_w = oo_w[:, js], dd_w[:, js]
        if widx is not None:
            w3 = widx.unsqueeze(-1).expand(-1, -1, 3)
        pm_w = oo_w + t_w.clamp(max=1e6).unsqueeze(-1) * dd_w
        unlit_w = None
        if sl >= 0 and P > 1:
            unlit_w = self._occluded(sc, pm_w, L, sl, xpos, xmat, sizes, skip=i_w, eps=sc.eps)
        col_w = _shade(pm_w, n_w, -dd_w, L, mat_rgb[bi, i_w], spec[i_w], shin[i_w], unlit_w) \
            * tex_rgb[i_w] * 255.0
        if sl >= 0:
            cand, dark = self._shadow_candidates(sc, ss)
            if cand.numel():
                stc = st[:, cand]
                ps = oo[:, cand] + stc.clamp(max=1e6).unsqueeze(-1) * dd[:, cand]
                occ = self._occluded(sc, ps, L, sl, xpos, xmat, sizes, eps=sc.eps) \
                    & (stc < R3.INF)
                base[:, cand] = torch.where(occ.unsqueeze(-1), dark, base[:, cand])
        if sc.reflectance > 0:
            base = self._reflect(sc, base, xpos, xmat, sizes, mat_rgb, tex_rgb, spec,
                                 shin, L, ss, bi)

        # The movers go on LAST, over the shadowed and mirrored plate, and they go straight into it:
        # ``where(cover, col, base)`` is the identity wherever ``cover`` is False, and outside the
        # index set — a screen window or the union of covered rays — it provably is. So the composite
        # is a scatter of the set, without a second full-size canvas and a full-frame select.
        if widx is None:                       # nothing was restricted: no index to scatter through
            frame = torch.where(cover_w.unsqueeze(-1), col_w, base)
        else:
            frame = base.scatter_(1, w3, torch.where(cover_w.unsqueeze(-1), col_w,
                                                     base.gather(1, w3)))
        frame = frame.view(B, size, ss, size, ss, 3).sum((2, 4)) / (ss * ss)
        return torch.clamp(torch.round(frame), 0, 255).to(torch.uint8)

    def _floor_tensors(self, sc: _Scene):
        """``(size, pos, mat)`` of the reflective plane, as device tensors, built once per scene.

        Three ``torch.tensor(..., device=cuda)`` calls per chunk otherwise, and a pageable
        host-to-device copy blocks until the stream drains — see the note in ``raycast3d``.
        """
        hit = sc.const.get("floor")
        if hit is None:
            fsize, fpos, fmat = sc.floor
            t = lambda x: torch.tensor(x, device=self.device, dtype=torch.float32)   # noqa: E731
            hit = sc.const["floor"] = (t(fsize), t(fpos), t(fmat))
        return hit

    def _reflect(self, sc, base, xpos, xmat, sizes, mat_rgb, tex_rgb, spec, shin, L, ss, bi):
        """Put the movers into the reflective floor, using MuJoCo's own compositing rule.

        ``frame = reflectance * mirror + (1 - reflectance) * surface``. The un-reflected plate IS
        ``surface``, so a mover that appears in the mirror replaces the mirror term outright instead
        of being blended into a number that would have to be guessed. The alternative — scaling the
        mover's colour into the existing plate — cannot be right, because the plate already holds
        whatever the mirror showed at that pixel (skybox, a rail, the far floor) and there is no way
        to subtract it back out.

        Everything geometric is precomputed by ``_floor_rays``; what is left per frame is one fused
        cast down the reflected rays and a shade of the samples that came back with a hit.
        """
        idx, ro, rd, surf = self._floor_rays(sc, ss)
        if idx.numel() == 0:
            return base
        B = base.shape[0]
        t_r, n_r, i_r = self._cast(sc, ro.expand(B, -1, 3), rd.expand(B, -1, 3), xpos, xmat, sizes)
        seen = t_r < R3.INF
        widx = idx.view(1, -1).expand(B, -1)
        # Narrow to the samples where a mover is ACTUALLY in the mirror. The floor is a third to a
        # half of the frame and the reflection reaches a few percent of it; ``seen`` is False for
        # every row outside the union, so the select below is the identity there and the shade —
        # ~75 elementwise ops — never runs on it.
        if self.COVER_COMPACTION:
            js = torch.nonzero(seen.any(0), as_tuple=False).reshape(-1)
            widx, seen, ro, rd = widx[:, js], seen[:, js], ro[:, js], rd[:, js]
            surf, t_r, n_r, i_r = surf[:, js], t_r[:, js], n_r[:, js], i_r[:, js]
        pr = ro + t_r.clamp(max=1e6).unsqueeze(-1) * rd
        colr = _shade(pr, n_r, -rd, L, mat_rgb[bi, i_r], spec[i_r], shin[i_r], None) \
            * tex_rgb[i_r] * 255.0
        r = float(sc.reflectance)
        val = r * colr + (1.0 - r) * surf
        w3 = widx.unsqueeze(-1).expand(-1, -1, 3)
        # in place: ``base`` is this render's own float canvas, and the out-of-place ``scatter``
        # would copy the whole frame to change a few percent of it
        return base.scatter_(1, w3, torch.where(seen.unsqueeze(-1), val, base.gather(1, w3)))


# ---------------------------------------------------------------- MJWarp TrackCom backend
# RESTORED 2026-08-11 after a scripted line-range splice in this file overshot its end boundary
# and removed this block (`_TCtx`, `_tc_context`, `_tc_frame` and the `DMControlTrackCom` header),
# orphaning the class's methods into `DMControlSuite` and taking the whole module's import down —
# and with it all 16 DM Control cells, not just the 5 this backend serves. The file is untracked,
# so there was no git object to recover from and both __pycache__ entries had already been
# regenerated from the broken source; the text below was recovered verbatim from a transcript that
# had read the file earlier, and verified by import, by the MJWarp round-trip and by the visual
# tier's own declared numbers rather than by inspection.

class _TCtx:
    """One CUDA-graph-captured MJWarp render context, plus the device views a render writes."""

    __slots__ = ("wm", "wd", "rc", "bgra", "graph", "nworld", "res", "wp", "mjw")


def _tc_context(cls, res: int, nworld: int) -> _TCtx:
    """One captured MJWarp context per (domain, task, resolution, worlds).

    Two things here are not optional and both are silent when omitted.

    **``com_pos`` between ``kinematics`` and ``camlight``.** These domains' camera is
    ``mjCAMLIGHT_TRACKCOM`` — it rides the body's centre of mass, which is the whole reason the plate
    design cannot serve them — and ``mujoco_warp.kinematics`` writes ``xpos``/``xquat``/``xmat``/
    ``xipos`` and NOT ``subtree_com``. Without the extra launch the camera never leaves the pose
    ``put_data`` uploaded and every state is drawn from it, with no error and a plausible-looking
    picture. ``MW.needs_com_pos`` answers the question from the model and is asserted below rather
    than assumed, because the failure has no other symptom.
    ``docs/research/mujoco_locomotion.md`` S6.2 measures the two frame loops: camera spread 0.0
    (frozen) against MuJoCo's own, and 0.0 distance FROM MuJoCo's once ``com_pos`` runs.

    **Graph capture.** Un-captured these five kernels run at roughly 1.4x the official renderer rather
    than 30x, because at this scene size the frame is launch-bound. A throughput number near 1.4x
    means the capture did not happen, not that the GPU is slow.
    """
    import mujoco
    import warp as wp

    import mujoco_warp as mjw

    key = (cls.DOMAIN, cls.TASK, int(res), int(nworld), tuple(sorted(cls.RENDER_FLAGS.items())))
    hit = _TC_CACHE.get(key)
    if hit is not None:
        return hit
    if len(_TC_CACHE) >= _TC_CACHE_MAX:
        _TC_CACHE.clear()
    _, m, _, pr, _, slots = cls._bundle(cls.default_size)
    for arr in {a for a, _ in slots}:                  # a canonical model, so the context is a
        getattr(m, arr)[:] = pr[arr]                   # function of the domain and nothing else
    m.mat_rgba[:] = pr["mat_rgba"]
    m.light_diffuse[:] = pr["light_diffuse"]
    if not MW.needs_com_pos(m):
        raise RuntimeError(
            f"{cls.__name__}: no camera or light on this model tracks subtree_com, so it does not "
            f"belong in the trackcom family; the plate design in DMControlSuite is strictly better "
            f"for it")
    m.vis.global_.offwidth = max(int(m.vis.global_.offwidth), int(res))
    m.vis.global_.offheight = max(int(m.vis.global_.offheight), int(res))

    # The zero-alpha pre-pass. MJWarp's shader never reads alpha, so ``swimmer``'s six ``inertial``
    # geoms and six sites and ``fish``'s target and torso site would be painted OPAQUE over geometry
    # the reference does not draw at all. Left applied for the life of the context: MJWarp reads the
    # geometry once, at create_render_context, so restoring would not un-draw anything.
    MW.prepare_model(m, VISIBLE_GROUPS)

    c = _TCtx()
    c.wp, c.mjw, c.nworld, c.res = wp, mjw, int(nworld), int(res)
    d0 = mujoco.MjData(m)
    d0.qpos[:] = m.qpos0
    mujoco.mj_forward(m, d0)
    batched = sorted({"mat_rgba", "light_diffuse"} | {a for a, _ in slots})
    c.wm = mjw.put_model(m, batch_sizes={k: int(nworld) for k in batched})
    c.wd = mjw.put_data(m, d0, nworld=int(nworld))
    # The rounding patch has to be visible while the megakernel's source is GENERATED (context
    # creation) and while it is COMPILED (first launch), which is why the warm-up frame is inside the
    # block. MJWarp packs float->uint8 with C's ``int()``, which truncates, and a GL pipeline rounds;
    # on Fetch that one rule was worth interior parity 0.791 -> 0.9988.
    with MW.gl_rounding():
        c.rc = MW.create_render_context(
            m, nworld=int(nworld), cam_res=(int(res), int(res)), render_rgb=True,
            cam_active=[i == int(cls.CAMERA) for i in range(m.ncam)],
            enabled_geom_groups=list(VISIBLE_GROUPS), use_textures=True, use_shadows=True,
            render_skybox=True, **dict(cls.RENDER_FLAGS))
        _tc_frame(c)
        wp.synchronize()
    import torch
    c.bgra = wp.to_torch(c.rc.rgb_data).view(torch.uint8).reshape(
        int(nworld), int(res), int(res), 4)
    with wp.ScopedCapture() as cap:
        _tc_frame(c)
    c.graph = cap.graph
    _TC_CACHE[key] = c
    return c


def _tc_frame(c: _TCtx) -> None:
    c.mjw.kinematics(c.wm, c.wd)
    c.mjw.com_pos(c.wm, c.wd)          # see _tc_context: the TRACKCOM camera reads subtree_com
    c.mjw.camlight(c.wm, c.wd)
    c.mjw.refit_bvh(c.wm, c.wd, c.rc)
    c.mjw.render(c.wm, c.wd, c.rc)



class DMControlTrackCom(_DMCModel):
    """DM Control's body-tracking domains, drawn by MJWarp. VISUAL TIER — not the official pixels.

    These six domains put camera 0 in ``mode="trackcom"``: it rides the body's centre of mass, so the
    "static" half of the scene moves with the state and there is no plate to bake — hide every mover
    and the remaining frame still changes by 51-220/255 between states. ``DMControlSuite``'s design
    is therefore structurally unavailable here, and the alternative is a backend that draws the whole
    scene every frame. MJWarp is that backend, and its ``mjCAMLIGHT_TRACKCOM`` camera lands on
    MuJoCo's ``cam_xpos`` to 1e-8 metres, so the camera is not what the residual is.

    WHAT THE RESIDUAL IS
    --------------------

    Two named mechanisms, measured per surface off MuJoCo's own segmentation buffer (84 px, 32 states
    per domain, ``scratch/dmcontrol/probe_trackcom.py``):

      * **the speckled gradient skybox**, which is 48-89% of the frame on cheetah, hopper and walker.
        DM Control's ``skybox`` is an 800x4800 ``builtin="gradient"`` texture with ``mark="random"``,
        i.e. white specks scattered by MuJoCo's own RNG, and MJWarp samples it with a single
        unfiltered tap. Mean 4.1-4.7/255 with 89-92% of those pixels within 8, and it is what holds
        SSIM down — a speck one pixel out of place costs structure twice.
      * **the minified checker floor**, the same mechanism that decides
        ``docs/research/mujoco_locomotion.md``: in the far band MuJoCo's mipmap has averaged the
        checker into one grey (luminance spread 0.1 on humanoid) and MJWarp point-samples it at full
        contrast (9.7). Floor mean 4.3-8.2/255.

    Neither is closable here. The floor needs level-of-detail selection in ``mujoco_warp``'s render
    kernel, which needs ray differentials the kernel does not compute; nvdiffrast's mipmapped sampler
    closes it to 0.66-0.81 of far-band pixels (``docs/research/nvdiffrast.md``) and is not exactness
    either, is 3-5x slower, and cannot carry these scenes' MOVING lights, which are ``trackcom`` too.

    Supersampling is deliberately not offered. It was measured on the same mechanism for Gymnasium
    locomotion (``mujoco_locomotion.md`` S6.5): 16 samples per pixel take the far field's mean error
    from 100 to 8-14/255 and reach 0.6-1.4% exact, while whole-frame exactness FALLS on every domain,
    because a box average over 16 samples is not GL's mipmap pyramid and everything the renderer had
    right gets blurred.
    """

    fidelity_tier = "visual"
    parity_interior_min = None
    default_supersample = 1
    #: No camera knob. ``cam_fovy`` is not among the model arrays ``mujoco_warp.put_model`` will carry
    #: per world, so a per-environment field of view would need one render context per distinct value
    #: — and a variation that silently costs a context rebuild is worse than one that is not offered.
    CAMERA_KNOB = False

    #: Per-domain overrides for ``create_render_context``, folded into the cache key so two domains
    #: that disagree cannot share a context. Empty for every domain but swimmer; see
    #: ``SPECULAR_NOTE`` for what swimmer sets and why it is not set family-wide.
    RENDER_FLAGS: dict = {}

    def __init__(self, *a, **kw):
        super().__init__(*a, **kw)
        self._context(self.size)                     # build and capture up front, not mid-render

    def _context(self, size: int) -> _TCtx:
        return _tc_context(type(self), int(size), MJW_CHUNK)

    # ---- render -----------------------------------------------------------

    @torch.no_grad()
    def render(self, states=None, size: int | None = None, supersample: int | None = None,
               variation: dict | None = None, style=None, out=None) -> torch.Tensor:
        """``(...,state_dim)`` -> ``(...,H,W,3)`` uint8. Style may differ per environment."""
        s = self._state if states is None else torch.as_tensor(states, device=self.device,
                                                               dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim).to(torch.float32)
        B = s.shape[0]
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        if ss != 1:
            raise ValueError(
                f"{type(self).__name__}: supersample={ss} is not offered. MJWarp casts one ray per "
                f"pixel and a box average over sub-samples is not GL's mipmap pyramid: measured on "
                f"the identical mechanism it lowers the far floor's mean error and lowers whole-frame "
                f"exactness at the same time (docs/research/mujoco_locomotion.md S6.5).")
        sty = self.resolve_style(style, variation, batch=B)

        if out is None:
            out = torch.empty((B, size, size, 3), dtype=torch.uint8, device=self.device)
        buf = out.view(B, size, size, 3)
        c = self._context(size)
        for lo in range(0, B, MJW_CHUNK):
            hi = min(B, lo + MJW_CHUNK)
            self._upload(c, s[lo:hi], sty, lo, hi)
            torch.cuda.synchronize()
            c.wp.capture_launch(c.graph)
            c.wp.synchronize()
            # the BGRA -> RGB swap, ~3% of frame time and never optional: a byte view of MJWarp's
            # rgb_data is B,G,R,A, and reading it as RGB scores 0.0% exact on every domain
            buf[lo:hi] = c.bgra[:hi - lo][..., [2, 1, 0]]
        return buf.view(*lead, size, size, 3) if lead else buf

    @classmethod
    def _static_slot_rows(cls) -> dict:
        """``(kind, element id) -> (base row, [(state index, component), ...])`` for slot elements
        whose world transform ``mujoco_warp.kinematics`` never recomputes.

        This is the one place the per-world model upload is not enough, and it is silent when missed.
        ``mujoco_warp``'s ``_geom_local_to_global`` EARLY-OUTS on geoms and sites that hang off the
        world body -- their world pose cannot change under kinematics, so it is not recomputed -- and
        ``swimmer``'s per-episode ``target`` geom is exactly such an element. Writing the drawn
        ``geom_pos`` per world therefore reached nothing: measured, moving ``target_pos_x`` by 1.5 m
        changed the official frame by 216/255 and ours by **0**, while the per-episode LIGHT in the
        same state vector moved ours by 102. So the derived ``data`` row has to be written as well,
        which is free here because the world body's frame is the identity.
        """
        hit = cls.__dict__.get("_slot_rows_cache")
        if hit is not None:
            return hit
        m, pr = cls._model(), cls._pristine()
        rows: dict = {}
        for j, (arr, k) in enumerate(cls._slots()):
            kind, eid = _slot_element(m, arr, k)
            if kind not in ("geom", "site") or not arr.endswith("_pos"):
                continue
            bodyid = m.geom_bodyid if kind == "geom" else m.site_bodyid
            if int(bodyid[eid]) != 0:
                continue                              # kinematics DOES recompute this one
            rows.setdefault((kind, eid),
                            [np.asarray(pr[arr], np.float64)[eid].copy(), []])[1].append(
                (j, int(k % _row_width(m, arr))))
        cls._slot_rows_cache = rows
        return rows

    def _upload(self, c: _TCtx, s: torch.Tensor, sty, lo: int, hi: int) -> None:
        """Write this chunk's state and appearance into the context's PER-WORLD arrays."""
        import mujoco
        n = hi - lo
        wp = c.wp
        wp.to_torch(c.wd.qpos)[:n] = s[:, :self.nq]
        for j, (arr, k) in enumerate(self._slots()):
            wp.to_torch(getattr(c.wm, arr))[:n].reshape(n, -1)[:, k] = s[:, self.nq + j]
        for (kind, eid), (base, comps) in self._static_slot_rows().items():
            dst = wp.to_torch(c.wd.geom_xpos if kind == "geom" else c.wd.site_xpos)
            row = torch.as_tensor(base, device=self.device, dtype=torch.float32).repeat(n, 1)
            for j, comp in comps:
                row[:, comp] = s[:, self.nq + j]
            dst[:n, eid] = row
        m, pr = self._model(), self._pristine()
        mr = wp.to_torch(c.wm.mat_rgba)
        for name in self._used_materials():
            i = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_MATERIAL, name)
            if i >= 0:
                mr[:n, i, :3] = sty[f"material.{name}"][lo:hi]
        base = torch.as_tensor(pr["light_diffuse"], device=self.device, dtype=torch.float32)
        wp.to_torch(c.wm.light_diffuse)[:n] = torch.clamp(
            base.unsqueeze(0) * sty["light.diffuse_scale"][lo:hi].reshape(n, 1, 1), 0, 1)


#: Conditions every ``visual_fidelity`` below was measured under. One string, so a reader cannot get a
#: number without the conditions that produced it.
#: swimmer's entry is measured under its own render context (``enable_specular=False``) and over four
#: draws rather than three, so it carries its own provenance string rather than borrowing ``_AT``.
_SWIMMER_AT = ("84 px, 64 states x 4 independent draws from sample_states, WORST of the four, "
               "default appearance, enable_specular=False, one H200, mujoco 3.11.0 / warp 1.16.0, "
               "against dm_control's physics.render at camera 0 "
               "(scratch/altrender/probe_declare.py --config nospec_native --size 84 --draws 4)")

#: Turning MuJoCo Warp's specular term off on swimmer, and ONLY on swimmer. The measurement, at 256 px
#: batch 1024 against dm_control's own renderer, one draw of 64 states per cell:
#:
#:     domain         speed    mean abs        within-8         SSIM
#:     swimmer6       1.30x    4.16 -> 2.31    0.854 -> 0.955   0.9869 -> 0.9935   ship
#:     walker-walk    1.19x    5.85 -> 5.78    0.762 -> 0.763   0.9335 -> 0.9336   marginal
#:     cheetah-run    1.16x    5.016 -> 5.008  0.882 -> 0.884   0.66098 -> 0.66093 a wash
#:     humanoid-stand 1.16x    4.35 -> 4.23    0.866 -> 0.878   0.9364 -> 0.9404   SSIM up, mixed at 84 px
#:     hopper-hop     1.15x    4.347 -> 4.355  0.9053 -> 0.9051 0.6821 -> 0.6818   A LOSS
#:
#: IT IS NOT A FAMILY-WIDE SWITCH, and hopper is the reason it is written down: the same flag makes
#: that domain measurably worse on all three metrics. The mechanism is that dm_control's GL
#: fixed-function specular and MJWarp's are different lobes, so removing the term moves a frame
#: toward the reference only where MJWarp was adding a highlight the reference does not have. That
#: happens on the two floor-dominated domains (swimmer floor_share 0.981, humanoid 0.959) and not on
#: the skybox-dominated ones — the same camera-and-scene dependence the RoboCasa investigation found
#: between its agentview and wrist cameras (``scratch/altrender/README.md``). Swimmer is the one
#: domain where the gain is large (mean |diff| down 37%) rather than in the third decimal, so it is
#: the one domain that sets the flag.
SPECULAR_NOTE = ("This domain renders with enable_specular=False, which is measured to be BETTER "
                 "than the default on all four declared metrics as well as 1.30x faster (see "
                 "SPECULAR_NOTE in stable_envs/envs/dmcontrol.py for the per-domain table and for "
                 "why hopper does not get it).")

#: The one domain that overrides ``create_render_context``. Keyed by class name so the table above and
#: the switch cannot drift apart.
TRACKCOM_RENDER_FLAGS: dict[str, dict] = {
    "DMCSwimmerSwimmer6": {"enable_specular": False},
}

_AT = ("84 px, 64 states per domain from sample_states (the domain's own initialize_episode plus a "
       "0-40 step random-action rollout), default appearance, H200 / driver 590.48.01 / MUJOCO_GL=egl, "
       "mujoco 3.11.0 + mujoco-warp 3.11.0, against dm_control's physics.render at camera 0")

#: The versions ``_AT`` / ``_SWIMMER_AT`` name, as fields the certifier can compare with the measuring
#: interpreter's ``reference_packages`` (``stable_envs.core.env.REFERENCE_VERSIONS_KEY``). Exactly what
#: the text says and nothing more -- neither string names a dm_control version, so none is declared.
#: Why it matters: these numbers were judged under mujoco 3.12.0 in Step 4 and read as three renderer
#: failures; 3.12.0 re-anchors finite-plane textures (changelog item 15, commit cc7fb98c) and the
#: reference's floor checker had phase-shifted while ours had not
#: (examples/results/step6/texture-fidelity/findings.md). With the field declared, the same run is
#: ``blocked_environment: reference version mismatch`` instead of FAIL.
_AT_VERSIONS = {"mujoco": "3.11.0", "mujoco_warp": "3.11.0"}
_SWIMMER_AT_VERSIONS = {"mujoco": "3.11.0", "warp": "1.16.0"}

#: MEASURED per domain, then the thresholds the conformance suite asserts. The thresholds are set from
#: the SPREAD over independent state draws rather than from one, which is why they are not the
#: measurements: ``scratch/dmcontrol/visual_calib.py`` re-draws three times and the ceiling sits clear
#: of the worst draw.
VISUAL_FIDELITY: dict[str, dict] = {
    # ``interior_exact_frac`` is reported for honesty and is not a threshold anywhere. On three of the
    # five it is UNDEFINED rather than low: the interior mask keeps only pixels whose 3x3 neighbourhood
    # in the reference is one flat colour, and a DM Control frame under a speckled skybox and a
    # spotlit checker floor has almost none (``interior_frac`` 0.0000-0.0329).
    "DMCCheetahRun": {
        "mean_abs": 5.01, "within8_frac": 0.882, "ssim": 0.661,
        "interior_exact_frac": None, "interior_frac": 0.0000, "all_exact_frac": 0.290,
        "skybox_share": 0.831, "floor_share": 0.127,
        "draw_spread": {"mean_abs": [4.986, 5.032], "within8_frac": [0.8813, 0.8826],
                        "ssim": [0.6597, 0.6615]},
        "mean_abs_max": 6.5, "within8_min": 0.84, "ssim_min": 0.62, "measured_at": _AT,
        "reference_versions": _AT_VERSIONS},
    "DMCWalkerWalk": {
        "mean_abs": 8.07, "within8_frac": 0.700, "ssim": 0.826,
        "interior_exact_frac": 0.0046, "interior_frac": 0.0010, "all_exact_frac": 0.205,
        "skybox_share": 0.478, "floor_share": 0.448,
        "draw_spread": {"mean_abs": [7.636, 8.302], "within8_frac": [0.6887, 0.7199],
                        "ssim": [0.8170, 0.8306]},
        "mean_abs_max": 10.5, "within8_min": 0.65, "ssim_min": 0.77, "measured_at": _AT,
        "reference_versions": _AT_VERSIONS},
    "DMCHopperHop": {
        "mean_abs": 4.35, "within8_frac": 0.905, "ssim": 0.684,
        "interior_exact_frac": None, "interior_frac": 0.0000, "all_exact_frac": 0.307,
        "skybox_share": 0.893, "floor_share": 0.082,
        "draw_spread": {"mean_abs": [4.327, 4.364], "within8_frac": [0.9044, 0.9051],
                        "ssim": [0.6826, 0.6866]},
        "mean_abs_max": 5.7, "within8_min": 0.86, "ssim_min": 0.64, "measured_at": _AT,
        "reference_versions": _AT_VERSIONS},
    "DMCHumanoidStand": {
        "mean_abs": 6.26, "within8_frac": 0.735, "ssim": 0.868,
        "interior_exact_frac": 0.00018, "interior_frac": 0.0297, "all_exact_frac": 0.023,
        "skybox_share": 0.000, "floor_share": 0.959,
        "draw_spread": {"mean_abs": [6.203, 6.300], "within8_frac": [0.7308, 0.7442],
                        "ssim": [0.8660, 0.8707]},
        "mean_abs_max": 8.2, "within8_min": 0.69, "ssim_min": 0.82, "measured_at": _AT,
        "reference_versions": _AT_VERSIONS},
    # swimmer is the one domain that renders with ``enable_specular=False`` (see SPECULAR_NOTE), so
    # its entry is the only one measured under a non-default render context. Everything here is the
    # WORST of 4 independent 64-state draws, natively at 84 px with no resampling
    # (``scratch/altrender/probe_declare.py``), and every number is BETTER than the shipped
    # specular-on measurement it replaces: mean 5.48 -> 3.45, within-8 0.813 -> 0.917,
    # SSIM 0.980 -> 0.986, interior mean 6.22 -> 0.91, interior-exact 0.464 -> 0.515,
    # all-exact 0.123 -> 0.148. The thresholds keep the family's own margin convention (ceiling
    # 1.37x the measurement, floors 0.93x / 0.95x of it) rather than being tightened onto the
    # measurement, so they are far below the old ones without being fragile.
    "DMCSwimmerSwimmer6": {
        "mean_abs": 3.45, "within8_frac": 0.917, "ssim": 0.986,
        "interior_exact_frac": 0.515, "interior_mean_abs": 0.91,
        "interior_frac": 0.0059, "all_exact_frac": 0.148,
        "skybox_share": 0.000, "floor_share": 0.981,
        # the widest spread of the five, and it is the widened state doing it: swimmer draws its
        # target geom AND its target light per episode, so two draws differ in the scene and not only
        # in the pose. The ceiling is set from the worst of the four rather than from the mean.
        "draw_spread": {"mean_abs": [3.223, 3.452], "within8_frac": [0.917, 0.921],
                        "ssim": [0.986, 0.986], "interior_exact_frac": [0.515, 0.657]},
        "mean_abs_max": 4.8, "within8_min": 0.85, "ssim_min": 0.94, "measured_at": _SWIMMER_AT,
        "reference_versions": _SWIMMER_AT_VERSIONS},
}

VISUAL_NOTE = (
    "The frame is MJWarp's raytrace of the same scene, not dm_control's own raster. Two named "
    "mechanisms carry the residual, both measured per surface off MuJoCo's segmentation buffer: "
    "(1) the skybox is an 800x4800 gradient with mark=\"random\" specks and MJWarp samples it with one "
    "unfiltered tap, which is 48-89% of the frame on cheetah/hopper/walker at mean 4.1-4.7/255 and is "
    "what holds SSIM down; (2) the checker floor is MINIFIED and MJWarp has no level-of-detail "
    "selection, so where MuJoCo's mipmap shows one grey (far-band luminance spread 0.1 on humanoid) "
    "MJWarp paints full contrast (9.7). Suitable for generating training data live -- the frame is a "
    "function of the state and the appearance knobs are per-environment tensors. NOT a substitute for "
    "a stored dataset when a policy will be evaluated on official dm_control pixels: the train and "
    "eval pixel distributions differ by the amount above and nothing here measures what that costs a "
    "policy. Closing it needs plane-texture level-of-detail in mujoco_warp's render kernel, which "
    "needs ray differentials it does not compute.")


#: All-pixel floors that differ from the suite's shared 0.90, with the mechanism in ``PARITY_NOTE``.
#: ONE cell, because one cell needs it. The INTERIOR floor is untouched at 1.0 on every cell here and
#: is the number that describes the renderer; this only stops a scene being failed for the size of its
#: robot, and the other two finger cells clear the shared bar by 0.021-0.026 and declare nothing.
ALL_EXACT_FLOOR = {"DMCFingerTurnEasy": 0.89}

PARITY_NOTE = (
    "parity_interior_min is NOT relaxed: interior-pixel agreement measures exactly 1.000000 -- 0 bad "
    "pixels out of 9,834,291 -- over 201 independent draws of 64 states at the shipped "
    "default_supersample = 1. What is declared is parity_all_exact_min, and the reason is arithmetic "
    "rather than a tolerance -- all-pixel exactness is interior exactness WEIGHTED BY HOW MUCH OF THE "
    "FRAME MOVES, and finger-turn_easy's seven movers cover 11.7% of an 84 px frame against "
    "cartpole's 2.5%. Measured over those 201 draws: all-pixel 0.9103 at WORST and 0.9119 median, "
    "i.e. it CLEARS the shared 0.90 bar on every draw -- the floor is declared so that the margin is "
    "the scene's rather than luck's, and it is set from a draw count rather than from one large "
    "evaluation, because a rare pixel event reported off a single draw is how MujocoPusher came to "
    "declare a floor it then failed 15 times in 101. 100.0% of every disagreeing pixel is inside the "
    "ray-cast mover pass and the baked plate is 1.00000 exact on the pixels it owns; the residual is a "
    "one-level rim on curved silhouettes, where MuJoCo shades a tessellated capsule per vertex and an "
    "analytic per-pixel evaluation lands within a few levels of it. finger-spin (0.9288 worst) and "
    "finger-turn_hard (0.9241 worst) are the same renderer with fewer mover pixels and declare "
    "nothing. Dropping the supersample IMPROVED all three -- at ss=2 the same draws gave 0.9053 / "
    "0.9249 / 0.9200 -- because an aliased edge is bit-identical to MuJoCo wherever MuJoCo's own MSAA "
    "happened to be fully covered, while a blended edge almost never is.")


def _make_classes():
    made = {}
    for cname, (dom, task, cam) in DOMAINS.items():
        extra = ({"parity_all_exact_min": ALL_EXACT_FLOOR[cname], "parity_note": PARITY_NOTE}
                 if cname in ALL_EXACT_FLOOR else {})
        cls = register(type(cname, (DMControlSuite,), dict(
            **extra,
            DOMAIN=dom, TASK=task, CAMERA=cam,
            name=f"dmc-{dom.replace('_', '-')}-{task.replace('_', '-')}",
            reference_id=f"dm_control/{dom}-{task}",
            __doc__=(f"DM Control {dom}-{task} at camera {cam}. State is qpos plus whatever the "
                     f"episode draws into the model; the frame is a measured pure function of it."),
        )))
        made[cname] = cls
    for cname, (dom, task, cam) in TRACKCOM_DOMAINS.items():
        vf = VISUAL_FIDELITY[cname]
        flags = TRACKCOM_RENDER_FLAGS.get(cname, {})
        cls = register(type(cname, (DMControlTrackCom,), dict(
            DOMAIN=dom, TASK=task, CAMERA=cam,
            name=f"dmc-{dom.replace('_', '-')}-{task.replace('_', '-')}",
            reference_id=f"dm_control/{dom}-{task}",
            visual_fidelity=vf,
            RENDER_FLAGS=flags,
            visual_note=VISUAL_NOTE + (" " + SPECULAR_NOTE if flags else ""),
            __doc__=(
                f"DM Control {dom}-{task} at camera {cam}, a mode=\"trackcom\" camera.\n\n"
                f"VISUAL TIER -- these are NOT the official pixels. The frame is MJWarp's raytrace of "
                f"dm_control's own scene at dm_control's own camera, which lands on MuJoCo's "
                f"cam_xpos to 1e-8 m; measured against physics.render it is mean "
                f"{vf['mean_abs']:.2f}/255 with {100 * vf['within8_frac']:.1f}% of pixels within "
                f"8/255 and SSIM {vf['ssim']:.3f}. The residual is the unfiltered speckled skybox and "
                f"the minified checker floor. Good for generating training data live; not "
                f"interchangeable with a stored dataset of official pixels. See visual_note."),
        )))
        made[cname] = cls
    return made


globals().update(_make_classes())
