"""How the five classic-control cells and the thirteen DM Control cells relate to their
``stable_worldmodel`` counterparts, leaf by leaf, with the measurement behind every classification.

Seventeen ``stable_worldmodel`` ids, 161 variation leaves. Every one of them is exactly one of:

``mapped``
    reachable through a stable-envs leaf. 46 leaves.
``unsupported``
    the leaf has a real effect upstream that THIS cell cannot express. 94 leaves, of three kinds, and
    the reason on every one of them says which: appearance the cell has no knob for (floor colour,
    floor tilt, ball size, a per-geom rgba, a target shape); appearance the cell has no knob for that
    arrives through the PHYSICS (the ten ``trackcom`` mass leaves and ball-in-cup's gravity, S4); and
    PHYSICS-ONLY leaves of a render-from-state cell, which are unsupported-by-this-cell rather than
    not-applicable because they do ask something of stable-envs — a dynamics surface these cells do
    not have — even though they move no pixel.
``dead_upstream``
    the leaf reaches no pixel and no state in ``stable_worldmodel`` ITSELF. The reason carries the
    measurement. 21 leaves.
``not_applicable``
    reserved by the taxonomy and UNUSED by this family: every leaf these seventeen ids offer does ask
    something of stable-envs, even the ones that ask for a simulator. 0 leaves.

Why the physics leaves are ``unsupported`` and not ``not_applicable``
---------------------------------------------------------------------

The five classic-control cells ship a batched simulator whose trajectories are certified against the
official env, so ``physics.gravity`` is a feature they could carry and do not. The thirteen DM
Control cells ship a renderer over states and no simulator at all (``hasattr(env, 'step')`` is
``False`` on every one), so ``agent.torso_density`` asks them for a dynamics surface that does not
exist. Both are the same answer to the caller — "this cell cannot honour that leaf" — and both carry
the mechanism and the measured 0-pixel result in their reason. Calling the second kind
``not_applicable`` would say the leaf asks nothing of stable-envs, and it does; it asks for a
simulator. The distinction between the two kinds is preserved in the REASON text, which names which
it is, rather than in the label.

The exception, and the most interesting measurement in this workstream, is the ``trackcom`` group
below: there a mass leaf is not physics-only at all.

Three things this module measured that were not previously written down
-----------------------------------------------------------------------

**1. Sixteen SWM DM Control leaves never reach the compiled model.** Probed by hashing every
``numpy`` array on ``MjModel`` (110+ arrays, plus ``opt.gravity`` and the size scalars) at the leaf's
default and at an interior probe value, through ``reset(options={'variation_values': ...})`` on the
live wrapper. Two distinct mechanisms, both dead:

* *the wrapper never reads it* — all seven ``*_locked`` leaves, ``BallInCup.ball.color``,
  ``Cartpole.agent.cart_shape``, ``Pendulum.agent.mass_shape``, ``Manipulator.agent.upper_arm_length``
  (the last one already recorded in Step 7 with the ``# TODO should modify length of arm`` beside it).
* *MuJoCo ignores the write* — ``Acrobot.agent.{upper,lower}_arm_density``,
  ``Cartpole.agent.pole_density`` and ``Pendulum.agent.{mass,pole}_density`` are written onto the
  geom's ``density``, but those geoms carry an explicit ``mass``: acrobot's
  ``<default class="/"><geom type="capsule" mass="1"/>``, cartpole's ``pole`` class, pendulum's three
  ``mass=`` attributes. An explicit mass wins, so the recompiled model is bit-identical.

**2. Five more leaves, in two groups, reach the model and no pixel because the element is not drawn.**
``Acrobot.target.{color,shape}`` move ``mat_rgba``/``site_type``/``site_size`` on a site in **group 3**,
and ``BallInCup.{rendering.render_target, target.color, target.shape}`` move ``site_rgba``/``site_type``
on a site in **group 4**. ``mjvOption``'s default ``geomgroup`` draws groups 0-2, so neither site is
ever rasterised: 0 px over 8 states, and 0 px for ball-in-cup even with ``render_target`` forced to 1.
Contrast finger, reacher and manipulator, whose targets are in group 0 and hidden by a material
*alpha* instead — there the same leaf moves 150-459 px and is mapped.

**3. On the five ``trackcom`` cells a mass leaf moves pixels at a fixed state.** ``cheetah``,
``walker``, ``hopper``, ``humanoid`` and ``quadruped`` all render through a ``mode="trackcom"`` camera,
whose position is derived from ``subtree_com``. Changing a density changes ``body_mass``, which moves
the camera, which moves every pixel — 2,115 to 137,533 disagreeing pixels over 8 states with ``qpos``
and ``geom_xpos`` verified bit-identical. So these leaves are NOT "physics with no pixel": they are
appearance-affecting leaves this cell cannot express, and they are classified accordingly.

Classic control: an exact-match recolor is not the same picture as drawing in the colour
-----------------------------------------------------------------------------------------

``GymControlWrapper.render`` renders the official frame and then swaps pixels **exactly equal** to a
source RGB for the target colour. stable-envs draws the primitive IN the colour. The two agree
everywhere the reference wrote the source colour verbatim and disagree on the anti-aliased boundary,
where swm keeps a blend of the ORIGINAL colour and we produce a blend of the new one. Measured over 8
states per env, one leaf at a time at 90 % of its range:

======================== ============ ======================
env                      default      one leaf at a probe
======================== ============ ======================
Acrobot (8x500x500)      **0 px**     2-2,725 px (<= 0.14 %)
CartPole (8x400x600)     **0 px**     64-957 px (<= 0.05 %)
Pendulum (8x500x500)     **0 px**     64-2,726 px (<= 0.14 %)
MountainCar (8x400x600)  **0 px**     11,033-11,326 px (0.6 %)
======================== ============ ======================

The registered default is **byte-identical on every one of the five**, which is the property
certification asks for; the divergence is confined to non-default styles and to shape boundaries. It is recorded
here rather than smoothed over, because reproducing it would mean rendering the default frame and
then recoloring it, which is a renderer change and a worse picture.

``visuals.fg`` and ``visuals.cart`` are ONE-TO-MANY. swm swaps a source colour wherever it occurs, and
black occurs on more than one element: MountainCar's ``fg`` is the car, the terrain AND the flag pole,
CartPole's ``cart`` is the cart AND the track. stable-envs holds those as separate leaves, so one swm
leaf drives three of ours. The inverse names the first, which is the same appearance.
"""

from __future__ import annotations

import numpy as np

from ..variation import VariationMap

# ===================================================================== classic control

#: Every classic-control cell offers exactly this and swm does not, on every one of the five.
_AA_ONLY = {
    "rendering.antialias": "exact anti-aliasing on or off; swm's envs are pygame and have no "
    "such switch. See _Classic._aa_space for what it costs."
}

#: The mapped-colour caveat every classic-control cell carries. See the module docstring for the
#: measured table; the numbers here are the worst case over the five envs.
_RECOLOR_NOTE = (
    "MAPPED, and NOT bit-identical away from the default. swm recolours the rendered frame by EXACT "
    "uint8 match; stable-envs draws the primitive in the colour. The two agree wherever the reference "
    "wrote the source colour verbatim and differ on anti-aliased boundaries, where swm keeps a blend "
    "of the ORIGINAL colour. Measured over 8 states, one leaf at a time at 90 % of its range: 0 "
    "disagreeing pixels at the registered default on all five envs, and at most 0.6 % of the frame "
    "(11,326 px of 8x400x600, MountainCar) at a probe value. The conversion itself is exact: this "
    "leaf uses `rgb255`, which applies swm's own uint8(v * 255 + 0.5)."
)


_DYNAMICS_MAPPED = (
    "MAPPED, identical path and units: the value is written onto each lane's Gymnasium env exactly as "
    "swm writes it (a float32 Box value widened to float64), sampled per lane under `variation` or set "
    "under `variation_values`, and plain resets restore the default. Gated by "
    "tests/integration/test_variation_pilots.py::test_classic_physics_matches_swm: reset State and every "
    "step output equal swm's exactly over 24 steps, with lanes at 10/50/90 % of the range."
)


def _dyn(*names: str, notes: dict[str, str], geometry: dict[str, str] | None = None) -> dict:
    """``identical``, ``notes`` and ``partial`` for the physics leaves of one classic cell."""
    leaves = tuple(f"physics.{n}" for n in names)
    return {
        "identical": leaves,
        "notes": {**{leaf: _DYNAMICS_MAPPED for leaf in leaves}, **notes},
        "partial": {f"physics.{n}": why for n, why in (geometry or {}).items()},
    }


_CLASSIC_MAPS = {
    "Acrobot": VariationMap(
        swm_id="swm/AcrobotControl-v1",
        rgb255={
            "background.color": "visuals.bg",
            "link.color": "visuals.link",
            "joint.color": "visuals.joint",
            "ground.color": "visuals.line",
        },
        **_dyn(
            "link_length_1",
            "link_length_2",
            "link_mass_1",
            "link_mass_2",
            "link_com_pos_1",
            "link_com_pos_2",
            "link_moi",
            geometry={
                "link_length_1": "dynamics only: AcrobotEnv.render also draws each link at LINK_LENGTH_1 * scale, "
                "and moving it to 1.45 moves 37,966 px of 8x500x500 in the official frame. The stable-envs "
                "renderer holds SCALE and the unit-length link quad as class constants, so the frame does "
                "not follow the leaf.",
                "link_length_2": "dynamics only: the same on the second link, 38,258 px of 8x500x500 at 1.45.",
            },
            notes={k: _RECOLOR_NOTE for k in ("visuals.bg", "visuals.link", "visuals.joint", "visuals.line")},
        ),
        ours_only=_AA_ONLY,
    ),
    "CartPole": VariationMap(
        swm_id="swm/CartPoleControl-v1",
        rgb255={
            "background.color": "visuals.bg",
            "cart.color": "visuals.cart",
            # ONE-TO-MANY: the reference paints the track black too, so swm's ``cart`` swap hits
            # it as well. Measured: with visuals.cart at a probe value the track changes colour in
            # the swm arm, and a map that moved only ``cart.color`` would leave it black.
            "track.color": "visuals.cart",
            "pole.color": "visuals.pole",
            "axle.color": "visuals.axle",
        },
        **_dyn(
            "gravity",
            "masscart",
            "masspole",
            "length",
            "force_mag",
            "tau",
            geometry={
                "length": "dynamics only: CartPoleEnv.render also sets polelen = scale * 2 * length, and 0.725 "
                "moves 6,621 px of 8x400x600 in the official frame. POLELEN is a class constant on the "
                "stable-envs renderer, so the frame does not follow the leaf."
            },
            notes={
                "visuals.bg": _RECOLOR_NOTE,
                "visuals.pole": _RECOLOR_NOTE,
                "visuals.axle": _RECOLOR_NOTE,
                "visuals.cart": _RECOLOR_NOTE + " ONE-TO-MANY: this leaf also drives `track.color`, "
                "because the reference paints the track black too.",
            },
        ),
        ours_only=_AA_ONLY,
    ),
    "Pendulum": VariationMap(
        swm_id="swm/PendulumControl-v1",
        rgb255={"background.color": "visuals.bg", "rod.color": "visuals.rod", "axle.color": "visuals.hub"},
        **_dyn(
            "g",
            "m",
            "l",
            "max_torque",
            "max_speed",
            "dt",
            notes={k: _RECOLOR_NOTE for k in ("visuals.bg", "visuals.rod", "visuals.hub")},
        ),
        ours_only=_AA_ONLY,
    ),
}


def _mountain_car(swm_id: str, *dyn: str) -> VariationMap:
    """MountainCar and MountainCarContinuous share a renderer and differ in their dynamics leaves.

    ``visuals.fg`` is the widest one-to-many in the family: the reference paints the car body, the 99
    terrain segments and the flag POLE all in black, so one swm leaf drives three of ours. The flag
    cloth (204, 204, 0) and the wheels (128, 128, 128) are not black and swm therefore cannot recolour
    them at all; they are ``ours_only``.
    """
    return VariationMap(
        swm_id=swm_id,
        rgb255={
            "background.color": "visuals.bg",
            "car.color": "visuals.fg",
            "terrain.color": "visuals.fg",
            "flag.pole_color": "visuals.fg",
        },
        **_dyn(
            *dyn,
            notes={
                "visuals.bg": _RECOLOR_NOTE,
                "visuals.fg": _RECOLOR_NOTE + " ONE-TO-MANY: this leaf drives `car.color`, "
                "`terrain.color` AND `flag.pole_color`, because the "
                "reference paints all three black.",
            },
        ),
        ours_only={
            **_AA_ONLY,
            "flag.color": "the flag cloth is (204, 204, 0) in the reference and swm swaps only "
            "(255,255,255) and (0,0,0), so it has no leaf that reaches it",
            "wheel.color": "the wheels are (128, 128, 128); same reason",
        },
    )


_CLASSIC_MAPS["MountainCar"] = _mountain_car("swm/MountainCarControl-v0", "gravity", "force", "max_speed")
_CLASSIC_MAPS["MountainCarContinuous"] = _mountain_car("swm/MountainCarContinuousControl-v0", "power", "max_speed")

# ===================================================================== DM Control

#: ``light.intensity`` is an ABSOLUTE diffuse in [0, 1]; ``light.diffuse_scale`` MULTIPLIES the
#: compiled model's own diffuse and clips to [0, 1]. The factor is therefore ``1 / base``, and the
#: base is a property of the domain's XML, read off ``_pristine()["light_diffuse"]``:
#: 0.7 everywhere except ball_in_cup, finger and manipulator, which are 0.6. Every one of these models
#: has exactly one light (``nlight == 1``) with an equal diffuse on all three channels, so one scalar
#: scale really is the whole leaf.
_LIGHT_BASE = {
    "DMCAcrobotBalance": 0.7,
    "DMCBallInCupCatch": 0.6,
    "DMCCartpoleBalance": 0.7,
    "DMCCartpoleSwingup": 0.7,
    "DMCCheetahRun": 0.7,
    "DMCFingerTurnHard": 0.6,
    "DMCHopperHop": 0.7,
    "DMCHumanoidStand": 0.7,
    "DMCManipulatorBringBall": 0.6,
    "DMCPendulumSwingup": 0.7,
    "DMCQuadrupedWalk": 0.7,
    "DMCReacherHard": 0.7,
    "DMCWalkerWalk": 0.7,
}


def _light_reason(base: float) -> str:
    lo = 0.2 * base
    return (
        f"mapped to light.diffuse_scale with factor 1/{base} — the compiled model's own light "
        f"diffuse. RANGE GAP at the dark end: our scale is Box(0.2, 1.8), so the reachable "
        f"intensity is [{lo:.3f}, 1.0] (1.8 * {base} clips to 1.0 at the bright end, which costs "
        f"nothing because swm's range stops at 1.0), and swm's [0, {lo:.3f}) is not reachable. "
        f"swm's default 0.7 is reached at scale {0.7 / base:.6f}."
    )


#: The eight PARITY-TIER cells (``DMControlSuite``) that carry swm's ``floor.color``: seven swm ids,
#: eight of our cells. They bake three static plates through a ``mujoco.Renderer`` and ray-cast the
#: movers, so the floor is plate-only and a texel substitution on the plate rig reaches it. The other
#: five (``DMControlTrackCom``) hand the whole scene to mujoco_warp and have no plate.
#:
#: Written down rather than derived with ``issubclass`` because this module must import without
#: ``stable_envs.envs``; ``test_the_parity_floor_set_is_the_plate_baking_tier`` derives it from the
#: classes and asserts this set, so the transcription cannot drift.
_PARITY_FLOOR = frozenset(
    {
        "DMCAcrobotBalance",
        "DMCBallInCupCatch",
        "DMCCartpoleBalance",
        "DMCCartpoleSwingup",
        "DMCFingerTurnHard",
        "DMCManipulatorBringBall",
        "DMCPendulumSwingup",
        "DMCReacherHard",
    }
)

#: The floor, on the eight parity cells: MAPPED, identically, as of the capability patch.
_FLOOR_MAPPED = (
    "Our ``floor.color`` is swm's leaf in shape, order, units and default: (2,3) rgb1/rgb2 of the "
    "builtin ``grid`` checker, with the default READ from dm_control's own common/materials.xml -- "
    "the same file swm's init_value [[.1,.2,.3],[.2,.3,.4]] transcribes -- so it is `identical`, not "
    "a rename or a scale. swm honours the leaf by rewriting the MJCF and RECOMPILING; we substitute "
    "the compiled texture's texels on the plate rig's own model and re-upload the texture to its own "
    "GL context, which needs no recompile and no second render context. VERIFIED byte-for-byte, not "
    "argued: over all twelve parity cells, three probe colours and eight states at 84 px, the "
    "substituted ``tex_data`` and the rendered frames are IDENTICAL to a full recompile of the suite "
    "at the same floor.color (pre-repository-cleanup:tests/fixtures/regression/join/step8/capability/bake_vs_recompile_probe{0,1,2}.json). "
    "The suite's markrgb equals its rgb2, so the 1,196 edge-mark texels are separated positionally "
    "and left alone exactly as swm's recompile leaves them."
)

#: The floor, on the five trackcom cells: still unsupported, and now for a narrower reason.
_FLOOR_COLOR = (
    "UNSUPPORTED on this cell, and it is a RANGE gap rather than a default gap. swm writes the "
    "builtin ``grid`` TEXTURE's rgb1/rgb2 (2x3 = six free scalars) and recompiles. This cell is in "
    "the ``DMControlTrackCom`` tier: the camera rides the body, there is no static half of the scene "
    "and therefore no baked plate, and the whole scene is handed to mujoco_warp, which uploads its "
    "textures when the render context is BUILT -- so a per-value floor here is one render context "
    "per value, which is the one thing this repo will not scale. The capability that closed this "
    "leaf on the eight parity cells (per-value texel substitution on a baked plate) does not apply "
    "where there is no plate. What IS reachable is the one value that matters for certification: "
    "swm's init_value is character-for-character common/materials.xml's own grid texture, so the "
    "REGISTERED DEFAULT floor is the compiled model's floor and needs no leaf at all. "
    "DOCUMENTED_LIMITATION, five ids (Cheetah, Walker, Hopper, Humanoid, Quadruped)."
)

_GRAVITY_PHYS = (
    "PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or not-applicable: gravity "
    "is a real dynamics parameter of the swm env, and this cell is a render-from-state cell that "
    "exposes no dynamics parameters to map it through. Measured 0 disagreeing pixels over 8 states "
    "with qpos and geom_xpos bit-identical, so nothing about the FRAME is lost; what is not reachable "
    "is the trajectory."
)


_PHYSICS_MAPPED = (
    "MAPPED through the physics hook (stable_envs._core.physics): the value is written into each "
    "lane's compiled model as swm writes it after its recompile, and travels in State under "
    "('physics', *leaf) as float64, so a restored State keeps its physics. The style column is "
    "float32; an explicit float64 value set through reset options or State is kept exactly. "
    "Gated by tests/integration/test_variation_pilots.py: 0 State error over 40 steps against swm "
    "stepping the same per-lane values, with the knob moving swm's trajectory by 9.4e7 times the "
    "1e-9 tolerance"
)


def _density_phys(what: str) -> str:
    return (
        f"PHYSICS-ONLY, and therefore UNSUPPORTED-BY-THIS-CELL rather than dead or "
        f"not-applicable: {what} is a real dynamics parameter of the swm env, and this cell is a "
        f"render-from-state cell that exposes no dynamics parameters to map it through. Measured "
        f"0 disagreeing pixels over 8 states with qpos and geom_xpos bit-identical, so nothing "
        f"about the FRAME is lost; what is not reachable is the trajectory."
    )


def _density_trackcom(px: int, n_px: int) -> str:
    return (
        f'UNSUPPORTED, not merely physics: this cell renders through a mode="trackcom" camera, '
        f"whose position comes from subtree_com, so a mass change MOVES THE CAMERA. Measured "
        f"{px:,} disagreeing pixels of {n_px:,} over 8 states with qpos and geom_xpos bit-"
        f"identical. The cell declares no mass leaf and its camera follows the compiled model's "
        f"own COM, so the appearance is not reachable."
    )


_LOCKED_DEAD = (
    "DEAD UPSTREAM: the leaf is declared by the swm wrapper and its own "
    "``modify_mjcf_model`` never reads it (0 textual references outside the space "
    "literal). Measured: not one of the 110+ MjModel arrays changes between the leaf's "
    "two values, so it reaches neither a pixel nor a state."
)


def _never_read(extra: str = "") -> str:
    return _LOCKED_DEAD + (" " + extra if extra else "")


def _density_dead(where: str) -> str:
    return (
        "DEAD UPSTREAM: the wrapper writes the value onto the geom's ``density`` and recompiles, "
        f"but {where} carries an EXPLICIT mass, and in MuJoCo an explicit mass wins over density. "
        "Measured: not one MjModel array changes, body_mass included."
    )


def _target_group(group: int, extra: str = "") -> str:
    return (
        f"DEAD UPSTREAM: the leaf does reach the compiled model, and the element it reaches is "
        f"never drawn. The target is in geom/site GROUP {group}, and mjvOption's default "
        f"``geomgroup`` rasterises groups 0-2 only. Measured 0 disagreeing pixels over 8 states"
        + (" " + extra if extra else "")
        + "."
    )


def _dmc(
    cell: str,
    swm_id: str,
    *,
    mapped_target: str | None = None,
    visibility: str | None = None,
    shape: tuple[str, tuple[int, ...]] | None = None,
    unsupported=None,
    physics_only=None,
    physics_mapped=(),
    dead_upstream=None,
    notes=None,
    ours_only=None,
) -> VariationMap:
    """The shared skeleton. ``agent.color`` and ``light.intensity`` are mapped on all thirteen."""
    base = _LIGHT_BASE[cell]
    renamed = {"material.self": "agent.color"}
    if mapped_target:
        renamed[mapped_target] = "target.color"
    if visibility:
        renamed[visibility] = "rendering.render_target"
    parity = cell in _PARITY_FLOOR
    return VariationMap(
        swm_id=swm_id,
        identical=(("floor.color",) if parity else ()) + tuple(physics_mapped),
        renamed=renamed,
        scaled={"light.diffuse_scale": ("light.intensity", 1.0 / base)},
        enum=({shape[0]: ("target.shape", shape[1])} if shape else {}),
        notes={
            "light.intensity": _light_reason(base),
            **({"floor.color": _FLOOR_MAPPED} if parity else {}),
            **{leaf: _PHYSICS_MAPPED for leaf in physics_mapped},
            **(notes or {}),
        },
        # ``physics_only`` is merged into ``unsupported`` rather than kept as its own bucket: plan
        # S9's taxonomy has four labels and a physics leaf of a render-from-state cell is
        # "unsupported by this cell", not "not applicable" — the coordinator's ruling, and the right
        # one, because ``not_applicable`` would say the leaf asks nothing of stable-envs and it does.
        # The separate keyword survives so the REASON is uniform and the count is auditable.
        unsupported={
            **({} if parity else {"floor.color": _FLOOR_COLOR}),
            **(physics_only or {}),
            **(unsupported or {}),
        },
        dead_upstream=dict(dead_upstream or {}),
        ours_only=dict(ours_only or {}),
    )


#: ``ours_only`` entries that recur. ``material.grid`` is our floor knob and is deliberately NOT the
#: mapping for ``floor.color`` — see ``_FLOOR_COLOR``.
_GRID_OURS = {
    "material.grid": "a 3-vector multiplying the floor MATERIAL's rgba, which is a "
    "different knob from the checker TEXTURE and composes with it: swm "
    "has no material-level floor knob. Our floor.color is swm's "
    "floor.color and is mapped `identical` on the parity cells; on the "
    "five trackcom cells this is the only floor knob there is"
}
_FOVY_OURS = {
    "camera.fovy_scale": "a field-of-view multiplier; swm's DMControlWrapper renders at the "
    "model's own camera and offers no such knob"
}
_DECOR_OURS = {
    "material.decoration": "the rails/arena furniture material; swm recolours only "
    "material `self` and, where it has one, material `target`"
}
_EFFECTOR_OURS = {"material.effector": "the effector material; swm has no leaf that reaches it"}

_DMC_MAPS = {
    "DMCAcrobotBalance": _dmc(
        "DMCAcrobotBalance",
        "swm/AcrobotDMControl-v0",
        dead_upstream={
            "agent.upper_arm_locked": _never_read(),
            "agent.upper_arm_density": _density_dead(
                'acrobot\'s <default class="/"><geom type="capsule" mass="1"/> means every geom'
            ),
            "agent.lower_arm_density": _density_dead(
                'acrobot\'s <default class="/"><geom type="capsule" mass="1"/> means every geom'
            ),
            "target.color": _target_group(3, "(site `target`, material `target`)"),
            "target.shape": _target_group(3, "(site `target`; the leaf does move site_type and site_size)"),
        },
        physics_only={f"gravity.{a}": _GRAVITY_PHYS for a in "xyz"},
        ours_only={**_GRID_OURS, **_FOVY_OURS, **_DECOR_OURS},
    ),
    "DMCBallInCupCatch": _dmc(
        "DMCBallInCupCatch",
        "swm/BallInCupDMControl-v0",
        unsupported={
            "ball.size": "the ball's radius, which is GEOMETRY: 902 disagreeing pixels of 8x84x84 at "
            "0.0475. The cell declares colour, light and (elsewhere) visibility leaves "
            "and no size leaf, so the appearance is not reachable.",
            **{
                f"gravity.{a}": (
                    "UNSUPPORTED rather than physics-only: ball_in_cup's cup and ball are joined by a "
                    "SPATIAL TENDON, which MuJoCo draws as a hanging catenary through mjv_catenary, and "
                    "mjv_catenary reads the model's gravity. Measured at a fixed qpos: gravity.x moves "
                    "178 px and gravity.y 110 px of 8x84x84 (gravity.z alone moves 0, because scaling z "
                    "does not rotate the gravity direction). The cell reproduces mjv_catenary from the "
                    "compiled model's gravity and declares no gravity leaf."
                )
                for a in "xyz"
            },
        },
        dead_upstream={
            "ball.color": _never_read(
                "The ball is drawn with material `self`, which ``agent.color`` already reaches."
            ),
            "rendering.render_target": _target_group(
                4, "(site `target`), and 0 px again with render_target forced to 1"
            ),
            "target.color": _target_group(4, "(site `target`, rgba on the site itself)"),
            "target.shape": _target_group(4, "(site `target`; the leaf does move site_type)"),
        },
        physics_only={
            "agent.density": _density_phys("the cup's density"),
            "ball.density": _density_phys("the ball's density"),
        },
        ours_only={**_GRID_OURS, **_FOVY_OURS, **_EFFECTOR_OURS},
    ),
    "DMCCheetahRun": _dmc(
        "DMCCheetahRun",
        "swm/CheetahDMControl-v0",
        unsupported={
            "agent.torso_density": _density_trackcom(2115, 8 * 84 * 84),
            "agent.back_foot_density": _density_trackcom(4521, 8 * 84 * 84),
        },
        dead_upstream={"agent.back_foot_locked": _never_read()},
        physics_only={
            "floor.friction": _density_phys("the floor's sliding friction"),
            **{f"gravity.{a}": _GRAVITY_PHYS for a in "xyz"},
        },
        ours_only=_GRID_OURS,
    ),
    "DMCFingerTurnHard": _dmc(
        "DMCFingerTurnHard",
        "swm/FingerDMControl-v0",
        mapped_target="material.target",
        visibility="visibility.target",
        unsupported={
            "spinner.color": "the spinner's colour is a per-geom ``rgba`` OVERRIDE (the leaf moves "
            "geom_rgba, not mat_rgba) and this cell's colour knobs are material-"
            "level, one per material carried by a visible element. Measured 2,191 "
            "disagreeing pixels of 8x84x84, so the leaf is real; there is simply no "
            "leaf here that reaches a single geom's rgba.",
            "target.shape": "the target site's box/sphere alternative. Measured 459 disagreeing "
            "pixels of 8x84x84 WITH the target visible, so the leaf is real. This "
            "cell declares SHAPE_KNOBS = {} — only DMCManipulatorBringBall carries a "
            "shape leaf — so the default (sphere, index 1) is the compiled model's "
            "own and reproduces exactly, and the box alternative is unreachable.",
        },
        physics_only={
            "agent.fingertip_density": _density_phys("the fingertip's density"),
            "agent.proximal_density": _density_phys("the proximal link's density"),
            "spinner.density": _density_phys("the spinner's density"),
            "spinner.friction": _density_phys("the spinner joint's friction loss"),
        },
        ours_only={**_GRID_OURS, **_FOVY_OURS, **_DECOR_OURS, **_EFFECTOR_OURS},
    ),
    "DMCHopperHop": _dmc(
        "DMCHopperHop",
        "swm/HopperDMControl-v0",
        unsupported={
            "agent.torso_density": _density_trackcom(10918, 8 * 84 * 84),
            "agent.foot_density": _density_trackcom(11167, 8 * 84 * 84),
        },
        dead_upstream={"agent.foot_locked": _never_read()},
        physics_only={
            "floor.friction": _density_phys("the floor's sliding friction"),
            **{f"gravity.{a}": _GRAVITY_PHYS for a in "xyz"},
        },
        ours_only=_GRID_OURS,
    ),
    "DMCHumanoidStand": _dmc(
        "DMCHumanoidStand",
        "swm/HumanoidDMControl-v0",
        unsupported={
            "agent.torso_density": _density_trackcom(36302, 8 * 84 * 84),
            "agent.right_lower_arm_density": _density_trackcom(25419, 8 * 84 * 84),
        },
        dead_upstream={"agent.left_knee_locked": _never_read()},
        physics_only={
            "floor.friction": _density_phys("the floor's sliding friction"),
            **{f"gravity.{a}": _GRAVITY_PHYS for a in "xyz"},
        },
        ours_only=_GRID_OURS,
    ),
    "DMCManipulatorBringBall": _dmc(
        "DMCManipulatorBringBall",
        "swm/ManipulatorDMControl-v0",
        mapped_target="material.target",
        visibility="visibility.target_ball",
        # swm's Discrete(2) is (0 = box .015, 1 = sphere .015); ours is Discrete(3) whose third
        # alternative is the compiled model's own sphere .02, which keeps the DEFAULT STYLE a render
        # of the compiled model. Both swm alternatives are reachable, at the same indices.
        shape=("target_ball.shape", (0, 1)),
        notes={
            "target.shape": "MAPPED at both alternatives. swm's Discrete(2) is (0 = box .015, "
            "1 = sphere .015); ours is Discrete(3) whose index 2 is the compiled "
            "model's own sphere .02 and is the DEFAULT, which is what keeps the "
            "default style a render of the compiled model. swm's registered "
            "default is 1, so reaching it is a leaf write, not a no-op.",
            "rendering.render_target": "MAPPED to visibility.target_ball. swm's registered "
            "default is 0 (the target is NOT drawn), and our leaf's "
            "default is 1, so the registered default is reached "
            "through the leaf. Measured 274 disagreeing pixels of "
            "8x224x224 between the two values upstream.",
        },
        dead_upstream={
            "agent.upper_arm_length": _never_read(
                "The swm source carries ``# TODO should modify length of arm`` at the point it would be "
                "read; recorded in Step 7 and re-measured here."
            )
        },
        physics_only={
            "agent.hand_density": _density_phys("the hand's density"),
            "agent.upper_arm_density": _density_phys("the upper arm's density"),
        },
        ours_only={
            **_GRID_OURS,
            **_FOVY_OURS,
            **_DECOR_OURS,
            **_EFFECTOR_OURS,
            "material.background": "the arena backdrop material; swm has no leaf for it",
        },
    ),
    "DMCPendulumSwingup": _dmc(
        "DMCPendulumSwingup",
        "swm/PendulumDMControl-v0",
        dead_upstream={
            "agent.mass_shape": _never_read(),
            "agent.mass_density": _density_dead('pendulum.xml\'s <geom name="mass" ... mass="1"/>'),
            "agent.pole_density": _density_dead('pendulum.xml\'s <geom name="pole" ... mass="0"/>'),
        },
        physics_only={f"gravity.{a}": _GRAVITY_PHYS for a in "xyz"},
        ours_only={**_GRID_OURS, **_FOVY_OURS, **_DECOR_OURS, **_EFFECTOR_OURS},
    ),
    "DMCQuadrupedWalk": _dmc(
        "DMCQuadrupedWalk",
        "swm/QuadrupedDMControl-v0",
        unsupported={
            "agent.torso_density": _density_trackcom(135622, 8 * 224 * 224),
            "agent.foot_back_left_density": _density_trackcom(137533, 8 * 224 * 224),
        },
        dead_upstream={"agent.knee_back_left_locked": _never_read()},
        physics_only={
            "floor.friction": _density_phys("the floor's sliding friction"),
            **{f"gravity.{a}": _GRAVITY_PHYS for a in "xyz"},
        },
        ours_only={**_GRID_OURS, **_EFFECTOR_OURS},
    ),
    "DMCReacherHard": _dmc(
        "DMCReacherHard",
        "swm/ReacherDMControl-v0",
        mapped_target="material.target",
        visibility="visibility.target",
        unsupported={
            "target.shape": "the target geom's box/sphere alternative. Measured 236 "
            "disagreeing pixels of 8x84x84 WITH the target visible, so the "
            "leaf is real. This cell declares SHAPE_KNOBS = {}, so the "
            "default (sphere, index 1) is the compiled model's own and "
            "reproduces exactly, and the box alternative is unreachable."
        },
        dead_upstream={"agent.finger_locked": _never_read()},
        physics_only={
            "agent.arm_density": _density_phys("the arm's density"),
            "agent.finger_density": _density_phys("the finger's density"),
        },
        ours_only={**_GRID_OURS, **_FOVY_OURS, **_DECOR_OURS, **_EFFECTOR_OURS},
    ),
    "DMCWalkerWalk": _dmc(
        "DMCWalkerWalk",
        "swm/WalkerDMControl-v0",
        unsupported={
            "agent.torso_density": _density_trackcom(26328, 8 * 84 * 84),
            "agent.left_foot_density": _density_trackcom(21374, 8 * 84 * 84),
            "floor.rotation_y": "walker is the one domain whose swm wrapper TILTS the floor: the leaf "
            "writes the floor geom's quat and moves 24,004 disagreeing pixels of "
            "8x84x84 at 9 degrees. The cell bakes the floor plane from the "
            "compiled model and declares no pose leaf, so the tilt is not "
            "reachable. Its default, 0, is the compiled model's own.",
        },
        dead_upstream={"agent.right_knee_locked": _never_read()},
        physics_only={
            "floor.friction": _density_phys("the floor's sliding friction"),
            **{f"gravity.{a}": _GRAVITY_PHYS for a in "xyz"},
        },
        ours_only=_GRID_OURS,
    ),
}

# The two cartpole cells are one swm id. The balance and swingup tasks differ in their reset
# distribution and in nothing the appearance mapping can see, so they carry the same map built twice
# rather than one shared object — ``_LIGHT_BASE`` is keyed by cell and a shared instance would hide a
# future divergence behind an alias.
# Only the swing-up cell simulates; its gravity leaves go through the physics hook.
for _cell in ("DMCCartpoleBalance", "DMCCartpoleSwingup"):
    _gravity = tuple(f"gravity.{a}" for a in "xyz") if _cell == "DMCCartpoleSwingup" else ()
    _DMC_MAPS[_cell] = _dmc(
        _cell,
        "swm/CartpoleDMControl-v0",
        dead_upstream={
            "agent.cart_shape": _never_read(),
            "agent.pole_density": _density_dead(
                "cartpole.xml's ``pole`` default class, which sets an explicit mass on every pole geom"
            ),
        },
        physics_only={
            "agent.cart_mass": _density_phys("the cart's mass"),
            **{leaf: _GRAVITY_PHYS for leaf in (f"gravity.{a}" for a in "xyz") if leaf not in _gravity},
        },
        physics_mapped=_gravity,
        ours_only={**_GRID_OURS, **_FOVY_OURS, **_DECOR_OURS},
    )


#: stable-envs CLASS name -> its map. Installed by ``variation_maps.install``.
MAPS: dict[str, VariationMap] = {**_CLASSIC_MAPS, **_DMC_MAPS}


# ===================================================================== the light range gap

#: ``light.intensity`` is mapped, and its RANGE GAP is recorded here rather than in ``unsupported``,
#: because a leaf that is reachable over 86-88 % of its range is mapped with a documented gap, not
#: unsupported. ``cell -> (base, reachable_low, reachable_high, note)``.
LIGHT_RANGE: dict[str, tuple[float, float, float, str]] = {
    cell: (base, 0.2 * base, 1.0, _light_reason(base)) for cell, base in _LIGHT_BASE.items()
}

#: The four-way count this module claims, asserted by tests/unit/test_variation_maps_dmc_control.py
#: against the LIVE swm variation spaces rather than against this table.
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "swm/AcrobotControl-v1": {"total": 11, "mapped": 9, "unsupported": 0, "dead_upstream": 0, "not_applicable": 0},
    "swm/CartPoleControl-v1": {"total": 10, "mapped": 9, "unsupported": 0, "dead_upstream": 0, "not_applicable": 0},
    "swm/PendulumControl-v1": {"total": 9, "mapped": 9, "unsupported": 0, "dead_upstream": 0, "not_applicable": 0},
    "swm/MountainCarControl-v0": {"total": 5, "mapped": 5, "unsupported": 0, "dead_upstream": 0, "not_applicable": 0},
    "swm/MountainCarContinuousControl-v0": {
        "total": 4,
        "mapped": 4,
        "unsupported": 0,
        "dead_upstream": 0,
        "not_applicable": 0,
    },
    "swm/AcrobotDMControl-v0": {"total": 11, "mapped": 3, "unsupported": 3, "dead_upstream": 5, "not_applicable": 0},
    "swm/BallInCupDMControl-v0": {"total": 13, "mapped": 3, "unsupported": 6, "dead_upstream": 4, "not_applicable": 0},
    "swm/CartpoleDMControl-v0": {"total": 9, "mapped": 3, "unsupported": 4, "dead_upstream": 2, "not_applicable": 0},
    "swm/CheetahDMControl-v0": {"total": 10, "mapped": 2, "unsupported": 7, "dead_upstream": 1, "not_applicable": 0},
    "swm/FingerDMControl-v0": {"total": 11, "mapped": 5, "unsupported": 6, "dead_upstream": 0, "not_applicable": 0},
    "swm/HopperDMControl-v0": {"total": 10, "mapped": 2, "unsupported": 7, "dead_upstream": 1, "not_applicable": 0},
    "swm/HumanoidDMControl-v0": {"total": 10, "mapped": 2, "unsupported": 7, "dead_upstream": 1, "not_applicable": 0},
    "swm/ManipulatorDMControl-v0": {
        "total": 9,
        "mapped": 6,
        "unsupported": 2,
        "dead_upstream": 1,
        "not_applicable": 0,
    },
    "swm/PendulumDMControl-v0": {"total": 9, "mapped": 3, "unsupported": 3, "dead_upstream": 3, "not_applicable": 0},
    "swm/QuadrupedDMControl-v0": {"total": 10, "mapped": 2, "unsupported": 7, "dead_upstream": 1, "not_applicable": 0},
    "swm/ReacherDMControl-v0": {"total": 9, "mapped": 5, "unsupported": 3, "dead_upstream": 1, "not_applicable": 0},
    "swm/WalkerDMControl-v0": {"total": 11, "mapped": 2, "unsupported": 8, "dead_upstream": 1, "not_applicable": 0},
}


#: Cells whose counts differ from the first cell of their swm id, which EXPECTED_COUNTS records.
EXPECTED_COUNTS_BY_CELL: dict[str, dict[str, int]] = {
    "DMCCartpoleSwingup": {"total": 9, "mapped": 6, "unsupported": 1, "dead_upstream": 2, "not_applicable": 0},
}


#: The swm registered DEFAULT variation config for each id, as a dotted dict. Written down because it
#: is what "the registered default is reproducible through the mapping" is measured against, and
#: reading it off a live wrapper makes the test depend on the wrapper it is testing.
def swm_defaults(swm_id: str) -> dict:
    """The leaf defaults of ``swm_id``'s variation space, read from the live wrapper.

    Deliberately live rather than transcribed: the claim under test is about the CURRENT swm, and a
    transcribed table would keep passing after swm moved. The test that uses it also asserts the
    values it cares about against the numbers in this module's docstring.
    """
    import gymnasium
    import stable_worldmodel  # noqa: F401

    env = gymnasium.make(swm_id)
    node = env
    while node is not None and getattr(node, "variation_space", None) is None:
        node = getattr(node, "env", None)
    space = node.variation_space
    out = {}
    for name in space.names():
        leaf = space
        for part in name.split("."):
            leaf = leaf[part]
        out[name] = np.asarray(leaf.init_value)
    env.close()
    return out
