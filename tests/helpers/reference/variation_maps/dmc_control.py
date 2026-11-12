"""``VariationMap``s for the five classic-control cells and the thirteen DM Control cells.

Seventeen ``stable_worldmodel`` ids, 161 variation leaves: 140 ``mapped`` and 21 ``dead_upstream``;
none ``unsupported`` or ``not_applicable``. Every DM Control physics leaf (gravity, densities, masses,
friction, walker's floor tilt) goes through the physics hook and is gated against swm stepping the
same values (tests/integration/test_dmc_physics_variation.py).

The DM Control dead leaves:

* Sixteen never reach the compiled model (every ``MjModel`` array hashed at the default and at a
  probe value). The wrapper never reads the seven ``*_locked`` leaves, ``BallInCup.ball.color``,
  ``Cartpole.agent.cart_shape``, ``Pendulum.agent.mass_shape`` or
  ``Manipulator.agent.upper_arm_length``. ``Acrobot.agent.{upper,lower}_arm_density``,
  ``Cartpole.agent.pole_density`` and ``Pendulum.agent.{mass,pole}_density`` are written onto geoms
  that carry an explicit ``mass``, which wins over density.
* Five reach the model but no pixel: ``Acrobot.target.{color,shape}`` (a site in group 3) and
  ``BallInCup.{rendering.render_target, target.color, target.shape}`` (group 4). ``mjvOption``'s
  default ``geomgroup`` draws groups 0-2 only: 0 px over 8 states, also with ``render_target`` forced
  to 1. Finger, reacher and manipulator hide their group-0 targets by material alpha instead, so the
  same leaves move pixels there and are mapped.

On the five ``trackcom`` cells (cheetah, walker, hopper, humanoid, quadruped) the camera follows
``subtree_com``, so a density leaf changes ``body_mass`` and moves every pixel at a fixed state (2,115
to 137,533 px over 8 states with ``qpos`` unchanged); the renderer derives each row's masses from the
leaf.

Classic control: ``GymControlWrapper.render`` swaps pixels exactly equal to a source color, while
stable-envs draws the primitive in the color. The registered default is byte-identical on all five
envs; a non-default color differs on anti-aliased boundaries, where swm keeps a blend of the
original color (at most 0.6 % of the frame, MountainCar). ``visuals.fg`` and ``visuals.cart`` are
one-to-many: black is the car, terrain and flag pole on MountainCar, and the cart and track on
CartPole, so one swm leaf drives several of ours and the inverse names the first.
"""

from __future__ import annotations

import numpy as np

from ..variation import VariationMap

# ===================================================================== classic control

#: Every classic-control cell offers this and swm does not.
_AA_ONLY = {
    "rendering.antialias": "exact anti-aliasing on or off; swm's envs are pygame and have no "
    "such switch. See _Classic._aa_space for what it costs."
}

#: The caveat on every mapped classic-control color; the numbers are the worst case over the five envs.
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
    "tests/integration/test_classic_variation.py::test_classic_physics_matches_swm: reset State and every "
    "step output equal swm's exactly over 24 steps, with lanes at 10/50/90 % of the range."
)


_GEOMETRY_MAPPED = (
    _DYNAMICS_MAPPED + " The official render also draws this leaf ({what}), and so does ours, per lane; the same gate "
    "compares every reset and step frame with swm's own, explicit and sampled, at worst-frame "
    "normalized MSE 0."
)


def _dyn(*names: str, notes: dict[str, str], geometry: dict[str, str] | None = None) -> dict:
    """``identical`` and ``notes`` for the physics leaves of one classic cell."""
    leaves = tuple(f"physics.{n}" for n in names)
    drawn = {f"physics.{n}": _GEOMETRY_MAPPED.format(what=what) for n, what in (geometry or {}).items()}
    return {"identical": leaves, "notes": {**{leaf: _DYNAMICS_MAPPED for leaf in leaves}, **drawn, **notes}}


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
                "link_length_1": "the first link at LINK_LENGTH_1 * scale, with scale = 500 / (2 * (l1 + l2 + "
                "0.2)) resizing the whole picture, joint radius and ground line included",
                "link_length_2": "the second link at LINK_LENGTH_2 * scale, with the same picture scale",
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
            # it as well. Measured: with visuals.cart at a probe value the track changes color in
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
            geometry={"length": "the pole at polelen = scale * (2 * length)"},
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
    cloth (204, 204, 0) and the wheels (128, 128, 128) are not black and swm therefore cannot recolor
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

#: ``light.intensity`` is an absolute diffuse in [0, 1]; ``light.diffuse_scale`` multiplies the
#: compiled model's own diffuse and clips to [0, 1]. The factor is ``1 / base``, with the base read
#: off the domain's XML (``_pristine()["light_diffuse"]``): 0.7 except ball_in_cup, finger and
#: manipulator at 0.6. Each model has one light with an equal diffuse on all three channels.
_LIGHT_BASE = {
    "DMCAcrobotBalance": 0.7,
    "DMCBallInCupCatch": 0.6,
    "DMCCartpoleBalance": 0.7,
    "DMCCartpoleSwingup": 0.7,
    "DMCCheetahRun": 0.7,
    "DMCFingerTurnHard": 0.6,
    "DMCHopperHop": 0.7,
    "DMCHumanoidWalk": 0.7,
    "DMCManipulatorBringBall": 0.6,
    "DMCPendulumSwingup": 0.7,
    "DMCQuadrupedWalk": 0.7,
    "DMCReacher": 0.7,
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


#: The eight plate-baking cells (``DMControlSuite``), seven swm ids. They bake static plates through a
#: ``mujoco.Renderer`` and ray-cast the movers, so a texel substitution on the plate rig reaches the
#: floor. The other five (``DMControlTrackCom``) hand the whole scene to mujoco_warp and have no plate.
#: Transcribed because this module must import without ``stable_envs.envs``;
#: ``test_the_parity_floor_set_is_the_plate_baking_tier`` checks it against the classes.
_PARITY_FLOOR = frozenset(
    {
        "DMCAcrobotBalance",
        "DMCBallInCupCatch",
        "DMCCartpoleBalance",
        "DMCCartpoleSwingup",
        "DMCFingerTurnHard",
        "DMCManipulatorBringBall",
        "DMCPendulumSwingup",
        "DMCReacher",
    }
)

#: The floor, on the eight plate-baking cells.
_FLOOR_MAPPED = (
    "Our ``floor.color`` is swm's leaf in shape, order, units and default: (2,3) rgb1/rgb2 of the "
    "builtin ``grid`` checker, with the default READ from dm_control's own common/materials.xml -- "
    "the same file swm's init_value [[.1,.2,.3],[.2,.3,.4]] transcribes -- so it is `identical`, not "
    "a rename or a scale. swm honours the leaf by rewriting the MJCF and RECOMPILING; we substitute "
    "the compiled texture's texels on the plate rig's own model and re-upload the texture to its own "
    "GL context, which needs no recompile and no second render context. VERIFIED byte-for-byte, not "
    "argued: over all eight plate-baking cells, three probe colours and eight states at 84 px, the "
    "substituted ``tex_data`` and the rendered frames are IDENTICAL to a full recompile of the suite "
    "at the same floor.color. "
    "The suite's markrgb equals its rgb2, so the 1,196 edge-mark texels are separated positionally "
    "and left alone exactly as swm's recompile leaves them."
)

#: The floor, on the five trackcom cells.
_FLOOR_TRACKCOM = (
    "Our ``floor.color`` is swm's leaf in shape, order, units and default, read from dm_control's "
    "common/materials.xml as on the parity cells. swm rewrites the builtin ``grid`` texture's rgb1/rgb2 "
    "and recompiles; these cells hand the whole scene to mujoco_warp, which fixes its textures when the "
    "render context is built, so the floor is instead drawn from a second texture holding each texel's "
    "palette weights (rgb1, rgb2, and the edge marks, which keep markrgb as swm's recompile does). "
    "Texture filtering is linear in the texels, so weights times each row's colours, stored as the "
    "compiled texture stores them, are the substituted texture's own samples: one render context serves "
    "every colour; on CUDA the recoloured frames are bit-identical to a render context built on the "
    "substituted texture. The CPU renderer does the same per row and the OpenGL backend substitutes the "
    "texels. Gated by tests/integration/test_dmc_appearance_variation.py against swm's frames over 8 "
    "steps, normalized MSE mean (worst frame) over the five cells: a dim checker ((.3,.25,.05), "
    "(.05,.1,.35)) at most 0.00038 (0.00044) on CPU and 0.00056 (0.00061) on CUDA, near each cell's "
    "default residual (0.0001-0.0004); a saturated one ((.9,.1,.1), (.1,.8,.2)) at most 0.00053 "
    "(0.00074) on CPU and 0.0015 (0.0015, humanoid) on CUDA, whose floor shading residual grows with "
    "the floor's albedo; 0.002-0.12 with the leaf ignored. The OpenGL backend matches swm at MSE 0."
)

_PHYSICS_MAPPED = (
    "MAPPED through the physics hook (stable_envs._core.physics): the value is written into each "
    "lane's compiled model as swm writes it after its recompile (a direct field write, or an analytic "
    "mass change that agrees with the recompile to 1e-12), and travels in State under "
    "('physics', *leaf) as float64, so a restored State keeps its physics. The style column is "
    "float32; an explicit float64 value set through reset options or State is kept exactly. Gated by "
    "tests/integration/test_dmc_physics_variation.py against swm stepping the same per-lane values: "
    "direct writes 0 State error, mass leaves at most 3e-7 (relative, humanoid's torso over 40 "
    "contact-rich steps; 1e-12 in the model itself), while the leaf moves swm's own trajectory by 3e-3 "
    "to 2.7"
)

#: Leaves swm accepts and its own dynamics never feel, measured on its trajectory.
_INERT = {
    "gravity.y": "Mapped and exact, and inert upstream: every joint of this planar model moves in the "
    "x-z plane, so y gravity does no work and swm's own trajectory does not move (0 qpos shift).",
    "hopper": "Mapped and exact, and inert upstream: every hopper geom carries friction 1.0 and MuJoCo "
    "combines a contact's friction by max, so swm's own trajectory does not move (0 qpos shift over 200 "
    "steps at 0.25 and 0.9).",
    "quadruped": "Mapped and exact, and inert upstream: the feet carry friction 1.5 and MuJoCo combines "
    "a contact's friction by max, so swm's own trajectory does not move (0 qpos shift over 200 steps).",
}

_FLOOR_TILT = (
    _PHYSICS_MAPPED + ". swm writes the floor geom's zaxis = (sin, 0, cos) in float32 and recompiles; "
    "the physics hook writes the geom_quat that compile produces (MuJoCo's own z-to-vector rule, "
    "bit-identical), its frame flag and the world body's bounding box, which is 0 State error over the "
    "physics gate. The floor turns about its own centre (x = 248), so any tilt moves the walker's "
    "contact by metres. The renderer takes each row's floor orientation from the same value: the "
    "mujoco_warp path uploads the floor's per-world pose, which its floor shading, reflection and shadow "
    "passes already read; the CPU path intersects, textures, shades and reflects per-row planes. "
    "Against swm's frames over 8 steps at 0.125 and -0.25 degrees, normalized MSE mean 0.00036 and "
    "0.00041 on CPU and on CUDA, as at the default (0.00039), against 0.0051 and 0.0069 with the tilt "
    "left out of the render; the OpenGL backend at most 1.5e-9"
)

_TRACKCOM_MASS = (
    _PHYSICS_MAPPED + '. This cell renders through a mode="trackcom" camera whose position comes from '
    "subtree_com, so the mass also moves the camera: the renderer derives each row's body masses, "
    "centres of mass, subtree masses and the cameras' qpos0 offsets from the leaf with the same "
    "arithmetic as the simulator, and uploads them per world. Gated by tests/integration/test_dmc_appearance_variation.py against swm's own frames over 8 steps at 550 and 1450: the residual stays at the "
    "cell's registered-default residual (walker torso: normalized MSE 0.00039, as at the default, "
    "against 0.022 with the leaf ignored)"
)

_CATENARY = (
    _PHYSICS_MAPPED + ". The cup-ball tendon is drawn as a catenary hanging along gravity, so x and y "
    "move pixels at a fixed state; the renderer evaluates each row's catenary under that row's gravity. "
    "Gated by tests/integration/test_dmc_appearance_variation.py: gravity.x = 2.5 renders at normalized MSE 0.00026 against swm's "
    "frames (0.00025 at the default, 0.00036 with the leaf ignored)"
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
    mapped=(),
    unsupported=None,
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
        identical=("floor.color",) + tuple(mapped) + tuple(physics_mapped),
        renamed=renamed,
        scaled={"light.diffuse_scale": ("light.intensity", 1.0 / base)},
        enum=({shape[0]: ("target.shape", shape[1])} if shape else {}),
        notes={
            "light.intensity": _light_reason(base),
            "floor.color": _FLOOR_MAPPED if parity else _FLOOR_TRACKCOM,
            **{leaf: _PHYSICS_MAPPED for leaf in physics_mapped},
            **(notes or {}),
        },
        unsupported=dict(unsupported or {}),
        dead_upstream=dict(dead_upstream or {}),
        ours_only=dict(ours_only or {}),
    )


#: ``ours_only`` entries that recur. ``material.grid`` is our floor material knob, not the mapping for
#: ``floor.color``, which is the checker texture.
_GRID_OURS = {
    "material.grid": "a 3-vector multiplying the floor MATERIAL's rgba, which is a "
    "different knob from the checker TEXTURE and composes with it: swm "
    "has no material-level floor knob. Our floor.color is swm's "
    "floor.color and is mapped `identical`"
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

_GRAVITY = tuple(f"gravity.{a}" for a in "xyz")


def _trackcom(cell, swm_id, densities, *, ours_only, tilt=False, dead_upstream=None, floor_note=None):
    """The five trackcom cells: gravity, floor friction, two densities and walker's tilt, through the physics hook."""
    notes = {leaf: _TRACKCOM_MASS for leaf in densities}
    if tilt:
        notes["floor.rotation_y"] = _FLOOR_TILT
    if cell not in ("DMCHumanoidWalk", "DMCQuadrupedWalk"):
        notes["gravity.y"] = _INERT["gravity.y"]
    if floor_note:
        notes["floor.friction"] = floor_note
    return _dmc(
        cell,
        swm_id,
        physics_mapped=(*_GRAVITY, "floor.friction", *densities, *(("floor.rotation_y",) if tilt else ())),
        notes=notes,
        dead_upstream=dead_upstream,
        ours_only=ours_only,
    )


_SHAPE_NOTE = (
    "MAPPED at both alternatives: SHAPE_KNOBS declares swm's box and sphere at the task's target size, "
    "and the sphere is the compiled model's own, so the default style stays a render of it. Visible only "
    "with rendering.render_target = 1, which swm's registered default is not. The task reads only the "
    "target's first size, which both alternatives share, so the shape never reaches the dynamics. "
    "Normalized MSE against swm's frames with the box drawn: finger 0.00032, reacher 0.00025 (defaults "
    "0.00026 and 0.00016; 0.0011 and 0.00072 with the leaf ignored)."
)

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
        physics_mapped=_GRAVITY,
        notes={"gravity.y": _INERT["gravity.y"]},
        ours_only={**_GRID_OURS, **_FOVY_OURS, **_DECOR_OURS},
    ),
    "DMCBallInCupCatch": _dmc(
        "DMCBallInCupCatch",
        "swm/BallInCupDMControl-v0",
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
        physics_mapped=(*_GRAVITY, "agent.density", "ball.density", "ball.size"),
        notes={
            **{f"gravity.{a}": _CATENARY for a in "xyz"},
            "ball.size": _PHYSICS_MAPPED + ". The radius is geometry as well: the ball's size, collision "
            "bounds, mass (its density times the new volume) and inertia follow it in the simulator, and "
            "the renderer draws each row's ball at that row's radius: normalized MSE 0.00026 against swm's "
            "frames at 0.04 (0.00079 with the leaf ignored)",
        },
        ours_only={**_GRID_OURS, **_FOVY_OURS, **_EFFECTOR_OURS},
    ),
    "DMCCheetahRun": _trackcom(
        "DMCCheetahRun",
        "swm/CheetahDMControl-v0",
        ("agent.torso_density", "agent.back_foot_density"),
        dead_upstream={"agent.back_foot_locked": _never_read()},
        ours_only=_GRID_OURS,
    ),
    "DMCFingerTurnHard": _dmc(
        "DMCFingerTurnHard",
        "swm/FingerDMControl-v0",
        mapped_target="material.target",
        visibility="visibility.target",
        mapped=("spinner.color", "target.shape"),
        physics_mapped=("agent.fingertip_density", "agent.proximal_density", "spinner.density", "spinner.friction"),
        notes={
            "spinner.color": "MAPPED: the leaf writes cap1's and cap2's explicit rgba, as swm's does, so the "
            "spinner stops following material.self exactly where swm's stops following agent.color. "
            "Normalized MSE 0.00026 against swm's frames, the default's own residual (0.0064 with the "
            "leaf ignored).",
            "target.shape": _SHAPE_NOTE,
        },
        ours_only={**_GRID_OURS, **_FOVY_OURS, **_DECOR_OURS, **_EFFECTOR_OURS},
    ),
    "DMCHopperHop": _trackcom(
        "DMCHopperHop",
        "swm/HopperDMControl-v0",
        ("agent.torso_density", "agent.foot_density"),
        dead_upstream={"agent.foot_locked": _never_read()},
        floor_note=_INERT["hopper"],
        ours_only=_GRID_OURS,
    ),
    "DMCHumanoidWalk": _trackcom(
        "DMCHumanoidWalk",
        "swm/HumanoidDMControl-v0",
        ("agent.torso_density", "agent.right_lower_arm_density"),
        dead_upstream={"agent.left_knee_locked": _never_read()},
        ours_only=_GRID_OURS,
    ),
    "DMCManipulatorBringBall": _dmc(
        "DMCManipulatorBringBall",
        "swm/ManipulatorDMControl-v0",
        mapped_target="material.target",
        visibility="visibility.target_ball",
        # swm's Discrete(2) is (0 = box .015, 1 = sphere .015); ours is Discrete(3) whose third
        # alternative is the compiled model's own sphere .02, so the default style renders the
        # compiled model. Both swm alternatives are reachable at the same indices.
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
                "The swm source carries ``# TODO should modify length of arm`` at the point it would be read."
            )
        },
        physics_mapped=("agent.hand_density", "agent.upper_arm_density"),
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
        physics_mapped=_GRAVITY,
        notes={"gravity.y": _INERT["gravity.y"]},
        ours_only={**_GRID_OURS, **_FOVY_OURS, **_DECOR_OURS, **_EFFECTOR_OURS},
    ),
    "DMCQuadrupedWalk": _trackcom(
        "DMCQuadrupedWalk",
        "swm/QuadrupedDMControl-v0",
        ("agent.torso_density", "agent.foot_back_left_density"),
        dead_upstream={"agent.knee_back_left_locked": _never_read()},
        floor_note=_INERT["quadruped"],
        ours_only={**_GRID_OURS, **_EFFECTOR_OURS},
    ),
    "DMCReacher": _dmc(
        "DMCReacher",
        "swm/ReacherDMControl-v0",
        mapped_target="material.target",
        visibility="visibility.target",
        mapped=("target.shape",),
        notes={"target.shape": _SHAPE_NOTE},
        dead_upstream={"agent.finger_locked": _never_read()},
        physics_mapped=("agent.arm_density", "agent.finger_density"),
        ours_only={**_GRID_OURS, **_FOVY_OURS, **_DECOR_OURS, **_EFFECTOR_OURS},
    ),
    "DMCWalkerWalk": _trackcom(
        "DMCWalkerWalk",
        "swm/WalkerDMControl-v0",
        ("agent.torso_density", "agent.left_foot_density"),
        tilt=True,
        dead_upstream={"agent.right_knee_locked": _never_read()},
        ours_only=_GRID_OURS,
    ),
}

# The two cartpole cells are one swm id and differ only in their reset distribution, which the
# mapping cannot see; each gets its own map instance so a divergence is not hidden by an alias.
for _cell in ("DMCCartpoleBalance", "DMCCartpoleSwingup"):
    _DMC_MAPS[_cell] = _dmc(
        _cell,
        "swm/CartpoleDMControl-v0",
        dead_upstream={
            "agent.cart_shape": _never_read(),
            "agent.pole_density": _density_dead(
                "cartpole.xml's ``pole`` default class, which sets an explicit mass on every pole geom"
            ),
        },
        physics_mapped=(*_GRAVITY, "agent.cart_mass"),
        notes={"gravity.y": _INERT["gravity.y"]},
        ours_only={**_GRID_OURS, **_FOVY_OURS, **_DECOR_OURS},
    )


#: stable-envs class name -> its map. Installed by ``variation_maps.install``.
MAPS: dict[str, VariationMap] = {**_CLASSIC_MAPS, **_DMC_MAPS}


# ===================================================================== the light range gap

#: ``light.intensity`` is mapped over 86-88 % of its range; the gap is recorded here.
#: ``cell -> (base, reachable_low, reachable_high, note)``.
LIGHT_RANGE: dict[str, tuple[float, float, float, str]] = {
    cell: (base, 0.2 * base, 1.0, _light_reason(base)) for cell, base in _LIGHT_BASE.items()
}

#: The four-way count per swm id, asserted by tests/unit/test_variation_maps_dmc_control.py against
#: the live swm variation spaces.
EXPECTED_COUNTS: dict[str, dict[str, int]] = {
    "swm/AcrobotControl-v1": {"total": 11, "mapped": 11, "unsupported": 0, "dead_upstream": 0, "not_applicable": 0},
    "swm/CartPoleControl-v1": {"total": 10, "mapped": 10, "unsupported": 0, "dead_upstream": 0, "not_applicable": 0},
    "swm/PendulumControl-v1": {"total": 9, "mapped": 9, "unsupported": 0, "dead_upstream": 0, "not_applicable": 0},
    "swm/MountainCarControl-v0": {"total": 5, "mapped": 5, "unsupported": 0, "dead_upstream": 0, "not_applicable": 0},
    "swm/MountainCarContinuousControl-v0": {
        "total": 4,
        "mapped": 4,
        "unsupported": 0,
        "dead_upstream": 0,
        "not_applicable": 0,
    },
    "swm/AcrobotDMControl-v0": {"total": 11, "mapped": 6, "unsupported": 0, "dead_upstream": 5, "not_applicable": 0},
    "swm/BallInCupDMControl-v0": {"total": 13, "mapped": 9, "unsupported": 0, "dead_upstream": 4, "not_applicable": 0},
    "swm/CartpoleDMControl-v0": {"total": 9, "mapped": 7, "unsupported": 0, "dead_upstream": 2, "not_applicable": 0},
    "swm/CheetahDMControl-v0": {"total": 10, "mapped": 9, "unsupported": 0, "dead_upstream": 1, "not_applicable": 0},
    "swm/FingerDMControl-v0": {"total": 11, "mapped": 11, "unsupported": 0, "dead_upstream": 0, "not_applicable": 0},
    "swm/HopperDMControl-v0": {"total": 10, "mapped": 9, "unsupported": 0, "dead_upstream": 1, "not_applicable": 0},
    "swm/HumanoidDMControl-v0": {"total": 10, "mapped": 9, "unsupported": 0, "dead_upstream": 1, "not_applicable": 0},
    "swm/ManipulatorDMControl-v0": {
        "total": 9,
        "mapped": 8,
        "unsupported": 0,
        "dead_upstream": 1,
        "not_applicable": 0,
    },
    "swm/PendulumDMControl-v0": {"total": 9, "mapped": 6, "unsupported": 0, "dead_upstream": 3, "not_applicable": 0},
    "swm/QuadrupedDMControl-v0": {"total": 10, "mapped": 9, "unsupported": 0, "dead_upstream": 1, "not_applicable": 0},
    "swm/ReacherDMControl-v0": {"total": 9, "mapped": 8, "unsupported": 0, "dead_upstream": 1, "not_applicable": 0},
    "swm/WalkerDMControl-v0": {"total": 11, "mapped": 10, "unsupported": 0, "dead_upstream": 1, "not_applicable": 0},
}


def swm_defaults(swm_id: str) -> dict:
    """The registered default of every leaf in ``swm_id``'s variation space, read from the live wrapper.

    Read live rather than transcribed so the claim stays about the installed swm.
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
