"""The twelve ``swm/Fetch*-v3`` ids, mapped once as a family.

``stable_worldmodel`` reaches four gymnasium-robotics scenes through twelve ids, and the ``Dense``
and ``Dict`` suffixes change the reward and the observation container and not a pixel. Their
variation spaces are therefore *identical per scene* — measured, not assumed
(``pre-repository-cleanup:tests/fixtures/regression/join/step8/variations-fetch/leaf_classification.json`` carries the enumeration from
the live wrapper for all twelve) — so this module builds twelve maps from ONE vocabulary plus a
per-scene difference table, rather than writing twelve tables that would drift apart.

WHAT THE SWM WRAPPER ACTUALLY DOES, WHICH IS NOT WHAT ITS LEAF NAMES SAY
-----------------------------------------------------------------------

``stable_worldmodel.envs.gymnasium_robotics.fetch.FetchWrapper`` offers 11 leaves (8 on Reach, which
has no block). Four of them do not reach a pixel in the OFFICIAL renderer, and the mapping has to
say so rather than invent support:

``camera.angle_delta`` — **dead upstream.** The wrapper rotates ``model.cam_pos`` for every MODEL
camera. ``MujocoFetchEnv`` renders through the FREE camera it configures in
``DEFAULT_CAMERA_CONFIG`` (asserted independently by ``FetchBase._check_camera``), and never selects
a model camera. Measured on all four scenes at 480 px: ``angle_delta`` at ``[[0,0]]`` versus
``[[±10,±10]]`` is **0/255 over 0% of the frame**, while ``model.cam_pos`` really does move by
0.2257. Four dead model cameras (``ncam == 4``) is what makes this look alive from the model side.
We DO have a working camera (``camera.azimuth_delta`` / ``camera.elevation_delta`` /
``camera.distance_scale``), and mapping the swm leaf onto it would make our frame move where the
reference's does not — i.e. it would turn a truthful "the reference ignores this" into a silent
fidelity divergence. Same treatment as Rocket's ``environment.sky_color``.

``block.mass`` — **not applicable.** Pure dynamics. Measured at 0.01 versus 50.0 kg on Push, Slide
and PickAndPlace: **0/255 over 0% of the frame**. Our state is ``qpos + goal``; a mass is not in it
and reaches nothing we draw.

``object.color`` on **Reach** — **dead upstream on that scene only.** The wrapper resolves the
object by the ``block_mat``/``puck_mat`` material; ``reach.xml`` compiles neither onto any geom, so
the write lands on an empty geom list. Measured: two maximally different object colours on Reach
give **0/255 over 0% of the frame**. On Push and PickAndPlace it drives ``block_mat`` (geom 23) and
on Slide ``puck_mat`` (geom 23) — exactly the geom our ``object.color`` owns — and is live at
0.07-0.12% of the frame.

``rendering.transparent_arm`` — **live upstream, unsupported by our backend.** It is real in MuJoCo
(9.3-9.4% of pixels change between 0 and 1). It sets ``geom_rgba[i][3] = 0.3`` on every ``robot0:``
geom, and ``mujoco_warp``'s render kernel reads only ``color[0..2]`` and packs a constant 255 — it
has no alpha path at all. That is the same measured fact that makes this family hide the four
translucent ``robot0:mocap`` markers instead of drawing them opaque; see ``FetchBase.parity_note``.
This is a backend limitation with a named mechanism, not a modelling choice, and it is recorded
rather than approximated: painting the arm 30%-grey would change pixels that the alpha-blind
renderer cannot get right anyway.

THE THREE LEAVES THAT DO MAP, AND THE TWO UNIT SHIFTS IN THEM
-------------------------------------------------------------

``table.color`` is ours by name and by units (both write the table geom's colour in [0,1] floats).

``background.color`` -> ``floor.color`` is a **rename with a measured residual**. swm's "background"
is two things at once: the ``floor_mat`` geoms AND the skybox texture, recoloured by overwriting
``model.tex_data`` and re-uploading it. Measured split at 480 px, floor versus floor+skybox:
the floor is **9.2-9.4%** of the frame and the skybox a further **61.7-73.0%**. We render the floor
half exactly and offer no leaf for the skybox half.

**The reason for that was measured and replaced** — see
``pre-repository-cleanup:tests/fixtures/regression/join/step8/capability/fetch_skybox_probe.json``. It is NOT, as this module and
``FetchBase.make_variation_space`` used to say, that a per-value skybox costs one render context
per distinct value: the delivered frame's skybox comes from the MuJoCo-baked PLATE, not from the ``mujoco_warp``
pass, so a texel substitution on the plate rig reaches it with no second context, and that is
exactly the operation ``stable_worldmodel`` itself performs (it does not recompile either). What
blocks the leaf is the DEFAULT: swm FLATTENS the skybox to one colour and ours is a 16-colour
gradient, so no flat-fill leaf value reproduces our registered default appearance. Offering the leaf
moves this family's default frame, which is a frozen pixel oracle — a re-freeze decision, escalated
rather than taken.

The leaf is therefore declared ``partial``, not ``mapped``: the rename is real and the floor half is
genuinely what swm writes, and the skybox half is genuinely missing, and
``VariationMap.classification`` says so, so a generated completeness table cannot count this leaf as
full support. That field exists because of this leaf — Sami's ruling is that a feasible, semantically
required leaf is NOT declared unsupported merely because a renderer capability is missing. The
skybox is a missing renderer capability recorded rather than approximated; until a re-freeze
decision lands, the truthful label is ``partial`` and the measured residual travels with it.

``light.intensity`` -> ``light.diffuse`` is a **scale**, and the factor is not 1. swm assigns the
light's diffuse ABSOLUTELY (``light_diffuse[0][:3] = [i,i,i]``, ``Box(0, 1)``, declared default
0.7); ours is a MULTIPLIER on the model's own ``light_diffuse``, which is **0.8** on all four
scenes, with default 1.0 — because that is what makes "our default is the official appearance" an
identity a test can assert. So ``ours = theirs * 1.25`` and swm's full ``[0, 1]`` range maps onto
``[0, 1.25]``, comfortably inside our ``Box(0, 2)``: **the whole swm range is representable.**

WHAT OUR SPACE HAS THAT SWM'S DOES NOT
--------------------------------------

``goal.color`` (the red target sphere), ``arm.color`` (the five ``robot0:blue`` links) and the three
working camera knobs. swm has no counterpart for any of them; ``rendering.transparent_arm`` is the
nearest thing to an arm-appearance leaf and it is an alpha, not a colour.

FOUR SWM LEAVES ARE OUR STATE, NOT OUR STYLE
---------------------------------------------

``agent.start_position``, ``goal.start_position``, ``block.start_position`` and ``block.angle`` are
RESET DISTRIBUTIONS in swm — the wrapper writes them into ``initial_gripper_xpos``, ``data.qpos``
and ``env.goal`` at reset time. All four are live pixels, and all four are dimensions of our state
vector (``qpos + goal``), which is the same classification PushT's ``block.angle`` and
``agent.start_position`` already carry. A caller who wants them sets the state, which is strictly
more expressive than a reset distribution.

A MEASURED UPSTREAM CAVEAT THAT IS NOT A MAPPING QUESTION
----------------------------------------------------------

``FetchWrapper._apply_visual_variations`` unbinds materials (``model.geom_matid[i] = -1``) on the
table, floor and object geoms the first time ANY of its six visual leaves is active, permanently,
for the life of that env. That drops those geoms from their material colours (table 0.93, floor 0.2,
object 0.2) to the compiler default 0.5 grey: measured at **19.0-32.3% of the frame, up to 102/255**,
with every leaf still at its own declared default. Since ``DEFAULT_VARIATIONS`` activates five
visual leaves, a plain ``reset()`` on a swm Fetch id always triggers it. It means the appearance a
swm Fetch env shows after any variation activity is NOT the registered default appearance, and it is
why swm's declared leaf defaults (table 0.3, object 0.8/0.1/0.1, background 0.1, light 0.7) are not
the model's values (0.93, 0.2, 0.2 + skybox, 0.8). Our defaults are read off the compiled model, so
OUR default reproduces the official picture; a swm config's default VALUES do not, and
``style_from_variation`` is right to carry them through unchanged. Recorded here so that a later
parity number computed through the swm wrapper is not mistaken for a stable-envs regression.
"""

from __future__ import annotations

from ..variation import VariationMap

#: ``model.light_diffuse[0]`` on all four Fetch scenes, read off the compiled model (identical in
#: reach/push/slide/pick_and_place). swm's ``light.intensity`` is this quantity absolutely; ours is a
#: multiplier on it, so ``ours = theirs / LIGHT_DIFFUSE_0``.
LIGHT_DIFFUSE_0 = 0.8
#: ``ours = theirs * LIGHT_FACTOR``. Exactly representable in binary, so ``theirs = 0.8`` round-trips
#: to ``ours = 1.0`` with no error and the default stays byte-identical through the mapping.
LIGHT_FACTOR = 1.0 / LIGHT_DIFFUSE_0

#: The four scenes, and whether the compiled model carries an object geom under
#: ``block_mat``/``puck_mat``. Reach does not, which is the whole of its per-task difference.
SCENE_HAS_OBJECT = {"Reach": False, "Push": True, "Slide": True, "PickAndPlace": True}
#: Suffixes that select a reward (``Dense``) or an observation container (``Dict``) and reach no
#: pixel, so the three ids of a scene share one map body.
SUFFIXES = ("", "Dense", "Dict")

_DEAD_CAMERA = (
    "DEAD UPSTREAM (measured 0/255 over 0% of the frame at +-10 deg on all four scenes): the "
    "wrapper rotates model.cam_pos for all 4 MODEL cameras, and MujocoFetchEnv renders through the "
    "FREE camera from DEFAULT_CAMERA_CONFIG, which is never one of them. We have a working camera "
    "(camera.azimuth_delta / camera.elevation_delta / camera.distance_scale); mapping onto it would "
    "move our frame where the reference's does not, so it is classified rather than fabricated"
)
_ARM_ALPHA = (
    "UNSUPPORTED BY BACKEND. Live in MuJoCo (9.3-9.4% of pixels between 0 and 1) and unreachable "
    "here: it writes geom_rgba[.][3]=0.3 on every robot0: geom, and mujoco_warp's render kernel "
    "reads color[0..2] and packs a constant 255 -- no alpha path at all. Same measured fact that "
    "makes this family hide the four translucent robot0:mocap markers; see FetchBase.parity_note"
)
_BLOCK_MASS = (
    "NOT APPLICABLE: dynamics only. Measured at 0.01 vs 50.0 kg on Push, Slide and PickAndPlace: "
    "0/255 over 0% of the frame. Our state is qpos + goal and carries no mass"
)
_STATE = "a RESET DISTRIBUTION in swm, and a dimension of our STATE (qpos + goal) here -- "

# The four unmapped buckets below are the four-way taxonomy, and this module uses them rather than the
# older undifferentiated ``swm_only``: ``classification()`` reads them directly, so the Gate-F matrix
# and every generated support table get the distinction between "upstream moves and we cannot follow"
# (unsupported), "upstream does not move at all" (dead_upstream) and "not a question a renderer
# answers" (not_applicable) without anybody parsing a reason string for a shouted prefix.

#: swm leaves with a real upstream effect that THIS cell cannot express. One, and it names its
#: mechanism.
_UNSUPPORTED = {"rendering.transparent_arm": _ARM_ALPHA}

#: swm leaves that reach no pixel and no state IN SWM ITSELF, each carrying the measurement.
_DEAD_UPSTREAM = {"camera.angle_delta": _DEAD_CAMERA}
#: Reach's only extra: swm still OFFERS object.color there and it reaches nothing.
_REACH_OBJECT = {
    "object.color": "DEAD UPSTREAM on Reach (measured 0/255 over 0% of the frame between two "
    "maximally different colours): the wrapper resolves the object through the "
    "block_mat/puck_mat material, and reach.xml compiles neither onto a geom, so "
    "the write lands on an empty geom list. Live on the other three scenes",
}

#: swm leaves that are not appearance questions at all here: reset distributions that are dimensions
#: of our STATE, and one pure dynamics parameter. The same classification PushT already carries.
_NOT_APPLICABLE = {
    "agent.start_position": _STATE + "the wrapper writes it into initial_gripper_xpos; ours is the "
    "arm's qpos. Note its declared default (1.3418, 0.7491) is not the "
    "env's own initial gripper xy: applying it at its default moves up to "
    "55/255 on Reach",
    "goal.start_position": _STATE + "the wrapper writes env.goal and model.site_pos[target0]; ours "
    "is state dimensions goal_x/goal_y/goal_z, which the goal sphere reads",
}
#: ...and the three that only exist on a scene with a block.
_BLOCK_NOT_APPLICABLE = {
    "block.start_position": _STATE + "the wrapper writes data.qpos[object0:joint][0:2]; ours is "
    "state dimensions object_x/object_y",
    "block.angle": _STATE + "the wrapper writes the object's quaternion about z; ours is state "
    "dimensions object_quat_w..object_quat_z",
    "block.mass": _BLOCK_MASS,
}

#: Knobs we expose that ``FetchWrapper`` has no leaf for.
_OURS_ONLY = {
    "goal.color": "the red target sphere's colour; swm exposes no goal appearance, only goal.start_position",
    "arm.color": "the five robot0:blue links; swm's only arm leaf is rendering.transparent_arm, "
    "which is an alpha and not a colour",
    "camera.distance_scale": "a working camera knob; swm's camera.angle_delta is dead upstream "
    "(see swm_only) and has no distance term at all",
    "camera.azimuth_delta": "a working camera knob; see camera.distance_scale",
    "camera.elevation_delta": "a working camera knob; see camera.distance_scale",
}

#: ``our path -> swm path`` for the leaves that mean the same thing under different names. The
#: residual on this one is measured and stated in the module docstring: swm's "background" is the
#: floor geoms (9.2-9.4% of the frame, which we render exactly) AND the skybox texture (a further
#: 61.7-73.0%, which we do not render because a per-value skybox costs one render context per value).
_RENAMED = {"floor.color": "background.color"}
#: ``swm path -> what is NOT reproduced``. The mapping above is kept — the floor half is what swm
#: writes and we write it exactly — and the classification is downgraded from ``mapped`` to
#: ``partial`` so that nothing generated from this map can report the leaf as fully supported.
_PARTIAL = {
    "background.color": (
        "the floor half only. swm's background is the floor_mat geoms AND the skybox texture; the "
        "floor is 9.2-9.4 % of the frame at 480 px and we reproduce it exactly, the skybox is a "
        "further 61.7-73.0 % and we do not offer a leaf for it. "
        "THE REASON CHANGED IN STEP 8 AND THE NEW ONE IS NARROWER. The old one -- that a per-value "
        "skybox costs one render context per distinct value -- was measured WRONG, and the "
        "measurement is "
        "pre-repository-cleanup:tests/fixtures/regression/join/step8/capability/fetch_skybox_probe.json. The skybox of the DELIVERED "
        "frame comes from the MuJoCo-baked plate, not from the mujoco_warp pass (the composite takes "
        "mujoco_warp only under the mover mask), so substituting the skybox texels on the plate rig "
        "and re-uploading moves 62.2-74.5 % of the frame with no second context -- and that IS the "
        "operation swm performs, which does not recompile either. "
        "What blocks the leaf is the DEFAULT, not the mechanism. swm FLATTENS the skybox to one "
        "colour, and this family's skybox is a 16-colour gradient (.44 .85 .56 to .46 .87 .58 over "
        "6,144 texels), so no value of a flat-fill leaf reproduces our registered default "
        "appearance and no two-colour masked substitution reproduces swm's flat sky. Offering the "
        "leaf therefore moves this family's DEFAULT frame, which is a frozen pixel oracle on the "
        "frozen corpora. That is a re-freeze decision, not an "
        "implementation decision, and it is escalated rather than taken."
    ),
}
#: ``our path -> (swm path, factor)`` with ``ours = theirs * factor``.
_SCALED = {"light.diffuse": ("light.intensity", LIGHT_FACTOR)}


def _map_for(scene: str, suffix: str) -> VariationMap:
    """The map for one of the twelve ids, from the family vocabulary plus the scene's difference."""
    has_object = SCENE_HAS_OBJECT[scene]
    identical = ("table.color",) + (("object.color",) if has_object else ())
    not_applicable = dict(_NOT_APPLICABLE)
    dead_upstream = dict(_DEAD_UPSTREAM)
    if has_object:
        not_applicable.update(_BLOCK_NOT_APPLICABLE)
    else:
        dead_upstream.update(_REACH_OBJECT)
    return VariationMap(
        swm_id=f"swm/Fetch{scene}{suffix}-v3",
        identical=identical,
        renamed=dict(_RENAMED),
        scaled=dict(_SCALED),
        partial=dict(_PARTIAL),
        unsupported=dict(_UNSUPPORTED),
        dead_upstream=dead_upstream,
        not_applicable=not_applicable,
        ours_only=dict(_OURS_ONLY),
    )


#: our class name -> how its appearance relates to the official swm env's. Twelve entries, one per
#: registered Fetch id, built from one vocabulary so a change to the family cannot reach eleven of
#: the twelve.
MAPS: dict[str, VariationMap] = {
    f"Fetch{scene}{suffix}": _map_for(scene, suffix) for scene in SCENE_HAS_OBJECT for suffix in SUFFIXES
}


def swm_variation_space(swm_id: str):
    """The live ``variation_space`` of a ``swm/Fetch*-v3`` id.

    Constructs the id and hands the live env to ``interop.swm_variation_space``, which
    walks the wrapper chain. It has to be walked rather than read off ``unwrapped``: ``FetchWrapper``
    is a ``gym.Wrapper`` around the official ``MujocoFetchEnv``, so ``unwrapped`` is the official env
    and has no ``variation_space`` at all (it raised ``AttributeError`` for all twelve ids until
    ``check_mapping`` is fixed centrally). This is the id-taking convenience over that
    function, and deliberately not a second implementation of it — the discovery rule lives in one
    place so a future wrapper change cannot fix ``check_mapping`` and leave this module behind.

    Closes the env on every path, including the failure path.
    """
    import gymnasium
    import stable_worldmodel  # noqa: F401

    from ..variation import swm_variation_space as _walk

    env = gymnasium.make(swm_id)
    try:
        return _walk(env)
    finally:
        env.close()
