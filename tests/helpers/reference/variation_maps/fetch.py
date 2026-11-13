"""``VariationMap``s for the twelve ``swm/Fetch*-v3`` ids.

The ids cover four gymnasium-robotics scenes; the ``Dense`` and ``Dict`` suffixes change the reward
and the observation container but no pixel, and the variation spaces are identical per scene, so the
twelve maps come from one vocabulary plus a per-scene difference.

Dead upstream: ``camera.angle_delta`` rotates ``model.cam_pos`` for the model cameras, but
``MujocoFetchEnv`` renders through the free camera (0/255 at +-10 deg on all four scenes).
``object.color`` on Reach targets ``block_mat``/``puck_mat``, which ``reach.xml`` binds to no geom.
Our reset still accepts ``camera.angle_delta``, because naming it unbinds the materials (below).

``background.color`` drives two of our leaves, ``floor.color`` and ``sky.color``. Our ``sky.color``
default draws the registered gradient, which has no swm spelling; any other value fills the sky flat,
as swm does. ``light.intensity`` is absolute in swm and a multiplier on the model's 0.8 here, so
``ours = theirs * 1.25``. The start-position, ``block.angle`` and ``block.mass`` leaves are reset
options: the first four land in State, the mass is installed in the simulator.

Reset differences: any of swm's six visual leaves unbinds the table, floor and object materials
(``geom_matid = -1``), which our reset reproduces for the episode. swm keeps them unbound, and keeps
a moved ``initial_gripper_xpos``, for the life of the env; ours restores both on the next reset that
does not name them. A plain swm ``reset()`` samples ``DEFAULT_VARIATIONS``; ours samples only the
leaves named in ``options["variation"]``.
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

#: swm leaves that reach no pixel and no state in swm itself, each carrying the measurement.
_DEAD_UPSTREAM = {"camera.angle_delta": _DEAD_CAMERA}
#: On Reach swm offers object.color, and it reaches nothing.
_REACH_OBJECT = {
    "object.color": "DEAD UPSTREAM on Reach (measured 0/255 over 0% of the frame between two "
    "maximally different colours): the wrapper resolves the object through the "
    "block_mat/puck_mat material, and reach.xml compiles neither onto a geom, so "
    "the write lands on an empty geom list. Live on the other three scenes",
}

#: Reset options with swm's names that land in our State rather than in our style.
STATE_LEAVES = ("agent.start_position", "goal.start_position")
BLOCK_STATE_LEAVES = ("block.start_position", "block.angle")
_NOTES = {
    "agent.start_position": "a reset option that writes initial_gripper_xpos before the source reset; "
    "it lands in State (qpos, task.initial_gripper_xpos), and an episode that does not name it "
    "returns to the registered anchor where swm keeps the moved one",
    "goal.start_position": "a reset option that lands in State (goal)",
    "background.color": "drives floor.color and sky.color together; sky.color's default, the "
    "registered gradient, has no swm spelling",
    "rendering.transparent_arm": "alpha 0.3 on every robot0: geom; MJWarp blends every front face "
    "in MuJoCo's transparent order over the robot-free scene",
}
_BLOCK_NOTES = {
    "block.start_position": "a reset option that lands in State (qpos of object0:joint)",
    "block.angle": "a reset option that lands in State (object quaternion about z)",
    "block.mass": "installed in the simulator on reset; a dynamics leaf that reaches no pixel",
}

#: Knobs we expose that ``FetchWrapper`` has no leaf for.
_OURS_ONLY = {
    "goal.color": "the red target sphere's colour; swm exposes no goal appearance, only goal.start_position",
    "arm.color": "the five robot0:blue links; swm's only arm leaf is rendering.transparent_arm, "
    "which is an alpha and not a colour",
    "camera.distance_scale": "a working camera knob; swm's camera.angle_delta is dead upstream "
    "(see dead_upstream) and has no distance term at all",
    "camera.azimuth_delta": "a working camera knob; see camera.distance_scale",
    "camera.elevation_delta": "a working camera knob; see camera.distance_scale",
}

#: ``our path -> swm path``. swm's ``background.color`` is both the floor and the sky; the floor
#: comes last so that the inverse mapping reads the floor color.
_RENAMED = {"sky.color": "background.color", "floor.color": "background.color"}
#: ``our path -> (swm path, factor)`` with ``ours = theirs * factor``.
_SCALED = {"light.diffuse": ("light.intensity", LIGHT_FACTOR)}


def _map_for(scene: str, suffix: str) -> VariationMap:
    """The map for one of the twelve ids, from the family vocabulary plus the scene's difference."""
    has_object = SCENE_HAS_OBJECT[scene]
    identical = ("table.color", "rendering.transparent_arm") + STATE_LEAVES
    notes = dict(_NOTES)
    dead_upstream = dict(_DEAD_UPSTREAM)
    if has_object:
        identical += ("object.color", "block.mass") + BLOCK_STATE_LEAVES
        notes.update(_BLOCK_NOTES)
    else:
        dead_upstream.update(_REACH_OBJECT)
    return VariationMap(
        swm_id=f"swm/Fetch{scene}{suffix}-v3",
        identical=identical,
        renamed=dict(_RENAMED),
        scaled=dict(_SCALED),
        dead_upstream=dead_upstream,
        notes=notes,
        ours_only=dict(_OURS_ONLY),
    )


#: our class name -> its map, one per registered Fetch id.
MAPS: dict[str, VariationMap] = {
    f"Fetch{scene}{suffix}": _map_for(scene, suffix) for scene in SCENE_HAS_OBJECT for suffix in SUFFIXES
}


def swm_variation_space(swm_id: str):
    """The live ``variation_space`` of a ``swm/Fetch*-v3`` id.

    ``FetchWrapper`` wraps the official ``MujocoFetchEnv``, so ``unwrapped`` has no variation space;
    this delegates to ``variation.swm_variation_space``, which walks the wrapper chain. Closes the env
    on every path.
    """
    import gymnasium
    import stable_worldmodel  # noqa: F401

    from ..variation import swm_variation_space as _walk

    env = gymnasium.make(swm_id)
    try:
        return _walk(env)
    finally:
        env.close()
