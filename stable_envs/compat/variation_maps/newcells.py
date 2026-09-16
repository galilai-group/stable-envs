"""VariationMaps for the Step-7 new cells and the OGB / Rocket / maze families.

Five swm ids: ``OGBCube-v0``, ``OGBScene-v0``, ``PFRocketLanding-v0``, ``OGBMaze-v0`` and
``SimplePointMaze-v0``. Every leaf each of them offers is placed in exactly one of four buckets, and
the bucket is a MEASUREMENT rather than a reading of the upstream docstring:

``mapped``
    we render it, and ``VARIATION_MAPS`` carries the path pair that reaches it.
``unsupported``
    reachable in principle, deliberately not offered; the reason names the mechanism that would have
    to change.
``dead_upstream``
    swm samples the leaf and no pixel of the OFFICIAL renderer moves. Every such claim here cites the
    probe that measured it, with the pixel counts, because "we could not find a reader" and "we
    measured that nothing reads it" are different claims and only the second one is evidence.
``not_applicable``
    the leaf describes something this registration does not have (a goal marker that is switched
    off, an appearance that is our STATE rather than our style).

``LEAF_CLASSIFICATION`` below carries that four-way verdict with its evidence.
``VariationMap.swm_only`` can only hold one flat reason string per leaf, so the distinction between
"we chose not to" and "upstream measured dead" would be lost if the reasons were the only record;
the structured table is what the docs page and ``leaf_classification.json`` are generated from.
See ``docs/research/step8_variations_newcells.md``.

WHY FOUR OF THE FIVE IDS MAP NOTHING BUT STATE
---------------------------------------------
``OGBCube``/``OGBScene`` declare ``make_variation_space() -> Dict({})``. That is not an oversight
and it is not laziness: their frame is a MuJoCo-baked static plate composited with one mujoco_warp
launch for the movers, and BOTH halves are built from a single compiled ``MjModel`` whose appearance
arrays ``_appearance_digest`` asserts never change. swm applies ``floor.color``, ``agent.color``,
``cube.size``, ``camera.angle_delta`` and ``light.intensity`` by rewriting the MJCF XML and
RECOMPILING (``CubeEnv.modify_mjcf_model`` -> ``mark_dirty`` -> ``compile_model``), so honouring them
per environment means one compiled model, one plate rig and one captured mujoco_warp graph PER
APPEARANCE VECTOR. Which of those are cheap and which are a renderer change is recorded leaf by leaf
below, measured rather than guessed.
"""

from __future__ import annotations

from ..worldmodel import VariationMap

__all__ = ["NEWCELL_VARIATION_MAPS", "LEAF_CLASSIFICATION"]


# ====================================================================== evidence vocabulary

#: The probes these classifications rest on. Quoted in the reasons so a reader can re-run the one
#: that produced a verdict rather than taking the verdict on trust.
PROBES = {
    "rocket_liveness": (
        "examples/results/step8/variations-newcells/probes/live_rocket.json -- each leaf resampled "
        "alone under 5 seeds, 3 frames of 480x480 (691,200 px) per reset, compared against the "
        "no-variation reset"),
    "rocket_pad": (
        "examples/results/step8/variations-newcells/probes/pad_color.json -- pad.color over 3 seeds, "
        "120 steps, 6 frames of 480x480, plus pybullet getVisualShapeData on the landing pad body"),
    "ogb_model": (
        "examples/results/step8/variations-newcells/probes/ogb_model_{Cube,Scene}.json -- every "
        "MjModel array diffed between a no-variation reset and a one-leaf-extreme reset, with each "
        "changed geom classified as a MOVER (drawn per world by mujoco_warp) or STATIC (baked into "
        "the plate)"),
    "ogb_agent_color": (
        "examples/results/step8/variations-newcells/probes/agent_color_dirty_flag.json -- "
        "agent.color=RED alone vs agent.color=RED co-varied with light.intensity"),
    "scene_noise": (
        "examples/results/step8/variations-newcells/probes/scene_noise.json -- two IDENTICAL "
        "seed=0 SceneEnv resets diffed, to separate reset noise from leaf effect"),
}


# ====================================================================== OGB manipulation

#: Leaves that reach the OGB frame ONLY through qpos / mocap_pos / the button bits, i.e. through our
#: STATE. swm samples them at reset and they are part of the registered default variation set, which
#: is why `sample_states` draws them explicitly (`_variation_values`) instead of letting swm resample
#: them from OS entropy.
_OGB_STATE_LEAVES = {
    "agent.ee_start_position": "a RESET pose for the arm; it reaches the frame only through qpos, "
                               "which is our state. sample_states draws it explicitly",
    "cube.start_position": "reset placement of the cube; reaches the frame through qpos (state)",
    "cube.start_yaw": "reset orientation of the cube; reaches the frame through qpos (state)",
    "cube.goal_position": "reset placement of the goal marker; reaches the frame through the mocap "
                          "pose, which is state",
    "cube.goal_yaw": "reset orientation of the goal marker; mocap pose, i.e. state",
}

#: The appearance half. Each entry names the array swm's recompile actually writes, and whether the
#: geoms it writes are movers or static, because that is what decides the cost of honouring it.
_OGB_APPEARANCE = {
    "floor.color": (
        "the checker floor texture (swm rewrites the MJCF grid texture rgb1/rgb2 and recompiles). "
        "MEASURED: it changes exactly ONE model array, tex_data, and nothing else -- the floor is "
        "STATIC, so this is a plate-only leaf and the tractable one. Not offered yet because the "
        "plate cache and _PlateRig are keyed on (cell, resolution) alone; honouring it means adding "
        "the appearance vector to that key, which is an additive change this workstream is NOT "
        "authorised to make to a certified renderer without a pixel re-baseline. ESCALATED"),
    "light.intensity": (
        "the global directional light's diffuse term (MEASURED: writes light_diffuse row 0 and "
        "nothing else). It lights BOTH halves of the frame, so it is not plate-only: the plate can "
        "be re-baked per value, but the mover pass would need light_diffuse to vary PER WORLD inside "
        "one mujoco_warp context. That is a renderer change, not a cache key. ESCALATED"),
    "camera.angle_delta": (
        "perturbs the front_pixels camera's xyaxes (MEASURED: writes cam_mat0 and cam_quat). Same "
        "shape of problem as light.intensity -- the plate is a picture taken FROM this camera and "
        "the mover pass renders through the same model camera, so per-environment camera angles "
        "need a per-world camera in the mujoco_warp context. ESCALATED"),
    "cube.size": (
        "the cube's geom half-extents (MEASURED: writes geom_size on 2 MOVER geoms plus 12 derived "
        "inertia/bounding arrays). Movers are drawn by mujoco_warp from the context's shared model, "
        "so per-environment sizes need a per-world geom_size. ESCALATED"),
    "agent.color": (
        "the arm's `ur5e/robotiq/black` and `ur5e/robotiq/pad_gray` materials, which are carried "
        "only by MOVER geoms. UPSTREAM DEFECT: applied alone it reaches NO model array at all -- "
        "CubeEnv.modify_mjcf_model computes `agent_color_changed = np.allclose(current, desired)`, "
        "which is True exactly when the colour did NOT change, so the dirty flag is never set and "
        "the staged MJCF edit is never compiled. MEASURED: agent.color=(1,0,0) alone leaves mat_rgba "
        "at the default (0.61,0.28,0.82); the same request co-varied with light.intensity (which "
        "sets the flag correctly) does turn the arm red. Conditionally dead upstream"),
    "cube.color": (
        "the cube geoms' rgba. MEASURED: swm writes it DIRECTLY on the compiled model in "
        "initialize_episode with no recompile, on 2 MOVER geoms (Cube). This is the one appearance "
        "leaf reachable through machinery this cell ALREADY has -- _upload_button_colours writes "
        "per-world geom_rgba/mat_rgba into the mujoco_warp context for the Scene buttons by exactly "
        "this route. Not offered yet only because the cell's style space is empty and adding the "
        "first style dim moves the declaration (style_dim, style_digest) for a certified cell"),
    "lock_color": (
        "Scene only: the two lock materials, carried by 6 MOVER geoms. Same mechanism and same "
        "answer as cube.color -- reachable through the existing per-world rgba upload. Attribution "
        "caveat: SceneEnv's reset is not seed-reproducible, so geom_rgba/mat_rgba move between two "
        "IDENTICAL resets and this probe cannot separate the leaf from that noise by array diff "
        "alone; the mover-geom membership is the solid half of the measurement"),
}

_OGB_CUBE_SWM_ONLY = {
    **_OGB_STATE_LEAVES,
    **{k: v for k, v in _OGB_APPEARANCE.items() if k != "lock_color"},
}
_OGB_SCENE_SWM_ONLY = {**_OGB_STATE_LEAVES, **_OGB_APPEARANCE}


# ====================================================================== Rocket

_ROCKET_SWM_ONLY = {
    "environment.sky_color": (
        "DEAD UPSTREAM, MEASURED. swm declares an RGBBox and nothing ever reads it: the string "
        "`sky_color` occurs exactly once in stable_worldmodel (its own declaration) and not at all "
        "in PyFlyt, and pybullet's TinyRenderer clear colour is fixed. Probe: resampling it alone "
        "over 5 seeds spanning the RGB cube -- drawn values as far apart as (43,202,87) and "
        "(149,165,44) -- moved 0 of 691,200 pixels, max channel difference 0, and all three frame "
        "stacks hash to the same digest. Offering it would advertise a knob that cannot work"),
    "environment.start_height_ratio": (
        "a RESET distribution for the drop altitude, not an appearance. It reaches the frame only by "
        "moving the rocket, which is our state; sample_states covers 1.9-489 m directly"),
    "environment.start_horizontal_offset": (
        "a RESET distribution for the lateral start, not an appearance; our state carries the "
        "position and sample_states covers lateral drift to 190 m"),
    "environment.start_tilt": (
        "a RESET distribution for the initial attitude; our state carries the quaternion and "
        "sample_states draws tumbling and near-upright attitudes"),
}


# ====================================================================== the maps

#: OUR class name -> how its appearance relates to the official swm env's.
NEWCELL_VARIATION_MAPS: dict[str, VariationMap] = {
    "OGBCube": VariationMap(
        swm_id="swm/OGBCube-v0",
        swm_only=_OGB_CUBE_SWM_ONLY,
    ),
    "OGBScene": VariationMap(
        swm_id="swm/OGBScene-v0",
        swm_only=_OGB_SCENE_SWM_ONLY,
    ),
    "PFRocketLanding": VariationMap(
        swm_id="swm/PFRocketLanding-v0",
        # Same dotted path and the same 0-255 units on both sides. `pad.color` is included on the
        # strength of the MODEL measurement, not a pixel one: swm's landing pad is never in the
        # reference render camera's view in the sampled regime, so a pixel probe cannot see it, but
        # getVisualShapeData confirms our material_table reproduces the 3-decimal URDF quantisation
        # swm applies (e.g. (104,60,3) -> 0.408/0.235/0.012, matching f"{v:.3f}").
        identical=("rocket.body_color", "rocket.fin_color", "rocket.leg_color",
                   "rocket.booster_color", "pad.color"),
        swm_only=_ROCKET_SWM_ONLY,
        ours_only={
            "flame.ignited": "swm has no ignition leaf: PyFlyt recolours the flame link from the "
                             "thrust command at step time, so upstream it is a consequence of the "
                             "action rather than a knob. We carry it as a style bit because the "
                             "renderer is a pure function of (state, style) and the flame's colour "
                             "is not derivable from the 13-dim state",
        },
    ),
}


# ====================================================================== structured classification

def _cls(bucket, reason, evidence=None, mechanism=None):
    d = {"classification": bucket, "reason": reason}
    if evidence:
        d["evidence"] = evidence
    if mechanism:
        d["mechanism"] = mechanism
    return d


LEAF_CLASSIFICATION: dict[str, dict] = {
    "swm/PFRocketLanding-v0": {
        "our_envs": ["PFRocketLanding"],
        "leaves": {
            "rocket.body_color": _cls("mapped", "style dim, identical path and units (0-255)",
                                      PROBES["rocket_liveness"] + " -- 35,551 px moved, maxdiff 241"),
            "rocket.fin_color": _cls("mapped", "style dim, identical path and units",
                                     PROBES["rocket_liveness"] + " -- 16,858 px moved, maxdiff 186"),
            "rocket.leg_color": _cls("mapped", "style dim, identical path and units",
                                     PROBES["rocket_liveness"] + " -- 228 px moved, maxdiff 218"),
            "rocket.booster_color": _cls("mapped", "style dim, identical path and units",
                                         PROBES["rocket_liveness"] + " -- 573 px moved, maxdiff 191"),
            "pad.color": _cls(
                "mapped",
                "style dim, identical path and units. Reaches the reference MODEL (verified) but no "
                "reference PIXEL in the sampled regime: the landing pad is not in the render "
                "camera's view over 120 steps from a 430 m drop",
                PROBES["rocket_pad"] + " -- 0 of 1,382,400 px moved; getVisualShapeData shows the "
                "quantised rgba applied, e.g. (104,60,3) -> (0.408,0.235,0.012,1.0)",
                mechanism="URDF material rewrite, 3-decimal quantised, reproduced by material_table"),
            "environment.sky_color": _cls(
                "dead_upstream",
                "swm samples it and no reader exists anywhere in swm, PyFlyt or pybullet",
                PROBES["rocket_liveness"] + " -- 0 of 691,200 px moved over 5 seeds spanning the "
                "RGB cube; identical frame digests",
                mechanism="declared once in pyflyt_rocketlanding.py:127 and read nowhere; "
                          "TinyRenderer's clear colour is fixed"),
            "environment.start_height_ratio": _cls(
                "not_applicable", "a reset distribution; the resulting pose is our STATE",
                PROBES["rocket_liveness"] + " -- moves pixels (18 px) only by moving the rocket"),
            "environment.start_horizontal_offset": _cls(
                "not_applicable", "a reset distribution; the resulting pose is our STATE",
                PROBES["rocket_liveness"] + " -- moves pixels (28 px) only by moving the rocket"),
            "environment.start_tilt": _cls(
                "not_applicable", "a reset distribution; the resulting attitude is our STATE",
                PROBES["rocket_liveness"] + " -- moves pixels (55,257 px) only by re-posing the rocket"),
        },
    },
    "swm/OGBCube-v0": {
        "our_envs": ["OGBCube"],
        "leaves": {
            **{k: _cls("not_applicable", v, PROBES["ogb_model"] + " -- 0 appearance arrays changed")
               for k, v in _OGB_STATE_LEAVES.items()},
            "floor.color": _cls("unsupported", _OGB_APPEARANCE["floor.color"],
                                PROBES["ogb_model"] + " -- changes tex_data only (1 row, max 235)",
                                mechanism="plate cache key extension; escalation candidate"),
            "light.intensity": _cls("unsupported", _OGB_APPEARANCE["light.intensity"],
                                    PROBES["ogb_model"] + " -- changes light_diffuse only (1 row)",
                                    mechanism="per-world model array in the mjwarp context; "
                                              "escalation candidate"),
            "camera.angle_delta": _cls("unsupported", _OGB_APPEARANCE["camera.angle_delta"],
                                       PROBES["ogb_model"] + " -- changes cam_mat0 and cam_quat",
                                       mechanism="per-world camera; escalation candidate"),
            "cube.size": _cls("unsupported", _OGB_APPEARANCE["cube.size"],
                              PROBES["ogb_model"] + " -- geom_size on 2 MOVER geoms + 12 derived",
                              mechanism="per-world geom_size; escalation candidate"),
            "agent.color": _cls("dead_upstream", _OGB_APPEARANCE["agent.color"],
                                PROBES["ogb_agent_color"] + " -- 0 model arrays changed when applied "
                                "alone; mat_rgba rows 8 and 9 move only when co-varied",
                                mechanism="inverted dirty-flag test in CubeEnv.modify_mjcf_model; "
                                          "conditionally dead"),
            "cube.color": _cls("unsupported", _OGB_APPEARANCE["cube.color"],
                               PROBES["ogb_model"] + " -- geom_rgba on 2 MOVER geoms, no recompile",
                               mechanism="reachable through the EXISTING per-world rgba upload; the "
                                         "cheapest of the six and the first one to offer"),
        },
    },
    "swm/OGBScene-v0": {
        "our_envs": ["OGBScene"],
        "leaves": {
            **{k: _cls("not_applicable", v, PROBES["ogb_model"])
               for k, v in _OGB_STATE_LEAVES.items()},
            "floor.color": _cls("unsupported", _OGB_APPEARANCE["floor.color"],
                                PROBES["ogb_model"] + " -- changes tex_data only"),
            "light.intensity": _cls("unsupported", _OGB_APPEARANCE["light.intensity"],
                                    PROBES["ogb_model"] + " -- changes light_diffuse"),
            "camera.angle_delta": _cls("unsupported", _OGB_APPEARANCE["camera.angle_delta"],
                                       PROBES["ogb_model"] + " -- changes cam_mat0 and cam_quat"),
            "cube.size": _cls("unsupported", _OGB_APPEARANCE["cube.size"],
                              PROBES["ogb_model"] + " -- geom_size on 2 MOVER geoms"),
            "agent.color": _cls("dead_upstream", _OGB_APPEARANCE["agent.color"],
                                PROBES["ogb_agent_color"] + " -- 0 model arrays changed beyond the "
                                "measured Scene reset-noise set when applied alone",
                                mechanism="inverted dirty-flag test in CubeEnv.modify_mjcf_model; "
                                          "conditionally dead"),
            "cube.color": _cls("unsupported", _OGB_APPEARANCE["cube.color"],
                               PROBES["ogb_model"] + " -- geom_rgba on 3 MOVER geoms"),
            "lock_color": _cls("unsupported", _OGB_APPEARANCE["lock_color"],
                               PROBES["ogb_model"] + " -- mat_rgba on 2 materials carried by 6 "
                               "MOVER geoms; " + PROBES["scene_noise"]),
        },
    },
}


# ====================================================================== the leaf inventory

#: What each swm id's ``variation_space.names()`` ACTUALLY returned, enumerated from the live wrapper
#: on 2026-09-16. Recorded rather than only asserted live because ``stable_worldmodel`` is not
#: importable in every environment this repo is tested in -- without this, the completeness test
#: (every offered leaf is classified) would silently skip exactly where it matters. The live test
#: runs too, wherever swm IS importable, and asserts this table still matches the wrapper.
RECORDED_SWM_LEAVES: dict[str, tuple[str, ...]] = {
    "swm/OGBCube-v0": (
        "agent.color", "agent.ee_start_position", "camera.angle_delta", "cube.color",
        "cube.goal_position", "cube.goal_yaw", "cube.size", "cube.start_position",
        "cube.start_yaw", "floor.color", "light.intensity"),
    "swm/OGBScene-v0": (
        "agent.color", "agent.ee_start_position", "camera.angle_delta", "cube.color",
        "cube.goal_position", "cube.goal_yaw", "cube.size", "cube.start_position",
        "cube.start_yaw", "floor.color", "light.intensity", "lock_color"),
    "swm/PFRocketLanding-v0": (
        "environment.sky_color", "environment.start_height_ratio",
        "environment.start_horizontal_offset", "environment.start_tilt", "pad.color",
        "rocket.body_color", "rocket.booster_color", "rocket.fin_color", "rocket.leg_color"),
}

#: The four buckets, and nothing else. A verdict outside this set is a typo, not a new category.
BUCKETS = ("mapped", "unsupported", "dead_upstream", "not_applicable")
