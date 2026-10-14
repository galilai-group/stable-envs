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
See ``pre-repository-cleanup:docs/research/step8_variations_newcells.md``.

Cube/Scene recovery now maps the registered public semantic adapters. Their
independent compiled models feed the unchanged frozen renderer; historical model
probes below remain provenance. Reset configuration is part of Variation, with
one value per lane through ``variation_values``.
"""

from __future__ import annotations

from ..variation import VariationMap

__all__ = ["NEWCELL_VARIATION_MAPS", "LEAF_CLASSIFICATION"]


# ====================================================================== evidence vocabulary

#: The probes these classifications rest on. Quoted in the reasons so a reader can re-run the one
#: that produced a verdict rather than taking the verdict on trust.
PROBES = {
    "rocket_liveness": (
        "pre-repository-cleanup:tests/fixtures/regression/join/step8/variations-newcells/probes/live_rocket.json -- each leaf resampled "
        "alone under 5 seeds, 3 frames of 480x480 (691,200 px) per reset, compared against the "
        "no-variation reset"
    ),
    "rocket_pad": (
        "pre-repository-cleanup:tests/fixtures/regression/join/step8/variations-newcells/probes/pad_color.json -- pad.color over 3 seeds, "
        "120 steps, 6 frames of 480x480, plus pybullet getVisualShapeData on the landing pad body"
    ),
    "ogb_model": (
        "pre-repository-cleanup:tests/fixtures/regression/join/step8/variations-newcells/probes/ogb_model_{Cube,Scene}.json -- every "
        "MjModel array diffed between a no-variation reset and a one-leaf-extreme reset, with each "
        "changed geom classified as a MOVER (drawn per world by mujoco_warp) or STATIC (baked into "
        "the plate)"
    ),
    "ogb_agent_color": (
        "pre-repository-cleanup:tests/fixtures/regression/join/step8/variations-newcells/probes/agent_color_dirty_flag.json -- "
        "agent.color=RED alone vs agent.color=RED co-varied with light.intensity"
    ),
    "scene_noise": (
        "pre-repository-cleanup:tests/fixtures/regression/join/step8/variations-newcells/probes/scene_noise.json -- two IDENTICAL "
        "seed=0 SceneEnv resets diffed, to separate reset noise from leaf effect"
    ),
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
        "authorised to make to a certified renderer without a pixel re-baseline. ESCALATED"
    ),
    "light.intensity": (
        "the global directional light's diffuse term (MEASURED: writes light_diffuse row 0 and "
        "nothing else). It lights BOTH halves of the frame, so it is not plate-only: the plate can "
        "be re-baked per value, but the mover pass would need light_diffuse to vary PER WORLD inside "
        "one mujoco_warp context. That is a renderer change, not a cache key. ESCALATED"
    ),
    "camera.angle_delta": (
        "perturbs the front_pixels camera's xyaxes (MEASURED: writes cam_mat0 and cam_quat). Same "
        "shape of problem as light.intensity -- the plate is a picture taken FROM this camera and "
        "the mover pass renders through the same model camera, so per-environment camera angles "
        "need a per-world camera in the mujoco_warp context. ESCALATED"
    ),
    "cube.size": (
        "the cube's geom half-extents (MEASURED: writes geom_size on 2 MOVER geoms plus 12 derived "
        "inertia/bounding arrays). Movers are drawn by mujoco_warp from the context's shared model, "
        "so per-environment sizes need a per-world geom_size. ESCALATED"
    ),
    "agent.color": (
        "the arm's `ur5e/robotiq/black` and `ur5e/robotiq/pad_gray` materials, which are carried "
        "only by MOVER geoms. UPSTREAM DEFECT: applied alone it reaches NO model array at all -- "
        "CubeEnv.modify_mjcf_model computes `agent_color_changed = np.allclose(current, desired)`, "
        "which is True exactly when the colour did NOT change, so the dirty flag is never set and "
        "the staged MJCF edit is never compiled. MEASURED: agent.color=(1,0,0) alone leaves mat_rgba "
        "at the default (0.61,0.28,0.82); the same request co-varied with light.intensity (which "
        "sets the flag correctly) does turn the arm red. Conditionally dead upstream"
    ),
    "cube.color": (
        "the cube geoms' rgba. MEASURED: swm writes it DIRECTLY on the compiled model in "
        "initialize_episode with no recompile, on 2 MOVER geoms (Cube). This is the one appearance "
        "leaf reachable through machinery this cell ALREADY has -- _upload_button_colours writes "
        "per-world geom_rgba/mat_rgba into the mujoco_warp context for the Scene buttons by exactly "
        "this route. Not offered yet only because the cell's style space is empty and adding the "
        "first style dim moves the declaration (style_dim, style_digest) for a certified cell"
    ),
    "lock_color": (
        "Scene only: the two lock materials, carried by 6 MOVER geoms. Same mechanism and same "
        "answer as cube.color -- reachable through the existing per-world rgba upload. Attribution "
        "caveat: SceneEnv's reset is not seed-reproducible, so geom_rgba/mat_rgba move between two "
        "IDENTICAL resets and this probe cannot separate the leaf from that noise by array diff "
        "alone; the mover-geom membership is the solid half of the measurement"
    ),
}

_OGB_CUBE_MAPPED = tuple(sorted((*_OGB_STATE_LEAVES, *(k for k in _OGB_APPEARANCE if k != "lock_color"))))
_OGB_SCENE_MAPPED = tuple(sorted((*_OGB_STATE_LEAVES, *_OGB_APPEARANCE)))
_OGB_PLACEMENT_NOTES = {
    k: "SWM CubeEnv/SceneEnv read this leaf only in data_collection mode; the registered env runs in task "
    "mode, where it is sampled, stored and seeded but moves no physics. Ours carries the same value "
    "and RNG stream in State (gate: tests/helpers/reference/gates/ogb_variation.py)."
    for k in _OGB_STATE_LEAVES
    if k != "agent.ee_start_position"
}


# ====================================================================== Rocket

#: Reset distributions for the drop pose. They are Variation on both sides (per lane, sampled with
#: ``seed + lane`` or set through ``variation_values``) and reach pixels by moving the rocket.
ROCKET_START_LEAVES = (
    "environment.start_height_ratio",
    "environment.start_horizontal_offset",
    "environment.start_tilt",
)
ROCKET_GATE = (
    "tests/helpers/reference/gates/variation_pilots.py::rocket_reset_leaf_gate -- explicit per-lane "
    "values, seeded per-lane samples and a plain reset back to the defaults give bit-identical "
    "observations to swm over a 40-step rollout"
)

_ROCKET_SWM_ONLY = {
    "environment.sky_color": (
        "DEAD UPSTREAM, MEASURED. swm declares an RGBBox and nothing ever reads it: the string "
        "`sky_color` occurs exactly once in stable_worldmodel (its own declaration) and not at all "
        "in PyFlyt, and pybullet's TinyRenderer clear colour is fixed. Probe: resampling it alone "
        "over 5 seeds spanning the RGB cube -- drawn values as far apart as (43,202,87) and "
        "(149,165,44) -- moved 0 of 691,200 pixels, max channel difference 0, and all three frame "
        "stacks hash to the same digest. Offering it would advertise a knob that cannot work"
    ),
}


# ====================================================================== the maps

#: OUR class name -> how its appearance relates to the official swm env's.
NEWCELL_VARIATION_MAPS: dict[str, VariationMap] = {
    "OGBCube": VariationMap(
        swm_id="swm/OGBCube-v0",
        identical=_OGB_CUBE_MAPPED,
        notes=_OGB_PLACEMENT_NOTES,
    ),
    "OGBScene": VariationMap(
        swm_id="swm/OGBScene-v0",
        identical=_OGB_SCENE_MAPPED,
        notes=_OGB_PLACEMENT_NOTES,
    ),
    "PFRocketLanding": VariationMap(
        swm_id="swm/PFRocketLanding-v0",
        # Same dotted path and the same 0-255 units on both sides. `pad.color` is included on the
        # strength of the MODEL measurement, not a pixel one: swm's landing pad is never in the
        # reference render camera's view in the sampled regime, so a pixel probe cannot see it, but
        # getVisualShapeData confirms our material_table reproduces the 3-decimal URDF quantisation
        # swm applies (e.g. (104,60,3) -> 0.408/0.235/0.012, matching f"{v:.3f}").
        identical=(
            "rocket.body_color",
            "rocket.fin_color",
            "rocket.leg_color",
            "rocket.booster_color",
            "pad.color",
            *ROCKET_START_LEAVES,
        ),
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
            "rocket.body_color": _cls(
                "mapped",
                "style dim, identical path and units (0-255)",
                PROBES["rocket_liveness"] + " -- 35,551 px moved, maxdiff 241",
            ),
            "rocket.fin_color": _cls(
                "mapped",
                "style dim, identical path and units",
                PROBES["rocket_liveness"] + " -- 16,858 px moved, maxdiff 186",
            ),
            "rocket.leg_color": _cls(
                "mapped",
                "style dim, identical path and units",
                PROBES["rocket_liveness"] + " -- 228 px moved, maxdiff 218",
            ),
            "rocket.booster_color": _cls(
                "mapped",
                "style dim, identical path and units",
                PROBES["rocket_liveness"] + " -- 573 px moved, maxdiff 191",
            ),
            "pad.color": _cls(
                "mapped",
                "style dim, identical path and units. Reaches the reference MODEL (verified) but no "
                "reference PIXEL in the sampled regime: the landing pad is not in the render "
                "camera's view over 120 steps from a 430 m drop",
                PROBES["rocket_pad"] + " -- 0 of 1,382,400 px moved; getVisualShapeData shows the "
                "quantised rgba applied, e.g. (104,60,3) -> (0.408,0.235,0.012,1.0)",
                mechanism="URDF material rewrite, 3-decimal quantised, reproduced by material_table",
            ),
            "environment.sky_color": _cls(
                "dead_upstream",
                "swm samples it and no reader exists anywhere in swm, PyFlyt or pybullet",
                PROBES["rocket_liveness"] + " -- 0 of 691,200 px moved over 5 seeds spanning the "
                "RGB cube; identical frame digests",
                mechanism="declared once in pyflyt_rocketlanding.py:127 and read nowhere; "
                "TinyRenderer's clear colour is fixed",
            ),
            "environment.start_height_ratio": _cls(
                "mapped",
                "reset distribution for the drop altitude, identical path and units",
                ROCKET_GATE,
            ),
            "environment.start_horizontal_offset": _cls(
                "mapped",
                "reset distribution for the lateral start, identical path and units",
                ROCKET_GATE,
            ),
            "environment.start_tilt": _cls(
                "mapped",
                "reset distribution for the initial attitude, identical path and units",
                ROCKET_GATE,
            ),
        },
    },
}


# The recovery records supersede only the two manipulation cells. Other families'
# classifications and immutable historical probe files remain unchanged.
for _family, _mapped in [("Cube", _OGB_CUBE_MAPPED), ("Scene", _OGB_SCENE_MAPPED)]:
    _evidence = (
        "pre-repository-cleanup:docs/research/semantic_recovery_evidence/ogb_renderer_retry/remote/evidence/"
        + _family
        + "/result.json and cpu.log: exact source physics arrays and frozen pixels"
    )
    _leaves = {
        k: _cls(
            "mapped",
            "Identical public persistent/reset configuration and source behavior; exhaustive State "
            "carries its current/anchor values, and named RenderState carries its pixel effect.",
            _evidence,
        )
        for k in _mapped
    }
    _leaves["agent.color"]["reason"] = (
        "Mapped with the source's inverted np.allclose dirty predicate preserved: pending MJCF "
        "color is distinct from compiled color. Repeated/co-varied application changes pixels; "
        "an isolated pending change must not be rendered before source compilation."
    )
    for _k, _note in _OGB_PLACEMENT_NOTES.items():
        _leaves[_k]["reason"] = _note
    LEAF_CLASSIFICATION["swm/OGB" + _family + "-v0"] = {"our_envs": ["OGB" + _family], "leaves": _leaves}


# ====================================================================== the leaf inventory

#: What each swm id's ``variation_space.names()`` ACTUALLY returned, enumerated from the live wrapper
#: on 2026-09-16. Recorded rather than only asserted live because ``stable_worldmodel`` is not
#: importable in every environment this repo is tested in -- without this, the completeness test
#: (every offered leaf is classified) would silently skip exactly where it matters. The live test
#: runs too, wherever swm IS importable, and asserts this table still matches the wrapper.
RECORDED_SWM_LEAVES: dict[str, tuple[str, ...]] = {
    "swm/OGBCube-v0": (
        "agent.color",
        "agent.ee_start_position",
        "camera.angle_delta",
        "cube.color",
        "cube.goal_position",
        "cube.goal_yaw",
        "cube.size",
        "cube.start_position",
        "cube.start_yaw",
        "floor.color",
        "light.intensity",
    ),
    "swm/OGBScene-v0": (
        "agent.color",
        "agent.ee_start_position",
        "camera.angle_delta",
        "cube.color",
        "cube.goal_position",
        "cube.goal_yaw",
        "cube.size",
        "cube.start_position",
        "cube.start_yaw",
        "floor.color",
        "light.intensity",
        "lock_color",
    ),
    "swm/PFRocketLanding-v0": (
        "environment.sky_color",
        "environment.start_height_ratio",
        "environment.start_horizontal_offset",
        "environment.start_tilt",
        "pad.color",
        "rocket.body_color",
        "rocket.booster_color",
        "rocket.fin_color",
        "rocket.leg_color",
    ),
}

#: The four buckets, and nothing else. A verdict outside this set is a typo, not a new category.
BUCKETS = ("mapped", "unsupported", "dead_upstream", "not_applicable")


# ====================================================================== OGB maze / SimplePointMaze

_MAZE_PROBES = {
    "ogbmaze": (
        "pre-repository-cleanup:tests/fixtures/regression/join/step8/variations-newcells/probes/probe_ogbmaze.json -- each leaf set to "
        "explicit values through options['variation_values'], 224x224 (50,176 px), with the task id "
        "and the qpos/qvel pinned and the global numpy seed fixed so the ~3,000-5,500 px reset noise "
        "floor (ogbench draws cur_task_id from the GLOBAL np.random) does not swamp the leaf; "
        "base-vs-base maxdiff 0"
    ),
    "spm": (
        "pre-repository-cleanup:tests/fixtures/regression/join/step8/variations-newcells/probes/probe_spm.json -- 500x500 (250,000 px). "
        "SimplePointMazeEnv honours ONLY options['variation'] (a resample list) and silently ignores "
        "options['variation_values'], so exact values are set on the leaf after a no-variation reset; "
        "base-vs-base maxdiff 0"
    ),
}

#: Gate evidence for the appearance leaves, measured on the CPU ray-caster against MuJoCo 3.12.0 /
#: ogbench 1.2.1 / swm 176d545 at 224x224 (three state-scene lanes, two pixel-scene lanes per case),
#: each lane a fresh swm env.
_OGBMAZE_GATE = (
    "tests/helpers/reference/gates/ogbench_maze.py: appearance_gate (state scene, PointMazeLargeNavigate: "
    "same seed and per-lane variation_values give the exact reset State and a 20-step rollout, since the "
    "recompile never reaches physics) and pixel_appearance_gate (pixel scenes, point/ant/humanoid at swm's "
    "qpos). Worst-frame normalized MSE <= 1e-3 and every varied lane moves swm's frame off our default by more than "
    "our residual"
)

_OGBMAZE_NOTES = {
    "floor.color": "the checker's rgb1/rgb2, written as texels into the baked plate. Drawn wherever swm "
    "draws it: always in the state scene, and in pixel scenes only at floor.pixel_encoding=0 -- the "
    "position-encoding gradient (encoding 1) overwrites the checker on both sides",
    "floor.pixel_encoding": "same switch, sense and default as swm (1 gradient, 0 XML checker). Like swm, the state scene ignores it",
    "light.intensity": "the global light's diffuse term, written into the model before the plate bake. "
    "Static pixels are MuJoCo's; agent shading in pixel scenes scales the global share of the fitted "
    "diffuse law, and in the state scene reads the scene's lights directly",
}

_OGBMAZE_OURS_ONLY = {
    "floor.tint": "Multiplicative tint of the bound texture, not SWM checker RGB replacement",
    "rendering.decolor_floor": "Flat slate floor, not SWM pixel_encoding=0 two-colour checker; "
    "an affine inversion alone cannot reproduce its pixels",
    "agent.scale": "rjepa's `mark_agent_red` enlarges the agent so a 6-pixel body is findable by a "
    "fixed sky camera; swm has no counterpart",
    "camera.distance_scale": "swm does not expose the camera; ours is needed to reproduce the "
    "published sky-camera framings",
    "camera.elevation": "as camera.distance_scale",
}

_SPM_AGENT_POSITION = (
    "a per-lane reset distribution (Env.reset_leaves) that seeds State, not a style column: "
    "reset(options={'variation': ['agent.position']}) draws it per lane from the same seeded "
    "tree swm samples, and options['variation_values'] sets it explicitly. swm's "
    "SimplePointMazeEnv ignores options['variation_values'], so the matching swm start is the "
    "leaf's value. Gated by tests/integration/test_variation_pilots.py: reset State exact and "
    "first-frame interior exact"
)

_SPM_GOAL_POSITION = (
    "a per-lane reset leaf (Env.reset_leaves) that seeds State columns goal_x, goal_y. Not drawn by "
    "default, as in swm: reset(options={'variation': ['goal.position']}) draws it per lane from the "
    "seeded tree with swm's wall-collision rejection at the goal radius, and "
    "options['variation_values'] sets it explicitly (swm ignores variation_values, so the matching "
    "swm goal is the leaf's value). Gated by tests/integration/test_variation_navigation.py: reset "
    "State, rollout State and first frame"
)

NEWCELL_VARIATION_MAPS.update(
    {
        "SimplePointMaze": VariationMap(
            swm_id="swm/SimplePointMaze-v0",
            # Every one of these is the same dotted path, the same units and the same default on both
            # sides: RGBBox 0-255 for the colours, world units on the 5x5 domain for the radii and
            # positions, Discrete(3, start=4) for walls.number. Verified leaf by leaf against the live
            # space rather than assumed from the names.
            identical=(
                "agent.color",
                "agent.radius",
                "agent.speed",
                "agent.position",
                "background.color",
                "walls.color",
                "walls.number",
                "walls.shape",
                "walls.positions",
                "goal.color",
                "goal.radius",
                "goal.position",
            ),
            notes={"agent.position": _SPM_AGENT_POSITION, "goal.position": _SPM_GOAL_POSITION},
        ),
        **{
            name: VariationMap(
                swm_id="swm/OGBMaze-v0",
                identical=("agent.color", "floor.color", "floor.pixel_encoding", "light.intensity"),
                # swm spells it `wall.color`, singular; we spell it `walls.color`. Same units (float 0-1)
                # and the same absolute semantics -- MEASURED: swm writes the `wall` MATERIAL's mat_rgba and
                # never geom_rgba, and setting it to our default 0.6 reproduces the default frame bit-exactly
                # (0 of 50,176 px), which is what settles the apparent default conflict: swm reads its
                # init_value from the loaded XML, and ogbench rewrites wall rgba to 0.6 only in pixels mode,
                # so the 1.0 that a states-mode env reports is not the appearance we render.
                renamed={"walls.color": "wall.color"},
                ours_only=_OGBMAZE_OURS_ONLY,
                notes=_OGBMAZE_NOTES,
            )
            for name in ("PointmazeLargeNavigate", "PointmazeLargeStitch")
        },
    }
)

RECORDED_SWM_LEAVES.update(
    {
        "swm/OGBMaze-v0": ("agent.color", "floor.color", "floor.pixel_encoding", "light.intensity", "wall.color"),
        "swm/SimplePointMaze-v0": (
            "agent.color",
            "agent.position",
            "agent.radius",
            "agent.speed",
            "background.color",
            "goal.color",
            "goal.position",
            "goal.radius",
            "walls.color",
            "walls.number",
            "walls.positions",
            "walls.shape",
        ),
    }
)

LEAF_CLASSIFICATION.update(
    {
        "swm/OGBMaze-v0": {
            "our_envs": ["PointmazeLargeNavigate", "PointmazeLargeStitch"],
            "config_note": (
                "Gated in both scenes. The registered swm/OGBMaze-v0 default is ob_type='states', which "
                "PointMazeLargeNavigate renders; the other cells render ob_type='pixels'. UPSTREAM FINDING: "
                "swm's compile_model() rebuilds ogbench's `custom_renderer` only for 'pixels', so in "
                "'states' a leaf reaches swm's frame only if that env had not rendered before the "
                "recompile; the gate uses a fresh swm env per lane and case"
            ),
            "leaves": {
                "agent.color": _cls(
                    "mapped",
                    "identical path, float 0-1, same default [0.7,0.5,0.3]",
                    _MAZE_PROBES["ogbmaze"] + " -- 59 px moved at maxdiff 255",
                    mechanism="MJCF material 'self' -> mat_rgba row 1, carried by 1 geom (pointbody)",
                ),
                "wall.color": _cls(
                    "mapped",
                    "our walls.color; same float 0-1 units and absolute (not tint) semantics",
                    _MAZE_PROBES["ogbmaze"] + " -- 26,310 px moved at black/red; setting it to our "
                    "default 0.6 moves 0 of 50,176 px",
                    mechanism="MJCF material 'wall' -> mat_rgba row 3, carried by 62 block geoms; "
                    "geom_rgba is never written",
                ),
                "floor.color": _cls(
                    "mapped",
                    _OGBMAZE_NOTES["floor.color"],
                    _OGBMAZE_GATE + " -- state scene worst 1.4e-4 (swm shift >= 0.079); pixel scenes at "
                    "encoding 0 worst 7.3e-5 (shift >= 0.059)",
                    mechanism="checker texels of texture 'grid' in the baked plate",
                ),
                "floor.pixel_encoding": _cls(
                    "mapped",
                    _OGBMAZE_NOTES["floor.pixel_encoding"],
                    _OGBMAZE_GATE + " -- encoding 0 worst 7.3e-5 (shift >= 0.059); gradient worst 7.6e-5",
                    mechanism="plate floor texture: ogbench gradient (1) or XML checker (0)",
                ),
                "light.intensity": _cls(
                    "mapped",
                    _OGBMAZE_NOTES["light.intensity"],
                    _OGBMAZE_GATE + " -- state scene worst 2.1e-5 (shift >= 8.0e-4); pixel scenes worst "
                    "3.3e-5 (shift >= 0.045)",
                    mechanism="light_diffuse of light 'global' in the bake and the agent shading law",
                ),
            },
        },
        "swm/SimplePointMaze-v0": {
            "our_envs": ["SimplePointMaze"],
            "config_note": (
                "This class is the show_goal=True configuration, the swm default; reference_env passes "
                "show_goal=True so fidelity compares like with like. The goal disc is drawn at alpha 0.5"
            ),
            "leaves": {
                "agent.color": _cls(
                    "mapped",
                    "identical path, RGBBox 0-255, default (255,0,0)",
                    _MAZE_PROBES["spm"] + " -- 344 px moved at maxdiff 255",
                ),
                "agent.radius": _cls(
                    "mapped",
                    "identical path, world units 0.05-0.5 on the 5x5 domain",
                    _MAZE_PROBES["spm"] + " -- 284 px at r=0.05, 7,712 px at r=0.5",
                ),
                "background.color": _cls(
                    "mapped", "identical path, RGBBox 0-255", _MAZE_PROBES["spm"] + " -- 211,173 px moved"
                ),
                "walls.color": _cls(
                    "mapped", "identical path, RGBBox 0-255", _MAZE_PROBES["spm"] + " -- 37,552 px moved"
                ),
                "walls.number": _cls(
                    "mapped",
                    "identical path, Discrete(3, start=4), init 5",
                    _MAZE_PROBES["spm"] + " -- 3,671 px at n=6, 27 px at n=4",
                ),
                "walls.shape": _cls(
                    "mapped",
                    "identical path, (6,2) in world units 0.5-1.5",
                    _MAZE_PROBES["spm"] + " -- 15,909 px moved",
                ),
                "walls.positions": _cls(
                    "mapped", "identical path, (6,2) in world units 0-5", _MAZE_PROBES["spm"] + " -- 46,153 px moved"
                ),
                "agent.position": _cls(
                    "mapped",
                    _SPM_AGENT_POSITION,
                    _MAZE_PROBES["spm"] + " -- 0 of 250,000 px when set after reset",
                    mechanism="render() draws self.state; the leaf only seeds it at reset",
                ),
                "agent.speed": _cls(
                    "mapped",
                    "identical path; dynamics-only Variation leaf that scales the move",
                    _MAZE_PROBES["spm"] + " -- 0 of 250,000 px at 0.05 and at 2.0 (no pixel effect)",
                    mechanism="step(): state + speed*action; now a native variation leaf",
                ),
                "goal.color": _cls(
                    "mapped",
                    "identical path, RGBBox 0-255, composited over the background at alpha 0.5",
                    _MAZE_PROBES["spm"] + " -- 1,324 px moved with show_goal=True",
                ),
                "goal.radius": _cls(
                    "mapped",
                    "identical path, world units 0.05-0.5; also sets `terminated`",
                    _MAZE_PROBES["spm"] + " -- 1,264-6,812 px moved with show_goal=True",
                ),
                "goal.position": _cls(
                    "mapped",
                    _SPM_GOAL_POSITION,
                    _MAZE_PROBES["spm"] + " -- 0 px with show_goal=False, 2,648 px with show_goal=True",
                    mechanism="render() draws the goal leaf's value; reset samples it only when named",
                ),
            },
        },
    }
)

#: The two symbolic-adjacent Craftax pixel ids. They are recorded here rather than left out so the
#: completeness count can say "every id is accounted for" without an implicit exception.
#: MEASURED LIVE: `CraftaxWrapper.__init__` sets `self.variation_space = None` (craftax.py:59) and
#: there is no `swm.spaces` import anywhere in stable_worldmodel/envs/craftax/, so both ids offer
#: ZERO leaves and there is no Gate-F variation gap to close for either.
CRAFTAX_NO_LEAVES = {
    "swm/CraftaxClassicPixels-v1": {
        "our_envs": ["CraftaxClassic"],
        "n_swm_leaves": 0,
        "classification": "not_applicable",
        "evidence": "live: CraftaxClassicPixelsWrapper.variation_space is None; frame (63,63,3)",
    },
    "swm/CraftaxPixels-v1": {
        "our_envs": ["CraftaxFull"],
        "n_swm_leaves": 0,
        "classification": "not_applicable",
        "evidence": "live: CraftaxPixelsWrapper.variation_space is None; frame (130,110,3)",
    },
}


#: Our OWN variation-space leaves, per cell, recorded from ``make_variation_space``.
#:
#: Recorded rather than enumerated live for the same reason ``RECORDED_SWM_LEAVES`` is: the OGB
#: manipulation cells refuse to construct without CUDA, so a live enumeration would skip exactly
#: where the completeness check matters. The live check runs as well wherever the class CAN be built
#: and asserts this table still matches.
#:
#: This is the half of the classification the reviewer found missing. ``swm_only`` makes every swm
#: leaf account for itself; without this, an OUR-side leaf can exist with no statement of whether
#: swm has a counterpart, which is the same gap in the other direction.
RECORDED_OURS_LEAVES: dict[str, tuple[str, ...]] = {
    "OGBCube": _OGB_CUBE_MAPPED,
    "OGBScene": _OGB_SCENE_MAPPED,
    "PointmazeLargeNavigate": (
        "agent.color",
        "agent.scale",
        "floor.tint",
        "floor.color",
        "floor.pixel_encoding",
        "light.intensity",
        "walls.color",
        "camera.distance_scale",
        "camera.elevation",
        "rendering.decolor_floor",
    ),
    "PointmazeLargeStitch": (
        "agent.color",
        "agent.scale",
        "floor.tint",
        "floor.color",
        "floor.pixel_encoding",
        "light.intensity",
        "walls.color",
        "camera.distance_scale",
        "camera.elevation",
        "rendering.decolor_floor",
    ),
}

#: ``NEWCELL_VARIATION_MAPS`` under the name the ``variation_maps`` package installer looks for.
#: The old name is kept because this module's own tests and findings cite it.
MAPS = NEWCELL_VARIATION_MAPS
