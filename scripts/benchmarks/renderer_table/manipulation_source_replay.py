"""Render saved numeric State with the pinned upstream Cube/Scene renderer.

The source constructs its own model from each original reset recipe. Only the
recorded numeric state is installed; no candidate model or image is copied.
Capture every State leaf back from the source and require a bit-exact match.
"""

import argparse
import hashlib
import inspect
import json
import os
from pathlib import Path

import mujoco
import numpy as np
import torch
from OpenGL import GL

from scripts.benchmarks.renderer_table import collect
from scripts.benchmarks.renderer_table.common import catalog, load_state, profile, rgb, state_hash, timed, write
from scripts.benchmarks.renderer_table.special import ManipulationBridge
from stable_envs._adapters.ogbench.manipulation import FLAGS, pcg_decode, rngs, site_ids, task_payload
from stable_envs._adapters.ogbench.compatibility import CUBE_SAMPLES, MODEL_VALUES


def instrument_model_values(raw):
    # Read-only path accessor needed by the existing numeric State capture schema.
    def walk(path):
        node = raw.variation_space
        for part in path.split("."):
            node = node[part]
        return node

    raw.variation_space._walk = walk
    compile_source = raw.compile_model_and_data
    modify_source = raw.modify_mjcf_model

    def values():
        return {k: np.asarray(raw.variation_space._walk(k).value, dtype=np.float64).copy() for k in MODEL_VALUES}

    def compile_record(*args, **kwargs):
        raw._model_values = values()
        return compile_source(*args, **kwargs)

    def modify_record(*args, **kwargs):
        raw._mjcf_values = values()
        return modify_source(*args, **kwargs)

    raw.compile_model_and_data = compile_record
    raw.modify_mjcf_model = modify_record


def install(raw, row):
    for name in raw.supported_variations:
        node = raw.variation_space._walk(name)
        node.set_init_value(row["variation", "anchor", name].copy())
        node.set_value(row["variation", "current", name].copy())
    for attr, key in [("cur_task_id", "id"), ("_reward_task_id", "reward_id")]:
        value = int(row["task", key][0])
        setattr(raw, attr, None if value == -1 else value)
    raw.cur_task_info = task_payload(raw, {k[-1]: v for k, v in row.items() if k[:2] == ("task", "payload")})
    raw._success = bool(row["task", "success"][0])
    raw._cur_goal_ob = row["task", "goal_observation"].copy()
    raw._prev_qpos = row["output", "prev_qpos"].copy()
    raw._prev_qvel = row["output", "prev_qvel"].copy()
    raw._reset_next_step = bool(row["lifecycle", "reset_next_step"][0])
    if raw.family == "Scene":
        for name in ("cur_button_states", "prev_button_states", "target_button_states"):
            setattr(raw, "_" + name, row["scene", name].copy())
        for name in ("target_drawer_pos", "target_window_pos"):
            setattr(raw, "_" + name, float(row["scene", name][0]))
        raw.model.site("drawer_handle_center_target").pos[1] = raw._target_drawer_pos
        raw.model.site("window_handle_center_target").pos[0] = raw._target_window_pos
        raw._apply_button_states()
    for ids in raw._cube_target_geom_ids_list:
        raw.model.geom_rgba[ids, 3] = 0.2
    for pass_index in range(2):
        for name, flag in FLAGS.items():
            if ("physics", name) in row:
                mujoco.mj_setState(raw.model, raw.data, row["physics", name].astype(np.float64), flag)
        if pass_index == 0:
            mujoco.mj_forward(raw.model, raw.data)
    pos, rot = site_ids(raw)
    raw.data.site_xpos[pos] = row["controller", "site_xpos"]
    raw.data.site_xmat[rot] = row["controller", "site_xmat"]
    raw.data.body("ur5e/robotiq/right_pad").cfrc_ext[:] = row["output", "gripper_force"]
    for name, rng in rngs(raw).items():
        rng.bit_generator.state = pcg_decode(
            {k: row["rng", name, k] for k in ("state", "increment", "has_uint32", "uinteger")}
        )
    raw.task_rng.set_state(
        (
            "MT19937",
            row["rng", "task", "key"].astype(np.uint32),
            int(row["rng", "task", "position"][0]),
            int(row["rng", "task", "has_gauss"][0]),
            float(row["rng", "task", "cached_gaussian"][0]),
        )
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("registration")
    p.add_argument("corpus", type=Path)
    p.add_argument("output", type=Path)
    p.add_argument("--frames", type=int, default=1000)
    a = p.parse_args()
    a.output.mkdir(parents=True, exist_ok=False)
    row = catalog()[a.registration]
    assert row["name"] in ("OGBCube", "OGBScene")
    corpus = json.loads((a.corpus / "result.json").read_text())
    result = dict(
        registration=a.registration,
        role="source_gpu",
        backend="upstream-opengl",
        profile=profile(),
        job_id=os.environ.get("SLURM_JOB_ID"),
        status="RUNNING",
        frames=[],
        corpus_sha256=hashlib.sha256((a.corpus / "result.json").read_bytes()).hexdigest(),
        method="Exact numeric State install in pinned upstream instance; source-owned model built from original recipe; full leaf capture and render nonmutation",
    )
    gl_probe = mujoco.Renderer(
        mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>'),
        height=32,
        width=32,
    )
    gl_probe._gl_context.make_current()
    result["gl_vendor"] = GL.glGetString(GL.GL_VENDOR).decode()
    result["gl_renderer"] = GL.glGetString(GL.GL_RENDERER).decode()
    assert "nvidia" in result["gl_vendor"].lower()
    write(a.output / "campaign.json", result)
    output = dict(corpus, status="RUNNING", frames=[])
    env = collect.candidate(row, "cpu", 0)
    adapter = env._simulator
    active = None
    bridges = []  # Keep GL owners alive until process exit.
    for f in corpus["frames"][: a.frames]:
        if f["case"] != active:
            recipe = corpus["recipes"][f["case"]]
            bridge = ManipulationBridge(row, f["seed"], recipe["reset_options"])
            bridges.append(bridge)
            raw = bridge.source
            raw.family = row["name"][3:]
            raw.supported_variations = set(raw.variation_space.names())
            if raw.family == "Scene":
                raw.supported_variations -= set(CUBE_SAMPLES)
            instrument_model_values(raw)
            for _ in range(recipe["reset_count"]):
                bridge.reset(f["seed"])
            active = f["case"]
        state = load_state(a.corpus / f["state_file"], "cpu", env.get_state())
        assert state_hash(state) == f["state_hash"]
        values = {k: v[0].numpy().copy() for k, v in state.items(True, True)}
        # Compiled and pending MJCF configuration must come from the source's own calls.
        for namespace, actual in [("compiled", raw._model_values), ("mjcf", raw._mjcf_values)]:
            for k in MODEL_VALUES:
                assert np.array_equal(actual[k], values["model", namespace, k]), (namespace, k)
        install(raw, values)
        captured = torch.stack([adapter._row(raw)])
        assert state_hash(captured) == f["state_hash"], [
            k
            for k in state.keys(True, True)
            if state[k].dtype != captured[k].dtype or state[k].numpy().tobytes() != captured[k].numpy().tobytes()
        ]
        before = bridge.snapshot()
        timing, image = timed(bridge.render, gpu=True, samples=1, warmups=1)
        assert bridge.snapshot() == before
        assert state_hash(torch.stack([adapter._row(raw)])) == f["state_hash"]
        if not result.get("source_paths"):
            paths = {inspect.getsourcefile(type(raw).__init__), inspect.getsourcefile(type(raw).render)}
            result["source_paths"] = {s: hashlib.sha256(Path(s).read_bytes()).hexdigest() for s in paths if s}
        image = rgb(image).copy()
        name = f"frame-{f['index']:04d}.npy"
        np.save(a.output / name, image, allow_pickle=False)
        output["frames"].append(
            dict(
                f,
                oracle_file=name,
                oracle_sha256=hashlib.sha256(image.tobytes()).hexdigest(),
                original_timing=timing,
                state_witness="EXACT_FULL_STATE_RECAPTURE",
            )
        )
        write(a.output / "result.json", output)
        print(a.registration, f["index"], flush=True)
    output["status"] = result["status"] = "COMPLETE"
    result["collected_frames"] = len(output["frames"])
    write(a.output / "result.json", output)
    write(a.output / "campaign.json", result)


if __name__ == "__main__":
    main()
