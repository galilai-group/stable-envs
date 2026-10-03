"""CPU source-worker chunk stopping versus OGBench adapters/shared dispatcher.

Moves task targets to the current configuration in selected lanes, without
patching source step/termination functions. The default is a render-free adapter
fixture. --public uses real public constructors on Linux; they may open GL.
"""

import argparse
import contextlib
import hashlib
import importlib.metadata
import json
import platform
import types
from functools import partial
from pathlib import Path
from unittest.mock import patch

import mujoco
import numpy as np
import torch

import stable_envs as se
from scripts.benchmarks.online_stepping.ogbench_parallel_source import OGBSource, ParallelOGBSource
from scripts.benchmarks.online_stepping.ogbench_source_transport_gate import equal
from scripts.benchmarks.renderer_table.common import PIN, equal_state
from stable_envs._adapters.ogbench import OGBMazeAdapter
from stable_envs._adapters.ogbench_manipulation import OGBManipulationAdapter
from stable_envs._stepping import step as dispatch_step
from tests.unit.test_ogbench_manipulation_cpu import assert_oracle

NAMES = {"Maze": "PointMazeLargeNavigate", "Cube": "OGBCube", "Scene": "OGBScene"}


def adapter_fixture(family):
    """Exercise the unchanged dispatcher without claiming public construction."""
    adapter = OGBMazeAdapter(2, seed=17) if family == "Maze" else OGBManipulationAdapter(family, 2, seed=17)
    fixture = types.SimpleNamespace(num_envs=2, device=torch.device("cpu"), has_dynamics=True,
        execution_mode="DELEGATED_CPU", action_spec=types.SimpleNamespace(shape=(2 if family == "Maze" else 5,),
        dtype=torch.float64), _simulator=adapter, _check_open=lambda _: adapter._check_open(),
        _step=adapter.step, _advance_without_snapshot=partial(adapter.step, capture_state=False),
        get_state=adapter.get_state, set_state=adapter.set_state, close=adapter.close)
    fixture.step = partial(dispatch_step, fixture)
    return fixture


def reached_goal(env, family):
    if family == "Maze":
        env.cur_goal_xy = env.data.qpos[:2].copy()
        env.model.geom("target").pos[:2] = env.cur_goal_xy
    else:
        for i in range(env._num_cubes):
            env.data.mocap_pos[env._cube_target_mocap_ids[i]] = env.data.joint(f"object_joint_{i}").qpos[:3]
        if family == "Scene":
            env._target_button_states = env._cur_button_states.copy()
            env._target_drawer_pos = float(env.data.joint("drawer_slide").qpos[0])
            env._target_window_pos = float(env.data.joint("window_slide").qpos[0])
            env.model.site("drawer_handle_center_target").pos[1] = env._target_drawer_pos
            env.model.site("window_handle_center_target").pos[0] = env._target_window_pos
    mujoco.mj_forward(env.model, env.data)


def public_tasks(result, horizon):
    tasks = {}
    for key, value in result.exclude("state", "states", "state_valid").items(True, True):
        name = ".".join(key) if isinstance(key, tuple) else key
        tasks[name] = value.unsqueeze(1) if horizon == 1 and name != "valid" else value
    return tasks


def compare(actual, expected):
    assert set(actual) == set(expected)
    for key in actual:
        value = torch.from_numpy(expected[key]) if isinstance(expected[key], np.ndarray) else expected[key]
        torch.testing.assert_close(actual[key], value, rtol=0, atol=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--public", action="store_true", help="Use public constructors on Linux instead of the adapter fixture")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.public and platform.system() != "Linux":
        p.error("Public constructors open GL; run this mode on an Oscar compute node")
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest["commit"] == PIN
    for rel, digest in manifest["files"].items():
        assert hashlib.sha256((args.source_root / rel).read_bytes()).hexdigest() == digest, rel
    torch.set_num_threads(1)
    paths = [Path(__file__).relative_to(Path.cwd()),
             Path("scripts/benchmarks/online_stepping/ogbench_parallel_source.py"),
             Path("scripts/benchmarks/online_stepping/ogbench_source_prerequisite.py"),
             Path("scripts/benchmarks/online_stepping/ogbench_source_transport_gate.py"),
             Path("scripts/benchmarks/online_stepping/classic_parallel_source.py"),
             Path("tests/helpers/reference/ogbench_maze_semantics.py"),
             Path("tests/unit/test_ogbench_manipulation_cpu.py"),
             Path("docs/research/semantic_recovery_evidence/ogb_rng_feasibility/ownership.py"),
             Path("docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/probe.py"),
             Path("stable_envs/env.py"), Path("stable_envs/_stepping.py"),
             *Path("stable_envs/_adapters").glob("ogbench*.py"),
             Path("stable_envs/envs/ogbench/maze.py"),
             Path("stable_envs/envs/ogbench/manipulation_semantics.py")]
    report = dict(status="RUNNING", host=platform.node(), source_pin=PIN,
        source_files=manifest["files"], runtime_files={},
        candidate_files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        versions={n: importlib.metadata.version(n) for n in
                  ("ogbench", "mujoco", "dm-control", "torch", "numpy", "gymnasium", "tensordict")},
        construction="PUBLIC_CPU" if args.public else "ADAPTER_DISPATCH_FIXTURE",
        scope="CPU B2/P2 no-frame source termination/zero padding/State history; no timing or CUDA",
        fixture="move task goals to current object/joint configuration, one or both lanes; no source function patching",
        cases=[], worker_closures=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        guard = contextlib.nullcontext() if args.public else patch(
            "mujoco.Renderer", side_effect=AssertionError("No rendering in adapter fixture gate"))
        with guard:
            for family in NAMES:
                for horizon in (1, 8):
                    with contextlib.closing(OGBSource(family, args.source_root, 2)) as serial, \
                            contextlib.closing(se.make(NAMES[family], num_envs=2, device="cpu", seed=17)
                                if args.public else adapter_fixture(family)) as env, \
                            ParallelOGBSource(family, args.source_root, 2, 2, horizon) as workers:
                        original_source, original_public = serial.snapshot(), env.get_state()
                        report["runtime_files"].update(workers.runtime_files)

                        def witness():
                            if family == "Maze":
                                equal_state(env.get_state(), serial.reference.get_state())
                            else:
                                for candidate, raw in zip(env._simulator.envs, serial.envs, strict=True):
                                    assert_oracle(candidate, raw, serial.module)

                        witness()
                        for reached in (1, 2):
                            serial.restore(original_source)
                            env.set_state(original_public)
                            for lane in range(reached):
                                raw = serial.envs[lane].env if family == "Maze" else serial.envs[lane]
                                reached_goal(raw, family)
                                reached_goal(env._simulator.envs[lane], family)
                            witness()
                            initial_source, initial_public = serial.snapshot(), env.get_state()
                            for dtype in (np.float32, np.float64):
                                serial.restore(initial_source)
                                workers.restore(initial_source)
                                env.set_state(initial_public)
                                equal_state(initial_public, env.get_state())
                                actions = np.zeros((2, horizon, workers.action_dim), dtype=dtype)
                                expected = serial.step(actions)
                                actual = workers.step(actions)
                                result = env.step(actions[:, 0] if horizon == 1 else actions,
                                                  states=slice(None), frames=None)
                                compare(actual, expected)
                                compare(public_tasks(result, horizon), expected)
                                equal(workers.snapshot(), serial.snapshot())
                                # The accepted witness reads source action_space,
                                # incrementing its diagnostic access counter. Run
                                # it only after both full source-snapshot comparisons.
                                assert bool(result["terminated"][:reached].all()) if horizon == 1 else \
                                    bool(result["terminated"][:reached, 0].all())
                                expected_valid = torch.ones((2, horizon), dtype=torch.bool)
                                expected_valid[:reached, 1:] = False
                                assert torch.equal(result["valid"], expected_valid)
                                assert torch.equal(result["state_valid"], expected_valid)
                                for lane in range(reached):
                                    for t in range(1, horizon):
                                        equal_state(result["states"][lane, 0], result["states"][lane, t])
                                held = {key: value.clone() for key, value in actual.items()}
                                held_state = result["state"].clone()
                                # A new call begins normally, including after a prior
                                # terminal result. No container auto-reset is invented.
                                continuation = np.full_like(actions, .01)
                                expected_next = serial.step(continuation)
                                actual_next = workers.step(continuation)
                                public_next = env.step(continuation[:, 0] if horizon == 1 else continuation,
                                                       states=None, frames=None)
                                compare(actual_next, expected_next)
                                compare(public_tasks(public_next, horizon), expected_next)
                                assert bool(actual_next["valid"][:, 0].all())
                                compare(actual, held)
                                equal_state(result["state"], held_state)
                                equal(workers.snapshot(), serial.snapshot())
                                witness()
                                inspection = list(workers.vector.call("inspection"))
                                assert all(row["renderers_absent"] and row["torch_threads"] == 1 for row in inspection)
                                report["cases"].append(dict(family=family, batch=2, workers=2, horizon=horizon,
                                    reached_lanes=reached, action_dtype=np.dtype(dtype).name,
                                    valid_steps=int(expected_valid.sum()), task_state_witness="PASS",
                                    zero_padding_and_validity="PASS", terminal_state_hold="PASS",
                                    next_call_source_lifecycle="PASS", output_ownership="PASS", worker_info=inspection))
                                save()
                                print(family, horizon, reached, np.dtype(dtype).name, "PASS", flush=True)
                    closure = [dict(pid=p.pid, exitcode=p.exitcode, alive=p.is_alive()) for p in workers.vector.processes]
                    assert all(row["exitcode"] == 0 and not row["alive"] for row in closure)
                    report["worker_closures"].append(dict(family=family, horizon=horizon, workers=closure))
                    save()
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
