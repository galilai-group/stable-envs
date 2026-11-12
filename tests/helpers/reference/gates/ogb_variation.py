"""Cube/Scene Variation leaves against pinned stable_worldmodel, per lane.

Each lane of ours is compared with its own stable-worldmodel env (``OwnedCube``/``OwnedScene``
owns entropy the way our lane does): MuJoCo integration state, observations,
infos, model arrays, every RNG stream and every Variation node, at reset and
along a short rollout.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from stable_envs.envs.ogbench import OGBCube, OGBScene
from stable_envs.envs.ogbench.simulation.compatibility import (
    COMPILED,
    COMPILED_FIELDS,
    CUBE_SAMPLES,
    CompatibleCube,
    CompatibleScene,
)
from tests.helpers.reference.ogbench_manipulation_semantics import assert_oracle, load_oracle

CLASSES = {"Cube": OGBCube, "Scene": OGBScene}
LANES = 3
STEPS = 8


def _node(env, path):
    node = env.variation_space
    for part in path.split("."):
        node = node[part]
    return node


def _lane(value, lane, path, env):
    value = np.asarray(value, np.float64)
    return value[lane] if value.ndim > np.ndim(_node(env, path).value) else value


def _same_variation(ours, theirs, oracle):
    mine, other = dict(oracle.spaces(ours.variation_space)), dict(oracle.spaces(theirs.variation_space))
    assert mine.keys() == other.keys()
    for name, node in mine.items():
        assert node.np_random.bit_generator.state == other[name].np_random.bit_generator.state, name
        if not hasattr(node, "spaces"):
            assert np.array_equal(node.value, other[name].value), name
            assert np.array_equal(node.init_value, other[name].init_value), name


def _compare(env, refs, oracle):
    for lane, ref in enumerate(refs):
        assert_oracle(env._simulator.envs[lane], ref, oracle)
        _same_variation(env._simulator.envs[lane], ref, oracle)


def _rollout(env, refs, oracle, seed):
    actions = np.random.default_rng(seed).uniform(-1, 1, (STEPS, len(refs), 5))
    for action in actions:
        env.step(action, frames=None)
        for lane, ref in enumerate(refs):
            ref.step(action[lane])
    _compare(env, refs, oracle)


def _refs(family, seed, oracle):
    cls = oracle.OwnedCube if family == "Cube" else oracle.OwnedScene
    return [cls(entropy_seed=seed + lane) for lane in range(LANES)]


def _reset_refs(refs, seed, options):
    """stable-worldmodel per lane: per-lane ``variation_values`` rows are split, everything else is shared."""
    for lane, ref in enumerate(refs):
        lane_options = dict(options)
        if "variation_values" in options:
            lane_options["variation_values"] = {
                k: _lane(v, lane, k, ref) for k, v in options["variation_values"].items()
            }
        ref.reset(seed=seed + lane, options=lane_options)


def _case(family, seed, options, oracle):
    refs = _refs(family, seed, oracle)
    try:
        with CLASSES[family](num_envs=LANES, seed=seed) as env:
            state = env.reset(seed=seed, options=options)
            _reset_refs(refs, seed, options)
            _compare(env, refs, oracle)
            _rollout(env, refs, oracle, seed)
            return state
    finally:
        for ref in refs:
            ref.close()


def reset_options_gate(family, seed=31) -> dict:
    """Sampled and explicit per-lane leaves through reset options."""
    oracle = load_oracle()
    evidence = {"family": family, "seed": seed, "lanes": LANES, "steps": STEPS}
    sampled = [*CUBE_SAMPLES, "cube.size", "cube.color", "light.intensity", "camera.angle_delta", "floor.color"]
    if family == "Scene":
        sampled.append("lock_color")
    state = _case(family, seed, {"task_id": 2, "variation": sampled}, oracle)
    for path in sampled:
        drawn = state["variation", "current", path].reshape(LANES, -1)
        assert len({tuple(row.tolist()) for row in drawn}) == LANES, path
    evidence["sampled"] = sorted(sampled)

    explicit = {
        "cube.start_position": np.array([[[0.32, -0.05]], [[0.4, 0.1]], [[0.44, 0.17]]]),
        "cube.start_yaw": np.array([[0.3], [2.0], [5.5]]),
        "cube.goal_position": np.array([[0.35, 0.0]]),
        "cube.goal_yaw": np.array([[1.0], [0.0], [6.0]]),
        "cube.size": np.array([[0.012], [0.02], [0.029]]),
        "light.intensity": np.array([[0.1], [0.55], [1.0]]),
        "camera.angle_delta": np.array([[[1.0, -2.0]], [[0.0, 0.0]], [[-4.5, 3.0]]]),
        "agent.ee_start_position": np.array([[0.3, -0.1, 0.25], [0.45, 0.0, 0.3], [0.55, 0.15, 0.22]]),
    }
    if family == "Scene":
        explicit["lock_color"] = np.array([[0.1, 0.2, 0.3], [0.9, 0.8, 0.1]])
    _case(family, seed + 1, {"task_id": 4, "variation": [], "variation_values": explicit}, oracle)
    evidence["explicit_per_lane"] = sorted(explicit)
    evidence["placement_leaves_read_in_task_mode"] = _placement_moves_engine(family, oracle, seed)
    return evidence


def _placement_moves_engine(family, oracle, seed) -> bool:
    """stable-worldmodel reads the cube placement leaves only in data_collection mode."""
    cls = oracle.OwnedCube if family == "Cube" else oracle.OwnedScene
    a, b = cls(entropy_seed=seed), cls(entropy_seed=seed)
    try:
        a.reset(seed=seed, options={"task_id": 2, "variation": []})
        b.reset(
            seed=seed,
            options={
                "task_id": 2,
                "variation": [],
                "variation_values": {
                    "cube.start_position": np.array([[0.44, 0.15]]),
                    "cube.start_yaw": np.array([2.5]),
                    "cube.goal_position": np.array([[0.31, -0.06]]),
                    "cube.goal_yaw": np.array([4.0]),
                },
            },
        )
        return bool(oracle.differences(oracle.engine(a), oracle.engine(b)))
    finally:
        a.close()
        b.close()


def _fresh_compile(family, live):
    fresh = (CompatibleCube if family == "Cube" else CompatibleScene)(seed=0)
    for path in (*COMPILED, "agent.color"):
        fresh.variation_space._walk(path)._value = np.array(live._model_values[path], np.float64)
    fresh._mjcf_model = fresh.modify_mjcf_model(fresh.build_mjcf_model())
    fresh.compile_model_and_data()
    return fresh


def set_variation_gate(family, seed=37) -> dict:
    """``set_variation`` mid-episode: immediate model, exact restore, and stable-worldmodel's next reset."""
    oracle = load_oracle()
    values = {
        "cube.size": np.array([[0.014], [0.02], [0.028]]),
        "light.intensity": np.array([0.35]),
        "camera.angle_delta": np.array([[[3.0, -2.0]], [[0.0, 0.0]], [[-1.5, 4.0]]]),
        "floor.color": np.array([[0.5, 0.2, 0.1], [0.9, 0.9, 0.3]]),
        "cube.color": np.array([[0.1, 0.9, 0.2]]),
        "agent.color": np.array([0.9, 0.1, 0.1]),
        "agent.ee_start_position": np.array([0.5, 0.05, 0.3]),
    }
    if family == "Scene":
        values["lock_color"] = np.array([[0.2, 0.3, 0.9], [0.8, 0.1, 0.6]])
    evidence = {"family": family, "seed": seed, "leaves": sorted(values)}
    rng = np.random.default_rng(seed)
    refs = _refs(family, seed, oracle)
    try:
        with CLASSES[family](num_envs=LANES, seed=seed) as env:
            env.reset(seed=seed, options={"task_id": 2})
            _reset_refs(refs, seed, {"task_id": 2})
            for action in rng.uniform(-1, 1, (3, LANES, 5)):
                env.step(action, frames=None)
                for lane, ref in enumerate(refs):
                    ref.step(action[lane])
            before = env.get_state()
            env.set_variation(values)
            state = env.get_state()

            # Immediate: current values, the compiled record and the live model move now.
            immediate = [k for k in values if k not in ("agent.color", "agent.ee_start_position")]
            for path in immediate:
                want = torch.as_tensor(np.stack([_lane(values[path], i, path, refs[i]) for i in range(LANES)]))
                assert torch.equal(state["variation", "current", path].cpu(), want), path
                if path in COMPILED:
                    assert torch.equal(state["model", "compiled", path].cpu(), want), path
            for path in ("agent.color", "agent.ee_start_position"):
                assert torch.equal(state["variation", "current", path], before["variation", "current", path])
            for name in ("physics", "controller", "task", "rng"):
                for key in before[name].keys(True, True):
                    assert torch.equal(state[name][key], before[name][key]), (name, key)
            for live in env._simulator.envs:
                fresh = _fresh_compile(family, live)
                try:
                    for field in COMPILED_FIELDS:
                        assert np.array_equal(getattr(live.model, field), getattr(fresh.model, field)), field
                finally:
                    fresh.close()
            evidence["model_equals_fresh_compile"] = list(COMPILED_FIELDS)

            # The running episode equals one restored from its State, which recompiles from scratch.
            actions = rng.uniform(-1, 1, (STEPS, LANES, 5))
            continued = [env.step(action, frames=None)["state"] for action in actions]
            env.set_state(state)
            restored = [env.step(action, frames=None)["state"] for action in actions]
            for a, b in zip(continued, restored, strict=True):
                for key in a.keys(True, True):
                    assert torch.equal(a[key], b[key]), key
            evidence["continuation_equals_restore_steps"] = STEPS

            # stable-worldmodel spells set_variation as set_init_value; the next reset must match it exactly.
            for lane, ref in enumerate(refs):
                for path, value in values.items():
                    _node(ref, path).set_init_value(_lane(value, lane, path, ref))
            env.reset(seed=seed + 10, options={"task_id": 3})
            _reset_refs(refs, seed + 10, {"task_id": 3})
            _compare(env, refs, oracle)
            _rollout(env, refs, oracle, seed + 10)

            # agent.color alone is not compiled by stable-worldmodel's dirty rule; ours follows it.
            env.set_variation({"agent.color": np.array([0.2, 0.8, 0.3])})
            for ref in refs:
                _node(ref, "agent.color").set_init_value(np.array([0.2, 0.8, 0.3]))
            env.reset(seed=seed + 20, options={"task_id": 1})
            _reset_refs(refs, seed + 20, {"task_id": 1})
            _compare(env, refs, oracle)
            evidence["next_reset_matches_swm"] = True
    finally:
        for ref in refs:
            ref.close()
    return evidence


def run_gate() -> dict:
    return {
        family: {"reset_options": reset_options_gate(family), "set_variation": set_variation_gate(family)}
        for family in CLASSES
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--output")
    args = parser.parse_args()
    text = json.dumps(run_gate(), indent=2)
    if args.output:
        Path(args.output).write_text(text + "\n")
    print(text)
