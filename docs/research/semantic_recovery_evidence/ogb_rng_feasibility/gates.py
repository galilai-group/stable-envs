"""Research replay and actual threaded isolation gates; no global patches."""

from __future__ import annotations

import copy
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, get_ident

import gymnasium as gym
import numpy as np
from gymnasium.utils import seeding
from ownership import (
    BOX_SAMPLE,
    GETTER,
    LastCube,
    LastScene,
    OwnedCube,
    OwnedScene,
    capture,
    previous,
    restore,
    result,
)

OUT = Path(__file__).resolve().parent


def plain(value):
    return previous.encode(value)


def digest(value):
    return hashlib.sha256(json.dumps(plain(value), sort_keys=True).encode()).hexdigest()


def compare(left, right, path=""):
    # Exact values, dtype, shape, keys, and sequence lengths; no tolerance.
    if digest(left) == digest(right):
        return {}
    if isinstance(left, dict) and isinstance(right, dict):
        diff = {}
        for key in left.keys() | right.keys():
            if key not in left or key not in right:
                diff[path + "/" + str(key)] = "missing key"
            else:
                diff.update(compare(left[key], right[key], path + "/" + str(key)))
        return diff
    if isinstance(left, (list, tuple)) and isinstance(right, (list, tuple)):
        diff = {} if len(left) == len(right) else {path + "/length": [len(left), len(right)]}
        for i, (a, b) in enumerate(zip(left, right)):
            diff.update(compare(a, b, path + "/" + str(i)))
        return diff
    if isinstance(left, np.ndarray) and isinstance(right, np.ndarray):
        if left.shape != right.shape or left.dtype != right.dtype:
            return {path: "dtype/shape differ"}
        return {
            path: {
                "count": int(np.count_nonzero(left != right)),
                "max_abs": float(np.max(np.abs(left.astype(float) - right.astype(float)))),
            }
        }
    return {path: {"left": plain(left), "right": plain(right)}}


def run_reset(env, options=None, seed=None):
    return result(env, env.reset(seed=seed, options=options))


def replay(cls, seed, options, design="A"):
    a, b = cls(entropy_seed=seed, design=design), cls(entropy_seed=seed + 101, design=design)
    try:
        a.reset(seed=seed, options={"variation": [], "task_id": 2})
        b.reset(seed=seed + 101, options={"variation": ["cube.size"], "task_id": 4})
        # Different model geometry cannot be restored as the same persistent config.
        b.reset(options={"variation": [], "task_id": 4})
        for _ in range(3):
            b.step(np.array([0.2, -0.1, -0.5, 0.7, 0.8]))
        state = capture(a)
        original_hash = digest(state)
        first = run_reset(a, options)
        assert digest(state) == original_hash, "snapshot aliases live state"
        restore(a, state)
        again = run_reset(a, options)
        restore(b, state)
        independent = run_reset(b, options)
        diffs = {"same_instance": compare(first, again), "independent_history": compare(first, independent)}
        assert not any(diffs.values()), diffs
        assert len(first["settling_actions"]) == 2
        assert first["sequence"] == ["initialize_arm", "step", "step", "initialize_arm"]
        return {
            "seed": seed,
            "options": options,
            "checkpoint": state,
            "result": first,
            "replayed_sha256": digest(again),
            "independent_sha256": digest(independent),
            "diffs": diffs,
        }
    finally:
        a.close()
        b.close()


def metadata_reads(cls, design, reads):
    env = cls(entropy_seed=17, design=design)
    try:
        env.reset(seed=17, options={"variation": [], "task_id": 2})
        state = capture(env)
        before = digest(state["ownership"])
        boxes = [env.action_space for _ in range(reads)]
        rng_changed = digest(capture(env)["ownership"]) != before
        with_reads = run_reset(env, {"variation": []})
        restore(env, state)
        without = run_reset(env, {"variation": []})
        diff = compare(with_reads, without)
        assert bool(diff) == (design == "B"), (design, reads, diff)
        assert all(type(box) is gym.spaces.Box and box.sample.__func__ is BOX_SAMPLE for box in boxes)
        return {
            "design": design,
            "reads": reads,
            "result_with_reads": with_reads,
            "result_without_reads": without,
            "diff": diff,
            "ownership_including_diagnostic_counter_changed": rng_changed,
            "same_box_on_all_reads": all(box is boxes[0] for box in boxes),
        }
    finally:
        env.close()


def last_box(cls):
    env = cls()
    try:
        env.reset(seed=17, options={"variation": [], "task_id": 2})
        original_box = env.last_box
        state = capture(env)
        output1 = env.reset(options={"variation": [], "task_id": 2})
        first_box = env.last_box
        first_rng = copy.deepcopy(first_box.np_random.bit_generator.state)
        first = result(env, output1)
        env.last_box = original_box
        restore(env, state)
        retained_rng = copy.deepcopy(original_box.np_random.bit_generator.state)
        output2 = env.reset(options={"variation": [], "task_id": 2})
        second_box = env.last_box
        second = result(env, output2)
        assert first_box is not second_box and original_box is not second_box
        assert retained_rng == original_box.np_random.bit_generator.state
        diff = compare(first, second)
        assert diff
        return {
            "checkpoint": state,
            "before_box_id": id(original_box),
            "first_box_id": id(first_box),
            "second_box_id": id(second_box),
            "first_box_post_sample_rng": first_rng,
            "retained_box_rng_unchanged": True,
            "result1": first,
            "result2": second,
            "diff": diff,
        }
    finally:
        env.close()


def isolation():
    global_before = copy.deepcopy(np.random.get_state())
    envs = {
        "cube0": OwnedCube(entropy_seed=42),
        "cube1": OwnedCube(entropy_seed=42),
        "scene0": OwnedScene(entropy_seed=42),
        "scene1": OwnedScene(entropy_seed=42),
    }
    try:
        for env in envs.values():
            env.reset(seed=42, options={"variation": [], "task_id": 2})
        states = {k: capture(e) for k, e in envs.items()}
        same_seed = {
            "cube": compare(states["cube0"], states["cube1"]),
            "scene": compare(states["scene0"], states["scene1"]),
        }
        assert not any(same_seed.values()), same_seed
        orders = [list(envs), list(reversed(envs))]
        results = []
        for order in orders:
            for key, env in envs.items():
                restore(env, states[key])
            row = {}
            # Reads across siblings/families must be observational.
            for key in order:
                for _ in range(5):
                    _ = envs[key].action_space.shape
            for key in order:
                row[key] = [
                    run_reset(envs[key], {"variation": ["cube.start_position"]}),
                    run_reset(envs[key], {"variation": []}),
                ]
            results.append(row)
        order_diffs = {k: compare(results[0][k], results[1][k]) for k in envs}
        assert not any(order_diffs.values()), order_diffs
        for key, env in envs.items():
            restore(env, states[key])
        # Genuine concurrent execution in one process, with no diagnostic patches.
        barrier = Barrier(4)
        thread_ids = {}

        def concurrent_reset(key, env):
            thread_ids[key] = get_ident()
            barrier.wait(timeout=60)
            return [run_reset(env, {"variation": ["cube.start_position"]}), run_reset(env, {"variation": []})]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {k: executor.submit(concurrent_reset, k, e) for k, e in envs.items()}
            threaded = {k: f.result() for k, f in futures.items()}
        assert len(set(thread_ids.values())) == 4
        threaded_diffs = {k: compare(results[0][k], threaded[k]) for k in envs}
        assert not any(threaded_diffs.values()), threaded_diffs
        # Different public seeds, then perturb only one family/instance.
        distinct = {k: run_reset(e, {"variation": []}, seed=17 if k.endswith("0") else 42) for k, e in envs.items()}
        assert compare(distinct["cube0"], distinct["cube1"])
        assert compare(distinct["scene0"], distinct["scene1"])
        unaffected = {k: capture(e) for k, e in envs.items() if k != "cube0"}
        run_reset(envs["cube0"], {"variation": ["cube.size"]}, seed=0)
        untouched_diffs = {k: compare(s, capture(envs[k])) for k, s in unaffected.items()}
        assert not any(untouched_diffs.values())
        assert not compare(global_before, np.random.get_state()), "global RNG mutated"
        return {
            "same_seed_diffs": same_seed,
            "orders": orders,
            "sequential_results": results,
            "threaded_results": threaded,
            "order_diffs": order_diffs,
            "threaded_diffs": threaded_diffs,
            "thread_ids": thread_ids,
            "different_seed_results": distinct,
            "unaffected_instance_diffs": untouched_diffs,
            "global_rng_unchanged": True,
        }
    finally:
        for env in envs.values():
            env.close()


def task_routes(cls):
    records = []
    for reward_id in [None, 0, 1, 5]:
        env = cls(entropy_seed=17, reward_task_id=reward_id)
        try:
            env.reset(seed=17, options={"variation": [], "task_id": 2})
            custom = copy.deepcopy(env.task_infos[3])
            for options in [{"variation": [], "task_id": 3}, {"variation": [], "task_info": custom}]:
                state = capture(env)
                before = copy.deepcopy(env.task_rng.get_state())
                first = run_reset(env, options)
                assert not compare(before, env.task_rng.get_state())
                restore(env, state)
                second = run_reset(env, options)
                diff = compare(first, second)
                assert not diff
                records.append(
                    {
                        "reward_id_argument": reward_id,
                        "options": options,
                        "result": first,
                        "diff": diff,
                        "task_rng_unchanged": True,
                    }
                )
        finally:
            env.close()
    return records


def cold_reads(cls):
    records = []
    for count in [1, 5]:
        a, b = cls(entropy_seed=17), cls(entropy_seed=17)
        try:
            assert a.action_box is None and b.action_box is None
            for _ in range(count):
                _ = a.action_space.dtype
            first, second = run_reset(a), run_reset(b)
            diff = compare(first, second)
            assert not diff
            records.append({"reads_before_first_reset": count, "first": first, "second": second, "diff": diff})
        finally:
            a.close()
            b.close()
    return records


def unseeded_construction(cls):
    a, b = cls(entropy_seed=None), cls(entropy_seed=None)
    try:
        # All entropy is materialized during construction, before any source reset.
        assert a._np_random is not None and b._np_random is not None
        assert a.entropy_seed != b.entropy_seed
        a.reset(options={"variation": [], "task_id": 2})
        b.reset(options={"variation": [], "task_id": 4})
        state = capture(a)
        first = run_reset(a)
        restore(a, state)
        second = run_reset(a)
        restore(b, state)
        independent = run_reset(b)
        diffs = {"same_instance": compare(first, second), "independent": compare(first, independent)}
        assert not any(diffs.values()), diffs
        return {"checkpoint": state, "result": first, "diffs": diffs, "constructor_entropy_captured": True}
    finally:
        a.close()
        b.close()


def main():
    assert gym.spaces.Box.sample is BOX_SAMPLE
    assert GETTER.__module__ == "ogbench.manipspace.envs.manipspace_env"
    assert seeding.np_random.__module__ == "gymnasium.utils.seeding"
    data = {"scope": "research RNG feasibility only; no public State acceptance", "families": {}}
    for name, cls, last in [("Cube", OwnedCube, LastCube), ("Scene", OwnedScene, LastScene)]:
        family = {
            "replay": [],
            "property_reads": [],
            "last_box": last_box(last),
            "design_b_replay": replay(cls, 17, {"variation": []}, design="B"),
        }
        data["families"][name] = family
        family["cold_reads"] = cold_reads(cls)
        family["unseeded_construction"] = unseeded_construction(cls)
        family["task_routes"] = task_routes(cls)
        for seed in [0, 17, 42]:
            for options in [
                None,
                {"variation": []},
                {"variation": ["cube.start_position", "cube.size"], "task_id": 3},
            ]:
                family["replay"].append(replay(cls, seed, options))
                print(name, "Design A replay", seed, options, "PASS", flush=True)
        for design in ["A", "B"]:
            for reads in [1, 5]:
                family["property_reads"].append(metadata_reads(cls, design, reads))
                print(name, design, "metadata reads", reads, "PASS" if design == "A" else "REJECTED", flush=True)
        # Task-native option/reward routes bypass random default selection.
        env = cls(entropy_seed=17)
        try:
            env.reset(seed=17, options={"variation": [], "task_id": 1})
            custom = copy.deepcopy(env.task_infos[4])
        finally:
            env.close()
        for options in [{"variation": [], "task_id": 5}, {"variation": [], "task_info": custom}]:
            family["replay"].append(replay(cls, 17, options))
        (OUT / "replay_results.json").write_text(json.dumps(plain(data), indent=2) + "\n")
    data["isolation"] = isolation()
    (OUT / "replay_results.json").write_text(json.dumps(plain(data), indent=2) + "\n")
    print("OGB_RNG_UNPATCHED_REPLAY_AND_THREADED_ISOLATION_PASS", flush=True)


if __name__ == "__main__":
    main()
