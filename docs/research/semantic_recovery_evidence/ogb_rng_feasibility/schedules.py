"""Explicit round-robin and different-seed concurrent cross-family isolation."""

import copy
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from threading import Barrier

import numpy as np
from gates import compare, plain, run_reset
from ownership import BOX_SAMPLE, OwnedCube, OwnedScene, capture, gym, restore, result

OUT = Path(__file__).resolve().parent


def main():
    assert gym.spaces.Box.sample is BOX_SAMPLE
    global_before = copy.deepcopy(np.random.get_state())
    envs = {
        "cube0": OwnedCube(entropy_seed=17),
        "cube1": OwnedCube(entropy_seed=42),
        "scene0": OwnedScene(entropy_seed=17),
        "scene1": OwnedScene(entropy_seed=42),
    }
    try:
        for key, env in envs.items():
            env.reset(seed=17 if key.endswith("0") else 42, options={"variation": [], "task_id": 2})
        states = {k: capture(e) for k, e in envs.items()}
        options = [{"variation": ["cube.start_position"]}, {"variation": []}]
        baseline = {k: [run_reset(e, option) for option in options] for k, e in envs.items()}
        for key, env in envs.items():
            restore(env, states[key])
        order = ["cube0", "scene1", "cube1", "scene0"]
        scheduled = {k: [] for k in envs}
        events = []
        for cycle in [0, 1]:
            for key in order if cycle == 0 else reversed(order):
                before = {k: result(e, None) for k, e in envs.items() if k != key}
                for other in envs:
                    for _ in range(3):
                        _ = envs[other].action_space.low
                scheduled[key].append(run_reset(envs[key], options[cycle]))
                diffs = {k: compare(value, result(envs[k], None)) for k, value in before.items()}
                assert not any(diffs.values()), diffs
                events.append({"cycle": cycle, "actor": key, "nonactor_diffs": diffs})
        scheduled_diffs = {k: compare(baseline[k], scheduled[k]) for k in envs}
        assert not any(scheduled_diffs.values()), scheduled_diffs
        for key, env in envs.items():
            restore(env, states[key])
        barrier = Barrier(4)

        def run(env):
            barrier.wait(timeout=60)
            return [run_reset(env, option) for option in options]

        with ThreadPoolExecutor(max_workers=4) as executor:
            futures = {k: executor.submit(run, e) for k, e in envs.items()}
            concurrent = {k: future.result() for k, future in futures.items()}
        concurrent_diffs = {k: compare(baseline[k], concurrent[k]) for k in envs}
        assert not any(concurrent_diffs.values()), concurrent_diffs
        assert not compare(global_before, np.random.get_state())
        output = {
            "baseline": baseline,
            "round_robin": scheduled,
            "events": events,
            "scheduled_diffs": scheduled_diffs,
            "threaded_diffs": concurrent_diffs,
            "threaded_results": concurrent,
            "global_rng_unchanged": True,
        }
        (OUT / "schedule_results.json").write_text(json.dumps(plain(output), indent=2) + "\n")
        print("OGB_DIFFERENT_SEED_ROUND_ROBIN_AND_CONCURRENT_ISOLATION_PASS")
    finally:
        for env in envs.values():
            env.close()


if __name__ == "__main__":
    main()
