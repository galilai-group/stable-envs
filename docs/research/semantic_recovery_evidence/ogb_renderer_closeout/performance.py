"""CPU-only restore characterization; no production changes or optimization."""

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from stable_envs._adapters.ogbench_manipulation import OGBManipulationAdapter

HERE = Path(__file__).resolve().parent


def summary(samples):
    a = np.asarray(samples)
    return dict(
        n=len(a),
        first_ms=float(a[0] * 1000),
        median_ms=float(np.median(a) * 1000),
        p95_ms=float(np.percentile(a, 95) * 1000),
        max_ms=float(a.max() * 1000),
        operations_per_second=float(1 / np.median(a)),
    )


def timed(call):
    start = time.perf_counter()
    call()
    return time.perf_counter() - start


def exact(a, b):
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    assert all(torch.equal(a[k], b[k]) for k in a.keys(True, True))


def measure(family, batch):
    construct = []
    cold = {k: [] for k in ["step", "get_state", "set_state_same", "set_state_physical_change"]}
    for _ in range(5):
        start = time.perf_counter()
        env = OGBManipulationAdapter(family, batch, seed=17)
        construct.append(time.perf_counter() - start)
        saved = env.get_state()
        cold["step"].append(timed(lambda: env.step(np.zeros((batch, 5)))))
        cold["get_state"].append(timed(env.get_state))
        cold["set_state_same"].append(timed(lambda: env.set_state(saved)))
        env.configure([{"cube.size": np.array([0.03])} for _ in range(batch)])
        env.reset(17, {"variation": []})
        changed = env.get_state()
        cold["set_state_physical_change"].append(timed(lambda: env.set_state(saved)))
        exact(saved, env.get_state())
        env.close()
    env = OGBManipulationAdapter(family, batch, seed=17)
    saved = env.get_state()
    try:
        warm = {}
        operations = {
            "step": lambda: env.step(np.zeros((batch, 5))),
            "get_state": env.get_state,
            "set_state_same": lambda: env.set_state(saved),
        }
        for name, call in operations.items():
            call()
            warm[name] = summary([timed(call) for _ in range(30)])
        # Alternate physical configurations; both directions require changed geometry.
        next_state = changed
        times = []
        for _ in range(30):
            times.append(timed(lambda: env.set_state(next_state)))
            exact(next_state, env.get_state())
            next_state = saved if next_state is changed else changed
        warm["set_state_physical_change"] = summary(times)
        stage_names = ["validation", "constructor", "compilation", "state_row_copy"]
        stage_times = {k: [] for k in stage_names}
        stage_calls = {k: 0 for k in stage_names}
        originals = {
            "validation": env.validate,
            "constructor": env.cls.__init__,
            "compilation": env.cls.compile_model_and_data,
            "state_row_copy": env._row,
        }

        def observer(name):
            def call(*args, **kwargs):
                start = time.perf_counter()
                result = originals[name](*args, **kwargs)
                stage_times[name].append(time.perf_counter() - start)
                stage_calls[name] += 1
                return result

            return call

        with (
            patch.object(env, "validate", observer("validation")),
            patch.object(env.cls, "__init__", observer("constructor")),
            patch.object(env.cls, "compile_model_and_data", observer("compilation")),
            patch.object(env, "_row", observer("state_row_copy")),
        ):
            totals = [timed(lambda: env.set_state(saved)) for _ in range(10)]
        stages = {
            k: dict(
                summary(stage_times[k]),
                calls_per_restore=stage_calls[k] / 10,
                mean_ms_per_batched_restore=sum(stage_times[k]) * 1000 / 10,
            )
            for k in stage_names
        }
        stages["other_reconstruction_and_swap_mean_ms"] = (
            sum(totals) - sum(sum(v) for v in stage_times.values())
        ) * 100
        state_copy = summary([timed(saved.clone) for _ in range(100)])
        for name in ["set_state_same", "set_state_physical_change"]:
            warm[name]["relative_to_step"] = warm[name]["median_ms"] / warm["step"]["median_ms"]
            warm[name]["individual_world_restores_per_second"] = batch * warm[name]["operations_per_second"]
        return dict(
            family=family,
            batch=batch,
            cold_definition="First public operation of its kind on each of five newly constructed adapters, in this already-loaded process; not fresh-process import timing.",
            constructor=summary(construct),
            cold={k: summary(v) for k, v in cold.items()},
            warm=warm,
            instrumented_restore_stages=stages,
            tensordict_clone=state_copy,
            exact_restores=True,
        )
    finally:
        env.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, default=HERE / "performance.json")
    args = parser.parse_args()
    result = dict(
        profile={
            p: version(p) for p in ["mujoco", "ogbench", "numpy", "gymnasium", "dm-control", "torch", "tensordict"]
        },
        units="milliseconds except explicitly named throughput/ratios",
        timing_scope="CPU wall time; no GPU, no optimization; per-call batched throughput",
        rows=[],
    )
    for family in ["Cube", "Scene"]:
        for batch in [1, 3]:
            row = measure(family, batch)
            result["rows"].append(row)
            args.output.write_text(json.dumps(result, indent=2) + "\n")
            print(
                f"RESTORE_PERF {family} B={batch} step_ms={row['warm']['step']['median_ms']:.3f} get_ms={row['warm']['get_state']['median_ms']:.3f} same_ms={row['warm']['set_state_same']['median_ms']:.3f} changed_ms={row['warm']['set_state_physical_change']['median_ms']:.3f}",
                flush=True,
            )
    print("RESTORE_CHARACTERIZATION_PASS", flush=True)


if __name__ == "__main__":
    main()
