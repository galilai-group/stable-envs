"""Bounded raw upstream eager/JIT characterization; no public eager-mode claim."""

from __future__ import annotations

import argparse
import json
import time
from contextlib import nullcontext
from pathlib import Path

import jax
import numpy as np
from craftax.craftax_env import make_craftax_env_from_name

from stable_envs._adapters.craftax import numeric_leaves


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--classic", action="store_true")
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    env = make_craftax_env_from_name(
        "Craftax-" + ("Classic-" if args.classic else "") + "Symbolic-v1", auto_reset=False
    )
    key = jax.random.PRNGKey(17)
    results, states = {}, {}
    state = None
    for mode in ["jit", "eager"]:
        record = {}
        with jax.disable_jit() if mode == "eager" else nullcontext():
            for operation in ["reset", "step"]:
                times = []
                for _ in range(2):
                    start = time.perf_counter()
                    result = (
                        env.reset(key, env.default_params)
                        if operation == "reset"
                        else env.step(key, state, 0, env.default_params)
                    )
                    jax.block_until_ready(result)
                    times.append(time.perf_counter() - start)
                if operation == "reset":
                    state = result[1]
                values = dict(numeric_leaves(result[1]))
                np.savez_compressed(
                    args.output / f"{mode}-{operation}.npz",
                    **{"engine/" + "/".join(k): v for k, v in values.items()},
                    output=np.asarray(result[0]),
                )
                states[mode, operation] = values
                record[operation] = {"first_seconds": times[0], "second_seconds": times[1]}
                print(mode, operation, times, flush=True)
        results[mode] = record
    for operation in ["reset", "step"]:
        results[operation + "_eager_jit_differences"] = {
            "/".join(k): float(np.max(np.abs(v.astype(float) - states["eager", operation][k].astype(float))))
            for k, v in states["jit", operation].items()
            if not np.array_equal(v, states["eager", operation][k])
        }
    results["scope"] = (
        "Two raw upstream calls per path, one process; first includes compilation/runtime warmup. "
        "Eager is diagnostic only. SWM and public adapter use JIT. Independent eager/JIT worlds "
        "may differ; arrays preserved, no eager/JIT or cross-device identity claim."
    )
    (args.output / "result.json").write_text(json.dumps(results, indent=2) + "\n")


if __name__ == "__main__":
    main()
