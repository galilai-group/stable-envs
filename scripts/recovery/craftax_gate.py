"""Collect real per-registration Craftax arrays and synchronized measurements."""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata as metadata
import json
import os
import platform
import time
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--performance", action="store_true")
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("JAX_PLATFORMS", "cuda" if args.device == "cuda" else "cpu")
    os.environ["CRAFTAX_JAX_BACKEND"] = "gpu" if args.device == "cuda" else "cpu"
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    import jax
    import numpy as np
    import torch

    import stable_envs as se
    from stable_envs._adapters.craftax import kernel, numeric_leaves
    from stable_envs._adapters.craftax_render import fields as draw_fields
    from tests.helpers.reference.craftax_source import (
        ACTIONS,
        assert_engine,
        assert_state_equal,
        fixture,
        oracle,
        restore_oracle,
        symbolic_output,
    )

    def collect_numpy(value):
        # JAX collection is explicit; Torch CPU transfers also synchronize CUDA.
        if isinstance(value, torch.Tensor):
            return value.detach().cpu().numpy()
        jax.block_until_ready(value)
        return np.asarray(value)

    def capture(label, env, source, expected):
        full = env.get_state()
        actual = (
            collect_numpy(env._simulator.output_fn(env._simulator.states[0]))
            if env.PIXELS
            else symbolic_output(env, state=full)
        )
        arrays = {"state/" + "/".join(k): collect_numpy(v) for k, v in full.items(True, True)}
        arrays["source_output"], arrays["candidate_output"] = expected, actual
        arrays.update({"source_engine/" + "/".join(k): collect_numpy(v) for k, v in numeric_leaves(source._state)})
        arrays["source_wrapper_key"] = collect_numpy(source._key)
        # Preserve both sides before assertions so failed GPU gates remain diagnosable.
        np.savez_compressed(args.output / (label + ".npz"), **arrays)
        assert_engine(source, env, full)
        if env.PIXELS:
            draw = env.to_render_state(full)
            candidate = collect_numpy(env.render(full))[0]
            private = collect_numpy(env.render(draw))[0]
            reference = (np.clip(expected, 0, 1) * 255).astype(np.uint8)
            arrays.update(source_frame=reference, full_frame=candidate, render_frame=private)
            np.savez_compressed(args.output / (label + ".npz"), **arrays)
            np.testing.assert_array_equal(candidate, reference)
            np.testing.assert_array_equal(candidate, private)
            output_exact = bool(np.array_equal(actual, expected))
            exact = True
        else:
            np.savez_compressed(args.output / (label + ".npz"), **arrays)
            np.testing.assert_array_equal(actual, expected)
            output_exact = exact = True
        assert_state_equal(full, env.get_state())
        return {
            "label": label,
            "state_exact": True,
            "required_output_exact": exact,
            "normalized_float_output_exact": output_exact,
            "output_shape": list(actual.shape),
            "output_dtype": str(actual.dtype),
            "output_min": float(actual.min()),
            "output_max": float(actual.max()),
            "nonmutation": True,
            "array_file": label + ".npz",
        }

    profile = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "packages": {
            n: metadata.version(n)
            for n in ["craftax", "jax", "jaxlib", "flax", "gymnax", "chex", "numpy", "torch", "tensordict"]
        },
        "devices": [str(d) for d in jax.devices()],
        "backend": jax.default_backend(),
        "x64": jax.config.jax_enable_x64,
        "prng": jax.config.jax_default_prng_impl,
        "threefry_partitionable": jax.config.jax_threefry_partitionable,
        "environment": {
            k: os.environ.get(k)
            for k in ["JAX_PLATFORMS", "JAX_ENABLE_X64", "XLA_FLAGS", "XLA_PYTHON_CLIENT_PREALLOCATE"]
        },
    }
    if args.device == "cuda":
        assert jax.default_backend() == "gpu" and torch.cuda.is_available()
        profile["gpu"] = torch.cuda.get_device_name()
        profile["jax_memory_before"] = jax.devices()[0].memory_stats()
    (args.output / "profile.json").write_text(json.dumps(profile, indent=2) + "\n")
    results = []
    source = oracle(args.name)
    started = time.perf_counter()
    with se.make(args.name, device=args.device) as env:
        constructor_seconds = time.perf_counter() - started
        schema = [{"key": list(k), "shape": list(s), "dtype": str(d)} for k, (s, d) in env._simulator.fields.items()]
        render_schema = (
            [
                {
                    "key": ["draw", *k],
                    "shape": list(v.shape),
                    "dtype": "torch.int64" if v.dtype == np.uint32 else str(torch.from_numpy(v.copy()).dtype),
                }
                for k, v in draw_fields(env._simulator).items()
            ]
            if env.PIXELS
            else []
        )
        (args.output / "schema.json").write_text(
            json.dumps(
                {
                    "state": schema,
                    "render_fields": render_schema,
                    "render_signature": {"key": ["renderer", "signature"], "shape": [32], "dtype": "torch.uint8"}
                    if env.PIXELS
                    else None,
                    "params": dataclasses.asdict(env._simulator.upstream.default_params),
                    "static": dataclasses.asdict(env._simulator.upstream.static_env_params),
                },
                indent=2,
            )
            + "\n"
        )
        for seed in [0, 17, 42]:
            expected, _ = source.reset(seed=seed)
            env.reset(seed=seed)
            results.append(capture("reset-" + str(seed), env, source, expected))
        for case, action in ACTIONS.items():
            source.reset(seed=999)
            env.reset(seed=999)
            state = fixture(env, args.name, case)
            restore_oracle(source, env, state)
            env.set_state(state)
            if not bool(state["lifecycle", "terminated"][0]):
                expected, reward, done, truncated, _ = source.step(action)
                result = env.step([action], capture="states")
            else:
                expected = np.asarray(
                    jax.jit(source._env.get_obs, backend=os.environ["CRAFTAX_JAX_BACKEND"])(source._state)
                )
            results.append(capture(case, env, source, expected))
            if not bool(state["lifecycle", "terminated"][0]):
                assert float(result["reward"][0]) == reward
                assert bool(result["terminated"][0]) == done and not truncated
    performance = {"constructor_seconds_in_current_process": constructor_seconds, "batches": {}}
    if args.performance:

        def timed(function, repeats=10):
            values = []
            for _ in range(repeats + 1):
                start = time.perf_counter()
                result = function()
                if isinstance(result, jax.Array):
                    jax.block_until_ready(result)
                if args.device == "cuda":
                    torch.cuda.synchronize()
                values.append(time.perf_counter() - start)
            return {
                "first_seconds": values[0],
                "warm_median_seconds": float(np.median(values[1:])),
                "warm_p95_seconds": float(np.percentile(values[1:], 95)),
                "warm_calls_per_second": float(1 / np.median(values[1:])),
                "samples_seconds": values,
            }

        for batch in [1, 3]:
            with se.make(args.name, num_envs=batch, device=args.device) as env:
                state = env.reset(seed=17)
                record = {
                    "step": timed(lambda: env.step([0] * batch, capture="states")),
                    "reset": timed(lambda: env.reset(seed=17)),
                    "get_state": timed(env.get_state),
                    "set_state": timed(lambda: env.set_state(state)),
                }
                symbolic = kernel(env.CLASSIC, False, env._simulator.platform)[3]

                def symbolic_batch():
                    outputs = [symbolic(s) for s in env._simulator.states]
                    jax.block_until_ready(outputs)
                    return outputs

                record["symbolic_output"] = timed(symbolic_batch)
                record["pixel_render"] = (
                    timed(env.render) if env.PIXELS else {"status": "SOURCE_EXPLICITLY_UNSUPPORTED"}
                )
                performance["batches"][str(batch)] = record
        performance["public_execution"] = "JIT only, matching SWM; scalar source kernels with explicit public batching"
    report = {
        "registration": "swm/" + args.name + "-v1",
        "name": args.name,
        "status": "PASS",
        "cases": results,
        "profile": profile,
        "performance": performance,
        "resource_policy": "Instance close drops State/keys; bounded family/device kernel cache and upstream texture/JIT caches are process-lifetime runtime resources, never semantic State.",
        "cross_device_exactness": "NOT_CLAIMED; compare preserved CPU/GPU arrays separately",
        "marker": "CRAFTAX_" + args.name.upper() + "_" + ("GPU" if args.device == "cuda" else "CPU") + "_GATE_PASS",
    }
    (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
    paths = sorted(p for p in args.output.rglob("*") if p.is_file() and p.name != "SHA256SUMS")
    (args.output / "SHA256SUMS").write_text(
        "".join(
            hashlib.sha256(p.read_bytes()).hexdigest() + "  " + str(p.relative_to(args.output)) + "\n" for p in paths
        )
    )
    print(report["marker"], flush=True)


if __name__ == "__main__":
    main()
