"""Descriptive public-State benchmarks; immutable outputs and explicit native scope."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import io
import json
import os
import platform
import subprocess
import sys
import time
import traceback
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"


def catalog():
    rows = json.loads((ROOT / "docs/research/semantic_recovery_implementation_map.json").read_text())["rows"]
    return {r["historical_id"]: r for r in rows if r["current_status"] == "VERIFIED"}


def synchronize(value=None):
    import torch

    if torch.cuda.is_available():
        torch.cuda.synchronize()
    if "jax" in sys.modules:
        import jax

        jax.block_until_ready(value)
    return value


def measure(fn, *, prepare=None, warmup=10, iterations=50, minimum_seconds=0.25):
    import numpy as np

    cold = None
    sink = None
    samples = []
    for i in range(1 + warmup + iterations):
        if prepare is not None:
            prepare()
        synchronize()
        started = time.perf_counter_ns()
        sink = synchronize(fn())
        elapsed = time.perf_counter_ns() - started
        if i == 0:
            cold = elapsed
        elif i > warmup:
            samples.append(elapsed)
    # Adapt upwards for very fast operations, with a bounded maximum run length.
    while sum(samples) < minimum_seconds * 1e9 and len(samples) < max(iterations, 1000):
        if prepare is not None:
            prepare()
        synchronize()
        started = time.perf_counter_ns()
        sink = synchronize(fn())
        samples.append(time.perf_counter_ns() - started)
    median = float(np.median(samples))
    # Returned values stay live until completion; no unused lazy/JIT call is timed.
    return {
        "status": "MEASURED",
        "median_ns": median,
        "p95_ns": float(np.percentile(samples, 95)),
        "operations_per_second": 1e9 / median,
        "mean_ns": float(np.mean(samples)),
        "coefficient_of_variation": float(np.std(samples) / np.mean(samples)),
        "cold_first_call_ns": cold,
        "warmup_iterations": warmup,
        "measured_iterations": len(samples),
        "samples_ns": samples,
        "construction_included": False,
        "cold_first_call_may_include_compilation": True,
        "warm_compilation_included": False,
        "prepare_excluded": prepare is not None,
        "result_type": type(sink).__name__,
    }


def profile():
    import torch

    names = [
        "torch",
        "tensordict",
        "numpy",
        "gymnasium",
        "mujoco",
        "dm-control",
        "gymnasium-robotics",
        "ogbench",
        "craftax",
        "jax",
        "jaxlib",
        "flax",
        "pymunk",
        "pybullet",
        "mujoco-warp",
        "warp-lang",
        "matplotlib",
        "pillow",
        "pygame",
        "glfw",
    ]
    versions = {}
    for name in names:
        try:
            versions[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            pass
    cpu = platform.processor()
    if sys.platform == "darwin":
        cpu = subprocess.check_output(["sysctl", "-n", "machdep.cpu.brand_string"]).decode().strip()
    elif Path("/proc/cpuinfo").exists():
        cpu = next(
            (
                line.split(":", 1)[1].strip()
                for line in Path("/proc/cpuinfo").read_text().splitlines()
                if line.startswith("model name")
            ),
            cpu,
        )
    return {
        "python": platform.python_version(),
        "executable": sys.executable,
        "platform": platform.platform(),
        "cpu": cpu,
        "gpu": torch.cuda.get_device_name() if torch.cuda.is_available() else None,
        "packages": versions,
        "torch_threads": torch.get_num_threads(),
        "environment": {
            k: os.environ.get(k)
            for k in [
                "MUJOCO_GL",
                "JAX_PLATFORMS",
                "JAX_ENABLE_X64",
                "CRAFTAX_JAX_BACKEND",
                "OMP_NUM_THREADS",
                "SLURM_JOB_ID",
            ]
        },
    }


def authoritative(registration):
    if "Craftax" in registration:
        # Same bounded oracle loader used by the accepted Craftax gates; avoid
        # unrelated SWM learner imports in the pinned minimal Craftax profile.
        import importlib.util

        path = Path(os.environ["SWM_ROOT"]) / "stable_worldmodel/envs/craftax/craftax.py"
        spec = importlib.util.spec_from_file_location("state_bench_craftax_oracle", path)
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        name = registration.split("/")[-1].split("-v")[0] + "Wrapper"
        return getattr(module, name)(backend=os.environ.get("CRAFTAX_JAX_BACKEND", "cpu"))
    import gymnasium as gym
    import stable_worldmodel.envs  # noqa: F401

    return gym.make(registration)


def engine_chain(env):
    seen = set()
    queue = [env]
    while queue:
        obj = queue.pop(0)
        if id(obj) in seen:
            continue
        seen.add(id(obj))
        yield obj
        for name in ["env", "_env", "unwrapped"]:
            child = getattr(obj, name, None)
            if child is not None and child is not obj:
                queue.append(child)


def native_baseline(source):
    """Official lower bounds only; wrapper/task/RNG omissions are never hidden."""
    import numpy as np

    for obj in engine_chain(source):
        physics = getattr(obj, "physics", None)
        if physics is not None and hasattr(physics, "get_state"):
            snapshot = physics.get_state()
            return (
                {
                    "classification": "NATIVE_PARTIAL",
                    "mechanism": "dm_control.Physics.get_state/set_state",
                    "like_for_like": False,
                    "omits": "task, RNG, lifecycle, model changes and warm-start state",
                },
                physics.get_state,
                lambda: physics.set_state(snapshot),
            )
        model, data = getattr(obj, "model", None), getattr(obj, "data", None)
        if model is not None and data is not None:
            import mujoco

            mask = mujoco.mjtState.mjSTATE_INTEGRATION
            size = mujoco.mj_stateSize(model, mask)

            def get():
                array = np.empty(size, dtype=np.float64)
                mujoco.mj_getState(model, data, array, mask)
                return array

            snapshot = get()
            return (
                {
                    "classification": "NATIVE_PARTIAL",
                    "mechanism": "MuJoCo mjSTATE_INTEGRATION get/set",
                    "like_for_like": False,
                    "omits": "task, RNG, wrapper lifecycle, model configuration and output caches",
                },
                get,
                lambda: mujoco.mj_setState(model, data, snapshot, mask),
            )
        if hasattr(obj, "craftax_env"):
            import jax
            import jax.numpy as jnp

            def get():
                return jax.tree.map(lambda x: jnp.array(x, copy=True), obj._state)

            snapshot = get()

            def restore():
                obj._state = jax.tree.map(lambda x: jnp.array(x, copy=True), snapshot)
                return obj._state

            return (
                {
                    "classification": "NATIVE_PARTIAL",
                    "mechanism": "JAX explicit engine pytree copy/install",
                    "like_for_like": False,
                    "omits": "wrapper PRNG key, lifecycle and immediate observation caches",
                },
                get,
                restore,
            )
    return (
        {
            "classification": "NONE",
            "mechanism": None,
            "like_for_like": False,
            "omits": "No documented complete native snapshot operation in the pinned source",
        },
        None,
        None,
    )


def run(args):
    import numpy as np

    import stable_envs as se

    row = catalog()[args.registration]
    names = row.get("accepted_classes") or row["stable_envs_classes"]
    name = names[0]
    output = {
        "schema": 1,
        "registration": args.registration,
        "stable_envs_class": name,
        "family": row["family"],
        "execution_mode": row["execution_mode"],
        "device": args.device,
        "oracle_commit": PIN,
        "source_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT).decode().strip()
        if (ROOT / ".git").exists()
        else os.environ.get("STATE_BENCH_SOURCE_COMMIT"),
        "command": sys.argv,
        "profile": profile(),
        "batches": {},
        "batch_64": f"DEFERRED_RUNTIME: requested public batches {args.batches}; not emulated",
        "notes": [
            "Descriptive measurement, not acceptance or an optimization gate.",
            "Oracle native API is scalar; oracle/native timings use B=1, never an emulated batch.",
            "No extra render call in dynamics; required pixel observations inside oracle step are identified.",
        ],
    }
    args.output.mkdir(parents=True, exist_ok=False)

    def save():
        (args.output / "result.json").write_text(json.dumps(output, indent=2) + "\n")

    def timed(fn, prepare=None):
        return measure(
            fn, prepare=prepare, warmup=args.warmup, iterations=args.iterations, minimum_seconds=args.minimum_seconds
        )

    save()
    source_actions = None
    for batch in args.batches:
        cell = {"batch_size": batch, "operations": {}}
        output["batches"][str(batch)] = cell
        a = b = None
        try:
            t = time.perf_counter_ns()
            a = se.make(name, num_envs=batch, device=args.device)
            synchronize()
            cell["cold_construction_ns"] = time.perf_counter_ns() - t
            b = se.make(name, num_envs=batch, device=args.device)
            a.reset(seed=17)
            b.reset(seed=42)
            rng = np.random.default_rng(12345)
            actions = [a.action_spec.sample(batch, rng=rng) for _ in range(args.history_steps + 2)]
            if batch == 1:
                source_actions = [v[0].detach().cpu().numpy().copy() for v in actions]
                output["actions_lane_zero"] = [v.tolist() for v in source_actions]
            # Reach a nontrivial checkpoint using the public API.
            for action in actions[:2]:
                a.step(action)
            snapshot = a.get_state()
            payload = {
                json.dumps(list(k) if isinstance(k, tuple) else [k]): v.detach().cpu().numpy()
                for k, v in snapshot.items(True, True)
            }
            buffer = io.BytesIO()
            np.savez(buffer, **payload)
            cell["state_size"] = {
                "numeric_payload_bytes": sum(v.nbytes for v in payload.values()),
                "serialized_npz_uncompressed_bytes": buffer.tell(),
                "leaf_count": len(payload),
                "format": "named numeric NPZ, no pickle",
                "sha256": hashlib.sha256(buffer.getvalue()).hexdigest(),
            }
            cell["implementation"] = (
                type(getattr(a, "_simulator", a)).__module__ + "." + type(getattr(a, "_simulator", a)).__name__
            )
            ops = cell["operations"]

            def divergence(target):
                for action in actions[: args.history_steps]:
                    target.step(action)

            def branch():
                saved = a.get_state()
                divergence(a)
                a.set_state(saved)
                result = None
                for action in actions[-2:]:
                    result = a.step(action)
                return result

            operations = (
                []
                if args.render_only
                else [
                    ("reset", lambda: a.reset(seed=17), None),
                    ("step", lambda: a.step(actions[-1]), lambda: a.set_state(snapshot)),
                    ("get_state", a.get_state, lambda: a.set_state(snapshot)),
                    ("set_state_same", lambda: a.set_state(snapshot), lambda: divergence(a)),
                    ("set_state_independent", lambda: b.set_state(snapshot), lambda: divergence(b)),
                    ("get_then_set", lambda: a.set_state(a.get_state()), lambda: a.set_state(snapshot)),
                    ("branch_cycle", branch, lambda: a.set_state(snapshot)),
                ]
            )
            for label, fn, prepare in operations:
                ops[label] = timed(fn, prepare)
                ops[label]["environment_operations_per_second"] = batch * ops[label]["operations_per_second"]
                print(args.registration, batch, label, ops[label]["median_ns"], flush=True)
                save()
            for label, fn in [
                ("to_render_state", lambda: a.to_render_state(snapshot)),
                ("render", lambda: a.render(snapshot)),
            ]:
                try:
                    ops[label] = timed(fn)
                except (NotImplementedError, AttributeError, ImportError) as exc:
                    ops[label] = {"status": "UNSUPPORTED", "reason": str(exc)}
                save()
            if not args.render_only:
                step = ops["step"]["median_ns"]
                cell["ratios"] = {
                    k + "_over_step": ops[k]["median_ns"] / step
                    for k in ["get_state", "set_state_same", "set_state_independent", "get_then_set"]
                }
                cell["branch"] = {
                    "divergent_steps": args.history_steps,
                    "continuation_steps": 2,
                    "median_ns": ops["branch_cycle"]["median_ns"],
                }
            cell["status"] = "MEASURED"
        except Exception as exc:
            cell.update(status="ERROR", error=repr(exc), traceback=traceback.format_exc())
            print(cell["traceback"], file=sys.stderr, flush=True)
        finally:
            for env in [a, b]:
                if env is not None:
                    env.close()
            save()
    if not args.render_only:
        source = None
        try:
            t = time.perf_counter_ns()
            source = authoritative(args.registration)
            output["oracle"] = {
                "batch_size": 1,
                "cold_construction_ns": time.perf_counter_ns() - t,
                "step_includes_required_pixel_observation": "Pixels" in args.registration,
                "explicit_render_calls": 0,
            }
            source.reset(seed=17)
            if source_actions is None:
                raise ValueError("Include batch 1 for matched authoritative actions")
            import gymnasium as gym

            def oracle_action(value):
                return int(value.item()) if isinstance(source.action_space, gym.spaces.Discrete) else value

            action = oracle_action(source_actions[-1])
            oracle = output["oracle"]
            oracle["reset"] = timed(lambda: source.reset(seed=17))

            # Same reset seed and two-action prefix as the public benchmark checkpoint.
            def prepare_oracle():
                source.reset(seed=17)
                for value in source_actions[:2]:
                    source.step(oracle_action(value))

            oracle["step"] = timed(lambda: source.step(action), prepare_oracle)
            classification, get, restore = native_baseline(source)
            oracle["native"] = classification
            if get is not None:
                oracle["native"]["get"] = timed(get)
                oracle["native"]["set"] = timed(restore)
            for cell in output["batches"].values():
                if cell.get("status") == "MEASURED" and cell["batch_size"] == 1 and get is not None:
                    cell["native_ratios_not_like_for_like"] = {
                        "get_over_native_get": cell["operations"]["get_state"]["median_ns"]
                        / oracle["native"]["get"]["median_ns"],
                        "set_over_native_set": cell["operations"]["set_state_same"]["median_ns"]
                        / oracle["native"]["set"]["median_ns"],
                    }
        except Exception as exc:
            output["oracle_error"] = {"error": repr(exc), "traceback": traceback.format_exc()}
            print(traceback.format_exc(), file=sys.stderr, flush=True)
        finally:
            if source is not None:
                source.close()
            save()
    output["status"] = (
        "COMPLETE"
        if all(c.get("status") == "MEASURED" for c in output["batches"].values()) and "oracle_error" not in output
        else "INCOMPLETE"
    )
    save()
    print("STATE_BENCHMARK_" + output["status"], args.registration, flush=True)
    return 0 if output["status"] == "COMPLETE" else 1


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--registration", choices=sorted(catalog()), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    parser.add_argument("--warmup", type=int, default=10)
    parser.add_argument("--iterations", type=int, default=50)
    parser.add_argument("--minimum-seconds", type=float, default=0.25)
    parser.add_argument("--history-steps", type=int, default=8)
    parser.add_argument("--render-only", action="store_true")
    args = parser.parse_args()
    if min(args.batches) < 1 or args.iterations < 1 or args.warmup < 0 or args.history_steps < 2:
        parser.error("Invalid measurement counts")
    raise SystemExit(run(args))


if __name__ == "__main__":
    main()
