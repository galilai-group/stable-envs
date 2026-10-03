"""Navigation online source comparison on equal CPU allocations.

Unmodified source registrations expose scalar CPU rendering and stepping. Compare
serial and persistent shared-memory source workers with our native batched path.
Source CUDA-output rows include CPU rendering and delivery, not native GPU work.
The ordinary library additionally returns complete final State. Oracle export,
restore, construction and warmup stay outside all timed regions.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from scripts.benchmarks.online_stepping.classic_source_pipeline import image_errors
from scripts.benchmarks.online_stepping.navigation_parallel_source import (
    NAMES,
    ParallelNavigationSource,
    import_source,
    source,
    source_step,
)
from scripts.benchmarks.renderer_table.common import PIN, equal_state


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=NAMES, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    p.add_argument("--batches", nargs="+", type=int, default=[16, 64])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cpu-threads", type=int, default=8,
                   help="Parent Torch threads for both output devices; scalar-source workers remain at one")
    p.add_argument("--gate-only", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.workers < 1 or any(b < args.workers or b % args.workers for b in args.batches):
        p.error("Batches must be positive multiples of worker count")
    if args.cpu_threads < 1:
        p.error("Require a positive parent CPU thread count")
    torch.set_num_threads(1)
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest["commit"] == PIN
    for rel, digest in manifest["files"].items():
        assert hashlib.sha256((args.source_root / rel).read_bytes()).hexdigest() == digest, rel
    import_source(args.source_root)
    paths = [Path(__file__).relative_to(Path.cwd()),
             Path("scripts/benchmarks/online_stepping/navigation_parallel_source.py"),
             Path("scripts/benchmarks/online_stepping/classic_parallel_source.py"),
             Path("scripts/benchmarks/online_stepping/classic_source_pipeline.py"),
             Path("scripts/benchmarks/renderer_table/common.py"),
             Path("tests/helpers/reference/native_navigation_semantics.py"),
             Path("stable_envs/envs/navigation.py"), Path("stable_envs/env.py"),
             Path("stable_envs/_stepping.py")]
    report = dict(status="RUNNING", domain=args.domain, name=NAMES[args.domain],
                  host=platform.node(), job=os.environ.get("SLURM_JOB_ID"),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
                  source_pin=PIN, source_files=manifest["files"], source_gpu_supported=False,
                  source_render_device="cpu", source_workers=args.workers, source_native_batch=False,
                  source_device_api=False, source_threads_per_worker=1,
                  parent_torch_threads={"cpu": args.cpu_threads, "cuda": args.cpu_threads},
                  source_executor="Gymnasium AsyncVectorEnv, spawn, shared_memory=True, copy=True, no autoreset",
                  source_state_export_in_timing=False, ours_mandatory_final_state=True, ours_state_history=False,
                  gate_only=args.gate_only,
                  scope="equal CPU allocation; scalar serial/parallel source versus native batched Navigation",
                  initial_state_policy="seed 17 + lane; no changes to sampled reset State",
                  action_policy="float32 +/-0.01 alternating over eight steps",
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "gymnasium", "matplotlib", "tensordict")},
                  candidate_files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
                  source_runtime_files={}, gates=[], components=[], timings=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def sync(device):
        if device == "cuda":
            torch.cuda.synchronize()

    save()
    try:
        for device in args.devices:
            torch.set_num_threads(args.cpu_threads)
            for batch in args.batches:
                ref = source(args.domain, batch)
                try:
                    with se.make(NAMES[args.domain], num_envs=batch, seed=17, device=device) as env:
                        initial = env.reset(seed=17).cpu()
                        equal_state(initial, ref.get_state())
                        reset_state = initial.clone()
                        # Both callers hold terminal lanes and zero invalid task slots.
                        actions = np.full((batch, 8, 2), 0.01, dtype=np.float32)
                        actions[:, 1::2] *= -1
                        if not args.gate_only:
                            # These are candidate stage costs, not source-speedup ratios.
                            # Export is measured explicitly but never inserted into the
                            # source online baseline. Rendering uses full saved State.
                            selected_state = initial.to(device)
                            ops = {"raw_task_step_with_state": lambda: env._step(actions[:, 0]),
                                   "get_state": env.get_state,
                                   "render_saved_state": lambda: env.render(selected_state)}
                            for operation, call in ops.items():
                                env.set_state(selected_state)
                                call()
                                seconds = []
                                for _ in range(6):
                                    env.set_state(selected_state)
                                    sync(device)
                                    start = time.perf_counter()
                                    result = call()
                                    sync(device)
                                    seconds.append(time.perf_counter() - start)
                                    del result
                                report["components"].append(dict(device=device, batch=batch,
                                    operation=operation, seconds=seconds,
                                    rate_mean=statistics.mean(batch / t for t in seconds),
                                    rate_sd=statistics.stdev(batch / t for t in seconds)))
                            save()
                        for horizon in (1, 8):
                            action = actions[:, :horizon]
                            public = action[:, 0] if horizon == 1 else action
                            for selection, indices, frame_slice in (("none", (), None),
                                    ("final", (horizon - 1,), slice(-1, None)),
                                    ("all", tuple(range(horizon)), slice(None))):
                                with ParallelNavigationSource(args.domain, args.source_root, batch, args.workers,
                                                    horizon, indices) as parallel:
                                    assert len(parallel.vector.processes) == args.workers
                                    report["source_runtime_files"].update(parallel.runtime_files)
                                    # Workers use lane-offset seeds; compare before any restore.
                                    equal_state(reset_state, parallel.snapshot())
                                    restore = {"source_serial": lambda: ref.set_state(initial),
                                               "source_parallel": lambda: parallel.restore(initial),
                                               "ours": lambda: env.set_state(initial.to(device))}
                                    functions = {"source_serial": lambda: source_step(ref, action, indices, device),
                                                 "source_parallel": lambda: parallel.step(action, device),
                                                 "ours": lambda: env.step(public, states=None, frames=frame_slice)}
                                    for mode in restore:
                                        restore[mode]()
                                    expected = functions["source_serial"]()
                                    batched = functions["source_parallel"]()
                                    actual = functions["ours"]()
                                    valid_steps = int(actual["valid"].sum())
                                    assert valid_steps > 0
                                    equal_state(ref.get_state(), parallel.snapshot())
                                    equal_state(actual["state"], ref.get_state())
                                    for key in expected:
                                        torch.testing.assert_close(batched[key], expected[key], rtol=0, atol=0)
                                    for key in expected:
                                        if key == "frames":
                                            continue
                                        path = tuple(key.split(".")) if "." in key else key
                                        value = actual[path].unsqueeze(1) if horizon == 1 and key != "valid" else actual[path]
                                        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                                    errors = image_errors(actual["frames"], expected["frames"]) if indices else []
                                    assert all(e <= .001 for e in errors)
                                    del expected, actual
                                    # A subsequent shared-buffer write must not overwrite
                                    # any returned task/image tensor on either destination.
                                    held = {k: v.clone() for k, v in batched.items()}
                                    changed = action.copy()
                                    changed[:, :, 0] = 0.015
                                    next_output = parallel.step(changed, device)
                                    assert not torch.equal(next_output["observation"], held["observation"])
                                    for key in held:
                                        torch.testing.assert_close(batched[key], held[key], rtol=0, atol=0)
                                    report["gates"].append(dict(device=device, batch=batch, horizon=horizon,
                                        selection=selection, valid_steps=valid_steps, complete_state_task="PASS", parallel_source_exact="PASS",
                                        source_output_ownership="PASS", frames_checked=len(errors),
                                        mse_mean=statistics.mean(errors) if errors else None,
                                        mse_sd=statistics.stdev(errors) if len(errors) > 1 else (0 if errors else None),
                                        mse_max=max(errors) if errors else None))
                                    del batched, held, next_output
                                    if not args.gate_only:
                                        samples = {key: [] for key in functions}
                                        for block in range(2):
                                            order = tuple(functions) if block == 0 else tuple(reversed(functions))
                                            for mode in order:
                                                restore[mode]()
                                                functions[mode]()
                                                for _ in range(3):
                                                    restore[mode]()
                                                    sync(device)
                                                    start = time.perf_counter()
                                                    result = functions[mode]()
                                                    sync(device)
                                                    elapsed = time.perf_counter() - start
                                                    if mode == "ours":
                                                        assert int(result["valid"].sum()) == valid_steps
                                                    samples[mode].append(elapsed)
                                                    del result
                                        for mode, seconds in samples.items():
                                            steps = valid_steps
                                            frame_count = batch * len(indices)
                                            rates = [steps / s for s in seconds]
                                            report["timings"].append(dict(device=device, batch=batch, horizon=horizon,
                                                selection=selection, mode=mode, seconds=seconds, steps=steps,
                                                frames=frame_count, steps_per_s_mean=statistics.mean(rates),
                                                steps_per_s_sd=statistics.stdev(rates),
                                                fps_mean=statistics.mean(frame_count / s for s in seconds) if indices else None,
                                                fps_sd=statistics.stdev(frame_count / s for s in seconds) if indices else None,
                                                ours_workers=0))
                                    save()
                                    print(args.domain, device, batch, horizon, selection, "PASS", flush=True)
                finally:
                    ref.close()
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
