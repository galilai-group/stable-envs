"""Equal-allocation classic source comparison, including shared-memory source workers.

All strategies get the same CPU allocation. Source pygame rendering is CPU only;
CUDA rows measure delivery to CUDA for source and native CUDA rendering for ours.
The ordinary library additionally returns complete final State. No oracle export
or restore is inside a timing region. No production route is changed here.
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
from scripts.benchmarks.online_stepping.classic_parallel_source import ParallelSource
from scripts.benchmarks.online_stepping.classic_source_pipeline import (
    NAMES,
    PIN,
    actions_for,
    equal_state,
    image_errors,
    import_source,
    source,
    source_step,
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=NAMES, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    p.add_argument("--batches", nargs="+", type=int, default=[64, 256])
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--cpu-threads", type=int, default=1,
                   help="Parent Torch intra-op threads for CPU output; spawned source/simulator workers remain at one")
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
             Path("scripts/benchmarks/online_stepping/classic_parallel_source.py"),
             Path("scripts/benchmarks/online_stepping/classic_source_pipeline.py"),
             Path("scripts/benchmarks/online_stepping/classic_state_export.py"),
             Path("scripts/benchmarks/renderer_table/common.py"),
             Path("tests/helpers/reference/native_classic_semantics.py"),
             Path("stable_envs/envs/_classic_semantics.py"), Path("stable_envs/envs/classic_control.py"),
             Path("stable_envs/_stepping.py"), Path("stable_envs/_adapters/process_requests.py"),
             Path("stable_envs/_adapters/classic/process.py"),
             *Path("stable_envs/_adapters").rglob("*.py")]
    report = dict(status="RUNNING", domain=args.domain, name=NAMES[args.domain],
                  host=platform.node(), job=os.environ.get("SLURM_JOB_ID"),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
                  source_pin=PIN, source_files=manifest["files"], source_gpu_supported=False,
                  source_render_device="cpu", source_workers=args.workers,
                  parent_torch_threads={"cpu": args.cpu_threads, "cuda": 1},
                  source_executor="Gymnasium AsyncVectorEnv, spawn, shared_memory=True, copy=True, no autoreset",
                  source_state_export_in_timing=False, ours_mandatory_final_state=True, ours_state_history=False,
                  gate_only=args.gate_only,
                  scope="equal CPU allocation; serial and bounded parallel pinned source versus ordinary adapter",
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "gymnasium", "pygame", "tensordict")},
                  candidate_files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
                  source_runtime_files={}, gates=[], timings=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def sync(device):
        if device == "cuda":
            torch.cuda.synchronize()

    save()
    try:
        for device in args.devices:
            torch.set_num_threads(args.cpu_threads if device == "cpu" else 1)
            for batch in args.batches:
                ref = source(args.domain, batch)
                try:
                    with se.make(NAMES[args.domain], num_envs=batch, seed=17, device=device) as env:
                        initial = env.get_state().cpu()
                        equal_state(initial, ref.get_state())
                        actions = actions_for(args.domain, batch, 31)
                        if args.domain == "cartpole":
                            actions[:] = np.arange(8)[None, :, None] % 2
                        for horizon in (1, 8):
                            action = actions[:, :horizon]
                            public = action[:, 0] if horizon == 1 else action
                            for selection, indices, frame_slice in (("none", (), None),
                                    ("final", (horizon - 1,), slice(-1, None)),
                                    ("all", tuple(range(horizon)), slice(None))):
                                with ParallelSource(args.domain, args.source_root, batch, args.workers,
                                                    horizon, indices) as parallel:
                                    assert len(parallel.vector.processes) == args.workers
                                    report["source_runtime_files"].update(parallel.runtime_files)
                                    equal_state(initial, parallel.snapshot())
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
                                    assert bool(actual["valid"].all())
                                    assert not bool(expected["terminated"][:, :-1].any() or expected["truncated"][:, :-1].any())
                                    equal_state(ref.get_state(), parallel.snapshot())
                                    equal_state(actual["state"], ref.get_state())
                                    for key in expected:
                                        torch.testing.assert_close(batched[key], expected[key], rtol=0, atol=0)
                                    for key in ("observation", "reward", "terminated", "truncated", "info_state"):
                                        value = actual[key].unsqueeze(1) if horizon == 1 else actual[key]
                                        torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                                    errors = image_errors(actual["frames"], expected["frames"]) if indices else []
                                    assert all(e <= .001 for e in errors)
                                    del expected, actual
                                    # A subsequent shared-buffer write must not overwrite
                                    # any returned task/image tensor on either destination.
                                    held = {k: v.clone() for k, v in batched.items()}
                                    next_output = parallel.step(action, device)
                                    assert not torch.equal(next_output["observation"], held["observation"])
                                    for key in held:
                                        torch.testing.assert_close(batched[key], held[key], rtol=0, atol=0)
                                    report["gates"].append(dict(device=device, batch=batch, horizon=horizon,
                                        selection=selection, complete_state_task="PASS", parallel_source_exact="PASS",
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
                                                        assert bool(result["valid"].all())
                                                    samples[mode].append(elapsed)
                                                    del result
                                        for mode, seconds in samples.items():
                                            steps = batch * horizon
                                            frame_count = batch * len(indices)
                                            rates = [steps / s for s in seconds]
                                            report["timings"].append(dict(device=device, batch=batch, horizon=horizon,
                                                selection=selection, mode=mode, seconds=seconds, steps=steps,
                                                frames=frame_count, steps_per_s_mean=statistics.mean(rates),
                                                steps_per_s_sd=statistics.stdev(rates),
                                                fps_mean=statistics.mean(frame_count / s for s in seconds) if indices else None,
                                                fps_sd=statistics.stdev(frame_count / s for s in seconds) if indices else None,
                                                ours_workers=len(env._simulator.worker_info) if hasattr(env._simulator, "worker_info") else 0))
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
