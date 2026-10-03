"""Sustained snapshot comparison, one strategy/cell per fresh interpreter.

Normal GC stays enabled. There is no competing baseline population or restore
between calls. Seeded reset is outside each window and separately reported.
This diagnoses two pending classic candidates; it is not an upstream baseline.
"""

import argparse
import gc
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

from tests.helpers.reference.classic_snapshot import make


def digest(value):
    result = hashlib.sha256()
    for key in sorted(value.keys(True, True), key=str):
        tensor = value[key].detach().cpu().contiguous()
        result.update(str((key, tuple(tensor.shape), str(tensor.dtype))).encode())
        result.update(tensor.numpy().tobytes())
    return result.hexdigest()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=("pendulum", "acrobot"), required=True)
    p.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    p.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--horizon", type=int, choices=(1, 8), required=True)
    p.add_argument("--states", choices=("none", "all"), required=True)
    p.add_argument("--frames", choices=("none", "final"), required=True)
    p.add_argument("--transitions", type=int, default=128)
    p.add_argument("--samples", type=int, default=6)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if not (0 < args.transitions < 200) or args.transitions % args.horizon:
        p.error("Require a positive multiple of horizon below either task's time limit")
    if args.batch < 1 or args.samples < 2:
        p.error("Require positive batch and at least two samples")
    torch.set_num_threads(1)
    assert gc.isenabled()
    files = ["stable_envs/envs/_classic_semantics.py", "stable_envs/_adapters/acrobot_process.py",
             "stable_envs/_adapters/classic_execution.py", "stable_envs/_adapters/process_requests.py",
             "stable_envs/envs/classic_control.py", "stable_envs/_stepping.py",
             "tests/helpers/reference/classic_before_snapshot.py",
             "tests/helpers/reference/acrobot_before_snapshot.py",
             "tests/helpers/reference/classic_snapshot.py", str(Path(__file__).relative_to(Path.cwd()))]
    report = dict(status="RUNNING", domain=args.domain, mode=args.mode, device=args.device,
                  batch=args.batch, horizon=args.horizon, states=args.states, frames=args.frames,
                  transitions_per_lane=args.transitions, samples=[],
                  host=platform.node(), job=os.environ.get("SLURM_JOB_ID"), pid=os.getpid(),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
                  gc_enabled=True, gc_thresholds=gc.get_threshold(),
                  scope="independent-process sustained candidate vs frozen library; no source speedup",
                  timing="synchronized window; consecutive calls including result disposal; reset outside",
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "gymnasium", "tensordict")},
                  source_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def sync():
        if args.device == "cuda":
            torch.cuda.synchronize()

    # Zero torque keeps this bounded workload away from early success. Gates
    # still count every validity flag; no padded transition is counted as work.
    action = np.full((args.batch, args.horizon, 1), 1 if args.domain == "acrobot" else 0,
                     dtype=np.int64 if args.domain == "acrobot" else np.float32)
    if args.horizon == 1:
        action = action[:, 0]
    states = slice(None) if args.states == "all" else None
    frames = slice(-1, None) if args.frames == "final" else None
    calls = args.transitions // args.horizon
    save()
    try:
        with make(args.domain, args.batch, args.device, legacy=args.mode == "baseline") as env:
            report["workers"] = len(env._simulator.worker_info) if hasattr(env._simulator, "worker_info") else 0
            report["renderer_backend"] = env.backend
            # Untimed full-trajectory fingerprints compare every selected State,
            # task output and image across the separate strategy interpreters.
            env.reset(17)
            hashes = []
            for _ in range(calls):
                output = env.step(action, states=states, frames=frames)
                assert bool(output["valid"].all())
                hashes.append(digest(output))
                del output
            report["trajectory_sha256"] = hashes
            report["gate_final_state_sha256"] = digest(env.get_state())
            for sample in range(args.samples):
                sync()
                start = time.perf_counter()
                env.reset(17)
                sync()
                reset_seconds = time.perf_counter() - start
                before_gc = gc.get_stats()
                valid_flags = []
                sync()
                start = time.perf_counter()
                for _ in range(calls):
                    output = env.step(action, states=states, frames=frames)
                    # Keep only tiny owned validity tensors for counting after
                    # timing; intermediate images and States are released here.
                    valid_flags.append(output["valid"])
                    del output
                sync()
                seconds = time.perf_counter() - start
                after_gc = gc.get_stats()
                valid_steps = sum(int(v.sum()) for v in valid_flags)
                assert valid_steps == args.batch * args.transitions
                final = env.get_state()
                assert bool((final["lifecycle", "elapsed_steps"] == args.transitions).all())
                final_digest = digest(final)
                assert final_digest == report["gate_final_state_sha256"]
                emitted_frames = args.batch * calls if frames is not None else 0
                report["samples"].append(dict(index=sample, seconds=seconds, reset_seconds=reset_seconds,
                    valid_steps=valid_steps, frames=emitted_frames, calls=calls,
                    steps_per_s=valid_steps / seconds, fps=emitted_frames / seconds if emitted_frames else None,
                    final_state_sha256=final_digest,
                    gc_collections=[b["collections"] - a["collections"] for a, b in zip(before_gc, after_gc, strict=True)]))
                del final, valid_flags
                save()
            rates = [s["steps_per_s"] for s in report["samples"]]
            report["steps_per_s_mean"], report["steps_per_s_sd"] = statistics.mean(rates), statistics.stdev(rates)
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
