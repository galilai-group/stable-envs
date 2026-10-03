"""Revisit retained slow cells using original actions and one fresh strategy process.

Every call restores the original initial State outside timing, matching the
original paired benchmark. All 192 measured call durations are retained; no
forced collection or co-resident control population is used.
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

import torch

from scripts.benchmarks.online_stepping.classic_snapshot import actions_for
from scripts.benchmarks.online_stepping.classic_snapshot_sustained import digest
from tests.helpers.reference.classic_snapshot import make

CELLS = (
    ("mountain_car_continuous", "cpu", 256, 1, "none", "final"),
    ("mountain_car_continuous", "cpu", 1024, 8, "all", "final"),
    ("mountain_car_continuous", "cuda", 1024, 8, "all", "final"),
    ("acrobot", "cpu", 256, 8, "all", "none"),
    ("acrobot", "cuda", 1024, 8, "all", "none"),
    ("acrobot", "cuda", 1024, 1, "all", "final"),
    ("acrobot", "cpu", 64, 1, "all", "none"),
)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--cell", type=int, choices=range(len(CELLS)), required=True)
    p.add_argument("--mode", choices=("baseline", "candidate"), required=True)
    p.add_argument("--windows", type=int, default=6)
    p.add_argument("--calls", type=int, default=32)
    p.add_argument("--smoke", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.windows, args.calls) < 2:
        p.error("Require at least two windows and two calls")
    domain, device, batch, horizon, states_name, frames_name = CELLS[args.cell]
    if args.smoke:
        device, batch = "cpu", 4
    torch.set_num_threads(1)
    assert gc.isenabled()
    files = ["stable_envs/envs/_classic_semantics.py", "stable_envs/_adapters/acrobot_process.py",
             "stable_envs/_adapters/classic_execution.py", "stable_envs/_adapters/process_requests.py",
             "stable_envs/envs/classic_control.py", "stable_envs/_stepping.py",
             "tests/helpers/reference/classic_before_snapshot.py", "tests/helpers/reference/acrobot_before_snapshot.py",
             "tests/helpers/reference/classic_snapshot.py", "scripts/benchmarks/online_stepping/classic_snapshot.py",
             "scripts/benchmarks/online_stepping/classic_snapshot_sustained.py", str(Path(__file__).relative_to(Path.cwd()))]
    report = dict(status="RUNNING", cell=args.cell, mode=args.mode, domain=domain, device=device,
                  batch=batch, horizon=horizon, states=states_name, frames=frames_name, smoke=args.smoke,
                  host=platform.node(), job=os.environ.get("SLURM_JOB_ID"), pid=os.getpid(),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  gpu=torch.cuda.get_device_name() if device == "cuda" else None,
                  gc_thresholds=gc.get_threshold(), gc_enabled=True, windows=[],
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "gymnasium", "tensordict")},
                  source_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
                  scope="original reset/action workload in fresh strategy interpreter; not sustained online or source speedup")
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def sync():
        if device == "cuda":
            torch.cuda.synchronize()

    save()
    try:
        with make(domain, batch, device, legacy=args.mode == "baseline") as env:
            report["workers"] = len(env._simulator.worker_info) if hasattr(env._simulator, "worker_info") else 0
            initial = env.get_state()
            actions = actions_for(domain, batch)
            action = actions[:, 0] if horizon == 1 else actions
            states = slice(None) if states_name == "all" else None
            frames = slice(-1, None) if frames_name == "final" else None
            expected = env.step(action, states=states, frames=frames)
            assert bool(expected["valid"].all())
            report["output_sha256"] = digest(expected)
            report["initial_sha256"] = digest(initial)
            del expected
            for window in range(args.windows):
                times, restores = [], []
                for _ in range(args.calls):
                    sync()
                    start = time.perf_counter()
                    env.set_state(initial)
                    sync()
                    restores.append(time.perf_counter() - start)
                    start = time.perf_counter()
                    output = env.step(action, states=states, frames=frames)
                    sync()
                    times.append(time.perf_counter() - start)
                    assert bool(output["valid"].all())
                    del output
                env.set_state(initial)
                checked = env.step(action, states=states, frames=frames)
                assert digest(checked) == report["output_sha256"]
                del checked
                report["windows"].append(dict(index=window, seconds=times, restore_seconds=restores,
                    valid_steps=batch * horizon * args.calls,
                    emitted_frames=batch * args.calls if frames is not None else 0,
                    steps_per_s=batch * horizon * args.calls / sum(times),
                    exact_output="PASS", gc_collections=[s["collections"] for s in gc.get_stats()]))
                save()
            rates = [w["steps_per_s"] for w in report["windows"]]
            report["steps_per_s_mean"], report["steps_per_s_sd"] = statistics.mean(rates), statistics.stdev(rates)
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
