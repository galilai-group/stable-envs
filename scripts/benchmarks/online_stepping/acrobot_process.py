"""Paired Acrobot public-step throughput: accepted local exporter vs process owners.

Physics is CPU-side on both output devices. All intermediate States are retained;
rendering is disabled in timings. Exact native-frame gates run in the accompanying
test suite before this benchmark. No source-renderer or end-to-end claim follows.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import stable_envs as se
import stable_envs.envs._classic_semantics as semantics
from stable_envs._adapters.classic import execution as classic_execution
from stable_envs._adapters.classic.process import AcrobotProcessAdapter
from tests.unit.test_classic_state_export import equal


def make(batch, device, processes, cpus):
    options = dict(num_envs=batch, device=device, seed=17)
    if not processes:
        with patch.object(classic_execution, "process_count", return_value=0):
            return se.make("Acrobot", **options)
    with (patch.object(classic_execution, "process_count", return_value=0),
          patch.object(semantics, "ClassicControlAdapter",
                       partial(AcrobotProcessAdapter, processes=processes, cpus=cpus))):
        env = se.make("Acrobot", **options)
    env._step_pipeline = lambda: env._simulator
    return env


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    p.add_argument("--batches", nargs="+", type=int, default=[64, 256, 1024])
    p.add_argument("--processes", nargs="+", type=int, default=[4, 8])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 2 or min(args.batches) < max(args.processes) or min(args.processes) < 1:
        p.error("Require >=2 repeats and positive process counts no larger than the smallest batch")
    torch.set_num_threads(1)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if cpus is not None and len(cpus) < max(args.processes):
        p.error("Insufficient allocated CPUs")
    files = [Path(name) for name in (
        "stable_envs/_adapters/classic/process.py", "stable_envs/_adapters/process_requests.py",
        "stable_envs/_adapters/classic/execution.py",
        "stable_envs/envs/_classic_semantics.py", "stable_envs/envs/classic_control.py",
        "stable_envs/_stepping.py", "tests/unit/test_acrobot_process.py")]
    files.append(Path(__file__).relative_to(Path.cwd()))
    report = dict(status="RUNNING", env="swm/AcrobotControl-v1", host=platform.node(),
                  job=os.environ.get("SLURM_JOB_ID"), affinity=cpus,
                  gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
                  versions={name: importlib.metadata.version(name) for name in
                            ("torch", "numpy", "gymnasium", "tensordict")},
                  source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                  scope="CPU physics; owned all-State public steps, frames=None; direct-column local control; no upstream speedup",
                  timings=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for device in args.devices:
            for batch in args.batches:
                modes = [0, *args.processes]
                samples = {(mode, horizon): [] for mode in modes for horizon in (1, 8)}
                actions = np.random.default_rng(31).integers(0, 3, (batch, 8, 1))
                with make(batch, device, 0, cpus) as control:
                    initial = control.get_state()
                    expected = {}
                    for horizon in (1, 8):
                        control.set_state(initial)
                        expected[horizon] = control.step(actions[:, 0] if horizon == 1 else actions, frames=None)
                for block in range(2):
                    for mode in modes if block == 0 else reversed(modes):
                        with make(batch, device, mode, cpus) as env:
                            for horizon in (1, 8):
                                a = actions[:, 0] if horizon == 1 else actions
                                env.set_state(initial)
                                equal(expected[horizon], env.step(a, frames=None))
                                for _ in range(args.repeats):
                                    env.set_state(initial)
                                    sync(device)
                                    start = time.perf_counter()
                                    result = env.step(a, frames=None)
                                    sync(device)
                                    elapsed = time.perf_counter() - start
                                    assert bool(result["valid"].all())
                                    samples[mode, horizon].append(elapsed)
                for (mode, horizon), seconds in samples.items():
                    rates = [batch * horizon / t for t in seconds]
                    report["timings"].append(dict(device=device, batch=batch, horizon=horizon, processes=mode,
                                                 seconds=seconds, valid_transitions=batch * horizon,
                                                 steps_per_s_mean=statistics.mean(rates),
                                                 steps_per_s_sd=statistics.stdev(rates),
                                                 exact_state_task_gate="PASS"))
                save()
                print(device, batch, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
