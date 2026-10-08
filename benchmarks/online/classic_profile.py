"""Separate classic-control transition, State export and restore costs.

Diagnostic only: no rendering or upstream speedup is measured. The no-export
cell substitutes an already owned State inside adapter.step solely to isolate
task/physics cost. It is never exposed as a valid transitioned State.
"""

from __future__ import annotations

import argparse
import cProfile
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import pstats
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from stable_envs.envs.classic_control.simulation.local import CLASSIC_FIELDS, ClassicControlAdapter


def make(family, batch, device):
    adapter = ClassicControlAdapter(family, batch, device, 17)
    adapter.reset(seed=17)
    return adapter


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family", choices=CLASSIC_FIELDS, required=True)
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 64, 256])
    parser.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 2 or min(args.batches) < 1:
        parser.error("Require positive batches and at least two samples")
    torch.set_num_threads(1)
    report = dict(
        status="RUNNING",
        family=args.family,
        host=platform.node(),
        job=os.environ.get("SLURM_JOB_ID"),
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
        versions={p: importlib.metadata.version(p) for p in ("torch", "numpy", "gymnasium", "tensordict")},
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path("stable_envs/envs/classic_control/semantics.py"), Path(__file__).relative_to(Path.cwd())]
        },
        scope="adapter component diagnostic, CPU physics for both output devices; no rendering or source speedup",
        cells=[],
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for batch in args.batches:
            for device in args.devices:
                start = time.perf_counter()
                adapter = make(args.family, batch, device)
                construction = time.perf_counter() - start
                try:
                    initial = adapter.get_state()
                    action = (
                        np.random.default_rng(31).integers(0, 2 if args.family == "cartpole" else 3, (batch, 1))
                        if args.family in ("cartpole", "acrobot", "mountain_car")
                        else np.random.default_rng(31).uniform(-0.5, 0.5, (batch, 1))
                    )
                    adapter.step(action)
                    cell = dict(
                        batch=batch,
                        device=device,
                        construction_s=construction,
                        state_bytes=sum(v.numel() * v.element_size() for v in initial.values(True, True)),
                        state_leaves=len(initial.keys(True, True)),
                        timings={},
                    )

                    def no_export():
                        with patch.object(adapter, "get_state", return_value=initial):
                            return adapter.step(action)

                    for name, function in (
                        ("step_with_state", lambda: adapter.step(action)),
                        ("step_without_export_DIAGNOSTIC", no_export),
                        ("get_state", adapter.get_state),
                        ("set_state", lambda: adapter.set_state(initial)),
                    ):
                        samples = []
                        for _ in range(args.repeats):
                            adapter.set_state(initial)
                            sync(device)
                            start = time.perf_counter()
                            result = function()
                            sync(device)
                            samples.append(time.perf_counter() - start)
                            del result
                        rates = [batch / t for t in samples]
                        cell["timings"][name] = dict(
                            seconds=samples,
                            seconds_mean=statistics.mean(samples),
                            seconds_sd=statistics.stdev(samples),
                            lanes_per_s_mean=statistics.mean(rates),
                            lanes_per_s_sd=statistics.stdev(rates),
                        )
                    adapter.set_state(initial)
                    profile = cProfile.Profile()
                    profile.enable()
                    adapter.step(action)
                    sync(device)
                    profile.disable()
                    output = io.StringIO()
                    pstats.Stats(profile, stream=output).sort_stats("cumulative").print_stats(40)
                    cell["step_cprofile"] = output.getvalue()
                    adapter.set_state(initial)
                    first = adapter.step(action)
                    adapter.set_state(initial)
                    replay = adapter.step(action)
                    assert first.batch_size == replay.batch_size
                    assert set(first.keys(True, True)) == set(replay.keys(True, True))
                    for key in first.keys(True, True):
                        torch.testing.assert_close(first[key], replay[key], rtol=0, atol=0, equal_nan=True)
                    cell["restore_step_replay"] = "PASS"
                    report["cells"].append(cell)
                    save()
                    print(args.family, batch, device, "PASS", flush=True)
                finally:
                    adapter.close()
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"], report["error"] = "FAIL", repr(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
