"""Profile accepted DMC adapters before changing their execution strategy.

This is a component diagnostic, not the public pipeline/source-speedup table.
Timers synchronize CUDA output; cProfile runs separately from timed samples.
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

import numpy as np
import torch

from stable_envs._adapters.dmc.aligned import CONFIG as ALIGNED
from stable_envs._adapters.dmc.aligned import DMCAlignedAdapter
from stable_envs._adapters.dmc.continuation import CONFIG as CONTINUATION
from stable_envs._adapters.dmc.continuation import DMCContinuationAdapter
from stable_envs._adapters.dmc.first_family import FirstFamilyAdapter, ReacherAdapter
from stable_envs._adapters.dmc.humanoid import DMCHumanoidAdapter

DOMAINS = ["acrobot", "pendulum", "cartpole", "reacher", *CONTINUATION, *ALIGNED, "humanoid"]


def make(domain, batch, device):
    if domain in ("acrobot", "pendulum", "cartpole"):
        return FirstFamilyAdapter(domain, batch, device, 17)
    if domain == "reacher":
        return ReacherAdapter(batch, device, 17)
    if domain in CONTINUATION:
        return DMCContinuationAdapter(domain, batch, device, 17)
    if domain in ALIGNED:
        return DMCAlignedAdapter(domain, batch, device, 17)
    return DMCHumanoidAdapter(batch, device, 17)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--domain", choices=DOMAINS, required=True)
    parser.add_argument("--batches", nargs="+", type=int, default=[1, 64, 256])
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    parser.add_argument("--repeats", type=int, default=6)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 2 or min(args.batches) < 1:
        parser.error("Require positive batches and at least two repeats")
    torch.set_num_threads(1)
    report = dict(
        status="RUNNING",
        domain=args.domain,
        host=platform.node(),
        job=os.environ.get("SLURM_JOB_ID"),
        affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
        versions={
            name: importlib.metadata.version(name) for name in ("torch", "numpy", "dm-control", "mujoco", "tensordict")
        },
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [*Path("stable_envs/_adapters").rglob("*.py"), *Path("stable_envs/_execution").rglob("*.py"), Path(__file__).relative_to(Path.cwd())]
        },
        scope="adapter.step with complete State; diagnostic only, not public step or a source speedup",
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
                adapter = make(args.domain, batch, device)
                construction = time.perf_counter() - start
                try:
                    initial = adapter.get_state()
                    actions = np.random.default_rng(21).uniform(
                        -0.5, 0.5, (batch, adapter.envs[0].action_spec().shape[0])
                    )
                    adapter.step(actions)
                    cell = dict(
                        batch=batch,
                        device=device,
                        construction_s=construction,
                        state_bytes=sum(x.numel() * x.element_size() for x in initial.values(True, True)),
                        timings={},
                    )
                    for name, fn in (
                        ("step_with_state", lambda: adapter.step(actions)),
                        ("get_state", adapter.get_state),
                        ("set_state", lambda: adapter.set_state(initial)),
                    ):
                        samples = []
                        for _ in range(args.repeats):
                            adapter.set_state(initial)
                            sync(device)
                            start = time.perf_counter()
                            result = fn()
                            sync(device)
                            samples.append(time.perf_counter() - start)
                            del result
                        rates = [batch / seconds for seconds in samples]
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
                    adapter.step(actions)
                    sync(device)
                    profile.disable()
                    output = io.StringIO()
                    pstats.Stats(profile, stream=output).sort_stats("cumulative").print_stats(35)
                    cell["step_cprofile"] = output.getvalue()
                    # Explicitly exercise wrapper reset-next-step boundaries. These
                    # registrations expose false done flags even when DMC resets.
                    boundary = initial.clone()
                    boundary["lifecycle", "step_count"][:] = int(adapter.envs[0]._env._step_limit) - 1
                    boundary["lifecycle", "reset_next_step"].fill_(False)
                    adapter.set_state(boundary)
                    first = adapter.step(actions)
                    adapter.set_state(boundary)
                    replay = adapter.step(actions)
                    for key in first.keys(True, True):
                        torch.testing.assert_close(first[key], replay[key], rtol=0, atol=0, equal_nan=True)
                    cell["boundary_replay"] = "PASS"
                    cell["boundary_terminated"] = first["terminated"].cpu().tolist()
                    cell["boundary_truncated"] = first["truncated"].cpu().tolist()
                    report["cells"].append(cell)
                    save()
                    print(
                        args.domain, batch, device, cell["timings"]["step_with_state"]["lanes_per_s_mean"], flush=True
                    )
                finally:
                    adapter.close()
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = repr(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
