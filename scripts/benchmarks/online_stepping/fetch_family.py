"""Registration-specific Fetch executor and online pipeline comparison.

Run one task per allocation, all its suffixes sequentially. Both output devices
and all execution layouts share the same CPUs. Rendering is measured in that
same allocation. Full State and task equality are checked outside every sample.
"""

from __future__ import annotations

import argparse
import gc
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
from fetch_process_adapter import equal, sync

import stable_envs as snv
import stable_envs.envs._fetch_semantics as semantics
from stable_envs._adapters.fetch.adapter import FetchAdapter
from stable_envs._adapters.fetch.process import FetchProcessAdapter


def make(name, batch, device, layout, cpus, size):
    constructor = FetchAdapter
    if layout != "current":
        constructor = partial(FetchProcessAdapter, processes=int(layout.removeprefix("process")), cpus=cpus)
    with patch.object(semantics, "FetchAdapter", constructor):
        return snv.make(name, num_envs=batch, seed=17, device=device, backend="opengl", size=size)


def gate(name, device, cpus, size):
    batch = 64 if cpus is not None else 8
    actions = np.random.default_rng(31).uniform(-1, 1, (batch, 6, 4)).astype(np.float32)
    checks = []
    for layout in ("process4", "process8"):
        with make(name, batch, device, "current", cpus, size) as reference:
            with make(name, batch, device, layout, cpus, size) as candidate:
                initial = reference.get_state()
                equal(initial, candidate.get_state())
                for selection in (slice(None), slice(None, None, -2), None):
                    for env in (reference, candidate):
                        env.set_state(initial)
                    expected = reference.step(actions, states=selection, frames=None)
                    actual = candidate.step(actions, states=selection, frames=None)
                    equal(expected, actual)
                owned = actual.clone()
                candidate.reset(seed=42)
                equal(owned, actual)
                for env in (reference, candidate):
                    env.set_state(initial)
                equal(reference.reset(), candidate.reset())
                for env in (reference, candidate):
                    env.set_state(initial)
                    env._simulator._set_episode_limits(np.arange(batch) % 3 + 1)
                expected = reference.step(actions, frames=None)
                actual = candidate.step(actions, frames=None)
                equal(expected, actual)
                assert actual["valid"].sum(1).tolist() == (np.arange(batch) % 3 + 1).tolist()
                variation = {"table.color": np.array([0.2, 0.7, 0.4])}
                if "Reach" not in name:
                    variation.update({"block.mass": np.array([0.9]), "object.color": np.array([0.8, 0.1, 0.3])})
                for env in (reference, candidate):
                    env._simulator._set_episode_limits(np.full(batch, 50))
                equal(
                    reference.reset(seed=42, options={"variation_values": variation}),
                    candidate.reset(seed=42, options={"variation_values": variation}),
                )
                equal(
                    reference.step(actions.astype(np.float64), frames=None),
                    candidate.step(actions.astype(np.float64), frames=None),
                )
                frame_mse = None
                if device == "cuda":
                    assert torch.equal(reference.render(), candidate.render())
                    frame_mse = 0.0
                checks.append(
                    dict(
                        env=name,
                        device=device,
                        layout=layout,
                        batch=batch,
                        state_task_rng_replay_lifecycle_variation="PASS",
                        frame_mse_vs_current=frame_mse,
                    )
                )
    return checks


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--task", choices=["Reach", "Push", "Slide", "PickAndPlace"], required=True)
    parser.add_argument(
        "--suffixes", nargs="+", choices=["plain", "Dense", "Dict"], default=["plain", "Dense", "Dict"]
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    parser.add_argument("--batches", type=int, nargs="+", default=[1, 8, 32, 64, 128, 256, 512, 1024])
    parser.add_argument("--pipeline-batches", type=int, nargs="*", default=[64, 256, 1024])
    parser.add_argument("--horizons", type=int, nargs="+", default=[1, 8])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--size", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert args.repeats >= 2 and args.blocks >= 1 and all(1 <= h < 50 for h in args.horizons)
    torch.set_num_threads(1)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if cpus is not None:
        assert len(cpus) == 8, cpus
    import stable_envs._adapters.fetch.adapter as local_module
    import stable_envs._adapters.fetch.process as process_module

    files = [Path(__file__), Path(local_module.__file__), Path(process_module.__file__), Path(semantics.__file__)]
    report = dict(
        status="RUNNING",
        task=args.task,
        host=platform.node(),
        affinity=cpus,
        job=os.environ.get("SLURM_JOB_ID"),
        partition=os.environ.get("SLURM_JOB_PARTITION"),
        cpu_info=Path("/proc/cpuinfo").read_text().split("\n\n")[0]
        if Path("/proc/cpuinfo").exists()
        else platform.processor(),
        gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
        versions={
            p: importlib.metadata.version(p) for p in ("torch", "numpy", "gymnasium", "gymnasium-robotics", "mujoco")
        },
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        configuration=vars(args) | {"output": str(args.output)},
        gates=[],
        timings=[],
        scope="Complete public Fetch step, all State/task histories. CPU physics with CPU/CUDA outputs. Pipeline: all frames, OpenGL, same CPU allocation. Render-only uses prepared final State. Restoration, startup, warmup, equality checks excluded. No policy inference or simulation/render overlap. Frame reference is current executor, not new upstream parity.",
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for suffix in args.suffixes:
        name = "Fetch" + args.task + ("" if suffix == "plain" else suffix)
        for device in args.devices:
            report["gates"].extend(gate(name, device, cpus, args.size))
            print("GATE_PASS", name, device, flush=True)
            save()
            for batch in args.batches:
                sequence = np.random.default_rng(51).uniform(-1, 1, (batch, max(args.horizons), 4)).astype(np.float32)
                modes = ["step_state"]
                if device == "cuda" and batch in args.pipeline_batches:
                    modes += ["step_render_all", "render_only"]
                expected = {}
                with make(name, batch, device, "current", cpus, args.size) as reference:
                    initial = reference.get_state().cpu()
                    for horizon in args.horizons:
                        actions = sequence[:, 0] if horizon == 1 else sequence[:, :horizon]
                        for mode in modes:
                            reference.set_state(initial)
                            if mode == "render_only":
                                state = expected[horizon, "step_state"]["state"].to(device)
                                expected[horizon, mode] = reference.render(state).cpu()
                            else:
                                expected[horizon, mode] = reference.step(
                                    actions,
                                    states=slice(None),
                                    frames=slice(None) if mode == "step_render_all" else None,
                                ).cpu()
                for block in range(args.blocks):
                    layouts = ["current"] + [f"process{p}" for p in (4, 8) if batch >= p]
                    if block % 2:
                        layouts.reverse()
                    for layout in layouts:
                        started = time.perf_counter()
                        with make(name, batch, device, layout, cpus, args.size) as env:
                            startup = time.perf_counter() - started
                            checkpoint = initial.to(device)
                            for horizon in args.horizons:
                                actions = sequence[:, 0] if horizon == 1 else sequence[:, :horizon]
                                render_state = expected[horizon, "step_state"]["state"].to(device)
                                for mode in modes:

                                    def run():
                                        if mode == "render_only":
                                            return env.render(render_state)
                                        return env.step(
                                            actions,
                                            states=slice(None),
                                            frames=slice(None) if mode == "step_render_all" else None,
                                        )

                                    def check(result):
                                        reference = expected[horizon, mode]
                                        if mode == "render_only":
                                            assert torch.equal(reference, result.cpu())
                                        else:
                                            equal(reference, result)

                                    env.set_state(checkpoint)
                                    check(run())
                                    sync(device)
                                    seconds = []
                                    for _ in range(args.repeats):
                                        env.set_state(checkpoint)
                                        sync(device)
                                        start = time.perf_counter()
                                        result = run()
                                        sync(device)
                                        seconds.append(time.perf_counter() - start)
                                        check(result)
                                        del result
                                    transitions = 0 if mode == "render_only" else batch * horizon
                                    frames = (
                                        batch
                                        if mode == "render_only"
                                        else transitions
                                        if mode == "step_render_all"
                                        else 0
                                    )
                                    row = dict(
                                        env=name,
                                        device=device,
                                        batch=batch,
                                        horizon=horizon,
                                        layout=layout,
                                        stage=mode,
                                        block=block,
                                        seconds=seconds,
                                        transitions=transitions,
                                        frames=frames,
                                        startup_seconds=startup,
                                        exact_parity="PASS",
                                        frame_mse_vs_current=0.0 if frames else None,
                                        workers=getattr(env._simulator, "worker_info", None),
                                    )
                                    for label, count in (("steps_s", transitions), ("frames_s", frames)):
                                        values = [count / t for t in seconds]
                                        row[label + "_mean"] = statistics.mean(values)
                                        row[label + "_sd"] = statistics.stdev(values)
                                    report["timings"].append(row)
                                    print(
                                        "TIMING",
                                        name,
                                        device,
                                        batch,
                                        horizon,
                                        layout,
                                        mode,
                                        round(row["steps_s_mean"], 1),
                                        round(row["frames_s_mean"], 1),
                                        flush=True,
                                    )
                                    save()
                        gc.collect()
                expected.clear()
    report["status"] = "PASS"
    save()
    print("FETCH_FAMILY_BENCH_PASS", args.task, flush=True)


if __name__ == "__main__":
    main()
