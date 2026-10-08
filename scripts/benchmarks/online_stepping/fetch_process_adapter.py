"""Full-adapter process/hybrid comparison; never substitutes partial State.

Each sample restores an identical checkpoint outside timing. Timed regions
include physics, observation/reward/lifecycle handling, every State field,
IPC and parent-owned outputs. Public API samples additionally retain all
intermediate States. Rendering, startup and restoration are excluded.
"""

from __future__ import annotations

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

import stable_envs as snv
import stable_envs.envs._fetch_semantics as semantics
from stable_envs._adapters.fetch.adapter import FetchAdapter
from stable_envs._adapters.fetch.process import FetchProcessAdapter

LAYOUTS = {"threads_1x4": (1, 4), "pure_8x1": (8, 1), "hybrid_4x2": (4, 2)}


def equal(a, b):
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    for key in a.keys(True, True):
        x, y = a[key].cpu(), b[key].cpu()
        assert x.dtype == y.dtype and torch.equal(x, y), key


def make(name, batch, device, layout, cpus):
    processes, threads = LAYOUTS[layout]
    constructor = (
        FetchAdapter
        if processes == 1
        else partial(FetchProcessAdapter, processes=processes, threads_per_process=threads, cpus=cpus)
    )
    with patch.object(semantics, "FetchAdapter", constructor):
        env = snv.make(name, num_envs=batch, seed=17, device=device, backend="opengl", size=64)
    if processes == 1:
        env._simulator._set_workers(min(batch, threads))
    return env


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    parser.add_argument("--batches", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    parser.add_argument("--repeats", type=int, default=3)
    parser.add_argument("--blocks", type=int, default=2)
    parser.add_argument("--horizon", type=int, default=8)
    parser.add_argument("--gate-only", action="store_true")
    parser.add_argument("--gate-batch", type=int, default=64)
    args = parser.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert args.repeats >= 2 and 1 <= args.horizon <= 49
    torch.set_num_threads(1)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if cpus is not None:
        assert len(cpus) == 8, cpus
    files = [Path(__file__), Path(semantics.__file__)]
    import stable_envs._adapters.fetch.adapter as local_module
    import stable_envs._adapters.fetch.process as process_module

    files += [Path(process_module.__file__), Path(local_module.__file__)]
    report = dict(
        status="RUNNING",
        device=args.device,
        host=platform.node(),
        affinity=cpus,
        job=os.environ.get("SLURM_JOB_ID"),
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        versions={
            p: importlib.metadata.version(p) for p in ("torch", "numpy", "mujoco", "gymnasium", "gymnasium-robotics")
        },
        source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in files},
        scope="FetchPush; full State/task outputs, transport and owned copies; no rendering, startup or restoration. CPU simulation even with CUDA outputs. Known actions, not policy throughput.",
        gates=[],
        timings=[],
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    if args.gate_only:
        batch = args.gate_batch
        for task in ("Reach", "Push", "Slide", "PickAndPlace"):
            for suffix in ("", "Dense", "Dict"):
                name = "Fetch" + task + suffix
                actions = np.random.default_rng(31).uniform(-1, 1, (batch, 6, 4)).astype(np.float32)
                for layout in ("pure_8x1", "hybrid_4x2"):
                    with make(name, batch, args.device, "threads_1x4", cpus) as reference:
                        initial = reference.get_state()
                        expected = reference.step(actions, frames=None).cpu()
                        with make(name, batch, args.device, layout, cpus) as env:
                            equal(initial, env.get_state())
                            actual = env.step(actions, frames=None)
                            equal(expected, actual)
                            frozen = actual.clone()
                            env.reset(seed=42)
                            equal(actual, frozen)
                            for owner in (reference, env):
                                owner.set_state(initial)
                            equal(reference.reset(), env.reset())
                            for owner in (reference, env):
                                owner.set_state(initial)
                                owner._simulator._set_episode_limits(np.arange(batch) % 3 + 1)
                            equal(reference.step(actions, frames=None), env.step(actions, frames=None))
                            for owner in (reference, env):
                                owner._simulator._set_episode_limits(np.full(batch, 50))
                            variation = {"table.color": np.array([0.2, 0.7, 0.4])}
                            if task != "Reach":
                                variation.update(
                                    {"block.mass": np.array([0.9]), "object.color": np.array([0.8, 0.1, 0.3])}
                                )
                            equal(
                                reference.reset(seed=42, options={"variation_values": variation}),
                                env.reset(seed=42, options={"variation_values": variation}),
                            )
                            equal(
                                reference.step(actions, states=None, frames=None),
                                env.step(actions, states=None, frames=None),
                            )
                            # Identical State/style must render identically, independently per registration.
                            frame_mse = None
                            if args.device == "cuda":
                                a, b = reference.render(), env.render()
                                assert torch.equal(a, b)
                                frame_mse = 0.0
                            row = dict(
                                env=name,
                                layout=layout,
                                batch=batch,
                                state_task_replay_endpoint_variation="PASS",
                                frame_mse=frame_mse,
                            )
                            report["gates"].append(row)
                            print("GATE_PASS", json.dumps(row), flush=True)
                            save()
        report["status"] = "PASS"
        save()
        print("FULL_FETCH_PROCESS_GATE_PASS", flush=True)
        return

    for batch in args.batches:
        actions = np.random.default_rng(51).uniform(-1, 1, (batch, args.horizon, 4)).astype(np.float32)
        # One reference is closed before any timed owner is constructed.
        with make("FetchPush", batch, args.device, "threads_1x4", cpus) as reference:
            initial = reference.get_state().cpu()
            expected_public = reference.step(actions, states=slice(None), frames=None).cpu()
            reference.set_state(initial)
            expected_adapter = [reference._simulator.step(actions[:, t]).cpu() for t in range(args.horizon)]
        for block in range(args.blocks):
            layouts = list(LAYOUTS)
            if block % 2:
                layouts.reverse()
            for layout in layouts:
                started = time.perf_counter()
                with make("FetchPush", batch, args.device, layout, cpus) as env:
                    startup = time.perf_counter() - started
                    checkpoint = initial.to(args.device)
                    for stage in ("adapter_full_state", "public_all_states"):

                        def run():
                            if stage == "public_all_states":
                                return env.step(actions, states=slice(None), frames=None)
                            return [env._simulator.step(actions[:, t]) for t in range(args.horizon)]

                        def check(result):
                            if stage == "public_all_states":
                                equal(expected_public, result)
                            else:
                                for a, b in zip(expected_adapter, result):
                                    equal(a, b)

                        env.set_state(checkpoint)
                        check(run())
                        sync(args.device)
                        seconds = []
                        for _ in range(args.repeats):
                            env.set_state(checkpoint)
                            sync(args.device)
                            start = time.perf_counter()
                            result = run()
                            sync(args.device)
                            seconds.append(time.perf_counter() - start)
                            check(result)  # Every sample checked, outside timing.
                            del result
                        rates = [batch * args.horizon / t for t in seconds]
                        row = dict(
                            batch=batch,
                            horizon=args.horizon,
                            block=block,
                            layout=layout,
                            stage=stage,
                            seconds=seconds,
                            steps_s_mean=statistics.mean(rates),
                            steps_s_sd=statistics.stdev(rates),
                            startup_seconds=startup,
                            exact_parity="PASS",
                            workers=getattr(env._simulator, "worker_info", None),
                        )
                        report["timings"].append(row)
                        print("TIMING", json.dumps(row), flush=True)
                        save()
    report["status"] = "PASS"
    save()
    print("FULL_FETCH_PROCESS_BENCH_PASS", flush=True)


if __name__ == "__main__":
    main()
