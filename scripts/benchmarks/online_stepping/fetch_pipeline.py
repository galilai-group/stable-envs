"""Compare overlapped Fetch stepping to identical, non-overlapped process owners.

The deferred control fuses State export exactly like the pipeline but launches
the next transition only after rendering. This isolates overlap from RPC savings.
"""

from __future__ import annotations

import argparse
import gc
import hashlib
import json
import os
import platform
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from fetch_family import make
from fetch_process_adapter import equal, sync

import stable_envs as snv
from stable_envs._adapters.fetch.process import FetchProcessAdapter


class Deferred:
    def __init__(self, owner):
        self.owner = owner
        self.pending = None

    def submit_step(self, actions, **kwargs):
        assert self.pending is None
        self.pending = actions, kwargs

    def receive_step(self):
        actions, kwargs = self.pending
        self.pending = None
        self.owner.submit_step(actions, **kwargs)
        return self.owner.receive_step()

    def abort_step(self):
        self.owner.abort_step()


def executor(env, mode):
    if mode == "serial_legacy":
        return lambda: None
    if mode == "serial_fused":
        return lambda: Deferred(env._simulator)
    return lambda: env._simulator


def gate(name, device, cpus, size, production=False):
    batch = 64 if cpus else 8
    with make(name, batch, device, "current", cpus, size) as reference:
        candidate = (
            snv.make(name, num_envs=batch, seed=17, device=device, backend="opengl", size=size)
            if production
            else make(name, batch, device, "process8", cpus, size)
        )
        with candidate:
            assert isinstance(candidate._simulator, FetchProcessAdapter)
            actions = np.random.default_rng(21).uniform(-1, 1, (batch, 8, 4)).astype(np.float64)
            initial = reference.get_state()
            equal(initial, candidate.get_state())
            selections = [None, slice(None), slice(-1, None), slice(None, None, -2), slice(20, 30)]
            cases = 0
            for horizon in (1, 8):
                for selection in selections:
                    frames = selection if device == "cuda" else None
                    for env in (reference, candidate):
                        env.set_state(initial)
                    a = actions[:, 0] if horizon == 1 else actions
                    expected = reference.step(a, states=selection, frames=frames)
                    actual = candidate.step(a, states=selection, frames=frames)
                    equal(expected, actual)
                    held = actual.clone()
                    candidate.step(actions[:, 0], frames=None)
                    equal(held, actual)
                    assert candidate._simulator._pending is None
                    cases += 1
            for env in (reference, candidate):
                env.set_state(initial)
                env._simulator._set_episode_limits(np.arange(batch) % 3 + 1)
            # CUDA actions as well as NumPy; several lanes finish mid-chunk.
            a = torch.as_tensor(actions, device=device, dtype=torch.float32)
            frames = slice(None) if device == "cuda" else None
            expected = reference.step(a, frames=frames)
            actual = candidate.step(a, frames=frames)
            equal(expected, actual)
            assert actual["valid"].sum(1).tolist() == (np.arange(batch) % 3 + 1).tolist()
            for env in (reference, candidate):
                env.set_state(initial)
            equal(reference.reset(), candidate.reset())
            variation = {"table.color": np.array([0.2, 0.7, 0.4])}
            if "Reach" not in name:
                variation.update({"block.mass": np.array([0.9]), "object.color": np.array([0.8, 0.1, 0.3])})
            for env in (reference, candidate):
                env._simulator._set_episode_limits(np.full(batch, 50))
            equal(
                reference.reset(seed=42, options={"variation_values": variation}),
                candidate.reset(seed=42, options={"variation_values": variation}),
            )
            equal(reference.step(a, frames=frames), candidate.step(a, frames=frames))
            return dict(
                env=name,
                device=device,
                cases=cases + 3,
                state_task_replay_ownership="PASS",
                frame_mse_vs_serial=0.0 if device == "cuda" else None,
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--task", choices=["Reach", "Push", "Slide", "PickAndPlace"], required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    p.add_argument("--batches", nargs="+", type=int, default=[64, 256, 1024])
    p.add_argument("--horizons", nargs="+", type=int, default=[1, 8])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--size", type=int, default=64)
    p.add_argument("--production", action="store_true", help="Gate the unpatched public factory on Linux")
    p.add_argument("--gates-only", action="store_true")
    p.add_argument("--suffixes", nargs="+", choices=["plain", "Dense", "Dict"], default=["plain", "Dense", "Dict"])
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert args.repeats >= 2 and args.blocks >= 1
    torch.set_num_threads(1)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    if cpus is not None:
        assert len(cpus) == 8, cpus
    report = dict(
        status="RUNNING",
        task=args.task,
        host=platform.node(),
        affinity=cpus,
        job=os.environ.get("SLURM_JOB_ID"),
        partition=os.environ.get("SLURM_JOB_PARTITION"),
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        configuration=vars(args) | {"output": str(args.output)},
        gates=[],
        timings=[],
        source_sha256={
            str(p): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path("stable_envs/_stepping.py"), Path("stable_envs/_adapters/fetch/process.py")]
        },
        scope="CPU simulation, eight single-threaded process owners. All States retained. Same allocation and rendering fidelity for every schedule. CUDA synchronized timing; reset, warmup and exact equality outside timing. No policy inference. Render reference is serial executor, not a new upstream parity study.",
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for suffix in args.suffixes:
        name = "Fetch" + args.task + ("" if suffix == "plain" else suffix)
        if args.production and args.device != "cpu":
            report["gates"].append(gate(name, "cpu", cpus, args.size, production=True))
        report["gates"].append(gate(name, args.device, cpus, args.size, production=args.production))
        print("PIPELINE_GATE_PASS", name, flush=True)
        save()
        # Every registration is gated; throughput uses the task's plain variant.
        if suffix != "plain" or args.gates_only:
            continue
        for batch in args.batches:
            env = (
                snv.make(name, num_envs=batch, seed=17, device=args.device, backend="opengl", size=args.size)
                if args.production
                else make(name, batch, args.device, "process8", cpus, args.size)
            )
            with env:
                assert isinstance(env._simulator, FetchProcessAdapter)
                checkpoint = env.get_state()
                actions = np.random.default_rng(31).uniform(-1, 1, (batch, max(args.horizons), 4)).astype(np.float32)
                for horizon in args.horizons:
                    a = actions[:, 0] if horizon == 1 else actions[:, :horizon]
                    selections = {"none": None}
                    if args.device == "cuda":
                        selections.update(final=slice(-1, None), all=slice(None))
                    for selection, frames in selections.items():
                        env.set_state(checkpoint)
                        with patch.object(env, "_step_pipeline", lambda: None):
                            expected = env.step(a, frames=frames).cpu()
                        for block in range(args.blocks):
                            modes = ["serial_legacy", "serial_fused", "pipeline"]
                            if block % 2:
                                modes.reverse()
                            for mode in modes:
                                with patch.object(env, "_step_pipeline", executor(env, mode)):
                                    env.set_state(checkpoint)
                                    equal(expected, env.step(a, frames=frames))
                                    seconds = []
                                    for _ in range(args.repeats):
                                        env.set_state(checkpoint)
                                        sync(args.device)
                                        start = time.perf_counter()
                                        result = env.step(a, frames=frames)
                                        sync(args.device)
                                        seconds.append(time.perf_counter() - start)
                                        equal(expected, result)
                                        del result
                                count = batch * (horizon if selection == "all" else int(selection == "final"))
                                row = dict(
                                    env=name,
                                    batch=batch,
                                    horizon=horizon,
                                    frames=selection,
                                    mode=mode,
                                    block=block,
                                    seconds=seconds,
                                    steps=batch * horizon,
                                    images=count,
                                    steps_s_mean=statistics.mean(batch * horizon / s for s in seconds),
                                    frames_s_mean=statistics.mean(count / s for s in seconds),
                                    exact_parity="PASS",
                                    frame_mse=0.0 if count else None,
                                )
                                report["timings"].append(row)
                                print(
                                    "PIPELINE_TIMING",
                                    name,
                                    batch,
                                    horizon,
                                    selection,
                                    mode,
                                    round(row["steps_s_mean"], 1),
                                    round(row["frames_s_mean"], 1),
                                    flush=True,
                                )
                                save()
                        del expected
            del env, checkpoint
            gc.collect()
    report["status"] = "PASS"
    save()
    print("FETCH_PIPELINE_PASS", args.task, flush=True)


if __name__ == "__main__":
    main()
