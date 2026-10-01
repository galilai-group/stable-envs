"""Bounded Fetch API/transport/executor proof against an immutable old adapter.

Run on Oscar for CUDA. No reset, restoration or warmup is in timed regions.
Source steps and candidates use identical seeds/actions and CPU allocations.
This is an action-chunk benchmark, not a visual-policy benchmark.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import stable_envs as snv


def equal(a, b):
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    for key in a.keys(True, True):
        x, y = a[key].cpu(), b[key].cpu()
        assert x.dtype == y.dtype and torch.equal(x, y), key


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def measure(fn, setup, count, repeats, device):
    setup()
    fn()
    sync(device)
    seconds = []
    for _ in range(repeats):
        setup()
        sync(device)
        start = time.perf_counter()
        fn()
        sync(device)
        seconds.append(time.perf_counter() - start)
    rates = [count / t for t in seconds]
    return dict(
        seconds=seconds,
        rate_mean=statistics.mean(rates),
        rate_sd=statistics.stdev(rates),
        count=count,
        repeats=repeats,
    )


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline-fetch", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--device", choices=["cpu", "cuda"], default="cpu")
    p.add_argument("--batches", type=int, nargs="+", default=[64, 256])
    p.add_argument("--workers", type=int, nargs="+", default=[1, 2, 4, 8])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--gate-only", action="store_true")
    p.add_argument("--gate-batch", type=int, default=3)
    args = p.parse_args()
    assert args.repeats >= 2
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    module = importlib.util.spec_from_file_location("baseline_fetch", args.baseline_fetch)
    baseline = importlib.util.module_from_spec(module)
    module.loader.exec_module(baseline)
    report = dict(
        status="RUNNING",
        device=args.device,
        host=platform.node(),
        cpu_affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        baseline_sha256=hashlib.sha256(args.baseline_fetch.read_bytes()).hexdigest(),
        gpu=torch.cuda.get_device_name() if args.device == "cuda" else None,
        torch=torch.__version__,
        numpy=np.__version__,
        gates=[],
        timings=[],
        scope="Known-action chunks; not closed-loop policy throughput. Frames use existing OpenGL backend.",
    )

    def save():
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    gate_batch = args.gate_batch
    # Independent proof for every Fetch task/reward registration.
    for task in ("Reach", "Push", "Slide", "PickAndPlace"):
        for suffix in ("", "Dense", "Dict"):
            name = "Fetch" + task + suffix
            old = baseline.FetchAdapter(gate_batch, "cpu", 17, task, "dense" if suffix == "Dense" else "sparse")
            with snv.make(name, num_envs=gate_batch, device=args.device, backend="opengl") as env:
                selected_workers = env._simulator._workers
                initial = env.reset(seed=17)
                old.reset(seed=17)
                equal(initial, old.get_state())
                actions = np.random.default_rng(19).uniform(-1, 1, (gate_batch, 6, 4)).astype(np.float64)
                expected = [old.step(actions[:, t]) for t in range(6)]
                got = env.step(actions, capture="states")
                for t, reference in enumerate(expected):
                    equal(got["states"][:, t], reference["state"])
                    for key in ("reward", "terminated", "truncated", "success"):
                        assert torch.equal(got[key][:, t].cpu(), reference[key])
                frozen = got.clone()
                env.set_state(initial)
                env._simulator._set_workers(min(3, gate_batch))
                threaded = env.step(actions, capture="states")
                equal(got, threaded)
                equal(frozen, got)
                # Different per-lane endpoints must skip, not step-and-restore.
                env.set_state(initial)
                for i, upstream in enumerate(env._simulator.envs):
                    upstream._max_episode_steps = i % 3 + 1
                ended = env.step(actions, capture="states")
                assert ended["valid"].sum(1).tolist() == [i % 3 + 1 for i in range(gate_batch)]
                assert ended["state"]["lifecycle", "elapsed_steps"].flatten().tolist() == [
                    i % 3 + 1 for i in range(gate_batch)
                ]
            old.close()
            report["gates"].append(
                dict(
                    env=name, batch=gate_batch, default_workers=selected_workers, state_task_thread_and_endpoint="PASS"
                )
            )
            print("GATE_PASS", name, flush=True)
            save()

    if args.device == "cuda":
        with snv.FetchPush(num_envs=3, device="cuda", backend="opengl", size=64) as env:
            initial = env.reset(seed=17)
            actions = np.random.default_rng(31).uniform(-1, 1, (3, 4, 4)).astype(np.float32)
            steps = [env.step(actions[:, t]) for t in range(4)]
            reference_frames = torch.stack([s["frame"] for s in steps], dim=1)
            env.set_state(initial)
            out = env.step(actions, capture="all_frames")
            assert torch.equal(out["frames"], reference_frames)
            env.set_state(initial)
            original_export = env._simulator.get_state
            exports = []

            def counted():
                exports.append(1)
                return original_export()

            env._simulator.get_state = counted
            final = env.step(actions, capture="final_frame")
            assert len(exports) == 1, len(exports)
            equal(final["state"], out["state"])
            assert torch.equal(final["frame"], reference_frames[:, -1])
            report["gates"].append(
                dict(env="FetchPush", frame_mse_vs_repeated_step=0.0, final_frame_snapshot_exports=len(exports))
            )
        save()

    if args.gate_only:
        report["status"] = "PASS"
        save()
        print("ONLINE_FETCH_GATE_PASS", flush=True)
        return

    for batch in args.batches:
        old = baseline.FetchAdapter(batch, args.device, 17, "Push")
        with snv.FetchPush(num_envs=batch, device=args.device, backend="opengl", size=64) as env:
            initial = env.reset(seed=17)
            actions = np.random.default_rng(51).uniform(-1, 1, (batch, args.horizon, 4)).astype(np.float32)
            setup = lambda: env.set_state(initial)
            count = batch * args.horizon
            row = dict(
                batch=batch,
                workers=1,
                stage="baseline_snapshot",
                unit="snapshots/s",
                **measure(old.get_state, lambda: None, batch, args.repeats, args.device),
            )
            report["timings"].append(row)
            report["timings"].append(
                dict(
                    batch=batch,
                    workers=1,
                    stage="bulk_snapshot",
                    unit="snapshots/s",
                    **measure(env.get_state, lambda: None, batch, args.repeats, args.device),
                )
            )
            report["timings"].append(
                dict(
                    batch=batch,
                    workers=1,
                    stage="baseline_steps",
                    unit="transitions/s",
                    **measure(
                        lambda: [old.step(actions[:, t]) for t in range(args.horizon)],
                        lambda: old.set_state(initial.cpu()),
                        count,
                        args.repeats,
                        args.device,
                    ),
                )
            )
            for workers in args.workers:
                if workers > batch:
                    continue
                env._simulator._set_workers(workers)

                def source_steps():
                    adapter = env._simulator
                    for t in range(args.horizon):
                        if adapter._executor is None:
                            adapter._advance_shard(range(batch), actions[:, t])
                        else:
                            futures = [
                                adapter._executor.submit(adapter._advance_shard, shard, actions[:, t])
                                for shard in np.array_split(np.arange(batch), workers)
                            ]
                            for f in futures:
                                f.result()

                for stage, fn in [
                    ("source_task_steps", source_steps),
                    ("steps_with_states", lambda: env.step(actions, capture="states")),
                ]:
                    report["timings"].append(
                        dict(
                            batch=batch,
                            workers=workers,
                            stage=stage,
                            unit="transitions/s",
                            **measure(fn, setup, count, args.repeats, args.device),
                        )
                    )
                if args.device == "cuda":
                    for stage, mode, frames in [
                        ("steps_final_frame", "final_frame", batch),
                        ("steps_all_frames", "all_frames", count),
                    ]:
                        timing = measure(
                            lambda: env.step(actions, capture=mode), setup, count, args.repeats, args.device
                        )
                        timing["frames_per_second"] = [frames / t for t in timing["seconds"]]
                        report["timings"].append(
                            dict(batch=batch, workers=workers, stage=stage, unit="transitions/s", **timing)
                        )
                print("TIMED", batch, workers, flush=True)
                save()
            if args.device == "cuda":
                setup()
                prepared = env.get_state()
                report["timings"].append(
                    dict(
                        batch=batch,
                        workers=1,
                        stage="render_only",
                        unit="frames/s",
                        **measure(lambda: env.render(prepared), lambda: None, batch, args.repeats, args.device),
                    )
                )
        old.close()
        save()
    report["status"] = "PASS"
    save()
    print("ONLINE_FETCH_PROBE_PASS", flush=True)


if __name__ == "__main__":
    main()
