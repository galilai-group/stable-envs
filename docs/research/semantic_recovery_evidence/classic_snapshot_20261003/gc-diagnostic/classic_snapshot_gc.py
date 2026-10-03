"""Pair snapshot-aware classic stepping against frozen pre-change adapters."""

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

from stable_envs.envs._classic_semantics import _PHYSICS_SPACES
from tests.helpers.reference.classic_snapshot import NAMES, make
from tests.unit.test_classic_state_export import equal


def actions_for(domain, batch):
    rng = np.random.default_rng(31)
    if domain in ("cartpole", "acrobot", "mountain_car"):
        a = rng.integers(0, 2 if domain == "cartpole" else 3, (batch, 8, 1))
        if domain == "cartpole":
            a[:] = np.arange(8)[None, :, None] % 2
        return a
    return rng.uniform(-.5, .5, (batch, 8, 1)).astype(np.float32)


def gate(domain, device):
    batch = 64 if domain == "acrobot" and hasattr(os, "sched_getaffinity") else 4
    with make(domain, batch, device, legacy=True) as old, make(domain, batch, device) as new:
        initial = old.get_state()
        equal(initial, new.get_state())
        with make(domain, 1, "cpu", legacy=True) as prototype:
            limit = prototype._simulator.envs[0]._max_episode_steps
        initial["lifecycle", "elapsed_steps"][:4, 0] = torch.tensor([0, limit - 1, limit, 0], device=device)
        initial["rng", "has_uint32"][-1] = 1
        initial["rng", "uinteger"][-1] = 2**32 - 1
        if domain == "cartpole":
            initial["x"][3] = 2.5
        elif domain == "acrobot":
            initial["theta1"][3] = np.pi
        elif domain in ("mountain_car", "mountain_car_continuous"):
            initial["position"][3] = .55
        actions = actions_for(domain, batch)
        cases = 0
        for horizon in (1, 8):
            a = actions[:, 0] if horizon == 1 else actions
            for states, frames in ((None, None), (None, slice(-1, None)), (slice(None), None),
                                   (slice(None), slice(None)), (slice(None, None, -2), slice(None, None, 3)),
                                   (slice(10, 12), slice(10, 12)), (slice(-1, None), slice(None))):
                for env in (old, new):
                    env.set_state(initial)
                expected = old.step(a, states=states, frames=frames)
                actual = new.step(a, states=states, frames=frames)
                equal(expected, actual)
                assert actual["valid"][1:3].sum(1).tolist() == [1, 1]
                held = actual.clone()
                torch.testing.assert_close(old.render(), new.render(), rtol=0, atol=0)
                new.reset(42)
                equal(actual, held)
                cases += 1
        for env in (old, new):
            env.set_state(initial)
        equal(old.reset(), new.reset())
        key, bounds = next(iter(_PHYSICS_SPACES[domain].items()))
        for env in (old, new):
            env.set_variation({"physics." + key: [bounds[2] * 1.1]})
            env.reset(31)
        device_actions = torch.as_tensor(actions, device=device)
        equal(old.step(device_actions), new.step(device_actions))
        torch.testing.assert_close(old.render(), new.render(), rtol=0, atol=0)
        if domain in ("cartpole", "acrobot", "mountain_car"):
            equal(old.step(device_actions[:, 0, 0], states=None, frames=None),
                  new.step(device_actions[:, 0, 0], states=None, frames=None))
        workers = len(new._simulator.worker_info) if hasattr(new._simulator, "worker_info") else 0
        old_workers = len(old._simulator.worker_info) if hasattr(old._simulator, "worker_info") else 0
        assert workers == old_workers
        return dict(device=device, batch=batch, cases=cases, workers=workers,
                    complete_state_task_rng_lifecycle_ownership="PASS", frame_mse_vs_baseline=0,
                    reset_variation_device_actions="PASS", current_render="PASS")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=NAMES, required=True)
    p.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 64, 256, 1024])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.batches) < 1 or args.repeats < 2:
        p.error("positive batches and >=2 repeats required")
    torch.set_num_threads(1)
    files = ["stable_envs/envs/_classic_semantics.py", "stable_envs/_adapters/acrobot_process.py",
             "stable_envs/_adapters/classic_execution.py", "stable_envs/_adapters/process_requests.py",
             "stable_envs/envs/classic_control.py", "stable_envs/_stepping.py",
             "tests/helpers/reference/classic_before_snapshot.py", "tests/helpers/reference/acrobot_before_snapshot.py",
             "tests/helpers/reference/classic_snapshot.py", "tests/unit/test_classic_snapshot.py"]
    files.append(str(Path(__file__).relative_to(Path.cwd())))
    report = dict(status="RUNNING", domain=args.domain, name=NAMES[args.domain],
                  host=platform.node(), job=os.environ.get("SLURM_JOB_ID"),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "gymnasium", "tensordict")},
                  source_sha256={f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in files},
                  scope="snapshot-aware candidate vs frozen library adapters; same workers and renderer; no upstream speedup",
                  gates=[], timings=[])
    gc_events, gc_started = [], {}
    report["gc_events"] = gc_events
    report["scope"] += "; GC-instrumented diagnostic, not acceptance throughput"

    def observe_gc(phase, info):
        generation = info["generation"]
        if phase == "start":
            gc_started[generation] = time.perf_counter()
        else:
            gc_events.append(dict(start=gc_started.pop(generation), end=time.perf_counter(), **info))

    gc.callbacks.append(observe_gc)
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def sync(device):
        if device == "cuda":
            torch.cuda.synchronize()

    save()
    try:
        for device in args.devices:
            report["gates"].append(gate(args.domain, device))
            save()
            for batch in args.batches:
                actions = actions_for(args.domain, batch)
                with make(args.domain, batch, device, legacy=True) as old, make(args.domain, batch, device) as new:
                    initial = old.get_state()
                    for horizon in (1, 8):
                        a = actions[:, 0] if horizon == 1 else actions
                        for state_name, states in (("none", None), ("all", slice(None))):
                            for frame_name, frames in (("none", None), ("final", slice(-1, None))):
                                for env in (old, new):
                                    env.set_state(initial)
                                expected = old.step(a, states=states, frames=frames)
                                actual = new.step(a, states=states, frames=frames)
                                equal(expected, actual)
                                assert bool(actual["valid"].all())
                                del expected, actual
                                samples = {"baseline": [], "candidate": []}
                                windows = {"baseline": [], "candidate": []}
                                for block in range(2):
                                    order = [("baseline", old), ("candidate", new)]
                                    for mode, env in order if block == 0 else reversed(order):
                                        env.set_state(initial)
                                        env.step(a, states=states, frames=frames)
                                        for _ in range(args.repeats):
                                            env.set_state(initial)
                                            sync(device)
                                            start = time.perf_counter()
                                            result = env.step(a, states=states, frames=frames)
                                            sync(device)
                                            end = time.perf_counter()
                                            samples[mode].append(end - start)
                                            windows[mode].append([start, end])
                                            assert bool(result["valid"].all())
                                            del result
                                for mode, seconds in samples.items():
                                    rates = [batch * horizon / s for s in seconds]
                                    report["timings"].append(dict(device=device, batch=batch, horizon=horizon,
                                        states=state_name, frames=frame_name, mode=mode, seconds=seconds, timing_windows=windows[mode],
                                        valid_steps=batch * horizon, frame_count=batch if frames is not None else 0,
                                        steps_per_s_mean=statistics.mean(rates), steps_per_s_sd=statistics.stdev(rates),
                                        exact_state_task_frame="PASS"))
                                save()
                print(args.domain, device, batch, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        gc.callbacks.remove(observe_gc)
        save()


if __name__ == "__main__":
    main()
