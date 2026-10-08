"""Exact classic-control public-API/export comparison against frozen 6ce4e829 rows.

CPU physics is used for both output devices. Timings exclude rendering; frame
gates compare export paths, not fresh upstream image-fidelity certification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
import time
import types
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from benchmarks.online.classic_profile import sync
from stable_envs.envs._classic_semantics import _PHYSICS_SPACES
from tests.helpers.reference.classic_state_export import classic_state

NAMES = {"cartpole": "CartPole", "pendulum": "Pendulum", "acrobot": "Acrobot",
         "mountain_car": "MountainCar", "mountain_car_continuous": "MountainCarContinuous"}
DISCRETE = {"cartpole": 2, "acrobot": 3, "mountain_car": 3}


def actions_for(domain, batch, seed):
    rng = np.random.default_rng(seed)
    if domain in DISCRETE:
        return rng.integers(0, DISCRETE[domain], (batch, 8, 1))
    return rng.uniform(-0.5, 0.5, (batch, 8, 1)).astype(np.float32)

def equal(a, b):
    assert a.batch_size == b.batch_size
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    for key in a.keys(True, True):
        x, y = a[key], b[key]
        assert x.shape == y.shape and x.dtype == y.dtype and x.device == y.device, key
        torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True, msg=str(key))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=NAMES, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 64, 256])
    p.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--no-frames", action="store_true", help="CPU-only local smoke; GPU acceptance must include frames")
    p.add_argument("--reference-classic", type=Path, help="Frozen accepted column-export module for a newer candidate")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 2 or min(args.batches) < 1:
        p.error("Require positive batches and at least two repeats")
    reference_class = None
    if args.reference_classic is not None:
        spec = importlib.util.spec_from_file_location("stable_envs.envs._classic_export_reference", args.reference_classic)
        reference_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(reference_module)
        reference_class = reference_module.ClassicControlAdapter
    reference_files = [args.reference_classic] if reference_class is not None else [Path("tests/helpers/reference/classic_state_export.py")]
    torch.set_num_threads(1)
    report = dict(status="RUNNING", domain=args.domain, env=NAMES[args.domain],
                  host=platform.node(), job=os.environ.get("SLURM_JOB_ID"),
                  gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
                  versions={name: importlib.metadata.version(name) for name in ("torch", "numpy", "gymnasium", "tensordict")},
                  reference_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in reference_files},
                  candidate_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                      Path("stable_envs/envs/_classic_semantics.py"), Path("stable_envs/envs/classic_control.py"),
                      Path("stable_envs/_adapters/state_columns.py"), Path("stable_envs/_stepping.py")]},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="frozen accepted exporter vs column exporter; public step frames=None; same CPU physics for both output devices",
                  gates=[], timings=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def make(batch, device, legacy=False):
        env = se.make(NAMES[args.domain], num_envs=batch, device=device, seed=17, backend="auto")
        if legacy:
            adapter = env._simulator
            if reference_class is None:
                adapter.get_state = types.MethodType(classic_state, adapter)
            else:
                adapter.get_state = types.MethodType(reference_class.get_state, adapter)
                adapter._state_values = types.MethodType(reference_class._state_values, adapter)
        return env

    save()
    try:
        for device in args.devices:
            with make(4, device, True) as reference, make(4, device) as candidate:
                initial = reference.get_state()
                equal(initial, candidate.get_state())
                actions = actions_for(args.domain, 4, 12)
                initial["rng", "has_uint32"].fill_(1)
                initial["rng", "uinteger"].fill_(2**32 - 1)
                limit = reference._simulator.envs[0]._max_episode_steps
                initial["lifecycle", "elapsed_steps"][:, 0] = torch.tensor([0, limit - 1, limit, 4], device=device)
                if args.domain == "cartpole":
                    initial["x"][3, 0] = 2.5
                elif args.domain == "acrobot":
                    initial["theta1"][3, 0] = np.pi
                elif args.domain in ("mountain_car", "mountain_car_continuous"):
                    initial["position"][3, 0] = 0.55
                cases = 0
                for horizon in (1, 8):
                    for selection in (None, slice(None), slice(-1, None), slice(None, None, -2), slice(9, 12)):
                        for env in (reference, candidate):
                            env.set_state(initial)
                        a = actions[:, 0] if horizon == 1 else actions
                        frames = None if args.no_frames else selection
                        expected = reference.step(a, states=selection, frames=frames)
                        actual = candidate.step(a, states=selection, frames=frames)
                        equal(expected, actual)
                        held = actual.clone()
                        candidate.step(actions[:, 0], frames=None)
                        candidate.reset(seed=31)
                        equal(held, actual)
                        candidate.set_state(expected["state"])
                        equal(expected["state"], candidate.get_state())
                        cases += 1
                for env in (reference, candidate):
                    env.set_state(initial)
                equal(reference.reset(), candidate.reset())
                key, bounds = next(iter(_PHYSICS_SPACES[args.domain].items()))
                for env in (reference, candidate):
                    env.set_variation({"physics." + key: [bounds[2] * 1.1]})
                    env.reset(seed=42)
                equal(reference.get_state(), candidate.get_state())
                equal(reference.step(actions, frames=None if args.no_frames else slice(None)),
                      candidate.step(actions, frames=None if args.no_frames else slice(None)))
                if args.domain in DISCRETE:
                    for env in (reference, candidate):
                        env.reset(seed=17)
                    equal(reference.step(actions[:, 0, 0], frames=None),
                          candidate.step(actions[:, 0, 0], frames=None))
                report["gates"].append(dict(device=device, cases=cases, state_task_rng_lifecycle_ownership="PASS",
                                             frame_mse_vs_legacy=None if args.no_frames else 0.0))
                save()
            for batch in args.batches:
                with make(batch, device, True) as reference, make(batch, device) as candidate:
                    initial = reference.get_state()
                    actions = actions_for(args.domain, batch, 23)
                    for horizon in (1, 8):
                        a = actions[:, 0] if horizon == 1 else actions
                        for operation in ("get_state", "public_step_states_all"):
                            samples = {"legacy": [], "column": []}
                            for block in range(2):
                                order = [("legacy", reference), ("column", candidate)]
                                if block:
                                    order.reverse()
                                for mode, env in order:
                                    env.set_state(initial)
                                    fn = env.get_state if operation == "get_state" else lambda: env.step(a, frames=None)
                                    fn()
                                    for _ in range(args.repeats):
                                        env.set_state(initial)
                                        sync(device)
                                        start = time.perf_counter()
                                        result = fn()
                                        sync(device)
                                        samples[mode].append(time.perf_counter() - start)
                                        if operation == "public_step_states_all":
                                            assert bool(result["valid"].all()), "Padded slots cannot count as steps"
                                        del result
                            for mode, seconds in samples.items():
                                count = batch * (horizon if operation != "get_state" else 1)
                                rates = [count / t for t in seconds]
                                report["timings"].append(dict(batch=batch, device=device, horizon=horizon,
                                    operation=operation, mode=mode, seconds=seconds,
                                    rate_mean=statistics.mean(rates), rate_sd=statistics.stdev(rates)))
                            save()
                print(args.domain, device, batch, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = repr(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
