"""Paired public Navigation calls against the frozen c39880f9 step dispatcher.

Restore/warmup is outside timing. Exact State/task/frame gates include early
termination. Rates count valid transitions; final-frame FPS counts one per lane.
"""

import argparse
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import platform
import statistics
import time
from pathlib import Path

import torch

import stable_envs as se
from stable_envs._stepping import step


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def equal(actual, expected):
    assert actual.batch_size == expected.batch_size
    assert set(actual.keys(True, True)) == set(expected.keys(True, True))
    for key in actual.keys(True, True):
        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)


def gate(name, device, legacy):
    cases = 0
    with se.make(name, num_envs=3, device=device, seed=17) as env:
        initial = env.reset(seed=17)
        for terminal in (False, True):
            start = initial.clone()
            if terminal:
                target = "goal" if name == "SimplePointMaze" else "target"
                start["agent", "position"][0].copy_(start[target, "position"][0])
            for horizon in (1, 8):
                actions = torch.zeros((3, 2) if horizon == 1 else (3, horizon, 2), device=device)
                for states, frames in ((None, None), (slice(None), slice(-1, None)),
                                       (slice(None, None, -1), slice(None)), (slice(9, 9), slice(9, 9))):
                    env.set_state(start)
                    expected = legacy(env, actions, states=states, frames=frames)
                    env.set_state(start)
                    actual = step(env, actions, states=states, frames=frames)
                    equal(actual, expected)
                    if terminal and horizon == 8:
                        assert actual["valid"][0].sum().item() == 1
                    held = actual.clone()
                    env.reset(seed=31)
                    equal(actual, held)
                    cases += 1
    return dict(device=device, cases=cases, exact_state_task_frames="PASS", frame_mse_vs_legacy=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--env", choices=("TwoRoom", "Piecewise", "SimplePointMaze"), required=True)
    p.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 16, 64])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 2 or min(args.batches) < 1:
        p.error("Positive batches and at least two repetitions required")
    torch.set_num_threads(1)
    frozen = Path("tests/helpers/reference/step_capture.py")
    spec = importlib.util.spec_from_file_location("stable_envs._reference_stepping", frozen)
    reference = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(reference)
    modes = {"legacy": reference.step, "candidate": step}
    paths = [Path("stable_envs/_stepping.py"), Path("stable_envs/env.py"),
             Path("stable_envs/envs/navigation.py"), frozen, Path(__file__).relative_to(Path.cwd())]
    report = dict(status="RUNNING", env=args.env, host=platform.node(), job=os.environ.get("SLURM_JOB_ID"),
                  gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
                  source_sha256={str(path): hashlib.sha256(path.read_bytes()).hexdigest() for path in paths},
                  versions={name: importlib.metadata.version(name) for name in ("torch", "numpy", "tensordict")},
                  scope="public native stepping: frozen dispatcher vs activity-aware padding; no upstream comparison",
                  gates=[], timings=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for device in args.devices:
            report["gates"].append(gate(args.env, device, reference.step))
            save()
            for batch in args.batches:
                with se.make(args.env, num_envs=batch, device=device, seed=17) as env:
                    initial = env.reset(seed=17)
                    for horizon in (1, 8):
                        actions = torch.full((batch, 2) if horizon == 1 else (batch, horizon, 2), 0.01, device=device)
                        for operation in ("states_all", "states_none", "pipeline_final_frame"):
                            samples, counts = {m: [] for m in modes}, {m: [] for m in modes}
                            kwargs = dict(states=None if operation == "states_none" else slice(None),
                                          frames=slice(-1, None) if operation == "pipeline_final_frame" else None)
                            for block in range(2):
                                for mode in modes if block == 0 else reversed(modes):
                                    env.set_state(initial)
                                    modes[mode](env, actions, **kwargs)
                                    for _ in range(args.repeats):
                                        env.set_state(initial)
                                        sync(device)
                                        start = time.perf_counter()
                                        result = modes[mode](env, actions, **kwargs)
                                        sync(device)
                                        samples[mode].append(time.perf_counter() - start)
                                        counts[mode].append(int(result["valid"].sum().item()))
                                        del result
                            assert counts["legacy"] == counts["candidate"]
                            for mode in modes:
                                rates = [n / t for n, t in zip(counts[mode], samples[mode], strict=True)]
                                row = dict(device=device, batch=batch, horizon=horizon, operation=operation, mode=mode,
                                           seconds=samples[mode], valid_transitions=counts[mode],
                                           steps_per_s_mean=statistics.mean(rates), steps_per_s_sd=statistics.stdev(rates))
                                if operation == "pipeline_final_frame":
                                    fps = [batch / t for t in samples[mode]]
                                    row.update(frames_per_s_mean=statistics.mean(fps), frames_per_s_sd=statistics.stdev(fps))
                                report["timings"].append(row)
                            save()
                    print(args.env, device, batch, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"], report["error"] = "FAIL", repr(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
