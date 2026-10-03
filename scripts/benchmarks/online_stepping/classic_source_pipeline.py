"""Direct pinned-source classic stepping/rendering versus the public adapter.

The source baseline calls real scalar wrappers and never the oracle's step()
or get_state() inside timing. Both paths materialize task outputs and requested
images on the selected destination. Ours additionally returns its mandatory
complete final State. Source is serial CPU (pygame has no GPU renderer).
These rows do not establish superiority over a parallel source implementation.
"""

import argparse
import hashlib
import importlib
import importlib.metadata
import inspect
import json
import os
import platform
import statistics
import sys
import time
import types
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from scripts.benchmarks.online_stepping.classic_state_export import NAMES, actions_for
from scripts.benchmarks.renderer_table.common import PIN, equal_state
from tests.helpers.reference.native_classic_remaining_semantics import NativeClassicRemainingReference
from tests.helpers.reference.native_classic_semantics import NativeClassicReference


def import_source(root):
    # Load the unchanged environment modules without unrelated policy/data deps.
    for name, sub in (("stable_worldmodel", ""), ("stable_worldmodel.envs", "envs"),
                      ("stable_worldmodel.envs.gymnasium_control", "envs/gymnasium_control")):
        package = types.ModuleType(name)
        package.__path__ = [str(root / "stable_worldmodel" / sub)]
        sys.modules[name] = package
    importlib.import_module("stable_worldmodel.utils")


def source(domain, batch):
    cls = NativeClassicReference if domain in ("cartpole", "pendulum") else NativeClassicRemainingReference
    ref = cls(domain, num_envs=batch)
    for i, env in enumerate(ref.envs):
        ref.envs[i] = type(env)(render_mode="rgb_array")
        env.close()
    ref.reset(17)
    return ref


def source_step(ref, actions, frame_indices, device):
    """Scalar upstream calls; no oracle export, restore, or State conversion."""
    outputs, frames = [], []
    discrete = ref.domain in ("cartpole", "acrobot", "mountain_car")
    for t in range(actions.shape[1]):
        rows = [env.step(actions[i, t, 0].item() if discrete else actions[i, t])
                for i, env in enumerate(ref.envs)]
        outputs.append(rows)
        if t in frame_indices:
            frames.append(np.stack([env.render() for env in ref.envs]))
    tasks = {}
    for field, index, dtype in (("observation", 0, None), ("reward", 1, np.float64),
                                ("terminated", 2, bool), ("truncated", 3, bool)):
        values = np.stack([np.stack([r[index] for r in rows]) for rows in outputs], axis=1)
        tasks[field] = torch.as_tensor(values.astype(dtype) if dtype else values, device=device)
    tasks["info_state"] = torch.as_tensor(
        np.stack([np.stack([r[4]["state"] for r in rows]) for rows in outputs], axis=1), device=device)
    if frames:
        tasks["frames"] = torch.as_tensor(np.stack(frames, axis=1), device=device)
    return tasks


def image_errors(a, b):
    assert a.shape == b.shape and a.dtype == b.dtype == torch.uint8
    # Keep float64 scratch bounded to one frame, even at large B*T.
    a, b = a.flatten(0, 1), b.flatten(0, 1)
    errors = []
    for x, y in zip(a, b, strict=True):
        difference = (x.cpu().numpy().astype(np.float64) - y.cpu().numpy()) / 255
        errors.append(float(np.mean(difference * difference)))
    return errors


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=NAMES, required=True)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--devices", nargs="+", choices=("cpu", "cuda"), default=["cpu", "cuda"])
    p.add_argument("--batches", nargs="+", type=int, default=[1, 64, 256])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.batches) < 1 or args.repeats < 2:
        p.error("positive batches and >=2 repeats required")
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest["commit"] == PIN
    for rel, digest in manifest["files"].items():
        assert hashlib.sha256((args.source_root / rel).read_bytes()).hexdigest() == digest, rel
    import_source(args.source_root)
    torch.set_num_threads(1)
    report = dict(status="RUNNING", domain=args.domain, name=NAMES[args.domain],
                  host=platform.node(), job=os.environ.get("SLURM_JOB_ID"),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
                  source_pin=PIN, source_files=manifest["files"], source_gpu_supported=False,
                  source_render_device="cpu", source_execution="serial scalar CPU wrappers",
                  source_active_simulation_workers=1,
                  source_state_export_in_timing=False, candidate_state_history=False,
                  candidate_mandatory_final_state=True,
                  scope="direct serial upstream vs ordinary public adapter; matched output destination; no parallel-source superiority claim",
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "gymnasium", "pygame", "tensordict")},
                  gates=[], timings=[], components=[], source_runtime_files={})
    paths = [Path(__file__).relative_to(Path.cwd()), Path("stable_envs/envs/_classic_semantics.py"),
             Path("stable_envs/envs/classic_control.py"), Path("stable_envs/_stepping.py"),
             *Path("stable_envs/_adapters").glob("*classic*.py"),
             Path("stable_envs/_adapters/acrobot_process.py"), Path("stable_envs/_adapters/process_requests.py"),
             Path("scripts/benchmarks/online_stepping/classic_state_export.py"),
             Path("scripts/benchmarks/renderer_table/common.py"),
             Path("tests/helpers/reference/native_classic_semantics.py"),
             Path("tests/helpers/reference/native_classic_remaining_semantics.py")]
    report["candidate_files"] = {str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths}
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    def measure(fn, device):
        if device == "cuda":
            torch.cuda.synchronize()
        start = time.perf_counter()
        result = fn()
        if device == "cuda":
            torch.cuda.synchronize()
        return time.perf_counter() - start, result

    def stats(samples, count):
        rates = [count / s for s in samples]
        return dict(seconds=samples, count=count, rate_mean=statistics.mean(rates), rate_sd=statistics.stdev(rates))

    save()
    try:
        for device in args.devices:
            for batch in args.batches:
                ref = source(args.domain, batch)
                try:
                    for cls in (type(ref.envs[0]), type(ref.envs[0]).__bases__[0], type(ref.envs[0].unwrapped)):
                        f = Path(inspect.getsourcefile(cls))
                        report["source_runtime_files"][str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()
                    with se.make(NAMES[args.domain], num_envs=batch, seed=17, device=device) as env:
                        initial = env.get_state().cpu()
                        equal_state(initial, ref.get_state())
                        actions = actions_for(args.domain, batch, 31)
                        if args.domain == "cartpole":
                            actions[:] = np.arange(8)[None, :, None] % 2
                        for horizon in (1, 8):
                            a = actions[:, :horizon]
                            public_actions = a[:, 0] if horizon == 1 else a
                            for selection, indices, frame_slice in (("none", (), None),
                                    ("final", (horizon - 1,), slice(-1, None)),
                                    ("all", tuple(range(horizon)), slice(None))):
                                ref.set_state(initial)
                                env.set_state(initial.to(device))
                                expected = source_step(ref, a, indices, device)
                                actual = env.step(public_actions, states=None, frames=frame_slice)
                                assert bool(actual["valid"].all())
                                equal_state(actual["state"], ref.get_state())
                                for key in ("observation", "reward", "terminated", "truncated", "info_state"):
                                    value = actual[key].unsqueeze(1) if horizon == 1 else actual[key]
                                    torch.testing.assert_close(value, expected[key], rtol=0, atol=0)
                                assert not bool(expected["terminated"][:, :-1].any() or expected["truncated"][:, :-1].any())
                                errors = image_errors(actual["frames"], expected["frames"]) if indices else []
                                report["gates"].append(dict(device=device, batch=batch, horizon=horizon, selection=selection,
                                    state_task="PASS", frames_checked=len(errors), mse_mean=statistics.mean(errors) if errors else None,
                                    mse_sd=statistics.stdev(errors) if len(errors) > 1 else (0 if errors else None),
                                    mse_max=max(errors) if errors else None,
                                    mse_le_0_001=all(x <= .001 for x in errors) if errors else None))
                                del actual, expected
                                samples = {"source_serial": [], "ours": []}
                                for block in range(2):
                                    for mode in (("source_serial", "ours") if block == 0 else ("ours", "source_serial")):
                                        restore = (lambda: ref.set_state(initial)) if mode == "source_serial" else (lambda: env.set_state(initial.to(device)))
                                        fn = (lambda: source_step(ref, a, indices, device)) if mode == "source_serial" else (lambda: env.step(public_actions, states=None, frames=frame_slice))
                                        restore()
                                        fn()
                                        for _ in range(args.repeats):
                                            restore()
                                            elapsed, result = measure(fn, device)
                                            samples[mode].append(elapsed)
                                            del result
                                for mode, seconds in samples.items():
                                    report["timings"].append(dict(device=device, batch=batch, horizon=horizon,
                                        selection=selection, mode=mode, **stats(seconds, batch * horizon),
                                        frames=batch * len(indices), fps_mean=statistics.mean([batch * len(indices) / s for s in seconds]) if indices else None,
                                        fps_sd=statistics.stdev([batch * len(indices) / s for s in seconds]) if indices else None,
                                        candidate_processes=len(env._simulator.worker_info) if hasattr(env._simulator, "worker_info") else 0))
                                save()
                        final = env.get_state()
                        ref.set_state(final.cpu())
                        functions = (("ours_state_export", env.get_state),
                                     ("ours_render_saved", lambda: env.render(final)),
                                     ("source_render_saved", lambda: torch.as_tensor(np.stack([e.render() for e in ref.envs]), device=device)))
                        for operation, fn in functions:
                            fn()
                            samples = []
                            for _ in range(2 * args.repeats):
                                elapsed, result = measure(fn, device)
                                samples.append(elapsed)
                                del result
                            report["components"].append(dict(device=device, batch=batch, operation=operation, **stats(samples, batch)))
                        save()
                        print(args.domain, device, batch, "PASS", flush=True)
                finally:
                    ref.close()
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
