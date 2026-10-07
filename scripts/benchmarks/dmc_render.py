"""Compare two checkouts on saved source rollout States, on the same device."""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

DOMAINS = ("walker", "hopper", "cheetah", "humanoid", "quadruped")


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _code(root: Path, package: str) -> dict[str, str]:
    return {str(path.relative_to(root)): _hash(path) for path in sorted((root / package).rglob("*.py"))}


def _hardware(device: str) -> dict:
    packages = ("torch", "numpy", "mujoco", "dm-control", "gymnasium", "mujoco-warp", "warp-lang")
    versions = {}
    for package in packages:
        try:
            versions[package] = importlib.metadata.version(package)
        except importlib.metadata.PackageNotFoundError:
            versions[package] = None
    return {
        "host": platform.node(),
        "device": device,
        "versions": versions,
        "gpu": torch.cuda.get_device_name() if device == "cuda" else None,
        "torch_threads": torch.get_num_threads(),
    }


def _worker(plan: dict) -> None:
    root = Path(plan["root"])
    sys.path.insert(0, str(root))
    from tensordict import TensorDict

    import stable_envs as se

    assert Path(se.__file__).resolve().is_relative_to(root), se.__file__
    path = Path(plan["corpus"])
    with np.load(path, allow_pickle=False) as saved:
        source = saved["source_frames"]
        assert source.dtype == np.uint8 and source.shape[1:] == (64, 64, 3), "Invalid source RGB corpus"
        states = TensorDict(
            {
                tuple(key.removeprefix("state/").split("/")): torch.from_numpy(saved[key])
                for key in saved.files
                if key.startswith("state/")
            },
            batch_size=[len(source)],
        ).to(plan["device"])

    def sync() -> None:
        if plan["device"] == "cuda":
            torch.cuda.synchronize()

    report = {
        **_hardware(plan["device"]),
        "root": str(root),
        "code_sha256": _code(root, "stable_envs"),
        "corpus_sha256": _hash(path),
        "unique_source_states": len(source),
        "measurements": [],
    }
    with se.make(plan["name"], num_envs=1, size=64, device=plan["device"]) as env:
        actual = env.render(states).cpu().numpy()
        assert actual.shape == source.shape and actual.dtype == source.dtype, "Renderer output cannot broadcast"
        mse = np.mean(((actual.astype(np.float64) - source) / 255) ** 2, axis=(1, 2, 3))
        report["mse_vs_source"] = {
            "per_state": mse.tolist(),
            "mean": float(mse.mean()),
            "sd": float(mse.std(ddof=1)),
            "max": float(mse.max()),
            "within_0_001": int((mse <= 0.001).sum()),
        }
        batches = (1, 64) if plan["device"] == "cpu" else (1, 64, 256)
        for batch in batches:
            samples = [
                states[(torch.arange(batch, device=plan["device"]) + i * batch) % len(states)]
                for i in range(plan["repeats"])
            ]
            for _ in range(3):
                env.render(samples[0])
            seconds = []
            for state in samples:
                sync()
                start = time.perf_counter()
                env.render(state)
                sync()
                seconds.append(time.perf_counter() - start)
            report["measurements"].append(
                {
                    "batch": batch,
                    "seconds": seconds,
                    "seconds_mean": float(np.mean(seconds)),
                    "seconds_sd": float(np.std(seconds, ddof=1)),
                    "fps": batch / float(np.mean(seconds)),
                }
            )
    assert report["code_sha256"] == _code(root, "stable_envs"), "Checkout changed during measurement"
    with Path(plan["result"]).open("x") as stream:
        json.dump(report, stream, indent=2, allow_nan=False)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--baseline-root", type=Path)
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--device", choices=("cpu", "cuda"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--repeats", type=int, default=12)
    parser.add_argument("--worker", type=Path, help=argparse.SUPPRESS)
    args = parser.parse_args()
    if args.worker:
        _worker(json.loads(args.worker.read_text()))
        return
    if not all((args.baseline_root, args.source_root, args.device, args.output)) or args.repeats < 2:
        parser.error("baseline-root, source-root, device, output and repeats >= 2 are required")
    candidate = Path(__file__).resolve().parents[2]
    source_root, baseline = args.source_root.resolve(), args.baseline_root.resolve()
    sys.path[:0] = [str(candidate), str(source_root)]
    from stable_envs.testing.evidence import EvidenceRun
    from stable_envs.testing.provenance import pinned_source_identity
    from tests.helpers.reference.bridges import load_source
    from tests.helpers.reference.dmc_pixels import dmc_trace
    from tests.helpers.reference.profiles import PINS, PROFILES
    from tests.helpers.reference.saved import SWM_PIN

    source_identity = pinned_source_identity(source_root, SWM_PIN, PINS["dmc"])
    load_source(source_root, "dmc")
    evidence = EvidenceRun(args.output)
    hardware = _hardware(args.device)
    report = {
        **hardware,
        "baseline_root": str(baseline),
        "candidate_root": str(candidate),
        "source_root": str(source_root),
        "source_identity": source_identity,
        "runner_sha256": _hash(Path(__file__)),
        "warmups": 3,
        "repeats": args.repeats,
        "output_size": 64,
        "seeds": [17, 42],
        "horizon": 8,
        "lanes": 2,
        "timing_scope": "Public render of device-resident full States plus output synchronization; includes State packing",
        "excluded": "Source rollout, disk loading, initial device transfer, batch indexing and initialization",
        "comparison": "Our renderer before/after; MSE versus source; source throughput is not measured",
        "sampling": "32 distinct reached States per domain; larger batches repeat those States",
        "rows": [],
    }
    for index, domain in enumerate(DOMAINS):
        case = next(case for case in PROFILES["dmc"] if case.domain == domain)
        traces, identities = [], []
        for seed in (17, 42):
            trace, identity = dmc_trace(case, seed, "float64", 8, args.device)
            traces.append(trace)
            identities.append(identity)
        assert identities[0] == identities[1]
        corpus = torch.cat([torch.stack(trace.states, dim=1).reshape(-1) for trace in traces])
        corpus_name = f"{domain}/states.npz"
        evidence.arrays(
            corpus_name,
            source_frames=np.concatenate([trace.frames.reshape(-1, 64, 64, 3) for trace in traces]),
            seed=np.repeat([17, 42], 16),
            lane=np.tile(np.repeat([0, 1], 8), 2),
            step=np.tile(np.arange(1, 9), 4),
            **{
                "state/" + ("/".join(key) if isinstance(key, tuple) else key): value.numpy()
                for key, value in corpus.items(include_nested=True, leaves_only=True)
            },
        )
        results = {}
        order = ("before", "after") if index % 2 == 0 else ("after", "before")
        for label in order:
            root = baseline if label == "before" else candidate
            plan = {
                "root": str(root),
                "corpus": str(evidence.root / corpus_name),
                "name": case.name,
                "device": args.device,
                "repeats": args.repeats,
                "result": str(evidence.root / domain / f"{label}.json"),
            }
            plan_name = f"{domain}/{label}-plan.json"
            evidence.json(plan_name, plan)
            environment = dict(os.environ, PYTHONPATH=os.pathsep.join((str(root), str(source_root))))
            with (evidence.root / domain / f"{label}.log").open("x") as stream:
                subprocess.run(
                    [sys.executable, str(Path(__file__).resolve()), "--worker", str(evidence.root / plan_name)],
                    cwd=root,
                    env=environment,
                    stdout=stream,
                    stderr=subprocess.STDOUT,
                    check=True,
                )
            result = json.loads(Path(plan["result"]).read_text())
            assert all(result[key] == hardware[key] for key in hardware), "Worker runtime changed"
            assert result["corpus_sha256"] == _hash(evidence.root / corpus_name)
            results[label] = result
        for before, after in zip(results["before"]["measurements"], results["after"]["measurements"], strict=True):
            assert before["batch"] == after["batch"]
            row = {
                "env": case.registration,
                "domain": domain,
                "batch": before["batch"],
                "before": before,
                "after": after,
                "speedup": after["fps"] / before["fps"],
                "mse_before": results["before"]["mse_vs_source"],
                "mse_after": results["after"]["mse_vs_source"],
                "source_gl": identities[0],
                "order": list(order),
            }
            report["rows"].append(row)
            print(json.dumps(row), flush=True)
    assert pinned_source_identity(source_root, SWM_PIN, PINS["dmc"])["files"] == source_identity["files"]
    evidence.json("result.json", report)
    evidence.seal()


if __name__ == "__main__":
    main()
