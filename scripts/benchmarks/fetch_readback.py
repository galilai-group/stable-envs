"""Compare Fetch OpenGL implementations on the same saved rollout States."""

import argparse
import hashlib
import importlib.util
import json
import platform
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import stable_envs as se
from stable_envs.envs import fetch_opengl


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=("cpu", "cuda"), required=True)
    parser.add_argument("--baseline-file", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--repeats", type=int, default=12)
    args = parser.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    spec = importlib.util.spec_from_file_location("stable_envs.envs._fetch_readback_baseline", args.baseline_file)
    baseline = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(baseline)
    implementations = {"baseline": baseline.render, "candidate": fetch_opengl.render}
    report = {
        "host": platform.node(),
        "device": args.device,
        "gpu": torch.cuda.get_device_name() if args.device == "cuda" else None,
        "baseline_sha256": hashlib.sha256(args.baseline_file.read_bytes()).hexdigest(),
        "candidate_sha256": hashlib.sha256(Path(fetch_opengl.__file__).read_bytes()).hexdigest(),
        "output_size": 64,
        "warmups": 3,
        "repeats": args.repeats,
        "source_comparison": False,
        "rows": [],
    }

    def sync() -> None:
        if args.device == "cuda":
            torch.cuda.synchronize()

    for scene in ("Reach", "Push", "Slide", "PickAndPlace"):
        with se.make(f"Fetch{scene}", num_envs=32, size=64, device=args.device, backend="opengl") as env:
            env.reset(seed=17)
            actions = torch.rand((32, 32, 4), generator=torch.Generator().manual_seed(42)) * 2 - 1
            corpus = env.step(actions, frames=None)["states"].reshape(-1).cpu()
            corpus_path = args.output / f"{scene}-states.npz"
            np.savez_compressed(
                corpus_path,
                **{
                    "/".join(key) if isinstance(key, tuple) else key: value.numpy()
                    for key, value in corpus.items(include_nested=True, leaves_only=True)
                },
            )
            # Reload the saved corpus; rollout and device transfer are outside render timing.
            from tensordict import TensorDict

            with np.load(corpus_path) as saved:
                corpus = TensorDict(
                    {tuple(key.split("/")): torch.from_numpy(saved[key]) for key in saved.files},
                    batch_size=[1024],
                ).to(args.device)
            for batch in (1, 64) if args.device == "cpu" else (1, 64, 256):
                timings = {name: [] for name in implementations}
                errors = []
                for name, implementation in implementations.items():
                    with patch.object(fetch_opengl, "render", implementation):
                        for _ in range(3):
                            env.render(corpus[:batch])
                for repeat in range(args.repeats):
                    indices = (torch.arange(batch, device=args.device) + repeat * batch) % len(corpus)
                    states = corpus[indices]
                    frames = {}
                    for name in list(implementations)[:: 1 if repeat % 2 else -1]:
                        with patch.object(fetch_opengl, "render", implementations[name]):
                            sync()
                            start = time.perf_counter()
                            image = env.render(states)
                            sync()
                            timings[name].append(time.perf_counter() - start)
                            frames[name] = image.cpu().numpy().astype(np.float64)
                    error = (frames["candidate"] - frames["baseline"]) / 255
                    errors.extend(np.mean(error**2, axis=(1, 2, 3)).tolist())
                assert max(errors) < 1e-5, f"Resize regression: {scene} B{batch}: {max(errors)}"
                row = {
                    "scene": scene,
                    "batch": batch,
                    "states_sha256": hashlib.sha256(corpus_path.read_bytes()).hexdigest(),
                    "mse_mean_vs_baseline": float(np.mean(errors)),
                    "mse_max_vs_baseline": max(errors),
                    "measurements": {
                        name: {
                            "seconds": values,
                            "seconds_mean": float(np.mean(values)),
                            "seconds_sd": float(np.std(values, ddof=1)),
                            "fps": batch / float(np.mean(values)),
                        }
                        for name, values in timings.items()
                    },
                }
                report["rows"].append(row)
                (args.output / "result.json").write_text(json.dumps(report, indent=2) + "\n")
                print(json.dumps(row), flush=True)


if __name__ == "__main__":
    main()
