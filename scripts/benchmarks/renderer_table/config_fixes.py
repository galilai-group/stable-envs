"""Re-render immutable matched States for the Maze/Finger/Scene configuration fixes.

Each invocation owns a fresh process and output directory. Source images are
the independently captured, hash-recorded CPU/GPU images from the fresh corpus.
"""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import torch

from scripts.benchmarks.renderer_table.collect import candidate
from scripts.benchmarks.renderer_table.common import (
    catalog,
    load_state,
    profile,
    public_render,
    rgb,
    state_hash,
    write,
)


def sha(image):
    return hashlib.sha256(image.tobytes()).hexdigest()


def mse(a, b):
    return float(np.mean(((a.astype(np.float64) - b.astype(np.float64)) / 255) ** 2))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("campaign", type=Path)
    p.add_argument("name")
    p.add_argument("device", choices=["cpu", "cuda"])
    p.add_argument("output", type=Path)
    args = p.parse_args()
    args.output.mkdir(parents=True, exist_ok=False)
    row = next(r for r in catalog().values() if r["name"] == args.name)
    manipulation = args.name in ("OGBCube", "OGBScene")
    original = args.campaign / ("run-6751001_3" if manipulation else "run-6750648_1") / "results" / args.name
    meta_path = original / "cpu/result.json"
    meta = json.loads(meta_path.read_text())
    assert meta["status"] == "COMPLETE"
    env = candidate(row, args.device, 17)
    before = state_hash(env.get_state())
    result = dict(
        status="RUNNING",
        name=args.name,
        registration=row["registration"],
        device=args.device,
        profile=profile(),
        corpus_sha256=hashlib.sha256(meta_path.read_bytes()).hexdigest(),
        frames=[],
    )
    write(args.output / "result.json", result)
    try:
        for f in meta["frames"]:
            index = f["index"]
            state = load_state(original / "cpu" / f["state_file"], args.device, env.get_state())
            assert state_hash(state) == f["state_hash"]
            kw = {
                k: torch.tensor(v, dtype=torch.float32, device=args.device) if k == "style" else v
                for k, v in f["render_kwargs"].items()
            }
            image = rgb(public_render(env, state, **kw)).copy()
            again = rgb(public_render(env, state, **kw)).copy()
            np.testing.assert_array_equal(image, again)
            assert state_hash(state) == f["state_hash"] and state_hash(env.get_state()) == before
            filename = f"frame-{index:04d}.npy"
            old_path = original / "gpu" / f"frame-{index:04d}-gpu.npy"
            if manipulation:
                source_cpu_path = original / "cpu" / f["oracle_file"]
                source_gpu_path = original / "source_gpu" / f["oracle_file"]
            else:
                source_cpu_path = args.campaign / "isolated-6750890_0/results" / args.name / filename
                source_gpu_path = (
                    args.campaign
                    / ("isolated-gpu-6750891" if args.name == "DMCFingerTurnHard" else "isolated-gpu-checked-6751052")
                    / "results"
                    / args.name
                    / filename
                )
                if args.device == "cpu":
                    old_path = args.campaign / "isolated-6750890_1/results" / args.name / filename
            sources = {
                "source_cpu": np.load(source_cpu_path, allow_pickle=False),
                "source_gpu": np.load(source_gpu_path, allow_pickle=False),
                "old": np.load(old_path, allow_pickle=False),
            }
            for label, a in {**sources, "new": image}.items():
                np.save(args.output / f"frame-{index:04d}-{label}.npy", a, allow_pickle=False)
            if args.name == "OGBCube":
                np.testing.assert_array_equal(image, sources["old"])
            result["frames"].append(
                dict(
                    index=index,
                    case=f["case"],
                    seed=f["seed"],
                    step=f["step"],
                    state_hash=f["state_hash"],
                    render_kwargs=f["render_kwargs"],
                    repeated_frame_exact=True,
                    hashes={k: sha(a) for k, a in {**sources, "new": image}.items()},
                    source_paths={
                        "source_cpu": str(source_cpu_path),
                        "source_gpu": str(source_gpu_path),
                        "old": str(old_path),
                    },
                    **{
                        f"{side}_vs_{label}": mse(a, sources[label])
                        for side, a in [("old", sources["old"]), ("new", image)]
                        for label in ("source_cpu", "source_gpu")
                    },
                )
            )
            write(args.output / "result.json", result)
        if args.name == "OGBScene":
            # Independent lock colors and all four bit combinations in a single
            # heterogeneous batch must match individual renders and preserve State.
            rs = env.to_render_state(state).expand(4).clone()
            rs["pose", "buttons"][:] = torch.tensor([[0, 0], [0, 1], [1, 0], [1, 1]], device=args.device)
            rs["appearance", "lock_color"][:] = torch.tensor([[0.15, 0.7, 0.3], [0.2, 0.4, 0.8]], device=args.device)
            digest = state_hash(rs)
            images = env.render(rs).cpu().numpy()
            for i in range(4):
                np.testing.assert_array_equal(images[i], rgb(env.render(rs[i : i + 1])))
            assert state_hash(rs) == digest
            result["heterogeneous_button_batch"] = "PASS"
        result["status"] = "PASS"
        write(args.output / "result.json", result)
        print(args.name, args.device, len(result["frames"]), "matched frames PASS", flush=True)
    finally:
        env.close()


if __name__ == "__main__":
    main()
