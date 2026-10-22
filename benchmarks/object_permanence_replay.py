"""Compare upload-inclusive G34 rasterization with the legacy native/raycast paths.

Default nvdiff execution needs no MuJoCo. Legacy backends are opt-in baselines.
JSON and optional GIFs are local outputs, not committed reference fixtures.
"""

from __future__ import annotations

import argparse
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from stable_envs.envs.object_permanence.replay import G34Replay

from ._common import configure_threads, measure, positive_ints, write_json


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--size", type=int, default=90)
    parser.add_argument(
        "--window-size", type=int, default=None, help="optional fixed nvdiff window side in source pixels"
    )
    parser.add_argument("--batches", type=positive_ints, default=[1, 16, 64, 256])
    parser.add_argument("--backends", default="nvdiff", help="comma-separated nvdiff,raycast,opengl")
    parser.add_argument("--modes", default="rgb", help="comma-separated rgb,depth,ids (multiple modes require nvdiff)")
    parser.add_argument(
        "--separate", action="store_true", help="time separate single-mode calls instead of render_many"
    )
    parser.add_argument("--coverage", choices=("scaled", "refined", "native"), default="scaled")
    parser.add_argument("--background-coverage", choices=("scaled", "refined", "native"), default="refined")
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--repeats", type=int, default=21)
    parser.add_argument("--output", default="-")
    parser.add_argument("--gif", default=None, help="optional nvdiff full-trajectory GIF path")
    args = parser.parse_args()
    if args.warmup < 0 or args.repeats < 1:
        parser.error("warmup must be nonnegative and repeats positive")
    configure_threads(1)
    device = torch.device("cuda")
    backends = args.backends.split(",")
    modes = tuple(args.modes.split(","))
    if any(b not in ("nvdiff", "raycast", "opengl") for b in backends):
        parser.error("unknown backend")
    if len(set(modes)) != len(modes) or any(m not in ("rgb", "depth", "ids") for m in modes):
        parser.error("modes must be unique rgb/depth/ids")
    if modes != ("rgb",) and any(b != "nvdiff" for b in backends):
        parser.error("diagnostic modes require --backends nvdiff")
    result = {
        "benchmark": "g34-engine-free-replay-upload",
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvdiffrast": version("nvdiffrast"),
        "size": [args.size, args.size * 16 // 9],
        "threads": 1,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "modes": modes,
        "separate": args.separate,
        "methodology": {
            "timed": "CPU checkpoint validation/projection + rendering + all pose/camera or RGB uploads into reused GPU RGB output",
            "source": "same evenly sampled phases of the nominal 120-frame trajectory",
            "excluded": "trajectory creation, constructor, first draw, output allocation, GIF download/encoding",
            "notes": "synchronized wall clock; native baseline is OSMesa, not NVIDIA EGL or Blender; no training measured",
            "nvdiff_upload": "28 pose bytes/frame plus derived crop projections/indices and shared camera/light tensors; all included in timings",
            "nvdiff_precision": {"dynamic": args.coverage, "background": args.background_coverage},
            "memory": "peak_cuda_allocated_bytes is Torch allocations, not total device/context workspace or process memory",
            "startup": "first draw after constructor/trajectory setup; legacy caches can be warm from a preceding backend, not independent cold-start comparisons",
        },
        "points": [],
        "startup": {},
    }
    for backend in backends:
        if backend == "nvdiff":
            env = G34Replay(
                size=args.size,
                window_size=args.window_size,
                coverage_mode=args.coverage,
                background_coverage=args.background_coverage,
            )
            bank = env.state_at(torch.arange(120))
        else:
            import stable_envs as se

            env = se.MuJoCoOcclusionBall(
                size=args.size, backend=backend, device="cpu" if backend == "opengl" else "cuda"
            )
            initial = env.reset()
            sequence = env.step(np.zeros((1, 119, 1)), frames=None)["states"]
            bank = torch.cat([initial, sequence.squeeze(0)], 0).cpu()
        with env:

            def render(state, output=None):
                if backend == "nvdiff" and not args.separate:
                    return env.render_many(state, modes=modes, out=output)
                if backend == "nvdiff":
                    return {m: env.render(state, mode=m, out=None if output is None else output[m]) for m in modes}
                return {"rgb": env.render(state, out=None if output is None else output["rgb"])}

            torch.cuda.synchronize()
            start = time.perf_counter()
            render(bank[:1])
            torch.cuda.synchronize()
            result["startup"][backend] = {"first_draw_ms": (time.perf_counter() - start) * 1000}
            if backend == "nvdiff":
                result["nvdiff_viewport"] = {
                    "window_size": env._renderer.patch,
                    "coverage_scale": env._renderer.renderer.coverage_scale,
                    "chunk": env._renderer.chunk,
                }
            for batch in args.batches:
                indices = torch.linspace(0, 119, batch).round().long()
                state = bank[indices].clone()
                first = render(state)
                output = {m: torch.empty_like(v) for m, v in first.items()}
                delivered = {m: v if v.is_cuda else torch.empty_like(v, device="cuda") for m, v in output.items()}

                def draw():
                    render(state, output)
                    for mode in modes:
                        if delivered[mode] is not output[mode]:
                            delivered[mode].copy_(output[mode])

                timing = measure(draw, device=device, warmup=args.warmup, repeats=args.repeats)
                result["points"].append(
                    {
                        "backend": backend,
                        "batch": batch,
                        "timing": timing,
                        "frames_per_second": batch * 1000 / timing["median_ms"],
                        "pose_upload_bytes": 28 * batch * (len(modes) if args.separate and backend == "nvdiff" else 1)
                        if backend != "opengl"
                        else 0,
                        "rgb_upload_bytes": batch * args.size * (args.size * 16 // 9) * 3
                        if backend == "opengl"
                        else 0,
                        "peak_cuda_allocated_bytes": torch.cuda.max_memory_allocated(),
                    }
                )
            if args.gif and backend == "nvdiff":
                from PIL import Image

                frames = []
                for state in bank.split(16):
                    frames.extend(Image.fromarray(im) for im in env.render(state).cpu().numpy())
                path = Path(args.gif)
                path.parent.mkdir(parents=True, exist_ok=True)
                # A shared palette prevents temporal color flicker. GIF stores
                # durations in 10ms units: alternate 40/50ms to total 5s at 24Hz.
                palette = frames[0].quantize(colors=128)
                frames = [im.quantize(palette=palette, dither=Image.Dither.NONE) for im in frames]
                durations = [50 if i % 6 == 5 else 40 for i in range(120)]
                frames[0].save(
                    path, save_all=True, append_images=frames[1:], duration=durations, loop=0, optimize=False
                )
                result["gif"] = str(path.resolve())
    write_json(result, args.output, pretty=True)


if __name__ == "__main__":
    main()
