"""Measure recorded-scene GPU rendering, including CPU pose upload.

Native Blender preview PNGs are optional visual references, never timing inputs.
Outputs and profiler reports are local artifacts, not frozen reference fixtures.
"""

from __future__ import annotations

import argparse
import json
import time
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer

from ._common import configure_threads, measure, positive_ints


def comparison_gif(clip, source, output, *, size, visibility_contract=None):
    """Align native Blender PNGs and shared RGB at the same original frame IDs."""
    from PIL import Image, ImageDraw

    metadata = json.loads((source / "manifest.json").read_text())
    recorded_source = clip.manifest["source"]
    if metadata["task"] != recorded_source["task"] or metadata["source_commit"] != recorded_source["commit"]:
        raise ValueError("source preview task or pinned source differs")
    if (
        metadata.get("member", 0) != recorded_source.get("member", 0)
        or metadata.get("entry_profile", "nominal") != recorded_source.get("entry_profile", "nominal")
        or metadata["source_script_sha256"] != recorded_source["script_sha256"]
    ):
        raise ValueError("source preview member/profile/script differs")
    if metadata.get("source_execution_context") != recorded_source.get("execution_context"):
        raise ValueError("source preview and recording execution contexts differ")
    if metadata.get("source_realization") != recorded_source.get("realization"):
        raise ValueError("source preview and recording native realizations differ")
    if (visibility_contract == "stable-envs/opaque-pixel-rays-v2" or clip.manifest.get("geometry_conversion")) and (
        metadata.get("source_preparation_recipe") != recorded_source.get("preparation_recipe")
        or metadata["source_script_sha256"] != recorded_source["script_sha256"]
    ):
        raise ValueError("source preview and recording preparation definitions differ")
    if metadata["frames"] != list(clip.frames) or metadata["size"] != [size, size * 16 // 9]:
        raise ValueError("source preview frame numbers or dimensions differ")
    with RigidSceneRenderer(clip, size=size, visibility_contract=visibility_contract) as renderer:
        rgb = renderer.render(np.arange(len(clip))).cpu().numpy()
    frames = []
    for frame, pixels in zip(clip.frames, rgb, strict=True):
        left = Image.open(source / f"{frame:04d}.png").convert("RGB")
        right = Image.fromarray(pixels)
        canvas = Image.new("RGB", (left.width * 2, left.height + 24), "white")
        canvas.paste(left, (0, 24))
        canvas.paste(right, (left.width, 24))
        draw = ImageDraw.Draw(canvas)
        draw.text((6, 6), f"Source Blender / frame {frame}", fill="black")
        label = (
            "Shared / " + visibility_contract.removeprefix("stable-envs/opaque-").replace("-v", " v")
            if visibility_contract
            else "Shared nvdiff / same frame"
        )
        draw.text((left.width + 6, 6), label, fill="black")
        frames.append(canvas)
    # Initially occluded objects can introduce colors absent from frame one.
    # Include the complete clip so GIF quantization does not erase their identity.
    palette_source = Image.new("RGB", (frames[0].width, frames[0].height * len(frames)))
    for index, frame in enumerate(frames):
        palette_source.paste(frame, (0, index * frame.height))
    palette = palette_source.quantize(colors=256)
    frames = [frame.quantize(palette=palette, dither=Image.Dither.NONE) for frame in frames]
    output.parent.mkdir(parents=True, exist_ok=True)
    frames[0].save(
        output,
        save_all=True,
        append_images=frames[1:],
        duration=[50 if i % 6 == 5 else 40 for i in range(len(frames))],
        loop=0,
        optimize=False,
    )


def profile_draw(draw):
    """Capture operator totals separately, without perturbing throughput samples."""
    from torch.profiler import ProfilerActivity, profile

    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as p:
        draw()
        torch.cuda.synchronize()
    events = p.key_averages()
    return {
        "gpu_operators": events.table(sort_by="self_device_time_total", row_limit=12),
        "cpu_operators": events.table(sort_by="self_cpu_time_total", row_limit=12),
        "item_calls": sum(event.count for event in events if event.key == "aten::item"),
    }


def compare(draws, *, warmup, repeats):
    """Time retained outputs, alternating variants to limit sample-order bias."""
    for _ in range(warmup):
        for draw in draws.values():
            draw()
    times = {name: [] for name in draws}
    for sample in range(repeats):
        order = list(draws) if sample % 2 == 0 else list(reversed(draws))
        for name in order:
            torch.cuda.synchronize()
            start = time.perf_counter()
            draws[name]()
            torch.cuda.synchronize()
            times[name].append((time.perf_counter() - start) * 1000)
    return times


def benchmark(args):
    configure_threads(1)
    modes = tuple(args.modes.split(","))
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "nvdiffrast": version("nvdiffrast"),
        "size": [args.size, args.size * 16 // 9],
        "warmup": args.warmup,
        "repeats": args.repeats,
        "threads": 1,
        "modes": modes,
        "methodology": {
            "included": "warm CPU frame lookup/pose expansion, CPU-to-GPU upload, transforms, rendering into reused CUDA outputs",
            "excluded": "download, constructor, first draw/JIT/static bake, output allocation, GIF/CPU image download",
            "sampling": "same evenly spaced nominal frame indices, synchronized wall-clock median",
            "precision": "shared refined cached coverage; shared-full uncached full-frame",
            "notes": "different simple shaders; timings are rendering throughput, not training speedups; shared and shared-full use chunk size 4",
        },
        "points": [],
        "profiles": {},
    }
    clips = getattr(args, "clips", ("g34-nominal", "g62-nominal"))
    cases = [(clip, backend) for backend in args.backends.split(",") for clip in clips]
    for clip_name, backend in cases:
        task = clip_name.removesuffix("-nominal")
        with RigidTrajectory(args.recordings, clip_name) as clip:
            manifest = clip.manifest
            owned = RigidSceneRenderer(
                clip,
                size=args.size,
                cache_static=backend == "shared",
                visibility_contract=manifest.get("visibility_contract"),
            )

            def render(indices, out=None):
                return owned.render_many(indices, modes=modes, out=out)

            try:
                torch.cuda.synchronize()
                first_start = time.perf_counter()
                render(np.array([0]))
                torch.cuda.synchronize()
                first_ms = (time.perf_counter() - first_start) * 1000
                for batch in args.batches:
                    frames = np.linspace(0, len(clip) - 1, batch).round().astype(np.int64)
                    first = render(frames)
                    output = {m: torch.empty_like(value) for m, value in first.items()}

                    def draw():
                        render(frames, output)

                    timing = measure(draw, device=torch.device("cuda"), warmup=args.warmup, repeats=args.repeats)
                    point = {
                        "task": task,
                        "backend": backend,
                        "batch": batch,
                        "first_draw_ms": first_ms,
                        "timing": timing,
                        "frames_per_second": batch * 1000 / timing["median_ms"],
                        "pose_upload_bytes": batch
                        * len(manifest["dynamic_nodes"])
                        * (128 if "transforms" in manifest["arrays"] else 28 + (12 if clip.has_scales else 0)),
                    }
                    result["points"].append(point)
                    print(
                        task,
                        backend,
                        batch,
                        f"{timing['median_ms']:.3f} ms / {point['frames_per_second']:.0f} fps",
                        flush=True,
                    )
                    if args.profile and backend in ("shared", "shared-full") and batch == 256:
                        result["profiles"][f"{task}-{backend}"] = profile_draw(draw)
            finally:
                owned.close()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    if args.source_previews:
        for task in ("g34", "g62"):
            with RigidTrajectory(args.recordings, f"{task}-nominal") as clip:
                comparison_gif(
                    clip,
                    args.source_previews / f"source-{task}",
                    args.output.parent / f"{task}-source-vs-shared.gif",
                    size=args.size,
                )


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--clips", nargs="+", default=["g34-nominal", "g62-nominal"])
    parser.add_argument("--batches", type=positive_ints, default=[1, 16, 64, 256, 1024])
    parser.add_argument("--size", type=int, default=180)
    parser.add_argument("--modes", default="rgb")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--backends", default="shared,shared-full")
    parser.add_argument("--source-previews", type=Path)
    args = parser.parse_args()
    if args.size <= 0 or args.warmup < 0 or args.repeats < 1:
        parser.error("invalid size/warmup/repeats")
    modes = args.modes.split(",")
    if not modes or len(set(modes)) != len(modes) or any(m not in ("rgb", "depth", "ids") for m in modes):
        parser.error("modes must be unique rgb/depth/ids")
    if any(b not in ("shared", "shared-full") for b in args.backends.split(",")):
        parser.error("unknown backend")
    benchmark(args)


if __name__ == "__main__":
    main()
