"""Compare shared replay with G34 specializations, including CPU pose upload.

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


def comparison_gif(clip, source, output, *, size):
    """Align native Blender PNGs and shared RGB at the same original frame IDs."""
    from PIL import Image, ImageDraw

    metadata = json.loads((source / "manifest.json").read_text())
    recorded_source = clip.manifest["source"]
    if metadata["task"] != recorded_source["task"] or metadata["source_commit"] != recorded_source["commit"]:
        raise ValueError("source preview task or pinned source differs")
    if metadata["frames"] != list(clip.frames) or metadata["size"] != [size, size * 16 // 9]:
        raise ValueError("source preview frame numbers or dimensions differ")
    with RigidSceneRenderer(clip, size=size) as renderer:
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
        draw.text((left.width + 6, 6), "Shared nvdiff / same frame", fill="black")
        frames.append(canvas)
    palette = frames[0].quantize(colors=256)
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


def diagnose(args):
    """Report full-sequence legacy discrepancies without changing source gates."""
    from stable_envs.envs.object_permanence.rendering.g34_nvdiff import G34Renderer
    from stable_envs.envs.object_permanence.rendering.g34_raycast import render_raycast
    from stable_envs.envs.object_permanence.scenes.g34 import authored_pose
    from stable_envs.testing.geometry import depth_mse, mask_iou

    configure_threads(1)
    manifest = json.loads((args.oracle / "manifest.json").read_text())
    with np.load(args.oracle / manifest["file"], allow_pickle=False) as archive:
        source = dict(archive)
    depth = (source["depth_xor"] ^ source["depth_base"].view(np.uint32)).view(np.float32)
    with RigidTrajectory(args.recordings, "g34-nominal") as clip:
        imported = clip.poses(np.arange(len(clip)))[:, 0]
    authored = np.concatenate(authored_pose(np.arange(120)), -1).astype(np.float32)
    cases = {}
    for name, rows in (("cropped-imported", imported), ("cropped-authored", authored)):
        with G34Renderer(size=args.size) as renderer:
            cases[name] = {m: renderer.render(torch.from_numpy(rows), mode=m).cpu().numpy() for m in ("depth", "ids")}
    cases["raycast-imported"] = {
        m: render_raycast(
            torch.from_numpy(imported), height=args.size, width=args.size * 16 // 9, device="cuda", mode=m
        )
        .cpu()
        .numpy()
        for m in ("depth", "ids")
    }
    import stable_envs as se

    with se.MuJoCoOcclusionBall(size=args.size) as env:
        initial = env.reset()
        states = env.step(np.zeros((1, 119, 1)), frames=None)["states"]
        bank = torch.cat([initial, states.squeeze(0)], 0)
        fields = env.geometry(bank)
        cases["native-authored"] = {"depth": fields["depth"].numpy(), "ids": fields["object_ids"].numpy()}
    report = {}
    for name, fields in cases.items():
        bad = []
        for index, frame in enumerate(source["geometry_frames"]):
            ids = fields["ids"][index]
            expected = source["object_ids"][index]
            roi = ((ids >= 4) | (expected >= 4)) & (depth[index] > 0)
            mse = float(depth_mse(fields["depth"][index], depth[index], roi))
            iou = float(mask_iou(ids == 5, expected == 5))
            if mse > 1e-8 or (not np.isnan(iou) and iou < 0.99):
                bad.append(
                    {"frame": int(frame), "depth_mse": mse, "ball_iou": iou, "id_pixels": int((ids != expected).sum())}
                )
        report[name] = bad
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps(report, indent=2))


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
            "precision": "shared refined cached coverage; shared-full uncached full-frame; G34 cropped scaled coverage/refined static; legacy fused exact-facet raycast",
            "notes": "different simple shaders; timings are rendering throughput, not training speedups; shared and shared-full use chunk size 4",
        },
        "points": [],
        "profiles": {},
    }
    cases = [
        ("g34", "shared"),
        ("g62", "shared"),
        ("g34", "shared-full"),
        ("g62", "shared-full"),
        ("g34", "cropped"),
        ("g34", "cropped-refined"),
        ("g34", "raycast"),
    ]
    cases = [(task, backend) for task, backend in cases if backend in args.backends.split(",")]
    for task, backend in cases:
        with RigidTrajectory(args.recordings, f"{task}-nominal") as clip:
            owned = None
            if backend in ("shared", "shared-full"):
                owned = RigidSceneRenderer(clip, size=args.size, cache_static=backend == "shared")

                def render(indices, out=None):
                    return owned.render_many(indices, modes=modes, out=out)
            elif backend in ("cropped", "cropped-refined"):
                from stable_envs.envs.object_permanence.rendering.g34_nvdiff import G34Renderer

                owned = G34Renderer(
                    size=args.size, coverage_mode="refined" if backend == "cropped-refined" else "scaled"
                )

                def render(indices, out=None):
                    rows = torch.from_numpy(clip.poses(indices)[:, 0])
                    return owned.render_many(rows, modes=modes, out=out)
            else:
                from stable_envs.envs.object_permanence.rendering.g34_raycast import render_raycast

                def render(indices, out=None):
                    rows = torch.from_numpy(clip.poses(indices)[:, 0])
                    return {
                        m: render_raycast(
                            rows,
                            height=args.size,
                            width=args.size * 16 // 9,
                            device="cuda",
                            mode=m,
                            out=None if out is None else out[m],
                        )
                        for m in modes
                    }

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
                        * 28
                        * (
                            len(clip.manifest["objects"])
                            if backend == "shared-full"
                            else len(clip.manifest["dynamic_nodes"])
                        )
                        * (len(modes) if backend == "raycast" else 1),
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
                if owned is not None:
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
    parser.add_argument("--batches", type=positive_ints, default=[1, 16, 64, 256, 1024])
    parser.add_argument("--size", type=int, default=180)
    parser.add_argument("--modes", default="rgb")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=11)
    parser.add_argument("--profile", action="store_true")
    parser.add_argument("--backends", default="shared,shared-full,cropped,cropped-refined,raycast")
    parser.add_argument("--source-previews", type=Path)
    parser.add_argument("--oracle", type=Path, help="instead run a full legacy diagnostic against this G34 oracle")
    args = parser.parse_args()
    if args.size <= 0 or args.warmup < 0 or args.repeats < 1:
        parser.error("invalid size/warmup/repeats")
    modes = args.modes.split(",")
    if not modes or len(set(modes)) != len(modes) or any(m not in ("rgb", "depth", "ids") for m in modes):
        parser.error("modes must be unique rgb/depth/ids")
    if any(
        b not in ("shared", "shared-full", "cropped", "cropped-refined", "raycast") for b in args.backends.split(",")
    ):
        parser.error("unknown backend")
    if args.oracle:
        diagnose(args)
    else:
        benchmark(args)


if __name__ == "__main__":
    main()
