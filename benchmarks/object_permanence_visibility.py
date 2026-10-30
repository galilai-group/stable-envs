"""Compare full and cached versioned visibility, including CPU pose upload.

Both cases keep static transforms resident and upload only dynamic poses. This
isolates visibility caching from the older pose-upload optimization. Alternating
sample order reduces case-order bias; output allocation/JIT/bake are excluded.
"""

import argparse
import json
from contextlib import ExitStack
from importlib.metadata import version
from pathlib import Path

import numpy as np
import torch

from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT

from ._common import configure_threads, positive_ints
from .object_permanence_recordings import compare, profile_draw


def run(args):
    configure_threads(1)
    modes = tuple(args.modes.split(","))
    if args.warmup < 0 or args.repeats < 1:
        raise ValueError("warmup must be nonnegative and repeats positive")
    report: dict = {
        "visibility_contract": CONTRACT,
        "gpu": torch.cuda.get_device_name(),
        "versions": {k: version(k) for k in ("torch", "triton", "nvdiffrast")},
        "size": [args.size, args.size * 16 // 9],
        "modes": modes,
        "methodology": {
            "included": "CPU frame lookup, dynamic pose upload, transforms, cache validation, rendering into reused outputs",
            "excluded": "constructor, first draw/JIT/bake, caller output allocation, image download and encoding",
            "comparison": "same retained static transforms and dynamic-only pose upload; full coverage disables only visibility seeds",
            "sampling": "synchronized wall time; alternate full/cached sample order; same evenly spaced frame indices",
            "chunk_size": 4,
            "warmup": args.warmup,
            "repeats": args.repeats,
            "cpu_threads": 1,
        },
        "points": [],
        "profiles": {},
    }
    for task in args.tasks.split(","):
        with RigidTrajectory(args.recordings, f"{task.lower()}-nominal") as clip:
            for batch in args.batches:
                frames = np.linspace(0, len(clip) - 1, batch).round().astype(np.int64)
                with ExitStack() as stack:
                    renderers = {
                        name: stack.enter_context(
                            RigidSceneRenderer(clip, size=args.size, visibility_contract=CONTRACT)
                        )
                        for name in ("full", "cached")
                    }
                    renderers["full"].renderer._ray_coverage.clear()
                    renderers["full"].renderer._ray_coverage = None
                    buffers = {name: r.render_many(frames, modes=modes) for name, r in renderers.items()}
                    assert all(torch.equal(buffers["full"][m], buffers["cached"][m]) for m in modes)

                    def draw(name):
                        renderers[name].render_many(frames, modes=modes, out=buffers[name])

                    times = compare(
                        {name: lambda name=name: draw(name) for name in renderers},
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    seed = renderers["cached"].renderer._ray_coverage
                    memory = {
                        "pixel_seed_bytes": sum(x.numel() * x.element_size() for x in (seed._depth, seed._winner)),
                        "validation_snapshot_bytes": sum(x.numel() * x.element_size() for x in seed._snapshot),
                        "static_faces": len(seed.static),
                        "dynamic_faces": len(seed.dynamic),
                        "seed_builds": seed.builds,
                    }
                    point = {
                        "task": task.upper(),
                        "source_commit": clip.manifest["source"]["commit"],
                        "batch": batch,
                        "pose_upload_bytes": batch * 28 * len(clip.manifest["dynamic_nodes"]),
                        "cache": memory,
                        "outputs_bit_identical": True,
                        "timing": {
                            name: {
                                "median_ms": float(np.median(values)),
                                "p25_ms": float(np.quantile(values, 0.25)),
                                "p75_ms": float(np.quantile(values, 0.75)),
                                "samples_ms": values,
                                "fps": batch * 1000 / float(np.median(values)),
                            }
                            for name, values in times.items()
                        },
                    }
                    point["speedup"] = point["timing"]["full"]["median_ms"] / point["timing"]["cached"]["median_ms"]
                    report["points"].append(point)
                    print(
                        task,
                        batch,
                        {name: round(t["fps"]) for name, t in point["timing"].items()},
                        "speedup",
                        round(point["speedup"], 2),
                        flush=True,
                    )
                    if args.profile and batch == max(args.batches):
                        for name in renderers:
                            report["profiles"][f"{task}-{name}"] = profile_draw(lambda: draw(name))
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--recordings", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--tasks", default="g82,g84,g86")
    parser.add_argument("--batches", type=positive_ints, default=[1, 256])
    parser.add_argument("--size", type=int, default=180)
    parser.add_argument("--modes", default="rgb")
    parser.add_argument("--warmup", type=int, default=3)
    parser.add_argument("--repeats", type=int, default=7)
    parser.add_argument("--profile", action="store_true")
    run(parser.parse_args())


if __name__ == "__main__":
    main()
