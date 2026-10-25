"""Measure optical replay including CPU frame selection and CUDA state upload."""

import argparse
import json
import statistics
import time
from pathlib import Path

import numpy as np
import torch

from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer

VARIANTS = ("full", "static-torch", "static-fused-unbounded", "static-fused")


def storage_bytes(renderer):
    """Count retained CUDA storage once, including views shared by cache entries."""
    seen = set()

    def count(value):
        if torch.is_tensor(value) and value.is_cuda:
            storage = value.untyped_storage()
            key = (value.device, storage.data_ptr())
            if key not in seen:
                seen.add(key)
                return storage.nbytes()
        elif isinstance(value, (tuple, list)):
            return sum(count(v) for v in value)
        return 0

    result = {name: count(value) for name, value in vars(renderer).items()}
    result["query"] = sum(count(value) for value in vars(renderer.query).values())
    return {name: size for name, size in result.items() if size}


def benchmark(root, clip, *, size=180, batches=(1, 4, 16), repeats=3, variants=VARIANTS):
    rows = []
    for variant in variants:
        with RigidTrajectory(root, clip) as trajectory:
            start = time.perf_counter()
            with RigidSceneRenderer(
                trajectory,
                size=size,
                visibility_contract=trajectory.manifest["visibility_contract"],
                cache_static=variant != "full",
            ) as renderer:
                renderer.renderer.fuse_cache = variant.startswith("static-fused")
                renderer.renderer.use_ray_limits = variant == "static-fused"
                torch.cuda.synchronize()
                initialization = time.perf_counter() - start
                start = time.perf_counter()
                renderer.render([44])
                torch.cuda.synchronize()
                cold = time.perf_counter() - start
                renderer.render([0, 44, 119])
                torch.cuda.synchronize()
                for batch in batches:
                    frames = (np.arange(batch) * 37 + 44) % len(trajectory)
                    durations = []
                    torch.cuda.reset_peak_memory_stats()
                    resident = torch.cuda.memory_allocated()
                    for _ in range(repeats):
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        output = renderer.render(frames)
                        torch.cuda.synchronize()
                        durations.append(time.perf_counter() - start)
                        del output
                    median = statistics.median(durations)
                    rows.append(
                        {
                            "clip": clip,
                            "variant": variant,
                            "height": size,
                            "batch": batch,
                            "modes": ["rgb"],
                            "cpu_selection_and_upload_included": True,
                            "ray_limits": variant == "static-fused",
                            "median_seconds": median,
                            "frames_per_second": batch / median,
                            "seconds": durations,
                            "initialization_seconds": initialization,
                            "first_call_seconds": cold,
                            "resident_mib": resident / 2**20,
                            "peak_allocated_mib": torch.cuda.max_memory_allocated() / 2**20,
                            "cache_builds": renderer.renderer.builds,
                            "retained_storage_bytes": storage_bytes(renderer.renderer),
                        }
                    )
                    print(clip, variant, batch, round(batch / median, 2), "frames/s", flush=True)
    return rows


def profile(root, clip, output, *, size=180):
    """Attribute cached RGB work after warmup; profiling is separate from FPS."""
    with RigidTrajectory(root, clip) as trajectory:
        with RigidSceneRenderer(
            trajectory, size=size, visibility_contract=trajectory.manifest["visibility_contract"]
        ) as renderer:
            frames = (np.arange(16) * 37 + 44) % len(trajectory)
            renderer.render(frames)
            torch.cuda.synchronize()
            with torch.profiler.profile(
                activities=[torch.profiler.ProfilerActivity.CPU, torch.profiler.ProfilerActivity.CUDA],
                record_shapes=True,
                profile_memory=True,
            ) as measured:
                renderer.render(frames)
                torch.cuda.synchronize()
            output.parent.mkdir(parents=True, exist_ok=True)
            averages = measured.key_averages()
            output.write_text(
                averages.table(sort_by="self_device_time_total", row_limit=15)
                + "\n\n"
                + averages.table(sort_by="self_cpu_time_total", row_limit=15)
            )


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recordings", type=Path, required=True)
    p.add_argument("--clips", nargs="+", default=["g01-nominal", "g01-member1", "g03-member1"])
    p.add_argument("--batches", type=int, nargs="+", default=[1, 4, 16])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--height", type=int, default=180)
    p.add_argument("--variants", nargs="+", choices=VARIANTS, default=VARIANTS)
    p.add_argument("--report", type=Path, required=True)
    p.add_argument("--profile", type=Path, default=None)
    a = p.parse_args()
    rows = []
    for clip in a.clips:
        rows.extend(
            benchmark(a.recordings, clip, size=a.height, batches=a.batches, repeats=a.repeats, variants=a.variants)
        )
    result = {
        "gpu": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "cuda": torch.version.cuda,
        "rows": rows,
    }
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(result, indent=2) + "\n")
    if a.profile is not None:
        profile(a.recordings, a.clips[0], a.profile, size=a.height)


if __name__ == "__main__":
    main()
