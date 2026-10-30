"""Measure original/fused transforms including CPU selection and pose upload.

One renderer is retained at a time. Both variants use the same static visibility
cache, precision, frames and outputs; only geometry preparation differs. Peak
and resident numbers are PyTorch allocations, not total driver memory.
"""

import argparse
import gc
import json
import time
from pathlib import Path

import numpy as np
import torch

from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer
from stable_envs.envs.object_permanence.rendering.visibility import CONTRACT

from ._common import configure_threads, measure, positive_ints


def run(root, *, tasks, batches, size, repeats):
    configure_threads(1)
    rows = []
    for task in tasks:
        with RigidTrajectory(root, task.lower() + "-nominal") as clip:
            for batch in batches:
                frames = np.linspace(0, len(clip) - 1, batch).round().astype(np.int64)
                digest = None
                for fused in (False, True):
                    gc.collect()
                    torch.cuda.empty_cache()
                    with RigidSceneRenderer(clip, size=size, visibility_contract=CONTRACT) as renderer:
                        renderer.renderer.fuse_geometry = fused
                        output = renderer.render(frames)
                        pixels = output.cpu().numpy()
                        if digest is None:
                            digest = pixels
                        else:
                            np.testing.assert_array_equal(digest, pixels)
                        del output
                        gc.collect()
                        torch.cuda.synchronize()
                        resident = torch.cuda.memory_allocated()
                        torch.cuda.reset_peak_memory_stats()

                        def draw():
                            return renderer.render(frames)

                        timing = measure(draw, device=renderer.device, warmup=3, repeats=repeats)
                        row = {
                            "task": task.upper(),
                            "batch": batch,
                            "height": size,
                            "variant": "fused" if fused else "reference",
                            "modes": ["rgb"],
                            "cpu_selection_and_upload_included": True,
                            "timing": timing,
                            "resident_mib": resident / 2**20,
                            "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
                            "static_builds": renderer.renderer._ray_coverage.builds,
                            "outputs_bit_identical": True,
                        }
                        # measure() reports latency in milliseconds, including all
                        # caller-visible work and output allocation.
                        row["fps"] = batch * 1000 / timing["median_ms"]
                        rows.append(row)
                        print(
                            task, batch, row["variant"], row["resident_mib"], row["peak_mib"], row["fps"], flush=True
                        )
    return {"gpu": torch.cuda.get_device_name(), "torch": torch.__version__, "cuda": torch.version.cuda, "rows": rows}


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recordings", type=Path, required=True)
    p.add_argument("--tasks", default="g98,g132")
    p.add_argument("--batches", type=positive_ints, default=[1, 16])
    p.add_argument("--height", type=int, default=180)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--report", type=Path, required=True)
    a = p.parse_args()
    started = time.perf_counter()
    report = run(a.recordings, tasks=a.tasks.split(","), batches=a.batches, size=a.height, repeats=a.repeats)
    report["elapsed_seconds"] = time.perf_counter() - started
    a.report.parent.mkdir(parents=True, exist_ok=True)
    a.report.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
