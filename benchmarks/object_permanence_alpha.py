"""Compare full-capacity, active-layer and cached alpha rendering with upload."""

import argparse
import json
import time
from contextlib import ExitStack
from pathlib import Path

import numpy as np
import torch

from stable_envs.envs.object_permanence import RigidTrajectory
from stable_envs.envs.object_permanence.rendering.rigid_nvdiff import RigidSceneRenderer

from ._common import nonnegative_int, positive_int, positive_ints
from .object_permanence_recordings import compare


def run(args):
    torch.set_num_threads(1)
    report = {
        "device": torch.cuda.get_device_name(),
        "torch": torch.__version__,
        "source": "824baddcdad087810d35638bf8f273b884f74975",
        "height": args.size,
        "chunk_size": args.chunk_size,
        "modes": args.modes.split(","),
        "includes": "CPU frame selection, pose upload, CUDA transforms, coverage, shading and composition",
        "excludes": "construction, JIT, first cache build and caller output allocation; internal scratch allocation is included",
        "points": [],
    }
    modes = tuple(args.modes.split(","))
    for task in args.tasks.split(","):
        with RigidTrajectory(args.recordings, task.lower() + "-nominal") as clip:
            with ExitStack() as stack:
                renderers = {
                    name: stack.enter_context(
                        RigidSceneRenderer(
                            clip,
                            size=args.size,
                            visibility_contract="stable-envs/transparent-pixel-rays-v1",
                            cache_static=name == "cached",
                            max_layers=args.max_layers,
                            chunk_size=args.chunk_size,
                        )
                    )
                    for name in ("baseline", "active_layers", "cached")
                }
                renderers["baseline"].renderer.shade_empty_layers = True
                for batch in args.batches:
                    frames = np.linspace(0, len(clip) - 1, batch).round().astype(np.int64)
                    buffers = {name: r.render_many(frames, modes=modes) for name, r in renderers.items()}
                    assert all(
                        torch.equal(buffers["baseline"][m], buffers[name][m]) for name in renderers for m in modes
                    )
                    times = compare(
                        {
                            name: lambda r=r, name=name: r.render_many(frames, modes=modes, out=buffers[name])
                            for name, r in renderers.items()
                        },
                        warmup=args.warmup,
                        repeats=args.repeats,
                    )
                    point = {
                        "task": task,
                        "batch": batch,
                        "resolution": [renderers["cached"].height, renderers["cached"].width],
                        "cpu_pose_upload_bytes": batch * 28 * len(clip.manifest["dynamic_nodes"]),
                        "outputs_bit_identical": True,
                        "timings": {
                            name: {
                                "median_ms": float(np.median(t)),
                                "fps": batch * 1000 / float(np.median(t)),
                                "samples_ms": t,
                            }
                            for name, t in times.items()
                        },
                    }
                    cache = renderers["cached"].renderer._layer_cache
                    point["cache"] = {
                        "builds": cache.builds,
                        "fallbacks": cache.fallbacks,
                        "static_layers": len(cache._depth),
                        "pixel_bytes": sum(t.numel() * t.element_size() for t in (cache._depth, cache._winner)),
                        "static_faces": len(cache.static),
                        "dynamic_faces": len(cache.dynamic),
                    }
                    point["speedup"] = (
                        point["timings"]["baseline"]["median_ms"] / point["timings"]["cached"]["median_ms"]
                    )
                    point["peak_scratch_bytes"] = {}
                    for name, r in renderers.items():
                        torch.cuda.synchronize()
                        torch.cuda.reset_peak_memory_stats()
                        allocated = torch.cuda.memory_allocated()
                        r.render_many(frames, modes=modes, out=buffers[name])
                        torch.cuda.synchronize()
                        point["peak_scratch_bytes"][name] = torch.cuda.max_memory_allocated() - allocated
                    report["points"].append(point)
                    print(
                        task,
                        batch,
                        {k: round(v["fps"], 1) for k, v in point["timings"].items()},
                        "speedup",
                        round(point["speedup"], 2),
                        flush=True,
                    )
                if args.profile:
                    # CUDA event ranges include time waiting for host launches.
                    profiles = {}
                    for name, r in renderers.items():
                        events = []
                        original = r.renderer._layers

                        def measured(*a, _original=original, **kw):
                            start = torch.cuda.Event(enable_timing=True)
                            end = torch.cuda.Event(enable_timing=True)
                            start.record()
                            result = _original(*a, **kw)
                            end.record()
                            events.append((start, end))
                            return result

                        r.renderer._layers = measured
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        r.render_many(frames, modes=modes, out=buffers[name])
                        torch.cuda.synchronize()
                        profiles[name] = {
                            "visibility_ms": sum(a.elapsed_time(b) for a, b in events),
                            "total_ms": (time.perf_counter() - start) * 1000,
                        }
                        r.renderer._layers = original
                    report.setdefault("profiles", {})[task] = profiles
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, indent=2) + "\n")


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--recordings", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--tasks", default="g90,g36,g54,g61")
    p.add_argument("--batches", type=positive_ints, default=[1, 32])
    p.add_argument("--size", type=positive_int, default=180)
    p.add_argument("--max-layers", type=positive_int, default=16)
    p.add_argument("--chunk-size", type=positive_int, default=4)
    p.add_argument("--modes", default="rgb")
    p.add_argument("--warmup", type=nonnegative_int, default=2)
    p.add_argument("--repeats", type=positive_int, default=7)
    p.add_argument("--profile", action="store_true")
    run(p.parse_args())


if __name__ == "__main__":
    main()
