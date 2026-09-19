"""Throughput of the generic scene rasterizer, per env in the registry.

The number to keep honest here is what generality costs. The PushT-specific kernel reached 14.1 M
frames/s by baking the static layers into the background and knowing there were exactly two quads and
one disc. This kernel walks a runtime primitive list and re-evaluates every primitive per subpixel, so
it is expected to be slower — the question is by how much, and whether it is still far enough past disk
bandwidth that storing frames remains pointless.

``--ss native`` (the default) leaves every env at its own ``default_supersample``, which is what
``se.make(name)`` gives a user. Forcing a number instead measures a picture 44 of the registered
classes do not produce — they declare 2, because their reference is MSAA'd — and that substitution was
live in this repo's published decode table until it was found by two of its own benchmarks disagreeing
on the same env.
"""

import argparse
import statistics
import time

import torch

import stable_envs as se

ap = argparse.ArgumentParser()
ap.add_argument("--sizes", default="96,224")
ap.add_argument("--batches", default="256,1024,4096,16384")
ap.add_argument(
    "--ss", default="native", help="'native' uses each env's own default_supersample; a number forces that value"
)
ap.add_argument("--reps", type=int, default=10)
a = ap.parse_args()
SS = None if str(a.ss).lower() in ("native", "default", "env", "none") else int(a.ss)

print(f"{torch.cuda.get_device_name(0)}   supersample={a.ss}\n")
for name in sorted(se.registry.REGISTRY):
    for size in [int(s) for s in a.sizes.split(",")]:
        print(f"--- {name} @ {size}px ---")
        print(f"  {'batch':>7} {'ms/frame':>10} {'frames/s':>13} {'GB/s':>8} {'peak GB':>9} {'ss':>3}")
        for B in [int(b) for b in a.batches.split(",")]:
            try:
                kw = {} if SS is None else {"supersample": SS}
                env = se.make(name, num_envs=B, size=size, device="cuda", **kw)
                st = torch.as_tensor(env.sample_states(B), device="cuda")
                out = torch.empty((B, size, size, 3), dtype=torch.uint8, device="cuda")
                for _ in range(3):
                    env.render(st, out=out)
                torch.cuda.synchronize()
                torch.cuda.reset_peak_memory_stats()
                ts = []
                for _ in range(a.reps):
                    torch.cuda.synchronize()
                    t0 = time.perf_counter()
                    env.render(st, out=out)
                    torch.cuda.synchronize()
                    ts.append(time.perf_counter() - t0)
                dt = statistics.median(ts)
                gb = B * size * size * 3 / 1e9
                print(
                    f"  {B:>7} {dt / B * 1e3:10.5f} {B / dt:13,.0f} {gb / dt:8.1f} "
                    f"{torch.cuda.max_memory_allocated() / 1e9:9.2f} {env.ss:>3}"
                )
                del env, st, out
                torch.cuda.empty_cache()
            except torch.OutOfMemoryError:
                print(f"  {B:>7} {'OOM':>10}")
                torch.cuda.empty_cache()
