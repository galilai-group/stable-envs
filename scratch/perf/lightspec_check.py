"""Is the light specialisation bit-identical to stock mujoco_warp, and what does it buy?

The claim is that deleting an any-hit BVH traversal that never executes cannot change a pixel. That
is an argument; this is the measurement. For each env the SAME states are rendered twice in one
process -- once with ``STABLE_ENVS_LIGHTSPEC=0`` (stock ``mujoco_warp``) and once with it on -- and
the frames are compared byte for byte. Throughput is timed the way ``examples/bench_storage.py``
times it, so the number is comparable to ``examples/results/storage/storage_*.json``.

The env-level render is what is compared, not the megakernel, because that is what a caller gets:
on Meta-World it is a composite over a MuJoCo plate, so a kernel difference that the composite
happens to discard would not count as a win and a difference it amplifies would not be missed.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se

FAMILIES = ("MetaWorld", "HumanoidBench", "Fetch", "RoboCasa")


def module_for(name):
    import importlib
    for f in FAMILIES:
        if name.startswith(f):
            return importlib.import_module({
                "MetaWorld": "stable_envs.envs.metaworld",
                "HumanoidBench": "stable_envs.envs.humanoidbench",
                "Fetch": "stable_envs.envs.fetch",
                "RoboCasa": "stable_envs.envs.robocasa"}[f])
    raise SystemExit(f"unknown family for {name}")


def timed(fn, reps, warm=5):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", required=True)
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--reps", type=int, default=11)
    ap.add_argument("--n-pixels", type=int, default=64, help="states for the byte comparison")
    ap.add_argument("--control", action="store_true",
                    help="run BOTH halves on stock mujoco_warp. Any difference this reports is a "
                         "run-to-run flake in the comparison, not the specialisation, and it is the "
                         "only way to read a 1/255 result honestly.")
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    from stable_envs.render import mjwarp_lightspec as LS
    print(f"# mjwarp_lightspec pinned to mujoco-warp {LS.UPSTREAM_VERSION}", flush=True)

    out = {"device": torch.cuda.get_device_name(0), "size": a.size, "batch": a.batch,
           "upstream": LS.UPSTREAM_VERSION, "envs": {}}
    for name in a.envs.split(","):
        mod = module_for(name)
        row = {}
        frames = {}
        for on in (False, True):
            os.environ["STABLE_ENVS_LIGHTSPEC"] = "1" if (on and not a.control) else "0"
            mod._CTX_CACHE.clear()
            for cache in ("_PLATE_CACHE", "_RASTER_CACHE"):
                if hasattr(mod, cache):
                    getattr(mod, cache).clear()
            torch.cuda.empty_cache()
            key = "lightspec" if on else "stock"
            try:
                env = se.make(name, batch=a.batch, size=a.size, device="cuda")
                st = env.sample_states(a.n_pixels, rng=np.random.default_rng(0))
                frames[key] = env.render(st).clone()
                pool = env.sample_states(min(512, a.batch))
                big = torch.as_tensor(np.resize(pool, (a.batch, pool.shape[1])), device="cuda")
                buf = torch.empty((a.batch, a.size, a.size, 3), dtype=torch.uint8, device="cuda")
                dt = timed(lambda: env.render(big, out=buf), a.reps)
                row[key] = {"fps": a.batch / dt, "ratio_vs_8worker_jpeg": a.batch / dt / 16889.0}
                del env, big, buf
            except Exception as e:
                import traceback
                traceback.print_exc()
                row[key] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                mod._CTX_CACHE.clear()
                torch.cuda.empty_cache()
        os.environ["STABLE_ENVS_LIGHTSPEC"] = "1"
        if "stock" in frames and "lightspec" in frames:
            d = (frames["lightspec"].to(torch.int16) - frames["stock"].to(torch.int16)).abs()
            row["pixels"] = {
                "max_abs": int(d.max()), "mean_abs": float(d.to(torch.float32).mean()),
                "bad_px": int((d.amax(-1) > 0).sum()), "n_states": a.n_pixels,
                "identical": bool(int(d.max()) == 0),
            }
        if "fps" in row.get("stock", {}) and "fps" in row.get("lightspec", {}):
            row["speedup"] = row["lightspec"]["fps"] / row["stock"]["fps"]
        out["envs"][name] = row
        p = row.get("pixels", {})
        print(f"{name:34s} stock {row.get('stock', {}).get('fps', float('nan')):9,.0f} -> "
              f"lightspec {row.get('lightspec', {}).get('fps', float('nan')):9,.0f} f/s  "
              f"x{row.get('speedup', float('nan')):4.2f}   "
              f"pixels {'IDENTICAL' if p.get('identical') else 'DIFFER max ' + str(p.get('max_abs'))}"
              f" over {p.get('n_states')} states", flush=True)
    print("JSON " + json.dumps(out), flush=True)
    if a.out:
        with open(a.out, "w") as f:
            json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
