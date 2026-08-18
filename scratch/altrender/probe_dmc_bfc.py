"""Is ``enable_backface_culling=False`` free on the five DM Control TrackCom cells, byte for byte?

The frontier says it is worth 1.12-1.24x with all four fidelity metrics unchanged to four decimals.
Metric equality is weaker than what is actually being claimed, so this asserts the strong form
instead: the two configurations produce the SAME BYTES. If they do, the declaration in
``VISUAL_FIDELITY`` does not change and there is nothing to re-measure — which is the only kind of
"free" this repo accepts.

Checked at BOTH sizes that matter and on four independent state draws, because the declaration is
measured at 84 px (``_AT``) while the storage comparison lives at 256 px, and a claim of identity has
to hold wherever it is going to be relied on.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

from probe_budget_frontier import ctx_cache, env_module, patched

ap = argparse.ArgumentParser()
ap.add_argument("--cells", default="DMCCheetahRun,DMCWalkerWalk,DMCHopperHop,"
                                   "DMCHumanoidStand,DMCSwimmerSwimmer6")
ap.add_argument("--sizes", default="84,256")
ap.add_argument("--draws", type=int, default=4)
ap.add_argument("--states", type=int, default=64)
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--out", default="")
a = ap.parse_args()

import stable_envs as se

FREE = dict(enable_backface_culling=False)
res = {}
print(f"node {os.uname().nodename}  loadavg {os.getloadavg()}\n")
for cell in [c for c in a.cells.split(",") if c]:
    mod = env_module(cell)
    res[cell] = {}
    for size in [int(x) for x in a.sizes.split(",")]:
        worst = 0
        npix = 0
        for k in range(a.draws):
            probe = se.make(cell, batch=1, size=size, device="cuda", seed=k)
            states = probe.sample_states(a.states)
            del probe
            ctx_cache(mod).clear()
            torch.cuda.empty_cache()
            frames = {}
            for tag, ov in (("shipped", {}), ("nobfc", FREE)):
                orig = patched(mod, ov, 1)
                try:
                    env = se.make(cell, batch=1, size=size, device="cuda", seed=k)
                    frames[tag] = env.render(states).clone()
                    del env
                finally:
                    mod.MW.create_render_context = orig
                    ctx_cache(mod).clear()
                    torch.cuda.empty_cache()
            d = (frames["shipped"].to(torch.int16) - frames["nobfc"].to(torch.int16)).abs()
            worst = max(worst, int(d.max()))
            npix += int(d.numel())
            del frames, d
            torch.cuda.empty_cache()
        res[cell][size] = {"max_abs_diff": worst, "bytes_compared": npix,
                           "draws": a.draws, "states": a.states}
        print(f"{cell:22s} {size:4d}px  max |shipped - nobfc| = {worst}/255 over "
              f"{npix:,} bytes ({a.draws} draws x {a.states} states)", flush=True)

    # throughput at the operating point the storage baseline is quoted at
    for tag, ov in (("shipped", {}), ("nobfc", FREE)):
        orig = patched(mod, ov, 1)
        try:
            env = se.make(cell, batch=a.batch, size=256, device="cuda")
            st = torch.as_tensor(env.sample_states(a.batch, rng=np.random.default_rng(0)),
                                 device="cuda", dtype=torch.float32)
            buf = torch.empty((a.batch, 256, 256, 3), dtype=torch.uint8, device="cuda")
            for _ in range(2):
                env.render(st, out=buf)
            torch.cuda.synchronize()
            ts = []
            for _ in range(5):
                t0 = time.perf_counter()
                env.render(st, out=buf)
                torch.cuda.synchronize()
                ts.append(time.perf_counter() - t0)
            res[cell][tag + "_fps"] = a.batch / float(np.median(ts))
            del env, st, buf
        finally:
            mod.MW.create_render_context = orig
            ctx_cache(mod).clear()
            torch.cuda.empty_cache()
    sp = res[cell]["nobfc_fps"] / res[cell]["shipped_fps"]
    res[cell]["speedup"] = sp
    print(f"{cell:22s} 256px b{a.batch}: shipped {res[cell]['shipped_fps']:,.0f} -> nobfc "
          f"{res[cell]['nobfc_fps']:,.0f} f/s  = {sp:.2f}x\n", flush=True)

res["_meta"] = {"node": os.uname().nodename, "loadavg": os.getloadavg(),
                "gpu": torch.cuda.get_device_name(0)}
if a.out:
    with open(a.out, "w") as f:
        json.dump(res, f, indent=1)
    print("wrote", a.out)
