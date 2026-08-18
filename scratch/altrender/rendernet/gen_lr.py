"""Re-render the CHEAP half of an existing split at a different resolution. The official half is kept.

The frontier this project has to draw is fidelity against end-to-end throughput, and the two knobs
that move along it are the render resolution and the size of the network. Sweeping the first one
naively means regenerating everything, and the official frames are 97 % of the generation cost -- 30
f/s through ``mujoco.Renderer`` against 19,000 f/s through MJWarp. They are also the one thing that
does NOT depend on the render resolution: the target is a 256 px official frame of a given state,
whatever the input to the corrector happens to be.

So this reads ``states_*.npy`` and ``hr_*.npy`` out of a split that already exists, renders the cheap
half again at a new resolution, and writes a new split beside it. Same states, same targets, same
order, different input -- which is what makes two points on the resolution axis comparable rather
than merely adjacent.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")
os.environ.setdefault("SDL_VIDEODRIVER", "dummy")

import numpy as np
import torch

import data as D
from probe_budget_frontier import ctx_cache, env_module, patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("env")
    ap.add_argument("--split", required=True)
    ap.add_argument("--src-tag", required=True)
    ap.add_argument("--render-size", type=int, required=True)
    ap.add_argument("--tex-lod", type=int, default=1)
    ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args()

    import stable_envs as se
    mod = env_module(a.env)
    src = D.split_dir(a.env, a.split, a.src_tag)
    dst = D.split_dir(a.env, a.split, f"r{a.render_size}_tex{a.tex_lod}"
                      + ("_sty" if a.src_tag.endswith("_sty") else ""))
    os.makedirs(dst, exist_ok=True)

    shards = sorted(glob.glob(os.path.join(src, "states_*.npy")),
                    key=lambda p: int(p.rsplit("_", 1)[1][:-4]))
    if not shards:
        raise SystemExit(f"no states under {src}")
    ov = dict(enable_backface_culling=False)
    orig = patched(mod, ov, a.tex_lod)
    try:
        env = se.make(a.env, batch=1, size=256, device="cuda", seed=0)
        for sp in shards:
            k = sp.rsplit("_", 1)[1][:-4]
            states = np.load(sp)
            sty = None
            stp = os.path.join(src, f"style_{k}.npy")
            if os.path.exists(stp):
                sty = np.load(stp)
            lr = np.empty((len(states), a.render_size, a.render_size, 3), np.uint8)
            t0 = time.perf_counter()
            for i in range(0, len(states), a.chunk):
                j = min(i + a.chunk, len(states))
                lr[i:j] = env.render(states[i:j], size=a.render_size,
                                     style=None if sty is None else sty[i:j]).cpu().numpy()
            np.save(os.path.join(dst, f"lr_{k}.npy"), lr)
            for kind in ("states", "hr", "style"):
                s = os.path.join(src, f"{kind}_{k}.npy")
                if os.path.exists(s):
                    d = os.path.join(dst, f"{kind}_{k}.npy")
                    if not os.path.exists(d):
                        # A hard link, not a copy: the official half is 2.4 GB per split and the
                        # whole point of this script is that it is the expensive half.
                        os.link(s, d)
            meta = json.load(open(os.path.join(src, f"meta_{k}.json")))
            meta.update(render_size=a.render_size, tex_lod=a.tex_lod, rerendered_from=src)
            with open(os.path.join(dst, f"meta_{k}.json"), "w") as f:
                json.dump(meta, f, indent=1)
            print(f"shard {k}: {len(lr)} at {a.render_size} px in "
                  f"{time.perf_counter() - t0:.1f}s", flush=True)
        del env
    finally:
        mod.MW.create_render_context = orig
        ctx_cache(mod).clear()
        torch.cuda.empty_cache()
    print("wrote", dst, flush=True)


if __name__ == "__main__":
    main()
