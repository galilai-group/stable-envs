"""Is the culled-vs-full difference culling, or is it the same expression compiled twice?

Two specializations of the SAME source that both draw every primitive are compared: only the tile
shape differs, so if they disagree on the same pixels the difference cannot be the bounding box.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import stable_envs as se
from stable_envs.render import raster2d as R


def diff(a, b):
    d = (a.to(torch.int16) - b.to(torch.int16)).abs()
    return int((d > 0).any(-1).sum()), int(d.max())


def main(name, B=1024):
    env = se.make(name, batch=1, device="cuda")
    st = torch.as_tensor(env.sample_states(B, rng=np.random.default_rng(0)), device="cuda")
    sc = env.scene(st)
    hw = (env.H, env.W)
    full = R.render_triton(sc, hw, env.ss, cull=False).clone()
    culled = R.render_triton(sc, hw, env.ss, cull=True).clone()
    n1, m1 = diff(full, culled)
    # same kernel, culling ON in both, but a different tile shape -> a different compilation
    old = R._CULL_TILE
    try:
        R._CULL_TILE = (8, 128, 4)
        alt = R.render_triton(sc, hw, env.ss, cull=True).clone()
    finally:
        R._CULL_TILE = old
    n2, m2 = diff(culled, alt)
    # and the unculled path against ITSELF at a second block size
    other = R.render_triton(sc, hw, env.ss, cull=False, block=256).clone()
    n3, m3 = diff(full, other)
    print(f"{name:14s} {hw} N={sc.n_prims} B={B} px={B*hw[0]*hw[1]}", flush=True)
    print(f"   full   vs culled        {n1:4d} px  max {m1}", flush=True)
    print(f"   culled vs culled(8x128) {n2:4d} px  max {m2}", flush=True)
    print(f"   full   vs full(blk 256) {n3:4d} px  max {m3}", flush=True)


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), flush=True)
    for n in (sys.argv[1:] or ["Intersection", "Racetrack", "Roundabout"]):
        main(n)
