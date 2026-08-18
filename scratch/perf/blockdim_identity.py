"""Is ``block_dim.render`` really a launch geometry and not arithmetic? Same state, two block sizes.

``robocasa.py`` sets ``block_dim.render = 32`` on the claim that the megakernel is a plain
``wp.launch(dim=(nworld, rays))``, one ray per thread, so the block size changes how threads are
packed onto SMs and changes no float. That claim is worth one probe rather than one sentence.
"""
from __future__ import annotations

import argparse
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="RoboCasaCounterToSink")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--dims", default="32,64,128")
    a = ap.parse_args()

    import mujoco_warp as mjw

    from stable_envs.envs import robocasa as RC

    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    cls = type(env)
    orig = mjw.put_model
    ref = None
    for bd in [int(x) for x in a.dims.split(",")]:
        RC._CTX_CACHE.clear()
        torch.cuda.empty_cache()

        def pm(m, _bd=bd, **kw):
            wm = orig(m, **kw)
            wm.block_dim.render = _bd
            return wm
        mjw.put_model = pm
        try:
            c = RC._context(cls.KITCHEN, cls.CAMERA, a.size, a.nworld)
            c.wp.capture_launch(c.graph)
            c.wp.synchronize()
            f = c.bgra[: a.nworld][..., [2, 1, 0]].clone()
            if ref is None:
                ref = f
                print(f"block_dim={bd:4d}: reference", flush=True)
            else:
                d = (f.to(torch.int16) - ref.to(torch.int16)).abs()
                print(f"block_dim={bd:4d}: max {int(d.max())}/255 on "
                      f"{int((d.amax(-1) > 0).sum())} px", flush=True)
        finally:
            mjw.put_model = orig
            RC._CTX_CACHE.clear()
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
