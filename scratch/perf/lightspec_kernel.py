"""Does the light specialisation change the MEGAKERNEL's own bytes, or only the composite's?

``scratch/perf/lightspec_check.py`` compares what a caller gets, which on Meta-World is a composite
over a MuJoCo plate with a translucent-mover blend on two tasks. If a difference shows up there, the
next question is whether MJWarp's raw frame moved at all -- a kernel that is byte-identical would put
the residual in the composite's own float32 rounding and mean something completely different from a
kernel that is not.

So this compares ``c.bgra`` and ``c.seg`` straight out of the context, on identical ``qpos``, with
the specialisation off and on. No plate, no blend, no channel swap.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="MetaWorldWindowOpen,MetaWorldReach")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    a = ap.parse_args()

    from stable_envs.envs import metaworld as MWE

    out = {}
    for name in a.envs.split(","):
        env = se.make(name, batch=8, size=a.size, device="cuda")
        task = type(env).TASK
        b = MWE._build_bundle(task)
        st = env.sample_states(a.nworld, rng=np.random.default_rng(0))
        frames = {}
        for on in (False, True):
            os.environ["STABLE_ENVS_LIGHTSPEC"] = "1" if on else "0"
            MWE._CTX_CACHE.clear()
            torch.cuda.empty_cache()
            c = MWE._context(task, a.size, a.nworld)
            q = torch.as_tensor(np.asarray(st, np.float32)[:, :b.nq], device="cuda")
            c.wp.to_torch(c.wd.qpos)[:] = q
            torch.cuda.synchronize()
            c.wp.capture_launch(c.graph)
            c.wp.synchronize()
            frames["on" if on else "off"] = (c.bgra.clone(), c.seg.clone())
            MWE._CTX_CACHE.clear()
            torch.cuda.empty_cache()
        os.environ["STABLE_ENVS_LIGHTSPEC"] = "1"
        (a0, s0), (a1, s1) = frames["off"], frames["on"]
        d = (a1.to(torch.int16) - a0.to(torch.int16)).abs()
        out[name] = {
            "rgb_max_abs": int(d.max()), "rgb_bad_px": int((d.amax(-1) > 0).sum()),
            "rgb_px_total": int(d[..., 0].numel()),
            "seg_differs": int((s1 != s0).sum()),
            "identical": bool(int(d.max()) == 0 and int((s1 != s0).sum()) == 0),
        }
        print(f"{name:28s} kernel rgb max {out[name]['rgb_max_abs']} on "
              f"{out[name]['rgb_bad_px']}/{out[name]['rgb_px_total']} px, seg differs on "
              f"{out[name]['seg_differs']} px -> "
              f"{'IDENTICAL' if out[name]['identical'] else 'DIFFERS'}", flush=True)
        del env
        torch.cuda.empty_cache()
    print("JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
