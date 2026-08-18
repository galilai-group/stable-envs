"""Does the CAPTURED graph carry ``gl_rounding``'s kernel, or the truncating one it was meant to replace?

``mujoco_warp.render`` DEFINES its megakernel inside the call -- ``@wp.kernel(module="unique")`` on a
closure -- so the kernel's source is generated per call, from whatever ``pack_rgba_to_uint32`` is bound
at that moment. Every family in this repo builds its context like this::

    with MW.gl_rounding():
        rc = create_render_context(...)
        _frame(c)                 # generates + compiles the ROUNDING kernel
    with wp.ScopedCapture() as cap:
        _frame(c)                 # generates a SECOND kernel, outside the patch

If the second generation is a different module, the graph every later render replays is the truncating
one and the patch reaches nothing. The signed difference between the two frames decides it: a
quantisation swap is ``{0, -1}`` and nothing else.
"""
from __future__ import annotations

import argparse
import collections
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se
import stable_envs.render.mjwarp as MW


def module_for(name):
    import importlib
    for f, m in (("MetaWorld", "metaworld"), ("HumanoidBench", "humanoidbench"),
                 ("Fetch", "fetch"), ("RoboCasa", "robocasa")):
        if name.startswith(f):
            return importlib.import_module("stable_envs.envs." + m)
    raise SystemExit(name)


def build(mod, env, res, nworld):
    cls = type(env)
    if hasattr(cls, "ROBOT"):
        return mod._context(cls.ROBOT, cls.TASK, res, nworld)
    if hasattr(cls, "KITCHEN"):
        return mod._context(cls.KITCHEN, cls.CAMERA, res, nworld)
    if hasattr(cls, "SCENE"):
        return mod._context(cls.SCENE, res, nworld)
    return mod._context(cls.TASK, res, nworld)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="FetchReach")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=8)
    a = ap.parse_args()

    mod = module_for(a.env)
    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    mod._CTX_CACHE.clear()
    c = build(mod, env, a.size, a.nworld)

    def frame_direct(patched: bool):
        if patched:
            with MW.gl_rounding():
                mod._frame(c)
                c.wp.synchronize()
        else:
            mod._frame(c)
            c.wp.synchronize()
        return c.bgra[: a.nworld][..., [2, 1, 0]].clone()

    def frame_graph():
        c.wp.capture_launch(c.graph)
        c.wp.synchronize()
        return c.bgra[: a.nworld][..., [2, 1, 0]].clone()

    g = frame_graph()
    rounded = frame_direct(True)
    trunc = frame_direct(False)

    def signed(x, y, label):
        d = (x.to(torch.int16) - y.to(torch.int16)).reshape(-1)
        v, ct = torch.unique(d, return_counts=True)
        tot = int(d.numel())
        hist = {int(a_): round(float(b_) / tot, 6) for a_, b_ in zip(v.tolist(), ct.tolist())}
        print(f"{label:34s} maxabs {int(d.abs().max())}  signed {hist}", flush=True)
        return hist

    print(f"# {a.env} {a.size}px nworld={a.nworld}", flush=True)
    h1 = signed(g, rounded, "captured graph - ROUNDED direct")
    h2 = signed(g, trunc, "captured graph - TRUNCATING direct")
    h3 = signed(rounded, trunc, "ROUNDED direct - TRUNCATING direct")
    print(json.dumps({"env": a.env, "graph_vs_rounded": h1, "graph_vs_trunc": h2,
                      "rounded_vs_trunc": h3}), flush=True)


if __name__ == "__main__":
    main()
