"""Two render contexts built from the SAME model at the SAME settings do not draw the same frame.

``docs/research/humanoidbench.md`` §5.2 records 2-3 of ~660 tests failing per suite run, always
``test_determinism`` or ``test_frame_i_is_independent_of_batch_size``, by 1-2/255, on a different cell
each run -- and the whole family is gated behind ``STABLE_ENVS_HUMANOIDBENCH=1`` because of it. This
isolates the mechanism: same process, same model, same qpos, two contexts, and then the same question
asked of every stage between them, so the answer is a named array rather than "the GPU".
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


def frame(c, n, qpos=None):
    if qpos is not None:
        t = c.wp.to_torch(c.wd.qpos)
        t[:] = qpos[: t.shape[0]]
    c.wp.synchronize()
    c.wp.capture_launch(c.graph)
    c.wp.synchronize()
    return c.bgra[:n][..., [2, 1, 0]].clone()


def diff(a, b):
    d = (a.to(torch.int16) - b.to(torch.int16)).abs()
    return {"max": int(d.max()), "npix": int((d.any(-1)).sum()),
            "frac": float((d.any(-1)).float().mean())}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HumanoidBenchH1Walk")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    ap.add_argument("--builds", type=int, default=4)
    a = ap.parse_args()

    mod = module_for(a.env)
    env = se.make(a.env, batch=8, size=a.size, device="cuda")
    st = env.sample_states(a.nworld)
    q = torch.as_tensor(np.asarray(st, np.float32)[:, :env.state_dim], device="cuda")

    out = {"env": a.env, "nworld": a.nworld}
    ctxs, frames, bvhs = [], [], []
    for i in range(a.builds):
        mod._CTX_CACHE.clear()
        c = build(mod, env, a.size, a.nworld)
        nq = c.wp.to_torch(c.wd.qpos).shape[1]
        f1 = frame(c, a.nworld, q[:, :nq])
        f2 = frame(c, a.nworld, q[:, :nq])
        same = diff(f1, f2)
        bv = {k: c.wp.to_torch(getattr(c.rc, k)).clone()
              for k in ("lower", "upper", "group", "group_root") if hasattr(c.rc, k)}
        bv["enabled_geom_ids"] = c.wp.to_torch(c.rc.enabled_geom_ids).clone()
        ctxs.append(c)
        frames.append(f1)
        bvhs.append(bv)
        print(f"build {i}: same-context repeat {same}", flush=True)
        out[f"repeat_{i}"] = same

    for i in range(1, a.builds):
        d = diff(frames[0], frames[i])
        arr = {k: (float((bvhs[0][k].float() - bvhs[i][k].float()).abs().max())
                   if bvhs[0][k].shape == bvhs[i][k].shape else -1.0)
               for k in bvhs[0]}
        print(f"build 0 vs {i}: frame {d}  arrays {arr}", flush=True)
        out[f"build0_vs_{i}"] = {"frame": d, "arrays": arr}

    # Is the BVH itself the thing that moved? ``wp.Bvh`` exposes its node bounds.
    try:
        b0, b1 = ctxs[0].rc.bvh, ctxs[1].rc.bvh
        for attr in ("lowers", "uppers", "node_lowers", "node_uppers", "primitive_indices"):
            if hasattr(b0, attr):
                x = ctxs[0].wp.to_torch(getattr(b0, attr)).float()
                y = ctxs[1].wp.to_torch(getattr(b1, attr)).float()
                same_shape = x.shape == y.shape
                m = float((x - y).abs().max()) if same_shape else -1.0
                print(f"  bvh.{attr}: shape {tuple(x.shape)} vs {tuple(y.shape)} maxdiff {m}",
                      flush=True)
                out[f"bvh_{attr}"] = m
    except Exception as e:
        print(f"  bvh introspection: {type(e).__name__}: {e}", flush=True)

    print(json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
