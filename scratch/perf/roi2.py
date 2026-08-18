"""What fraction of the frame does the DYNAMIC half of a split actually decide, and how big is its box?

``scratch/perf/split_static.py`` says a static/dynamic split is worth 1.5x on a RoboCasa room camera
and 3.1x on its wrist camera. The next question is whether the dynamic pass has to cover the whole
frame. It does not have to cover more than the pixels the composite takes from it -- the movers, plus
(where a light casts) the static pixels those movers shadow -- so this measures that mask, its
per-state bounding box, and the UNION of the boxes over a chunk, which is the box a launch with one
shared ray table would have to use.

``scratch/perf/roi.py`` asked the same question for Meta-World through the env's own masks and got
21% mask / 89% box; this is the general version, driven off the segmentation and depth buffers, over
the states the env really reaches rather than the context's construction pose.
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
import stable_envs.render.mjwarp as MW

import sys; sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from split_static import (build, family, module_for, model_of, moving_geoms)  # noqa: E402


def bbox_frac(mask: torch.Tensor, size: int) -> float:
    rows = mask.any(-1).nonzero()
    cols = mask.any(-2).nonzero()
    if not len(rows) or not len(cols):
        return 0.0
    h = int(rows[-1]) - int(rows[0]) + 1
    w = int(cols[-1]) - int(cols[0]) + 1
    return h * w / float(size * size)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="RoboCasaCounterToSink")
    ap.add_argument("--size", type=int, default=256)
    ap.add_argument("--nworld", type=int, default=64)
    a = ap.parse_args()

    import warp as wp

    mod = module_for(a.env)
    env = se.make(a.env, batch=a.nworld, size=a.size, device="cuda")
    m = model_of(mod, env)
    fam = family(a.env)
    mv, states = moving_geoms(mod, env, m, k=a.nworld)
    vis_groups = set(int(g) for g in mod.VISIBLE_GROUPS)
    mv_vis = np.array([g for g in mv if int(m.geom_group[g]) in vis_groups], np.int64)

    orig_crc = MW.create_render_context

    def crc(mm, **kw):
        kw["render_seg"] = True
        kw["render_depth"] = True
        return orig_crc(mm, **kw)
    MW.create_render_context = crc
    mod._CTX_CACHE.clear()
    try:
        c = build(mod, env, a.size, a.nworld)
    finally:
        MW.create_render_context = orig_crc

    # Real states into every world. The construction pose puts all 64 worlds in one place, which is
    # exactly the case a union box flatters.
    q = wp.to_torch(c.wd.qpos)
    if fam == "RoboCasa":
        s = mod._scene(type(env).KITCHEN)
        full = np.repeat(s.pinned[None], a.nworld, 0)
        full[:, env._live] = states[:a.nworld]
        q[:] = torch.as_tensor(full, device=q.device, dtype=q.dtype)
    elif fam == "HumanoidBench":
        q[:] = torch.as_tensor(np.asarray(states[:a.nworld], np.float32)[:, :int(m.nq)],
                               device=q.device, dtype=q.dtype)
    torch.cuda.synchronize()
    c.wp.capture_launch(c.graph)
    c.wp.synchronize()

    seg = c.seg[:a.nworld] if hasattr(c, "seg") else \
        wp.to_torch(c.rc.seg_data).reshape(a.nworld, a.size, a.size, 2)[..., 0]
    rgb_full = c.bgra[:a.nworld][..., [2, 1, 0]].clone()
    seg = seg.clone()

    lut = torch.zeros(int(m.ngeom) + 1, dtype=torch.bool, device=seg.device)
    lut[torch.as_tensor(mv_vis + 1, device=seg.device, dtype=torch.long)] = True
    mover = lut[(seg + 1).clamp(min=0).long()]

    # Movers off, by re-pointing their BVH slots at a static geom -- metaworld.py's own trick.
    ids = wp.to_torch(c.rc.enabled_geom_ids)
    ids0 = ids.clone()
    mset = set(int(g) for g in mv_vis)
    slots = [i for i, g in enumerate(ids0.tolist()) if int(g) in mset]
    rest = [int(g) for g in ids0.tolist() if int(g) not in mset]
    shadow = torch.zeros_like(mover)
    if slots and rest:
        ids[torch.as_tensor(slots, device=ids.device, dtype=torch.long)] = rest[0]
        torch.cuda.synchronize()
        c.wp.capture_launch(c.graph)
        c.wp.synchronize()
        off = c.bgra[:a.nworld][..., [2, 1, 0]].clone()
        ids.copy_(ids0)
        torch.cuda.synchronize()
        shadow = (rgb_full != off).any(-1) & ~mover

    used = mover | shadow
    per = [bbox_frac(used[i], a.size) for i in range(a.nworld)]
    out = {
        "env": a.env, "size": a.size, "nworld": a.nworld,
        "n_movers": int(len(mv_vis)),
        "mover_frac": float(mover.float().mean()),
        "shadow_frac": float(shadow.float().mean()),
        "used_frac": float(used.float().mean()),
        "bbox_frac_mean": float(np.mean(per)), "bbox_frac_max": float(np.max(per)),
        "union_bbox_frac": bbox_frac(used.any(0), a.size),
        "mover_bbox_frac_mean": float(np.mean([bbox_frac(mover[i], a.size)
                                               for i in range(a.nworld)])),
        "mover_union_bbox_frac": bbox_frac(mover.any(0), a.size),
    }
    print(json.dumps(out, indent=1), flush=True)
    print("JSON " + json.dumps(out), flush=True)


if __name__ == "__main__":
    main()
