"""Can the movers' SHADOW be put back into the plate exactly, without rendering the state in MuJoCo?

``diag_residual.py`` says every non-mover disagreement on reach-v3 is a pixel the movers reach in the
official frame and the plate cannot contain: hiding a geom hides its shadow, and one of Meta-World's
three lights casts. It is 8.6% of the frame at mean 18.9/255, which is a visible missing shadow rather
than a quantisation residual.

The proposal, all of it MuJoCo arithmetic on both branches:

  * bake a SECOND plate per episode with the CASTING light's diffuse and specular zeroed. A MuJoCo
    pixel is ``ambient + shadow * (diffuse + specular)`` per light, so that second bake is exactly what
    MuJoCo draws where the casting light is fully occluded -- not an approximation of it.
  * ask MJWarp which static pixels the movers shadow, by rendering the same scene twice: once as
    usual, and once with the movers removed from the BVH's ``enabled_geom_ids`` (each mover slot
    re-pointed at a static geom, so the tree keeps its shape and a static surface is simply listed
    twice). The movers-off frame depends on the episode and the appearance and NOT on qpos, so it is
    cached with the plate; the movers-on frame is the one the composite already renders.
  * composite ``mover ? warp : (shadowed ? dark : bright)``.

The error left is then the SHADOW BOUNDARY -- MuJoCo filters a 4096 shadow map, MJWarp casts one ray
per light -- rather than the whole shadow. This measures whether that is true.
"""
from __future__ import annotations

import json
import os
import sys

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))


def run(task: str, size: int, n: int) -> dict:
    import mujoco
    import torch

    import stable_envs as se
    from stable_envs.envs.metaworld import (HIDDEN_GROUP, _apply_state, _build_bundle, _class_name,
                                            _context, _raster)
    from stable_envs.testing import conformance as C

    cls = se.REGISTRY[_class_name(task)]
    env = cls(batch=8, size=size, device="cuda")
    b = _build_bundle(task)
    m, d = b.model, b.data
    states = env.sample_states(n, rng=np.random.default_rng(0))
    ref = C.reference_env(cls, size)
    official = cls.reference_render(ref, states).astype(np.int16)

    cast = np.flatnonzero(np.asarray(m.light_castshadow, np.int64) != 0)
    rec = {"task": task, "size": size, "n": n, "nlight": int(m.nlight),
           "casting_lights": cast.tolist()}

    c = _context(task, size, 8)
    wp = c.wp
    ids = wp.to_torch(c.rc.enabled_geom_ids)
    ids0 = ids.clone()
    mover_set = set(int(g) for g in b.movers)
    local_movers = [i for i in range(len(ids0)) if int(ids0[i]) in mover_set]
    substitute = int([int(g) for g in ids0.tolist() if int(g) not in mover_set][0])
    rec["bvh_geoms"] = int(len(ids0))
    rec["bvh_mover_slots"] = len(local_movers)

    r = _raster(task, size, m)

    def bake(state_row, dark: bool) -> np.ndarray:
        _apply_state(b, m, d, np.asarray(state_row, np.float64), forward=False)
        keep = np.array(m.light_diffuse, copy=True), np.array(m.light_specular, copy=True)
        if dark:
            m.light_diffuse[cast] = 0.0
            m.light_specular[cast] = 0.0
        g0, s0 = np.array(m.geom_group, copy=True), np.array(m.site_group, copy=True)
        m.geom_group[b.movers] = HIDDEN_GROUP
        if len(b.mover_sites):
            m.site_group[b.mover_sites] = HIDDEN_GROUP
        mujoco.mj_forward(m, d)
        r.update_scene(d, camera=b.cam)
        img = r.render().copy()
        m.geom_group[:], m.site_group[:] = g0, s0
        m.light_diffuse[:], m.light_specular[:] = keep
        return img

    out = {"composite": [], "shadow_fixed": [], "mask_iou": []}
    # ONE batch of exactly ``n`` states, so the buffers read below belong to the context this render
    # used: ``render`` picks its context by batch size, and reading ``c.seg`` after a render of a
    # different width is reading a stale frame from a context nobody launched.
    ours_all = env.render(states[:n]).cpu().numpy().astype(np.int16)
    mv_all = torch.isin(c.seg[:n], torch.as_tensor(b.movers, device="cuda").to(c.seg.dtype))
    on_all = c.bgra[:n][..., [2, 1, 0]].clone()
    ids[local_movers] = substitute
    torch.cuda.synchronize()
    wp.capture_launch(c.graph)
    wp.synchronize()
    off_all = c.bgra[:n][..., [2, 1, 0]].clone()
    ids.copy_(ids0)
    torch.cuda.synchronize()
    shadow_all = ((on_all.to(torch.int16) - off_all.to(torch.int16)).abs().amax(-1) > 0) & (~mv_all)

    for i in range(n):
        st = states[i]
        canon = np.concatenate([b.qpos0[:b.nq], st[b.nq:]]).astype(np.float32)
        bright = torch.as_tensor(bake(canon, False), device="cuda")
        dark = torch.as_tensor(bake(canon, True), device="cuda")
        fixed = torch.where(mv_all[i].unsqueeze(-1), on_all[i],
                            torch.where(shadow_all[i].unsqueeze(-1), dark, bright))
        fixed = fixed.cpu().numpy().astype(np.int16)

        # MuJoCo's OWN shadow mask, for the IoU: the official frame against the bright plate
        truth = (np.abs(official[i] - bright.cpu().numpy().astype(np.int16)).max(-1) > 0)
        truth = truth & (~mv_all[i].cpu().numpy())
        got = shadow_all[i].cpu().numpy()
        inter = float((truth & got).sum())
        out["mask_iou"].append(round(inter / max(float((truth | got).sum()), 1.0), 4))
        out["composite"].append(C.pixel_agreement(ours_all[i][None].astype(np.uint8),
                                                  official[i][None].astype(np.uint8)))
        out["shadow_fixed"].append(C.pixel_agreement(fixed[None].astype(np.uint8),
                                                     official[i][None].astype(np.uint8)))

    def agg(rows, k):
        return round(float(np.mean([x[k] for x in rows])), 6)

    rec["mask_iou"] = round(float(np.mean(out["mask_iou"])), 4)
    for name in ("composite", "shadow_fixed"):
        rec[name] = {k: agg(out[name], k) for k in
                     ("exact_frac", "mean_abs", "within8_frac", "interior_exact_frac")}
    return rec


if __name__ == "__main__":
    task = sys.argv[1] if len(sys.argv) > 1 else "reach-v3"
    size = int(sys.argv[2]) if len(sys.argv) > 2 else 480
    n = int(sys.argv[3]) if len(sys.argv) > 3 else 4
    print("RESULT " + json.dumps(run(task, size, n)), flush=True)
