"""Exercise the packaged env the way the conformance suite will, and report AGENT-PIXEL parity.

Whole-frame agreement on a mostly-static plate is not the number worth quoting, so the agent's own
pixels are scored separately, using MuJoCo's segmentation buffer to say which pixels those are rather
than inferring them from our own renderer.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np
import torch

OUT = "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/ogbench/out"


def agent_mask(ref_env, states, agent_ids, size):
    import mujoco
    u = ref_env.unwrapped
    m = u.model
    m.vis.quality.offsamples = 0
    r = mujoco.Renderer(m, width=size, height=size)
    r.enable_segmentation_rendering()
    out = []
    for st in np.asarray(states, np.float64):
        u.set_state(st, np.zeros(m.nv))
        r.update_scene(u.data, camera=u.custom_camera)
        seg = r.render()
        gid = np.where(seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM), seg[..., 0], -1)
        out.append(np.isin(gid, agent_ids))
    r.disable_segmentation_rendering()
    m.vis.quality.offsamples = 4
    return np.stack(out)


def erode(msk):
    e = msk.copy()
    e[:, 1:] &= msk[:, :-1]
    e[:, :-1] &= msk[:, 1:]
    e[:, :, 1:] &= msk[:, :, :-1]
    e[:, :, :-1] &= msk[:, :, 1:]
    return e


def report(name, n=32, ss=None):
    import stable_envs as se
    from stable_envs.testing import conformance as C
    e = se.make(name, batch=16, device="cuda", supersample=ss) if ss else \
        se.make(name, batch=16, device="cuda")
    cls = type(e)
    res = dict(env=name, state_dim=e.state_dim, size=e.size, supersample=e.ss)

    st = e.sample_states(n)
    ours = e.render(st).cpu().numpy()
    ref = cls.reference_env(size=e.size)
    official = cls.reference_render(ref, st)
    res["whole_frame"] = {k: (round(v, 6) if isinstance(v, float) else v)
                          for k, v in C.pixel_agreement(ours, official).items()}

    sc = e._scene()
    am = agent_mask(ref, st, sc.agent_ids, e.size)
    d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
    ai = erode(am)
    for nm, k in (("agent_all", am), ("agent_interior", ai)):
        if k.sum() == 0:
            res[nm] = None
            continue
        res[nm] = dict(n=int(k.sum()), per_frame=round(float(k.sum()) / len(st), 1),
                       exact=round(float((d[k] == 0).mean()), 5),
                       le1=round(float((d[k] <= 1).mean()), 5),
                       le2=round(float((d[k] <= 2).mean()), 5),
                       le8=round(float((d[k] <= 8).mean()), 5),
                       mean=round(float(d[k].mean()), 4), max=int(d[k].max()))
    # everything that is NOT the agent: the baked plate, which should be exact
    nb = ~am
    res["non_agent"] = dict(n=int(nb.sum()), exact=round(float((d[nb] == 0).mean()), 6),
                            mean=round(float(d[nb].mean()), 5), max=int(d[nb].max()))

    res["roundtrip"] = C.check_state_roundtrip(e)
    res["dead_dims"] = C.check_state_dims_matter(e)["dead_dims"]
    res["batch_invariance"] = C.check_batch_invariance(e)
    res["determinism"] = C.check_determinism(e)
    res["inert_variations"] = C.check_variations_do_something(e)["inert_variations"]

    # throughput: renderer only (poses reused) and end to end (qpos -> frame)
    for B in (256, 1024):
        s = torch.as_tensor(e.sample_states(B), device="cuda")
        for _ in range(2):
            e.render(s)
        torch.cuda.synchronize()
        t0 = time.time()
        for _ in range(5):
            e.render(s)
        torch.cuda.synchronize()
        res[f"fps_end_to_end_B{B}"] = round(B * 5 / (time.time() - t0), 1)
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    log = open(os.path.join(OUT, "smoke_pkg.log"), "w")
    names = sys.argv[1:] or ["PointmazeMediumNavigate", "PointmazeLargeStitch",
                             "AntmazeLargeNavigate", "HumanoidmazeLargeNavigate"]
    allr = []
    for nm in names:
        for ss in (1, 2, 4):
            try:
                r = report(nm, ss=ss)
            except Exception:  # noqa: BLE001
                import traceback
                log.write(f"FAIL {nm} ss={ss}\n{traceback.format_exc()}\n")
                log.flush()
                continue
            allr.append(r)
            log.write(json.dumps(r, indent=1) + "\n")
            log.flush()
    log.close()
    with open(os.path.join(OUT, "smoke_pkg.json"), "w") as f:
        json.dump(allr, f, indent=1)


if __name__ == "__main__":
    main()
