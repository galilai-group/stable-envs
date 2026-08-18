"""Where the shipped residual lives, through the ENV as registered, and a clean throughput baseline.

``parity_note`` has to name a mechanism, so the residual is split by the geom class each pixel belongs
to using MuJoCo's own segmentation buffer on the reference model — not by inspection of a picture.
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

import numpy as np
import torch
import mujoco

import stable_envs as se
from stable_envs.testing import conformance as C

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/fetch"
NAMES = ("FetchReach", "FetchPush", "FetchSlide", "FetchPickAndPlace")
N = 32
KEEP = []


def classify(m, seg):
    cls = np.full(seg.shape, "background", dtype=object)
    for g in range(m.ngeom):
        nm = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or f"geom{g}"
        t = int(m.geom_type[g])
        if nm == "floor0":
            c = "floor"
        elif nm == "object0":
            c = "object"
        elif int(m.body_mocapid[m.geom_bodyid[g]]) >= 0:
            c = "mocap_marker"
        elif nm.startswith("robot0:") and t == 7:
            c = "robot_mesh"
        elif nm.startswith("robot0:"):
            c = "robot_box"
        elif t == 6:
            c = "table"
        else:
            c = nm
        cls[seg == g] = c
    return cls


def main():
    rep = {}
    for name in NAMES:
        env = se.make(name, batch=8, device="cuda")
        cls_ = type(env)
        ref = C.reference_env(cls_, env.size)
        u = ref.unwrapped
        st = env.sample_states(N, rng=np.random.default_rng(3))
        official = cls_.reference_render(ref, st)
        ours = env.render(st).cpu().numpy()

        sg = mujoco.Renderer(u.model, width=env.size, height=env.size)
        sg.enable_segmentation_rendering()
        KEEP.append(sg)
        segs = []
        for s in st:
            nq = int(u.model.nq)
            u.data.qpos[:] = np.asarray(s, np.float64)[:nq]
            u.data.qvel[:] = 0.0
            u.goal = np.asarray(s, np.float64)[nq:nq + 3].copy()
            mujoco.mj_forward(u.model, u.data)
            from gymnasium_robotics.utils import mujoco_utils
            mujoco_utils.reset_mocap2body_xpos(u.model, u.data)
            mujoco.mj_forward(u.model, u.data)
            # the env's own render callback places the target site; do the same before segmenting
            u._render_callback()
            sg.update_scene(u.data, camera=u.mujoco_renderer.viewer.cam)
            segs.append(sg.render()[..., 0].copy())
        segs = np.stack(segs)
        klass = classify(u.model, segs)

        d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
        im = C.interior_mask(official)
        di = d[:, :-1, :-1]
        ki = klass[:, :-1, :-1]
        tot = float(di[im].sum())
        split = {}
        for c in sorted(set(klass.reshape(-1).tolist())):
            allm = klass == c
            intm = im & (ki == c)
            split[c] = dict(
                frac=float(allm.mean()),
                all_exact=float((d[allm] == 0).mean()) if allm.any() else None,
                interior_frac_of_interior=float(intm.sum() / max(im.sum(), 1)),
                interior_exact=float((di[intm] == 0).mean()) if intm.any() else None,
                interior_err_share=float(di[intm].sum() / max(tot, 1)) if intm.any() else 0.0,
                mean=float(d[allm].mean()) if allm.any() else None)
        rep[name] = dict(overall=C.pixel_agreement(ours, official), split=split)

        # clean throughput for the official renderer: warm, then 32 frames of render() alone
        cls_.reference_render(ref, st[:4])
        t0 = time.perf_counter()
        cls_.reference_render(ref, st)
        rep[name]["fps_official"] = N / (time.perf_counter() - t0)

        print(f"\n=== {name}  official {rep[name]['fps_official']:.1f} f/s", flush=True)
        for c, v in split.items():
            print(f"  {c:14s} frame {v['frac']:.4f}  all_exact {v['all_exact']}  "
                  f"int_share_of_interior {v['interior_frac_of_interior']:.4f}  "
                  f"int_exact {v['interior_exact']}  int_err_share {v['interior_err_share']:.3f}",
                  flush=True)
    with open(f"{OUT}/attribute.json", "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    print("WROTE attribute.json")


if __name__ == "__main__":
    main()
