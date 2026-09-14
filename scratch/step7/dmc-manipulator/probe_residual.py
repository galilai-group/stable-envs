"""Where the prototype's residual lives: silhouette or shading, floor or wall, ss=1 or ss=2.

Answers three questions the headline numbers cannot:
  1. of the mover-region pixels, how many survive the INTERIOR mask -- i.e. is the 8.8% mover-region
     exactness a silhouette/MSAA effect or a shading error?
  2. does a supersample move it (MuJoCo renders at offsamples=4)?
  3. does the residual sit on the reflective WALLS, which the shipped bake drops (it keeps the first
     reflective plane only)?
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import torch

from probe_prototype import patched_episode_slots


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="examples/results/step7/dmc-manipulator/phase0")
    a = ap.parse_args()

    import mujoco
    import scipy.ndimage as ndi
    from stable_envs.envs import dmcontrol as D
    from stable_envs.testing import conformance as C

    patched_episode_slots(D)
    Cell = type("DMCManipulatorBringBall", (D.DMControlSuite,), dict(
        DOMAIN="manipulator", TASK="bring_ball", CAMERA=0, default_size=a.size,
        name="dmc-manipulator-bring-ball",
        reference_id="dm_control/manipulator-bring_ball", __doc__="phase-0 prototype"))

    ref = Cell.reference_env(a.size)
    out = {"size": a.size, "n": a.n}
    states = None
    for ss in (1, 2):
        env = Cell(batch=1, size=a.size, supersample=ss, device="cpu", seed=0)
        if states is None:
            states = env.sample_states(a.n, np.random.default_rng(1000))
            want = Cell.reference_render(ref, states)
            sc0 = env._scene()
            plate = sc0.plate.cpu().numpy().astype(np.int16)
            mover_mask = np.abs(want.astype(np.int16) - plate[None]).max(-1) > 0
            # interior mask of the reference, aligned to (N,H-1,W-1)
            im = C.interior_mask(want)
            mm = mover_mask[:, :-1, :-1]
            out["mover_region_px"] = int(mover_mask.sum())
            out["mover_region_interior_px"] = int((mm & im).sum())
            out["mover_region_interior_frac"] = float((mm & im).sum() / max(1, mm.sum()))
            # which STATIC surface each disagreeing pixel sits on: segment the reference with
            # MuJoCo's own segmentation of the mover-hidden scene
            out["note_segments"] = ("mover_region = pixels where the official frame differs from "
                                    "MuJoCo's own movers-hidden plate; it therefore contains the "
                                    "movers, their shadows and their reflections")
        got = env.render(torch.as_tensor(states)).cpu().numpy()
        d = np.abs(got.astype(np.int16) - want.astype(np.int16)).max(-1)
        pa = C.pixel_agreement(got, want)
        dm = d[:, :-1, :-1]
        rec = dict(
            exact_frac=pa["exact_frac"], mean_abs=pa["mean_abs"], max_abs=pa["max_abs"],
            interior_exact_frac=pa["interior_exact_frac"], interior_frac=pa["interior_frac"],
            interior_bad_px=pa["interior_bad_px"],
            mover_region_exact=float((d[mover_mask] == 0).mean()),
            mover_region_within8=float((d[mover_mask] <= 8).mean()),
            mover_region_mean_abs=float(d[mover_mask].mean()),
            mover_interior_exact=(float((dm[mm & im] == 0).mean()) if (mm & im).any() else None),
            mover_interior_px=int((mm & im).sum()),
            plate_region_exact=float((d[~mover_mask] == 0).mean()),
            bad_px=int((d > 0).sum()),
        )
        # how much of the mover region is a SILHOUETTE pixel of the reference (3x3 not flat)
        rec["mover_region_edge_frac"] = float((mm & ~im).sum() / max(1, mm.sum()))
        out[f"ss{ss}"] = rec
        print(f"ss={ss}", json.dumps(rec, indent=1))

    # is the residual on the walls? erode the reference's own geom segmentation
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"residual_{a.size}.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print("wrote", p)


if __name__ == "__main__":
    main()
