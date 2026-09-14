"""Do the two reflective WALLS matter? The shipped ``_bake`` keeps the FIRST reflective plane only.

manipulator's floor, wall1 and wall2 all wear ``material grid`` with ``reflectance=.2``. ``_bake``
loops the static geoms and ``break``s on the first plane it finds, so a mover mirrored in a WALL is
not composited. This measures the size of that hole three ways:

  1. MuJoCo's own frame with ``mat_reflectance = 0`` against its frame with reflectance on, at real
     states with the movers PRESENT -- the whole reflection signal, floor and walls together;
  2. the same with only the FLOOR's reflectance zeroed -- the part the shipped pass already models;
  3. our frame with ``sc.reflectance`` forced to 0 against our frame as shipped -- what the pass buys.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "glfw")

import numpy as np
import torch

from probe_prototype import patched_episode_slots


def diff(a, b):
    d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1)
    return dict(px=int((d > 0).sum()), frac=float((d > 0).mean()), max=int(d.max()),
                mean=float(d.mean()))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n", type=int, default=16)
    ap.add_argument("--out", default="examples/results/step7/dmc-manipulator/phase0")
    a = ap.parse_args()

    import mujoco
    from stable_envs.envs import dmcontrol as D

    patched_episode_slots(D)
    Cell = type("DMCManipulatorBringBall", (D.DMControlSuite,), dict(
        DOMAIN="manipulator", TASK="bring_ball", CAMERA=0, default_size=a.size,
        name="dmc-manipulator-bring-ball",
        reference_id="dm_control/manipulator-bring_ball", __doc__="phase-0 prototype"))

    env = Cell(batch=a.n, size=a.size, device="cpu", seed=0)
    st = env.sample_states(a.n, np.random.default_rng(33))
    ref = Cell.reference_env(a.size)

    sc = env._scene()
    m = sc.model
    names = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, i): i for i in range(m.ngeom)}
    mats = {mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_MATERIAL, i): i for i in range(m.nmat)}
    out = {"size": a.size, "n": a.n,
           "grid_material_worn_by": [g for g in ("floor", "wall1", "wall2")
                                     if int(m.geom_matid[names[g]]) == mats["grid"]]}

    p = ref.physics
    rm = p.model.ptr if hasattr(p.model, "ptr") else p.model
    keep = np.array(rm.mat_reflectance, copy=True)
    on = Cell.reference_render(ref, st)
    rm.mat_reflectance[:] = 0.0
    off_all = Cell.reference_render(ref, st)
    rm.mat_reflectance[:] = keep
    out["reference_reflection_signal_total"] = diff(on, off_all)

    # the shipped pass' own contribution
    ours = env.render(torch.as_tensor(st)).cpu().numpy()
    r_keep = sc.reflectance
    sc.reflectance = 0.0
    ours_norefl = env.render(torch.as_tensor(st)).cpu().numpy()
    sc.reflectance = r_keep
    out["our_reflection_pass_contribution"] = diff(ours, ours_norefl)
    out["parity_with_reflection_pass"] = diff(ours, on)
    out["parity_without_reflection_pass"] = diff(ours_norefl, on)

    os.makedirs(a.out, exist_ok=True)
    pth = os.path.join(a.out, f"reflect_{a.size}.json")
    json.dump(out, open(pth, "w"), indent=1, default=float)
    print(json.dumps(out, indent=1, default=float))
    print("wrote", pth)


if __name__ == "__main__":
    main()
