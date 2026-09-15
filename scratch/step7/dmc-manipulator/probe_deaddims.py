"""Why ``check_state_dims_matter`` calls three dimensions dead, and whether that is our defect.

``ball_y`` (the ball prop's hinge) and the two ``target_ball`` quaternion slots all orient a SPHERE.
An analytic per-pixel sphere is exactly rotationally invariant; MuJoCo's is a TESSELLATED sphere
shaded per vertex, so rotating it rotates the mesh and moves a few pixels. This measures both sides
of that on the same rotations.
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
    from stable_envs.envs import dmcontrol as D

    patched_episode_slots(D)
    Cell = type("DMCManipulatorBringBall", (D.DMControlSuite,), dict(
        DOMAIN="manipulator", TASK="bring_ball", CAMERA=0, default_size=a.size,
        name="dmc-manipulator-bring-ball",
        reference_id="dm_control/manipulator-bring_ball", __doc__="phase-0 prototype"))

    env = Cell(batch=a.n, size=a.size, device="cpu", seed=0)
    ref = Cell.reference_env(a.size)
    names = list(env.state_names)
    st = env.sample_states(a.n, np.random.default_rng(21))

    base_ours = env.render(torch.as_tensor(st)).cpu().numpy()
    base_ref = Cell.reference_render(ref, st)

    out = {"size": a.size, "n": a.n, "state_names": names, "dims": {}}
    for d, nm in enumerate(names):
        pert = st.copy()
        if nm == "ball_y":
            pert[:, d] = pert[:, d] + 0.15
        elif nm.startswith("target_ball_quat"):
            # rotate the target body about y by 0.4 rad, keeping the quaternion a unit quaternion
            iw = names.index("target_ball_quat_qw")
            iy = names.index("target_ball_quat_qy")
            ang = 2.0 * np.arctan2(pert[:, iy], pert[:, iw]) + 0.4
            pert[:, iw] = np.cos(ang / 2)
            pert[:, iy] = np.sin(ang / 2)
        else:
            pert[:, d] = pert[:, d] + 0.15
        ours = env.render(torch.as_tensor(pert)).cpu().numpy()
        theirs = Cell.reference_render(ref, pert)
        do = np.abs(ours.astype(np.int16) - base_ours.astype(np.int16)).max(-1)
        dr = np.abs(theirs.astype(np.int16) - base_ref.astype(np.int16)).max(-1)
        agree = np.abs(ours.astype(np.int16) - theirs.astype(np.int16)).max(-1)
        out["dims"][nm] = dict(
            ours_px_moved=int((do > 0).sum()), ours_max=int(do.max()),
            ref_px_moved=int((dr > 0).sum()), ref_max=int(dr.max()),
            parity_after_perturb_exact_frac=float((agree == 0).mean()),
            parity_after_perturb_mean_abs=float(agree.mean()))
        print(nm, json.dumps(out["dims"][nm]))

    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"dead_dims_{a.size}.json")
    json.dump(out, open(p, "w"), indent=1, default=float)
    print("wrote", p)


if __name__ == "__main__":
    main()
