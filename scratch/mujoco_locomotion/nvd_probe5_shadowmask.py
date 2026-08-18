"""WHERE does our shadow land, and where does MuJoCo's? The two masks, per surface.

The shadow pass helps HalfCheetah (mean 4.85 -> 3.79) and hurts the other five (Hopper 4.51 -> 7.08).
That is either a bug in the pass or a fact about MuJoCo, and one measurement separates them: take the
pixels OUR pass changes (ours with shadows against ours without) and the pixels MUJOCO's changes (the
reference against the same reference at ``mjRND_SHADOW = 0``), and split both by surface off MuJoCo's
own segmentation buffer. A pass that shadows the floor where MuJoCo shadows the floor is right and
merely mis-shaped; a pass that darkens the BODY while MuJoCo darkens the floor is a different thing
entirely.
"""
from __future__ import annotations

import argparse
import json
import os
import sys

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/nvdloco"


def emit(tag, obj):
    print(f"@@{tag} " + json.dumps(obj, default=float), flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--cls", default="MujocoHopper")
    ap.add_argument("--size", type=int, default=240)
    ap.add_argument("--states", type=int, default=8)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    import mujoco

    from stable_envs.envs import mujoco_locomotion as M

    env_id = M.VISUAL_ENVS[args.cls]

    def build(shadows):
        cls = type(args.cls, (M.GymMujocoVisual,), dict(
            ENV_ID=env_id, name=M._hyphen_name(env_id), reference_id=f"gymnasium/{env_id}",
            SHADOWS=bool(shadows), __doc__="VISUAL TIER shadow-mask probe"))
        M._NVD_CACHE.clear()
        return cls, cls(batch=8, size=args.size, device="cuda")

    cls_off, env_off = build(False)
    states = env_off.sample_states(int(args.states))
    ours_off = env_off.render(states).cpu().numpy()
    del env_off
    cls_on, env_on = build(True)
    ours_on = env_on.render(states).cpu().numpy()
    m = env_on._model()
    cam_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "track")
    floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    d = mujoco.MjData(m)
    ren = mujoco.Renderer(m, height=args.size, width=args.size)
    ref_on, ref_off, seg = [], [], []
    for q in np.asarray(states, np.float64):
        d.qpos[:] = q
        d.qvel[:] = 0
        mujoco.mj_forward(m, d)
        ren.update_scene(d, camera=int(cam_id))
        ref_on.append(ren.render().copy())
        ren.update_scene(d, camera=int(cam_id))
        ren.scene.flags[mujoco.mjtRndFlag.mjRND_SHADOW] = 0
        ref_off.append(ren.render().copy())
        ren.enable_segmentation_rendering()
        ren.update_scene(d, camera=int(cam_id))
        seg.append(ren.render()[..., 0].copy())
        ren.disable_segmentation_rendering()
    ref_on, ref_off, seg = np.stack(ref_on), np.stack(ref_off), np.stack(seg)

    regions = {"frame": np.ones(seg.shape, bool), "floor": seg == floor,
               "body": (seg >= 0) & (seg != floor), "sky": seg < 0}
    mask_ours = (ours_on != ours_off).any(-1)
    mask_ref = (ref_on != ref_off).any(-1)
    emit("masks", {"ours_frac": float(mask_ours.mean()), "ref_frac": float(mask_ref.mean()),
                   "iou": float((mask_ours & mask_ref).sum() / max(1, (mask_ours | mask_ref).sum())),
                   "ours_only": float((mask_ours & ~mask_ref).mean()),
                   "ref_only": float((mask_ref & ~mask_ours).mean())})
    for name, r in regions.items():
        emit(f"region.{name}", {
            "px_frac": float(r.mean()),
            "ours_mask_here": float((mask_ours & r).sum() / max(1, mask_ours.sum())),
            "ref_mask_here": float((mask_ref & r).sum() / max(1, mask_ref.sum())),
            "err_off": float(np.abs(ours_off.astype(np.int16) - ref_on.astype(np.int16)).max(-1)[r].mean()),
            "err_on": float(np.abs(ours_on.astype(np.int16) - ref_on.astype(np.int16)).max(-1)[r].mean())})

    os.makedirs(OUT, exist_ok=True)
    np.save(f"{OUT}/mask_{args.cls}.npy",
            np.stack([ours_off[0], ours_on[0], ref_on[0], ref_off[0]]))


if __name__ == "__main__":
    main()
