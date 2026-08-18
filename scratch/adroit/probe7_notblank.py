"""Are the fixed-camera frames real pictures, and why did the segmentation buffer read empty?

probe6 reported 10/10 bit-identical frames under camera_name="fixed" AND background=1.0 from the
segmentation pass. Bit-identical frames are worthless as evidence if the frames are blank, so this
checks the frames themselves (unique colours, per-channel spread, PNG dump) and separately re-reads
the segmentation buffer with its documented layout, since a wrong decode of that buffer explains
"no geom pixels" just as well as an empty scene does.
"""
from __future__ import annotations

import os
os.environ.setdefault("MUJOCO_GL", "egl")

import glob
import json

import numpy as np

TMP = "/mnt/vast/home/randall.balestriero/se_logs/adroit"
SCENES = ("Door", "Hammer", "Pen", "Relocate")


def main():
    import mujoco
    import gymnasium, gymnasium_robotics                                        # noqa: F401
    gymnasium.register_envs(gymnasium_robotics)
    REF = {"Door": "AdroitHandDoor-v1", "Hammer": "AdroitHandHammer-v1",
           "Pen": "AdroitHandPen-v1", "Relocate": "AdroitHandRelocate-v1"}
    out = {}
    for s in SCENES:
        r = {}
        # --- the saved fixed-camera frames --------------------------------------------
        f = np.load(os.path.join(TMP, f"fix_{s}_0_0.npz"))["frame"]
        r["fixed_frame"] = {"shape": list(f.shape), "n_unique_colors": int(
            len(np.unique(f.reshape(-1, 3), axis=0))), "min": f.min(axis=(0, 1)).tolist(),
            "max": f.max(axis=(0, 1)).tolist(), "mean": f.mean(axis=(0, 1)).round(2).tolist()}
        # and the free-camera frame from probe4, for comparison
        g = np.load(os.path.join(TMP, f"frame_{s}_0.npz"))["frame"]
        r["free_frame"] = {"n_unique_colors": int(len(np.unique(g.reshape(-1, 3), axis=0))),
                           "mean": g.mean(axis=(0, 1)).round(2).tolist()}
        try:
            import imageio.v2 as iio
            iio.imwrite(os.path.join(TMP, f"view_fixed_{s}.png"), f)
            iio.imwrite(os.path.join(TMP, f"view_free_{s}.png"), g)
        except Exception as e:
            r["png"] = f"skipped: {e}"

        # --- re-read the segmentation buffer with the documented layout ----------------
        env = gymnasium.make(REF[s])
        u = env.unwrapped
        u.reset(seed=0)
        m = u.model
        cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, "fixed")
        rend = mujoco.Renderer(m, width=480, height=480)
        rend.enable_segmentation_rendering()
        rend.update_scene(u.data, camera=cid)
        seg = rend.render()
        rend.disable_segmentation_rendering()
        r["seg"] = {"shape": list(seg.shape), "dtype": str(seg.dtype),
                    "ch0_unique": np.unique(seg[..., 0]).tolist()[:12],
                    "ch1_unique_count": int(len(np.unique(seg[..., 1]))),
                    "ch1_min": int(seg[..., 1].min()), "ch1_max": int(seg[..., 1].max())}
        # mjOBJ_GEOM's numeric value, so the comparison in probe6 can be checked
        r["seg"]["mjOBJ_GEOM"] = int(mujoco.mjtObj.mjOBJ_GEOM)
        r["seg"]["mjOBJ_UNKNOWN"] = int(mujoco.mjtObj.mjOBJ_UNKNOWN)
        out[s] = r
        print(f"\n===== {s} =====")
        print(json.dumps(r, indent=2))
    with open(os.path.join(TMP, "probe7.json"), "w") as fh:
        json.dump(out, fh, indent=2)


if __name__ == "__main__":
    main()
