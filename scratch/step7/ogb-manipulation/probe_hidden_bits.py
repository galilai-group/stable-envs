"""Step 7 / W5 phase 0 -- name the hidden bits.

`probe_purity_fps.py` found that two `reset(seed=7)` calls on `swm/OGBCube-v0` give
frames that differ on 6.4 % of pixels while `qpos`, `qvel`, `mocap_pos`, `mocap_quat`
and every `MjModel` appearance array are bit-identical.  Something else in `MjModel` or
`MjData` carries the difference.  This walks every array of both structs, diffs them
between the two resets, and localises the pixel difference.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np


def arrays(obj):
    for n in dir(obj):
        if n.startswith("_"):
            continue
        try:
            v = getattr(obj, n)
        except Exception:
            continue
        if isinstance(v, np.ndarray) and v.size and v.dtype.kind in "fiub":
            yield n, v


def snap(obj):
    return {n: v.copy() for n, v in arrays(obj)}


def diff(a, b):
    out = []
    for k in a:
        if k not in b:
            continue
        if a[k].shape != b[k].shape:
            out.append((k, "shape", None)); continue
        if not np.array_equal(a[k], b[k]):
            try:
                d = float(np.abs(a[k].astype(np.float64) - b[k].astype(np.float64)).max())
            except Exception:
                d = None
            out.append((k, "value", d))
    return out


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    import mujoco
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="swm/OGBCube-v0")
    ap.add_argument("--seed", type=int, default=7)
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    env = gym.make(a.env); u = env.unwrapped
    env.reset(seed=a.seed)
    # NOTE: `compile_model_and_data` REBINDS `_model`/`_data` whenever the env is dirty,
    # so the structs must be re-fetched after every reset or the diff compares an object
    # with itself.
    m1o, d1o = u.model, u.data
    f1 = u.render().copy(); M1, D1 = snap(m1o), snap(d1o)
    env.reset(seed=a.seed)
    m, d = u.model, u.data
    f2 = u.render().copy(); M2, D2 = snap(m), snap(d)
    rec_ids = {"model_object_reused": m is m1o, "data_object_reused": d is d1o,
               "dirty_after_reset": bool(getattr(u, "_dirty", None))}

    rec = {"env": a.env, "seed": a.seed,
           "frames_identical": bool(np.array_equal(f1, f2)),
           "frac_pixels_differing": float((f1 != f2).any(-1).mean()),
           "max_abs_diff": int(np.abs(f1.astype(np.int16) - f2.astype(np.int16)).max()),
           "model_arrays_differing": [[k, w, v] for k, w, v in diff(M1, M2)],
           "data_arrays_differing": [[k, w, v] for k, w, v in diff(D1, D2)],
           **rec_ids}

    # where on screen?
    msk = (f1 != f2).any(-1)
    ys, xs = np.nonzero(msk)
    if ys.size:
        rec["diff_bbox_yxyx"] = [int(ys.min()), int(xs.min()), int(ys.max()), int(xs.max())]
        h, w = msk.shape
        rec["diff_bbox_frac_of_frame"] = float((ys.max()-ys.min()+1)*(xs.max()-xs.min()+1)/(h*w))

    # Now: force EVERY differing MjData/MjModel array from reset 1 back, re-render.
    for k, w, _ in rec["model_arrays_differing"]:
        if w == "value":
            getattr(m, k)[:] = M1[k]
    for k, w, _ in rec["data_arrays_differing"]:
        if w == "value":
            try:
                getattr(d, k)[:] = D1[k]
            except Exception:
                pass
    mujoco.mj_forward(m, d)
    f3 = u.render().copy()
    rec["identical_after_forcing_all_differing_arrays"] = bool(np.array_equal(f1, f3))
    rec["frac_pixels_differing_after_forcing"] = float((f1 != f3).any(-1).mean())

    # And the cheapest sufficient set: qpos+qvel+mocap only, from a clean reset.
    env.reset(seed=a.seed + 1)
    nb = getattr(u, "_num_buttons", 0)
    kw = {f"button_state_{i}": list(getattr(u, "_cur_button_states", [0]*nb))[i] for i in range(nb)}
    u.set_state(D1["qpos"], D1["qvel"], **kw) if nb else u.set_state(D1["qpos"], D1["qvel"])
    d.mocap_pos[:] = D1["mocap_pos"]; d.mocap_quat[:] = D1["mocap_quat"]
    mujoco.mj_forward(m, d)
    f4 = u.render().copy()
    rec["identical_from_other_episode_with_qpos_qvel_mocap"] = bool(np.array_equal(f1, f4))
    rec["frac_pixels_differing_from_other_episode"] = float((f1 != f4).any(-1).mean())

    np.savez_compressed(out / f"hidden_bits_{a.env.split('/')[-1]}.npz", f1=f1, f2=f2, f3=f3, f4=f4)
    (out / f"hidden_bits_{a.env.split('/')[-1]}.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({k: v for k, v in rec.items() if k not in ("model_arrays_differing", "data_arrays_differing")}, indent=1))
    print("model differing:", [k for k, _, _ in rec["model_arrays_differing"]])
    print("data differing:", [k for k, _, _ in rec["data_arrays_differing"]])


if __name__ == "__main__":
    main()
