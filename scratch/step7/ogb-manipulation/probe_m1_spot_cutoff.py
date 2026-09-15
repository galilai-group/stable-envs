"""M1, the half that needs no GPU: does MuJoCo's own renderer honour a spot cutoff above 90 degrees?

`mujoco_warp` attenuates a spot as `cos_theta < cos(cutoff)` -> unlit, then `pow(cos_theta, exp)`.
`swm/OGBCube-v0`'s spot is cutoff 100 deg, exponent 10. OpenGL's `GL_SPOT_CUTOFF` accepts [0, 90] or
the sentinel 180 and rejects everything else with `GL_INVALID_VALUE`, leaving the default 180 = "not
a spotlight" -- so if MuJoCo goes through that path, cutoff 100 and cutoff 180 must render the SAME
frame while a valid 89 must render a different one. That is a statement about MuJoCo alone and is
therefore measurable on any machine, which is why it is not in the GPU job.
"""
from __future__ import annotations

import argparse, json
from pathlib import Path

import numpy as np


def main():
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--env", default="swm/OGBCube-v0")
    a = ap.parse_args()
    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)

    env = gym.make(a.env); u = env.unwrapped
    env.reset(seed=21); m, d = u.model, u.data
    for _ in range(8):
        env.step(env.action_space.sample())
    spot = [i for i in range(m.nlight) if int(m.light_type[i]) == 0]
    rec = {"env": a.env, "spot_ids": spot,
           "cutoff": [float(m.light_cutoff[i]) for i in spot],
           "exponent": [float(m.light_exponent[i]) for i in spot],
           "diffuse": [m.light_diffuse[i].tolist() for i in spot],
           "pos": [m.light_pos[i].tolist() for i in spot],
           "dir": [m.light_dir[i].tolist() for i in spot],
           "cutoff_sweep": [], "exponent_sweep": []}
    base = u.render().copy()
    for cut in (100.0, 180.0, 90.0, 89.0, 45.0, 10.0):
        for i in spot:
            m.light_cutoff[i] = cut
        f = u.render().copy()
        rec["cutoff_sweep"].append({
            "cutoff": cut,
            "frac_differing_from_100": float((f != base).any(-1).mean()),
            "max_abs": int(np.abs(f.astype(np.int16) - base.astype(np.int16)).max()),
            "frame_mean": float(f.mean())})
    for i in spot:
        m.light_cutoff[i] = 100.0
    base2 = u.render().copy()
    for ex in (10.0, 0.0, 1.0, 50.0):
        for i in spot:
            m.light_exponent[i] = ex
        f = u.render().copy()
        rec["exponent_sweep"].append({
            "exponent": ex,
            "frac_differing_from_10": float((f != base2).any(-1).mean()),
            "max_abs": int(np.abs(f.astype(np.int16) - base2.astype(np.int16)).max())})
    for i in spot:
        m.light_exponent[i] = 10.0
    sv = m.light_active.copy()
    for i in spot:
        m.light_active[i] = 0
    f = u.render().copy()
    rec["spot_off"] = {"frac_differing": float((f != base2).any(-1).mean()),
                       "max_abs": int(np.abs(f.astype(np.int16) - base2.astype(np.int16)).max())}
    m.light_active[:] = sv
    c100 = [r for r in rec["cutoff_sweep"] if r["cutoff"] == 100.0][0]
    c180 = [r for r in rec["cutoff_sweep"] if r["cutoff"] == 180.0][0]
    c89 = [r for r in rec["cutoff_sweep"] if r["cutoff"] == 89.0][0]
    rec["verdict_100_equals_180"] = bool(c180["frac_differing_from_100"] < 1e-3)
    rec["verdict_89_is_honoured"] = bool(c89["frac_differing_from_100"] > 0.5)
    (out / "m1_mujoco_spot_cutoff.json").write_text(json.dumps(rec, indent=1, sort_keys=True))
    print(json.dumps({k: rec[k] for k in ("cutoff", "exponent", "verdict_100_equals_180",
                                          "verdict_89_is_honoured", "spot_off")}, indent=1))
    for r in rec["cutoff_sweep"]:
        print(f"  cutoff {r['cutoff']:6.1f}  differs {r['frac_differing_from_100']*100:6.2f}%  "
              f"max {r['max_abs']:3d}  mean {r['frame_mean']:.3f}")


if __name__ == "__main__":
    main()
