"""Is the horizon band a HAZE pass MJWarp does not implement, and what does the throughput curve look like?

``probe6`` finds the far-most floor band (the 24 rows nearest the horizon) at 0.053 exact where every
other band is 0.63-0.94, with the reference's luminance spread there at 0.46 -- i.e. MuJoCo draws it
FLAT and MJWarp draws texture. ``mjRND_HAZE`` is ON in MuJoCo's default scene flags (measured), it
blends geometry into the skybox near the horizon, and ``mujoco_warp`` has no haze pass at all.
"""
from __future__ import annotations

import sys
import time

import numpy as np

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/humanoidbench")

from hb_common import (LOGS, MjwRenderer, agreement, interior_mask, jsave, make_env, mj_render,
                       mj_segment, renderer, resolved_model, sample_qpos, set_state)

SIZE = 256
NSTATE = 16
#: one CPU core of JPEG decode at 256 px, from ``docs/research/when_does_this_pay_off.md``
DECODE_FPS_256 = 5486.0


def main():
    task = sys.argv[1] if len(sys.argv) > 1 else "walk"
    robot = sys.argv[2] if len(sys.argv) > 2 else "h1hand"
    import mujoco
    res = {"task": task, "robot": robot}

    env, u = make_env(task, robot, SIZE)
    cam_name, path = u.camera_name, u.fullpath
    Q = sample_qpos(task, robot, NSTATE, seed=0, steps=60)
    env.close()

    m, _ = resolved_model(path, SIZE)
    d = mujoco.MjData(m)
    cam = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam_name)
    ref = mj_render(m, d, cam, SIZE, Q, tag="ref")
    seg = mj_segment(m, d, cam, SIZE, Q, tag="seg")
    floor = (seg[..., 1] == int(mujoco.mjtObj.mjOBJ_GEOM)) & (seg[..., 0] == 0)

    r = renderer(m, SIZE, "ref")
    out = []
    try:
        for q in Q:
            set_state(m, d, q)
            r.update_scene(d, camera=int(cam))
            r.scene.flags[int(mujoco.mjtRndFlag.mjRND_HAZE)] = 0
            out.append(np.ascontiguousarray(r.render()))
    finally:
        r.update_scene(d, camera=int(cam))
        r.scene.flags[int(mujoco.mjtRndFlag.mjRND_HAZE)] = 1
    nohaze = np.stack(out)
    dh = np.abs(nohaze.astype(np.int16) - ref.astype(np.int16)).max(-1)
    im = interior_mask(ref)
    res["haze_pass"] = dict(pixels_changed=round(float((dh > 0).mean()), 5), max=int(dh.max()),
                            mean=round(float(dh.mean()), 4),
                            floor_pixels_changed=round(float((dh[floor] > 0).mean()), 5),
                            interior_pixels_changed=round(float((dh[:, :-1, :-1][im] > 0).mean()), 5))

    R = MjwRenderer(m, cam, SIZE, NSTATE)
    got = R.render(Q)
    res["vs_official"] = agreement(got, ref)
    res["vs_nohaze_reference"] = agreement(got, nohaze)
    del R

    # ---- throughput: the (ratio, batch) pair and its crossover ------------------------------
    speed = {}
    for B in (1, 4, 16, 64, 256, 1024):
        try:
            Rb = MjwRenderer(m, cam, SIZE, B, capture=True)
            Rb.upload(Q[np.arange(B) % NSTATE])
            Rb.wp.capture_launch(Rb.graph)
            Rb.wp.synchronize()
            reps = max(3, int(1200 / B) + 1)
            t0 = time.perf_counter()
            for _ in range(reps):
                Rb.wp.capture_launch(Rb.graph)
            Rb.wp.synchronize()
            speed[str(B)] = round(B * reps / (time.perf_counter() - t0), 1)
            del Rb
        except Exception as e:                                          # pragma: no cover
            speed[str(B)] = f"{type(e).__name__}: {e}"
    r = renderer(m, SIZE, "ref")
    set_state(m, d, Q[0])
    r.update_scene(d, camera=int(cam))
    r.render()
    t0 = time.perf_counter()
    for _ in range(120):
        r.update_scene(d, camera=int(cam))
        r.render()
    off = round(120 / (time.perf_counter() - t0), 1)
    res["throughput"] = dict(mjwarp=speed, official_render_only=off,
                             decode_fps_per_core=DECODE_FPS_256,
                             ratio_vs_official={k: (round(v / off, 1) if isinstance(v, float) else v)
                                                for k, v in speed.items()},
                             cores_of_decode={k: (round(v / DECODE_FPS_256, 2) if isinstance(v, float)
                                                  else v) for k, v in speed.items()})
    jsave(f"{LOGS}/probe8_{robot}_{task}.json", res)
    for k, v in res.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
