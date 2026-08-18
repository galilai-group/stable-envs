"""How expensive is one official RoboCasa frame, and where does the time go?

The stable-envs pitch is "don't store images, regenerate them". That only pays if regenerating is
cheap relative to reading a stored frame. So this separates the three costs that get conflated:

  * scene construction  -- create_env + the first compile (paid once per process)
  * hard reset          -- robocasa re-runs _load_model() and recompiles on every reset, so this is
                           paid once per *episode*, not once per run
  * physics step        -- mj_step at the env's control rate
  * render              -- the EGL offscreen rasterization itself, at several resolutions and
                           camera counts

and reports frames/s for each, plus the render-only figure that a GPU rasterizer would have to beat.
"""
from __future__ import annotations

import argparse
import json
import os
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np


def timeit(fn, n, warmup=3):
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    dt = time.perf_counter() - t0
    return dt / n


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env-name", default="PickPlaceCounterToSink")
    ap.add_argument("--n", type=int, default=50)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/code/stable-envs/scratch/robocasa/out")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    import mujoco
    from robocasa.utils.env_utils import create_env

    rec = {"gpu": os.environ.get("CUDA_VISIBLE_DEVICES", "?")}
    try:
        import subprocess
        rec["gpu_name"] = subprocess.check_output(
            ["nvidia-smi", "--query-gpu=name", "--format=csv,noheader"]).decode().strip().split("\n")[0]
    except Exception as e:
        rec["gpu_name"] = f"unknown ({e})"

    t0 = time.time()
    env = create_env(env_name=args.env_name, robots="PandaOmron",
                     camera_names=["robot0_agentview_left"],
                     camera_widths=256, camera_heights=256, seed=0, render_onscreen=False)
    rec["t_create_env_s"] = time.time() - t0

    t0 = time.time()
    env.reset()
    rec["t_first_reset_s"] = time.time() - t0

    m = env.sim.model._model
    rec["ngeom"] = int(m.ngeom)
    rec["nmesh"] = int(m.nmesh)
    rec["nq"] = int(m.nq)

    # hard reset cost (robocasa default hard_reset=True -> recompiles the model each episode)
    ts = []
    for _ in range(3):
        t0 = time.time()
        env.reset()
        ts.append(time.time() - t0)
    rec["t_reset_s_mean"] = float(np.mean(ts))
    rec["t_reset_s_all"] = ts

    # physics
    a = np.zeros(env.action_dim)
    rec["t_step_s"] = timeit(lambda: env.step(a), args.n)
    rec["steps_per_s"] = 1.0 / rec["t_step_s"]
    rec["control_freq"] = float(getattr(env, "control_freq", -1))
    rec["n_mjstep_per_control"] = int(getattr(env, "control_timestep", 0) / m.opt.timestep) if m.opt.timestep else -1
    rec["mj_timestep"] = float(m.opt.timestep)

    # render, several resolutions
    cam = "robot0_agentview_left"
    rec["render"] = {}
    for res in (64, 128, 224, 256, 512):
        try:
            t = timeit(lambda: env.sim.render(width=res, height=res, camera_name=cam), args.n)
            rec["render"][str(res)] = {"s_per_frame": t, "fps": 1.0 / t}
            print(f"render {res}x{res}: {1.0/t:8.1f} fps ({t*1e3:.2f} ms)")
        except Exception as e:
            rec["render"][str(res)] = {"error": str(e)}
            print(f"render {res}: FAILED {e}")

    # raw mujoco.Renderer (no robosuite wrapper) for reference
    try:
        d = env.sim.data._data
        for res in (128, 224, 256):
            r = mujoco.Renderer(m, height=res, width=res)
            cid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, cam)

            def f():
                r.update_scene(d, camera=cid)
                r.render()
            t = timeit(f, args.n)
            rec["render"][f"raw_{res}"] = {"s_per_frame": t, "fps": 1.0 / t}
            print(f"raw mujoco.Renderer {res}x{res}: {1.0/t:8.1f} fps")
            r.close()
    except Exception as e:
        rec["render"]["raw_error"] = str(e)

    # step+render, 3 cameras, as used in the datasets
    try:
        env3 = create_env(env_name=args.env_name, robots="PandaOmron",
                          camera_names=["robot0_agentview_left", "robot0_agentview_right",
                                        "robot0_eye_in_hand"],
                          camera_widths=128, camera_heights=128, seed=0, render_onscreen=False)
        env3.reset()
        cams = ["robot0_agentview_left", "robot0_agentview_right", "robot0_eye_in_hand"]

        def f3():
            for c in cams:
                env3.sim.render(width=128, height=128, camera_name=c)
        t = timeit(f3, args.n)
        rec["render"]["3cam_128"] = {"s_per_frame_set": t, "sets_per_s": 1.0 / t}
        print(f"3 cams @128: {1.0/t:.1f} obs/s")
    except Exception as e:
        rec["render"]["3cam_error"] = str(e)

    # what fraction of an env step is rendering?
    r224 = rec["render"].get("224", {}).get("s_per_frame")
    if r224:
        rec["render_share_of_step"] = r224 / (rec["t_step_s"] + r224)

    with open(os.path.join(args.out, "speed.json"), "w") as f:
        json.dump(rec, f, indent=2)
    print(json.dumps(rec, indent=2))


if __name__ == "__main__":
    main()
