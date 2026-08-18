"""Clean throughput of the OFFICIAL renderer, warm, with nothing else on the GPU."""
import os, sys, time
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
import numpy as np, mujoco, gymnasium, gymnasium_robotics
gymnasium.register_envs(gymnasium_robotics)
for eid in ("FetchReach-v4", "FetchPush-v4", "FetchSlide-v4", "FetchPickAndPlace-v4"):
    for size in (480, 224):
        e = gymnasium.make(eid, render_mode="rgb_array", width=size, height=size)
        e.reset(seed=0)
        u = e.unwrapped
        for _ in range(10):
            u.render()
        t0 = time.perf_counter()
        for _ in range(60):
            u.render()
        dt = time.perf_counter() - t0
        print(f"{eid:24s} {size}px  render-only {60/dt:8.1f} f/s", flush=True)
