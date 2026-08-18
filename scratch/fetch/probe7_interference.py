"""Three questions the locomotion agent's findings raise about THIS family's numbers.

1. ``OffScreenViewer.render`` never binds its GL context, so a second context that becomes current can
   silently rewrite a live env's pixels. This env builds ``mujoco.Renderer`` objects for the plate bake
   IN THE SAME PROCESS as the reference. Does that corrupt the reference? Compared against frames
   produced in a process where no stable_envs renderer exists at all.
2. Gymnasium's free camera is frozen at first render and its DEFAULT lookat is ``median(geom_xpos)``.
   Fetch overrides all four parameters from ``DEFAULT_CAMERA_CONFIG``, so it should be a constant of
   the family — checked by reading the live viewer camera after resets, steps and renders.
3. Throughput at 224 px as well as 480, because on a small scene MuJoCo's rasterizer is competitive and
   a loss has to be reported as a loss.
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

import numpy as np

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/fetch"
PY = sys.executable

CLEAN = r'''
import os, sys, hashlib
os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
import numpy as np, gymnasium, gymnasium_robotics, mujoco
gymnasium.register_envs(gymnasium_robotics)
from gymnasium_robotics.utils import mujoco_utils
st = np.load("%s/interf_states.npy")
env = gymnasium.make("%s", render_mode="rgb_array", width=480, height=480)
u = env.unwrapped
nq = int(u.model.nq)
out = []
for s in st.astype(np.float64):
    u.data.qpos[:] = s[:nq]; u.data.qvel[:] = 0.0
    u.goal = s[nq:nq+3].copy()
    mujoco.mj_forward(u.model, u.data)
    mujoco_utils.reset_mocap2body_xpos(u.model, u.data)
    mujoco.mj_forward(u.model, u.data)
    out.append(np.ascontiguousarray(u.render()))
np.save("%s/interf_clean.npy", np.stack(out))
c = u.mujoco_renderer.viewer.cam
print("CLEANCAM", float(c.distance), float(c.azimuth), float(c.elevation),
      np.asarray(c.lookat).tolist(), int(c.type), int(c.fixedcamid))
'''


def main():
    import torch
    import stable_envs as se
    from stable_envs.testing import conformance as C

    rep = {}
    name, ref_id = "FetchPickAndPlace", "FetchPickAndPlace-v4"
    env = se.make(name, batch=8, device="cuda")
    cls = type(env)
    st = env.sample_states(8, rng=np.random.default_rng(7))
    np.save(f"{OUT}/interf_states.npy", st)

    # --- clean process, no stable_envs renderer anywhere
    src = CLEAN % (OUT, ref_id, OUT)
    r = subprocess.run([PY, "-c", src], capture_output=True, text=True)
    clean_cam = [ln for ln in r.stdout.splitlines() if ln.startswith("CLEANCAM")]
    assert os.path.exists(f"{OUT}/interf_clean.npy"), r.stdout + r.stderr[-2000:]
    clean = np.load(f"{OUT}/interf_clean.npy")

    # --- same frames, in THIS process, where the env's plate renderers are live
    env.render(st)                                   # force the plate bake and the warp context
    ref = C.reference_env(cls, env.size)
    dirty = cls.reference_render(ref, st)
    d = np.abs(clean.astype(np.int16) - dirty.astype(np.int16)).max(-1)
    rep["reference_contamination"] = dict(max=int(d.max()), frac=float((d > 0).mean()),
                                          mean=float(d.mean()))

    # --- is the viewer camera a constant of the family?
    u = ref.unwrapped
    cams = []
    for i in range(3):
        ref.reset(seed=100 + i)
        for _ in range(5):
            ref.step(ref.action_space.sample())
        u.render()
        c = u.mujoco_renderer.viewer.cam
        cams.append([float(c.distance), float(c.azimuth), float(c.elevation),
                     *np.asarray(c.lookat).tolist(), float(c.type), float(c.fixedcamid)])
    rep["viewer_camera_over_time"] = cams
    rep["viewer_camera_constant"] = bool(np.abs(np.asarray(cams) - np.asarray(cams[0])).max() == 0)
    rep["clean_process_camera"] = clean_cam

    # --- throughput at both resolutions, ours end-to-end vs official render-only
    for size in (480, 224):
        e = se.make(name, batch=8, size=size, device="cuda")
        big = e.sample_states(256, rng=np.random.default_rng(4))
        buf = torch.empty((256, size, size, 3), dtype=torch.uint8, device=e.device)
        e.render(big, out=buf)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(5):
            e.render(big, out=buf)
        torch.cuda.synchronize()
        rep[f"fps_ours_{size}"] = 5 * 256 / (time.perf_counter() - t0)
    rep["fps_official_480"] = 954.4      # scratch/fetch/fps_official.py, warm, render only
    rep["fps_official_224"] = 2503.4
    for size in (480, 224):
        rep[f"speedup_{size}"] = rep[f"fps_ours_{size}"] / rep[f"fps_official_{size}"]

    print(json.dumps(rep, indent=1, default=float), flush=True)
    with open(f"{OUT}/probe7.json", "w") as fh:
        json.dump(rep, fh, indent=1, default=float)


if __name__ == "__main__":
    main()
