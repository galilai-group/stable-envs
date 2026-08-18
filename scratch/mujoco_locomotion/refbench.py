"""Reference (gymnasium EGL) throughput at the sizes people train at, for the doc's table."""
import os, sys, time, json
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
import stable_envs as se
from stable_envs.envs import mujoco_locomotion as ML

for name in ML.ENVS:
    cls = se.REGISTRY[name]
    rec = {"env": name}
    for size in (84, 240):
        env, u = cls._fresh(size)
        st = cls(batch=1, size=size, device="cuda").sample_states(200, rng=np.random.default_rng(0))
        u.set_state(st[0].astype(np.float64), np.zeros(u.model.nv)); cls._shot(u)
        t0 = time.perf_counter()
        for s in st:
            u.set_state(s.astype(np.float64), np.zeros(u.model.nv)); cls._shot(u)
        rec[f"fps_reference_{size}"] = round(200 / (time.perf_counter() - t0), 1)
    print(json.dumps(rec), flush=True)
