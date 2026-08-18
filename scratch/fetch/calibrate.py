"""Fetch calibration: the numbers that ship, plus the attribution of what is left.

Everything is scored through the ENV as registered — no probe-only code path — against
``reference_render``, i.e. the official env with the mocap canonicalised the way the env's own
``reset_mocap2body_xpos`` canonicalises it.

Three columns are produced per scene:

  vs official              what the conformance suite asserts
  vs official, markers off the same comparison against a reference whose four TRANSLUCENT mocap
                           markers are hidden — the one thing this renderer deliberately does not draw
  throughput               our end-to-end render (plate composite included) against the official EGL
                           renderer on the same node
"""
from __future__ import annotations

import json
import os
import sys
import time

os.environ.setdefault("MUJOCO_GL", "egl")
sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

import numpy as np
import torch
import mujoco

import stable_envs as se
from stable_envs.testing import conformance as C

OUT = "/mnt/vast/home/randall.balestriero/stable_envs_logs/fetch"
NAMES = ("FetchReach", "FetchPush", "FetchSlide", "FetchPickAndPlace")
N = int(os.environ.get("FETCH_N", 64))


def main():
    rep = {}
    for name in NAMES:
        env = se.make(name, batch=8, device="cuda")
        cls = type(env)
        ref = C.reference_env(cls, env.size)
        st = env.sample_states(N, rng=np.random.default_rng(3))
        official = cls.reference_render(ref, st)
        ours = env.render(st).cpu().numpy()
        r = dict(vs_official=C.pixel_agreement(ours, official))

        # the same reference with the four translucent markers hidden
        u = ref.unwrapped
        m = u.model
        keep = m.geom_rgba.copy()
        mocap_geoms = [g for g in range(m.ngeom)
                       if int(m.body_mocapid[m.geom_bodyid[g]]) >= 0]
        m.geom_rgba[mocap_geoms, 3] = 0.0
        nomark = cls.reference_render(ref, st)
        m.geom_rgba[:] = keep
        r["vs_official_markers_off"] = C.pixel_agreement(ours, nomark)
        r["marker_cost"] = C.pixel_agreement(nomark, official)

        # throughput: ours (end to end, plate composite included) and the official EGL renderer
        big = env.sample_states(256, rng=np.random.default_rng(4))
        buf = torch.empty((256, env.size, env.size, 3), dtype=torch.uint8, device=env.device)
        env.render(big, out=buf)
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            env.render(big, out=buf)
        torch.cuda.synchronize()
        r["fps_ours_256"] = 3 * 256 / (time.perf_counter() - t0)
        env.render(st[:64])
        torch.cuda.synchronize()
        t0 = time.perf_counter()
        for _ in range(3):
            env.render(st[:64])
        torch.cuda.synchronize()
        r["fps_ours_64"] = 3 * 64 / (time.perf_counter() - t0)
        t0 = time.perf_counter()
        cls.reference_render(ref, st[:16])
        r["fps_official"] = 16 / (time.perf_counter() - t0)

        rep[name] = r
        print(f"\n=== {name}", flush=True)
        for k in ("vs_official", "vs_official_markers_off", "marker_cost"):
            v = r[k]
            print(f"  {k:26s} exact {v['exact_frac']:.5f}  interior {v['interior_exact_frac']:.6f}  "
                  f"mean {v['mean_abs']:.4f}  within8 {v['within8_frac']:.5f}  "
                  f"int_mean {v['interior_mean_abs']:.5f}", flush=True)
        print(f"  fps ours@256 {r['fps_ours_256']:.0f}  ours@64 {r['fps_ours_64']:.0f}  "
              f"official {r['fps_official']:.1f}  speedup {r['fps_ours_256']/r['fps_official']:.0f}x",
              flush=True)
        np.save(f"{OUT}/cal_ours_{name}.npy", ours[:4])
        np.save(f"{OUT}/cal_official_{name}.npy", official[:4])
    with open(f"{OUT}/calibrate.json", "w") as fh:
        json.dump(rep, fh, indent=1, default=float)
    worst = min(rep[n]["vs_official"]["interior_exact_frac"] for n in NAMES)
    print(f"\nWORST interior_exact_frac = {worst:.6f}", flush=True)


if __name__ == "__main__":
    main()
