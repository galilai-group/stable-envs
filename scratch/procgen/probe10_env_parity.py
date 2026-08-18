"""Parity + throughput of the torch env module against the shipped procgen renderer."""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing import conformance as C

NAMES = ["ProcgenMaze", "ProcgenHeist", "ProcgenMiner"]


def main():
    print("import errors:", se.envs.import_errors())
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 64
    for name in NAMES:
        env = se.make(name, batch=16, device="cuda")
        st = env.sample_states(n)
        ref = C.reference_env(type(env), env.size)
        official = type(env).reference_render(ref, st)
        ours = env.render(st).cpu().numpy()
        r = C.pixel_agreement(ours, official)
        d = np.abs(ours.astype(np.int16) - official.astype(np.int16))
        bad = np.argwhere(d.max(-1) > 0)
        print(f"\n{name}: state_dim={env.state_dim} n={n}")
        print(f"  exact_frac {r['exact_frac']:.6f}  interior {r['interior_exact_frac']}  "
              f"mean_abs {r['mean_abs']:.5f}  max_abs {r['max_abs']}  bad_px {len(bad)}")
        if len(bad):
            for f_, y, x in bad[:8]:
                print(f"    frame {f_} px({y},{x}) ours={ours[f_,y,x].tolist()} "
                      f"ref={official[f_,y,x].tolist()}")
            per = np.bincount(bad[:, 0], minlength=n)
            print(f"    frames with any bad px: {(per>0).sum()}/{n}, worst frame {per.max()} px")

        # throughput
        for B in (1, 64, 1024, 4096):
            e = se.make(name, batch=B, device="cuda")
            s = torch.as_tensor(e.sample_states(B), device="cuda")
            for _ in range(3):
                e.render(s)
            torch.cuda.synchronize()
            t0 = time.time()
            reps = 20 if B <= 1024 else 5
            for _ in range(reps):
                e.render(s)
            torch.cuda.synchronize()
            dt = (time.time() - t0) / reps
            print(f"  B={B:5d}: {B/dt:12.0f} frames/s  ({1000*dt:8.2f} ms/call)")


if __name__ == "__main__":
    main()
