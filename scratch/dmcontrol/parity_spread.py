"""All-pixel and interior agreement over INDEPENDENT state draws, for the parity cells.

``parity_all_exact_min`` is the one floor an env may legitimately declare below the shared bar, and
the reason is arithmetic rather than a tolerance: all-pixel exactness is interior exactness WEIGHTED
BY HOW MUCH OF THE FRAME MOVES. ``finger`` has seven movers covering 9.5-11.7% of an 84 px frame
against cartpole's 3.0%, so it cannot reach cartpole's number on the identical backend with the
identical per-pixel residual. Declaring that needs the SPREAD, not one draw: a floor set at a single
measurement flakes on the next.
"""
from __future__ import annotations
import argparse
import json
import os
os.environ.setdefault("MUJOCO_GL", "egl")
import numpy as np


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", default="")
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--draws", type=int, default=3)
    ap.add_argument("--size", type=int, default=84)
    ap.add_argument("--ss", type=int, default=2)
    ap.add_argument("--out", default="/mnt/vast/home/randall.balestriero/logs/dmc2/parity_spread.json")
    a = ap.parse_args()
    import stable_envs as se
    from stable_envs.envs import dmcontrol as DM
    from stable_envs.testing import conformance as C
    names = a.envs.split(",") if a.envs else sorted(DM.DOMAINS)
    out = {}
    for name in names:
        env = se.make(name, batch=8, size=a.size, supersample=a.ss, device="cuda")
        cls = type(env)
        ref = C.reference_env(cls, a.size)
        rows = []
        for k in range(a.draws):
            st = env.sample_states(a.n, rng=np.random.default_rng(200 + k))
            r = C.pixel_agreement(env.render(st).cpu().numpy(), cls.reference_render(ref, st))
            rows.append(r)
        out[name] = {
            "exact": [round(r["exact_frac"], 5) for r in rows],
            "exact_worst": round(min(r["exact_frac"] for r in rows), 5),
            "interior": [r["interior_exact_frac"] for r in rows],
            "interior_frac": [round(r["interior_frac"], 6) for r in rows],
            "mean_abs": [round(r["mean_abs"], 4) for r in rows],
            "mean_worst": round(max(r["mean_abs"] for r in rows), 4),
            "max_abs": [r["max_abs"] for r in rows],
        }
        print(f"{name:22s} exact={out[name]['exact']} worst={out[name]['exact_worst']} "
              f"interior={out[name]['interior']} mean={out[name]['mean_abs']}", flush=True)
    os.makedirs(os.path.dirname(a.out), exist_ok=True)
    with open(a.out, "w") as f:
        json.dump(out, f, indent=1, default=float)


if __name__ == "__main__":
    main()
