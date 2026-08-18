"""When HumanoidBench's fidelity moves inside a long session, WHICH side of the comparison moved?

``humanoidbench.py`` declares that this family is not bit-reproducible inside a long conformance
session and registers only behind an opt-in for it, but what is recorded is the SYMPTOM (two tests
fail by 1-2/255 on a different cell each run) and not which renderer drifts. It matters: a drift in
``env.render`` is a defect in this repo, and a drift in ``reference_render`` is a defect in the
reference env we compare against, and the two need opposite fixes.

One process, ONE set of states, held fixed. Every round re-renders them on both sides and compares
each side against its OWN first round, so the two drifts are separated. Other envs are constructed
part-way through, because that is what a suite does and what a single-env loop does not.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("STABLE_ENVS_HUMANOIDBENCH", "1")

import numpy as np
import torch

import stable_envs as se
from stable_envs.testing import conformance as C


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="HumanoidBenchH1Walk")
    ap.add_argument("--others", default="HumanoidBenchH1Stand,HumanoidBenchH1HandStair")
    ap.add_argument("--rounds", type=int, default=10)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--intrude-at", type=int, default=3)
    a = ap.parse_args()

    env = se.make(a.env, batch=16, device="cuda", seed=0)
    cls = type(env)
    states = env.sample_states(a.n)
    ref = C.reference_env(cls, env.size)

    ours0 = off0 = None
    rows = []
    for k in range(a.rounds):
        if k == a.intrude_at:
            for other in [x for x in a.others.split(",") if x]:
                e2 = se.make(other, batch=16, device="cuda", seed=0)
                e2.render(e2.sample_states(8))
                r2 = C.reference_env(type(e2), e2.size)
                type(e2).reference_render(r2, e2.sample_states(4))
            print(f"  -- built {a.others} after round {k}", flush=True)
        ours = env.render(states).cpu().numpy()
        off = cls.reference_render(ref, states)
        if ours0 is None:
            ours0, off0 = ours, off
        row = {
            "round": k,
            "mean_abs_vs_official": float(np.abs(ours.astype(np.int32)
                                                 - off.astype(np.int32)).mean()),
            "ours_drift_max": int(np.abs(ours.astype(np.int32) - ours0.astype(np.int32)).max()),
            "ours_drift_mean": float(np.abs(ours.astype(np.int32) - ours0.astype(np.int32)).mean()),
            "ref_drift_max": int(np.abs(off.astype(np.int32) - off0.astype(np.int32)).max()),
            "ref_drift_mean": float(np.abs(off.astype(np.int32) - off0.astype(np.int32)).mean()),
        }
        rows.append(row)
        print(f"  round {k}: mean|ours-official| {row['mean_abs_vs_official']:.4f}  "
              f"ours vs round0 max {row['ours_drift_max']} mean {row['ours_drift_mean']:.4f}  "
              f"reference vs round0 max {row['ref_drift_max']} mean {row['ref_drift_mean']:.4f}",
              flush=True)
    print("JSON " + json.dumps({"env": a.env, "rows": rows}), flush=True)


if __name__ == "__main__":
    main()
