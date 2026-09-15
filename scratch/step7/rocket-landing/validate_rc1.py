"""RC-1 evidence: the reference bridge on FLOAT32 states, at measure_tier's exact seed, and on the corpus.

(a) ``PFRocketLanding.sample_states(64)`` with ``rng = default_rng(1000)`` (draw 0 of measure_tier.py),
    float32 as the env holds them -> number of pose-assertion rejections in ``_reference_frame``
    (was 36/64 before the fix, per the review).
(b) the first 100 corpus states, from the float64 obs (the frames' own states) and from the float32
    ``state13``: exact frames, rejections, worst differing-pixel count.
Writes examples/results/step7/rocket-landing/reference/rc1_evidence.json.
"""
from __future__ import annotations

import json
import os
import sys
import time

import numpy as np

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", ".."))
sys.path.insert(0, ROOT)
REF = os.path.join(ROOT, "examples", "results", "step7", "rocket-landing", "reference")
from stable_envs.envs import rocket_landing as RL  # noqa: E402


def main():
    t0 = time.time()
    env = RL.PFRocketLanding.reference_env(480)
    out = {"env_head": os.popen(f"git -C {ROOT} rev-parse --short HEAD").read().strip()}
    # (a) the exact measure_tier draw-0 states
    fake = object.__new__(RL.PFRocketLanding)
    fake.rng = np.random.default_rng(1000)
    st = fake.sample_states(64)                                                  # float32
    assert st.dtype == np.float32
    sty = np.tile(RL._default_style_row(), (64, 1))
    rej, errs = 0, []
    for i in range(64):
        try:
            RL._reference_frame(env, st[i].astype(np.float64), sty[i])
        except AssertionError as e:
            rej += 1
            errs.append(str(e)[:120])
    out["draw0_seed1000_n64_float32"] = {"rejections": rej, "first_errors": errs[:3],
                                          "quat_norm_dev_max": float(np.abs(np.linalg.norm(st[:, 3:7].astype(np.float64), axis=1) - 1).max())}
    # (b) corpus, float64 vs float32
    S = np.load(os.path.join(REF, "states.npz"))
    F = np.load(os.path.join(REF, "frames.npz"))["frames"]
    sty = RL.corpus_style_to_env(S["style"])
    for label, states in (("float64_obs", S["obs30"][:100, :13].astype(np.float64)),
                          ("float32_state13", S["state13"][:100].astype(np.float64))):
        rej, exact, diffs = 0, 0, []
        for i in range(100):
            try:
                fr = RL._reference_frame(env, states[i], sty[i])
            except AssertionError:
                rej += 1
                continue
            d = int(np.any(fr != F[i], axis=-1).sum())
            diffs.append(d)
            exact += d == 0
        out[f"corpus100_{label}"] = {"rejections": rej, "exact": exact, "rendered": len(diffs),
                                     "diff_px_max": max(diffs) if diffs else None,
                                     "diff_px_mean": float(np.mean(diffs)) if diffs else None}
    out["seconds"] = round(time.time() - t0, 1)
    with open(os.path.join(REF, "rc1_evidence.json"), "w") as fh:
        json.dump(out, fh, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
