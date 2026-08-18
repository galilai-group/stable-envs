"""Interior-exactness of a CORRECTED frame, over many independent draws.

One held-out set reporting ``interior_exact_frac = 1.0`` is not evidence that a corrector preserves a
parity claim, and this repository already has the counterexample written down: ``MujocoPusher``
declared 1.0 off a single draw and then failed 15 of 101 seeded ones at 1-2 pixels in 3 M. A learned
decoder is a strictly worse thing to trust on one draw than a rasterizer is, because its residual is a
continuous function that happens to quantise away rather than a value that is structurally zero.

So the claim is tested the way the repo tests a rare pixel event: ``--draws`` independent 64-state
draws at the parity seed and its successors, each one scored on the three assertions the test makes,
with the COUNT of disagreeing interior pixels pooled over all of them. A corrector "keeps the claim"
only if that count is zero across every draw.
"""
from __future__ import annotations

import argparse
import json
import os
import time

import common as K  # noqa: E402

import numpy as np
import torch

import nets_ss  # noqa: E402
import train_ss as T  # noqa: E402
from stable_envs.testing import conformance as C  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--shrink", type=float, required=True)
    ap.add_argument("--draws", type=int, default=41)
    ap.add_argument("--n-states", type=int, default=64)
    ap.add_argument("--out", default="")
    a = ap.parse_args()

    ck = torch.load(a.ckpt, map_location="cuda", weights_only=False)
    cell, size, net = ck["cell"], ck["size"], ck["net"]
    model = nets_ss.build(net).cuda().to(memory_format=torch.channels_last)
    model.load_state_dict(ck["state_dict"])
    model = model.eval()
    cls = K.cls_of(cell)
    floor = float(getattr(cls, "parity_interior_min", 1.0))
    all_floor = float(getattr(cls, "parity_all_exact_min", None) or 0.90)
    env = K.make(cell, batch=1, size=size)
    ref = cls.reference_env(int(size))

    rows = []
    t0 = time.perf_counter()
    for k in range(a.draws):
        seed = K.PARITY_STATE_SEED + k
        st = np.asarray(env.sample_states(a.n_states, rng=np.random.default_rng(seed)), np.float32)
        official = cls.reference_render(ref, st)
        ss1 = env.render(torch.as_tensor(st, device="cuda"), size=size, supersample=1)
        corr = T.predict(model, ss1, a.shrink).cpu().numpy()
        r = C.pixel_agreement(corr, official)
        ie = r["interior_exact_frac"]
        rows.append({"seed": seed, "interior_exact_frac": ie,
                     "interior_px": r["interior_px"], "interior_bad_px": r["interior_bad_px"],
                     "exact_frac": r["exact_frac"], "mean_abs": r["mean_abs"],
                     "max_abs": r["max_abs"], "within8_frac": r["within8_frac"],
                     "interior_ok": True if ie is None else bool(ie >= floor),
                     "all_exact_ok": bool(r["exact_frac"] >= all_floor),
                     "mean_ok": bool(r["mean_abs"] <= 4.0)})
        if (k + 1) % 10 == 0:
            print(f"  draw {k+1}/{a.draws}  {(k+1)/(time.perf_counter()-t0):.2f}/s", flush=True)

    n_int = sum(r["interior_px"] or 0 for r in rows)
    n_bad = sum(r["interior_bad_px"] or 0 for r in rows)
    out = {"ckpt": os.path.basename(a.ckpt), "cell": cell, "size": size, "net": net,
           "shrink": a.shrink, "draws": a.draws, "floors": {"interior": floor, "all_exact": all_floor},
           "n_fail_interior": sum(not r["interior_ok"] for r in rows),
           "n_fail_all_exact": sum(not r["all_exact_ok"] for r in rows),
           "n_fail_mean": sum(not r["mean_ok"] for r in rows),
           "interior_px_total": n_int, "interior_bad_px_total": n_bad,
           "interior_exact_pooled": None if n_int == 0 else 1.0 - n_bad / n_int,
           "worst_exact_frac": min(r["exact_frac"] for r in rows),
           "worst_mean_abs": max(r["mean_abs"] for r in rows),
           "worst_max_abs": max(r["max_abs"] for r in rows),
           "median_mean_abs": float(np.median([r["mean_abs"] for r in rows])),
           "rows": rows, "host": os.uname().nodename}
    print(f"\n{cell} {size}px {net} shrink {a.shrink}: interior fail {out['n_fail_interior']}"
          f"/{a.draws}, all-exact fail {out['n_fail_all_exact']}, mean fail {out['n_fail_mean']} | "
          f"interior bad {n_bad}/{n_int} | worst exact {out['worst_exact_frac']:.4f} "
          f"worst mean {out['worst_mean_abs']:.4f} worst max {out['worst_max_abs']}", flush=True)
    p = a.out or os.path.join(K.LOGS, "corrected_draws",
                              f"{os.path.basename(a.ckpt)[:-3]}_sh{a.shrink:g}.json")
    os.makedirs(os.path.dirname(p), exist_ok=True)
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print("wrote", p, flush=True)


if __name__ == "__main__":
    main()
