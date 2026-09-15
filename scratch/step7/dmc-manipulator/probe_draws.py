"""The DRAW COUNT that declares ``DMCManipulatorBringBall``'s parity floors.

The family's rule, from ``PARITY_NOTE``: a parity floor is set from MANY INDEPENDENT DRAWS and not
from one large evaluation, because a rare pixel event reported off a single draw is how MujocoPusher
came to declare a floor it then failed 15 times in 101. So this runs ``--draws`` seeded draws of
``--n`` states each and reports the WORST and the median of every metric, plus the per-draw series.

Run on Oscar under ``oscar_certify311.sbatch`` at the declared pin (mujoco 3.11.0 + dm_control 1.0.44),
which is the arm the declaration is taken from; the local Mac numbers are 3.12.0 / 1.0.45 and are
recorded separately.
"""
from __future__ import annotations

import argparse
import json
import os

os.environ.setdefault("MUJOCO_GL", "egl")

import numpy as np
import torch

CELL = "DMCManipulatorBringBall"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--size", type=int, default=224)
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--draws", type=int, default=41)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--supersample", type=int, default=None)
    ap.add_argument("--out", default="examples/results/step7/dmc-manipulator/oscar")
    a = ap.parse_args()

    import mujoco
    from importlib.metadata import version as _v
    import stable_envs as se
    from stable_envs.testing import conformance as C

    cls = se.REGISTRY[CELL]
    env = cls(batch=a.n, size=a.size, supersample=a.supersample, device=a.device, seed=0)
    ref = C.reference_env(cls, a.size)
    res = dict(cell=CELL, size=a.size, n=a.n, draws=a.draws, device=a.device,
               supersample=int(env.ss), mujoco=mujoco.__version__,
               dm_control=_v("dm_control"), per_draw=[])

    for k in range(a.draws):
        st = env.sample_states(a.n, np.random.default_rng(1000 + k))
        got = env.render(torch.as_tensor(st, device=env.device)).cpu().numpy()
        want = cls.reference_render(ref, st)
        pa = C.pixel_agreement(got, want)
        pa["ssim"] = float(C.ssim(got, want))
        pa["draw"] = k
        res["per_draw"].append(pa)
        print(f"draw {k:3d}: exact {pa['exact_frac']:.6f} interior {pa['interior_exact_frac']} "
              f"bad {pa['interior_bad_px']}/{pa['interior_px']} mean {pa['mean_abs']:.4f} "
              f"max {pa['max_abs']} ssim {pa['ssim']:.5f}", flush=True)

    def agg(key):
        v = [p[key] for p in res["per_draw"] if p.get(key) is not None]
        return None if not v else dict(worst=float(np.min(v)), median=float(np.median(v)),
                                       best=float(np.max(v)))
    res["summary"] = {k: agg(k) for k in ("exact_frac", "interior_exact_frac", "interior_frac",
                                          "ssim")}
    for k in ("mean_abs", "max_abs"):
        v = [p[k] for p in res["per_draw"]]
        res["summary"][k] = dict(worst=float(np.max(v)), median=float(np.median(v)),
                                 best=float(np.min(v)))
    res["interior_px_total"] = int(sum(p["interior_px"] or 0 for p in res["per_draw"]))
    res["interior_bad_px_total"] = int(sum(p["interior_bad_px"] or 0 for p in res["per_draw"]))
    res["declaration"] = dict(
        parity_interior_min=1.0 if res["interior_bad_px_total"] == 0 else None,
        all_exact_worst=res["summary"]["exact_frac"]["worst"],
        clears_shared_0_90=bool(res["summary"]["exact_frac"]["worst"] >= 0.90),
        note=("parity_all_exact_min is declared ONLY if the worst draw does not clear the suite's "
              "shared 0.90 bar; see ALL_EXACT_FLOOR / PARITY_NOTE"))
    os.makedirs(a.out, exist_ok=True)
    p = os.path.join(a.out, f"draws_{a.size}_{a.device}.json")
    json.dump(res, open(p, "w"), indent=1, default=float)
    print(json.dumps(res["summary"], indent=1))
    print(f"interior: {res['interior_bad_px_total']} bad of {res['interior_px_total']}")
    print("wrote", p)


if __name__ == "__main__":
    main()
