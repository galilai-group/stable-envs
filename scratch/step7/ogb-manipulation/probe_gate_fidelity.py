"""Measure the visual-tier declaration with THE GATE'S OWN INSTRUMENT, then characterise the spread.

WHY THIS EXISTS. The declaration shipped for OGBCube was measured by `probe_remeasure_chain.py` with
`sample_states(n, rng=default_rng(1000+k))` and came out at 2.41-2.44/255 across every draw it took.
The gate measures `check_visual_fidelity`, whose draw is `env.sample_states(int(n))` with NO rng --
which `sample_states` turns into `default_rng(0)`. That is one fixed draw, it is the ONLY draw the
test ever takes, and on it the same env scores 3.295. So the declared ceiling of 3.2 was set from a
band of draws that excludes the single draw it is checked against, and no amount of widening by the
stated rule fixes that: worst-of-eight x1.3 over seeds 1000-1007 is 3.17, which still fails.

This is the same defect as the Gate-A gap found the same day: a probe that measures something ADJACENT
to what the gate measures will report green indefinitely. So this probe calls the gate's own function
for the authoritative number, and re-derives that number with an explicit-rng copy of the same body
as an INSTRUMENT AGREEMENT CHECK -- if the copy disagrees at seed 0, the spread it reports is
measuring something else and the run is refused rather than declared.
"""
from __future__ import annotations

import argparse
import json
import os
import pathlib
import socket

import numpy as np


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--cells", nargs="+", default=["OGBCube", "OGBScene"])
    ap.add_argument("--n", type=int, default=64)
    ap.add_argument("--draws", type=int, default=8)
    # ATTRIBUTION. The declaration is taken at seeds 0..draws-1 because seed 0 is the draw the gate
    # checks. `--seed-base 1000` re-takes the SAME measurement at the seeds job 6420033 used on the
    # pre-fix tree, which is the only way to tell a pixel change caused by this fix apart from the
    # much larger change caused by measuring different seeds.
    ap.add_argument("--seed-base", type=int, default=0)
    a = ap.parse_args()

    import stable_envs as se
    import stable_envs.testing.conformance as C

    out_dir = pathlib.Path(a.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    job = f"job {os.environ.get('SLURM_JOB_ID')} on {socket.gethostname()}"
    report: dict = {}

    def measure(env, cls, ref, states) -> dict:
        """`check_visual_fidelity`'s body, with the draw handed in instead of taken at seed 0."""
        official = cls.reference_render(ref, states)
        ours = env.render(states).cpu().numpy()
        assert ours.shape == official.shape, f"{ours.shape} vs {official.shape}"
        r = C.pixel_agreement(ours, official)
        r["ssim"] = C.ssim(ours, official)
        return r

    for cname in a.cells:
        cls = se.REGISTRY[cname]
        env = se.make(cname, batch=a.n, device="cuda")
        ref = cls.reference_env()

        # WARM UP THE REFERENCE RENDERER FIRST. Measured on CPU with two renders of one fixed state:
        # the FIRST render in a process differs from the second by 6 pixels of +/-1, and every render
        # after that is bit-identical to the one before it. It is an upstream first-frame effect, it
        # is far too small to move `mean_abs` (unchanged at 12 digits) and far too large to ignore in
        # an equality check -- it moved `interior_px` by one pixel in 2.5 million and `ssim` by 5e-9,
        # which is what made the first attempt at this probe REFUSE OGBScene. Burning one frame before
        # anything is measured keeps the effect out of every draw instead of inside the first one.
        cls.reference_render(ref, env.sample_states(2, np.random.default_rng(999))[:1])

        gate = C.check_visual_fidelity(env, n=a.n)            # THE number the test compares
        mine = measure(env, cls, ref, env.sample_states(a.n, np.random.default_rng(0)))
        # TOLERANCE, chosen to separate the two things this check exists to tell apart. A DIFFERENT
        # INSTRUMENT shows up as a gross difference -- the old chain read mean_abs 0.89 low because it
        # averaged |diff| over channels where `pixel_agreement` averages the per-pixel channel MAX.
        # The first-frame effect shows up at 1e-9. 1e-6 is comfortably below the former and above the
        # latter, and the observed delta is recorded either way rather than discarded.
        agree = {k: [gate[k], mine[k]] for k in ("mean_abs", "within8_frac", "ssim")
                 if abs(float(gate[k]) - float(mine[k])) > 1e-6}
        if agree:
            report[cname] = {"REFUSED": "instrument disagreement at seed 0", "delta": agree,
                             "gate": gate, "copy": mine, "job": job}
            print(f"{cname} REFUSED: {agree}", flush=True)
            continue

        base = int(a.seed_base)
        draws = [dict(gate, seed=0)] if base == 0 else []
        for k in range(1 if base == 0 else 0, a.draws):
            sd = base + k
            draws.append(dict(measure(env, cls, ref,
                                      env.sample_states(a.n, np.random.default_rng(sd))), seed=sd))
            print(f"{cname} draw seed={k}: mean_abs={draws[-1]['mean_abs']:.4f} "
                  f"ssim={draws[-1]['ssim']:.4f} within8={draws[-1]['within8_frac']:.4f}", flush=True)

        ma = [float(d["mean_abs"]) for d in draws]
        w8 = [float(d["within8_frac"]) for d in draws]
        ss = [float(d["ssim"]) for d in draws]
        ie = [float(d["interior_exact_frac"]) for d in draws]
        ifr = [float(d["interior_frac"]) for d in draws]
        ex = [float(d["exact_frac"]) for d in draws]
        # THE RULE, stated once and applied to every cell: the declared value is the WORST draw, and
        # the threshold is that worst x1.3 for a ceiling (rounded UP to the next 0.05) or the best
        # draw minus 0.04 for a floor (rounded DOWN to the next 0.05). The gate's own seed-0 draw is
        # one of the draws, so the threshold covers the draw that is actually checked BY CONSTRUCTION
        # and not by luck.
        report[cname] = {
            "mean_abs": max(ma), "within8_frac": min(w8), "ssim": min(ss),
            "interior_exact_frac": min(ie), "interior_frac": float(np.mean(ifr)),
            "all_exact_frac": min(ex),
            "draw_spread": {"mean_abs": [min(ma), max(ma)], "ssim": [min(ss), max(ss)],
                            "within8_frac": [min(w8), max(w8)]},
            # The SHIPPED rule, spelled exactly as `_fidelity` computes it, so the artifact and the
            # class cannot disagree about what the threshold is.
            "mean_abs_max": round(max(ma) * 1.3, 1),
            "within8_min": float(np.floor((min(w8) - 0.04) * 20) / 20),
            "ssim_min": float(np.floor((min(ss) - 0.04) * 20) / 20),
            "gate_draw": {"seed": 0, "mean_abs": float(gate["mean_abs"]),
                          "ssim": float(gate["ssim"]), "within8_frac": float(gate["within8_frac"])},
            "n_states": a.n, "draws": a.draws, "seed_base": base, "job": job,
            "instrument_agreement": {k: [float(gate[k]), float(mine[k])]
                                     for k in ("mean_abs", "within8_frac", "ssim")},
        }
        print(f"{cname} WORST-OF-{a.draws} mean_abs={max(ma):.4f} -> ceiling "
              f"{report[cname]['mean_abs_max']}; gate draw was {gate['mean_abs']:.4f}", flush=True)

    (out_dir / "paste_me.json").write_text(json.dumps(report, indent=1, sort_keys=True))
    print(json.dumps(report, indent=1, sort_keys=True), flush=True)
    return 0 if all("REFUSED" not in v for v in report.values()) else 5


if __name__ == "__main__":
    raise SystemExit(main())
