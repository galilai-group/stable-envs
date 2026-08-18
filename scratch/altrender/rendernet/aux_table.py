"""The aux question in one table: does the out-of-distribution failure close?

Everything else about the corrector is settled. This joins, per architecture, the RGB-only checkpoint
against its depth / descriptor / learned-embedding siblings on the SAME evaluation sets, so the only
thing that differs down a column is what the network was allowed to see. The in-distribution row is
included because it is cheap and because a lever that helps in distribution and hurts out of it is a
different finding from one that does nothing.

Read the last three columns, not the first. The mean is what a resampling filter moves; SSIM and
within-8 are what caught the hallucination in the RGB-only study, and they are what will catch it
again here if the aux channels have only made the network more confidently wrong.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

OUT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet"
#: The order the report reads them in: no aux, then one signal at a time.
AUX_ORDER = ("none", "depth", "descriptor", "learned")


def rows_for(env: str):
    out = {}
    for p in glob.glob(f"{OUT}/eval_*.json"):
        j = json.load(open(p))
        if j["env"] != env:
            continue
        ck = os.path.basename(j["ckpt"])[:-3]
        aux = j.get("aux", "none")
        net = j["net"]
        sty = "_sty" in ck
        for r in j["splits"] + j["ood"]:
            if "error" in r:
                continue
            key = (net, sty, aux)
            tgt = r["env"] if "ood_of" in r else f"[{r['split']}] {j['tag']}"
            out.setdefault(key, {})[tgt] = r["variants"]
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="RoboCasaCounterToSink")
    ap.add_argument("--targets", default="[heldout] r112_tex2,RoboCasaCounterToSinkRight,"
                                         "RoboCasaCounterToSinkL45S33,RoboCasaCounterToSinkWrist")
    a = ap.parse_args()
    data = rows_for(a.env)
    targets = a.targets.split(",")
    for tgt in targets:
        base = next((v[tgt] for v in data.values() if tgt in v), None)
        if base is None:
            print(f"\n### {tgt}: no evaluation found")
            continue
        print(f"\n### {tgt}")
        print(f"{'architecture':32s} {'aux':11s} {'mean':>7s} {'dyn':>7s} {'w8':>7s} {'ssim':>7s} "
              f"{'iexact':>7s} {'max':>5s}")
        for k in ("bilinear", "shipped_256", "jpeg_q95_of_official"):
            if k not in base:
                continue
            v = base[k]
            print(f"{'-- ' + k:32s} {'':11s} {v['mean_abs']:7.2f} {v['mean_abs_dynamic']:7.2f} "
                  f"{v['within8_frac']:7.3f} {v['ssim']:7.4f} "
                  f"{(v['interior_exact_frac'] or 0):7.3f} {v['max_abs']:5d}")
        for (net, sty, aux), vals in sorted(data.items(),
                                            key=lambda z: (z[0][0], z[0][1],
                                                           AUX_ORDER.index(z[0][2])
                                                           if z[0][2] in AUX_ORDER else 9)):
            if tgt not in vals:
                continue
            v = vals[tgt]["corrector"]
            print(f"{net[:30] + ('*' if sty else ''):32s} {aux:11s} {v['mean_abs']:7.2f} "
                  f"{v['mean_abs_dynamic']:7.2f} {v['within8_frac']:7.3f} {v['ssim']:7.4f} "
                  f"{(v['interior_exact_frac'] or 0):7.3f} {v['max_abs']:5d}")
    print("\n* = trained under the env's declared camera and light variation")


if __name__ == "__main__":
    main()
