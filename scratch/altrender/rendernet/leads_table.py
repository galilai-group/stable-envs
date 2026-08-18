"""The two leads, each as the one table that decides it.

**(a) One corrector across a kitchen's three cameras.** Per cell, the specialist fitted on that cell
against the joint checkpoint fitted on all three, on the same held-out states and the same five
metrics -- plus the BUDGET-MATCHED joint model, which saw ``16384 // 3`` rows per cell, i.e. one
specialist's data spread over three cells. The first comparison says whether one checkpoint CAN cover
a kitchen; the second says whether doing so costs three times less, which is the claim in the lead.

**(b) Randomising the aux table rows.** The recorded failure is not the in-distribution number, it is
the pair "best mean, worst SSIM" on a kitchen the corrector never saw, and the blank-frame probe that
explains it: a descriptor model shown a table it never fitted emits 86/255 of residual on a grey
frame. So the table here is three columns wide -- in distribution, the blank probe, and zero-shot --
because a repair that moves only the first one has not moved the lead.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

FIVE = ("mean_abs", "within8_frac", "ssim", "interior_mean_abs", "interior_exact_frac")


def load(pattern):
    out = {}
    for p in sorted(glob.glob(pattern)):
        try:
            out[os.path.basename(p)] = json.load(open(p))
        except Exception:                                          # noqa: BLE001
            pass
    return out


def held(res):
    for r in res["splits"]:
        if r["split"] == "heldout":
            return r["variants"]
    return None


def decl(res):
    for r in res["splits"]:
        if r["split"] == "decl" and "worst_of_draws" in r:
            return r["worst_of_draws"]
    return None


def line(tag, m, fps=None):
    if m is None:
        return f"{tag:52s}      --"
    f = "" if fps is None else f"{fps:8.0f}"
    return (f"{tag:52s} {f:>8s} {m['mean_abs']:7.2f} {m['within8_frac']:6.3f} {m['ssim']:7.4f} "
            f"{m['interior_mean_abs']:6.2f} {m['interior_exact_frac']:6.3f}")


HDR = (f"{'':52s} {'f/s':>8s} {'mean':>7s} {'w8':>6s} {'ssim':>7s} {'imean':>6s} {'iexact':>6s}")


def lead_a(d, cells):
    print("LEAD (a)  ONE CORRECTOR ACROSS L52S03's THREE CAMERAS\n")
    for key, title in (("heldout", "held out (disjoint recorded states, default appearance)"),
                       ("decl", "declared draw (sample_states x4, worst of the four)")):
        print(f"-- {title}")
        print(HDR)
        for cell in cells:
            got = False
            for name, res in d.items():
                if res.get("env") != cell:
                    continue
                v = held(res) if key == "heldout" else decl(res)
                if v is None:
                    continue
                if not got:
                    ship = v.get("shipped_256") or v.get("shipped_256")
                    print(line(f"{cell}  SHIPPED 256 px", ship,
                               res["throughput"]["shipped_256_fps"]))
                    bil = v.get("bilinear")
                    if bil is not None:
                        print(line("   bilinear 112 -> 256", bil))
                    got = True
                stem = os.path.basename(res["ckpt"])[:-3]
                print(line("   " + stem, v["corrector"],
                           res["throughput"]["render_plus_corrector"]["fps"]))
            print()


def lead_b(d):
    print("\nLEAD (b)  RANDOMISING THE AUX TABLE ROWS THE WAY THE STYLE IS RANDOMISED\n")
    print("-- in distribution (RoboCasaCounterToSink held out, default appearance)")
    print(HDR + "   blank-residual  moved>8")
    for name, res in sorted(d.items()):
        v = held(res)
        if v is None:
            continue
        stem = os.path.basename(res["ckpt"])[:-3]
        bp = res.get("hallucination", {}).get("blank_input", {}).get("grey", {})
        extra = (f"   {bp.get('residual_mean_abs', float('nan')):8.1f}      "
                 f"{bp.get('frac_moved_over_8', float('nan')):5.2f}") if bp else ""
        print(line("   " + stem, v["corrector"],
                   res["throughput"]["render_plus_corrector"]["fps"]) + extra)
    print("\n-- zero-shot, target scene's OWN descriptor table swapped in")
    for name, res in sorted(d.items()):
        if not res.get("ood"):
            continue
        stem = os.path.basename(res["ckpt"])[:-3]
        print(f"   {stem}")
        for row in res["ood"]:
            if "error" in row:
                print(f"      {row['env']:44s} {row['error']}")
                continue
            v = row["variants"]
            for k in ("shipped_256", "bilinear", "corrector"):
                if k in v:
                    print(line(f"      {row['env']} / {k}", v[k]))
        for k, v in res.get("hallucination", {}).items():
            if k.startswith("overlay_vs_"):
                print(f"      overlay corr {k[11:]:38s} r={v['mean_map_pearson']:+.3f} "
                      f"home {v['home_mean_residual']:.2f} foreign {v['foreign_mean_residual']:.2f}")
        print()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--which", default="ab")
    a = ap.parse_args()
    if "a" in a.which:
        d = load(os.path.join(a.dir, "eval_RoboCasa*L52S03x3*.json"))
        d.update(load(os.path.join(a.dir, "eval_RoboCasaCounterToSink_*sty_e300_on_default.json")))
        d.update(load(os.path.join(a.dir, "eval_RoboCasaCounterToSinkRight_*sty_e300_on_default.json")))
        d.update(load(os.path.join(a.dir, "eval_RoboCasaCounterToSinkWrist_*sty_e300_on_default.json")))
        d.update(load(os.path.join(a.dir, "eval_*jointloop*.json")))
        lead_a(d, ["RoboCasaCounterToSink", "RoboCasaCounterToSinkRight",
                   "RoboCasaCounterToSinkWrist"])
    if "b" in a.which:
        d = load(os.path.join(a.dir, "eval_*aux-descriptor*.json"))
        d.update(load("/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/"
                      "eval_RoboCasaCounterToSink_unet_112t128_w48_d2_u2_sty_aux-descriptor_e300"
                      "_on_default.json"))
        d.update(load("/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/"
                      "eval_RoboCasaCounterToSink_unet_112t128_w48_d2_u2_sty_e300_RGBONLY"
                      "_on_default.json"))
        lead_b(d)


if __name__ == "__main__":
    main()
