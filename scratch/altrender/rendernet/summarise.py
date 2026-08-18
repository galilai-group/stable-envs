"""One table over every checkpoint and every evaluation, so nothing is read off a log tail."""
from __future__ import annotations

import argparse
import glob
import json
import os

CK = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/ckpt"
OUT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet"


def training():
    rows = []
    for p in sorted(glob.glob(os.path.join(CK, "*.json"))):
        j = json.load(open(p))
        h = j["history"]
        if not h:
            continue
        best = min(h, key=lambda r: r["mean_abs"])
        rows.append({"name": os.path.basename(p)[:-5], "net": j["args"]["net"],
                     "lr": j["args"]["lr"], "epochs": len(h), "params": j["params"],
                     "gflops": j["gflops_per_frame"],
                     "bilinear": j["bilinear_baseline"]["mean_abs"],
                     "bilinear_dyn": j["bilinear_baseline"]["mean_abs_dynamic"],
                     "best_mean": best["mean_abs"], "best_ep": best["epoch"],
                     "dyn": best["mean_abs_dynamic"], "static": best["mean_abs_static"],
                     "w8": best["within8_frac"], "exact": best["exact_frac"],
                     "final_mean": h[-1]["mean_abs"], "sec": h[-1]["seconds"]})
    rows.sort(key=lambda r: r["best_mean"])
    print(f"{'checkpoint':58s} {'lr':>6s} {'par':>8s} {'GF':>5s} {'bilin':>6s} -> {'best':>6s} "
          f"{'(ep)':>5s} {'dyn':>6s} {'stat':>6s} {'w8':>6s} {'exact':>6s} {'final':>6s}")
    for r in rows:
        print(f"{r['name']:58s} {r['lr']:6.0e} {r['params']:8d} {r['gflops']:5.2f} "
              f"{r['bilinear']:6.2f} -> {r['best_mean']:6.3f} {r['best_ep']:5d} {r['dyn']:6.2f} "
              f"{r['static']:6.2f} {r['w8']:6.3f} {r['exact']:6.3f} {r['final_mean']:6.3f}")
    return rows


def evals():
    for p in sorted(glob.glob(os.path.join(OUT, "eval_*.json"))):
        j = json.load(open(p))
        t = j["throughput"]
        print(f"\n=== {os.path.basename(p)}  net {j['net']}  {j['params']} params, "
              f"{j['gflops_per_frame']:.2f} GFLOP/frame")
        print(f"    shipped 256px {t['shipped_256_fps']:8.0f} f/s | render-only "
              f"{t['render_only_fps']:8.0f} | render+corrector "
              f"{t['render_plus_corrector']['fps']:8.0f} f/s")
        for row in j["splits"] + j["ood"]:
            if "error" in row:
                print(f"  {row}")
                continue
            head = f"  [{row['split']}] {row['env']} n={row['n']}"
            print(head + (f"   OOD of {row['ood_of']}" if "ood_of" in row else ""))
            for k, v in row["variants"].items():
                print(f"    {k:24s} mean {v['mean_abs']:7.3f} dyn {v['mean_abs_dynamic']:7.3f} "
                      f"stat {v['mean_abs_static']:7.3f} w8 {v['within8_frac']:.3f} "
                      f"ssim {v['ssim']:.4f} iexact {str(v['interior_exact_frac'])[:6]:>6s} "
                      f"max {v['max_abs']:3d} p99.9 {v['p999']:6.1f}")
        if j.get("shrink_sweep"):
            print("  shrink sweep (held out):")
            for s in j["shrink_sweep"]:
                print(f"    shrink {s['shrink']:4.1f}  mean {s['mean_abs']:7.3f}  "
                      f"iexact {str(s['interior_exact_frac'])[:6]}  ssim {s['ssim']:.4f}  "
                      f"w8 {s['within8_frac']:.3f}")
        if j.get("hallucination"):
            print("  hallucination controls:")
            for k, v in j["hallucination"].items():
                print(f"    {k}: {json.dumps(v)}")
        if j.get("encoders"):
            print("  frozen-encoder cosine distance, as a fraction of one state change:")
            for enc, row in j["encoders"].items():
                if "error" in row:
                    print(f"    {enc}: {row['error']}")
                    continue
                s = "  ".join(f"{k} {row[k]['frac_of_state_change']:.2f}"
                              for k in ("jpeg_q95", "shipped_256", "bilinear_lr", "corrector",
                                        "blank") if k in row)
                print(f"    {enc:36s} {s}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--what", default="both", choices=("train", "eval", "both"))
    a = ap.parse_args()
    if a.what in ("train", "both"):
        training()
    if a.what in ("eval", "both"):
        evals()
