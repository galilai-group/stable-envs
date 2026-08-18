"""Assemble every measurement in this directory into the tables the report quotes.

Reads only what other scripts wrote (``frontier_*.json``, ``window_*.json``, ``fps_*.json``,
``ckpt/*.json``, ``eval_*.json``) and prints; it computes nothing that was not measured.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics as st

import common as C

L = C.LOGS
BARS = json.load(open(os.path.join(L, "bars.json")))


def frontier():
    rows = []
    for p in sorted(glob.glob(f"{L}/frontier_*.json")):
        d = json.load(open(p))
        R = {r["config"]: r for r in d["rows"] if r.get("kind") == "render" and "fps" in r}
        if "r96" not in R:
            continue
        rows.append((d["env"], BARS[d["env"]]["stored"], R))
    print(f"\n### resolution frontier, {len(rows)} of 50 tasks, batch 1024, 256 px out\n")
    keys = ["r256", "r160", "r128", "r112", "r96"]
    print(f"{'':16s} " + "".join(f"{k:>24s}" for k in keys))
    for label, f in (("f/s (median)", lambda r, k: r[2][k]["fps"]),
                     ("x stored bar", lambda r, k: r[2][k]["fps"] / r[1]),
                     ("mean |diff|", lambda r, k: r[2][k]["mean_abs"]),
                     ("SSIM", lambda r, k: r[2][k]["ssim"]),
                     ("interior-exact", lambda r, k: r[2][k]["interior_exact_frac"]),
                     ("all-pixel exact", lambda r, k: r[2][k]["exact_frac"])):
        cells = []
        for k in keys:
            v = [f(r, k) for r in rows if k in r[2]]
            fmt = "{:.0f}" if label.startswith("f/s") else "{:.3f}"
            cells.append((fmt + " [" + fmt + "," + fmt + "]").format(st.median(v), min(v), max(v))
                         if v else "")
        print(f"{label:16s} " + "".join(f"{c:>24s}" for c in cells))
    n = {k: sum(1 for r in rows if k in r[2] and r[2][k]["fps"] >= r[1]) for k in keys}
    print(f"{'clears bar':16s} " + "".join(f"{str(n[k]) + '/' + str(len(rows)):>24s}" for k in keys))
    ls = [(r[0], r[2]["r112_lightspec"]["fps"] / r[2]["r112"]["fps"])
          for r in rows if "r112_lightspec" in r[2]]
    if ls:
        print(f"\nlightspec at r112: median {st.median([x[1] for x in ls]):.3f}x "
              f"[{min(x[1] for x in ls):.3f}, {max(x[1] for x in ls):.3f}] over {len(ls)} tasks")
    print("\nworst five tasks by r96/bar:")
    for e, b, R in sorted(rows, key=lambda r: r[2]["r96"]["fps"] / r[1])[:5]:
        print(f"   {e:32s} shipped {R['r256']['fps']:6.0f} ({R['r256']['fps'] / b:.2f}x)  "
              f"r96 {R['r96']['fps']:6.0f} ({R['r96']['fps'] / b:.2f}x)  bar {b:.0f}")
    # where the frame's time goes: the captured graph against the whole render
    print("\nray-cast share of the frame (captured graph alone / end-to-end), median over tasks:")
    for k in keys:
        v = [r[2][k]["fps"] / r[2][k]["graph_fps"] for r in rows if k in r[2]]
        if v:
            print(f"   {k:6s} end-to-end is {st.median(v):.2f}x the bare graph rate "
                  f"-> ray-cast is {100 * st.median(v):.0f} % of the frame")


def training():
    print("\n### per-task correctors, held-out EPISODES (disjoint seeds), fp32 in-loop metric\n")
    rows = []
    for p in sorted(glob.glob(f"{L}/ckpt/pt_*.json")):
        d = json.load(open(p))
        if "final" not in d:
            continue
        f, b = d["final"]["agg"], d["bilinear_baseline"]["agg"]
        s = (d.get("final_shrink0.5") or {}).get("agg") or {}
        rows.append(dict(env=d["envs"][0], net=d["args"]["net"], tag=d["args"]["tag"],
                         bil=b["mean_abs"], mean=f["mean_abs"], w8=f["within8_frac"],
                         ex=f["exact_frac"], ie=f["interior_exact_frac"],
                         s_mean=s.get("mean_abs"), s_ie=s.get("interior_exact_frac")))
    for (net, tag) in sorted({(r["net"], r["tag"]) for r in rows}):
        sel = [r for r in rows if r["net"] == net and r["tag"] == tag]
        m = [r["mean"] for r in sel]
        print(f"{net:26s} {tag:14s} n={len(sel):2d}  bilinear {st.median([r['bil'] for r in sel]):.2f}"
              f" -> mean {st.median(m):.2f} [{min(m):.2f},{max(m):.2f}]  "
              f"w8 {st.median([r['w8'] for r in sel]):.4f}  exact {st.median([r['ex'] for r in sel]):.3f}"
              f"  iexact {st.median([r['ie'] for r in sel]):.4f}"
              + (f"  | shrink0.5 mean {st.median([r['s_mean'] for r in sel]):.2f}"
                 f" iexact {st.median([r['s_ie'] for r in sel]):.4f}" if sel[0]["s_mean"] else ""))


def multi():
    print("\n### one corrector for six tasks against six specialists (held-out episodes)\n")
    per = {}
    for p in glob.glob(f"{L}/ckpt/pt_*.json"):
        d = json.load(open(p))
        if "final" in d:
            per[(d["envs"][0], d["args"]["net"], d["args"]["tag"])] = d["final"]["agg"]["mean_abs"]
    for p in sorted(glob.glob(f"{L}/ckpt/multi6_*.json")):
        d = json.load(open(p))
        if "final" not in d:
            continue
        net, tag = d["args"]["net"], d["args"]["tag"]
        print(f"-- {net} {tag}")
        gaps = []
        for e, v in d["final"]["per_env"].items():
            o = per.get((e, net, tag))
            g = (v["mean_abs"] / o - 1) if o else None
            gaps.append(g) if g is not None else None
            print(f"   {e.replace('MetaWorld', ''):14s} specialist {o if o else float('nan'):.3f}"
                  f"   one-model {v['mean_abs']:.3f}   "
                  + (f"{100 * g:+.1f} %" if g is not None else ""))
        if gaps:
            print(f"   one model costs {100 * st.mean(gaps):+.1f} % of mean |diff| on average, "
                  f"worst {100 * max(gaps):+.1f} %")


def fps():
    print("\n### throughput with the corrector inside, batch 1024, 25 reps, lightspec ON\n")
    files = sorted(glob.glob(f"{L}/fps_*.json"))
    nets = []
    for p in files:
        for r in json.load(open(p))["rows"]:
            if "net" in r and r["net"] not in nets:
                nets.append(r["net"])
    print(f"{'task':16s} {'bar':>6s} {'ship':>6s} " + "".join(f"{n[:22]:>24s}" for n in nets))
    for p in files:
        d = json.load(open(p))
        R = {(r.get("net"), r.get("lightspec")): r for r in d["rows"] if "net" in r}
        ship = [r for r in d["rows"] if r["config"] == "shipped_256"][0]["fps"]
        cells = []
        for n in nets:
            r = R.get((n, True))
            cells.append(f"{r['fps']:8.0f} {'ok' if r['clears_bar'] else '--'} "
                         f"{r['fps'] / d['stored_bar']:5.2f}x" if r else "")
        print(f"{d['env'].replace('MetaWorld', '')[:15]:16s} {d['stored_bar']:6.0f} {ship:6.0f} "
              + "".join(f"{c:>24s}" for c in cells))
    print("\ncorrector cost, standalone fp16 CUDA-graph replay at batch 1024 (us/frame):")
    seen = {}
    for p in files:
        d = json.load(open(p))
        for r in d["rows"]:
            if "net" in r:
                seen.setdefault(r["net"], []).append(r["corrector_us_standalone"])
    for n, v in seen.items():
        print(f"   {n:26s} {st.median(v):5.2f} [{min(v):.2f},{max(v):.2f}] over {len(v)} runs")


def evals():
    for p in sorted(glob.glob(f"{L}/eval_*.json")):
        d = json.load(open(p))
        print(f"\n### {os.path.basename(p)}  net {d['net']} r{d['render_size']} "
              f"params {d['params']} trained on {len(d['trained_on'])} task(s)")
        for split in ("heldout", "decl"):
            rows = [r for r in d["rows"] if r.get("split") == split]
            if not rows:
                continue
            envs = sorted({r["env"] for r in rows})
            print(f"-- {split}")
            print(f"   {'env':16s} {'seen':5s} {'config':22s} {'mean':>6s} {'w8':>7s} {'ssim':>7s} "
                  f"{'iexact':>7s} {'exact':>6s} {'p999':>5s} {'max':>4s} {'dyn':>6s} {'stat':>6s}")
            for e in envs:
                for r in [x for x in rows if x["env"] == e]:
                    print(f"   {e.replace('MetaWorld', '')[:15]:16s} {str(r['seen_task']):5s} "
                          f"{r['config']:22s} {r['mean_abs']:6.3f} {r['within8_frac']:7.4f} "
                          f"{r['ssim']:7.4f} {r['interior_exact_frac']:7.4f} {r['exact_frac']:6.3f} "
                          f"{r['p999']:5.0f} {r['max_abs']:4.0f} {r['mean_dynamic']:6.2f} "
                          f"{r['mean_static']:6.2f}")
        for r in d["rows"]:
            if r.get("kind") == "control":
                print(f"   control {r['config']:12s} residual mean {r['residual_mean']:6.2f}/255  "
                      f"max {r['residual_max']:5.1f}  moved>8 {100 * r['frac_moved_gt8']:5.1f} %")
            if r.get("kind") == "throughput":
                print(f"   throughput {r['env'].replace('MetaWorld', ''):14s} shipped "
                      f"{r['shipped_256_fps']:7.0f}  render {r['render_only_fps']:7.0f}  "
                      f"+corrector {r['corrector_fps']:7.0f} ({r['corrector_us_per_frame']:.1f} us)"
                      f"  bar {r['stored_bar']}  clears {r['clears_bar']}")


def window():
    print("\n### the two window tasks: the translucent second pass is the whole gap\n")
    for p in sorted(glob.glob(f"{L}/window_*.json")):
        d = json.load(open(p))
        bar = BARS[d["env"]]["stored"]
        print(f"-- {d['env']} (bar {bar:.0f})")
        for r in d["rows"]:
            if "fps" not in r:
                continue
            print(f"   {r['config']:26s} {r['fps']:7.0f} f/s ({r['fps'] / bar:.2f}x bar)  "
                  f"mean {r['mean_abs']:5.3f}  ssim {r['ssim']:.4f}  "
                  f"iexact {r['interior_exact_frac']:.5f}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", default="")
    a = ap.parse_args()
    want = a.only.split(",") if a.only else ["frontier", "window", "training", "multi", "fps",
                                             "evals"]
    for name in want:
        globals()[name]()
