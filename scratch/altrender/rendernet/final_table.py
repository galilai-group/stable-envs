"""The frontier in one table: end-to-end throughput against all four declared metrics.

Two files have to be joined for a row to mean anything. ``finalfps_<env>.json`` holds the throughput,
measured for every candidate in ONE process on ONE node so the rows are comparable to each other;
``eval_*.json`` holds the fidelity, measured on state sets the training loop never saw. Joining them
by checkpoint name is the only way a reader sees the trade rather than one half of it.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

OUT = "/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet"


def load_fps(env):
    p = f"{OUT}/finalfps_{env}.json"
    if not os.path.exists(p):
        raise SystemExit(f"{p} is missing; run probe_final_fps.py {env} first")
    j = json.load(open(p))
    fps = {r["config"]: r for r in j["rows"] if "error" not in r}
    return fps, j["target"], fps["shipped_256"]["fps"]


def load_evals():
    out = {}
    for p in glob.glob(f"{OUT}/eval_*.json"):
        j = json.load(open(p))
        out[os.path.basename(p)[5:-5]] = j
    return out


def row_for(j, split, env=None):
    for r in j["splits"] + j["ood"]:
        if "error" in r:
            continue
        if r["split"] == split and (env is None or r["env"] == env):
            return r
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--env", default="RoboCasaCounterToSink")
    ap.add_argument("--split", default="heldout")
    ap.add_argument("--tag", default="", help="only evaluations run at this data tag")
    a = ap.parse_args()
    fps, target, shipped_fps = load_fps(a.env)
    evs = load_evals()

    print(f"=== {a.env}  split={a.split}  tag={a.tag or 'any'}  "
          f"stored-frame target {target} f/s  "
          f"(shipped 256 px renders at {shipped_fps:.0f} f/s)")
    print(f"{'evaluation':56s} {'fps':>7s} {'x ship':>6s} {'us':>6s} | {'mean':>6s} {'dyn':>6s} "
          f"{'stat':>6s} {'w8':>6s} {'ssim':>6s} {'iexact':>6s} {'p99.9':>6s} {'max':>4s}")
    rows = []
    for name, j in sorted(evs.items()):
        if j["env"] != a.env or (a.tag and j["tag"] != a.tag):
            continue
        r = row_for(j, a.split, a.env)
        if r is None:
            continue
        ck = os.path.basename(j["ckpt"])[:-3]
        f = fps.get(ck)
        v = r["variants"]["corrector"]
        rows.append((f["fps"] if f else j["throughput"]["render_plus_corrector"]["fps"],
                     name, v, f))
    # the baselines, taken from any evaluation of this env and split (they do not depend on the net)
    anyj = next((j for j in evs.values()
                 if j["env"] == a.env and (not a.tag or j["tag"] == a.tag)
                 and row_for(j, a.split, a.env)), None)
    if anyj:
        base = row_for(anyj, a.split, a.env)["variants"]
        for k, lbl, fk in (("shipped_256", "-- shipped 256 px MJWarp", "shipped_256"),
                           ("bilinear", "-- bilinear render->256", None),
                           ("lanczos3", "-- Lanczos-3 render->256", None),
                           ("fxaa", "-- FXAA on bilinear", None),
                           ("jpeg_q95_of_official", "-- JPEG q95 of official (storage)", None)):
            if k not in base:
                continue
            v = base[k]
            fv = fps.get(fk, {}).get("fps") if fk else None
            print(f"{lbl:56s} {(f'{fv:7.0f}' if fv else '      -')} "
                  f"{(f'{fv / shipped_fps:6.2f}' if fv else '     -')} {'':>6s} | "
                  f"{v['mean_abs']:6.3f} {v['mean_abs_dynamic']:6.3f} {v['mean_abs_static']:6.3f} "
                  f"{v['within8_frac']:6.3f} {v['ssim']:6.4f} "
                  f"{(v['interior_exact_frac'] or 0):6.3f} {v['p999']:6.1f} {v['max_abs']:4d}")
    for f_, name, v, f in sorted(rows, key=lambda z: z[2]["mean_abs"]):
        mark = "  CLEARS" if (target and f_ >= target) else ""
        print(f"{name[:56]:56s} {f_:7.0f} {f_ / shipped_fps:6.2f} "
              f"{(f['corrector_us_per_frame'] if f else float('nan')):6.2f} | "
              f"{v['mean_abs']:6.3f} {v['mean_abs_dynamic']:6.3f} {v['mean_abs_static']:6.3f} "
              f"{v['within8_frac']:6.3f} {v['ssim']:6.4f} "
              f"{(v['interior_exact_frac'] or 0):6.3f} {v['p999']:6.1f} {v['max_abs']:4d}{mark}")


if __name__ == "__main__":
    main()
