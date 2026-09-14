"""RAYCAST-5's acceptance table: the bound off vs on, per row, against the 0.97x bar.

Separate from ``r4_summary.py`` because the question is different. R4 asked "how much does this buy";
R5 asks "is any row worse than it was", which is a PASS/FAIL with a stated bar, and the answer has to
be computed rather than eyeballed off a table where 0.96 and 1.04 look alike.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
A = ROOT / "examples/results/step6/raycast3d"
sys.path.insert(0, str(Path(__file__).resolve().parent))
import r4_summary as S                                                   # noqa: E402

BAR = 0.97


def sweep(leg):
    out, d = {}, A / "profile_after_r5" / leg
    if not d.is_dir():
        return out
    for f in sorted(d.glob("*.json")):
        n = json.loads(f.read_text())
        rows = {}
        for B, r in n["batches"].items():
            tests = (r.get("facts", {}).get("ray_primitive_tests") or {}).get("sum_per_render")
            px = int(n["size"]) ** 2
            rows[int(B)] = {"fps": r["fps"], "tests": tests,
                            "tests_per_px": (tests / (int(B) * px)) if tests else None}
        out[n["env"]] = {"size": n["size"], "from": n.get("stable_envs_from"),
                         "host": n.get("host"), "job": n.get("slurm_job"),
                         "switches": n.get("switches"), "rows": rows}
    return out


def main():
    print("=== identity ===")
    for name, f in (("cpu: PRIMITIVE_BOUNDS on vs off, 21 configs", "r5_cpu_bounds_identity.json"),
                    ("cpu: branch(bound on) vs pristine baseline ", "r5_cpu_branch_vs_baseline.json"),
                    ("cuda: branch vs pristine baseline, 21      ", "r5_cuda_pixel_diff.json")):
        d = S.load(A / "replay" / f)
        print(f"  {name}  " + ("MISSING" if d is None else
                               f"all_identical={d.get('all_identical')} n={d.get('n_compared')}"))

    print("\n=== the per-kind cost table the predicate came from ===")
    m = S.load(A / "experiments/r5_kind_cost.json")
    if m:
        print(f"  margin {m['margin']['value']:.3f} (worst rsd {m['margin']['worst_rsd']:.4f}), "
              f"{m['host'].split('.')[0]}, job {m['slurm_job']}, {m['threads']} thread(s)")
        for k, on in m["predicate"].items():
            det = m["predicate_detail"][k]
            r = (f"ns_intersect/ns_bound {det['ratio_min']:6.2f} - {det['ratio_max']:6.2f}"
                 if "ratio_min" in det else det["reason"])
            print(f"    {k:9s} bounded={str(on):5s}  {r}")

    off, on = sweep("bounds_off"), sweep("bounds_on")
    if not (off and on):
        print("\n(no r5 sweep yet)")
        return 0
    print(f"\n=== acceptance: same branch, bound off vs on, bar {BAR:.2f}x ===")
    print(f"{'env':26s} {'B':>5s} {'off fps':>9s} {'on fps':>9s} {'x':>6s} "
          f"{'tests/px off':>13s} {'on':>7s} {'verdict':>8s}")
    worst, fails = {}, []
    for env in sorted(off):
        for B in sorted(off[env]["rows"]):
            rb, ra = off[env]["rows"][B], on.get(env, {}).get("rows", {}).get(B)
            if not ra:
                continue
            x = ra["fps"] / rb["fps"]
            ok = x >= BAR
            if not ok:
                fails.append((env, B, x))
            worst[env] = min(worst.get(env, 9e9), x)
            tb, ta = rb["tests_per_px"], ra["tests_per_px"]
            print(f"{env:26s} {B:5d} {rb['fps']:9.1f} {ra['fps']:9.1f} {x:6.2f} "
                  f"{(tb or 0):13.2f} {(ta or 0):7.2f} {'ok' if ok else 'FAIL':>8s}")
    print(f"\n{'env':26s} {'worst x over batches':>22s}")
    for env in sorted(worst):
        print(f"{env:26s} {worst[env]:22.2f}  {'ok' if worst[env] >= BAR else 'FAIL'}")
    print(f"\nACCEPTANCE: {'PASS' if not fails else 'FAIL'} "
          f"({len(fails)} row-batch pair(s) below {BAR:.2f}x)")
    for env, B, x in fails:
        print(f"  below bar: {env} B={B} {x:.2f}x")

    print("\n=== provenance, per row ===")
    for leg, d in (("off", off), ("on", on)):
        for env in sorted(d):
            v = d[env]
            sw = (v.get("switches") or {}).get("PRIMITIVE_BOUNDS")
            print(f"{env:26s} {leg:4s} {str(v['host']).split('.')[0]:10s} job {v['job']} "
                  f"PRIMITIVE_BOUNDS={sw}  {v['from']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
