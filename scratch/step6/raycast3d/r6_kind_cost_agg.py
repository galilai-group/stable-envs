"""Aggregate independent runs of r5_kind_cost.py and put an error bar on the RATIO.

The margin in the first version was ``1 + 3 * worst_rsd``, with ``rsd`` the spread of an individual
timing. That is the wrong statistic and a re-run proved it: the same measurement taken on a busier
node returned worst_rsd 0.25 instead of 0.076, which moved the margin from 1.23 to 1.75 and would
have un-bounded every kind -- a predicate decided by how loaded the node was that evening.

The quantity the decision uses is the RATIO of two medians, not a single timing, and its spread is
what has to be bounded. So: run the whole benchmark N times independently and report, per kind, the
spread of the worst-case ratio ACROSS RUNS. That is an empirical error bar on the number the rule
consumes.
"""
from __future__ import annotations

import json
import statistics
import sys
from pathlib import Path

KINDS = ("PLANE", "SPHERE", "BOX", "CYLINDER", "CAPSULE")


def main(argv) -> int:
    out = None
    argv = list(argv)
    if "--out" in argv:
        i = argv.index("--out")
        out = Path(argv[i + 1])
        del argv[i:i + 2]
    paths = [Path(p) for p in argv[1:]]
    if not paths:
        print("usage: r6_kind_cost_agg.py [--out agg.json] run1.json run2.json ...")
        return 2
    runs = [json.loads(p.read_text()) for p in paths]
    print(f"{len(runs)} run(s), shape={runs[0].get('shape')}, "
          f"host={runs[0]['host'].split('.')[0]}, job={runs[0]['slurm_job']}")
    print(f"\n{'kind':9s} {'worst ratio per run':>34s} {'min':>7s} {'max':>7s} {'spread':>8s} "
          f"{'> 1 in every run?':>18s}")
    table = {}
    for k in KINDS:
        if "ratio_min" not in runs[0]["predicate_detail"][k]:
            table[k] = False
            print(f"{k:9s} {'no finite bounding sphere':>34s} {'-':>7s} {'-':>7s} {'-':>8s} "
                  f"{'n/a (unbounded)':>18s}")
            continue
        w = [r["predicate_detail"][k]["ratio_min"] for r in runs]
        lo, hi = min(w), max(w)
        # THE DECISION: the necessary condition is cost_bound < cost_intersect, i.e. ratio > 1, and it
        # has to hold in EVERY run and at EVERY width. No tuned margin: "the cheap side never reaches
        # 1 and the dear side never falls below it" is a statement the data can support or refuse.
        table[k] = lo > 1.0
        print(f"{k:9s} {'  '.join(f'{x:.2f}' for x in w):>34s} {lo:7.2f} {hi:7.2f} "
              f"{(hi - lo) / lo * 100:7.1f}% {str(table[k]):>18s}")
    sph = max(max(r["predicate_detail"]["SPHERE"]["ratio_max"] for r in runs), 0)
    bounded = [k for k in KINDS if table.get(k)]
    if bounded:
        worst_bounded = min(min(r["predicate_detail"][k]["ratio_min"] for r in runs) for k in bounded)
        print(f"\nSEPARATION: the dearest UNBOUNDED kind reaches {sph:.2f} at its best; the cheapest "
              f"BOUNDED kind never falls below {worst_bounded:.2f}.")
        print(f"  gap = {worst_bounded / sph:.2f}x, and the line at 1.0 sits inside it.")
    print(f"\nPREDICATE: {{{', '.join(sorted(bounded))}}}")
    if out is not None:
        # THE AUTHORITATIVE TABLE. Each individual run file carries a ``predicate`` computed with the
        # SUPERSEDED margin rule, and those three disagree with each other -- all-False, capsule-only,
        # all-False -- while their ratios agree to 1-3 %. That is the whole argument for this file
        # existing: the runs are the measurement, the rule is separate, and the rule that stands is
        # the necessary condition rather than a margin scaled by how loaded the node was.
        agg = {
            "tool": "r6_kind_cost_agg",
            "supersedes": "the per-run `predicate` fields, which use the superseded "
                          "1 + 3*worst_rsd margin",
            "rule": "bounded(kind) iff ns_intersect/ns_bound > 1 at every measured (B, N) in EVERY "
                    "independent run: cost_bound < cost_intersect is the condition that holds for "
                    "any scene, since the fraction of rays kept is >= 0",
            "shape": runs[0].get("shape"),
            "runs": [str(q) for q in paths],
            "host": runs[0]["host"], "slurm_job": runs[0]["slurm_job"],
            "threads": runs[0]["threads"], "cpu": runs[0].get("cpu"),
            "predicate": {k: bool(table.get(k, False)) for k in KINDS},
            "ratio_worst_per_run": {
                k: [r["predicate_detail"][k]["ratio_min"] for r in runs]
                for k in KINDS if "ratio_min" in runs[0]["predicate_detail"][k]},
            "ratio_best_per_run": {
                k: [r["predicate_detail"][k]["ratio_max"] for r in runs]
                for k in KINDS if "ratio_max" in runs[0]["predicate_detail"][k]},
            "separation": {"dearest_unbounded_at_its_best": sph,
                           "cheapest_bounded_at_its_worst": worst_bounded if bounded else None,
                           "gap_x": (worst_bounded / sph) if bounded and sph else None},
        }
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps(agg, indent=1))
        print("wrote", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
