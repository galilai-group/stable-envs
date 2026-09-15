"""Judge a paired CPU tier measurement, or quarantine it.

The absolute throughput of an Oscar batch node is not a property of this renderer: the first attempt
measured an unchanged uint8 gather at 6,219 f/s where a laptop measures 126,845. What IS a property
of the renderer is the RATIO between two trees measured in the same job on the same node, and that
ratio is only trustworthy if the node held still while it was taken.

So the baseline is measured twice, on either side of the branch. Their spread is the node's noise
floor. Above 1.25x the number is quarantined rather than reported -- a tier taken on a node that
moved under the measurement is not a tier.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

OUT = "examples/results/step7/craftax-classic-full/sweep"
SPREAD_MAX = 1.25


def load(pat):
    f = sorted(glob.glob(os.path.join(OUT, pat)))
    if not f:
        return None
    return json.load(open(f[-1]))


def cell(rec):
    """(certified batch, fps) for CraftaxClassic out of a bench_step3 record."""
    for env in (rec.get("envs") or rec.get("results") or []):
        if env.get("env") != "CraftaxClassic" and env.get("name") != "CraftaxClassic":
            continue
        best = None
        for c in env.get("cells", []):
            if "fps" in c and c.get("fps"):
                if best is None or c["fps"] > best[1]:
                    best = (c.get("batch"), c["fps"])
        return best
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    a = ap.parse_args()
    recs = {k: load(f"bench_step3_cpu_{k}_{a.host}.json") for k in ("baseA", "branch", "baseB")}
    out = {"host": a.host, "spread_max": SPREAD_MAX,
           "loadavg_at_verdict": list(os.getloadavg())}
    cells = {k: (cell(v) if v else None) for k, v in recs.items()}
    out["cells"] = {k: (None if c is None else {"batch": c[0], "fps": c[1]})
                    for k, c in cells.items()}
    if not all(cells.values()):
        out["verdict"] = "INCOMPLETE"
        out["why"] = "a leg produced no CraftaxClassic cell; nothing to judge"
    else:
        a_, b_ = cells["baseA"][1], cells["baseB"][1]
        spread = max(a_, b_) / min(a_, b_)
        out["baseline_spread"] = round(spread, 4)
        out["ratio_branch_over_baseline"] = round(
            cells["branch"][1] / ((a_ + b_) / 2), 4)
        if spread > SPREAD_MAX:
            out["verdict"] = "QUARANTINED"
            out["why"] = (f"the two baseline legs differ by {spread:.2f}x (> {SPREAD_MAX}); the node "
                          f"moved under the measurement, so neither leg describes the renderer")
        else:
            out["verdict"] = "USABLE"
            out["why"] = (f"the two baseline legs agree to {spread:.3f}x, so the node held still and "
                          f"the branch/baseline ratio is a property of the renderer")
    p = os.path.join(OUT, f"paired_verdict_{a.host}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
