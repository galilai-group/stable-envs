"""Judge a paired CPU tier measurement, or quarantine it.

The absolute throughput of an Oscar batch node is not a property of this renderer: one attempt
measured an unchanged uint8 gather at 6,219 f/s where a laptop measures 126,845. What IS a property
of the renderer is the RATIO between two trees measured in the same job on the same node -- but only
if the node was actually running them.

**Three gates, because the first two are not enough and job 6395321 proved it.** That job's baseline
legs agreed to 1.178x, passing the spread gate, while measuring 1,484 and 1,259 f/s -- about a
hundredth of what Step 6 certified for the SAME TREE on this cluster, at loadavg 15.55 on a 4-cpu
allocation. Its ratio came out 0.444 where a laptop measures 0.011 for the same pair, because when
both legs are dominated by scheduler contention they converge towards each other. **A contended node
does not merely add noise to the ratio; it moves the ratio towards 1 and makes the slower tree look
better than it is.** Spread alone cannot see that, since contention that is steady is still steady.

  1. **spread** -- the two baseline legs, either side of the branch, within 1.25x. Catches a node
     that moved under the measurement.
  2. **baseline floor** -- the baseline leg re-measures a tree whose answer is already known
     (115,166 f/s at the certified batch, Step 6, this cluster). Within 4x of it, or the measurement
     is not of the renderer. Catches steady contention, which gate 1 cannot -- and caught something
     better on its first run: job 6395321 pointed at ``$SCR/se6/baseline``, which is the PRE-Step-6
     tree rather than this workstream's parent, and measured it at 1,371 f/s. That looked exactly
     like contention (and was diagnosed as such) until this gate compared it with the known answer.
     A leg that re-measures a known tree has a known answer, and checking it is how a wrong baseline
     is caught rather than reported.
  3. **load** -- loadavg per allocated cpu under 1.5. Catches the cause directly, and is the one a
     reader can check against the log without re-deriving anything.

A number that fails any of them is reported as QUARANTINED, not as a tier.
"""
from __future__ import annotations

import argparse
import glob
import json
import os

OUT = "examples/results/step7/craftax-classic-full/sweep"
SPREAD_MAX = 1.25
#: The Step-6 certified CPU throughput for this env on this cluster, at its certified batch
#: (``examples/results/step6/tilemap/findings.md`` section 5: 115,166 f/s on node1831, 154,086 on the
#: quieter node2333). The BASELINE leg re-measures that same tree, so it is a known answer, and a
#: node that cannot reproduce it to within this factor is not measuring the renderer.
BASELINE_KNOWN_FPS = 115166.0
BASELINE_FLOOR_FRAC = 0.25
#: Load average per allocated CPU above which the node is oversubscribed. A batch allocation of 4
#: cpus on a node at loadavg 15 is not four cores.
LOAD_PER_CPU_MAX = 1.5


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
        base_mean = (a_ + b_) / 2
        load1 = out["loadavg_at_verdict"][0]
        cpus = int(os.environ.get("SLURM_CPUS_PER_TASK") or os.cpu_count() or 1)
        load_per_cpu = load1 / max(cpus, 1)
        out.update(baseline_spread=round(spread, 4), baseline_mean_fps=round(base_mean),
                   baseline_known_fps=BASELINE_KNOWN_FPS,
                   baseline_frac_of_known=round(base_mean / BASELINE_KNOWN_FPS, 4),
                   cpus_allocated=cpus, load_per_cpu=round(load_per_cpu, 2),
                   ratio_branch_over_baseline=round(cells["branch"][1] / base_mean, 4))
        fail = []
        if spread > SPREAD_MAX:
            fail.append(f"the two baseline legs differ by {spread:.2f}x (> {SPREAD_MAX}): the node "
                        f"moved under the measurement")
        if base_mean < BASELINE_KNOWN_FPS * BASELINE_FLOOR_FRAC:
            fail.append(f"the baseline leg re-measures a tree Step 6 certified at "
                        f"{BASELINE_KNOWN_FPS:,.0f} f/s on this cluster and got {base_mean:,.0f} "
                        f"({base_mean / BASELINE_KNOWN_FPS:.3f}x): the node is not measuring the "
                        f"renderer, and a ratio taken under that contention is biased towards 1")
        if load_per_cpu > LOAD_PER_CPU_MAX:
            fail.append(f"loadavg {load1:.2f} over {cpus} allocated cpus is {load_per_cpu:.2f} per "
                        f"cpu (> {LOAD_PER_CPU_MAX}): the allocation is oversubscribed")
        out["verdict"] = "QUARANTINED" if fail else "USABLE"
        out["why"] = ("; ".join(fail) if fail else
                      f"baseline legs agree to {spread:.3f}x, the baseline reproduces its known "
                      f"{BASELINE_KNOWN_FPS:,.0f} f/s to {base_mean / BASELINE_KNOWN_FPS:.2f}x, and "
                      f"loadavg is {load_per_cpu:.2f} per allocated cpu")
        out["gates"] = {"spread": spread <= SPREAD_MAX,
                        "baseline_floor": base_mean >= BASELINE_KNOWN_FPS * BASELINE_FLOOR_FRAC,
                        "load": load_per_cpu <= LOAD_PER_CPU_MAX}
    p = os.path.join(OUT, f"paired_verdict_{a.host}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
