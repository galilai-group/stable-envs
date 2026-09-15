"""Judge a paired CPU tier measurement for a cell that has no baseline counterpart.

**The gates, their constants and the reasoning behind them are W1's**
(``scratch/step7/craftax-classic-full/paired_verdict.py`` on the integration branch, commit
`84ecb03`), and they are not re-derived here. What differs is what the pair is *for*.

W1 measures the same env on two trees and reports the RATIO, because ``CraftaxClassic`` exists on
both. ``CraftaxFull`` is a **new cell**: it does not exist on ``$SCR/se7/baseline`` at all, so there
is no before to compare against and no ratio to report. The baseline legs are therefore used purely
as a **node-qualification instrument** — they re-measure a tree whose answer on this cluster is
already known, on either side of the branch leg, and they say whether the node was in a state where
*any* absolute throughput means anything. The branch leg's number is then reported as the tier, or
quarantined.

The three gates, unchanged:

  1. **spread** — the two baseline legs within 1.25x. Catches a node that moved under the
     measurement.
  2. **baseline floor** — the baseline leg re-measures ``CraftaxClassic`` on ``$SCR/se7/baseline``
     (76dc511, Step 6 closed), which Step 6 certified at 115,166 f/s on this cluster. Within 4x, or
     the node is not measuring a renderer. This is the gate that knows what the answer should be, and
     it is the one that matters: a contended node does not merely add noise, it drags every leg
     towards the same number, which spread alone cannot see.
  3. **load** — ``/proc/loadavg`` per **NODE** core under 1.5. Per node core, not per allocated cpu:
     loadavg is a whole-machine number and an Oscar batch node has 48-192 cores.

``$SCR/se7/baseline`` and never ``$SCR/se6/baseline`` — the latter is the PRE-Step-6 tree, whose
``CraftaxClassic`` is the per-frame blend renderer Step 6 replaced. W1's job 6395321 used it by
mistake, measured 1,371 f/s and read as contention on a node that was fine.

    python scratch/step7/craftax-full/paired_verdict.py --host $(hostname) [--loadavg L --nproc N]
"""
from __future__ import annotations

import argparse
import glob
import json
import os

OUT = "examples/results/step7/craftax-full/sweep"
SPREAD_MAX = 1.25
#: Step 6's certified CPU throughput for CraftaxClassic on this cluster at its certified batch
#: (``examples/results/step6/tilemap/findings.md``: 115,166 f/s on node1831, 154,086 on node2333).
#: The baseline legs re-measure that same tree, so this is a KNOWN answer.
BASELINE_KNOWN_FPS = 115166.0
BASELINE_FLOOR_FRAC = 0.25
LOAD_PER_CORE_MAX = 1.5
BASELINE_ENV = "CraftaxClassic"
BRANCH_ENV = "CraftaxFull"


def _node_cores() -> int:
    """The NODE's online core count. ``os.cpu_count()`` is right here and ``sched_getaffinity`` is
    not: the latter reports the cgroup allocation, which is the number that must NOT be used."""
    try:
        return int(os.sysconf("SC_NPROCESSORS_ONLN"))
    except (ValueError, OSError):
        return int(os.cpu_count() or 1)


def load(pat):
    f = sorted(glob.glob(os.path.join(OUT, pat)))
    return json.load(open(f[-1])) if f else None


def cell(rec, want):
    """``(batch, fps, official_fps, decode_fps)`` for ``want`` out of a ``bench_step3`` record."""
    for env in (rec.get("envs") or rec.get("results") or []):
        if (env.get("env") or env.get("name")) != want:
            continue
        best = None
        for c in env.get("cells", []):
            if c.get("fps") and (best is None or c["fps"] > best[1]):
                best = (c.get("batch"), c["fps"])
        off = ((env.get("official_cpu") or {}).get("set_state_render") or {}).get("fps")
        dec = (env.get("decode") or {}).get("decode_fps_per_core")
        return None if best is None else (best[0], best[1], off, dec)
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--host", required=True)
    ap.add_argument("--nproc", type=int, default=0)
    ap.add_argument("--loadavg", type=float, default=0.0)
    a = ap.parse_args()
    recs = {k: load(f"bench_step3_cpu_{k}_{a.host}.json") for k in ("baseA", "branch", "baseB")}
    out = {"host": a.host, "spread_max": SPREAD_MAX,
           "baseline_env": BASELINE_ENV, "branch_env": BRANCH_ENV,
           "protocol": "the baseline legs qualify the NODE; they do not form a ratio, because "
                       "CraftaxFull is a new cell and does not exist on the baseline tree",
           "loadavg_at_verdict": ([a.loadavg] * 3 if a.loadavg else list(os.getloadavg())),
           "loadavg_source": "job log" if a.loadavg else "this process"}
    cells = {"baseA": cell(recs["baseA"], BASELINE_ENV) if recs["baseA"] else None,
             "baseB": cell(recs["baseB"], BASELINE_ENV) if recs["baseB"] else None,
             "branch": cell(recs["branch"], BRANCH_ENV) if recs["branch"] else None}
    out["cells"] = {k: (None if c is None else
                        {"batch": c[0], "fps": c[1], "official_fps": c[2], "decode_fps": c[3]})
                    for k, c in cells.items()}
    if not all(cells.values()):
        out["verdict"] = "INCOMPLETE"
        out["why"] = "a leg produced no cell; nothing to judge"
    else:
        a_, b_ = cells["baseA"][1], cells["baseB"][1]
        spread = max(a_, b_) / min(a_, b_)
        base_mean = (a_ + b_) / 2
        load1 = out["loadavg_at_verdict"][0]
        cores = a.nproc or _node_cores()
        lpc = load1 / max(cores, 1)
        br_batch, br_fps, br_off, br_dec = cells["branch"]
        out.update(baseline_spread=round(spread, 4), baseline_mean_fps=round(base_mean),
                   baseline_known_fps=BASELINE_KNOWN_FPS,
                   baseline_frac_of_known=round(base_mean / BASELINE_KNOWN_FPS, 4),
                   node_cores=cores, load_per_core=round(lpc, 2),
                   cpus_allocated=int(os.environ.get("SLURM_CPUS_PER_TASK") or 0) or None,
                   tier_fps=round(br_fps), tier_batch=br_batch)
        if br_off:
            out["S_cpu_vs_unjitted_reference"] = round(br_fps / br_off, 3)
            out["S_cpu_note"] = (
                "bench_step3's official denominator goes through cls.reference_render, which this "
                "cell (like CraftaxClassic since W1) returns UN-JITTED because the parity claim "
                "requires it. Against the jitted renderer the same ratio is ~500x smaller. NEITHER "
                "certifies anything until Sami rules; both are recorded.")
        if br_dec:
            out["cores_cpu"] = round(br_fps / br_dec, 3)
        fail = []
        if spread > SPREAD_MAX:
            fail.append(f"the two baseline legs differ by {spread:.2f}x (> {SPREAD_MAX}): the node "
                        f"moved under the measurement")
        if base_mean < BASELINE_KNOWN_FPS * BASELINE_FLOOR_FRAC:
            fail.append(f"the baseline leg re-measures a tree Step 6 certified at "
                        f"{BASELINE_KNOWN_FPS:,.0f} f/s on this cluster and got {base_mean:,.0f} "
                        f"({base_mean / BASELINE_KNOWN_FPS:.3f}x): the node is not measuring a "
                        f"renderer, and an absolute number taken under that contention is not a tier")
        if lpc > LOAD_PER_CORE_MAX:
            fail.append(f"loadavg {load1:.2f} over {cores} node cores is {lpc:.2f} per core "
                        f"(> {LOAD_PER_CORE_MAX}): the node is oversubscribed")
        out["verdict"] = "QUARANTINED" if fail else "USABLE"
        out["why"] = ("; ".join(fail) if fail else
                      f"baseline legs agree to {spread:.3f}x, the baseline reproduces its known "
                      f"{BASELINE_KNOWN_FPS:,.0f} f/s to {base_mean / BASELINE_KNOWN_FPS:.2f}x, and "
                      f"loadavg is {lpc:.2f} per node core")
        out["gates"] = {"spread": spread <= SPREAD_MAX,
                        "baseline_floor": base_mean >= BASELINE_KNOWN_FPS * BASELINE_FLOOR_FRAC,
                        "load": lpc <= LOAD_PER_CORE_MAX}
    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, f"paired_verdict_{a.host}.json")
    with open(p, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
