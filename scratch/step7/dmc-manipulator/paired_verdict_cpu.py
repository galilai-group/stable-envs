"""Judge the paired CPU tier measurement for DMCManipulatorBringBall, or quarantine it.

The three gates are W1's (``scratch/step7/craftax-classic-full/paired_verdict.py``), reused rather
than re-derived, because the reasoning behind them was paid for by two quarantined jobs there:

  1. **spread** -- the two baseline legs, either side of the branch leg, within 1.25x. Catches a node
     that MOVED under the measurement.
  2. **baseline floor** -- the baseline leg re-measures a tree whose answer is already known, and must
     land within 4x of it. This is the gate that catches STEADY contention, which spread cannot see:
     when both legs are dominated by scheduler contention they converge towards each other, so the
     ratio moves towards 1 and the slower tree looks better than it is.
  3. **load** -- ``/proc/loadavg`` per NODE core under 1.5. Per node core, because loadavg is a
     whole-machine number and dividing it by the allocation's cpu count overstates it by one to two
     orders of magnitude.

WHAT IS DIFFERENT HERE, and it changes what the ratio means. ``DMCManipulatorBringBall`` does not
exist in the baseline tree -- it is the cell this branch adds -- so there is no before/after ratio to
take for it and none is reported. The baseline legs instead re-measure a FLOOR CELL that exists in
both trees and whose Step-6 CPU throughput is on record, and their job is solely to establish that the
NODE was measuring a renderer at all. The branch leg then carries the number that is being declared.
That is the honest shape: the gates certify the machine, not a speedup.

``DMCReacherHard`` is the floor cell. It is the same family, the same backend (plate + raycast3d) and
the same 84 px / 1-thread configuration Step 6 certified, so a node that cannot reproduce it cannot be
trusted with the new cell either.
"""
from __future__ import annotations

import argparse
import json
import os

SPREAD_MAX = 1.25
#: Step-6's certified CPU throughput for the FLOOR cell on this cluster, at 84 px, one thread, at its
#: best batch (B=16). Taken from the shipped configuration -- ``profile_after_r5/bounds_on`` -- which
#: measured 111.0 f/s on node1831 and 130.4 on the quieter node2333
#: (``examples/results/step6/raycast3d/profile_after_r5/``). The LOWER of the two is used, so the gate
#: is conservative about what a healthy node owes.
FLOOR_CELL = "DMCReacherHard"
FLOOR_KNOWN_FPS = 111.0
FLOOR_FRAC = 0.25
LOAD_PER_CORE_MAX = 1.5


def _node_cores(rec) -> int:
    return int(rec.get("node_cores") or os.sysconf("SC_NPROCESSORS_ONLN"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True)
    ap.add_argument("--host", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    legs = {}
    for k in ("baseA", "branch", "baseB"):
        p = os.path.join(a.dir, f"cpu_tier_{k}_{a.host}.json")
        legs[k] = json.load(open(p)) if os.path.exists(p) else None

    out = {"host": a.host, "spread_max": SPREAD_MAX, "floor_cell": FLOOR_CELL,
           "floor_known_fps": FLOOR_KNOWN_FPS, "floor_frac": FLOOR_FRAC,
           "load_per_core_max": LOAD_PER_CORE_MAX,
           "note": ("the branch cell does not exist in the baseline tree, so NO before/after ratio is "
                    "reported for it; the baseline legs measure the floor cell and their only job is "
                    "to establish that this node was measuring a renderer")}
    out["legs"] = {k: (None if v is None else
                       {"env": v.get("env"), "size": v.get("size"),
                        "certified_batch": v.get("certified_batch"), "fps": v.get("fps"),
                        "official_fps": (v.get("official") or {}).get("fps"),
                        "decode_fps_per_core": (v.get("decode") or {}).get("decode_fps_per_core"),
                        "S_cpu": v.get("S_cpu"), "cores_cpu": v.get("cores_cpu"),
                        "loadavg_before": v.get("loadavg_before"),
                        "loadavg_after": v.get("loadavg_after"),
                        "error": v.get("error") or v.get("official_error")})
                   for k, v in legs.items()}

    if not all(legs.values()) or any(l.get("fps") is None for l in legs.values()):
        out["verdict"] = "INCOMPLETE"
        out["why"] = "a leg produced no throughput; nothing to judge"
        json.dump(out, open(a.out, "w"), indent=1, default=str)
        print(json.dumps(out, indent=1, default=str))
        return

    A, Br, B = legs["baseA"], legs["branch"], legs["baseB"]
    for leg, rec in (("baseA", A), ("baseB", B)):
        if rec.get("env") != FLOOR_CELL:
            out["verdict"] = "INCOMPLETE"
            out["why"] = f"{leg} measured {rec.get('env')}, not the floor cell {FLOOR_CELL}"
            json.dump(out, open(a.out, "w"), indent=1, default=str)
            print(json.dumps(out, indent=1, default=str))
            return

    a_, b_ = A["fps"], B["fps"]
    spread = max(a_, b_) / min(a_, b_)
    base_mean = (a_ + b_) / 2
    loads = [x for rec in (A, Br, B) for x in (rec.get("loadavg_before") or [])[:1]
             ] + [x for rec in (A, Br, B) for x in (rec.get("loadavg_after") or [])[:1]]
    load1 = max(loads) if loads else 0.0
    cores = _node_cores(Br)
    lpc = load1 / max(cores, 1)

    out.update(baseline_spread=round(spread, 4), baseline_mean_fps=round(base_mean, 2),
               baseline_frac_of_known=round(base_mean / FLOOR_KNOWN_FPS, 4),
               node_cores=cores, loadavg_worst=round(load1, 2), load_per_core=round(lpc, 3),
               branch_env=Br.get("env"), branch_fps=round(Br["fps"], 2),
               branch_certified_batch=Br.get("certified_batch"), branch_size=Br.get("size"),
               S_cpu=(round(Br["S_cpu"], 4) if Br.get("S_cpu") else None),
               cores_cpu=(round(Br["cores_cpu"], 4) if Br.get("cores_cpu") else None))

    fail = []
    if spread > SPREAD_MAX:
        fail.append(f"the two baseline legs differ by {spread:.2f}x (> {SPREAD_MAX}): the node moved "
                    f"under the measurement")
    if base_mean < FLOOR_KNOWN_FPS * FLOOR_FRAC:
        fail.append(f"the baseline leg re-measures {FLOOR_CELL}, which Step 6 certified at "
                    f"{FLOOR_KNOWN_FPS:,.0f} f/s on this cluster, and got {base_mean:,.1f} "
                    f"({base_mean / FLOOR_KNOWN_FPS:.3f}x): this node is not measuring the renderer")
    if lpc > LOAD_PER_CORE_MAX:
        fail.append(f"loadavg {load1:.2f} over {cores} node cores is {lpc:.2f} per core "
                    f"(> {LOAD_PER_CORE_MAX}): the node is oversubscribed")
    out["gates"] = {"spread": spread <= SPREAD_MAX,
                    "baseline_floor": base_mean >= FLOOR_KNOWN_FPS * FLOOR_FRAC,
                    "load": lpc <= LOAD_PER_CORE_MAX}
    out["verdict"] = "QUARANTINED" if fail else "USABLE"
    out["why"] = ("; ".join(fail) if fail else
                  f"baseline legs agree to {spread:.3f}x, the floor cell reproduces its known "
                  f"{FLOOR_KNOWN_FPS:,.0f} f/s to {base_mean / FLOOR_KNOWN_FPS:.2f}x, and loadavg is "
                  f"{lpc:.2f} per node core")

    # the tier, at the frozen definitions (certification.md S6.2/S6.3)
    if out["verdict"] == "USABLE" and out["S_cpu"] and out["cores_cpu"]:
        S, co = out["S_cpu"], out["cores_cpu"]
        if S >= 20 and co >= 10:
            tier = "CPU_EXCEPTIONAL"
        elif S >= 5 and co >= 1:
            tier = "CPU_FAST"
        elif S >= 2:
            tier = "CPU_FASTER"
        elif S >= 1:
            tier = "CPU_PARITY"
        else:
            tier = "CPU_REGRESSION"
        out["tier"] = tier
        out["tier_basis"] = (f"S_cpu = {S:.2f} (ours {out['branch_fps']:,.1f} f/s / official "
                             f"{out['legs']['branch']['official_fps']:,.1f} f/s), cores_cpu = "
                             f"{co:.2f} (ours / {out['legs']['branch']['decode_fps_per_core']:,.1f} "
                             f"f/s/core of JPEG-q90 decode), both at 1 thread on {a.host}")
        # the NO_REFERENCE row, reported alongside, since it is the axis an env without an official
        # renderer would be judged on and a reader should be able to see both
        out["tier_if_no_reference"] = ("CPU_EXCEPTIONAL" if co >= 10 else
                                       "CPU_FAST" if co >= 3 else
                                       "CPU_FASTER" if co >= 1 else "below CPU_FASTER")
    else:
        out["tier"] = "QUARANTINED" if out["verdict"] == "QUARANTINED" else "CPU_UNMEASURED"

    json.dump(out, open(a.out, "w"), indent=1, default=str)
    print(json.dumps(out, indent=1, default=str))


if __name__ == "__main__":
    main()
