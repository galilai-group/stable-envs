"""Commit (f) assembly: label the quiet-window tier shards with the FROZEN Step-3 evaluator, apply the
paired-protocol gate (A/C spread <= 1.25x on ours and on the official, floor vs the certified
node2333 numbers), aggregate the kind-cost runs into the canonical file, and print one summary.

    python scratch/step7/dmc-quadruped/assemble_f.py --gpu phase1/tier_gpu --cpu phase1/tier_cpu

Reads only; writes ``tiers_*.json`` beside each shard, ``tier_gate_<device>.json`` per device and,
when kind-cost runs are present, ``phase1/cpu_oscar/kind_cost_ellipsoid.json`` (the canonical
aggregate the bounds test reads) -- replacing the loaded-node aggregate only if the new runs carry a
lower load. Nothing is relabelled by hand: the label is the evaluator's, and where it says
``unknown`` (no census entry for a cell that landed after the Step-4 freeze) the would-be label under
the class-S ladder is computed by the evaluator's own ``_tier_from`` and reported as such.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / "scratch" / "step3"))
PY = sys.executable
KNOWN_CPU = {"DMCCheetahRun": (115.6, 23.4), "DMCHopperHop": (187.8, 27.4),
             "DMCWalkerWalk": (63.1, 25.0), "DMCHumanoidStand": (45.8, 10.9)}     # node2333, cpu-3d findings


def label(shard: Path) -> dict:
    out = shard.with_name("tiers_" + shard.name.replace("paired_", ""))
    subprocess.run([PY, str(ROOT / "scratch/step6/cpu-3d/tiers_after.py"), "--shard", str(shard), "--out", str(out)],
                   check=True, capture_output=True, env=dict(os.environ, MUJOCO_GL=os.environ.get("MUJOCO_GL", "glfw")))
    return json.loads(out.read_text())


def rows(shard: Path) -> dict:
    s = json.loads(shard.read_text())
    return s, {e["env"]: e for e in s.get("envs", [])}


def best(e):
    return max(c["fps"] for c in e["cells"] if c.get("fps"))


def official(e):
    return ((e.get("official_cpu") or {}).get("paired") or {}).get("fps_median")


def gate(device: str, d: Path) -> dict:
    import certify as CT
    legs = {}
    for tag in ("A_baseline", "B_branch", "C_baseline"):
        fs = sorted(d.glob(f"paired_{tag}_*.json"))
        if fs:
            legs[tag[0]] = fs[-1]
    rep = {"device": device, "legs": {k: str(v.relative_to(ROOT)) for k, v in legs.items()}, "four": {}, "quadruped": None,
           "quarantined": [str(p.relative_to(ROOT)) for p in d.glob("QUARANTINED_LOAD_*")]}
    if not legs:
        return rep
    shards = {k: rows(v) for k, v in legs.items()}
    rep["certifiable"] = {k: s[0].get("certifiable") for k, s in shards.items()}
    rep["load_at_start"] = {k: s[0].get("load_average_at_start") for k, s in shards.items()}
    labels = {k: {l["env"]: l for l in label(v)["labels"]} for k, v in legs.items()}
    worst = 1.0
    for env in KNOWN_CPU:
        r = {}
        for k, (s, m) in shards.items():
            if env in m:
                r[f"{k}_best"] = best(m[env]); r[f"{k}_official"] = official(m[env])
                r[f"{k}_tier"] = labels[k][env]["tier"]; r[f"{k}_S"] = labels[k][env]["evidence"].get("ratio")
        if "A_best" in r and "C_best" in r:
            r["spread_ours_AC"] = max(r["A_best"] / r["C_best"], r["C_best"] / r["A_best"])
            worst = max(worst, r["spread_ours_AC"])
            if r.get("A_official") and r.get("C_official"):
                r["spread_official_AC"] = max(r["A_official"] / r["C_official"], r["C_official"] / r["A_official"])
                worst = max(worst, r["spread_official_AC"])
        if device == "cpu" and "A_best" in r:
            kb, ko = KNOWN_CPU[env]
            r["floor_ratio_ours_vs_node2333"] = r["A_best"] / kb
            r["floor_ratio_official_vs_node2333"] = (r["A_official"] / ko) if r.get("A_official") else None
        rep["four"][env] = r
    rep["worst_spread_AC"] = worst
    rep["spread_ok"] = worst <= 1.25
    if "B" in shards and "DMCQuadrupedWalk" in shards["B"][1]:
        e = shards["B"][1]["DMCQuadrupedWalk"]; l = labels["B"]["DMCQuadrupedWalk"]
        ev = l["evidence"]
        q = {"size": e["size"], "cells": [(c["batch"], round(c["fps"], 1), c.get("loadavg")) for c in e["cells"]],
             "best": best(e), "official_same_node": official(e),
             "official_spread": ((e.get("official_cpu") or {}).get("paired") or {}).get("spread"),
             "decode_fps_per_core": (e.get("decode") or {}).get("decode_fps_per_core"),
             "evaluator_tier": l["tier"], "qualifiers": l["qualifiers"], "S": ev.get("ratio"), "cores": ev.get("cores")}
        if l["tier"] == "unknown" and ev.get("ratio") is not None:
            S = CT.SPEC
            if device == "gpu":
                order = ["GPU_EXCEPTIONAL", "GPU_FAST", "GPU_FASTER", "GPU_PARITY", "GPU_REGRESSION"]
                q["would_be_tier_class_S"] = CT._tier_from(S["gpu"]["tiers"]["S"], order, ev["ratio"], ev.get("cores"), "R", "cores")
            else:
                order = ["CPU_EXCEPTIONAL", "CPU_FAST", "CPU_FASTER", "CPU_PARITY", "CPU_REGRESSION"]
                q["would_be_tier"] = CT._tier_from(S["cpu"]["tiers"], order, ev["ratio"], ev.get("cores"), "S_cpu", "cores_cpu")
            q["why_unknown"] = "no census entry for the cell (Step-8 freeze item); the ladder walk above is the evaluator's own"
        rep["quadruped"] = q
    (d / f"tier_gate_{device}.json").write_text(json.dumps(rep, indent=1, default=str))
    return rep


def kind_cost(d: Path) -> dict | None:
    runs = sorted(d.glob("kind_cost_*_run[123].json"))
    if len(runs) < 3:
        return None
    agg = d / "kind_cost_quiet_agg.json"
    p = subprocess.run([PY, str(ROOT / "scratch/step6/raycast3d/r6_kind_cost_agg.py"), *map(str, runs), "--out", str(agg)],
                       capture_output=True, text=True)
    print(p.stdout[-1200:])
    a = json.loads(agg.read_text())
    loads = [l for f in (d / "load_wait_placeholder",) for l in []]
    return a


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--gpu", default=str(ROOT / "examples/results/step7/dmc-quadruped/phase1/tier_gpu"))
    ap.add_argument("--cpu", default=str(ROOT / "examples/results/step7/dmc-quadruped/phase1/tier_cpu"))
    a = ap.parse_args()
    for device, d in (("gpu", Path(a.gpu)), ("cpu", Path(a.cpu))):
        if not d.is_dir():
            print(device, "no directory yet"); continue
        rep = gate(device, d)
        print(f"== {device}: legs {list(rep['legs'])} certifiable {rep.get('certifiable')} load {rep.get('load_at_start')} "
              f"worst spread A/C {rep.get('worst_spread_AC')} ok={rep.get('spread_ok')} quarantined={rep['quarantined']}")
        for env, r in rep["four"].items():
            print(f"   {env:18s} " + " ".join(f"{k}={round(v, 3) if isinstance(v, float) else v}" for k, v in r.items()))
        print("   quadruped:", json.dumps(rep["quadruped"], default=str))
        if device == "cpu":
            k = kind_cost(d)
            if k:
                print("   kind-cost predicate:", k["predicate"], "ELLIPSOID worst", k["ratio_worst_per_run"].get("ELLIPSOID"))


if __name__ == "__main__":
    main()
