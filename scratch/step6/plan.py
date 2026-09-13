"""Rank the Step 6 workstreams from Step 3's measurements, not from an impression.

Step 6 optimises SHARED RENDER PATHS, but Step 3 measured ENVIRONMENT ROWS and Step 4 froze
RENDER-DISTINCT CONFIGURATIONS. Those are three different units and the brief is explicit that
they must not be conflated: a path is worth a workstream in proportion to how many *configurations*
inherit its improvement and how far below tier they currently sit.

So this joins them. For every backend on every device it reports how many rows are below the
frozen Step 2 target, how far, and which Step 4 configurations a change to that path would put at
risk -- which is also the Step 14 replay list, computed rather than recalled.

    python scratch/step6/plan.py --out examples/results/step6/plan.json
"""
from __future__ import annotations

import argparse
import json
from collections import Counter, defaultdict
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

#: Tiers that clear the Step 6 target, per device. Anything else is work.
GPU_OK = {"GPU_FAST", "GPU_EXCEPTIONAL"}
CPU_OK = {"CPU_FAST", "CPU_EXCEPTIONAL"}

#: backend -> the workstream branch that owns it, from section 10. A backend absent here is not
#: claimed by any workstream, which is itself a finding rather than a default.
OWNER = {
    "raycast3d": "step6/raycast3d",
    "mjwarp": "step6/cpu-3d",
    "gfxdraw": "step6/gfxdraw",
    "scene": "step6/raster2d",
    "tilemap": "step6/tilemap",
    "nvdiff": None,
}

#: Five envs declare no backend list at all in the Step 3 rows. Resolved by name rather than left
#: "(unclaimed)", which is how 20 CPU rows below tier were nearly filed under nobody.
BY_ENV = {
    "CraftaxClassic": "step6/tilemap",
    "Piecewise": "step6/raster2d",
    "PiecewiseZones": "step6/raster2d",
    "TwoRoom": "step6/raster2d",
    "TwoRoomTarget": "step6/raster2d",
}


def load_rows(p: Path) -> list[dict]:
    return json.loads(p.read_text())["labels"]


def workstream_for(backends: list[str], env: str, device: str) -> str | None:
    """Which workstream owns this row.

    Fetch splits BY DEVICE and the split is the whole point: section 10.2 owns its GPU
    update/batching pathology, section 10.3 owns the fact that it has no CPU route at all. Filing
    both under one workstream would hide an architecture problem inside an optimisation task.
    """
    if env.startswith("Fetch"):
        return "step6/fetch-gpu" if device == "gpu" else "step6/cpu-3d"
    if env in BY_ENV:
        return BY_ENV[env]
    for b in backends:
        if b in OWNER:
            return OWNER[b]
    return None


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--step3", type=Path, default=ROOT / "examples/results/step3/certification.json")
    ap.add_argument("--step4", type=Path, default=ROOT / "examples/results/step4/certification.json")
    ap.add_argument("--out", type=Path, default=ROOT / "examples/results/step6/plan.json")
    args = ap.parse_args()

    rows = load_rows(args.step3)
    # Which envs does Step 4 hold a frozen corpus for? Those are the replay obligations.
    # Step 4's `configs` is keyed BY configuration id, not a list -- iterate its items.
    frozen: dict[str, list[str]] = {}
    if args.step4.is_file():
        for cid, c in json.loads(args.step4.read_text()).get("configs", {}).items():
            frozen[cid] = list(c.get("backends") or [])

    paths: dict[tuple, dict] = defaultdict(
        lambda: {"rows": [], "below": [], "unmeasured": [], "ok": 0, "class": None,
                 "backends": set(), "blockers": Counter()})
    for r in rows:
        dev = r["device"]
        key = (workstream_for(r.get("backends", []), r["env"], dev), dev)
        p = paths[key]
        p["class"] = r.get("backend_class")
        p["backends"].update(r.get("backends", []))
        ok = r["tier"] in (GPU_OK if dev == "gpu" else CPU_OK)
        # UNMEASURED carries EMPTY evidence: the row was never benchmarked, usually because the
        # path does not exist on that device. Optimising a path is not the same job as building
        # one, so these are counted apart rather than summed into "below tier".
        unmeasured = r["tier"].endswith("UNMEASURED")
        ev = r.get("evidence") or {}
        rec = {"env": r["env"], "tier": r["tier"], "qualifiers": r.get("qualifiers", []),
               "ratio": ev.get("ratio"), "cores": ev.get("cores"), "se_fps": ev.get("se_fps"),
               "official_fps": ev.get("official_cpu_fps"), "best_batch": ev.get("best_batch")}
        p["rows"].append(rec)
        if ok:
            p["ok"] += 1
        elif unmeasured:
            p["unmeasured"].append(rec)
        else:
            p["below"].append(rec)
            for q in (r.get("qualifiers") or ["(no qualifier)"]):
                p["blockers"][q] += 1

    out = []
    for (ws, dev), p in paths.items():
        below = p["below"]
        # Rank by how many rows are below tier, then by how far the worst one is. "How far" is the
        # shortfall against the frozen bar for the row's class, so a class-S row needing 4x and a
        # class-R row needing 500x are comparable.
        need = 4.0 if p["class"] == "S" else 500.0
        worst = min((b["ratio"] for b in below if b["ratio"] is not None), default=None)
        # Section 14: a shared-path change must replay EVERY frozen configuration built on that
        # path, not only the one used to profile it. Computed here so the obligation is a list
        # rather than a memory.
        # The join cannot be on backends alone: five Step 4 configurations declare an EMPTY
        # backend list (CraftaxClassic and the four swm_nav scenes), so a naive intersection files
        # them under nobody and understates the replay obligation for tilemap and raster2d. Resolve
        # those by name, exactly as the Step 3 rows are.
        replay = sorted(
            cid for cid, bks in frozen.items()
            if (set(bks) & p["backends"]) or (not bks and BY_ENV.get(cid) == ws)
        )
        out.append({
            "workstream": ws, "device": dev, "backend_class": p["class"],
            "backends": sorted(p["backends"]),
            "step4_replay_configs": replay, "step4_replay_n": len(replay),
            "rows_total": len(p["rows"]), "rows_at_tier": p["ok"], "rows_below": len(below),
            "rows_unmeasured": len(p["unmeasured"]),
            "blocking_qualifiers": dict(p["blockers"]),
            "required_ratio": need if dev == "gpu" else 5.0,
            "worst_ratio": worst,
            "shortfall_x": (need / worst) if (worst and worst > 0 and dev == "gpu") else
                           ((5.0 / worst) if (worst and worst > 0) else None),
            "below": sorted(below, key=lambda b: (b["ratio"] is None, b["ratio"] or 0))[:12],
        })
    out.sort(key=lambda o: (-(o["rows_below"] + o["rows_unmeasured"]), -(o["shortfall_x"] or 0)))

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"ranked": out, "step3": str(args.step3.name)}, indent=2))

    hdr = (f"{'workstream':<20}{'dev':<5}{'cls':<4}{'at':>4}{'slow':>6}{'none':>6}"
           f"{'worst R':>9}{'gap':>7}{'replay':>8}  blocking")
    print(hdr); print("-" * len(hdr))
    for o in out:
        ws = o["workstream"] or "(UNCLAIMED)"
        w = f"{o['worst_ratio']:.3f}" if o["worst_ratio"] is not None else "-"
        g = f"{o['shortfall_x']:.0f}x" if o["shortfall_x"] else "-"
        bl = ", ".join(f"{k}x{v}" for k, v in sorted(o["blocking_qualifiers"].items(),
                                                     key=lambda kv: -kv[1])[:2]) or "-"
        print(f"{ws:<20}{o['device']:<5}{str(o['backend_class']):<4}{o['rows_at_tier']:>4}"
              f"{o['rows_below']:>6}{o['rows_unmeasured']:>6}{w:>9}{g:>7}"
              f"{o['step4_replay_n']:>8}  {bl}")
    print(f"\nwrote {args.out}")


if __name__ == "__main__":
    main()
