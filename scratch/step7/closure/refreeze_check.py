"""Compare a fresh render of the six Mujoco locomotion nvdiff cells against the Step-7 RE-FREEZE
artifact that W6 committed, cell by cell.

The six cells are NOT among the 49 frozen Step-4 render-distinct configurations -- they have no
``examples/results/step4/states/<Config>.npz`` and appear in no ``conformance/<shard>.json``; the
only Step-4 artifact naming them is ``bounds.downsample_exact_frac``. Their frames were re-baselined
in Step 7 under Sami's correctness re-freeze ruling (the batch-invariant mvp), so the thing a closure
sweep can hold them to is the re-freeze artifact itself:
``examples/results/step7/rocket-landing/refreeze/branch.json``.

Input: the ``branch.json`` that ``scratch/step7/rocket-landing/refreeze_locomotion.py --phase render
--tag branch`` writes, produced HERE on the integration HEAD. Both files carry, per cell, the
sha256 of the identity-draw frames AND of the states they were rendered from, so a mismatch is
attributed to the frames or to the draw rather than being ambiguous.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
FROZEN = ROOT / "examples" / "results" / "step7" / "rocket-landing" / "refreeze" / "branch.json"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fresh", required=True, type=Path)
    ap.add_argument("--frozen", type=Path, default=FROZEN)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    frozen = json.loads(a.frozen.read_text())["cells"]
    fresh = json.loads(a.fresh.read_text())
    prov = fresh.get("provenance", {})
    fresh = fresh["cells"]

    rows = []
    for name in sorted(set(frozen) | set(fresh)):
        f, g = frozen.get(name, {}), fresh.get(name, {})
        ff, gf = (f.get("frames") or {}), (g.get("frames") or {})
        d = {"cell": name,
             "frozen_frames_sha256": ff.get("sha256"), "fresh_frames_sha256": gf.get("sha256"),
             "frozen_states_sha256": ff.get("states_sha256"),
             "fresh_states_sha256": gf.get("states_sha256"),
             "frozen_mean": ff.get("mean"), "fresh_mean": gf.get("mean"),
             "missing_here": g.get("missing")}
        if d["missing_here"]:
            d["verdict"] = "NOT RENDERED"
        elif not ff.get("sha256") or not gf.get("sha256"):
            d["verdict"] = "NO DIGEST"
        elif d["frozen_states_sha256"] != d["fresh_states_sha256"]:
            d["verdict"] = "DIFFERENT DRAW"          # the states moved: the frames are not comparable
        elif d["frozen_frames_sha256"] == d["fresh_frames_sha256"]:
            d["verdict"] = "identical"
        else:
            d["verdict"] = "FRAMES MOVED"
        # The declared fidelity is carried through so a moved frame arrives with its consequence.
        for key in ("visual_fidelity", "downsample", "declared"):
            if key in g:
                d[f"fresh_{key}"] = g[key]
            if key in f:
                d[f"frozen_{key}"] = f[key]
        rows.append(d)

    n_ident = sum(1 for d in rows if d["verdict"] == "identical")
    doc = {"what_this_is": "the six locomotion nvdiff cells re-rendered on the integration HEAD and "
                           "compared to the Step-7 re-freeze artifact they were re-baselined to",
           "frozen_artifact": str(a.frozen.relative_to(ROOT)),
           "fresh_provenance": prov, "n_cells": len(rows), "n_identical": n_ident, "rows": rows}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(f"{n_ident}/{len(rows)} identical to the re-freeze artifact")
    for d in rows:
        print(f"  {d['cell']:22s} {d['verdict']:16s} frames "
              f"{str(d['frozen_frames_sha256'])[:12]} -> {str(d['fresh_frames_sha256'])[:12]}")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
