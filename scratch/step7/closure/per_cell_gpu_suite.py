"""Per-cell view of `tests/gpu` for the seven cells Step 7 added, across every interpreter that ran.

The file-level diff against Step 6 answers "did anything new break". This answers the question a
reader of the coverage table actually has: **for THIS cell, what fails, and in which interpreter was
it even constructible?** A cell that skips everywhere is not a passing cell, and the `-` in a column
below means "this interpreter could not build it", never "it passed".

    python scratch/step7/closure/per_cell_gpu_suite.py --dir DIR --out DIR/per_cell.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
CELLS = ("CraftaxClassic", "CraftaxFull", "DMCManipulatorBringBall", "DMCQuadrupedWalk",
         "OGBCube", "OGBScene", "PFRocketLanding")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    out = {c: {} for c in CELLS}
    for p in sorted(a.dir.glob("test_*.results.txt")):
        if not p.stat().st_size:
            continue
        stem, _, rest = p.name.partition(".")
        venv = rest.split(".")[0]
        txt = p.read_text()
        failed = re.findall(r"^FAILED (\S+)", txt, re.M)
        ran = re.search(r"(\d+) passed", txt)
        for c in CELLS:
            # The id carries the cell either as the whole parameter, `[Cell]`, or as one segment of a
            # composite parameter, `[raster2d-tile-culling-ss2-Cell-native]`. Both count.
            hits = sorted({f.split("::")[-1] for f in failed
                           if re.search(rf"[\[\-]{re.escape(c)}[\]\-]", f)})
            constructed = bool(re.search(rf"[\[\-]{re.escape(c)}[\]\-]", txt))
            out[c][f"{stem}/{venv}"] = {
                "constructed_here": constructed,
                "failures": hits,
                "suite_passed": int(ran.group(1)) if ran else None}
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True))
    for c in CELLS:
        allf = sorted({f for v in out[c].values() for f in v["failures"]})
        where = sorted(k for k, v in out[c].items() if v["constructed_here"])
        print(f"## {c}: {len(allf)} distinct failing test(s); built in {len(where)} run(s)")
        for f in allf:
            print(f"   - {f}")
    print("wrote", a.out)


if __name__ == "__main__":
    main()
