"""Diff a closure `tests/gpu` run against the Step-6 run of the SAME files in the SAME interpreter.

"nothing else failing" is a claim about a DIFFERENCE, and the only way to make it without re-running
the suite on the parent tree is to compare against a run that already exists and was taken the same
way. `examples/results/step6/verify/gpu/*.results.txt` is that run: Step 6's integration verification
executed these files in `$SCR/venvs/se6-verify`, the same chained interpreter this sweep uses, and
committed the filtered output. The error COUNT is part of the comparison, because 437 of the lines
are the 23 environments that interpreter cannot construct at all, and a change there would mean the
interpreter moved rather than the code.

    python scratch/step7/closure/compare_gpu_suite.py --now DIR --out DIR/gpu_suite_diff.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
STEP6 = ROOT / "examples" / "results" / "step6" / "verify" / "gpu"

#: closure file stem -> the Step-6 file that ran the same tests in the same interpreter
PAIRS = {
    "test_conformance.verify": ["test_conformance_gfx.results.txt",
                                "test_conformance_swmnav.results.txt"],
    "test_reproducibility.verify": ["test_reproducibility_gfx.results.txt",
                                    "test_reproducibility_final1.results.txt"],
    "test_fast_paths.verify": ["test_fast_paths_gfx.results.txt",
                               "test_fast_paths_final1.results.txt"],
}


def counts(p: Path) -> dict:
    t = p.read_text() if p.is_file() else ""
    out = {"file": str(p.relative_to(ROOT)) if p.is_relative_to(ROOT) else str(p),
           "exists": p.is_file(),
           "failed_ids": sorted(set(re.findall(r"^FAILED (\S+)", t, re.M))),
           "error_ids": sorted(set(re.findall(r"^ERROR (\S+)", t, re.M)))}
    m = re.findall(r"(\d+) (passed|failed|skipped|errors?|error)", t)
    tot = {}
    for n, w in m:
        tot[w.rstrip("s")] = int(n)
    out["totals"] = tot
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--now", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()
    rows = []
    for stem, olds in PAIRS.items():
        now = counts(a.now / f"{stem}.results.txt")
        ref = None
        for o in olds:
            c = counts(STEP6 / o)
            if c["exists"]:
                ref = c
                break
        row = {"file": stem, "now": now, "step6_reference": ref}
        if ref and now["exists"]:
            nf, rf = set(now["failed_ids"]), set(ref["failed_ids"])
            row["new_failures"] = sorted(nf - rf)
            row["fixed_failures"] = sorted(rf - nf)
            row["carried_failures"] = sorted(nf & rf)
            row["error_count_now"] = len(now["error_ids"])
            row["error_count_step6"] = len(ref["error_ids"])
            row["error_set_unchanged"] = set(now["error_ids"]) == set(ref["error_ids"])
        rows.append(row)
        print(f"## {stem}")
        print(f"   now   {now['totals']}")
        print(f"   step6 {ref['totals'] if ref else 'NO REFERENCE COMMITTED'}")
        if "new_failures" in row:
            print(f"   NEW failures  : {row['new_failures'] or 'none'}")
            print(f"   carried       : {len(row['carried_failures'])}")
            print(f"   fixed         : {row['fixed_failures'] or 'none'}")
            print(f"   error set unchanged: {row['error_set_unchanged']} "
                  f"({row['error_count_now']} vs {row['error_count_step6']})")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps({
        "what_this_is": "closure tests/gpu vs the Step-6 run of the same files in the same "
                        "interpreter ($SCR/venvs/se6-verify)", "rows": rows},
        indent=1, sort_keys=True))
    print("wrote", a.out)


if __name__ == "__main__":
    main()
