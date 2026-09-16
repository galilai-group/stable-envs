"""Diff this closure run against the previous one, file by file and test id by test id.

The Step-6 comparator (`compare_gpu_suite.py`) answers "is anything failing that Step 6 did not".
After a re-sweep on a later integration SHA the more useful question is the narrow one: **what moved
between MY OWN two runs**, on trees that differ by a known set of commits. Anything that moved and is
not explained by those commits is the finding.

Works on the `*.results.txt` files of two closure directories and pairs them by filename, so the
interpreter and the file are held fixed across the comparison.

    python scratch/step7/closure/compare_runs.py --old DIR --new DIR --out DIR/x.json
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]


def parse(p: Path) -> dict:
    t = p.read_text() if p.is_file() else ""
    tot = {}
    for n, w in re.findall(r"(\d+) (passed|failed|skipped|errors?|error)", t):
        tot[w.rstrip("s")] = int(n)
    return {"exists": p.is_file() and bool(t.strip()),
            "failed": sorted(set(re.findall(r"^FAILED (\S+)", t, re.M))),
            "error": sorted(set(re.findall(r"^ERROR (\S+)", t, re.M))),
            "totals": tot}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--old", required=True, type=Path)
    ap.add_argument("--new", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args()

    names = sorted({p.name for d in (a.old, a.new) if d.is_dir()
                    for p in d.glob("*.results.txt")})
    rows, clean = [], True
    for n in names:
        o, w = parse(a.old / n), parse(a.new / n)
        row = {"file": n, "old": o["totals"], "new": w["totals"],
               "old_exists": o["exists"], "new_exists": w["exists"]}
        if o["exists"] and w["exists"]:
            of, wf = set(o["failed"]), set(w["failed"])
            row["fixed"] = sorted(of - wf)
            row["new_failures"] = sorted(wf - of)
            row["carried"] = sorted(of & wf)
            row["error_set_unchanged"] = set(o["error"]) == set(w["error"])
            row["n_errors_old"], row["n_errors_new"] = len(o["error"]), len(w["error"])
            if row["new_failures"] or not row["error_set_unchanged"]:
                clean = False
        else:
            row["comparable"] = False
        rows.append(row)
        print(f"## {n}")
        print(f"   old {o['totals'] or '(absent)'}")
        print(f"   new {w['totals'] or '(absent)'}")
        if "new_failures" in row:
            print(f"   NEW     : {row['new_failures'] or 'none'}")
            print(f"   FIXED   : {row['fixed'] or 'none'}")
            print(f"   carried : {len(row['carried'])}; errors "
                  f"{row['n_errors_old']} -> {row['n_errors_new']} "
                  f"({'unchanged' if row['error_set_unchanged'] else 'CHANGED'})")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(
        {"what_this_is": "this closure run against the previous one, per file, per test id",
         "old_dir": str(a.old), "new_dir": str(a.new),
         "no_new_failures_anywhere": clean, "rows": rows}, indent=1, sort_keys=True))
    print(f"\nno new failures anywhere: {clean}\nwrote {a.out}")


if __name__ == "__main__":
    main()
