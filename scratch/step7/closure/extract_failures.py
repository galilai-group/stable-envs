"""Pull the first assertion line out of each named failure in a raw pytest log, and nothing else.

The filtered `*.results.txt` files that reach the repository deliberately carry only test ids and
totals -- a raw pytest log is full of absolute `$SCR` interpreter paths and would fail the citation
scanner in `tests/unit/test_repo_contracts.py`. So the MECHANISM of each failure has to be lifted out
of the raw log under `$SCR/logs/` separately, scrubbed, and recorded as a short quotation.

    python scratch/step7/closure/extract_failures.py --log RAW.txt --out DIR/x.json [--only ID,ID]
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

#: pytest heads each failure section with the test id between underscore rules. The rule can be
#: short and the id can carry brackets, so the match is on "a line that is mostly underscores and
#: contains a test id", not on an exact width.
_HDR = re.compile(r"^_{3,}\s*(\S.*?)\s*_{3,}$")
_ERR = re.compile(r"^_{3,}\s*ERROR at \w+ of (\S.*?)\s*_{3,}$")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--log", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--only", default="", help="comma-separated test ids (the part after ::)")
    ap.add_argument("--max-chars", type=int, default=700)
    a = ap.parse_args()
    want = set(x for x in a.only.split(",") if x)

    lines = a.log.read_text(errors="replace").splitlines()
    starts = []
    for i, l in enumerate(lines):
        m = _ERR.match(l) or _HDR.match(l)
        if m:
            starts.append((i, m.group(1).strip()))
    out = {}
    for k, (i, name) in enumerate(starts):
        j = starts[k + 1][0] if k + 1 < len(starts) else len(lines)
        if want and name not in want:
            continue
        ev = [l for l in lines[i:j] if l.startswith("E ")]
        if not ev:
            continue
        # The first E-line is the exception; anything after it is the assert repr, which for an
        # image comparison is a megabyte of pixels. One line, scrubbed of absolute paths, truncated.
        msg = ev[0][2:].strip()
        msg = re.sub(r"/oscar/scratch/\S+", "<SCR>/…", msg)
        msg = re.sub(r"/mnt/\S+", "<absolute path>", msg)
        # Last section wins: pytest prints the same id once under FAILURES and, for a teardown
        # error, again under ERRORS. Keying by name without this, an earlier unrelated section
        # whose header merely CONTAINED the id could be attributed to it -- which happened, and is
        # why the match is exact and the sections are not searched fuzzily.
        out[name] = msg[:a.max_chars] + ("…" if len(msg) > a.max_chars else "")
    a.out.parent.mkdir(parents=True, exist_ok=True)
    a.out.write_text(json.dumps(out, indent=1, sort_keys=True))
    for k, v in sorted(out.items()):
        print(f"## {k}\n   {v[:400]}")
    print(f"\n{len(out)} failure(s) -> {a.out}")


if __name__ == "__main__":
    main()
