"""Rewrite absolute cluster paths in a measurement artifact to ``$SCR`` / ``$HOME`` before committing.

    python scratch/step8/provenance/sanitize.py <dir-or-file>...

A committed path into one person's scratch is not reproducible by anyone else, and the repo
contract test refuses it. The paths appear in RECORDED text -- import errors, dependency-hint
messages, warp's kernel-cache line -- not in anything this repo wrote, so they are rewritten here
rather than suppressed at the source, which would also drop the sentence around them.

HOSTNAMES are provenance and are left alone: ``gpu2264.oscar.ccv.brown.edu`` is how a reader knows
which card a number came from.
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

#: (pattern, replacement). Ordered: the scratch root before the generic /oscar prefix.
RULES = (
    (re.compile(r"/oscar/scratch/[A-Za-z0-9_.-]+"), "$SCR"),
    (re.compile(r"/oscar/home/[A-Za-z0-9_.-]+"), "$HOME"),
    (re.compile(r"/users?/[A-Za-z0-9_.-]+"), "$HOME"),
    (re.compile(r"/mnt/vast/home/[A-Za-z0-9_.-]+"), "$HOME"),
)


def scrub(text: str) -> str:
    for pattern, repl in RULES:
        text = pattern.sub(repl, text)
    return text


def main(argv: list[str]) -> int:
    targets: list[Path] = []
    for a in argv:
        p = Path(a)
        targets.extend(sorted(q for q in p.rglob("*") if q.is_file()) if p.is_dir() else [p])
    changed = 0
    for p in targets:
        try:
            before = p.read_text()
        except UnicodeDecodeError:
            continue
        after = scrub(before)
        if after != before:
            p.write_text(after)
            changed += 1
            print("scrubbed", p)
    print(f"{changed} file(s) changed of {len(targets)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
