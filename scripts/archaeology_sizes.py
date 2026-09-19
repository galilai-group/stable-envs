#!/usr/bin/env python3
"""Print working-tree sizes for the v0 weight trees. Classification only; deletes nothing."""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
TREES = (
    "scratch",
    "examples/results",
    "examples/robocasa_visual",
    "docs/assets",
)


def du(path: Path) -> str:
    out = subprocess.check_output(["du", "-sh", str(path)], text=True)
    return out.split()[0]


def main() -> int:
    print(f"root\t{du(ROOT)}\t{ROOT}")
    print(f"stable_envs\t{du(ROOT / 'stable_envs')}\t{ROOT / 'stable_envs'}")
    print()
    for rel in TREES:
        p = ROOT / rel
        if not p.exists():
            print(f"{rel}\tMISSING")
            continue
        print(f"{rel}\t{du(p)}")
        children = sorted(
            [c for c in p.iterdir() if not c.name.startswith(".")],
            key=lambda c: (-int(subprocess.check_output(["du", "-sk", str(c)]).split()[0]), c.name),
        )
        for child in children:
            print(f"  {rel}/{child.name}\t{du(child)}")
        print()
    return 0


if __name__ == "__main__":
    sys.exit(main())
