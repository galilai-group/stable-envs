"""Splice generated base64 blobs into stable_envs/envs/crafter.py.

The blobs are thousands of characters long; hand-copying one is how a texture table ends up truncated
and the atlas silently wrong, so the constants are only ever written by this script.

    python scratch/crafter/splice.py NAME=path/to/blob_module.py:CONST ...
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

TARGET = Path("stable_envs/envs/crafter.py")


def block(text: str, const: str) -> str:
    m = re.search(rf"^{re.escape(const)} = \((.*?)^\)$", text, re.S | re.M)
    if m is None:
        raise SystemExit(f"{const} not found")
    return m.group(0)


def main(argv):
    src = TARGET.read_text()
    for arg in argv:
        const, path = arg.split("=", 1)
        blob = Path(path).read_text()
        new = block(blob, const)
        old = block(src, const)
        src = src.replace(old, new)
        print(f"{const}: {len(old)} -> {len(new)} chars")
    TARGET.write_text(src)


if __name__ == "__main__":
    main(sys.argv[1:])
