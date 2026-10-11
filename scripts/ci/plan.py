"""Choose the test profiles a change needs, from ci/families.toml.

    python scripts/ci/plan.py --base origin/main --head HEAD            # GitHub outputs
    python scripts/ci/plan.py --base origin/main --profiles-only        # e.g. "core dmc"
    python scripts/ci/plan.py --scope all

Docs-only changes need no profile. A change to a shared path selects every profile. Otherwise
the profiles whose source paths or test files the change touches are selected (other files under
``tests/`` by the family name or an alias in their path), and ``core`` always runs. Without a
usable base (a new branch, a force push) everything runs.
"""

from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
from functools import cache
from pathlib import Path, PurePosixPath

try:
    import tomllib
except ImportError:  # Python 3.10
    import tomli as tomllib

ROOT = Path(__file__).resolve().parents[2]
CONFIG = tomllib.loads((ROOT / "ci" / "families.toml").read_text())
FAMILIES = CONFIG["family"]
ALL = list(FAMILIES)


@cache
def _regex(pattern: str) -> re.Pattern:
    """A path glob: ``**`` spans directories, ``*`` and ``?`` stay within one."""
    out = []
    for part in re.split(r"(\*\*|\*|\?)", pattern):
        out.append({"**": ".*", "*": "[^/]*", "?": "[^/]"}.get(part, re.escape(part)))
    return re.compile("".join(out) + r"\Z")


def _match(path: str, patterns) -> bool:
    return any(_regex(pattern).match(path) for pattern in patterns)


def changed_files(base: str, head: str) -> list[str] | None:
    if not base or set(base) == {"0"}:
        return None
    result = subprocess.run(
        ["git", "diff", "--name-only", f"{base}...{head}"], cwd=ROOT, capture_output=True, text=True
    )
    if result.returncode != 0:
        return None
    return [line for line in result.stdout.splitlines() if line]


def families_for(path: str) -> set[str]:
    name = PurePosixPath(path).name
    found = {
        family
        for family, spec in FAMILIES.items()
        if _match(path, spec.get("paths", ())) or (path.startswith("tests/") and _match(name, spec.get("tests", ())))
    }
    if not found and path.startswith("tests/"):
        lowered = path.lower()
        found = {
            family
            for family, spec in FAMILIES.items()
            if family != "core" and any(word in lowered for word in (family, *spec.get("aliases", ())))
        }
    return found


def plan(files: list[str] | None) -> dict:
    if files is None:
        return {"profiles": ALL, "docs_only": False, "reason": "no usable base; everything runs"}
    if files and all(_match(path, CONFIG["docs"]["paths"]) for path in files):
        return {"profiles": [], "docs_only": True, "reason": "documentation only"}
    shared = [path for path in files if _match(path, CONFIG["shared"]["paths"])]
    if shared:
        return {"profiles": ALL, "docs_only": False, "reason": f"shared path {shared[0]}"}
    selected = {"core"}
    for path in files:
        selected |= families_for(path)
    return {"profiles": [p for p in ALL if p in selected], "docs_only": False, "reason": "affected families"}


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base", default="origin/main")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--scope", choices=("affected", "all"), default="affected")
    parser.add_argument("--profiles-only", action="store_true", help="print the selected profiles, space separated")
    args = parser.parse_args(argv)

    result = plan(None if args.scope == "all" else changed_files(args.base, args.head))
    if args.scope == "all":
        result["reason"] = "full run requested"
    if args.profiles_only:
        print(" ".join(result["profiles"]))
        return 0
    families = [p for p in result["profiles"] if p != "core"]
    print(f"core={'true' if 'core' in result['profiles'] else 'false'}")
    print(f"families={json.dumps({'include': [{'profile': p} for p in families]}, separators=(',', ':'))}")
    print(f"docs_only={'true' if result['docs_only'] else 'false'}")
    print(f"reason={result['reason']}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
