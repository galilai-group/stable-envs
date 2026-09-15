"""Does any of the 49 frozen Step-4 configs come from a family the sweeps could not reach?

Reviewer item 5 left RoboCasa, MetaWorld and HumanoidBench unswept: none of them imports in any
interpreter available here, each for a structural reason of its own. The coordinator's bound on that
residual risk is this question -- if one of the 49 frozen render-distinct configs belonged to those
families, its digest would have to be replayed before the shared change could be trusted.

Answered STATICALLY, from the source that registers the cells, because the modules cannot be
imported to be asked. Each env module names the classes it registers; this matches the frozen 49
against the cells each module defines, so a family that cannot import is still accounted for.
"""
from __future__ import annotations

import ast
import json
import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
UNSWEPT = ("robocasa", "metaworld", "humanoidbench")


def registered_names(path: Path) -> set[str]:
    """Class names a module registers, from its source: `class X(...)` plus generated `type("X", ...)`
    and f-string cell-name patterns."""
    src = path.read_text()
    out = set()
    try:
        tree = ast.parse(src)
    except SyntaxError:
        return out
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef):
            out.add(node.name)
        # register(type("Name", ...)) and type(cname, ...) generated families
        if isinstance(node, ast.Call) and isinstance(node.func, ast.Name) and node.func.id == "type":
            if node.args and isinstance(node.args[0], ast.Constant) and isinstance(node.args[0].value, str):
                out.add(node.args[0].value)
    # literal cell-name tables, e.g. CELLS = (("FetchReach", ...), ...) and "H1Walk" style constants
    out |= set(re.findall(r'"([A-Z][A-Za-z0-9]{2,})"', src))
    out |= set(re.findall(r"'([A-Z][A-Za-z0-9]{2,})'", src))
    return out


def main():
    frozen = sorted(json.loads((ROOT / "examples/results/step4/certification.json").read_text())["configs"])
    envs = ROOT / "stable_envs" / "envs"
    per_module = {p.stem: registered_names(p) for p in sorted(envs.glob("*.py"))
                  if not p.stem.startswith("_")}
    rows, overlap = [], []
    for name in frozen:
        owners = sorted(m for m, names in per_module.items() if name in names)
        unswept = [m for m in owners if m in UNSWEPT]
        rows.append({"config": name, "candidate_modules": owners, "unswept_owner": unswept})
        if unswept:
            overlap.append(name)
    doc = {
        "n_frozen_configs": len(frozen),
        "unswept_families": list(UNSWEPT),
        "frozen_configs_from_unswept_families": overlap,
        "verdict": ("NONE of the 49 frozen Step-4 render-distinct configs is registered by "
                    "robocasa.py, metaworld.py or humanoidbench.py, so no frozen digest depends on "
                    "a family the sweeps could not reach."
                    if not overlap else
                    f"{len(overlap)} frozen configs come from unswept families and MUST be replayed: "
                    f"{overlap}"),
        "rows": rows,
    }
    out = ROOT / "examples/results/step7/ogb-manipulation/m2"
    out.mkdir(parents=True, exist_ok=True)
    (out / "frozen_family_overlap.json").write_text(json.dumps(doc, indent=1, sort_keys=True))
    print(doc["verdict"])
    for m in UNSWEPT:
        owned = sorted(n for n in frozen if m in (r["unswept_owner"] for r in rows if r["config"] == n))
        print(f"  {m}: registers {len(per_module.get(m, ()))} candidate names; frozen overlap {owned or 'none'}")
    unowned = [r["config"] for r in rows if not r["candidate_modules"]]
    if unowned:
        print(f"  NOTE: {len(unowned)} frozen configs matched no module by this static rule "
              f"(they are still not from the three families, but the rule did not place them): {unowned}")


if __name__ == "__main__":
    main()
