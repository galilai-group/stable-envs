"""Dump ``VariationMap.classification()`` for every class Step 8 registered, so
``build_gate_f.py`` can read it as plain JSON instead of importing ``stable_envs`` (which needs
``numpy`` and a device-specific interpreter).

    MUJOCO_GL=glfw PYTHONPATH=. /Users/samib/se6-venv/bin/python \
        scratch/step8/swm/probe_variation_classification.py

Re-run this whenever ``VARIATION_MAPS`` changes (e.g. after T3 lands) and re-run
``build_gate_f.py`` after that -- this is the seam the brief asks for so the Gate-F matrix can be
regenerated once T3 (plate texel capability) merges without hand-editing anything.
"""
from __future__ import annotations

import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[3]
OUT = ROOT / "examples" / "results" / "step8" / "swm" / "variation_classification.json"


def _sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=ROOT,
                          capture_output=True, text=True).stdout.strip()


def main() -> None:
    from stable_envs.compat.worldmodel import VARIATION_MAPS

    classes = {}
    for name, vm in sorted(VARIATION_MAPS.items()):
        classification = vm.classification()
        counts: dict[str, int] = {}
        for label in classification.values():
            counts[label] = counts.get(label, 0) + 1
        classes[name] = {
            "swm_id": vm.swm_id,
            "n_leaves": len(classification),
            "counts": counts,
            "classification": classification,
        }
    out = {
        "what_this_is": __doc__,
        "baseline_sha": _sha(),
        "source": "stable_envs.compat.worldmodel.VARIATION_MAPS[*].classification()",
        "classes": classes,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=2, sort_keys=True) + "\n")
    print("wrote", OUT, "-", len(classes), "classes")


if __name__ == "__main__":
    main()
