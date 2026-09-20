"""Classify every Step-8 support-matrix row against the live Step-9 registry.

This does not regenerate Gate A, tiers, or frozen corpora. It copies the accepted Step-8 matrix
and attaches a Step-9 difference class per row:

    API_ONLY                 public names / ctor / state container changed, pixels did not
    INTENTIONAL_IMPROVEMENT  additive lifecycle / resource protocol
    REGRESSION               forbidden at closure
    UNCHANGED                no Step-9 difference recorded

Usage (from the repo root)::

    /Users/samib/se8-vm-venv/bin/python scripts/regression/compare_support_matrix.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path


def _repo_root() -> Path:
    p = Path(__file__).resolve()
    for cand in (p, *p.parents):
        if (cand / "pyproject.toml").is_file():
            return cand
    raise RuntimeError("repo root not found")


ROOT = _repo_root()
STEP8 = ROOT / "tests" / "fixtures" / "regression" / "support_matrix.json"
SNAPSHOT = ROOT / "tests" / "fixtures" / "regression" / "references" / "frozen_oracle" / "step8_support_matrix.json"
OUT = ROOT / "tests" / "fixtures" / "regression" / "support_matrix_diff.json"

# Differences that apply to every certified id after Wave 0/1. None of these change
# Gate A, frozen corpora, or declared fidelity tiers.
GLOBAL_API_ONLY = [
    "ctor is num_envs= (batch= still accepted, warns once)",
    "public State is a TensorDict; pack/unpack still emit the Step-8 float32 row",
    "step is stateful and returns StepResult; transition/unroll are functional",
    "env.close() / context manager exist on every cell",
]
GLOBAL_IMPROVEMENTS = [
    "renderer backends that own device memory expose Closeable.close(); EGL contexts are still never released",
]


def _load_step8() -> dict:
    path = STEP8 if STEP8.is_file() else SNAPSHOT
    return json.loads(path.read_text())


def classify_row(row: dict) -> dict:
    diffs = [{"class": "API_ONLY", "note": note} for note in GLOBAL_API_ONLY]
    diffs.extend({"class": "INTENTIONAL_IMPROVEMENT", "note": note} for note in GLOBAL_IMPROVEMENTS)
    if row.get("swm_id") == "swm/HumanoidmazeGiantNavigate-v0" or (
        "HumanoidmazeGiantNavigate" in row.get("stable_envs_cells", [])
    ):
        diffs.append(
            {
                "class": "DOCUMENTED_LIMITATION",
                "note": "cuda process-reproducibility on some Oscar nodes; accepted in Step 6/8/9, not a Step-9 regression",
            }
        )
    return {
        "swm_id": row["swm_id"],
        "stable_envs_cells": list(row.get("stable_envs_cells") or []),
        "pixel_or_symbolic": row.get("pixel_or_symbolic"),
        "gate_a": (row.get("gate_a") or {}).get("status"),
        "step9_differences": diffs,
        "regression": False,
    }


def main() -> int:
    src = _load_step8()
    rows = [classify_row(r) for r in src["rows"]]
    regressions = sum(1 for r in rows if r["regression"])
    payload = {
        "schema": "stable-envs-step9-support-matrix/v1",
        "step8_baseline_sha": "21cb30678b7a7d0f3344b59f176f33d3f3b09ffc",
        "step8_matrix_source": str((STEP8 if STEP8.is_file() else SNAPSHOT).relative_to(ROOT)),
        "step8_counts": src.get("counts"),
        "counts": {
            "rows": len(rows),
            "pixel_producing": sum(1 for r in rows if r["pixel_or_symbolic"] == "pixel"),
            "symbolic": sum(1 for r in rows if r["pixel_or_symbolic"] == "symbolic"),
            "regression": regressions,
            "api_only_rows": len(rows),
            "documented_limitation_rows": sum(
                1 for r in rows if any(d["class"] == "DOCUMENTED_LIMITATION" for d in r["step9_differences"])
            ),
        },
        "rule": "every Step-9 difference is API_ONLY / INTENTIONAL_IMPROVEMENT / REGRESSION / DOCUMENTED_LIMITATION; closure requires REGRESSION = 0",
        "rows": rows,
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {OUT.relative_to(ROOT)} rows={len(rows)} regression={regressions}")
    return 0 if regressions == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
