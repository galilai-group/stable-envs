"""Judge saved Step-7 measurements with the frozen judge, including family checks.

No rendering, threshold changes, or edits to the original measurement artifacts.
"""
from __future__ import annotations

import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / "examples/results/step8/certification/gate_a_rejudged.json"


def build():
    spec = importlib.util.spec_from_file_location("step8_frozen_judge", ROOT / "scratch/step4/certify.py")
    judge = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(judge)
    judges = judge.judges(judge.B.load())
    records = []
    for path in sorted((ROOT / "examples/results/step7/closure2/gate_a").glob("*.json")):
        raw = path.read_bytes()
        doc = json.loads(raw)
        for cell, row in sorted(doc.get("configs", {}).items()):
            result = judge.verdict(row, judges, doc.get("provenance"))
            records.append({
                "cell": cell, "device": doc["device"],
                "source": str(path.relative_to(ROOT)),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "measurement_provenance": doc.get("provenance", {}),
                "previous_mandatory_only_verdict": doc.get("verdicts", {}).get(cell, {}).get("verdict"),
                "judgment": result,
            })
    return {
        "baseline_sha": "43ac95238f7472137619f3e69e7636e11dd088b1",
        "judge": "scratch/step4/certify.py:verdict",
        "judge_sha256": hashlib.sha256((ROOT / "scratch/step4/certify.py").read_bytes()).hexdigest(),
        "generated_at_head_sha": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "scope": "Rejudgment of unchanged historical measurements, not a fresh Step-8 renderer certification. Both mandatory and family_specific groups are judged; unresolved applicability/reference issues remain visible.",
        "records": records,
    }


if __name__ == "__main__":
    doc = build()
    OUTPUT.write_text(json.dumps(doc, indent=2, sort_keys=True) + "\n")
    for record in doc["records"]:
        verdict = record["judgment"]
        print(record["cell"], record["device"], verdict["gate_a"], verdict["failures"], verdict["blocked"])
