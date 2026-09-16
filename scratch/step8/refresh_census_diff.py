"""Refresh final-classifier columns of the recorded before/after census comparison."""
import importlib.util
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]


def build():
    spec = importlib.util.spec_from_file_location("step8_classifier", ROOT / "scratch/step3/certify.py")
    classifier = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(classifier)
    path = ROOT / "examples/results/step8/certification/census_class_diff.json"
    doc = json.loads(path.read_text())
    census = classifier.read_censuses(sorted((ROOT / "scratch/census").glob("census_step8-*.json")))
    for row in doc["rows"]:
        row["new_class_new_census"] = classifier.backend_class(row["env"], census)[0]
    doc["class_changed"] = [r for r in doc["rows"] if r["old_class_old_census"] not in (None, "unknown")
                            and r["old_class_old_census"] != r["new_class_new_census"]]
    doc["unknown_resolved"] = [r for r in doc["rows"] if r["old_class_new_census"] == "unknown"
                               and r["new_class_new_census"] != "unknown"]
    doc["still_unknown"] = [r for r in doc["rows"] if r["new_class_new_census"] == "unknown"]
    return path, doc


if __name__ == "__main__":
    path, doc = build()
    path.write_text(json.dumps(doc, indent=1) + "\n")
    print("changed", len(doc["class_changed"]), "resolved", len(doc["unknown_resolved"]),
          "unknown", len(doc["still_unknown"]))
