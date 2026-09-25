"""Recompute native feasibility conclusions from retained numeric counterexamples."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
EVIDENCE = ROOT / "docs/research/semantic_recovery_evidence"


def compare(left, right):
    assert set(left.files) == set(right.files)
    return {
        key: {
            "count": int(np.count_nonzero(left[key] != right[key])),
            "max_abs": float(np.abs(left[key].astype(float) - right[key].astype(float)).max()),
        }
        for key in left.files
        if not np.array_equal(left[key], right[key])
    }


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    r = EVIDENCE / "rocket_native_feasibility"
    original = EVIDENCE / "rocket_source_audit/feasibility-final"
    comparisons = 0
    for path in (r / "v2-baseline/feasibility").glob("*.npz"):
        other = original / path.name
        if other.exists():
            with np.load(path) as x, np.load(other) as y:
                assert set(x.files) == set(y.files)
                for key in x.files:
                    assert np.array_equal(x[key], y[key], equal_nan=True), (path.name, key)
                    comparisons += 1
    rocket = {}
    rocket_array_pairs = 0
    for directory in r.iterdir():
        path = directory / "feasibility/result.json"
        if not path.exists():
            continue
        result = json.loads(path.read_text())
        rocket[directory.name] = {
            name: {route: len(v["trajectory_differences"]) for route, v in case["routes"].items()}
            for name, case in result["cases"].items()
        }
        for folder, case_key in [
            (directory / "feasibility", "cases"),
            (directory / "controls", "landing_contact_replay"),
        ]:
            records = json.loads((folder / "result.json").read_text())
            cases = records[case_key] if case_key == "cases" else {"landing_contact": records[case_key]}
            for name, case in cases.items():
                with (
                    np.load(folder / (name + "-snapshot.npz")) as snapshot,
                    np.load(folder / (name + "-reference.npz")) as reference,
                ):
                    for mode, record in case["routes"].items():
                        with (
                            np.load(folder / f"{name}-{mode}-immediate.npz") as immediate,
                            np.load(folder / f"{name}-{mode}.npz") as replay,
                        ):
                            assert compare(snapshot, immediate) == record["immediate_differences"]
                            assert compare(reference, replay) == record["trajectory_differences"]
                            rocket_array_pairs += 2
            if case_key == "landing_contact_replay":
                with (
                    np.load(folder / "independent-reference.npz") as left,
                    np.load(folder / "independent-restored.npz") as right,
                ):
                    assert (
                        compare(left, right) == records["independent_official_file_restore"]["trajectory_differences"]
                    )
                with np.load(folder / "source-frame-restore.npz") as frames:
                    assert (
                        np.count_nonzero(frames["before"] != frames["after"])
                        == records["official_restore_immediate_pixels"]["changed_channels"]
                    )
                rocket_array_pairs += 2
    best = json.loads((r / "v2-empty_contacts/feasibility/result.json").read_text())
    assert best["cases"]["near_contact"]["routes"]["memory"]["trajectory_differences"]
    assert best["cases"]["crash"]["routes"]["memory"]["trajectory_differences"]
    x = json.loads((EVIDENCE / "pusht_native_feasibility/final/result.json").read_text())
    checked = 0
    for row in x["cases"]:
        assert "failure" not in row, row
        path = EVIDENCE / "pusht_native_feasibility/final" / f"{row['case']}-{row['mode']}.npz"
        with np.load(path) as arrays:
            assert row["continuation_exact"] == np.array_equal(arrays["source"], arrays["target"])
            assert row["future_native_exact"] == np.array_equal(arrays["source_native"], arrays["target_native"])
            assert row["immediate_pixels_exact"] == np.array_equal(arrays["source_frame"], arrays["target_frame"])
            assert row["max_output_delta"] == float(np.abs(arrays["source"] - arrays["target"]).max())
            checked += 1
    positives = [v for v in x["cases"] if v["mode"] == "clean_independent"]
    assert len(positives) == 11 and all(v["continuation_exact"] and v["future_native_exact"] for v in positives)
    failing = next(
        v for v in x["cases"] if v["mode"] == "plus_active_order" and v["case"] == "preserved_multiple_contact"
    )
    assert not failing["continuation_exact"] and failing["native_immediate_exact"]
    assert failing["active_order_restore_code"] == 0
    a.output.write_text(
        json.dumps(
            dict(
                rocket_original_baseline_leaf_comparisons=comparisons,
                rocket_native_array_pairs=rocket_array_pairs,
                rocket_differing_trajectory_leaf_counts=rocket,
                pusht_arrays_checked=checked,
                pusht_clean_independent_controls=len(positives),
                pusht_remaining_counterexample={
                    k: failing[k]
                    for k in [
                        "case",
                        "mode",
                        "max_output_delta",
                        "reward_delta",
                        "done_exact",
                        "native_immediate_exact",
                        "immediate_pixels_exact",
                        "contact_cache_exact",
                    ]
                },
                rocket_classification="EXACT_RESTORE_NOT_YET_FEASIBLE",
                pusht_classification=x["classification"],
            ),
            indent=2,
        )
        + "\n"
    )
    print("PASS: baseline fidelity and 55 independent numeric PushT records; both feasibility gates remain blocked")


if __name__ == "__main__":
    main()
