"""Read-only final State-performance evidence verification; print JSON to stdout."""

import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
BASE = ROOT / "docs/research/semantic_recovery_evidence"
FINAL = BASE / "state_performance_final"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def read(path):
    return json.loads(path.read_text())


def main():
    start = read(BASE / "state_performance_start/start.json")
    for group in ("artifact_hashes", "production_hashes"):
        for path, expected in start[group].items():
            assert digest(ROOT / path) == expected, (group, path)
    for name in ("semantic_recovery_matrix.json", "semantic_recovery_implementation_map.json"):
        path = "docs/research/" + name
        original = subprocess.check_output(["git", "show", start["starting_HEAD"] + ":" + path], cwd=ROOT)
        assert (ROOT / path).read_bytes() == original, path

    summary = read(BASE / "state_performance_report/summary.json")
    expected_ids = set(start["per_registration_evidence"])
    assert len(summary["rows"]) == len(expected_ids) == 39
    assert {r["registration"] for r in summary["rows"]} == expected_ids
    artifact_count = 0
    renders = 0
    for row in summary["rows"]:
        for device in ("cpu", "gpu"):
            result = row[device]
            assert result["registration"] == row["registration"]
            assert result["status"] == "COMPLETE"
            assert digest(ROOT / result["artifact"]) == result["artifact_sha256"]
            artifact_count += 1
            batches = {"1", "4"} if row["registration"] in ("swm/OGBCube-v0", "swm/OGBScene-v0") else {"1", "16"}
            assert set(result["batches"]) == batches
        if "Symbolic" not in row["registration"]:
            for batch in row["gpu"]["batches"].values():
                assert batch["operations"]["render"]["status"] == "MEASURED"
            renders += 1
    assert renders == 37
    cold = summary["cold_craftax"]
    assert len(cold) == len({(r["registration"], r["device"], r["kind"]) for r in cold}) == 16
    for result in cold:
        assert result["status"] == "COMPLETE"
        assert digest(ROOT / result["artifact"]) == result["artifact_sha256"]
    assert digest(ROOT / summary["dmc_integration_artifact"]) == summary["dmc_integration_sha256"]

    tracked = set(subprocess.check_output(["git", "ls-files", "-z"], cwd=ROOT).decode().split("\0"))
    seal_counts = {}
    seals = read(FINAL / "benchmark-seals.json")["local_seals"]
    seal_paths = []
    for entry in seals.values():
        path = ROOT / entry["seal"]
        assert digest(path) == entry["sha256"], path
        seal_paths.append(path)
    for name in ("rocket_native_feasibility", "pusht_native_feasibility"):
        seal_paths.append(BASE / name / "SHA256SUMS")
    for seal in seal_paths:
        count = 0
        for line in seal.read_text().splitlines():
            expected, relative = line.split("  ", 1)
            path = seal.parent / relative
            assert digest(path) == expected, path
            assert str(path.relative_to(ROOT)) in tracked, ("untracked sealed artifact", path)
            count += 1
        seal_counts[str(seal.relative_to(ROOT))] = count
    native = read(FINAL / "native-verification.json")
    assert native["rocket_original_baseline_leaf_comparisons"] == 5096
    assert native["rocket_native_array_pairs"] == 550
    assert native["pusht_arrays_checked"] == 55
    assert native["pusht_clean_independent_controls"] == 11
    assert native["rocket_classification"] == native["pusht_classification"] == "EXACT_RESTORE_NOT_YET_FEASIBLE"

    changed = sorted(
        subprocess.check_output(
            ["git", "diff", "--name-only", start["starting_HEAD"]], cwd=ROOT, text=True
        ).splitlines()
    )
    inventory = (FINAL / "changed-files.txt").read_text().splitlines()
    assert changed == inventory, "Changed-file inventory does not match Git"
    assert all(
        p.startswith(
            (
                "docs/research/",
                "scripts/benchmarks/",
                "scripts/research_native/",
                "tests/unit/test_state_ops_benchmark.py",
            )
        )
        for p in changed
    )
    result = {
        "status": "STATE_PERFORMANCE_AND_NATIVE_COUNTEREXAMPLES_VERIFIED",
        "starting_commit": start["starting_HEAD"],
        "pre_handoff_commit": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
        "matrix_counts": {"VERIFIED": 39, "INCOMPLETE": 3},
        "cpu_registrations": 39,
        "gpu_registration_records": 39,
        "gpu_renderers_measured": renders,
        "warm_source_artifacts_verified": artifact_count,
        "fresh_process_craftax_cases": len(cold),
        "accepted_artifact_hashes": len(start["artifact_hashes"]),
        "production_hashes": len(start["production_hashes"]),
        "seals_verified": seal_counts,
        "all_sealed_artifacts_tracked": True,
        "changed_file_count": len(changed),
        "production_changes": [],
        "shared_contract_changes": [],
        "promotions": [],
        "push_performed": False,
    }
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
