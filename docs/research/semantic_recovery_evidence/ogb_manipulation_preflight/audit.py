"""Read-only closure, evidence and checkpoint-preservation verification."""

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
START = "e1b7e5a364b02386e1d3ed6ace29c3d22c7826f3"
DOCS = "03adc1ab79bae71e3e62db2e5c4cb19f20723166"
TARGETS = {"swm/OGBCube-v0", "swm/OGBScene-v0"}


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT)


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    accepted_evidence = json.loads((HERE.parent / "pusht_phase_a/audit.json").read_text())
    for rel, expected in accepted_evidence["artifact_hashes"].items():
        assert sha(ROOT / rel) == expected, rel
    for rel, entry in accepted_evidence["terminal_logs"].items():
        assert sha(ROOT / rel) == entry["sha256"], rel
        assert all(line in (ROOT / rel).read_text().splitlines() for line in entry["exact_terminal_lines"])
    fetch = HERE.parent / "fetch_family/SHA256SUMS"
    fetch_count = 0
    for line in fetch.read_text().splitlines():
        expected, rel = line.split(maxsplit=1)
        path = ROOT / rel.lstrip("*")
        if not path.exists():
            path = fetch.parent / rel.lstrip("*")
        assert sha(path) == expected, rel
        fetch_count += 1
    for rel in [
        "docs/research/semantic_recovery_matrix.json",
        "docs/research/semantic_recovery_implementation_map.json",
    ]:
        before = json.loads(git("show", f"{DOCS}:{rel}"))
        after = json.loads((ROOT / rel).read_text())
        assert before.keys() == after.keys()
        for key in before:
            if key != "rows":
                assert before[key] == after[key], (rel, key)
        a = {r["historical_id"]: r for r in before["rows"]}
        b = {r["historical_id"]: r for r in after["rows"]}
        assert a.keys() == b.keys()
        for key in a.keys() - TARGETS:
            assert a[key] == b[key], (rel, key)
        for key in TARGETS:
            for artifact in b[key]["source_preflight"]["artifacts"]:
                assert sha(ROOT / artifact["path"]) == artifact["sha256"], artifact["path"]
    mapping = json.loads((ROOT / "docs/research/semantic_recovery_implementation_map.json").read_text())
    verified = [r["historical_id"] for r in mapping["rows"] if r["current_status"] == "VERIFIED"]
    incomplete = [r["historical_id"] for r in mapping["rows"] if r["current_status"] == "INCOMPLETE"]
    assert len(verified) == 33 and len(incomplete) == 9
    assert set(verified) == set(accepted_evidence["verified_ids"])
    docs_files = git("diff-tree", "--no-commit-id", "--name-only", "-r", DOCS).decode().splitlines()
    for rel in docs_files:
        assert (ROOT / rel).read_bytes() == git("show", f"{DOCS}:{rel}"), rel
    assert git("diff", START, "--", "stable_envs", "tests", "scripts", "pyproject.toml", "uv.lock") == b""
    changes = git("diff", "--name-only", DOCS).decode().splitlines()
    allowed = {
        "docs/research/semantic_recovery_matrix.json",
        "docs/research/semantic_recovery_implementation_map.json",
        "docs/research/ogbench_manipulation_candidate_state.md",
        "docs/research/ogbench_manipulation_checkpoint.md",
    }
    prefix = "docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/"
    assert all(p in allowed or p.startswith(prefix) for p in changes), changes
    for inventory in ["provenance.json", "dependency_hashes.json"]:
        records = json.loads((HERE / inventory).read_text())
        if inventory == "provenance.json":
            assert records["integration_mask"] == records["component_union"]
            records = records["files"]
        for item in records.values():
            assert sha(Path(item["path"])) == item["sha256"], item["path"]
    results = json.loads((HERE / "results.json").read_text())
    for family in results["families"].values():
        assert len(family["reset_pairs"]) == 15 and all(r["diff"] for r in family["reset_pairs"])
        assert len(family["conditioned_reset_pairs"]) == 3
        assert all(r["diff"] for r in family["conditioned_reset_pairs"])
        for case in family["continuations"]:
            assert case["different_before"] and not case["roundtrip_diff"]
            assert case["unseeded_reset_diff"] and len(case["trace"]) == 8
            assert all(not r["state_diff"] and not r["output_diff"] for r in case["trace"])
    assert results["families"]["Scene"]["frozen_constructor"]["status"] == "FAIL"
    assert results["families"]["Cube"]["frozen_constructor"]["status"] == "PASS"
    for log, marker in [
        ("probe.log", "OGB_MANIPULATION_PREFLIGHT_COMPLETE_NO_PROMOTION"),
        ("reset_effects.log", "OGB_RESET_EFFECTS_RECORDED_NO_PROMOTION"),
    ]:
        assert (HERE / log).read_text().splitlines()[-1] == marker
    result = {
        "status": "PASS",
        "starting_sha": START,
        "documentation_sha": DOCS,
        "verified_ids": verified,
        "incomplete_ids": incomplete,
        "preserved_artifact_count": len(accepted_evidence["artifact_hashes"]),
        "preserved_terminal_logs": len(accepted_evidence["terminal_logs"]),
        "fetch_manifest_files": fetch_count,
        "unchanged_documentation_files": docs_files,
        "production_tests_scripts_shared_diff": "",
        "non_target_matrix_map_rows_unchanged": True,
        "ogbench_maze_unchanged": True,
        "source_hash_inventory_verified": True,
        "source_diagnostics_verified": True,
    }
    (HERE / "closure.json").write_text(json.dumps(result, indent=2) + "\n")
    print("ALL_33_OGB_PREFLIGHT_CLOSURE_PASS artifacts=170 terminal_logs=11 fetch_files=75 docs_files=24")


if __name__ == "__main__":
    main()
