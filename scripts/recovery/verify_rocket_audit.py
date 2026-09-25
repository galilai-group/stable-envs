"""Independently check Rocket counterexamples and preserve the 39 accepted rows."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
START = "badc193af3ee450ac486a75270bc88f47d9a1816"
EVIDENCE = ROOT / "docs/research/semantic_recovery_evidence/rocket_source_audit"
TARGET = "swm/PFRocketLanding-v0"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def compare(a, b):
    assert set(a.files) == set(b.files)
    return {
        k: {
            "count": int(np.count_nonzero(a[k] != b[k])),
            "max_abs": float(np.abs(a[k].astype(float) - b[k].astype(float)).max()),
        }
        for k in a.files
        if not np.array_equal(a[k], b[k])
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    assert not args.output.exists()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT)

    allowed = {
        "docs/research/index.md",
        "docs/research/semantic_recovery_matrix.json",
        "docs/research/semantic_recovery_implementation_map.json",
    }
    unchanged = 0
    for line in git("ls-tree", "-r", START).decode().splitlines():
        meta, rel = line.split("\t")
        if rel in allowed:
            continue
        data = (ROOT / rel).read_bytes()
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == meta.split()[2], rel
        unchanged += 1
    for name in ["semantic_recovery_matrix.json", "semantic_recovery_implementation_map.json"]:
        rel = "docs/research/" + name
        before = json.loads(git("show", START + ":" + rel))
        after = json.loads((ROOT / rel).read_text())
        old = {r["historical_id"]: r for r in before["rows"]}
        new = {r["historical_id"]: r for r in after["rows"]}
        assert old.keys() == new.keys()
        assert all(new[k] == v for k, v in old.items() if k != TARGET)
        if name == "semantic_recovery_matrix.json":
            counts = dict(Counter(r["state_status"] for r in new.values()))
            assert counts == {"VERIFIED": 39, "INCOMPLETE": 3}
    manifests = 0
    for family in ["CraftaxClassicPixels", "CraftaxClassicSymbolic", "CraftaxPixels", "CraftaxSymbolic"]:
        e = ROOT / "docs/research/semantic_recovery_evidence/craftax_gpu_6679082"
        r = json.loads((e / (family + "-acceptance.json")).read_text())
        assert r["status"] == "ACCEPTED"
        for item in r["artifacts"]:
            assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
        assert (
            f"CRAFTAX_REGISTRATION_ACCEPTANCE_PASS {family} 6679082"
            in (e / "slurm-6679082.log").read_text().splitlines()
        )
        manifests += 1
    for name in ["Cube", "Scene"]:
        e = ROOT / "docs/research/semantic_recovery_evidence/ogb_renderer_retry"
        r = json.loads((e / (name + "-acceptance.json")).read_text())
        assert set(r["gates"].values()) == {"PASS"}
        for item in r["artifacts"]:
            assert sha(ROOT / item["path"]) == item["sha256"], item["path"]
        manifests += 1
    prior = json.loads((ROOT / "docs/research/semantic_recovery_evidence/pusht_phase_a/audit.json").read_text())
    for rel, h in prior["artifact_hashes"].items():
        assert sha(ROOT / rel) == h
    for rel, item in prior["terminal_logs"].items():
        assert sha(ROOT / rel) == item["sha256"]
        assert all(s in (ROOT / rel).read_text().splitlines() for s in item["exact_terminal_lines"])
    # Prior seals include the live bookkeeping tables: resolve those against the
    # containing commit, while checking every immutable evidence artifact on disk.
    craftax_seals = {}
    for name in ["SHA256SUMS", "DOWNLOAD-SHA256SUMS"]:
        path = ROOT / "docs/research/semantic_recovery_evidence/craftax_gpu_6679082" / name
        entries = path.read_text().splitlines()
        for line in entries:
            expected, rel = line.split("  ", 1)
            data = git("show", START + ":" + rel) if rel in allowed else (ROOT / rel).read_bytes()
            assert hashlib.sha256(data).hexdigest() == expected, rel
        craftax_seals[name] = len(entries)
    # Counterexample verification uses retained arrays, not the collector's summary alone.
    comparisons = 0
    for directory, case_path in [("feasibility-final", "cases"), ("controls-final", "landing_contact_replay")]:
        p = EVIDENCE / directory
        r = json.loads((p / "result.json").read_text())
        cases = r[case_path] if case_path == "cases" else {"landing_contact": r[case_path]}
        for name, case in cases.items():
            with (
                np.load(p / (name + "-snapshot.npz"), allow_pickle=False) as snapshot,
                np.load(p / (name + "-reference.npz"), allow_pickle=False) as reference,
            ):
                for mode, record in case["routes"].items():
                    with (
                        np.load(p / (name + "-" + mode + "-immediate.npz"), allow_pickle=False) as immediate,
                        np.load(p / (name + "-" + mode + ".npz"), allow_pickle=False) as replay,
                    ):
                        assert compare(snapshot, immediate) == record["immediate_differences"]
                        assert compare(reference, replay) == record["trajectory_differences"]
                        comparisons += 2
    r = json.loads((EVIDENCE / "feasibility-final/result.json").read_text())
    near = r["cases"]["near_contact"]["routes"]["numeric"]
    assert set(near["immediate_differences"]) == {"diagnostic/cached_links"}
    assert (
        "wrapper/reward" in near["trajectory_differences"] and "wrapper/termination" in near["trajectory_differences"]
    )
    assert max(near["different_history_contacts"]) > 0
    p = EVIDENCE / "controls-final"
    r = json.loads((p / "result.json").read_text())
    assert r["clean_independent_same_seed_actions"] == {}
    with (
        np.load(p / "independent-reference.npz", allow_pickle=False) as a,
        np.load(p / "independent-restored.npz", allow_pickle=False) as b,
    ):
        assert compare(a, b) == r["independent_official_file_restore"]["trajectory_differences"]
        comparisons += 1
    with np.load(p / "source-frame-restore.npz", allow_pickle=False) as frames:
        assert (
            np.count_nonzero(frames["before"] != frames["after"])
            == r["official_restore_immediate_pixels"]["changed_channels"]
            > 0
        )
    assert r["successful_landing"]["info"]["env_complete"]
    provenance = json.loads((EVIDENCE / "provenance.json").read_text())
    for rel, item in provenance["files"].items():
        assert sha(Path(item["path"])) == item["sha256"], rel
    result = {
        "status": "ROCKET_BLOCKER_AND_PRESERVATION_VERIFIED",
        "starting_commit": START,
        "matrix_counts": counts,
        "accepted_rows_preserved": 39,
        "non_rocket_rows_preserved": 41,
        "unchanged_starting_files": unchanged,
        "craftax_cube_scene_acceptance_manifests": manifests,
        "craftax_containing_commit_seals": craftax_seals,
        "earlier_artifact_hashes": len(prior["artifact_hashes"]),
        "earlier_terminal_logs": len(prior["terminal_logs"]),
        "independent_array_comparisons": comparisons,
        "source_files_hashed": len(provenance["files"]),
        "production_changes": [],
        "shared_contract_changes": [],
        "promotion": False,
        "push_performed": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
