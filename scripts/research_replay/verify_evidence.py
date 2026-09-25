"""Read-only checks of compact clone/replay evidence and preservation since checkpoint."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
BASE = ROOT / "docs/research/semantic_recovery_evidence"
START = "31153d38c800f44e5c4c0ae1acdd0edca4c5f656"


def read(name):
    return json.loads((BASE / name).read_text())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    assert not a.output.exists()
    changed = subprocess.check_output(["git", "diff", "--name-only", START], cwd=ROOT, text=True).splitlines()
    untracked = subprocess.check_output(
        ["git", "ls-files", "--others", "--exclude-standard"], cwd=ROOT, text=True
    ).splitlines()
    allowed_files = {
        "scripts/benchmarks/rendered_state_ops.py",
        "scripts/benchmarks/run_rendered_matrix.py",
        "scripts/benchmarks/report_rendered_state.py",
        "tests/unit/test_rendered_state_benchmark.py",
    }
    outside_research = [
        path
        for path in set(changed + untracked)
        if not path.startswith(("docs/research/", "scripts/research_replay/")) and path not in allowed_files
    ]
    assert not outside_research, outside_research
    prior = set(
        subprocess.check_output(["git", "ls-tree", "-r", "--name-only", START], cwd=ROOT, text=True).splitlines()
    )
    prior_modified = [v for v in changed if v in prior and v != "docs/research/index.md"]
    assert not prior_modified, prior_modified
    for path in [
        "docs/research/semantic_recovery_matrix.json",
        "docs/research/semantic_recovery_implementation_map.json",
    ]:
        assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", START + ":" + path], cwd=ROOT)
    matrix = json.loads((ROOT / "docs/research/semantic_recovery_matrix.json").read_text())
    counts = dict(Counter(row["state_status"] for row in matrix["rows"]))
    assert counts == {"VERIFIED": 39, "INCOMPLETE": 3}, counts
    start = read("state_performance_start/start.json")
    for group in ["artifact_hashes", "production_hashes"]:
        for path, h in start[group].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == h, path
    seals = {}
    for name in [
        "clone_rocket",
        "clone_pusht",
        "clone_rocket_ownership",
        "clone_pusht_ownership",
        "clone_rocket_ownership_verified",
        "clone_pusht_ownership_verified",
        "replay_rocket",
        "replay_rocket_v2",
        "replay_rocket_v3",
        "replay_pusht",
    ]:
        d = BASE / name
        entries = (d / "SHA256SUMS").read_text().splitlines()
        for line in entries:
            h, path = line.split("  ", 1)
            assert hashlib.sha256((d / path).read_bytes()).hexdigest() == h, (name, path)
        seals[name] = len(entries)
    for engine in ["rocket", "pusht"]:
        clone = read("clone_" + engine + "/result.json")["records"]
        assert len(clone) == 20
        assert all(r["classification"] != "EXACT_INDEPENDENT_CLONE" and r["process_exit_code"] == 0 for r in clone)
        shallow = [r for r in clone if r["method"] == "copy"]
        assert all(r["native_aliased"] for r in shallow)
    ownership = read("clone_rocket_ownership_verified/result.json")["records"]
    assert len(ownership) == 12
    assert all(
        not r["clone_usable_after_source_close"] and not r["source_usable_after_clone_close"] for r in ownership
    )
    portable = [r for r in ownership if r["method"] == "cloudpickle"]
    assert len(portable) == 4
    assert all(not r["fresh_process"]["step_usable"] for r in portable)
    rocket = read("replay_rocket_v3/result.json")
    assert rocket["classification"] == "RESET_ANCHOR_NOT_REPRODUCIBLE"
    assert len(rocket["records"]) == 12
    for r in rocket["records"]:
        assert r["immediate_exact"] and r["suffix_exact"] and r["independent_clients"] and r["independent_storage"]
        assert r["future_rng_draw_exact"] and r["independent_rng"] and all(r["future_seeded_resets_exact"])
        assert r["pixels_exact"] is not False
        assert r["render_between_steps_nonmutating"]
    assert all(r["rejected"] and r["destination_unchanged"] for r in rocket["atomic_invalid_candidates"])
    pairs = 0
    with np.load(BASE / "replay_rocket_v3/unseeded-reset-counterexample.npz", allow_pickle=False) as data:
        for key in data.files:
            assert data[key].dtype.kind in "biuf"
            if key.startswith("source/"):
                assert not np.array_equal(data[key], data[key.replace("source/", "destination/", 1)]), key
                pairs += 1
    assert pairs == len(rocket["future_unseeded_reset"]["numeric_mismatch_keys"]) + len(
        rocket["future_unseeded_reset"]["variation_rng_mismatch_paths"]
    )
    pusht = read("replay_pusht/result.json")["records"]
    assert len(pusht) == 6
    for r in pusht:
        assert r["classification"] == "UNBOUNDED_PUBLIC_HISTORY" and r["spec_max_episode_steps"] is None
        assert r["first_success"] and r["physics_steps_after_success"] == 10240 and not r["truncated_ever"]
        assert r["finite_prefix_replay_exact"] and r["final_render_exact"]
    with np.load(BASE / "replay_pusht/unbounded-history-witness.npz", allow_pickle=False) as data:
        for name in ["swm/PushT-v1", "swm/PushT-Discrete-v1"]:
            assert np.array_equal(data[name + "/source_tail"], data[name + "/replay_tail"])
            assert np.diff(data[name + "/physics_stamps"])[0] == 10240
    result = dict(
        status="COMPACT_RESEARCH_FAILURES_AND_PRESERVATION_VERIFIED",
        starting_commit=START,
        prior_tracked_file_count=len(prior),
        prior_modified_files=prior_modified,
        outside_research_changes=outside_research,
        matrix_counts=counts,
        accepted_artifact_hashes=len(start["artifact_hashes"]),
        production_hashes=len(start["production_hashes"]),
        clone_records=40,
        rocket_verified_ownership_records=12,
        fresh_process_deserializations=4,
        rocket_seeded_case_records=12,
        invalid_candidate_records=6,
        rocket_independently_rechecked_unequal_archive_pairs=pairs,
        pusht_registration_seed_records=6,
        pusht_independently_rechecked_exact_tail_pairs=2,
        seals_verified=seals,
        production_changes=[],
        shared_contract_changes=[],
        promotions=[],
        push_performed=False,
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
