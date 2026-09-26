"""Read-only preservation audit for the approval-blocked Rocket preflight."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = "3d0a799d329bb00fbecbf708e970be18a14ce611"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    assert not a.output.exists()

    def git(*args):
        return subprocess.check_output(["git", *args], cwd=ROOT, text=True)

    tracked = git("ls-tree", "-r", "--name-only", START).splitlines()
    changed = git("diff", "--name-only", START).splitlines()
    assert changed == ["docs/research/index.md"], changed
    start = json.loads(
        (ROOT / "docs/research/semantic_recovery_evidence/state_performance_start/start.json").read_text()
    )
    for group in ["artifact_hashes", "production_hashes"]:
        for path, digest in start[group].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
    seal = ROOT / "docs/research/semantic_recovery_evidence/branch_replay_final/REPOSITORY-SHA256SUMS"
    count = 0
    for line in seal.read_text().splitlines():
        digest, path = line.split("  ", 1)
        if path == "docs/research/index.md":
            old = subprocess.check_output(["git", "show", START + ":" + path], cwd=ROOT)
            assert (ROOT / path).read_bytes().startswith(old)
        else:
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == digest, path
        count += 1
    matrix = json.loads((ROOT / "docs/research/semantic_recovery_matrix.json").read_text())
    counts = dict(Counter(r["state_status"] for r in matrix["rows"]))
    assert counts == {"VERIFIED": 39, "INCOMPLETE": 3}
    for path in [
        "docs/research/semantic_recovery_matrix.json",
        "docs/research/semantic_recovery_implementation_map.json",
    ]:
        assert (ROOT / path).read_bytes() == subprocess.check_output(["git", "show", START + ":" + path], cwd=ROOT)
    metadata = json.loads((ROOT / "docs/research/reset_replay_classification.json").read_text())
    assert len(metadata["rows"]) == 42
    for row, accepted in zip(metadata["rows"], matrix["rows"], strict=True):
        assert row["historical_id"] == accepted["historical_id"]
        assert row["existing_reset_parity"] == accepted["reset_parity"]
        assert row["existing_rng_lifecycle"] == accepted["rng_lifecycle"]
    result = dict(
        status="PRESERVATION_PASS_PRODUCTION_AND_GPU_PENDING",
        head=git("rev-parse", "HEAD").strip(),
        branch=git("branch", "--show-current").strip(),
        worktree=str(ROOT),
        matrix_counts=counts,
        prior_tracked_files=len(tracked),
        prior_changed_files=changed,
        prior_root_seal_entries_checked=count,
        accepted_artifact_hashes=len(start["artifact_hashes"]),
        production_hashes=len(start["production_hashes"]),
        matrix_map_unchanged=True,
        reset_metadata_rows=42,
        production_changes=[],
        pushT_changes=[],
        new_commits=[],
        push_performed=False,
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
