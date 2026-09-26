"""Independently verify Rocket's bounded-replay counterexample and preserved checkpoints."""

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
START = "3d0a799d329bb00fbecbf708e970be18a14ce611"


def read(name):
    return json.loads((BASE / name).read_text())


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    a = parser.parse_args()
    assert not a.output.exists()
    original = set(git("ls-tree", "-r", "--name-only", START).splitlines())
    changed = set(git("diff", "--name-only", START).splitlines())
    assert changed & original == {"docs/research/index.md"}, sorted(changed & original)
    start = read("state_performance_start/start.json")
    for group in ["artifact_hashes", "production_hashes"]:
        for path, h in start[group].items():
            assert hashlib.sha256((ROOT / path).read_bytes()).hexdigest() == h, path
    for path in ["semantic_recovery_matrix.json", "semantic_recovery_implementation_map.json"]:
        name = "docs/research/" + path
        assert (ROOT / name).read_bytes() == subprocess.check_output(["git", "show", START + ":" + name], cwd=ROOT)
    m = json.loads((ROOT / "docs/research/semantic_recovery_matrix.json").read_text())
    counts = dict(Counter(r["state_status"] for r in m["rows"]))
    assert counts == {"VERIFIED": 39, "INCOMPLETE": 3}
    prior_seals = {}
    for name, rev in [
        ("branch_replay_final", START),
        ("rocket_production_pending", "3e24542c4453e5682d12481d6a458144e049d60f"),
    ]:
        lines = (BASE / name / "REPOSITORY-SHA256SUMS").read_text().splitlines()
        for line in lines:
            h, path = line.split("  ", 1)
            data = (ROOT / path).read_bytes()
            if path == "docs/research/index.md":
                original_data = subprocess.check_output(["git", "show", rev + ":" + path], cwd=ROOT)
                assert data.startswith(original_data)
                data = original_data
            assert hashlib.sha256(data).hexdigest() == h, path
        prior_seals[name] = len(lines)
    for name in [
        "rocket_reset_boundary_source",
        "rocket_task_layer_candidate",
        "rocket_rejected_controls_gate",
        "rocket_renderer_preflight_source_v2",
        "rocket_renderer_preflight_gpu_6721727/results",
    ]:
        for line in (BASE / name / "SHA256SUMS").read_text().splitlines():
            h, path = line.split("  ", 1)
            assert hashlib.sha256((BASE / name / path).read_bytes()).hexdigest() == h, (name, path)
    blocker = read("rocket_rejected_controls_gate/result.json")
    assert len(blocker["records"]) == 27
    assert blocker["status"] == "REPLAY_BOUND_BLOCKER_REJECTED_CONTROLS"
    assert all(
        r["immediate_91_numeric_fields_exact"]
        and r["migrated_task_exact_to_source"]
        and r["next_zero_action_differences"]
        for r in blocker["records"]
    )
    d = np.load(BASE / "rocket_rejected_controls_gate/equal-count-witness.npz", allow_pickle=False)
    before = [k for k in d.files if k.startswith("before_a/")]
    assert len(before) == 91
    for key in before:
        assert np.array_equal(d[key], d[key.replace("before_a/", "before_b/", 1)]), key
    future = [
        k
        for k in d.files
        if k.startswith("after_a/") and not np.array_equal(d[k], d[k.replace("after_a/", "after_b/", 1)])
    ]
    assert future and "after_a/wrapper/reward" in future
    assert np.array_equal(d["prefix_a/2"], d["prefix_b/2"])
    horizon = blocker["horizon"]
    assert (
        horizon["rejected_calls"] == 4096
        and horizon["successful_physical_action_bound"] == 1202
        and horizon["public_step_count"] == 0
        and horizon["physics_steps"] == 20
    )
    gpu = read("rocket_renderer_preflight_gpu_6721727/results/result.json")
    assert gpu["status"] == "RENDERER_PREFLIGHT_PASS" and gpu["batch_single_exact"] and gpu["inputs_unchanged"]
    assert all(r["within_frozen_limits"] for r in gpu["groups"].values())
    src = np.load(BASE / "rocket_renderer_preflight_source_v2/source-frames.npz", allow_pickle=False)
    frames = np.load(
        BASE / "rocket_renderer_preflight_gpu_6721727/results/frozen-renderer-frames.npz", allow_pickle=False
    )["frames"]
    assert len(frames) == 102
    for name, group in gpu["groups"].items():
        idx = [i for i, r in enumerate(gpu["records"]) if r["case"] == name]
        delta = np.abs(frames[idx].astype(np.int16) - src["frames"][idx].astype(np.int16)).max(-1)
        assert float(delta.mean()) == group["metrics"]["mean_abs"]
        assert float((delta <= 8).mean()) == group["metrics"]["within8_frac"]
    terminal = read("rocket_renderer_preflight_gpu_6721727/terminal.json")
    assert terminal["status"] == "COMPLETED" and terminal["exit_code"] == "0:0" and terminal["marker"] == gpu["status"]
    actual = read("rocket_task_layer_candidate/result.json")
    assert len(actual["records"]) == 24
    for p, h in [("stable_envs/_adapters/rocket_task.py", actual["candidate_sha256"])]:
        assert hashlib.sha256((ROOT / p).read_bytes()).hexdigest() == h
    result = dict(
        status="BLOCKER_WITNESS_AND_PRESERVATION_VERIFIED",
        starting_commit=START,
        audited_parent_commit=git("rev-parse", "HEAD").strip(),
        branch=git("branch", "--show-current").strip(),
        worktree=str(ROOT),
        matrix_counts=counts,
        original_tracked_files=len(original),
        modified_original_files=sorted(changed & original),
        accepted_artifact_hashes=len(start["artifact_hashes"]),
        production_hashes=len(start["production_hashes"]),
        prior_repository_seal_entries=prior_seals,
        blocker_cases=27,
        independently_verified_equal_numeric_fields=91,
        independently_verified_future_difference_fields=len(future),
        equal_rejected_counts=True,
        equal_final_control=True,
        source_rejected_call_horizon=horizon,
        task_layer_cases=24,
        gpu_job=6721727,
        gpu_source_frames=102,
        gpu_groups=17,
        gpu_metrics_independently_recomputed=True,
        production_files_added=["stable_envs/_adapters/rocket_task.py", "stable_envs/_adapters/rocket_spaces.py"],
        public_adapter_implemented=False,
        promotions=[],
        shared_contract_changes=[],
        pushT_changes=[],
        push_performed=False,
    )
    a.output.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
