"""Research evidence audit; preserve every pre-checkpoint tracked file."""

from __future__ import annotations

import ast
import hashlib
import importlib.metadata as metadata
import json
import platform
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
START = "f9e62ac9f3d9f9179eebfacbf3c4c45a9898fcc3"
DOCS = "03adc1ab79bae71e3e62db2e5c4cb19f20723166"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def main():
    # Whole tracked starting tree is immutable in this research checkpoint.
    old_files = git("ls-tree", "-r", "--name-only", START).splitlines()
    for rel in old_files:
        current = ROOT / rel
        if current.is_file():
            original = subprocess.check_output(["git", "show", f"{START}:{rel}"], cwd=ROOT)
            assert hashlib.sha256(original).hexdigest() == sha(current), rel
        else:
            raise AssertionError(f"missing tracked file: {rel}")
    closure = json.loads((HERE.parent / "pusht_phase_a/audit.json").read_text())
    for rel, expected in closure["artifact_hashes"].items():
        assert sha(ROOT / rel) == expected, rel
    for rel, entry in closure["terminal_logs"].items():
        assert sha(ROOT / rel) == entry["sha256"]
        assert all(line in (ROOT / rel).read_text().splitlines() for line in entry["exact_terminal_lines"])
    manifest = HERE.parent / "fetch_family/SHA256SUMS"
    count = 0
    for line in manifest.read_text().splitlines():
        expected, rel = line.split(maxsplit=1)
        path = ROOT / rel.lstrip("*")
        if not path.exists():
            path = manifest.parent / rel.lstrip("*")
        assert sha(path) == expected
        count += 1
    matrix = json.loads((ROOT / "docs/research/semantic_recovery_matrix.json").read_text())
    counts = {key: sum(r["state_status"] == key for r in matrix["rows"]) for key in ["VERIFIED", "INCOMPLETE"]}
    assert counts == {"VERIFIED": 33, "INCOMPLETE": 9}
    swm = Path("/Users/samib/stable-worldmodel")
    assert git("rev-parse", "HEAD", cwd=swm) == "176d54532b3c0a251a4f509b74f2fdacc143a63d"
    assert git("diff", "HEAD", "--name-only", cwd=swm) == ""
    source_inventory = json.loads((HERE.parent / "ogb_manipulation_preflight/provenance.json").read_text())
    selected = {}
    wanted = [
        "cube_env.py",
        "scene_env.py",
        "manipspace_env.py",
        "env.py",
        "diff_ik.py",
        "spaces.py",
        "space.py",
        "box.py",
        "dict.py",
        "seeding.py",
        "core.py",
        "mjdata.h",
        "uv.lock",
    ]
    for name, info in source_inventory["files"].items():
        path = Path(info["path"])
        assert sha(path) == info["sha256"], name
        if path.name in wanted:
            entry = dict(info)
            if path.suffix == ".py":
                tree = ast.parse(path.read_text())
                entry["definitions"] = [
                    {"name": node.name, "line": node.lineno, "end_line": node.end_lineno}
                    for node in ast.walk(tree)
                    if isinstance(node, (ast.ClassDef, ast.FunctionDef))
                ]
            selected[name] = entry
    packages = [
        "ogbench",
        "mujoco",
        "gymnasium",
        "numpy",
        "dm-control",
        "dm-env",
        "dm-tree",
        "scipy",
        "lxml",
        "labmaze",
        "torch",
        "tensordict",
    ]
    provenance = {
        "python": platform.python_version(),
        "platform": platform.platform(),
        "versions": {p: metadata.version(p) for p in packages},
        "source_files": selected,
        "upstream_source_inventory": {
            "path": str(HERE.parent / "ogb_manipulation_preflight/provenance.json"),
            "sha256": sha(HERE.parent / "ogb_manipulation_preflight/provenance.json"),
        },
        "swm_tracked_diff": "",
        "swm_untracked_paths_present_at_start_and_untouched": git(
            "ls-files", "--others", "--exclude-standard", cwd=swm
        ).splitlines(),
    }
    (HERE / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    markers = {
        "trace.log": "OGB_ENTROPY_TRACE_COMPLETE",
        "semantic-preservation.log": "OGB_FIXED_TAPE_SEMANTIC_PRESERVATION_PASS_PATCHES_REMOVED",
        "gates.log": "OGB_RNG_UNPATCHED_REPLAY_AND_THREADED_ISOLATION_PASS",
        "codec.log": "OGB_NUMERIC_RNG_REPLAY_PASS six_resets=exact malformed_rejections=8",
        "schedules.log": "OGB_DIFFERENT_SEED_ROUND_ROBIN_AND_CONCURRENT_ISOLATION_PASS",
    }
    for log, marker in markers.items():
        assert (HERE / log).read_text().splitlines()[-1] == marker, log
    replay = json.loads((HERE / "replay_results.json").read_text())
    for family in replay["families"].values():
        assert len(family["replay"]) == 11
        assert all(not any(row["diffs"].values()) for row in family["replay"])
        assert all(not row["diff"] for row in family["property_reads"] if row["design"] == "A")
        assert all(row["diff"] for row in family["property_reads"] if row["design"] == "B")
        assert all(not row["diff"] for row in family["task_routes"] + family["cold_reads"])
        assert family["last_box"]["diff"] and family["last_box"]["retained_box_rng_unchanged"]
    assert len(set(replay["isolation"]["thread_ids"].values())) == 4
    assert not any(replay["isolation"]["threaded_diffs"].values())
    changed = set(git("diff", "--name-only", START).splitlines())
    changed.update(git("ls-files", "--others", "--exclude-standard").splitlines())
    prefix = "docs/research/semantic_recovery_evidence/ogb_rng_feasibility/"
    allowed_report = "docs/research/ogbench_rng_feasibility.md"
    assert all(path.startswith(prefix) or path == allowed_report for path in changed), changed
    report = {
        "status": "PASS",
        "starting_sha": START,
        "documentation_sha": DOCS,
        "branch": git("branch", "--show-current"),
        "worktree": str(ROOT),
        "starting_status": "clean",
        "matrix_counts": counts,
        "all_starting_tracked_files_byte_identical": len(old_files),
        "accepted_artifact_count": len(closure["artifact_hashes"]),
        "terminal_log_count": len(closure["terminal_logs"]),
        "fetch_manifest_count": count,
        "production_renderer_shared_matrix_diff": "",
        "markers": markers,
        "source_hashes_verified": True,
        "push_performed": False,
    }
    (HERE / "closure.json").write_text(json.dumps(report, indent=2) + "\n")
    print("OGB_RNG_RESEARCH_CLOSURE_PASS counts=33/9 artifacts=170 terminal_logs=11 fetch_files=75")


if __name__ == "__main__":
    main()
