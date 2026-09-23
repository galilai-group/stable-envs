"""Read-only accepted-evidence closure and exact source identity audit."""

import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
START = "b45500ca84d8c6b3216d90196acac747e8fb3d5b"


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


p = json.loads((ROOT / "docs/research/semantic_recovery_implementation_map.json").read_text())
artifacts, markers, verified = {}, {}, []
for row in p["rows"]:
    if row["current_status"] != "VERIFIED":
        continue
    verified.append(row["historical_id"])
    for item in row["evidence_manifest"]["artifacts"]:
        path, expected = item["path"], item["sha256"]
        assert sha(ROOT / path) == expected, path
        artifacts[path] = expected
    for key in ("closure_evidence", "family_closure_evidence"):
        if key not in row:
            continue
        closure = row[key]
        path = closure["terminal_log"]
        marker = closure["actual_success_marker"]
        lines = (ROOT / path).read_text().splitlines()
        matches = [line for line in lines if line == marker or line.startswith(marker + " ")]
        assert matches, (path, marker)
        actual = sha(ROOT / path)
        expected = closure.get("terminal_log_sha256", artifacts.get(path))
        assert expected is not None and actual == expected, path
        markers[path] = {"sha256": actual, "map_marker": marker, "exact_terminal_lines": matches}
        if "renderer_sha256" in closure:
            assert sha(ROOT / closure["renderer_manifest"]) == closure["renderer_sha256"]
assert len(verified) == 33
# Whole accepted Fetch tree, including duplicate GPU evidence, remains intact.
fetch_manifest = ROOT / "docs/research/semantic_recovery_evidence/fetch_family/SHA256SUMS"
fetch_count = 0
for line in fetch_manifest.read_text().splitlines():
    expected, path = line.split(maxsplit=1)
    path = path.lstrip("*")
    candidate = ROOT / path
    if not candidate.exists():
        candidate = fetch_manifest.parent / path
    assert sha(candidate) == expected, path
    fetch_count += 1
swm = Path("/Users/samib/stable-worldmodel")
engine = Path("/private/tmp/pusht-pymunk-source-7.2.0")
assert git("rev-parse", "HEAD", cwd=swm) == "176d54532b3c0a251a4f509b74f2fdacc143a63d"
assert git("diff", "--name-only", "HEAD", cwd=swm) == ""
swm_files = [
    "stable_worldmodel/envs/pusht/env.py",
    "stable_worldmodel/envs/pusht/env_discrete.py",
    "stable_worldmodel/envs/quantizer.py",
    "stable_worldmodel/envs/pusht/expert_policy.py",
    "stable_worldmodel/envs/__init__.py",
    "stable_worldmodel/world/world.py",
    "stable_worldmodel/spaces.py",
    "uv.lock",
]
engine_files = [
    "pymunk/body.py",
    "pymunk/space.py",
    "pymunk/_pickle.py",
    "pymunk_cffi/chipmunk_cdef.h",
    "pymunk_cffi/extensions.c",
    "Munk2D/src/cpBody.c",
    "Munk2D/src/cpSpaceStep.c",
    "Munk2D/src/cpArbiter.c",
    "Munk2D/include/chipmunk/chipmunk_structs.h",
]
installed = Path("/private/tmp/pusht-authority-swm-lock/lib/python3.12/site-packages/pymunk")
result = {
    "status": "PASS",
    "starting_branch": "repair/environment-semantics",
    "starting_head": START,
    "fetch_promotion_sha": START,
    "verified_ids": verified,
    "incomplete_ids": [r["historical_id"] for r in p["rows"] if r["current_status"] != "VERIFIED"],
    "artifact_hashes": artifacts,
    "terminal_logs": markers,
    "artifact_count": len(artifacts),
    "terminal_log_count": len(markers),
    "fetch_manifest_sha256": sha(fetch_manifest),
    "fetch_manifest_files_checked": fetch_count,
    "swm_commit": git("rev-parse", "HEAD", cwd=swm),
    "swm_tracked_diff": "",
    "swm_source_hashes": {f: sha(swm / f) for f in swm_files},
    "pymunk_release_commit": git("rev-parse", "HEAD", cwd=engine),
    "munk2d_commit": git("rev-parse", "HEAD", cwd=engine / "Munk2D"),
    "engine_source_hashes": {f: sha(engine / f) for f in engine_files},
    "installed_pymunk_hashes": {
        str(f.relative_to(installed)): sha(f) for f in sorted(installed.iterdir()) if f.suffix in (".py", ".so")
    },
    "production_diff_from_start": git("diff", START, "--", "stable_envs", "tests", "pyproject.toml"),
    "accepted_map_diff_from_start": git(
        "diff",
        START,
        "--",
        "docs/research/semantic_recovery_implementation_map.json",
        "docs/research/semantic_recovery_matrix.json",
    ),
}
assert result["production_diff_from_start"] == result["accepted_map_diff_from_start"] == ""
(HERE / "audit.json").write_text(json.dumps(result, indent=2) + "\n")
print(f"ALL_33_CLOSURE_PASS artifacts={len(artifacts)} terminal_logs={len(markers)} fetch_files={fetch_count}")
