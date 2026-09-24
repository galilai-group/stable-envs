"""Read-only revalidation of the 35 accepted registrations around Craftax work."""

from __future__ import annotations

import argparse
import ast
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
START = "bcf8699d9983f95e97b3c71607521a61c354247b"
TARGETS = {
    "swm/CraftaxClassicPixels-v1",
    "swm/CraftaxClassicSymbolic-v1",
    "swm/CraftaxPixels-v1",
    "swm/CraftaxSymbolic-v1",
}


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    allowed = {
        "pyproject.toml",
        "docs/research/index.md",
        "docs/research/semantic_recovery_matrix.json",
        "docs/research/semantic_recovery_implementation_map.json",
    }
    unchanged = []
    for line in git("ls-tree", "-r", START).decode().splitlines():
        meta, rel = line.split("\t")
        if rel in allowed:
            continue
        data = (ROOT / rel).read_bytes()
        assert hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest() == meta.split()[2], rel
        unchanged.append(rel)
    for name in ["semantic_recovery_matrix.json", "semantic_recovery_implementation_map.json"]:
        rel = "docs/research/" + name
        before = json.loads(git("show", START + ":" + rel))
        after = json.loads((ROOT / rel).read_text())
        old = {r["historical_id"]: r for r in before["rows"]}
        new = {r["historical_id"]: r for r in after["rows"]}
        assert old.keys() == new.keys()
        assert all(old[k] == new[k] for k in old if k not in TARGETS)
        if name == "semantic_recovery_matrix.json":
            assert sum(r["state_status"] == "VERIFIED" for r in old.values()) == 35
            counts = dict(Counter(r["state_status"] for r in new.values()))
            assert counts == {"VERIFIED": 35, "INCOMPLETE": 7}
    evidence = ROOT / "docs/research/semantic_recovery_evidence"
    accepted = json.loads((evidence / "pusht_phase_a/audit.json").read_text())
    for rel, sha in accepted["artifact_hashes"].items():
        assert digest((ROOT / rel).read_bytes()) == sha, rel
    for rel, item in accepted["terminal_logs"].items():
        assert digest((ROOT / rel).read_bytes()) == item["sha256"], rel
        assert all(s in (ROOT / rel).read_text().splitlines() for s in item["exact_terminal_lines"])
    for family in ["Cube", "Scene"]:
        accepted_cube = json.loads((evidence / "ogb_renderer_retry" / (family + "-acceptance.json")).read_text())
        assert set(accepted_cube["gates"].values()) == {"PASS"}
        for item in accepted_cube["artifacts"]:
            assert digest((ROOT / item["path"]).read_bytes()) == item["sha256"], item["path"]
    manifests, historical = {}, []
    for rel in [
        "ogb_renderer_retry/GPU-SHA256SUMS",
        "ogb_renderer_retry/SHA256SUMS",
        "craftax_source_audit/SHA256SUMS",
    ]:
        manifest = evidence / rel
        for line in manifest.read_text().splitlines():
            sha, path = line.split(maxsplit=1)
            path = path.lstrip("*")
            p = ROOT / path
            if not p.exists():
                p = manifest.parent / path
            data = p.read_bytes()
            relative = str(p.relative_to(ROOT))
            if digest(data) != sha and relative in allowed:
                data = git("show", START + ":" + relative)
                historical.append(relative)
            assert digest(data) == sha, (rel, path)
        manifests[rel] = len(manifest.read_text().splitlines())
    provenance = json.loads((evidence / "craftax_source_audit/provenance.json").read_text())
    for rel, item in provenance["files"].items():
        assert digest(Path(item["path"]).read_bytes()) == item["sha256"], rel
    swm = Path("/Users/samib/stable-worldmodel")
    assert git("rev-parse", "HEAD", cwd=swm).decode().strip() == provenance["swm_commit"]
    assert not git("diff", "HEAD", "--name-only", cwd=swm)
    imports = []
    for p in (ROOT / "stable_envs").rglob("*.py"):
        for node in ast.walk(ast.parse(p.read_text())):
            names = (
                [a.name for a in node.names]
                if isinstance(node, ast.Import)
                else [node.module or ""]
                if isinstance(node, ast.ImportFrom)
                else []
            )
            imports.extend(str(p) for n in names if n.startswith("stable_worldmodel"))
    assert not imports
    result = {
        "status": "CRAFTAX_PRESERVATION_PASS",
        "starting_commit": START,
        "unchanged_starting_files": len(unchanged),
        "allowed_starting_changes": sorted(allowed),
        "previously_accepted_rows_unchanged": 35,
        "non_craftax_rows_unchanged": 38,
        "accepted_artifacts": len(accepted["artifact_hashes"]),
        "accepted_terminal_logs": len(accepted["terminal_logs"]),
        "cube_scene_acceptances": "all gates PASS; all referenced artifact hashes exact",
        "manifests": manifests,
        "mutable_manifest_entries_resolved_at_original_commit": historical,
        "upstream_source_files_unchanged": len(provenance["files"]),
        "matrix_counts": counts,
        "production_swm_imports": imports,
        "shared_contract_changes": [],
        "frozen_renderer_changed": False,
        "push_performed": False,
    }
    args.output.write_text(json.dumps(result, indent=2) + "\n")
    print(result["status"])


if __name__ == "__main__":
    main()
