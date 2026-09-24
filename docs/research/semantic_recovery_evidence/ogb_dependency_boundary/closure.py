"""Verify unchanged starting tree and all accepted evidence; do not rewrite old records."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
START = "434cc72974360ab98e47cad7aebcc8b1b62bb1de"
DOCS = "03adc1ab79bae71e3e62db2e5c4cb19f20723166"


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def main():
    subprocess.run(["git", "merge-base", "--is-ancestor", DOCS, START], cwd=ROOT, check=True)
    old = git("ls-tree", "-r", "--name-only", START).splitlines()
    for rel in old:
        data = subprocess.check_output(["git", "show", f"{START}:{rel}"], cwd=ROOT)
        assert (ROOT / rel).read_bytes() == data, rel
    accepted = json.loads((HERE.parent / "pusht_phase_a/audit.json").read_text())
    for rel, digest in accepted["artifact_hashes"].items():
        assert sha(ROOT / rel) == digest, rel
    for rel, entry in accepted["terminal_logs"].items():
        assert sha(ROOT / rel) == entry["sha256"], rel
        assert all(s in (ROOT / rel).read_text().splitlines() for s in entry["exact_terminal_lines"])
    manifest_counts = {}
    for family in ["fetch_family", "ogb_rng_feasibility"]:
        manifest = HERE.parent / family / "SHA256SUMS"
        lines = manifest.read_text().splitlines()
        for line in lines:
            digest, rel = line.split(maxsplit=1)
            path = ROOT / rel.lstrip("*")
            if not path.exists():
                path = manifest.parent / rel.lstrip("*")
            assert sha(path) == digest, str(path)
        manifest_counts[family] = len(lines)
    matrix = json.loads((ROOT / "docs/research/semantic_recovery_matrix.json").read_text())
    counts = {key: sum(r["state_status"] == key for r in matrix["rows"]) for key in ["VERIFIED", "INCOMPLETE"]}
    assert counts == {"VERIFIED": 33, "INCOMPLETE": 9}
    rows = [r for r in matrix["rows"] if r["historical_id"] in ["swm/OGBCube-v0", "swm/OGBScene-v0"]]
    assert len(rows) == 2 and all(r["adapter_status"] == "NOT_IMPLEMENTED" for r in rows)
    imports = []
    for path in (ROOT / "stable_envs").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports.extend(
                    (str(path.relative_to(ROOT)), a.name) for a in node.names if a.name.startswith("stable_worldmodel")
                )
            elif isinstance(node, ast.ImportFrom) and (node.module or "").startswith("stable_worldmodel"):
                imports.append((str(path.relative_to(ROOT)), node.module))
    assert not imports, imports
    swm = Path("/Users/samib/stable-worldmodel")
    assert git("rev-parse", "HEAD", cwd=swm) == "176d54532b3c0a251a4f509b74f2fdacc143a63d"
    assert not git("diff", "HEAD", "--name-only", cwd=swm)
    prefix = str(HERE.relative_to(ROOT)) + "/"
    report = "docs/research/ogbench_dependency_boundary.md"
    changed = set(git("diff", "--name-only", START).splitlines())
    changed.update(git("ls-files", "--others", "--exclude-standard").splitlines())
    assert all(p.startswith(prefix) or p == report for p in changed), changed
    record = {
        "result": "PRESERVATION_PASS",
        "stage_a": "FAIL_NO_COMPLIANT_RESET_DELEGATION",
        "stage_b": "NOT_STARTED",
        "start": START,
        "rng_checkpoint": START,
        "documentation_checkpoint": DOCS,
        "branch": git("branch", "--show-current"),
        "worktree": str(ROOT),
        "starting_status": "clean",
        "unchanged_starting_files": len(old),
        "accepted_artifacts": len(accepted["artifact_hashes"]),
        "terminal_logs": len(accepted["terminal_logs"]),
        "manifests": manifest_counts,
        "matrix": counts,
        "production_swm_imports": imports,
        "production_diff": "",
        "push_performed_this_checkpoint": False,
        "changed_inventory": sorted([report] + [str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.is_file()]),
    }
    (HERE / "closure.json").write_text(json.dumps(record, indent=2) + "\n")
    print(
        f"OGB_BOUNDARY_PRESERVATION_PASS tracked={len(old)} accepted_artifacts=170 terminal_logs=11 fetch_files=75 matrix=33/9"
    )


if __name__ == "__main__":
    main()
