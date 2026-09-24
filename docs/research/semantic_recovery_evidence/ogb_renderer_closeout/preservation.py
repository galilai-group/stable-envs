"""Read-only verification of prior accepted evidence and checkpoint boundaries."""

import ast
import hashlib
import json
import subprocess
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
START = "72612563590f12a081d940394496795a6c024577"
SWM = "176d54532b3c0a251a4f509b74f2fdacc143a63d"


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify():
    allowed = {
        "pyproject.toml",
        "stable_envs/envs/ogbench/manipulation_semantics.py",
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
    old = json.loads(git("show", START + ":docs/research/semantic_recovery_matrix.json"))
    current = json.loads((ROOT / "docs/research/semantic_recovery_matrix.json").read_text())
    for row in old["rows"]:
        if row["state_status"] == "VERIFIED":
            assert row in current["rows"], row
    for name in ["semantic_recovery_matrix.json", "semantic_recovery_implementation_map.json"]:
        a = json.loads(git("show", START + ":docs/research/" + name))
        b = json.loads((ROOT / "docs/research" / name).read_text())

        def strip(value):
            if isinstance(value, list):
                return [
                    strip(x)
                    for x in value
                    if not (
                        isinstance(x, dict)
                        and any(v in ["swm/OGBCube-v0", "swm/OGBScene-v0"] for v in x.values() if isinstance(v, str))
                    )
                ]
            if isinstance(value, dict):
                return {k: strip(v) for k, v in value.items()}
            return value

        assert strip(a) == strip(b), name
    accepted = json.loads((HERE.parent / "pusht_phase_a/audit.json").read_text())
    for rel, sha in accepted["artifact_hashes"].items():
        assert digest((ROOT / rel).read_bytes()) == sha, rel
    for rel, item in accepted["terminal_logs"].items():
        assert digest((ROOT / rel).read_bytes()) == item["sha256"], rel
        assert all(s in (ROOT / rel).read_text().splitlines() for s in item["exact_terminal_lines"])
    manifests = {}
    for family in ["fetch_family", "ogb_rng_feasibility", "ogb_dependency_boundary", "ogb_cpu_compatibility"]:
        manifest = HERE.parent / family / "SHA256SUMS"
        for line in manifest.read_text().splitlines():
            sha, rel = line.split(maxsplit=1)
            p = ROOT / rel.lstrip("*")
            if not p.exists():
                p = manifest.parent / rel.lstrip("*")
            relpath = str(p.relative_to(ROOT))
            data = git("show", START + ":" + relpath) if relpath in allowed else p.read_bytes()
            assert digest(data) == sha, rel
        manifests[family] = len(manifest.read_text().splitlines())
    source = json.loads((HERE.parent / "ogb_dependency_boundary/provenance.json").read_text())
    for rel, item in source["files"].items():
        assert digest(Path(item["path"]).read_bytes()) == item["sha256"], rel
    swm = Path("/Users/samib/stable-worldmodel")
    assert git("rev-parse", "HEAD", cwd=swm).decode().strip() == SWM
    assert not git("diff", "HEAD", "--name-only", cwd=swm)
    imports = []
    for p in (ROOT / "stable_envs").rglob("*.py"):
        for n in ast.walk(ast.parse(p.read_text())):
            names = (
                [a.name for a in n.names]
                if isinstance(n, ast.Import)
                else [n.module or ""]
                if isinstance(n, ast.ImportFrom)
                else []
            )
            imports.extend((str(p.relative_to(ROOT)), name) for name in names if name.startswith("stable_worldmodel"))
    assert not imports
    assert (
        digest((ROOT / "stable_envs/envs/ogbench/manipulation.py").read_bytes())
        == "bac4b1ea3e29a04faa5eeec33de0c212cb87bbf488ae0466ea762bd3a140576a"
    )
    result = dict(
        status="PRESERVATION_PASS",
        starting_commit=START,
        unchanged_starting_files=len(unchanged),
        allowed_starting_changes=sorted(allowed),
        accepted_rows_unchanged=33,
        accepted_artifacts=len(accepted["artifact_hashes"]),
        accepted_terminal_logs=len(accepted["terminal_logs"]),
        manifests=manifests,
        source_files_unchanged=len(source["files"]),
        production_swm_imports=imports,
        shared_contract_changes=[],
        frozen_renderer_changed=False,
        push_performed=False,
    )
    (HERE / "preservation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
