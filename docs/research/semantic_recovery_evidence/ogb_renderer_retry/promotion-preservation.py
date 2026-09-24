"""Audit the bounded Cube/Scene promotion without rewriting sealed evidence."""

import ast
import dataclasses
import hashlib
import json
import subprocess
import sys
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[4]
HERE = Path(__file__).resolve().parent
START = "b59df7599eac1951dfd1ac5682b41452db0ac8f5"
CPU = "72612563590f12a081d940394496795a6c024577"
TARGETS = {"swm/OGBCube-v0", "swm/OGBScene-v0"}
sys.path.insert(0, str(ROOT))


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd)


def digest(data):
    return hashlib.sha256(data).hexdigest()


def verify():
    from tests.helpers.reference import VARIATION_MAPS
    from tests.helpers.reference.readiness import inventory

    allowed = {
        "docs/research/semantic_recovery_matrix.json",
        "docs/research/semantic_recovery_implementation_map.json",
        "docs/research/step9_replacement_readiness.md",
        "docs/research/index.md",
        "tests/helpers/reference/readiness.py",
        "tests/helpers/reference/variation_maps/newcells.py",
        "tests/unit/test_variation_maps_newcells.py",
    }
    unchanged = []
    for line in git("ls-tree", "-r", START).decode().splitlines():
        meta, rel = line.split("\t")
        if rel in allowed:
            continue
        data = (ROOT / rel).read_bytes()
        actual = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert actual == meta.split()[2], rel
        unchanged.append(rel)
    counts = None
    for name in ["semantic_recovery_matrix.json", "semantic_recovery_implementation_map.json"]:
        path = "docs/research/" + name
        before = json.loads(git("show", START + ":" + path))
        after = json.loads((ROOT / path).read_text())
        old = {r["historical_id"]: r for r in before["rows"]}
        new = {r["historical_id"]: r for r in after["rows"]}
        assert old.keys() == new.keys()
        assert all(old[k] == new[k] for k in old if k not in TARGETS), name
        if name == "semantic_recovery_matrix.json":
            assert sum(r["state_status"] == "VERIFIED" for r in old.values()) == 33
            counts = dict(Counter(r["state_status"] for r in new.values()))
            assert counts == {"VERIFIED": 35, "INCOMPLETE": 7}
            remaining = sorted(k for k, r in new.items() if r["state_status"] != "VERIFIED")
            assert remaining == sorted(
                [
                    "swm/PushT-v1",
                    "swm/PushT-Discrete-v1",
                    "swm/CraftaxClassicPixels-v1",
                    "swm/CraftaxClassicSymbolic-v1",
                    "swm/CraftaxPixels-v1",
                    "swm/CraftaxSymbolic-v1",
                    "swm/PFRocketLanding-v0",
                ]
            )
            assert {k: v for k, v in after.items() if k not in {"rows", "ogbench_manipulation_acceptance"}} == {
                k: v for k, v in before.items() if k != "rows"
            }
        else:
            assert after["updates"][:-1] == before["updates"]
            assert {k: v for k, v in after.items() if k not in {"rows", "current_counts", "updates"}} == {
                k: v for k, v in before.items() if k not in {"rows", "current_counts", "updates"}
            }
    taxonomy = json.loads((HERE / "taxonomy-before.json").read_text())
    current_maps = json.loads(json.dumps({k: dataclasses.asdict(v) for k, v in VARIATION_MAPS.items()}))
    assert current_maps.keys() == taxonomy["maps"].keys()
    for k, v in taxonomy["maps"].items():
        if k not in {"OGBCube", "OGBScene"}:
            assert current_maps[k] == v, k
    current_rows = {r["swm_id"]: r for r in json.loads(json.dumps(inventory()))}
    for row in taxonomy["readiness"]:
        if row["swm_id"] not in TARGETS:
            assert current_rows[row["swm_id"]] == row, row["swm_id"]
    accepted = json.loads((HERE.parent / "pusht_phase_a/audit.json").read_text())
    for rel, sha in accepted["artifact_hashes"].items():
        assert digest((ROOT / rel).read_bytes()) == sha, rel
    for rel, item in accepted["terminal_logs"].items():
        assert digest((ROOT / rel).read_bytes()) == item["sha256"], rel
        assert all(s in (ROOT / rel).read_text().splitlines() for s in item["exact_terminal_lines"])
    manifests = {}
    historical = {
        "pyproject.toml",
        "stable_envs/envs/ogbench/manipulation_semantics.py",
        "docs/research/semantic_recovery_matrix.json",
        "docs/research/semantic_recovery_implementation_map.json",
    }
    for family in [
        "fetch_family",
        "ogb_rng_feasibility",
        "ogb_dependency_boundary",
        "ogb_cpu_compatibility",
        "ogb_renderer_closeout",
        "ogb_renderer_retry",
    ]:
        filename = "GPU-SHA256SUMS" if family == "ogb_renderer_retry" else "SHA256SUMS"
        manifest = HERE.parent / family / filename
        for line in manifest.read_text().splitlines():
            sha, rel = line.split(maxsplit=1)
            p = ROOT / rel.lstrip("*")
            if not p.exists():
                p = manifest.parent / rel.lstrip("*")
            relpath = str(p.relative_to(ROOT))
            data = p.read_bytes()
            if digest(data) != sha and relpath in historical and family != "ogb_renderer_retry":
                base = START if family == "ogb_renderer_closeout" else CPU
                data = git("show", base + ":" + relpath)
            assert digest(data) == sha, (family, rel)
        manifests[family] = len(manifest.read_text().splitlines())
    for family in ["Cube", "Scene"]:
        acceptance = json.loads((HERE / (family + "-acceptance.json")).read_text())
        assert set(acceptance["gates"].values()) == {"PASS"}
        for item in acceptance["artifacts"]:
            assert digest((ROOT / item["path"]).read_bytes()) == item["sha256"], item["path"]
    source = json.loads((HERE.parent / "ogb_dependency_boundary/provenance.json").read_text())
    for rel, item in source["files"].items():
        assert digest(Path(item["path"]).read_bytes()) == item["sha256"], rel
    swm = Path("/Users/samib/stable-worldmodel")
    assert git("rev-parse", "HEAD", cwd=swm).decode().strip() == "176d54532b3c0a251a4f509b74f2fdacc143a63d"
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
    assert digest((ROOT / "stable_envs/envs/ogbench/manipulation.py").read_bytes()) == (
        "bac4b1ea3e29a04faa5eeec33de0c212cb87bbf488ae0466ea762bd3a140576a"
    )
    assert not git("diff", START, "--", "stable_envs", "pyproject.toml")
    changed = set(git("diff", START, "--name-only").decode().splitlines())
    changed.update(git("ls-files", "--others", "--exclude-standard").decode().splitlines())
    prefix = "docs/research/semantic_recovery_evidence/ogb_renderer_retry/"
    new_allowed = {
        "docs/research/ogbench_renderer_acceptance.md",
        "tests/unit/test_ogbench_manipulation_acceptance.py",
    }
    assert all(p in allowed | new_allowed or p.startswith(prefix) for p in changed), sorted(changed)
    result = dict(
        status="PROMOTION_PRESERVATION_PASS",
        starting_commit=START,
        unchanged_starting_files=len(unchanged),
        allowed_starting_changes=sorted(allowed),
        actual_changed_files=sorted(changed),
        expected_new_files=sorted(new_allowed),
        expected_new_evidence_prefix=prefix,
        previously_accepted_rows_unchanged=33,
        all_other_rows_unchanged=40,
        accepted_artifacts=len(accepted["artifact_hashes"]),
        accepted_terminal_logs=len(accepted["terminal_logs"]),
        manifests=manifests,
        source_files_unchanged=len(source["files"]),
        matrix_counts=counts,
        remaining_incomplete=remaining,
        non_target_readiness_and_variation_maps_unchanged=True,
        production_diff=[],
        production_swm_imports=imports,
        ogbench_fork=False,
        shared_contract_changes=[],
        frozen_renderer_changed=False,
        push_performed=False,
    )
    (HERE / "promotion-preservation.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    verify()
