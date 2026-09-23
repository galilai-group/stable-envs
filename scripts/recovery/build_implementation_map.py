"""Build the Phase A 21-row closure snapshot without simulator runs.

Run at the pre-Fetch-promotion matrix. Later accepted rows update the canonical
map with their own acceptance manifests, preserving this original closure.
"""

import argparse
import ast
import hashlib
import json
import subprocess
from collections import Counter
from pathlib import Path

import tomllib

PIN = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
BASELINE = "6d80b68b8b3ccd07d4207be94cbf4f0760758a06"
ROOT = Path(__file__).resolve().parents[2]
EVID = Path("docs/research/semantic_recovery_evidence")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def git(*args):
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True).strip()


def read(p):
    return json.loads((ROOT / p).read_text())


def ast_hash(data):
    return sha(ast.dump(ast.parse(data), include_attributes=False).encode())


def main(swm):
    assert subprocess.check_output(["git", "-C", str(swm), "rev-parse", "HEAD"], text=True).strip() == PIN
    assert not subprocess.check_output(["git", "-C", str(swm), "status", "--short", "--untracked-files=no"], text=True)
    matrix = read("docs/research/semantic_recovery_matrix.json")
    counts = Counter((r["family"], r["state_status"]) for r in matrix["rows"])
    assert counts == {
        ("dmcontrol", "VERIFIED"): 12,
        ("classic_control", "VERIFIED"): 5,
        ("navigation", "VERIFIED"): 3,
        ("ogbench", "VERIFIED"): 1,
        ("fetch", "INCOMPLETE"): 12,
        ("ogbench", "INCOMPLETE"): 2,
        ("craftax", "INCOMPLETE"): 4,
        ("rocket", "INCOMPLETE"): 1,
        ("pusht", "INCOMPLETE"): 2,
    }, counts
    assert sha((ROOT / matrix["source_audit"]).read_bytes()) == matrix["source_audit_sha256"]
    # Execute only the registration table: no package imports or environment construction.
    inventory = {}

    def register(id, entry_point, **kw):
        inventory[id] = {"entry_point": entry_point, **kw}

    tree = ast.parse((swm / "stable_worldmodel/envs/__init__.py").read_text())
    tree.body = [n for n in tree.body if isinstance(n, (ast.Assign, ast.Expr, ast.For))]
    exec(compile(tree, "pinned-registration-table", "exec"), {"register": register})
    assert set(inventory) == {r["historical_id"] for r in matrix["rows"]}
    lock = tomllib.loads((swm / "uv.lock").read_text())
    versions = {p["name"]: p["version"] for p in lock["package"]}
    repos = {
        "dmcontrol": ("dm-control", "https://github.com/google-deepmind/dm_control"),
        "classic_control": ("gymnasium", "https://github.com/Farama-Foundation/Gymnasium"),
        "fetch": ("gymnasium-robotics", "https://github.com/Farama-Foundation/Gymnasium-Robotics"),
        "ogbench": ("ogbench", "https://github.com/seohongpark/ogbench"),
        "craftax": ("craftax", "https://github.com/MichaelTMatthews/Craftax"),
        "rocket": ("pyflyt", "https://github.com/jjshoots/PyFlyt"),
        "pusht": ("pymunk", "https://github.com/viblo/pymunk"),
        "navigation": (None, None),
    }
    catalog = {}

    def source(path, repository=swm, oracle=True):
        key = ("swm:" if oracle else "candidate:") + str(path)
        if key in catalog:
            return key
        data = (repository / path).read_text()
        tree = ast.parse(data)
        methods = {}
        for cls in tree.body:
            if isinstance(cls, ast.ClassDef):
                for f in cls.body:
                    if isinstance(f, ast.FunctionDef) and (
                        f.name
                        in (
                            "__init__",
                            "reset",
                            "step",
                            "set_state",
                            "get_state",
                            "get_reward",
                            "_get_obs",
                            "_set_action",
                            "_reset_sim",
                            "_sample_goal",
                        )
                        or "reward" in f.name
                    ):
                        methods[cls.name + "." + f.name] = {
                            "line": f.lineno,
                            "end_line": f.end_lineno,
                            "signature": ast.unparse(f.args),
                            "source": ast.get_source_segment(data, f),
                        }
        catalog[key] = {"path": str(path), "sha256": sha(data.encode()), "methods": methods}
        if oracle:
            catalog[key]["url"] = f"https://github.com/galilai-group/stable-worldmodel/blob/{PIN}/{path}"
        return key

    # Verify immutable hashes, including reference-only corpus arrays.
    checks = []
    for p in sorted((ROOT / EVID).rglob("SHA256SUMS")):
        for line in p.read_text().splitlines():
            if not line.strip():
                continue
            h, name = line.split(maxsplit=1)
            q = p.parent / name.lstrip("*")
            assert q.exists() and sha(q.read_bytes()) == h, (p, name)
            checks.append(str(q.relative_to(ROOT)))
    for p in sorted((ROOT / "tests/fixtures/semantics").glob("*/manifest.json")):
        m = json.loads(p.read_text())
        assert m["reference_commit"] == PIN, p
        entries = [m] + list(m.get("domains", {}).values()) + [m.get("domain", {})]
        for e in entries:
            if "file" in e and "sha256" in e:
                assert sha((p.parent / e["file"]).read_bytes()) == e["sha256"], p
                checks.append(str((p.parent / e["file"]).relative_to(ROOT)))
    # Latest DMC run includes all previous DMC semantic regression gates.
    stability = {}
    groups = {
        "dmcontrol": (
            "07d7098",
            list((ROOT / "stable_envs/envs/dmcontrol").glob("*.py"))
            + list((ROOT / "stable_envs/_adapters").glob("dmcontrol*.py")),
        ),
        "classic_control": (
            "98c65cb",
            [ROOT / "stable_envs/envs/classic_control.py", ROOT / "stable_envs/envs/_classic_semantics.py"],
        ),
        "navigation": ("6ee8d8cb", [ROOT / "stable_envs/envs/navigation.py"]),
        "ogbench": ("7f2f3ae", [ROOT / "stable_envs/_adapters/ogbench.py", ROOT / "stable_envs/envs/ogbench/maze.py"]),
    }
    for family, (rev, paths) in groups.items():
        files = {}
        for p in paths:
            rel = str(p.relative_to(ROOT))
            old = subprocess.check_output(["git", "show", f"{rev}:{rel}"], cwd=ROOT)
            assert ast_hash(old) == ast_hash(p.read_bytes()), (family, rel)
            files[rel] = {"sha256": sha(p.read_bytes()), "ast_sha256": ast_hash(old)}
        stability[family] = {
            "covered_revision": git("rev-parse", rev),
            "production_ast_unchanged": True,
            "files": files,
        }
    rows = []
    for r in matrix["rows"]:
        ident = r["historical_id"]
        reg = inventory[ident]
        module, cls = reg["entry_point"].split(":")
        path = Path(module.replace(".", "/") + ".py")
        if not (swm / path).exists():
            path = Path(module.replace(".", "/")) / "__init__.py"
        source_key = source(path)
        entry = catalog[source_key]
        init = entry["methods"].get(cls + ".__init__", {})
        family = r["family"]
        package, repo = repos[family]
        domain = path.stem
        if ident.startswith("swm/MountainCarContinuous"):
            domain = "mountain_car_continuous"
        if family == "navigation":
            domain = {
                "TwoRoomEnv": "two_room",
                "PiecewiseEnv": "piecewise",
                "SimplePointMazeEnv": "simple_point_maze",
            }[cls]
        manifest = r.get("oracle_corpus")
        corpus = read(manifest) if manifest and (ROOT / manifest).exists() else {}
        cell = corpus.get("domains", {}).get(domain, corpus.get("domain", corpus))
        fields = {k.removeprefix("state/"): v for k, v in cell.get("arrays", {}).items() if k.startswith("state/")}
        evidence = list(r.get("evidence", []))
        for e in evidence:
            assert (ROOT / e).exists(), e
        if family == "dmcontrol":
            adapter = (
                "dmcontrol_humanoid"
                if domain == "humanoid"
                else "dmcontrol_aligned"
                if domain in ("ball_in_cup", "finger", "manipulator", "quadruped")
                else "dmcontrol_continuation"
                if domain in ("cheetah", "hopper", "walker")
                else "dmcontrol"
            )
            production = ["stable_envs/_adapters/" + adapter + ".py"]
            order = [
                "validate full named State",
                "restore qpos/qvel/time and task model coordinates/activation where present",
                "physics.forward()",
                "restore qacc_warmstart after forward",
                "restore lifecycle counters, cumulative reward and complete MT19937 state",
            ]
            action = "Two upstream steps per public action; cast to action_spec dtype; sum rewards. Finite out-of-range actions pass through; no extra clipping."
            common = source(Path("stable_worldmodel/envs/dmcontrol/dmcontrol.py"))
        elif family == "classic_control":
            production = ["stable_envs/envs/_classic_semantics.py"]
            order = [
                "validate full named State",
                "restore physical state and task cache",
                "restore TimeLimit and complete PCG64 state",
            ]
            action = (
                "One Gymnasium step; task-specific discrete/continuous validation and clipping delegated unchanged."
            )
            common = None
        elif family == "navigation":
            production = ["stable_envs/envs/navigation.py"]
            order = [
                "validate every named field and dtype",
                "restore position, goal and cached reset/goal fields without recomputing them",
            ]
            action = "Pinned native step arithmetic, agent.speed and domain collision/zone rules; no wrapper repeat."
            common = None
        elif family == "ogbench" and r["state_status"] == "VERIFIED":
            production = ["stable_envs/_adapters/ogbench.py"]
            order = [
                "restore qpos/qvel/time/target/task id",
                "mj_forward",
                "restore warmstart",
                "restore PCG64 and virtualized MT19937",
            ]
            action = "Point: add 0.2 * action; zero qvel; five MuJoCo substeps; no invented TimeLimit."
            common = None
        elif ident == "swm/FetchPickAndPlace-v3":
            production = ["stable_envs/_adapters/fetch_pickandplace.py"]
            order = [
                "qpos/qvel/goal; zero unsupported external applied forces",
                "mj_forward",
                "warmstart",
                "controller cached position/quaternion",
                "TimeLimit and complete named PCG64",
            ]
            action = "Clip [-1,1]; XYZ *0.05; fixed quaternion [1,0,1,0]; duplicate gripper command; 20 substeps, dt0.04; preserve input dtype."
            common = None
        else:
            production = []
            order = [
                "UNPROVED: no accepted full public State restoration; inspect pinned methods before implementation"
            ]
            action = "UNACCEPTED: pinned wrapper step implementation indexed below; no candidate parity claim."
            common = None
        candidate_refs = [source(Path(p), ROOT, False) for p in production]
        method_refs = [source_key] + ([common] if common else [])
        known = list(r["remaining_issues"])
        result = {
            "historical_id": ident,
            "current_status": r["state_status"],
            "family": family,
            "pinned_swm_commit": PIN,
            "swm_implementation": {
                "path": str(path),
                "url": entry["url"],
                "class": cls,
                "source_catalog": method_refs,
            },
            "upstream": {
                "repository": repo,
                "package": package,
                "pinned_lock_version": versions.get(package),
                "tested_profile_versions": corpus.get("versions", {}),
                "version_scope": "Lock version is historical provenance, not a claim of candidate certification; tested CPU/GPU versions remain in evidence.",
            },
            "stable_envs_classes": r["stable_envs_classes"],
            "accepted_classes": r.get("verified_classes", []) if r["state_status"] == "VERIFIED" else [],
            "historical_task_default_kwargs": {
                "registration_kwargs": reg.get("kwargs", {}),
                "constructor_signature": init.get("signature"),
                "accepted_task": r.get("verification_scope", {}).get("task"),
                "source_constructor": source_key + ":" + cls + ".__init__",
            },
            "state_schema": {
                "status": "ACCEPTED_BOUNDED"
                if r["state_status"] == "VERIFIED"
                else "PROVISIONAL"
                if production
                else "UNPROVED",
                "corpus_fields": fields,
                "candidate_source_catalog": candidate_refs,
                "note": r.get("state_schema", r["state_contract"]),
            },
            "restoration_order": order,
            "action_scaling_clipping_repeat": {
                "summary": action,
                "parity": r["action_parity"],
                "source_catalog": method_refs,
            },
            "reset_rng_semantics": {
                "reset_parity": r["reset_parity"],
                "rng_lifecycle": r["rng_lifecycle"],
                "source_catalog": method_refs,
                "rng_structure_policy": "Complete documented named RNG state is atomic; individual-word minimality is not required.",
            },
            "reward_success_terminated_truncated": {
                "reward": r["reward_parity"],
                "lifecycle": r["terminated_truncated_parity"],
                "source_catalog": method_refs,
            },
            "accepted_variation_scope": r["variation_parity"],
            "execution_mode": r["execution_mode"],
            "evidence_manifest": {
                "oracle": manifest,
                "artifacts": [{"path": e, "sha256": sha((ROOT / e).read_bytes())} for e in evidence],
            },
            "cpu_semantic_result": r["step_parity"] if r["state_status"] == "VERIFIED" else "NOT_ACCEPTED",
            "gpu_semantic_result": {
                "status": r.get(
                    "oscar_result", "PASS_EXISTING_EVIDENCE" if r["state_status"] == "VERIFIED" else "NOT_ACCEPTED"
                ),
                "job": r.get("oscar_job"),
                "note": "CUDA State placement is not GPU physics for DELEGATED_CPU.",
            },
            "frozen_renderer_result": r["renderer_regression"],
            "known_limitations": known,
            "remaining_gates": known
            if r["state_status"] == "INCOMPLETE"
            else ["No remaining gate within accepted bounded scope; broader scopes and benchmarks remain unclaimed."],
            "closure_code_check": stability.get(family) if r["state_status"] == "VERIFIED" else None,
        }
        rows.append(result)
    # Reconcile actual terminal markers and renderer cells, including older logs
    # retrieved from scratch. Earlier failures remain evidence, never pass inputs.
    for row in rows:
        if row["current_status"] != "VERIFIED":
            continue
        ident = row["historical_id"]
        family = row["family"]
        if family == "navigation":
            folder = EVID / "navigation_zone_acceptance"
            log = folder / "simple-6642897.out"
            marker = "FAMILY_GPU_GATE_PASS navigation-zone-fix-6ee8d8cb"
            render = folder / "navigation-zone-fix-6ee8d8cb-renderer-cuda.json"
            baseline = folder / "navigation-zone-fix-6ee8d8cb-renderer-baseline-cuda.json"
        elif ident == "swm/OGBMaze-v0":
            log = EVID / "implementation_closure/family-6604606.out"
            marker = "FAMILY_GPU_GATE_PASS ogbench-7f2f3ae"
            render = EVID / "first_family_renderer_cuda.json"
            baseline = None
        elif ident in (
            "swm/ReacherDMControl-v0",
            "swm/AcrobotDMControl-v0",
            "swm/PendulumDMControl-v0",
            "swm/CartpoleDMControl-v0",
        ):
            log = (
                EVID
                / "implementation_closure"
                / ("reacher-6597787.out" if "Reacher" in ident else "family-6605333.out")
            )
            marker = (
                "REACHER_AND_FROZEN_RENDERER_GATE_PASS"
                if "Reacher" in ident
                else "FAMILY_GPU_GATE_PASS acrobot-7221759"
            )
            render = EVID / ("renderer_cuda.json" if "Reacher" in ident else "acrobot_identity_oscar.json")
            baseline = None
        else:
            er = [x["path"] for x in row["evidence_manifest"]["artifacts"]]
            log = Path(next(x for x in reversed(er) if x.endswith(".out")))
            marker = "FAMILY_GPU_GATE_PASS"
            render = Path(next(x for x in reversed(er) if x.endswith("-renderer-cuda.json")))
            baseline = Path(str(render).replace("-renderer-cuda.json", "-renderer-baseline-cuda.json"))
        assert marker in (ROOT / log).read_text(), log
        rd = read(render)
        if baseline:
            assert rd == read(baseline), (render, baseline)
        for cls in row["accepted_classes"]:
            assert cls in json.dumps(rd), (ident, cls, render)
        row["closure_evidence"] = {
            "terminal_log": str(log),
            "terminal_log_sha256": sha((ROOT / log).read_bytes()),
            "actual_success_marker": marker,
            "renderer_manifest": str(render),
            "renderer_sha256": sha((ROOT / render).read_bytes()),
            "separate_baseline": str(baseline) if baseline else None,
            "baseline_equality": "PASS: direct dictionary equality"
            if baseline
            else "PASS: preserved equality assertion and cells in immutable original manifest and terminal log",
            "later_code": "Production AST unchanged at latest covering gate; DMC later regression coverage in job 6628919.",
        }
        row["evidence_manifest"]["closure_manifest"] = (
            "docs/research/semantic_recovery_evidence/implementation_closure/closure.json"
        )
    closure = {
        "starting_branch": git("branch", "--show-current"),
        "starting_sha": git("rev-parse", "HEAD"),
        "swm_commit": PIN,
        "counts": {"VERIFIED": 21, "INCOMPLETE": 21},
        "immutable_hashes_checked": checks,
        "production_stability": stability,
        "status": "PASS",
        "closure_discrepancies": [],
        "supplemental_evidence": "Five original Oscar terminal logs retrieved; no accepted gate rerun or production modification.",
    }
    out = {
        "schema": "stable-envs-implementation-map/v1",
        "matrix": "docs/research/semantic_recovery_matrix.json",
        "pinned_swm_commit": PIN,
        "registration_url": f"https://github.com/galilai-group/stable-worldmodel/blob/{PIN}/stable_worldmodel/envs/__init__.py",
        "audit": closure,
        "source_catalog": catalog,
        "rows": rows,
    }
    (ROOT / "docs/research/semantic_recovery_implementation_map.json").write_text(json.dumps(out, indent=2) + "\n")
    closure_path = ROOT / EVID / "implementation_closure/closure.json"
    if not closure_path.exists():
        closure_path.write_text(json.dumps(closure, indent=2) + "\n")
    print("42 source-mapped rows; 21 accepted scopes; immutable hashes and production AST checks pass")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    p.add_argument("--swm", type=Path, required=True)
    main(p.parse_args().swm)
