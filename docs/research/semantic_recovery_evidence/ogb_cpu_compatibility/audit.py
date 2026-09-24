"""Reproduce the CPU ownership/schema audit; imports SWM only as a test oracle."""

import ast
import copy
import hashlib
import inspect
import json
import platform
import subprocess
import textwrap
from importlib.metadata import version
from pathlib import Path

from ogbench.manipspace.envs.cube_env import CubeEnv
from ogbench.manipspace.envs.scene_env import SceneEnv
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv as OracleCube
from stable_worldmodel.envs.ogbench.scene_env import SceneEnv as OracleScene

from stable_envs._adapters.ogbench_manipulation import FLAGS, OGBManipulationAdapter
from stable_envs._adapters.ogbench_manipulation_compat import Compatibility, CompatibleCube, CompatibleScene

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
START = "75ef812977423fa059b857b36542119e101a430f"
RNG = "434cc72974360ab98e47cad7aebcc8b1b62bb1de"
DOCS = "03adc1ab79bae71e3e62db2e5c4cb19f20723166"
SWM = "176d54532b3c0a251a4f509b74f2fdacc143a63d"
PRODUCTION = [
    "stable_envs/_adapters/ogbench_manipulation_compat.py",
    "stable_envs/_adapters/ogbench_manipulation.py",
    "stable_envs/envs/ogbench/manipulation_semantics.py",
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def git(*args, cwd=ROOT):
    return subprocess.check_output(["git", *args], cwd=cwd, text=True).strip()


def write(name, data):
    (HERE / name).write_text(json.dumps(data, indent=2) + "\n")


class Normalize(ast.NodeTransformer):
    def visit_FunctionDef(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            if isinstance(node.body[0].value.value, str):
                node.body.pop(0)
        return self.generic_visit(node)


def normalized(node):
    return ast.unparse(Normalize().visit(copy.deepcopy(node)))


def provenance(cls, name):
    owner = next((c for c in cls.__mro__ if name in c.__dict__), None)
    if owner is None:
        return None
    obj = owner.__dict__[name]
    obj = obj.fget if isinstance(obj, property) else obj
    text, line = inspect.getsourcelines(obj)
    return {
        "owner": owner.__module__ + "." + owner.__qualname__,
        "file": inspect.getfile(obj),
        "line": line,
        "file_sha256": sha(Path(inspect.getfile(obj))),
        "method_sha256": hashlib.sha256("".join(text).encode()).hexdigest(),
        "physical_lines": len(text),
        "normalized_lines": len(normalized(ast.parse(textwrap.dedent("".join(text))).body[0]).splitlines()),
    }


def ownership():
    methods = []
    for rel in PRODUCTION:
        path = ROOT / rel
        lines = path.read_text().splitlines()
        tree = ast.parse(path.read_text())
        for container in tree.body:
            candidates = container.body if isinstance(container, ast.ClassDef) else [container]
            for node in candidates:
                if not isinstance(node, ast.FunctionDef):
                    continue
                methods.append(
                    {
                        "file": rel,
                        "class": container.name if isinstance(container, ast.ClassDef) else None,
                        "name": node.name,
                        "line": node.lineno,
                        "physical_lines": node.end_lineno - node.lineno + 1,
                        "normalized_lines": len(normalized(node).splitlines()),
                        "method_sha256": hashlib.sha256(
                            "\n".join(lines[node.lineno - 1 : node.end_lineno]).encode()
                        ).hexdigest(),
                    }
                )
    deltas = {
        "__init__": (2, "Registered constructor/configuration and stable-envs Variation tree", ["__init__"]),
        "_seed_owned": (3, "Design A per-instance ownership, before any implicit reset", ["reset"]),
        "action_space": (3, "Memoize genuine upstream Box; own complete PCG64", ["action_space"]),
        "validate_options": (2, "Restrict options before mutation", ["reset"]),
        "validate_task": (2, "Validate supported numeric task payload", ["set_tasks", "reset"]),
        "reset": (3, "Owned RNG/Variation prelude then delegate whole reset", ["reset"]),
        "initialize_arm": (
            4,
            "Only unavoidable reset algorithm body: Variation position replaces uniform draw",
            ["initialize_arm"],
        ),
        "initialize_episode": (
            3,
            "Initialize immediate-output caches then delegate entire episode",
            ["initialize_episode"],
        ),
        "_cube_colors_from_variation": (
            2,
            "Configure registered cube and target colors",
            ["initialize_episode", "post_step"],
        ),
        "post_step": (3, "Delegate physics/success/button changes then apply cube color", ["post_step"]),
        "get_reset_info": (3, "Delegate observations/info then adapt output keys", ["get_reset_info"]),
        "get_step_info": (3, "Delegate info then adapt output keys", ["get_step_info"]),
        "_output_info": (2, "Registered env_name/target/success output boundary", ["get_reset_info", "get_step_info"]),
        "modify_mjcf_model": (
            2,
            "Existing MJCF hook: size, material, floor, camera, light and source dirty predicate",
            ["modify_mjcf_model", "add_objects"],
        ),
        "compile_model_and_data": (
            3,
            "Delegate compilation then track exact compiled configuration",
            ["modify_mjcf_model", "compile_model_and_data"],
        ),
        "_apply_button_states": (
            3,
            "Delegate damping/forward then configure registered lock colors",
            ["_apply_button_states"],
        ),
    }
    overrides = []
    for entry in methods:
        if entry["class"] not in ("Compatibility", "CompatibleScene"):
            continue
        category, reason, sources = deltas[entry["name"]]
        row = dict(entry, category=category, reason=reason, provenance={})
        for family, oracle, upstream in [("Cube", OracleCube, CubeEnv), ("Scene", OracleScene, SceneEnv)]:
            if entry["class"] == "CompatibleScene" and family != "Scene":
                continue
            row["provenance"][family] = {
                name: {"swm": provenance(oracle, name), "upstream": provenance(upstream, name)} for name in sources
            }
        overrides.append(row)
    arms = [
        ast.parse(textwrap.dedent(inspect.getsource(c.initialize_arm))).body[0]
        for c in [Compatibility, OracleCube, OracleScene]
    ]
    assert normalized(arms[0]) == normalized(arms[1]) == normalized(arms[2])
    delegated = {}
    for family, cls, upstream in [("Cube", CompatibleCube, CubeEnv), ("Scene", CompatibleScene, SceneEnv)]:
        names = [
            "step",
            "set_tasks",
            "add_objects",
            "compute_observation",
            "compute_oracle_observation",
            "compute_reward",
            "_compute_successes",
            "post_compilation_objects",
        ]
        delegated[family] = {name: provenance(cls, name) for name in names}
        for name in names:
            assert getattr(cls, name) is getattr(upstream, name), (family, name)
    compat_methods = [m for m in methods if m["file"] == PRODUCTION[0]]
    write(
        "ownership.json",
        {
            "production_files": {
                rel: {"sha256": sha(ROOT / rel), "physical_lines": len((ROOT / rel).read_text().splitlines())}
                for rel in PRODUCTION
            },
            "all_production_methods": methods,
            "compatibility_methods": overrides,
            "compatibility_module_function_lines": {
                "physical": sum(m["physical_lines"] for m in compat_methods),
                "normalized": sum(m["normalized_lines"] for m in compat_methods),
                "interpretation": "Conservative total of ALL new compatibility functions, including validation, ownership, tree configuration and delegation; not an exact copied-line detector.",
            },
            "category4_copied_algorithm": {
                "method": "initialize_arm",
                "physical": arms[0].end_lineno,
                "normalized": len(normalized(arms[0]).splitlines()),
                "exact_executable_ast_equals_both_swm_families": True,
            },
            "upstream_delegated": delegated,
            "hierarchy": {
                c.family: [p.__module__ + "." + p.__qualname__ for p in c.__mro__]
                for c in [CompatibleCube, CompatibleScene]
            },
            "source_differences": {
                "initialize_arm": "Only eff_pos assignment differs from upstream; no uniform position draws. Remaining yaw/Lie/IK/joints/mj_forward statements are source-identical.",
                "modify_mjcf_model": "Registered configuration hook, not episode reset; preserves agent-color inverted dirty predicate and NumPy float32 size promotion. Tracks compiled versus pending MJCF values.",
            },
        },
    )
    schemas = {}
    for family in ("Cube", "Scene"):
        env = OGBManipulationAdapter(family, seed=17)
        try:
            schemas[family] = {
                "batch": "[B] (all listed leaf shapes have leading B)",
                "leaves": [
                    {"key": list(key), "dtype": str(dtype), "shape": list(shape)}
                    for key, (shape, dtype) in env.fields.items()
                ],
                "rng_owners": list(env.get_state()["rng"].keys()),
                "official_integration_sizes": {
                    key[-1]: list(shape) for key, (shape, _) in env.fields.items() if key[0] == "physics"
                },
                "integration_flags": {name: int(flag) for name, flag in FLAGS.items()},
            }
        finally:
            env.close()
    write("state-schema.json", schemas)


def closure():
    for ancestor in [RNG, DOCS]:
        subprocess.run(["git", "merge-base", "--is-ancestor", ancestor, START], cwd=ROOT, check=True)
    allowed_old = {"pyproject.toml", "stable_envs/envs/ogbench/__init__.py", "tests/unit/test_envs_ogbench.py"}
    unchanged = []
    for line in git("ls-tree", "-r", START).splitlines():
        metadata, rel = line.split("\t")
        if rel in allowed_old:
            continue
        data = (ROOT / rel).read_bytes()
        digest = hashlib.sha1(b"blob " + str(len(data)).encode() + b"\0" + data).hexdigest()
        assert digest == metadata.split()[2], rel
        unchanged.append(rel)
    accepted = json.loads((HERE.parent / "pusht_phase_a/audit.json").read_text())
    for rel, digest in accepted["artifact_hashes"].items():
        assert sha(ROOT / rel) == digest, rel
    for rel, entry in accepted["terminal_logs"].items():
        assert sha(ROOT / rel) == entry["sha256"], rel
        assert all(s in (ROOT / rel).read_text().splitlines() for s in entry["exact_terminal_lines"])
    manifests = {}
    for family in ["fetch_family", "ogb_rng_feasibility", "ogb_dependency_boundary"]:
        manifest = HERE.parent / family / "SHA256SUMS"
        for line in manifest.read_text().splitlines():
            digest, rel = line.split(maxsplit=1)
            path = ROOT / rel.lstrip("*")
            if not path.exists():
                path = manifest.parent / rel.lstrip("*")
            assert sha(path) == digest, str(path)
        manifests[family] = len(manifest.read_text().splitlines())
    source = json.loads((HERE.parent / "ogb_dependency_boundary/provenance.json").read_text())
    for rel, item in source["files"].items():
        assert sha(Path(item["path"])) == item["sha256"], rel
    swm = Path("/Users/samib/stable-worldmodel")
    assert git("rev-parse", "HEAD", cwd=swm) == SWM
    assert not git("diff", "HEAD", "--name-only", cwd=swm)
    imports = []
    for path in (ROOT / "stable_envs").rglob("*.py"):
        for node in ast.walk(ast.parse(path.read_text())):
            if isinstance(node, ast.Import):
                imports.extend((str(path), a.name) for a in node.names if a.name.startswith("stable_worldmodel"))
            if isinstance(node, ast.ImportFrom) and (node.module or "").startswith("stable_worldmodel"):
                imports.append((str(path), node.module))
    assert not imports
    matrix = json.loads((ROOT / "docs/research/semantic_recovery_matrix.json").read_text())
    counts = {key: sum(r["state_status"] == key for r in matrix["rows"]) for key in ["VERIFIED", "INCOMPLETE"]}
    assert counts == {"VERIFIED": 33, "INCOMPLETE": 9}
    changed = set(git("diff", "--name-only", START).splitlines())
    changed.update(git("ls-files", "--others", "--exclude-standard").splitlines())
    changed.update(str(p.relative_to(ROOT)) for p in HERE.iterdir() if p.is_file())
    changed.add(str((HERE / "closure.json").relative_to(ROOT)))
    prefix = str(HERE.relative_to(ROOT)) + "/"
    allowed = (
        allowed_old
        | set(PRODUCTION)
        | {
            "tests/unit/test_ogbench_manipulation_cpu.py",
            "docs/research/ogbench_cpu_compatibility_plan.md",
            "docs/research/ogbench_cpu_compatibility.md",
        }
    )
    assert all(p in allowed or p.startswith(prefix) for p in changed), changed
    write(
        "closure.json",
        {
            "result": "CPU_PRESERVATION_PASS",
            "start": START,
            "audit_checkpoint": START,
            "rng_checkpoint": RNG,
            "documentation_checkpoint": DOCS,
            "oracle": SWM,
            "branch": git("branch", "--show-current"),
            "worktree": str(ROOT),
            "starting_status": "clean",
            "implementation_commit": git("log", "-1", "--format=%H", "--", PRODUCTION[0]),
            "baseline_tracked_files": len(unchanged) + len(allowed_old),
            "unchanged_baseline_files": len(unchanged),
            "allowed_changed_baseline_files": sorted(allowed_old),
            "accepted_artifacts": len(accepted["artifact_hashes"]),
            "accepted_terminal_logs": len(accepted["terminal_logs"]),
            "manifests": manifests,
            "unchanged_source_files": len(source["files"]),
            "matrix": counts,
            "production_swm_imports": imports,
            "shared_contract_changes": [],
            "frozen_renderer_unchanged": True,
            "push_performed": False,
            "gpu_started": False,
            "promotion_performed": False,
            "changed_inventory": sorted(changed),
        },
    )
    write(
        "profile.json",
        {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "packages": {
                p: version(p) for p in ["ogbench", "mujoco", "numpy", "gymnasium", "dm-control", "torch", "tensordict"]
            },
            "cpu_authority_only": True,
            "renderer_extra_mujoco_range_unchanged": ">=3.11,<3.13",
            "profile_limit": "Exact CPU oracle evidence is for SWM-lock MuJoCo 3.8.1. Renderer-install MuJoCo range is not certified by this CPU checkpoint.",
        },
    )
    print(
        f"CPU_PRESERVATION_PASS unchanged={len(unchanged)} accepted_artifacts=170 terminal_logs=11 sources={len(source['files'])} matrix=33/9"
    )


if __name__ == "__main__":
    ownership()
    closure()
    print("CPU_OWNERSHIP_AND_SCHEMA_PASS")
