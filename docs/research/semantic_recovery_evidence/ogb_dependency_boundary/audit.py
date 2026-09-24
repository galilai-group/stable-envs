"""Read-only dependency audit against installed OGBench; never a production adapter."""

import ast
import copy
import difflib
import hashlib
import importlib.metadata as metadata
import inspect
import json
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import stable_worldmodel.envs  # noqa: F401 -- register the pinned oracle
from ogbench.manipspace import mjcf_utils
from ogbench.manipspace.envs.cube_env import CubeEnv as UpstreamCube
from ogbench.manipspace.envs.scene_env import SceneEnv as UpstreamScene
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv as OracleCube
from stable_worldmodel.envs.ogbench.scene_env import SceneEnv as OracleScene

HERE = Path(__file__).resolve().parent


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write(name, value):
    (HERE / name).write_text(json.dumps(value, indent=2) + "\n")


class Normalize(ast.NodeTransformer):
    """Remove only docstrings; optionally canonicalize literal dict keyword syntax."""

    def __init__(self, dictionaries=False):
        self.dictionaries = dictionaries

    def visit_FunctionDef(self, node):
        if node.body and isinstance(node.body[0], ast.Expr) and isinstance(node.body[0].value, ast.Constant):
            if isinstance(node.body[0].value.value, str):
                node.body.pop(0)
        return self.generic_visit(node)

    def visit_Call(self, node):
        node = self.generic_visit(node)
        if self.dictionaries and isinstance(node.func, ast.Name) and node.func.id == "dict" and not node.args:
            if all(k.arg is not None for k in node.keywords):
                return ast.copy_location(
                    ast.Dict([ast.Constant(k.arg) for k in node.keywords], [k.value for k in node.keywords]), node
                )
        return node


def normalized(node, dictionaries=False):
    return ast.unparse(Normalize(dictionaries).visit(copy.deepcopy(node))) + "\n"


def methods(cls):
    tree = ast.parse(Path(inspect.getfile(cls)).read_text())
    definition = next(n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == cls.__name__)
    return {n.name: n for n in definition.body if isinstance(n, ast.FunctionDef)}


def qualified(cls):
    return cls.__module__ + "." + cls.__qualname__


def owner(cls, name):
    base = next(c for c in cls.__mro__ if name in c.__dict__)
    member = base.__dict__[name]
    if isinstance(member, property):
        member = member.fget
    try:
        path = Path(inspect.getfile(member)).resolve()
        return {
            "owner": qualified(base),
            "file": str(path),
            "line": inspect.getsourcelines(member)[1],
            "sha256": sha(path),
        }
    except (TypeError, OSError):
        return {"owner": qualified(base)}


def serial(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, dict):
        return {k: serial(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [serial(v) for v in value]
    return value


def main():
    result = {
        "versions": {n: metadata.version(n) for n in ["ogbench", "mujoco", "gymnasium", "numpy", "dm-control"]},
        "families": {},
    }
    for family, oracle, upstream, env_type in [
        ("cube", OracleCube, UpstreamCube, "single"),
        ("scene", OracleScene, UpstreamScene, "scene"),
    ]:
        source_path, upstream_path = Path(inspect.getfile(oracle)), Path(inspect.getfile(upstream))
        source_text, upstream_text = source_path.read_text(), upstream_path.read_text()
        (HERE / f"{family}.exact.diff").write_text(
            "".join(
                difflib.unified_diff(
                    upstream_text.splitlines(True),
                    source_text.splitlines(True),
                    fromfile=str(upstream_path),
                    tofile=str(source_path),
                    n=0,
                )
            )
        )
        om, um = methods(oracle), methods(upstream)
        comparison = []
        normalized_diff = []
        for name, node in om.items():
            old = um.get(name)
            if old is None:
                inherited = next((b for b in upstream.__mro__[1:] if name in b.__dict__), None)
                if inherited is not None:
                    member = inherited.__dict__[name]
                    if inspect.isfunction(member):
                        old = methods(inherited)[name]
            new_code = normalized(node)
            old_code = normalized(old) if old else ""
            status = (
                "identical_AST"
                if new_code == old_code
                else "dict_syntax_only"
                if old and normalized(node, True) == normalized(old, True)
                else "changed"
                if old
                else "SWM_only"
            )
            comparison.append(
                {
                    "method": name,
                    "status": status,
                    "swm_lines": [node.lineno, node.end_lineno],
                    "swm_normalized_lines": len(new_code.splitlines()),
                    "upstream_normalized_lines": len(old_code.splitlines()),
                    "upstream_owner": owner(upstream, name) if old else None,
                }
            )
            normalized_diff.extend(
                difflib.unified_diff(
                    old_code.splitlines(True),
                    new_code.splitlines(True),
                    fromfile=f"upstream:{name}",
                    tofile=f"SWM:{name}",
                    n=0,
                )
            )
        (HERE / f"{family}.normalized.diff").write_text("".join(normalized_diff))
        spec = gym.spec("swm/OGBCube-v0" if family == "cube" else "swm/OGBScene-v0")
        # Default gym.make intentionally checks the real wrapper chain, including implicit reset.
        wrapped = gym.make(spec.id)
        chain = []
        current = wrapped
        while isinstance(current, gym.Wrapper):
            chain.append(qualified(type(current)))
            current = current.env
        chain.append(qualified(type(current)))
        direct, up = oracle(), upstream(env_type=env_type)
        names = sorted(
            {
                name
                for c in oracle.__mro__
                for name, member in c.__dict__.items()
                if not name.startswith("__") and (inspect.isfunction(member) or isinstance(member, property))
            }
        )
        graph = {name: owner(oracle, name) for name in names}
        compiled = {}
        for label, env in [("swm", direct), ("upstream", up)]:
            model_xml = env.modify_mjcf_model(env.build_mjcf_model())
            model = mujoco.MjModel.from_xml_string(mjcf_utils.to_string(model_xml), mjcf_utils.get_assets(model_xml))
            compiled[label] = {
                "cube_geoms": [
                    {"name": model.geom(i).name, "size": model.geom_size[i].tolist()}
                    for i in range(model.ngeom)
                    if model.geom_bodyid[i] == model.body("object_0").id
                ],
                "cube_mass": float(model.body("object_0").mass[0]),
                "cube_inertia": model.body("object_0").inertia.tolist(),
                "camera_names": [model.camera(i).name for i in range(model.ncam)],
            }
        assert compiled["swm"]["cube_geoms"] != compiled["upstream"]["cube_geoms"]
        # Task tables are checked for every concrete constructor variant, without resets.
        tables = []
        for typ in ["single", "double", "triple", "quadruple", "octuple"] if family == "cube" else ["scene"]:
            a, b = oracle(env_type=typ), upstream(env_type=typ)
            equal = serial(a.task_infos) == serial(b.task_infos)
            assert equal
            tables.append({"env_type": typ, "task_count": a.num_tasks, "exact_numeric_payload_equality": equal})
        desc = direct._desc_dir.resolve()
        assets = {
            str(p.relative_to(desc)): {"sha256": sha(p), "bytes": p.stat().st_size}
            for p in sorted(desc.rglob("*"))
            if p.is_file()
        }
        assert up._desc_dir.resolve() == desc
        result["families"][family] = {
            "registration": {
                "id": spec.id,
                "entry_point": spec.entry_point,
                "kwargs": spec.kwargs,
                "max_episode_steps": spec.max_episode_steps,
                "order_enforce": spec.order_enforce,
                "disable_env_checker": spec.disable_env_checker,
            },
            "oracle_mro": [qualified(c) for c in oracle.__mro__],
            "upstream_mro": [qualified(c) for c in upstream.__mro__],
            "oracle_is_upstream_concrete_subclass": issubclass(oracle, upstream),
            "wrapper_chain": chain,
            "checker_construction_compiled_model": wrapped.unwrapped._model is not None,
            "direct_construction_compiled_model": direct._model is not None,
            "owners": graph,
            "methods": comparison,
            "compiled_default_models": compiled,
            "raw_file_lines": {"swm": len(source_text.splitlines()), "upstream": len(upstream_text.splitlines())},
            "method_totals": {
                s: {
                    "count": sum(r["status"] == s for r in comparison),
                    "swm_normalized_lines": sum(r["swm_normalized_lines"] for r in comparison if r["status"] == s),
                }
                for s in ["identical_AST", "dict_syntax_only", "changed", "SWM_only"]
            },
            "controller": {
                "class": qualified(type(direct._ik)),
                "file": inspect.getfile(type(direct._ik)),
                "sha256": sha(Path(inspect.getfile(type(direct._ik)))),
            },
            "asset_root": str(desc),
            "assets": assets,
            "task_tables": tables,
            "upstream_has_variation_space": hasattr(up, "variation_space"),
            "swm_default_ee_position": direct.variation_space["agent"]["ee_start_position"].value.tolist(),
            "upstream_arm_sampling_bounds": up._arm_sampling_bounds.tolist(),
            "swm_cube_size": direct.variation_space["cube"]["size"].value.tolist(),
        }
        wrapped.close()
    # A degenerate sampling bound still consumes draws and changes yaw: configuration alone is insufficient.
    counterexamples = []
    for seed in [0, 17, 42]:
        a, b = np.random.default_rng(seed), np.random.default_rng(seed)
        position = np.array([0.425, 0, 0.275])
        degenerate = b.uniform(position, position)
        swm_yaw, degenerate_yaw = a.uniform(-np.pi, np.pi), b.uniform(-np.pi, np.pi)
        assert swm_yaw != degenerate_yaw
        counterexamples.append(
            {
                "seed": seed,
                "degenerate_position": degenerate.tolist(),
                "swm_yaw": swm_yaw,
                "upstream_degenerate_bounds_yaw": degenerate_yaw,
            }
        )
    result["degenerate_bound_counterexamples"] = counterexamples
    write("dependency_graph.json", result)
    prior = json.loads((HERE.parent / "ogb_manipulation_preflight/provenance.json").read_text())
    provenance = {}
    for name, info in prior["files"].items():
        path = Path(info["path"])
        assert sha(path) == info["sha256"], name
        provenance[name] = info
    for path in {
        Path(inspect.getfile(gym.envs.registration)),
        Path(inspect.getfile(gym.wrappers.PassiveEnvChecker)),
        Path(inspect.getfile(gym.wrappers.OrderEnforcing)),
        Path(inspect.getfile(gym.wrappers.TimeLimit)),
    }:
        provenance[str(path)] = {"path": str(path), "sha256": sha(path)}
    write("provenance.json", {"files": provenance, "prior_inventory_verified": len(prior["files"])})
    print("OGB_DEPENDENCY_GRAPH_COMPLETE families=2 task_tables=6 default_wrappers=2")
    print("OGB_CONFIGURATION_COUNTEREXAMPLE_PASS seeds=0,17,42")


if __name__ == "__main__":
    main()
