"""CPU prerequisite for OGBench online source comparisons, without rendering.

Exercises actual source step calls and the existing independent source restore
witness. No source State export belongs inside the future timing region. This
does not benchmark parallelism or establish image/GPU/performance acceptance.
"""

import argparse
import contextlib
import copy
import hashlib
import importlib
import importlib.metadata
import importlib.util
import inspect
import json
import platform
import sys
import types
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.benchmarks.renderer_table.common import PIN, equal_state
from stable_envs._adapters.ogbench import OGBMazeAdapter
from stable_envs._adapters.ogbench_manipulation import OGBManipulationAdapter
from tests.helpers.reference.ogbench_maze_semantics import OGBMazeReference
from tests.unit.test_ogbench_manipulation_cpu import assert_oracle


def import_source(root):
    for name, sub in (("stable_worldmodel", ""), ("stable_worldmodel.envs", "envs"),
                      ("stable_worldmodel.envs.ogbench", "envs/ogbench")):
        if name in sys.modules:
            raise RuntimeError(f"Use a fresh interpreter; {name} is already loaded")
        package = types.ModuleType(name)
        package.__path__ = [str(root / "stable_worldmodel" / sub)]
        sys.modules[name] = package
    importlib.import_module("stable_worldmodel.utils")
    sys.modules["stable_worldmodel"].spaces = importlib.import_module("stable_worldmodel.spaces")


def ownership_module():
    path = Path("docs/research/semantic_recovery_evidence/ogb_rng_feasibility/ownership.py")
    spec = importlib.util.spec_from_file_location("ogbench_online_source_ownership", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def source_tasks(rows, family):
    tasks = {"reward": torch.tensor([r[1] for r in rows], dtype=torch.float64),
             "terminated": torch.tensor([r[2] for r in rows], dtype=torch.bool),
             "truncated": torch.tensor([r[3] for r in rows], dtype=torch.bool),
             "success": torch.tensor([r[4]["success"] for r in rows],
                                     dtype=torch.float64 if family == "Maze" else torch.bool)}
    if family == "Maze":
        tasks["observation"] = torch.from_numpy(np.stack([r[0] for r in rows]))
        for name in ("xy", "prev_qpos", "prev_qvel", "qpos", "qvel"):
            tasks["info", name] = torch.from_numpy(np.stack([r[4][name] for r in rows]))
    return tasks


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest["commit"] == PIN
    for rel, digest in manifest["files"].items():
        assert hashlib.sha256((args.source_root / rel).read_bytes()).hexdigest() == digest, rel
    import_source(args.source_root)
    module = ownership_module()
    torch.set_num_threads(1)
    paths = [Path(__file__).relative_to(Path.cwd()),
             Path("scripts/benchmarks/renderer_table/common.py"),
             Path("tests/helpers/reference/ogbench_maze_semantics.py"),
             Path("tests/unit/test_ogbench_manipulation_cpu.py"),
             Path("docs/research/semantic_recovery_evidence/ogb_rng_feasibility/ownership.py"),
             Path("docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/probe.py"),
             *Path("stable_envs/_adapters").glob("ogbench*.py")]
    report = dict(status="RUNNING", host=platform.node(), source_pin=PIN,
        scope="CPU B2 direct-source/restore prerequisite only; no rendering, parallel transport, timing or CUDA",
        versions={n: importlib.metadata.version(n) for n in
                  ("ogbench", "mujoco", "dm-control", "torch", "numpy", "gymnasium", "tensordict")},
        source_files=manifest["files"], runtime_files={},
        candidate_files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
        renderer_construction_guard=True, source_close_probes={}, cases=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        with patch("mujoco.Renderer", side_effect=AssertionError("CPU prerequisite must not construct a renderer")):
            for family in ("Maze", "Cube", "Scene"):
                for horizon in (1, 8):
                    for dtype in (np.float32, np.float64):
                        with contextlib.ExitStack() as stack:
                            if family == "Maze":
                                source = OGBMazeReference(num_envs=2, seed=17)
                                stack.callback(source.close)
                                source.reset(17)
                                raw = source.envs
                                capture = source.get_state
                                restore = source.set_state

                                def source_step(actions):
                                    return [source._in_legacy_rng(i, env.step, action)
                                            for i, (env, action) in enumerate(zip(raw, actions, strict=True))]

                                adapter = OGBMazeAdapter(2, seed=17)
                            else:
                                raw = []
                                for lane in range(2):
                                    env = getattr(module, "Owned" + family)(entropy_seed=17 + lane)
                                    stack.callback(env.close)
                                    env.reset(seed=17 + lane)
                                    raw.append(env)
                                capture = lambda: [module.capture(env) for env in raw]

                                def restore(state):
                                    for env, row in zip(raw, state, strict=True):
                                        module.restore(env, row)

                                def source_step(actions):
                                    return [env.step(action) for env, action in zip(raw, actions, strict=True)]

                                adapter = OGBManipulationAdapter(family, 2, seed=17)
                            stack.callback(adapter.close)
                            inspected = raw + ([env.env for env in raw] if family == "Maze" else [])
                            for env in inspected:
                                for cls in type(env).__mro__:
                                    for method in ("__init__", "step", "reset", "render", "close"):
                                        fn = cls.__dict__.get(method)
                                        if inspect.isfunction(fn):
                                            f = Path(inspect.getsourcefile(fn))
                                            report["runtime_files"][str(f)] = hashlib.sha256(f.read_bytes()).hexdigest()

                            def verify():
                                if family == "Maze":
                                    equal_state(adapter.get_state(), source.get_state())
                                else:
                                    for a, b in zip(adapter.envs, raw, strict=True):
                                        assert_oracle(a, b, module)

                            verify()
                            source_initial, initial = capture(), adapter.get_state()
                            actions = np.random.default_rng(31).uniform(-.5, .5,
                                (2, horizon, 2 if family == "Maze" else 5)).astype(dtype)
                            first_tasks = []
                            for replay in range(2):
                                if replay:
                                    restore(source_initial)
                                    adapter.set_state(initial)
                                    equal_state(adapter.get_state(), initial)
                                    if family == "Maze":
                                        equal_state(capture(), source_initial)
                                    else:
                                        assert not module.previous.differences(source_initial, capture())
                                    verify()
                                for t in range(horizon):
                                    outside = copy.deepcopy(np.random.get_state())
                                    tasks = source_tasks(source_step(actions[:, t]), family)
                                    assert not bool(tasks["terminated"].any() or tasks["truncated"].any()), \
                                        "This bounded source workload must remain nonterminal"
                                    assert not module.previous.differences(outside, np.random.get_state())
                                    result = adapter.step(actions[:, t], capture_state=False)
                                    assert set(result.keys(True, True)) == set(tasks)
                                    for key, value in tasks.items():
                                        torch.testing.assert_close(result[key], value, rtol=0, atol=0)
                                    verify()
                                    if replay:
                                        for key in tasks:
                                            torch.testing.assert_close(tasks[key], first_tasks[t][key], rtol=0, atol=0)
                                    else:
                                        first_tasks.append({k: v.clone() for k, v in tasks.items()})
                                if not replay:
                                    endpoint = adapter.get_state()
                                else:
                                    equal_state(endpoint, adapter.get_state())
                            report["cases"].append(dict(family=family, batch=2, horizon=horizon,
                                action_dtype=np.dtype(dtype).name, source_task_parity="PASS",
                                independent_engine_rng_witness="PASS", restore_replay="PASS",
                                source_global_rng_preserved="PASS", valid_steps=2 * horizon))
                            if family not in report["source_close_probes"]:
                                # Structural cleanup probe only: no actual GL resource or
                                # memory measurement. The sentinel cannot render.
                                target = raw[0].env if family == "Maze" else raw[0]
                                field = "custom_renderer" if family == "Maze" else "_renderer"
                                assert getattr(target, field) is None
                                sentinel = types.SimpleNamespace(close_calls=0)

                                def close_sentinel():
                                    sentinel.close_calls += 1

                                sentinel.close = close_sentinel
                                setattr(target, field, sentinel)
                                try:
                                    raw[0].close()
                                    fn = type(target).close
                                    report["source_close_probes"][family] = dict(
                                        render_field=field, close_calls=sentinel.close_calls,
                                        reference_cleared=getattr(target, field) is None,
                                        delegated_close=f"{fn.__module__}.{fn.__qualname__}",
                                        scope="sentinel only; no actual renderer or memory allocated")
                                finally:
                                    setattr(target, field, None)
                            save()
                            print(family, horizon, np.dtype(dtype).name, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
