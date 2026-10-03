"""CPU B2/P2 source transport prerequisite; no timing or rendering acceptance."""

import argparse
import contextlib
import copy
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

from scripts.benchmarks.online_stepping.ogbench_parallel_source import OGBSource, ParallelOGBSource
from scripts.benchmarks.renderer_table.common import PIN


def equal(a, b):
    if isinstance(a, dict):
        assert isinstance(b, dict) and set(a) == set(b)
        for key in a:
            equal(a[key], b[key])
    elif isinstance(a, (tuple, list)):
        assert type(a) is type(b) and len(a) == len(b)
        for x, y in zip(a, b, strict=True):
            equal(x, y)
    elif isinstance(a, np.ndarray):
        assert isinstance(b, np.ndarray) and a.dtype == b.dtype and a.shape == b.shape
        np.testing.assert_array_equal(a, b, strict=True)
    else:
        assert a == b


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-root", type=Path, required=True)
    p.add_argument("--source-manifest", type=Path, required=True)
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--empty-frames", action="store_true",
                   help="Exercise zero-length shared frame columns without initializing GL")
    p.add_argument("--render-size", type=int, default=None)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    manifest = json.loads(args.source_manifest.read_text())
    assert manifest["commit"] == PIN
    for rel, digest in manifest["files"].items():
        assert hashlib.sha256((args.source_root / rel).read_bytes()).hexdigest() == digest, rel
    frame_indices = () if args.empty_frames else None
    torch.set_num_threads(1)
    files = [Path(__file__).relative_to(Path.cwd()),
             Path("scripts/benchmarks/online_stepping/ogbench_parallel_source.py"),
             Path("scripts/benchmarks/online_stepping/ogbench_source_prerequisite.py"),
             Path("scripts/benchmarks/online_stepping/classic_parallel_source.py"),
             Path("tests/helpers/reference/ogbench_maze_semantics.py"),
             Path("docs/research/semantic_recovery_evidence/ogb_rng_feasibility/ownership.py"),
             Path("docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/probe.py")]
    report = dict(status="RUNNING", host=platform.node(), parent_pid=os.getpid(),
        scope="CPU B2/P2 source-to-source task transport, owned snapshots and nonterminal restore/replay only",
        source_pin=PIN, source_files=manifest["files"], runtime_files={},
        candidate_files={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        versions={n: importlib.metadata.version(n) for n in
                  ("ogbench", "mujoco", "dm-control", "torch", "numpy", "gymnasium", "tensordict")},
        no_rendering=True, no_timing=True, empty_frames=args.empty_frames,
        requested_render_size=args.render_size, state_rpc="untimed pickle, task columns use shared memory",
        cases=[], worker_closures=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        with patch("mujoco.Renderer", side_effect=AssertionError("No parent renderer in transport prerequisite")):
            for family in ("Maze", "Cube", "Scene"):
                for horizon in (1, 8):
                    with contextlib.closing(OGBSource(family, args.source_root, 2, render_size=args.render_size)) as serial, \
                            ParallelOGBSource(family, args.source_root, 2, 2, horizon,
                                              frame_indices=frame_indices, render_size=args.render_size) as workers:
                        assert workers.vector.shared_memory and workers.vector.copy
                        assert workers.vector.autoreset_mode.name == "DISABLED"
                        initial = serial.snapshot()
                        immutable = copy.deepcopy(initial)
                        workers.restore(initial)
                        equal(initial, workers.snapshot())
                        report["runtime_files"].update(workers.runtime_files)
                        for dtype in (np.float32, np.float64):
                            serial.restore(initial)
                            workers.restore(initial)
                            actions = np.random.default_rng(31).uniform(-.5, .5,
                                (2, horizon, workers.action_dim)).astype(dtype)
                            outside = copy.deepcopy(np.random.get_state())
                            expected = serial.step(actions, frame_indices)
                            actual = workers.step(actions)
                            equal(outside, np.random.get_state())
                            assert set(actual) == set(expected)
                            for key in actual:
                                torch.testing.assert_close(actual[key], torch.from_numpy(expected[key]), rtol=0, atol=0)
                            endpoint = workers.snapshot()
                            equal(endpoint, serial.snapshot())
                            held = {key: value.clone() for key, value in actual.items()}
                            immutable_endpoint = copy.deepcopy(endpoint)
                            changed = workers.step(-actions)
                            assert set(changed) == set(held)
                            moved = workers.snapshot()
                            old_qpos = endpoint[("root", "position")] if family == "Maze" else endpoint[0]["engine"]["QPOS"]
                            new_qpos = moved[("root", "position")] if family == "Maze" else moved[0]["engine"]["QPOS"]
                            assert not np.array_equal(old_qpos, new_qpos)
                            for key in actual:
                                torch.testing.assert_close(actual[key], held[key], rtol=0, atol=0)
                            equal(endpoint, immutable_endpoint)
                            workers.restore(initial)
                            replay = workers.step(actions)
                            for key in replay:
                                torch.testing.assert_close(replay[key], held[key], rtol=0, atol=0)
                            equal(endpoint, workers.snapshot())
                            equal(initial, immutable)
                            inspection = list(workers.vector.call("inspection"))
                            assert len({row["pid"] for row in inspection}) == 2
                            for lane, row in enumerate(inspection):
                                assert row["pid"] != os.getpid() and row["torch_threads"] == 1
                                assert row["start"] == lane and row["size"] == 1 and row["renderers_absent"]
                            report["cases"].append(dict(family=family, batch=2, workers=2, horizon=horizon,
                                action_dtype=np.dtype(dtype).name, render_size=serial.render_size, source_tasks="PASS", source_snapshot="PASS",
                                restore_replay="PASS", shared_output_ownership="PASS", state_ownership="PASS",
                                global_rng_preserved="PASS", worker_info=inspection))
                            save()
                            print(family, horizon, np.dtype(dtype).name, "PASS", flush=True)
                    closure = [dict(pid=process.pid, exitcode=process.exitcode, alive=process.is_alive())
                               for process in workers.vector.processes]
                    assert len(closure) == 2 and all(row["exitcode"] == 0 and not row["alive"] for row in closure)
                    report["worker_closures"].append(dict(family=family, horizon=horizon, workers=closure))
                    save()
        report["status"] = "PASS"
    except BaseException as error:
        report.update(status="FAIL", error=repr(error))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
