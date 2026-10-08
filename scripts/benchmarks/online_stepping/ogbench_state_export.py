"""Exact OGBench public-API/export comparison against frozen f0716916 rows.

CPU physics is used for both output devices. Timings exclude rendering; frame
gates compare export paths, not fresh upstream image-fidelity certification.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
import types
from pathlib import Path

import numpy as np
import torch
from ogbench_profile import sync

import stable_envs as se
from tests.helpers.reference.ogbench_state_export import manipulation_state, maze_state

NAMES = {"Maze": "PointMazeLargeNavigate", "Cube": "OGBCube", "Scene": "OGBScene"}

def equal(a, b):
    assert a.batch_size == b.batch_size
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    for key in a.keys(True, True):
        x, y = a[key], b[key]
        assert x.shape == y.shape and x.dtype == y.dtype and x.device == y.device, key
        torch.testing.assert_close(x, y, rtol=0, atol=0, equal_nan=True, msg=str(key))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=NAMES, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 16])
    p.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--no-frames", action="store_true", help="CPU-only local smoke; GPU acceptance must include frames")
    p.add_argument("--verify-gl", action="store_true", help="Require software GL for CPU and NVIDIA GL for CUDA")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 2 or min(args.batches) < 1:
        p.error("Require positive batches and at least two repeats")
    if args.verify_gl and (len(args.devices) != 1 or args.no_frames):
        p.error("--verify-gl requires one device and enabled frame gates")
    torch.set_num_threads(1)
    report = dict(status="RUNNING", domain=args.domain, env=NAMES[args.domain],
                  host=platform.node(), job=os.environ.get("SLURM_JOB_ID"),
                  gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
                  versions={name: importlib.metadata.version(name) for name in ("torch", "numpy", "gymnasium", "ogbench", "mujoco", "tensordict")},
                  reference_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [Path("tests/helpers/reference/ogbench_state_export.py")]},
                  candidate_hashes={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [*Path("stable_envs/_adapters").rglob("*.py"), Path("stable_envs/_adapters/state_columns.py"), Path("stable_envs/_stepping.py"), *Path("stable_envs/envs/ogbench").glob("*.py")]},
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  scope="frozen accepted exporter vs column exporter; public step frames=None; same CPU physics for both output devices",
                  gates=[], timings=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def make(batch, device, legacy=False):
        env = se.make(NAMES[args.domain], num_envs=batch, device=device, seed=17, size=64, backend="auto" if args.domain == "Maze" else "opengl")
        if legacy:
            adapter = env._simulator
            adapter.get_state = types.MethodType(maze_state if args.domain == "Maze" else manipulation_state, adapter)
        return env

    save()
    try:
        if args.verify_gl:
            import mujoco
            from OpenGL import GL

            model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
            with mujoco.Renderer(model, height=32, width=32) as probe:
                probe._gl_context.make_current()
                vendor = GL.glGetString(GL.GL_VENDOR).decode()
                renderer = GL.glGetString(GL.GL_RENDERER).decode()
                software = any(x in renderer.lower() for x in ("llvmpipe", "softpipe", "swrast"))
                assert software if args.devices == ["cpu"] else "nvidia" in vendor.lower() and not software
                report["opengl"] = dict(vendor=vendor, renderer=renderer, software=software)
                save()
        for device in args.devices:
            with make(4, device, True) as reference, make(4, device) as candidate:
                initial = reference.get_state()
                equal(initial, candidate.get_state())
                actions = np.random.default_rng(12).uniform(-0.5, 0.5, (4, 8, candidate.action_dim))
                if args.domain == "Maze":
                    initial["rng", "global_has_gauss"].fill_(True)
                    initial["rng", "global_cached_gaussian"].fill_(0.125)
                else:
                    initial["rng", "task", "has_gauss"].fill_(1)
                    initial["rng", "task", "cached_gaussian"].fill_(0.125)
                cases = 0
                for horizon in (1, 8):
                    for selection in (None, slice(None), slice(-1, None), slice(None, None, -2), slice(9, 12)):
                        for env in (reference, candidate):
                            env.set_state(initial)
                        a = actions[:, 0] if horizon == 1 else actions
                        frames = None if args.no_frames else selection
                        expected = reference.step(a, states=selection, frames=frames)
                        actual = candidate.step(a, states=selection, frames=frames)
                        equal(expected, actual)
                        held = actual.clone()
                        candidate.step(actions[:, 0], frames=None)
                        candidate.reset(seed=31)
                        equal(held, actual)
                        candidate.set_state(expected["state"])
                        equal(expected["state"], candidate.get_state())
                        cases += 1
                for env in (reference, candidate):
                    env.set_state(initial)
                equal(reference.reset(), candidate.reset())
                if args.domain != "Maze":
                    for env in (reference, candidate):
                        env.reset(seed=42, options={"task_id": 4, "variation": ["cube.size", "camera.angle_delta"]})
                    equal(reference.get_state(), candidate.get_state())
                    equal(reference.step(actions, frames=None if args.no_frames else slice(None)),
                          candidate.step(actions, frames=None if args.no_frames else slice(None)))
                report["gates"].append(dict(device=device, cases=cases, state_task_rng_lifecycle_ownership="PASS",
                                             frame_mse_vs_legacy=None if args.no_frames else 0.0))
                save()
            for batch in args.batches:
                with make(batch, device, True) as reference, make(batch, device) as candidate:
                    initial = reference.get_state()
                    actions = np.random.default_rng(23).uniform(-0.5, 0.5, (batch, 8, candidate.action_dim))
                    for horizon in (1, 8):
                        a = actions[:, 0] if horizon == 1 else actions
                        for operation in ("get_state", "public_step_states_all"):
                            samples = {"legacy": [], "column": []}
                            for block in range(2):
                                order = [("legacy", reference), ("column", candidate)]
                                if block:
                                    order.reverse()
                                for mode, env in order:
                                    env.set_state(initial)
                                    fn = env.get_state if operation == "get_state" else lambda: env.step(a, frames=None)
                                    fn()
                                    for _ in range(args.repeats):
                                        env.set_state(initial)
                                        sync(device)
                                        start = time.perf_counter()
                                        result = fn()
                                        sync(device)
                                        samples[mode].append(time.perf_counter() - start)
                                        del result
                            for mode, seconds in samples.items():
                                count = batch * (horizon if operation != "get_state" else 1)
                                rates = [count / t for t in seconds]
                                report["timings"].append(dict(batch=batch, device=device, horizon=horizon,
                                    operation=operation, mode=mode, seconds=seconds,
                                    rate_mean=statistics.mean(rates), rate_sd=statistics.stdev(rates)))
                            save()
                print(args.domain, device, batch, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"] = "FAIL"
        report["error"] = repr(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
