"""Gate private Cube/Scene process owners and measure paired public State throughput."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from functools import partial
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import stable_envs as se
import stable_envs._adapters.ogbench.manipulation as manipulation
from stable_envs._adapters.ogbench import execution as ogbench_execution
from stable_envs._adapters.ogbench.manipulation_process import OGBManipulationProcessAdapter


def equal(a, b):
    assert a.batch_size == b.batch_size
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    for key in a.keys(True, True):
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, equal_nan=True)


def sync(device):
    if device == "cuda":
        torch.cuda.synchronize()


def make(family, batch, device, processes, cpus, *, backend="opengl"):
    options = dict(num_envs=batch, device=device, seed=17, size=64, backend=backend)
    if not processes:
        with patch.object(ogbench_execution, "process_count", return_value=0):
            return se.make("OGB" + family, **options)
    with (patch.object(ogbench_execution, "process_count", return_value=0),
          patch.object(manipulation, "OGBManipulationAdapter",
                       partial(OGBManipulationProcessAdapter, processes=processes, cpus=cpus))):
        return se.make("OGB" + family, **options)


def gate(family, device, cpus, *, production=False):
    batch = 64 if production else 2
    backend = "auto" if production else "opengl"
    with make(family, batch, device, 0, cpus, backend=backend) as local:
        remote = (se.make("OGB" + family, num_envs=batch, device=device, seed=17, size=64)
                  if production else make(family, batch, device, 2, cpus))
        with remote:
            return _gate_pair(device, local, remote, cpus, production)


def _gate_pair(device, local, remote, cpus, production):
    batch = local.num_envs
    if production:
        assert isinstance(remote._simulator, OGBManipulationProcessAdapter)
        assert len(remote._simulator.worker_info) == min(8, len(cpus))
        assert local.backend == remote.backend == "auto"
    initial = local.get_state()
    equal(initial, remote.get_state())
    initial["rng", "task", "has_gauss"][-1] = 1
    initial["rng", "task", "cached_gaussian"][-1] = 0.25
    initial["rng", "environment", "has_uint32"][-1] = 1
    initial["rng", "environment", "uinteger"][-1] = 2**32 - 1
    cases, overlaps = 0, 0
    original = remote.render

    def render(state):
        nonlocal overlaps
        if remote._simulator._pending is not None:
            assert all(conn.poll(30) for conn in remote._simulator._connections)
            overlaps += 1
        return original(state)

    with patch.object(remote, "render", render):
        for dtype in (np.float32, np.float64):
            actions = np.random.default_rng(31).uniform(-0.2, 0.2, (batch, 8, 5)).astype(dtype)
            for horizon in (1, 8):
                for selection in (None, slice(None), slice(-1, None), slice(None, None, -2), slice(9, 12)):
                    for env in (local, remote):
                        env.set_state(initial)
                    action = actions[:, 0] if horizon == 1 else actions
                    expected = local.step(action, states=selection, frames=selection)
                    actual = remote.step(action, states=selection, frames=selection)
                    equal(actual, expected)
                    held = actual.clone()
                    remote.reset(41)
                    equal(actual, held)
                    assert remote._simulator._pending is None
                    cases += 1
    assert overlaps > 0
    for env in (local, remote):
        env.set_state(initial)
    equal(local.reset(), remote.reset())
    values = {"cube.size": [0.025], "agent.color": [0.2, 0.7, 0.4]}
    torch.testing.assert_close(local.set_variation(values), remote.set_variation(values), rtol=0, atol=0)
    equal(local.reset(31, {"variation": []}), remote.reset(31, {"variation": []}))
    torch.testing.assert_close(local.get_style(), remote.get_style(), rtol=0, atol=0)
    device_actions = torch.as_tensor(actions, device=device, dtype=torch.float64)
    equal(local.step(device_actions), remote.step(device_actions))
    invalid = initial.clone()
    invalid["physics", "time"][0] += 0.25
    invalid["task", "id"][-1] = 999
    before = remote.get_state()
    try:
        remote.set_state(invalid)
    except ValueError:
        pass
    else:
        raise AssertionError("Invalid last-shard checkpoint accepted")
    equal(before, remote.get_state())
    assert not remote._simulator._closed
    return dict(device=device, cases=cases + 2, overlaps=overlaps,
                batch=batch, production_default=production, processes=len(remote._simulator.worker_info),
                backend=remote.backend,
                state_task_rng_lifecycle_ownership="PASS", failed_restore_keeps_owners="PASS",
                frame_mse_vs_local=0)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=["Cube", "Scene"], required=True)
    p.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu", "cuda"])
    p.add_argument("--batches", nargs="+", type=int, default=[16, 64])
    p.add_argument("--processes", nargs="+", type=int, default=[4, 8])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--verify-gl", action="store_true")
    p.add_argument("--gates-only", action="store_true")
    p.add_argument("--production", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.verify_gl and len(args.devices) != 1:
        p.error("Use one GL/device mode per interpreter")
    if args.production and not args.gates_only:
        p.error("Production routing checks require --gates-only")
    torch.set_num_threads(1)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    files = list(Path("stable_envs/_adapters").rglob("*.py"))
    files += [Path("stable_envs/_stepping.py"), Path("stable_envs/envs/ogbench/manipulation_semantics.py"),
              Path(__file__).relative_to(Path.cwd())]
    files += [Path("stable_envs/envs/ogbench") / (name + ".py") for name in
              ("manipulation_renderer", "manipulation", "manipulation_opengl", "manipulation_fidelity")]
    report = dict(status="RUNNING", domain=args.domain, host=platform.node(), job=os.environ.get("SLURM_JOB_ID"),
                  affinity=cpus, gpu=torch.cuda.get_device_name() if "cuda" in args.devices else None,
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "mujoco", "ogbench", "tensordict")},
                  source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                  scope="paired public steps with full State history and no frames; local vs persistent process owners",
                  gates=[], timings=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        if args.verify_gl:
            import mujoco
            from OpenGL import GL

            model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
            with mujoco.Renderer(model, height=32, width=32) as probe:
                probe._gl_context.make_current()
                vendor, renderer = GL.glGetString(GL.GL_VENDOR).decode(), GL.glGetString(GL.GL_RENDERER).decode()
                software = any(x in renderer.lower() for x in ("llvmpipe", "softpipe", "swrast"))
                assert software if args.devices == ["cpu"] else "nvidia" in vendor.lower() and not software
                report["opengl"] = dict(vendor=vendor, renderer=renderer, software=software)
        for device in args.devices:
            report["gates"].append(gate(args.domain, device, cpus, production=args.production))
            save()
            if args.gates_only:
                continue
            for batch in args.batches:
                modes = [0, *args.processes]
                samples = {(mode, horizon): [] for mode in modes for horizon in (1, 8)}
                counts = {key: [] for key in samples}
                for block in range(2):
                    for mode in modes if block == 0 else reversed(modes):
                        with make(args.domain, batch, device, mode, cpus) as env:
                            initial = env.get_state()
                            actions = np.random.default_rng(31).uniform(-0.2, 0.2, (batch, 8, 5))
                            for horizon in (1, 8):
                                action = actions[:, 0] if horizon == 1 else actions
                                env.step(action, frames=None)
                                for _ in range(args.repeats):
                                    env.set_state(initial)
                                    sync(device)
                                    start = time.perf_counter()
                                    result = env.step(action, frames=None)
                                    sync(device)
                                    samples[mode, horizon].append(time.perf_counter() - start)
                                    counts[mode, horizon].append(int(result["valid"].sum().item()))
                                    del result
                for (mode, horizon), seconds in samples.items():
                    assert counts[mode, horizon] == counts[0, horizon]
                    rates = [n / t for n, t in zip(counts[mode, horizon], seconds, strict=True)]
                    report["timings"].append(dict(device=device, batch=batch, horizon=horizon, processes=mode,
                                                  seconds=seconds, valid_transitions=counts[mode, horizon],
                                                  steps_per_s_mean=statistics.mean(rates), steps_per_s_sd=statistics.stdev(rates)))
                save()
                print(args.domain, device, batch, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"], report["error"] = "FAIL", repr(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
