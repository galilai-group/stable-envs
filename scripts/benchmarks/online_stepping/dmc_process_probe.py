"""Gate private DMC process owners and measure their actual public step calls.

No factory defaults are changed. Results compare accepted local adapters with
process owners on one host; these are not source-renderer speedup numbers.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch
from dmc_profile import DOMAINS, sync
from dmc_state_export import NAMES, equal

import stable_envs as se
from stable_envs._adapters.dmc import execution as dmcontrol_execution
from stable_envs._adapters.dmc.process import DMCProcessAdapter


def make(domain, batch, device, processes, cpus):
    name = NAMES[domain]
    if not processes:
        with patch.object(dmcontrol_execution, "process_count", return_value=0):
            return se.make(name, num_envs=batch, device=device, seed=17, size=64, backend="opengl")
    if domain in ("acrobot", "pendulum", "cartpole"):
        module, attribute, has_domain = "_first_family", "FirstFamilyAdapter", True
    elif domain in ("cheetah", "hopper", "walker"):
        module, attribute, has_domain = "_continuation", "DMCContinuationAdapter", True
    elif domain in ("ball_in_cup", "finger", "manipulator", "quadruped"):
        module, attribute, has_domain = "_aligned", "DMCAlignedAdapter", True
    elif domain == "reacher":
        module, attribute, has_domain = "reacher", "ReacherAdapter", False
    else:
        module, attribute, has_domain = "_humanoid", "DMCHumanoidAdapter", False

    def factory(*args, **kwargs):
        return DMCProcessAdapter(domain, *(args[1:] if has_domain else args),
                                 processes=processes, cpus=cpus, **kwargs)

    with (patch.object(dmcontrol_execution, "process_count", return_value=0),
          patch.object(importlib.import_module("stable_envs.envs.dmcontrol." + module), attribute, factory)):
        return se.make(name, num_envs=batch, device=device, seed=17, size=64, backend="opengl")


def gate(domain, device, cpus, frames=True, *, production=False):
    batch = 64 if production else 5
    with make(domain, batch, device, 0, cpus) as local:
        if production:
            # No patch surrounds this public construction: the frozen candidate
            # constructors must select their actual default route themselves.
            remote = se.make(NAMES[domain], num_envs=batch, device=device, seed=17, size=64, backend="opengl")
        else:
            remote = make(domain, batch, device, 2, cpus)
        with remote:
            return _gate_pair(domain, device, cpus, local, remote, frames, production)


def _gate_pair(domain, device, cpus, local, remote, frames, production):
    if production:
        assert isinstance(remote._simulator, DMCProcessAdapter)
        assert len(remote._simulator.worker_info) == min(8, len(cpus))
    batch = local.num_envs
    initial = local.get_state()
    equal(initial, remote.get_state())
    initial["lifecycle", "reset_next_step"][3] = True
    initial["rng", "has_gauss"][3] = True
    initial["rng", "cached_gaussian"][3] = 0.125
    initial["lifecycle", "step_count"][4] = int(local._simulator.envs[0]._env._step_limit) - 1
    actions = np.random.default_rng(21).uniform(-0.5, 0.5, (batch, 8, local.action_dim))
    cases, overlaps = 0, 0
    original_render = remote.render

    def checked_render(state):
        nonlocal overlaps
        if remote._simulator._pending is not None:
            # Workers may finish/overwrite transport while the parent holds
            # the preceding owned State. No reply is consumed here.
            assert all(conn.poll(30) for conn in remote._simulator._connections)
            overlaps += 1
        return original_render(state)

    with patch.object(remote, "render", checked_render):
        for horizon in (1, 8):
            for selection in (None, slice(None), slice(-1, None), slice(None, None, -2), slice(9, 12)):
                for env in (local, remote):
                    env.set_state(initial)
                action = torch.as_tensor(actions[:, 0] if horizon == 1 else actions, dtype=torch.float32, device=device)
                expected = local.step(action, states=selection, frames=selection if frames else None)
                actual = remote.step(action, states=selection, frames=selection if frames else None)
                equal(expected, actual)
                retained = actual.clone()
                remote.step(actions[:, 0], frames=None)
                remote.reset(seed=41)
                equal(retained, actual)
                assert remote._simulator._pending is None
                cases += 1
    if frames:
        assert overlaps > 0
    for env in (local, remote):
        env.set_state(initial)
    equal(local.reset(), remote.reset())
    physics = {
        "reacher": {"agent.arm_density": [1100.0], "agent.finger_density": [900.0]},
        "finger": {"agent_physics.proximal_density": [1100.0]},
        "manipulator": {"agent_physics.upper_arm_density": [1100.0]},
    }.get(domain, {"gravity.x": [0.25], "gravity.z": [-8.0]})
    for variation in (physics, {"light.diffuse_scale": [1.1]}):
        for env in (local, remote):
            env.set_variation(variation)
        equal(local.get_state(), remote.get_state())
        torch.testing.assert_close(local.get_style(), remote.get_style(), rtol=0, atol=0)
        equal(local.reset(seed=37), remote.reset(seed=37))
        expected = local.step(actions, frames=slice(None) if frames else None)
        actual = remote.step(actions, frames=slice(None) if frames else None)
        equal(expected, actual)
        cases += 1
    return dict(domain=domain, device=device, cases=cases, overlaps=overlaps,
                batch=batch, production_default=production, processes=len(remote._simulator.worker_info),
                state_task_rng_lifecycle_ownership="PASS", physics_and_visual_variations="PASS",
                frame_mse_vs_local=0 if frames else None)


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=DOMAINS, required=True)
    p.add_argument("--devices", nargs="+", choices=["cpu", "cuda"], default=["cpu"])
    p.add_argument("--batches", nargs="+", type=int, default=[64, 256])
    p.add_argument("--processes", nargs="+", type=int, default=[4, 8])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--gates-only", action="store_true")
    p.add_argument("--production", action="store_true", help="Require unpatched B64 default routing; gates only")
    p.add_argument("--no-frames", action="store_true")
    p.add_argument("--verify-gl", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    if a.production and (not a.gates_only or not a.verify_gl or a.no_frames
                         or a.domain not in dmcontrol_execution.PROCESS_DOMAINS):
        p.error("Production gate requires --gates-only --verify-gl, frames and an eligible domain; Quadruped remains excluded")
    if a.output.exists():
        raise FileExistsError(a.output)
    if a.verify_gl and (len(a.devices) != 1 or a.no_frames):
        p.error("GL proof requires one device and enabled frame gates")
    torch.set_num_threads(1)
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    report = dict(status="RUNNING", domain=a.domain, host=platform.node(), affinity=cpus,
                  job=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name() if "cuda" in a.devices else None,
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "dm-control", "mujoco", "tensordict")},
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [
                      *Path("stable_envs/_adapters").glob("dmcontrol*.py"), Path("stable_envs/_adapters/process_requests.py"),
                      Path("stable_envs/_stepping.py"), *Path("stable_envs/envs/dmcontrol").glob("*.py"), Path(__file__).relative_to(Path.cwd())]},
                  scope="public step + full States, frames disabled; local column exporter vs pure-process candidate",
                  production_default_gate=a.production, gate_batch=64 if a.production else 5,
                  gates=[], timings=[])
    a.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        a.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        if a.verify_gl:
            import mujoco
            from OpenGL import GL

            model = mujoco.MjModel.from_xml_string('<mujoco><worldbody><geom type="sphere" size=".1"/></worldbody></mujoco>')
            with mujoco.Renderer(model, height=32, width=32) as probe:
                probe._gl_context.make_current()
                vendor, renderer = GL.glGetString(GL.GL_VENDOR).decode(), GL.glGetString(GL.GL_RENDERER).decode()
                software = any(x in renderer.lower() for x in ("llvmpipe", "softpipe", "swrast"))
                assert software if a.devices == ["cpu"] else "nvidia" in vendor.lower() and not software
                report["opengl"] = dict(vendor=vendor, renderer=renderer, software=software)
        for device in a.devices:
            report["gates"].append(gate(a.domain, device, cpus, not a.no_frames, production=a.production))
            save()
            if a.gates_only:
                continue
            for batch in a.batches:
                modes = [0, *a.processes]
                samples = {(mode, horizon): [] for mode in modes for horizon in (1, 8)}
                # Sequential construction bounds memory. Reverse order in the
                # second block; do not retain B worlds for every strategy.
                for block in range(2):
                    for mode in modes if not block else reversed(modes):
                        with make(a.domain, batch, device, mode, cpus) as env:
                            initial = env.get_state()
                            actions = np.random.default_rng(31).uniform(-0.5, 0.5, (batch, 8, env.action_dim))
                            for horizon in (1, 8):
                                action = actions[:, 0] if horizon == 1 else actions
                                env.step(action, frames=None)
                                for _ in range(a.repeats):
                                    env.set_state(initial)
                                    sync(device)
                                    start = time.perf_counter()
                                    result = env.step(action, frames=None)
                                    sync(device)
                                    samples[mode, horizon].append(time.perf_counter() - start)
                                    del result
                for (mode, horizon), seconds in samples.items():
                    rates = [batch * horizon / t for t in seconds]
                    report["timings"].append(dict(batch=batch, device=device, horizon=horizon, processes=mode,
                                                  seconds=seconds, steps_per_s_mean=statistics.mean(rates),
                                                  steps_per_s_sd=statistics.stdev(rates)))
                save()
                print(a.domain, device, batch, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as exc:
        report["status"], report["error"] = "FAIL", repr(exc)
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
