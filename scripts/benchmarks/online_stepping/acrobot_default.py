"""Gate ordinary Acrobot construction against the accepted local implementation."""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
from pathlib import Path
from unittest.mock import patch

import numpy as np
import torch

import stable_envs as se
from stable_envs._adapters import classic_execution
from stable_envs._adapters.acrobot_process import AcrobotProcessAdapter
from stable_envs.envs._classic_semantics import ClassicControlAdapter
from tests.unit.test_classic_state_export import equal


def gate(device):
    batch = 64
    with patch.object(classic_execution, "process_count", return_value=0):
        local = se.make("Acrobot", num_envs=batch, seed=17, device=device)
    try:
        with se.make("Acrobot", num_envs=batch, seed=17, device=device) as remote:
            assert isinstance(local._simulator, ClassicControlAdapter)
            assert isinstance(remote._simulator, AcrobotProcessAdapter)
            assert len(remote._simulator.worker_info) == 8
            assert remote._step_pipeline() is remote._simulator
            assert local._step_pipeline() is None
            assert local.size == remote.size == 500
            initial = local.get_state()
            equal(initial, remote.get_state())
            initial["lifecycle", "elapsed_steps"][:4, 0] = torch.tensor([0, 499, 500, 4], device=device)
            initial["theta1"][3] = np.pi
            initial["rng", "has_uint32"][-1] = 1
            initial["rng", "uinteger"][-1] = 2**32 - 1
            cases, overlaps = 0, 0
            original = remote.render

            def render(state):
                nonlocal overlaps
                if remote._simulator._pending is not None:
                    assert all(conn.poll(30) for conn in remote._simulator._connections)
                    overlaps += 1
                return original(state)

            with patch.object(remote, "render", render):
                for dtype in (np.int32, np.int64):
                    actions = np.random.default_rng(31).integers(0, 3, (batch, 8, 1)).astype(dtype)
                    for horizon in (1, 8):
                        a = actions[:, 0] if horizon == 1 else actions
                        for selection in (None, slice(None), slice(-1, None), slice(None, None, -2), slice(10, 12)):
                            for env in (local, remote):
                                env.set_state(initial)
                            expected = local.step(a, states=selection, frames=selection)
                            actual = remote.step(a, states=selection, frames=selection)
                            equal(expected, actual)
                            if horizon == 8:
                                assert actual["valid"][1:4].sum(1).tolist() == [1, 1, 1]
                            held = actual.clone()
                            remote.reset(42)
                            equal(held, actual)
                            assert remote._simulator._pending is None
                            cases += 1
            assert overlaps > 0
            for states, frames in ((None, slice(None)), (slice(None), None)):
                for env in (local, remote):
                    env.set_state(initial)
                equal(local.step(actions, states=states, frames=frames), remote.step(actions, states=states, frames=frames))
                cases += 1
            torch.testing.assert_close(local.render(), remote.render(), rtol=0, atol=0)
            for env in (local, remote):
                env.set_state(initial)
            equal(local.reset(), remote.reset())
            for env in (local, remote):
                env.set_variation({"physics.link_length_1": [1.15], "physics.link_mass_1": [1.3]})
                env.reset(31)
            device_actions = torch.as_tensor(actions, device=device)
            equal(local.step(device_actions), remote.step(device_actions))
            equal(local.step(device_actions[:, 0, 0]), remote.step(device_actions[:, 0, 0]))
            before = remote.get_state()
            invalid = before.clone()
            invalid["theta1"][0] += .25
            invalid["rng", "state"][-1].fill_(-1)
            try:
                remote.set_state(invalid)
            except (ValueError, OverflowError):
                pass
            else:
                raise AssertionError("Invalid last-shard State accepted")
            equal(before, remote.get_state())
            assert not remote._simulator._closed
            return dict(device=device, cases=cases + 3, overlaps=overlaps, batch=batch, processes=8,
                        native_resolution=[500, 500], unpatched_constructor=True,
                        complete_state_task_rng_lifecycle_ownership="PASS", frame_mse_vs_local=0,
                        variation_and_device_actions="PASS", failed_restore_keeps_owners="PASS")
    finally:
        local.close()


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) == 8 and torch.cuda.is_available()
    paths = [Path(n) for n in ("stable_envs/_adapters/classic_execution.py", "stable_envs/_adapters/acrobot_process.py",
                              "stable_envs/_adapters/process_requests.py", "stable_envs/envs/_classic_semantics.py",
                              "stable_envs/envs/classic_control.py", "stable_envs/_stepping.py")]
    paths.append(Path(__file__).relative_to(Path.cwd()))
    report = dict(status="RUNNING", env="swm/AcrobotControl-v1", host=platform.node(),
                  job=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name(), affinity=cpus,
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "gymnasium", "tensordict")},
                  source_sha256={str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}, gates=[])
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    try:
        for device in ("cpu", "cuda"):
            report["gates"].append(gate(device))
            save()
            print(device, "PASS", flush=True)
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
