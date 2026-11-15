"""A step of several actions on CUDA that runs on a host simulator gives the CPU environment's results."""

import numpy as np
import pytest
import torch

import stable_envs as se

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize(
    "name",
    [
        pytest.param("TwoRoom", marks=pytest.mark.family("navigation")),
        pytest.param("DMCReacher", marks=pytest.mark.family("dmc")),
        pytest.param("OGBCube", marks=pytest.mark.family("ogbench")),
    ],
)
def test_a_cuda_sequence_on_the_host_matches_the_cpu_env(name):
    with se.make(name, num_envs=2, device="cuda") as gpu, se.make(name, num_envs=2, device="cpu") as cpu:
        assert gpu._steps_on_host()
        spec = cpu.action_spec
        low = np.broadcast_to(np.asarray(getattr(spec, "low", -1.0), np.float64), spec.shape)
        high = np.broadcast_to(np.asarray(getattr(spec, "high", 1.0), np.float64), spec.shape)
        actions = torch.as_tensor(np.random.default_rng(0).uniform(low, high, (2, 8, *spec.shape)), dtype=spec.dtype)
        for e in (gpu, cpu):
            e.reset(seed=3)
        got, want = gpu.step(actions.cuda(), frames=None), cpu.step(actions, frames=None)
        assert all(leaf.device.type == "cuda" for leaf in got.values(True, True))
        assert set(got.keys(True, True)) == set(want.keys(True, True))
        for key in want.keys(True, True):
            assert torch.equal(got[key].cpu(), want[key]), key
        # The next step continues from the State the sequence reached.
        got, want = gpu.step(actions[:, 0].cuda(), frames=None), cpu.step(actions[:, 0], frames=None)
        for key in want.keys(True, True):
            assert torch.equal(got[key].cpu(), want[key]), key


@pytest.mark.family("ogbench")
def test_a_device_simulator_steps_its_sequences_on_the_device():
    pytest.importorskip("mujoco_warp")
    with se.make("OGBCube", num_envs=2, device="cuda", simulation_backend="mjwarp") as env:
        assert not env._steps_on_host()
        env.reset(seed=0)
        result = env.step(torch.zeros(2, 3, 5, device="cuda"), frames=None)
        assert result["state"].device.type == "cuda"


@pytest.mark.family("ogbench")
def test_a_device_masked_sequence_equals_the_sequence_that_reads_its_masks_back(monkeypatch):
    """Lane 0 succeeds on its first action and holds; the device-masked loop gives the same bytes as the readback."""
    pytest.importorskip("mujoco_warp")
    actions = torch.as_tensor(np.random.default_rng(1).uniform(-1, 1, (4, 12, 5)), dtype=torch.float32).cuda()
    runs = []
    for device_masks in (True, False):
        with se.make("OGBCube", num_envs=4, device="cuda", simulation_backend="mjwarp") as env:
            assert env._masks_on_device()
            monkeypatch.setattr(type(env), "_masks_on_device", lambda self, value=device_masks: value)
            env.reset(seed=3)
            sim = env._simulator
            state = env.get_state().clone()
            qpos, mocap = state["physics", "qpos"], state["physics", "mocap_pos"]
            moved = mocap.reshape(4, -1, 3).clone()
            moved[0, sim._target] = qpos[0, sim._cube : sim._cube + 3]
            state["physics", "mocap_pos"] = moved.reshape(mocap.shape)
            env.set_state(state)
            runs.append(env.step(actions, frames=None))
        monkeypatch.undo()
    masked, readback = runs
    assert not bool(masked["valid"][0, 1:].any()) and bool(masked["valid"][1:, 0].all())
    assert set(masked.keys(True, True)) == set(readback.keys(True, True))
    for key in readback.keys(True, True):
        assert torch.equal(masked[key], readback[key]), key
