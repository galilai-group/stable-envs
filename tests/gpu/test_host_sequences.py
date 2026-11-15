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
