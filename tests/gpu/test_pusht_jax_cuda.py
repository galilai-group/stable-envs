import jax
import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.envs.pusht.simulation import execution
from stable_envs.envs.pusht.simulation.jax_batch import from_torch
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("batch", [1, 64])
def test_public_cuda_steps_use_dlpack_without_host_staging(batch, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Host staging during a warmed PushT transition")

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), se.PushT(num_envs=batch, size=32, device="cuda") as env:
        actions = torch.full((batch, 4), 0.25, device="cuda")[:, ::2]
        contiguous = actions.contiguous()
        assert from_torch(contiguous).unsafe_buffer_pointer() == contiguous.data_ptr()
        env.step(actions, frames=None)
        checkpoint = env.get_state()
        with monkeypatch.context() as patch, jax.transfer_guard("disallow"):
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "numpy", forbidden)
            result = env.step(actions, frames=None)
            held = result.clone()
            env.step(-actions, frames=None)
        stream.synchronize()
        assert_same_state(result, held)
        env.set_state(checkpoint)
        assert_same_state(result, env.step(actions, frames=None))
        assert result["state", "current", "observation"].is_cuda


@pytest.mark.parametrize("name", ["PushT", "PushTDiscrete"])
def test_public_cuda_env_with_reset_workers_matches_local(name, monkeypatch):
    envs = []
    for count in (0, 2):
        with monkeypatch.context() as patch:
            patch.setattr(execution, "process_count", lambda num_envs, cpus, count=count, **_: count)
            envs.append(se.make(name, num_envs=4, size=64, device="cuda", seed=17))
    local, remote = envs
    workers = remote._simulator._pool._processes
    try:
        assert local._simulator._pool is None
        rng = np.random.default_rng(3)
        if name == "PushTDiscrete":
            actions = torch.as_tensor(rng.integers(0, 16, (4, 6, 2)), device="cuda")
        else:
            actions = torch.as_tensor(rng.uniform(-1, 1, (4, 6, 2)).astype(np.float32), device="cuda")
        assert_same_state(remote.reset(seed=5), local.reset(seed=5))
        for env in envs:
            env.set_variation({"agent.color": np.arange(12, dtype=np.uint8).reshape(4, 3), "agent.scale": 25.0})
        torch.testing.assert_close(remote.get_variation(), local.get_variation(), rtol=0, atol=0)
        assert_same_state(remote.reset(seed=6), local.reset(seed=6))
        for states, frames in ((slice(None), slice(-1, None)), (None, None), (slice(None, None, 2), slice(None))):
            assert_same_state(
                remote.step(actions, states=states, frames=frames), local.step(actions, states=states, frames=frames)
            )
        torch.testing.assert_close(remote.render(), local.render(), rtol=0, atol=0)
        saved = local.get_state()
        lanes = {"variation": [], "variation_values": {"block.shape": np.asarray([1, 4, 2, 6])}}
        assert_same_state(remote.reset(seed=8, options=lanes), local.reset(seed=8, options=lanes))
        for env in envs:
            env.set_state(saved)
        assert_same_state(remote.reset(seed=9), local.reset(seed=9))
        assert_same_state(remote.step(actions[:, 0]), local.step(actions[:, 0]))
        torch.testing.assert_close(remote.render(), local.render(), rtol=0, atol=0)
    finally:
        for env in envs:
            env.close()
    assert not any(process.is_alive() for process in workers)
