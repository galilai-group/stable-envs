import jax
import pytest
import torch

import stable_envs as se
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
