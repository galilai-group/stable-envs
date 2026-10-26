import jax
import pytest
import torch

import stable_envs as se
from stable_envs.envs.pusht.jax import _from_torch
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("batch", [1, 64])
def test_cuda_step_and_render_use_dlpack_without_host_staging(batch, monkeypatch):
    def forbidden(*args, **kwargs):
        raise AssertionError("Host staging in a warmed PushTJax transition")

    stream = torch.cuda.Stream()
    with torch.cuda.stream(stream), se.PushTJax(num_envs=batch, size=32, device="cuda") as env:
        actions = torch.full((batch, 4), 256.0, device="cuda")[:, ::2]
        contiguous = actions.contiguous()
        assert _from_torch(contiguous).unsafe_buffer_pointer() == contiguous.data_ptr()
        env.step(actions)
        checkpoint = env.get_state()
        with monkeypatch.context() as patch, jax.transfer_guard("disallow"):
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "numpy", forbidden)
            result = env.step(actions)
            held = result.clone()
            actions.fill_(100)
            env.step(actions)
        stream.synchronize()
        assert_same_state(result, held)
        env.set_state(checkpoint)
        actions.fill_(256)
        assert_same_state(result, env.step(actions))
        assert result["frames"].dtype == torch.uint8
        assert result["frames"].device.type == "cuda"
