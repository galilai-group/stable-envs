import hashlib

import jax
import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")


@pytest.mark.parametrize("batch", [1, 64])
def test_cuda_reacher_uses_mjx_without_host_staging(batch, monkeypatch):
    pytest.importorskip("mujoco.mjx")
    from stable_envs.envs.dmcontrol.simulation.reacher_mjx import _from_torch

    def forbidden(*args, **kwargs):
        raise AssertionError("Host staging in a steady-state CUDA Reacher step")

    sha256 = hashlib.sha256

    def hash_empty_metadata(value=b"", **kwargs):
        assert not value, "Static model hashing in a warmed CUDA Reacher step"
        return sha256(value, **kwargs)

    stream = torch.cuda.Stream()
    with (
        torch.cuda.stream(stream),
        se.make("DMCReacher", num_envs=batch, device="cuda", simulation_backend="mjx") as env,
    ):
        assert env.execution_mode == "JAX_MJX"
        env.reset(seed=7)
        action = torch.zeros(batch, 4, device="cuda")[:, ::2]
        env.step(action, frames=None)
        contiguous = action.contiguous()
        assert _from_torch(contiguous).unsafe_buffer_pointer() == contiguous.data_ptr()
        checkpoint = env.get_state()
        with monkeypatch.context() as patch, jax.transfer_guard("disallow"):
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "numpy", forbidden)
            patch.setattr("hashlib.sha256", hash_empty_metadata)
            actual = env.step(action, frames=None)
            held = actual.clone()
            action.fill_(0.25)
            env.step(action, frames=None)
        stream.synchronize()
        assert_same_state(actual, held)
        env.set_state(checkpoint)
        action.zero_()
        assert_same_state(env.step(action, frames=None), actual)
        frame = env.render()
        assert frame.device.type == "cuda"
        assert frame.dtype == torch.uint8


def test_cuda_reacher_tracks_native_physics_variation_and_resets():
    pytest.importorskip("mujoco.mjx")
    from stable_envs.envs.dmcontrol.simulation.first_family import ReacherAdapter

    reference = ReacherAdapter(3, "cpu", 7)
    try:
        with se.make("DMCReacher", num_envs=3, device="cuda", simulation_backend="mjx") as env:
            values = np.array([[500, 1500], [1500, 500], [850, 1150]])
            reference.configure_physics(values)
            reference.reset(7)
            state = reference.get_state()
            state["lifecycle", "step_count"][:, 0] = torch.tensor([997, 998, 999])
            reference.set_state(state)
            env.set_state(state.cuda())
            for action in np.random.default_rng(43).uniform(-1, 1, (32, 3, 2)):
                expected = reference.step(action)
                actual = env.step(torch.as_tensor(action, device="cuda"), frames=None).cpu()
                for key in ("qpos", "qvel"):
                    torch.testing.assert_close(actual["state", key], expected["state", key], rtol=0, atol=3e-6)
                torch.testing.assert_close(actual["observation"], expected["observation"], rtol=0, atol=3e-6)
                for key in ("reward", "score", "terminated", "truncated"):
                    assert torch.equal(actual[key], expected[key])
                for key in ("rng", "lifecycle", "time", ("task", "target_pos")):
                    assert_same_state(actual["state"][key], expected["state"][key])
    finally:
        reference.close()


@pytest.mark.parametrize("batch,backend", [(1, "native"), (31, "native"), (32, "mjx")])
def test_auto_reacher_selects_physics_and_preserves_cuda_outputs(batch, backend):
    with se.DMCReacher(num_envs=batch, device="cuda") as env:
        assert env.simulation_backend == backend
        assert env.execution_mode == ("JAX_MJX" if backend == "mjx" else "DELEGATED_CPU")
        checkpoint = env.reset(seed=7)
        action = torch.zeros(batch, 2, device="cuda")
        actual = env.step(action, frames=None)
        assert all(leaf.device.type == "cuda" for leaf in actual.values(True, True))
        env.set_state(checkpoint)
        assert_same_state(env.step(action, frames=None), actual)
