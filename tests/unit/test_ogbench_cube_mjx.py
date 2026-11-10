from contextlib import closing

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")
jax = pytest.importorskip("jax")
pytest.importorskip("mujoco.mjx")
pytest.importorskip("mujoco_warp")

from stable_envs.envs.ogbench.simulation.cube_mjx import CubeMJXAdapter, _from_torch
from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
from tests.helpers.reference.semantics import assert_same_state


@pytest.fixture(scope="module", params=["cpu", "cuda"])
def env(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with closing(CubeMJXAdapter(num_envs=2, device=request.param, seed=17)) as owner:
        yield owner


def test_controller_and_physics_track_native_with_different_cube_sizes(env):
    options = [
        dict(variation=[], task_id=2, variation_values={"cube.size": np.array([size])}) for size in (0.013, 0.029)
    ]
    env.reset(17, options)
    control = jax.jit(jax.vmap(env._set_control))
    with closing(OGBManipulationAdapter("Cube", 2, seed=17)) as ref:
        ref.reset(17, options)
        for action in ([0.2, -0.3, 0.1, 0.4, -0.5], [0, 0, 0, 12, 0], [2, -3, 0.6, 4, -2]):
            actions = torch.tensor([action] * 2, dtype=torch.float32, device=env.device)
            predicted = control(env._data, _from_torch(actions))
            for lane, native in enumerate(ref.envs):
                native.set_control(np.asarray(action, np.float32))
                np.testing.assert_allclose(np.asarray(predicted.ctrl[lane]), native.data.ctrl, atol=2e-4, rtol=2e-4)
            got = env.step(actions)
            want = ref.step(actions.cpu())
            np.testing.assert_allclose(
                got["state", "physics", "qpos"].cpu(), want["state", "physics", "qpos"], atol=2e-3, rtol=2e-3
            )
            np.testing.assert_allclose(
                got["state", "physics", "qvel"].cpu(), want["state", "physics", "qvel"], atol=2e-2, rtol=2e-2
            )
            assert torch.equal(got["reward"].cpu(), want["reward"])
            assert torch.equal(got["terminated"].cpu(), want["terminated"])


def test_checkpoint_continuation_mask_and_owned_outputs(env, monkeypatch):
    env.reset(
        17,
        [dict(variation=[], task_id=2, variation_values={"cube.size": np.array([size])}) for size in (0.015, 0.025)],
    )
    actions = torch.full((2, 10), 0.1, device=env.device)[:, ::2]
    for _ in range(8):
        env.step(actions, capture_state=False)
    checkpoint = env.get_state()
    expected = env.step(actions)
    held = expected.clone()
    env.step(-actions)
    assert_same_state(expected, held)

    def forbidden(*args, **kwargs):
        raise AssertionError("Host staging or native physics during an MJX transition")

    with monkeypatch.context() as patch:
        patch.setattr("mujoco.mj_step", forbidden)
        env.set_state(checkpoint)
    assert_same_state(checkpoint, env.get_state())
    assert_same_state(expected, env.step(actions))
    before = env.get_state()
    result = env.step(actions, active=torch.tensor([True, False], device=env.device))
    assert_same_state(before[1], result["state"][1])

    contiguous = actions.contiguous()
    assert _from_torch(contiguous).unsafe_buffer_pointer() == contiguous.data_ptr()
    with monkeypatch.context() as patch, jax.transfer_guard("disallow"):
        patch.setattr(torch.Tensor, "cpu", forbidden)
        patch.setattr(torch.Tensor, "numpy", forbidden)
        patch.setattr("mujoco.mj_step", forbidden)
        env.step(actions)
    before = env.get_state()
    expected_reset = env.reset(None, dict(variation=[]))
    env.set_state(before)
    assert_same_state(expected_reset, env.reset(None, dict(variation=[])))


def test_native_checkpoint_import_and_atomic_validation(env):
    with closing(OGBManipulationAdapter("Cube", 2, seed=29)) as cpu:
        cpu.step(torch.full((2, 5), -0.2))
        source = cpu.get_state()
        env.set_state(source)
        torch.testing.assert_close(
            env.get_state()["physics", "qpos"].cpu(), source["physics", "qpos"], atol=1e-6, rtol=0
        )
        with pytest.raises(ValueError, match="different solver"):
            cpu.set_state(env.get_state().cpu())
    before = env.get_state()
    bad = before.clone()
    bad["physics", "qpos"][-1, 0] = float("nan")
    with pytest.raises(ValueError):
        env.set_state(bad)
    assert_same_state(before, env.get_state())
    with pytest.raises(ValueError):
        env.step(torch.full((2, 5), float("nan"), device=env.device))
    assert_same_state(before, env.get_state())
    env.configure([{"cube.size": np.array([0.025])}] * 2)
    torch.testing.assert_close(env.get_state()["physics", "qpos"], before["physics", "qpos"], atol=0, rtol=0)
    env.step(torch.zeros(2, 5, device=env.device))


def test_goal_rewards_and_mixed_reward_tasks(env):
    with (
        closing(OGBManipulationAdapter("Cube", 1, seed=17)) as regular,
        closing(OGBManipulationAdapter("Cube", 1, seed=19, reward_task_id=2)) as reward_task,
    ):
        for owner in (regular, reward_task):
            native = owner.envs[0]
            start = int(native.model.joint("object_joint_0").qposadr[0])
            native.data.qpos[start : start + 3] = native.data.mocap_pos[0]
            native._success = False
        source = torch.cat([regular.get_state(), reward_task.get_state()])
        env.set_state(source)
        result = env.step(torch.zeros(2, 5, device=env.device))
        expected = torch.cat([regular.step(torch.zeros(1, 5)), reward_task.step(torch.zeros(1, 5))])
        for key in ("reward", "success", "terminated", "truncated"):
            assert torch.equal(result[key].cpu(), expected[key])
        assert result["reward"].tolist() == [1.0, 0.0]
        assert result["success"].tolist() == [True, True]


def test_public_state_and_render_state_with_mjx_adapter(env, monkeypatch):
    import stable_envs as se
    from stable_envs.envs.ogbench.simulation import render_state

    monkeypatch.setattr(
        "stable_envs.envs.ogbench.simulation.execution.make_manipulation_adapter",
        lambda family, n, device, seed, reward_id, local: CubeMJXAdapter(family, n, device, seed, reward_id),
    )
    with se.OGBCube(num_envs=2, device=env.device, seed=17) as public:
        state = public.step(torch.zeros(2, 5, device=env.device), frames=None)["state"]
        assert public.state_spec.contains(state)
        projected = public.to_render_state(state)
        render_state.validate("Cube", projected)
        assert torch.equal(projected["pose", "qpos"], state["physics", "qpos"])
        assert public.execution_mode == "NATIVE_JAX"


def test_missing_cuda_fails_before_native_owners_are_constructed(monkeypatch):
    def unavailable(*args, **kwargs):
        raise RuntimeError("CUDA unavailable")

    def forbidden(*args, **kwargs):
        raise AssertionError("CPU owners constructed before checking JAX CUDA")

    monkeypatch.setattr(jax, "devices", unavailable)
    monkeypatch.setattr(OGBManipulationAdapter, "__init__", forbidden)
    with pytest.raises(RuntimeError, match="ogbench-manipulation-gpu"):
        CubeMJXAdapter(device="cuda:0")
