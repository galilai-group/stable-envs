"""Parity and buffer regressions specific to the TwoRoom JAX port."""

import jax
import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.family("navigation")


@pytest.mark.parametrize("batch", [1, 17])
def test_torch_and_jax_have_identical_public_steps_and_restore(batch):
    with (
        se.TwoRoom(num_envs=batch, device="cpu", simulation_backend="torch") as native,
        se.TwoRoom(
            num_envs=batch,
            device="cpu",
            simulation_backend="jax",
        ) as accelerated,
    ):
        assert native.execution_mode == "NATIVE_TORCH"
        assert accelerated.execution_mode == "NATIVE_JAX"
        variation = {
            "wall.axis": np.arange(batch).reshape(-1, 1) % 2,
            "agent.speed": np.linspace(1.25, 7.75, batch).reshape(-1, 1),
            "agent.radius": np.linspace(7, 12.25, batch).reshape(-1, 1),
        }
        for env in (native, accelerated):
            env.set_variation(variation)
            env.reset(seed=17, options={"state": [60, 60], "target_state": [180, 180]})
        checkpoint = native.get_state()
        actions = torch.from_numpy(np.random.default_rng(43).uniform(-2, 2, (batch, 8, 2)).astype(np.float32))
        expected = native.step(actions, states=slice(None), frames=None)
        assert_same_state(accelerated.step(actions, states=slice(None), frames=None), expected)
        for env in (native, accelerated):
            env.set_state(checkpoint)
            assert_same_state(env.step(actions, states=slice(None), frames=None), expected)


def test_invalid_simulation_backend():
    with pytest.raises(ValueError, match="simulation_backend"):
        se.TwoRoom(simulation_backend="unknown")


@pytest.mark.parametrize(
    "batch,threshold,backend,expected",
    [
        (1, 1, "auto", "numba"),
        (7, 8, "auto", "numba"),
        (8, 8, "auto", "numba"),
        (1, 8, "jax", "jax"),
        (8, 8, "torch", "torch"),
        (8, 1, "numba", "numba"),
    ],
)
def test_batch_selection_matches_explicit_backend_and_restores(batch, threshold, backend, expected):
    with (
        se.TwoRoom(
            num_envs=batch,
            device="cpu",
            simulation_backend=backend,
            jax_min_envs=threshold,
        ) as env,
        se.TwoRoom(num_envs=batch, device="cpu", simulation_backend=expected) as reference,
    ):
        assert env.simulation_backend == expected
        checkpoint = env.reset(seed=17)
        reference.reset(seed=17)
        action = torch.linspace(-1, 1, batch * 2).reshape(batch, 2)
        actual = env.step(action, frames=None)
        assert_same_state(actual, reference.step(action, frames=None))
        env.set_state(checkpoint)
        assert_same_state(env.step(action, frames=None), actual)


@pytest.mark.parametrize("threshold", [0, -1, True, 1.5, "8"])
def test_invalid_jax_cutoff(threshold):
    with pytest.raises(ValueError, match="jax_min_envs"):
        se.TwoRoom(jax_min_envs=threshold)


@pytest.mark.parametrize("backend", ["torch", "jax", "numba"])
def test_collisions_with_fractional_per_lane_geometry(backend):
    positions = [[95, 100], [130, 100], [100, 49], [50, 21], [202, 100], [49, 100], [100, 130]]
    actions = torch.tensor([[1, 0], [-1, 0], [1, 0], [0, -5], [5, 0], [0, 1], [0, -1]], dtype=torch.float64)
    variation = {
        "agent.radius": [[7.125], [7.5], [8.25], [7], [7], [7], [8.125]],
        "agent.speed": [[5.5], [6.25], [4.125], [5], [5], [5], [7.25]],
        "wall.axis": [[1], [1], [1], [1], [1], [0], [0]],
        "wall.thickness": [[11.75], [10.25], [9.75], [10], [10], [13.5], [15.75]],
        "door.number": [[0.9], [1], [2.75], [1], [1], [3], [2]],
        "door.position": [[49, 112, 180]] * 7,
        "door.size": [[14.25, 10.5, 7.75]] * 7,
    }
    with se.TwoRoom(num_envs=7, device="cpu", simulation_backend=backend) as env:
        env.set_variation(variation)
        env.reset(seed=0, options={"state": positions, "target_state": [180, 180]})
        for step in range(1, 7):
            result = env.step(actions, frames=None)
            expected = torch.tensor(
                [
                    [99.375, 100],
                    [125, 100],
                    [100 + 4.125 * step, 49],
                    [50, 21],
                    [203, 100],
                    [49, 100 + 5 * step],
                    [100, 130 - 7.25 * step],
                ]
            )
            assert_same_state(result["state", "agent", "position"], expected)


def test_sequence_masks_freeze_each_lane_after_success():
    options = {"state": [[60, 112], [60, 112], [60, 112]], "target_state": [[60, 112], [85, 112], [180, 112]]}
    actions = torch.tensor([[[1.0, 0.0]] * 6] * 3)
    with se.TwoRoom(num_envs=3, device="cpu") as env:
        env.reset(seed=0, options=options)
        result = env.step(actions, frames=None)
        assert_same_state(result["state", "agent", "position"], torch.tensor([[65, 112], [70, 112], [90, 112.0]]))
        assert result["valid"].sum(dim=1).tolist() == [1, 2, 6]


def test_dlpack_shares_contiguous_inputs_and_step_has_no_host_conversion(monkeypatch):
    from stable_envs.envs.navigation.tworoom_jax import from_torch

    value = torch.ones(4, 2)
    borrowed = from_torch(value)
    assert borrowed.unsafe_buffer_pointer() == value.data_ptr()
    assert torch.from_dlpack(borrowed).data_ptr() == value.data_ptr()
    with se.TwoRoom(num_envs=4, device="cpu", simulation_backend="jax") as env:
        env.reset(seed=0)
        action = torch.ones(4, 2, requires_grad=True)
        env.step(action, states=None, frames=None)  # Compile before guarding the runtime path.
        x64 = jax.config.x64_enabled

        def forbidden(*args, **kwargs):
            raise AssertionError("host conversion in JAX TwoRoom step")

        with monkeypatch.context() as patch, jax.transfer_guard("disallow"):
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "numpy", forbidden)
            patch.setattr(torch.Tensor, "item", forbidden)
            patch.setattr(torch.Tensor, "__bool__", forbidden)
            result = env.step(action, states=None, frames=None)
            held = result.clone()
            env.step(-action, states=None, frames=None)
        assert_same_state(result, held)
        assert jax.config.x64_enabled == x64


@pytest.mark.parametrize("name", ["TwoRoom", "Piecewise"])
@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_task_snapshots_and_output_leaves_have_independent_storage(name, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    with se.make(name, num_envs=3, device=device) as env:
        env.reset(seed=0)
        action = torch.zeros(3, 2, device=device)
        result = env.step(action, states=None, frames=None)
        held = result.clone()
        result["info", "proprio"].zero_()
        assert_same_state(result.exclude("info"), held.exclude("info"))
        assert_same_state(result["info", "state"], held["info", "state"])
        result["state", "target", "position"].zero_()
        assert_same_state(result["info", "goal_state"], held["info", "goal_state"])
        assert_same_state(env.get_state(), held["state"])
        modified = result.clone()
        env.step(action, states=None, frames=None)
        assert_same_state(result, modified)


def test_noncontiguous_actions_and_one_lane_offset_views():
    with se.TwoRoom(num_envs=1, device="cpu") as env:
        env.reset(seed=0, options={"state": [50, 50], "target_state": [180, 180]})
        action = torch.arange(8, dtype=torch.float32).reshape(4, 2)[1:2]
        assert action.storage_offset() > 0
        result = env.step(action, frames=None)
        assert_same_state(result["state", "agent", "position"], torch.tensor([[55.0, 55.0]]))
        actions = (torch.arange(20, dtype=torch.float32).reshape(1, 5, 4) / 32)[..., ::2]
        assert not actions.is_contiguous()
        result = env.step(actions, frames=None)
        assert_same_state(result["state", "agent", "position"], torch.tensor([[61.25, 62.8125]]))


@pytest.mark.parametrize("name", ["TwoRoom", "Piecewise"])
def test_shared_distance_matches_cpu_norm_without_host_conversion(name, monkeypatch):
    angles = np.linspace(0, 2 * np.pi, 128, endpoint=False)
    target = (np.stack((np.cos(angles), np.sin(angles)), -1) * 16 + 60).astype(np.float32)
    target = np.concatenate((target, np.random.default_rng(7).uniform(14, 209, (512, 2)).astype(np.float32)))
    with se.make(name, num_envs=len(target), device="cpu") as env:
        state = env.reset(seed=0, options={"state": [60, 60], "target_state": target})
        delta = state["agent", "position"] - state["target", "position"]
        expected = torch.tensor([float(torch.norm(row)) for row in delta], dtype=torch.float64)

        def forbidden(*args, **kwargs):
            raise AssertionError("host conversion in shared navigation outputs")

        with monkeypatch.context() as patch:
            patch.setattr(torch.Tensor, "cpu", forbidden)
            patch.setattr(torch.Tensor, "numpy", forbidden)
            patch.setattr(torch.Tensor, "item", forbidden)
            result = env._outputs(state)
        assert_same_state(result["info", "distance_to_target"], expected)
        assert_same_state(result["terminated"], expected < 16)


@pytest.mark.parametrize("backend", ["torch", "jax", "numba"])
def test_step_uses_the_observation_hook(monkeypatch, backend):
    with se.TwoRoom(device="cpu", simulation_backend=backend) as env:
        env.reset(seed=0)
        expected = torch.arange(10, dtype=torch.float32).reshape(1, 10)
        monkeypatch.setattr(env, "_observation", lambda agent, target: expected.clone())
        assert_same_state(env.step(torch.zeros(1, 2), frames=None)["observation"], expected)


@pytest.mark.parametrize("backend", ["torch", "jax", "numba"])
def test_continuous_action_multiply_rounds_before_position_addition(backend):
    options = {"state": [53.47578811645508, 60], "target_state": [180, 180]}
    action = torch.tensor([[-0.9868243932723999, 0.0]])
    with se.TwoRoom(device="cpu", simulation_backend=backend) as env:
        env.reset(seed=0, options=options)
        actual = env.step(action, frames=None)
        assert actual["state", "agent", "position"][0, 0].item() == 48.541664123535156
