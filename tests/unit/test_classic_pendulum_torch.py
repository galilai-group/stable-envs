"""Batched Torch Pendulum steps against the Gymnasium adapter: same resets, State and outputs to rounding."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("pygame")
torch = pytest.importorskip("torch")

import stable_envs as se  # noqa: E402
from stable_envs.envs.classic_control.simulation.local import ClassicControlAdapter  # noqa: E402
from stable_envs.envs.classic_control.simulation.pendulum_torch import (  # noqa: E402
    MAX_EPISODE_STEPS,
    PendulumTorchAdapter,
    pendulum_backend,
)
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402

#: Exact fields: every one a step does not compute with sin, cos or remainder.
EXACT = ("torque", ("task", "has_last_u"), ("lifecycle", "elapsed_steps"))
PHYSICS = {
    "g": [10.0, 5.5, 14.25, 9.81],
    "m": [1.0, 0.6, 1.4, 1.1],
    "l": [1.0, 1.3, 0.55, 0.9],
    "max_torque": [2.0, 1.5, 3.7, 2.5],
    "max_speed": [8.0, 5.0, 11.0, 8.0],
    "dt": [0.05, 0.03, 0.08, 0.05],
}


def _pair(device="cpu", lanes=4, seed=7):
    reference = ClassicControlAdapter("pendulum", lanes, "cpu", seed)
    fast = PendulumTorchAdapter(lanes, device, seed)
    values = {name: np.asarray(rows[:lanes], np.float64) for name, rows in PHYSICS.items()}
    for item in (reference, fast):
        item.configure(values)
        item.reset(seed)
    return reference, fast


def _assert_agree(actual, expected):
    actual, expected = actual.cpu(), expected.cpu()
    ours, theirs = actual["state"], expected["state"]
    for key in EXACT:
        assert_same_state(ours[key], theirs[key])
    for key in ("theta", "theta_dot"):
        torch.testing.assert_close(ours[key], theirs[key], rtol=0, atol=1e-9)
    torch.testing.assert_close(actual["observation"], expected["observation"], rtol=0, atol=1e-6)
    torch.testing.assert_close(actual["reward"], expected["reward"], rtol=0, atol=1e-9)
    assert torch.equal(actual["terminated"], expected["terminated"])
    assert torch.equal(actual["truncated"], expected["truncated"])
    for key in theirs.keys(True, True):
        if key[0] == "rng":
            assert_same_state(ours[key], theirs[key])


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
def test_steps_match_gymnasium_through_truncation_and_clipping(dtype):
    reference, fast = _pair()
    try:
        assert_same_state(fast.get_state(), reference.get_state())
        actions = torch.as_tensor(
            np.random.default_rng(3).uniform(-4.5, 4.5, (MAX_EPISODE_STEPS + 30, 4, 1)), dtype=dtype
        )
        for action in actions:
            _assert_agree(fast.step(action), reference.step(action))
    finally:
        reference.close()
        fast.close()


def test_inactive_lanes_hold_and_report_zero_outputs():
    reference, fast = _pair()
    try:
        actions = torch.as_tensor(np.random.default_rng(5).uniform(-2, 2, (40, 4, 1)), dtype=torch.float32)
        for t, action in enumerate(actions):
            active = torch.tensor([t % 2 == 0, True, t % 3 == 0, False])
            _assert_agree(fast.step(action, active=active), reference.step(action, active=active))
    finally:
        reference.close()
        fast.close()


def test_state_restores_across_the_two_adapters():
    reference, fast = _pair()
    try:
        rng = np.random.default_rng(11)
        for action in torch.as_tensor(rng.uniform(-2, 2, (25, 4, 1)), dtype=torch.float32):
            fast.step(action)
        reference.set_state(fast.get_state())
        fast.set_state(reference.get_state())
        for action in torch.as_tensor(rng.uniform(-2, 2, (25, 4, 1)), dtype=torch.float32):
            _assert_agree(fast.step(action), reference.step(action))
        reference.reset(None)
        fast.reset(None)
        assert_same_state(fast.get_state(), reference.get_state())
    finally:
        reference.close()
        fast.close()


def test_public_backend_selection():
    assert pendulum_backend("cpu") == "gym" and pendulum_backend("cuda") == "torch"
    with pytest.raises(ValueError, match="simulation_backend"):
        pendulum_backend("cpu", "mjx")
    with se.Pendulum(num_envs=2, device="cpu") as env:
        assert env.simulation_backend == "gym" and env.execution_mode == "DELEGATED_CPU"
        assert type(env._simulator) is ClassicControlAdapter
    with se.Pendulum(num_envs=2, device="cpu", simulation_backend="torch") as env:
        assert env.simulation_backend == "torch" and env.execution_mode == "NATIVE_TORCH"
        assert isinstance(env._simulator, PendulumTorchAdapter)


def test_public_torch_env_matches_gymnasium_env():
    actions = torch.as_tensor(np.random.default_rng(9).uniform(-2, 2, (3, 30, 1)), dtype=torch.float32)
    with (
        se.Pendulum(num_envs=3, device="cpu") as gym_env,
        se.Pendulum(num_envs=3, device="cpu", simulation_backend="torch") as fast_env,
    ):
        gym_env.reset(seed=4)
        fast_env.reset(seed=4)
        a = gym_env.step(actions, frames=slice(None))
        b = fast_env.step(actions, frames=slice(None))
        torch.testing.assert_close(b["state", "theta"], a["state", "theta"], rtol=0, atol=1e-9)
        torch.testing.assert_close(b["reward"], a["reward"], rtol=0, atol=1e-9)
        # Frames draw [theta, torque]; theta differences far below a pixel leave them equal.
        assert torch.equal(b["frames"], a["frames"])


@pytest.mark.gpu
def test_cuda_steps_match_gymnasium():
    if not torch.cuda.is_available():
        pytest.skip("CUDA required")
    reference, fast = _pair("cuda")
    try:
        actions = torch.as_tensor(
            np.random.default_rng(13).uniform(-3, 3, (MAX_EPISODE_STEPS + 5, 4, 1)), dtype=torch.float32
        )
        for action in actions:
            _assert_agree(fast.step(action.cuda()), reference.step(action))
        with se.Pendulum(num_envs=2, device="cuda") as env:
            assert env.simulation_backend == "torch" and env.execution_mode == "NATIVE_TORCH"
    finally:
        reference.close()
        fast.close()
