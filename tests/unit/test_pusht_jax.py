import jax
import jax.numpy as jnp
import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.envs.pusht.simulation.jax_geometry import coverage
from tests.helpers.reference.semantics import assert_same_state


@pytest.mark.parametrize(
    "position,angle,expected",
    [
        ((256, 256), np.pi / 4, 1),
        ((220, 270), -0.4, 0.38762307),
        ((300, 210), 1.2, 0.15608072),
        ((256, 196), np.pi / 4, 1 / 7),
        ((100, 100), 0, 0),
    ],
)
def test_coverage_matches_shapely_reference(position, angle, expected):
    actual = jax.jit(coverage)(
        jnp.asarray(position, dtype=jnp.float32),
        jnp.float32(angle),
        jnp.array([256, 256], dtype=jnp.float32),
        jnp.float32(np.pi / 4),
    )
    assert float(actual) == pytest.approx(expected, abs=2e-6)


# Reference trajectories: gym-pusht 0.1.6 with Pymunk 6.11.1.
@pytest.mark.parametrize(
    "pose,action,steps,expected,tolerance",
    [
        ((256, 420, 256, 240, 0), (256, 300), 1, (256, 384.51938569, 256, 240, 0), 1e-3),
        ((256, 420, 256, 240, 0), (256, 300), 20, (256, 300.00006224, 256, 165.00006627, 0), 0.25),
        ((100, 215, 200, 200, 0), (260, 215), 10, (259.837773, 215, 330.378174, 247.653892, 0.699824951), 0.30),
    ],
)
def test_physics_tracks_gym_pusht_reference(pose, action, steps, expected, tolerance, device):
    with se.PushTJax(device=device, size=32) as env:
        env.reset(seed=0, options={"state": pose})
        for _ in range(steps):
            result = env.step(torch.tensor([action]), states=None, frames=None)
        actual = result["observation"][0].cpu().numpy()
    np.testing.assert_allclose(actual[:4], expected[:4], atol=tolerance, rtol=0)
    assert float(actual[4]) == pytest.approx(expected[4], abs=8e-4)


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return request.param


def test_checkpoint_continuation_rng_ownership_and_atomic_rejection(device):
    with (
        se.PushTJax(num_envs=3, size=32, device=device) as env,
        se.PushTJax(num_envs=3, size=32, device=device, seed=99) as restored,
    ):
        state = env.reset(seed=7)
        actions = torch.tensor([[256, 300], [400, 200], [100, 350]], device=device)
        expected = env.step(actions)
        retained = expected.clone()
        restored.set_state(state)
        assert_same_state(expected, restored.step(actions))
        env.step(actions)
        assert_same_state(expected, retained)
        assert_same_state(env.reset(), restored.reset())
        before = restored.get_state()
        bad = before.clone()
        bad["agent_vel"][1, 0] = float("nan")
        with pytest.raises(ValueError):
            restored.set_state(bad)
        assert_same_state(before, restored.get_state())
        bad = before.clone()
        bad["rng"][0, 0] = 2**32
        with pytest.raises(ValueError):
            restored.set_state(bad)
        assert_same_state(before, restored.get_state())
        assert torch.equal(restored.render(), restored.render(before))


def test_goal_reward_clock_and_inactive_lanes(device):
    with se.PushTJax(num_envs=2, size=32, device=device) as env:
        state = env.reset(seed=0, options={"state": [0, 0, 256, 256, np.pi / 4]})
        state["time"][1] = 299
        env.set_state(state)
        result = env._step(torch.zeros(2, 2, device=device), active=torch.tensor([False, True], device=device))
        assert_same_state(state[0], result["state"][0])
        assert result["reward"].tolist() == [0, 1]
        assert result["terminated"].tolist() == [False, True]
        assert result["truncated"].tolist() == [False, True]
        env.reset(seed=0, options={"state": [0, 0, 256, 256, np.pi / 4]})
        result = env.step(torch.zeros(2, 4, 2, device=device), frames=slice(None))
        assert result["valid"].tolist() == [[True, False, False, False]] * 2
        assert result["state", "time"].tolist() == [[1], [1]]
