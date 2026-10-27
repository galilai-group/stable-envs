import numpy as np
import pytest
import torch

from stable_envs.envs.pusht.simulation.jax_batch import PushTJaxAdapter
from stable_envs.envs.pusht.simulation.replay_batch import PushTReplayAdapter
from tests.helpers.reference.semantics import assert_same_state


@pytest.fixture(params=["cpu", "cuda"])
def device(request):
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    return request.param


@pytest.mark.parametrize("discrete", [False, True])
def test_replay_ownership_and_atomic_restore(device, discrete):
    env = PushTJaxAdapter(2, device, 8, discrete=discrete, config=dict(resolution=32))
    try:
        a = torch.tensor([[3, 4], [8, 9]] if discrete else [[0.2, -0.3], [-0.4, 0.5]], device=device)
        env.step(a)
        env.step(a.to(torch.int32 if discrete else torch.float64))
        checkpoint = env.get_state()
        expected = env.step(a)
        env.set_state(checkpoint)
        assert_same_state(expected, env.step(a))
        retained = expected.clone()
        env.step(a)
        assert_same_state(retained, expected)
        before = env.get_state()
        bad = before.clone()
        bad["current", "body", "block", "position"][1, 0] += 1
        with pytest.raises(ValueError, match="replay disagrees"):
            env.set_state(bad)
        assert_same_state(before, env.get_state())
        with pytest.raises(ValueError):
            env.step(a * 1000)
        assert_same_state(before, env.get_state())
        expected_reset = env.reset(seed=17)
        env.set_state(before)
        assert_same_state(expected_reset, env.reset(seed=17))
    finally:
        env.close()


@pytest.mark.parametrize(
    "initial,action",
    [
        ([480, 100, 200, 250, 0, 0, 0], [1.0, 0.0]),
        ([100, 215, 200, 200, 0, 0, 0], [0.5, 0.0]),
    ],
)
def test_relative_controls_and_contact_track_main(device, initial, action):
    env = PushTJaxAdapter(1, device, 0, config=dict(resolution=32))
    ref = PushTReplayAdapter(1, "cpu", 0, config=dict(resolution=32))
    options = dict(
        variation=[], state=np.array(initial, np.float64), goal_state=np.array([100, 100, 300, 250, np.pi / 4, 0, 0])
    )
    try:
        env.reset(0, options)
        ref.reset(0, options)
        for _ in range(8):
            got = env.step(torch.tensor([action], device=device))
            want = ref.step(np.array([action], np.float32))
            np.testing.assert_allclose(
                got["state", "current", "observation"].cpu(),
                want["state", "current", "observation"],
                atol=5e-4,
                rtol=0,
            )
            assert got["reward"].item() == pytest.approx(want["reward"].item(), abs=5e-4)
            assert got["success"].item() == want["success"].item()
            assert not got["truncated"].any()
    finally:
        env.close()
        ref.close()


def test_shapes_variations_reward_mask_and_source_checkpoint(device, monkeypatch):
    env = PushTJaxAdapter(8, device, 0, config=dict(resolution=32))
    ref = PushTReplayAdapter(8, "cpu", 0, config=dict(resolution=32))
    options = [
        dict(
            variation=[],
            variation_values={"agent.shape": i, "block.shape": i % 7 + 1},
            state=np.array([100, 100, 350, 300, 0.2, 0, 0], np.float64),
            goal_state=np.array([100, 100, 300, 250, 0.7, 0, 0]),
        )
        for i in range(8)
    ]
    try:
        source = ref.reset(0, options)
        env.set_state(source.to(device))
        np.testing.assert_array_equal(
            env.get_state()["current", "observation"].cpu(), source["current", "observation"]
        )
        before = env.get_state()

        def forbidden(*args, **kwargs):
            raise AssertionError("CPU physics during a JAX transition")

        for owner in env.envs:
            monkeypatch.setattr(owner.env, "advance", forbidden)
            monkeypatch.setattr(owner.env.space, "step", forbidden)
        result = env.step(torch.full((8, 2), 0.2, device=device), active=torch.arange(8, device=device) != 0)
        state = result["state"]
        assert torch.isfinite(state["current", "observation"]).all()
        for key, value in before["current"].items(True, True):
            assert torch.equal(value[0], state["current"][key][0])
        expected = -(state["current", "observation"] - state["current", "goal", "value"]).norm(dim=1)
        torch.testing.assert_close(result["reward"], expected, rtol=0, atol=0)
        assert not result["truncated"].any()
        env.set_state(state)
        assert_same_state(state, env.get_state())
    finally:
        env.close()
        ref.close()
