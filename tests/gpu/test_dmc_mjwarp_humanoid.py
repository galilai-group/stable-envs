"""Humanoid checkpoints must reproduce contact-rich public CUDA rollouts."""

import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state

pytestmark = [pytest.mark.family("dmc"), pytest.mark.skipif(not torch.cuda.is_available(), reason="requires CUDA")]


@pytest.mark.parametrize("seed", [91000, 91010, 91020])
def test_humanoid_replays_100_public_calls(seed):
    with se.make("DMCHumanoidWalk", num_envs=3, device="cuda", simulation_backend="mjwarp", size=32) as env:
        assert env.simulation_backend == "mjwarp"
        env.reset(seed=seed)
        # Preserve the original 100-call evaluator's RNG, five-call prefix and
        # three lanes. These are public calls, each with two control rounds.
        actions = np.random.default_rng(seed + 91717).uniform(-1, 1, (105, 3, env.action_dim))
        for action in actions[:5]:
            env.step(torch.as_tensor(action, device="cuda"), states=None, frames=None)
        checkpoint = env.get_state()
        expected = []
        for action in actions[5:]:
            result = env.step(torch.as_tensor(action, device="cuda"), states=None, frames=None).cpu()
            for key in (("state", "qpos"), ("state", "qvel"), "observation", "reward"):
                assert torch.isfinite(result[key]).all(), key
            expected.append(result)

        env.reset(seed=seed + 200000)
        env.set_state(checkpoint)
        assert_same_state(env.get_state(), checkpoint)
        for call, (action, want) in enumerate(zip(actions[5:], expected, strict=True), start=1):
            actual = env.step(torch.as_tensor(action, device="cuda"), states=None, frames=None).cpu()
            try:
                assert_same_state(actual, want)
            except AssertionError as error:
                raise AssertionError(f"Humanoid seed {seed}, replay public call {call}: {error}") from error


@pytest.mark.parametrize("reset_together", [False, True], ids=["scratch-width2", "scratch-width1"])
def test_humanoid_mixed_resets_preserve_continuing_lane_across_streams(reset_together):
    device = torch.device("cuda", torch.cuda.current_device())
    first, second = torch.cuda.Stream(device=device), torch.cuda.Stream(device=device)
    with se.make("DMCHumanoidWalk", num_envs=3, device=device, simulation_backend="mjwarp", size=32) as env:
        with torch.cuda.stream(first):
            env.reset(seed=91000)
            prefix = np.random.default_rng(91000 + 91717).uniform(-1, 1, (5, 3, env.action_dim))
            for action in prefix:
                env.step(torch.as_tensor(action, device=device), states=None, frames=None)
            initial = env.get_state()
            limit = int(env._simulator._limits[0])
            # Staggered resets advance two scratch lanes in each round. When
            # both reset in round one, only lane 2 advances in that scratch.
            initial["lifecycle", "step_count"][:, 0] = torch.tensor(
                [limit if reset_together else limit - 1, limit, 6], device=device
            )
            initial["lifecycle", "reset_next_step"][:, 0] = torch.tensor([reset_together, True, False], device=device)
            initial["task", "cumulative_reward"][:, 0] = torch.tensor([11.0, 17.0, 23.0], device=device)
            action = torch.linspace(-0.5, 0.5, 6 * env.action_dim, dtype=torch.float64, device=device).reshape(3, -1)[
                :, ::2
            ]
            assert not action.is_contiguous()
            env.set_state(initial)

        # Adapter operations must order their own work across caller streams.
        # There is no explicit caller synchronization until both runs finish.
        with torch.cuda.stream(second):
            expected = env.step(action, states=None, frames=None)
            following = env.step(action, states=None, frames=None)
            env.set_state(initial)
        with torch.cuda.stream(first):
            restored = env.get_state()
            actual = env.step(action, states=None, frames=None)
        with torch.cuda.stream(second):
            replay_following = env.step(action, states=None, frames=None)
        first.synchronize()
        second.synchronize()
        assert_same_state(restored, initial)
        assert_same_state(actual, expected)
        assert_same_state(replay_following, following)
        counts = [1 if reset_together else 0, 1, 8]
        assert actual["state", "lifecycle", "step_count"].flatten().tolist() == counts
        assert following["state", "lifecycle", "step_count"].flatten().tolist() == [count + 2 for count in counts]
        assert_same_state(actual["state", "rng"][2], initial["rng"][2])

        # Keep the same batch width but let all lanes advance persistently.
        # The third lane must not depend on the other lanes' reset decisions.
        ordinary = initial.clone()
        ordinary["lifecycle", "step_count"][:2] = 0
        ordinary["lifecycle", "reset_next_step"][:2] = False
        env.set_state(ordinary)
        persistent = env.step(action, states=None, frames=None)
        persistent_following = env.step(action, states=None, frames=None)
        assert_same_state(actual[2], persistent[2])
        assert_same_state(following[2], persistent_following[2])
