"""Device task formulas agree with authoritative dm_control observations and rewards."""

import numpy as np
import pytest
import torch

pytest.importorskip("dm_control")

from dm_control.utils import rewards

from stable_envs.envs.dmcontrol.simulation.factory import FACTORIES, make_local
from stable_envs.envs.dmcontrol.simulation.mjwarp_tasks import TASKS, _tolerance, bind_task


class SourceSnapshot:
    """The runtime's B-first field contract, populated from independent native lanes."""

    def __init__(self, envs, dtype=torch.float64, device="cpu"):
        self.envs, self.dtype, self.device = envs, dtype, device
        self.cache = {}

    def _field(self, owner, name):
        key = (owner, name)
        if key not in self.cache:
            values = [np.array(getattr(getattr(env.physics, owner), name)) for env in self.envs]
            self.cache[key] = torch.as_tensor(np.stack(values), dtype=self.dtype, device=self.device)
        return self.cache[key]

    def data(self, name):
        return self._field("data", name)

    def model(self, name):
        return self._field("model", name)


DEVICES = [
    "cpu",
    pytest.param("cuda", marks=pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")),
]


@pytest.mark.parametrize("key", sorted(FACTORIES))
@pytest.mark.parametrize("device", DEVICES)
def test_task_outputs_match_source_after_steps_and_resets(key, device):
    """Different lane targets, source sensors and reset poses exercise each registered task."""
    lanes = make_local(key, 3, "cpu", 7)
    try:
        task = bind_task(key, lanes.envs[0])
        rng = np.random.default_rng(5)
        for episode in range(2):
            lanes.reset(7 + 31 * episode)
            for _ in range(3):
                actions = rng.uniform(-1, 1, (lanes.num_envs, lanes.action_dim))
                outputs = [env.step(action) for env, action in zip(lanes.envs, actions, strict=True)]
                expected_observation = torch.from_numpy(
                    np.stack([np.concatenate([v.flatten() for v in out.observation.values()]) for out in outputs])
                ).to(device=device, dtype=torch.float32)
                expected_reward = torch.tensor([out.reward for out in outputs], dtype=torch.float64, device=device)
                snapshot = SourceSnapshot(lanes.envs, device=device)
                actual_observation = task.observation(snapshot)
                actual_reward = task.reward(snapshot)
                assert actual_observation.dtype == torch.float32
                assert actual_reward.dtype == torch.float64
                torch.testing.assert_close(actual_observation, expected_observation, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(actual_reward, expected_reward, atol=2e-12, rtol=2e-12)
                infos = [lanes._info(env, i) for i, env in enumerate(lanes.envs)]
                actual_info = task.info(snapshot)
                if infos[0] is None:
                    assert actual_info is None
                else:
                    success, fields = actual_info
                    expected_success = torch.tensor([v[0] for v in infos], dtype=torch.float64, device=device)
                    torch.testing.assert_close(success, expected_success, equal_nan=True)
                    assert set(fields) == set(infos[0][1])
                    for name, values in fields.items():
                        expected = torch.as_tensor(np.stack([v[1][name] for v in infos]), device=device)
                        torch.testing.assert_close(values, expected, atol=2e-12, rtol=2e-12, equal_nan=True)
                # Warp uses float32 fields. This compares formula precision independently of solver drift.
                fp32 = SourceSnapshot(lanes.envs, dtype=torch.float32, device=device)
                torch.testing.assert_close(task.observation(fp32), expected_observation, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(task.reward(fp32), expected_reward, atol=2e-6, rtol=2e-6)
    finally:
        lanes.close()


@pytest.mark.parametrize("sigmoid,value", [("gaussian", 0.1), ("long_tail", 0.1), ("linear", 0.5), ("quadratic", 0.0)])
def test_tolerance_matches_source_curves_with_per_world_bounds(sigmoid, value):
    x = np.random.default_rng(2).uniform(-3, 3, (19, 4))
    upper = np.linspace(0.015, 0.6, 19)[:, None]
    margin = np.where(np.arange(19)[:, None] % 3 == 0, 0.0, 5 * upper)
    expected = np.array(
        [
            [rewards.tolerance(v, (0, u), m, sigmoid, value) for v in row]
            for row, u, m in zip(x, upper[:, 0], margin[:, 0], strict=True)
        ]
    )
    actual = _tolerance(torch.from_numpy(x), (0, torch.from_numpy(upper)), torch.from_numpy(margin), sigmoid, value)
    torch.testing.assert_close(actual, torch.from_numpy(expected), atol=2e-15, rtol=2e-15)


def test_task_bindings_cover_registered_simulators_and_reject_unknown_keys():
    assert set(TASKS) == set(FACTORIES)
    with pytest.raises(ValueError, match="Unsupported DMC MuJoCo Warp task"):
        bind_task("unsupported", None)
