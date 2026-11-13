"""Batched task outputs match native DMC tasks, including sparse positive states."""

from contextlib import closing

import numpy as np
import pytest
import torch

pytest.importorskip("dm_control")

from stable_envs.envs.dmcontrol.simulation.factory import FACTORIES, make_local
from stable_envs.envs.dmcontrol.simulation.mjwarp_tasks import bind_task


class SourceSnapshot:
    """Native fields in the runtime's batched array layout."""

    def __init__(self, envs, dtype=torch.float64):
        self.envs, self.dtype = envs, dtype
        self.cache = {}

    def _field(self, owner, name):
        key = (owner, name)
        if key not in self.cache:
            values = [np.array(getattr(getattr(env.physics, owner), name)) for env in self.envs]
            self.cache[key] = torch.as_tensor(np.stack(values), dtype=self.dtype)
        return self.cache[key]

    def data(self, name):
        return self._field("data", name)

    def model(self, name):
        return self._field("model", name)


@pytest.mark.parametrize("key", sorted(FACTORIES))
def test_task_outputs_match_source_after_steps_and_resets(key):
    with closing(make_local(key, 3, "cpu", 7)) as lanes:
        task = bind_task(key, lanes.envs[0])
        rng = np.random.default_rng(5)
        for seed in (7, 38):
            lanes.reset(seed)
            for _ in range(3):
                actions = rng.uniform(-1, 1, (lanes.num_envs, lanes.action_dim))
                outputs = [env.step(action) for env, action in zip(lanes.envs, actions, strict=True)]
                observation = torch.from_numpy(
                    np.stack([np.concatenate([v.flatten() for v in out.observation.values()]) for out in outputs])
                ).float()
                reward = torch.tensor([out.reward for out in outputs], dtype=torch.float64)
                snapshot = SourceSnapshot(lanes.envs)
                torch.testing.assert_close(task.observation(snapshot), observation, atol=1e-6, rtol=1e-6)
                torch.testing.assert_close(task.reward(snapshot), reward, atol=2e-12, rtol=2e-12)
                infos = [lanes._info(env, i) for i, env in enumerate(lanes.envs)]
                actual_info = task.info(snapshot)
                if infos[0] is None:
                    assert actual_info is None
                else:
                    success, fields = actual_info
                    expected = torch.tensor([v[0] for v in infos], dtype=torch.float64)
                    torch.testing.assert_close(success, expected, equal_nan=True)
                    assert set(fields) == set(infos[0][1])
                    for name, values in fields.items():
                        expected = torch.as_tensor(np.stack([v[1][name] for v in infos]))
                        torch.testing.assert_close(values, expected, atol=2e-12, rtol=2e-12, equal_nan=True)
                # Test the float32 field boundary separately from integration drift.
                fp32 = SourceSnapshot(lanes.envs, dtype=torch.float32)
                torch.testing.assert_close(task.observation(fp32), observation, atol=2e-5, rtol=2e-5)
                torch.testing.assert_close(task.reward(fp32), reward, atol=2e-6, rtol=2e-6)


def test_reacher_sparse_reward_uses_each_lanes_target():
    with closing(make_local("reacher", 3, "cpu", 7)) as lanes:
        for env, target in zip(lanes.envs, ((0.26, 0, 0.01), (0.27, 0, 0.01), (0.24, 0.1, 0.01)), strict=True):
            physics = env.physics
            physics.data.qpos[:] = 0
            physics.data.qvel[:] = 0
            physics.named.model.geom_pos["target"] = target
            physics.after_reset()
        task = bind_task("reacher", lanes.envs[0])
        reward = torch.tensor([env.task.get_reward(env.physics) for env in lanes.envs], dtype=torch.float64)
        assert reward.tolist() == [1.0, 0.0, 0.0]
        for dtype in (torch.float32, torch.float64):
            torch.testing.assert_close(task.reward(SourceSnapshot(lanes.envs, dtype=dtype)), reward, atol=0, rtol=0)


@pytest.mark.parametrize("key", ["ball_in_cup", "finger-spin"])
def test_sparse_reward_positives_and_negatives_match_native(key):
    with closing(make_local(key, 3, "cpu", 7)) as lanes:
        for index, env in enumerate(lanes.envs):
            physics = env.physics
            physics.data.qvel[:] = 0
            physics.data.qacc_warmstart[:] = 0
            if key == "ball_in_cup":
                # Target center, then clearly outside along each task axis.
                target = physics.named.data.site_xpos["target", ["x", "z"]].copy()
                ball = physics.named.data.xpos["ball", ["x", "z"]].copy()
                offset = np.asarray(((0.0, 0.0), (0.2, 0.0), (0.0, 0.2))[index])
                physics.named.data.qpos[["ball_x", "ball_z"]] += target - ball + offset
            else:
                # Fast negative, slow negative and fast positive spin; far from the threshold.
                physics.data.qpos[:] = [-0.5, 0.2, 0.0]
                physics.named.data.qvel["hinge"] = (-30.0, -2.0, 30.0)[index]
            physics.after_reset()
        reward = torch.tensor([env.task.get_reward(env.physics) for env in lanes.envs], dtype=torch.float64)
        success = torch.tensor([lanes._info(env, i)[0] for i, env in enumerate(lanes.envs)], dtype=torch.float64)
        assert reward.tolist() == [1.0, 0.0, 0.0]
        if key == "ball_in_cup":
            assert success.tolist() == [1.0, 0.0, 0.0]
        task = bind_task(key, lanes.envs[0])
        snapshot = SourceSnapshot(lanes.envs, dtype=torch.float32)
        torch.testing.assert_close(task.reward(snapshot), reward, atol=0, rtol=0)
        torch.testing.assert_close(task.info(snapshot)[0], success, atol=0, rtol=0, equal_nan=True)
