"""Exercise sparse positives through native task oracles and real Warp field views."""

from contextlib import closing

import numpy as np
import pytest
import torch

pytest.importorskip("dm_control")

from stable_envs.envs.dmcontrol.simulation.factory import make_local
from stable_envs.envs.dmcontrol.simulation.mjwarp_runtime import WarpRuntime
from stable_envs.envs.dmcontrol.simulation.mjwarp_tasks import bind_task

DEVICES = [
    "cpu",
    pytest.param(
        "cuda",
        marks=[pytest.mark.gpu, pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")],
    ),
]


def _prepare_sparse_cases(lanes, key):
    for index, env in enumerate(lanes.envs):
        physics = env.physics
        physics.data.qvel[:] = 0
        physics.data.qacc_warmstart[:] = 0
        if key == "ball_in_cup":
            # Place the ball at the native target center, then clearly outside
            # along each independent task axis. No reward formula is reproduced.
            target = physics.named.data.site_xpos["target", ["x", "z"]].copy()
            ball = physics.named.data.xpos["ball", ["x", "z"]].copy()
            offset = np.asarray(((0.0, 0.0), (0.2, 0.0), (0.0, 0.2))[index])
            physics.named.data.qpos[["ball_x", "ball_z"]] += target - ball + offset
        else:
            # Collision-free finger pose; fast negative spin, slow negative
            # spin, and fast positive spin are far from the task threshold.
            physics.data.qpos[:] = [-0.5, 0.2, 0.0]
            physics.named.data.qvel["hinge"] = (-30.0, -2.0, 30.0)[index]
        physics.after_reset()


def _assert_native_outputs(task, snapshot, expected_reward, expected_success):
    torch.testing.assert_close(task.reward(snapshot), expected_reward, atol=0, rtol=0)
    success, _ = task.info(snapshot)
    torch.testing.assert_close(success, expected_success, atol=0, rtol=0, equal_nan=True)


@pytest.mark.parametrize("key", ["ball_in_cup", "finger-spin"])
@pytest.mark.parametrize("device", DEVICES)
def test_sparse_reward_positives_and_negatives_match_native(key, device, monkeypatch):
    # Import only when executing a runtime test, so collection alone does not
    # load the optional GPU dependency in otherwise native CPU test profiles.
    pytest.importorskip("mujoco_warp")
    with closing(make_local(key, 3, "cpu", 7)) as lanes:
        _prepare_sparse_cases(lanes, key)
        expected_reward = torch.tensor(
            [env._env.task.get_reward(env.physics) for env in lanes.envs], dtype=torch.float64, device=device
        )
        expected_success = torch.tensor(
            [lanes._info(env, i)[0] for i, env in enumerate(lanes.envs)], dtype=torch.float64, device=device
        )
        # Assert that the source fixture actually contains both outcomes.
        assert expected_reward.tolist() == [1.0, 0.0, 0.0]
        if key == "ball_in_cup":
            assert expected_success.tolist() == [1.0, 0.0, 0.0]
        else:
            assert bool(torch.isnan(expected_success).all())

        models = [env.physics.model.ptr for env in lanes.envs]
        task = bind_task(key, lanes.envs[0])
        with closing(WarpRuntime(models, device, lanes.envs[0]._env._n_sub_steps)) as runtime:
            runtime.load(lanes.get_state())
            snapshot = runtime.snapshot()
            _assert_native_outputs(task, snapshot, expected_reward, expected_success)

            # These probes establish that an always-zero reward or reversed
            # predicate cannot pass merely because ordinary resets are negative.
            observed = task.reward(snapshot)
            for faulty in (torch.zeros_like(observed), 1 - observed):
                with monkeypatch.context() as patch:
                    patch.setattr(task, "reward", lambda _, value=faulty: value)
                    with pytest.raises(AssertionError):
                        _assert_native_outputs(task, snapshot, expected_reward, expected_success)
            if key == "ball_in_cup":
                _, info = task.info(snapshot)
                with monkeypatch.context() as patch:
                    patch.setattr(task, "info", lambda _: (torch.zeros_like(expected_success), info))
                    with pytest.raises(AssertionError):
                        _assert_native_outputs(task, snapshot, expected_reward, expected_success)
