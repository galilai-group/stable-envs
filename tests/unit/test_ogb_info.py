"""``OGBCube.info`` and ``OGBScene.info`` equal OGBench's ``compute_ob_info`` bit for bit."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")
pytest.importorskip("mujoco")

import stable_envs as snv
from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter

pytestmark = pytest.mark.unit

FAMILIES = ("Cube", "Scene")


def _upstream(env):
    info = env.compute_ob_info()
    info["success"] = env._success
    info["goal"] = env._cur_goal_ob
    return info


def _check(info, envs):
    for i, env in enumerate(envs):
        want = _upstream(env)
        assert list(info.keys()) == list(want.keys())
        for key, value in want.items():
            got = info[key][i].numpy()
            value = np.asarray(value)
            assert got.dtype == value.dtype and got.shape == value.shape, key
            assert np.array_equal(got, value), key


@pytest.mark.parametrize("family", FAMILIES)
def test_info_equals_ogbench_compute_ob_info(family):
    rng = np.random.default_rng(0)
    ref = OGBManipulationAdapter(family, 3, seed=5)
    env = snv.make(f"OGB{family}", num_envs=3, device="cpu")
    try:
        _check(env.info(ref.get_state()), ref.envs)
        for t in range(40):
            action = np.clip(rng.normal(0, 0.6, (3, 5)), -1, 1)
            if t % 10 < 4:
                action[:, 2] = -1.0
            state = ref.step(action)["state"]
            if t % 8 == 7:
                _check(env.info(state), ref.envs)
    finally:
        ref.close()
        env.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_info_of_rows_equals_the_rows_of_info(family):
    env = snv.make(f"OGB{family}", num_envs=4, device="cpu")
    try:
        env.reset(seed=1)
        state = env.step(torch.full((4, 5), 0.3))["state"]
        full = env.info(state)
        rows = torch.tensor([3, 1])
        part = env.info(state[rows])
        assert part.batch_size == torch.Size([2])
        for key in full.keys():
            assert torch.equal(part[key], full[key][rows]), key
        before = snv.get_field(state, "physics/qpos").clone()
        full["qpos"].zero_()
        assert torch.equal(snv.get_field(state, "physics/qpos"), before)
    finally:
        env.close()
