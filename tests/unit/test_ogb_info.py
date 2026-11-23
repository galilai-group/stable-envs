"""``OGBCube.source_info`` and ``OGBScene.source_info`` equal OGBench's ``compute_ob_info`` bit for bit."""

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


def _toward(env, target, gripper):
    """A normalized action that moves the pinch site of the OGBench object ``env`` toward ``target``."""
    delta = target - env._data.site_xpos[env._pinch_site_id]
    return env.normalize_action(np.array([*delta, 0.0, gripper]))


def _script(env, t):
    """Close the gripper and press it on the table, then, in Scene, push the first button down."""
    if env.family == "Scene" and t >= 30:
        top = env._data.site_xpos[env._button_site_ids[0]].copy()
        return _toward(env, top + np.array([0.0, 0.0, 0.08 if t < 55 else -0.05]), 1.0)
    low = env._data.site_xpos[env._pinch_site_id].copy()
    low[2] = 0.0
    return _toward(env, low, 1.0)


@pytest.mark.parametrize("family", FAMILIES)
def test_source_info_of_the_default_backend_equals_compute_ob_info(family):
    """``snv.make`` steps on ``mujoco.batch``; OGBench objects with the same seed take the same actions."""
    ref = OGBManipulationAdapter(family, 2, seed=5)
    env = snv.make(f"OGB{family}", num_envs=2, device="cpu", seed=5)
    try:
        state = env.reset(seed=8)
        ref.reset(seed=8)
        _check(env.source_info(state), ref.envs)
        contact, pressed = 0.0, False
        for t in range(90):
            actions = np.stack([_script(e, t) for e in ref.envs]).astype(np.float32)
            ref.step(actions)
            state = env.step(torch.from_numpy(actions))["state"]
            info = env.source_info(state)
            _check(info, ref.envs)
            contact = max(contact, float(info["proprio/gripper_contact"].max()))
            if family == "Scene":
                pressed |= bool((info["button_states"] != info["prev_button_states"]).any())
        assert contact > 0.0
        assert pressed or family == "Cube"
    finally:
        ref.close()
        env.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_source_info_of_rows_equals_the_rows_of_source_info(family):
    env = snv.make(f"OGB{family}", num_envs=4, device="cpu")
    try:
        env.reset(seed=1)
        state = env.step(torch.full((4, 5), 0.3))["state"]
        full = env.source_info(state)
        rows = torch.tensor([3, 1])
        part = env.source_info(state[rows])
        assert part.batch_size == torch.Size([2])
        for key in full.keys():
            assert torch.equal(part[key], full[key][rows]), key
        before = snv.get_field(state, "physics/qpos").clone()
        full["qpos"].zero_()
        assert torch.equal(snv.get_field(state, "physics/qpos"), before)
    finally:
        env.close()


def test_source_info_refuses_a_state_that_is_not_a_full_state():
    env = snv.make("OGBCube", num_envs=2, device="cpu")
    try:
        state = env.reset(seed=0)
        with pytest.raises(TypeError):
            env.source_info(snv.get_field(state, "physics/qpos"))
        broken = state.clone()
        snv.get_field(broken, "physics/qpos")[0, 0] = float("nan")
        with pytest.raises(ValueError):
            env.source_info(broken)
    finally:
        env.close()
