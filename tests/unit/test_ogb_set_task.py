"""``OGBCube.set_task`` and ``OGBScene.set_task`` against OGBench objects in data-collection mode."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")
pytest.importorskip("mujoco")

import stable_envs as snv
from stable_envs.envs.ogbench.simulation.compatibility import TARGET_TASKS
from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.unit

FAMILIES = ("Cube", "Scene")


def _collect(env, seed, **options):
    """OGBench's own data-collection target draw on the object ``env``, with Gymnasium's seeding."""
    from gymnasium.utils import seeding

    env._mode = "data_collection"
    env._np_random, env._np_random_seed = seeding.np_random(seed)
    env.set_new_target(return_info=False, **options)


def _toward(env, target, gripper):
    delta = target - env._data.site_xpos[env._pinch_site_id]
    return env.normalize_action(np.array([*delta, 0.0, gripper]))


def _press(env, t, button):
    """Push the top of ``button`` down from above."""
    top = env._data.site_xpos[env._button_site_ids[button]].copy()
    return _toward(env, top + np.array([0.0, 0.0, 0.08 if t < 25 else -0.05]), 1.0)


def _same_outputs(got, want):
    for key in ("reward", "success", "terminated", "truncated"):
        assert torch.equal(got[key].cpu(), want[key].cpu()), key


def _upstream_info(env):
    info = env.compute_ob_info()
    info["privileged/target_task"] = TARGET_TASKS.index(str(info["privileged/target_task"]))
    info["success"] = env._success
    info["goal"] = env._cur_goal_ob
    return info


@pytest.mark.parametrize("family", FAMILIES)
def test_targets_and_rollout_equal_ogbench_data_collection(family):
    """Targets per seed, then a rollout: State, reward, success and terminated equal OGBench's, bit for bit."""
    rng = np.random.default_rng(1)
    ref = OGBManipulationAdapter(family, 3, seed=5)
    env = snv.make(f"OGB{family}", num_envs=3, device="cpu", seed=5)
    try:
        env.reset(seed=8)
        ref.reset(seed=8)
        for _ in range(4):
            action = rng.uniform(-1, 1, (3, 5)).astype(np.float32)
            ref.step(action)
            env.step(torch.from_numpy(action))
        state = env.set_task([2, 0], seed=[21, 22])
        _collect(ref.envs[2], 21)
        _collect(ref.envs[0], 22)
        ref._forget_static(ref.envs)
        assert_same_state(state, ref.get_state())
        assert snv.get_field(state, "task/mode").reshape(-1).tolist() == [1, 0, 1]
        info = env.source_info(state)
        for i in (0, 2):
            for key, value in _upstream_info(ref.envs[i]).items():
                assert np.array_equal(info[key][i].numpy(), np.asarray(value)), key
        for _ in range(40):
            action = rng.uniform(-1, 1, (3, 5)).astype(np.float32)
            want, got = ref.step(action), env.step(torch.from_numpy(action))
            assert_same_state(got["state"], want["state"])
            _same_outputs(got, want)
        # A new reset returns every row to task mode.
        state = env.reset(seed=3)
        assert not snv.get_field(state, "task/mode").any()
    finally:
        ref.close()
        env.close()


def test_scene_scores_the_target_only():
    """A Scene row with a button target succeeds when that button changes, as OGBench scores it."""
    probe = OGBManipulationAdapter("Scene", 1, seed=0)
    try:
        probe.reset(seed=4)
        seed = next(s for s in range(200) if _picks_button_zero(probe, s))
    finally:
        probe.close()
    ref = OGBManipulationAdapter("Scene", 2, seed=0)
    env = snv.make("OGBScene", num_envs=2, device="cpu", seed=0)
    try:
        env.reset(seed=4)
        ref.reset(seed=4)
        env.set_task(torch.tensor([True, False]), seed=[seed])
        _collect(ref.envs[0], seed)
        ref._forget_static(ref.envs)
        succeeded = False
        for t in range(60):
            action = np.stack([_press(e, t, 0) for e in ref.envs]).astype(np.float32)
            want, got = ref.step(action), env.step(torch.from_numpy(action))
            assert_same_state(got["state"], want["state"])
            _same_outputs(got, want)
            succeeded |= bool(got["success"][0])
            assert not bool(got["success"][1])
        assert succeeded
    finally:
        ref.close()
        env.close()


def _picks_button_zero(adapter, seed):
    env = adapter.envs[0]
    saved = adapter.get_state()
    _collect(env, seed)
    found = env._target_task == "button" and env._target_button == 0
    adapter.set_state(saved)
    return found


@pytest.mark.parametrize("family", FAMILIES)
def test_rows_equal_the_same_rows_of_a_full_set_task(family):
    """``set_task`` on some rows equals those rows of ``set_task`` on every row with the same seeds."""
    a = snv.make(f"OGB{family}", num_envs=3, device="cpu")
    b = snv.make(f"OGB{family}", num_envs=3, device="cpu")
    try:
        start = a.reset(seed=2)
        b.reset(seed=2)
        part = a.set_task([1], seed=[40])
        full = b.set_task([0, 1, 2], seed=[30, 40, 50])
        for key, value in part.items(True, True):
            assert torch.equal(value[1], full[key][1]), key
            assert torch.equal(value[[0, 2]], start[key][[0, 2]]), key
    finally:
        a.close()
        b.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_set_task_state_restores_and_has_no_goal_frame(family):
    env = snv.make(f"OGB{family}", num_envs=2, device="cpu")
    other = snv.make(f"OGB{family}", num_envs=2, device="cpu")
    try:
        env.reset(seed=6)
        state = env.set_task([1], seed=[3], options={"p_stack": 0.0})
        other.reset(seed=1)
        other.set_state(state)
        assert_same_state(other.get_state(), state)
        action = torch.full((2, 5), 0.2)
        assert_same_state(other.step(action)["state"], env.step(action)["state"])
        with pytest.raises(ValueError, match=r"rows \[1\]"):
            env.goal_render_state(state)
        env.goal_render_state(state[[0]])
    finally:
        env.close()
        other.close()


def test_set_task_refuses_bad_arguments():
    env = snv.make("OGBCube", num_envs=2, device="cpu")
    try:
        env.reset(seed=0)
        with pytest.raises(ValueError, match="one nonnegative integer seed"):
            env.set_task([0, 1], seed=[1])
        with pytest.raises(ValueError, match="Unknown set_task options"):
            env.set_task([0], seed=[1], options={"p_put": 0.1})
        with pytest.raises(ValueError, match="distinct"):
            env.set_task([0, 0], seed=[1, 2])
        with pytest.raises(ValueError, match="mask"):
            env.set_task(torch.tensor([True]), seed=[1])
    finally:
        env.close()
