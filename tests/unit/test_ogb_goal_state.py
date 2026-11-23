"""``OGBCube.goal_render_state`` and ``OGBScene.goal_render_state``: the scene of OGBench's task-mode goal."""

from __future__ import annotations

import mujoco
import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

import stable_envs as snv
from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
from stable_envs.render._resources import retain_process_lifetime

pytestmark = pytest.mark.unit

FAMILIES = ("Cube", "Scene")
SCENE = ("qpos", "mocap_pos", "mocap_quat")


def _frame(env, data):
    env._renderer.update_scene(data, camera="front_pixels")
    return np.asarray(env._renderer.render())[..., :3].copy()


def _record_goals(env):
    """Record, at each goal observation of ``env``'s resets, the scene and two upstream frames of it.

    ``drawn`` is what OGBench draws as its goal image there. ``forward`` draws the same ``qpos`` after
    ``mj_forward``, on a copy of the data.
    """
    goals, last = [], {}
    observe, initialize = env.compute_observation, env.initialize_episode

    def compute_observation():
        for name in SCENE:
            last[name] = getattr(env._data, name).ravel().copy()
        if env.family == "Scene":
            last["button_states"] = np.asarray(env._cur_button_states, np.int64).copy()
        last["drawn"] = _frame(env, env._data)
        data = mujoco.MjData(env._model)
        data.qpos[:], data.mocap_pos[:], data.mocap_quat[:] = env._data.qpos, env._data.mocap_pos, env._data.mocap_quat
        mujoco.mj_forward(env._model, data)
        last["forward"] = _frame(env, data)
        return observe()

    def initialize_episode():
        result = initialize()
        goals.append(dict(last))
        return result

    env.compute_observation, env.initialize_episode = compute_observation, initialize_episode
    return goals


@pytest.mark.parametrize("family", FAMILIES)
def test_goal_render_state_draws_the_scene_of_the_ogbench_goal(family, record_property):
    ref = OGBManipulationAdapter(family, 2, seed=3)
    # Closing the source renderers would release GL contexts that other live contexts share.
    retain_process_lifetime(ref, f"OGBench {family} goal-image reference")
    records = [_record_goals(env) for env in ref.envs]
    state = ref.reset(seed=7)
    env = snv.make(f"OGB{family}", num_envs=2, device="cpu")
    try:
        names = [*SCENE, *(["button_states"] if family == "Scene" else [])]
        for i, goals in enumerate(records):
            for name in names:
                assert np.array_equal(snv.get_field(state, f"task/goal_scene/{name}")[i].numpy(), goals[-1][name])
        frames = env.render(env.goal_render_state(state, size=224), backend="opengl").numpy().astype(np.float64)
        drawn, forward = (
            np.stack([goals[-1][key] for goals in records]).astype(np.float64) for key in ("drawn", "forward")
        )
        np.testing.assert_array_equal(frames, forward)
        # Cube's step runs no mj_forward after its last substep, so OGBench's goal image draws the poses of the
        # substep before the stored qpos. Scene's post_step runs mj_forward, so its goal image is the same frame.
        mse = float(((frames - drawn) ** 2).mean())
        record_property("goal_mse_vs_ogbench_goal_image", mse)
        if family == "Scene":
            assert mse == 0.0
    finally:
        env.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_goal_scene_keeps_through_steps_and_follows_each_reset(family):
    env = snv.make(f"OGB{family}", num_envs=3, device="cpu")
    ref = OGBManipulationAdapter(family, 3, seed=0)
    try:
        first = env.reset(seed=4)
        names = [name for name in snv.schema_of(first).names if name.startswith("task/goal_scene/")]
        assert len(names) == (4 if family == "Scene" else 3)
        state = env.step(torch.full((3, 6, 5), 0.4))["state"]
        for name in names:
            assert torch.equal(snv.get_field(state, name), snv.get_field(first, name)), name
        before = state.clone()
        goal = env.goal_render_state(state)
        assert all(torch.equal(a, b) for a, b in zip(state.values(True, True), before.values(True, True)))
        rows = torch.tensor([2, 0])
        assert torch.equal(env.render(env.goal_render_state(state[rows])), env.render(goal)[rows])
        with pytest.raises((TypeError, ValueError)):
            env.set_state(goal)
        again = env.reset(seed=9)
        fresh = ref.reset(seed=9)
        for name in names:
            assert torch.equal(snv.get_field(again, name), snv.get_field(fresh, name)), name
            assert not torch.equal(
                snv.get_field(again, "task/goal_scene/qpos"), snv.get_field(first, "task/goal_scene/qpos")
            )
        env.set_state(state)
        restored = env.get_state()
        assert all(torch.equal(a, b) for a, b in zip(restored.values(True, True), state.values(True, True)))
    finally:
        ref.close()
        env.close()
