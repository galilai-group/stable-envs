"""``OGBCube.goal_state`` and ``OGBScene.goal_state``: the scene where OGBench draws its task-mode goal image."""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")
pytest.importorskip("mujoco")

import stable_envs as snv
from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
from stable_envs.render._resources import retain_process_lifetime

pytestmark = pytest.mark.unit

FAMILIES = ("Cube", "Scene")


def _record_goals(env):
    """Record the scene and the upstream frame at each goal observation of ``env``'s resets."""
    goals, last = [], {}
    observe, initialize = env.compute_observation, env.initialize_episode

    def compute_observation():
        last["qpos"] = env._data.qpos.copy()
        last["mocap_pos"] = env._data.mocap_pos.ravel().copy()
        last["mocap_quat"] = env._data.mocap_quat.ravel().copy()
        if env.family == "Scene":
            last["button_states"] = np.asarray(env._cur_button_states, np.int64).copy()
        last["frame"] = np.asarray(env.render())[..., :3].copy()
        return observe()

    def initialize_episode():
        result = initialize()
        goals.append(dict(last))
        return result

    env.compute_observation, env.initialize_episode = compute_observation, initialize_episode
    return goals


@pytest.mark.parametrize("family", FAMILIES)
def test_goal_state_is_the_scene_of_the_ogbench_goal_image(family, record_property):
    ref = OGBManipulationAdapter(family, 2, seed=3)
    # Closing the source renderers would release GL contexts that other live contexts share.
    retain_process_lifetime(ref, f"OGBench {family} goal-image reference")
    records = [_record_goals(env) for env in ref.envs]
    state = ref.reset(seed=7)
    env = snv.make(f"OGB{family}", num_envs=2, device="cpu")
    try:
        goal = env.goal_state(state)
        names = ["qpos", "mocap_pos", "mocap_quat"] + (["button_states"] if family == "Scene" else [])
        targets = {"button_states": "scene/cur_button_states"}
        for i, goals in enumerate(records):
            for name in names:
                want = goals[-1][name]
                assert np.array_equal(snv.get_field(state, f"task/goal/{name}")[i].numpy(), want), name
                assert np.array_equal(snv.get_field(goal, targets.get(name, f"physics/{name}"))[i].numpy(), want)
        frames = env.render(goal, backend="opengl").numpy().astype(np.float64)
        want = np.stack([goals[-1]["frame"] for goals in records]).astype(np.float64)
        mse = float(((frames - want) ** 2).mean())
        record_property("goal_mse", mse)
        assert mse == 0.0
    finally:
        env.close()


@pytest.mark.parametrize("family", FAMILIES)
def test_goal_state_keeps_through_steps_and_restores(family):
    env = snv.make(f"OGB{family}", num_envs=3, device="cpu")
    try:
        first = env.reset(seed=4)
        state = env.step(torch.full((3, 4, 5), 0.4))["state"]
        before = state.clone()
        goal = env.goal_state(state)
        for name in ("task/goal/qpos", "task/goal/mocap_pos"):
            assert torch.equal(snv.get_field(state, name), snv.get_field(first, name))
        assert all(torch.equal(a, b) for a, b in zip(state.values(True, True), before.values(True, True)))
        assert torch.equal(snv.get_field(goal, "physics/qpos"), snv.get_field(state, "task/goal/qpos"))
        rows = torch.tensor([2, 0])
        part = env.goal_state(state[rows])
        for key, value in part.items(True, True):
            assert torch.equal(value, goal[key][rows]), key
        env.set_state(state)
        restored = env.get_state()
        assert all(torch.equal(a, b) for a, b in zip(restored.values(True, True), state.values(True, True)))
    finally:
        env.close()
