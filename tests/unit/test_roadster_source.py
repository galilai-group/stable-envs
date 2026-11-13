"""Every Roadster cell matches the source's ``RoadsterPlanEnv`` driven directly, lane by lane.

The source gate of the family: reset, State, reward, success and both done flags over a rollout of the
scripted expert with noise, under wind and rain, and the frame MuJoCo draws of the live simulation at the
same State.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

import stable_envs as snv
from stable_envs._core.mujoco_gl import mujoco_renderer
from stable_envs.envs.roadster.envs import CELLS
from stable_envs.envs.roadster.source import GoalExpert, RoadsterPlanEnv
from stable_envs.envs.roadster.source.weather import apply_render_flags

pytestmark = pytest.mark.unit

LANES = 2
SEED = 11
STEPS = 8
#: Values in ``variation_values`` reach the source exactly (float64 physics leaves).
WIND, RAIN = 5.0, 0.3
#: Renderers of the live source simulation, kept for the process: releasing one can invalidate other GL contexts.
_SOURCE_RENDERERS: dict = {}


def _lane_actions(expert, lane, rng):
    return np.clip(expert(lane) + 0.3 * rng.standard_normal(4), -1, 1).astype(np.float32)


@pytest.mark.parametrize("name,task", CELLS)
def test_lanes_match_the_source_called_directly(name, task):
    with snv.make(name, num_envs=LANES, device="cpu") as env:
        state = env.reset(seed=SEED, options={"variation_values": {"weather.wind": [WIND], "weather.rain": [RAIN]}})
        sources = [RoadsterPlanEnv(task=task) for _ in range(LANES)]
        for lane, src in enumerate(sources):
            src.reset(seed=SEED + lane, wind=WIND, rain=RAIN)
            np.testing.assert_array_equal(state["sim", "qpos"][lane].numpy(), src.data.qpos)
            np.testing.assert_array_equal(state["goal", "position"][lane].numpy(), src.goal_pos)
            np.testing.assert_array_equal(state["weather", "wind_mean"][lane].numpy(), src.sim.weather.wind_mean)
            assert state["task", "status"][lane].item() == list(("drive", "flight")).index(src.sim.status)
        expert, rng = GoalExpert(), np.random.default_rng(SEED)
        for t in range(STEPS):
            actions = np.stack([_lane_actions(expert, src, rng) for src in sources])
            result = env.step(actions, frames=None)
            for lane, src in enumerate(sources):
                reward, terminated, truncated, info = src.step(actions[lane])
                np.testing.assert_array_equal(
                    result["state"]["sim", "qpos"][lane].numpy(), src.data.qpos, err_msg=str(t)
                )
                np.testing.assert_array_equal(result["state"]["sim", "qvel"][lane].numpy(), src.data.qvel)
                assert result["reward"][lane].item() == reward, t
                assert result["success"][lane].item() == info["success"]
                assert result["failed"][lane].item() == info["failed"]
                assert result["terminated"][lane].item() == terminated
                assert result["truncated"][lane].item() == truncated
                assert result["cost"][lane].item() == info["cost"]
                assert result["state"]["lifecycle", "elapsed_steps"][lane].item() == src.steps
                assert result["state"]["task", "fuel"][lane].item() == src.sim.fuel
        np.testing.assert_array_equal(
            result["state"]["sim", "mocap_pos"][0].numpy(), sources[0].data.mocap_pos.reshape(-1)
        )


def test_frames_match_the_live_simulation_drawn_directly():
    """``render(state)`` from the compact rows equals MuJoCo drawing the source's own data at that State
    (smoke and rain included, the flicker seeded); the rows hold float32, so a few edge pixels may differ."""
    with snv.RoadsterTrafficJam(num_envs=1, device="cpu") as env:
        env.reset(seed=SEED, options={"variation_values": {"weather.rain": [0.6]}})
        src = RoadsterPlanEnv(task="traffic")
        src.reset(seed=SEED, rain=0.6)
        expert = GoalExpert()
        for _ in range(12):
            action = expert(src).astype(np.float32)
            src.step(action)
            result = env.step(action[None], frames=None)
        np.testing.assert_array_equal(result["state"]["sim", "qpos"][0].numpy(), src.data.qpos)
        src.model.vis.quality.offsamples = 0
        renderer = _SOURCE_RENDERERS.setdefault("traffic", mujoco_renderer(src.model, height=96, width=96))
        src.sim.draw_effects(np.random.default_rng(0))
        renderer.update_scene(src.data, camera="wm")
        apply_render_flags(renderer.scene, src.sim.weather)
        expected = torch.from_numpy(renderer.render().copy())
        frame = env.render(result["state"])[0]
        differing = (frame != expected).any(-1).float().mean().item()
        assert differing < 0.01, f"{differing:.1%} of the pixels differ from the source frame"
        assert (result["state"]["fx", "count"] == 60).all() and result["state"]["sim", "ctrl"].abs().max() > 0.05
