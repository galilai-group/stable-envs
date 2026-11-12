"""Roadster physics routing, the CPU batch against the source after early terminations, and frames from
render workers and from steps."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

pytest.importorskip("mujoco")

import stable_envs as snv
from stable_envs import tuning
from stable_envs._execution import render_pool
from stable_envs.envs.roadster import render as roadster_render
from stable_envs.envs.roadster.simulation.adapter import _from_limbs
from stable_envs.envs.roadster.simulation.batch import RoadsterNativeAdapter
from stable_envs.envs.roadster.simulation.execution import MJWARP_MIN_ENVS, roadster_backend
from stable_envs.envs.roadster.source import RoadsterPlanEnv
from tests.helpers.reference.semantics import assert_same_state

pytestmark = pytest.mark.unit

LANES = 4
SEED = 7
#: Per-environment (wind speed, rain): calm, windy and wet, windy, heavy rain.
WEATHER = np.array([[0.0, 0.0], [6.0, 0.3], [12.0, 0.0], [3.0, 0.9]])


def _actions(steps: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1, 1, (LANES, 4)).astype(np.float32) for _ in range(steps)]


def test_the_backend_follows_the_device_and_the_cutoff():
    assert roadster_backend(8, "cpu") == "native"
    assert roadster_backend(8, "cpu", "auto") == "native"
    assert roadster_backend(1, "cuda") == "mjwarp"
    assert roadster_backend(MJWARP_MIN_ENVS - 1, "cuda", "auto") == "native"
    assert roadster_backend(MJWARP_MIN_ENVS, "cuda", "auto") == "mjwarp"
    assert roadster_backend(4, "cuda", "auto", mjwarp_min_envs=4) == "mjwarp"
    assert roadster_backend(4096, "cuda", "native") == "native"
    with pytest.raises(ValueError, match="cuda"):
        roadster_backend(8, "cpu", "mjwarp")
    with pytest.raises(ValueError, match="simulation_backend"):
        roadster_backend(8, "cpu", "source")
    with snv.RoadsterLoop(num_envs=2, device="cpu") as env:
        assert env.simulation_backend == "native" and env.execution_mode == "DELEGATED_CPU"
        assert isinstance(env._simulator, RoadsterNativeAdapter)


def test_environments_that_end_within_an_action_continue_as_the_source():
    """Random commands crash the car within an action; the source then stops that action's substeps and its
    gust draws. The batch keeps every environment's State, its gust generator included, equal to the source."""
    weather = np.array([[8.0, 0.2], [0.0, 0.0], [4.0, 0.0]])
    batch = RoadsterNativeAdapter("loop", len(weather))
    sources = [RoadsterPlanEnv(task="loop") for _ in weather]
    try:
        batch.configure_physics(weather)
        batch.reset(SEED)
        for i, src in enumerate(sources):
            src._loop_goal_state()  # the batch takes the reference run once, before any episode
            src.reset(seed=SEED + i, wind=weather[i, 0], rain=weather[i, 1])
        rng = np.random.default_rng(2)
        ended = np.zeros(len(weather), bool)
        for t in range(40):
            actions = rng.uniform(-1, 1, (len(weather), 4)).astype(np.float32)
            out = batch.step(actions)
            state = out["state"]
            for i, src in enumerate(sources):
                reward, terminated, _, _ = src.step(actions[i])
                ended[i] |= terminated
                np.testing.assert_array_equal(state["sim", "qpos"][i].numpy(), src.data.qpos, err_msg=str(t))
                np.testing.assert_array_equal(state["weather", "gust"][i].numpy(), src.sim.weather.gust)
                rng_state = src.sim.weather._wind_rng.bit_generator.state["state"]["state"]
                assert _from_limbs(state["weather", "rng_state"][i].numpy()) == rng_state
                assert out["reward"][i].item() == reward and out["terminated"][i].item() == terminated
        assert ended[0], "the windy environment should crash under random commands"
    finally:
        batch.close()


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="render workers run on Linux only")
@pytest.mark.parametrize("look", ["wm", "cinematic"])
def test_render_workers_draw_the_in_process_frames(look):
    with snv.RoadsterTrafficJam(num_envs=LANES, device="cpu") as env:
        env._simulator.configure_physics(WEATHER)
        state = env.reset(seed=SEED)
        for action in _actions(4):
            state = env.step(action, frames=None)["state"]
        rows = env._pack_render_state(state)
    host = rows.to(torch.float64).numpy()
    expected = np.empty((LANES, 64, 64, 3), np.uint8)
    roadster_render.renderer(64, look).draw(host, expected)
    pool = render_pool.RenderPool(roadster_render.Drawer(64, look), 2, software_gl=True, name="test render")
    try:
        frames = pool.render(LANES, (64, 64, 3), lambda part: host[part])
    finally:
        pool.close()
    np.testing.assert_array_equal(frames, expected)
    with tuning.override(render_workers=2, render_worker_min_rows=1):
        pooled = roadster_render.render_rows(rows, 64, "cpu", look=look)
    np.testing.assert_array_equal(pooled.numpy(), expected)


def test_step_frames_match_render_of_the_step_states():
    """Frames a step returns, drawn from the live render-row fields without a State snapshot, equal
    render(state) of that step's State; frame-only steps leave the final State as a step that captures every
    State would."""
    actions = np.stack(_actions(6, seed=3), 1)
    options = {"variation_values": {"weather.wind": [6.0], "weather.rain": [0.5]}}
    with snv.RoadsterTrafficJam(num_envs=LANES, device="cpu", size=48) as env:
        env.set_variation({"rendering.cinematic": [1, 0, 0, 1]})  # both looks in one batch
        env.reset(seed=SEED, options=options)
        both = env.step(actions, states=slice(None), frames=slice(None))
        for t in range(actions.shape[1]):
            assert torch.equal(both["frames"][:, t], env.render(both["states"][:, t])), t
        env.reset(seed=SEED, options=options)
        frames_only = env.step(actions, states=None, frames=slice(None))
        assert torch.equal(frames_only["frames"], both["frames"])
        assert_same_state(frames_only["state"], both["state"])
