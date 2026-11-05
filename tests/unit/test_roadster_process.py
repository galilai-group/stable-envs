"""Roadster lanes in worker processes, and frames from render workers, match one process exactly."""

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
from stable_envs.envs.roadster.simulation import execution
from stable_envs.envs.roadster.simulation.adapter import RoadsterAdapter
from stable_envs.envs.roadster.simulation.process import RoadsterProcessAdapter
from tests.helpers.reference.semantics import assert_same_state

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not sys.platform.startswith("linux"), reason="worker processes run on Linux only"),
]

LANES = 4
SEED = 7
#: Per-lane (wind speed, rain): calm, windy and wet, windy, heavy rain.
WEATHER = np.array([[0.0, 0.0], [6.0, 0.3], [12.0, 0.0], [3.0, 0.9]])


def _actions(steps: int, seed: int = 0) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    return [rng.uniform(-1, 1, (LANES, 4)).astype(np.float32) for _ in range(steps)]


@pytest.fixture(scope="module")
def pair():
    local = RoadsterAdapter("traffic", LANES)
    remote = RoadsterProcessAdapter("traffic", LANES, processes=2)
    try:
        yield local, remote
    finally:
        remote.close()
        local.close()


def test_workers_step_the_lanes_of_one_local_batch(pair):
    local, remote = pair
    for adapter in pair:
        adapter.configure_physics(WEATHER)
    assert_same_state(remote.reset(SEED), local.reset(SEED))
    for t, action in enumerate(_actions(6)):
        active = None if t < 3 else torch.tensor([True, False, True, True])
        assert_same_state(remote.step(action, active=active), local.step(action, active=active))
    action = _actions(1, seed=5)[0]
    quiet = remote.step(action, capture_state=False)
    assert "state" not in quiet.keys()
    assert_same_state(quiet, local.step(action, capture_state=False))
    assert_same_state(remote.get_state(), local.get_state())


def test_restore_reaches_every_worker_or_none(pair):
    local, remote = pair
    for adapter in pair:
        adapter.configure_physics(WEATHER)
        adapter.reset(SEED)
    for action in _actions(3):
        local.step(action)
    saved = local.get_state()
    remote.set_state(saved)
    assert_same_state(remote.get_state(), saved)
    for action in _actions(3, seed=9):
        assert_same_state(remote.step(action), local.step(action))
    live = remote.get_state()
    bad = live.clone()
    bad["task", "route_idx"][3] = 7
    with pytest.raises(ValueError, match="route_idx"):
        remote.set_state(bad)
    assert_same_state(remote.get_state(), live)
    with pytest.raises(ValueError):
        remote.configure_physics(np.full((LANES, 2), -1.0))
    with pytest.raises((TypeError, ValueError)):
        remote.step(np.zeros((LANES, 3), np.float32))
    assert_same_state(remote.get_state(), live)


def test_a_process_env_matches_an_in_process_env(monkeypatch):
    """With workers chosen, stepping through the pipeline gives the in-process outputs, States and frames."""
    actions = np.stack(_actions(5, seed=2), 1)
    options = {"variation_values": {"weather.wind": [5.0], "weather.rain": [0.4]}}
    # The policy needs 4 CPUs; pick the worker count directly so the test runs on any Linux machine.
    monkeypatch.setattr(execution, "cpu_affinity", lambda: None)
    results = []
    for workers in (0, 2):
        monkeypatch.setattr(execution, "process_count", lambda *args, workers=workers, **kwargs: workers)
        env = snv.RoadsterCarHop(num_envs=LANES, device="cpu", size=48)
        with env:
            assert isinstance(env._simulator, RoadsterProcessAdapter) == bool(workers)
            env.reset(seed=SEED, options=options)
            results.append(env.step(actions, states=slice(None), frames=slice(None)))
    assert_same_state(results[1], results[0])


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


@pytest.mark.parametrize("workers", [0, 2])
def test_step_frames_match_render_of_the_step_states(monkeypatch, workers):
    """Frames a step returns, drawn without a State snapshot (in-process from the live lanes, or in a process
    batch from the render-row fields), equal render(state) of that step's State; frame-only steps leave the
    final State as a step that captures every State would."""
    monkeypatch.setattr(execution, "cpu_affinity", lambda: None)
    monkeypatch.setattr(execution, "process_count", lambda *args, **kwargs: workers)
    actions = np.stack(_actions(6, seed=3), 1)
    options = {"variation_values": {"weather.wind": [6.0], "weather.rain": [0.5]}}
    with snv.RoadsterTrafficJam(num_envs=LANES, device="cpu", size=48) as env:
        assert isinstance(env._simulator, RoadsterProcessAdapter) == bool(workers)
        env.set_variation({"rendering.cinematic": [1, 0, 0, 1]})  # both looks in one batch
        env.reset(seed=SEED, options=options)
        both = env.step(actions, states=slice(None), frames=slice(None))
        for t in range(actions.shape[1]):
            assert torch.equal(both["frames"][:, t], env.render(both["states"][:, t])), t
        env.reset(seed=SEED, options=options)
        frames_only = env.step(actions, states=None, frames=slice(None))
        assert torch.equal(frames_only["frames"], both["frames"])
        assert_same_state(frames_only["state"], both["state"])
        _FRAMES[workers] = both["frames"]
    if len(_FRAMES) == 2:
        assert torch.equal(_FRAMES[0], _FRAMES[2]), "process batches draw the in-process frames"


#: Step frames per worker count, compared across the two parametrizations.
_FRAMES: dict = {}
