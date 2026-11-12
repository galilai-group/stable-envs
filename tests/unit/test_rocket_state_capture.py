"""Rocket State reads only the leaves a step changes and keeps the rest from the last full read."""

import numpy as np
import pytest
import torch

pytest.importorskip("PyFlyt")
pytest.importorskip("pybullet")

from stable_envs.envs.rocket.simulation.adapter import RocketAdapter  # noqa: E402


def _read(adapter):
    """The State of every lane read in full from the engines."""
    rows = [env.read_state() for env in adapter.envs]
    return {tuple(key.split("/")): np.stack([row[key] for row in rows]) for key in rows[0]}


def _assert_full_read(adapter):
    state = adapter.get_state()
    expected = _read(adapter)
    leaves = dict(state.items(True, True))
    assert leaves.keys() == expected.keys()
    for key, value in expected.items():
        actual = leaves[key].cpu().numpy()
        assert actual.dtype == value.dtype, key
        assert np.array_equal(actual, value, equal_nan=True), key


def _actions(rng, batch, ignite):
    actions = rng.uniform([-1, -1, -1, 0, 0, -1, -1], 1, (batch, 7))
    actions[:, 3] = ignite
    return actions


@pytest.mark.parametrize("config", [None, {"render_mode": "rgb_array", "render_resolution": (16, 16)}])
def test_state_matches_a_full_read_through_an_episode(config):
    torch.set_num_threads(1)
    rng = np.random.default_rng(3)
    adapter = RocketAdapter(3, "cpu", 11, config)
    try:
        _assert_full_read(adapter)
        for t in range(40):
            # Lane 0 ignites at step 5, which recolours the flame; lane 2 never ignites.
            ignite = np.array([float(t >= 5), 1.0, 0.0])
            adapter.step(_actions(rng, 3, ignite), capture_state=t % 7 == 0)
            if t % 5 == 0:
                _assert_full_read(adapter)
        adapter.configure([{"rocket.body_color": np.array([10, 20, 30], np.uint8)}, {}, {}])
        _assert_full_read(adapter)
        saved = adapter.get_state()
        adapter.reset(4, {"variation": ["all"]})
        _assert_full_read(adapter)
        adapter.step(_actions(rng, 3, np.ones(3)))
        _assert_full_read(adapter)
        adapter.set_state(saved)
        _assert_full_read(adapter)
    finally:
        adapter.close()


def test_state_matches_a_full_read_after_landing_and_terminal_steps():
    adapter = RocketAdapter(2, "cpu", 0)
    try:
        # Free fall ends the episode; later steps are post-terminal.
        idle = np.zeros((2, 7))
        for _ in range(400):
            out = adapter.step(idle, capture_state=False)
            if bool(out["terminated"].all() or out["truncated"].all()):
                break
        assert bool((out["terminated"] | out["truncated"]).all())
        _assert_full_read(adapter)
        adapter.step(idle)
        _assert_full_read(adapter)
    finally:
        adapter.close()


def test_static_leaves_are_the_ones_a_step_does_not_change():
    adapter = RocketAdapter(1, "cpu", 0)
    try:
        static = set(adapter.process_static_leaves)
        assert static.isdisjoint(adapter.process_step_state())
        assert static | set(adapter.process_step_state()) == set(adapter.fields)
        assert ("current", "wrapper", "current_goal") in static
        assert ("history", "actions") not in static
    finally:
        adapter.close()
