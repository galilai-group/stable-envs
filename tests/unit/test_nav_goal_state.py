"""``goal_state`` of TwoRoom, TwoRoomTarget and Piecewise draws the source's target image."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as snv

pytestmark = pytest.mark.unit

CLASSES = {"TwoRoom": "two_room", "Piecewise": "piecewise"}


@pytest.mark.swm
@pytest.mark.parametrize("name", sorted(CLASSES))
@pytest.mark.parametrize("seed", [0, 9])
def test_goal_frames_against_the_source_target_image(name, seed, record_property):
    pytest.importorskip("stable_worldmodel")
    from tests.helpers.reference.native_navigation_semantics import NativeNavigationReference

    source = NativeNavigationReference(CLASSES[name], 3)
    with snv.make(name, num_envs=3, device="cpu") as env:
        state = env.reset(seed=seed)
        source.reset(seed)
        for field in ("agent/position", "target/position"):
            assert torch.equal(snv.get_field(state, field), snv.get_field(source.get_state(), field)), field
        got = env.render(env.goal_state(state)).numpy().astype(np.float64)
        want = np.stack([e._target_img.permute(1, 2, 0).numpy() for e in source.envs]).astype(np.float64)
        mse = float(((got - want) ** 2).mean())
        record_property("goal_mse", mse)
        # The source-pixel gate of these classes allows the same bound.
        assert mse <= 1e-3


@pytest.mark.parametrize("name", ["TwoRoom", "TwoRoomTarget", "Piecewise"])
def test_goal_state_puts_the_agent_on_the_target_and_changes_nothing_else(name):
    with snv.make(name, num_envs=4, device="cpu") as env:
        env.reset(seed=2)
        state = env.step(torch.full((4, 3, 2), 0.7))["state"]
        before = state.clone()
        goal = env.goal_state(state)
        assert torch.equal(snv.get_field(goal, "agent/position"), snv.get_field(state, "target/position"))
        assert torch.equal(snv.get_field(goal, "target/position"), snv.get_field(state, "target/position"))
        assert all(torch.equal(a, b) for a, b in zip(state.values(True, True), before.values(True, True)))
        rows = torch.tensor([2, 1])
        part = env.goal_state(state[rows])
        for key, value in part.items(True, True):
            assert torch.equal(value, goal[key][rows]), key
