"""Snapshot selection must preserve exact continuation and return owned State."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

from stable_envs.envs._classic_semantics import ClassicControlAdapter
from tests.helpers.reference.classic_snapshot import NAMES, make
from tests.unit.test_classic_state_export import equal


@pytest.mark.parametrize("domain", NAMES)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_requested_snapshots_only_and_current_render(domain, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requires Oscar")
    with make(domain, 4, device, legacy=True) as old, make(domain, 4, device) as new:
        initial = old.get_state()
        limit = old._simulator.envs[0]._max_episode_steps
        initial["lifecycle", "elapsed_steps"][:, 0] = torch.tensor([0, limit - 1, limit, 0], device=device)
        initial["rng", "has_uint32"][-1] = 1
        initial["rng", "uinteger"][-1] = 2**32 - 1
        discrete = domain in ("cartpole", "acrobot", "mountain_car")
        actions = np.ones((4, 8, 1), np.int64) if discrete else np.full((4, 8, 1), .1, np.float32)
        if domain == "cartpole":
            actions[:] = np.arange(8)[None, :, None] % 2
        for horizon in (1, 8):
            a = actions[:, 0] if horizon == 1 else actions
            for states, frames in ((None, None), (None, slice(-1, None)),
                                   (slice(None), None), (slice(None, None, -2), slice(None, None, 3)),
                                   (slice(10, 12), slice(10, 12)), (None, slice(None))):
                for env in (old, new):
                    env.set_state(initial)
                with patch.object(new._simulator, "get_state", wraps=new._simulator.get_state) as exports:
                    actual = new.step(a, states=states, frames=frames)
                    selected = {horizon - 1}
                    for selection in (states, frames):
                        if selection is not None:
                            selected.update(range(horizon)[selection])
                    assert exports.call_count == len(selected)
                expected = old.step(a, states=states, frames=frames)
                equal(actual, expected)
                assert actual["valid"][1:3].sum(1).tolist() == [1, 1]
                torch.testing.assert_close(old.render(), new.render(), rtol=0, atol=0)
                held = actual.clone()
                new.reset(42)
                equal(actual, held)
        for env in (old, new):
            env.set_state(initial)
        equal(old.reset(), new.reset())


@pytest.mark.parametrize("domain", NAMES)
def test_advance_really_omits_export_and_matches_full_step(domain):
    old, new = ClassicControlAdapter(domain, 3), ClassicControlAdapter(domain, 3)
    try:
        equal(old.reset(17), new.reset(17))
        action = np.ones((3, 1), np.int64) if domain in ("cartpole", "acrobot", "mountain_car") else np.full((3, 1), .1, np.float32)
        with patch.object(new, "get_state", side_effect=AssertionError("unrequested export")):
            for t in range(8):
                active = None if t == 0 else torch.tensor([True, False, True])
                expected = old.step(action, active=active)
                actual = new.step(action, active=active, capture_state=False)
                assert "state" not in actual.keys()
                equal(expected.exclude("state"), actual)
        equal(old.get_state(), new.get_state())
    finally:
        old.close()
        new.close()
