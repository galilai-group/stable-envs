"""A Craftax sequence in one jitted call gives the results of the same steps one at a time."""

from __future__ import annotations

import pytest
import torch

import stable_envs as se
from stable_envs.env import Env

pytest.importorskip("jax")
pytest.importorskip("craftax")

from tests.helpers.reference.semantics import assert_same_state  # noqa: E402

pytestmark = pytest.mark.unit

LANES = 3
STEPS = 9
SEED = 5


def _pair(name):
    fused = se.make(name, num_envs=LANES, device="cpu")
    stepped = se.make(name, num_envs=LANES, device="cpu")
    for env in (fused, stepped):
        state = env.reset(seed=SEED)
        # Lane 1 dies on its first step, so the rest of its sequence is padding.
        state["engine", "player_health"][1] = 0
        env.set_state(state)
    return fused, stepped


def _assert_equal(got, expected):
    assert set(got.keys()) == set(expected.keys())
    for key in expected.keys():
        if key in ("state", "states"):
            assert_same_state(got[key], expected[key])
        else:
            assert torch.equal(got[key], expected[key]), key


@pytest.mark.parametrize("name", ["CraftaxClassicSymbolic", "CraftaxClassicPixels", "CraftaxSymbolic"])
@pytest.mark.parametrize(
    "states, frames",
    [
        (slice(None), slice(-1, None)),
        (None, None),
        (None, slice(None)),
        (slice(-1, None), slice(None)),
        (slice(None), None),
    ],
)
def test_a_fused_sequence_equals_its_steps_one_at_a_time(name, states, frames):
    fused, stepped = _pair(name)
    with fused, stepped:
        g = torch.Generator().manual_seed(SEED)
        for _ in range(2):
            acts = torch.randint(0, fused.action_high + 1, (LANES, STEPS, 1), generator=g)
            got = fused.step(acts, states=states, frames=frames)
            expected = Env.step(stepped, acts, states=states, frames=frames)
            assert not got["valid"][1, 1:].any()
            _assert_equal(got, expected)
        assert_same_state(fused.get_state(), stepped.get_state())


def test_other_selections_step_one_at_a_time():
    fused, stepped = _pair("CraftaxClassicSymbolic")
    with fused, stepped:
        acts = torch.randint(0, fused.action_high + 1, (LANES, STEPS, 1), generator=torch.Generator().manual_seed(1))
        got = fused.step(acts, states=slice(0, None, 2), frames=slice(1, 3))
        expected = Env.step(stepped, acts, states=slice(0, None, 2), frames=slice(1, 3))
        _assert_equal(got, expected)


def test_a_sequence_with_an_action_out_of_range_raises():
    with se.make("CraftaxClassicSymbolic", num_envs=LANES, device="cpu") as env:
        before = env.reset(seed=SEED)
        acts = torch.zeros((LANES, STEPS, 1), dtype=torch.int64)
        acts[2, 4] = env.action_high + 1
        with pytest.raises(ValueError):
            env.step(acts)
        assert_same_state(env.get_state(), before)
