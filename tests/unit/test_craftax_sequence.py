"""A Craftax sequence in one jitted call gives the results of the same steps one at a time."""

from __future__ import annotations

import pytest
import torch

import stable_envs as se

pytest.importorskip("jax")
pytest.importorskip("craftax")

from tests.helpers.reference.craftax import assert_same_sequence, per_step  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402

pytestmark = pytest.mark.unit

LANES = 3
STEPS = 9
SEED = 5
SELECTIONS = [
    (slice(None), slice(-1, None)),
    (None, None),
    (None, slice(None)),
    (slice(-1, None), slice(None)),
    (slice(None), None),
    (slice(0, None, 2), slice(1, 3)),
    (slice(None, None, -1), slice(None, None, -3)),
    (slice(4, 5), slice(-1, None)),
]


def _pair(name):
    fused = se.make(name, num_envs=LANES, device="cpu")
    stepped = per_step(se.make(name, num_envs=LANES, device="cpu"))
    for env in (fused, stepped):
        state = env.reset(seed=SEED)
        # Lane 1 dies on its first step, so the rest of its sequence is padding.
        state["engine", "player_health"][1] = 0
        env.set_state(state)
    return fused, stepped


# Each case compiles its own programs, so the selections run on one game and the other games take two of them.
CASES = [("CraftaxClassicPixels", *selection) for selection in SELECTIONS] + [
    (name, *SELECTIONS[i]) for name in ("CraftaxClassicSymbolic", "CraftaxSymbolic") for i in (0, 5)
]


@pytest.mark.parametrize("name, states, frames", CASES)
def test_a_fused_sequence_equals_its_steps_one_at_a_time(name, states, frames):
    fused, stepped = _pair(name)
    with fused, stepped:
        g = torch.Generator().manual_seed(SEED)
        for length in (STEPS, 2):
            acts = torch.randint(0, fused.action_high + 1, (LANES, length, 1), generator=g)
            got = fused.step(acts, states=states, frames=frames)
            assert not got["valid"][1, 1:].any()
            assert_same_sequence(got, stepped.step(acts, states=states, frames=frames))
        assert_same_state(fused.get_state(), stepped.get_state())


def _compiled_sequences(env) -> int:
    return sum(fn._cache_size() for (name, _), fn in env._k._jitted.items() if name[0] == "sequence")


def test_lengths_up_to_a_power_of_two_share_one_program():
    with se.make("CraftaxClassicSymbolic", num_envs=LANES, device="cpu") as env:
        env.reset(seed=SEED)
        env.step(torch.zeros((LANES, 5, 1), dtype=torch.int64), frames=None)
        compiled = _compiled_sequences(env)
        for length in (6, 7, 8):
            out = env.step(torch.zeros((LANES, length, 1), dtype=torch.int64), frames=None)
            assert out["reward"].shape == (LANES, length) and out["valid"].shape == (LANES, length)
        assert _compiled_sequences(env) == compiled


@pytest.mark.parametrize("states, frames", SELECTIONS)
def test_a_host_action_out_of_range_raises_before_any_step(states, frames):
    with se.make("CraftaxClassicSymbolic", num_envs=LANES, device="cpu") as env:
        before = env.reset(seed=SEED)
        acts = torch.zeros((LANES, STEPS, 1), dtype=torch.int64)
        acts[2, 4] = env.action_high + 1
        with pytest.raises(ValueError, match="actions must be integers"):
            env.step(acts, states=states, frames=frames)
        assert_same_state(env.get_state(), before)


def test_the_hook_builds_no_final_state_when_not_asked():
    with se.make("CraftaxClassicSymbolic", num_envs=LANES, device="cpu") as env:
        env.reset(seed=SEED)
        acts = torch.zeros((LANES, 3, 1), dtype=torch.int64)
        for length in (1, 3):
            out = env._step_sequence(acts[:, :length], None, None, False)
            assert "state" not in out and out["reward"].shape == (LANES, length)
