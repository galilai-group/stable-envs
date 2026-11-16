"""On CUDA, a Craftax sequence in one jitted call gives the results of the same steps one at a time."""

from __future__ import annotations

import pytest
import torch

import stable_envs as se
from stable_envs import get_field

jax = pytest.importorskip("jax")
pytest.importorskip("craftax")

from tests.helpers.reference.craftax import assert_same_sequence, per_step  # noqa: E402
from tests.helpers.reference.semantics import assert_same_state  # noqa: E402

pytestmark = pytest.mark.gpu


def _gpu() -> bool:
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


if not torch.cuda.is_available() or not _gpu():
    pytest.skip("Craftax on CUDA needs CUDA and the craftax-gpu extra", allow_module_level=True)

NUM_ENVS = 5
STEPS = 11
SEED = 3


def _pair(name: str):
    fused = se.make(name, num_envs=NUM_ENVS, device="cuda")
    stepped = per_step(se.make(name, num_envs=NUM_ENVS, device="cuda"))
    for env in (fused, stepped):
        state = env.reset(seed=SEED)
        # Environment 2 dies on its first step, so it holds its state and gives zero outputs after it.
        get_field(state, "engine/player_health")[2] = 0
        env.set_state(state)
    return fused, stepped


@pytest.mark.parametrize("name", ["CraftaxClassicPixels", "CraftaxSymbolic"])
@pytest.mark.parametrize("device_actions", [False, True], ids=["host-actions", "cuda-actions"])
def test_a_fused_sequence_with_every_state_and_frame_equals_its_steps(name, device_actions):
    fused, stepped = _pair(name)
    with fused, stepped:
        g = torch.Generator().manual_seed(SEED)
        for _ in range(2):
            acts = torch.randint(0, fused.action_high + 1, (NUM_ENVS, STEPS, 1), generator=g)
            if device_actions:
                acts = acts.cuda()
            got = fused.step(acts, states=slice(None), frames=slice(None))
            assert got["frames"].device.type == "cuda" and got["frames"].shape[1] == STEPS
            assert got["valid"][2, 0] and not got["valid"][2, 1:].any()
            assert not got["reward"][2, 1:].any() and not got["terminated"][2, 1:].any()
            assert_same_sequence(got, stepped.step(acts, states=slice(None), frames=slice(None)))
        assert_same_state(fused.get_state(), stepped.get_state())


def test_a_cuda_sequence_with_an_action_out_of_range_changes_nothing_and_the_next_call_raises():
    with se.make("CraftaxClassicSymbolic", num_envs=NUM_ENVS, device="cuda") as env:
        before = env.reset(seed=SEED)
        acts = torch.zeros((NUM_ENVS, STEPS, 1), dtype=torch.int64, device="cuda")
        acts[1, 7] = env.action_high + 1
        out = env.step(acts, states=slice(None), frames=None)
        assert out["valid"].all() and not out["reward"].any()
        assert_same_state(env.get_state(), before)
        with pytest.raises(ValueError, match="actions must be integers"):
            env.step(acts[:, :1])
