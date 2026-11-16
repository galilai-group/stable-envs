"""Craftax rejects out-of-range actions without moving, and its State histories match leafwise stacking."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se
from tests.helpers.reference.semantics import assert_same_state

pytest.importorskip("jax")
pytest.importorskip("craftax")

from tests.helpers.reference.craftax import NAMES, actions  # noqa: E402

pytestmark = pytest.mark.unit

LANES = 3
CUDA = pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA")
INPUTS = [
    pytest.param("cpu", "numpy", id="cpu-numpy"),
    pytest.param("cpu", "cpu", id="cpu-tensor"),
    pytest.param("cuda", "numpy", id="cuda-numpy", marks=CUDA),
    pytest.param("cuda", "cpu", id="cuda-cpu-tensor", marks=CUDA),
    pytest.param("cuda", "cuda", id="cuda-tensor", marks=CUDA),
]


def _as(values: np.ndarray, kind: str):
    return values if kind == "numpy" else torch.as_tensor(values, device=kind)


@pytest.mark.parametrize(("device", "kind"), INPUTS)
def test_an_out_of_range_action_raises_and_leaves_every_lane_unchanged(device, kind):
    with se.make("CraftaxClassicSymbolic", num_envs=LANES, device=device) as env:
        env.reset(seed=7)
        before = env.get_state()
        # 2**32 wraps to 0 as int32, so the range check must see the caller's dtype.
        good = actions(env, 1, 7)[0]
        for bad in (-1, env.action_high + 1, 2**32):
            values = np.zeros((LANES, 1), np.int64)
            values[1, 0] = bad
            if kind == "cuda":
                # Device actions are checked on the device; the next step raises without a sync on this one.
                rejected = env.step(_as(values, kind))
                assert rejected["valid"].all() and not rejected["reward"].any() and not rejected["terminated"].any()
                assert_same_state(rejected["state"], before)
                assert_same_state(env.get_state(), before)
                values = good
            with pytest.raises(ValueError, match="actions must be integers"):
                env.step(_as(values, kind))
            assert_same_state(env.get_state(), before)
        with pytest.raises(ValueError, match="actions must be integers"):
            env.step(_as(np.zeros((LANES, 1), np.float32), kind))
        assert_same_state(env.get_state(), before)
        stepped = env.step(_as(good, kind))
        with se.make("CraftaxClassicSymbolic", num_envs=LANES, device=device) as fresh:
            fresh.reset(seed=7)
            assert_same_state(stepped, fresh.step(_as(good, kind)))


@CUDA
def test_close_raises_for_a_rejected_step_that_no_call_read():
    env = se.make("CraftaxClassicSymbolic", num_envs=LANES, device="cuda")
    env.reset(seed=7)
    values = torch.zeros((LANES, 1), dtype=torch.int64, device="cuda")
    values[0, 0] = -1
    env.step(values)
    with pytest.raises(ValueError, match="actions must be integers"):
        env.close()
    with pytest.raises(RuntimeError, match="closed"):
        env.reset(seed=7)
    env.close()


@pytest.mark.parametrize("name", NAMES)
def test_state_histories_own_their_storage(name):
    with se.make(name, num_envs=LANES, device="cpu") as env:
        env.reset(seed=7)
        sequence = torch.as_tensor(actions(env, 4, 7)).permute(1, 0, 2)
        for out, length in ((env.step(sequence), 4), (env.step(sequence[:, 0]), 1)):
            states, state = out["states"], out["state"]
            assert states.batch_size == torch.Size([LANES, length])
            assert_same_state(states[:, -1], state)
            held = {state[key].untyped_storage().data_ptr() for key in state.keys(True, True)}
            for key in states.keys(True, True):
                assert states[key].untyped_storage().data_ptr() not in held, key
            before = state.clone()
            states.apply_(lambda x: x.zero_())
            assert_same_state(state, before)
