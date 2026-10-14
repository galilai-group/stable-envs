"""Exact owned column export for each classic-control registration."""

import numpy as np
import pytest
import torch

from stable_envs.envs.classic_control.simulation.local import CLASSIC_FIELDS, ClassicControlAdapter
from tests.helpers.reference.classic_state_export import classic_state
from tests.helpers.reference.semantics import assert_same_state


@pytest.mark.parametrize("domain", CLASSIC_FIELDS)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_classic_export_continuation_and_ownership(domain, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
    adapter = ClassicControlAdapter(domain, 3, device, 17)
    try:
        adapter.reset(17)
        cached = adapter.envs[1].unwrapped.np_random.bit_generator.state
        cached["has_uint32"], cached["uinteger"] = 1, 2**32 - 1
        adapter.envs[1].unwrapped.np_random.bit_generator.state = cached
        initial = adapter.get_state()
        held = initial.clone()
        assert_same_state(initial, classic_state(adapter))
        if domain == "cartpole":
            adapter.envs[2].unwrapped.steps_beyond_terminated = 7
            assert_same_state(adapter.get_state(), classic_state(adapter))
            adapter.set_state(initial)
        action = (
            np.ones((3, 1), np.int64)
            if domain in ("cartpole", "acrobot", "mountain_car")
            else np.full((3, 1), 0.25, np.float32)
        )
        for _ in range(8):
            result = adapter.step(action)
            assert_same_state(result["state"], classic_state(adapter))
        if domain == "mountain_car_continuous":
            assert result["state", "task", "state_is_float32"].all()
            assert not initial["task", "state_is_float32"].any()
        assert_same_state(initial, held)
        adapter.set_state(initial)
        assert_same_state(adapter.get_state(), initial)
        first = adapter.step(action)
        adapter.set_state(initial)
        assert_same_state(first, adapter.step(action))
        adapter.reset()
        assert_same_state(adapter.get_state(), classic_state(adapter))
        assert_same_state(initial, held)
        changed = adapter.get_state()
        before = classic_state(adapter)
        for value in changed.values(True, True):
            value.zero_()
        assert_same_state(adapter.get_state(), before)
        adapter.close()
        with pytest.raises(RuntimeError, match="closed"):
            adapter.get_state()
    finally:
        adapter.close()
