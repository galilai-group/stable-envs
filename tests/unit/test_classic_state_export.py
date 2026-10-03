"""Exact owned column export for each recovered classic-control registration."""

import numpy as np
import pytest
import torch

from stable_envs.envs._classic_semantics import CLASSIC_FIELDS, ClassicControlAdapter
from tests.helpers.reference.classic_state_export import classic_state


def equal(a, b):
    assert a.batch_size == b.batch_size and a.device == b.device
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    for key in a.keys(True, True):
        assert a[key].dtype == b[key].dtype and a[key].shape == b[key].shape
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("domain", CLASSIC_FIELDS)
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_classic_export_continuation_and_ownership(domain, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requires Oscar")
    adapter = ClassicControlAdapter(domain, 3, device, 17)
    try:
        adapter.reset(17)
        cached = adapter.envs[1].unwrapped.np_random.bit_generator.state
        cached["has_uint32"], cached["uinteger"] = 1, 2**32 - 1
        adapter.envs[1].unwrapped.np_random.bit_generator.state = cached
        initial = adapter.get_state()
        held = initial.clone()
        equal(initial, classic_state(adapter))
        if domain == "cartpole":
            adapter.envs[2].unwrapped.steps_beyond_terminated = 7
            equal(adapter.get_state(), classic_state(adapter))
            adapter.set_state(initial)
        action = np.ones((3, 1), np.int64) if domain in ("cartpole", "acrobot", "mountain_car") else np.full((3, 1), 0.25, np.float32)
        for _ in range(8):
            result = adapter.step(action)
            equal(result["state"], classic_state(adapter))
        if domain == "mountain_car_continuous":
            assert result["state", "task", "state_is_float32"].all()
            assert not initial["task", "state_is_float32"].any()
        equal(initial, held)
        adapter.set_state(initial)
        equal(adapter.get_state(), initial)
        first = adapter.step(action)
        adapter.set_state(initial)
        equal(first, adapter.step(action))
        adapter.reset()
        equal(adapter.get_state(), classic_state(adapter))
        equal(initial, held)
        changed = adapter.get_state()
        before = classic_state(adapter)
        for value in changed.values(True, True):
            value.zero_()
        equal(adapter.get_state(), before)
        adapter.close()
        with pytest.raises(RuntimeError, match="closed"):
            adapter.get_state()
    finally:
        adapter.close()
