"""Focused adapter tests for the aligned DMC cohort."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs._adapters.dmcontrol_aligned import ALIGNED_FIELDS, CONFIG, DMCAlignedAdapter


@pytest.mark.parametrize("domain", ALIGNED_FIELDS)
def test_aligned_adapter_roundtrip_independence_and_validation(domain):
    adapter = DMCAlignedAdapter(domain, 2)
    try:
        state = adapter.reset(seed=17)
        adapter.step(np.full((2, CONFIG[domain]["nu"]), 0.25))
        adapter.set_state(state)
        for key in state.keys(True, True):
            assert torch.equal(adapter.get_state()[key], state[key])
        saved, live = adapter.get_state(), adapter.get_state()
        saved["qpos"].add_(1)
        saved["warmstart"].add_(1)
        saved["rng", "keys"].add_(1)
        for key in live.keys(True, True):
            assert torch.equal(adapter.get_state()[key], live[key])
        with pytest.raises((TypeError, ValueError)):
            adapter.set_state(TensorDict({"qpos": state["qpos"]}, [2]))
        with pytest.raises(ValueError):
            adapter.step(np.zeros((2, CONFIG[domain]["nu"] + 1)))
    finally:
        adapter.close()


def test_production_adapter_has_no_swm_runtime_import():
    import inspect

    import stable_envs._adapters.dmcontrol_aligned as module

    assert "stable_worldmodel" not in inspect.getsource(module)
