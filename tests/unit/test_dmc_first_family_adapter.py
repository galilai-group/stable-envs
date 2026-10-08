"""Fast adapter checks independent of the historical wrapper import."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs.envs.dmcontrol.simulation.first_family import FIRST_FAMILY_FIELDS, FirstFamilyAdapter


@pytest.mark.parametrize("domain", ["acrobot", "pendulum", "cartpole"])
def test_adapter_full_state_roundtrip_and_rejects_partial_state(domain):
    adapter = FirstFamilyAdapter(domain, 2)
    try:
        state = adapter.reset(seed=17)
        assert set(state.keys(True, True)) == {
            key if isinstance(key, tuple) else key for key in FIRST_FAMILY_FIELDS[domain]
        }
        adapter.step(np.full((2, 1), 0.25))
        adapter.set_state(state)
        for key in state.keys(True, True):
            assert torch.equal(adapter.get_state()[key], state[key])
        snapshot = adapter.get_state()
        live = adapter.get_state()
        snapshot["qpos"].add_(1)
        snapshot["rng", "keys"].add_(1)
        for key in live.keys(True, True):
            assert torch.equal(adapter.get_state()[key], live[key])
        with pytest.raises((TypeError, ValueError)):
            adapter.set_state(TensorDict({"qpos": state["qpos"]}, [2]))
    finally:
        adapter.close()
