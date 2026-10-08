"""Fast Humanoid adapter checks independent of the historical wrapper import."""

import inspect

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs.envs.dmcontrol.simulation.humanoid import HUMANOID_CONFIG, HUMANOID_FIELDS, DMCHumanoidAdapter


def test_adapter_full_state_roundtrip_snapshot_independence_and_validation():
    adapter = DMCHumanoidAdapter(2)
    try:
        state = adapter.reset(seed=17)
        assert set(state.keys(True, True)) == {
            key if isinstance(key, tuple) else key for key in HUMANOID_FIELDS
        }
        adapter.step(np.full((2, HUMANOID_CONFIG["nu"]), 0.25))
        adapter.set_state(state)
        for key in state.keys(True, True):
            assert torch.equal(adapter.get_state()[key], state[key])

        snapshot, live = adapter.get_state(), adapter.get_state()
        snapshot["qpos"].add_(1)
        snapshot["warmstart"].add_(1)
        snapshot["rng", "keys"].add_(1)
        for key in live.keys(True, True):
            assert torch.equal(adapter.get_state()[key], live[key])

        with pytest.raises((TypeError, ValueError)):
            adapter.set_state(TensorDict({"qpos": state["qpos"]}, [2]))
        with pytest.raises(ValueError):
            adapter.step(np.zeros((2, 22)))
        with pytest.raises(ValueError):
            adapter.step(np.full((2, 21), np.nan))
    finally:
        adapter.close()


def test_production_adapter_does_not_import_stable_worldmodel():
    import stable_envs.envs.dmcontrol.simulation.humanoid as module

    assert "stable_worldmodel" not in inspect.getsource(module)
