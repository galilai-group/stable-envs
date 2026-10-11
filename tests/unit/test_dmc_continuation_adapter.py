"""Fast adapter checks independent of the historical wrapper import."""

import numpy as np
import pytest
import torch
from tensordict import TensorDict

from stable_envs.envs.dmcontrol.simulation.continuation import CONFIG, CONTINUATION_FIELDS, DMCContinuationAdapter


@pytest.mark.parametrize("domain", ["cheetah", "hopper", "walker"])
def test_adapter_full_state_roundtrip_snapshot_independence_and_validation(domain):
    adapter = DMCContinuationAdapter(domain, 2)
    try:
        state = adapter.reset(seed=17)
        assert set(state.keys(True, True)) == {
            key if isinstance(key, tuple) else key for key in CONTINUATION_FIELDS[domain]
        }
        adapter.step(np.full((2, CONFIG[domain]["nu"]), 0.25))
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
            adapter.step(np.zeros((2, CONFIG[domain]["nu"] + 1)))
        with pytest.raises(ValueError):
            adapter.step(np.full((2, CONFIG[domain]["nu"]), np.nan))
    finally:
        adapter.close()


def test_production_adapter_does_not_import_stable_worldmodel():
    source = __import__(
        "inspect"
    ).getsource(__import__("stable_envs.envs.dmcontrol.simulation.continuation", fromlist=["*"]))
    assert "stable_worldmodel" not in source


@pytest.mark.parametrize(
    "name, color",
    [
        ("DMCCheetahRun", [0.7, 0.5, 0.3, 1.0]),
        ("DMCHopperHop", [0.7, 0.5, 0.3, 1.0]),
        ("DMCWalkerWalk", [0.7, 0.5, 0.3, 1.0]),
    ],
)
def test_registered_render_model_material(name: str, color: list[float]) -> None:
    """Renderer models must use pinned SWM defaults."""
    from stable_envs.envs import dmcontrol

    source = getattr(dmcontrol, name)._load_env(17)
    try:
        np.testing.assert_array_equal(source.physics.named.model.mat_rgba["self"], np.asarray(color, np.float32))
    finally:
        source.close()
