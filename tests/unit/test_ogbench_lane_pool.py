"""Cube/Scene lanes sharing compiled models behave exactly like lanes that own them."""

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter


def model_arrays(model):
    names = [n for n in dir(model) if not n.startswith("_")]
    return {n: getattr(model, n).copy() for n in names if isinstance(getattr(model, n), np.ndarray)}


@pytest.mark.parametrize("family", ["Cube", "Scene"])
def test_shared_lanes_match_owned_lanes(family):
    pooled = OGBManipulationAdapter(family, 3, seed=17)
    owned = [OGBManipulationAdapter(family, 1, seed=17 + i) for i in range(3)]
    rng = np.random.default_rng(0)

    def run(steps):
        for _ in range(steps):
            action = rng.uniform(-1, 1, (3, 5))
            pooled.step(action, capture_state=False)
            for i, adapter in enumerate(owned):
                adapter.step(action[i : i + 1], capture_state=False)

    def check(models):
        assert len(pooled._pool.models) == models
        state = pooled.get_state()
        for i, adapter in enumerate(owned):
            for key, value in adapter.get_state().items(True, True):
                assert torch.equal(state[key][i : i + 1], value), key
            ours, theirs = model_arrays(pooled.envs[i].model), model_arrays(adapter.envs[0].model)
            assert [n for n in theirs if not np.array_equal(ours[n], theirs[n], equal_nan=True)] == []

    try:
        run(12)
        check(1)
        sizes = [
            {"variation": [], "variation_values": {"cube.size": np.array([0.02 + 0.005 * (i // 2)])}} for i in range(3)
        ]
        pooled.reset(5, sizes)
        for i, adapter in enumerate(owned):
            adapter.reset(5 + i, [sizes[i]])
        run(4)
        check(2)
        light = [{"light.intensity": np.array([0.3])}, {"light.intensity": np.array([0.7])}, {}]
        pooled.configure(light)
        for i, adapter in enumerate(owned):
            adapter.configure([light[i]])
        run(4)
        check(3)
        saved = pooled.get_state()
        pooled.reset(3)
        pooled.set_state(saved)
        for i, adapter in enumerate(owned):
            adapter.set_state(saved[i : i + 1])
        run(4)
        check(3)
    finally:
        for adapter in [pooled, *owned]:
            adapter.close()
