"""Cube/Scene lanes sharing compiled models behave exactly like lanes that own them."""

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

from stable_envs.envs.ogbench.simulation.compatibility import COMPILED_FIELDS
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


@pytest.mark.parametrize("family", ["Cube", "Scene"])
def test_configure_matches_recompiling_from_source(family):
    values = {
        "cube.size": np.array([0.027]),
        "floor.color": np.array([[0.3, 0.2, 0.1], [0.6, 0.5, 0.4]]),
        "camera.angle_delta": np.array([[3.0, -2.0]]),
        "light.intensity": np.array([0.35]),
    }
    configured = OGBManipulationAdapter(family, 1, seed=3)
    recompiled = OGBManipulationAdapter(family, 1, seed=3)
    try:
        default_diffuse = configured.envs[0].model.light_diffuse.copy()
        configured.configure([values])
        assert not np.allclose(configured.envs[0].model.light_diffuse, default_diffuse)
        recompiled.reset(3, [{"variation": [], "variation_values": values}])
        for name in COMPILED_FIELDS:
            got, want = (getattr(a.envs[0].model, name) for a in (configured, recompiled))
            assert np.allclose(got, want, rtol=1e-6, atol=1e-9), name
    finally:
        configured.close()
        recompiled.close()


def test_closing_lanes_releases_their_shared_models():
    adapter = OGBManipulationAdapter("Cube", 2, seed=1)
    pool = adapter._pool
    adapter.configure([{"light.intensity": np.array([0.3])}, {}])
    assert len(pool.models) == 2
    adapter.envs[0].close()
    assert len(pool.models) == 1
    adapter.close()
    assert not pool.models
