"""Column export preserves every pre-change OGBench field and owned snapshot."""

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
from stable_envs.envs.ogbench.simulation.maze import OGBMazeAdapter
from tests.helpers.reference.ogbench_state_export import manipulation_state, maze_state


def equal(a, b):
    assert a.batch_size == b.batch_size and a.device == b.device
    assert set(a.keys(True, True)) == set(b.keys(True, True))
    for key in a.keys(True, True):
        assert a[key].shape == b[key].shape and a[key].dtype == b[key].dtype
        torch.testing.assert_close(a[key], b[key], rtol=0, atol=0, equal_nan=True)


@pytest.mark.parametrize("family", ["Maze", "Cube", "Scene"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_owned_export(family, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requires Oscar")
    adapter = OGBMazeAdapter(3, device, 17) if family == "Maze" else OGBManipulationAdapter(family, 3, device, 17)
    reference = maze_state if family == "Maze" else manipulation_state
    try:
        if family == "Maze":
            cached = list(adapter._global_rng[2])
            cached[3:] = [1, 0.125]
            adapter._global_rng[2] = tuple(cached)
        else:
            cached = list(adapter.envs[2].task_rng.get_state())
            cached[3:] = [1, 0.125]
            adapter.envs[2].task_rng.set_state(tuple(cached))
        initial = adapter.get_state()
        retained = initial.clone()
        equal(initial, reference(adapter))
        action = np.random.default_rng(21).uniform(-0.5, 0.5, (3, 2 if family == "Maze" else 5))
        for _ in range(8):
            result = adapter.step(action)
            equal(result["state"], reference(adapter))
        equal(initial, retained)
        adapter.set_state(initial)
        equal(adapter.get_state(), reference(adapter))
        equal(adapter.get_state(), initial)
        adapter.reset(seed=None)
        equal(adapter.get_state(), reference(adapter))
        equal(initial, retained)
        edited = adapter.get_state()
        before = reference(adapter)
        for value in edited.values(True, True):
            value.zero_()
        equal(adapter.get_state(), before)
        if family != "Maze":
            adapter.reset(seed=42, options={"task_id": 4, "variation": ["cube.size", "camera.angle_delta"]})
            equal(adapter.get_state(), reference(adapter))
        adapter.close()
        with pytest.raises(RuntimeError, match="closed"):
            adapter.get_state()
    finally:
        adapter.close()
