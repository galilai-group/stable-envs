"""Column export matches the reference OGBench fields and returns owned snapshots."""

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter
from stable_envs.envs.ogbench.simulation.maze import OGBMazeAdapter
from tests.helpers.reference.ogbench_state_export import manipulation_state, maze_state
from tests.helpers.reference.semantics import assert_same_state


@pytest.mark.parametrize("family", ["Maze", "Cube", "Scene"])
@pytest.mark.parametrize("device", ["cpu", "cuda"])
def test_complete_owned_export(family, device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("requires CUDA")
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
        assert_same_state(initial, reference(adapter))
        action = np.random.default_rng(21).uniform(-0.5, 0.5, (3, 2 if family == "Maze" else 5))
        for _ in range(8):
            result = adapter.step(action)
            assert_same_state(result["state"], reference(adapter))
        assert_same_state(initial, retained)
        adapter.set_state(initial)
        assert_same_state(adapter.get_state(), reference(adapter))
        assert_same_state(adapter.get_state(), initial)
        adapter.reset(seed=None)
        assert_same_state(adapter.get_state(), reference(adapter))
        assert_same_state(initial, retained)
        edited = adapter.get_state()
        before = reference(adapter)
        for value in edited.values(True, True):
            value.zero_()
        assert_same_state(adapter.get_state(), before)
        if family != "Maze":
            adapter.reset(seed=42, options={"task_id": 4, "variation": ["cube.size", "camera.angle_delta"]})
            assert_same_state(adapter.get_state(), reference(adapter))
        adapter.close()
        with pytest.raises(RuntimeError, match="closed"):
            adapter.get_state()
    finally:
        adapter.close()
