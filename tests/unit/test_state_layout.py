"""StateLayout exports a complete State that owns its memory, from host or device values."""

import numpy as np
import pytest
import torch

from stable_envs._core.state_layout import StateLayout

LEAVES = [
    ("qpos", (3,), torch.float64),
    (("lifecycle", "step_count"), (1,), torch.int64),
    ("rng/keys", (4,), torch.int64),
    ("task/done", (1,), torch.bool),
]


def _values(n, offset=0):
    return {
        "qpos": np.arange(3 * n, dtype=np.float64).reshape(n, 3) + offset,
        ("lifecycle", "step_count"): np.full((n, 1), 7 + offset, np.int64),
        "rng/keys": np.arange(4 * n, dtype=np.int64).reshape(n, 4) + offset,
        "task/done": np.zeros((n, 1), np.bool_),
    }


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_export_builds_the_tree_and_owns_its_memory(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    layout = StateLayout(LEAVES, 2, static=frozenset({"rng/keys"}))
    values = _values(2)
    layout.load_static(values)
    state = layout.export(values, device)
    assert state.batch_size == (2,) and state.device.type == device
    assert list(state.keys()) == ["qpos", "lifecycle", "rng", "task"]
    for key, value in values.items():
        path = tuple(key.split("/")) if isinstance(key, str) else key
        assert torch.equal(state[path].cpu(), torch.from_numpy(value))
        assert state[path].dtype == layout.dtypes[key]
    values["qpos"] += 100
    values["rng/keys"] += 100
    assert state["qpos"][0, 0].item() == 0 and state["rng", "keys"][0, 0].item() == 0
    again = layout.export(values, device)
    assert again["qpos"][0, 0].item() == 100 and again["rng", "keys"][0, 0].item() == 0
    layout.load_static(values)
    assert layout.export(values, device)["rng", "keys"][0, 0].item() == 100


def test_columns_reads_a_state_back_and_checks_it():
    layout = StateLayout(LEAVES, 2)
    state = layout.export(_values(2), "cpu")
    columns = layout.columns(state)
    assert set(columns) == {key for key, _, _ in LEAVES}
    with pytest.raises(ValueError, match="qpos"):
        layout.columns(state.set("qpos", torch.zeros(2, 4, dtype=torch.float64)))


def test_export_needs_the_static_leaves():
    layout = StateLayout(LEAVES, 2, static=frozenset({"qpos"}))
    with pytest.raises(RuntimeError, match="load_static"):
        layout.export(_values(2), "cpu")


def test_a_leaf_without_a_shape_takes_the_width_of_each_export():
    layout = StateLayout([("qpos", (3,), torch.float64), ("history/actions", None, torch.float64)], 2)
    for width in (0, 1, 5):
        values = {"qpos": np.zeros((2, 3)), "history/actions": np.ones((2, width, 2))}
        state = layout.export(values, "cpu")
        assert state["history", "actions"].shape == (2, width, 2)
        assert torch.equal(state["history", "actions"], torch.ones(2, width, 2, dtype=torch.float64))
