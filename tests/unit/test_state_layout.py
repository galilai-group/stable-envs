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


def _storages(state):
    return {state[key].untyped_storage().data_ptr() for key in state.keys(True, True)}


@pytest.mark.parametrize("num_envs", [2, 8192])
def test_a_state_from_tensor_values_shares_no_storage_with_them(num_envs):
    """Every dtype is copied, including a dtype with a single leaf, for small and for reused large buffers."""
    leaves = [("a", (3,), torch.float32), ("b", (2,), torch.float32), ("flag", (1,), torch.bool)]
    layout = StateLayout(leaves, num_envs)
    values = {
        "a": torch.rand(num_envs, 3),
        "b": torch.rand(num_envs, 2),
        "flag": torch.zeros(num_envs, 1, dtype=torch.bool),
    }
    sources = {value.untyped_storage().data_ptr() for value in values.values()}
    for _ in range(3):
        state = layout.export(values, "cpu")
        assert not _storages(state) & sources
        before = state.clone()
        for value in values.values():
            value.logical_not_() if value.dtype == torch.bool else value.add_(1)
        assert (state == before).all()


def test_a_large_buffer_in_shared_memory_is_not_reused():
    """Another process may read a State moved to shared memory, though the use count doesn't show it."""
    n = 1 << 15
    layout = StateLayout([("x", (2,), torch.float32), ("y", (2,), torch.float32)], n)
    values = {"x": torch.rand(n, 2), "y": torch.rand(n, 2)}
    state = layout.export(values, "cpu")
    state["x"].untyped_storage().share_memory_()
    shared = state["x"].untyped_storage().data_ptr()
    del state
    for _ in range(3):
        assert shared not in _storages(layout.export(values, "cpu"))


@pytest.mark.parametrize("device", ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)])
def test_a_layout_without_static_leaves_exports_in_state_order(device):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA required")
    layout = StateLayout(LEAVES, 2)
    values = _values(2)
    state = layout.export(values, device)
    keys = [key if isinstance(key, tuple) else (key,) for key in state.keys(True, True)]
    assert keys == [("qpos",), ("lifecycle", "step_count"), ("rng", "keys"), ("task", "done")]
    values["qpos"] += 100
    assert state["qpos"][0, 0].item() == 0


def test_host_leaves_give_each_leaf_its_declared_shape_and_dtype():
    layout = StateLayout(LEAVES, 2)
    values = _values(2)
    values[("lifecycle", "step_count")] = values[("lifecycle", "step_count")].reshape(2).astype(np.int32)
    leaves = layout.host_leaves(values, ("state",))
    paths = [("qpos",), ("lifecycle", "step_count"), ("rng", "keys"), ("task", "done")]
    assert list(leaves) == [("state", *path) for path in paths]
    assert leaves[("state", "lifecycle", "step_count")].shape == (2, 1)
    assert leaves[("state", "lifecycle", "step_count")].dtype == np.int64
    with pytest.raises(RuntimeError, match="static"):
        StateLayout(LEAVES, 2, static=frozenset({"rng/keys"})).host_leaves(values)
