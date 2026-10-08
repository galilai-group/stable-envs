"""Column-wise validation must match the frozen validator and own host data."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

from stable_envs._adapters.ogbench.manipulation import OGBManipulationAdapter
from tests.helpers.reference.ogbench_before_validation_columns import OGBManipulationAdapter as Before


@pytest.fixture(scope="module", params=[("Cube", "cpu"), ("Scene", "cpu"), ("Cube", "cuda"), ("Scene", "cuda")])
def adapter(request):
    family, device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requires Oscar")
    owner = OGBManipulationAdapter(family, 3, device=device, seed=17)
    yield owner
    owner.close()


def same_rows(a, b):
    assert len(a) == len(b)
    for left, right in zip(a, b, strict=True):
        assert left.keys() == right.keys()
        for key in left:
            assert left[key].dtype == right[key].dtype and left[key].shape == right[key].shape
            np.testing.assert_array_equal(left[key], right[key])


def compare_rejection(owner, bad):
    with pytest.raises((ValueError, TypeError)) as baseline:
        Before.validate(owner, bad)
    with pytest.raises(type(baseline.value)) as candidate:
        owner.validate(bad)
    assert str(candidate.value) == str(baseline.value)


def test_rows_owned_and_one_transfer_per_column(adapter):
    state = adapter.get_state()
    original = state.clone()
    expected = Before.validate(adapter, state)
    calls = []
    cpu = torch.Tensor.cpu

    def count(tensor, *args, **kwargs):
        calls.append(tuple(tensor.shape))
        return cpu(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "cpu", count):
        rows = adapter.validate(state)
    assert len(calls) == len(adapter.fields)
    assert all(shape[0] == 3 for shape in calls)
    same_rows(rows, expected)
    for key in rows[0]:
        assert not np.shares_memory(rows[0][key], rows[1][key])
        rows[0][key].fill(0)
        np.testing.assert_array_equal(rows[1][key], expected[1][key])
    for key in state.keys(True, True):
        assert torch.equal(state[key], original[key])
    rows = adapter.validate(state)
    for value in state.values(True, True):
        value.zero_()
    same_rows(rows, expected)


def test_noncontiguous_and_metadata_batch_override(adapter):
    state = adapter.get_state()
    wide = state.repeat_interleave(2, dim=0)
    noncontiguous = wide[::2]
    assert not noncontiguous["physics", "qpos"].is_contiguous()
    same_rows(adapter.validate(noncontiguous), Before.validate(adapter, noncontiguous))
    same_rows(adapter.validate(wide, _batch_size=6), Before.validate(adapter, wide, _batch_size=6))


def test_every_leaf_structure_dtype_and_nonfinite_rejection(adapter):
    state = adapter.get_state()
    for key in state.keys(True, True):
        bad = state.clone()
        del bad[key]
        compare_rejection(adapter, bad)
        bad = state.clone()
        value = bad[key]
        bad[key] = value.to(torch.float32 if value.dtype != torch.float32 else torch.float64)
        compare_rejection(adapter, bad)
        bad = state.clone()
        bad[key] = bad[key].unsqueeze(-1)
        compare_rejection(adapter, bad)
        if value.is_floating_point() and value.numel():
            for nonfinite in (float("nan"), float("inf"), -float("inf")):
                bad = state.clone()
                bad[key][-1].reshape(-1)[0] = nonfinite
                compare_rejection(adapter, bad)
    bad = state.clone()
    bad["unexpected"] = torch.zeros((3, 1), device=state.device)
    compare_rejection(adapter, bad)
    after = adapter.get_state()
    for key in state.keys(True, True):
        assert torch.equal(state[key], after[key])


def test_semantic_final_lane_rejections(adapter):
    state = adapter.get_state()
    edits = [
        (("physics", "time"), -1), (("task", "id"), 6), (("task", "reward_id"), 6),
        (("physics", "mocap_quat"), 0), (("controller", "site_xmat"), 0),
        (("rng", "action", "increment"), 0), (("rng", "environment", "has_uint32"), 2),
        (("rng", "action", "uinteger"), 2**32), (("rng", "task", "position"), 625),
        (("rng", "task", "key"), 0), (("rng", "task", "has_gauss"), 2),
        (("compatibility", "signature"), 0), (("lifecycle", "reset_next_step"), True),
        (("variation", "current", "cube.size"), .3), (("variation", "anchor", "cube.size"), .3),
        (("model", "compiled", "cube.size"), .3), (("model", "mjcf", "cube.size"), .3),
    ]
    if adapter.family == "Scene":
        edits += [(("scene", "cur_button_states"), 2), (("scene", "prev_button_states"), 2),
                  (("scene", "target_button_states"), 2), (("scene", "target_drawer_pos"), 1),
                  (("scene", "target_window_pos"), 1)]
    for key, value in edits:
        bad = state.clone()
        bad[key][-1] = value
        compare_rejection(adapter, bad)


def test_variation_rng_and_restored_continuation(adapter):
    for options in ({"task_id": 1}, {"task_id": 4, "variation": ["cube.size", "camera.angle_delta"]}):
        state = adapter.reset(42, options)
        state["rng", "environment", "has_uint32"][-1] = 1
        state["rng", "environment", "uinteger"][-1] = 2**32 - 1
        state["rng", "task", "has_gauss"][-1] = True
        state["rng", "task", "cached_gaussian"][-1] = .25
        same_rows(adapter.validate(state), Before.validate(adapter, state))
        adapter.set_state(state)
        action = np.zeros((3, 5))
        expected = adapter.step(action)
        with patch.object(OGBManipulationAdapter, "validate", Before.validate):
            adapter.set_state(state)
            actual = adapter.step(action)
        for key in expected.keys(True, True):
            torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0)
    adapter.reset(17)
