"""Column-wise Cube/Scene State validation: ownership, transfers and rejection of invalid input."""

from unittest.mock import patch

import numpy as np
import pytest
import torch

pytest.importorskip("ogbench")

from stable_envs.envs.ogbench.simulation.manipulation import OGBManipulationAdapter


@pytest.fixture(scope="module", params=[("Cube", "cpu"), ("Scene", "cpu"), ("Cube", "cuda"), ("Scene", "cuda")])
def adapter(request):
    family, device = request.param
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA requires Oscar")
    owner = OGBManipulationAdapter(family, 3, device=device, seed=17)
    yield owner
    owner.close()


def rejected(owner, bad):
    with pytest.raises((ValueError, TypeError)):
        owner.validate(bad)


def test_rows_are_owned_host_copies_with_one_transfer_per_column(adapter):
    state = adapter.get_state()
    original = state.clone()
    calls = []
    cpu = torch.Tensor.cpu

    def count(tensor, *args, **kwargs):
        calls.append(tuple(tensor.shape))
        return cpu(tensor, *args, **kwargs)

    with patch.object(torch.Tensor, "cpu", count):
        rows = adapter.validate(state)
    assert len(rows) == 3 and len(calls) == len(adapter.fields)
    assert all(shape[0] == 3 for shape in calls)
    for key in rows[0]:
        np.testing.assert_array_equal(rows[0][key], original[key][0].cpu().numpy())
        assert not np.shares_memory(rows[0][key], rows[1][key])
    snapshot = [{key: value.copy() for key, value in row.items()} for row in rows]
    for value in state.values(True, True):
        value.zero_()
    for row, saved in zip(rows, snapshot):
        for key in row:
            np.testing.assert_array_equal(row[key], saved[key])


def test_noncontiguous_input_and_batch_override(adapter):
    state = adapter.get_state()
    wide = state.repeat_interleave(2, dim=0)
    noncontiguous = wide[::2]
    assert not noncontiguous["physics", "qpos"].is_contiguous()
    rows = adapter.validate(noncontiguous)
    for key, value in state.items(True, True):
        np.testing.assert_array_equal(np.stack([row[key] for row in rows]), value.cpu().numpy())
    assert len(adapter.validate(wide, _batch_size=6)) == 6


def test_every_leaf_structure_dtype_shape_and_nonfinite_value_is_rejected(adapter):
    state = adapter.get_state()
    for key in state.keys(True, True):
        bad = state.clone()
        del bad[key]
        rejected(adapter, bad)
        bad = state.clone()
        value = bad[key]
        bad[key] = value.to(torch.float32 if value.dtype != torch.float32 else torch.float64)
        rejected(adapter, bad)
        bad = state.clone()
        bad[key] = bad[key].unsqueeze(-1)
        rejected(adapter, bad)
        if value.is_floating_point() and value.numel():
            for nonfinite in (float("nan"), float("inf"), -float("inf")):
                bad = state.clone()
                bad[key][-1].reshape(-1)[0] = nonfinite
                rejected(adapter, bad)
    bad = state.clone()
    bad["unexpected"] = torch.zeros((3, 1), device=state.device)
    rejected(adapter, bad)


def test_out_of_range_values_in_the_final_lane_are_rejected(adapter):
    state = adapter.get_state()
    edits = [
        (("physics", "time"), -1),
        (("task", "id"), 6),
        (("task", "reward_id"), 6),
        (("physics", "mocap_quat"), 0),
        (("controller", "site_xmat"), 0),
        (("rng", "action", "increment"), 0),
        (("rng", "environment", "has_uint32"), 2),
        (("rng", "action", "uinteger"), 2**32),
        (("rng", "task", "position"), 625),
        (("rng", "task", "key"), 0),
        (("rng", "task", "has_gauss"), 2),
        (("compatibility", "signature"), 0),
        (("lifecycle", "reset_next_step"), True),
        (("variation", "current", "cube.size"), 0.3),
        (("variation", "anchor", "cube.size"), 0.3),
        (("model", "compiled", "cube.size"), 0.3),
        (("model", "mjcf", "cube.size"), 0.3),
    ]
    if adapter.family == "Scene":
        edits += [
            (("scene", "cur_button_states"), 2),
            (("scene", "prev_button_states"), 2),
            (("scene", "target_button_states"), 2),
            (("scene", "target_drawer_pos"), 1),
            (("scene", "target_window_pos"), 1),
        ]
    for key, value in edits:
        bad = state.clone()
        bad[key][-1] = value
        rejected(adapter, bad)
