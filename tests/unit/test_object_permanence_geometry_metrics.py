"""Geometry metrics retain small-object errors and distinguish missing/empty data."""

import numpy as np
import pytest

from stable_envs.testing.geometry import depth_mse, mask_iou


def test_mask_iou_does_not_count_hidden_frames_as_visible_agreement():
    reference = np.zeros((3, 4, 4), bool)
    reference[0, 1, 1:3] = True
    actual = reference.copy()
    actual[0, 1, 2] = False
    actual[2, 1, 1] = True  # erroneous appearance while the reference is hidden
    values = mask_iou(actual, reference)
    assert values[0] == 0.5 and np.isnan(values[1]) and values[2] == 0


def test_depth_roi_does_not_dilute_errors_with_background_and_rejects_missing_surfaces():
    reference = np.full((3, 100, 100), 8.0)
    actual = reference.copy()
    valid = np.zeros(reference.shape, bool)
    valid[:, 1, 1] = True
    actual[0, 1, 1] += 0.01
    actual[1, 1, 1] = 0
    valid[2] = False
    values = depth_mse(actual, reference, valid)
    assert values[0] == pytest.approx(1e-4)
    assert np.isinf(values[1]) and np.isnan(values[2])
    actual[0, 1, 1] = np.nan
    with pytest.raises(ValueError, match="finite"):
        depth_mse(actual, reference, valid)
