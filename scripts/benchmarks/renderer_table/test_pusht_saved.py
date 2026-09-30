"""Regression checks for exhaustive, bounded-memory post-timing image checks."""

import numpy as np
import pytest
import torch

from scripts.benchmarks.renderer_table.pusht_saved import check_outputs, mse


def test_chunked_mse_preserves_full_frame_reductions_and_tail():
    rng = np.random.default_rng(17)
    a, b = [rng.integers(0, 256, (65, 9, 11, 3), dtype=np.uint8) for _ in range(2)]
    expected = np.mean(((a.astype(np.float64) - b.astype(np.float64)) / 255) ** 2, axis=(1, 2, 3))
    np.testing.assert_array_equal(mse(a, b), expected)


def test_output_checks_preserve_cycling_and_detect_last_pixel_error():
    rng = np.random.default_rng(42)
    source = rng.integers(0, 256, (65, 9, 11, 3), dtype=np.uint8)
    ours = source.copy()
    ours[:, 0, 0, 0] ^= 1
    indices = np.arange(128) % 65
    images = [torch.from_numpy(ours[ids]) for ids in indices.reshape(-1, 64)]
    errors = mse(ours, source)
    np.testing.assert_array_equal(check_outputs(images, indices, source, errors, "ours"), errors[indices])
    images = [torch.from_numpy(source[ids].copy()) for ids in indices.reshape(-1, 64)]
    assert check_outputs(images, indices, source, errors, "source") is None
    images[-1][-1, -1, -1, -1] ^= 1
    with pytest.raises(AssertionError):
        check_outputs(images, indices, source, errors, "source")
