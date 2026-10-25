"""Small rigid-transform arithmetic preserves broadcasting and numeric meaning."""

import numpy as np
import pytest
import torch

from stable_envs.envs.object_permanence.rendering.transforms import row_matrix3


@pytest.mark.parametrize("dtype", [torch.float32, torch.float64])
@pytest.mark.parametrize("batch", [1, 3])
def test_three_component_rows_match_independent_numpy_product_with_broadcasting(dtype, batch):
    rng = np.random.default_rng(122)
    rows = torch.tensor(rng.normal(size=(11, 3)), dtype=dtype)
    # Noncontiguous matrices are valid and must not alter the operation's meaning.
    matrices = torch.tensor(rng.normal(size=(batch, 11, 3, 3)), dtype=dtype).transpose(-1, -2)
    actual = row_matrix3(rows, matrices)
    expected = np.einsum("fi,bfij->bfj", rows.numpy().astype(np.float64), matrices.numpy().astype(np.float64))
    assert actual.shape == (batch, 11, 3) and actual.dtype == dtype
    np.testing.assert_allclose(actual.numpy(), expected, atol=2e-6 if dtype == torch.float32 else 2e-15, rtol=0)
    single = row_matrix3(rows[0], matrices[0, 0])
    assert single.shape == (3,)
    torch.testing.assert_close(single, actual[0, 0], atol=0, rtol=0)
