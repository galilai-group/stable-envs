"""With the default backend, row 0 of a seeded CUDA rollout is the same bytes at every batch size."""

import pytest
import torch

from tests.helpers.batch_invariance import assert_batch_invariant

pytestmark = [pytest.mark.family("dmc"), pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")]


def test_dmc_row_0_is_the_same_at_every_batch_size():
    assert_batch_invariant("DMCReacher")
