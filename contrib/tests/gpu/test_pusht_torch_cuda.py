"""contrib.pusht_torch on CUDA: frames equal the public PushT render at its default size."""

from __future__ import annotations

import platform

import pytest
import torch

from contrib.tests.unit.test_pusht_torch import _frames_match
from contrib.tests.unit.test_pusht_torch import pytestmark as _requirements

pytestmark = [
    _requirements,
    pytest.mark.skipif(not torch.cuda.is_available(), reason="needs CUDA"),
    pytest.mark.skipif(
        platform.system() != "Linux" or platform.machine() != "x86_64",
        reason="OpenCV's generic resize path is the reference on Linux x86-64",
    ),
]


@pytest.mark.parametrize("size", [224, 96])
def test_cuda_frames_equal_the_public_render(size):
    _frames_match("cuda", size, size)
