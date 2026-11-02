"""Every class renders its golden render rows on CUDA to the recorded frames exactly."""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

import stable_envs as se
from tests.helpers import golden as G


@pytest.mark.parametrize("name", sorted(se.registry.REGISTRY))
def test_cuda_frames_match_the_golden_frames(name):
    G.check_class(name, "cuda")
