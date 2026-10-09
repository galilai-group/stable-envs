"""The tile-atlas case (``MiniGridEmpty``) of ``tests/gpu/test_distributed.py``, run through the same test body.

Only the two-devices-in-one-process check: the DDP check spawns fresh interpreters that build the env by
name, and those children do not load ``contrib``.
"""

from __future__ import annotations

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.gpu

if not torch.cuda.is_available():
    pytest.skip("stable-envs GPU tests require a CUDA device", allow_module_level=True)

import test_distributed as _main


@pytest.mark.parametrize("name", ["MiniGridEmpty"])
def test_two_devices_in_one_process(name):
    _main.test_two_devices_in_one_process(name)
