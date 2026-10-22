"""Every class that renders on CPU draws its golden render rows to the recorded frames exactly."""

from __future__ import annotations

import pytest

pytest.importorskip("torch")

import stable_envs as se
from tests.helpers import golden as G

CPU_NAMES = sorted(n for n, cls in se.registry.REGISTRY.items() if not getattr(cls, "requires_cuda", False))


@pytest.mark.parametrize("name", CPU_NAMES)
def test_cpu_frames_match_the_golden_frames(name):
    G.check_class(name, "cpu")
