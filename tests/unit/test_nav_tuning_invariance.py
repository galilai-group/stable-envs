"""Navigation cells give the same frames for every render chunk size."""

from __future__ import annotations

import pytest

pytest.importorskip("matplotlib")
torch = pytest.importorskip("torch")

import stable_envs as snv
from stable_envs import tuning

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", ["TwoRoom", "SimplePointMaze"])
def test_render_chunk_keeps_frames(name):
    with snv.make(name, num_envs=7, device="cpu") as env:
        state = env.reset(seed=4)
        reference = env.render(state)
        for chunk in (1, 2, 3, 7, 256):
            with tuning.override(nav_render_chunk=chunk):
                assert torch.equal(env.render(state), reference), chunk
