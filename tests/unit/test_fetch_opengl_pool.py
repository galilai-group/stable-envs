"""Fetch OpenGL render workers return the frames the main process draws, in row order."""

from __future__ import annotations

import sys

import numpy as np
import pytest
import torch

pytest.importorskip("gymnasium_robotics")

import stable_envs as snv
from stable_envs import tuning

pytestmark = pytest.mark.unit


@pytest.mark.parametrize("name", ["FetchPickAndPlace", "FetchReach"])
def test_worker_frames_equal_in_process_frames(name):
    if not sys.platform.startswith("linux"):
        pytest.skip("Render workers run on Linux only")
    from stable_envs._execution import render_pool

    with snv.make(name, num_envs=2, size=64, device="cpu", seed=17) as env:
        rng = np.random.default_rng(5)
        rows = torch.as_tensor(env.sample_states(24, rng=rng))
        style = torch.as_tensor(env.sample_style(24, rng=rng), dtype=env.get_style().dtype)
        style[::3] = env.get_style()[0]
        with tuning.override(render_workers=0):
            alone = env._render(rows, style=style)
        with tuning.override(render_workers=3, render_worker_min_rows=2):
            if render_pool.worker_count(len(rows), "fetch-opengl", 64) == 0:
                pytest.skip("Fewer than 2 CPUs in the process affinity")
            shared = env._render(rows, style=style)
            again = env._render(rows[3:19], style=style[3:19])
            out = torch.empty_like(alone)
            into = env._render(rows, style=style, out=out)
        assert shared.dtype == torch.uint8 and shared.shape == alone.shape
        assert torch.equal(shared, alone)
        assert torch.equal(again, alone[3:19])
        assert into is out and torch.equal(out, alone)
