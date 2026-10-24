"""Render worker processes give the frames that one process draws, and stay off where they must."""

import os
import sys

import numpy as np
import pytest
import torch

import stable_envs as snv
from stable_envs import tuning
from stable_envs._execution import render_pool

pytestmark = pytest.mark.integration


def test_worker_count_follows_knobs_and_context(monkeypatch):
    if not sys.platform.startswith("linux"):
        assert render_pool.worker_count(4096, "classic-render", 600) == 0
        return
    with tuning.override(render_worker_min_rows=32, render_workers=2):
        assert render_pool.worker_count(31, "classic-render", 600) == 0
        expected = 2 if len(os.sched_getaffinity(0)) >= 2 else 0
        assert render_pool.worker_count(32, "classic-render", 600) == expected
    with tuning.override(render_workers=1):
        assert render_pool.worker_count(4096, "classic-render", 600) == 0
    monkeypatch.setattr(render_pool, "IN_WORKER", True)
    assert render_pool.worker_count(4096, "classic-render", 600) == 0


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="render workers run on Linux only")
def test_workers_draw_the_in_process_frames():
    env = snv.make("CartPole", num_envs=1, device="cpu")
    rows = torch.as_tensor(env.sample_states(37, np.random.default_rng(3)))
    style = env.resolve_style(batch=len(rows)).values.numpy()
    pool = render_pool.RenderPool(render_pool.EnvRows("CartPole", size=env.size), 3, name="test render")
    try:
        frames = pool.render(len(rows), (*env.native, 3), lambda part: (rows[part].numpy(), style[part], {}))
    finally:
        pool.close()
    with tuning.override(render_workers=0):
        expected = env._render(rows).numpy()
    env.close()
    assert np.array_equal(frames, expected)


@pytest.mark.skipif(not sys.platform.startswith("linux"), reason="render workers run on Linux only")
@pytest.mark.parametrize("name", ["Acrobot", "CartPole", "MountainCar", "MountainCarContinuous", "Pendulum"])
def test_source_workers_draw_the_port_frames(name):
    pytest.importorskip("pygame")
    env = snv.make(name, num_envs=1, device="cpu")
    rows = torch.as_tensor(env.sample_states(41, np.random.default_rng(5)))
    try:
        with tuning.override(render_workers=3, render_worker_min_rows=2):
            pooled = env._render(rows).numpy()
        with tuning.override(render_workers=0):
            single = env._render(rows[:1]).numpy()
            expected = env._render(rows, backend="port").numpy()
    finally:
        env.close()
    assert np.array_equal(pooled, expected)
    assert np.array_equal(single, expected[:1])
