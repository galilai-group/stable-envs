"""The Numba navigation dot compositor draws the Torch path's CPU frames byte for byte."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.render import cpu_raster

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not cpu_raster.available(), reason="the CPU compositor needs numba"),
]

CONFIGS = ("TwoRoom", "TwoRoomTarget", "Piecewise", "PiecewiseZones")


def _pair(cfg, B):
    return se.make(cfg, num_envs=B, device="cpu"), se.make(cfg, num_envs=B, device="cpu", backend="torch")


def _differ(fast, slow):
    return (fast != slow).flatten(1).any(1).nonzero().flatten().tolist()


@pytest.mark.parametrize("cfg", CONFIGS)
@pytest.mark.parametrize("B", [1, 7, 300])
def test_sampled_states(cfg, B):
    fast, slow = _pair(cfg, B)
    assert fast._numba_dots() and not slow._numba_dots()
    rows = torch.as_tensor(np.asarray(fast.sample_states(B, np.random.default_rng(B)), np.float32))
    assert not _differ(fast._render(rows), slow._render(rows)), cfg


@pytest.mark.parametrize("cfg", CONFIGS)
def test_states_the_env_would_never_sample(cfg):
    """Dots off the frame, on and between pixel centers, and stacked on each other."""
    pos = [
        (112.0, 112.0),
        (112.5, 112.5),
        (0.0, 0.0),
        (223.0, 223.0),
        (223.5, -0.5),
        (-60.0, 112.0),
        (112.0, -60.0),
        (280.0, 280.0),
        (-3000.0, 4000.0),
        (1e9, -1e9),
    ]
    fast, slow = _pair(cfg, len(pos))
    rows = np.zeros((len(pos), fast.state_dim), np.float32)
    for i, (x, y) in enumerate(pos):
        rows[i, 0], rows[i, 1] = x, y
        if fast.state_dim >= 4:
            rows[i, 2], rows[i, 3] = (x, y) if i % 2 else (223.0 - x, 223.0 - y)
    rows = torch.as_tensor(rows)
    assert not _differ(fast._render(rows), slow._render(rows)), cfg


@pytest.mark.parametrize("cfg", CONFIGS)
def test_per_row_styles(cfg):
    """Per-row colors, radii and backgrounds; a few rows share a style so both base paths draw."""
    B = 12
    fast, slow = _pair(cfg, B)
    rows = np.tile(np.asarray(fast.default_style(), np.float32), (B, 1))
    lay = {p: (o, w) for p, o, w in fast.variation_space.flat_layout()}
    rng = np.random.default_rng(6)
    for path in ("agent.color", "agent.radius", "target.color", "target.radius", "background.color"):
        if path in lay:
            o, w = lay[path]
            values = rng.uniform(0, 255, (B, w)) if path.endswith("color") else rng.uniform(7.0, 14.0, (B, w))
            rows[: B // 2, o : o + w] = values[: B // 2]
    states = torch.as_tensor(np.asarray(fast.sample_states(B, np.random.default_rng(1)), np.float32))
    got = fast._render(states, style=rows)
    assert not _differ(got, slow._render(states, style=rows)), cfg
    assert not _differ(fast._render(states, style=rows[:1]), slow._render(states, style=rows[:1])), cfg


@pytest.mark.parametrize("flags", [("render_target",), ("render_bias_field",), ("render_target", "render_bias_field")])
def test_piecewise_layers(flags):
    pytest.importorskip("matplotlib")
    frames = []
    for backend in ("auto", "torch"):
        with se.make("Piecewise", num_envs=5, device="cpu", backend=backend) as env:
            env.set_variation({f"rendering.{f}": 1 for f in flags})
            frames.append(env.render(env.reset(seed=3)))
    assert not _differ(*frames)


@pytest.mark.parametrize("cfg", CONFIGS)
def test_steps_with_frames(cfg):
    """Frames returned by ``step`` match, so the online path draws through the same compositor."""
    frames = []
    for backend in ("auto", "torch"):
        with se.make(cfg, num_envs=16, device="cpu", backend=backend) as env:
            env.reset(seed=2)
            actions = torch.as_tensor(np.random.default_rng(0).uniform(-1, 1, (16, 6, 2)), dtype=torch.float32)
            frames.append(env.step(actions, states=None, frames=slice(None))["frames"])
    assert torch.equal(*frames), cfg


def test_render_writes_into_out():
    with se.make("TwoRoomTarget", num_envs=4, device="cpu") as env:
        state = env.reset(seed=0)
        out = torch.zeros((4, 224, 224, 3), dtype=torch.uint8)
        got = env.render(state, out=out)
        assert got.data_ptr() == out.data_ptr()
        assert torch.equal(out, env.render(state))
