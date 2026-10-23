"""PushT frames drawn on CUDA from device State tensors equal the pygame frames byte for byte."""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.envs.pusht import raster

pytestmark = pytest.mark.skipif(
    not torch.cuda.is_available() or not raster.HAVE_TRITON, reason="the device rasterizer needs CUDA and Triton"
)

NAMES = ("PushT", "PushTDiscrete")
LANES = 64


@pytest.fixture(autouse=True)
def _in_process():
    # Worker processes do not change frames; in-process scenes keep the file fast.
    with se.tuning.override(workers=0):
        yield


def _actions(name, rng, steps):
    if name == "PushTDiscrete":
        return torch.as_tensor(rng.integers(0, 16, (LANES, steps, 2), dtype=np.int64))
    return torch.as_tensor(rng.uniform(-1.0, 1.0, (LANES, steps, 2)))


@functools.cache
def _seeded(name):
    with se.make(name, num_envs=LANES, size=224, device="cpu") as env:
        env.reset(seed=11)
        rng = np.random.default_rng(5)
        for _ in range(4):
            env.step(_actions(name, rng, 8))
        return env.get_state().clone()


def _states(name):
    """64 seeded States after random pushes, then edge cases written into the first lanes."""
    state = _seeded(name).clone()
    block = state["current", "body", "block", "position"]
    agent = state["current", "body", "agent", "position"]
    angle = state["current", "body", "block", "angle"]
    # Block partly above the canvas, upside down: its outline lines are clipped.
    block[0] = torch.tensor([300.0, 20.0], dtype=block.dtype)
    angle[0] = math.pi + 0.05
    # Block across the left and bottom edges.
    block[1] = torch.tensor([-10.0, 505.0], dtype=block.dtype)
    angle[1] = 0.7
    # Agent off the canvas.
    agent[2] = torch.tensor([-60.0, 600.0], dtype=agent.dtype)
    # Agent over the block.
    agent[3] = block[3].clone()
    # Goal not drawn.
    state["owned", "variation", "rendering", "render_goal", "current", "value"][4] = 0
    return state


def _counting(env):
    rows = []
    draw = env._draw_pygame

    def counted(fields, size):
        rows.append(len(fields["recipe/config"]))
        return draw(fields, size)

    env._draw_pygame = counted
    return rows


@pytest.mark.parametrize("name", NAMES)
def test_device_frames_equal_the_pygame_frames(name):
    state = _states(name)
    assert int((state["current", "render_segment_count"] > 0).sum()) > 0, "no State has contact lines"
    with (
        se.make(name, num_envs=LANES, size=224, device="cuda") as env,
        se.make(name, num_envs=LANES, size=224, device="cpu") as ref,
    ):
        fallback = _counting(env)
        for size in (224, 96):
            got = env.render(state.to("cuda"), size=size).cpu()
            want = ref.render(state, size=size)
            bad = torch.nonzero((got != want).flatten(1).any(1)).view(-1).tolist()
            assert not bad, f"{name} size {size}: lanes {bad} differ from pygame"
        assert env._device_renderer is not None, "render did not use the device renderer"
        assert sum(fallback) == 0, f"{name}: {sum(fallback)} rows used pygame"


@pytest.mark.parametrize("name", NAMES)
def test_step_frames_equal_the_pygame_frames(name):
    rng = np.random.default_rng(9)
    actions = _actions(name, rng, 8)
    with (
        se.make(name, num_envs=LANES, size=224, device="cuda") as env,
        se.make(name, num_envs=LANES, size=224, device="cpu") as ref,
    ):
        env.reset(seed=4)
        ref.reset(seed=4)
        got = env.step(actions.to("cuda"), frames=slice(None))["frames"].cpu()
        want = ref.step(actions, frames=slice(None))["frames"]
        assert torch.equal(got, want), f"{name}: step frames differ from pygame"


def test_action_markers_use_pygame():
    with (
        se.make("PushT", num_envs=8, size=96, device="cuda", render_action=True) as env,
        se.make("PushT", num_envs=8, size=96, device="cpu", render_action=True) as ref,
    ):
        env.reset(seed=2)
        ref.reset(seed=2)
        actions = _actions("PushT", np.random.default_rng(1), 2)[:8]
        env.step(actions.to("cuda"))
        ref.step(actions)
        state = env.get_state()
        fallback = _counting(env)
        got = env.render(state).cpu()
        assert sum(fallback) == 8, "frames with action markers must be drawn by pygame"
        assert torch.equal(got, ref.render(state.to("cpu")))
