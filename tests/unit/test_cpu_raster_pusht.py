"""The Numba PushT renderer draws the pygame frames byte for byte on CPU."""

from __future__ import annotations

import functools
import math

import numpy as np
import pytest
import torch

import stable_envs as se
from stable_envs.render import cpu_raster

pytestmark = [
    pytest.mark.unit,
    pytest.mark.skipif(not cpu_raster.available(), reason="the CPU rasterizer needs numba"),
]

NAMES = ("PushT", "PushTDiscrete")
LANES = 48


@pytest.fixture(autouse=True)
def _in_process():
    # Worker processes do not change frames; in-process pygame keeps the file fast.
    with se.tuning.override(workers=0):
        yield


def _actions(name, rng, steps):
    if name == "PushTDiscrete":
        return torch.as_tensor(rng.integers(0, 16, (LANES, steps, 2), dtype=np.int64))
    return torch.as_tensor(rng.uniform(-1.0, 1.0, (LANES, steps, 2)))


@functools.cache
def _seeded(name):
    with se.make(name, num_envs=LANES, size=224, device="cpu") as env:
        env.reset(seed=13)
        rng = np.random.default_rng(7)
        for _ in range(4):
            env.step(_actions(name, rng, 8))
        return env.get_state().clone()


def _states(name):
    """Seeded States after random pushes, with edge cases written into the first lanes."""
    state = _seeded(name).clone()
    block = state["current", "body", "block", "position"]
    agent = state["current", "body", "agent", "position"]
    angle = state["current", "body", "block", "angle"]
    block[0] = torch.tensor([300.0, 20.0], dtype=block.dtype)
    angle[0] = math.pi + 0.05
    block[1] = torch.tensor([-10.0, 505.0], dtype=block.dtype)
    angle[1] = 0.7
    agent[2] = torch.tensor([-60.0, 600.0], dtype=agent.dtype)
    agent[3] = block[3].clone()
    state["owned", "variation", "rendering", "render_goal", "current", "value"][4] = 0
    angle[5] = math.pi / 2
    angle[6] = -math.pi
    return state


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("size", [224, 96])
def test_the_numba_frames_equal_the_pygame_frames(name, size):
    state = _states(name)
    with se.make(name, num_envs=LANES, size=size, device="cpu") as numba:
        fast = numba.render(state)
    with se.make(name, num_envs=LANES, size=size, device="cpu", backend="source") as source:
        slow = source.render(state)
    differ = (fast != slow).flatten(1).any(1)
    assert not bool(differ.any()), f"{name} at {size}: frames {differ.nonzero().flatten().tolist()} differ"


def test_auto_uses_the_numba_renderer_on_cpu():
    with se.make("PushT", num_envs=2, size=224, device="cpu") as env:
        assert env._cpu_port(224)
    with se.make("PushT", num_envs=2, size=224, device="cpu", backend="source") as env:
        assert not env._cpu_port(224)
