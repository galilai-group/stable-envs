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
#: The contact line colour of pymunk's debug draw.
LINE = (231.0, 76.0, 60.0, 255.0)


@pytest.fixture(autouse=True)
def _in_process():
    # Worker processes do not change frames; in-process pygame keeps the file fast.
    with se.tuning.override(workers=0):
        yield


def _actions(name, rng, steps, lanes=LANES):
    if name == "PushTDiscrete":
        return torch.as_tensor(rng.integers(0, 16, (lanes, steps, 2), dtype=np.int64))
    return torch.as_tensor(rng.uniform(-1.0, 1.0, (lanes, steps, 2)))


@functools.cache
def _seeded(name):
    with se.make(name, num_envs=LANES, size=224, device="cpu") as env:
        env.reset(seed=13)
        rng = np.random.default_rng(7)
        for _ in range(4):
            env.step(_actions(name, rng, 8))
        return env.get_state().clone()


@functools.cache
def _sampled(name):
    """States after random pushes, each with its own sampled Variation: shapes, scales, colours and goal."""
    with se.make(name, num_envs=LANES, size=224, device="cpu") as env:
        rng = np.random.default_rng(17)
        env.set_variation(env.sample_variation(rng=rng))
        env.reset(seed=19)
        for _ in range(3):
            env.step(_actions(name, rng, 6), frames=None)
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
    # The agent far from the block, so their redrawn boxes stay apart.
    agent[7] = torch.tensor([40.0, 470.0], dtype=agent.dtype)
    block[7] = torch.tensor([440.0, 60.0], dtype=block.dtype)
    return state


def _with_lines(state, seed, lanes):
    """Write contact lines into ``lanes``: some clipped by the canvas, some steep, one of zero length per lane."""
    rng = np.random.default_rng(seed)
    segments = state["current", "render_segments"]
    counts = state["current", "render_segment_count"]
    for lane in lanes:
        k = int(rng.integers(1, 6))
        ends = rng.uniform(-40.0, 552.0, (k, 4))
        ends[0, 2:] = ends[0, :2]
        colors = np.tile(LINE, (k, 1))
        colors[1:, :3] = rng.integers(0, 256, (k - 1, 3))
        segments[lane] = 0
        segments[lane, :k] = torch.as_tensor(np.concatenate([ends, colors], 1), dtype=segments.dtype)
        counts[lane] = k
    return state


def _all_states(name):
    return {
        "seeded": _with_lines(_states(name), 1, range(8, 16)),
        "sampled": _with_lines(_sampled(name).clone(), 2, range(0, LANES, 3)),
    }


def _pygame(name, state, size, **config):
    with se.make(name, num_envs=LANES, size=size, device="cpu", backend="source", **config) as source:
        return source.render(state)


def _differ(got, want):
    return torch.nonzero((got != want).flatten(1).any(1)).view(-1).tolist()


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


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("kind", ["seeded", "sampled"])
def test_varied_frames_and_contact_lines_equal_the_pygame_frames(name, kind):
    state = _all_states(name)[kind]
    assert int((state["current", "render_segment_count"] > 0).sum()) >= 8
    want = _pygame(name, state, 224)
    with se.make(name, num_envs=LANES, size=224, device="cpu") as env:
        # The second render copies the goal-and-wall frames that the first one kept.
        for call in range(2):
            got = env.render(state)
            assert not _differ(got, want), f"{name} {kind} call {call}: lanes {_differ(got, want)} differ"


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("threads", [1, 3])
def test_small_batches_equal_the_pygame_frames(name, threads):
    import numba

    before = numba.get_num_threads()
    numba.set_num_threads(min(threads, numba.config.NUMBA_NUM_THREADS))
    try:
        with se.make(name, num_envs=LANES, size=224, device="cpu") as env:
            for kind, state in _all_states(name).items():
                want = _pygame(name, state, 224)
                # Repeated batches draw whole frames, then keep their goal-and-wall frames and copy them.
                for lanes in ([0], [0], [0], [5, 6], [5, 6], [7, 9, 11], list(range(8, 16)), list(range(8, 16))):
                    got = env.render(state[torch.as_tensor(lanes)])
                    bad = _differ(got, want[lanes])
                    assert not bad, f"{name} {kind} lanes {lanes}: {bad} differ ({threads} threads)"
    finally:
        numba.set_num_threads(before)


def _one_pass(fields, size):
    """Frames of the renderer that builds the records of a whole batch first, then draws every frame."""
    from stable_envs.envs.pusht import raster, raster_cpu

    records, _, unsupported = raster.primitives(fields)
    background = raster.pack(fields["owned/variation/background/color/current/value"].astype(np.int64))
    frames = np.zeros((len(records), size, size, 3), np.uint8)
    raster_cpu.draw(records, background, raster.contact_writes(fields), frames)
    return torch.from_numpy(frames), unsupported


@pytest.mark.parametrize("name", NAMES)
@pytest.mark.parametrize("size", [224, 96])
def test_frames_equal_the_one_pass_renderer(name, size):
    from stable_envs.envs.pusht import raster

    with se.make(name, num_envs=LANES, size=size, device="cpu") as env:
        for kind, state in _all_states(name).items():
            fields = env._render_fields_of(state)
            want, unsupported = _one_pass(fields, size)
            got = raster.render_cpu(fields, size, lambda rows: np.zeros((len(rows["recipe/config"]), size, size, 3)))
            keep = torch.as_tensor(~unsupported)
            assert bool(keep.any())
            bad = _differ(got[keep], want[keep])
            assert not bad, f"{name} {kind} at {size}: drawn lanes {bad} differ"


@pytest.mark.parametrize("name", NAMES)
def test_action_markers_equal_the_pygame_frames(name):
    rng = np.random.default_rng(3)
    with (
        se.make(name, num_envs=8, size=96, device="cpu", render_action=True) as env,
        se.make(name, num_envs=8, size=96, device="cpu", render_action=True, backend="source") as source,
    ):
        env.reset(seed=2)
        states = [env.get_state()]
        env.step(_actions(name, rng, 3, lanes=8), frames=None)
        states.append(env.get_state())
        assert bool(states[1]["current", "latest_present"].all())
        for state in states:
            for size in (96, 224):
                got, want = env.render(state, size=size), source.render(state, size=size)
                assert not _differ(got, want), f"{name} at {size}: lanes {_differ(got, want)} differ"


@pytest.mark.parametrize("name", NAMES)
def test_step_frames_equal_the_pygame_frames(name):
    rng = np.random.default_rng(5)
    with (
        se.make(name, num_envs=16, size=224, device="cpu") as env,
        se.make(name, num_envs=16, size=224, device="cpu", backend="source") as source,
    ):
        env.reset(seed=4)
        rollout = env.step(_actions(name, rng, 4, lanes=16), states=slice(None), frames=slice(None))
        want = torch.stack([source.render(rollout["states"][:, t]) for t in range(4)], 1)
        assert torch.equal(rollout["frames"], want), f"{name}: step frames differ from pygame"


def test_auto_uses_the_numba_renderer_on_cpu():
    with se.make("PushT", num_envs=2, size=224, device="cpu") as env:
        assert env._cpu_port(224)
    with se.make("PushT", num_envs=2, size=224, device="cpu", backend="source") as env:
        assert not env._cpu_port(224)
