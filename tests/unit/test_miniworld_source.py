"""MiniWorldMovingBlockWorld matches the fork's ``MovingBlockWorld`` driven directly, lane by lane.

The source gate of the family. The source envs are built as the fork's ``scripts/generate_videos.py``
builds them (``build_env`` and the heading handling of ``run_rollout``), or with the registered defaults,
and compared on every step: the agent, each block, the step count, reward, both done flags and the frame
``env.render()`` returns.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from stable_envs.envs.miniworld.simulation.adapter import configure_pyglet

configure_pyglet()  # before miniworld imports pyglet.gl
pytest.importorskip("miniworld.envs.moving_blockworld")

from miniworld.entity import Box  # noqa: E402
from miniworld.envs import MovingBlockWorld  # noqa: E402
from miniworld.params import DEFAULT_PARAMS  # noqa: E402

import stable_envs as snv  # noqa: E402

pytestmark = pytest.mark.unit

LANES = 2
SEED = 11
SIZE = 48
#: Source envs, kept for the process: each holds a GL context.
_SOURCES: dict = {}


def _flowm_source(lane: int):
    """``generate_videos.build_env`` for the README's textured training command (8 blocks)."""
    key = ("flowm", lane)
    if key not in _SOURCES:
        params = DEFAULT_PARAMS.no_random()
        params.set("turn_step", 90.0, 90.0, 90.0)
        params.set("forward_step", 1.0, 1.0, 1.0)
        params.set("cam_fov_y", 60.0, 60.0, 60.0)
        env = MovingBlockWorld(
            view="agent",
            render_mode="rgb_array",
            obs_width=SIZE,
            obs_height=SIZE,
            window_width=SIZE,
            window_height=SIZE,
            size=16,
            box_tex_randomize=True,
            wall_tex_randomize=True,
            floor_tex_randomize=True,
            box_and_ball=True,
            box_allow_overlap=True,
            agent_box_allow_overlap=True,
            grid_mode=True,
            grid_vel_min=-1,
            grid_vel_max=1,
            grid_cardinal_only=True,
            num_blocks=8,
            ensure_base_palette=True,
            block_size_xy=0.7,
            block_height=1.5,
            params=params,
        )
        _SOURCES[key] = env
    env = _SOURCES[key]
    env.max_episode_steps = 500
    return env


def _upstream_source(lane: int, max_episode_steps: int = 250):
    key = ("upstream", lane)
    if key not in _SOURCES:
        _SOURCES[key] = MovingBlockWorld(render_mode="rgb_array", window_width=SIZE, window_height=SIZE)
    env = _SOURCES[key]
    env.max_episode_steps = max_episode_steps
    return env


def _source_frame(src):
    """``src.render()`` with its own room: every env compiles its room into GL display list 1, shared by all
    contexts of the process, so the list is recompiled first as a lone env in its process would hold it."""
    src.shadow_window.switch_to()
    src._render_static()
    return src.render()


def _wrap(env):
    env.agent.dir = float(env.agent.dir % (2.0 * math.pi))


def _assert_lane(state, lane, src, t):
    msg = f"lane {lane}, step {t}"
    np.testing.assert_array_equal(state["agent", "pos"][lane].numpy(), src.agent.pos, err_msg=msg)
    assert state["agent", "dir"][lane].item() == src.agent.dir, msg
    blocks = src.entities[1:]
    np.testing.assert_array_equal(
        state["blocks", "pos"][lane].numpy().reshape(-1, 3), np.stack([b.pos for b in blocks]), err_msg=msg
    )
    np.testing.assert_array_equal(
        state["blocks", "velocity"][lane].numpy().reshape(-1, 3), np.stack([b.velocity for b in blocks]), err_msg=msg
    )
    assert state["blocks", "dir"][lane].tolist() == [b.dir for b in blocks], msg
    assert state["blocks", "kind"][lane].tolist() == [0 if isinstance(b, Box) else 1 for b in blocks], msg
    carried = src.agent.carrying
    assert state["agent", "carrying"][lane].item() == (-1 if carried is None else blocks.index(carried)), msg
    assert state["lifecycle", "elapsed_steps"][lane].item() == src.step_count, msg


def _run(env, sources, actions, *, wrap):
    state = env.get_state()
    for lane, src in enumerate(sources):
        _assert_lane(state, lane, src, 0)
    np.testing.assert_array_equal(env.render().numpy(), np.stack([_source_frame(src) for src in sources]))
    for t, action in enumerate(actions, start=1):
        result = env.step(action[:, None])
        for lane, src in enumerate(sources):
            _, reward, terminated, truncated, _ = src.step(int(action[lane]))
            if wrap:
                _wrap(src)
            _assert_lane(result["state"], lane, src, t)
            assert result["reward"][lane].item() == reward
            assert result["terminated"][lane].item() == terminated
            assert result["truncated"][lane].item() == truncated
            np.testing.assert_array_equal(result["frames"][lane, -1].numpy(), _source_frame(src), err_msg=f"step {t}")
    return result


def test_lanes_match_the_source_called_directly():
    """The FloWM preset: grid motion, textures, boxes and balls; every action id, pick up and drop included."""
    with snv.MiniWorldMovingBlockWorld(num_envs=LANES, size=SIZE, device="cpu") as env:
        env.reset(seed=SEED)
        sources = [_flowm_source(lane) for lane in range(LANES)]
        for lane, src in enumerate(sources):
            src.reset(seed=SEED + lane)
            src.agent.dir = 0.0  # --heading-zero
            _wrap(src)
        rng = np.random.default_rng(SEED)
        actions = rng.integers(0, 9, size=(24, LANES))
        actions[:4] = [2, 5]  # walk, and try to pick up a block
        _run(env, sources, actions, wrap=True)


def test_the_upstream_preset_matches_the_registered_env():
    """``preset="upstream"``: continuous motion in a 12 x 12 room, random block sizes, untextured boxes."""
    with snv.MiniWorldMovingBlockWorld(num_envs=LANES, size=SIZE, device="cpu", preset="upstream") as env:
        env.reset(seed=SEED)
        sources = [_upstream_source(lane) for lane in range(LANES)]
        for lane, src in enumerate(sources):
            src.reset(seed=SEED + lane)
        actions = np.random.default_rng(SEED + 1).integers(0, 5, size=(16, LANES))
        _run(env, sources, actions, wrap=False)


def test_the_step_limit_truncates_as_the_source_does():
    """The truncating step returns before the blocks move, as ``MovingBlockWorld.step`` does."""
    with snv.MiniWorldMovingBlockWorld(
        num_envs=LANES, size=SIZE, device="cpu", preset="upstream", max_episode_steps=3
    ) as env:
        env.reset(seed=SEED)
        sources = [_upstream_source(lane, max_episode_steps=3) for lane in range(LANES)]
        for lane, src in enumerate(sources):
            src.reset(seed=SEED + lane)
        result = _run(env, sources, np.full((3, LANES), 4), wrap=False)
        assert result["truncated"].all() and not result["terminated"].any()
