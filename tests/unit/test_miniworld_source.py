"""MiniWorldMovingBlockWorld steps as the fork's ``MovingBlockWorld`` driven directly, environment by environment.

The source gate of the family. The source envs are built as the fork's ``scripts/generate_videos.py``
builds them (``build_env`` and the heading handling of ``run_rollout``), or with the registered defaults,
reset by the fork, and their State is restored into the batch. Then both step the same actions and are compared
on every step: the agent, each block, the step count, reward, both done flags and the frame ``env.render()``
returns. Resets draw from another generator than the fork's; ``test_resets_sample_the_source_distribution``
compares the distributions.
"""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

from stable_envs.envs.miniworld.simulation.adapter import TEXTURES, configure_pyglet, read_instance

configure_pyglet()  # before miniworld imports pyglet.gl
pytest.importorskip("miniworld.envs.moving_blockworld")
pytest.importorskip("numba")

from miniworld.entity import Ball, Box  # noqa: E402
from miniworld.envs import MovingBlockWorld  # noqa: E402
from miniworld.params import DEFAULT_PARAMS  # noqa: E402

import stable_envs as snv  # noqa: E402
from stable_envs import get_field  # noqa: E402
from stable_envs.envs.miniworld.simulation.world import BALL_RADIUS_PER_HEIGHT  # noqa: E402

pytestmark = pytest.mark.unit

NUM_ENVS = 2
SEED = 11
SIZE = 48
#: Source envs, kept for the process: each holds a GL context.
_SOURCES: dict = {}
_TEXTURED_ROOMS = {(f, w) for f in ("cardboard", "grass", "concrete") for w in ("brick_wall", "wood_planks", "wood")}


def _flowm_source(instance: int, textured: bool):
    """``generate_videos.build_env`` for the paper's dynamic ("SUNDAY V2") or textured ("TEX") split."""
    key = ("flowm", textured, instance)
    if key not in _SOURCES:
        tex = dict.fromkeys(("box_tex_randomize", "wall_tex_randomize", "floor_tex_randomize", "box_and_ball"), True)
        params = DEFAULT_PARAMS.no_random()
        params.set("turn_step", 90.0, 90.0, 90.0)
        params.set("forward_step", 1.0, 1.0, 1.0)
        params.set("cam_fov_y", 60.0, 60.0, 60.0)
        # num_blocks is the most an environment draws; each test sets the count before reset.
        env = MovingBlockWorld(
            view="agent",
            render_mode="rgb_array",
            obs_width=SIZE,
            obs_height=SIZE,
            window_width=SIZE,
            window_height=SIZE,
            size=16,
            box_allow_overlap=True,
            agent_box_allow_overlap=True,
            grid_mode=True,
            grid_vel_min=-1,
            grid_vel_max=1,
            grid_cardinal_only=True,
            num_blocks=10,
            ensure_base_palette=True,
            block_size_xy=0.7,
            block_height=1.5,
            params=params,
            **(tex if textured else {}),
        )
        _SOURCES[key] = env
    env = _SOURCES[key]
    env.max_episode_steps = 500
    return env


def _upstream_source(instance: int, max_episode_steps: int = 250):
    key = ("upstream", instance)
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


def _adopt(env, sources):
    """Restore the sources' worlds and motion into ``env``; the generator fields keep ``env``'s values."""
    schema = env._simulator.schema
    rows = [read_instance(src, env._simulator.num_blocks) for src in sources]
    state = env.get_state()
    columns = {}
    for name, field in schema.slices.items():
        if name.startswith("rng/"):
            columns[name] = get_field(state, name)
        else:
            values = np.asarray([row[name.replace("/", ".")] for row in rows])
            columns[name] = torch.tensor(values.reshape(len(rows), *field.shape), dtype=field.dtype)
    env.set_state(schema.fold(columns, [len(rows)]))


def _assert_instance(state, instance, src, t, atol=0.0):
    msg = f"environment {instance}, step {t}"

    def close(a, b):
        np.testing.assert_allclose(a, b, rtol=0, atol=atol, err_msg=msg)

    close(get_field(state, "agent/pos")[instance].numpy(), src.agent.pos)
    close(get_field(state, "agent/dir")[instance].numpy(), [src.agent.dir])
    blocks = src.entities[1:]
    n = len(blocks)
    assert get_field(state, "blocks/present")[instance].tolist() == [True] * n + [False] * (
        get_field(state, "blocks/present").shape[1] - n
    )
    close(get_field(state, "blocks/pos")[instance].numpy().reshape(-1, 3)[:n], np.stack([b.pos for b in blocks]))
    close(
        get_field(state, "blocks/velocity")[instance].numpy().reshape(-1, 3)[:n],
        np.stack([b.velocity for b in blocks]),
    )
    close(get_field(state, "blocks/dir")[instance].numpy()[:n], [b.dir for b in blocks])
    assert get_field(state, "blocks/kind")[instance].tolist()[:n] == [
        0 if isinstance(b, Box) else 1 for b in blocks
    ], msg
    carried = src.agent.carrying
    assert get_field(state, "agent/carrying")[instance].item() == (-1 if carried is None else blocks.index(carried)), (
        msg
    )
    assert get_field(state, "lifecycle/elapsed_steps")[instance].item() == src.step_count, msg


def _run(env, sources, actions, *, wrap, frames=True, atol=0.0):
    state = env.get_state()
    for instance, src in enumerate(sources):
        _assert_instance(state, instance, src, 0)
    if frames:
        np.testing.assert_array_equal(env.render().numpy(), np.stack([_source_frame(src) for src in sources]))
    result = None
    for t, action in enumerate(actions, start=1):
        result = env.step(action[:, None], frames=slice(-1, None) if frames else None)
        for instance, src in enumerate(sources):
            _, reward, terminated, truncated, _ = src.step(int(action[instance]))
            if wrap:
                _wrap(src)
            _assert_instance(result["state"], instance, src, t, atol)
            assert result["reward"][instance].item() == reward
            assert result["terminated"][instance].item() == terminated
            assert result["truncated"][instance].item() == truncated
            if frames:
                np.testing.assert_array_equal(
                    result["frames"][instance, -1].numpy(), _source_frame(src), err_msg=f"step {t}"
                )
    return result


def _flowm_sources(preset, wrap):
    sources = [_flowm_source(instance, textured=preset == "flowm-textured") for instance in range(NUM_ENVS)]
    for instance, src in enumerate(sources):
        src.num_blocks = 6 + 2 * instance  # --num-blocks-min 6 --num-blocks-max 10
        src.reset(seed=SEED + instance)
        src.agent.dir = 0.0  # --heading-zero
        if wrap:
            _wrap(src)
    return sources


def _upstream_sources(max_episode_steps=250):
    sources = [_upstream_source(instance, max_episode_steps) for instance in range(NUM_ENVS)]
    for instance, src in enumerate(sources):
        src.reset(seed=SEED + instance)
    return sources


@pytest.mark.parametrize(
    "preset,wrap",
    [("flowm", False), ("flowm-textured", False), ("flowm", True)],
    ids=["dynamic", "textured", "wrapped-heading"],
)
def test_environments_step_as_the_source_called_directly(preset, wrap):
    """The FloWM presets: grid motion, plain or textured with balls; every action id, pick up and drop included.

    ``wrap`` adds the [0, 2pi) heading wrap of the fork's later script (``wrap_heading=True``).
    """
    with snv.MiniWorldMovingBlockWorld(
        num_envs=NUM_ENVS, size=SIZE, device="cpu", preset=preset, wrap_heading=wrap
    ) as env:
        assert env.simulation_backend == "numba"
        sources = _flowm_sources(preset, wrap)
        _adopt(env, sources)
        rng = np.random.default_rng(SEED)
        actions = rng.integers(0, 9, size=(24, NUM_ENVS))
        actions[:4] = [2, 5]  # walk, and try to pick up a block
        _run(env, sources, actions, wrap=wrap)


def test_the_upstream_preset_steps_as_the_registered_env():
    """``preset="upstream"``: continuous motion in a 12 x 12 room, random block sizes, untextured boxes."""
    with snv.MiniWorldMovingBlockWorld(num_envs=NUM_ENVS, size=SIZE, device="cpu", preset="upstream") as env:
        sources = _upstream_sources()
        _adopt(env, sources)
        actions = np.random.default_rng(SEED + 1).integers(0, 5, size=(16, NUM_ENVS))
        _run(env, sources, actions, wrap=False)


def test_the_step_limit_truncates_as_the_source_does():
    """The truncating step returns before the blocks move, as ``MovingBlockWorld.step`` does."""
    with snv.MiniWorldMovingBlockWorld(
        num_envs=NUM_ENVS, size=SIZE, device="cpu", preset="upstream", max_episode_steps=3
    ) as env:
        sources = _upstream_sources(max_episode_steps=3)
        _adopt(env, sources)
        result = _run(env, sources, np.full((3, NUM_ENVS), 4), wrap=False)
        assert result["truncated"].all() and not result["terminated"].any()


#: A fixed scripted policy: walk, turn, pick up, carry, drop and back up, repeated.
SCRIPT = np.array([2, 2, 5, 2, 0, 2, 2, 1, 2, 6, 3, 2, 2, 0, 5, 2, 1, 1, 2, 6], np.int64)


@pytest.mark.parametrize("backend", ["numba", "torch"])
@pytest.mark.parametrize("preset", ["flowm", "flowm-textured", "upstream"])
def test_a_scripted_200_step_rollout_stays_on_the_source(backend, preset):
    """200 steps of a fixed policy from the source's reset: Numba gives the source's values; Torch agrees to 1e-9."""
    atol = 0.0 if backend == "numba" else 1e-9
    with snv.MiniWorldMovingBlockWorld(
        num_envs=NUM_ENVS, size=SIZE, device="cpu", preset=preset, simulation_backend=backend
    ) as env:
        sources = _upstream_sources() if preset == "upstream" else _flowm_sources(preset, False)
        _adopt(env, sources)
        actions = np.tile(np.stack([np.roll(SCRIPT, 3 * instance) for instance in range(NUM_ENVS)], 1), (10, 1))
        result = _run(env, sources, actions, wrap=False, frames=False, atol=atol)
        assert not result["truncated"].any()


def test_ball_radius_matches_the_fork():
    for color in ("blue", "green", "grey", "purple", "red", "yellow"):
        assert float(Ball(color=color, size=0.85).radius) == BALL_RADIUS_PER_HEIGHT * 0.85


def test_resets_sample_the_source_distribution():
    """FloWM textured resets: counts, kinds, palette, room textures, grid cells and velocities as the fork draws them."""
    episodes = 64
    source = _flowm_source(0, textured=True)
    src = {"count": [], "ball": [], "pos": [], "vel": set(), "agent_pos": [], "rooms": set()}
    rng = np.random.default_rng(0)
    for e in range(episodes):
        source.num_blocks = int(rng.integers(6, 11))
        source.reset(seed=1000 + e)
        blocks = source.entities[1:]
        src["count"].append(len(blocks))
        src["ball"] += [isinstance(b, Ball) for b in blocks]
        src["pos"] += [b.pos[[0, 2]] for b in blocks]
        src["vel"] |= {tuple(float(v) for v in b.velocity) for b in blocks}
        src["agent_pos"].append(source.agent.pos[[0, 2]])
        src["rooms"].add((source.rooms[0].floor_tex_name, source.rooms[0].wall_tex_name))
    with snv.MiniWorldMovingBlockWorld(num_envs=episodes, size=SIZE, device="cpu", preset="flowm-textured") as env:
        state = env.reset(seed=1000)
    present = get_field(state, "blocks/present")
    count = present.sum(-1)
    pos = get_field(state, "blocks/pos").reshape(episodes, -1, 3)[present][:, [0, 2]].numpy()
    vel = get_field(state, "blocks/velocity").reshape(episodes, -1, 3)[present].numpy()
    assert set(count.tolist()) <= set(range(6, 11)) and len(set(count.tolist())) >= 4
    cardinal = {(1.0, 0.0, 0.0), (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0), (0.0, 0.0, -1.0)}
    assert {tuple(float(x) for x in v) for v in vel} == src["vel"] == cardinal
    for p in (pos, np.stack(src["pos"]), get_field(state, "agent/pos")[:, [0, 2]].numpy(), np.stack(src["agent_pos"])):
        assert (p == np.round(p)).all() and p.min() >= 1 and p.max() <= 15
        assert abs(p.mean() - 8) < 1.0
    ball = (get_field(state, "blocks/kind")[present] == 1).double().mean().item()
    assert abs(ball - 1 / 3) < 0.1 and abs(np.mean(src["ball"]) - 1 / 3) < 0.1
    # ensure_base_palette: the first five blocks take the five base colours in some order.
    assert all(len(set(row)) == 5 for row in get_field(state, "blocks/color")[:, :5].tolist())
    rooms = {(TEXTURES[f], TEXTURES[w]) for f, w, _ in get_field(state, "room/textures").tolist()}
    assert rooms <= _TEXTURED_ROOMS and src["rooms"] <= _TEXTURED_ROOMS and len(rooms) >= 6
