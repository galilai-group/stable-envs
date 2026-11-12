"""MovingBlockWorld State and frames: the State schema, its checks, and the fork's own GL drawing of a State.

A State row is drawn by rebuilding the world in a ``miniworld.envs.MovingBlockWorld`` (hlillemark/Miniworld at its
``flowm_release`` branch, the code frozen for the FloWM paper) as ``MiniWorldEnv.reset`` builds it after
``_gen_world``: one rectangular room, the agent first in the entity list, the blocks after it in placement order,
then the default (non-randomised) parameters and the static display list. Render envs hold a GL context each and
are kept for the process.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from tensordict import TensorDict

#: Fork commit the family runs and its reference gate compares against: the head of ``flowm_release``.
#: The fork's master changes the base palette's "gray" to "grey", which changes the drawn block colours.
FORK_COMMIT = "bc7972050155b67e242c07bfdccfe1eeddbf9882"
FORK_URL = "https://github.com/hlillemark/Miniworld"

#: Texture names of the Miniworld package; State stores an index into this tuple (-1: untextured box).
TEXTURES = (
    "airduct_grate",
    "asphalt",
    "brick_wall",
    "cardboard",
    "ceiling_tile_noborder",
    "ceiling_tiles",
    "checkerboard",
    "cinder_blocks",
    "concrete",
    "concrete_tiles",
    "door_doom",
    "door_garage_red",
    "door_garage_white",
    "door_steel_brown",
    "door_steel_grey",
    "door_steel_red",
    "drywall",
    "floor_tiles_bw",
    "floor_tiles_white",
    "grass",
    "lava",
    "logo_mila",
    "marble",
    "metal_grill",
    "picket_fence",
    "rock",
    "slime",
    "stucco",
    "water",
    "white",
    "wood",
    "wood_planks",
)
#: ``miniworld.entity.COLOR_NAMES``; State stores an index into this tuple.
COLORS = ("blue", "green", "grey", "purple", "red", "yellow")
BOX, BALL = 0, 1
#: Highest action id of the fork's ``MiniWorldEnv.Actions`` (``done``).
ACTION_HIGH = 8

#: The FloWM dynamic dataset ("SUNDAY V2" in the fork's ``scripts/old_generation_commands.py``): plain
#: coloured boxes in the env's white-walled, concrete-floored room. Like the script's
#: ``--num-blocks-min 6 --num-blocks-max 10``, each lane draws its block count uniformly from [6, 10] at reset.
FLOWM = {
    "room_size": 16,
    "grid_mode": True,
    "grid_vel_min": -1,
    "grid_vel_max": 1,
    "grid_cardinal_only": True,
    "block_size_xy": 0.7,
    "block_height": 1.5,
    "box_allow_overlap": True,
    "agent_box_allow_overlap": True,
    "num_blocks": (6, 10),
    "ensure_base_palette": True,
    # --turn-step-deg, --forward-step, --cam-fov-y: fixed (non-randomised) parameters
    "turn_step": 90.0,
    "forward_step": 1.0,
    "cam_fov_y": 60.0,
    # --heading-zero; the paper's script does not wrap the heading (master's later script wraps it to [0, 2pi))
    "heading_zero": True,
    "wrap_heading": False,
    # --steps 500 (the script drops the frame after the last step, so a 500-step limit gives the same frames)
    "max_episode_steps": 500,
}
#: The FloWM textured dataset ("TEX", the README's command): FLOWM with random wall, floor and block
#: textures, and balls among the blocks.
FLOWM_TEXTURED = {
    **FLOWM,
    "box_tex_randomize": True,
    "wall_tex_randomize": True,
    "floor_tex_randomize": True,
    "box_and_ball": True,
}
#: ``MovingBlockWorld()`` as the fork registers it (12 x 12 room, continuous motion, PutNext's 250 steps).
UPSTREAM = {
    "turn_step": None,
    "forward_step": None,
    "cam_fov_y": None,
    "heading_zero": False,
    "wrap_heading": False,
    "max_episode_steps": 250,
}
PRESETS = {"flowm": FLOWM, "flowm-textured": FLOWM_TEXTURED, "upstream": UPSTREAM}
#: Keys handled here rather than passed to ``MovingBlockWorld``.
_HARNESS = ("turn_step", "forward_step", "cam_fov_y", "heading_zero", "wrap_heading", "max_episode_steps")
#: ``MovingBlockWorld`` keyword arguments a configuration may set (``room_size`` is its ``size``).
WORLD_KEYS = (
    "room_size",
    "floor_tex",
    "wall_tex",
    "ceil_tex",
    "box_tex",
    "box_tex_randomize",
    "wall_tex_randomize",
    "floor_tex_randomize",
    "box_and_ball",
    "box_speed_scale",
    "box_allow_overlap",
    "agent_box_allow_overlap",
    "box_random_orientation",
    "blocks_static",
    "block_torus_wrap",
    "spawn_wall_buffer",
    "grid_mode",
    "grid_vel_min",
    "grid_vel_max",
    "grid_cardinal_only",
    "num_blocks",
    "allow_color_repeat",
    "color_pool",
    "ensure_base_palette",
    "near_margin",
    "block_size_xy",
    "block_height",
    "agent_center_start",
)
#: Frame buffer of the observation the source renders inside ``step``; frames are drawn separately at
#: the requested size, so this one is kept small.
_OBS_SIZE = 8


def configure_pyglet() -> None:
    """Draw offscreen through EGL on Linux unless ``PYGLET_HEADLESS`` says otherwise; before pyglet's window import."""
    if sys.platform.startswith("linux") and "PYGLET_HEADLESS" not in os.environ:
        import pyglet

        pyglet.options["headless"] = True


def resolve_config(preset: str, overrides: dict) -> dict:
    """The full configuration: ``preset`` (a key of ``PRESETS``) updated with ``overrides``."""
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}, got {preset!r}")
    unknown = set(overrides) - set(WORLD_KEYS) - set(_HARNESS)
    if unknown:
        raise TypeError(f"unknown MovingBlockWorld option(s) {sorted(unknown)}; known: {[*WORLD_KEYS, *_HARNESS]}")
    config = {**UPSTREAM, **PRESETS[preset], **overrides}
    if "color_pool" in config and config["color_pool"] is not None:
        config["color_pool"] = tuple(config["color_pool"])
    # An int fixes the count; a (min, max) pair draws it per lane and reset, like the dataset script.
    count = config.get("num_blocks", 6)
    lo, hi = (int(count), int(count)) if isinstance(count, (int, np.integer)) else (int(count[0]), int(count[1]))
    if not 1 <= lo <= hi:
        raise ValueError(f"num_blocks must be a positive int or a (min, max) pair with 1 <= min <= max, got {count!r}")
    config["num_blocks"] = (lo, hi)
    return config


def source_kwargs(config: dict, size: int) -> dict:
    """``MovingBlockWorld`` keyword arguments for ``config``, with ``size`` x ``size`` frames.

    The parameters follow ``generate_videos.build_env``: a fixed turn or forward step starts from
    ``DEFAULT_PARAMS.no_random()``, a field of view alone from ``DEFAULT_PARAMS.copy()``.
    """
    from miniworld.params import DEFAULT_PARAMS

    kwargs = {k: v for k, v in config.items() if k in WORLD_KEYS}
    kwargs["num_blocks"] = int(config["num_blocks"][1])  # each reset sets the lane's own count
    if "room_size" in kwargs:
        kwargs["size"] = int(kwargs.pop("room_size"))
    if "color_pool" in kwargs and kwargs["color_pool"] is not None:
        kwargs["color_pool"] = list(kwargs["color_pool"])
    params = None
    if config["turn_step"] is not None or config["forward_step"] is not None:
        params = DEFAULT_PARAMS.no_random()
        for name in ("turn_step", "forward_step"):
            if config[name] is not None:
                v = float(config[name])
                params.set(name, v, v, v)
    if config["cam_fov_y"] is not None:
        params = DEFAULT_PARAMS.copy() if params is None else params
        v = float(config["cam_fov_y"])
        params.set("cam_fov_y", v, v, v)
    if params is not None:
        kwargs["params"] = params
    return dict(kwargs, obs_width=_OBS_SIZE, obs_height=_OBS_SIZE, window_width=int(size), window_height=int(size))


#: The room that GL display list 1 holds, as ``(configuration, room textures)``, or None when unknown.
#: Every Miniworld env compiles its static scene (room and lights) into list 1, and pyglet puts all
#: contexts of a process in one share group, so the last env to compile it supplies every env's room.
_COMPILED: list = [None]


def _track_display_list() -> None:
    """Clear the record of list 1 whenever any Miniworld env compiles it (resets of envs this family does not own)."""
    from miniworld.miniworld import MiniWorldEnv

    if getattr(MiniWorldEnv._render_static, "_se_tracked", False):
        return
    compile_static = MiniWorldEnv._render_static

    def tracked(self):
        compile_static(self)
        _COMPILED[0] = None

    tracked._se_tracked = True
    MiniWorldEnv._render_static = tracked


def make_source(config: dict, size: int):
    """A ``MovingBlockWorld`` for ``config`` drawing ``size`` px frames, as the dataset script builds it."""
    configure_pyglet()
    from miniworld.envs import MovingBlockWorld

    _track_display_list()
    env = MovingBlockWorld(**source_kwargs(config, size))
    env.max_episode_steps = math.inf if config["max_episode_steps"] is None else int(config["max_episode_steps"])
    env._se_config = repr(sorted(config.items()))
    set_room(env)
    _COMPILED[0] = env._se_room  # the constructor's reset compiled list 1 with this room
    return env


def set_room(env) -> None:
    """Record the room ``env`` holds, the key of its display list in ``_COMPILED``."""
    room = env.rooms[0]
    names = (room.floor_tex_name, room.wall_tex_name, room.ceil_tex_name)
    env._se_room = (env._se_config, tuple(TEXTURES.index(n) for n in names))


def _pool_key(config: dict, size: int) -> tuple:
    return (repr(sorted(config.items())), int(size))


def frame(env) -> np.ndarray:
    """The agent view ``env.render()`` returns in ``rgb_array`` mode: ``render_obs`` into the window buffer.

    Display list 1 is recompiled only when it holds another room (see ``_COMPILED``), so each env draws its
    own room, as a lone env in its process does, and lanes that share a room compile it once.
    """
    env.shadow_window.switch_to()
    room = env._se_room
    if _COMPILED[0] != room:
        env._render_static()
        _COMPILED[0] = room
    return env.render_obs(env.vis_fb)


def state_fields(num_blocks: int) -> dict[tuple[str, str], tuple[int, torch.dtype]]:
    """The State schema: ``{(group, name): (width, dtype)}``."""
    f64, i64, m = torch.float64, torch.int64, int(num_blocks)
    return {
        ("agent", "pos"): (3, f64),
        ("agent", "dir"): (1, f64),
        ("agent", "carrying"): (1, i64),  # index of the carried block, -1 for none
        ("room", "textures"): (3, i64),  # floor, wall, ceiling: indices into TEXTURES
        ("blocks", "present"): (m, torch.bool),  # the lane's blocks first; absent slots hold fixed padding
        ("blocks", "kind"): (m, i64),  # BOX or BALL
        ("blocks", "color"): (m, i64),  # index into COLORS
        ("blocks", "texture"): (m, i64),  # index into TEXTURES, -1 for a plain box or a ball
        ("blocks", "size"): (3 * m, f64),  # box (sx, sy, sz); ball (height, height, height)
        ("blocks", "pos"): (3 * m, f64),
        ("blocks", "dir"): (m, f64),
        ("blocks", "velocity"): (3 * m, f64),
        ("rng", "seed"): (1, i64),  # the reset draws hash (seed, episode, draw index); see rng.py
        ("rng", "episode"): (1, i64),
        ("lifecycle", "elapsed_steps"): (1, i64),
    }


def _ball_colors() -> dict[int, str]:
    from miniworld.objmesh import ObjMesh

    return {id(ObjMesh.get(f"ball_{c}")): c for c in COLORS}


def read_lane(env, num_blocks: int) -> dict:
    """One environment's State values, read from a source env; ``num_blocks`` is the padded width.

    A source env has no counter-based generator, so the ``rng`` fields are 0.
    """
    from miniworld.entity import Ball, Box

    blocks = env.entities[1:]
    if env.entities[0] is not env.agent or not 1 <= len(blocks) <= num_blocks:
        raise RuntimeError("MovingBlockWorld lists the agent first and then at most num_blocks blocks")
    pad = num_blocks - len(blocks)
    balls = _ball_colors()
    kind, color, texture, size = [], [], [], []
    for ent in blocks:
        if isinstance(ent, Box):
            kind.append(BOX)
            color.append(COLORS.index(ent.color))
            texture.append(-1 if ent.tex_name is None else TEXTURES.index(ent.tex_name))
            size.extend(np.asarray(ent.size, np.float64).reshape(3))
        elif isinstance(ent, Ball):
            kind.append(BALL)
            color.append(COLORS.index(balls[id(ent.mesh)]))
            texture.append(-1)
            size.extend([ent.height] * 3)
        else:
            raise RuntimeError(f"unexpected MovingBlockWorld entity {type(ent).__name__}")
    kind, color, texture, size = kind + [BOX] * pad, color + [0] * pad, texture + [-1] * pad, size + [1.0] * 3 * pad
    zeros = [0.0] * 3 * pad
    room = env.rooms[0]
    carrying = env.agent.carrying
    return {
        ("agent", "pos"): env.agent.pos,
        ("agent", "dir"): [env.agent.dir],
        ("agent", "carrying"): [-1 if carrying is None else blocks.index(carrying)],
        ("room", "textures"): [
            TEXTURES.index(n) for n in (room.floor_tex_name, room.wall_tex_name, room.ceil_tex_name)
        ],
        ("blocks", "present"): [True] * len(blocks) + [False] * pad,
        ("blocks", "kind"): kind,
        ("blocks", "color"): color,
        ("blocks", "texture"): texture,
        ("blocks", "size"): size,
        ("blocks", "pos"): np.concatenate([*(ent.pos for ent in blocks), zeros]),
        ("blocks", "dir"): [ent.dir for ent in blocks] + [0.0] * pad,
        ("blocks", "velocity"): np.concatenate([*(ent.velocity for ent in blocks), zeros]),
        ("rng", "seed"): [0],
        ("rng", "episode"): [0],
        ("lifecycle", "elapsed_steps"): [env.step_count],
    }


def _world(row: dict) -> tuple:
    """The parts of a lane's State that fix its static world (room, block kinds and sizes)."""
    return tuple(
        tuple(np.asarray(row[key]).reshape(-1).tolist())
        for key in (
            ("room", "textures"),
            ("blocks", "present"),
            ("blocks", "kind"),
            ("blocks", "color"),
            ("blocks", "texture"),
            ("blocks", "size"),
        )
    )


def build_world(env, row: dict) -> None:
    """Rebuild ``env``'s room and entities from State, as ``MiniWorldEnv.reset`` builds them after ``_gen_world``.

    The room (geometry, textures, collision segments and lights) is rebuilt only when its textures differ:
    lanes of one configuration differ only in their blocks, which are drawn per frame. ``frame`` compiles
    the room's display list when it draws.
    """
    from miniworld.entity import Agent, Ball, Box

    textures = tuple(int(i) for i in row["room", "textures"])
    if getattr(env, "_se_room", (None, None))[1] != textures:
        floor, wall, ceil = (TEXTURES[i] for i in textures)
        env.rooms, env.wall_segs = [], []
        env.add_rect_room(
            min_x=0, max_x=env.size, min_z=0, max_z=env.size, floor_tex=floor, wall_tex=wall, ceil_tex=ceil
        )
        env._gen_static_data()  # place_entity generates it before placing the agent
        rand = env.np_random if env.domain_rand else None
        env.params.sample_many(rand, env, ["sky_color", "light_pos", "light_color", "light_ambient"])
        env.max_forward_step = env.params.get_max("forward_step")
        env.min_x, env.max_x = min(r.min_x for r in env.rooms), max(r.max_x for r in env.rooms)
        env.min_z, env.max_z = min(r.min_z for r in env.rooms), max(r.max_z for r in env.rooms)
        set_room(env)
    env.agent = Agent()
    env.entities = [env.agent]
    sizes = np.asarray(row["blocks", "size"], np.float64).reshape(-1, 3)
    count = int(np.asarray(row["blocks", "present"]).sum())
    env.num_blocks = count
    columns = (row["blocks", "kind"], row["blocks", "color"], row["blocks", "texture"], sizes)
    for kind, color, tex, size in list(zip(*columns))[:count]:
        if int(kind) == BALL:
            ent = Ball(color=COLORS[int(color)], size=float(size[0]))
        else:
            ent = Box(color=COLORS[int(color)], size=size.copy(), texture=None if tex < 0 else TEXTURES[int(tex)])
        env.entities.append(ent)
    rand = env.np_random if env.domain_rand else None
    for ent in env.entities:
        ent.randomize(env.params, rand)
    env._se_world = _world(row)


def write_lane(env, row: dict) -> None:
    """Write validated State values into a source env, rebuilding the world only when it differs."""
    if getattr(env, "_se_world", None) != _world(row):
        build_world(env, row)
    blocks = env.entities[1:]
    env.agent.pos = np.asarray(row["agent", "pos"], np.float64).copy()
    env.agent.dir = float(row["agent", "dir"][0])
    carrying = int(row["agent", "carrying"][0])
    env.agent.carrying = None if carrying < 0 else blocks[carrying]
    pos = np.asarray(row["blocks", "pos"], np.float64).reshape(-1, 3)
    vel = np.asarray(row["blocks", "velocity"], np.float64).reshape(-1, 3)
    for k, ent in enumerate(blocks):
        ent.pos = pos[k].copy()
        ent.dir = float(row["blocks", "dir"][k])
        ent.velocity = vel[k].copy()
    env.step_count = int(row["lifecycle", "elapsed_steps"][0])
    set_room(env)


def validate_state(state, fields: dict, count_range: tuple[int, int], *, batch: int | None = None) -> None:
    """Require the complete typed schema with consistent values; nothing is changed."""
    if not isinstance(state, TensorDict):
        raise TypeError("Expected a full State TensorDict")
    if len(state.batch_size) != 1 or (batch is not None and state.batch_size != torch.Size([batch])):
        raise ValueError(f"Expected State batch [{batch or 'N'}], got {list(state.batch_size)}")
    have, want = set(state.keys(True, True)), set(fields)
    if have != want:
        missing, extra = sorted(want - have, key=str), sorted(have - want, key=str)
        raise ValueError(
            "State keys do not match the MovingBlockWorld schema"
            + (f"; missing {missing}" if missing else "")
            + (f"; unexpected {extra}" if extra else "")
        )
    n = int(state.batch_size[0])
    for key, (width, dtype) in fields.items():
        value = state[key]
        if tuple(value.shape) != (n, width) or value.dtype != dtype:
            raise ValueError(
                f"State field {key!r} must be {dtype} of shape ({n}, {width}), got {value.dtype} {tuple(value.shape)}"
            )
    host = state.cpu()
    for key, (_, dtype) in fields.items():
        if dtype.is_floating_point and not bool(torch.isfinite(host[key]).all()):
            raise ValueError(f"State field {key!r} contains non-finite values")

    def within(key, lo, hi):
        v = host[key]
        if bool((v < lo).any() or (v > hi).any()):
            raise ValueError(f"State field {key!r} must lie in [{lo}, {hi}]")

    num_blocks = fields["blocks", "present"][0]
    present = host["blocks", "present"]
    count = present.sum(-1)
    lo, hi = count_range
    if bool((present != (torch.arange(num_blocks) < count[:, None])).any()):
        raise ValueError("State field ('blocks', 'present') must list an environment's blocks first")
    if bool(((count < lo) | (count > hi)).any()):
        raise ValueError(f"State field ('blocks', 'present') must mark between {lo} and {hi} blocks")
    if bool((host["agent", "carrying"][:, 0] >= count).any()):
        raise ValueError("State field ('agent', 'carrying') must name a present block or -1")
    within(("agent", "carrying"), -1, num_blocks - 1)
    within(("room", "textures"), 0, len(TEXTURES) - 1)
    within(("blocks", "kind"), BOX, BALL)
    within(("blocks", "color"), 0, len(COLORS) - 1)
    within(("blocks", "texture"), -1, len(TEXTURES) - 1)
    within(("blocks", "size"), np.finfo(np.float64).tiny, np.inf)
    within(("rng", "episode"), 0, np.iinfo(np.int64).max)
    within(("lifecycle", "elapsed_steps"), 0, np.iinfo(np.int64).max)
    if bool(((host["blocks", "kind"] == BALL) & (host["blocks", "texture"] >= 0)).any()):
        raise ValueError("State field ('blocks', 'texture') must be -1 for a ball")


def pack_rows(state: TensorDict, fields: dict) -> np.ndarray:
    """A validated State as ``(n, D)`` float64 rows for the render workers; every integer field is exact."""
    return np.concatenate([state[key].detach().cpu().numpy().astype(np.float64) for key in fields], axis=1)


def unpack_rows(rows: np.ndarray, fields: dict) -> list[dict]:
    """Inverse of :func:`pack_rows`: one State value dict per row."""
    lanes, offset = [{} for _ in range(len(rows))], 0
    for key, (width, dtype) in fields.items():
        part = rows[:, offset : offset + width]
        if dtype == torch.bool:
            part = part > 0.5
        elif not dtype.is_floating_point:
            part = np.rint(part).astype(np.int64)
        for lane, value in zip(lanes, part, strict=True):
            lane[key] = value
        offset += width
    return lanes


def draw_rows(env, rows: list[dict], out: np.ndarray) -> None:
    """Draw State value dicts into ``out`` with ``env``.

    Rows are drawn in order of their world, room first, so each room's display list is compiled once and
    each world's blocks are built once.
    """
    for i in sorted(range(len(rows)), key=lambda i: _world(rows[i])):
        write_lane(env, rows[i])
        out[i] = frame(env)


class RenderRows:
    """Picklable factory of a render worker's drawer: one render env for ``config`` and ``size``."""

    def __init__(self, config: dict, size: int):
        self.config, self.size = config, int(size)

    def __call__(self):
        env = make_source(self.config, self.size)
        fields = state_fields(int(self.config["num_blocks"][1]))

        def draw(rows, frames):
            draw_rows(env, unpack_rows(rows, fields), frames)

        return draw


#: Render-only envs per configuration and size, for ``render(state)`` in this process; never released.
_RENDERERS: dict[tuple, object] = {}


def render_here(config: dict, size: int, rows: np.ndarray) -> np.ndarray:
    """Draw packed rows with this process's render env for ``config`` and ``size``."""
    key = _pool_key(config, size)
    env = _RENDERERS.get(key)
    if env is None:
        env = _RENDERERS[key] = make_source(config, size)
    out = np.empty((len(rows), int(size), int(size), 3), np.uint8)
    draw_rows(env, unpack_rows(rows, state_fields(int(config["num_blocks"][1]))), out)
    return out
