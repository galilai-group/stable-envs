"""MovingBlockWorld lanes over the Miniworld fork's own simulator, with complete continuation State.

Each lane is one ``miniworld.envs.MovingBlockWorld`` (hlillemark/Miniworld) configured as the FloWM
dataset script ``scripts/generate_videos.py`` builds it, or with the upstream defaults. State holds the
episode's world (room textures, each block's kind, colour, size and texture), every moving quantity (agent
pose and carried block, block poses and velocities), the env's PCG64 generator and its step count.

The world is rebuilt from State as ``MiniWorldEnv.reset`` builds it after ``_gen_world``: one rectangular
room, the agent first in the entity list, the blocks after it in placement order, then the default
(non-randomised) parameters and the static display list.

Envs hold a GL context each. They live in a process-wide pool per configuration and frame size and are
reused by later instances, never destroyed, so instances do not create contexts per call or per state.
"""

from __future__ import annotations

import math
import os
import sys

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.protocol import SimulatorAdapter

#: Fork commit the family runs and its reference gate compares against.
FORK_COMMIT = "b28988d5cd9d0d261a53b8cfa4f162e7fe14853e"
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

#: The FloWM textured dataset (``generate_videos.py`` flags of the README's training command); the
#: script draws ``num_blocks`` in [6, 10] per video, this preset fixes it per instance.
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
    "num_blocks": 8,
    "ensure_base_palette": True,
    "box_tex_randomize": True,
    "wall_tex_randomize": True,
    "floor_tex_randomize": True,
    "box_and_ball": True,
    # --turn-step-deg, --forward-step, --cam-fov-y: fixed (non-randomised) parameters
    "turn_step": 90.0,
    "forward_step": 1.0,
    "cam_fov_y": 60.0,
    # --heading-zero, and the [0, 2pi) heading wrap the script applies after reset and every step
    "heading_zero": True,
    "wrap_heading": True,
    # --steps 500 (the script drops the frame after the last step, so a 500-step limit gives the same frames)
    "max_episode_steps": 500,
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
PRESETS = {"flowm": FLOWM, "upstream": UPSTREAM}
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
    """The full configuration: ``preset`` (``"flowm"`` or ``"upstream"``) updated with ``overrides``."""
    if preset not in PRESETS:
        raise ValueError(f"preset must be one of {sorted(PRESETS)}, got {preset!r}")
    unknown = set(overrides) - set(WORLD_KEYS) - set(_HARNESS)
    if unknown:
        raise TypeError(f"unknown MovingBlockWorld option(s) {sorted(unknown)}; known: {[*WORLD_KEYS, *_HARNESS]}")
    config = {**UPSTREAM, **PRESETS[preset], **overrides}
    if "color_pool" in config and config["color_pool"] is not None:
        config["color_pool"] = tuple(config["color_pool"])
    num_blocks = int(config.get("num_blocks", 6))
    if num_blocks < 1:
        raise ValueError("num_blocks must be at least 1")
    config["num_blocks"] = num_blocks
    return config


def source_kwargs(config: dict, size: int) -> dict:
    """``MovingBlockWorld`` keyword arguments for ``config``, with ``size`` x ``size`` frames.

    The parameters follow ``generate_videos.build_env``: a fixed turn or forward step starts from
    ``DEFAULT_PARAMS.no_random()``, a field of view alone from ``DEFAULT_PARAMS.copy()``.
    """
    from miniworld.params import DEFAULT_PARAMS

    kwargs = {k: v for k, v in config.items() if k in WORLD_KEYS}
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


def make_source(config: dict, size: int):
    """A ``MovingBlockWorld`` for ``config`` drawing ``size`` px frames, as the dataset script builds it."""
    configure_pyglet()
    from miniworld.envs import MovingBlockWorld

    env = MovingBlockWorld(**source_kwargs(config, size))
    env.max_episode_steps = math.inf if config["max_episode_steps"] is None else int(config["max_episode_steps"])
    return env


_POOL: dict[tuple, list] = {}


def _pool_key(config: dict, size: int) -> tuple:
    return (repr(sorted(config.items())), int(size))


def acquire(config: dict, size: int, count: int) -> list:
    """``count`` source envs for ``config`` and ``size``, reused from the pool where possible."""
    free = _POOL.setdefault(_pool_key(config, size), [])
    envs = [free.pop() for _ in range(min(count, len(free)))]
    try:
        while len(envs) < count:
            envs.append(make_source(config, size))
    except BaseException:
        free.extend(envs)
        raise
    return envs


def release(config: dict, size: int, envs: list) -> None:
    """Return ``envs`` to the pool; their GL contexts stay alive for later instances."""
    _POOL.setdefault(_pool_key(config, size), []).extend(envs)


def frame(env) -> np.ndarray:
    """The agent view ``env.render()`` returns in ``rgb_array`` mode: ``render_obs`` into the window buffer.

    Every Miniworld env compiles its room into GL display list 1, and pyglet puts all contexts of a process
    in one share group, so with several envs the last one reset would supply every room. The list is
    recompiled before each frame, so each env draws its own room as a lone env in its process does.
    """
    env.shadow_window.switch_to()
    env._render_static()
    return env.render_obs(env.vis_fb)


def state_fields(num_blocks: int) -> dict[tuple[str, str], tuple[int, torch.dtype]]:
    """The State schema: ``{(group, name): (width, dtype)}``."""
    f64, i64, m = torch.float64, torch.int64, int(num_blocks)
    return {
        ("agent", "pos"): (3, f64),
        ("agent", "dir"): (1, f64),
        ("agent", "carrying"): (1, i64),  # index of the carried block, -1 for none
        ("room", "textures"): (3, i64),  # floor, wall, ceiling: indices into TEXTURES
        ("blocks", "kind"): (m, i64),  # BOX or BALL
        ("blocks", "color"): (m, i64),  # index into COLORS
        ("blocks", "texture"): (m, i64),  # index into TEXTURES, -1 for a plain box or a ball
        ("blocks", "size"): (3 * m, f64),  # box (sx, sy, sz); ball (height, height, height)
        ("blocks", "pos"): (3 * m, f64),
        ("blocks", "dir"): (m, f64),
        ("blocks", "velocity"): (3 * m, f64),
        ("rng", "state"): (4, i64),  # the env's PCG64, as four uint32 limbs, least significant first
        ("rng", "increment"): (4, i64),
        ("rng", "has_uint32"): (1, i64),
        ("rng", "uinteger"): (1, i64),
        ("lifecycle", "elapsed_steps"): (1, i64),
    }


def _limbs(value: int) -> list[int]:
    return [(int(value) >> (32 * i)) & 0xFFFF_FFFF for i in range(4)]


def _from_limbs(value) -> int:
    return sum(int(part) << (32 * i) for i, part in enumerate(np.asarray(value, np.int64).reshape(4)))


def _ball_colors() -> dict[int, str]:
    from miniworld.objmesh import ObjMesh

    return {id(ObjMesh.get(f"ball_{c}")): c for c in COLORS}


def read_lane(env, num_blocks: int) -> dict:
    """One lane's State values, read from the source env."""
    from miniworld.entity import Ball, Box

    blocks = env.entities[1:]
    if env.entities[0] is not env.agent or len(blocks) != num_blocks:
        raise RuntimeError("MovingBlockWorld lists the agent first and then exactly num_blocks blocks")
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
    room = env.rooms[0]
    rng = env.np_random.bit_generator.state
    if rng["bit_generator"] != "PCG64":
        raise RuntimeError(f"expected a PCG64 generator, got {rng['bit_generator']!r}")
    carrying = env.agent.carrying
    return {
        ("agent", "pos"): env.agent.pos,
        ("agent", "dir"): [env.agent.dir],
        ("agent", "carrying"): [-1 if carrying is None else blocks.index(carrying)],
        ("room", "textures"): [
            TEXTURES.index(n) for n in (room.floor_tex_name, room.wall_tex_name, room.ceil_tex_name)
        ],
        ("blocks", "kind"): kind,
        ("blocks", "color"): color,
        ("blocks", "texture"): texture,
        ("blocks", "size"): size,
        ("blocks", "pos"): np.concatenate([ent.pos for ent in blocks]),
        ("blocks", "dir"): [ent.dir for ent in blocks],
        ("blocks", "velocity"): np.concatenate([ent.velocity for ent in blocks]),
        ("rng", "state"): _limbs(rng["state"]["state"]),
        ("rng", "increment"): _limbs(rng["state"]["inc"]),
        ("rng", "has_uint32"): [rng["has_uint32"]],
        ("rng", "uinteger"): [rng["uinteger"]],
        ("lifecycle", "elapsed_steps"): [env.step_count],
    }


def _world(row: dict) -> tuple:
    """The parts of a lane's State that fix its static world (room, block kinds and sizes)."""
    return tuple(
        tuple(np.asarray(row[key]).reshape(-1).tolist())
        for key in (
            ("room", "textures"),
            ("blocks", "kind"),
            ("blocks", "color"),
            ("blocks", "texture"),
            ("blocks", "size"),
        )
    )


def build_world(env, row: dict) -> None:
    """Rebuild ``env``'s room and entities from State, as ``MiniWorldEnv.reset`` builds them after ``_gen_world``."""
    from miniworld.entity import Agent, Ball, Box

    floor, wall, ceil = (TEXTURES[int(i)] for i in row["room", "textures"])
    env.agent = Agent()
    env.entities, env.rooms, env.wall_segs = [], [], []
    env.add_rect_room(min_x=0, max_x=env.size, min_z=0, max_z=env.size, floor_tex=floor, wall_tex=wall, ceil_tex=ceil)
    env._gen_static_data()  # place_entity generates it before placing the agent
    env.entities.append(env.agent)
    sizes = np.asarray(row["blocks", "size"], np.float64).reshape(-1, 3)
    for kind, color, tex, size in zip(row["blocks", "kind"], row["blocks", "color"], row["blocks", "texture"], sizes):
        if int(kind) == BALL:
            ent = Ball(color=COLORS[int(color)], size=float(size[0]))
        else:
            ent = Box(color=COLORS[int(color)], size=size.copy(), texture=None if tex < 0 else TEXTURES[int(tex)])
        env.entities.append(ent)
    rand = env.np_random if env.domain_rand else None
    env.params.sample_many(rand, env, ["sky_color", "light_pos", "light_color", "light_ambient"])
    env.max_forward_step = env.params.get_max("forward_step")
    for ent in env.entities:
        ent.randomize(env.params, rand)
    env.min_x, env.max_x = min(r.min_x for r in env.rooms), max(r.max_x for r in env.rooms)
    env.min_z, env.max_z = min(r.min_z for r in env.rooms), max(r.max_z for r in env.rooms)
    env._render_static()
    env._se_world = _world(row)


def write_lane(env, row: dict) -> None:
    """Restore one lane from validated State values, rebuilding the world only when it differs."""
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
    env.np_random.bit_generator.state = {
        "bit_generator": "PCG64",
        "state": {"state": _from_limbs(row["rng", "state"]), "inc": _from_limbs(row["rng", "increment"])},
        "has_uint32": int(row["rng", "has_uint32"][0]),
        "uinteger": int(row["rng", "uinteger"][0]),
    }


def wrap_heading(env) -> None:
    """``generate_videos._wrap_angle_0_2pi`` applied to the agent's heading."""
    env.agent.dir = float(env.agent.dir % (2.0 * math.pi))


def host_actions(actions, num_envs: int) -> np.ndarray:
    """Actions as ``(num_envs,)`` int64 action ids, rejecting floats and ids outside [0, ACTION_HIGH]."""
    if torch.is_tensor(actions):
        actions = actions.detach().cpu().numpy()
    a = np.asarray(actions)
    if a.size != num_envs or a.dtype.kind not in "iu":
        raise ValueError(f"MovingBlockWorld actions must be {num_envs} integer action ids, got {a.dtype} {a.shape}")
    a = a.reshape(num_envs).astype(np.int64)
    if ((a < 0) | (a > ACTION_HIGH)).any():
        raise ValueError(f"MovingBlockWorld action ids lie in [0, {ACTION_HIGH}]")
    return a


class MovingBlockWorldAdapter(SimulatorAdapter):
    """``num_envs`` source envs on one configuration; lane ``i`` of ``reset(seed)`` is ``reset(seed=seed + i)``."""

    def __init__(self, config: dict, num_envs: int, size: int, device="cpu"):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        self.config, self.num_envs, self.size = config, num_envs, int(size)
        self.num_blocks = int(config["num_blocks"])
        self.device = torch.device(device)
        self.fields = state_fields(self.num_blocks)
        self.envs = acquire(config, self.size, num_envs)
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("MovingBlockWorldAdapter is closed")

    def close(self) -> None:
        if not self._closed:
            release(self.config, self.size, self.envs)
        self.envs, self._closed = [], True

    def _after_reset(self, env) -> None:
        if self.config["heading_zero"]:
            env.agent.dir = 0.0
        if self.config["wrap_heading"]:
            wrap_heading(env)
        env._se_world = _world(read_lane(env, self.num_blocks))

    def reset(self, seed: int | None = None) -> TensorDict:
        self._check_open()
        for i, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else int(seed) + i)
            self._after_reset(env)
        return self.get_state()

    def get_state(self) -> TensorDict:
        self._check_open()
        return self.pack([read_lane(env, self.num_blocks) for env in self.envs])

    def pack(self, rows: list[dict]) -> TensorDict:
        columns = {
            key: torch.tensor(np.asarray([row[key] for row in rows]).reshape(len(rows), width), dtype=dtype)
            for key, (width, dtype) in self.fields.items()
        }
        return TensorDict({k: v.to(self.device) for k, v in columns.items()}, [len(rows)], device=self.device)

    def unpack(self, state: TensorDict) -> list[dict]:
        host = {key: state[key].detach().cpu().numpy() for key in self.fields}
        return [{key: value[i] for key, value in host.items()} for i in range(int(state.batch_size[0]))]

    def validate(self, state, *, batch: int | None = None) -> None:
        """Require the complete typed schema with consistent values; nothing is changed."""
        if not isinstance(state, TensorDict):
            raise TypeError("Expected a full State TensorDict")
        if len(state.batch_size) != 1 or (batch is not None and state.batch_size != torch.Size([batch])):
            raise ValueError(f"Expected State batch [{batch or 'N'}], got {list(state.batch_size)}")
        have, want = set(state.keys(True, True)), set(self.fields)
        if have != want:
            missing, extra = sorted(want - have, key=str), sorted(have - want, key=str)
            raise ValueError(
                "State keys do not match the MovingBlockWorld schema"
                + (f"; missing {missing}" if missing else "")
                + (f"; unexpected {extra}" if extra else "")
            )
        n = int(state.batch_size[0])
        for key, (width, dtype) in self.fields.items():
            value = state[key]
            if tuple(value.shape) != (n, width) or value.dtype != dtype:
                raise ValueError(
                    f"State field {key!r} must be {dtype} of shape ({n}, {width}), got {value.dtype} {tuple(value.shape)}"
                )
            if value.is_floating_point() and not bool(torch.isfinite(value).all()):
                raise ValueError(f"State field {key!r} contains non-finite values")
        host = state.cpu()

        def within(key, lo, hi):
            v = host[key]
            if bool((v < lo).any() or (v > hi).any()):
                raise ValueError(f"State field {key!r} must lie in [{lo}, {hi}]")

        within(("agent", "carrying"), -1, self.num_blocks - 1)
        within(("room", "textures"), 0, len(TEXTURES) - 1)
        within(("blocks", "kind"), BOX, BALL)
        within(("blocks", "color"), 0, len(COLORS) - 1)
        within(("blocks", "texture"), -1, len(TEXTURES) - 1)
        within(("blocks", "size"), np.finfo(np.float64).tiny, np.inf)
        within(("rng", "state"), 0, 0xFFFF_FFFF)
        within(("rng", "increment"), 0, 0xFFFF_FFFF)
        within(("rng", "has_uint32"), 0, 1)
        within(("rng", "uinteger"), 0, 0xFFFF_FFFF)
        within(("lifecycle", "elapsed_steps"), 0, np.iinfo(np.int64).max)
        if bool(((host["blocks", "kind"] == BALL) & (host["blocks", "texture"] >= 0)).any()):
            raise ValueError("State field ('blocks', 'texture') must be -1 for a ball")
        if not bool((host["rng", "increment"][:, 0] % 2 == 1).all()):
            raise ValueError("State field ('rng', 'increment') must be odd (a PCG64 increment)")

    def set_state(self, state: TensorDict) -> None:
        self._check_open()
        self.validate(state, batch=self.num_envs)
        for env, row in zip(self.envs, self.unpack(state), strict=True):
            write_lane(env, row)

    def step(self, actions, *, active=None) -> TensorDict:
        """Advance one action per lane; an inactive lane holds its state and reports zero outputs."""
        self._check_open()
        actions = host_actions(actions, self.num_envs)
        if active is not None:
            active = np.asarray(active.detach().cpu() if torch.is_tensor(active) else active, dtype=bool)
        reward = np.zeros(self.num_envs)
        terminated = np.zeros(self.num_envs, bool)
        truncated = np.zeros(self.num_envs, bool)
        for i, env in enumerate(self.envs):
            if active is not None and not active[i]:
                continue
            _, reward[i], terminated[i], truncated[i], _ = env.step(int(actions[i]))
            if self.config["wrap_heading"]:
                wrap_heading(env)
        dev = self.device
        return TensorDict(
            {
                "state": self.get_state(),
                "reward": torch.tensor(reward, dtype=torch.float64, device=dev),
                "terminated": torch.tensor(terminated, device=dev),
                "truncated": torch.tensor(truncated, device=dev),
            },
            batch_size=[self.num_envs],
            device=dev,
        )

    def frames(self) -> torch.Tensor:
        """The live lanes' agent views, ``(num_envs, size, size, 3)`` uint8 on the adapter's device."""
        self._check_open()
        return torch.from_numpy(np.stack([frame(env) for env in self.envs])).to(self.device)


#: Render-only envs per configuration and size, for ``render(state)``; built on first use, never released.
_RENDERERS: dict[tuple, object] = {}


def render_rows(config: dict, size: int, adapter: MovingBlockWorldAdapter, state: TensorDict) -> torch.Tensor:
    """Draw a validated State of any batch with the process-wide render env for ``config`` and ``size``."""
    key = _pool_key(config, size)
    env = _RENDERERS.get(key)
    if env is None:
        env = _RENDERERS[key] = make_source(config, size)
    rows = adapter.unpack(state)
    out = np.empty((len(rows), int(size), int(size), 3), np.uint8)
    # Rows that share a world are drawn together, so the display list is rebuilt once per world.
    for i in sorted(range(len(rows)), key=lambda i: _world(rows[i])):
        write_lane(env, rows[i])
        out[i] = frame(env)
    return torch.from_numpy(out)
