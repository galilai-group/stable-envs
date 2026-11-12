"""The fixed numbers of a MovingBlockWorld configuration, as the batched kernels read them."""

from __future__ import annotations

import math

import numpy as np

#: ``Ball(size=0.85).radius / 0.85`` for the fork's ball meshes (``MeshEnt``: the mesh's x-z half-diagonal over its
#: height, in float32); times 0.85 it gives the source's radius exactly.
BALL_RADIUS_PER_HEIGHT = 0.708539275562062
#: Defaults of ``miniworld.params.DEFAULT_PARAMS`` that a configuration may replace.
DEFAULT_FORWARD_STEP, DEFAULT_FORWARD_MAX, DEFAULT_TURN_STEP = 0.15, 0.17, 15.0
#: ``MovingBlockWorld`` texture defaults and pools.
FLOOR_TEX, WALL_TEX, CEIL_TEX = "concrete", "white", "ceiling_tile_noborder"
FLOOR_POOL = ("cardboard", "grass", "concrete")
WALL_POOL = ("brick_wall", "wood_planks", "wood")
BOX_POOL = ("ceiling_tiles", "airduct_grate", "checkerboard")
#: ``Room`` defaults, used when a configuration sets a room texture to None.
ROOM_DEFAULTS = {"floor_tex": "floor_tiles_bw", "wall_tex": "concrete", "ceil_tex": "concrete_tiles"}
#: ``ensure_base_palette`` order; the fork's "gray" is not a Miniworld colour name, so it never joins.
BASE_PALETTE = ("green", "red", "yellow", "blue", "purple", "gray")

(
    P_ROOM,
    P_GRID,
    P_CARDINAL,
    P_VMIN,
    P_VMAX,
    P_BOX_OVERLAP,
    P_AGENT_OVERLAP,
    P_RANDOM_ORIENT,
    P_STATIC,
    P_TORUS,
    P_SPAWN_BUFFER,
    P_REPEAT,
    P_ENSURE_BASE,
    P_BLOCK_XY,
    P_BLOCK_H,
    P_AGENT_CENTER,
    P_FLOOR_TEX,
    P_FLOOR_TEX_RANDOM,
    P_WALL_TEX,
    P_WALL_TEX_RANDOM,
    P_CEIL_TEX,
    P_BOX_TEX,
    P_BOX_TEX_RANDOM,
    P_BOX_AND_BALL,
    P_SPEED_SCALE,
    P_FWD,
    P_MAX_FWD,
    P_TURN,
    P_MAX_STEPS,
    P_HEADING_ZERO,
    P_WRAP_HEADING,
    P_BALL_RADIUS,
) = range(32)
P_SIZE = 32


def _room_texture(config: dict, key: str) -> str:
    name = config.get(key, {"floor_tex": FLOOR_TEX, "wall_tex": WALL_TEX, "ceil_tex": CEIL_TEX}[key])
    return ROOM_DEFAULTS[key] if name is None else str(name)


class World:
    """A configuration's parameter vector ``P`` and index tables, shared by both kernels."""

    def __init__(self, config: dict, textures: tuple[str, ...], colors: tuple[str, ...]):
        c = config
        get = c.get
        tex = textures.index
        P = np.zeros(P_SIZE)
        P[P_ROOM] = float(get("room_size", 12))
        P[P_GRID] = bool(get("grid_mode", False))
        P[P_CARDINAL] = bool(get("grid_cardinal_only", False))
        P[P_VMIN], P[P_VMAX] = int(get("grid_vel_min", -1)), int(get("grid_vel_max", 1))
        P[P_BOX_OVERLAP] = bool(get("box_allow_overlap", False))
        P[P_AGENT_OVERLAP] = bool(get("agent_box_allow_overlap", False))
        P[P_RANDOM_ORIENT] = bool(get("box_random_orientation", False))
        P[P_STATIC] = bool(get("blocks_static", False))
        P[P_TORUS] = bool(get("block_torus_wrap", False))
        P[P_SPAWN_BUFFER] = -1.0 if get("spawn_wall_buffer") is None else float(c["spawn_wall_buffer"])
        P[P_REPEAT] = bool(get("allow_color_repeat", False))
        P[P_ENSURE_BASE] = bool(get("ensure_base_palette", False))
        P[P_BLOCK_XY] = -1.0 if get("block_size_xy") is None else float(c["block_size_xy"])
        P[P_BLOCK_H] = -1.0 if get("block_height") is None else float(c["block_height"])
        P[P_AGENT_CENTER] = bool(get("agent_center_start", False))
        P[P_FLOOR_TEX] = tex(_room_texture(c, "floor_tex"))
        P[P_FLOOR_TEX_RANDOM] = bool(get("floor_tex_randomize", False))
        P[P_WALL_TEX] = tex(_room_texture(c, "wall_tex"))
        P[P_WALL_TEX_RANDOM] = bool(get("wall_tex_randomize", False))
        P[P_CEIL_TEX] = tex(_room_texture(c, "ceil_tex"))
        P[P_BOX_TEX] = -1 if get("box_tex") is None else tex(c["box_tex"])
        P[P_BOX_TEX_RANDOM] = bool(get("box_tex_randomize", False))
        P[P_BOX_AND_BALL] = bool(get("box_and_ball", False))
        P[P_SPEED_SCALE] = float(get("box_speed_scale", 1.0))
        # generate_videos.build_env: a fixed turn or forward step starts from DEFAULT_PARAMS.no_random().
        fixed = c["turn_step"] is not None or c["forward_step"] is not None
        P[P_FWD] = DEFAULT_FORWARD_STEP if c["forward_step"] is None else float(c["forward_step"])
        P[P_MAX_FWD] = P[P_FWD] if fixed else DEFAULT_FORWARD_MAX
        P[P_TURN] = DEFAULT_TURN_STEP if c["turn_step"] is None else float(c["turn_step"])
        P[P_MAX_STEPS] = math.inf if c["max_episode_steps"] is None else int(c["max_episode_steps"])
        P[P_HEADING_ZERO] = bool(c["heading_zero"])
        P[P_WRAP_HEADING] = bool(c["wrap_heading"])
        P[P_BALL_RADIUS] = BALL_RADIUS_PER_HEIGHT
        self.P = P
        self.room = int(P[P_ROOM])
        self.count_range = tuple(int(v) for v in c["num_blocks"])
        pool = get("color_pool")
        pool = list(colors) if pool is None else list(pool)
        self.pool = np.array([colors.index(n) for n in pool], np.int64)
        self.base_palette = np.array([colors.index(n) for n in BASE_PALETTE if n in pool], np.int64)
        self.floor_pool = np.array([tex(n) for n in FLOOR_POOL], np.int64)
        self.wall_pool = np.array([tex(n) for n in WALL_POOL], np.int64)
        self.box_pool = np.array([tex(n) for n in BOX_POOL], np.int64)
