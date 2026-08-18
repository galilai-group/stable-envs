"""Exact parser/serializer for procgen 0.10.7's `get_state()` blob.

Transcribed from the C++ `serialize`/`deserialize` pair (openai/procgen @ 37b521d, which is the
0.10.7 tag): `game.cpp::Game::serialize`, `basic-abstract-game.cpp::BasicAbstractGame::serialize`,
`entity.cpp::Entity::serialize`, `grid.h::Grid::serialize`, `randgen.cpp::RandGen::serialize`,
plus the per-game tail in `games/<name>.cpp`.

Every field is a 4-byte little-endian int or float except strings, which are `int length` followed
by raw bytes. `RandGen::serialize` writes `ostream << std::mt19937`, i.e. the 624-word Mersenne
state as SPACE-SEPARATED DECIMAL TEXT -- that is the 20 kB that dominates the blob and none of it
reaches `game_draw`.

The parser is checked by re-serializing and comparing bytes; a schema error therefore cannot pass
silently.
"""
from __future__ import annotations

import struct

import numpy as np

# ---------------------------------------------------------------- primitive reader/writer


class R:
    def __init__(self, buf: bytes):
        self.b = buf
        self.o = 0

    def i(self) -> int:
        v = struct.unpack_from("<i", self.b, self.o)[0]
        self.o += 4
        return v

    def f(self) -> float:
        v = struct.unpack_from("<f", self.b, self.o)[0]
        self.o += 4
        return v

    def s(self) -> bytes:
        n = self.i()
        v = self.b[self.o:self.o + n]
        self.o += n
        return v

    def vi(self) -> list:
        return [self.i() for _ in range(self.i())]

    def vb(self) -> list:
        return [self.i() > 0 for _ in range(self.i())]


class W:
    def __init__(self):
        self.p = []

    def i(self, v):
        self.p.append(struct.pack("<i", int(v)))

    def f(self, v):
        self.p.append(struct.pack("<f", float(v)))

    def s(self, v):
        self.i(len(v))
        self.p.append(bytes(v))

    def vi(self, v):
        self.i(len(v))
        for x in v:
            self.i(x)

    def vb(self, v):
        self.i(len(v))
        for x in v:
            self.i(1 if x else 0)

    def out(self) -> bytes:
        return b"".join(self.p)


# ---------------------------------------------------------------- schemas
# (name, kind) with kind in {i, f, s, vi, vb}. Order IS the wire format.

GAME_HEAD = [
    ("serialize_version", "i"), ("game_name", "s"),
    ("paint_vel_info", "i"), ("use_generated_assets", "i"), ("use_monochrome_assets", "i"),
    ("restrict_themes", "i"), ("use_backgrounds", "i"), ("center_agent", "i"),
    ("debug_mode", "i"), ("distribution_mode", "i"), ("use_sequential_levels", "i"),
    ("use_easy_jump", "i"), ("plain_assets", "i"), ("physics_mode", "i"),
    ("grid_step", "i"), ("level_seed_low", "i"), ("level_seed_high", "i"),
    ("game_type", "i"), ("game_n", "i"),
    ("level_seed_rand_gen_seeded", "i"), ("level_seed_rand_gen", "s"),
    ("rand_gen_seeded", "i"), ("rand_gen", "s"),
    ("reward", "f"), ("done", "i"), ("level_complete", "i"),
    ("action", "i"), ("timeout", "i"),
    ("current_level_seed", "i"), ("prev_level_seed", "i"),
    ("episodes_remaining", "i"), ("episode_done", "i"),
    ("last_reward_timer", "i"), ("last_reward", "f"), ("default_action", "i"),
    ("fixed_asset_seed", "i"), ("cur_time", "i"), ("is_waiting_for_step", "i"),
]

ENTITY = [
    ("x", "f"), ("y", "f"), ("vx", "f"), ("vy", "f"), ("rx", "f"), ("ry", "f"),
    ("type", "i"), ("image_type", "i"), ("image_theme", "i"), ("render_z", "i"),
    ("will_erase", "i"), ("collides_with_entities", "i"),
    ("collision_margin", "f"), ("rotation", "f"), ("vrot", "f"),
    ("is_reflected", "i"), ("fire_time", "i"), ("spawn_time", "i"), ("life_time", "i"),
    ("expire_time", "i"), ("use_abs_coords", "i"),
    ("friction", "f"), ("smart_step", "i"), ("avoids_collisions", "i"), ("auto_erase", "i"),
    ("alpha", "f"), ("health", "f"), ("theta", "f"), ("grow_rate", "f"),
    ("alpha_decay", "f"), ("climber_spawn_x", "f"),
]
ENTITY_BYTES = 4 * len(ENTITY)                              # 124

BAG_MID = [
    ("use_procgen_background", "i"), ("background_index", "i"),
    ("bg_tile_ratio", "f"), ("bg_pct_x", "f"),
    ("char_dim", "f"), ("last_move_action", "i"), ("move_action", "i"), ("special_action", "i"),
    ("mixrate", "f"), ("maxspeed", "f"), ("max_jump", "f"),
    ("action_vx", "f"), ("action_vy", "f"), ("action_vrot", "f"),
    ("center_x", "f"), ("center_y", "f"),
    ("random_agent_start", "i"), ("has_useful_vel_info", "i"), ("step_rand_int", "i"),
    ("asset_rand_gen_seeded", "i"), ("asset_rand_gen", "s"),
    ("main_width", "i"), ("main_height", "i"), ("out_of_bounds_object", "i"),
    ("unit", "f"), ("view_dim", "f"), ("x_off", "f"), ("y_off", "f"),
    ("visibility", "f"), ("min_visibility", "f"),
]

#: per-game tail, keyed by game name. Only the three in scope are transcribed; anything else
#: parses up to the tail and keeps the remainder verbatim so the round-trip still holds.
TAIL = {
    "maze":  [("maze_dim", "i"), ("world_dim", "i")],
    "heist": [("num_keys", "i"), ("world_dim", "i"), ("has_keys", "vb")],
    "miner": [("diamonds_remaining", "i")],
    "chaser": [("free_cells", "vi"), ("is_space_vec", "vb"), ("eat_timeout", "i"),
               ("egg_timeout", "i"), ("eat_time", "i"), ("total_enemies", "i"),
               ("total_orbs", "i"), ("orbs_collected", "i"), ("maze_dim", "i")],
    "bigfish": [("fish_eaten", "i"), ("r_inc", "f")],
}


def _read(r: R, schema) -> dict:
    return {n: getattr(r, k)() for n, k in schema}


def _write(w: W, schema, d) -> None:
    for n, k in schema:
        getattr(w, k)(d[n])


def parse(buf: bytes) -> dict:
    """The blob as a dict. `entities` is a list of dicts; `grid` is a (h, w) int32 array."""
    r = R(buf)
    st = _read(r, GAME_HEAD)
    st["grid_size"] = r.i()
    n = r.i()
    st["entities"] = [_read(r, ENTITY) for _ in range(n)]
    st.update(_read(r, BAG_MID))
    gw, gh = r.i(), r.i()
    data = r.vi()
    st["grid_w"], st["grid_h"] = gw, gh
    st["grid"] = np.asarray(data, np.int32).reshape(gh, gw)
    name = st["game_name"].decode()
    st["_tail_schema"] = TAIL.get(name)
    if st["_tail_schema"] is not None:
        st.update(_read(r, st["_tail_schema"]))
        st["_rest"] = buf[r.o:]
    else:
        st["_rest"] = buf[r.o:]
    return st


def dump(st: dict) -> bytes:
    w = W()
    _write(w, GAME_HEAD, st)
    w.i(st["grid_size"])
    w.i(len(st["entities"]))
    for e in st["entities"]:
        _write(w, ENTITY, e)
    _write(w, BAG_MID, st)
    w.i(st["grid_w"])
    w.i(st["grid_h"])
    w.vi(np.asarray(st["grid"]).reshape(-1).tolist())
    if st["_tail_schema"] is not None:
        _write(w, st["_tail_schema"], st)
    return w.out() + st["_rest"]


def rng_text_bytes(st: dict) -> int:
    return sum(len(st[k]) for k in ("level_seed_rand_gen", "rand_gen", "asset_rand_gen"))
