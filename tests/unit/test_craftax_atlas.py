"""The composited-tile atlas must reproduce the per-frame blend exactly.

``CraftaxClassic._render_blend`` is the per-frame arithmetic; every case asserts the atlas gather
matches it on every pixel, not approximately. Cases that would each hide an invisible bug:

* ``mobs_visible = 0`` is folded into mob id 0, which is the identity only because mob 0's alpha
  table is exactly zero, so the table is checked.
* ``player_visible = 0`` and every non-centre tile share slot :data:`NO_PLAYER`, which must be the
  block texture untouched, not the texture blended with a transparent sprite.
* ``is_sleeping`` leaves the uint8 fast path because the desaturation multiplies before rounding;
  the fuzz mixes sleeping and awake rows inside one batch.
* The table is a cache: rendering must not write through it, so its bytes are checked afterwards.

The fuzz includes out-of-range class ids, which ``render`` wraps.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stable_envs._core.style import StyleTensor
from stable_envs.envs.craftax.classic import (
    ATLAS_MAX_TILE_PX,
    N_BLOCK,
    N_COUNT,
    N_FACING,
    N_KEY_HALVES,
    N_MOB,
    N_PLAYER,
    N_STRIP,
    N_TILE,
    NO_PLAYER,
    OBS_H,
    OBS_W,
    STRIP_H,
    CraftaxClassic,
    _Assets,
    _Composite,
)

#: 7 and 16 are the sizes craftax ships textures for; 32 and 64 exercise the nearest-resample path
#: and the top of the atlas budget; 65 is the first size that must fall back to the blend path.
SIZES = (7, 16, 32, 64)


def _env(size: int, num_envs: int = 8) -> CraftaxClassic:
    return CraftaxClassic(num_envs=num_envs, size=size, device="cpu")


def _with(st: np.ndarray, **kw) -> np.ndarray:
    """``sample_states`` output with ``is_sleeping`` and/or ``light_level`` pinned."""
    out = np.array(st, np.float32, copy=True)
    j = OBS_H * OBS_W * 2 + 1 + N_STRIP
    if "is_sleeping" in kw:
        out[:, j] = kw["is_sleeping"]
    if "light" in kw:
        out[:, j + 1] = kw["light"]
    return out


def _style(env: CraftaxClassic, n: int, **fields) -> torch.Tensor:
    """A ``(n, style_dim)`` style values tensor with named variations overwritten per row."""
    base = env.resolve_style(None, None, batch=n)
    vals = base.values.clone()
    for path, arr in fields.items():
        off, width = base.layout[path]
        vals[:, off : off + width] = torch.as_tensor(np.asarray(arr, np.float32)).reshape(n, width)
    return vals


def _cases(rng: np.random.Generator, env: CraftaxClassic, n: int):
    """(name, states, style values) over the reachable states and the deliberately unreachable ones."""
    dim = env.state_dim
    states = {
        "sampled": env.sample_states(n, rng),
        "uniform_ids": np.stack(
            [
                np.concatenate(
                    [
                        rng.integers(0, N_BLOCK, OBS_H * OBS_W),
                        rng.integers(0, N_MOB, OBS_H * OBS_W),
                        rng.integers(0, N_FACING, 1),
                        rng.integers(0, N_COUNT, N_STRIP),
                        rng.integers(0, 2, 1),
                        rng.random(1),  # light_level anywhere in [0, 1]
                        rng.integers(0, 65536, N_KEY_HALVES),
                    ]
                ).astype(np.float32)
                for _ in range(n)
            ]
        ),
        # every row asleep, and every row awake, so neither path is only ever seen mixed
        "all_asleep": _with(env.sample_states(n, rng), is_sleeping=1.0),
        "all_awake": _with(env.sample_states(n, rng), is_sleeping=0.0),
        # the three lighting bands, each on its own, and one batch holding all of them
        "all_night": _with(env.sample_states(n, rng), light=0.1),
        "all_dusk": _with(env.sample_states(n, rng), light=0.75),
        "all_daylight": _with(env.sample_states(n, rng), light=1.0, is_sleeping=0.0),
        "mixed_light": _with(env.sample_states(n, rng), light=np.where(rng.random(n) < 0.5, 1.0, rng.random(n))),
        # out of range in both directions, and fractional so the round() is exercised too
        "out_of_range": (rng.random((n, dim)).astype(np.float32) * 80.0 - 30.0),
    }
    styles = {
        "default": {},
        "gain": {"channel_gain": rng.random((n, 3)) * 2.0},
        "gain_extreme": {"channel_gain": np.repeat([[0.0, 1.0, 2.0]], n, 0)},
        "no_player": {"player_visible": np.zeros((n, 1))},
        "no_mobs": {"mobs_visible": np.zeros((n, 1))},
        "nothing_visible": {"player_visible": np.zeros((n, 1)), "mobs_visible": np.zeros((n, 1))},
        # the case a per-environment style exists for: one batch, several appearances
        "mixed_per_env": {
            "player_visible": rng.integers(0, 2, (n, 1)),
            "mobs_visible": rng.integers(0, 2, (n, 1)),
            "channel_gain": np.where(rng.random((n, 3)) < 0.5, 1.0, rng.random((n, 3)) * 2.0),
        },
    }
    for sname, st in states.items():
        for vname, fields in styles.items():
            yield f"{sname}/{vname}", np.asarray(st, np.float32), _style(env, n, **fields)


def _decode(env: CraftaxClassic, states: np.ndarray):
    """The class-id decode ``render`` does, so the two paths are fed identical integers."""
    s = torch.as_tensor(states, dtype=torch.float32).reshape(-1, env.state_dim)
    return env._decode(s, s.shape[0])


@pytest.mark.parametrize("size", SIZES)
def test_atlas_path_equals_the_per_frame_composite(size):
    """Same pixels, not similar ones."""
    rng = np.random.default_rng(20260913 + size)
    env = _env(size)
    n = 8
    for name, states, vals in _cases(rng, env, n):
        dec = _decode(env, states)
        sty = StyleTensor(vals, env.resolve_style(None, None, batch=n)._layout_list())
        fast = env._render_atlas(*dec, size, sty, None)
        slow = env._render_blend(*dec, size, sty)
        assert fast.shape == slow.shape == (n, (OBS_H + STRIP_H) * size, OBS_W * size, 3)
        diff = (fast.int() - slow.int()).abs().max().item()
        assert diff == 0, f"size={size} case={name}: max|diff| = {diff}, atlas != per-frame blend"


def test_mob_zero_alpha_is_exactly_zero():
    """``mobs_visible = 0`` is folded into mob id 0, which is the identity only if mob 0's alpha is exactly zero."""
    for size in SIZES:
        *_, malpha = _Assets.get(torch.device("cpu"), size)
        assert torch.equal(malpha[0], torch.zeros_like(malpha[0])), (
            f"mob id 0 has a non-zero alpha at size {size}; hiding mobs would not be a no-op"
        )


@pytest.mark.parametrize("size", (7, 16))
def test_no_player_slot_is_the_block_texture_untouched(size):
    """Non-centre tiles and ``player_visible = 0`` both use NO_PLAYER, which must add nothing."""
    block, *_ = _Assets.get(torch.device("cpu"), size)
    f32 = _Composite.get(torch.device("cpu"), size, want_float=True)[0]
    # only the world half of the table has this shape; the strip cells follow it
    world = f32[: (N_PLAYER + 1) * N_BLOCK * N_MOB * size]
    tiles = world.reshape(N_PLAYER + 1, N_BLOCK, N_MOB, size, size, 3)
    assert torch.equal(tiles[NO_PLAYER, :, 0], block), (
        "the NO_PLAYER slot with no mob is not the block texture; the identity is broken"
    )


@pytest.mark.parametrize("size", (7, 16))
def test_atlas_is_not_written_through_by_a_render(size):
    """The table is shared across every render in the process; a write through it is silent."""
    env = _env(size)
    dev = torch.device("cpu")
    f32, u8 = _Composite.get(dev, size, want_float=True)
    before_f32, before_u8 = f32.clone(), u8.clone()
    states = env.sample_states(8, np.random.default_rng(4))
    env._render(states)
    env._render(states, variation={"channel_gain": [0.25, 1.5, 2.0]})
    env._render(states, variation={"player_visible": 0, "mobs_visible": 0})
    after_f32, after_u8 = _Composite.get(dev, size, want_float=True)
    assert torch.equal(before_f32, after_f32) and torch.equal(before_u8, after_u8), (
        "rendering mutated the composited atlas; every later frame in the process is wrong"
    )


def test_atlas_shape_and_budget():
    """The table has 976 tiles and is not built above the tile-size budget."""
    from stable_envs.envs.craftax.classic import N_COMPOSITE, STRIP_BASE

    assert N_COMPOSITE == (N_PLAYER + 1) * N_BLOCK * N_MOB == 816
    assert N_TILE == STRIP_BASE + N_STRIP * N_COUNT == 976
    f32, u8 = _Composite.get(torch.device("cpu"), 16, want_float=True)
    assert f32.shape == (N_TILE * 16, 16 * 3) and u8.shape == f32.shape
    assert u8.numel() == N_TILE * 16 * 16 * 3
    # above the budget render must not build a table at all
    env = _env(ATLAS_MAX_TILE_PX + 1, num_envs=1)
    states = env.sample_states(1, np.random.default_rng(0))
    img = env._render(states)
    assert img.shape == (1, (OBS_H + STRIP_H) * (ATLAS_MAX_TILE_PX + 1), OBS_W * (ATLAS_MAX_TILE_PX + 1), 3)
    assert (torch.device("cpu").__str__(), ATLAS_MAX_TILE_PX + 1) not in _Composite._cache, (
        "a tile size past the budget still baked an atlas"
    )


@pytest.mark.parametrize("size", (7, 16))
def test_batch_order_and_chunk_invariance_on_the_gather(size):
    """Folding the visibility flags into the gather index must not couple rows to each other."""
    env = _env(size, num_envs=12)
    rng = np.random.default_rng(99)
    states = env.sample_states(12, rng)
    vals = _style(env, 12, player_visible=rng.integers(0, 2, (12, 1)), mobs_visible=rng.integers(0, 2, (12, 1)))
    whole = env._render(states, style=vals)
    # one row at a time
    per_row = torch.cat([env._render(states[i : i + 1], style=vals[i : i + 1]) for i in range(12)])
    assert torch.equal(whole, per_row), "a row's frame depends on what else is in the batch"
    # chunked
    chunks = torch.cat([env._render(states[i : i + 5], style=vals[i : i + 5]) for i in range(0, 12, 5)])
    assert torch.equal(whole, chunks), "chunking the batch changes the frames"
    # permuted
    perm = np.array([7, 0, 11, 3, 2, 9, 1, 8, 5, 4, 10, 6])
    assert torch.equal(env._render(states[perm], style=vals[perm]), whole[perm]), (
        "permuting the batch changes the frames"
    )


@pytest.mark.parametrize("size", (7, 16))
def test_out_buffer_receives_the_frame_and_is_the_returned_tensor(size):
    """``out=`` is written in place on both the uint8 gather and the gain path."""
    env = _env(size, num_envs=5)
    states = env.sample_states(5, np.random.default_rng(12))
    shape = (5, (OBS_H + STRIP_H) * size, OBS_W * size, 3)
    for variation in ({}, {"channel_gain": [0.5, 1.0, 1.75]}):
        buf = torch.full(shape, 7, dtype=torch.uint8)
        got = env._render(states, variation=variation or None, out=buf)
        assert got.shape == shape
        assert torch.equal(buf, got)
        assert torch.equal(buf, env._render(states, variation=variation or None)), (
            "the out= path and the allocating path disagree"
        )


def test_leading_dimensions_are_preserved():
    """``(...,127)`` in, ``(...,H,W,3)`` out — the reshape around the gather must not flatten them."""
    env = _env(16, num_envs=6)
    states = env.sample_states(6, np.random.default_rng(1)).reshape(2, 3, env.state_dim)
    img = env._render(states)
    assert img.shape == (2, 3, (OBS_H + STRIP_H) * 16, OBS_W * 16, 3)
    assert torch.equal(
        img.reshape(6, (OBS_H + STRIP_H) * 16, OBS_W * 16, 3), env._render(states.reshape(6, env.state_dim))
    )


def test_the_table_cache_is_bounded():
    """A caller sweeping tile sizes must not retain one table per size for the life of the process."""
    _Composite._cache.clear()
    env = _env(16, num_envs=1)
    states = env.sample_states(1, np.random.default_rng(0))
    for size in range(4, 4 + _Composite._MAX_ENTRIES + 3):
        env._render(states, size=size)
        assert len(_Composite._cache) <= _Composite._MAX_ENTRIES, (
            f"the composited-atlas cache grew to {len(_Composite._cache)} entries"
        )
    # and the frames are still right after an eviction
    for size in (7, 16):
        dec = _decode(env, states)
        sty = env.resolve_style(None, None, batch=1)
        assert torch.equal(env._render_atlas(*dec, size, sty, None), env._render_blend(*dec, size, sty))
