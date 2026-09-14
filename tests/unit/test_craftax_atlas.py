"""The Craftax composited-tile atlas is the pre-Step-6 per-frame composite, precomputed.

Step 6 replaced a dozen ``B x 7t x 9t x 3`` float32 temporaries per frame with a gather from a
680-entry table of already-composited tiles. That is only admissible because the table holds exactly
what the per-frame arithmetic produced, so this module is the differential test that says so —
``CraftaxClassic._render_blend`` is the pre-Step-6 render body kept verbatim, and every case below
asserts the atlas path reproduces it on every pixel rather than approximately.

Three of the assertions are not about speed at all and would each hide a plausible, invisible bug:

* ``mobs_visible = 0`` is folded into mob id 0 rather than into a zeroed alpha. That is the identity
  only because mob 0's alpha table is exactly zero, so the table is checked, not assumed.
* ``player_visible = 0`` and every non-centre tile share player slot 4, which must be the block
  texture untouched — not "the block texture blended with a transparent sprite", which would be a
  different float32 expression.
* The table is a cache. A render that wrote through it would corrupt every later frame in the
  process, and nothing downstream would flag it, so the table's bytes are checked after rendering.

The fuzz deliberately includes out-of-range class ids: ``render`` wraps them (see the module
docstring in ``stable_envs/envs/craftax.py``) and wrapping has to survive the new index arithmetic.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from stable_envs.core.style import StyleTensor
from stable_envs.envs.craftax import (ATLAS_MAX_TILE_PX, N_BLOCK, N_COMPOSITE, N_MOB, N_PLAYER,
                                      NO_PLAYER, OBS_H, OBS_W, CraftaxClassic, _Assets, _Composite)

#: 7 and 16 are the sizes craftax ships textures for; 32 and 64 exercise the nearest-resample path
#: and the top of the atlas budget; 65 is the first size that must fall back to the blend path.
SIZES = (7, 16, 32, 64)


def _env(size: int, batch: int = 8) -> CraftaxClassic:
    return CraftaxClassic(batch=batch, size=size, device="cpu")


def _style(env: CraftaxClassic, n: int, **fields) -> torch.Tensor:
    """A ``(n, style_dim)`` style values tensor with named variations overwritten per row."""
    base = env.resolve_style(None, None, batch=n)
    vals = base.values.clone()
    for path, arr in fields.items():
        off, width = base.layout[path]
        vals[:, off:off + width] = torch.as_tensor(np.asarray(arr, np.float32)).reshape(n, width)
    return vals


def _cases(rng: np.random.Generator, env: CraftaxClassic, n: int):
    """(name, states, style values) over the reachable states and the deliberately unreachable ones."""
    dim = env.state_dim
    states = {
        "sampled": env.sample_states(n, rng),
        "uniform_ids": np.stack([
            np.concatenate([rng.integers(0, N_BLOCK, OBS_H * OBS_W),
                            rng.integers(0, N_MOB, OBS_H * OBS_W),
                            rng.integers(0, N_PLAYER, 1)]).astype(np.float32)
            for _ in range(n)]),
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
        "mixed_per_env": {"player_visible": rng.integers(0, 2, (n, 1)),
                          "mobs_visible": rng.integers(0, 2, (n, 1)),
                          "channel_gain": np.where(rng.random((n, 3)) < 0.5, 1.0,
                                                   rng.random((n, 3)) * 2.0)},
    }
    for sname, st in states.items():
        for vname, fields in styles.items():
            yield f"{sname}/{vname}", np.asarray(st, np.float32), _style(env, n, **fields)


def _decode(env: CraftaxClassic, states: np.ndarray):
    """The class-id decode ``render`` does, so the two paths are fed identical integers."""
    s = torch.as_tensor(states, dtype=torch.float32).reshape(-1, env.state_dim)
    n, nb = s.shape[0], OBS_H * OBS_W
    return (s[:, :nb].round().remainder(N_BLOCK).long().view(n, OBS_H, OBS_W),
            s[:, nb:2 * nb].round().remainder(N_MOB).long().view(n, OBS_H, OBS_W),
            s[:, 2 * nb].round().remainder(N_PLAYER).long())


@pytest.mark.parametrize("size", SIZES)
def test_atlas_path_equals_the_pre_step6_per_frame_composite(size):
    """The differential test the patch rests on: same pixels, not similar ones."""
    rng = np.random.default_rng(20260913 + size)
    env = _env(size)
    n = 8
    for name, states, vals in _cases(rng, env, n):
        blk, mob, pid = _decode(env, states)
        sty = StyleTensor(vals, env.resolve_style(None, None, batch=n)._layout_list())
        fast = env._render_atlas(blk, mob, pid, size, sty, None)
        slow = env._render_blend(blk, mob, pid, size, sty)
        assert fast.shape == slow.shape == (n, OBS_H * size, OBS_W * size, 3)
        diff = (fast.int() - slow.int()).abs().max().item()
        assert diff == 0, f"size={size} case={name}: max|diff| = {diff}, atlas != per-frame blend"


def test_mob_zero_alpha_is_exactly_zero():
    """``mobs_visible = 0`` is folded into mob id 0. That is the identity only if this holds.

    The pre-Step-6 path multiplied the mob alpha by the flag, so hiding mobs computed
    ``px*(1-0) + tex*0``. The atlas path instead looks up the "no mob" composite, which computes
    ``px*(1-malpha[0]) + tex[0]*malpha[0]``. The two agree for every possible texture exactly when
    ``malpha[0]`` is zero — not small, zero — because a 1-ULP alpha would move a pixel by less than
    half a level in most tiles and by a whole level in some.
    """
    for size in SIZES:
        *_, malpha = _Assets.get(torch.device("cpu"), size)
        assert torch.equal(malpha[0], torch.zeros_like(malpha[0])), \
            f"mob id 0 has a non-zero alpha at size {size}; hiding mobs is no longer a no-op"


@pytest.mark.parametrize("size", (7, 16))
def test_player_slot_4_is_the_block_texture_untouched(size):
    """Non-centre tiles and ``player_visible = 0`` both use slot 4, which must add nothing."""
    block, *_ = _Assets.get(torch.device("cpu"), size)
    f32 = _Composite.get(torch.device("cpu"), size, want_float=True)[0]
    tiles = f32.reshape(N_PLAYER + 1, N_BLOCK, N_MOB, size, size, 3)
    assert torch.equal(tiles[NO_PLAYER, :, 0], block), \
        "player slot 4 with no mob is not the block texture; the 'no player' identity is broken"


@pytest.mark.parametrize("size", (7, 16))
def test_atlas_is_not_written_through_by_a_render(size):
    """The table is shared across every render in the process; a write through it is silent."""
    env = _env(size)
    dev = torch.device("cpu")
    f32, u8 = _Composite.get(dev, size, want_float=True)
    before_f32, before_u8 = f32.clone(), u8.clone()
    states = env.sample_states(8, np.random.default_rng(4))
    env.render(states)
    env.render(states, variation={"channel_gain": [0.25, 1.5, 2.0]})
    env.render(states, variation={"player_visible": 0, "mobs_visible": 0})
    after_f32, after_u8 = _Composite.get(dev, size, want_float=True)
    assert torch.equal(before_f32, after_f32) and torch.equal(before_u8, after_u8), \
        "rendering mutated the composited atlas; every later frame in the process is now wrong"


def test_atlas_shape_and_budget():
    """The table is 680 tiles, and the size at which it stops being a table is documented."""
    assert N_COMPOSITE == (N_PLAYER + 1) * N_BLOCK * N_MOB == 680
    f32, u8 = _Composite.get(torch.device("cpu"), 16, want_float=True)
    assert f32.shape == (N_COMPOSITE * 16, 16 * 3) and u8.shape == f32.shape
    assert u8.numel() == N_COMPOSITE * 16 * 16 * 3
    # above the budget render must not build a table at all
    env = _env(ATLAS_MAX_TILE_PX + 1, batch=1)
    states = env.sample_states(1, np.random.default_rng(0))
    img = env.render(states)
    assert img.shape == (1, OBS_H * (ATLAS_MAX_TILE_PX + 1), OBS_W * (ATLAS_MAX_TILE_PX + 1), 3)
    assert (torch.device("cpu").__str__(), ATLAS_MAX_TILE_PX + 1) not in _Composite._cache, \
        "a tile size past the budget still baked an atlas"


@pytest.mark.parametrize("size", (7, 16))
def test_batch_order_and_chunk_invariance_on_the_gather(size):
    """Folding the visibility flags into the gather index must not couple rows to each other."""
    env = _env(size, batch=12)
    rng = np.random.default_rng(99)
    states = env.sample_states(12, rng)
    vals = _style(env, 12, player_visible=rng.integers(0, 2, (12, 1)),
                  mobs_visible=rng.integers(0, 2, (12, 1)))
    whole = env.render(states, style=vals)
    # one row at a time
    per_row = torch.cat([env.render(states[i:i + 1], style=vals[i:i + 1]) for i in range(12)])
    assert torch.equal(whole, per_row), "a row's frame depends on what else is in the batch"
    # chunked
    chunks = torch.cat([env.render(states[i:i + 5], style=vals[i:i + 5]) for i in range(0, 12, 5)])
    assert torch.equal(whole, chunks), "chunking the batch changes the frames"
    # permuted
    perm = np.array([7, 0, 11, 3, 2, 9, 1, 8, 5, 4, 10, 6])
    assert torch.equal(env.render(states[perm], style=vals[perm]), whole[perm]), \
        "permuting the batch changes the frames"


@pytest.mark.parametrize("size", (7, 16))
def test_out_buffer_receives_the_frame_and_is_the_returned_tensor(size):
    """``out=`` is written in place on both the uint8 gather and the gain path."""
    env = _env(size, batch=5)
    states = env.sample_states(5, np.random.default_rng(12))
    shape = (5, OBS_H * size, OBS_W * size, 3)
    for variation in ({}, {"channel_gain": [0.5, 1.0, 1.75]}):
        buf = torch.full(shape, 7, dtype=torch.uint8)
        got = env.render(states, variation=variation or None, out=buf)
        assert got.shape == shape
        assert torch.equal(buf, got)
        assert torch.equal(buf, env.render(states, variation=variation or None)), \
            "the out= path and the allocating path disagree"


def test_leading_dimensions_are_preserved():
    """``(...,127)`` in, ``(...,H,W,3)`` out — the reshape around the gather must not flatten them."""
    env = _env(16, batch=6)
    states = env.sample_states(6, np.random.default_rng(1)).reshape(2, 3, env.state_dim)
    img = env.render(states)
    assert img.shape == (2, 3, OBS_H * 16, OBS_W * 16, 3)
    assert torch.equal(img.reshape(6, OBS_H * 16, OBS_W * 16, 3),
                       env.render(states.reshape(6, env.state_dim)))
