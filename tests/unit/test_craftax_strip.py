"""The inventory strip and the fifth (sleeping) player sprite.

These are the two things Step 7 added to the world-layer renderer, and both are the kind of change
that produces a plausible wrong picture rather than an obvious one: a strip cell drawn from the
wrong count still looks like an inventory, and a sleeping player drawn without the desaturation (or
desaturated without the sprite) still looks like Craftax. So the assertions here are structural —
what the strip is a function of, and what it is *not* a function of — rather than eyeball checks.

No jax: the parity run against the official renderer lives in the certification harness, and these
run everywhere.
"""
from __future__ import annotations

import numpy as np
import pytest
import torch

from stable_envs.envs.craftax import (
    N_BLOCK, N_COUNT, N_FACING, N_MOB, N_STRIP, N_TILE, NO_PLAYER, OBS_H, OBS_W, PLAYER_ASLEEP,
    STRIP_BASE, STRIP_H, STRIP_LAYOUT, STRIP_NAMES, CraftaxClassic, _Composite, _strip_cells,
    strip_geometry)

SIZES = (7, 16, 32)
DEV = torch.device("cpu")


@pytest.fixture(scope="module")
def env():
    return CraftaxClassic(device="cpu")


@pytest.fixture(scope="module")
def states(env):
    return env.sample_states(24, np.random.default_rng(31337))


def _set(st, **kw):
    """A copy of ``st`` with named strip counts / is_sleeping overwritten."""
    out = np.array(st, np.float32, copy=True)
    base = OBS_H * OBS_W * 2 + 1
    for k, v in kw.items():
        if k == "is_sleeping":
            out[:, base + N_STRIP] = v
        else:
            out[:, base + STRIP_NAMES.index(k)] = v
    return out


# ---------------------------------------------------------------- the table


@pytest.mark.parametrize("t", SIZES)
def test_the_strip_lives_in_the_same_table_as_the_world_tiles(t):
    """One table, one gather. A strip cell is a tile indexed by (cell, count), nothing more."""
    tab = _Composite.get(DEV, t, want_float=False)[1]
    assert tab.shape == (N_TILE * t, t * 3)
    assert STRIP_BASE == (N_FACING + 2) * N_BLOCK * N_MOB == 816
    assert N_TILE == STRIP_BASE + N_STRIP * N_COUNT == 976


@pytest.mark.parametrize("t", SIZES)
def test_a_zero_count_cell_is_exactly_black(t):
    """Which is why the two cells the reference never writes need no table entry of their own."""
    cells = _strip_cells(DEV, t)
    zeros = cells.view(N_STRIP, N_COUNT, t, t, 3)[:, 0]
    assert float(zeros.abs().max()) == 0.0
    # and every non-zero count is not black, or the count would not reach the frame
    for c in range(N_STRIP):
        for v in range(1, N_COUNT):
            assert float(cells.view(N_STRIP, N_COUNT, t, t, 3)[c, v].abs().max()) > 0.0, (c, v)


@pytest.mark.parametrize("t", SIZES)
def test_every_cell_count_pair_is_a_distinct_picture(t):
    """A count that does not change its cell's picture is a count the frame cannot show."""
    cells = _strip_cells(DEV, t).view(N_STRIP, N_COUNT, -1)
    for c in range(N_STRIP):
        seen = {cells[c, v].numpy().tobytes() for v in range(N_COUNT)}
        assert len(seen) == N_COUNT, f"cell {STRIP_NAMES[c]} at t={t} collapses counts"


@pytest.mark.parametrize("t", SIZES)
def test_a_cells_writes_stay_inside_its_own_tile(t):
    """The property that makes a per-cell table a valid factorisation of sequential .at[] writes.

    The reference draws the strip as 32 in-place updates in sequence. That is order-independent —
    and therefore a table — only if no two cells touch the same pixel. Checked here on the geometry
    the renderer actually uses, at every size it will be asked for.
    """
    lo, hi, dlo, dhi = strip_geometry(t)
    assert 0 <= lo < hi <= t, (t, lo, hi)
    assert 0 <= dlo < dhi <= t, (t, dlo, dhi)
    occ = np.zeros((STRIP_H * t, OBS_W * t), np.int32)
    for (r, c) in STRIP_LAYOUT:
        occ[r * t + lo:r * t + hi, c * t + lo:c * t + hi] += 1
        occ[r * t + dlo:r * t + dhi, c * t + dlo:c * t + dhi] += 1
    assert occ.max() <= 2, f"strip cells overlap at t={t}"


def test_the_strip_atlas_resamples_by_exact_replication():
    """At 2S the strip must be the S strip with every pixel doubled, or the frozen
    ``DOWNSAMPLE_EXACT_FRAC`` moves for a reason that is not the renderer.

    Craftax's own strip geometry does NOT scale by two — ``int(0.8*bps)`` is 12 at 16 px and 25 at
    32 px, ``int(0.6*bps)`` is 9 and 19 — which is exactly why the finished cell is resampled rather
    than re-laid-out. This asserts the consequence.
    """
    assert int(0.8 * 32) != 2 * int(0.8 * 16) and int(0.6 * 32) != 2 * int(0.6 * 16)
    small = _strip_cells(DEV, 16)
    big = _strip_cells(DEV, 32)
    assert torch.equal(big, small.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2))


# ---------------------------------------------------------------- the frame


@pytest.mark.parametrize("t", SIZES)
def test_the_frame_has_the_strip_below_the_world(env, states, t):
    f = env.render(states, size=t)
    assert tuple(f.shape) == (len(states), (OBS_H + STRIP_H) * t, OBS_W * t, 3)
    assert f.dtype == torch.uint8


@pytest.mark.parametrize("t", (7, 16))
def test_each_count_reaches_its_own_cell_and_no_other(env, states, t):
    """Changing one count must change exactly one cell of the strip and nothing else in the frame.

    This is the assertion that catches a transposed layout, an off-by-one in the cell order, or a
    count wired to the wrong field — all of which still draw a plausible inventory.
    """
    w = OBS_H * t
    lay = {i: STRIP_LAYOUT[i] for i in range(N_STRIP)}
    for i, name in enumerate(STRIP_NAMES):
        a = env.render(_set(states, **{name: 0}), size=t).numpy()
        b = env.render(_set(states, **{name: 7}), size=t).numpy()
        d = (np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1) > 0)
        assert d[:, :w].sum() == 0, f"{name} reached the world layer"
        r, c = lay[i]
        mine = d[:, w + r * t:w + (r + 1) * t, c * t:(c + 1) * t]
        assert mine.any(), f"{name} did not reach its own cell {lay[i]}"
        rest = d[:, w:].sum() - mine.sum()
        assert rest == 0, f"{name} reached {rest} pixels outside cell {lay[i]}"


@pytest.mark.parametrize("t", (7, 16))
def test_the_two_unwritten_cells_stay_black_whatever_the_state(env, states, t):
    w = OBS_H * t
    written = set(STRIP_LAYOUT)
    f = env.render(states, size=t).numpy()
    for r in range(STRIP_H):
        for c in range(OBS_W):
            if (r, c) in written:
                continue
            patch = f[:, w + r * t:w + (r + 1) * t, c * t:(c + 1) * t]
            assert int(patch.max()) == 0, f"cell ({r},{c}) is written but the reference never writes it"


@pytest.mark.parametrize("t", (7, 16))
def test_the_strip_is_not_lit_and_not_desaturated(env, states, t):
    """The reference concatenates the strip AFTER the lighting chain, so sleep cannot touch it.

    Crafter's port found this by failing (`docs/research/crafter.md`); here it is asserted.
    """
    w = OBS_H * t
    awake = env.render(_set(states, is_sleeping=0.0), size=t).numpy()
    asleep = env.render(_set(states, is_sleeping=1.0), size=t).numpy()
    assert np.array_equal(awake[:, w:], asleep[:, w:]), "sleeping changed the inventory strip"
    assert not np.array_equal(awake[:, :w], asleep[:, :w]), "sleeping did not change the world layer"


# ---------------------------------------------------------------- the sleeping player


@pytest.mark.parametrize("t", (7, 16))
def test_sleeping_changes_the_centre_tile_sprite_and_desaturates_everything_else(env, states, t):
    """Both halves. A renderer with only one of them draws a plausible frame and a wrong one."""
    w = OBS_H * t
    cr, cc = OBS_H // 2, OBS_W // 2
    awake = env.render(_set(states, is_sleeping=0.0), size=t).numpy()[:, :w]
    asleep = env.render(_set(states, is_sleeping=1.0), size=t).numpy()[:, :w]

    # (a) the desaturation reaches the whole world layer, not only the player's tile
    d = np.abs(awake.astype(np.int16) - asleep.astype(np.int16)).max(-1) > 0
    off_centre = d.copy()
    off_centre[:, cr * t:(cr + 1) * t, cc * t:(cc + 1) * t] = False
    assert off_centre.any(), "is_sleeping did not desaturate outside the player's tile"

    # (b) a sleeping world row is grey-plus-blue by construction: R == G everywhere
    assert np.array_equal(asleep[..., 0], asleep[..., 1]), "the sleep transform is not a luminance"
    assert (asleep[..., 2].astype(np.int16) >= asleep[..., 0].astype(np.int16)).all()

    # (c) the SPRITE changed too, and not only the tint: composite the awake centre tile through the
    #     same desaturation and it must still differ from the sleeping one
    a = awake[:, cr * t:(cr + 1) * t, cc * t:(cc + 1) * t].astype(np.float64)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    tinted = np.stack([lum * 0.5, lum * 0.5, lum * 0.5 + 8], -1)
    s = asleep[:, cr * t:(cr + 1) * t, cc * t:(cc + 1) * t].astype(np.float64)
    assert np.abs(tinted - s).max() > 2.0, (
        "desaturating the awake centre tile reproduces the sleeping one, so the fifth player "
        "sprite is not being drawn")


def test_the_facing_slot_wraps_mod_four_not_mod_five(env, states):
    """Slot 4 is the sleeping sprite and must not be reachable from a facing value.

    ``player_facing`` is decoded ``% N_FACING``; if it were ``% N_PLAYER`` a facing of 4 would draw
    a sleeping player on an awake state, and the conformance probe's +25 would find it.
    """
    base = OBS_H * OBS_W * 2
    st = np.array(states, np.float32, copy=True)
    st[:, base] = 0.0
    st[:, OBS_H * OBS_W * 2 + 1 + N_STRIP] = 0.0
    a = env.render(st, size=16)
    st4 = np.array(st, copy=True)
    st4[:, base] = 4.0
    assert torch.equal(a, env.render(st4, size=16)), "facing 4 is not facing 0; it aliased the sprite"
    assert PLAYER_ASLEEP == 4 and NO_PLAYER == 5


def test_player_visible_still_hides_a_sleeping_player(env, states):
    """The ablation has to keep working on the new slot, or it is inert on a quarter of the pool."""
    st = _set(states, is_sleeping=1.0)
    on = env.render(st, size=16, variation={"player_visible": 1})
    off = env.render(st, size=16, variation={"player_visible": 0})
    assert not torch.equal(on, off)


# ---------------------------------------------------------------- the state contract


def test_the_first_127_slots_kept_their_meaning(env, states):
    """Appended, never reordered — which is what makes the frozen Step-4 world rows replayable."""
    assert env.state_dim == 144
    assert env.state_names[:127] == (
        tuple(f"block_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
        + tuple(f"mob_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
        + ("player_facing",))
    assert env.state_names[127:143] == tuple(f"inv_{n}" for n in STRIP_NAMES)
    assert env.state_names[143] == "is_sleeping"


def test_a_step4_shaped_row_reproduces_the_old_world_layer(env):
    """A 127-slot row zero-extended to 144 is daylight, awake, empty-handed — the Step-4 state.

    The world rows it renders must be exactly what the pre-Step-7 renderer produced, and the strip
    must be entirely black. This is the replay obligation, asserted in the repo rather than only in
    the certification harness.
    """
    rng = np.random.default_rng(4)
    st = env.sample_states(8, rng)
    old = np.zeros_like(st)
    old[:, :127] = st[:, :127]                      # counts 0, not sleeping
    for t in (7, 16):
        f = env.render(old, size=t).numpy()
        w = OBS_H * t
        assert int(f[:, w:].max()) == 0, "an empty inventory must draw a black strip"
        assert int(f[:, :w].max()) > 0


def test_the_sampler_covers_every_count_and_both_sleep_states(env):
    """A pool that does not is indistinguishable from a wrong state contract; see the docstring."""
    st = env.sample_states(512, np.random.default_rng(9))
    base = OBS_H * OBS_W * 2 + 1
    for i, name in enumerate(STRIP_NAMES):
        seen = set(np.unique(np.round(st[:, base + i]).astype(int)).tolist())
        assert seen == set(range(N_COUNT)), f"{name} only reached {sorted(seen)}"
    sleep = st[:, base + N_STRIP]
    assert 0.05 < float((sleep > 0.5).mean()) < 0.5
    assert set(np.unique(sleep).tolist()) == {0.0, 1.0}


def test_every_new_state_dimension_reaches_a_pixel(env):
    """``check_state_dims_matter`` in miniature, run here so a failure names the dimension."""
    rng = np.random.default_rng(11)
    st = env.sample_states(6, rng)
    base = OBS_H * OBS_W * 2 + 1
    ref = env.render(st, size=16)
    for j in range(base, env.state_dim):
        probe = np.array(st, copy=True)
        probe[:, j] = (probe[:, j] + 25.0)
        assert not torch.equal(ref, env.render(probe, size=16)), (
            f"state dim {j} ({env.state_names[j]}) reaches no pixel under a +25 probe")
