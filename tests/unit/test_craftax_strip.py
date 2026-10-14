"""Craftax Classic's inventory strip, sleeping sprite and lighting bands.

The assertions are structural (what the strip depends on and what it does not), since a wrong strip
still looks like an inventory. No jax; parity against the official renderer is checked elsewhere.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from stable_envs.envs.craftax.classic import (
    N_COUNT,
    N_STRIP,
    N_TILE,
    NO_PLAYER,
    OBS_H,
    OBS_W,
    PLAYER_ASLEEP,
    STRIP_H,
    STRIP_LAYOUT,
    STRIP_NAMES,
    CraftaxClassic,
    _Composite,
    _strip_cells,
    strip_geometry,
)

SIZES = (7, 16, 32)
DEV = torch.device("cpu")


@pytest.fixture(scope="module")
def env():
    return CraftaxClassic(device="cpu")


@pytest.fixture(scope="module")
def states(env):
    return env.sample_states(24, np.random.default_rng(31337))


def _set(st, **kw):
    """A copy of ``st`` with named strip counts / is_sleeping / light_level overwritten."""
    out = np.array(st, np.float32, copy=True)
    base = OBS_H * OBS_W * 2 + 1
    for k, v in kw.items():
        if k == "is_sleeping":
            out[:, base + N_STRIP] = v
        elif k == "light":
            out[:, base + N_STRIP + 1] = v
        else:
            out[:, base + STRIP_NAMES.index(k)] = v
    return out


# ---------------------------------------------------------------- the table


@pytest.mark.parametrize("t", SIZES)
def test_the_strip_lives_in_the_same_table_as_the_world_tiles(t):
    """One table, one gather. A strip cell is a tile indexed by (cell, count), nothing more."""
    tab = _Composite.get(DEV, t, want_float=False)[1]
    assert tab.shape == (N_TILE * t, t * 3)
    assert N_TILE == 976


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
    """No two strip cells write the same pixel, so the reference's 32 sequential writes form a table."""
    lo, hi, dlo, dhi = strip_geometry(t)
    assert 0 <= lo < hi <= t, (t, lo, hi)
    assert 0 <= dlo < dhi <= t, (t, dlo, dhi)
    occ = np.zeros((STRIP_H * t, OBS_W * t), np.int32)
    for r, c in STRIP_LAYOUT:
        occ[r * t + lo : r * t + hi, c * t + lo : c * t + hi] += 1
        occ[r * t + dlo : r * t + dhi, c * t + dlo : c * t + dhi] += 1
    assert occ.max() <= 2, f"strip cells overlap at t={t}"


def test_the_strip_atlas_resamples_by_exact_replication():
    """The 32 px strip is the 16 px strip with every pixel doubled.

    Craftax's own strip geometry does not scale by two (``int(0.8*t)`` is 12 at 16 px and 25 at
    32 px), so the finished cell is resampled rather than re-laid-out.
    """
    assert int(0.8 * 32) != 2 * int(0.8 * 16) and int(0.6 * 32) != 2 * int(0.6 * 16)
    small = _strip_cells(DEV, 16)
    big = _strip_cells(DEV, 32)
    assert torch.equal(big, small.repeat_interleave(2, dim=1).repeat_interleave(2, dim=2))


# ---------------------------------------------------------------- the frame


@pytest.mark.parametrize("t", SIZES)
def test_the_frame_has_the_strip_below_the_world(env, states, t):
    f = env._render(states, size=t)
    assert tuple(f.shape) == (len(states), (OBS_H + STRIP_H) * t, OBS_W * t, 3)
    assert f.dtype == torch.uint8


@pytest.mark.parametrize("t", (7, 16))
def test_each_count_reaches_its_own_cell_and_no_other(env, states, t):
    """Changing one count changes exactly its own strip cell and nothing else in the frame."""
    w = OBS_H * t
    lay = {i: STRIP_LAYOUT[i] for i in range(N_STRIP)}
    for i, name in enumerate(STRIP_NAMES):
        a = env._render(_set(states, **{name: 0}), size=t).numpy()
        b = env._render(_set(states, **{name: 7}), size=t).numpy()
        d = np.abs(a.astype(np.int16) - b.astype(np.int16)).max(-1) > 0
        assert d[:, :w].sum() == 0, f"{name} reached the world layer"
        r, c = lay[i]
        mine = d[:, w + r * t : w + (r + 1) * t, c * t : (c + 1) * t]
        assert mine.any(), f"{name} did not reach its own cell {lay[i]}"
        rest = d[:, w:].sum() - mine.sum()
        assert rest == 0, f"{name} reached {rest} pixels outside cell {lay[i]}"


@pytest.mark.parametrize("t", (7, 16))
def test_the_two_unwritten_cells_stay_black_whatever_the_state(env, states, t):
    w = OBS_H * t
    written = set(STRIP_LAYOUT)
    f = env._render(states, size=t).numpy()
    for r in range(STRIP_H):
        for c in range(OBS_W):
            if (r, c) in written:
                continue
            patch = f[:, w + r * t : w + (r + 1) * t, c * t : (c + 1) * t]
            assert int(patch.max()) == 0, f"cell ({r},{c}) is written but the reference never writes it"


@pytest.mark.parametrize("t", (7, 16))
def test_the_strip_is_not_lit_and_not_desaturated(env, states, t):
    """The reference concatenates the strip after the lighting chain, so sleep cannot touch it."""
    w = OBS_H * t
    awake = env._render(_set(states, is_sleeping=0.0), size=t).numpy()
    asleep = env._render(_set(states, is_sleeping=1.0), size=t).numpy()
    assert np.array_equal(awake[:, w:], asleep[:, w:]), "sleeping changed the inventory strip"
    assert not np.array_equal(awake[:, :w], asleep[:, :w]), "sleeping did not change the world layer"


# ---------------------------------------------------------------- the sleeping player


@pytest.mark.parametrize("t", (7, 16))
def test_sleeping_changes_the_centre_tile_sprite_and_desaturates_everything_else(env, states, t):
    """Sleeping both swaps the centre sprite and desaturates the world rows."""
    w = OBS_H * t
    cr, cc = OBS_H // 2, OBS_W // 2
    awake = env._render(_set(states, is_sleeping=0.0), size=t).numpy()[:, :w]
    asleep = env._render(_set(states, is_sleeping=1.0), size=t).numpy()[:, :w]

    # (a) the desaturation reaches the whole world layer, not only the player's tile
    d = np.abs(awake.astype(np.int16) - asleep.astype(np.int16)).max(-1) > 0
    off_centre = d.copy()
    off_centre[:, cr * t : (cr + 1) * t, cc * t : (cc + 1) * t] = False
    assert off_centre.any(), "is_sleeping did not desaturate outside the player's tile"

    # (b) a sleeping world row is grey-plus-blue by construction: R == G everywhere
    assert np.array_equal(asleep[..., 0], asleep[..., 1]), "the sleep transform is not a luminance"
    assert (asleep[..., 2].astype(np.int16) >= asleep[..., 0].astype(np.int16)).all()

    # (c) the sprite changed, not only the tint: the desaturated awake centre tile still differs
    #     from the sleeping one
    a = awake[:, cr * t : (cr + 1) * t, cc * t : (cc + 1) * t].astype(np.float64)
    lum = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    tinted = np.stack([lum * 0.5, lum * 0.5, lum * 0.5 + 8], -1)
    s = asleep[:, cr * t : (cr + 1) * t, cc * t : (cc + 1) * t].astype(np.float64)
    assert np.abs(tinted - s).max() > 2.0, (
        "desaturating the awake centre tile reproduces the sleeping one, so the fifth player sprite is not being drawn"
    )


def test_the_facing_slot_wraps_mod_four_not_mod_five(env, states):
    """Slot 4 is the sleeping sprite and must not be reachable from a facing value.

    ``player_facing`` is decoded ``% N_FACING``; if it were ``% N_PLAYER`` a facing of 4 would draw
    a sleeping player on an awake state, and the conformance probe's +25 would find it.
    """
    base = OBS_H * OBS_W * 2
    st = np.array(states, np.float32, copy=True)
    st[:, base] = 0.0
    st[:, OBS_H * OBS_W * 2 + 1 + N_STRIP] = 0.0
    a = env._render(st, size=16)
    st4 = np.array(st, copy=True)
    st4[:, base] = 4.0
    assert torch.equal(a, env._render(st4, size=16)), "facing 4 is not facing 0; it aliased the sprite"
    assert PLAYER_ASLEEP == 4 and NO_PLAYER == 5


def test_player_visible_still_hides_a_sleeping_player(env, states):
    """``player_visible = 0`` also hides the sleeping sprite."""
    st = _set(states, is_sleeping=1.0)
    on = env._render(st, size=16, variation={"player_visible": 1})
    off = env._render(st, size=16, variation={"player_visible": 0})
    assert not torch.equal(on, off)


# ---------------------------------------------------------------- the state contract


def test_state_slots_are_world_facing_inventory_then_sleep_light_and_rng(env, states):
    """The 149 State slots in order: block and mob grids, facing, inventory, then sleep, light and RNG."""
    assert env.state_dim == 149
    assert env.state_names[:127] == (
        tuple(f"block_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
        + tuple(f"mob_r{r}c{c}" for r in range(OBS_H) for c in range(OBS_W))
        + ("player_facing",)
    )
    assert env.state_names[127:143] == tuple(f"inv_{n}" for n in STRIP_NAMES)
    assert env.state_names[143:] == ("is_sleeping", "light_level", "rng_half0", "rng_half1", "rng_half2", "rng_half3")


def test_a_zero_extended_world_row_renders_daylight_with_a_black_strip(env):
    """World slots only, zero counts and full daylight: a lit world over a black strip."""
    rng = np.random.default_rng(4)
    st = env.sample_states(8, rng)
    old = np.zeros_like(st)
    old[:, :127] = st[:, :127]  # counts 0, not sleeping, light_level 0...
    old[:, 127 + N_STRIP + 1] = 1.0  # ...except daylight
    for t in (7, 16):
        f = env._render(old, size=t).numpy()
        w = OBS_H * t
        assert int(f[:, w:].max()) == 0, "an empty inventory must draw a black strip"
        assert int(f[:, :w].max()) > 0


def test_the_sampler_covers_every_count_and_both_sleep_states(env):
    """Sampled states reach every count 0..9 and both sleep states."""
    st = env.sample_states(512, np.random.default_rng(9))
    base = OBS_H * OBS_W * 2 + 1
    for i, name in enumerate(STRIP_NAMES):
        seen = set(np.unique(np.round(st[:, base + i]).astype(int)).tolist())
        assert seen == set(range(N_COUNT)), f"{name} only reached {sorted(seen)}"
    sleep = st[:, base + N_STRIP]
    assert 0.05 < float((sleep > 0.5).mean()) < 0.5
    assert set(np.unique(sleep).tolist()) == {0.0, 1.0}


def test_every_new_state_dimension_reaches_a_pixel(env):
    """Every strip, sleep, light and key slot reaches a pixel under a +25 probe.

    The states are forced to night, because the PRNG-key slots reach no pixel at ``light_level >= 0.5``.
    """
    rng = np.random.default_rng(11)
    st = _set(env.sample_states(6, rng), light=0.15)
    base = OBS_H * OBS_W * 2 + 1
    ref = env._render(st, size=16)
    for j in range(base, env.state_dim):
        probe = np.array(st, copy=True)
        probe[:, j] = probe[:, j] + 25.0
        assert not torch.equal(ref, env._render(probe, size=16)), (
            f"state dim {j} ({env.state_names[j]}) reaches no pixel under a +25 probe"
        )


# ---------------------------------------------------------------- night


def test_daylight_is_the_identity_and_keeps_the_uint8_fast_path(env, states):
    """``light == 1.0`` awake is bit-identical to the full float chain and stays on the uint8 gather."""
    st = _set(states, light=1.0, is_sleeping=0.0)
    fast = env._render(st, size=16)
    dec = env._decode(torch.as_tensor(st), len(st))
    sty = env.resolve_style(None, None, batch=len(st))
    # force the float path by asking for the blend implementation, which never short-circuits
    slow = env._render_blend(*dec, 16, sty)
    assert torch.equal(fast, slow)


@pytest.mark.parametrize("light", (1.0, 0.75, 0.5))
def test_no_noise_is_drawn_at_or_above_half_daylight(env, states, light):
    """``intensity = max(2*(0.5 - light), 0)`` is exactly 0 there, so the PRNG key reaches no pixel."""
    a = _set(states, light=light)
    b = np.array(a, copy=True)
    assert env.state_names[OBS_H * OBS_W * 2 + 3 + N_STRIP] == "rng_half0"
    b[:, OBS_H * OBS_W * 2 + 3 + N_STRIP :] = 12345.0
    assert torch.equal(env._render(a, size=16), env._render(b, size=16))


def test_the_key_reaches_the_world_layer_below_half_daylight(env, states):
    """Below 0.5 the PRNG key changes the world rows and never the strip."""
    a = _set(states, light=0.1)
    b = np.array(a, copy=True)
    assert env.state_names[OBS_H * OBS_W * 2 + 3 + N_STRIP] == "rng_half0"
    b[:, OBS_H * OBS_W * 2 + 3 + N_STRIP :] = 4321.0
    fa, fb = env._render(a, size=16).numpy(), env._render(b, size=16).numpy()
    w = OBS_H * 16
    changed = np.abs(fa.astype(np.int16) - fb.astype(np.int16)).max(-1) > 0
    assert changed[:, :w].mean() > 0.3, "the night noise field barely reaches the frame"
    assert changed[:, w:].sum() == 0, "the PRNG key reached the inventory strip"


def test_the_lighting_bands_are_three_different_pictures(env, states):
    """Day, dusk and night render differently: at dusk the enhance/tint/blend still run."""
    day = env._render(_set(states, light=1.0), size=16)
    dusk = env._render(_set(states, light=0.75), size=16)
    night = env._render(_set(states, light=0.1), size=16)
    assert not torch.equal(day, dusk) and not torch.equal(dusk, night)
    w = OBS_H * 16
    for a, b in ((day, dusk), (dusk, night)):
        assert torch.equal(a[:, w:], b[:, w:]), "the lighting reached the inventory strip"


def test_light_level_decodes_by_a_tent_that_is_the_identity_in_range(env, states):
    """In-range values pass through bit-exact; out-of-range ones fold back and still move pixels.

    A clamp would leave daylight insensitive to the +25 probe, and ``x - floor(x)`` would send 1.0 to 0.0.
    """
    for v in (0.0, 0.25, 0.5, 0.75, 1.0):
        _, _, _, _, _, light, _ = env._decode(torch.as_tensor(_set(states, light=v)), len(states))
        assert torch.equal(light, torch.full_like(light, v)), v
    # 1.0 + 25 = 26; 26 % 2 == 0; tent(0) == 0 -> full night, so the probe reaches the frame
    _, _, _, _, _, probed, _ = env._decode(torch.as_tensor(_set(states, light=26.0)), len(states))
    assert float(probed.max()) == 0.0
    assert not torch.equal(
        env._render(_set(states, light=1.0), size=16), env._render(_set(states, light=26.0), size=16)
    )


def test_a_batch_may_mix_day_dusk_night_and_sleep(env, states):
    """A row renders the same in a mixed batch as alone, whichever path the batch takes."""
    n = len(states)
    mixed = np.array(states, np.float32, copy=True)
    j = OBS_H * OBS_W * 2 + 1 + N_STRIP
    mixed[:, j] = (np.arange(n) % 4 == 0).astype(np.float32)
    mixed[:, j + 1] = np.tile([1.0, 0.75, 0.3, 0.0], n // 4 + 1)[:n]
    together = env._render(mixed, size=16)
    for i in range(n):
        assert torch.equal(together[i], env._render(mixed[i : i + 1], size=16)[0]), i


def test_the_night_chain_is_shape_parameterised_for_the_full_renderer():
    """``night_and_sleep`` accepts any window size and an optional ``light_map``."""
    from stable_envs.envs.craftax.classic import night_and_sleep, night_vignette

    for h, w in ((49, 63), (112, 144), (63, 77), (90, 110)):
        world = torch.rand(3, h, w, 3) * 255
        out = night_and_sleep(
            world,
            torch.tensor([1.0, 0.75, 0.1]),
            torch.tensor([False, True, False]),
            torch.tensor([[1, 2], [3, 4], [5, 6]], dtype=torch.int32),
        )
        assert out.shape == world.shape
        assert torch.equal(out[0], world[0]), "light=1.0 is not the identity"
        assert night_vignette(DEV, h, w).shape == (h, w, 1)
    # light_map of 1.0 is the identity
    world = torch.rand(2, 49, 63, 3) * 255
    keys = torch.tensor([[7, 8], [9, 10]], dtype=torch.int32)
    lit = torch.tensor([0.2, 0.2])
    slp = torch.tensor([False, False])
    assert torch.equal(
        night_and_sleep(world, lit, slp, keys),
        night_and_sleep(world, lit, slp, keys, light_map=torch.ones(1, 1, 1, 1)),
    )


def test_the_strip_grid_cache_is_not_written_through(env, states):
    """The cached strip layout tensors are reused and never modified by a render."""
    dev = torch.device("cpu")
    cell_of, blank = env._strip_grid(dev)
    before = (cell_of.clone(), blank.clone())
    for t in (7, 16):
        env._render(states, size=t)
        env._render(_set(states, light=0.1), size=t)
        env._render(states, size=t, variation={"channel_gain": [0.5, 1.0, 2.0]})
    again = env._strip_grid(dev)
    assert again[0] is cell_of and again[1] is blank, "the grid was rebuilt instead of cached"
    assert torch.equal(cell_of, before[0]) and torch.equal(blank, before[1])


def test_the_declared_resolution_applicability_matches_measurement(env, states):
    """The renderer meets the resolution bound at ``light_level >= 0.5`` and not below it.

    ``check_resolution_consistency`` is declared not applicable below 0.5 because the official
    renderer is itself inconsistent there.
    """
    from stable_envs.testing.conformance import check_resolution_consistency

    d = CraftaxClassic.resolution_consistency_applicability
    assert set(d) >= {"light_level >= 0.5", "light_level < 0.5", "evidence", "measured_against"}
    assert "not_applicable" in d["light_level < 0.5"]
    for light in (1.0, 0.75, 0.5):
        r = check_resolution_consistency(env, _set(states, light=light))
        assert r["exact_frac"] >= 1.0 - 0.005, (light, r)
    for light in (0.45, 0.1):
        r = check_resolution_consistency(env, _set(states, light=light))
        assert r["exact_frac"] < 1.0 - 0.005, (
            f"light={light} meets the bound; the declared non-applicability is stale",
            r,
        )


def test_the_strip_is_resolution_consistent_in_every_band(env, states):
    """The strip is exactly resolution-consistent in every lighting band.

    craftax re-lays the strip out at each size and scores about 0.69 under the check; resampling the
    finished cell keeps this renderer at 1.0.
    """
    from stable_envs.testing.conformance import _box_downsample

    w = OBS_H * 16
    for light in (1.0, 0.75, 0.1):
        st = _set(states, light=light)
        a = env._render(st, size=16).numpy()[:, w:]
        b = env._render(st, size=32).numpy()[:, OBS_H * 32 :]
        d = np.abs(_box_downsample(b, 2).astype(np.int16) - a.astype(np.int16))
        assert float((d.max(-1) == 0).mean()) == 1.0, (light, int(d.max()))


def test_the_declared_certification_distribution_is_the_one_the_sampler_produces(env):
    """The declared lighting and sleep shares match what ``sample_states(64)`` produces."""
    from stable_envs.envs.craftax.classic import CraftaxClassic as C

    d = C.certification_state_distribution
    assert d["daylight_share"] == 0.25
    for seed in range(20):
        e = C(device="cpu", seed=seed)
        p = e.sample_states(64)
        light, sleep = p[:, OBS_H * OBS_W * 2 + 2 + N_STRIP], p[:, OBS_H * OBS_W * 2 + 1 + N_STRIP]
        # daylight is exact because its band is pinned by row index; the other bands draw from
        # craftax's day cycle and get the declared range
        assert float((light == 1.0).mean()) == 0.25, seed
        assert 0.33 <= float(((light >= 0.5) & (light < 1.0)).mean()) <= 0.44, seed
        assert 0.31 <= float((light < 0.5).mean()) <= 0.43, seed
        assert 0.31 <= float((sleep > 0.5).mean()) <= 0.46, seed
        # rows below full daylight are about 3/4
        assert 0.70 <= float((light < 1.0).mean()) <= 0.80, seed
