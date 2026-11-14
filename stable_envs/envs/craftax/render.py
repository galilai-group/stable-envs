"""Craftax and Craftax-Classic pixel renderers that draw the map with one gather per pixel.

Upstream paints each block type, then each item type, over the whole frame. A pass for a type that a
pixel does not hold adds exactly 0.0 to it, or scales it by exactly 1.0 and adds 0.0. So each pixel ends
with the block texture of its own cell, blended once with the item texture of its own cell. These
renderers read those two textures with one gather each and apply upstream's blend once. The player,
mob, light, night, sleep and inventory drawing is upstream's, in the same order. Frames equal upstream's
bit for bit.

:class:`Look` holds appearance inputs. ``look=None`` and the default ``Look`` both draw upstream's frame.
"""

from __future__ import annotations

import importlib
from collections.abc import Callable
from typing import Any, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


class Look(NamedTuple):
    """Appearance inputs; the defaults draw upstream's frame."""

    #: (3,) factor on the RGB of every block and item texture.
    tint: Any = (1.0, 1.0, 1.0)
    #: Scalar factor on the RGB of every block and item texture.
    brightness: Any = 1.0
    #: Scalar factor on the night darkening.
    night: Any = 1.0


def _constants(classic: bool):
    return importlib.import_module(f"craftax.{'craftax_classic' if classic else 'craftax'}.constants")


def textures(classic: bool, upstream_renderer: Callable | None = None) -> dict:
    """Upstream's texture set for the agent tile size.

    ``upstream_renderer`` is the function an upstream pixel environment holds as ``_render_fn``; its
    textures are reused, so the texture cache is not read a second time.
    """
    if upstream_renderer is not None:
        cells = dict(zip(upstream_renderer.__code__.co_freevars, upstream_renderer.__closure__ or (), strict=True))
        if "textures" in cells:
            return cells["textures"].cell_contents
    constants = _constants(classic)
    return constants.load_all_textures()[constants.BLOCK_PIXEL_SIZE_AGENT]


def _tile_table(full, size: int) -> np.ndarray:
    """``(types, h, w, C)`` tiles of ``(types, H, W, C)`` full-view textures.

    Upstream builds each full-view texture by tiling one block tile, so ``h = w = size``. The full
    textures are kept when they are not such a tiling.
    """
    full = np.asarray(full, dtype=np.float32)
    tile = full[:, :size, :size]
    reps = (1, full.shape[1] // size, full.shape[2] // size, 1)
    return tile if np.array_equal(np.tile(tile, reps), full) else full


def _cell_gather(table, view, size: int):
    """``table[view[y // size, x // size], y % h, x % w]`` for every pixel; types outside the table read 0."""
    rows, cols = view.shape[0] * size, view.shape[1] * size
    index = jnp.repeat(jnp.repeat(view, size, axis=0), size, axis=1)
    ys = (np.arange(rows) % table.shape[1])[:, None]
    xs = (np.arange(cols) % table.shape[2])[None, :]
    return table.at[index, ys, xs].get(mode="fill", fill_value=0.0)


def _materialized(x):
    """``x`` written out before later ops read it, as upstream's scan writes its result.

    XLA would otherwise fuse the next blend with the lookup, and XLA:CPU rounds that fused blend
    differently from upstream's.
    """
    return jax.lax.optimization_barrier(x)


def _scale(look: Look | None):
    if look is None:
        return None
    return jnp.asarray(look.tint, jnp.float32) * jnp.asarray(look.brightness, jnp.float32)


def _mob_blend(pixels, local_position, on_screen, texture, alpha, size: int):
    texture = texture * on_screen
    with_background = 1 - alpha * on_screen
    corner = (local_position[0] * size, local_position[1] * size, 0)
    with_background = with_background * jax.lax.dynamic_slice(pixels, corner, (size, size, 3))
    with_background = with_background + texture * alpha
    return jax.lax.dynamic_update_slice(pixels, with_background, corner)


def _oriented(texture, alpha, direction):
    """Upstream's flip and transpose of a projectile texture for its direction."""
    flip = jnp.logical_or(direction[0] > 0, direction[1] > 0)
    texture = jax.lax.select(flip, jnp.flip(texture, axis=0), texture)
    alpha = jax.lax.select(flip, jnp.flip(alpha, axis=0), alpha)
    sideways = direction[1] != 0
    texture = jax.lax.select(sideways, jnp.transpose(texture, (1, 0, 2)), texture)
    alpha = jax.lax.select(sideways, jnp.transpose(alpha, (1, 0, 2)), alpha)
    return texture, alpha


def classic_renderer(tex: dict) -> Callable[..., jax.Array]:
    """``render(state, look=None)``: Craftax-Classic's ``(63, 63, 3)`` float frame in [0, 255]."""
    c = _constants(True)
    size = c.BLOCK_PIXEL_SIZE_AGENT
    obs = c.OBS_DIM
    blocks = jnp.asarray(_tile_table(tex["full_map_block_textures"], size))

    def render(state, look: Look | None = None) -> jax.Array:
        obs_dim_array = jnp.array([obs[0], obs[1]], dtype=jnp.int32)
        padded_grid = jnp.pad(
            state.map, (c.MAX_OBS_DIM + 2, c.MAX_OBS_DIM + 2), constant_values=c.BlockType.OUT_OF_BOUNDS.value
        )
        tl_corner = state.player_position - obs_dim_array // 2 + c.MAX_OBS_DIM + 2
        map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, obs)

        map_pixels = _cell_gather(blocks, map_view, size)
        scale = _scale(look)
        if scale is not None:
            map_pixels = map_pixels * scale
        map_pixels = _materialized(map_pixels)

        player_texture_index = jax.lax.select(state.is_sleeping, 4, state.player_direction - 1)
        map_pixels = (
            map_pixels * (1 - tex["full_map_player_textures_alpha"][player_texture_index])
            + tex["full_map_player_textures"][player_texture_index]
            * tex["full_map_player_textures_alpha"][player_texture_index]
        )

        def on_screen_at(mobs, index):
            local_position = (
                mobs.position[index] - state.player_position + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
            )
            on_screen = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all()
            return local_position, on_screen * mobs.mask[index]

        for mobs, name in ((state.zombies, "zombie"), (state.cows, "cow"), (state.skeletons, "skeleton")):

            def add_mob(pixels, index, mobs=mobs, name=name):
                local_position, on_screen = on_screen_at(mobs, index)
                texture, alpha = tex[f"{name}_texture"], tex[f"{name}_texture_alpha"]
                return _mob_blend(pixels, local_position, on_screen, texture, alpha, size), None

            map_pixels, _ = jax.lax.scan(add_mob, map_pixels, jnp.arange(mobs.mask.shape[0]))

        def add_arrow(pixels, index):
            local_position, on_screen = on_screen_at(state.arrows, index)
            texture, alpha = _oriented(tex["arrow_texture"], tex["arrow_texture_alpha"], state.arrow_directions[index])
            return _mob_blend(pixels, local_position, on_screen, texture, alpha, size), None

        map_pixels, _ = jax.lax.scan(add_arrow, map_pixels, jnp.arange(state.arrows.mask.shape[0]))

        daylight = state.light_level
        night_static_intensity = 2 * (0.5 - daylight)
        night_static_intensity = jnp.maximum(night_static_intensity, 0.0)
        if look is not None:
            night_static_intensity = night_static_intensity * jnp.asarray(look.night, jnp.float32)
        night_with_static = jax.random.uniform(state.state_rng, map_pixels.shape[:2]) * 95 + 32
        night_static_mask = night_static_intensity * tex["night_noise_intensity_texture"]
        night_with_static = (1 - night_static_mask) * map_pixels + night_static_mask * night_with_static[:, :, None]
        night_pixels = jax.lax.select(daylight < 0.5, night_with_static, map_pixels)

        enhance_factor = 0.4
        lum = 0.299 * night_pixels[:, :, 0] + 0.587 * night_pixels[:, :, 1] + 0.114 * night_pixels[:, :, 2]
        lum = jnp.expand_dims(lum, axis=-1).repeat(3, axis=-1)
        night_pixels = night_pixels * enhance_factor + (1 - enhance_factor) * lum
        night_pixels = 0.5 * night_pixels + 0.5 * tex["night_texture"]
        map_pixels = daylight * map_pixels + (1 - daylight) * night_pixels

        sleep_pixels = 0.299 * map_pixels[:, :, 0] + 0.587 * map_pixels[:, :, 1] + 0.114 * map_pixels[:, :, 2]
        sleep_pixels = (0.5 * sleep_pixels)[:, :, None] + (0.5 * jnp.array([0, 0, 16]))[None, None, :]
        map_pixels = (1 - state.is_sleeping) * map_pixels + state.is_sleeping * sleep_pixels

        left = (size - int(0.8 * size)) // 2 - 1
        right = size - int(0.8 * size) - left
        inv_pixels = jnp.zeros((c.INVENTORY_OBS_HEIGHT * size, obs[1] * size, 3), dtype=jnp.float32)
        number_offset = size - int(size * 0.6)

        def render_number(pixels, number, x, y):
            rows = slice(y * size + number_offset - 1, (y + 1) * size - 1)
            cols = slice(x * size + number_offset - 1, (x + 1) * size - 1)
            pixels = pixels.at[rows, cols].mul(1 - tex["number_textures_alpha"][number])
            return pixels.at[rows, cols].add(tex["number_textures"][number])

        inv = state.inventory
        block = tex["smaller_block_textures"]
        entries = (
            (state.player_health, tex["health_texture"], 0, 0),
            (state.player_food, tex["hunger_texture"], 1, 0),
            (state.player_drink, tex["thirst_texture"], 2, 0),
            (state.player_energy, tex["energy_texture"], 3, 0),
            (inv.wood, block[c.BlockType.WOOD.value], 5, 0),
            (inv.stone, block[c.BlockType.STONE.value], 6, 0),
            (inv.coal, block[c.BlockType.COAL.value], 7, 0),
            (inv.iron, block[c.BlockType.IRON.value], 8, 0),
            (inv.diamond, block[c.BlockType.DIAMOND.value], 0, 1),
            (inv.sapling, tex["sapling_texture"], 4, 0),
            (inv.wood_pickaxe, tex["wood_pickaxe_texture"], 1, 1),
            (inv.stone_pickaxe, tex["stone_pickaxe_texture"], 2, 1),
            (inv.iron_pickaxe, tex["iron_pickaxe_texture"], 3, 1),
            (inv.wood_sword, tex["wood_sword_texture"], 4, 1),
            (inv.stone_sword, tex["stone_sword_texture"], 5, 1),
            (inv.iron_sword, tex["iron_sword_texture"], 6, 1),
        )
        for value, texture, x, y in entries:
            icon = jax.lax.select(value > 0, texture, tex["smaller_empty_texture"])
            inv_pixels = inv_pixels.at[
                y * size + left : (y + 1) * size - right, x * size + left : (x + 1) * size - right
            ].set(icon)
            inv_pixels = render_number(inv_pixels, value, x, y)

        return jnp.concatenate([map_pixels, inv_pixels], axis=0)

    return render


def craftax_renderer(tex: dict, do_night_noise: bool = True) -> Callable[..., jax.Array]:
    """``render(state, look=None)``: Craftax's ``(130, 110, 3)`` float frame in [0, 255]."""
    from craftax.craftax.util.game_logic_utils import is_boss_vulnerable

    c = _constants(False)
    size = c.BLOCK_PIXEL_SIZE_AGENT
    obs = c.OBS_DIM
    blocks = jnp.asarray(_tile_table(tex["full_map_block_textures"], size))
    items = _tile_table(tex["full_map_item_textures"], size).copy()
    # Upstream draws no item pass for NONE; alpha 0 leaves the pixel as it is.
    items[c.ItemType.NONE.value] = 0.0
    items = jnp.asarray(items)

    def render(state, look: Look | None = None) -> jax.Array:
        obs_dim_array = jnp.array([obs[0], obs[1]], dtype=jnp.int32)
        pad = (c.MAX_OBS_DIM + 2, c.MAX_OBS_DIM + 2)
        padded_grid = jnp.pad(state.map[state.player_level], pad, constant_values=c.BlockType.OUT_OF_BOUNDS.value)
        tl_corner = state.player_position - obs_dim_array // 2 + c.MAX_OBS_DIM + 2
        map_view = jax.lax.dynamic_slice(padded_grid, tl_corner, obs)

        boss_block = jax.lax.select(
            is_boss_vulnerable(state), c.BlockType.NECROMANCER_VULNERABLE.value, c.BlockType.NECROMANCER.value
        )
        map_view_boss = map_view == c.BlockType.NECROMANCER.value
        map_view = map_view_boss * boss_block + (1 - map_view_boss) * map_view

        padded_item_map = jnp.pad(state.item_map[state.player_level], pad, constant_values=c.ItemType.NONE.value)
        item_map_view = jax.lax.dynamic_slice(padded_item_map, tl_corner, obs)
        is_ladder_down_open = state.monsters_killed[state.player_level] >= c.MONSTERS_KILLED_TO_CLEAR_LEVEL
        ladder_down_item = jax.lax.select(
            is_ladder_down_open, c.ItemType.LADDER_DOWN.value, c.ItemType.LADDER_DOWN_BLOCKED.value
        )
        item_map_view_is_ladder_down = item_map_view == c.ItemType.LADDER_DOWN.value
        item_map_view = (
            item_map_view_is_ladder_down * ladder_down_item + (1 - item_map_view_is_ladder_down) * item_map_view
        )

        map_pixels = _cell_gather(blocks, map_view, size)
        item = _cell_gather(items, item_map_view, size)
        item_rgb, item_alpha = item[:, :, :3], item[:, :, 3]
        scale = _scale(look)
        if scale is not None:
            map_pixels = map_pixels * scale
            item_rgb = item_rgb * scale
        map_pixels, item_rgb, item_alpha = _materialized((map_pixels, item_rgb, item_alpha))
        map_pixels = map_pixels * (1 - item_alpha)[:, :, None]
        map_pixels = _materialized(map_pixels + item_rgb * item_alpha[:, :, None])

        player_texture_index = jax.lax.select(state.is_sleeping, 4, state.player_direction - 1)
        map_pixels = (
            map_pixels * (1 - tex["full_map_player_textures_alpha"][player_texture_index])
            + tex["full_map_player_textures"][player_texture_index]
            * tex["full_map_player_textures_alpha"][player_texture_index]
        )

        def on_screen_at(mobs, index):
            local_position = (
                mobs.position[state.player_level, index]
                - state.player_position
                + jnp.ones((2,), dtype=jnp.int32) * (obs_dim_array // 2)
            )
            on_screen = jnp.logical_and(local_position >= 0, local_position < obs_dim_array).all()
            return local_position, on_screen * mobs.mask[state.player_level, index]

        # The mob and projectile tables travel in the scan carry, as upstream passes them.
        for mobs, name in (
            (state.melee_mobs, "melee_mob"),
            (state.passive_mobs, "passive_mob"),
            (state.ranged_mobs, "ranged_mob"),
        ):

            def add_mob(carry, index):
                pixels, mobs, textures, alphas = carry
                local_position, on_screen = on_screen_at(mobs, index)
                type_id = mobs.type_id[state.player_level, index]
                pixels = _mob_blend(pixels, local_position, on_screen, textures[type_id], alphas[type_id], size)
                return (pixels, mobs, textures, alphas), None

            carry = (map_pixels, mobs, tex[f"{name}_textures"], tex[f"{name}_texture_alphas"])
            (map_pixels, *_), _ = jax.lax.scan(add_mob, carry, jnp.arange(mobs.mask.shape[1]))

        def add_projectile(carry, index):
            pixels, projectiles, directions = carry
            local_position, on_screen = on_screen_at(projectiles, index)
            type_id = projectiles.type_id[state.player_level, index]
            texture, alpha = _oriented(
                tex["projectile_textures"][type_id],
                tex["projectile_texture_alphas"][type_id],
                directions[state.player_level, index],
            )
            pixels = _mob_blend(pixels, local_position, on_screen, texture, alpha, size)
            return (pixels, projectiles, directions), None

        for projectiles, directions in (
            (state.mob_projectiles, state.mob_projectile_directions),
            (state.player_projectiles, state.player_projectile_directions),
        ):
            carry = (map_pixels, projectiles, directions)
            (map_pixels, *_), _ = jax.lax.scan(add_projectile, carry, jnp.arange(projectiles.mask.shape[1]))

        padded_light_map = jnp.pad(state.light_map[state.player_level], pad, constant_values=False)
        light_map_view = jax.lax.dynamic_slice(padded_light_map, tl_corner, obs)
        light_map_pixels = light_map_view.repeat(size, axis=0).repeat(size, axis=1)
        map_pixels = (light_map_pixels)[:, :, None] * map_pixels

        night_pixels = tex["night_texture"]
        daylight = state.light_level
        daylight = jax.lax.select(state.player_level == 0, daylight, 1.0)
        night_intensity = 2 * (0.5 - daylight)
        night_intensity = jnp.maximum(night_intensity, 0.0)
        if look is not None:
            night_intensity = night_intensity * jnp.asarray(look.night, jnp.float32)
        if do_night_noise:
            night_noise = jax.random.uniform(state.state_rng, night_pixels.shape[:2]) * 95 + 32
            night_noise = jnp.expand_dims(night_noise, axis=-1).repeat(3, axis=-1)
            night_mask = tex["night_noise_intensity_texture"] * night_intensity
        else:
            night_noise = jnp.ones(night_pixels.shape[:2]) * 64
            night_noise = jnp.expand_dims(night_noise, axis=-1).repeat(3, axis=-1)
            night_mask = jnp.ones_like(tex["night_noise_intensity_texture"]) * night_intensity * 0.5
        night = (1.0 - night_mask) * map_pixels + night_mask * night_noise
        night = night_pixels * 0.5 + 0.5 * night
        map_pixels = daylight * map_pixels + (1 - daylight) * night

        sleep_pixels = jnp.zeros_like(map_pixels)
        sleep_level = 1.0 - state.is_sleeping * 0.5
        map_pixels = sleep_level * map_pixels + (1 - sleep_level) * sleep_pixels

        return jnp.concatenate([map_pixels, _craftax_inventory(state, tex, c)], axis=0)

    return render


def _craftax_inventory(state, tex: dict, c) -> jax.Array:
    """Upstream's inventory panel for full Craftax, drawn in upstream's order."""
    size = c.BLOCK_PIXEL_SIZE_AGENT
    left = (size - int(0.8 * size)) // 2
    right = size - int(0.8 * size) - left
    pixels = jnp.zeros((c.INVENTORY_OBS_HEIGHT * size, c.OBS_DIM[1] * size, 3), dtype=jnp.float32)
    number_size = int(size * 0.4)
    number_offset = size - number_size
    number_double_offset = size - 2 * number_size
    empty = tex["smaller_empty_texture"]

    def digit(pixels, number, x, y):
        rows = slice(y * size + number_offset, (y + 1) * size)
        cols = slice(x * size + number_offset, (x + 1) * size)
        pixels = pixels.at[rows, cols].mul(1 - tex["number_textures_alpha"][number])
        return pixels.at[rows, cols].add(tex["number_textures"][number])

    def two_digits(pixels, number, x, y):
        tens = number // 10
        ones = number % 10
        ones_textures = jax.lax.select(number == 0, tex["number_textures"], tex["number_textures_with_zero"])
        ones_alpha = jax.lax.select(number == 0, tex["number_textures_alpha"], tex["number_textures_alpha_with_zero"])
        rows = slice(y * size + number_offset, (y + 1) * size)
        cols = slice(x * size + number_offset, (x + 1) * size)
        pixels = pixels.at[rows, cols].mul(1 - ones_alpha[ones])
        pixels = pixels.at[rows, cols].add(ones_textures[ones])
        cols = slice(x * size + number_double_offset, x * size + number_offset)
        pixels = pixels.at[rows, cols].mul(1 - tex["number_textures_alpha"][tens])
        return pixels.at[rows, cols].add(tex["number_textures"][tens])

    def cell(x, y):
        return slice(size * y + left, size * (y + 1) - right), slice(size * x + left, size * (x + 1) - right)

    def icon(pixels, texture, x, y):
        return pixels.at[cell(x, y)].set(texture)

    def icon_with_alpha(pixels, texture, x, y):
        existing = pixels[cell(x, y)]
        new = existing * (1 - texture[:, :, 3][:, :, None]) + texture[:, :, :3] * texture[:, :, 3][:, :, None]
        return pixels.at[cell(x, y)].set(new)

    def counted(pixels, value, texture, x, y):
        pixels = icon(pixels, jax.lax.select(value > 0, texture, empty), x, y)
        return two_digits(pixels, value, x, y)

    inv = state.inventory
    block = tex["smaller_block_textures"]
    player_health = jnp.maximum(jnp.floor(state.player_health), 1).astype(int)
    pixels = counted(pixels, player_health, tex["health_texture"], 0, 0)
    pixels = counted(pixels, state.player_food, tex["hunger_texture"], 1, 0)
    pixels = counted(pixels, state.player_drink, tex["thirst_texture"], 2, 0)
    pixels = counted(pixels, state.player_energy, tex["energy_texture"], 3, 0)
    pixels = counted(pixels, state.player_mana, tex["mana_texture"], 4, 0)
    for value, block_type, x, y in (
        (inv.wood, c.BlockType.WOOD, 0, 2),
        (inv.stone, c.BlockType.STONE, 1, 2),
        (inv.coal, c.BlockType.COAL, 0, 1),
        (inv.iron, c.BlockType.IRON, 1, 1),
        (inv.diamond, c.BlockType.DIAMOND, 2, 1),
        (inv.sapphire, c.BlockType.SAPPHIRE, 3, 1),
        (inv.ruby, c.BlockType.RUBY, 4, 1),
    ):
        pixels = counted(pixels, value, block[block_type.value], x, y)
    pixels = counted(pixels, inv.sapling, tex["sapling_texture"], 5, 1)

    pixels = icon(pixels, tex["pickaxe_textures"][inv.pickaxe], 8, 2)
    pixels = icon(pixels, tex["sword_textures"][inv.sword], 8, 1)
    pixels = icon(pixels, tex["bow_textures"][inv.bow], 6, 1)
    pixels = counted(pixels, inv.arrows, tex["player_projectile_textures"][0], 6, 2)
    for i in range(4):
        pixels = icon(pixels, tex["armour_textures"][inv.armour[i], i], 7, i)
    pixels = counted(pixels, inv.torches, tex["torch_inv_texture"], 2, 2)
    for i in range(6):
        pixels = counted(pixels, inv.potions[i], tex["potion_textures"][i], i, 3)
    pixels = counted(pixels, inv.books, tex["book_texture"], 3, 2)
    pixels = icon(pixels, jax.lax.select(state.learned_spells[0], tex["fireball_inv_texture"], empty), 4, 2)
    pixels = icon(pixels, jax.lax.select(state.learned_spells[1], tex["iceball_inv_texture"], empty), 5, 2)

    pixels = icon_with_alpha(pixels, tex["sword_enchantment_textures"][state.sword_enchantment], 8, 1)
    arrow_enchantment_level = state.bow_enchantment * (inv.arrows > 0)
    pixels = icon_with_alpha(pixels, tex["arrow_enchantment_textures"][arrow_enchantment_level], 6, 2)
    for i in range(4):
        pixels = icon_with_alpha(pixels, tex["armour_enchantment_textures"][state.armour_enchantments[i], i], 7, i)

    pixels = digit(pixels, state.player_level, 6, 0)
    pixels = icon(pixels, jax.lax.select(state.player_xp > 0, tex["xp_texture"], empty), 9, 0)
    pixels = digit(pixels, state.player_xp, 9, 0)
    for texture, value, y in (
        ("dex_texture", state.player_dexterity, 1),
        ("str_texture", state.player_strength, 2),
        ("int_texture", state.player_intelligence, 3),
    ):
        pixels = icon(pixels, tex[texture], 9, y)
        pixels = digit(pixels, value, 9, y)
    return pixels


def renderer(classic: bool, upstream_renderer: Callable | None = None) -> Callable[..., jax.Array]:
    """``render(state, look=None)`` for one upstream ``EnvState``: a float32 frame in [0, 255]."""
    tex = textures(classic, upstream_renderer)
    return classic_renderer(tex) if classic else craftax_renderer(tex)
