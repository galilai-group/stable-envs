"""The gather renderers draw upstream Craftax's pixel frames bit for bit, on random rollouts and varied states."""

from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("jax")
pytest.importorskip("craftax")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402

pytestmark = pytest.mark.unit

GAMES = ("Craftax-Classic-Symbolic-v1", "Craftax-Symbolic-v1")
DIRECTIONS = jnp.array([[0, 1], [0, -1], [1, 0], [-1, 0]], dtype=jnp.int32)
DEVICES = ["cpu", pytest.param("cuda", marks=pytest.mark.gpu)]


def _device(name):
    try:
        return jax.devices("gpu" if name == "cuda" else "cpu")[0]
    except RuntimeError:
        pytest.skip(f"JAX has no {name} device")


def _renderers(game):
    from craftax.craftax_env import make_craftax_env_from_name

    from stable_envs.envs.craftax.kernel import texture_lock
    from stable_envs.envs.craftax.render import Look, renderer

    classic = "Classic" in game
    with texture_lock(classic):
        upstream = make_craftax_env_from_name(game.replace("Symbolic", "Pixels"), auto_reset=False)
        ours = renderer(classic)
    return (
        jax.jit(jax.vmap(upstream._render_fn)),
        jax.jit(jax.vmap(ours)),
        jax.jit(jax.vmap(lambda s: ours(s, Look()))),
        jax.jit(jax.vmap(upstream.get_obs)),
    )


def _ints(key, like, low, high):
    return jax.random.randint(key, jnp.shape(like), low, high).astype(jnp.asarray(like).dtype)


def _near(key, mobs, centre):
    """Mob positions around the player, some on screen and some off it."""
    return (centre + jax.random.randint(key, mobs.position.shape, -7, 8)).astype(mobs.position.dtype)


def _vary_classic(state, key):
    from craftax.craftax_classic.constants import BlockType

    k = iter(jax.random.split(key, 64))
    position = _ints(next(k), state.player_position, 0, state.map.shape[0])

    def mobs(m):
        return m.replace(position=_near(next(k), m, position), mask=jax.random.bernoulli(next(k), 0.6, m.mask.shape))

    inventory = state.inventory.replace(
        **{f: _ints(next(k), getattr(state.inventory, f), 0, 12) for f in state.inventory.__dataclass_fields__}
    )
    return state.replace(
        map=_ints(next(k), state.map, 0, len(BlockType)),
        player_position=position,
        player_direction=_ints(next(k), state.player_direction, 1, 5),
        player_health=_ints(next(k), state.player_health, 0, 12),
        player_food=_ints(next(k), state.player_food, 0, 12),
        player_drink=_ints(next(k), state.player_drink, 0, 12),
        player_energy=_ints(next(k), state.player_energy, 0, 12),
        is_sleeping=jax.random.bernoulli(next(k), 0.3),
        inventory=inventory,
        zombies=mobs(state.zombies),
        cows=mobs(state.cows),
        skeletons=mobs(state.skeletons),
        arrows=mobs(state.arrows),
        arrow_directions=DIRECTIONS[jax.random.randint(next(k), state.arrow_directions.shape[:1], 0, 4)],
        light_level=jax.random.uniform(next(k), dtype=jnp.asarray(state.light_level).dtype),
        state_rng=jax.random.split(next(k))[0],
    )


def _vary_craftax(state, key):
    from craftax.craftax.constants import BlockType, ItemType

    k = iter(jax.random.split(key, 96))
    position = _ints(next(k), state.player_position, 0, state.map.shape[1])

    def mobs(m):
        return m.replace(
            position=_near(next(k), m, position),
            mask=jax.random.bernoulli(next(k), 0.6, m.mask.shape),
            type_id=_ints(next(k), m.type_id, 0, 8),
        )

    def directions(d):
        return DIRECTIONS[jax.random.randint(next(k), d.shape[:-1], 0, 4)]

    inv = state.inventory
    ranges = {"pickaxe": 5, "sword": 5, "bow": 2, "armour": 3}
    inventory = inv.replace(
        **{f: _ints(next(k), getattr(inv, f), 0, ranges.get(f, 12)) for f in inv.__dataclass_fields__}
    )
    scalar = state.player_health
    return state.replace(
        map=_ints(next(k), state.map, 0, len(BlockType)),
        item_map=_ints(next(k), state.item_map, 0, len(ItemType)),
        light_map=jax.random.uniform(next(k), state.light_map.shape, dtype=state.light_map.dtype),
        monsters_killed=_ints(next(k), state.monsters_killed, 0, 12),
        player_position=position,
        player_level=_ints(next(k), state.player_level, 0, state.map.shape[0]),
        player_direction=_ints(next(k), state.player_direction, 1, 5),
        player_health=jax.random.uniform(next(k), (), jnp.asarray(scalar).dtype, 0.0, 12.0),
        player_food=_ints(next(k), state.player_food, 0, 12),
        player_drink=_ints(next(k), state.player_drink, 0, 12),
        player_energy=_ints(next(k), state.player_energy, 0, 12),
        player_mana=_ints(next(k), state.player_mana, 0, 12),
        is_sleeping=jax.random.bernoulli(next(k), 0.3),
        player_xp=_ints(next(k), state.player_xp, 0, 12),
        player_dexterity=_ints(next(k), state.player_dexterity, 0, 12),
        player_strength=_ints(next(k), state.player_strength, 0, 12),
        player_intelligence=_ints(next(k), state.player_intelligence, 0, 12),
        inventory=inventory,
        melee_mobs=mobs(state.melee_mobs),
        passive_mobs=mobs(state.passive_mobs),
        ranged_mobs=mobs(state.ranged_mobs),
        mob_projectiles=mobs(state.mob_projectiles),
        mob_projectile_directions=directions(state.mob_projectile_directions),
        player_projectiles=mobs(state.player_projectiles),
        player_projectile_directions=directions(state.player_projectile_directions),
        learned_spells=jax.random.bernoulli(next(k), 0.5, state.learned_spells.shape),
        sword_enchantment=_ints(next(k), state.sword_enchantment, 0, 3),
        bow_enchantment=_ints(next(k), state.bow_enchantment, 0, 3),
        armour_enchantments=_ints(next(k), state.armour_enchantments, 0, 3),
        boss_timesteps_to_spawn_this_round=_ints(next(k), state.boss_timesteps_to_spawn_this_round, -2, 3),
        light_level=jax.random.uniform(next(k), dtype=jnp.asarray(state.light_level).dtype),
        state_rng=jax.random.split(next(k))[0],
    )


def _assert_same(renderers, states, what):
    upstream, ours, default_look, _ = renderers
    expected = np.asarray(upstream(states))
    np.testing.assert_array_equal(np.asarray(ours(states)), expected, err_msg=what)
    np.testing.assert_array_equal(np.asarray(default_look(states)), expected, err_msg=f"{what}, default Look")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("name", ["CraftaxClassicSymbolic", "CraftaxSymbolic"])
def test_rollout_frames_equal_upstream(name, device):
    import torch

    import stable_envs as se
    from tests.helpers.reference.craftax import frame as frame_of

    dev = _device(device)
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("torch has no CUDA device")
    lanes, steps = 8, 64
    with se.make(name, num_envs=lanes, device=device) as env:
        renderers = _renderers(env.reference_id)
        with jax.default_device(dev):
            for seed in (0, 1):
                env.reset(seed=seed)
                rng = np.random.default_rng(seed)
                for t in range(steps):
                    action = rng.integers(0, env.action_high + 1, (lanes, 1))
                    if t % 4:
                        env.step(action, frames=None)
                        continue
                    _assert_same(renderers, env._state, f"seed {seed}, step {t}")
                    # A frame the step returns comes from the same jitted call as the step.
                    frame = env.step(action)["frames"][:, -1].cpu().numpy()
                    expected = frame_of(renderers[3](env._state))
                    np.testing.assert_array_equal(frame, expected, err_msg=f"seed {seed}, step {t}")


@pytest.mark.parametrize("device", DEVICES)
@pytest.mark.parametrize("game", GAMES)
def test_varied_states_equal_upstream(game, device):
    """Every block and item type, other levels, night, sleep, mobs and projectiles, inventory contents."""
    from craftax.craftax_env import make_craftax_env_from_name

    dev = _device(device)
    renderers = _renderers(game)
    env = make_craftax_env_from_name(game, auto_reset=False)
    params = env.default_params
    vary = _vary_classic if "Classic" in game else _vary_craftax
    varied = jax.jit(jax.vmap(lambda reset_key, key: vary(env.reset(reset_key, params)[1], key)))
    lanes = 64
    with jax.default_device(dev):
        for batch in range(3):
            keys = jax.random.split(jax.random.PRNGKey(100 + batch), 2 * lanes)
            _assert_same(renderers, varied(keys[:lanes], keys[lanes:]), f"batch {batch}")
