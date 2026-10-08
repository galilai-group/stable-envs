"""Named numeric render inputs for unchanged upstream Craftax pixel rendering."""

from __future__ import annotations

import hashlib

import numpy as np
import torch
from tensordict import TensorDict

from ..._adapters.craftax import decode

# Source-read inventory: both upstream render_craftax_pixels functions, including
# full Craftax's is_boss_vulnerable helper. Entity health/cooldowns are not drawn.
CLASSIC_ROOTS = frozenset(
    {
        "map",
        "player_position",
        "player_direction",
        "player_health",
        "player_food",
        "player_drink",
        "player_energy",
        "inventory",
        "is_sleeping",
        "light_level",
        "state_rng",
        "zombies",
        "cows",
        "skeletons",
        "arrows",
        "arrow_directions",
    }
)
FULL_ROOTS = frozenset(
    {
        "map",
        "item_map",
        "light_map",
        "player_position",
        "player_level",
        "player_direction",
        "player_health",
        "player_food",
        "player_drink",
        "player_energy",
        "player_mana",
        "player_xp",
        "player_dexterity",
        "player_strength",
        "player_intelligence",
        "inventory",
        "is_sleeping",
        "light_level",
        "state_rng",
        "melee_mobs",
        "passive_mobs",
        "ranged_mobs",
        "mob_projectiles",
        "mob_projectile_directions",
        "player_projectiles",
        "player_projectile_directions",
        "monsters_killed",
        "learned_spells",
        "sword_enchantment",
        "bow_enchantment",
        "armour_enchantments",
        "boss_timesteps_to_spawn_this_round",
    }
)
ENTITY_FIELDS = frozenset({"position", "mask", "type_id"})


def fields(adapter):
    roots = CLASSIC_ROOTS if adapter.classic else FULL_ROOTS
    return {
        key: value
        for key, value in adapter.engine.items()
        if key[0] in roots and (len(key) == 1 or key[0] == "inventory" or key[1] in ENTITY_FIELDS)
    }


def signature(adapter):
    return np.frombuffer(hashlib.sha256(b"craftax-render-v1" + adapter.signature.tobytes()).digest(), np.uint8).copy()


def project(adapter, state):
    adapter.validate(state)
    n = state.batch_size[0]
    data = {("draw", *key): state[("engine", *key)].clone() for key in fields(adapter)}
    data["renderer", "signature"] = torch.from_numpy(np.broadcast_to(signature(adapter), (n, 32)).copy()).to(
        adapter.device
    )
    return TensorDict(data, batch_size=[n], device=adapter.device)


def restore_draw(adapter, state):
    if not isinstance(state, TensorDict) or len(state.batch_size) != 1 or state.batch_size[0] < 1:
        raise ValueError("Craftax render inputs require a named numeric TensorDict batch")
    schema = {
        ("draw", *k): (v.shape, (torch.int64 if v.dtype == np.uint32 else torch.from_numpy(v.copy()).dtype))
        for k, v in fields(adapter).items()
    }
    schema["renderer", "signature"] = ((32,), torch.uint8)
    if set(state.keys(True, True)) != set(schema):
        raise ValueError("Invalid Craftax RenderState keys")
    n = state.batch_size[0]
    arrays = {}
    for key, (shape, dtype) in schema.items():
        value = state[key]
        if not torch.is_tensor(value) or value.dtype != dtype or value.shape != (n, *shape):
            raise ValueError(f"Invalid Craftax RenderState dtype/shape: {key}")
        arrays[key] = value.detach().cpu().numpy().copy()
        if not np.isfinite(arrays[key]).all():
            raise ValueError(f"Nonfinite Craftax RenderState: {key}")
    if not np.all(arrays["renderer", "signature"] == signature(adapter)):
        raise ValueError("Incompatible Craftax RenderState family/profile/device")
    if np.any(arrays["draw", "state_rng"] < 0) or np.any(arrays["draw", "state_rng"] >= 2**32):
        raise ValueError("Craftax render PRNG words must be int64 in [0, 2**32)")
    result = []
    for lane in range(n):
        row = {k: v.copy() for k, v in adapter.engine.items()}
        row.update({k: arrays[("draw", *k)][lane] for k in fields(adapter)})
        adapter.validate_engine(row, collision=False)
        result.append(decode(adapter.template, row))
    return result


def render(adapter, state, chunk_size=None):
    import jax

    states = restore_draw(adapter, state)
    if chunk_size is not None and (type(chunk_size) is not int or chunk_size <= 0):
        raise ValueError("chunk_size must be a positive integer")
    frames = []
    # Each source output has the same scalar compilation boundary regardless of
    # batch size/chunking. Do not assume vmap or eager floats round identically.
    for start in range(0, len(states), chunk_size or len(states)):
        outputs = [adapter.output_fn(s) for s in states[start : start + (chunk_size or len(states))]]
        jax.block_until_ready(outputs)
        frames.extend((np.clip(np.asarray(output), 0.0, 1.0) * 255.0).astype(np.uint8) for output in outputs)
    return torch.from_numpy(np.stack(frames)).to(adapter.device)
