"""Delegated Craftax 1.5.0 semantics and owning numeric snapshots; no game logic."""

from __future__ import annotations

import dataclasses
import hashlib
import importlib
import importlib.metadata
import json
import sys
from functools import lru_cache

import numpy as np
import torch
from tensordict import TensorDict

from . import SimulatorAdapter

PROFILE = {
    "craftax": "1.5.0",
    "jax": "0.10.1",
    "jaxlib": "0.10.1",
    "flax": "0.12.7",
    "gymnax": "0.0.9",
    "numpy": "2.2.6",
    "gymnasium": "1.2.3",
    "chex": "0.1.92",
    "torch": "2.14.0",
    "tensordict": "0.14.2",
}


def numeric_leaves(value, path=()):
    if dataclasses.is_dataclass(value):
        for field in dataclasses.fields(value):
            yield from numeric_leaves(getattr(value, field.name), (*path, field.name))
    elif isinstance(value, tuple):
        for index, part in enumerate(value):
            yield from numeric_leaves(part, (*path, str(index)))
    elif value is not None:
        yield path, np.asarray(value)


def decode(template, values, prefix=()):
    """Recreate immutable upstream structs from named leaves; no serialized objects."""
    import jax.numpy as jnp

    if dataclasses.is_dataclass(template):
        return template.replace(
            **{
                f.name: decode(getattr(template, f.name), values, (*prefix, f.name))
                for f in dataclasses.fields(template)
            }
        )
    if isinstance(template, tuple):
        return tuple(decode(v, values, (*prefix, str(i))) for i, v in enumerate(template))
    if template is None:
        return None
    return jnp.asarray(np.asarray(values[prefix]).astype(np.asarray(template).dtype))


def check_profile():
    import jax

    if sys.version_info[:2] != (3, 12):
        raise RuntimeError("Craftax recovery is pinned to Python 3.12")
    for name, version in PROFILE.items():
        actual = importlib.metadata.version(name)
        if (actual.split("+")[0] if name == "torch" else actual) != version:
            raise RuntimeError(f"Craftax recovery requires {name}=={version}; installed {actual}")
    if jax.config.jax_enable_x64 or jax.config.jax_default_prng_impl != "threefry2x32":
        raise RuntimeError("Craftax requires x64 disabled and the threefry2x32 PRNG")
    if not jax.config.jax_threefry_partitionable:
        raise RuntimeError("Craftax requires partitionable Threefry; changing global JAX configuration is unsupported")


@lru_cache(maxsize=8)
def kernel(classic, pixels, platform):
    """Immutable upstream configuration and compiled functions, never instance RNG/State."""
    import jax
    from craftax.craftax_env import make_craftax_env_from_name

    check_profile()
    if platform not in {"cpu", "gpu"} or not jax.devices(platform):
        raise RuntimeError(f"Craftax JAX backend unavailable: {platform}")
    name = "Craftax-" + ("Classic-" if classic else "") + ("Pixels" if pixels else "Symbolic") + "-v1"
    upstream = make_craftax_env_from_name(name, auto_reset=False)
    params = upstream.default_params

    def reset(key):
        retained, supplied = jax.random.split(key)
        output, state = upstream.reset(supplied, params)
        return output, state, retained

    def step(key, state, action):
        retained, supplied = jax.random.split(key)
        output, state, reward, done, info = upstream.step(supplied, state, action, params)
        return output, state, reward, done, info, retained

    reset_fn = jax.jit(reset, backend=platform)
    step_fn = jax.jit(step, backend=platform)
    output_fn = jax.jit(upstream.get_obs, backend=platform)
    terminal_fn = jax.jit(lambda state: upstream.is_terminal(state, params), backend=platform)
    _, template, _ = reset_fn(jax.random.PRNGKey(0))
    jax.block_until_ready(template)
    return upstream, reset_fn, step_fn, output_fn, terminal_fn, template


class CraftaxAdapter(SimulatorAdapter):
    def __init__(self, classic, pixels, batch, device, seed):
        import jax

        if isinstance(batch, bool) or not isinstance(batch, int) or batch < 1:
            raise ValueError("num_envs must be a positive integer")
        self.classic, self.pixels, self.num_envs = classic, pixels, batch
        self.retains_symbolic_output = not classic and not pixels
        self.symbolic_cache = None
        self.device = torch.device(device)
        if self.device.type not in {"cpu", "cuda"}:
            raise ValueError("Craftax supports CPU or CUDA through its pinned JAX profile")
        self.platform = "gpu" if self.device.type == "cuda" else "cpu"
        self.upstream, self.reset_fn, self.step_fn, self.output_fn, self.terminal_fn, self.template = kernel(
            classic, pixels, self.platform
        )
        self.engine = dict(numeric_leaves(self.template))
        self.fixed = {}
        for group, obj in [("params", self.upstream.default_params), ("static", self.upstream.static_env_params)]:
            self.fixed.update({(group, *key): value for key, value in numeric_leaves(obj)})
        descriptor = {
            "schema": 1,
            "key_encoding": "legacy uint32 words stored losslessly as int64 in [0, 2**32)",
            "profile": PROFILE,
            "python": "3.12",
            "classic": classic,
            "pixels": pixels,
            "platform": self.platform,
            "x64": False,
            "prng": "threefry2x32/partitionable/legacy-uint32-2",
            "params": dataclasses.asdict(self.upstream.default_params),
            "static": dataclasses.asdict(self.upstream.static_env_params),
            "engine": [(key, list(v.shape), str(v.dtype)) for key, v in self.engine.items()],
        }
        if self.retains_symbolic_output:
            descriptor["source_cache"] = "last_symbolic:float32[8268]; exact source-returned output"
        self.signature = np.frombuffer(
            hashlib.sha256(json.dumps(descriptor, sort_keys=True).encode()).digest(), np.uint8
        ).copy()
        self.fields = {
            ("engine", *k): (v.shape, (torch.int64 if v.dtype == np.uint32 else torch.from_numpy(v.copy()).dtype))
            for k, v in self.engine.items()
        }
        self.fields.update({k: (v.shape, torch.from_numpy(v.copy()).dtype) for k, v in self.fixed.items()})
        self.fields.update(
            {
                ("meta", "signature"): ((32,), torch.uint8),
                ("rng", "wrapper_key"): ((2,), torch.int64),
                ("lifecycle", "reward"): ((), torch.float32),
                ("lifecycle", "terminated"): ((), torch.bool),
                ("lifecycle", "truncated"): ((), torch.bool),
            }
        )
        if self.retains_symbolic_output:
            self.fields["source_cache", "last_symbolic"] = ((8268,), torch.float32)
        self.keys = [jax.random.PRNGKey(s) for s in self.seeds(seed)]
        self.states = None
        self.reward = np.zeros(batch, np.float32)
        self.terminated = np.zeros(batch, bool)
        self.truncated = np.zeros(batch, bool)

    def seeds(self, seed):
        if isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32:
            raise ValueError("seed must be an integer in [0, 2**32)")
        return [(int(seed) + i) % 2**32 for i in range(self.num_envs)]

    def reset(self, seed=None, options=None):
        import jax

        check_profile()
        if options:
            raise ValueError("Craftax registered defaults do not expose reset options")
        keys = self.keys if seed is None else [jax.random.PRNGKey(s) for s in self.seeds(seed)]
        results = [self.reset_fn(k) for k in keys]
        jax.block_until_ready(results)
        self._last_outputs = results
        cached = np.stack([np.asarray(x[0]) for x in results]) if self.retains_symbolic_output else None
        self.symbolic_cache = cached
        self.states = [x[1] for x in results]
        self.keys = [x[2] for x in results]
        self.reward.fill(0)
        self.terminated.fill(False)
        self.truncated.fill(False)
        return self.get_state()

    def get_state(self):
        if self.states is None:
            raise RuntimeError("Call reset before reading Craftax State")
        rows = [dict(numeric_leaves(s)) for s in self.states]
        values = {("engine", *k): np.stack([r[k] for r in rows]) for k in self.engine}
        values.update({k: np.broadcast_to(v, (self.num_envs, *v.shape)).copy() for k, v in self.fixed.items()})
        values.update(
            {
                ("meta", "signature"): np.broadcast_to(self.signature, (self.num_envs, 32)).copy(),
                ("rng", "wrapper_key"): np.stack([np.asarray(k) for k in self.keys]),
                ("lifecycle", "reward"): self.reward.copy(),
                ("lifecycle", "terminated"): self.terminated.copy(),
                ("lifecycle", "truncated"): self.truncated.copy(),
            }
        )
        if self.retains_symbolic_output:
            values["source_cache", "last_symbolic"] = self.symbolic_cache.copy()
        return TensorDict(
            {
                k: torch.from_numpy(v.astype(np.int64) if v.dtype == np.uint32 else v.copy()).to(self.device)
                for k, v in values.items()
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def validate(self, state, batch=None):
        check_profile()
        if not isinstance(state, TensorDict):
            raise TypeError("Craftax restoration requires exhaustive named numeric State")
        if len(state.batch_size) != 1 or state.batch_size[0] < 1 or batch is not None and state.batch_size[0] != batch:
            raise ValueError("Invalid Craftax State batch")
        if set(state.keys(True, True)) != set(self.fields):
            raise ValueError("Craftax State keys differ; neither symbolic output nor RenderState restores dynamics")
        n = state.batch_size[0]
        arrays = {}
        for key, (shape, dtype) in self.fields.items():
            value = state[key]
            if not torch.is_tensor(value) or value.dtype != dtype or value.shape != (n, *shape):
                raise ValueError(f"Craftax State dtype/shape mismatch: {key}")
            arrays[key] = value.detach().cpu().numpy().copy()
            if not np.isfinite(arrays[key]).all():
                raise ValueError(f"Non-finite Craftax State: {key}")
        if not np.all(arrays["meta", "signature"] == self.signature):
            raise ValueError("Incompatible Craftax engine, presentation, device, schema or profile")
        for k, v in self.fixed.items():
            if not np.all(arrays[k] == v):
                raise ValueError(f"Unsupported Craftax static/default configuration: {k}")
        for key in [("engine", "state_rng"), ("rng", "wrapper_key")]:
            if np.any(arrays[key] < 0) or np.any(arrays[key] >= 2**32):
                raise ValueError("Craftax PRNG words must be int64 in [0, 2**32)")
        decoded = []
        for lane in range(n):
            row = {k: arrays[("engine", *k)][lane] for k in self.engine}
            self.validate_engine(row)
            obj = decode(self.template, row)
            terminal = bool(np.asarray(self.terminal_fn(obj)))
            if terminal != bool(arrays["lifecycle", "terminated"][lane]) or arrays["lifecycle", "truncated"][lane]:
                raise ValueError("Craftax lifecycle flags disagree with source termination")
            decoded.append(obj)
        return arrays, decoded

    def validate_engine(self, row, *, collision=True):
        module = "craftax.craftax_classic.constants" if self.classic else "craftax.craftax.constants"
        constants = importlib.import_module(module)
        shape = self.engine["map",].shape[-2:]
        pos = row["player_position",]
        if np.any(pos < 0) or np.any(pos >= shape):
            raise ValueError("Craftax player position is outside the map")
        if not 1 <= row["player_direction",] <= 4 or row["timestep",] < 0:
            raise ValueError("Craftax direction or timestep is invalid")
        if not 0 <= row["light_level",] <= 1:
            raise ValueError("Craftax light level is outside [0, 1]")
        if np.any(row["map",] < 0) or np.any(row["map",] > max(v.value for v in constants.BlockType)):
            raise ValueError("Craftax map contains an invalid block enum")
        for key, value in row.items():
            if key[0] == "inventory" and (np.any(value < 0) or np.any(value > (9 if self.classic else 99))):
                raise ValueError(f"Craftax inventory out of source bounds: {key}")
        # Source bounds used by digit/texture indexing and intrinsic updates.
        limits = {"player_food": 9, "player_drink": 9, "player_energy": 9, "player_health": 9}
        if not self.classic:
            for attr in ["player_strength", "player_dexterity", "player_intelligence"]:
                if not 1 <= row[attr,] <= self.upstream.default_params.max_attribute:
                    raise ValueError(f"Craftax attribute out of source bounds: {attr}")
            limits.update(
                player_health=8 + row["player_strength",],
                player_food=7 + 2 * row["player_dexterity",],
                player_drink=7 + 2 * row["player_dexterity",],
                player_energy=7 + 2 * row["player_dexterity",],
                player_mana=6 + 3 * row["player_intelligence",],
            )
            for key in ["sword_enchantment", "bow_enchantment", "armour_enchantments"]:
                if np.any(row[key,] < 0) or np.any(row[key,] > 2):
                    raise ValueError("Craftax enchantment enum out of bounds")
            for key, maximum in [("pickaxe", 4), ("sword", 4), ("bow", 1), ("armour", 2)]:
                if np.any(row["inventory", key] > maximum):
                    raise ValueError("Craftax equipment tier out of bounds")
            if np.any(row["light_map",] < 0) or np.any(row["light_map",] > 1):
                raise ValueError("Craftax light map out of bounds")
            if np.any(row["monsters_killed",] < 0) or row["player_xp",] < 0:
                raise ValueError("Craftax progress counters must be nonnegative")
        for key, maximum in limits.items():
            value = row[key,]
            if value > maximum or (key != "player_health" and value < 0):
                raise ValueError(f"Craftax intrinsic out of source bounds: {key}")
        plant_mask = row["growing_plants_mask",]
        plants = row["growing_plants_positions",][plant_mask]
        if np.any(plants < 0) or np.any(plants >= shape) or np.any(row["growing_plants_age",] < 0):
            raise ValueError("Craftax plant state out of bounds")
        groups = ["zombies", "cows", "skeletons"] if self.classic else ["melee_mobs", "passive_mobs", "ranged_mobs"]
        projectiles = ["arrows"] if self.classic else ["mob_projectiles", "player_projectiles"]
        occupied = np.zeros_like(row["mob_map",])
        for group in groups + projectiles:
            mask, positions = row[group, "mask"], row[group, "position"]
            if np.any(positions[mask] < 0) or np.any(positions[mask] >= shape):
                raise ValueError(f"Craftax active entity out of map: {group}")
            if group in groups:
                for index in zip(*np.nonzero(mask), strict=True):
                    p = tuple(positions[index])
                    at = p if self.classic else (index[0], *p)
                    if occupied[at]:
                        raise ValueError("Craftax active collision entities overlap")
                    occupied[at] = True
        if collision and not np.array_equal(occupied, row["mob_map",]):
            raise ValueError("Craftax mob_map disagrees with active collision entities")
        if not self.classic:
            if not 0 <= row["player_level",] < self.engine["map",].shape[0]:
                raise ValueError("Craftax player level is invalid")
            if not np.array_equal(np.sort(row["potion_mapping",]), np.arange(6)):
                raise ValueError("Craftax potion mapping must be a permutation")
            if np.any(row["item_map",] < 0) or np.any(row["item_map",] > max(v.value for v in constants.ItemType)):
                raise ValueError("Craftax item map enum is invalid")
            for group in groups + projectiles:
                if np.any(row[group, "type_id"] < 0) or np.any(row[group, "type_id"] >= 8):
                    raise ValueError("Craftax entity type is invalid")

    def set_state(self, state):
        import jax.numpy as jnp

        arrays, decoded = self.validate(state, self.num_envs)
        keys = [jnp.asarray(k.astype(np.uint32)) for k in arrays["rng", "wrapper_key"]]
        # Every lane and all allocations have succeeded before changing owned state.
        rewards = arrays["lifecycle", "reward"].copy()
        terminated = arrays["lifecycle", "terminated"].copy()
        truncated = arrays["lifecycle", "truncated"].copy()
        cached = arrays["source_cache", "last_symbolic"].copy() if self.retains_symbolic_output else None
        self.states, self.keys = decoded, keys
        self.symbolic_cache = cached
        self.reward, self.terminated, self.truncated = rewards, terminated, truncated

    def step(self, actions, *, active=None):
        import jax

        check_profile()
        if self.states is None:
            raise RuntimeError("Call reset before stepping Craftax")
        active = np.ones(self.num_envs, bool) if active is None else np.asarray(active.detach().cpu(), dtype=bool)
        if self.terminated[active].any():
            raise RuntimeError("Craftax terminal State requires explicit reset")
        a = np.asarray(actions.detach().cpu() if torch.is_tensor(actions) else actions)
        if a.shape == (self.num_envs, 1):
            a = a[:, 0]
        if (
            a.shape != (self.num_envs,)
            or a.dtype.kind not in "iu"
            or np.any(a < 0)
            or np.any(a >= self.upstream.num_actions)
        ):
            raise ValueError("Craftax actions must be one valid discrete integer ID per lane")
        results = [
            self.step_fn(key, state, int(action)) if active[i] else self._last_outputs[i]
            for i, (key, state, action) in enumerate(zip(self.keys, self.states, a, strict=True))
        ]
        jax.block_until_ready(results)
        self._last_outputs = results
        cached = np.stack([np.asarray(r[0]) for r in results]) if self.retains_symbolic_output else None
        self.symbolic_cache = cached
        self.states = [r[1] for r in results]
        self.keys = [r[5] for r in results]
        self.reward = np.asarray([float(r[2]) for r in results], np.float32)
        self.terminated = np.asarray([bool(r[3]) for r in results], bool)
        return self.get_state()

    def close(self):
        self.symbolic_cache = None
        self.states = None
        self.keys = []
