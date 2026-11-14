"""The batched JAX source of the Craftax classes: upstream Craftax under one ``jit(vmap(...))``.

The upstream env is built with ``auto_reset=False``. Lane ``i`` follows stable-worldmodel's ``CraftaxWrapper``
built with ``seed + i``: ``PRNGKey(seed + i)``, split, keep the first half and spend the second. The key split runs
inside the jitted call. Without frames, the Pixels step drops its observation, so XLA removes its render; the
Symbolic step returns its symbolic observation. With frames, the Pixels step returns its observation as uint8 frames,
and the Symbolic step also draws the frame with the pixel env of its game. Frames are ``clip(obs, 0, 1) * 255``
truncated to uint8, as stable-worldmodel renders them.
"""

from __future__ import annotations

import os

import numpy as np

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")


def is_jax_source(name: str) -> bool:
    """Whether class ``name`` has a batched JAX source."""
    import stable_envs as snv

    return snv.info(name)["family"] == "craftax"


def jax_device(device: str):
    import jax

    return jax.devices("gpu" if device.startswith("cuda") else "cpu")[0]


class CraftaxSource:
    """Jitted, vmapped upstream reset, step and frames of one Craftax class; inputs are committed to one JAX device."""

    def __init__(self, name: str, device: str = "cpu"):
        import jax
        import jax.numpy as jnp
        from craftax.craftax_env import make_craftax_env_from_name

        import stable_envs as snv

        self.upstream_id = snv.info(name)["reference_id"]
        self.device = jax_device(device)
        env = make_craftax_env_from_name(self.upstream_id, auto_reset=False)
        symbolic = "Symbolic" in self.upstream_id
        pixels = (
            make_craftax_env_from_name(self.upstream_id.replace("Symbolic", "Pixels"), auto_reset=False)
            if symbolic
            else env
        )
        params = env.default_params
        self._template = jax.eval_shape(lambda k: env.reset(k, params)[1], jax.random.PRNGKey(0))

        def to_uint8(obs):
            return (jnp.clip(obs, 0.0, 1.0) * 255.0).astype(jnp.uint8)

        def reset_lane(seed):
            kept, used = jax.random.split(jax.random.PRNGKey(seed))
            return env.reset(used, params)[1], kept

        def step_lane(key, state, action, frames):
            kept, used = jax.random.split(key)
            obs, state, reward, done, _ = env.step(used, state, action, params)
            out = (state, kept, reward, done)
            if frames:
                return (*out, obs, to_uint8(pixels.get_obs(state))) if symbolic else (*out, to_uint8(obs))
            return (*out, obs) if symbolic else out

        self._reset = jax.jit(jax.vmap(reset_lane))
        self._steps = {f: jax.jit(jax.vmap(lambda k, s, a, f=f: step_lane(k, s, a, f))) for f in (False, True)}
        self._render = jax.jit(jax.vmap(lambda s: to_uint8(pixels.get_obs(s))))

    def put(self, array):
        import jax

        return jax.device_put(array, self.device)

    def reset(self, seed: int, lanes: int):
        """The upstream state and wrapper keys of ``lanes`` lanes reset with ``seed + i``."""
        return self._reset(self.put(np.arange(lanes, dtype=np.int32) + np.int32(seed)))

    def step(self, keys, state, actions, frames: bool):
        """``(state, keys, reward, done[, obs][, frames])``; the arrays are device arrays, not yet ready."""
        return self._steps[frames](keys, state, actions)

    def render(self, state):
        """``(N, H, W, 3)`` uint8 frames of a stacked upstream state."""
        return self._render(state)

    def state_from(self, td):
        """The stacked upstream state held in a stable-envs State TensorDict, on this source's device.

        Each upstream leaf is the State leaf ``engine/<path>``, one ``(N, width)`` row per lane; uint32 leaves are
        stored bit for bit as int32.
        """
        import jax

        paths, treedef = jax.tree_util.tree_flatten_with_path(self._template)
        leaves = []
        for path, leaf in paths:
            key = ("engine", *(_name(p) for p in path))
            value = np.ascontiguousarray(td[key].cpu().numpy())
            value = value.reshape(value.shape[0], *leaf.shape)
            leaves.append(self.put(value.view(np.uint32) if leaf.dtype == np.uint32 else value.astype(leaf.dtype)))
        return jax.tree_util.tree_unflatten(treedef, leaves)


def _name(entry) -> str:
    for attr in ("name", "idx", "key"):
        if hasattr(entry, attr):
            return str(getattr(entry, attr))
    return str(entry)
