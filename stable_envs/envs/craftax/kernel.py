"""The upstream Craftax simulator, vmapped over lanes and jitted, with State conversion to and from torch.

Arrays move between JAX and torch through DLPack, so they stay on their device.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import os
from dataclasses import dataclass

import torch

# JAX otherwise reserves most of the GPU at first use and leaves torch little of it.
os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.numpy as jnp  # noqa: E402
from craftax.craftax_env import make_craftax_env_from_name  # noqa: E402


@contextlib.contextmanager
def texture_lock(classic: bool):
    """Hold an exclusive lock beside Craftax's texture cache.

    Craftax builds its texture pickle on first use and writes it in place, so a process that finds
    the file while another is still writing it reads a truncated pickle. Building under this lock
    lets every process that goes through it see either no file or a complete one.
    """
    constants = importlib.import_module(f"craftax.{'craftax_classic' if classic else 'craftax'}.constants")
    try:
        import fcntl

        handle = open(constants.TEXTURE_CACHE_FILE + ".lock", "a")
    except (ImportError, OSError):
        yield
        return
    with handle:
        fcntl.flock(handle, fcntl.LOCK_EX)
        yield


@dataclass(frozen=True)
class Field:
    key: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype
    #: uint32 PRNG words, stored bit for bit as int32.
    words: bool

    @property
    def width(self) -> int:
        return int(torch.Size(self.shape).numel())


def _field(key, leaf) -> Field:
    words = leaf.dtype == jnp.uint32
    return Field(key, tuple(leaf.shape), torch.int32 if words else getattr(torch, str(leaf.dtype)), words)


def _strong(tree):
    """Drop weak types, so a reset, stepped or restored State reuses one compiled step."""
    return jax.tree.map(lambda x: jax.lax.convert_element_type(x, x.dtype), tree)


def _name(entry) -> str:
    for attr in ("name", "idx", "key"):
        if hasattr(entry, attr):
            return str(getattr(entry, attr))
    return str(entry)


class Kernel:
    """Jitted, vmapped reset, step and render for one Craftax id; holds no lane state."""

    def __init__(self, env_id: str):
        classic = "Classic" in env_id
        with texture_lock(classic):
            upstream = make_craftax_env_from_name(env_id, auto_reset=False)
            pixels = (
                upstream
                if "Pixels" in env_id
                else make_craftax_env_from_name(env_id.replace("Symbolic", "Pixels"), auto_reset=False)
            )
        params = upstream.default_params
        symbolic = pixels is not upstream
        self.num_actions = int(upstream.num_actions)

        # The key handling of stable-worldmodel's CraftaxWrapper: split, keep the first half, spend the second.
        def reset(key):
            kept, used = jax.random.split(key)
            return _strong(upstream.reset(used, params)[1]), kept

        def step(key, state, action):
            kept, used = jax.random.split(key)
            obs, state, reward, done, _ = upstream.step(used, state, action, params)
            return _strong(state), kept, reward, done, obs if symbolic else None

        def masked_step(key, state, action, active):
            new = jax.vmap(step)(key, state, action)

            def keep(fresh, old):
                return jnp.where(active.reshape(active.shape + (1,) * (fresh.ndim - 1)), fresh, old)

            state, key = jax.tree.map(keep, (new[0], new[1]), (state, key))
            outputs = [jnp.where(active, new[2], 0), new[3] & active]
            return state, key, *outputs, None if new[4] is None else keep(new[4], jnp.zeros_like(new[4]))

        def frame(state):
            return (jnp.clip(pixels.get_obs(state), 0.0, 1.0) * 255.0).astype(jnp.uint8)

        self._reset = jax.jit(jax.vmap(reset))
        self._step = jax.jit(jax.vmap(step))
        self._masked_step = jax.jit(masked_step)
        self._frame = jax.jit(jax.vmap(frame))
        template = jax.eval_shape(lambda k: upstream.reset(k, params)[1], jax.random.PRNGKey(0))
        paths, self._treedef = jax.tree_util.tree_flatten_with_path(template)
        self.fields = (
            *(_field(("engine", *map(_name, path)), leaf) for path, leaf in paths),
            _field(("key",), jax.eval_shape(jax.random.PRNGKey, 0)),
        )

    @staticmethod
    def device(device: torch.device):
        if device.type not in ("cpu", "cuda"):
            raise ValueError(f"Craftax runs on cpu or cuda, not {device}")
        try:
            return jax.devices("gpu" if device.type == "cuda" else "cpu")[device.index or 0]
        except RuntimeError as e:
            raise RuntimeError(
                "device='cuda' needs a CUDA build of JAX: pip install 'stable-envs[craftax-gpu]'"
            ) from e

    @staticmethod
    def seed_keys(seed: int, num_envs: int, device):
        with jax.default_device(device):
            keys = jnp.stack([jax.random.PRNGKey(seed + i) for i in range(num_envs)])
        return jax.device_put(keys, device)

    def reset(self, keys):
        return self._reset(keys)

    def step(self, keys, state, actions: torch.Tensor, active: torch.Tensor | None):
        actions = jnp.from_dlpack(actions.to(torch.int32).contiguous())
        if active is None:
            return self._step(keys, state, actions)
        return self._masked_step(keys, state, actions, jnp.from_dlpack(active.contiguous()))

    def render(self, state) -> torch.Tensor:
        return torch.from_dlpack(self._frame(state))

    def export(self, state, keys) -> dict[tuple[str, ...], torch.Tensor]:
        """Views of every leaf as ``(lanes, width)`` torch tensors; the caller copies them."""
        leaves = [*jax.tree.leaves(state), keys]
        out = {}
        for field, leaf in zip(self.fields, leaves, strict=True):
            value = torch.from_dlpack(jax.lax.bitcast_convert_type(leaf, jnp.int32) if field.words else leaf)
            out[field.key] = value.reshape(len(value), field.width)
        return out

    def restore(self, td, device: torch.device, *, copy: bool):
        """The stacked upstream state and wrapper keys held in a State TensorDict."""
        n = int(td.batch_size[0])
        leaves = []
        for field in self.fields:
            value = td[field.key].to(device=device, dtype=field.dtype).reshape(n, *field.shape)
            value = value.clone(memory_format=torch.contiguous_format) if copy else value.contiguous()
            array = jnp.from_dlpack(value)
            leaves.append(jax.lax.bitcast_convert_type(array, jnp.uint32) if field.words else array)
        return jax.tree_util.tree_unflatten(self._treedef, leaves[:-1]), leaves[-1]


@functools.cache
def kernel(env_id: str) -> Kernel:
    return Kernel(env_id)
