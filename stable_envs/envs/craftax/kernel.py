"""The upstream Craftax simulator, vmapped over lanes and jitted, with state conversion to and from torch.

Arrays move between JAX and torch through DLPack, so they stay on their device.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import torch
from craftax.craftax_env import make_craftax_env_from_name


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
    """One upstream leaf: its state key, per-lane shape, and its columns in a packed block."""

    key: tuple[str, ...]
    shape: tuple[int, ...]
    dtype: torch.dtype
    #: uint32 PRNG words, stored bit for bit as int32.
    words: bool
    block: int
    start: int

    @property
    def width(self) -> int:
        return int(torch.Size(self.shape).numel())


def _fields(leaves) -> tuple[tuple[Field, ...], tuple[torch.dtype, ...]]:
    blocks: list[torch.dtype] = []
    widths: list[int] = []
    fields = []
    for key, leaf in leaves:
        words = leaf.dtype == jnp.uint32
        dtype = torch.int32 if words else getattr(torch, str(leaf.dtype))
        if dtype not in blocks:
            blocks.append(dtype)
            widths.append(0)
        block = blocks.index(dtype)
        fields.append(Field(key, tuple(leaf.shape), dtype, words, block, widths[block]))
        widths[block] += fields[-1].width
    return tuple(fields), tuple(blocks)


def _strong(tree):
    """Drop weak types, so a reset, stepped or restored state reuses one compiled step."""
    return jax.tree.map(lambda x: jax.lax.convert_element_type(x, x.dtype), tree)


def _name(entry) -> str:
    for attr in ("name", "idx", "key"):
        if hasattr(entry, attr):
            return str(getattr(entry, attr))
    return str(entry)


class Kernel:
    """Jitted, vmapped reset, step and render for one Craftax id; holds no lane state.

    State leaves leave the jitted calls packed into one ``(lanes, columns)`` block per dtype, so a
    snapshot costs three DLPack hand-offs and no copy beyond the one XLA writes.
    """

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
        template = jax.eval_shape(lambda k: upstream.reset(k, params)[1], jax.random.PRNGKey(0))
        paths, treedef = jax.tree_util.tree_flatten_with_path(template)
        self.fields, self.blocks = _fields(
            [
                *((("engine", *map(_name, path)), leaf) for path, leaf in paths),
                (("key",), jax.eval_shape(jax.random.PRNGKey, 0)),
            ]
        )
        fields = self.fields
        self._slices = tuple((f.key, f.block, f.start, f.width) for f in fields)
        #: The first field of each block and the block's width.
        self._leads = tuple(
            (next(f.key for f in fields if f.block == b), sum(f.width for f in fields if f.block == b))
            for b in range(len(self.blocks))
        )

        def pack(state, key):
            columns = [[] for _ in self.blocks]
            for f, leaf in zip(fields, [*jax.tree.leaves(state), key], strict=True):
                leaf = leaf.reshape(leaf.shape[0], f.width)
                columns[f.block].append(jax.lax.bitcast_convert_type(leaf, jnp.int32) if f.words else leaf)
            return tuple(jnp.concatenate(c, axis=1) for c in columns)

        whole = {f.key for f in fields if f.width == sum(g.width for g in fields if g.block == f.block)}

        def unpack(blocks):
            leaves = []
            for f in fields:
                leaf = blocks[f.block][:, f.start : f.start + f.width].reshape(-1, *f.shape)
                if f.key in whole:
                    # A whole block would come back as the input buffer, which the next step cannot donate.
                    leaf = jnp.copy(leaf)
                leaves.append(jax.lax.bitcast_convert_type(leaf, jnp.uint32) if f.words else leaf)
            return jax.tree_util.tree_unflatten(treedef, leaves[:-1]), leaves[-1]

        # The key handling of stable-worldmodel's CraftaxWrapper: split, keep the first half, spend the second.
        def reset(key):
            kept, used = jax.random.split(key)
            return _strong(upstream.reset(used, params)[1]), kept

        def step(key, state, action):
            kept, used = jax.random.split(key)
            obs, state, reward, done, _ = upstream.step(used, state, action, params)
            return _strong(state), kept, reward, done, obs if symbolic else None

        def reset_all(keys):
            state, keys = jax.vmap(reset)(keys)
            return state, keys, pack(state, keys)

        high = int(upstream.action_space(params).n) - 1

        # One program serves full, masked and rejected steps, so no variant compiles in the middle of a run.
        def advance(keys, state, actions, active):
            invalid = jnp.any((actions < 0) | (actions > high))
            new = jax.vmap(step)(keys, state, actions)
            live = active & ~invalid

            def keep(fresh, old):
                return jnp.where(live.reshape(live.shape + (1,) * (fresh.ndim - 1)), fresh, old)

            state, keys = jax.tree.map(keep, (new[0], new[1]), (state, keys))
            obs = None if new[4] is None else keep(new[4], jnp.zeros_like(new[4]))
            return state, keys, jnp.where(live, new[2], 0), new[3] & live, obs, invalid

        def step_all(keys, state, actions, active):
            state, keys, *outputs = advance(keys, state, actions, active)
            return state, keys, *outputs, pack(state, keys)

        def frames(state):
            return (jnp.clip(jax.vmap(pixels.get_obs)(state), 0.0, 1.0) * 255.0).astype(jnp.uint8)

        def render_blocks(blocks):
            return frames(unpack(blocks)[0])

        # The live lanes are donated: each call writes the next state over the last one.
        self._reset = jax.jit(reset_all, donate_argnums=0)
        self._step = jax.jit(step_all, donate_argnums=(0, 1))
        self._advance = jax.jit(advance, donate_argnums=(0, 1))
        self._pack = jax.jit(pack)
        self._unpack = jax.jit(unpack)
        self._frames = jax.jit(frames)
        self._render_blocks = jax.jit(render_blocks)
        self._everyone: dict = {}

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
        """The keys ``PRNGKey(seed + i)``; threefry keys of seeds in ``[0, 2**31)`` are ``[0, seed + i]``."""
        if 0 <= seed and seed + num_envs <= 2**31 and jax.config.jax_default_prng_impl == "threefry2x32":
            keys = np.zeros((num_envs, 2), np.uint32)
            keys[:, 1] = np.arange(seed, seed + num_envs, dtype=np.uint32)
            return jax.device_put(keys, device)
        with jax.default_device(device):
            keys = jnp.stack([jax.random.PRNGKey(seed + i) for i in range(num_envs)])
        return jax.device_put(keys, device)

    def views(self, blocks) -> dict[tuple[str, ...], torch.Tensor]:
        """``(lanes, width)`` views of each leaf in freshly written blocks, which nothing else holds."""
        return self.split([torch.from_dlpack(b) for b in blocks])

    def split(self, blocks: list[torch.Tensor]) -> dict[tuple[str, ...], torch.Tensor]:
        """``(..., width)`` views of each leaf in torch blocks of shape ``(..., columns)``."""
        return {key: blocks[block].narrow(-1, start, width) for key, block, start, width in self._slices}

    def bases(self, td) -> list[torch.Tensor] | None:
        """The ``(lanes, columns)`` blocks behind a state that ``views`` made, or None for any other state."""
        n = int(td.batch_size[0])
        out = []
        for (key, width), dtype in zip(self._leads, self.blocks, strict=True):
            base = td.get(key)._base
            if base is None or base.dtype != dtype or tuple(base.shape) != (n, width) or not base.is_contiguous():
                return None
            out.append(base)
        return out

    def reset(self, keys):
        state, keys, blocks = self._reset(keys)
        return state, keys, self.views(blocks)

    def everyone(self, num_envs: int, device):
        """An all-true lane mask, made once per batch size and device."""
        mask = self._everyone.get((num_envs, device))
        if mask is None:
            mask = self._everyone[(num_envs, device)] = jax.device_put(np.ones(num_envs, np.bool_), device)
        return mask

    def step(self, keys, state, actions: torch.Tensor, active: torch.Tensor | None, device, snapshot: bool = True):
        """Advance the lanes and donate ``keys`` and ``state``.

        Every lane keeps its state when any action is outside the game's range, and the last output is
        that flag on the device.
        """
        actions = jnp.from_dlpack(actions.to(torch.int32).contiguous())
        if active is None:
            active = self.everyone(actions.shape[0], device)
        else:
            active = jnp.from_dlpack(active.to(torch.bool).contiguous())
        if snapshot:
            state, keys, reward, done, obs, invalid, blocks = self._step(keys, state, actions, active)
            views = self.views(blocks)
        else:
            state, keys, reward, done, obs, invalid = self._advance(keys, state, actions, active)
            views = None
        obs = None if obs is None else torch.from_dlpack(obs)
        return state, keys, torch.from_dlpack(reward), torch.from_dlpack(done), obs, views, invalid

    @staticmethod
    def read_flag(flag) -> bool:
        """Read a device flag with an explicit transfer, which waits only for the call that wrote it."""
        return bool(jax.device_get(flag))

    def export(self, state, keys) -> dict[tuple[str, ...], torch.Tensor]:
        return self.views(self._pack(state, keys))

    def _torch_blocks(self, td, device: torch.device) -> list[torch.Tensor]:
        """Copy a state TensorDict into fresh per-dtype blocks, checking its keys, shapes and values."""
        n = int(td.batch_size[0])
        columns = [[] for _ in self.blocks]
        for f in self.fields:
            try:
                value = td[f.key]
            except KeyError:
                raise ValueError(f"State is missing {'.'.join(f.key)}") from None
            if tuple(value.shape) != (n, f.width):
                raise ValueError(f"State field {'.'.join(f.key)} has shape {tuple(value.shape)}, not {(n, f.width)}")
            columns[f.block].append(value.to(device=device, dtype=f.dtype))
        blocks = [torch.cat(c, dim=1) for c in columns]
        if not all(bool(torch.isfinite(b).all()) for b in blocks if b.is_floating_point()):
            raise ValueError("State contains non-finite values")
        return blocks

    def _blocks(self, td, device: torch.device):
        return tuple(jnp.from_dlpack(b) for b in self._torch_blocks(td, device))

    def restore(self, td, device: torch.device):
        """The stacked upstream state and wrapper keys held in a state TensorDict, in buffers of their own."""
        return self._unpack(self._blocks(td, device))

    def render(self, state) -> torch.Tensor:
        return torch.from_dlpack(self._frames(state))

    def render_state(self, td, device: torch.device) -> torch.Tensor:
        """Frames of a state TensorDict; batches are padded to a power of two, so few sizes compile."""
        n = int(td.batch_size[0])
        blocks = self._torch_blocks(td, device)
        rows = 1 << max(n - 1, 0).bit_length()
        if rows != n:
            blocks = [torch.cat([b, b[:1].expand(rows - n, -1)]) for b in blocks]
        frames = torch.from_dlpack(self._render_blocks(tuple(jnp.from_dlpack(b.contiguous()) for b in blocks)))
        return frames if rows == n else frames[:n]


def has_gpu() -> bool:
    """Whether JAX has a GPU backend."""
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


@functools.cache
def kernel(env_id: str) -> Kernel:
    return Kernel(env_id)
