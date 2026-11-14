"""The upstream Craftax simulator, vmapped over lanes and jitted, with state conversion to and from torch.

Arrays move between JAX and torch through DLPack, so they stay on their device. On the CPU, the lanes
are split over one XLA host device per usable CPU, because one XLA:CPU program uses only a few cores.
"""

from __future__ import annotations

import contextlib
import functools
import importlib
import os
from dataclasses import dataclass

import jax
import jax.numpy as jnp
import numpy as np
import torch

from ..._core.config import log
from ..._core.cpus import cpu_limit


def _ask_for_host_devices() -> bool:
    """Ask JAX for one CPU device per CPU that this process can use; False when JAX started before.

    JAX reads the count once, when it starts its CPU backend, and the count holds for the whole process.
    A count that the user set (``jax_num_cpu_devices``, ``JAX_NUM_CPU_DEVICES`` or
    ``--xla_force_host_platform_device_count`` in ``XLA_FLAGS``) is kept.
    """
    count = getattr(jax.config, "jax_num_cpu_devices", None)
    if count is None or count >= 0 or "xla_force_host_platform_device_count" in os.environ.get("XLA_FLAGS", ""):
        return True
    try:
        jax.config.update("jax_num_cpu_devices", cpu_limit())
    except RuntimeError:
        return False
    return True


# Craftax makes JAX arrays when it is imported, which starts the backends, so this comes first.
_HOST_DEVICES_ASKED = _ask_for_host_devices()

from craftax.craftax_env import make_craftax_env_from_name  # noqa: E402

from .render import renderer  # noqa: E402


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
class Lanes:
    """Where the lanes of one env live: ``count`` lanes, padded to ``rows``, split evenly over ``sharding``.

    Without a sharding, the rows are the lanes and they live on ``device``. Outputs that go to torch are
    gathered on ``device`` and the padding rows are dropped.
    """

    count: int
    rows: int
    device: jax.Device
    sharding: jax.sharding.Sharding | None

    def put(self, x):
        """Place ``count`` rows (a torch tensor, a host array, or a tuple of them) on the lanes' devices.

        Padding rows copy the first row. A torch tensor on ``device`` crosses through DLPack.
        """
        return jax.tree.map(self._put, x)

    def _put(self, a):
        if self.sharding is None:
            return jnp.from_dlpack(a) if torch.is_tensor(a) else jax.device_put(a, self.device)
        a = np.asarray(a)
        if self.rows != self.count:
            a = np.concatenate([a, np.repeat(a[:1], self.rows - self.count, 0)])
        return jax.make_array_from_callback(a.shape, self.sharding, lambda index: a[index])

    def torch(self, x):
        """Lane-major outputs (or a tuple of them, with None kept) as torch tensors of ``count`` rows.

        Split outputs are joined on the host in one copy; the shards are in lane order.
        """
        return jax.tree.map(self._torch, x)

    def _torch(self, a):
        if self.sharding is None:
            return torch.from_dlpack(a)
        rows = torch.cat([torch.from_dlpack(a.addressable_data(i)) for i in range(self.parts)])
        return rows if self.rows == self.count else rows[: self.count]

    @property
    def parts(self) -> int:
        return 1 if self.sharding is None else len(self.sharding.device_set)


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
        symbolic = "Pixels" not in env_id
        with texture_lock(classic):
            upstream = make_craftax_env_from_name(env_id, auto_reset=False)
            draw = renderer(classic, None if symbolic else upstream._render_fn)
        params = upstream.default_params
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
            # The flag is repeated per lane, so every output has the lane axis first.
            return state, keys, jnp.where(live, new[2], 0), new[3] & live, obs, jnp.broadcast_to(invalid, live.shape)

        # Upstream's pixel observation, scaled to uint8 as stable-worldmodel scales it on the host. The
        # barrier keeps XLA from folding the division into the scaling, which would round differently.
        def frames(state):
            obs = jax.lax.optimization_barrier(jax.vmap(draw)(state) / 255.0)
            return (jnp.clip(obs, 0.0, 1.0) * 255.0).astype(jnp.uint8)

        # The step, plus the packed state and the new frames when they are asked for.
        def program(snapshot: bool, frame: bool):
            def run(keys, state, actions, active):
                out = advance(keys, state, actions, active)
                return (*out, pack(out[0], out[1]) if snapshot else None, frames(out[0]) if frame else None)

            return run

        def render_blocks(blocks):
            return frames(unpack(blocks)[0])

        #: Each function and its donated arguments. The step and the reset donate the live lanes: each
        #: call writes the next state over the last one.
        self._functions = {
            "reset": (reset_all, (0,)),
            "pack": (pack, ()),
            "unpack": (unpack, ()),
            "frames": (frames, ()),
            "render_blocks": (render_blocks, ()),
            **{("step", s, f): (program(s, f), (0, 1)) for s in (False, True) for f in (False, True)},
        }
        self._jitted: dict = {}
        self._everyone: dict = {}

    def _jit(self, name, lanes: Lanes):
        """The jitted function ``name``; on split lanes, every output keeps the lanes split as the inputs are."""
        key = (name, lanes.sharding)
        fn = self._jitted.get(key)
        if fn is None:
            function, donate = self._functions[name]
            split = {} if lanes.sharding is None else {"out_shardings": lanes.sharding}
            fn = self._jitted[key] = jax.jit(function, donate_argnums=donate, **split)
        return fn

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
    def lanes(count: int, device) -> Lanes:
        """The placement of ``count`` lanes on ``device``; CPU lanes are split evenly over the host devices."""
        devices = [device, *(d for d in jax.devices("cpu") if d != device)] if device.platform == "cpu" else [device]
        per = -(-count // len(devices))
        used = -(-count // per)
        if used == 1:
            return Lanes(count, count, device, None)
        mesh = jax.sharding.Mesh(np.array(devices[:used]), ("lanes",))
        return Lanes(count, used * per, device, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("lanes")))

    @staticmethod
    def seed_keys(seed: int, lanes: Lanes):
        """The keys ``PRNGKey(seed + i)``; threefry keys of seeds in ``[0, 2**31)`` are ``[0, seed + i]``."""
        num_envs = lanes.count
        if 0 <= seed and seed + num_envs <= 2**31 and jax.config.jax_default_prng_impl == "threefry2x32":
            keys = np.zeros((num_envs, 2), np.uint32)
            keys[:, 1] = np.arange(seed, seed + num_envs, dtype=np.uint32)
        else:
            with jax.default_device(lanes.device):
                keys = jnp.stack([jax.random.PRNGKey(seed + i) for i in range(num_envs)])
        return lanes.put(np.asarray(keys))

    def views(self, blocks: list[torch.Tensor]) -> dict[tuple[str, ...], torch.Tensor]:
        """``(lanes, width)`` views of each leaf in freshly written torch blocks, which nothing else holds."""
        return self.split(list(blocks))

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

    def reset(self, keys, lanes: Lanes):
        state, keys, blocks = self._jit("reset", lanes)(keys)
        return state, keys, self.views(lanes.torch(blocks))

    def everyone(self, lanes: Lanes):
        """An all-true lane mask, made once per placement."""
        mask = self._everyone.get(lanes)
        if mask is None:
            mask = self._everyone[lanes] = lanes.put(np.ones(lanes.count, np.bool_))
        return mask

    def step(
        self,
        keys,
        state,
        actions: torch.Tensor,
        active: torch.Tensor | None,
        lanes: Lanes,
        snapshot: bool = True,
        frame: bool = False,
    ):
        """Advance the lanes and donate ``keys`` and ``state``.

        With ``snapshot``, the call also packs the new state; with ``frame``, it also draws the new frames.
        Every lane keeps its state when any action is outside the game's range, and the last output is
        that flag on the device.
        """
        actions = actions.to(torch.int32).contiguous()
        if active is None:
            actions, active = lanes.put(actions), self.everyone(lanes)
        else:
            actions, active = lanes.put((actions, active.to(torch.bool).contiguous()))
        out = self._jit(("step", snapshot, frame), lanes)(keys, state, actions, active)
        state, keys, reward, done, obs, invalid, blocks, frames = out
        reward, done, obs, blocks, frames = lanes.torch((reward, done, obs, blocks, frames))
        views = None if blocks is None else self.views(blocks)
        return state, keys, reward, done, obs, views, frames, invalid

    @staticmethod
    def read_flag(flag) -> bool:
        """Read a per-lane device flag with an explicit transfer, which waits only for the call that wrote it."""
        return bool(np.any(jax.device_get(flag)))

    def export(self, state, keys, lanes: Lanes) -> dict[tuple[str, ...], torch.Tensor]:
        return self.views(lanes.torch(self._jit("pack", lanes)(state, keys)))

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

    def restore(self, td, device: torch.device, lanes: Lanes):
        """The stacked upstream state and wrapper keys held in a state TensorDict, in buffers of their own."""
        return self._jit("unpack", lanes)(lanes.put(tuple(self._torch_blocks(td, device))))

    def render(self, state, lanes: Lanes) -> torch.Tensor:
        return lanes.torch(self._jit("frames", lanes)(state))

    def render_state(self, td, device: torch.device) -> torch.Tensor:
        """Frames of a state TensorDict; batches are padded to a power of two, so few sizes compile."""
        n = int(td.batch_size[0])
        blocks = self._torch_blocks(td, device)
        rows = 1 << max(n - 1, 0).bit_length()
        if rows != n:
            blocks = [torch.cat([b, b[:1].expand(rows - n, -1)]) for b in blocks]
        lanes = self.lanes(rows, self.device(device))
        frames = lanes.torch(self._jit("render_blocks", lanes)(lanes.put(tuple(b.contiguous() for b in blocks))))
        return frames if rows == n else frames[:n]


@functools.cache
def report_host_devices() -> None:
    """Log once when the CPU lanes cannot be split, because JAX started before Craftax asked for host devices."""
    if not _HOST_DEVICES_ASKED and len(jax.devices("cpu")) == 1 and cpu_limit() > 1:
        log(
            "WARNING",
            f"JAX started with one CPU device, so Craftax steps its CPU lanes on 1 of {cpu_limit()} CPUs. "
            "Make the first Craftax env before other JAX work, or set "
            f"XLA_FLAGS=--xla_force_host_platform_device_count={cpu_limit()} before you start Python.",
        )


def has_gpu() -> bool:
    """Whether JAX has a GPU backend."""
    try:
        return bool(jax.devices("gpu"))
    except RuntimeError:
        return False


@functools.cache
def kernel(env_id: str) -> Kernel:
    return Kernel(env_id)
