"""The upstream Craftax simulator, vmapped over environments and jitted, with state conversion to and from torch.

Arrays move between JAX and torch through DLPack, so they stay on their device. On the CPU, the environments
are split over XLA host devices, because one XLA:CPU program uses only a few cores.
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

from ..._core.physics import PhysicsLayout
from ..._core.state_schema import Field as SchemaField
from ..._core.state_schema import StateSchema
from .params import CLASSIC_PARAMS, CRAFTAX_PARAMS
from .render import renderer


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


#: The fewest environments that one XLA host device steps. One XLA:CPU program that runs alone spreads its small
#: operations over the threads of XLA's pool, which is slower than programs that run at the same time on several
#: devices; each split also adds a cost per device and per call (placing the inputs, launching the program, joining the
#: outputs). docs/environments/craftax.md (Simulation) says how the value was measured.
MIN_ENVS_PER_HOST_DEVICE = 4

#: XLA:GPU reads the condition of a while loop on the host after each iteration. A Craftax step runs short loops
#: of fixed length (``lax.scan`` over mobs and block types), so the GPU step and reset unroll them and run without
#: these round trips. Unrolling changes no result. The sequence scan stays a loop, so its compile time stays short.
GPU_UNROLL = {"xla_gpu_enable_while_loop_unrolling": "WHILE_LOOP_UNROLLING_FULL_UNROLL"}


@dataclass(frozen=True)
class Instances:
    """Where the environments of one env live: ``count`` environments, padded to ``rows``, split evenly over ``sharding``.

    Without a sharding, the rows are the environments and they live on ``device``. Outputs that go to torch are
    gathered on ``device`` and the padding rows are dropped.
    """

    count: int
    rows: int
    device: jax.Device
    sharding: jax.sharding.Sharding | None

    def put(self, x):
        """Place ``count`` rows (a torch tensor, a host array, or a pytree of them) on the environments' devices.

        Padding rows copy the first row. A torch tensor on ``device`` crosses through DLPack.
        """
        return jax.tree.map(self._put, x)

    def _put(self, a):
        if self.sharding is None:
            return jnp.from_dlpack(a) if torch.is_tensor(a) else jax.device_put(a, self.device)
        a = np.asarray(a)
        if self.rows != self.count:
            a = np.concatenate([a, np.repeat(a[:1], self.rows - self.count, 0)])
        return jax.device_put(a, self.sharding)

    def torch(self, x):
        """Environment-major outputs (or a pytree of them, with None kept) as torch tensors of ``count`` rows.

        Split outputs are joined on the host in one copy, without the padding rows; the shards are in environment order.
        """
        return jax.tree.map(self._torch, x)

    def _torch(self, a):
        if self.sharding is None:
            return torch.from_dlpack(a)
        per = self.rows // self.parts
        shards = [torch.from_dlpack(a.addressable_data(i)) for i in range(self.parts)]
        shards[-1] = shards[-1][: self.count - per * (self.parts - 1)]
        return torch.cat(shards)

    @property
    def parts(self) -> int:
        return 1 if self.sharding is None else len(self.sharding.device_set)


@dataclass(frozen=True)
class Field:
    """One upstream leaf: its state key, per-environment shape, and its columns in a packed block."""

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


def _dtype(value) -> np.dtype:
    if isinstance(value, bool):
        return np.dtype(np.bool_)
    return np.dtype(np.int32 if isinstance(value, int) else np.float32)


def _strong(tree):
    """Drop weak types, so a reset, stepped or restored state reuses one compiled step."""
    return jax.tree.map(lambda x: jax.lax.convert_element_type(x, x.dtype), tree)


def _name(entry) -> str:
    for attr in ("name", "idx", "key"):
        if hasattr(entry, attr):
            return str(getattr(entry, attr))
    return str(entry)


class Kernel:
    """Jitted, vmapped reset, step and render for one Craftax id; holds no environment state.

    State leaves leave the jitted calls packed into one ``(environments, columns)`` block per dtype, so a
    snapshot costs three DLPack hand-offs and no copy beyond the one XLA writes.
    """

    def __init__(self, env_id: str):
        classic = "Classic" in env_id
        symbolic = "Pixels" not in env_id
        with texture_lock(classic):
            upstream = make_craftax_env_from_name(env_id, auto_reset=False)
            draw = renderer(classic, None if symbolic else upstream._render_fn)
        params = upstream.default_params
        #: ``(variation path, EnvParams field, dtype, default)`` of each per-environment parameter.
        self.params = tuple(
            (path, attr, _dtype(getattr(params, attr)), getattr(params, attr))
            for path, attr in (CLASSIC_PARAMS if classic else CRAFTAX_PARAMS)
        )
        n = len(self.params)
        #: The physics layout of the parameters: one column each, in ``params`` order.
        self.physics_layout = PhysicsLayout(
            leaves=tuple(path for path, *_ in self.params),
            widths=(1,) * n,
            offsets=tuple(range(n)),
            nominal=tuple(float(default) for *_, default in self.params),
        )
        engine = jax.eval_shape(lambda k: upstream.reset(k, params)[1], jax.random.PRNGKey(0))
        instance_params = {path: jax.ShapeDtypeStruct((), dtype) for path, _, dtype, _ in self.params}
        paths, treedef = jax.tree_util.tree_flatten_with_path((engine, instance_params))

        def key_of(path):
            if path[0].idx == 0:
                return ("engine", *map(_name, path[1:]))
            return ("physics", *path[1].key.split("."))

        self.fields, self.blocks = _fields(
            [*((key_of(path), leaf) for path, leaf in paths), (("key",), jax.eval_shape(jax.random.PRNGKey, 0))]
        )
        fields = self.fields
        section = "craftax_classic" if classic else "craftax"
        #: The State: one group per packed block, each field a slice of its block.
        self.schema = StateSchema(
            env_id,
            [
                SchemaField("/".join(f.key), f.shape, f.dtype, f"{section}/{str(f.dtype).removeprefix('torch.')}")
                for f in fields
            ],
        )
        #: The State key of each block, in block order.
        self._groups = tuple(group.path for group in self.schema.groups.values())

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

        # Default environments take the default EnvParams as constants, as upstream and stable-worldmodel do, so
        # XLA folds them the same way. Varied environments take theirs from the state.
        def instance(values, varied):
            if not varied:
                return params
            return params.replace(**{attr: values[path] for path, attr, _, _ in self.params})

        # The key handling of stable-worldmodel's CraftaxWrapper: split, keep the first half, spend the second.
        def reset(key, values, varied):
            kept, used = jax.random.split(key)
            return (_strong(upstream.reset(used, instance(values, varied))[1]), values), kept

        def step(key, state, action, varied):
            engine, values = state
            kept, used = jax.random.split(key)
            obs, engine, reward, done, _ = upstream.step(used, engine, action, instance(values, varied))
            return (_strong(engine), values), kept, reward, done, obs if symbolic else None

        def reset_all(keys, values, varied):
            state, keys = jax.vmap(functools.partial(reset, varied=varied))(keys, values)
            return state, keys, pack(state, keys)

        high = int(upstream.action_space(params).n) - 1

        #: The action that pads a sequence to its compiled length. It is outside the range that a device action
        #: is clamped to, so it cannot be a caller's action.
        self.pad_action = high + 2

        def rejected(actions):
            return jnp.any((actions < 0) | (actions > high))

        # Environments that are not live keep their state and keys, and give zero outputs.
        def transition(keys, state, actions, live, varied):
            new = jax.vmap(functools.partial(step, varied=varied))(keys, state, actions)

            def keep(fresh, old):
                return jnp.where(live.reshape(live.shape + (1,) * (fresh.ndim - 1)), fresh, old)

            state, keys = jax.tree.map(keep, (new[0], new[1]), (state, keys))
            obs = None if new[4] is None else keep(new[4], jnp.zeros_like(new[4]))
            return state, keys, jnp.where(live, new[2], 0), new[3] & live, obs

        # One program serves full, masked and rejected steps, so no variant compiles in the middle of a run.
        # The flag is repeated per environment, so every output has the environment axis first.
        def advance(keys, state, actions, active, varied):
            invalid = rejected(actions)
            out = transition(keys, state, actions, active & ~invalid, varied)
            return *out, jnp.broadcast_to(invalid, active.shape)

        # Upstream's pixel observation, scaled to uint8 as stable-worldmodel scales it on the host. The
        # barrier keeps XLA from folding the division into the scaling, which would round differently.
        def frames(state):
            obs = jax.lax.optimization_barrier(jax.vmap(draw)(state) / 255.0)
            return (jnp.clip(obs, 0.0, 1.0) * 255.0).astype(jnp.uint8)

        # The step, plus the packed state and the new frames when they are asked for.
        def program(snapshot: bool, frame: bool):
            def run(keys, state, actions, active, varied):
                out = advance(keys, state, actions, active, varied)
                return (*out, pack(out[0], out[1]) if snapshot else None, frames(out[0][0]) if frame else None)

            return run

        # The steps of ``(environments, P)`` actions in one call, as ``P`` calls of ``advance`` make them: an environment that
        # ends stops. Steps whose action is ``pad_action`` pad the sequence and change nothing. ``states`` packs
        # the state of every step, ``frame`` ("none", "last" or "all") draws the frames of the last or of every
        # step, and ``packed`` packs the last state. Every environment keeps its state when any action is outside the
        # game's range.
        def sequence(states: bool, frame: str, packed: bool):
            def run(keys, state, actions, varied):
                pad = actions == self.pad_action
                invalid = jnp.any(((actions < 0) | (actions > high)) & ~pad)

                def body(carry, xs):
                    keys, state, active = carry
                    a, padding = xs
                    live = active & ~invalid & ~padding
                    state, keys, reward, done, obs = transition(keys, state, jnp.where(padding, 0, a), live, varied)
                    extra = (pack(state, keys) if states else None, frames(state[0]) if frame == "all" else None)
                    return (keys, state, active & ~done), (active, reward, done, obs, *extra)

                active = jnp.ones(actions.shape[0], jnp.bool_)
                (keys, state, _), out = jax.lax.scan(body, (keys, state, active), (actions.T, pad.T))
                out = jax.tree.map(lambda x: jnp.swapaxes(x, 0, 1), out)
                last = frames(state[0]) if frame == "last" else None
                final = pack(state, keys) if packed else None
                return state, keys, *out, final, last, jnp.broadcast_to(invalid, active.shape)

            return run

        def render_blocks(blocks):
            return frames(unpack(blocks)[0][0])

        #: Each function and its donated arguments. A name that ends in ``varied`` steps with the parameter
        #: values in the state when ``varied`` is true, and with the defaults otherwise. The step and the
        #: reset donate the live environments: each call writes the next state over the last one.
        self._functions = {
            "pack": (pack, ()),
            "unpack": (unpack, ()),
            "frames": (frames, ()),
            "render_blocks": (render_blocks, ()),
        }
        for varied in (False, True):
            self._functions[("reset", varied)] = (functools.partial(reset_all, varied=varied), (0,))
            for s in (False, True):
                for f in (False, True):
                    run = functools.partial(program(s, f), varied=varied)
                    self._functions[("step", s, f, varied)] = (run, (0, 1))
            for s in (False, True):
                for f in ("none", "last", "all"):
                    for p in (False, True):
                        run = functools.partial(sequence(s, f, p), varied=varied)
                        self._functions[("sequence", s, f, p, varied)] = (run, (0, 1))
        self._jitted: dict = {}
        self._everyone: dict = {}
        one = jax.tree.map(lambda x: jax.ShapeDtypeStruct((1, *x.shape), x.dtype), engine)
        #: The ``(H, W, 3)`` shape of one frame.
        self.frame_shape = tuple(jax.eval_shape(frames, one).shape[1:])

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

    def _jit(self, name, instances: Instances):
        """The jitted function ``name``; on split environments, every output keeps the environments split as the inputs are."""
        unroll = instances.device.platform == "gpu" and isinstance(name, tuple) and name[0] in ("step", "reset")
        key = (name, instances.sharding, unroll)
        fn = self._jitted.get(key)
        if fn is None:
            function, donate = self._functions[name]
            options = {}
            if instances.sharding is not None:
                options["out_shardings"] = instances.sharding
            if unroll:
                options["compiler_options"] = GPU_UNROLL
            fn = self._jitted[key] = jax.jit(function, donate_argnums=donate, **options)
        return fn

    def instances(self, count: int, device) -> Instances:
        """The placement of ``count`` environments on ``device``.

        CPU environments are split evenly over the host devices, with at least ``MIN_ENVS_PER_HOST_DEVICE`` environments on
        each device.
        """
        devices = [device, *(d for d in jax.devices("cpu") if d != device)] if device.platform == "cpu" else [device]
        parts = max(1, min(len(devices), count // MIN_ENVS_PER_HOST_DEVICE))
        per = -(-count // parts)
        used = -(-count // per)
        if used == 1:
            return Instances(count, count, device, None)
        mesh = jax.sharding.Mesh(np.array(devices[:used]), ("instances",))
        return Instances(
            count, used * per, device, jax.sharding.NamedSharding(mesh, jax.sharding.PartitionSpec("instances"))
        )

    @staticmethod
    def seed_keys(seed: int, instances: Instances):
        """The keys ``PRNGKey(seed + i)``; threefry keys of seeds in ``[0, 2**31)`` are ``[0, seed + i]``."""
        num_envs = instances.count
        if 0 <= seed and seed + num_envs <= 2**31 and jax.config.jax_default_prng_impl == "threefry2x32":
            keys = np.zeros((num_envs, 2), np.uint32)
            keys[:, 1] = np.arange(seed, seed + num_envs, dtype=np.uint32)
        else:
            with jax.default_device(instances.device):
                keys = jnp.stack([jax.random.PRNGKey(seed + i) for i in range(num_envs)])
        return instances.put(np.asarray(keys))

    def leaves(self, blocks) -> dict[tuple[str, ...], torch.Tensor]:
        """The State leaves of torch blocks of shape ``(..., columns)``: each block under its group key."""
        return dict(zip(self._groups, blocks, strict=True))

    def varied(self, values: dict[str, np.ndarray]) -> bool:
        """Whether any environment has a parameter value other than the default."""
        return any(bool((np.asarray(values[path]) != default).any()) for path, _, _, default in self.params)

    def reset(self, keys, values: dict[str, np.ndarray], instances: Instances, varied: bool):
        """Reset every environment with its parameter values and donate ``keys``.

        The state keeps the values for later steps. Returns the state, the keys and the packed state in torch.
        """
        values = instances.put({path: np.asarray(values[path], dtype) for path, _, dtype, _ in self.params})
        state, keys, blocks = self._jit(("reset", varied), instances)(keys, values)
        return state, keys, instances.torch(blocks)

    def everyone(self, instances: Instances):
        """An all-true environment mask, made once per placement."""
        mask = self._everyone.get(instances)
        if mask is None:
            mask = self._everyone[instances] = instances.put(np.ones(instances.count, np.bool_))
        return mask

    def step(
        self,
        keys,
        state,
        actions: torch.Tensor,
        active: torch.Tensor | None,
        instances: Instances,
        snapshot: bool = True,
        frame: bool = False,
        varied: bool = False,
    ):
        """Advance the environments and donate ``keys`` and ``state``; returns the outputs of the jitted call.

        The new state and keys come first, so that the caller can hold them before ``outputs`` converts
        the rest. With ``snapshot``, the call also packs the new state; with ``frame``, it also draws the
        new frames. Every environment keeps its state when any action is outside the game's range. ``varied``
        steps with the parameter values in the state instead of the defaults.
        """
        actions = actions.to(torch.int32).contiguous()
        if active is None:
            actions, active = instances.put(actions), self.everyone(instances)
        else:
            actions, active = instances.put((actions, active.to(torch.bool).contiguous()))
        return self._jit(("step", snapshot, frame, varied), instances)(keys, state, actions, active)

    def outputs(self, out, instances: Instances):
        """The reward, ends, observation, State leaves and frames of a step in torch, and its range flag.

        The flag stays on the device.
        """
        _, _, reward, done, obs, invalid, blocks, frames = out
        reward, done, obs, blocks, frames = instances.torch((reward, done, obs, blocks, frames))
        views = None if blocks is None else self.leaves(blocks)
        return reward, done, obs, views, frames, invalid

    def sequence(
        self,
        keys,
        state,
        actions: torch.Tensor,
        instances: Instances,
        states: bool,
        frames: str,
        packed: bool,
        varied: bool = False,
    ):
        """Advance ``(environments, P)`` actions in one call and donate ``keys`` and ``state``.

        Returns the outputs of the jitted call, the new state and keys first. ``states`` packs the state of
        every step, ``frames`` ("none", "last" or "all") draws the frames of the last or of every step, and
        ``packed`` packs the last state. Steps whose action is ``pad_action`` change nothing.
        """
        actions = instances.put(actions.to(torch.int32).contiguous())
        return self._jit(("sequence", states, frames, packed, varied), instances)(keys, state, actions)

    def sequence_outputs(self, out, instances: Instances):
        """The outputs of ``sequence`` in torch, with the range flag still on the device.

        Returns the ``(environments, P)`` valid flags, rewards and ends, the observations, the packed states of every
        step, the frames of every step, the State leaves of the last state, the last frames, and the flag.
        """
        _, _, *rest, invalid = out
        valid, reward, done, obs, blocks, every, final, last = instances.torch(tuple(rest))
        return valid, reward, done, obs, blocks, every, None if final is None else self.leaves(final), last, invalid

    @staticmethod
    def read_flag(flag) -> bool:
        """Read a per-environment device flag with an explicit transfer, which waits only for the call that wrote it."""
        return bool(np.any(jax.device_get(flag)))

    def export(self, state, keys, instances: Instances) -> dict[tuple[str, ...], torch.Tensor]:
        return self.leaves(instances.torch(self._jit("pack", instances)(state, keys)))

    def _torch_blocks(self, td, device: torch.device) -> list[torch.Tensor]:
        """Copy the blocks of a state TensorDict to ``device``, checking its groups, shapes and values."""
        self.schema.validate(td, finite=True)
        # A copy, so that the jitted calls that donate the blocks never free the caller's State.
        return [td.get(key).to(device, copy=True).contiguous() for key in self._groups]

    def values(self, td) -> dict[str, np.ndarray]:
        """The per-environment parameter values that a checked state TensorDict holds, copied to the host at once."""
        n = int(td.batch_size[0])
        leaves = [
            self.schema.get(td, f"physics.{path}").detach().reshape(n, 1).to(torch.float64) for path, *_ in self.params
        ]
        return self.from_rows(torch.cat(leaves, dim=1).cpu().numpy())

    def from_rows(self, rows: np.ndarray) -> dict[str, np.ndarray]:
        """Per-environment parameter values from float64 rows with one column per parameter, in ``params`` order."""
        return {
            path: rows[:, i].astype(np.float32) if dtype == np.float32 else np.rint(rows[:, i]).astype(dtype)
            for i, (path, _, dtype, _) in enumerate(self.params)
        }

    def restore(self, td, device: torch.device, instances: Instances):
        """The stacked upstream state and wrapper keys held in a state TensorDict, in buffers of their own."""
        return self._jit("unpack", instances)(instances.put(tuple(self._torch_blocks(td, device))))

    def render(self, state, instances: Instances) -> torch.Tensor:
        return instances.torch(self._jit("frames", instances)(state[0]))

    def render_state(self, td, device: torch.device) -> torch.Tensor:
        """Frames of a state TensorDict; batches are padded to a power of two, so few sizes compile."""
        n = int(td.batch_size[0])
        blocks = self._torch_blocks(td, device)
        if n == 0:
            return torch.empty((0, *self.frame_shape), dtype=torch.uint8, device=device)
        rows = 1 << (n - 1).bit_length()
        if rows != n:
            blocks = [torch.cat([b, b[:1].expand(rows - n, -1)]) for b in blocks]
        instances = self.instances(rows, self.device(device))
        frames = instances.torch(
            self._jit("render_blocks", instances)(instances.put(tuple(b.contiguous() for b in blocks)))
        )
        return frames if rows == n else frames[:n]


@functools.cache
def kernel(env_id: str) -> Kernel:
    return Kernel(env_id)
