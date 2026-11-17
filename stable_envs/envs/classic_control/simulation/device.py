"""Classic-control environments held and stepped on a CUDA device, with each step replayed as one CUDA graph.

The live State is the group tensors of its schema, one per dtype. A step copies the actions into a fixed buffer
and replays a graph. The graph runs one Triton kernel for the domain, which also writes the step outputs into a
packed byte buffer and the environment's render rows, and copies the group tensors into the same buffer. One
copy of that buffer gives the State and outputs that the step returns, so they own their memory. The step reads
nothing back to the host. Without Triton, the graph replays the batched Torch kernel.

A reset draws from each environment's PCG64 stream on the host, with NumPy's seeding and ``uniform`` repeated on arrays
of generators (``pcg``). It gives Gymnasium's initial states and generator states, as the CPU backends do, so a
seed gives the same initial states on every backend. One copy moves them to the device; a reset reads nothing back.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from gymnasium.envs.classic_control import utils as gym_utils
from tensordict import TensorDict

from ...._core.config import log
from ...._core.graph_step import result_builder
from ...._core.tensordicts import build_tensordict, to_device
from ...._execution.protocol import SimulatorAdapter
from . import pcg
from .batched import _DISCRETE, _DYNAMIC, _FIELD, _KERNELS, MAX_EPISODE_STEPS, _Torch
from .local import (
    _PHYSICS_SPACES,
    BEYOND,
    CLASSIC_FIELDS,
    ELAPSED,
    HAS_BEYOND,
    HAS_LAST_U,
    IS_FLOAT32,
    RNG_HAS_UINT32,
    RNG_INCREMENT,
    RNG_STATE,
    RNG_UINTEGER,
    STATES,
    physics_field,
    validate_classic_state,
)
from .triton_step import HAVE_TRITON, launcher

_OBSERVATION_WIDTH = {"cartpole": 4, "pendulum": 3, "acrobot": 6, "mountain_car": 2, "mountain_car_continuous": 2}
_SIM_NAMES = {
    "cartpole": ("x", "x_dot", "theta", "theta_dot"),
    "pendulum": ("theta", "theta_dot", "torque"),
    "acrobot": ("theta1", "theta2", "dtheta1", "dtheta2"),
    "mountain_car": ("position", "velocity"),
    "mountain_car_continuous": ("position", "velocity"),
}
_RENDER_COLUMNS = {"cartpole": (0, 2), "pendulum": (0, 2), "acrobot": (0, 1), "mountain_car": (0,)}
_RENDER_COLUMNS["mountain_car_continuous"] = _RENDER_COLUMNS["mountain_car"]
_TORCH = _Torch(torch.device("cpu"))
#: Prefix of the packed-buffer regions that hold the second copy of each State group.
_STATES_COPY = "states:"
# The step kernels and the reset rely on the simulation variables being the first columns of the float64 group.
for _domain, _names in _SIM_NAMES.items():
    assert [STATES[_domain].slices[name].start for name in _names] == list(range(len(_names)))


def _regions(domain: str) -> list[tuple[str, torch.dtype, int]]:
    """``(name, dtype, width)`` of each region of the packed buffer: the State groups, then the step outputs.

    ``observation`` holds the observation and ``info_state`` side by side; ``flags`` holds ``terminated`` and
    ``truncated``. A second copy of each group (``states:<group key>``) and two all-true columns (``ones``) give
    the ``states`` and the masks of a single step, so its whole result is one copy of the buffer.
    """
    regions = [(key, group.dtype, group.size) for key, group in STATES[domain].groups.items()]
    width = _OBSERVATION_WIDTH[domain]
    return [
        *regions,
        ("reward", torch.float64, 1),
        ("observation", torch.float32, 2 * width),
        ("flags", torch.bool, 2),
        *((f"{_STATES_COPY}{key}", dtype, size) for key, dtype, size in regions),
        ("ones", torch.bool, 2),
    ]



def _offsets(regions, num_envs: int) -> tuple[list[int], int]:
    """Byte offset of each region in the packed buffer, and the buffer size; each starts at a multiple of 8."""
    offsets, offset = [], 0
    for _, dtype, width in regions:
        offsets.append(offset)
        offset += -(-num_envs * width * dtype.itemsize // 8) * 8
    return offsets, offset


def _blocks(buffer: torch.Tensor, regions, num_envs: int) -> dict[str, torch.Tensor]:
    """``{region name: [num_envs, width] view}`` of a packed byte buffer."""
    offsets, _ = _offsets(regions, num_envs)
    return {
        name: buffer[offset : offset + num_envs * width * dtype.itemsize].view(dtype).view(num_envs, width)
        for (name, dtype, width), offset in zip(regions, offsets, strict=True)
    }


def _outputs_of(blocks: dict, width: int) -> dict[str, torch.Tensor]:
    """The step outputs as views of the output regions in ``blocks``."""
    flags = blocks["flags"]
    return {
        "reward": blocks["reward"][:, 0],
        "terminated": flags[:, 0],
        "truncated": flags[:, 1],
        "observation": blocks["observation"][:, :width],
        "info_state": blocks["observation"][:, width:],
    }


def _advance(domain, fields, out, a, active, sim, render, sim_names, render_columns):
    """One step of every environment, written into the live ``fields``, the outputs ``out`` and the render rows.

    ``active=None`` steps all environments.
    """
    old = {key: fields[_FIELD.get(key, key)][:, 0] for key in _DYNAMIC[domain]}
    params = {name: fields[physics_field(name)][:, 0] for name in _PHYSICS_SPACES[domain]}
    if domain in _DISCRETE:
        a, dtype = a.long(), None
    else:
        a, dtype = _TORCH.floats(a)
    new, observation, reward, terminated = _KERNELS[domain](_TORCH, old, params, a, dtype)
    stored = fields[ELAPSED][:, 0]
    if active is None:
        elapsed = stored + 1
        truncated = elapsed >= MAX_EPISODE_STEPS[domain]
    else:
        new = {key: torch.where(active, new[key], old[key]) for key in old}
        elapsed = stored + active
        truncated = active & (elapsed >= MAX_EPISODE_STEPS[domain])
        observation = [torch.where(active, column, 0.0) for column in observation]
        reward = torch.where(active, reward, 0.0)
        terminated = active & terminated
    observation = torch.stack(observation, 1).float()
    rows = torch.stack([new[name] for name in sim_names], 1)
    drawn = torch.stack([new[sim_names[index]] for index in render_columns], 1).float()
    for key, value in new.items():
        old[key].copy_(value)
    stored.copy_(elapsed)
    out["reward"].copy_(reward)
    out["terminated"].copy_(terminated)
    out["truncated"].copy_(truncated)
    out["observation"].copy_(observation)
    out["info_state"].copy_(observation)
    sim.copy_(rows)
    render.copy_(drawn)


_warned: set = set()


def _warn_once(what: str, error: BaseException) -> None:
    if what not in _warned:
        _warned.add(what)
        log("WARNING", f"Classic control on CUDA: {what} failed ({type(error).__name__}: {error}). Steps are slower.")


def _reset_bounds(domain: str, options: dict | None):
    """Gymnasium's ``reset`` bounds of each initial-state variable, with its ``options``, in draw order."""
    if domain == "pendulum":
        x = gym_utils.verify_number_and_cast(options.get("x_init", np.pi) if options else np.pi)
        y = gym_utils.verify_number_and_cast(options.get("y_init", 1.0) if options else 1.0)
        return [(-x, x), (-y, y)]
    default = {"cartpole": (-0.05, 0.05), "acrobot": (-0.1, 0.1)}.get(domain, (-0.6, -0.4))
    low, high = gym_utils.maybe_parse_reset_bounds(options, *default)
    return [(low, high)] * (4 if domain in ("cartpole", "acrobot") else 1)


class DeviceClassicAdapter(SimulatorAdapter):
    """Batched classic-control environments whose State and steps stay on one CUDA device; Gymnasium draws the resets.

    ``sink`` is ``(sim, render_rows, sim_names, render_columns)`` of the environment. Each step also
    writes the new ``sim`` rows and their render rows, so the environment needs no copy of the State.
    """

    #: Steps write the environment's ``sim`` and render rows.
    writes_render_state = True

    def __init__(self, domain: str, num_envs: int, device="cuda", seed: int = 0, sink=None):
        if domain not in CLASSIC_FIELDS:
            raise ValueError(f"Unsupported classic-control domain {domain!r}")
        self.device = torch.device(device)
        if self.device.type != "cuda":
            raise ValueError(f"DeviceClassicAdapter needs a CUDA device, got {self.device}")
        if self.device.index is None:
            self.device = torch.device("cuda", torch.cuda.current_device())
        self.domain, self.num_envs, self.fields = domain, int(num_envs), CLASSIC_FIELDS[domain]
        self._schema = schema = STATES[domain]
        self._regions = _regions(domain)
        self._width = _OBSERVATION_WIDTH[domain]
        offsets, self._nbytes = _offsets(self._regions, self.num_envs)
        # Per region: its first element in the packed buffer viewed as the region dtype, and its width.
        self._splits = [
            (name, dtype, offset // dtype.itemsize, width)
            for (name, dtype, width), offset in zip(self._regions, offsets, strict=True)
        ]
        self._paths = {key: tuple(key.split("/")) for key in schema.groups}
        #: ``{group key: [num_envs, size] tensor}``: the live State, which the step kernels write.
        self._live = {
            key: torch.zeros(self.num_envs, group.size, dtype=group.dtype, device=self.device)
            for key, group in schema.groups.items()
        }
        #: ``{field name: [num_envs, width] view}`` of the live groups, cut once.
        self._fields = schema.views(self._live)
        self._packed = self._buffer()
        self._packed_blocks = _blocks(self._packed, self._regions, self.num_envs)
        self._outputs = _outputs_of(self._packed_blocks, self._width)
        # The State groups lead the buffer, and their second copy has the same byte layout.
        regions = [name for name, _, _ in self._regions]
        self._group_bytes = offsets[len(schema.groups)]
        self._copy_offset = offsets[regions.index(_STATES_COPY + next(iter(schema.groups)))]
        self._packed_blocks["ones"].fill_(True)
        self._results: dict = {}
        for name, (_, _, default) in _PHYSICS_SPACES[domain].items():
            self._fields[physics_field(name)].fill_(default)
        if sink is None:
            names = _SIM_NAMES[domain]
            sim = torch.zeros(self.num_envs, len(names), dtype=torch.float64, device=self.device)
            render = torch.zeros(self.num_envs, len(_RENDER_COLUMNS[domain]), dtype=torch.float32, device=self.device)
            sink = (sim, render, names, _RENDER_COLUMNS[domain])
        self._sim, self._render, self._sim_names, self._render_columns = sink
        # The kernels write the rows in the environment classes' layout.
        self._fused = (
            HAVE_TRITON
            and tuple(self._sim_names) == _SIM_NAMES[domain]
            and tuple(self._render_columns) == _RENDER_COLUMNS[domain]
            and self._sim.is_contiguous()
            and self._render.is_contiguous()
        )
        self._launch = None
        # Each environment's PCG64 generator, held on the host for resets: (state, inc, has_uint32, uinteger).
        self._seed = int(seed)
        self._rng: tuple | None = None
        self._active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._actions: dict = {}
        self._graphs: dict = {}
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("DeviceClassicAdapter is closed")

    def _buffer(self) -> torch.Tensor:
        return torch.empty(self._nbytes, dtype=torch.uint8, device=self.device)

    def _pack(self, blocks) -> None:
        for key, live in self._live.items():
            blocks[key].copy_(live)

    def _unpack(self, buffer: torch.Tensor, steps: int | None = None) -> dict:
        """Views of every region of packed buffers: one buffer, or ``steps`` buffers stacked in time."""
        n = self.num_envs
        flats: dict = {}
        out: dict = {}
        for name, dtype, offset, width in self._splits:
            flat = flats.get(dtype)
            if flat is None:
                flat = flats[dtype] = buffer.view(dtype)
            if steps is None:
                out[name] = flat.as_strided((n, width), (width, 1), offset)
            else:
                out[name] = flat.as_strided((n, steps, width), (width, self._nbytes // dtype.itemsize, 1), offset)
        return out

    def _state(self, views: dict, batch: list[int]) -> TensorDict:
        """The State whose group leaves are ``views``."""
        tree: dict = {}
        for key, path in self._paths.items():
            node = tree
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = views[key]
        return build_tensordict(tree, batch, self.device)

    def configure(self, values: Mapping[str, np.ndarray]) -> None:
        """Set each environment's physical constants; the live environments keep their state."""
        self._check_open()
        for name, rows in values.items():
            rows = np.asarray(rows, np.float64)
            if rows.shape != (self.num_envs,):
                raise ValueError("Physical parameters must have one scalar per environment")
            self._fields[physics_field(name)][:, 0].copy_(torch.from_numpy(rows))

    def reset(self, seed: int | None = None, options: dict | None = None) -> TensorDict:
        self._check_open()
        bounds = _reset_bounds(self.domain, options)
        if seed is None and self._rng is None:
            seed = self._seed
        n = self.num_envs
        if seed is not None:
            seed = int(seed)
            if seed < 0 or seed + n > 1 << 64:
                raise ValueError(f"Seed must be a non-negative integer below 2**64 - num_envs, got {seed}")
            state, inc = pcg.seed(np.arange(n, dtype=np.uint64) + np.uint64(seed))
            zero = np.zeros(n, np.int64)
            self._rng = (state, inc, zero, zero)
        assert self._rng is not None
        state, inc, has_uint32, uinteger = self._rng
        draws, state = pcg.doubles(state, inc, len(bounds))
        self._rng = (state, inc, has_uint32, uinteger)
        # ``Generator.uniform`` returns ``low + (high - low) * random()``.
        values = [low + (high - low) * u for (low, high), u in zip(bounds, draws, strict=True)]
        if self.domain == "acrobot":
            # Acrobot stores its initial state as float32.
            values = [value.astype(np.float32).astype(np.float64) for value in values]
        sim_names = _SIM_NAMES[self.domain]
        host = self._schema.new(n, "cpu")
        views = self._schema.views(host, numpy=True)
        for name, value in zip(sim_names, values, strict=False):
            views[name][:, 0] = value
        views[RNG_STATE][...], views[RNG_INCREMENT][...] = pcg.to_limbs(state), pcg.to_limbs(inc)
        views[RNG_HAS_UINT32][...], views[RNG_UINTEGER][...] = has_uint32[:, None], uinteger[:, None]
        drawn = to_device(host, self.device)
        # Gymnasium's reset clears the episode fields: elapsed steps, ``last_u`` and ``steps_beyond_terminated``.
        # The groups take the drawn values and zeros whole, except the physical constants after the simulation
        # variables in the float64 group.
        for key, group in self._schema.groups.items():
            live, value = self._live[key], drawn.get(self._paths[key])
            if group.dtype == torch.float64:
                live[:, : len(sim_names)].copy_(value[:, : len(sim_names)])
            else:
                live.copy_(value)
        return self.get_state()

    def get_state(self) -> TensorDict:
        """A State whose groups are views of one new copy of the live groups."""
        self._check_open()
        buffer = self._buffer()
        self._pack(_blocks(buffer, self._regions, self.num_envs))
        return self._state(self._unpack(buffer), [self.num_envs])

    def stack_states(self, states) -> TensorDict | None:
        """Stack States of this adapter's steps along a new time axis with one copy; None for other States."""
        first = next(iter(self._paths.values()))
        buffers = []
        for state in states:
            leaf = state.get(first, None)
            if not torch.is_tensor(leaf):
                return None
            storage = leaf.untyped_storage()
            if storage.nbytes() != self._nbytes or storage.device != self.device:
                return None
            buffers.append(torch.empty(0, dtype=torch.uint8, device=self.device).set_(storage))
        views = self._unpack(torch.stack(buffers).view(-1), len(buffers))
        return self._state(views, [self.num_envs, len(buffers)])

    def validate(self, state: TensorDict) -> None:
        self._check_open()
        validate_classic_state(self.domain, state, batch=self.num_envs)

    def set_state(self, state: TensorDict) -> None:
        self._check_open()
        validate_classic_state(self.domain, state, batch=self.num_envs)
        self._load(state)
        # The next reset continues each environment's restored PCG64 stream.
        key = f"{self.domain}/int64"
        rng = self._schema.views(
            {key: state.get(self._paths[key]).cpu()},
            [RNG_STATE, RNG_INCREMENT, RNG_HAS_UINT32, RNG_UINTEGER],
            numpy=True,
        )
        self._rng = (
            pcg.from_limbs(rng[RNG_STATE]),
            pcg.from_limbs(rng[RNG_INCREMENT]),
            rng[RNG_HAS_UINT32][:, 0].copy(),
            rng[RNG_UINTEGER][:, 0].copy(),
        )

    def _load(self, state: TensorDict) -> None:
        for key, live in self._live.items():
            live.copy_(state.get(self._paths[key]))
        # Hold the values as Gymnasium holds them: an absent ``last_u`` or ``steps_beyond_terminated`` reads as 0.
        fields = self._fields
        if self.domain == "cartpole":
            fields[BEYOND].mul_(fields[HAS_BEYOND])
        elif self.domain == "pendulum":
            fields["torque"].mul_(fields[HAS_LAST_U])
        elif self.domain == "mountain_car_continuous":
            narrow = fields[IS_FLOAT32]
            for name in ("position", "velocity"):
                column = fields[name]
                column.copy_(torch.where(narrow, column.float().double(), column))

    def _kernel(self, actions: torch.Tensor, masked: bool):
        """The step for one action buffer: the domain's Triton kernel, or the batched Torch kernel."""
        fused = self._fused and actions.dtype in (torch.int64, torch.float32, torch.float64)
        if fused and self._launch is None:
            self._launch = launcher(
                self.domain, self.num_envs, self.device, MAX_EPISODE_STEPS[self.domain], self._width
            )
        launch = self._launch if fused else None
        active, sim, render = self._active, self._sim, self._render
        narrow = actions.dtype == torch.float32
        live, packed = self._live, self._packed_blocks
        # The kernels take the float64, int64 and bool State groups; a domain with no bool field passes the flags.
        groups = (
            live[f"{self.domain}/float64"],
            live[f"{self.domain}/int64"],
            live.get(f"{self.domain}/bool", packed["flags"]),
            packed["reward"],
            packed["observation"],
            packed["flags"],
        )

        def step():
            if launch is not None:
                launch(groups, actions, active, sim, render, masked, narrow)
            else:
                _advance(
                    self.domain, self._fields, self._outputs, actions, active if masked else None, sim, render,
                    tuple(self._sim_names), tuple(self._render_columns),
                )  # fmt: skip
            self._pack(packed)
            copy = self._packed[self._copy_offset : self._copy_offset + self._group_bytes]
            copy.copy_(self._packed[: self._group_bytes])

        return step

    def _capture(self, dtype: torch.dtype, masked: bool):
        """Capture the step for one action dtype, with or without an active mask; None steps without a graph."""
        actions = self._actions.setdefault(dtype, torch.zeros(self.num_envs, dtype=dtype, device=self.device))
        step = self._kernel(actions, masked)
        held = tuple(value.clone() for value in (*self._live.values(), self._sim, self._render))
        graph = None
        try:
            with torch.cuda.device(self.device):
                stream = torch.cuda.Stream()
                stream.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(stream):
                    step()
                torch.cuda.current_stream().wait_stream(stream)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    step()
        except Exception as error:  # noqa: BLE001 - step without the graph, and say so
            _warn_once("CUDA graph capture of the step", error)
            graph = None
        finally:
            for live, value in zip((*self._live.values(), self._sim, self._render), held, strict=True):
                live.copy_(value)
        self._graphs[(dtype, masked)] = (graph, step, actions)
        return self._graphs[(dtype, masked)]

    def step(self, actions, *, active=None, capture_state=True) -> TensorDict:
        self._replay(actions, active)
        views = self._unpack(self._packed.clone())
        outputs = _outputs_of(views, self._width)
        if capture_state:
            outputs["state"] = self._state(views, [self.num_envs])
        return build_tensordict(outputs, [self.num_envs], self.device)

    def step_result(self, actions, sequence: bool, state_indices, final_state: bool) -> TensorDict:
        """The result of ``Env.step`` for one action of every environment, from one copy of the packed buffer.

        It holds the same leaves as the step loop builds: the State, the outputs (with a time axis for a
        ``sequence``), ``valid``, and the ``states`` and ``state_valid`` of ``state_indices``.
        """
        key = (sequence, None if state_indices is None else len(state_indices), final_state)
        build = self._results.get(key)
        if build is None:
            build = self._results[key] = self._result_builder(*key)
        self._replay(actions, None)
        return build(self._packed.clone())

    def _result_builder(self, sequence: bool, states: int | None, final_state: bool):
        """``build(buffer)``: the step result whose leaves are views of a copy of the packed buffer."""
        regions = {name: (dtype, offset, width) for name, dtype, offset, width in self._splits}

        def group(key, region):
            dtype, offset, width = regions[region]
            return (self._paths[key], dtype, (width,), (1,), width, offset)

        width = self._width
        reward, flags, observation = regions["reward"][1], regions["flags"][1], regions["observation"][1]
        outputs = [
            (("reward",), torch.float64, (), (), 1, reward),
            (("terminated",), torch.bool, (), (), 2, flags),
            (("truncated",), torch.bool, (), (), 2, flags + 1),
            (("observation",), torch.float32, (width,), (1,), 2 * width, observation),
            (("info_state",), torch.float32, (width,), (1,), 2 * width, observation + width),
        ]
        ones = regions["ones"][1]
        return result_builder(
            self.num_envs,
            self.device,
            [group(key, key) for key in self._paths],
            outputs,
            [group(key, _STATES_COPY + key) for key in self._paths],
            (ones, ones + self.num_envs),
            sequence=sequence,
            with_states=states,
            final_state=final_state,
        )

    def _replay(self, actions, active) -> None:
        """Advance every environment (those of ``active``) by one replay of the captured step."""
        self._check_open()
        a = actions if torch.is_tensor(actions) else torch.as_tensor(np.asarray(actions))
        if tuple(a.shape) != (self.num_envs, 1):
            raise ValueError(f"{self.domain} action must have shape ({self.num_envs}, 1), got {tuple(a.shape)}")
        if self.domain in _DISCRETE:
            if a.is_floating_point() or a.is_complex():
                # Gymnasium's Discrete check rejects float actions; Acrobot indexes a Python list with them.
                message = f"{self.domain} actions must be integers, got dtype {a.dtype}"
                raise TypeError(message) if self.domain == "acrobot" else AssertionError(message)
            dtype = torch.int64
        else:
            dtype = a.dtype if a.is_floating_point() else torch.float64
        if active is not None:
            active = torch.as_tensor(active)
            if tuple(active.shape) != (self.num_envs,) or active.dtype != torch.bool:
                raise ValueError("active must be a bool vector with shape [num_envs]")
        graph, step, buffer = self._graphs.get((dtype, active is not None)) or self._capture(dtype, active is not None)
        buffer.copy_(a[:, 0])
        if active is not None:
            self._active.copy_(active)
        if graph is None:
            step()
        else:
            graph.replay()

    def close(self) -> None:
        if not self._closed:
            self._graphs.clear()
            self._closed = True
