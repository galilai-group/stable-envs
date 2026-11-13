"""Classic-control lanes held, stepped and reset on a CUDA device, with each step replayed as one CUDA graph.

The State and the step outputs live in four device blocks, one per dtype. A step copies the actions
into a fixed buffer and replays a graph. The graph runs one Triton kernel for the domain, which also
writes the environment's render rows, and packs the blocks into one byte buffer. One copy of that
buffer gives the State and outputs that the step returns, so they own their memory. The step reads
nothing back to the host. Without Triton, the graph replays the batched Torch kernel.

Resets draw from a counter-based generator on the device. The ``rng`` State fields hold each lane's
key (``seed + lane``) and its reset count instead of a PCG64 state, so seeded resets repeat, but they
do not equal the Gymnasium resets. The initial states have the Gymnasium distributions.
"""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np
import torch
from gymnasium.envs.classic_control import utils as gym_utils
from tensordict import TensorDict

from ...._core.config import log
from ...._core.tensordicts import build_tensordict
from ...._execution.protocol import SimulatorAdapter
from .batched import _DISCRETE, _DYNAMIC, _ELAPSED, _KERNELS, MAX_EPISODE_STEPS, _Torch
from .local import _PHYSICS_SPACES, CLASSIC_FIELDS, physics_state_key, validate_classic_state
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
_BLOCK_DTYPES = (torch.float64, torch.int64, torch.float32, torch.bool)
_OUTPUTS = ("reward", "observation", "info_state", "terminated", "truncated")
_MASK = 0xFFFF_FFFF
_TORCH = _Torch(torch.device("cpu"))


def _columns(domain: str):
    """Place each State field and step output in the float64, int64, float32 or bool block."""
    groups: tuple[list, ...] = ([], [], [], [])
    for key, (width, dtype) in CLASSIC_FIELDS[domain].items():
        groups[_BLOCK_DTYPES.index(dtype)].append((key, width))
    width = _OBSERVATION_WIDTH[domain]
    groups[0].append(("reward", 1))
    groups[2].extend((("observation", width), ("info_state", width)))
    groups[3].extend((("terminated", 1), ("truncated", 1)))
    columns, widths = {}, []
    for block, entries in enumerate(groups):
        start = 0
        for key, count in entries:
            columns[key] = (block, start, count)
            start += count
        widths.append(start)
    return columns, tuple(widths)


_COLUMNS = {domain: _columns(domain) for domain in CLASSIC_FIELDS}


def _offsets(widths, num_envs: int) -> tuple[list[int], int]:
    """Byte offset of each block in the packed buffer, and the buffer size; each block starts at a multiple of 8."""
    offsets, offset = [], 0
    for dtype, width in zip(_BLOCK_DTYPES, widths, strict=True):
        offsets.append(offset)
        offset += -(-num_envs * width * dtype.itemsize // 8) * 8
    return offsets, offset


def _blocks(buffer: torch.Tensor, widths, num_envs: int) -> tuple[torch.Tensor, ...]:
    """View a packed byte buffer as the four ``[num_envs, width]`` blocks."""
    offsets, _ = _offsets(widths, num_envs)
    return tuple(
        buffer[offset : offset + num_envs * width * dtype.itemsize].view(dtype).view(num_envs, width)
        for dtype, width, offset in zip(_BLOCK_DTYPES, widths, offsets, strict=True)
    )


def _column(domain, blocks, key) -> torch.Tensor:
    block, start, _ = _COLUMNS[domain][0][key]
    return blocks[block][:, start]


def _advance(domain, blocks, a, active, sim, render, sim_names, render_columns):
    """One step of every lane, written into ``blocks`` and the render rows; ``active=None`` steps all lanes."""
    columns = _COLUMNS[domain][0]
    old = {key: _column(domain, blocks, key) for key in _DYNAMIC[domain]}
    params = {name: _column(domain, blocks, physics_state_key(name)) for name in _PHYSICS_SPACES[domain]}
    if domain in _DISCRETE:
        a, dtype = a.long(), None
    else:
        a, dtype = _TORCH.floats(a)
    new, observation, reward, terminated = _KERNELS[domain](_TORCH, old, params, a, dtype)
    elapsed = _column(domain, blocks, _ELAPSED)
    if active is None:
        elapsed = elapsed + 1
        truncated = elapsed >= MAX_EPISODE_STEPS[domain]
    else:
        new = {key: torch.where(active, new[key], old[key]) for key in old}
        elapsed = elapsed + active
        truncated = active & (elapsed >= MAX_EPISODE_STEPS[domain])
        observation = [torch.where(active, column, 0.0) for column in observation]
        reward = torch.where(active, reward, 0.0)
        terminated = active & terminated
    observation = torch.stack(observation, 1).float()
    rows = torch.stack([new[name] for name in sim_names], 1)
    drawn = torch.stack([new[sim_names[index]] for index in render_columns], 1).float()
    for key, value in new.items():
        _column(domain, blocks, key).copy_(value)
    _column(domain, blocks, _ELAPSED).copy_(elapsed)
    _column(domain, blocks, "reward").copy_(reward)
    _column(domain, blocks, "terminated").copy_(terminated)
    _column(domain, blocks, "truncated").copy_(truncated)
    for key in ("observation", "info_state"):
        block, start, width = columns[key]
        blocks[block][:, start : start + width].copy_(observation)
    sim.copy_(rows)
    render.copy_(drawn)


_warned: set = set()


def _warn_once(what: str, error: BaseException) -> None:
    if what not in _warned:
        _warned.add(what)
        log("WARNING", f"Classic control on CUDA: {what} failed ({type(error).__name__}: {error}). Steps are slower.")


def _mix(x: torch.Tensor) -> torch.Tensor:
    """A 32-bit integer hash of values in [0, 2**32); every product stays below 2**63."""
    x = x ^ (x >> 16)
    x = (x * 0x21F0AAAD) & _MASK
    x = x ^ (x >> 15)
    x = (x * 0x735A2D97) & _MASK
    return x ^ (x >> 15)


def _uniforms(key: torch.Tensor, counter: torch.Tensor, count: int) -> list[torch.Tensor]:
    """``count`` float64 uniforms in [0, 1) with 53 random bits, for each lane's key and reset count."""
    base = _mix((key & _MASK) ^ _mix(((key >> 32) & _MASK) ^ 0x85EBCA6B))
    base = _mix(base ^ _mix((counter & _MASK) ^ 0xC2B2AE35))
    out = []
    for k in range(count):
        high = _mix(base ^ (((2 * k + 1) * 0x27D4EB2F) & _MASK))
        low = _mix(high ^ 0x165667B1)
        out.append(((high >> 5) * 67108864 + (low >> 6)).double() / 9007199254740992.0)
    return out


def _reset_bounds(domain: str, options: dict | None):
    """Gymnasium's ``reset`` bounds of each initial-state variable, with its ``options``."""
    if domain == "pendulum":
        x = gym_utils.verify_number_and_cast(options.get("x_init", np.pi) if options else np.pi)
        y = gym_utils.verify_number_and_cast(options.get("y_init", 1.0) if options else 1.0)
        return [(-x, x), (-y, y)]
    default = {"cartpole": (-0.05, 0.05), "acrobot": (-0.1, 0.1)}.get(domain, (-0.6, -0.4))
    low, high = gym_utils.maybe_parse_reset_bounds(options, *default)
    return [(float(low), float(high))] * (4 if domain in ("cartpole", "acrobot") else 1)


class DeviceClassicAdapter(SimulatorAdapter):
    """Batched classic-control lanes whose State, steps and resets stay on one CUDA device.

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
        self._columns, self._widths = _COLUMNS[domain]
        offsets, self._nbytes = _offsets(self._widths, self.num_envs)
        # Per block: its first element in the packed buffer viewed as the block dtype, its keys and their widths.
        self._splits = [
            (
                offsets[block] // dtype.itemsize,
                [key for key, place in self._columns.items() if place[0] == block],
                [place[2] for place in self._columns.values() if place[0] == block],
            )
            for block, dtype in enumerate(_BLOCK_DTYPES)
        ]
        self._live = tuple(
            torch.zeros(self.num_envs, width, dtype=dtype, device=self.device)
            for dtype, width in zip(_BLOCK_DTYPES, self._widths, strict=True)
        )
        self._packed = self._buffer()
        self._packed_blocks = _blocks(self._packed, self._widths, self.num_envs)
        for name, (_, _, default) in _PHYSICS_SPACES[domain].items():
            self._field(physics_state_key(name)).fill_(default)
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
        self._active = torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
        self._actions: dict = {}
        self._graphs: dict = {}
        self._closed = False

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("DeviceClassicAdapter is closed")

    def _buffer(self) -> torch.Tensor:
        return torch.empty(self._nbytes, dtype=torch.uint8, device=self.device)

    def _field(self, key, blocks=None) -> torch.Tensor:
        """The ``[num_envs, width]`` view of a State field or output."""
        block, start, width = self._columns[key]
        return (self._live if blocks is None else blocks)[block][:, start : start + width]

    def _pack(self, blocks) -> None:
        for packed, live in zip(blocks, self._live, strict=True):
            packed.copy_(live)

    def _unpack(self, buffer: torch.Tensor, steps: int | None = None) -> dict:
        """Views of every field and output in packed buffers: one buffer, or ``steps`` buffers stacked in time."""
        n = self.num_envs
        out: dict = {}
        for (offset, keys, widths), dtype, row in zip(self._splits, _BLOCK_DTYPES, self._widths, strict=True):
            flat = buffer.view(dtype)
            if steps is None:
                block = flat.as_strided((n, row), (row, 1), offset)
            else:
                block = flat.as_strided((n, steps, row), (row, self._nbytes // dtype.itemsize, 1), offset)
            for key, value in zip(keys, block.split(widths, -1), strict=True):
                if isinstance(key, str):
                    out[key] = value
                else:
                    out.setdefault(key[0], {})[key[1]] = value
        return out

    @staticmethod
    def _state(views: dict) -> dict:
        return {key: value for key, value in views.items() if key not in _OUTPUTS}

    def configure(self, values: Mapping[str, np.ndarray]) -> None:
        """Set each lane's physical constants; the live lanes keep their state."""
        self._check_open()
        for name, rows in values.items():
            rows = np.asarray(rows, np.float64)
            if rows.shape != (self.num_envs,):
                raise ValueError("Physical parameters must have one scalar per lane")
            self._field(physics_state_key(name))[:, 0].copy_(torch.from_numpy(rows))

    def reset(self, seed: int | None = None, options: dict | None = None) -> TensorDict:
        self._check_open()
        bounds = _reset_bounds(self.domain, options)
        rng = self._field(("rng", "state"))
        if seed is not None:
            seed = int(seed)
            if seed < 0:
                raise ValueError(f"Seed must be a non-negative integer, got {seed}")
            lanes = torch.arange(self.num_envs, device=self.device, dtype=torch.int64) + seed % (1 << 62)
            rng[:, 0], rng[:, 1], rng[:, 2] = lanes & _MASK, lanes >> 32, 0
            self._field(("rng", "increment")).fill_(0)[:, 0] = 1
        else:
            rng[:, 2] += 1
        u = _uniforms(rng[:, 0] | (rng[:, 1] << 32), rng[:, 2], len(bounds))
        draws = [low + (high - low) * value for (low, high), value in zip(bounds, u, strict=True)]
        if self.domain == "acrobot":
            # Acrobot stores its initial state as float32.
            draws = [value.float().double() for value in draws]
        elif self.domain != "cartpole":
            # Pendulum's torque and the cars' velocity start at 0.
            draws.append(torch.zeros_like(draws[0]))
        names = _DYNAMIC[self.domain]
        for name, value in zip(names, draws, strict=False):
            self._field(name)[:, 0] = value
        for key in names[len(draws) :]:
            self._field(key).zero_()
        self._field(_ELAPSED).zero_()
        return self.get_state()

    def get_state(self) -> TensorDict:
        """A State whose fields are views of one new copy of the live blocks."""
        self._check_open()
        buffer = self._buffer()
        self._pack(_blocks(buffer, self._widths, self.num_envs))
        return build_tensordict(self._state(self._unpack(buffer)), [self.num_envs], self.device)

    def stack_states(self, states) -> TensorDict | None:
        """Stack States of this adapter's steps along a new time axis with one copy; None for other States."""
        buffers = []
        for state in states:
            storage = state["lifecycle", "elapsed_steps"].untyped_storage()
            if storage.nbytes() != self._nbytes or storage.device != self.device:
                return None
            buffers.append(torch.empty(0, dtype=torch.uint8, device=self.device).set_(storage))
        views = self._unpack(torch.stack(buffers).view(-1), len(buffers))
        return build_tensordict(self._state(views), [self.num_envs, len(buffers)], self.device)

    def validate(self, state: TensorDict) -> None:
        self._check_open()
        validate_classic_state(self.domain, state, batch=self.num_envs)

    def set_state(self, state: TensorDict) -> None:
        self._check_open()
        validate_classic_state(self.domain, state, batch=self.num_envs)
        for key in self.fields:
            self._field(key).copy_(state[key])
        # Hold the values as Gymnasium holds them: an absent ``last_u`` or ``steps_beyond_terminated`` reads as 0.
        if self.domain == "cartpole":
            beyond = self._field(("task", "steps_beyond_terminated"))
            beyond.mul_(self._field(("task", "has_steps_beyond_terminated")))
        elif self.domain == "pendulum":
            self._field("torque").mul_(self._field(("task", "has_last_u")))
        elif self.domain == "mountain_car_continuous":
            narrow = self._field(("task", "state_is_float32"))
            for name in ("position", "velocity"):
                column = self._field(name)
                column.copy_(torch.where(narrow, column.float().double(), column))

    def _kernel(self, actions: torch.Tensor, masked: bool):
        """The step for one action buffer: the domain's Triton kernel, or the batched Torch kernel."""
        fused = self._fused and actions.dtype in (torch.int64, torch.float32, torch.float64)
        if fused and self._launch is None:
            self._launch = launcher(
                self.domain, self._columns, self._widths, self.num_envs, self.device, MAX_EPISODE_STEPS[self.domain]
            )
        launch = self._launch if fused else None
        live, active, sim, render = self._live, self._active, self._sim, self._render
        narrow = actions.dtype == torch.float32

        def step():
            if launch is not None:
                launch(live, actions, active, sim, render, masked, narrow)
            else:
                _advance(
                    self.domain, live, actions, active if masked else None, sim, render,
                    tuple(self._sim_names), tuple(self._render_columns),
                )  # fmt: skip
            self._pack(self._packed_blocks)

        return step

    def _capture(self, dtype: torch.dtype, masked: bool):
        """Capture the step for one action dtype, with or without an active mask; None steps without a graph."""
        actions = self._actions.setdefault(dtype, torch.zeros(self.num_envs, dtype=dtype, device=self.device))
        step = self._kernel(actions, masked)
        held = tuple(value.clone() for value in (*self._live, self._sim, self._render))
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
            for live, value in zip((*self._live, self._sim, self._render), held, strict=True):
                live.copy_(value)
        self._graphs[(dtype, masked)] = (graph, step, actions)
        return self._graphs[(dtype, masked)]

    def step(self, actions, *, active=None, capture_state=True) -> TensorDict:
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
        views = self._unpack(self._packed.clone())
        outputs = {key: views[key] for key in ("observation", "info_state")}
        for key in ("reward", "terminated", "truncated"):
            outputs[key] = views[key][:, 0]
        if capture_state:
            outputs["state"] = build_tensordict(self._state(views), [self.num_envs], self.device)
        return build_tensordict(outputs, [self.num_envs], self.device)

    def close(self) -> None:
        if not self._closed:
            self._graphs.clear()
            self._closed = True
