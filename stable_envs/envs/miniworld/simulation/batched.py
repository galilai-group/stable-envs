"""MovingBlockWorld environments stepped together: a Numba kernel on CPU, a Triton kernel on CUDA, or Torch."""

from __future__ import annotations

import math

import numpy as np
import torch
from tensordict import TensorDict

from ...._core.tensordicts import build_tensordict
from ...._execution.protocol import SimulatorAdapter
from .adapter import COLORS, TEXTURES, state_fields, validate_state
from .world import P_BALL_RADIUS, World

BACKENDS = ("numba", "triton", "torch")
#: CUDA batches below this size step on the Numba kernel under ``simulation_backend="auto"``. On an L40S the
#: Triton step is as fast as the Numba step from one environment.
TRITON_MIN_ENVS = 1
#: Environments from which the Numba step runs on several threads.
PARALLEL_MIN_ENVS = 64
_EXECUTION_MODES = {"numba": "NATIVE_NUMBA", "triton": "NATIVE_TRITON", "torch": "NATIVE_TORCH"}


def miniworld_backend(
    device: str | torch.device, num_envs: int, backend: str | None = None, triton_min_envs: int = TRITON_MIN_ENVS
) -> str:
    """The step backend of a batch.

    ``None`` takes Triton on CUDA and Numba on CPU; ``"auto"`` also takes Numba for CUDA batches below
    ``triton_min_envs``. ``"torch"`` runs on any device; ``"triton"`` needs CUDA.
    """
    if backend not in (None, "auto", *BACKENDS):
        raise ValueError(f"simulation_backend must be None, 'auto', 'numba', 'triton' or 'torch', got {backend!r}")
    if isinstance(triton_min_envs, bool) or not isinstance(triton_min_envs, int) or triton_min_envs < 1:
        raise ValueError("triton_min_envs must be an int >= 1")
    cuda = torch.device(device).type == "cuda"
    if backend == "triton" and not cuda:
        raise ValueError("simulation_backend='triton' requires device='cuda'")
    if backend is None:
        return "triton" if cuda else "numba"
    if backend == "auto":
        return "triton" if cuda and num_envs >= triton_min_envs else "numba"
    return backend


def execution_mode(backend: str) -> str:
    return _EXECUTION_MODES[backend]


#: Arrays of the State dict by dtype, with their shapes after the environment axis.
_F64 = (("apos", (3,)), ("adir", ()), ("size", (-1, 3)), ("bpos", (-1, 3)), ("bdir", (-1,)), ("vel", (-1, 3)),
        ("rad", (-1,)), ("height", (-1,)))  # fmt: skip
_I64 = (("room", (3,)), ("kind", (-1,)), ("color", (-1,)), ("texture", (-1,)), ("count", ()), ("carry", ()),
        ("steps", ()), ("seed", ()), ("episode", ()))  # fmt: skip


def _layout(B: int, M: int) -> tuple:
    """Per dtype buffer: the arrays' names, shapes and sizes."""
    out = []
    for spec in (_F64, _I64):
        shapes = [(name, (B, *(M if n == -1 else n for n in tail))) for name, tail in spec]
        out.append((shapes, [math.prod(shape) for _, shape in shapes]))
    return tuple(out)


def _views(f64: torch.Tensor, i64: torch.Tensor, layout: tuple) -> dict:
    """The named arrays as contiguous views of the two flat buffers."""
    out = {}
    for flat, (shapes, sizes) in zip((f64, i64), layout, strict=True):
        for (name, shape), part in zip(shapes, flat.split(sizes), strict=True):
            out[name] = part.view(shape)
    return out


#: State dict entries the kernels change in a step.
_DYNAMIC = ("apos", "adir", "carry", "bpos", "bdir", "vel", "steps")


class BatchedMovingBlockWorldAdapter(SimulatorAdapter):
    """``num_envs`` environments in State-shaped arrays; environment ``i`` of ``reset(seed)`` uses seed ``seed + i``.

    ``backend="numba"`` keeps the arrays on the host and returns tensors on ``device``; ``"triton"`` and
    ``"torch"`` keep them on ``device``, and a CUDA step of every environment replays a captured graph.
    """

    action_dim = 1

    def __init__(self, config: dict, num_envs: int, size: int, device="cpu", backend: str = "numba"):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("num_envs must be a positive integer")
        if backend not in BACKENDS:
            raise ValueError(f"backend must be one of {BACKENDS}, got {backend!r}")
        self.config, self.num_envs, self.size, self.backend = config, num_envs, int(size), backend
        self.world = World(config, TEXTURES, COLORS)
        self.count_range = self.world.count_range
        self.num_blocks = M = int(self.count_range[1])
        self.frame_shape = (self.size, self.size, 3)
        self.device = torch.device(device)
        self.fields = state_fields(M)
        self.sim_device = torch.device("cpu") if backend == "numba" else self.device
        B, f64, i64, dev = num_envs, torch.float64, torch.int64, self.sim_device
        z = lambda *shape, dtype=f64: torch.zeros(shape, dtype=dtype, device=dev)  # noqa: E731
        # One float64 and one int64 buffer hold every array, so a State snapshot is two copies.
        self._layout = _layout(B, M)
        self._flat = tuple(z(sum(sizes), dtype=dtype) for (_, sizes), dtype in zip(self._layout, (f64, i64)))
        self.s = _views(*self._flat, self._layout)
        self._truncated = z(B, dtype=torch.bool)
        if backend == "triton":
            from . import kernel_triton
            from .world import P_AGENT_OVERLAP, P_BOX_OVERLAP, P_GRID, P_STATIC, P_TORUS, P_WRAP_HEADING

            P = self.world.P
            self.kernel = kernel_triton
            self._constants = kernel_triton.constants(P, dev)
            self._flags = {
                name: bool(P[index] > 0)
                for name, index in (
                    ("GRID", P_GRID),
                    ("TORUS", P_TORUS),
                    ("STATIC", P_STATIC),
                    ("AGENT_OV", P_AGENT_OVERLAP),
                    ("BOX_OV", P_BOX_OVERLAP),
                    ("WRAP", P_WRAP_HEADING),
                )
            }
            self._truncated8 = z(B, dtype=torch.int8)
            from .kernel_torch import TorchKernel

            self._resetter = TorchKernel(self.world, dev)
        elif backend == "torch":
            from .kernel_torch import TorchKernel

            self.kernel = TorchKernel(self.world, dev)
        else:
            from . import kernel_cpu

            self.kernel = kernel_cpu
            self._np = {k: v.numpy() for k, v in self.s.items()}
            self._numba_step = kernel_cpu.step_parallel if B >= PARALLEL_MIN_ENVS else kernel_cpu.step_serial
            self._all = np.ones(B, bool)
        self._graph = self._reset_graph = None
        self._closed = False

    # ------------------------------------------------------------------------------------------ lifecycle
    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError("MovingBlockWorld simulator is closed")

    def close(self) -> None:
        self._graph = self._reset_graph = None
        self._closed = True

    def reset(self, seed: int | None = None) -> TensorDict:
        self._check_open()
        s = self.s
        if seed is None:
            s["episode"] += 1
        else:
            s["seed"].copy_(int(seed) + torch.arange(self.num_envs, device=self.sim_device))
            s["episode"].zero_()
        if self.backend == "numba":
            n, w = self._np, self.world
            lo, hi = self.count_range
            self.kernel.reset_all(
                n["seed"], n["episode"], n["room"], n["kind"], n["color"], n["texture"], n["size"], n["apos"],
                n["adir"], n["carry"], n["count"], n["rad"], n["height"], n["bpos"], n["bdir"], n["vel"], n["steps"],
                w.P, lo, hi, w.pool, w.base_palette, w.floor_pool, w.wall_pool, w.box_pool,
            )  # fmt: skip
        elif self.sim_device.type == "cuda":
            if self._reset_graph is None:
                side = torch.cuda.Stream(self.sim_device)
                side.wait_stream(torch.cuda.current_stream(self.sim_device))
                with torch.cuda.stream(side):
                    self._reset_body()  # warm-up; the replay below draws the same episodes
                torch.cuda.current_stream(self.sim_device).wait_stream(side)
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph):
                    self._reset_body()
                self._reset_graph = graph
            self._reset_graph.replay()
        else:
            self._reset_body()
        return self.get_state()

    def _reset_body(self) -> None:
        """Each environment's next episode from its seed and episode number, in the State buffers."""
        resetter = self._resetter if self.backend == "triton" else self.kernel
        for key, value in resetter.reset(self.s).items():
            self.s[key].copy_(value)

    # ------------------------------------------------------------------------------------------ State
    def get_state(self) -> TensorDict:
        self._check_open()
        dev = self.device
        self._last = tuple(flat.to(dev, copy=True) for flat in self._flat)
        return self._snapshot(*self._last)

    def _snapshot(self, f64: torch.Tensor, i64: torch.Tensor, present: torch.Tensor | None = None, time=False):
        """A State whose leaves are views of the owned flat buffers ``f64`` and ``i64``.

        ``time`` adds a time axis of one.
        """
        B, M = self.num_envs, self.num_blocks
        s = _views(f64, i64, self._layout)
        if present is None:
            present = torch.arange(M, device=f64.device) < s["count"][:, None]
        tree = {
            "agent": {"pos": s["apos"], "dir": s["adir"][:, None], "carrying": s["carry"][:, None]},
            "room": {"textures": s["room"]},
            "blocks": {
                "present": present,
                "kind": s["kind"],
                "color": s["color"],
                "texture": s["texture"],
                "size": s["size"].view(B, 3 * M),
                "pos": s["bpos"].view(B, 3 * M),
                "dir": s["bdir"],
                "velocity": s["vel"].view(B, 3 * M),
            },
            "rng": {"seed": s["seed"][:, None], "episode": s["episode"][:, None]},
            "lifecycle": {"elapsed_steps": s["steps"][:, None]},
        }
        if time:
            tree = {group: {k: v[:, None] for k, v in leaves.items()} for group, leaves in tree.items()}
        return build_tensordict(tree, [B, 1] if time else [B], f64.device)

    def stack_one(self, state: TensorDict) -> TensorDict | None:
        """An owned copy of the last ``get_state`` snapshot ``state`` with a time axis of one, else None."""
        last = getattr(self, "_last", None)
        if last is None or state["agent", "pos"]._base is not last[0] or state["rng", "seed"]._base is not last[1]:
            return None
        f64, i64 = last
        return self._snapshot(f64.clone(), i64.clone(), state["blocks", "present"].clone(), time=True)

    def validate(self, state, *, batch: int | None = None) -> None:
        validate_state(state, self.fields, self.count_range, batch=batch)

    def set_state(self, state: TensorDict) -> None:
        self._check_open()
        self.validate(state, batch=self.num_envs)
        s, B, M, dev = self.s, self.num_envs, self.num_blocks, self.sim_device
        get = lambda key: state[key].to(dev)  # noqa: E731
        kind, size = get(("blocks", "kind")), get(("blocks", "size")).reshape(B, M, 3)
        rad, height = block_radius(kind, size, self.world.P[P_BALL_RADIUS])
        values = {
            "apos": get(("agent", "pos")),
            "adir": get(("agent", "dir"))[:, 0],
            "carry": get(("agent", "carrying"))[:, 0],
            "room": get(("room", "textures")),
            "count": get(("blocks", "present")).sum(-1),
            "kind": kind,
            "color": get(("blocks", "color")),
            "texture": get(("blocks", "texture")),
            "size": size,
            "rad": rad,
            "height": height,
            "bpos": get(("blocks", "pos")).reshape(B, M, 3),
            "bdir": get(("blocks", "dir")),
            "vel": get(("blocks", "velocity")).reshape(B, M, 3),
            "seed": get(("rng", "seed"))[:, 0],
            "episode": get(("rng", "episode"))[:, 0],
            "steps": get(("lifecycle", "elapsed_steps"))[:, 0],
        }
        for key, value in values.items():
            s[key].copy_(value)

    # ------------------------------------------------------------------------------------------ step
    def check_step(self, actions, active):
        """Actions as ``(num_envs,)`` int64 on the simulation device.

        Ids outside [0, 8] are refused when the actions are on the host. CUDA actions are not read back; there,
        such ids do nothing, as in the source env.
        """
        n = self.num_envs
        if torch.is_tensor(actions):
            a = actions.detach()
            if a.dtype.is_floating_point or a.dtype == torch.bool or a.numel() != n:
                raise ValueError(
                    f"MovingBlockWorld actions must be {n} integer action ids, got {a.dtype} {tuple(a.shape)}"
                )
            a = a.reshape(n).to(self.sim_device, torch.int64)
            host = actions.device.type == "cpu"
        else:
            arr = np.asarray(actions)
            if arr.size != n or arr.dtype.kind not in "iu":
                raise ValueError(
                    f"MovingBlockWorld actions must be {n} integer action ids, got {arr.dtype} {arr.shape}"
                )
            a = torch.from_numpy(arr.reshape(n).astype(np.int64)).to(self.sim_device)
            host = True
        if host and bool(((a < 0) | (a > 8)).any()):
            raise ValueError("MovingBlockWorld action ids lie in [0, 8]")
        if active is not None:
            active = torch.as_tensor(active).to(self.sim_device, torch.bool).reshape(-1)
            if active.shape != (n,):
                raise ValueError(f"active must have one entry per environment ({n}), got {tuple(active.shape)}")
        return a, active

    def probe_actions(self) -> np.ndarray:
        return np.full((1, 1), 4, np.int64)

    def step(self, actions, *, active=None, capture_state=True) -> TensorDict:
        """Advance one action per environment; an inactive environment holds its state and reports zero outputs."""
        self._check_open()
        a, active = self.check_step(actions, active)
        if self.backend == "numba":
            n = self._np
            live = self._all if active is None else active.numpy()
            self._numba_step(
                a.numpy(), live, n["apos"], n["adir"], n["carry"], n["count"], n["rad"], n["height"], n["bpos"],
                n["bdir"], n["vel"], n["steps"], self._truncated.numpy(), self.world.P,
            )  # fmt: skip
            truncated = self._truncated.clone()
        elif active is None and self.sim_device.type == "cuda":
            truncated = self._graph_step(a)
        elif self.backend == "triton":
            self.kernel.step(self.s, a, active.to(torch.int8), self._constants, self._flags, self._truncated8)
            truncated = self._truncated8.bool()
        else:
            out = self.kernel.step(self.s, a, active)
            for key in _DYNAMIC:
                self.s[key].copy_(out[key])
            truncated = out["truncated"]
        dev, B = self.device, self.num_envs
        out = {
            "reward": torch.zeros(B, dtype=torch.float64, device=dev),
            "terminated": torch.zeros(B, dtype=torch.bool, device=dev),
            "truncated": truncated.to(dev),
        }
        if capture_state:
            out["state"] = self.get_state()
        return build_tensordict(out, [B], dev)

    def _graph_step(self, a: torch.Tensor) -> torch.Tensor:
        """Step every environment through a CUDA graph captured on first use; the State buffers stay in place."""
        if self._graph is None:
            self._g_actions = torch.full_like(a, 4)
            saved = {key: self.s[key].clone() for key in _DYNAMIC}
            side = torch.cuda.Stream(self.sim_device)
            side.wait_stream(torch.cuda.current_stream(self.sim_device))
            with torch.cuda.stream(side):
                self._graph_body()  # warm-up, undone below
            torch.cuda.current_stream(self.sim_device).wait_stream(side)
            graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(graph):
                self._graph_body()
            for key in _DYNAMIC:
                self.s[key].copy_(saved[key])
            self._graph = graph
        self._g_actions.copy_(a)
        self._graph.replay()
        return self._truncated.clone()

    def _graph_body(self) -> None:
        if self.backend == "triton":
            self.kernel.step(self.s, self._g_actions, None, self._constants, self._flags, self._truncated8)
            self._truncated.copy_(self._truncated8)
            return
        out = self.kernel.step(self.s, self._g_actions, None)
        for key in _DYNAMIC:
            self.s[key].copy_(out[key])
        self._truncated.copy_(out["truncated"])


def block_radius(kind: torch.Tensor, size: torch.Tensor, ball_radius_per_height: float):
    """Collision radius and height of each block from its kind and ``(sx, sy, sz)`` size."""
    sx, sy, sz = size.unbind(-1)
    box = torch.sqrt(sx * sx + sz * sz) / 2
    ball = kind == 1
    return torch.where(ball, ball_radius_per_height * sy, box), sy.clone()
