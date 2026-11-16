"""MovingBlockWorld environments stepped together: a Numba kernel on CPU, a Triton kernel on CUDA, or Torch."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ...._core.tensordicts import build_tensordict, to_device
from ...._execution.protocol import SimulatorAdapter
from .adapter import COLORS, TEXTURES, state_schema, validate_state
from .world import P_BALL_RADIUS, World

BACKENDS = ("numba", "triton", "torch")
#: Environments from which the Numba step runs on several threads.
PARALLEL_MIN_ENVS = 64
_EXECUTION_MODES = {"numba": "NATIVE_NUMBA", "triton": "NATIVE_TRITON", "torch": "NATIVE_TORCH"}
#: Shipped CUDA switch point of ``"auto"``: the smallest batch that steps on Triton; smaller batches step with Numba.
#: None uses Triton at every batch size. The setting and the tuning cache take precedence.
#: Measured on RTX 3090 and RTX A5000 hosts with ``python -m benchmarks.backend_switch`` (docs/performance-tuning.md).
CUDA_MIN_ENVS: int | None = None


def _measure(config: dict, size: int, device):
    """Build and step MovingBlockWorld environments on one backend, for ``stable-envs tune``."""
    from ...._execution.backends import measure_adapter

    return measure_adapter(
        lambda backend, n: BatchedMovingBlockWorldAdapter(config, n, size, device, backend),
        lambda n: torch.full((n,), 2, dtype=torch.int64, device=device),
    )


def miniworld_backend(
    device: str | torch.device,
    num_envs: int,
    backend: str | None = "auto",
    config: dict | None = None,
    size: int | None = None,
) -> str:
    """The step backend of a batch.

    ``"auto"`` (or None) takes Numba on CPU. On CUDA it takes Numba below ``CUDA_MIN_ENVS`` and Triton from it.
    ``stable-envs tune`` measures with environments of ``config`` and frame ``size``. ``"torch"``
    runs on any device; ``"triton"`` needs CUDA.
    """
    from ...._execution.backends import BackendPair, select_backend

    if backend == "triton" and torch.device(device).type != "cuda":
        raise ValueError("simulation_backend='triton' requires device='cuda'")
    valid = type(num_envs) is int and num_envs > 0 and config is not None and size is not None
    measure = _measure(config, size, device) if valid else None
    return select_backend(
        "miniworld-sim/moving_blockworld",
        num_envs,
        device,
        backend,
        {"cpu": BackendPair("numba", "numba"), "cuda": BackendPair("numba", "triton", CUDA_MIN_ENVS)},
        choices=BACKENDS,
        measure=measure,
    )


def execution_mode(backend: str) -> str:
    return _EXECUTION_MODES[backend]


#: The State fields that the kernels read and write, by the name the kernels give them, and the shape of each
#: after the environment axis (-1 stands for the block count). Each is a view of a column range of its group.
_LIVE = {
    "apos": ("agent/pos", (3,)),
    "adir": ("agent/dir", ()),
    "size": ("blocks/size", (-1, 3)),
    "bpos": ("blocks/pos", (-1, 3)),
    "bdir": ("blocks/dir", (-1,)),
    "vel": ("blocks/velocity", (-1, 3)),
    "carry": ("agent/carrying", ()),
    "tex": ("room/textures", (3,)),
    "kind": ("blocks/kind", (-1,)),
    "color": ("blocks/color", (-1,)),
    "texture": ("blocks/texture", (-1,)),
    "seed": ("rng/seed", ()),
    "episode": ("rng/episode", ()),
    "steps": ("lifecycle/elapsed_steps", ()),
}


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
        self.schema = schema = state_schema(M)
        self.sim_device = torch.device("cpu") if backend == "numba" else self.device
        B, f64, i64, dev = num_envs, torch.float64, torch.int64, self.sim_device
        z = lambda *shape, dtype=f64: torch.zeros(shape, dtype=dtype, device=dev)  # noqa: E731
        # The kernels write the float64 and int64 groups of the State. A State snapshot copies the two groups and
        # derives the bool group from the block counts, which the collision tests also read with the radii and
        # heights.
        self._flat = tuple(
            z(B, group.size, dtype=group.dtype) for group in schema.groups.values() if group.dtype != torch.bool
        )
        self._groups = dict(zip((g.key for g in schema.groups.values() if g.dtype != torch.bool), self._flat))
        self._present = next(g.path for g in schema.groups.values() if g.dtype == torch.bool)
        views = schema.views(self._groups, [name for name, _ in _LIVE.values()])
        self.s = {key: self._shaped(views[name], tail) for key, (name, tail) in _LIVE.items()}
        self.s.update(count=z(B, dtype=i64), rad=z(B, M), height=z(B, M))
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

    def _shaped(self, view: torch.Tensor, tail: tuple[int, ...]) -> torch.Tensor:
        """A column range of a group as the array the kernels take: ``(B, *tail)`` with the block count filled in."""
        shape = tuple(self.num_blocks if n == -1 else n for n in tail)
        return view[:, 0] if not shape else view.view(self.num_envs, *shape)

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
                n["seed"], n["episode"], n["tex"], n["kind"], n["color"], n["texture"], n["size"], n["apos"],
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

    def get_state(self) -> TensorDict:
        self._check_open()
        dev = self.device
        self._last = tuple(flat.to(dev, copy=True) for flat in self._flat)
        return self._snapshot(*self._last, count=self.s["count"].to(dev))

    def _snapshot(
        self, f64: torch.Tensor, i64: torch.Tensor, present: torch.Tensor | None = None, *, count=None, time=False
    ):
        """A State whose leaves are the owned group buffers ``f64`` and ``i64`` and the block flags.

        ``present`` or ``count`` gives the flags. ``time`` adds a time axis of one.
        """
        M = self.num_blocks
        if present is None:
            present = torch.arange(M, device=f64.device) < count[:, None]
        leaves = {group.path: value for group, value in zip(self.schema.groups.values(), (f64, i64, present))}
        if time:
            leaves = {path: value[:, None] for path, value in leaves.items()}
        tree: dict = {}
        for path, value in leaves.items():
            node = tree
            for part in path[:-1]:
                node = node.setdefault(part, {})
            node[path[-1]] = value
        return build_tensordict(tree, [self.num_envs, 1] if time else [self.num_envs], f64.device)

    def stack_one(self, state: TensorDict) -> TensorDict | None:
        """An owned copy of the last ``get_state`` snapshot ``state`` with a time axis of one, else None."""
        last = getattr(self, "_last", None)
        first = state.get(next(iter(self.schema.groups.values())).path, None)
        if last is None or not torch.is_tensor(first) or first.data_ptr() != last[0].data_ptr():
            return None
        f64, i64 = last
        return self._snapshot(f64.clone(), i64.clone(), state.get(self._present).clone(), time=True)

    def validate(self, state, *, batch: int | None = None) -> None:
        validate_state(state, self.schema, self.count_range, batch=batch)

    def set_state(self, state: TensorDict) -> None:
        self._check_open()
        self.validate(state, batch=self.num_envs)
        s, dev = self.s, self.sim_device
        for key, flat in self._groups.items():
            flat.copy_(state.get(tuple(key.split("/"))))
        s["count"].copy_(state.get(self._present).to(dev).sum(-1))
        rad, height = block_radius(s["kind"], s["size"], self.world.P[P_BALL_RADIUS])
        s["rad"].copy_(rad)
        s["height"].copy_(height)

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
        if self.sim_device != dev:
            # The outputs and the State move from the host in one copy.
            out = {
                "reward": torch.zeros(B, dtype=torch.float64),
                "terminated": torch.zeros(B, dtype=torch.bool),
                "truncated": truncated,
            }
            if capture_state:
                out["state"] = self._snapshot(*(flat.clone() for flat in self._flat), count=self.s["count"].clone())
            return to_device(build_tensordict(out, [B], self.sim_device), dev)
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
