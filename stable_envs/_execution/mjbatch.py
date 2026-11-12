"""Batched MuJoCo physics for B environments: native ``mujoco.batch`` on CPU, MuJoCo Warp on CUDA.

A family builds one backend with :func:`make_backend` (or :class:`NativeBatch` / :class:`WarpBatch`) for one
compiled ``mujoco.MjModel``, or for one model per environment that differ only in parameters (sizes, masses,
frictions), and keeps its task logic (observations, reward, resets) in a thin adapter.

Interface, shared by both backends:

- ``step(ctrl=None, *, mocap_pos=None, mocap_quat=None, active=None)`` writes the inputs that are given and
  advances every environment, or the ``active`` ones (ids or a ``[B]`` bool mask), by ``frame_skip`` calls of
  ``mj_step``, then runs ``mj_rnePostConstraint`` when the backend was built with ``rne_postconstraint=True``.
  On MuJoCo Warp, inactive environments keep their :data:`PHYSICS_FIELDS` and ``outputs``; their other
  derived fields are not meaningful after the step.
- ``get_physics(env_ids=None)`` returns a snapshot ``{name: tensor}`` of :data:`PHYSICS_FIELDS` on the env
  device; ``set_physics(values, env_ids=None, forward=True)`` writes any of them for the given environments
  only, then forwards them. ``reset(env_ids=None)`` restores the initial state (model defaults or the
  ``keyframe`` given at construction) and forwards.
- ``data(name)`` is a live, zero-copy ``[B, ...]`` tensor over an ``MjData`` field (``qpos``, ``xpos``,
  ``site_xpos``, ``sensordata``, ``cfrc_ext`` ...) on the backend device: CPU for the native backend, CUDA
  for MuJoCo Warp. ``read(names, env_ids=None)`` returns copies on the env device. Derived fields must be
  named in ``outputs=`` at construction: the native batch copies out only the fields bound before a call.
- ``unstable_envs()`` returns the ids of environments whose physics diverged since their last reset or
  ``set_physics``. ``check_buffers()`` raises :class:`BufferOverflowError` if a MuJoCo Warp contact or
  constraint buffer overflowed; the step also raises it, one or two steps late, without a host sync.
- ``set_models(models)`` changes the per-environment model parameters; ``model_field(name)`` is a live
  ``[B, ...]`` tensor of per-environment values of one model field (on MuJoCo Warp, a field or ``"opt.<name>"``
  option named in ``batched_fields``) that a family writes between steps. ``close()`` releases the backend.

Native backend guarantee: the CPU backend steps float64 ``MjData`` exactly as ``mj_step(m, d, nstep)``
followed by ``mj_rnePostConstraint(m, d)`` (when requested) on one ``MjData`` per environment, bit for bit,
at any thread count. ``set_physics`` writes float64 values without rounding, and ``data(...)`` are the
float64 ``MjData`` values of each environment after the same calls as the source. Derived fields describe
the state before the last substep, as after ``mj_step``. Families that compute observations from these in
float64 NumPy in the source's operation order stay bit-exact with the source. A CPU backend on a CUDA env
device returns ``get_physics`` and ``read`` copies on CUDA.

MuJoCo Warp backend: float32, approximately equal to the source. ``put_model`` and ``put_data`` run once, from
the template ``MjData`` broadcast to B worlds; model fields that differ between environments are batched per
world. The ``frame_skip`` substeps run as one captured CUDA graph, launched on the current Torch stream, with
no host synchronization. ``naconmax`` (contacts of all worlds) and ``njmax`` (constraints per world) default
to sizes for the batch; an overflow raises :class:`BufferOverflowError` instead of dropping contacts silently.
``set_physics(..., forward=True)`` forwards every world, so derived fields of the other worlds then describe
their current state. Sensors are computed on the last substep only, as they do not act on the dynamics.
``exclude_pairs`` removes geom pairs from MuJoCo Warp's collision candidates, as an ``<exclude>`` would: a
family uses it for pairs whose bounding volumes overlap but which make no contact. ``batched_fields`` names model
fields (and options, as ``"opt.<name>"``) that hold one value per environment, for ``set_models`` and
``model_field``. ``step_fn(m, d)``, when given, runs in place of the ``frame_skip`` calls of ``mjw.step`` in the
step graph, for a family whose source steps in another call order.
Up to ``fused_tree_max_envs`` worlds, the graphs compute subtree centers of mass, composite inertias and RNE
internal forces with one launch per pass instead of one per body tree level (:mod:`.mjwarp_fused`), on the
MuJoCo Warp versions it supports.

Routing: :func:`resolve_backend`. Per-environment source resets: :class:`ResetPool`.

The native batch's threads do not survive ``fork``: build backends after forking, or in spawned processes.
"""

from __future__ import annotations

import gc
import os
import threading
import warnings
from collections.abc import Callable, Iterable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING

import numpy as np
import torch

from .policy import cpu_affinity

if TYPE_CHECKING:
    from .mjwarp_fused import TreeWalks

#: Integration state that ``get_physics`` returns and ``set_physics`` writes; mocap fields only for models with
#: mocap bodies.
PHYSICS_FIELDS = ("qpos", "qvel", "act", "time", "qacc_warmstart", "mocap_pos", "mocap_quat")
#: Most native batch threads: one per CPU in the affinity and per environment, up to this.
MAX_NATIVE_THREADS = 8
#: Fields that ``mj_rnePostConstraint`` computes.
RNE_POST_FIELDS = ("cacc", "cfrc_int", "cfrc_ext")
BACKENDS = ("default", "auto", "native", "mjwarp")
#: Largest MuJoCo Warp batch whose graphs use :mod:`.mjwarp_fused` tree passes.
FUSED_TREE_MAX_ENVS = 4096


class BufferOverflowError(RuntimeError):
    """A MuJoCo Warp contact or constraint buffer is too small for the batch."""


def resolve_backend(
    num_envs: int,
    device: str | torch.device,
    simulation_backend: str | None = "default",
    min_envs: int = 256,
    *,
    min_envs_name: str = "mjwarp_min_envs",
) -> str:
    """Return ``"native"`` (``mujoco.batch``) or ``"mjwarp"`` (MuJoCo Warp) for a batch.

    ``"default"`` (or None) selects MuJoCo Warp on CUDA and native on CPU. ``"auto"`` selects MuJoCo Warp for
    CUDA batches of ``min_envs`` or more and native for every other batch. ``"native"`` and ``"mjwarp"`` select
    explicitly; ``"mjwarp"`` requires a CUDA device.
    """
    if simulation_backend is None:
        simulation_backend = "default"
    if simulation_backend not in BACKENDS:
        raise ValueError("simulation_backend must be 'default', 'auto', 'native' or 'mjwarp'")
    if isinstance(min_envs, bool) or not isinstance(min_envs, int) or min_envs < 1:
        raise ValueError(f"{min_envs_name} must be an int >= 1")
    cuda = torch.device(device).type == "cuda"
    if simulation_backend == "mjwarp" and not cuda:
        raise ValueError("simulation_backend='mjwarp' requires device='cuda'")
    if simulation_backend == "default":
        return "mjwarp" if cuda else "native"
    if simulation_backend == "auto":
        return "mjwarp" if cuda and num_envs >= min_envs else "native"
    return simulation_backend


def native_threads(num_envs: int) -> int:
    """Native batch threads: one per CPU in the affinity and per environment, at most :data:`MAX_NATIVE_THREADS`."""
    cpus = len(cpu_affinity() or ()) or os.cpu_count() or 1
    return max(1, min(MAX_NATIVE_THREADS, cpus, int(num_envs)))


def make_backend(backend: str, models, num_envs: int, device, **kwargs) -> MjBatch:
    """Build the ``backend`` (``"native"`` or ``"mjwarp"``) that :func:`resolve_backend` returned."""
    if backend == "native":
        return NativeBatch(models, num_envs, device, **kwargs)
    if backend == "mjwarp":
        return WarpBatch(models, num_envs, device, **kwargs)
    raise ValueError(f"Unknown MuJoCo backend {backend!r}")


def _model_list(models, num_envs: int) -> list:
    import mujoco

    if isinstance(models, mujoco.MjModel):
        return [models] * num_envs
    models = list(models)
    if len(models) != num_envs:
        raise ValueError(f"Expected one model or {num_envs} models, got {len(models)}")
    return models


def _env_ids(env_ids, num_envs: int) -> np.ndarray | None:
    if env_ids is None:
        return None
    if torch.is_tensor(env_ids):
        env_ids = env_ids.detach().cpu().numpy()
    ids = np.asarray(env_ids)
    if ids.dtype == bool:
        if ids.shape != (num_envs,):
            raise ValueError(f"A boolean env_ids mask must have shape ({num_envs},)")
        return np.flatnonzero(ids)
    ids = ids.astype(np.int64).reshape(-1)
    if ids.size and (ids.min() < 0 or ids.max() >= num_envs):
        raise ValueError(f"env_ids must be in [0, {num_envs})")
    return ids


class MjBatch:
    """Common interface of the batched MuJoCo backends; see the module docstring."""

    backend: str
    num_envs: int
    device: torch.device
    frame_skip: int
    model: object
    fields: tuple[str, ...]

    def _check_open(self) -> None:
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    def get_physics(self, env_ids=None) -> dict[str, torch.Tensor]:
        """Return a snapshot of :data:`PHYSICS_FIELDS` on the env device, rows ``env_ids`` or all."""
        return self.read(self.fields, env_ids)

    def read(self, names: Iterable[str], env_ids=None) -> dict[str, torch.Tensor]:
        """Return copies of ``data(name)`` for ``names`` on the env device, rows ``env_ids`` or all."""
        self._check_open()
        ids = _env_ids(env_ids, self.num_envs)
        out = {}
        for name in names:
            value = self.data(name)
            value = value.clone() if ids is None else value[torch.as_tensor(ids, device=value.device)]
            out[name] = value.to(self.device)
        return out

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


class NativeBatch(MjBatch):
    """B float64 ``MjData`` stepped together in one ``mujoco.batch.Batch`` on a native thread pool."""

    backend = "native"

    def __init__(
        self,
        models,
        num_envs: int,
        device="cpu",
        *,
        frame_skip: int = 1,
        keyframe: int | None = None,
        rne_postconstraint: bool = False,
        outputs: Sequence[str] = (),
        nthread: int | None = None,
        **_warp_options,
    ):
        import mujoco
        from mujoco.batch import Batch

        self._closed = True
        self.num_envs, self.device, self.frame_skip = int(num_envs), torch.device(device), int(frame_skip)
        models = _model_list(models, self.num_envs)
        self.model = models[0]
        self._keyframe = -1 if keyframe is None else int(keyframe)
        self._rne_post = bool(rne_postconstraint)
        # Without a handler MuJoCo prints warnings and appends them to MUJOCO_LOG.TXT in the working directory.
        if mujoco.get_mju_user_warning() is None:
            mujoco.set_mju_user_warning(_log_warning)
        nthread = native_threads(self.num_envs) if nthread is None else int(nthread)
        # mj_rnePostConstraint reads every derived field of an environment's last step, which only a persistent
        # batch keeps.
        self.b = Batch(self.model, self.num_envs, nthread=nthread, persistent=self._rne_post)
        self._models = None
        if any(model is not self.model for model in models):
            self._set_models(models)
        self.fields = tuple(f for f in PHYSICS_FIELDS if not f.startswith("mocap") or self.model.nmocap)
        self._views: dict[str, torch.Tensor] = {}
        self._model_views: dict[str, torch.Tensor] = {}
        # The batch copies out only the derived fields bound before a call.
        self.outputs = frozenset(outputs) | set(self.fields) | {"ctrl"}
        for name in self.outputs - set(RNE_POST_FIELDS if self._rne_post else ()):
            getattr(self.b, name)
        if self._rne_post:
            self._scratch = mujoco.MjData(self.model)
            self._rne = {
                name: np.zeros((self.num_envs, *getattr(self._scratch, name).shape)) for name in RNE_POST_FIELDS
            }
        self._closed = False
        self.reset()

    @property
    def nthread(self) -> int:
        return self.b.nthread

    def data(self, name: str) -> torch.Tensor:
        self._check_open()
        view = self._views.get(name)
        if view is None:
            if self._rne_post and name in RNE_POST_FIELDS:
                array = self._rne[name]
            elif name not in self.outputs:
                raise KeyError(f"{name!r} is not bound; pass it in outputs= when building the backend")
            else:
                array = getattr(self.b, name)
            with warnings.catch_warnings():
                # Outputs are read-only arrays that the batch refreshes in place after every call.
                warnings.simplefilter("ignore", UserWarning)
                view = self._views[name] = torch.from_numpy(array)
        return view

    def numpy(self, name: str) -> np.ndarray:
        """The live float64 NumPy array of ``data(name)``."""
        return self.data(name).numpy()

    def set_models(self, models) -> None:
        """Run environment ``i`` on ``models[i]``: models compiled from one spec with other parameters."""
        self._check_open()
        self._set_models(_model_list(models, self.num_envs))

    def _set_models(self, models) -> None:
        self._models = models
        self.b.set_models(models)

    def model_of(self, env: int):
        """The ``MjModel`` environment ``env`` runs on."""
        return self.model if self._models is None else self._models[env]

    def model_field(self, name: str) -> torch.Tensor:
        """Live ``[B, ...]`` per-environment values of the model field ``name`` (``mujoco.batch`` ``expand``)."""
        self._check_open()
        view = self._model_views.get(name)
        if view is None:
            view = self._model_views[name] = torch.from_numpy(self.b.expand(name))
        return view

    def _baseline(self, ids) -> None:
        if ids is None:
            self._warnings = self.b.warning[:, :, 1].copy()
        else:
            self._warnings[ids] = self.b.warning[ids, :, 1]

    def reset(self, env_ids=None) -> None:
        self._check_open()
        ids = _env_ids(env_ids, self.num_envs)
        ids = None if ids is None else np.unique(ids)
        self.b.reset(ids, keyframe=self._keyframe)
        self._baseline(ids)
        if self._rne_post:
            self._run_rne_post(ids)

    def set_physics(self, values: Mapping, env_ids=None, *, forward: bool = True) -> None:
        self._check_open()
        ids = _env_ids(env_ids, self.num_envs)
        rows = slice(None) if ids is None else ids
        for name, value in values.items():
            if name not in self.fields:
                raise KeyError(f"{name!r} is not one of {self.fields}")
            target = getattr(self.b, name)
            if torch.is_tensor(value):
                value = value.detach().cpu().numpy()
            n = self.num_envs if ids is None else len(ids)
            target[rows] = np.asarray(value, np.float64).reshape(n, *target.shape[1:])
        # The batch takes sorted ids.
        ids = None if ids is None else np.unique(ids)
        self._baseline(ids)
        if forward:
            self.b.forward(ids)
            if self._rne_post:
                self._run_rne_post(ids)

    def forward(self, env_ids=None) -> None:
        self._check_open()
        ids = _env_ids(env_ids, self.num_envs)
        self.b.forward(None if ids is None else np.unique(ids))

    def step(self, ctrl=None, *, mocap_pos=None, mocap_quat=None, active=None) -> None:
        self._check_open()
        ids = _env_ids(active, self.num_envs)
        ids = None if ids is None else np.unique(ids)
        for name, value in (("ctrl", ctrl), ("mocap_pos", mocap_pos), ("mocap_quat", mocap_quat)):
            if value is None:
                continue
            target = getattr(self.b, name)
            if torch.is_tensor(value):
                # NumPy has no bfloat16.
                value = (value.float() if value.dtype == torch.bfloat16 else value).detach().cpu().numpy()
            target[:] = np.asarray(value).reshape(target.shape)
        self.b.step(ids, self.frame_skip)
        if self._rne_post:
            self._run_rne_post(ids)

    def _run_rne_post(self, ids) -> None:
        import mujoco

        data = self._scratch
        for i in range(self.num_envs) if ids is None else ids:
            self.b.get_data(int(i), data)
            mujoco.mj_rnePostConstraint(self.model_of(int(i)), data)
            for name in RNE_POST_FIELDS:
                self._rne[name][i] = getattr(data, name)

    def unstable_envs(self) -> np.ndarray:
        """Ids of environments whose MuJoCo warning counters rose since their last reset or ``set_physics``."""
        self._check_open()
        return np.flatnonzero((self.b.warning[:, :, 1] > self._warnings).any(axis=1))

    def check_buffers(self) -> None:
        self._check_open()

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        self._views = {}
        self._model_views = {}
        self.b = None


def _log_warning(message: str) -> None:
    from .._core.config import log

    log("WARNING", f"MuJoCo: {message}")


_OVERFLOW_KERNEL = None


def _overflow_kernel():
    global _OVERFLOW_KERNEL
    if _OVERFLOW_KERNEL is None:
        import warp as wp

        @wp.kernel
        def overflow(
            nacon: wp.array(dtype=int),
            nefc: wp.array(dtype=int),
            naconmax: int,
            njmax: int,
            flag: wp.array(dtype=int),
        ):
            world = wp.tid()
            if nefc[world] > njmax:
                flag[0] = flag[0] | 2
            if world == 0 and nacon[0] > naconmax:
                flag[0] = flag[0] | 1

        _OVERFLOW_KERNEL = overflow
    return _OVERFLOW_KERNEL


class _Scratch:
    """Device memory for the temporary arrays of a captured call, reused in the order of the first call.

    A buffer freed during the call serves a later request of the same size. MuJoCo Warp frees some temporary
    arrays only when the garbage collector breaks their reference cycles, so a request that finds no free buffer
    collects first. Once ``fixed``, every request must find a free buffer, as the captured call repeats the
    requests of the first call and collects at least as much garbage.
    """

    def __init__(self, device):
        self.device = device
        self.fixed = False
        self._owned: list[torch.Tensor] = []
        self._free: dict[int, list[int]] = {}

    def allocate(self, size_in_bytes):
        free = self._free.get(size_in_bytes)
        if not free:
            gc.collect(1)
            free = self._free.get(size_in_bytes)
        if not free and self.fixed:
            gc.collect()
            free = self._free.get(size_in_bytes)
        if free:
            return free.pop()
        if self.fixed:
            raise RuntimeError("A captured MuJoCo Warp call requested temporary memory that its first call did not")
        buffer = torch.empty(max(int(size_in_bytes), 1), dtype=torch.uint8, device=self.device)
        self._owned.append(buffer)
        return buffer.data_ptr()

    def deallocate(self, ptr, size_in_bytes):
        self._free.setdefault(size_in_bytes, []).append(ptr)


class WarpBatch(MjBatch):
    """B MuJoCo Warp worlds on one CUDA device, stepped by one captured CUDA graph."""

    backend = "mjwarp"

    def __init__(
        self,
        models,
        num_envs: int,
        device="cuda",
        *,
        frame_skip: int = 1,
        keyframe: int | None = None,
        rne_postconstraint: bool = False,
        outputs: Sequence[str] = (),
        naconmax: int | None = None,
        njmax: int | None = None,
        batched_fields: Sequence[str] = (),
        model_options: Mapping[str, object] | None = None,
        exclude_pairs: Iterable[tuple[int, int]] = (),
        fused_tree_max_envs: int = FUSED_TREE_MAX_ENVS,
        step_fn: Callable[[object, object], None] | None = None,
        **_native_options,
    ):
        import mujoco
        import mujoco_warp as mjw

        from .._core.warp_cuda import require_warp_cuda

        self._closed = True
        self.num_envs, self.device, self.frame_skip = int(num_envs), torch.device(device), int(frame_skip)
        if self.device.type != "cuda":
            raise ValueError("The MuJoCo Warp backend requires a CUDA device")
        self._wp = require_warp_cuda("MuJoCo Warp physics")
        self._mjw = mjw
        wp = self._wp
        self._wp_device = f"cuda:{self.device.index if self.device.index is not None else torch.cuda.current_device()}"
        models = _model_list(models, self.num_envs)
        self.model = models[0]
        self._rne_post = bool(rne_postconstraint)
        self.fused_tree = self.num_envs <= int(fused_tree_max_envs)
        self.fields = tuple(f for f in PHYSICS_FIELDS if not f.startswith("mocap") or self.model.nmocap)
        self.outputs = tuple(name for name in outputs if name not in self.fields)
        self._step_fn = step_fn
        self._batched_options = tuple(name[4:] for name in batched_fields if name.startswith("opt."))
        batched_fields = tuple(name for name in batched_fields if not name.startswith("opt."))
        template = mujoco.MjData(self.model)
        if keyframe is None:
            mujoco.mj_resetData(self.model, template)
        else:
            mujoco.mj_resetDataKeyframe(self.model, template, int(keyframe))
        mujoco.mj_forward(self.model, template)
        self._template = {name: np.array(getattr(template, name)) for name in self.fields}
        self.naconmax = (
            int(naconmax) if naconmax is not None else _default_naconmax(self.model, template, self.num_envs)
        )
        self.njmax = int(njmax) if njmax is not None else _default_njmax(self.model, template)
        torch.cuda.synchronize(self.device)
        with wp.ScopedDevice(self._wp_device):
            order = {id(m): m for m in models}
            variants, order = list(order.values()), {key: k for k, key in enumerate(order)}
            self._variant_index = torch.as_tensor([order[id(m)] for m in models], device=self.device)
            rows, self._batched = self._batched_fields(variants, batched_fields)
            try:
                self._m = mjw.put_model(self.model, batch_sizes={name: self.num_envs for name in self._batched})
                for name, value in (model_options or {}).items():
                    setattr(self._m.opt, name, value)
                for name in self._batched_options:
                    value = getattr(self._m.opt, name)
                    if not isinstance(value, wp.array):
                        raise ValueError(f"Option field {name!r} cannot be batched per environment")
                    per_env = np.repeat(value.numpy()[:1], self.num_envs, 0)
                    setattr(self._m.opt, name, wp.array(per_env, dtype=value.dtype))
                _exclude_pairs(self._m, self.model, exclude_pairs)
                self._write_variants(rows)
                self._d = mjw.put_data(
                    self.model, template, nworld=self.num_envs, naconmax=self.naconmax, njmax=self.njmax
                )
                self._flag = wp.zeros(1, dtype=int)
                torch.cuda.synchronize(self.device)
                wp.synchronize()
            except (MemoryError, RuntimeError, ValueError) as error:
                if isinstance(error, ValueError) and "overflow" in str(error):
                    raise BufferOverflowError(
                        f"MuJoCo Warp buffers are too small for the initial state of {self.num_envs} worlds: {error}"
                    ) from error
                if isinstance(error, MemoryError) or "memory" in str(error).lower():
                    raise BufferOverflowError(
                        f"MuJoCo Warp could not allocate {self.num_envs} worlds with naconmax={self.naconmax}, "
                        f"njmax={self.njmax} on {self._wp_device}: {error}. Use fewer environments or smaller "
                        "naconmax/njmax."
                    ) from error
                raise
            self._views: dict[str, torch.Tensor] = {}
            self._flag_t = wp.to_torch(self._flag)
            self._flag_host = torch.zeros(1, dtype=torch.int32, pin_memory=True)
            self._flag_event = torch.cuda.Event()
            self._flag_pending = False
            self._closed = False
            # Compile every kernel before the capture. Warp captures on its own stream; the graphs launch on
            # the current Torch stream.
            self._scratch: list[_Scratch] = []
            self._fused: list[TreeWalks] = []
            self._step_graph = self.capture(self._step_body)
            self._forward_graph = self.capture(self._forward_body)
        self._write_template(None)
        self._flag_t.zero_()
        self._launch(self._forward_graph)
        torch.cuda.synchronize(self.device)

    def capture(self, body: Callable[[], None]):
        """Run ``body`` once, then capture it in a Warp graph whose temporary arrays live in retained memory.

        MuJoCo Warp allocates temporary arrays in each call. Captured as they are, they become allocation nodes,
        and a graph with allocation nodes is slow to launch when other allocations happen between launches.
        """
        from .mjwarp_fused import tree_walks

        wp = self._wp
        scratch = _Scratch(self.device)
        # The scratch runs the garbage collector itself, so both calls free the same buffers at the same points.
        collecting = gc.isenabled()
        gc.collect()
        gc.disable()
        try:
            with (
                wp.ScopedDevice(self._wp_device),
                wp.ScopedAllocator(self._wp_device, scratch),
                tree_walks(self._m, self.fused_tree) as fused,
            ):
                self.fused_tree = fused is not None
                body()
                wp.synchronize()
                gc.collect()
                scratch.fixed = True
                with wp.ScopedCapture() as capture:
                    body()
                wp.synchronize()
        finally:
            if collecting:
                gc.enable()
        self._scratch.append(scratch)
        if fused is not None:
            self._fused.append(fused)
        return capture.graph

    def _batched_fields(self, variants, requested) -> tuple[list, tuple[str, ...]]:
        """Return the variants' Warp models and the model fields that differ between them, plus ``requested``."""
        import dataclasses

        mjw, wp = self._mjw, self._wp
        names = set(requested)
        rows = [mjw.put_model(m) for m in variants] if len(variants) > 1 or names else []
        if len(rows) > 1:
            base = rows[0]
            for field in dataclasses.fields(mjw.Model):
                value = getattr(base, field.name, None)
                if not isinstance(value, wp.array) or field.name in names:
                    continue
                first = value.numpy()
                if any(not np.array_equal(first, getattr(row, field.name).numpy()) for row in rows[1:]):
                    if not (getattr(field.type, "shape", ()) or ("",))[0] == "*":
                        raise ValueError(
                            f"Model field {field.name!r} differs between environments; MuJoCo Warp cannot batch it"
                        )
                    names.add(field.name)
            for field in dataclasses.fields(base.opt):
                values = [getattr(row.opt, field.name) for row in rows]
                values = [v.numpy() if isinstance(v, wp.array) else v for v in values]
                if any(not np.array_equal(values[0], v) for v in values[1:]):
                    raise ValueError(f"Option field {field.name!r} differs between environments")
        return rows, tuple(sorted(names))

    def _write_variants(self, rows) -> None:
        for name in self._batched:
            stacked = torch.stack([self._wp.to_torch(getattr(row, name))[0] for row in rows])
            self._wp.to_torch(getattr(self._m, name)).copy_(stacked[self._variant_index])

    def model_field(self, name: str) -> torch.Tensor:
        """A live ``[B, ...]`` view of a batched model field, or of a batched option as ``"opt.<name>"``.

        Writes take effect at the next step or forward on the current Torch stream.
        """
        self._check_open()
        view = self._views.get(f"model.{name}")
        if view is None:
            option = name.removeprefix("opt.")
            if name.startswith("opt.") and option in self._batched_options:
                array = getattr(self._m.opt, option)
            elif name in self._batched:
                array = getattr(self._m, name)
            else:
                raise KeyError(f"{name!r} is not batched; pass it in batched_fields= when building the backend")
            view = self._views[f"model.{name}"] = self._wp.to_torch(array)
        return view

    def set_models(self, models) -> None:
        """Write new per-environment parameters into the batched model fields; other fields must not change."""
        self._check_open()
        models = _model_list(models, self.num_envs)
        order = {id(m): m for m in models}
        variants, order = list(order.values()), {key: k for k, key in enumerate(order)}
        with self._wp.ScopedDevice(self._wp_device):
            rows, changed = self._batched_fields([self.model, *variants], self._batched)
            extra = sorted(set(changed) - set(self._batched))
            if extra:
                raise ValueError(f"Model fields {extra} change between environments; pass them in batched_fields")
            self._variant_index = torch.as_tensor([1 + order[id(m)] for m in models], device=self.device)
            torch.cuda.synchronize(self.device)
            self._write_variants(rows)

    def _stream(self):
        return self._wp.stream_from_torch(torch.cuda.current_stream(self.device))

    def _step_body(self) -> None:
        mjw, m, d = self._mjw, self._m, self._d
        if self._step_fn is not None:
            self._step_fn(m, d)
        else:
            # Sensors do not act on the dynamics: only the last substep computes them.
            flags = m.opt.disableflags
            if self.model.nsensor:
                m.opt.disableflags = flags | int(mjw.DisableBit.SENSOR)
            try:
                for substep in range(self.frame_skip):
                    if substep == self.frame_skip - 1:
                        m.opt.disableflags = flags
                    mjw.step(m, d)
            finally:
                m.opt.disableflags = flags
        if self._rne_post:
            mjw.rne_postconstraint(m, d)
        self._wp.launch(
            _overflow_kernel(),
            dim=self.num_envs,
            inputs=[d.nacon, d.nefc, self.naconmax, self.njmax, self._flag],
        )

    def _forward_body(self) -> None:
        self._mjw.forward(self._m, self._d)
        if self._rne_post:
            self._mjw.rne_postconstraint(self._m, self._d)

    def data(self, name: str) -> torch.Tensor:
        self._check_open()
        view = self._views.get(name)
        if view is None:
            view = self._views[name] = self._wp.to_torch(getattr(self._d, name))
        return view

    def _write_template(self, ids) -> None:
        rows = slice(None) if ids is None else ids
        for name in self.fields:
            target = self.data(name)
            target[rows] = torch.as_tensor(self._template[name], device=self.device, dtype=target.dtype)
        self.data("ctrl")[rows] = 0

    def _launch(self, graph) -> None:
        self._wp.capture_launch(graph, stream=self._stream())

    def reset(self, env_ids=None) -> None:
        self._check_open()
        ids = _env_ids(env_ids, self.num_envs)
        self._write_template(None if ids is None else torch.as_tensor(ids, device=self.device))
        self._launch(self._forward_graph)

    def set_physics(self, values: Mapping, env_ids=None, *, forward: bool = True) -> None:
        self._check_open()
        ids = _env_ids(env_ids, self.num_envs)
        index = None if ids is None else torch.as_tensor(ids, device=self.device)
        for name, value in values.items():
            if name not in self.fields:
                raise KeyError(f"{name!r} is not one of {self.fields}")
            target = self.data(name)
            n = self.num_envs if ids is None else len(ids)
            value = torch.as_tensor(value).to(self.device, target.dtype).reshape(n, *target.shape[1:])
            if index is None:
                target.copy_(value)
            else:
                target[index] = value
        if forward:
            self._launch(self._forward_graph)

    def forward(self, env_ids=None) -> None:
        self._check_open()
        self._launch(self._forward_graph)

    def step(self, ctrl=None, *, mocap_pos=None, mocap_quat=None, active=None) -> None:
        self._check_open()
        self._poll_overflow()
        for name, value in (("ctrl", ctrl), ("mocap_pos", mocap_pos), ("mocap_quat", mocap_quat)):
            if value is not None:
                target = self.data(name)
                target.copy_(torch.as_tensor(value).to(self.device, target.dtype).reshape(target.shape))
        if active is None:
            self._launch(self._step_graph)
        else:
            # Every world steps in the graph; inactive worlds then get their state and outputs back.
            active = torch.as_tensor(active, device=self.device)
            if active.dtype != torch.bool:
                active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device).index_fill_(0, active, True)
            names = (*self.fields, *self.outputs)
            saved = [self.data(name).clone() for name in names]
            self._launch(self._step_graph)
            for name, old in zip(names, saved, strict=True):
                new = self.data(name)
                new.copy_(torch.where(active.reshape(-1, *([1] * (new.ndim - 1))), new, old))
        if not self._flag_pending:
            self._flag_host.copy_(self._flag_t, non_blocking=True)
            self._flag_event.record()
            self._flag_pending = True

    def _poll_overflow(self) -> None:
        if self._flag_pending and self._flag_event.query():
            self._flag_pending = False
            self._raise_overflow(int(self._flag_host[0]))

    def _raise_overflow(self, flag: int) -> None:
        if not flag:
            return
        self._flag_t.zero_()
        problems = []
        if flag & 1:
            problems.append(f"more than naconmax={self.naconmax} contacts in the batch")
        if flag & 2:
            problems.append(f"more than njmax={self.njmax} constraints in an environment")
        raise BufferOverflowError(
            f"MuJoCo Warp buffers overflowed ({'; '.join(problems)}); pass larger naconmax/njmax"
        )

    def check_buffers(self) -> None:
        """Raise :class:`BufferOverflowError` if any step so far overflowed a buffer; synchronizes the stream."""
        self._check_open()
        self._raise_overflow(int(self._flag_t[0]))

    def unstable_envs(self) -> np.ndarray:
        """Ids of environments whose ``qpos`` or ``qvel`` is not finite; synchronizes the stream."""
        self._check_open()
        bad = ~(self.data("qpos").isfinite().all(1) & self.data("qvel").isfinite().all(1))
        return bad.nonzero().flatten().cpu().numpy()

    def close(self) -> None:
        if getattr(self, "_closed", True):
            return
        self._closed = True
        self._views = {}
        self._m = self._d = self._step_graph = self._forward_graph = None
        self._scratch = []
        self._fused = []


def _exclude_pairs(m, model, pairs) -> None:
    """Remove geom pairs from MuJoCo Warp's precomputed collision candidates, as an ``<exclude>`` would."""
    import warp as wp

    drop = {(min(int(a), int(b)), max(int(a), int(b))) for a, b in pairs}
    if not drop:
        return

    def host(value):
        return value.numpy() if isinstance(value, wp.array) else np.asarray(value)

    device = m.nxn_geom_pair_filtered.device
    geoms, ids = host(m.nxn_geom_pair_filtered), host(m.nxn_pairid_filtered)
    # Pairs that a collision sensor reads stay.
    keep = np.array([(int(a), int(b)) not in drop or s >= 0 for (a, b), (_, s) in zip(geoms, ids)], dtype=bool)
    counts = list(m.geom_pair_type_count)
    ntypes = int((np.sqrt(8 * len(counts) + 1) - 1) / 2)
    for a, b in geoms[~keep]:
        i, j = sorted((int(model.geom_type[a]), int(model.geom_type[b])))
        counts[(i * (2 * ntypes - i - 1)) // 2 + j] -= 1
    m.nxn_geom_pair_filtered = wp.array(geoms[keep], dtype=wp.vec2i, device=device)
    m.nxn_pairid_filtered = wp.array(ids[keep], dtype=wp.vec2i, device=device)
    pairid = host(m.nxn_pairid).copy()
    for k, (a, b) in enumerate(host(m.nxn_geom_pair)):
        if (int(a), int(b)) in drop and pairid[k, 1] < 0:
            pairid[k, 0] = -2
    m.nxn_pairid = wp.array(pairid, dtype=wp.vec2i, device=device)
    m.geom_pair_type_count = tuple(counts)


def _default_naconmax(model, data, num_envs: int) -> int:
    """Contacts of all worlds: per world the model's ``nconmax``, or MuJoCo Warp's default with room to double."""
    from mujoco_warp._src.io import _default_nconmax

    per_world = int(model.nconmax) if model.nconmax > 0 else max(_default_nconmax(model, data), 2 * int(data.ncon))
    return per_world * int(num_envs)


def _default_njmax(model, data) -> int:
    """Constraints per world: the model's ``njmax``, or MuJoCo Warp's default with room to double."""
    from mujoco_warp._src.io import _default_njmax

    return int(model.njmax) if model.njmax > 0 else max(_default_njmax(model, data), 2 * int(data.nefc))


class ResetPool:
    """Run a per-environment source reset across a few owners, each with one source environment.

    ``factory()`` builds one owner (a source environment); ``reset(owner, env_id, seed, *args)`` resets it for
    environment ``env_id`` and returns ``{name: array}``. ``run(env_ids, seeds)`` resets the environments on
    ``owners`` threads and returns ``{name: [len(env_ids), ...]}`` in ``env_ids`` order, ready for
    ``set_physics(values, env_ids)``. Threads help when the reset spends its time in MuJoCo calls, which
    release the GIL; ``processes=True`` runs owners in spawned worker processes instead, for Python-heavy
    resets (``factory`` and ``reset`` must then be picklable). Host memory holds ``owners`` source
    environments, not one per environment.
    """

    def __init__(self, factory: Callable, reset: Callable, owners: int = 1, *, processes: bool = False):
        self.owners = max(1, int(owners))
        self._reset = reset
        self._closed = False
        self._processes = bool(processes) and self.owners > 1
        if self._processes:
            import multiprocessing as mp

            ctx = mp.get_context("spawn")
            self._conns, self._procs = [], []
            for _ in range(self.owners):
                parent, child = ctx.Pipe()
                proc = ctx.Process(target=_reset_worker, args=(child, factory, reset), daemon=True)
                proc.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(proc)
            for conn in self._conns:
                ok, payload = conn.recv()
                if not ok:
                    self.close()
                    raise RuntimeError(f"Reset worker failed to start:\n{payload}")
        else:
            self._envs = [factory() for _ in range(self.owners)]
            self._locks = [threading.Lock() for _ in range(self.owners)]
            self._pool = ThreadPoolExecutor(self.owners) if self.owners > 1 else None

    @property
    def envs(self) -> list:
        """The owners' source environments, in this process (thread owners only)."""
        return self._envs

    def run(self, env_ids, seeds=None, *args) -> dict[str, np.ndarray]:
        if self._closed:
            raise RuntimeError("ResetPool is closed")
        ids = [int(i) for i in np.asarray(env_ids).reshape(-1)]
        if seeds is None:
            seeds = [None] * len(ids)
        seeds = [None if s is None else int(s) for s in seeds]
        if not ids:
            return {}
        chunks = [list(range(k, len(ids), self.owners)) for k in range(min(self.owners, len(ids)))]
        if self._processes:
            for conn, rows in zip(self._conns, chunks, strict=False):
                conn.send(("reset", [(ids[r], seeds[r]) for r in rows], args))
            replies = []
            for conn in self._conns[: len(chunks)]:
                ok, payload = conn.recv()
                if not ok:
                    raise RuntimeError(f"Reset worker failed:\n{payload}")
                replies.append(payload)
        else:

            def work(owner):
                env = self._envs[owner]
                with self._locks[owner]:
                    return [self._reset(env, ids[r], seeds[r], *args) for r in chunks[owner]]

            owners = range(len(chunks))
            replies = (
                list(self._pool.map(work, owners))
                if self._pool is not None and len(chunks) > 1
                else [work(k) for k in owners]
            )
        rows = [None] * len(ids)
        for chunk, reply in zip(chunks, replies, strict=True):
            for r, value in zip(chunk, reply, strict=True):
                rows[r] = value
        return {key: np.stack([np.asarray(row[key]) for row in rows]) for key in rows[0]}

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        if self._processes:
            for conn in self._conns:
                try:
                    conn.send(("close", None, ()))
                except (BrokenPipeError, OSError):
                    pass
            for proc in self._procs:
                proc.join(timeout=5)
                if proc.is_alive():
                    proc.kill()
        elif self._pool is not None:
            self._pool.shutdown()
        self._envs = []

    def __del__(self):
        try:
            self.close()
        except Exception:
            pass


def _reset_worker(conn, factory, reset) -> None:
    from .workers import pin_worker, send_failure

    try:
        pin_worker(None)
        env = factory()
        conn.send((True, None))
        while True:
            command, items, args = conn.recv()
            if command == "close":
                break
            try:
                conn.send((True, [reset(env, i, s, *args) for i, s in items]))
            except Exception:
                send_failure(conn)
    except Exception:
        send_failure(conn)
