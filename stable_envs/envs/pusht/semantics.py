"""PushT controls with batched Numba CPU physics and batched Triton or JAX CUDA physics."""

import copy

import numpy as np
import torch
from tensordict import TensorDict, TensorDictBase

from ... import tuning
from ..._execution.protocol import StepPipeline, step_pipeline
from ...env import Env
from ...registry import REGISTRY, register
from ...spaces import Box, Dict, Discrete
from ...specs import CompositeSpec, Spec
from . import legacy


def _to_host(tensors):
    """Copy ``{key: tensor}`` to host tensors with one transfer per dtype."""
    if not any(value.device.type != "cpu" for value in tensors.values()):
        return dict(tensors)
    groups = {}
    for key, value in tensors.items():
        groups.setdefault(value.dtype, []).append(key)
    host = {}
    for keys in groups.values():
        flat = torch.cat([tensors[key].reshape(-1) for key in keys]).cpu()
        offset = 0
        for key in keys:
            size = tensors[key].numel()
            host[key] = flat[offset : offset + size].view(tensors[key].shape)
            offset += size
    return {key: host[key] for key in tensors}


def _lookup(nodes, key):
    """``state.get(key, None)`` through ``nodes``: the children of each nested TensorDict already reached."""
    parent = key[:-1]
    if parent not in nodes:
        node = _lookup(nodes, parent)
        nodes[parent] = dict(node.items()) if isinstance(node, TensorDictBase) else None
    children = nodes[parent]
    return None if children is None else children.get(key[-1])


def _to_device(fields, device):
    """Copy host render ``fields`` to ``device`` with one transfer per dtype."""
    groups = {}
    for key, value in fields.items():
        groups.setdefault(value.dtype, []).append(key)
    out = {}
    for keys in groups.values():
        flat = torch.from_numpy(np.concatenate([np.ascontiguousarray(fields[key]).reshape(-1) for key in keys]))
        flat = flat.to(device, non_blocking=True)
        offset = 0
        for key in keys:
            size = fields[key].size
            out[key] = flat[offset : offset + size].view(fields[key].shape)
            offset += size
    return {key: out[key] for key in fields}


_EXECUTION_MODES = {"triton": "NATIVE_TRITON", "jax": "NATIVE_JAX", "numba": "NATIVE_NUMBA"}

#: The State leaves the renderers read, besides the ``current`` value and type of each Variation leaf.
DRAWN_LEAVES = frozenset(
    {
        "recipe/config",
        "anchor/start",
        "anchor/goal/value",
        "anchor/goal/type",
        *(f"current/body/{b}/{leaf}" for b in ("agent", "block") for leaf in ("position", "angle")),
        *(f"current/body/{b}/center_of_gravity" for b in ("agent", "block")),
        "current/latest_present",
        "current/latest/value",
        "current/latest/type",
        "current/render_segments",
        "current/render_segment_count",
    }
)


def _drawn(key):
    """Whether a renderer reads the State leaf ``key``.

    Body shapes and scales come from ``anchor/variation``; colours and the goal pose come from ``owned/variation``.
    """
    if key.startswith(("anchor/variation/", "owned/variation/")):
        return key.endswith(("/current/value", "/current/type"))
    return key in DRAWN_LEAVES


class PushT(Env):
    """Source-aligned continuous PushT with complete, replayable numeric State.

    Actions are relative controls in [-1, 1], not absolute pixel targets.
    Physics is a port of the source Pymunk contact solver: Triton (or JAX) on CUDA, Numba on CPU. They run
    the same math and differ from Pymunk numerically at contacts. Resets run the source reset. Checkpoints
    restore the solver cache.
    """

    family = domain = "pusht"
    name = "push-t"
    reference_id = "swm/PushT-v1"
    default_size = 224
    state_dim = 0
    execution_mode = "AUTO_NUMBA_TRITON"
    #: Every step returns new reward, success, terminated and truncated tensors.
    _step_outputs_owned = True
    cpu_render_support = "SUPPORTED_PYGAME"
    action_dim = 2
    action_dtype = torch.float32
    action_low, action_high = -1.0, 1.0
    discrete = False
    #: As in the source: every reset draws the start poses, and the goal is drawn with the block's shape
    #: and scale.
    undrawn_variation_leaves = frozenset(
        {
            "agent.start_position",
            "block.start_position",
            "block.angle",
            "goal.scale",
            "agent.angle",  # undrawn only because the default agent is round; other agent shapes show it
        }
    )

    def __init__(
        self,
        num_envs=1,
        size=None,
        supersample=1,
        device=None,
        backend="auto",
        seed=0,
        *,
        render_action=False,
        with_target=True,
        simulation_backend=None,
        triton_min_envs=None,
    ):
        """``simulation_backend=None`` steps Triton physics on CUDA and Numba physics on CPU.

        ``"triton"`` (CUDA only), ``"jax"`` or ``"numba"`` selects one backend; Numba on CUDA steps on the
        CPU and returns CUDA tensors. ``"auto"`` steps CUDA batches smaller than ``triton_min_envs`` with Numba.
        """
        from .projection import PushTProjection
        from .simulation.execution import TRITON_MIN_ENVS, make_pusht_adapter, pusht_backend

        if backend not in ("auto", "source", "pygame"):
            raise ValueError("PushT backend must be 'auto' or 'source' (pygame)")
        self._render_backend = "source" if backend == "pygame" else backend
        if supersample != 1:
            raise ValueError("The source PushT renderer has no supersample setting")
        device = torch.device(Env.resolve_device(device))
        triton_min_envs = TRITON_MIN_ENVS if triton_min_envs is None else triton_min_envs
        self.simulation_backend = pusht_backend(num_envs, device, simulation_backend, triton_min_envs)
        self.execution_mode = _EXECUTION_MODES[self.simulation_backend]
        # CPU batches step on Numba threads.
        self.native_step_threads = self.simulation_backend == "numba"
        # CPU renders return new storage.
        self._frames_owned = device.type == "cpu"
        resolution = self.default_size if size is None else size
        if render_action and resolution < 96:
            raise ValueError("Source PushT action markers require size >= 96")
        self._simulator = make_pusht_adapter(
            num_envs,
            device,
            seed,
            discrete=self.discrete,
            config=dict(resolution=resolution, render_action=render_action, with_target=with_target),
            simulation_backend=self.simulation_backend,
        )
        self._projection = PushTProjection()
        self._render_pool = None
        self._device_renderer = None
        try:
            super().__init__(num_envs, resolution, 1, device, "pygame", 0 if seed is None else seed)
            row = self._simulator.template.get_state()
            self._fixed_fields = {k: (v.shape, v.dtype) for k, v in row.items() if not k.startswith("history/")}
            self._render_fields = [
                (tuple(k.split("/")), shape, torch.from_numpy(np.empty((), dtype)).dtype)
                for k, (shape, dtype) in self._fixed_fields.items()
                if _drawn(k)
            ]
        except BaseException:
            self._simulator.close()
            self._projection.close()
            raise

    def make_variation_space(self):
        def convert(node):
            if hasattr(node, "spaces"):
                return Dict({k: convert(v) for k, v in node.spaces.items()})
            if hasattr(node, "n"):
                return Discrete(int(node.n), start=int(node.start), init_value=copy.deepcopy(node.init_value))
            return Box(node.low, node.high, dtype=node.dtype, init_value=copy.deepcopy(node.init_value))

        return convert(self._simulator.template.env.variation_space)

    @property
    def state_spec(self):
        # History has a growing axis; the spec describes the current checkpoint only.
        fields = {}
        for key, value in self.get_state().items(True, True):
            branch = fields
            for part in key[:-1]:
                branch = branch.setdefault(part, {})
            branch[key[-1]] = Spec(tuple(value.shape[1:]), dtype=value.dtype, device=self.device)
        return CompositeSpec(fields, device=self.device)

    @property
    def action_spec(self):
        return Spec(
            (2,),
            dtype=self.action_dtype,
            low=self.action_low,
            high=self.action_high,
            discrete_n=16 if self.discrete else None,
            device=self.device,
        )

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        return self._simulator.reset(seed, options)

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)

    def _step(self, actions, *, active=None):
        return self._simulator.step(actions, active=active)

    def _advance_without_snapshot(self, actions, *, active=None):
        return self._simulator.step(actions, active=active, capture_state=False)

    def _step_pipeline(self) -> StepPipeline | None:
        return step_pipeline(self._simulator)

    def _render_live(self):
        if self._rasterizes(self.size):
            from . import raster

            fields = self._simulator.render_fields()
            if isinstance(next(iter(fields.values())), torch.Tensor):
                return self._draw_device(fields, self.size)
            if raster.HAVE_TRITON:
                return self._draw_device(_to_device(fields, self.device), self.size)
            return raster.render(fields, self.size, self.device, lambda rows: self._draw_pygame(rows, self.size))
        if self.device.type == "cuda":
            return self.render()
        live = self._simulator.render_fields()
        fields = {"/".join(key): live["/".join(key)].numpy() for key, _, _ in self._render_fields}
        if self._cpu_port(self.size):
            from . import raster

            return raster.render_cpu(fields, self.size, lambda rows: self._draw_pygame(rows, self.size))
        return torch.as_tensor(self._draw_pygame(fields, self.size))

    @staticmethod
    def _stack_states(values):
        from .simulation.history import stack_states

        return stack_states(values)

    def _variation_rows(self, attribute):
        keys = [key for key, _, _ in self.variation_space.flat_layout()]
        return torch.as_tensor(self._simulator.variation_rows(attribute, keys), device=self.device)

    def get_variation(self):
        self._check_open("get_variation")
        return self._variation_rows("value")

    def sample_variation(self, batch=None, paths=None, rng=None):
        """Sample valid integer colors and source discrete ranges (including start=1)."""
        self._check_open("sample_variation")
        batch = self.B if batch is None else batch
        if type(batch) is not int or batch < 1:
            raise ValueError("PushT Variation sample batch must be positive")
        if paths is not None and set(paths) - set(self.variation_names):
            raise KeyError("Unknown PushT Variation path")
        rng = self.rng if rng is None else rng
        result = np.tile(self.default_variation(), (batch, 1))
        for key, offset, width in self.variation_space.flat_layout():
            if paths is not None and key not in paths:
                continue
            node = self.variation_space._walk(key)
            if hasattr(node, "n"):
                values = rng.integers(node.start, node.start + node.n, (batch, width))
            else:
                low, high = np.asarray(node.low).reshape(-1), np.asarray(node.high).reshape(-1)
                values = rng.uniform(low, high.astype(np.float64) + (node.dtype.kind in "iub"), (batch, width))
                if node.dtype.kind in "iub":
                    values = np.floor(values)
            result[:, offset : offset + width] = values
        return result

    #: Leaves only the frame reads; ``set_variation`` applies them to the running episode as well.
    #: Body shapes and scales need a rebuilt pymunk space, and start poses and velocity exist only
    #: at reset, so those wait for the next reset.
    immediate_variation_leaves = frozenset(
        {
            "agent.color",
            "block.color",
            "goal.color",
            "goal.scale",
            "goal.position",
            "goal.angle",
            "background.color",
            "rendering.render_goal",
        }
    )

    def set_variation(self, variation=None, **overrides):
        """Configure the values every later reset restores.

        Appearance leaves (``immediate_variation_leaves``) also apply to the running episode;
        the others take effect at the next reset.
        """
        from .simulation.replay_batch import resolve_variation

        self._check_open("set_variation")
        if isinstance(variation, dict):
            overrides = {**variation, **overrides}
            variation = None
        flat_keys = set()
        if variation is not None:
            rows = np.asarray(variation.detach().cpu() if torch.is_tensor(variation) else variation)
            if rows.shape == (self.variation_dim,):
                rows = np.broadcast_to(rows, (self.B, self.variation_dim))
            if rows.shape != (self.B, self.variation_dim):
                raise ValueError("Invalid PushT Variation batch")
            for key, off, width in self.variation_space.flat_layout():
                overrides[key] = rows[:, off : off + width].reshape(self.B, *self.variation_space._walk(key).shape)
                flat_keys.add(key)
        lanes = resolve_variation(self._simulator.template.env.variation_space, overrides, self.B, flat_keys)
        self._simulator.install_variation(lanes, self.immediate_variation_leaves)
        return self._variation_rows("init_value")

    def to_render_state(self, state=None):
        """Return an owned, history-free copy of the State fields the CPU renderer reads."""
        state = self.get_state() if state is None else state
        self._render_fields_of(state)
        return state.select(*(key for key, _, _ in self._render_fields)).clone()

    def _render_fields_of(self, state, on_device=False):
        self._check_open("to_render_state")
        state = self.get_state() if state is None else state
        if not isinstance(state, TensorDict) or len(state.batch_size) != 1 or not state.batch_size[0]:
            raise ValueError("PushT render requires a nonempty State batch")
        selected, error, nodes = {}, None, {(): dict(state.items())}
        batch = tuple(state.batch_size)
        for key, shape, dtype in self._render_fields:
            value = _lookup(nodes, key)
            if not isinstance(value, torch.Tensor):
                error = KeyError(key)
                break
            if value.shape != (*batch, *shape) or value.dtype != dtype:
                error = ValueError(f"Invalid PushT render field: {key}")
                break
            selected[key] = value
        if error is None and on_device and all(value.is_cuda for value in selected.values()):
            floats = [value.reshape(-1) for value in selected.values() if value.is_floating_point()]
            if floats and not bool(torch.isfinite(torch.cat(floats)).all()):
                key = next(k for k, v in selected.items() if v.is_floating_point() and not bool(v.isfinite().all()))
                raise ValueError(f"Nonfinite PushT render field: {key}")
            return {"/".join(key): value for key, value in selected.items()}
        host = _to_host(selected)
        fields = {}
        # Fields are checked in order, so an earlier nonfinite field is reported before a later layout error.
        for key, value in host.items():
            array = value.numpy()
            if array.dtype.kind == "f" and not np.isfinite(array).all():
                raise ValueError(f"Nonfinite PushT render field: {key}")
            fields["/".join(key)] = array
        if error is not None:
            raise error
        return fields

    def render(self, state=None, *, size=None, out=None):
        self._check_open("render")
        size = self.size if size is None else size
        if type(size) is not int or size < 1:
            raise ValueError("PushT render size must be a positive integer")
        if self._simulator.template.config["render_action"] and size < 96:
            raise ValueError("Source PushT action markers require size >= 96")
        fields = self._render_fields_of(state, on_device=self._rasterizes(size))
        if self._rasterizes(size):
            from . import raster

            if isinstance(next(iter(fields.values())), torch.Tensor):
                frames = self._draw_device(fields, size)
            else:
                frames = raster.render(fields, size, self.device, lambda rows: self._draw_pygame(rows, size))
        elif self._cpu_port(size):
            from . import raster

            frames = raster.render_cpu(fields, size, lambda rows: self._draw_pygame(rows, size), out)
        else:
            frames = torch.as_tensor(self._draw_pygame(fields, size), device=self.device)
        if out is not None:
            if (
                out.shape != frames.shape
                or out.dtype != frames.dtype
                or out.device != frames.device
                or not out.is_contiguous()
            ):
                raise ValueError(
                    "PushT render output buffer must be contiguous and match the frames in shape, dtype and device"
                )
            out.copy_(frames)
            return out
        return frames

    def _rasterizes(self, size):
        """Whether CUDA output uses the batched rasterizer, which draws the same bytes as pygame."""
        if self.device.type != "cuda":
            return False
        from . import raster

        return raster.HAVE_TRITON and raster.resize_supported(size)

    def _cpu_port(self, size):
        """Whether CPU output uses the Numba rasterizer, which draws the same bytes as pygame."""
        if self.device.type != "cpu" or self._render_backend == "source":
            return False
        from ...render import cpu_raster
        from . import raster

        return cpu_raster.available() and raster.resize_supported(size)

    def _draw_device(self, fields, size):
        """Draw device render fields with primitives built on the GPU."""
        from . import raster_device

        if self._device_renderer is None:
            self._device_renderer = raster_device.DeviceRenderer(self.device)
        return self._device_renderer(fields, size, lambda rows: self._draw_pygame(rows, size))

    def _draw_pygame(self, fields, size):
        """Draw host render fields with the source pygame code, in render workers for large batches."""
        count = len(fields["recipe/config"])
        pool = self._pool(count)
        if pool is not None:
            return pool.render(fields, size)
        return self._projection.render([{k: v[i] for k, v in fields.items()} for i in range(count)], size)

    def _pool(self, count):
        """Return the render worker pool for ``count`` rows, or None to draw in this process."""
        from ..._execution.policy import cpu_affinity, process_count

        if self._render_pool is None:
            cpus = cpu_affinity()
            processes = process_count(count, cpus, path="pusht-render")
            if processes < 2:
                return None
            from .render_pool import PushTRenderPool

            self._render_pool = PushTRenderPool(processes, cpus)
        elif count < int(tuning.get("worker_min_lanes", path="pusht-render", size=self.size, batch=count) or 0):
            return None
        return self._render_pool

    def _close(self):
        if self._render_pool is not None:
            self._render_pool.close()
        self._projection.close()
        self._simulator.close()


class PushTDiscrete(PushT):
    """Source-aligned PushT with two discrete polar-action indices in [0, 16).

    It steps with the same physics backends as ``PushT`` and restores the contact-solver cache.
    """

    name = "push-t-discrete"
    reference_id = "swm/PushT-Discrete-v1"
    discrete = True
    action_dtype = torch.int64
    action_low, action_high = 0, 15


if "PushT" in REGISTRY:
    assert REGISTRY["PushT"] is legacy.PushT
    del REGISTRY["PushT"]
register(PushT)
register(PushTDiscrete)
