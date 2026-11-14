from __future__ import annotations

import copy

import numpy as np
import torch
from tensordict import TensorDict

from ..._core.style import StyleTensor
from ..._core.tensordicts import stack_leaves
from ..._execution.process import zeros
from ..._execution.protocol import StepPipeline, step_pipeline
from ...env import Env, _split_appearance
from ...registry import REGISTRY, _require_extra, register
from ...specs import CompositeSpec, Spec
from . import legacy as frozen
from .simulation.execution import NUMBA_CUDA_MIN_ENVS, make_rocket_adapter, rocket_backend

#: ``execution_mode`` per backend and device; on CUDA the Numba kernel runs on the host.
_EXECUTION_MODES = {
    ("numba", "cpu"): "NATIVE_NUMBA",
    ("numba", "cuda"): "HOST_NUMBA",
    ("numba_cuda", "cuda"): "NATIVE_NUMBA_CUDA",
    ("pybullet", "cpu"): "DELEGATED_CPU",
    ("pybullet", "cuda"): "DELEGATED_CPU",
}

#: The full State leaf of the RenderState pose rows, which are its first 13 values.
POSE_KEY = ("current", "wrapper", "state")


class PyFlytRocketLanding(frozen.PyFlytRocketLanding):
    """Rocket CPU dynamics with CPU/CUDA rendering.

    ``simulation_backend="numba"`` (the CPU default) steps every environment in one Numba kernel call on the
    CPU, with physical State; on CUDA the results move to the device. ``"numba_cuda"`` (the CUDA default) steps
    them with the CUDA build of the kernel. ``"auto"`` selects ``"numba"`` for CUDA batches below
    ``numba_cuda_min_envs``. ``"pybullet"`` steps PyFlyt 0.29.0 and PyBullet 3.2.7 per environment, with
    replay-backed State that is exact to the source. Explicit body and cached camera poses drive TinyRenderer on
    CPU or the nvdiffrast path on CUDA.

    On CPU, ``backend="pybullet"`` (also ``"tiny"``) draws each frame with PyBullet's TinyRenderer, and ``"numba"``
    draws all frames in one call of a Numba port of it (``tiny_raster``). ``"auto"`` selects ``render.CPU_AUTO``.
    """

    requires_cuda = False
    execution_mode = "NATIVE_NUMBA"
    cpu_render_support = "SUPPORTED_TINYRENDERER"
    action_dim = 7
    action_dtype = torch.float64
    action_low = (-1.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0)
    action_high = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    dynamics_variation_leaves = frozenset(
        {"environment.start_height_ratio", "environment.start_horizontal_offset", "environment.start_tilt"}
    )
    undrawn_variation_leaves = frozenset({"environment.sky_color"})
    #: The source loads its original white pad unless reset options ask for color variations.
    DRAWN_PAD_DEFAULT = (255.0, 255.0, 255.0)
    _render_style = None

    def __init__(
        self,
        num_envs=1,
        size=None,
        supersample=None,
        device="cpu",
        backend="auto",
        seed=0,
        *,
        simulator_config=None,
        simulation_backend=None,
        numba_cuda_min_envs=NUMBA_CUDA_MIN_ENVS,
    ):
        from .render import check_backend

        _require_extra(type(self).__name__, "rocket", "PyFlyt", "pybullet", "numba", "scipy")
        device = Env.resolve_device(device)
        check_backend(backend, device)
        self.simulation_backend = rocket_backend(num_envs, device, simulation_backend, numba_cuda_min_envs)
        self.execution_mode = _EXECUTION_MODES[self.simulation_backend, torch.device(device).type]
        # CPU lanes draw their own step frames, at the size and supersample ``Env`` resolves.
        frame = (
            (int(size or self.default_size), int(1 if supersample is None else supersample))
            if torch.device(device).type == "cpu"
            else None
        )
        self._simulator = make_rocket_adapter(num_envs, device, seed, simulator_config, frame, self.simulation_backend)
        try:
            Env.__init__(
                self,
                num_envs,
                size,
                1 if supersample is None else supersample,
                device,
                backend,
                0 if seed is None else seed,
            )
        except BaseException:
            self._simulator.close()
            raise
        if self.device.type == "cuda" and hasattr(self._simulator, "draw_fields"):
            # Workers write each step's ``frame_fields``; the parent draws them while the next step runs.
            self._simulator.draw_fields = self._draw_fields

    def make_variation_space(self):
        return copy.deepcopy(self._simulator.template.env.variation_space)

    def _layout(self):
        return tuple((k, int(np.prod(shape)), None) for k, (shape, _) in self._simulator.fields.items())

    def _layout_dtypes(self):
        return {k: dtype for k, (_, dtype) in self._simulator.fields.items()}

    @property
    def state_spec(self):
        fields = {}
        for key, (shape, dtype) in self._simulator.fields.items():
            node = fields
            for part in key[:-1]:
                node = node.setdefault(part, {})
            node[key[-1]] = Spec(shape, dtype=dtype, device=self.device)
        return CompositeSpec(fields, device=self.device)

    def reset(self, seed=None, options=None):
        """Reset every lane.

        ``options["variation_values"]`` sets leaves for this episode only, with one value shared by
        the batch or one per lane.
        """
        self._check_open("reset")
        options = dict(options or {})
        values = options.pop("variation_values", None)
        if values is not None:
            if not isinstance(values, dict):
                raise ValueError("variation_values must map Variation paths to values")
            values = self._lane_values(values)
        return self._simulator.reset(seed, options, values)

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)

    def _step(self, actions, *, active=None):
        self._check_open("step")
        return self._simulator.step(actions, active=active)

    def _advance_without_snapshot(self, actions, *, active=None):
        self._check_open("step")
        return self._simulator.step(actions, active=active, capture_state=False)

    def _stack_states(self, values):
        """Stack State snapshots along a new time axis; zero leaves that a step does not change are not copied.

        The CUDA Numba simulator stacks its own States with one copy.
        """
        stack = getattr(self._simulator, "stack_states", None)
        stacked = None if stack is None else stack(values)
        if stacked is not None:
            return stacked
        zero = self._simulator.zero_state_leaves
        stacked = stack_leaves([value.exclude(*zero) for value in values])
        for key in zero:
            leaf = values[0][key]
            stacked[key] = zeros((*stacked.batch_size, *leaf.shape[1:]), leaf.dtype, leaf.device)
        return stacked

    def _step_pipeline(self) -> StepPipeline | None:
        return step_pipeline(self._simulator)

    def _render_live(self):
        if self.device.type == "cpu" and hasattr(self._simulator, "live_fields"):
            from . import render as rocket_render

            return rocket_render.render_cpu(self, *self._simulator.live_fields(), self.size, self.ss)
        if self.device.type == "cpu":
            return torch.from_numpy(self._simulator.render_live())
        if self.device.type == "cuda" and hasattr(self._simulator, "frame_fields"):
            return self._draw_fields(self._simulator.frame_fields())
        return self.render()

    def _draw_fields(self, fields):
        """Draw the live lanes' ``frame_fields`` on CUDA, without a State snapshot."""
        from . import render as rocket_render

        self._check_open("render")
        return rocket_render.draw_gpu(
            self, fields["body"], fields["link"], fields["rgba"], self.size, self.ss, views=fields["view"]
        )

    @torch.no_grad()
    def _render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None):
        """Draw ``(..., 13)`` pose rows on CUDA with the explicit-pose renderer of ``render(state)``.

        The rocket is at the row's pose and the camera at its link pose one physics substep earlier
        (``legacy.lagged_pose``), as the reference bridge places them. The colours come from the style rows.
        """
        if self.device.type != "cuda":
            return frozen.PyFlytRocketLanding._render(self, states, size, supersample, variation, style, out)
        from . import render as rocket_render

        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        batch = int(s.shape[0])
        size = int(size or self.size)
        ss = int(self.ss if supersample is None else supersample)
        style_rows = self.resolve_style(style, variation, batch=batch).values.detach().cpu().numpy()
        body, links, rgba = rocket_render.pose_row_fields(s.detach().cpu().numpy(), style_rows)
        frames = rocket_render.draw_gpu(self, body, links, rgba, size, ss)
        if out is not None:
            out.view(batch, size, size, 3).copy_(frames)
            frames = out.view(batch, size, size, 3)
        return frames.view(*lead, size, size, 3) if lead else frames

    def _variation_rows(self, attribute):
        keys = [key for key, _, _ in self.variation_space.flat_layout()]
        return torch.as_tensor(
            self._simulator.variation_rows(attribute, keys), dtype=torch.float32, device=self.device
        )

    def get_variation(self):
        self._check_open("get_variation")
        return self._variation_rows("value")

    def sample_variation(self, batch=None, paths=None, rng=None):
        """Sample configurations with integer RGB leaves using the supplied or metadata RNG.

        This does not consume simulator reset RNGs. Use reset options to randomize
        the next episode through those streams.
        """
        self._check_open("sample_variation")
        batch = self.B if batch is None else batch
        if type(batch) is not int or batch < 1:
            raise ValueError("Variation sample batch must be a positive integer")
        if paths is not None and set(paths) - set(self.variation_names):
            raise KeyError("Unknown Rocket Variation sample path")
        rng = self.rng if rng is None else rng
        result = np.tile(self.default_variation(), (batch, 1))
        for key, offset, width in self.variation_space.flat_layout():
            if paths is not None and key not in paths:
                continue
            node = self.variation_space._walk(key)
            low = np.asarray(node.low, np.float64).reshape(-1)
            high = np.asarray(node.high, np.float64).reshape(-1)
            values = rng.uniform(low, high + (1 if node.dtype.kind == "u" else 0), (batch, width))
            result[:, offset : offset + width] = np.floor(values) if node.dtype.kind == "u" else values
        return result.astype(np.float32)

    def set_variation(self, variation=None, **overrides):
        """Install defaults for later resets; appearance leaves also apply to the current episode.

        The start leaves describe how an episode begins, so they act from the next reset.
        """
        self._check_open("set_variation")
        if isinstance(variation, dict):
            overrides = {**variation, **overrides}
            variation = None
        if variation is not None:
            rows = np.asarray(variation.detach().cpu() if torch.is_tensor(variation) else variation)
            if rows.shape == (self.variation_space.flat_dim,):
                rows = np.broadcast_to(rows, (self.B, self.variation_space.flat_dim))
            if rows.shape != (self.B, self.variation_space.flat_dim):
                raise ValueError("Invalid Rocket Variation shape")
            for key, off, width in self.variation_space.flat_layout():
                overrides[key] = rows[:, off : off + width].reshape(self.B, *self.variation_space._walk(key).shape)
        self._simulator.configure(self._lane_values(overrides))
        return self._variation_rows("init_value")

    def _lane_values(self, overrides):
        """``{path: one value or one per lane}`` -> one validated ``{path: value}`` per lane."""
        values = [{} for _ in range(self.B)]
        for key, raw in overrides.items():
            if key not in self.variation_space.names():
                raise ValueError(f"Unknown Rocket Variation: {key}")
            node = self.variation_space._walk(key)
            value = np.asarray(raw.detach().cpu() if torch.is_tensor(raw) else raw)
            if value.shape == node.shape:
                value = np.broadcast_to(value, (self.B, *node.shape))
            if value.shape != (self.B, *node.shape) or not np.isfinite(value).all():
                raise ValueError(f"Invalid Rocket Variation shape/value: {key}")
            for i in range(self.B):
                if node.shape == ():
                    # Scalar initial values retain their precision; float64 height
                    # must not round through the float32 sampling distribution.
                    converted = raw if type(raw) is float else value[i].copy()
                    if not node.contains(converted):
                        raise ValueError(f"Invalid Rocket Variation range: {key}")
                    values[i][key] = copy.deepcopy(converted)
                    continue
                converted = value[i].astype(node.dtype)
                if (
                    np.any(converted < node.low)
                    or np.any(converted > node.high)
                    or node.dtype.kind == "u"
                    and not np.array_equal(converted, value[i])
                ):
                    raise ValueError(f"Invalid Rocket Variation range: {key}")
                values[i][key] = converted.copy()
        return values

    def _style_of(self, rgba):
        style = torch.zeros((self.B, 16), dtype=torch.float32, device=rgba.device)
        # Renderer style is alphabetical, not source/corpus order.
        layout = frozen.style_layout()
        for index, name in enumerate(frozen.CORPUS_STYLE_PATHS[:5]):
            offset, width = layout[name]
            style[:, offset : offset + width] = rgba[:, index, :3] * 255
        style[:, layout["flame.ignited"][0]] = torch.isclose(
            rgba[:, 5], torch.tensor(frozen.IGNITION_RGBA, device=rgba.device, dtype=rgba.dtype)
        ).all(-1)
        return style

    def to_render_state(self, state=None, *, size=None, supersample=None):
        from . import render as rocket_render

        self._check_open("to_render_state")
        state = self.get_state() if state is None else state
        rocket_render.host_leaves(state, {**rocket_render.STATE_READS, POSE_KEY: ((13,), True)}, self.B)
        return self._render_state_of(state, size, supersample)

    def _render_state_of(self, state, size, supersample):
        """The RenderState of a full state, without the checks of ``to_render_state``."""
        style = self._style_of(state["render", "rgba"])
        result = TensorDict(
            {
                ("renderer", "pose"): state[POSE_KEY][:, :13].to(torch.float32).clone(),
                ("renderer", "style"): style,
                ("renderer", "size"): torch.full(
                    (self.B,), self.size if size is None else size, dtype=torch.int64, device=self.device
                ),
                ("renderer", "supersample"): torch.full(
                    (self.B,), self.ss if supersample is None else supersample, dtype=torch.int64, device=self.device
                ),
            },
            [self.B],
            device=self.device,
        )
        from . import render as rocket_render

        result.update(rocket_render.project(state))
        return result

    def _pack_render_state(self, state):
        if isinstance(state, TensorDict) and "current" in state.keys():
            return self.to_render_state(state)["renderer", "pose"]
        if isinstance(state, TensorDict):
            return state["renderer", "pose"].to(device=self.device, dtype=torch.float32).clone()
        return torch.as_tensor(state, device=self.device, dtype=torch.float32).reshape(-1, 13).clone()

    def _unpack_render_state(self, rows):
        rows = torch.as_tensor(rows, device=self.device, dtype=torch.float32).reshape(-1, 13)
        # Appearance and all simulator leaves remain absent. This incomplete
        # projection cannot be passed to set_state or impersonate full state.
        return TensorDict({("renderer", "pose"): rows.clone()}, [len(rows)], device=self.device)

    def default_style(self):
        """The default variation row with the pad color the source actually draws by default."""
        row = np.array(super().default_style(), np.float32)
        off, width = next((o, w) for p, o, w in self.variation_space.flat_layout() if p == "pad.color")
        row[off : off + width] = self.DRAWN_PAD_DEFAULT
        return row

    def _as_rows(self, value):
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(value, np.float32), device=self.device)

    def _renderer_rows(self, rows):
        """Variation-layout rows -> renderer style rows (flame off; ignition is state, not appearance)."""
        rows = self._as_rows(rows).reshape(-1, self.variation_dim)
        source = {p: (o, w) for p, o, w in self.variation_space.flat_layout()}
        out = torch.zeros((len(rows), frozen.style_dim()), dtype=torch.float32, device=self.device)
        for path, (off, width) in frozen.style_layout().items():
            if path in source:
                o, w = source[path]
                out[:, off : off + width] = rows[:, o : o + w]
        return out

    def resolve_style(self, style=None, variation=None, batch=None):
        """Renderer appearance: an explicit style, else one set for rendering, else the state's.

        Supplied pose rows carry no ignition, so the state's appearance applies to them with the flame off.
        """
        overrides, style = _split_appearance(variation, style)
        if isinstance(style, StyleTensor):
            rows = style.values
        elif style is not None:
            rows = self._as_rows(style)
            rows = (
                rows.reshape(-1, frozen.style_dim())
                if rows.shape[-1] == frozen.style_dim()
                else self._renderer_rows(rows)
            )
        elif self._render_style is not None:
            rows = self._render_style
        else:
            # The live colors alone; exporting and validating the whole state is not needed for them.
            rows = self._style_of(self._simulator.rgba())
            rows[:, frozen.style_layout()["flame.ignited"][0]] = 0
        if overrides:
            rows = rows.clone()
            layout = frozen.style_layout()
            for path, value in overrides.items():
                if path not in self.variation_names:
                    raise KeyError(f"unknown Rocket Variation path {path!r}")
                if path in layout:
                    off, width = layout[path]
                    rows[:, off : off + width] = torch.as_tensor(value, dtype=rows.dtype, device=rows.device).reshape(
                        -1, width
                    )
        layout = frozen.PyFlytRocketLanding.make_variation_space(self).flat_layout()
        style = StyleTensor(rows, layout)
        return style if batch is None else style.for_batch(int(batch), owner=f"{type(self).__name__}.render")

    def set_style(self, style=None, **overrides):
        """Set the appearance used when rendering supplied rows; None returns to the state's."""
        if style is None and not overrides:
            self._render_style = None
            return self.to_render_state()["renderer", "style"]
        self._render_style = self.resolve_style(style, overrides or None, batch=self.B).values
        return self._render_style

    def _set_render_state(self, states, variation=None, style=None):
        Env._set_render_state(self, states)
        if style is not None or variation is not None:
            self._render_style = self.resolve_style(style, variation, batch=self._live_rows).values
        return self

    def _rows_unsupported(self):
        if self.device.type == "cpu":
            return "Rocket draws compact pose rows only on CUDA; on CPU it draws a full State or a RenderState"
        return None

    def _render_state_input(self, state, *, size=None, supersample=None, backend=None, chunk_size=None, out=None):
        self._check_open("render")
        from . import render as rocket_render

        backend = self.backend if backend is None else backend
        rocket_render.check_backend(backend, self.device)
        for value in (size, supersample, chunk_size):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("Render dimensions/chunk must be positive integers")
        if state is None or isinstance(state, TensorDict) and "current" in state.keys():
            frames = self._render_full(state, size, supersample, chunk_size, backend)
            if out is not None:
                if out.shape != frames.shape or out.dtype != frames.dtype or out.device != frames.device:
                    raise ValueError("Invalid Rocket render output buffer")
                out.copy_(frames)
                return out
            return frames
        b, complete, resolution, ss = self._render_dimensions(state, size, supersample)
        chunk = b if chunk_size is None else chunk_size
        frames = (
            rocket_render.render(self, state, resolution, ss, chunk_size, backend)
            if complete
            else torch.cat(
                [
                    self._render(
                        state["renderer", "pose"][i : i + chunk],
                        size=resolution,
                        supersample=ss,
                        style=state["renderer", "style"][i : i + chunk],
                    )
                    for i in range(0, b, chunk)
                ]
            )
        )
        if out is not None:
            if out.shape != frames.shape or out.dtype != frames.dtype or out.device != frames.device:
                raise ValueError("Invalid Rocket render output buffer")
            out.copy_(frames)
            return out
        return frames

    def _render_full(self, state, size, supersample, chunk_size, backend=None):
        """Draw full States; only the leaves the frame reads are checked."""
        from . import render as rocket_render

        state = self.get_state() if state is None else state
        leaves = rocket_render.host_leaves(state, rocket_render.STATE_READS, self.B)
        fields = {key[1]: value.numpy() for key, value in rocket_render.project(leaves).items()}
        args = (fields["source_pose"], fields["camera_pose"], fields["rgba"])
        size = self.size if size is None else size
        supersample = self.ss if supersample is None else supersample
        if self.device.type == "cpu":
            return rocket_render.render_cpu(self, *args, size, supersample, backend)
        return rocket_render.draw_gpu(self, *args, size, supersample, chunk_size)

    def _render_dimensions(self, state, size, supersample):
        """Check a RenderState; return its batch, whether it has explicit poses, and the frame size and supersample."""
        from . import render as rocket_render

        keys = {("renderer", k) for k in ["pose", "style", "size", "supersample"]}
        complete_keys = keys | {("renderer", k) for k in rocket_render.FIELDS}
        if (
            not isinstance(state, TensorDict)
            or state.ndim != 1
            or set(state.keys(True, True)) not in (keys, complete_keys)
        ):
            raise ValueError("Invalid Rocket RenderState")
        b = state.batch_size[0]
        if b < 1:
            raise ValueError("Rocket RenderState batch must be nonempty")
        complete = set(state.keys(True, True)) == complete_keys
        if not complete and self.device.type == "cpu":
            raise ValueError("Rocket CPU rendering requires explicit body and cached camera poses")
        if complete:
            for key, shape in rocket_render.FIELDS.items():
                value = state["renderer", key]
                if value.shape != (b, *shape) or value.dtype != torch.float64 or not torch.isfinite(value).all():
                    raise ValueError("Invalid Rocket RenderState leaf: " + key)
            if (state["renderer", "rgba"] < 0).any() or (state["renderer", "rgba"] > 1).any():
                raise ValueError("Rocket RenderState rgba must lie in [0, 1]")
        for key, shape, dtype in [
            ("pose", (b, 13), torch.float32),
            ("style", (b, 16), torch.float32),
            ("size", (b,), torch.int64),
            ("supersample", (b,), torch.int64),
        ]:
            value = state["renderer", key]
            if value.shape != shape or value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError("Invalid Rocket RenderState leaf: " + key)
        resolution = int(state["renderer", "size"][0]) if size is None else size
        ss = int(state["renderer", "supersample"][0]) if supersample is None else supersample
        if (
            resolution < 1
            or ss < 1
            or size is None
            and not (state["renderer", "size"] == resolution).all()
            or supersample is None
            and not (state["renderer", "supersample"] == ss).all()
        ):
            raise ValueError("RenderState requires uniform positive output dimensions")
        return b, complete, resolution, ss

    def _close(self):
        from . import render as rocket_render

        try:
            rocket_render.close(self)
        finally:
            self._simulator.close()


if "PyFlytRocketLanding" in REGISTRY:
    assert REGISTRY["PyFlytRocketLanding"] is frozen.PyFlytRocketLanding
    del REGISTRY["PyFlytRocketLanding"]
register(PyFlytRocketLanding)
