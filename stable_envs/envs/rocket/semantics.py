from __future__ import annotations

import copy
import functools

import numpy as np
import torch
from tensordict import TensorDict

from ..._core.style import StyleTensor
from ..._execution.protocol import StepPipeline, step_pipeline
from ...env import Env, _split_appearance
from ...registry import REGISTRY, register
from ...specs import CompositeSpec, Spec
from . import legacy as frozen
from .simulation.execution import make_rocket_adapter


class PyFlytRocketLanding(frozen.PyFlytRocketLanding):
    """Rocket CPU dynamics with replay-backed State and CPU/CUDA rendering.

    PyFlyt 0.29.0/PyBullet 3.2.7 run physics on CPU. Explicit body and cached camera
    poses drive TinyRenderer on CPU or the nvdiffrast path on CUDA.
    """

    requires_cuda = False
    execution_mode = "DELEGATED_CPU"
    cpu_render_support = "SUPPORTED_TINYRENDERER"
    action_dim = 7
    action_dtype = torch.float64
    action_low = (-1.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0)
    action_high = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)
    dynamics_variation_leaves = frozenset(
        {"environment.start_height_ratio", "environment.start_horizontal_offset", "environment.start_tilt"}
    )
    undrawn_variation_leaves = frozenset({"environment.sky_color"})
    #: The source loads its original white pad unless reset options ask for colour variations.
    DRAWN_PAD_DEFAULT = (255.0, 255.0, 255.0)
    _render_style = None

    def __init__(
        self, num_envs=1, size=None, supersample=None, device="cpu", backend="auto", seed=0, *, simulator_config=None
    ):
        device = Env.resolve_device(device)
        # CPU lanes draw their own step frames, at the size and supersample ``Env`` resolves.
        frame = (
            (int(size or self.default_size), int(1 if supersample is None else supersample))
            if torch.device(device).type == "cpu"
            else None
        )
        self._simulator = make_rocket_adapter(num_envs, device, seed, simulator_config, frame)
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

    def _step_pipeline(self) -> StepPipeline | None:
        return step_pipeline(self._simulator)

    def _render_live(self):
        if self.device.type != "cpu":
            return self.render()
        return torch.from_numpy(self._simulator.render_live())

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
        self._check_open("to_render_state")
        state = self.get_state() if state is None else state
        self._simulator.validate(state)
        return self._render_state_of(state, size, supersample)

    def _render_state_of(self, state, size, supersample):
        """The RenderState of a full State whose leaves have the recorded keys, shapes and dtypes."""
        style = self._style_of(state["render", "rgba"])
        result = TensorDict(
            {
                ("renderer", "pose"): state["current", "wrapper", "state"][:, :13].to(torch.float32).clone(),
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
        # projection cannot be passed to set_state or impersonate full State.
        return TensorDict({("renderer", "pose"): rows.clone()}, [len(rows)], device=self.device)

    def default_style(self):
        """The default Variation row with the pad colour the source actually draws by default."""
        row = np.array(super().default_style(), np.float32)
        off, width = next((o, w) for p, o, w in self.variation_space.flat_layout() if p == "pad.color")
        row[off : off + width] = self.DRAWN_PAD_DEFAULT
        return row

    def _as_rows(self, value):
        if torch.is_tensor(value):
            return value.to(device=self.device, dtype=torch.float32)
        return torch.as_tensor(np.asarray(value, np.float32), device=self.device)

    def _renderer_rows(self, rows):
        """Variation-layout rows -> renderer style rows (flame off; ignition is State, not appearance)."""
        rows = self._as_rows(rows).reshape(-1, self.variation_dim)
        source = {p: (o, w) for p, o, w in self.variation_space.flat_layout()}
        out = torch.zeros((len(rows), frozen.style_dim()), dtype=torch.float32, device=self.device)
        for path, (off, width) in frozen.style_layout().items():
            if path in source:
                o, w = source[path]
                out[:, off : off + width] = rows[:, o : o + w]
        return out

    def resolve_style(self, style=None, variation=None, batch=None):
        """Renderer appearance: an explicit style, else one set for rendering, else the State's.

        Supplied pose rows carry no ignition, so the State's appearance applies to them with the flame off.
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
            # The live colours alone; exporting and validating the whole State is not needed for them.
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
        """Set the appearance used when rendering supplied rows; None returns to the State's."""
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

    def render(self, state=None, *, size=None, supersample=None, backend=None, chunk_size=None, out=None):
        self._check_open("render")
        from . import render as rocket_render

        backend = self.backend if backend is None else backend
        allowed = ("auto", "tiny") if self.device.type == "cpu" else ("auto", "nvdiffrast")
        if backend not in allowed:
            raise ValueError("Rocket CPU rendering uses tiny; CUDA rendering uses nvdiffrast")
        for value in (size, supersample, chunk_size):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("Render dimensions/chunk must be positive integers")
        check = None
        if state is None or isinstance(state, TensorDict) and "current" in state.keys():
            full = self.get_state() if state is None else state
            self._simulator.check_fields(full)
            state = self._render_state_of(full, size, supersample)
            # The whole State is checked while the frames are drawn; no frame returns before it passes.
            check = functools.partial(self._simulator.validate, full)
        try:
            b, complete, resolution, ss = self._render_dimensions(state, size, supersample)
        except ValueError:
            # The State's own error comes first, as when ``to_render_state`` checks it.
            if check is not None:
                check()
            raise
        chunk = b if chunk_size is None else chunk_size
        frames = (
            rocket_render.render(self, state, resolution, ss, chunk_size, check)
            if complete
            else torch.cat(
                [
                    frozen.PyFlytRocketLanding._render(
                        self,
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
