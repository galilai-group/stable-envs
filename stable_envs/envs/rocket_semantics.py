from __future__ import annotations

import copy

import numpy as np
import torch
from tensordict import TensorDict

from .._adapters.rocket.adapter import RocketAdapter
from .._style import StyleTensor
from ..env import Env
from ..registry import REGISTRY, register
from ..specs import CompositeSpec, Spec
from . import rocket_landing as frozen


class PyFlytRocketLanding(frozen.PyFlytRocketLanding):
    """Rocket CPU dynamics with replay-backed State and CPU/CUDA rendering; VISUAL TIER.

    PyFlyt 0.29.0/PyBullet 3.2.7 run physics on CPU. Explicit body and cached camera
    poses drive TinyRenderer on CPU or the nvdiffrast path on CUDA."""

    requires_cuda = False
    execution_mode = "DELEGATED_CPU"
    cpu_render_support = "SUPPORTED_TINYRENDERER"
    action_dim = 7
    action_dtype = torch.float64
    action_low = (-1.0, -1.0, -1.0, 0.0, 0.0, -1.0, -1.0)
    action_high = (1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0)

    def __init__(
        self, num_envs=1, size=None, supersample=None, device="cpu", backend="auto", seed=0, *, simulator_config=None
    ):
        device = Env.resolve_device(device)
        self._simulator = RocketAdapter(num_envs, device, seed, simulator_config)
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
        return copy.deepcopy(self._simulator.envs[0].env.variation_space)

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
        self._check_open("reset")
        return self._simulator.reset(seed, options)

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)

    def _step(self, actions, *, active=None):
        self._check_open("step")
        return self._simulator.step(actions, active=active)

    def _variation_rows(self, attribute):
        rows = [
            np.concatenate(
                [
                    np.asarray(getattr(env.env.variation_space._walk(key), attribute)).reshape(-1)
                    for key, _, _ in self.variation_space.flat_layout()
                ]
            )
            for env in self._simulator.envs
        ]
        return torch.as_tensor(np.stack(rows), dtype=torch.float32, device=self.device)

    def get_variation(self):
        self._check_open("get_variation")
        return self._variation_rows("value")

    def sample_variation(self, batch=None, paths=None, rng=None):
        """Sample configurations with integer RGB leaves using the supplied or metadata RNG.

        This does not consume simulator reset RNGs. Use reset options to randomize
        the next episode through those streams."""
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
        self._check_open("set_variation")
        if isinstance(variation, dict):
            overrides = {**variation, **overrides}
            variation = None
        values = [{} for _ in range(self.B)]
        if variation is not None:
            rows = np.asarray(variation.detach().cpu() if torch.is_tensor(variation) else variation)
            if rows.shape == (self.variation_space.flat_dim,):
                rows = np.broadcast_to(rows, (self.B, self.variation_space.flat_dim))
            if rows.shape != (self.B, self.variation_space.flat_dim):
                raise ValueError("Invalid Rocket Variation shape")
            for key, off, width in self.variation_space.flat_layout():
                overrides[key] = rows[:, off : off + width].reshape(self.B, *self.variation_space._walk(key).shape)
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
        self._simulator.configure(values)
        return self._variation_rows("init_value")

    def to_render_state(self, state=None, *, size=None, supersample=None):
        self._check_open("to_render_state")
        state = self.get_state() if state is None else state
        self._simulator.validate(state)
        rgba = state["render", "rgba"]
        style = torch.zeros((self.B, 16), dtype=torch.float32, device=rgba.device)
        # Renderer style is alphabetical, not source/corpus order.
        layout = frozen.style_layout()
        for index, name in enumerate(frozen.CORPUS_STYLE_PATHS[:5]):
            offset, width = layout[name]
            style[:, offset : offset + width] = rgba[:, index, :3] * 255
        style[:, layout["flame.ignited"][0]] = torch.isclose(
            rgba[:, 5], torch.tensor(frozen.IGNITION_RGBA, device=rgba.device, dtype=rgba.dtype)
        ).all(-1)
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
        from . import rocket_render

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

    def resolve_style(self, style=None, variation=None, batch=None):
        if style is None:
            style = self.to_render_state()["renderer", "style"]
        if isinstance(style, StyleTensor):
            return style
        layout = frozen.PyFlytRocketLanding.make_variation_space(self).flat_layout()
        return StyleTensor(torch.as_tensor(style, dtype=torch.float32, device=self.device), layout)

    def render(self, state=None, *, size=None, supersample=None, backend=None, chunk_size=None, out=None):
        self._check_open("render")
        from . import rocket_render

        backend = self.backend if backend is None else backend
        allowed = ("auto", "tiny") if self.device.type == "cpu" else ("auto", "nvdiffrast")
        if backend not in allowed:
            raise ValueError("Rocket CPU rendering uses tiny; CUDA rendering uses nvdiffrast")
        for value in (size, supersample, chunk_size):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("Render dimensions/chunk must be positive integers")
        if state is None or isinstance(state, TensorDict) and "current" in state.keys():
            state = self.to_render_state(state, size=size, supersample=supersample)
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
        chunk = b if chunk_size is None else chunk_size
        frames = (
            rocket_render.render(self, state, resolution, ss, chunk_size)
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

    def _close(self):
        from . import rocket_render

        try:
            rocket_render.close(self)
        finally:
            self._simulator.close()


if "PyFlytRocketLanding" in REGISTRY:
    assert REGISTRY["PyFlytRocketLanding"] is frozen.PyFlytRocketLanding
    del REGISTRY["PyFlytRocketLanding"]
register(PyFlytRocketLanding)
