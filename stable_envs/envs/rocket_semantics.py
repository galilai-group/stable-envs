"""Public Rocket replay semantics over the unchanged frozen VISUAL tier renderer."""

from __future__ import annotations

import copy

import numpy as np
import torch
from tensordict import TensorDict

from .._adapters.rocket import RocketAdapter
from .._style import StyleTensor
from ..env import Env
from ..registry import REGISTRY, register
from ..specs import CompositeSpec, Spec
from . import rocket_landing as frozen


class PyFlytRocketLanding(frozen.PyFlytRocketLanding):
    """Official Rocket dynamics with bounded replay State; frozen renderer, VISUAL TIER.

    CPU dynamics use PyFlyt 0.29.0/PyBullet 3.2.7. GPU pixels retain the frozen
    renderer's measured residual; a CPU rendering request is unsupported.
    """

    requires_cuda = False
    execution_mode = "DELEGATED_CPU"
    cpu_render_support = "CPU_BACKEND_MISSING"
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

    def step(self, actions):
        self._check_open("step")
        return self._simulator.step(actions)

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
        return self.get_state()

    def to_render_state(self, state=None, *, size=None, supersample=None):
        self._check_open("to_render_state")
        state = self.get_state() if state is None else state
        self._simulator.validate(state)
        rgba = state["render", "rgba"]
        style = torch.zeros((self.B, 16), dtype=torch.float32, device=rgba.device)
        # The frozen renderer's layout is alphabetical, not source/corpus order.
        layout = frozen.style_layout()
        for index, name in enumerate(frozen.CORPUS_STYLE_PATHS[:5]):
            offset, width = layout[name]
            style[:, offset : offset + width] = rgba[:, index, :3] * 255
        style[:, layout["flame.ignited"][0]] = torch.isclose(
            rgba[:, 5], torch.tensor(frozen.IGNITION_RGBA, device=rgba.device, dtype=rgba.dtype)
        ).all(-1)
        return TensorDict(
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
        if self.device.type != "cuda":
            raise NotImplementedError("Rocket has no CPU textured-triangle renderer")
        if backend not in (None, "auto", "nvdiffrast"):
            raise ValueError("Rocket renderer requires nvdiffrast")
        for value in (size, supersample, chunk_size):
            if value is not None and (type(value) is not int or value < 1):
                raise ValueError("Render dimensions/chunk must be positive integers")
        if state is None or isinstance(state, TensorDict) and "current" in state.keys():
            state = self.to_render_state(state, size=size, supersample=supersample)
        keys = {("renderer", k) for k in ["pose", "style", "size", "supersample"]}
        if not isinstance(state, TensorDict) or state.ndim != 1 or set(state.keys(True, True)) != keys:
            raise ValueError("Invalid Rocket RenderState")
        b = state.batch_size[0]
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
        frames = torch.cat(
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
        if out is not None:
            if out.shape != frames.shape or out.dtype != frames.dtype or out.device != frames.device:
                raise ValueError("Invalid Rocket render output buffer")
            out.copy_(frames)
            return out
        return frames

    def _close(self):
        self._simulator.close()


if "PyFlytRocketLanding" in REGISTRY:
    assert REGISTRY["PyFlytRocketLanding"] is frozen.PyFlytRocketLanding
    del REGISTRY["PyFlytRocketLanding"]
register(PyFlytRocketLanding)
