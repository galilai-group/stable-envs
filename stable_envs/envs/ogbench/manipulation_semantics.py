from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ... import tuning
from ..._core.style import StyleTensor, rows_are_equal
from ..._execution.protocol import StepPipeline, step_pipeline
from ...env import Env, _split_appearance
from ...specs import CompositeSpec, Spec


class ManipulationSemantics:
    requires_cuda = False
    execution_mode = "DELEGATED_CPU"
    cpu_render_support = "SUPPORTED_SOFTWARE_OPENGL"
    action_dim = 5
    action_dtype = torch.float32
    action_low, action_high = -1.0, 1.0
    undrawn_variation_leaves = frozenset(
        {"agent.ee_start_position", "cube.goal_position", "cube.goal_yaw", "cube.start_position", "cube.start_yaw"}
    )
    _render_style = None
    _rows_installed = False

    def __init__(
        self, num_envs=1, size=None, supersample=None, device="cpu", backend="auto", seed=0, *, reward_task_id=None
    ):
        from .simulation.execution import make_manipulation_adapter
        from .simulation.manipulation import OGBManipulationAdapter

        device = Env.resolve_device(device)
        self._simulator = make_manipulation_adapter(
            self.CELL, num_envs, device, seed, reward_task_id, OGBManipulationAdapter
        )
        self._nq = self._simulator.nq
        self.state_dim = self._nq + 3 + (2 if self.CELL == "Scene" else 0)
        from .manipulation import render_state_names

        self.state_names = render_state_names(self._nq, 1, 2 if self.CELL == "Scene" else 0)
        if supersample is None:
            supersample = 2 if torch.device(device).type == "cuda" and backend == "auto" else 1
        # Env uses its seed only for metadata/render-state sampling; physical RNGs are adapter-owned.
        Env.__init__(
            self,
            num_envs,
            size,
            supersample,
            device,
            backend,
            0 if seed is None else seed,
        )
        self._sync_style()

    def make_variation_space(self):
        return self._simulator.make_variation_space()

    def _sync_style(self):
        layout = self.variation_space.flat_layout()
        self._style = StyleTensor(self._simulator.variation_values(layout).to(torch.float32), layout)
        self._render_style = None
        self._rows_installed = False

    def _drawn_columns(self):
        return [
            off + i
            for name, off, width in self.variation_space.flat_layout()
            if name not in self.undrawn_variation_leaves
            for i in range(width)
        ]

    def resolve_style(self, style=None, variation=None, batch=None):
        B = int(self.B if batch is None else batch)
        variation, style = _split_appearance(variation, style)
        if style is None:
            style = (self._style if self._render_style is None else self._render_style).values
            # Lanes differing only in reset-time leaves share one appearance.
            if len(style) != B and rows_are_equal(style[:, self._drawn_columns()]):
                style = style[:1]
        return StyleTensor.from_space(self.variation_space, B, device=self.device, values=style, overrides=variation)

    def _layout(self):
        offsets = {
            ("physics", "qpos"): 0,
            ("physics", "mocap_pos"): self._nq,
            ("scene", "cur_button_states"): self._nq + 3,
        }
        return tuple(
            (key, int(np.prod(shape)), offsets.get(key)) for key, (shape, _) in self._simulator.fields.items()
        )

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
        result = self._simulator.reset(seed, options)
        self._sync_style()
        return result

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)
        self._sync_style()

    def _step(self, actions, *, active=None):
        self._check_open("step")
        self._rows_installed = False
        return self._simulator.step(actions, active=active)

    def _advance_without_snapshot(self, actions, *, active=None):
        self._check_open("step")
        self._rows_installed = False
        return self._simulator.step(actions, active=active, capture_state=False)

    def _step_pipeline(self) -> StepPipeline | None:
        return step_pipeline(self._simulator)

    def set_variation(self, variation=None, **overrides):
        self._check_open("set_variation")
        if isinstance(variation, dict):
            overrides = {**variation, **overrides}
            variation = None
        values = [{} for _ in range(self.B)]
        if variation is not None:
            rows = np.asarray(variation.detach().cpu() if torch.is_tensor(variation) else variation, np.float64)
            if rows.shape == (self.variation_space.flat_dim,):
                rows = np.broadcast_to(rows, (self.B, self.variation_space.flat_dim))
            if rows.shape != (self.B, self.variation_space.flat_dim):
                raise ValueError("Variation must have shape [B, variation_dim] or [variation_dim]")
            for key, off, width in self.variation_space.flat_layout():
                shape = self.variation_space._walk(key).shape
                for i in range(self.B):
                    values[i][key] = rows[i, off : off + width].reshape(shape).copy()
        for key, value in overrides.items():
            if key not in self.variation_space.names():
                raise ValueError(f"Unsupported registered Variation: {key}")
            node = self.variation_space._walk(key)
            value = np.asarray(value.detach().cpu() if torch.is_tensor(value) else value, np.float64)
            if value.shape == node.shape:
                value = np.broadcast_to(value, (self.B, *node.shape))
            if value.shape != (self.B, *node.shape):
                raise ValueError(f"Invalid batched Variation shape: {key}")
            for i in range(self.B):
                values[i][key] = value[i].copy()
        self._simulator.configure(values)
        self._sync_style()
        return self._simulator.variation_values(self.variation_space.flat_layout(), anchor=True)

    def _pack_render_state(self, state):
        columns = [state["physics", "qpos"], state["physics", "mocap_pos"]]
        if self.CELL == "Scene":
            columns.append(state["scene", "cur_button_states"])
        return torch.cat(columns, dim=-1).to(device=self.device, dtype=torch.float32)

    def _unpack_render_state(self, rows):
        rows = torch.as_tensor(rows, device=self.device).reshape(-1, self.state_dim)
        values = {
            ("physics", "qpos"): rows[:, : self._nq].clone(),
            ("physics", "mocap_pos"): rows[:, self._nq : self._nq + 3].clone(),
        }
        if self.CELL == "Scene":
            values["scene", "cur_button_states"] = rows[:, -2:].clone()
        return TensorDict(values, batch_size=[len(rows)], device=self.device)

    def to_render_state(self, state=None, *, size=None, supersample=None):
        from .simulation import render_state as pixels

        self._check_open("to_render_state")
        state = self.get_state() if state is None else state
        self._simulator.validate(state)
        return pixels.project(
            self.CELL, state, self.size if size is None else size, self.ss if supersample is None else supersample
        )

    def render(self, state=None, *, size=None, supersample=None, backend=None, chunk_size=None, out=None):
        from . import manipulation_renderer
        from .simulation import render_state as pixels

        self._check_open("render")
        if any(value is not None and type(value) is not int for value in (size, supersample)):
            raise ValueError("Render size and supersample must be integers")
        if state is None or isinstance(state, TensorDict) and "physics" in state.keys():
            state = self.to_render_state(state, size=size, supersample=supersample)
        elif size is not None or supersample is not None:
            pixels.validate(self.CELL, state)
            state = state.clone()
            if size is not None:
                state["renderer", "size"].fill_(size)
            if supersample is not None:
                state["renderer", "supersample"].fill_(supersample)
        tuning.on_render("ogb-manipulation", int(state["renderer", "size"][0, 0]), len(state), self)
        result = manipulation_renderer.render(
            self.CELL, state, self.device, self.backend if backend is None else backend, chunk_size
        )
        if out is not None:
            if (
                out.shape != result.shape
                or out.dtype != result.dtype
                or out.device != result.device
                or not out.is_contiguous()
            ):
                raise ValueError("out must match the contiguous uint8 frame batch on the renderer device")
            out.copy_(result)
            return out
        return result

    def _set_render_state(self, states, variation=None, style=None):
        Env._set_render_state(self, states)
        self._render_style = (
            None if style is None and not variation else self.resolve_style(style, variation, self._live_rows)
        )
        self._rows_installed = True
        return self

    def _rows_to_render_state(self, rows, style, size, supersample):
        from .simulation import render_state as pixels

        family = self.CELL
        b, nq = len(rows), self._nq
        rows = rows.to(torch.float64)
        values = pixels.header(family, b, size, supersample, self.device)
        qpos = rows[:, :nq].clone()
        # MuJoCo normalises free-joint quaternions (identity below its minimum norm) before drawing.
        quat = qpos[:, pixels.OBJECT_QPOS_ADR + 3 : pixels.OBJECT_QPOS_ADR + 7]
        norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=self.device)
        quat.copy_(torch.where(norm < 1e-15, identity, quat / norm.clamp_min(1e-15)))
        values["pose", "qpos"] = qpos
        values["pose", "mocap_pos"] = rows[:, nq : nq + 3].clone()
        values["pose", "mocap_quat"] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=self.device)
        values["pose", "mocap_quat"] = values["pose", "mocap_quat"].expand(b, 4).clone()
        if family == "Scene":
            values["pose", "buttons"] = rows[:, nq + 3 :].round().clamp(0, 1).to(torch.int64)
        schema = pixels.fields(family)
        host = style.values.detach().cpu().numpy()
        for name, (off, width) in style.layout.items():
            key = ("appearance", name)
            if key in schema:
                # Shortest float32 decimals recover the float64 values the State path compiles,
                # so both paths share one compiled model.
                exact = np.array([float(str(v)) for v in host[:, off : off + width].ravel()], np.float64)
                values[key] = torch.as_tensor(exact, device=self.device).reshape(b, *schema[key][0])
        return TensorDict(values, [b], device=self.device)

    @torch.no_grad()
    def _render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None, **kwargs):
        if isinstance(states, TensorDict) or (
            states is None and not self._rows_installed and style is None and not variation
        ):
            return self.render(states, size=size, supersample=supersample, out=out, **kwargs)
        if states is None:
            rows = self._get_render_state() if self._rows_installed else self._pack_render_state(self.get_state())
        else:
            rows = torch.as_tensor(
                np.ascontiguousarray(states) if isinstance(states, np.ndarray) else states, device=self.device
            )
        lead = tuple(rows.shape[:-1])
        rows = rows.reshape(-1, self.state_dim)
        sty = self.resolve_style(style, variation, batch=len(rows))
        size = int(self.size if size is None else size)
        supersample = int(self.ss if supersample is None else supersample)
        frames = self.render(self._rows_to_render_state(rows, sty, size, supersample), out=out, **kwargs)
        return frames.reshape(*lead, *frames.shape[1:])

    @classmethod
    def reference_env(cls, size=None):
        """MuJoCo's own OpenGL renderer drawing the same compiled Cube/Scene model."""
        return cls(num_envs=1, size=size, device="cuda", backend="opengl")

    @classmethod
    def reference_render(cls, env, states):
        return env._render(states).cpu().numpy()

    def sample_states(self, batch, rng=None):
        self._check_open("sample_states")
        raise NotImplementedError("Cube/Scene dataset sampling is not supported")

    def _close(self):
        self._simulator.close()
        super()._close()


from .manipulation import OGBCube as _CubeRenderer
from .manipulation import OGBScene as _SceneRenderer


class OGBCube(ManipulationSemantics, _CubeRenderer):
    """Cube simulation with restorable State and CPU OpenGL or GPU MJWarp rendering.

    On the GPU, lighting and transparency can differ from MuJoCo.
    """


class OGBScene(ManipulationSemantics, _SceneRenderer):
    """Scene simulation with restorable State and CPU OpenGL or GPU MJWarp rendering.

    On the GPU, lighting, transparency, and button textures can differ from MuJoCo.
    """


CELL_CLASSES = {"OGBCube": OGBCube, "OGBScene": OGBScene}
