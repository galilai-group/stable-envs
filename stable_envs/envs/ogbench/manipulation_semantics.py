from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ..._style import StyleTensor
from ...env import Env
from ...specs import CompositeSpec, Spec


class ManipulationSemantics:
    requires_cuda = False
    execution_mode = "DELEGATED_CPU"
    cpu_render_support = "SUPPORTED_SOFTWARE_OPENGL"
    action_dim = 5
    action_dtype = torch.float32
    action_low, action_high = -1.0, 1.0

    def __init__(
        self, num_envs=1, size=None, supersample=None, device="cpu", backend="auto", seed=0, *, reward_task_id=None
    ):
        from ..._adapters.ogbench_execution import make_manipulation_adapter
        from ..._adapters.ogbench_manipulation import OGBManipulationAdapter

        device = Env.resolve_device(device)
        self._simulator = make_manipulation_adapter(
            self.CELL, num_envs, device, seed, reward_task_id, OGBManipulationAdapter
        )
        self._nq = self._simulator.nq
        self.state_dim = self._nq + 3 + (2 if self.CELL == "Scene" else 0)
        from .manipulation import render_state_names

        self.state_names = render_state_names(self._nq, 1, 2 if self.CELL == "Scene" else 0)
        # Env uses its seed only for metadata/render-state sampling; physical RNGs are adapter-owned.
        Env.__init__(
            self,
            num_envs,
            size,
            1 if supersample is None else supersample,
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
        return self._simulator.step(actions, active=active)

    def _advance_without_snapshot(self, actions, *, active=None):
        self._check_open("step")
        return self._simulator.step(actions, active=active, capture_state=False)

    def _step_pipeline(self):
        owner = self._simulator
        return owner if hasattr(owner, "submit_step") else None

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
        # Persistent reset anchors take effect on reset; current model/style remains unchanged.
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
        from ..._adapters import ogbench_render_state as pixels

        self._check_open("to_render_state")
        state = self.get_state() if state is None else state
        self._simulator.validate(state)
        return pixels.project(
            self.CELL, state, self.size if size is None else size, self.ss if supersample is None else supersample
        )

    def render(self, state=None, *, size=None, supersample=None, backend=None, chunk_size=None, out=None):
        from ..._adapters import ogbench_render_state as pixels
        from . import manipulation_renderer

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

    def _render(self, state=None, **kwargs):
        return self.render(state, **kwargs)

    def sample_states(self, batch, rng=None):
        self._check_open("sample_states")
        raise NotImplementedError("Historical Cube/Scene dataset sampling is outside the CPU compatibility checkpoint")

    def _close(self):
        self._simulator.close()
        super()._close()


from .manipulation import OGBCube as _CubeRenderer
from .manipulation import OGBScene as _SceneRenderer


class OGBCube(ManipulationSemantics, _CubeRenderer):
    """Cube simulation with restorable State and CPU OpenGL or GPU MJWarp rendering.

    GPU rendering is visual tier: lighting and transparency can differ from MuJoCo.
    """


class OGBScene(ManipulationSemantics, _SceneRenderer):
    """Scene simulation with restorable State and CPU OpenGL or GPU MJWarp rendering.

    GPU rendering is visual tier: lighting, transparency, and button textures can differ from MuJoCo.
    """


CELL_CLASSES = {"OGBCube": OGBCube, "OGBScene": OGBScene}
