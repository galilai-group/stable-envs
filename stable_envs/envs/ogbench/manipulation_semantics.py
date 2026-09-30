"""CPU public semantics; frozen manipulation renderer stays in manipulation.py."""

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
    cpu_render_support = "CPU_BACKEND_MISSING"
    action_dim = 5
    action_dtype = torch.float32
    action_low, action_high = -1.0, 1.0

    def __init__(
        self, num_envs=1, size=None, supersample=None, device="cpu", backend="auto", seed=0, *, reward_task_id=None
    ):
        from ..._adapters.ogbench_manipulation import OGBManipulationAdapter

        device = Env.resolve_device(device)
        self._simulator = OGBManipulationAdapter(self.CELL, num_envs, device, seed, reward_task_id)
        u = self._simulator.envs[0]
        self._nq = u.model.nq
        self.state_dim = self._nq + 3 + (2 if self.CELL == "Scene" else 0)
        self.state_names = (
            tuple(f"qpos_{i}" for i in range(self._nq))
            + ("goal_x", "goal_y", "goal_z")
            + (("button_0", "button_1") if self.CELL == "Scene" else ())
        )
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
        from ..._adapters.ogbench_manipulation_compat import CUBE_SAMPLES, variation_tree

        tree = variation_tree(self._simulator.envs[0])
        if self.CELL == "Scene":
            for key in CUBE_SAMPLES:
                del tree["cube"].spaces[key.split(".")[1]]
        return tree

    def _sync_style(self):
        rows = []
        for env in self._simulator.envs:
            rows.append(
                np.concatenate(
                    [
                        env.variation_space._walk(name).value.reshape(-1)
                        for name, _, _ in self.variation_space.flat_layout()
                    ]
                )
            )
        self._style = StyleTensor(
            torch.as_tensor(np.stack(rows), dtype=torch.float32, device=self.device),
            self.variation_space.flat_layout(),
        )

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
        return torch.as_tensor(
            np.stack(
                [
                    np.concatenate(
                        [
                            e.variation_space._walk(k).init_value.reshape(-1)
                            for k, _, _ in self.variation_space.flat_layout()
                        ]
                    )
                    for e in self._simulator.envs
                ]
            ),
            dtype=torch.float64,
            device=self.device,
        )

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


from .manipulation import OGBCube as _FrozenCube
from .manipulation import OGBScene as _FrozenScene


class OGBCube(ManipulationSemantics, _FrozenCube):
    __doc__ = (
        "CPU simulator semantics; renderer acceptance pending. Historical VISUAL TIER renderer description follows:\n\n"
        + _FrozenCube.__doc__
    )


class OGBScene(ManipulationSemantics, _FrozenScene):
    __doc__ = (
        "CPU simulator semantics; renderer acceptance pending. Historical VISUAL TIER renderer description follows:\n\n"
        + _FrozenScene.__doc__
    )


CELL_CLASSES = {"OGBCube": OGBCube, "OGBScene": OGBScene}
