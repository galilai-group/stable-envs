from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ... import tuning
from ..._core.style import StyleTensor, rows_are_equal
from ..._execution.protocol import StepPipeline, step_pipeline
from ...env import Env, _split_appearance
from ...specs import CompositeSpec, Spec


class _OwnStates:
    """The simulator's step pipeline; each State it returns is recorded as the env's own."""

    def __init__(self, env, pipeline):
        self._env, self._pipeline = env, pipeline

    def submit_step(self, actions, *, active=None, capture_state=True):
        self._pipeline.submit_step(actions, active=active, capture_state=capture_state)

    def receive_step(self):
        result = self._pipeline.receive_step()
        self._env._own_state = result.get("state", None)
        return result

    def abort_step(self):
        self._pipeline.abort_step()


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
    _own_state = None

    def __init__(
        self,
        num_envs=1,
        size=None,
        supersample=None,
        device="cpu",
        backend="auto",
        seed=0,
        *,
        reward_task_id=None,
        simulation_backend="default",
        mjwarp_min_envs=None,
    ):
        """Select physics at construction.

        ``simulation_backend="default"`` steps CUDA batches on MuJoCo Warp (float32) and CPU batches on native
        ``mujoco.batch``, equal to OGBench bit for bit. ``"auto"`` steps CUDA batches of ``mjwarp_min_envs`` or
        more on MuJoCo Warp and every other batch on native ``mujoco.batch``. ``"native"`` and ``"mjwarp"``
        select explicitly.
        """
        from ..._execution.mjbatch import resolve_backend
        from .simulation.execution import MJWARP_MIN_ENVS, make_manipulation_adapter

        device = Env.resolve_device(device)
        mjwarp_min_envs = MJWARP_MIN_ENVS if mjwarp_min_envs is None else mjwarp_min_envs
        self.simulation_backend = resolve_backend(num_envs, device, simulation_backend, mjwarp_min_envs)
        self._simulator = make_manipulation_adapter(
            self.CELL, num_envs, device, seed, reward_task_id, self.simulation_backend
        )
        self.execution_mode = self._simulator.execution_mode
        self._nq = self._simulator.nq
        self.state_dim = self._nq + 3 + (2 if self.CELL == "Scene" else 0)
        from .manipulation import render_state_names

        self.state_names = render_state_names(self._nq, 1, 2 if self.CELL == "Scene" else 0)
        if supersample is None:
            supersample = 1
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
        state = self._own_state = self._simulator.get_state()
        return state

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)
        self._sync_style()

    def _step(self, actions, *, active=None):
        self._check_open("step")
        self._rows_installed = False
        result = self._simulator.step(actions, active=active)
        self._own_state = result.get("state", None)
        return result

    def _advance_without_snapshot(self, actions, *, active=None):
        self._check_open("step")
        self._rows_installed = False
        return self._simulator.step(actions, active=active, capture_state=False)

    def _step_pipeline(self) -> StepPipeline | None:
        pipeline = step_pipeline(self._simulator)
        return None if pipeline is None else _OwnStates(self, pipeline)

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
        return self._project(state, size, supersample)

    def _project(self, state, size, supersample, host=False):
        from .simulation import render_state as pixels

        self._check_open("to_render_state")
        # The live State comes from this env's own simulator, so only a supplied State is checked.
        trusted = state is None
        size = self.size if size is None else size
        supersample = self.ss if supersample is None else supersample
        if trusted:
            state = self.get_state()
        elif state is not self.__dict__.get("_own_state"):
            # The frame reads only the projected leaves; set_state checks the leaves that restore simulation.
            pixels.require_leaves(self.CELL, state)
        if host:
            state = state.select(*pixels.projected_keys(self.CELL)).cpu()
        return pixels.project(self.CELL, state, size, supersample, check=not trusted)

    def _render_state_input(self, state, *, size=None, supersample=None, backend=None, chunk_size=None, out=None):
        return self._render_frames(
            state, False, size=size, supersample=supersample, backend=backend, chunk_size=chunk_size, out=out
        )

    def _render_frames(self, state, checked, *, size=None, supersample=None, backend=None, chunk_size=None, out=None):
        from . import manipulation_renderer
        from .simulation import render_state as pixels

        self._check_open("render")
        if any(value is not None and type(value) is not int for value in (size, supersample)):
            raise ValueError("Render size and supersample must be integers")
        # A projected State is validated once, by ``project``.
        validated = checked or state is None or isinstance(state, TensorDict) and "physics" in state.keys()
        if validated and not checked:
            sampling = self.ss if supersample is None else supersample
            host = manipulation_renderer.draws_on_host(backend or self.backend, self.device, sampling, self.CELL)
            if state is None and host and hasattr(self._simulator, "render_columns"):
                chosen = manipulation_renderer.host_backend(backend or self.backend, self.device)
                return self._draw_live(size, sampling, chunk_size, out, chosen)
            # OpenGL reads host inputs, so only the projected leaves leave the device.
            state = self._project(state, size, supersample, host)
        elif size is not None or supersample is not None:
            pixels.validate(self.CELL, state)
            state = state.clone()
            if size is not None:
                state["renderer", "size"].fill_(size)
            if supersample is not None:
                state["renderer", "supersample"].fill_(supersample)
        tuning.on_render("ogb-manipulation", int(state["renderer", "size"][0, 0]), len(state), self)
        result = manipulation_renderer.render(
            self.CELL, state, self.device, self.backend if backend is None else backend, chunk_size, validated
        )
        return self._into(result, out)

    def _render_live(self):
        """The stepping hook: frames of the live lanes."""
        return self.render()

    def _draw_live(self, size, supersample, chunk_size, out, chosen="opengl"):
        """Draw the live lanes from simulator columns straight into OpenGL, without a RenderState TensorDict."""
        from . import manipulation_opengl
        from .simulation import render_state as pixels

        if chunk_size is not None and (type(chunk_size) is not int or chunk_size < 1):
            raise ValueError("chunk_size must be a positive integer")
        size = self.size if size is None else size
        if not (1 <= size <= 4096 and 1 <= supersample <= 4):
            raise ValueError("RenderState requires one valid square resolution/supersampling per batch")
        columns = pixels.state_columns(self.CELL, self._simulator.render_columns(), size, supersample)
        tuning.on_render("ogb-manipulation", size, self.B, self)
        if chosen == "numba":
            from . import manipulation_numba

            return self._into(manipulation_numba.frames(self.CELL, columns, self.device), out)
        if chosen == "raster":
            from . import manipulation_raster

            return self._into(manipulation_raster.frames(self.CELL, columns, self.device, chunk_size), out)
        return self._into(manipulation_opengl.frames(self.CELL, columns, self.device, chosen == "opengl_fast"), out)

    def _into(self, result, out):
        if out is None:
            return result
        if (
            out.shape != result.shape
            or out.dtype != result.dtype
            or out.device != result.device
            or not out.is_contiguous()
        ):
            raise ValueError("out must match the contiguous uint8 frame batch on the renderer device")
        out.copy_(result)
        return out

    def _set_render_state(self, states, variation=None, style=None):
        Env._set_render_state(self, states)
        self._render_style = (
            None if style is None and not variation else self.resolve_style(style, variation, self._live_rows)
        )
        self._rows_installed = True
        return self

    def _appearance(self, style, b, size, supersample, device):
        """Renderer header and appearance columns of a RenderState batch drawn with ``style``."""
        from .simulation import render_state as pixels

        values = pixels.header(self.CELL, b, size, supersample, device)
        schema = pixels.fields(self.CELL)
        host = style.values.detach().cpu().numpy()
        for name, (off, width) in style.layout.items():
            key = ("appearance", name)
            if key in schema:
                # Shortest float32 decimals recover the float64 values the State path compiles,
                # so both paths share one compiled model.
                exact = np.array([float(str(v)) for v in host[:, off : off + width].ravel()], np.float64)
                values[key] = torch.as_tensor(exact, device=device).reshape(b, *schema[key][0])
        return values

    def _rows_to_render_state(self, rows, style, size, supersample, device=None, appearance=None):
        from .simulation import render_state as pixels

        family = self.CELL
        device = self.device if device is None else torch.device(device)
        b, nq = len(rows), self._nq
        rows = rows.to(torch.float64)
        values = dict(self._appearance(style, b, size, supersample, device) if appearance is None else appearance)
        qpos = rows[:, :nq].clone()
        # MuJoCo normalises free-joint quaternions (identity below its minimum norm) before drawing.
        quat = qpos[:, pixels.OBJECT_QPOS_ADR + 3 : pixels.OBJECT_QPOS_ADR + 7]
        norm = torch.linalg.vector_norm(quat, dim=-1, keepdim=True)
        identity = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=rows.device)
        quat.copy_(torch.where(norm < 1e-15, identity, quat / norm.clamp_min(1e-15)))
        # Rounding happens on the rows' device, so a host State holds the same bits.
        rows = torch.cat([qpos, rows[:, nq:]], dim=1).to(device)
        values["pose", "qpos"] = rows[:, :nq]
        values["pose", "mocap_pos"] = rows[:, nq : nq + 3]
        values["pose", "mocap_quat"] = torch.tensor([1.0, 0.0, 0.0, 0.0], dtype=torch.float64, device=device)
        values["pose", "mocap_quat"] = values["pose", "mocap_quat"].expand(b, 4).clone()
        if family == "Scene":
            values["pose", "buttons"] = rows[:, nq + 3 :].round().clamp(0, 1).to(torch.int64)
        return TensorDict(values, [b], device=device)

    def _stored_appearance(self, b, size, supersample, device):
        """Appearance columns of the stored style, cached until that style changes."""
        base = self._style if self._render_style is None else self._render_style
        key = (b, size, supersample, str(device))
        hit = self.__dict__.get("_appearance_cache")
        if hit is not None and hit[0] is base and hit[1] == base.values._version and hit[2] == key:
            return hit[3]
        appearance = self._appearance(self.resolve_style(batch=b), b, size, supersample, device)
        self.__dict__["_appearance_cache"] = (base, base.values._version, key, appearance)
        return appearance

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
        size = int(self.size if size is None else size)
        supersample = int(self.ss if supersample is None else supersample)
        from .manipulation_renderer import draws_on_host

        # OpenGL reads host inputs, so its RenderState is built on the host.
        device = (
            torch.device("cpu")
            if draws_on_host(kwargs.get("backend") or self.backend, self.device, supersample, self.CELL)
            else self.device
        )
        if style is None and not variation:
            appearance = self._stored_appearance(len(rows), size, supersample, device)
            state = self._rows_to_render_state(rows, None, size, supersample, device, appearance)
            # The env's own appearance is valid, so finite rows and a valid frame size need no full check.
            checked = self._render_style is None and 1 <= size <= 4096 and 1 <= supersample <= 4
            if states is None or states is not self._checked_rows:
                checked = checked and bool(torch.isfinite(state["pose", "qpos"]).all())
                checked = checked and bool(torch.isfinite(state["pose", "mocap_pos"]).all())
        else:
            sty = self.resolve_style(style, variation, batch=len(rows))
            state = self._rows_to_render_state(rows, sty, size, supersample, device)
            checked = False
        frames = self._render_frames(state, checked, out=out, **kwargs)
        return frames.reshape(*lead, *frames.shape[1:])

    def render_semantic(self, states):
        """Segmentation and planar depth of compact render rows, drawn by ``backend="raster"``.

        Returns ``(B, size, size, 2)`` ``(objid, objtype)`` pairs in the model of ``reference_env`` (-1 for
        background) and ``(B, size, size)`` depth along the optical axis, top row first.
        """
        from . import manipulation_raster

        rows = torch.as_tensor(np.ascontiguousarray(states) if isinstance(states, np.ndarray) else states)
        rows = rows.detach().to("cpu", torch.float32).reshape(-1, self.state_dim)
        cpu = torch.device("cpu")
        appearance = self._stored_appearance(len(rows), self.size, 1, cpu)
        state = self._rows_to_render_state(rows, None, self.size, 1, cpu, appearance)
        columns = {key: value.numpy() for key, value in state.items(True, True)}
        return manipulation_raster.semantic(self.CELL, columns, type(self))

    def sample_states(self, batch, rng=None):
        self._check_open("sample_states")
        raise NotImplementedError("Cube/Scene dataset sampling is not supported")

    def _close(self):
        self._simulator.close()
        super()._close()


from .manipulation import OGBCube as _CubeRenderer
from .manipulation import OGBScene as _SceneRenderer

# Per-frame MSE of default OpenGL frames against a second stable-envs OpenGL instance with the same
# compiled model and appearance. tests/fixtures/source_mse.json holds the CUDA measurement.
_OPENGL_MSE = {"Cube": (0.0, 0.0, 0.0), "Scene": (0.0, 0.0, 0.0)}
_OPENGL_NOTE = (
    "At supersample 1, visual_fidelity is an implementation-consistency measurement against another stable-envs "
    "MuJoCo OpenGL instance using the same compiled model and appearance; it is not an upstream-default-appearance "
    "comparison. With supersample 2 or more, MJWarp draws CUDA frames, and its lighting and transparency can "
    "differ from the same-style stable-envs OpenGL reference. On CPU the default backend is opengl_fast: MuJoCo "
    "OpenGL with simplified arm meshes and a 2048 shadow map. Its frames can differ from backend='opengl' "
    "(mean frame MSE about 1.2 for Cube and 3.1 for Scene); visual_fidelity holds the backend='opengl' values."
)


def _opengl_fidelity(renderer):
    cell = renderer.CELL
    measured = dict(zip(("mse_mean", "mse_p95", "mse_max"), _OPENGL_MSE[cell]))
    return {
        **renderer.visual_fidelity,
        **measured,
        "measured_at": (
            "mujoco 3.17.0 / ogbench 1.2.1; backend='opengl', supersample 1; per-frame MSE over pixels "
            "and channels on the 0-255 scale, over the 64 render rows of sample_render_rows(default_rng(20260811)); "
            "candidate and reference are stable-envs MuJoCo OpenGL instances with the same compiled appearance. "
            "Measured on CPU Mesa llvmpipe (LLVM 20.1.8, torch 2.15.0+cpu) and CUDA NVIDIA H200 / driver "
            "580.173.02 / MUJOCO_GL=egl (torch 2.15.0+cu132); tests/fixtures/source_mse.json retains the CUDA "
            "measurement."
        ),
    }


class OGBCube(ManipulationSemantics, _CubeRenderer):
    """Cube simulation with restorable State and MuJoCo OpenGL rendering on CPU and GPU.

    On CPU the default backend, ``opengl_fast``, draws simplified arm meshes; ``backend="numba"`` draws with a
    Numba rasterizer that follows MuJoCo OpenGL; ``backend="opengl"`` draws the source's frames. With ``supersample`` 2 or more on the GPU, MJWarp draws;
    its lighting and transparency can differ from MuJoCo.
    """

    reference_id = "ogbench/visual-cube-single-v0"
    visual_fidelity = _opengl_fidelity(_CubeRenderer)
    visual_note = _OPENGL_NOTE


class OGBScene(ManipulationSemantics, _SceneRenderer):
    """Scene simulation with restorable State and MuJoCo OpenGL rendering on CPU and GPU.

    On CPU the default backend, ``opengl_fast``, draws simplified arm meshes; ``backend="numba"`` draws with a
    Numba rasterizer that follows MuJoCo OpenGL; ``backend="opengl"`` draws the source's frames. With ``supersample`` 2 or more on the GPU, MJWarp draws;
    its lighting, transparency, and button textures can differ from MuJoCo.
    """

    reference_id = "ogbench/visual-scene-v0"
    visual_fidelity = _opengl_fidelity(_SceneRenderer)
    visual_note = _OPENGL_NOTE


CELL_CLASSES = {"OGBCube": OGBCube, "OGBScene": OGBScene}
