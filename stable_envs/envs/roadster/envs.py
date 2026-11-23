"""The five Roadster planning tasks: CPU MuJoCo simulation with restorable state and the world-model camera.

Each class fixes one task of ``source.RoadsterPlanEnv`` (the simulator, autopilot, reward, success and
horizon). The 4-D action ``[a_fwd, a_lat, a_up, yaw_rate]`` in [-1, 1] runs at
10 Hz. Wind and rain are physics variation leaves (``weather.wind`` in m/s, ``weather.rain`` in [0, 1]) that
reach the source at reset, as it sets the weather per episode; state records them under ``physics.weather``.
Physics steps every environment in one batch: native ``mujoco.batch`` by default, or MuJoCo Warp on CUDA
(``simulation_backend="mjwarp"``). Frames come from the ``wm`` camera (follows the car with a fixed world
orientation), 96 px by default, drawn by MuJoCo's OpenGL renderer on CPU; the exhaust smoke is drawn from the
history in state.
"""

from __future__ import annotations

import math
from collections.abc import Mapping

import torch
from tensordict import TensorDict

try:
    import mujoco  # noqa: F401
    import PIL  # noqa: F401
except ImportError as e:  # registered only with the extra, so import_errors() names it
    raise ImportError("Roadster needs the roadster extra: pip install 'stable-envs[roadster]'") from e

from ..._core.state_schema import get_field, schema_of
from ..._execution.protocol import StepPipeline, step_pipeline
from ...env import Env
from ...registry import register
from ...spaces import Box, Dict, Discrete
from . import render as roadster_render
from .simulation.adapter import PHYSICS_LEAVES, RAIN_MAX, RENDER_FIELDS, TASKS, WIND_MAX
from .simulation.execution import make_roadster_adapter, roadster_backend

#: The visual variation leaf that picks the cinematic look for a frame.
CINEMATIC = "rendering.cinematic"
#: State field names the render row carries, with the row column group each one fills.
_RENDER_KEYS = {
    "/".join(key): {("fx", "count"): "fx_count", ("fx", "history"): "fx_history"}.get(key, key[1])
    for key in RENDER_FIELDS
}


class RoadsterBase(Env):
    """One Roadster planning task on ``num_envs`` environments.

    The task logic runs on arrays of environments: NumPy with native ``mujoco.batch`` physics by default, bit for
    bit as the source; with ``simulation_backend="mjwarp"``, Torch with MuJoCo Warp physics on CUDA, approximately
    as the source.
    """

    family = "roadster"
    TASK: str = ""
    execution_mode = "AUTO_NATIVE_MJWARP"
    cpu_render_support = "SUPPORTED_SOFTWARE_OPENGL"
    action_dim = 4
    action_dtype = torch.float32
    action_low, action_high = -1.0, 1.0
    default_size = 96
    dynamics_variation_leaves = frozenset(PHYSICS_LEAVES)
    reset_value_paths = frozenset(PHYSICS_LEAVES)
    reset_sample_paths = frozenset(PHYSICS_LEAVES)
    simulation_backends = ("native", "mjwarp")
    state_solver_caches = ("sim/warmstart",)

    def __init__(
        self,
        num_envs=1,
        size=None,
        supersample=None,
        device=None,
        backend="auto",
        seed=0,
        *,
        simulation_backend="auto",
    ):
        """Build ``num_envs`` environments of the task.

        ``simulation_backend="auto"`` steps on native ``mujoco.batch`` on every device. On CUDA, one copy moves its
        results to the device. ``"native"`` and ``"mjwarp"`` select explicitly.

        Frames show the world-model camera; the ``rendering.cinematic`` variation leaf draws the same camera
        on the dark-space scene with shadows and MSAA (use a larger ``size``, e.g. 512).
        """
        device = Env.resolve_device(device)
        self.simulation_backend = roadster_backend(device, simulation_backend)
        self.execution_mode = "MUJOCO_WARP" if self.simulation_backend == "mjwarp" else "DELEGATED_CPU"
        self._simulator = make_roadster_adapter(self.TASK, int(num_envs), device, self.simulation_backend)
        model = self._simulator.model
        self.state_dim = roadster_render.row_dim(model)
        self.state_names = roadster_render.row_names(model)
        self._row_layout = roadster_render.row_layout(model)
        try:
            Env.__init__(
                self,
                num_envs=num_envs,
                size=size,
                supersample=1 if supersample is None else supersample,
                device=device,
                backend=backend,
                seed=seed,
            )
            if self.ss != 1:
                raise ValueError("Roadster renders at the requested size without supersampling")
            if self.backend not in ("auto", "opengl"):
                raise ValueError("Roadster draws with MuJoCo's OpenGL renderer; backend must be 'auto' or 'opengl'")
            self._configure_physics()
            self.reset(seed=seed)
        except BaseException:
            self._simulator.close()
            raise

    def make_variation_space(self):
        """The physics leaves ``weather.wind`` / ``weather.rain`` and the visual leaf ``rendering.cinematic``.

        ``rendering.cinematic`` (0 or 1) draws the frame on the dark-space scene instead of the world-model look.
        """
        return Dict(
            {
                "weather": Dict(
                    {
                        "wind": Box(low=0.0, high=WIND_MAX, shape=(1,), init_value=[0.0]),
                        "rain": Box(low=0.0, high=RAIN_MAX, shape=(1,), init_value=[0.0]),
                    }
                ),
                "rendering": Dict({"cinematic": Discrete(2, init_value=0)}),
            }
        )

    @property
    def _schema(self):
        return self._simulator.schema

    def _layout(self):
        return tuple(
            (
                tuple(name.split("/")),
                math.prod(where.shape),
                self._row_layout[_RENDER_KEYS[name]][0] if name in _RENDER_KEYS else None,
            )
            for name, where in self._schema.slices.items()
        )

    def _layout_dtypes(self):
        return {tuple(name.split("/")): where.dtype for name, where in self._schema.slices.items()}

    def _validate_state(self, state, *, batch=None):
        self._simulator.validate(state, batch=batch)

    def _pack_render_state(self, state):
        if not isinstance(state, TensorDict):
            return super()._pack_render_state(state)
        n = int(state.batch_size[0]) if state.batch_size else 1
        rows = torch.zeros(n, int(self.state_dim), device=self.device, dtype=torch.float32)
        if schema_of(state) is None:
            fields = {name: get_field(state, name) for name in _RENDER_KEYS}
        else:
            fields = self._schema.views(state, list(_RENDER_KEYS))
        for name, column in _RENDER_KEYS.items():
            off, width = self._row_layout[column]
            rows[:, off : off + width] = torch.as_tensor(
                fields[name], device=self.device, dtype=torch.float32
            ).reshape(n, width)
        return rows

    @property
    def state_spec(self):
        return self._schema.spec(self.device)

    def _reset(self, seed=None, options=None):
        """Reset every environment; ``options`` take ``variation`` / ``variation_values`` for the weather leaves."""
        self.draw_reset_values(seed, options)
        return self._simulator.reset(seed)

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def _set_state(self, state):
        self._simulator.set_state(state)
        self._adopt_physics_state(state)

    def _step_pipeline(self) -> StepPipeline | None:
        return step_pipeline(self._simulator)

    def _step(self, actions, *, active=None):
        self._check_open("step")
        return self._simulator.step(actions, active=active)

    def _advance_without_snapshot(self, actions, *, active=None):
        """Advance without exporting state, for sequences that need only the final snapshot."""
        self._check_open("step")
        return self._simulator.step(actions, active=active, capture_state=False)

    @torch.no_grad()
    def _render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None):
        """Draw ``(..., state_dim)`` render rows to ``(..., size, size, 3)`` uint8 frames.

        Each row's look comes from the ``rendering.cinematic`` leaf of the installed variation (or of ``style``,
        a compiled row). A ``variation`` mapping may override it and the drawn ``weather.rain`` /
        ``weather.wind`` (as a mean wind along +x).
        """
        if supersample not in (None, 1):
            raise ValueError("Roadster renders without supersampling")
        s = self._pack_render_state(self.get_state()) if states is None else states
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim).to(torch.float32)
        n = int(s.shape[0])
        if variation is not None and not isinstance(variation, Mapping):
            variation, style = None, variation  # a compiled row
        variation = dict(variation or {})
        unknown = set(variation) - {*PHYSICS_LEAVES, CINEMATIC}
        if unknown:
            raise KeyError(
                f"unknown Roadster Variation path(s) {sorted(unknown)}; known: {[*PHYSICS_LEAVES, CINEMATIC]}"
            )
        looks = {CINEMATIC: variation.pop(CINEMATIC)} if CINEMATIC in variation else None
        cinematic = self.resolve_style(style, looks, batch=n)[CINEMATIC].reshape(-1)[:n].cpu().numpy() > 0.5
        if variation:
            s = s.clone()
            value = lambda leaf: float(torch.as_tensor(variation[leaf], dtype=torch.float32).reshape(-1)[0])
            if "weather.rain" in variation:
                s[:, self._row_layout["rain"][0]] = value("weather.rain")
            if "weather.wind" in variation:
                off, _ = self._row_layout["wind_mean"]
                s[:, off : off + 3] = torch.tensor([value("weather.wind"), 0.0, 0.0])
        res = int(size or self.size)
        frames = torch.empty((n, res, res, 3), dtype=torch.uint8)
        for look, instances in (("wm", ~cinematic), ("cinematic", cinematic)):
            if instances.any():
                index = torch.from_numpy(instances.nonzero()[0])
                frames[index] = roadster_render.render_rows(s[index.to(s.device)], res, "cpu", look=look)
        if out is not None:
            out.view(-1, res, res, 3).copy_(frames.to(out.device))
            return out
        frames = frames.to(self.device)
        return frames.view(*lead, res, res, 3) if lead else frames

    def _rows_from_fields(self, fields) -> torch.Tensor:
        """Pack ``{State key: (n, width) array}`` render-row fields into ``(n, state_dim)`` rows.

        The rows equal what ``_pack_render_state`` packs from a state holding the same fields.
        """
        n = len(fields[RENDER_FIELDS[0]])
        rows = torch.zeros((n, self.state_dim), dtype=torch.float32)
        for key in RENDER_FIELDS:
            off, width = self._row_layout[_RENDER_KEYS["/".join(key)]]
            rows[:, off : off + width] = torch.as_tensor(fields[key]).reshape(n, width).to(torch.float32)
        return rows.to(self.device)

    def _draw_fields(self, fields) -> torch.Tensor:
        return self._render(self._rows_from_fields(fields))

    def _render_live(self):
        """Draw the running environments from their render-row fields, without building a state snapshot."""
        if hasattr(self._simulator, "render_fields"):
            return self._draw_fields(self._simulator.render_fields())
        return self._render(self._pack_render_state(self.get_state()))

    def _close(self):
        if hasattr(self, "_simulator"):
            self._simulator.close()


_DOCS = {
    "loop": "drive the half-pipe loop, leave it inverted and roll upright",
    "mission": "the full course: loop, three rings, pad landing",
    "traffic": "hop a traffic jam and land on the road beyond",
    "rooftop": "fly from a tower roof around a taller tower to a helipad",
    "carhop": "jump sideways over a parked car with a barrel roll",
}
CELLS = tuple(
    (f"Roadster{name}", task)
    for name, task in zip(("Loop", "Mission", "TrafficJam", "Rooftop", "CarHop"), TASKS, strict=True)
)


#: A State saved on one backend resumes on the other with different contact forces (issue #63).
_CONTACT_MISMATCH = frozenset({("native", "mjwarp"), ("mjwarp", "native")})


def _make_classes() -> dict[str, type]:
    made = {}
    for cname, task in CELLS:
        made[cname] = register(
            type(
                cname,
                (RoadsterBase,),
                dict(
                    TASK=task,
                    **({"state_backend_mismatches": _CONTACT_MISMATCH} if task == "rooftop" else {}),
                    name=f"roadster-{task}",
                    domain=task,
                    reference_id=f"roadster_jets/RoadsterPlan-v0:{task}",
                    __module__=__name__,
                    __doc__=f"Roadster ``{task}`` task: {_DOCS[task]}. Batched MuJoCo simulation with restorable State.",
                ),
            )
        )
    return made


globals().update(_make_classes())

__all__ = ["RoadsterBase", *(c for c, _ in CELLS)]
