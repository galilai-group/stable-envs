"""The five Roadster planning tasks: CPU MuJoCo simulation with restorable State and the world-model camera.

Each class fixes one task of ``source.RoadsterPlanEnv`` (the simulator, autopilot, reward, success and
horizon, ported from roadster-jets). The 4-D action ``[a_fwd, a_lat, a_up, yaw_rate]`` in [-1, 1] runs at
10 Hz. Wind and rain are physics Variation leaves (``weather.wind`` in m/s, ``weather.rain`` in [0, 1]) that
reach the source at reset, as it sets the weather per episode; State records them under ``physics.weather``.
Frames come from the ``wm`` camera (follows the car with a fixed world orientation), 96 px by default, drawn
by MuJoCo's OpenGL renderer on CPU; the exhaust smoke is drawn from the history in State.
"""

from __future__ import annotations

import torch

try:
    import mujoco  # noqa: F401
    import PIL  # noqa: F401
except ImportError as e:  # registered only with the extra, so import_errors() names it
    raise ImportError("Roadster needs the roadster extra: pip install 'stable-envs[roadster]'") from e

from ..._execution.protocol import StepPipeline, step_pipeline
from ...env import Env
from ...registry import register
from ...spaces import Box, Dict
from . import render as roadster_render
from .simulation.adapter import PHYSICS_LEAVES, RAIN_MAX, TASKS, WIND_MAX, RoadsterAdapter

#: State keys the render row carries, with the row column group each one fills.
_RENDER_KEYS = {
    ("sim", "time"): "time",
    ("sim", "qpos"): "qpos",
    ("sim", "qpos_pre"): "qpos_pre",
    ("sim", "has_qpos_pre"): "has_qpos_pre",
    ("sim", "ctrl"): "ctrl",
    ("sim", "mocap_pos"): "mocap_pos",
    ("sim", "mocap_quat"): "mocap_quat",
    ("weather", "seed"): "seed",  # float32 in the row: seeds above 2**24 draw the particles of the nearest one
    ("weather", "rain"): "rain",
    ("weather", "wind_mean"): "wind_mean",
    ("fx", "count"): "fx_count",
    ("fx", "history"): "fx_history",
}


class RoadsterBase(Env):
    """One Roadster planning task on ``num_envs`` lanes.

    Physics runs on CPU in the source env; ``device`` selects where State and frames are stored.
    """

    family = "roadster"
    TASK: str = ""
    execution_mode = "DELEGATED_CPU"
    cpu_render_support = "SUPPORTED_SOFTWARE_OPENGL"
    action_dim = 4
    action_dtype = torch.float32
    action_low, action_high = -1.0, 1.0
    default_size = 96
    dynamics_variation_leaves = frozenset(PHYSICS_LEAVES)
    reset_value_paths = frozenset(PHYSICS_LEAVES)
    reset_sample_paths = frozenset(PHYSICS_LEAVES)

    def __init__(self, num_envs=1, size=None, supersample=None, device=None, backend="auto", seed=0, look="wm"):
        """Build ``num_envs`` lanes of the task.

        ``look="wm"`` (default) draws the world-model camera; ``look="cinematic"`` draws the same camera on the
        dark-space scene with shadows and MSAA (use a larger ``size``, e.g. 512).
        """
        if look not in roadster_render.LOOKS:
            raise ValueError(f"look must be one of {roadster_render.LOOKS}, got {look!r}")
        self.look = look
        device = Env.resolve_device(device)
        self._simulator = RoadsterAdapter(self.TASK, int(num_envs), device)
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

    # ------------------------------------------------------------------ Variation
    def make_variation_space(self):
        return Dict(
            {
                "weather": Dict(
                    {
                        "wind": Box(low=0.0, high=WIND_MAX, shape=(1,), init_value=[0.0]),
                        "rain": Box(low=0.0, high=RAIN_MAX, shape=(1,), init_value=[0.0]),
                    }
                )
            }
        )

    # ------------------------------------------------------------------ State
    def _layout(self):
        return tuple(
            (key, width, self._row_layout[_RENDER_KEYS[key]][0] if key in _RENDER_KEYS else None)
            for key, (width, _) in self._simulator.fields.items()
        )

    def _layout_dtypes(self):
        return {key: dtype for key, (_, dtype) in self._simulator.fields.items()}

    def _validate_state(self, state, *, batch=None):
        self._simulator.validate(state, batch=batch)

    def _sync_render_rows(self, state) -> None:
        self._render_state.copy_(self._pack_render_state(state))

    def reset(self, seed=None, options=None):
        """Reset every lane; ``options`` take ``variation`` / ``variation_values`` for the weather leaves."""
        self._check_open("reset")
        self.draw_reset_values(seed, options)
        state = self._simulator.reset(seed)
        self._sync_render_rows(state)
        return state

    def get_state(self):
        self._check_open("get_state")
        return self._simulator.get_state()

    def set_state(self, state):
        self._check_open("set_state")
        self._simulator.set_state(state)
        self._sync_render_rows(state)
        self._adopt_physics_state(state)

    def _step_pipeline(self) -> StepPipeline | None:
        return step_pipeline(self._simulator)

    def _step(self, actions, *, active=None):
        self._check_open("step")
        result = self._simulator.step(actions, active=active)
        self._sync_render_rows(result["state"])
        return result

    # ------------------------------------------------------------------ rendering
    @torch.no_grad()
    def _render(self, states=None, size=None, supersample=None, variation=None, style=None, out=None):
        """Draw ``(..., state_dim)`` render rows to ``(..., size, size, 3)`` uint8 frames.

        ``variation`` may override the drawn ``weather.rain`` / ``weather.wind`` (as a mean wind along +x).
        """
        if supersample not in (None, 1):
            raise ValueError("Roadster renders without supersampling")
        if style is not None:
            raise TypeError("Roadster has no compiled appearance rows; pass weather overrides as variation=")
        s = self._get_render_state() if states is None else states
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim).to(torch.float32)
        if variation:
            unknown = set(variation) - set(PHYSICS_LEAVES)
            if unknown:
                raise KeyError(f"unknown Roadster Variation path(s) {sorted(unknown)}; known: {list(PHYSICS_LEAVES)}")
            s = s.clone()
            value = lambda leaf: float(torch.as_tensor(variation[leaf], dtype=torch.float32).reshape(-1)[0])
            if "weather.rain" in variation:
                s[:, self._row_layout["rain"][0]] = value("weather.rain")
            if "weather.wind" in variation:
                off, _ = self._row_layout["wind_mean"]
                s[:, off : off + 3] = torch.tensor([value("weather.wind"), 0.0, 0.0])
        res = int(size or self.size)
        flat = None if out is None else out.view(-1, res, res, 3)
        frames = roadster_render.render_rows(s, res, self.device, out=flat, look=self.look)
        if out is not None:
            return out
        return frames.view(*lead, res, res, 3) if lead else frames

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


def _make_classes() -> dict[str, type]:
    made = {}
    for cname, task in CELLS:
        made[cname] = register(
            type(
                cname,
                (RoadsterBase,),
                dict(
                    TASK=task,
                    name=f"roadster-{task}",
                    domain=task,
                    reference_id=f"roadster_jets/RoadsterPlan-v0:{task}",
                    __module__=__name__,
                    __doc__=f"Roadster ``{task}`` task: {_DOCS[task]}. CPU MuJoCo simulation with restorable State.",
                ),
            )
        )
    return made


globals().update(_make_classes())

__all__ = ["RoadsterBase", *(c for c, _ in CELLS)]
