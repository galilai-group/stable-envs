"""The Roadster State schema, its bounds and the checks of actions and physics rows.

State is ``source.RoadsterPlanEnv.get_state`` laid out as typed columns: MuJoCo's integration state split
into its components (time, qpos, qvel, warm start, ctrl, applied forces, mocap bodies), the pose the derived
quantities refer to after a step, the autopilot, the task bookkeeping, the goal and route, the traffic jam,
the weather with its gust RNG, the exhaust history the smoke is drawn from, and the elapsed steps.
"""

from __future__ import annotations

import numpy as np
import torch

from ...._core.physics import PhysicsLayout, state_key
from ..source.fx import ENTRY_DIM, HISTORY
from ..source.tasks import TASKS as TASKS

#: Episode status strings of the source env; ``("task", "status")`` indexes this tuple (``("task", "id")``
#: indexes the source's ``TASKS``).
STATUSES = (
    "drive",
    "flight",
    "landed",
    "success",
    "crash: hit obstacle",
    "crash: body impact",
    "crash: ditched",
    "crash: out of bounds",
)
#: Longest route a task defines (rooftop: four waypoints).
MAX_ROUTE = 4
#: The weather seed travels as two 24-bit halves, each exact in a float32 render row.
SEED_BITS = 24
SEED_MASK = (1 << SEED_BITS) - 1
MAX_SEED = (1 << 2 * SEED_BITS) - 1
#: Bounds of the ``weather.wind`` leaf (mean speed, m/s) and the ``weather.rain`` leaf (strength).
WIND_MAX = 20.0
RAIN_MAX = 1.0
#: The physics Variation leaves, their row layout and the source's defaults (calm, dry).
PHYSICS_LEAVES = ("weather.wind", "weather.rain")
PHYSICS_LAYOUT = PhysicsLayout(leaves=PHYSICS_LEAVES, widths=(1, 1), offsets=(0, 1), nominal=(0.0, 0.0))
MAX_TARGETS = 4  # three rings and the pad: ("task", "target_idx") is the next one
#: Exhaust history: the source keeps the last FX_HISTORY steps of (step, time, pose, velocity, jet commands,
#: wind), FX_ENTRY values each, and draws the smoke from them.
FX_HISTORY, FX_ENTRY = HISTORY, ENTRY_DIM

#: MuJoCo integration-state components in the order ``mj_getState`` writes them.
_COMPONENTS = (
    ("time", "mjSTATE_TIME"),
    ("qpos", "mjSTATE_QPOS"),
    ("qvel", "mjSTATE_QVEL"),
    ("act", "mjSTATE_ACT"),
    ("warmstart", "mjSTATE_WARMSTART"),
    ("ctrl", "mjSTATE_CTRL"),
    ("qfrc_applied", "mjSTATE_QFRC_APPLIED"),
    ("xfrc_applied", "mjSTATE_XFRC_APPLIED"),
    ("eq_active", "mjSTATE_EQ_ACTIVE"),
    ("mocap_pos", "mjSTATE_MOCAP_POS"),
    ("mocap_quat", "mjSTATE_MOCAP_QUAT"),
    ("userdata", "mjSTATE_USERDATA"),
    ("plugin", "mjSTATE_PLUGIN"),
)

_NUMPY_DTYPES = {torch.float64: np.float64, torch.int64: np.int64, torch.bool: np.bool_}

#: State fields a render row is packed from (``render.row_layout``): what a frame needs and nothing more.
RENDER_FIELDS = (
    ("sim", "time"),
    ("sim", "qpos"),
    ("sim", "qpos_pre"),
    ("sim", "has_qpos_pre"),
    ("sim", "ctrl"),
    ("sim", "mocap_pos"),
    ("sim", "mocap_quat"),
    ("weather", "seed_hi"),
    ("weather", "seed_lo"),
    ("weather", "rain"),
    ("weather", "wind_mean"),
    ("fx", "count"),
    ("fx", "history"),
)


def integration_layout(model) -> dict[str, tuple[int, int]]:
    """``{component: (offset, width)}`` of the nonempty components of the integration state vector."""
    import mujoco

    out, offset = {}, 0
    for name, flag in _COMPONENTS:
        width = int(mujoco.mj_stateSize(model, getattr(mujoco.mjtState, flag)))
        if width:
            out[name] = (offset, width)
            offset += width
    assert offset == int(mujoco.mj_stateSize(model, mujoco.mjtState.mjSTATE_INTEGRATION))
    return out


def state_fields(model) -> dict[tuple[str, str], tuple[int, torch.dtype]]:
    """The State schema for ``model``: ``{(group, name): (width, dtype)}``."""
    nq = int(model.nq)
    f64, i64, b = torch.float64, torch.int64, torch.bool
    fields: dict = {("sim", name): (width, f64) for name, (_, width) in integration_layout(model).items()}
    fields.update(
        {
            # Pose that xpos/xmat/contacts refer to after a step (one substep behind qpos), when it differs.
            ("sim", "qpos_pre"): (nq, f64),
            ("sim", "has_qpos_pre"): (1, b),
            ("autopilot", "yaw_cmd"): (1, f64),
            # The barrel-roll target; NaN upstream means "hold level", carried here as has_roll_cmd=False.
            ("autopilot", "roll_cmd"): (1, f64),
            ("autopilot", "has_roll_cmd"): (1, b),
            ("task", "id"): (1, i64),
            ("task", "fuel"): (1, f64),
            ("task", "target_idx"): (1, i64),
            ("task", "airborne_once"): (1, b),
            ("task", "status"): (1, i64),
            ("task", "sim_steps"): (1, i64),
            ("task", "prev_cost"): (1, f64),
            ("task", "loop_top_seen"): (1, b),
            ("task", "ground_hold"): (1, i64),
            ("task", "route_idx"): (1, i64),
            ("task", "roll_t0"): (1, f64),
            ("task", "has_roll_t0"): (1, b),
            ("task", "roll_done"): (1, b),
            ("task", "region_y"): (1, f64),
            ("task", "allow_ground_landing"): (1, b),
            ("goal", "position"): (3, f64),
            ("goal", "yaw"): (1, f64),
            ("goal", "velocity"): (3, f64),
            ("route", "count"): (1, i64),
            ("route", "positions"): (3 * MAX_ROUTE, f64),
            ("route", "velocities"): (3 * MAX_ROUTE, f64),
            ("route", "up"): (MAX_ROUTE, f64),
            ("route", "touch"): (MAX_ROUTE, b),  # touchdown waypoints
            ("jam", "present"): (1, b),
            ("jam", "extent"): (3, f64),  # x_start, x_end, top
            # The weather seed in two 24-bit halves, each exact in a float32 render row (seed < 2**48).
            ("weather", "seed_hi"): (1, i64),
            ("weather", "seed_lo"): (1, i64),
            ("weather", "rain"): (1, f64),
            ("weather", "wind_mean"): (3, f64),
            ("weather", "gust"): (3, f64),
            ("weather", "gust_std"): (1, f64),
            ("weather", "grip"): (1, f64),
            ("weather", "rng_state"): (4, i64),
            ("weather", "rng_increment"): (4, i64),
            ("weather", "rng_has_uint32"): (1, i64),
            ("weather", "rng_uinteger"): (1, i64),
            ("fx", "count"): (1, i64),
            ("fx", "history"): (FX_HISTORY * FX_ENTRY, f64),
            ("lifecycle", "elapsed_steps"): (1, i64),
            # The physics leaves the episode runs under (installed at its reset; float64, exact).
            **{state_key(leaf): (1, f64) for leaf in PHYSICS_LEAVES},
        }
    )
    return fields


def _limbs(value: int) -> np.ndarray:
    """Split a PCG64 unsigned 128-bit integer into four uint32 limbs, least significant first."""
    value = int(value)
    return np.asarray([(value >> (32 * i)) & 0xFFFF_FFFF for i in range(4)], dtype=np.int64)


def _from_limbs(value) -> int:
    parts = np.asarray(value, dtype=np.int64).reshape(4)
    return sum(int(part) << (32 * i) for i, part in enumerate(parts))


def physics_rows(values, num_envs: int) -> np.ndarray:
    """Return ``values`` as owned ``(num_envs, 2)`` float64 ``(wind speed, rain)`` rows, or raise."""
    rows = np.array(values, np.float64)
    if rows.shape != (num_envs, 2) or not np.isfinite(rows).all():
        raise ValueError(f"Roadster physics rows must be finite with shape ({num_envs}, 2)")
    if (rows < 0).any() or (rows[:, 0] > WIND_MAX).any() or (rows[:, 1] > RAIN_MAX).any():
        raise ValueError(f"Roadster weather.wind lies in [0, {WIND_MAX}] and weather.rain in [0, {RAIN_MAX}]")
    return rows


def host_actions(actions, num_envs: int) -> np.ndarray:
    """Return ``actions`` as a finite host array, rejecting any shape other than ``(num_envs, 4)``."""
    if torch.is_tensor(actions):
        actions = actions.detach().cpu().numpy()
    actions = np.asarray(actions)
    if actions.shape != (num_envs, 4):
        raise ValueError(f"Roadster actions require shape ({num_envs}, 4), got {actions.shape}")
    if actions.dtype.kind != "f":
        raise TypeError(f"Roadster actions must be floating point, got {actions.dtype}")
    if not np.isfinite(actions).all():
        raise ValueError("Roadster actions must contain only finite values")
    return actions
