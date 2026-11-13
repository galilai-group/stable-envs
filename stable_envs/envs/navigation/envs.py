from __future__ import annotations

import math

import numpy as np
import torch
from gymnasium import spaces as gym_spaces
from tensordict import TensorDict

from ... import tuning
from ..._core.tensordicts import build_tensordict
from ...env import Env, ResetLeaf, StepResult, check_states
from ...registry import register
from ...render import _resources as _RES
from ...render.scene import Scene2D, SceneBuilder
from ...spaces import Box, Dict, Discrete, RGBBox
from . import navdot
from .maze_goal import GoalPainter, goal_frames

IMG = 224
# Playfield inset, distinct from the border stroke width.
BORDER = 14
BORDER_T = 4
WALL_CENTER = 112
MAX_DOOR = 3
# Extra door clearance used by the reference collision test.
DOOR_MARGIN = 1.75
WALL_PUSHBACK = 0.5
GRID_N = 2  # PiecewiseEnv's default grid_n
N_ZONES = GRID_N * GRID_N

AGENT_C = (255, 0, 0)
TARGET_C = (0, 255, 0)
WALL_C = (0, 0, 0)
DOOR_C = (255, 255, 255)
BG_C = (255, 255, 255)
BORDER_C = (0, 0, 0)
AGENT_R = 7.0
TARGET_R = 7.0
AGENT_SPEED = 5.0
WALL_THICKNESS = 10
WALL_AXIS = 1  # 1 = vertical wall, 0 = horizontal
N_DOORS = 1
DOOR_POS = (49.0, 49.0, 49.0)
DOOR_SIZE = (14.0, 14.0, 14.0)
# Reference HSV zone colors, scaled by 255 and truncated to integers.
ZONE_C = ((255, 191, 191), (223, 255, 191), (191, 255, 255), (223, 191, 255))
# Per-zone motion bias; these vectors affect dynamics rather than rendering.
ZONE_BIAS = ((2.0, 0.0), (0.0, 2.0), (-2.0, 0.0), (-0.0, -2.0))

POS_LO = float(BORDER)
POS_HI = float(IMG - BORDER - 1)


def _grids(device) -> tuple[torch.Tensor, torch.Tensor]:
    """Return pixel-center coordinate grids ``(H, W)`` with matrix indexing."""
    a = torch.arange(IMG, dtype=torch.float32, device=device)
    gy, gx = torch.meshgrid(a, a, indexing="ij")
    return gy, gx


def _border_mask(device) -> torch.Tensor:
    """Return the mask for four-pixel border strokes at rows/columns 10–13 and 210–213."""
    m = torch.zeros(IMG, IMG, dtype=torch.bool, device=device)
    b, t = BORDER, BORDER_T
    m[:, b - t : b] = True
    m[:, IMG - b : IMG - b + t] = True
    m[b - t : b, :] = True
    m[IMG - b : IMG - b + t, :] = True
    return m


def _zone_map(device) -> torch.Tensor:
    """Return Piecewise zone indices ``(H, W)``, with -1 outside the playfield."""
    z = torch.full((IMG, IMG), -1, dtype=torch.long, device=device)
    play = IMG - 2 * BORDER
    for row in range(GRID_N):
        for col in range(GRID_N):
            y0 = BORDER + int(row * play / GRID_N)
            y1 = BORDER + int((row + 1) * play / GRID_N)
            x0 = BORDER + int(col * play / GRID_N)
            x1 = BORDER + int((col + 1) * play / GRID_N)
            z[y0:y1, x0:x1] = row * GRID_N + col
    return z


def _bias_field_rgba(biases) -> np.ndarray:
    """Draw the per-zone bias arrows as uint8 RGBA ``(H, W, 4)`` with Matplotlib Agg.

    The reference rasterizes a quiver plot, so the arrows are reproduced by the same
    rasterizer rather than approximated by primitives.
    """
    try:
        from matplotlib.backends.backend_agg import FigureCanvasAgg
        from matplotlib.figure import Figure
    except ImportError as exc:
        raise ImportError("Piecewise rendering.render_bias_field requires matplotlib") from exc

    samples = 3
    play = IMG - 2 * BORDER
    xs, ys, us, vs = [], [], [], []
    for row in range(GRID_N):
        for col in range(GRID_N):
            bx, by = (float(v) for v in biases[row * GRID_N + col])
            y_lo, y_hi = BORDER + row * play / GRID_N, BORDER + (row + 1) * play / GRID_N
            x_lo, x_hi = BORDER + col * play / GRID_N, BORDER + (col + 1) * play / GRID_N
            for si in range(samples):
                for sj in range(samples):
                    xs.append(x_lo + (sj + 0.5) * (x_hi - x_lo) / samples)
                    ys.append(y_lo + (si + 0.5) * (y_hi - y_lo) / samples)
                    us.append(bx)
                    vs.append(by)
    dpi = 100
    fig = Figure(figsize=(IMG / dpi, IMG / dpi), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, IMG)
    ax.set_ylim(IMG, 0)
    ax.axis("off")
    fig.patch.set_alpha(0.0)
    ax.patch.set_alpha(0.0)
    scale = 5.0 * math.sqrt(2) / (play / GRID_N * 0.15)
    ax.quiver(
        xs,
        ys,
        us,
        vs,
        color="#111111",
        alpha=0.85,
        scale=scale,
        scale_units="xy",
        angles="xy",
        width=0.006,
        headwidth=4,
        headlength=4,
        headaxislength=3.5,
        minlength=0.5,
    )
    canvas.draw()
    buf, (w, h) = canvas.print_to_buffer()
    rgba = np.frombuffer(buf, dtype=np.uint8).reshape(h, w, 4).copy()
    fig.clear()
    return rgba


_WALL_RESET_PATHS = ("wall.axis", "wall.thickness", "agent.radius")
_EXECUTION_MODES = {"torch": "NATIVE_TORCH", "jax": "NATIVE_JAX", "numba": "NATIVE_NUMBA"}
_NO_MASK = np.zeros(0, np.bool_)


def _numba_available() -> bool:
    from ...render import cpu_raster

    return cpu_raster.available()


def _u(env):
    return getattr(env, "unwrapped", env)


def _trunc_color(c: torch.Tensor) -> torch.Tensor:
    """Truncate color channels to match reference uint8 conversion."""
    return c.trunc()


def _gaussian_dot(gy, gx, pos, radius) -> torch.Tensor:
    """Compute normalized Gaussian alpha ``(B, H, W)``.

    Keep the squared distance in float32, but evaluate the divisor and exp
    in float64 before casting. Changing these rounding points can change
    uint8 pixels at truncation boundaries.
    """
    dx = gx.unsqueeze(0) - pos[:, 0].view(-1, 1, 1)
    dy = gy.unsqueeze(0) - pos[:, 1].view(-1, 1, 1)
    dist2 = dx * dx + dy * dy
    std = radius.double().clamp_min(1e-6)
    denom = (2.0 * std * std).to(torch.float32).view(-1, 1, 1)
    dot = torch.exp((-dist2 / denom).double()).to(torch.float32)
    m = dot.amax(dim=(1, 2), keepdim=True)
    return torch.where(m > 0, dot / m, dot)


# The uint8 alpha cutoff shared with the fused renderer.
_ALPHA_FLOOR = navdot.ALPHA_FLOOR


def _dot_window(pos, sigma):
    """Return pixel windows ``(k, 4)`` outside which a dot cannot change uint8 output."""
    return navdot.dot_window(pos, sigma.reshape(pos.shape[0]), IMG, IMG)


def _alpha_blend(img_u8: torch.Tensor, alpha: torch.Tensor, rgb: torch.Tensor) -> torch.Tensor:
    """Blend in float32 and return uint8.

    Quantize after each dot: combining consecutive blends before conversion
    changes pixels where the dots overlap.
    """
    a = alpha.clamp(0.0, 1.0).unsqueeze(-1)
    out = img_u8.to(torch.float32)
    out = out * (1.0 - a) + _trunc_color(rgb).view(-1, 1, 1, 3) * a
    return out.to(torch.uint8)


def _seed_leaf(shape=(1,), low=0.0, high=1.0, dtype=np.float32):
    """Build a Gymnasium leaf for the source-shaped reset RNG tree.

    Keep sibling leaves even when reset does not sample them: Dict.seed
    assigns child seeds according to the whole tree.
    """
    return gym_spaces.Box(low=low, high=high, shape=shape, dtype=dtype)


def _two_room_reset_tree():
    return gym_spaces.Dict(
        {
            "agent": gym_spaces.Dict(
                {
                    "color": _seed_leaf((3,), 0, 255, np.uint8),
                    "radius": _seed_leaf(),
                    "position": _seed_leaf((2,), POS_LO, POS_HI),
                    "speed": _seed_leaf(),
                }
            ),
            "background": gym_spaces.Dict({"color": _seed_leaf((3,), 0, 255, np.uint8)}),
            "door": gym_spaces.Dict(
                {
                    "color": _seed_leaf((3,), 0, 255, np.uint8),
                    "number": gym_spaces.Discrete(3),
                    "position": _seed_leaf((3,), 0, 223),
                    "size": _seed_leaf((3,), 1, 21),
                }
            ),
            "rendering": gym_spaces.Dict({"render_target": gym_spaces.Discrete(2)}),
            "target": gym_spaces.Dict(
                {
                    "color": _seed_leaf((3,), 0, 255, np.uint8),
                    "position": _seed_leaf((2,), POS_LO, POS_HI),
                    "radius": _seed_leaf(),
                }
            ),
            "task": gym_spaces.Dict({"min_steps": gym_spaces.Discrete(100)}),
            "wall": gym_spaces.Dict(
                {
                    "axis": gym_spaces.Discrete(2),
                    "border_color": _seed_leaf((3,), 0, 255, np.uint8),
                    "color": _seed_leaf((3,), 0, 255, np.uint8),
                    "thickness": gym_spaces.Discrete(35),
                }
            ),
        }
    )


def _piecewise_reset_tree():
    zones = {}
    for i in range(N_ZONES):
        zones[f"bias_{i}"] = _seed_leaf((2,), -4, 4)
        zones[f"color_{i}"] = _seed_leaf((3,), 0, 255, np.uint8)
    return gym_spaces.Dict(
        {
            "agent": gym_spaces.Dict(
                {
                    "color": _seed_leaf((3,), 0, 255, np.uint8),
                    "radius": _seed_leaf(),
                    "position": _seed_leaf((2,), POS_LO, POS_HI),
                    "speed": _seed_leaf(),
                }
            ),
            "background": gym_spaces.Dict({"color": _seed_leaf((3,), 0, 255, np.uint8)}),
            "border": gym_spaces.Dict({"color": _seed_leaf((3,), 0, 255, np.uint8)}),
            "rendering": gym_spaces.Dict(
                {
                    "render_bias_field": gym_spaces.Discrete(2),
                    "render_target": gym_spaces.Discrete(2),
                    "render_zones": gym_spaces.Discrete(2),
                }
            ),
            "target": gym_spaces.Dict(
                {
                    "color": _seed_leaf((3,), 0, 255, np.uint8),
                    "position": _seed_leaf((2,), POS_LO, POS_HI),
                    "radius": _seed_leaf(),
                }
            ),
            "task": gym_spaces.Dict({"min_steps": gym_spaces.Discrete(100)}),
            "zones": gym_spaces.Dict(zones),
        }
    )


def _simple_maze_reset_tree():
    return gym_spaces.Dict(
        {
            "agent": gym_spaces.Dict(
                {
                    "color": _seed_leaf((3,), 0, 255, np.uint8),
                    "position": _seed_leaf((2,), 0, 5),
                    "radius": _seed_leaf(shape=()),
                    "speed": _seed_leaf(shape=()),
                }
            ),
            "background": gym_spaces.Dict({"color": _seed_leaf((3,), 0, 255, np.uint8)}),
            "goal": gym_spaces.Dict(
                {
                    "color": _seed_leaf((3,), 0, 255, np.uint8),
                    "position": _seed_leaf((2,), 0, 5),
                    "radius": _seed_leaf(shape=()),
                }
            ),
            "walls": gym_spaces.Dict(
                {
                    "color": _seed_leaf((3,), 0, 255, np.uint8),
                    "number": gym_spaces.Discrete(MAX_WALLS - MIN_WALLS + 1),
                    "positions": _seed_leaf((MAX_WALLS, 2), 0, 5),
                    "shape": _seed_leaf((MAX_WALLS, 2), 0.5, 1.5),
                }
            ),
        }
    )


class _GymReferencePort(Env):
    """Shared Gymnasium reference construction, restoration and stepping hooks."""

    family = "navigation"

    reference_kwargs: dict = {}
    # Matplotlib references take a render mode argument; torch references take none.
    reference_render_args: tuple = ()

    step_reference_checkable = True
    step_reference_exact = True
    step_reference_control = "agent"
    step_reference_gate = "agent"
    step_reference_hold = 8
    step_reference_groups = (("agent", (0, 1), "euclidean"),)

    @classmethod
    def reference_env(cls, size: int | None = None):
        """Construct the reference supplied by the testing certification overlay."""
        del size
        gym_id = getattr(cls, "_se_cert_gym_id", None) or cls.reference_id
        if not gym_id:
            raise LookupError(
                f"{cls.__name__} has no production reference_id; certification gym ids live in "
                "tests.helpers.reference.certification"
            )
        import gymnasium

        return gymnasium.make(gym_id, **cls.reference_kwargs)

    @classmethod
    def reference_set_state(cls, u, st) -> None:
        """Write a compact pose to the unwrapped reference environment."""
        raise NotImplementedError

    @classmethod
    def reference_render(cls, env, states) -> np.ndarray:
        """Render reference states with default appearance.

        Reset with ``variation=[]`` to prevent the source from resampling
        appearance leaves before restoring the requested poses.
        """
        u = env.unwrapped
        env.reset(seed=0, options={"variation": []})
        out = []
        for st in np.asarray(states):
            cls.reference_set_state(u, st)
            got = cls.reference_sim_state(u)
            err = float(np.abs(np.asarray(got, np.float64) - np.asarray(st, np.float64)).max())
            if err > 1e-6:
                raise AssertionError(
                    f"the reference env did not reach the requested state (max err {err:.3e}); asked "
                    f"{np.asarray(st)}, got {np.asarray(got)}. Parity measured against this frame "
                    f"would be comparing two different scenes."
                )
            out.append(np.ascontiguousarray(u.render(*cls.reference_render_args)))
        return np.stack(out)

    @staticmethod
    def reference_step(ref, a):
        return ref.step(np.asarray(a, np.float32))

    @classmethod
    def step_reference_reset(cls, ref, seed):
        ref.reset(seed=int(seed))
        return np.asarray(cls.reference_sim_state(ref.unwrapped), np.float64)

    def _rows_and_style(self, state, owner: str):
        """Convert state to float32 rows ``(n, state_dim)`` and resolve style for those rows."""
        rows = (
            self._pack_render_state(state)
            if isinstance(state, TensorDict)
            else check_states(self, state, f"{type(self).__name__}.{owner}")
        )
        rows = rows.reshape(-1, self.state_dim)
        return rows, self._style.for_batch(int(rows.shape[0]), owner=f"{type(self).__name__}.{owner}")

    @torch.no_grad()
    def _step(self, action, *, active=None) -> TensorDict:
        """Advance compact navigation state and return it without task outputs.

        Task-aware subclasses override this method to supply reward and done flags.
        """
        self._check_open("step")
        act = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.B, self.action_dim)
        next_rows = self._advance_rows(self._render_state, act, self._style)
        if active is not None:
            next_rows = torch.where(active[:, None], next_rows, self._render_state)
        self._render_state.copy_(next_rows)
        return self._step_result(self.state)

    def _advance_rows(self, rows: torch.Tensor, action: torch.Tensor, sty) -> torch.Tensor:
        """Compute the next packed rows from state, action and style without mutation."""
        raise NotImplementedError


class _HiddenTargetSemantics:
    """Task outputs and full state for navigation environments with an invisible target.

    The target is absent from the renderer projection but required for
    observations, reward and termination, so it must survive state restoration.
    """

    execution_mode = "NATIVE_TORCH"
    _step_outputs_owned = True
    # TwoRoom observations append the door centers.
    _numba_doors = False
    state_layout = ((("agent", "position"), 2, 0), (("target", "position"), 2, None))
    state_bounds = {
        ("agent", "position"): (POS_LO, POS_HI),
        ("target", "position"): (POS_LO, POS_HI),
    }
    reset_leaves = (
        ResetLeaf("agent.position", (60.0, 112.0)),
        ResetLeaf("target.position", (164.0, 112.0)),
    )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._target_position = torch.full((self.B, 2), 0.0, device=self.device, dtype=torch.float32)
        self._target_position[:, 0] = 164.0
        self._target_position[:, 1] = 112.0
        if self.device.type == "cuda":
            from torch._inductor import list_options

            # Older compilers can eliminate the float32 rounding points below.
            # Keep their eager path; supported compilers fuse the same task code.
            if "emulate_precision_casts" in list_options():
                self._task_tensors = torch.compile(
                    self._task_tensors,
                    fullgraph=True,
                    options={"emulate_precision_casts": True, "aggressive_fusion": True, "triton.cudagraphs": False},
                )

    def _hidden_state(self):
        return {("target", "position"): self._target_position[: self._live_rows]}

    def _restore_state(self, state):
        self._set_render_state(self._pack_render_state(state))
        self._target_position.copy_(
            torch.as_tensor(state["target", "position"], device=self.device, dtype=torch.float32).reshape(self.B, 2)
        )

    @staticmethod
    def _option_position(value, lane, batch):
        value = np.asarray(value, np.float32)
        if value.shape == (2,):
            return value.copy()
        if value.shape == (batch, 2):
            return value[lane].copy()
        raise ValueError(f"reset position option must have shape (2,) or ({batch}, 2), got {value.shape}")

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        options = dict(options or {})
        values = self.draw_reset_values(seed, options)
        agent_rows, target_rows = list(values["agent.position"]), list(values["target.position"])
        for lane in range(self.B):
            if "state" in options:
                agent_rows[lane] = self._option_position(options["state"], lane, self.B)
            if "target_state" in options:
                target_rows[lane] = self._option_position(options["target_state"], lane, self.B)
        self._set_render_state(np.stack(agent_rows))
        self._target_position.copy_(torch.as_tensor(np.stack(target_rows), device=self.device))
        self._live_rows = self.B
        return self.get_state()

    def _full_state(self, agent):
        n = int(agent.shape[0])
        return build_tensordict(
            {
                "agent": {"position": agent.clone()},
                "target": {"position": self._target_position[:n].clone()},
            },
            batch_size=[n],
            device=self.device,
        )

    def _observation(self, agent, target):
        return torch.cat([agent, target], dim=1)

    #: The observation hook that the Numba step computes in its kernel; another hook runs after the kernel.
    _numba_observation = _observation

    def _task_tensors(self, agent, target):
        observation = self._observation(agent, target)
        # Match the two-element CPU torch.norm: round x*x, accumulate y*y with
        # one float32 rounding, then round sqrt. Reassociation can flip success at 16 px.
        delta = (agent - target).detach().double()
        first = delta[:, 0].square().float()
        squared = torch.addcmul(first.double(), delta[:, 1], delta[:, 1]).float()
        distance = squared.double().sqrt().float().double()
        n = int(agent.shape[0])
        return {
            "reward": torch.zeros(n, device=self.device, dtype=torch.float64),
            "terminated": distance < 16.0,
            "truncated": torch.zeros(n, device=self.device, dtype=torch.bool),
            "observation": observation,
            "info": {
                "proprio": agent.clone(),
                "state": agent.clone(),
                "goal_state": target.clone(),
                "distance_to_target": distance,
            },
        }

    def _outputs(self, state):
        values = self._task_tensors(state["agent", "position"], state["target", "position"])
        return build_tensordict({"state": state, **values}, state.batch_size, self.device)

    @torch.no_grad()
    def _step(self, action, *, active=None):
        self._check_open("step")
        act = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.B, self.action_dim)
        if self.simulation_backend == "numba":
            return self._numba_step(act, active)
        next_rows = self._advance_rows(self._render_state, act, self._style)
        if active is not None:
            next_rows = torch.where(active[:, None], next_rows, self._render_state)
        self._render_state.copy_(next_rows)
        # Physics already produces valid packed rows. Own the snapshot directly;
        # the public get_state() validation would synchronize CUDA every step.
        return self._outputs(self._full_state(self._render_state))

    def _numba_step(self, action, active):
        """Advance the lanes and build state and task outputs in one Numba call."""
        from . import navstep_cpu

        values = navstep_cpu.task_step(
            self._numba_kind, *self._numba_inputs(action, active), self._target_position.numpy(), self._numba_doors
        )
        agent, target, observation, proprio, state, goal_state, distance, terminated = map(torch.from_numpy, values)
        if getattr(self._observation, "__func__", None) is not type(self)._numba_observation:
            observation = self._observation(agent, target)
        n = self.B
        return build_tensordict(
            {
                "state": {"agent": {"position": agent}, "target": {"position": target}},
                "reward": torch.zeros(n, dtype=torch.float64),
                "terminated": terminated,
                "truncated": torch.zeros(n, dtype=torch.bool),
                "observation": observation,
                "info": {
                    "proprio": proprio,
                    "state": state,
                    "goal_state": goal_state,
                    "distance_to_target": distance,
                },
            },
            [n],
            self.device,
        )


class _TorchNavBase(_GymReferencePort):
    """Shared Gaussian-dot rendering for TwoRoom and Piecewise environments."""

    # Reference geometry and dot radii are defined at this pixel resolution.
    default_size = native_size = IMG
    action_dim = 2
    action_low = -1.0
    action_high = 1.0

    dynamics_variation_leaves = frozenset({"agent.speed"})

    #: Physics implementations that ``simulation_backend`` accepts in addition to ``"auto"``.
    simulation_backends: tuple[str, ...] = ("torch", "numba")
    #: Kernel selector and variation columns of the Numba step.
    _numba_kind = 0
    _numba_paths: tuple[str, ...] = ()

    def __init__(self, *args, simulation_backend="auto", **kwargs):
        """Select the physics implementation; ``"auto"`` uses Numba on CPU when it imports and Torch otherwise."""
        choices = ("auto", *self.simulation_backends)
        if simulation_backend not in choices:
            raise ValueError(f"simulation_backend must be one of {', '.join(map(repr, choices))}")
        super().__init__(*args, **kwargs)
        self._gy, self._gx = _grids(self.device)
        self._border = _border_mask(self.device)
        if simulation_backend == "auto":
            simulation_backend = self._auto_simulation_backend()
        if simulation_backend == "numba":
            if self.device.type != "cpu":
                raise ValueError(f"simulation_backend='numba' needs device='cpu', got {self.device}")
            if not _numba_available():
                raise ImportError("simulation_backend='numba' needs Numba; install stable-envs[navigation]")
            layout = self._style.layout
            self._numba_cols = np.asarray([layout[path][0] for path in self._numba_paths], np.int64)
        self.simulation_backend = simulation_backend
        self.execution_mode = _EXECUTION_MODES[simulation_backend]

    def _auto_simulation_backend(self) -> str:
        return "numba" if self.device.type == "cpu" and _numba_available() else "torch"

    def _numba_inputs(self, action, active):
        """Return the arrays that every Numba step reads; the kernel updates the live rows in place."""
        act = action.numpy()
        if not act.flags.c_contiguous:
            act = np.ascontiguousarray(act)
        mask = _NO_MASK if active is None else active.numpy()
        style = self._style.values.numpy()
        if style.shape[0] != self.B:
            raise ValueError(f"{type(self).__name__}: the installed Variation has {style.shape[0]} rows, not {self.B}")
        return self._render_state.numpy(), act, style, self._numba_cols, mask, active is not None

    def _advance_rows(self, rows, action, sty) -> torch.Tensor:
        numba = self.simulation_backend == "numba" and rows.dtype == action.dtype == torch.float32
        if numba and sty.batch == rows.shape[0] == action.shape[0]:
            from . import navstep_cpu

            out = rows.clone(memory_format=torch.contiguous_format)
            act = np.ascontiguousarray(action.numpy())
            navstep_cpu.advance(
                self._numba_kind, out.numpy(), act, sty.values.numpy(), self._numba_cols, _NO_MASK, False
            )
            return out
        return self._advance_rows_torch(rows, action, sty)

    def _advance_rows_torch(self, rows, action, sty) -> torch.Tensor:
        raise NotImplementedError

    @torch.no_grad()
    def _step(self, action, *, active=None) -> TensorDict:
        if self.simulation_backend != "numba" or self._live_rows != self.B or self.state_dim not in (2, 4):
            return super()._step(action, active=active)
        from . import navstep_cpu

        self._check_open("step")
        act = torch.as_tensor(action, device=self.device, dtype=torch.float32).reshape(self.B, self.action_dim)
        agent, rest, finite = navstep_cpu.state_step(self._numba_kind, *self._numba_inputs(act, active))
        if not finite:
            # get_state names the non-finite lanes and raises.
            return self._step_result(self.state)
        state = {"agent": {"position": torch.from_numpy(agent)}}
        if self.state_dim == 4:
            state["target"] = {"position": torch.from_numpy(rest)}
        return build_tensordict({"state": state}, [self.B], self.device)

    def _agent_space(self) -> dict:
        return {
            "color": RGBBox(init_value=AGENT_C),
            "radius": Box(low=7.0, high=14.0, shape=(1,), init_value=[AGENT_R]),
            "speed": Box(low=0.1, high=50.0, shape=(1,), init_value=[AGENT_SPEED]),
        }

    def base_image(self, sty, n: int) -> torch.Tensor:
        """Return uint8 background and structure ``(n, H, W, 3)`` before dot compositing."""
        raise NotImplementedError

    def scene(self, states, variation=None, style=None):
        raise NotImplementedError(
            f"{type(self).__name__} is not drawn from vector primitives: the reference composites a "
            f"normalised Gaussian alpha mask over the whole frame, which a Scene2D disc cannot express."
        )

    @torch.no_grad()
    def _render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        s = self._render_state if states is None else torch.as_tensor(states, device=self.device, dtype=torch.float32)
        lead = tuple(s.shape[:-1])
        s = s.reshape(-1, self.state_dim)
        n = int(s.shape[0])
        t = int(size or self.size)
        if t != self.native_size:
            raise ValueError(
                f"{type(self).__name__} renders at {self.native_size}px only (got {t}); every length in "
                f"the reference is in output pixels. Resize the frames afterwards instead."
            )
        if supersample not in (None, 1):
            raise ValueError(
                f"{type(self).__name__}.render: supersample={supersample} is not available; the frame "
                f"is an exact per-pixel formula on the reference's pixel lattice."
            )
        sty = self.resolve_style(style, variation, batch=n)

        if out is not None:
            if not out.is_contiguous():
                raise ValueError(
                    f"{type(self).__name__}.render(out=): the destination must be contiguous; a strided "
                    f"view such as frames[:, t] would be written linearly across neighbouring slices."
                )
            img = out.view(n, IMG, IMG, 3)
        else:
            img = torch.empty((n, IMG, IMG, 3), dtype=torch.uint8, device=self.device)

        # Chunks bound intermediate image memory independently of the batch size.
        tuning.on_render("navigation", IMG, n, self)
        chunk = tuning.get("nav_render_chunk", path="navigation", size=IMG, batch=n, device=self.device)
        for lo in range(0, n, chunk):
            dst = img[lo : min(lo + chunk, n)]
            part = self._render_chunk(s, sty, lo, min(lo + chunk, n), out=dst)
            if part is not dst:
                dst.copy_(part)
        return img.view(*lead, IMG, IMG, 3) if lead else img

    def _numba_dots(self) -> bool:
        """Draw CPU dots with the Numba compositor; ``backend="torch"`` selects the Torch path."""
        if self.device.type != "cpu" or self.backend == "torch":
            return False
        from ...render import cpu_raster

        return cpu_raster.available()

    def _render_chunk(self, s, sty, lo: int, hi: int, out=None) -> torch.Tensor:
        k = hi - lo
        sl = lambda p: sty[p][lo:hi]
        base = self._base_image(sty, lo, hi)
        if navdot.available(self.device):
            pos, sig, col = self._dots(s[lo:hi], sl, k)
            return navdot.blend_dots(base.expand(k, IMG, IMG, 3).contiguous(), pos, sig, col)
        if self._numba_dots():
            from . import navdot_cpu

            pos, sig, col = self._dots(s[lo:hi], sl, k)
            return navdot_cpu.blend_dots(base.expand(k, IMG, IMG, 3), pos, sig, col, out=out)
        return self.blend_dots(base, s[lo:hi], sl, k)

    def _dots(self, s, sl, k):
        """Return centers ``(k, ND, 2)``, radii ``(k, ND)`` and colors ``(k, ND, 3)``.

        Dots are listed in compositing order, with the agent last.
        """
        return (s[:, 0:2].reshape(k, 1, 2), sl("agent.radius").reshape(k, 1), sl("agent.color").reshape(k, 1, 3))

    _BAKE_CACHE_MAX = 64
    _BAKE_CACHE = _RES.BoundedCache(_BAKE_CACHE_MAX, name="navigation.bake")

    def _base_image(self, sty, lo: int, hi: int) -> torch.Tensor:
        """Return background rows [lo, hi), reusing images for identical CPU style rows.

        Cache keys include every style leaf so new appearance settings cannot
        reuse a stale image. Treat returned tensors as read-only: they may be
        views of shared cache entries.
        """
        k = hi - lo
        if self.device.type == "cuda":
            # Host-side grouping would synchronize the CUDA render loop.
            return self.base_image(lambda p: sty[p][lo:hi], k)
        rows = sty.values[lo:hi].detach().cpu().contiguous()
        # Compare bytes, as the cache keys do; one comparison replaces the per-row grouping for a shared style.
        raw = rows.view(torch.uint8).reshape(k, -1)
        if k > 1 and bool((raw == raw[:1]).all()):
            blob = raw[0].numpy().tobytes()
            order: dict = {blob: list(range(k))}
        else:
            blob = raw.numpy().tobytes()
            w = blob and len(blob) // k or 0
            keys = [blob[i * w : (i + 1) * w] for i in range(k)] if w else [b""] * k
            order = {}
            for i, key in enumerate(keys):
                order.setdefault(key, []).append(i)
        if len(order) == k and k > 1:
            return self.base_image(lambda p: sty[p][lo:hi], k)

        cls = type(self).__name__
        plates = []
        for key, idx in order.items():
            ck = (cls, int(self.native_size), str(self.device), key)
            hit = _TorchNavBase._BAKE_CACHE.get(ck)
            if hit is None:
                first = lo + idx[0]
                hit = _TorchNavBase._BAKE_CACHE.put(ck, self.base_image(lambda p, a=first: sty[p][a : a + 1], 1))
            plates.append(hit)
        if len(plates) == 1:
            return plates[0].expand(k, IMG, IMG, 3)
        pick = torch.empty(k, dtype=torch.long, device=self.device)
        for g, (_, idx) in enumerate(order.items()):
            pick[torch.as_tensor(idx, device=self.device)] = g
        return torch.cat(plates, dim=0)[pick]

    def blend_dots(self, img, s, sl, k) -> torch.Tensor:
        return self._blend_dot(img, s[:, 0:2], sl("agent.radius"), sl("agent.color"))

    def _blend_dot(self, img, pos, sigma, colour) -> torch.Tensor:
        """Blend a dot within windows where its alpha can affect uint8 pixels.

        Each window must contain the nearest in-frame pixel to the center. That
        keeps the normalization maximum equal to the full-frame maximum.
        """
        from ...render.raster2d import row_groups

        k = int(pos.shape[0])
        if self.device.type == "cuda":
            return _alpha_blend(img, _gaussian_dot(self._gy, self._gx, pos, sigma), colour)
        if self._numba_dots():
            from . import navdot_cpu

            return navdot_cpu.blend_dots(
                img.expand(k, IMG, IMG, 3), pos.reshape(k, 1, 2), sigma.reshape(k, 1), colour.reshape(k, 1, 3)
            )
        win = _dot_window(pos, sigma)
        wins = row_groups(
            win.cpu().numpy(), *(int(v) for v in (win[:, 0].min(), win[:, 1].max(), win[:, 2].min(), win[:, 3].max()))
        )
        if len(wins) == 1 and wins[0][0] is None:
            r0, r1, c0, c1 = wins[0][1:]
            if (r1 - r0) * (c1 - c0) >= IMG * IMG:
                return _alpha_blend(img, _gaussian_dot(self._gy, self._gx, pos, sigma), colour)
        # Always clone: contiguous() may alias a cached image, which must not be painted in place.
        out = img.expand(k, IMG, IMG, 3).clone()
        for rows, r0, r1, c0, c1 in wins:
            if r1 <= r0 or c1 <= c0:
                continue
            gy, gx = self._gy[r0:r1, c0:c1], self._gx[r0:r1, c0:c1]
            if rows is None:
                dot = _gaussian_dot(gy, gx, pos, sigma)
                out[:, r0:r1, c0:c1] = _alpha_blend(out[:, r0:r1, c0:c1], dot, colour)
            else:
                idx = torch.as_tensor(rows, device=out.device)
                dot = _gaussian_dot(gy, gx, pos[idx], sigma.reshape(-1)[idx])
                out[idx, r0:r1, c0:c1] = _alpha_blend(out[idx, r0:r1, c0:c1], dot, colour[idx])
        return out

    def _fill(self, k, color) -> torch.Tensor:
        c = _trunc_color(color).to(torch.uint8).view(k, 1, 1, 3)
        return c.expand(k, IMG, IMG, 3).contiguous()

    def _paint(self, img, mask, color) -> torch.Tensor:
        """Paint color through a shared ``(H, W)`` or batched ``(k, H, W)`` mask."""
        m = mask if mask.dim() == 3 else mask.unsqueeze(0)
        c = _trunc_color(color).to(torch.uint8).view(-1, 1, 1, 3)
        return torch.where(m.unsqueeze(-1), c, img)

    @classmethod
    def reference_set_state(cls, u, st) -> None:
        u.agent_position = torch.as_tensor(np.asarray(st, np.float32)[:2], dtype=torch.float32)

    def _clamp_border(self, pos, radius):
        """Clamp in float64, then cast, matching the source Python-float boundary arithmetic."""
        r = radius.double()
        lo = (float(BORDER) + r).to(torch.float32)
        hi = (float(IMG - BORDER) - r).to(torch.float32)
        return torch.maximum(torch.minimum(pos, hi.unsqueeze(1)), lo.unsqueeze(1))


class _TwoRoomBase(_TorchNavBase):
    reference_id = None
    _numba_paths = (
        "agent.speed",
        "agent.radius",
        "wall.thickness",
        "wall.axis",
        "door.number",
        "door.position",
        "door.size",
    )

    def _structure_space(self) -> dict:
        return {
            "background": Dict({"color": RGBBox(init_value=BG_C)}),
            "wall": Dict(
                {
                    "color": RGBBox(init_value=WALL_C),
                    "thickness": Discrete(35, start=7, init_value=WALL_THICKNESS),
                    "axis": Discrete(2, init_value=WALL_AXIS),
                }
            ),
            "door": Dict(
                {
                    "color": RGBBox(init_value=DOOR_C),
                    "number": Discrete(3, start=1, init_value=N_DOORS),
                    "position": Box(low=0.0, high=223.0, shape=(MAX_DOOR,), init_value=list(DOOR_POS)),
                    "size": Box(low=1.0, high=21.0, shape=(MAX_DOOR,), init_value=list(DOOR_SIZE)),
                }
            ),
        }

    def _masks(self, sl, k):
        """Return door and wall masks ``(k, H, W)``, including the arena border."""
        half = sl("wall.thickness").trunc().long().div(2, rounding_mode="floor").view(-1, 1, 1)
        axis = (sl("wall.axis") > 0.5).view(-1, 1, 1)
        gy = self._gy.unsqueeze(0)
        gx = self._gx.unsqueeze(0)
        lo = (WALL_CENTER - half).to(torch.float32)
        hi = (WALL_CENTER + half).to(torch.float32)
        stripe = torch.where(axis, (gx >= lo) & (gx <= hi), (gy >= lo) & (gy <= hi))

        pos = sl("door.position")
        size = sl("door.size")
        num = sl("door.number").trunc().view(-1, 1, 1)
        # Compute the coordinate along the wall once; duplicating this batch image wastes memory.
        along = torch.where(axis, gy, gx)
        span = torch.zeros(k, IMG, IMG, dtype=torch.bool, device=self.device)
        for i in range(MAX_DOOR):
            c = pos[:, i].view(-1, 1, 1)
            s = size[:, i].view(-1, 1, 1)
            span |= ((along >= (c - s)) & (along <= (c + s))) & (num > i)

        door = stripe & span
        wall = (stripe & ~span) | self._border.unsqueeze(0)
        return door, wall

    _reset_walls: dict | None = None

    def draw_reset_values(self, seed=None, options=None):
        self._reset_walls = {}
        try:
            return super().draw_reset_values(seed, options)
        finally:
            self._reset_walls = None

    def _reset_value_ok(self, path, value, lane):
        """Reject agent positions inside the band that the wall and the agent radius cover."""
        if path != "agent.position":
            return True
        # A reset reads the installed variation once, not once for each lane.
        walls = self._reset_walls
        if not walls:
            walls = {p: self._style[p].detach().cpu().numpy().reshape(-1) for p in _WALL_RESET_PATHS}
            if self._reset_walls is not None:
                self._reset_walls = walls
        axis = int(walls["wall.axis"][lane])
        thickness = int(walls["wall.thickness"][lane])
        radius = float(walls["agent.radius"][lane])
        lo = WALL_CENTER - thickness // 2 - radius
        hi = WALL_CENTER + thickness // 2 + radius
        return not (lo <= float(value[0 if axis == 1 else 1]) <= hi)

    def base_image(self, sl, k) -> torch.Tensor:
        img = self._fill(k, sl("background.color"))
        door, wall = self._masks(sl, k)
        img = self._paint(img, door, sl("door.color"))
        return self._paint(img, wall, sl("wall.color"))

    def _in_any_door(self, coord, sl):
        pos, size, num = sl("door.position"), sl("door.size"), sl("door.number").trunc()
        hit = torch.zeros_like(coord, dtype=torch.bool)
        for i in range(MAX_DOOR):
            inside = (coord >= pos[:, i] - size[:, i] - DOOR_MARGIN) & (coord <= pos[:, i] + size[:, i] + DOOR_MARGIN)
            hit |= inside & (num > i)
        return hit

    def _advance_rows_torch(self, rows, action, sty) -> torch.Tensor:
        """Move the agent through doors or clamp it to the wall face with pushback.

        Use the pre-move position to determine which side of the wall it starts on.
        """
        a = action.clamp(-1.0, 1.0)
        sl = lambda p: sty[p]
        r = sl("agent.radius").reshape(-1)
        pos = rows[:, 0:2]
        nxt = self._clamp_border(pos + a * sty["agent.speed"].reshape(-1, 1), r)

        half = sl("wall.thickness").trunc().long().div(2, rounding_mode="floor")
        axis1 = sl("wall.axis") > 0.5
        c = float(WALL_CENTER)
        rd = r.double()
        eff_lo = (c - half.double() - rd).to(torch.float32)
        eff_hi = (c + half.double() + rd).to(torch.float32)

        # Use x across a vertical wall and y across a horizontal wall; the other axis locates doors.
        moving = torch.where(axis1, nxt[:, 0], nxt[:, 1])
        started = torch.where(axis1, pos[:, 0], pos[:, 1]) < c
        other = torch.where(axis1, nxt[:, 1], nxt[:, 0])
        open_door = self._in_any_door(other, sl)
        block_lo = started & (moving > eff_lo) & ~open_door
        block_hi = (~started) & (moving < eff_hi) & ~open_door
        moved = torch.where(block_lo, eff_lo - WALL_PUSHBACK, torch.where(block_hi, eff_hi + WALL_PUSHBACK, moving))
        new = torch.stack([torch.where(axis1, moved, nxt[:, 0]), torch.where(axis1, nxt[:, 1], moved)], dim=1)
        out = rows.clone()
        out[:, 0:2] = new
        return out

    @staticmethod
    def reference_sim_state(env) -> np.ndarray:
        return np.asarray(_u(env).agent_position.detach().cpu().numpy(), np.float64)


@register
class TwoRoom(_HiddenTargetSemantics, _TwoRoomBase):
    """Two rooms separated by a wall with doors; the reward target is invisible.

    Full state includes the target even though the renderer reads only agent
    position. A compact render row cannot restore the task.
    """

    name = "two-room"
    execution_mode = "NATIVE_JAX"
    state_dim = 2
    state_names = ("agent_x", "agent_y")
    reference_kwargs = {"render_target": False}
    reset_tree_factory = staticmethod(_two_room_reset_tree)
    reset_value_paths = frozenset({"agent.speed"})
    dynamics_variation_leaves = frozenset(
        {
            "agent.speed",
            "agent.radius",
            "wall.thickness",
            "wall.axis",
            "door.number",
            "door.position",
            "door.size",
        }
    )

    simulation_backends = ("torch", "jax", "numba")
    _numba_doors = True

    def __init__(self, *args, simulation_backend="auto", jax_min_envs=1, **kwargs):
        """Select the physics implementation.

        With ``"auto"``, CPU lanes step in Numba when it imports. CUDA lanes step in Torch below
        ``jax_min_envs`` and in JAX at or above it; the cutoff defaults to 1.
        ``simulation_backend="torch"``, ``"jax"`` or ``"numba"`` selects that implementation at
        every batch size; Numba runs on CPU only.
        """
        if isinstance(jax_min_envs, bool) or not isinstance(jax_min_envs, int) or jax_min_envs < 1:
            raise ValueError("jax_min_envs must be an int >= 1")
        self.jax_min_envs = jax_min_envs
        super().__init__(*args, simulation_backend=simulation_backend, **kwargs)
        if self.simulation_backend == "jax":
            from .tworoom_jax import Kernel

            self._jax_kernel = Kernel(self.device, self._style.layout)

    def _auto_simulation_backend(self) -> str:
        if self.device.type == "cpu" and _numba_available():
            return "numba"
        return "jax" if self.B >= self.jax_min_envs else "torch"

    def make_variation_space(self) -> Dict:
        return Dict({"agent": Dict(self._agent_space()), **self._structure_space()})

    def _advance_rows(self, rows, action, sty):
        if self.simulation_backend == "jax":
            return self._jax_kernel.advance(rows, action, sty.values)
        return super()._advance_rows(rows, action, sty)

    def _observation(self, agent, target):
        n = int(agent.shape[0])
        axis = (self._style["wall.axis"][:n] > 0.5).reshape(n, 1)
        count = self._style["door.number"][:n].trunc().reshape(n, 1)
        positions = self._style["door.position"][:n]
        center = torch.full_like(positions, float(WALL_CENTER))
        doors = torch.stack((torch.where(axis, center, positions), torch.where(axis, positions, center)), dim=-1)
        live = torch.arange(MAX_DOOR, device=self.device) < count
        doors = torch.where(live[..., None], doors, 0.0)
        return torch.cat((agent, target, doors.flatten(1)), dim=1)

    _numba_observation = _observation

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample agent positions in [14, 209]^2, rejecting positions inside the central wall."""
        rng = rng or self.rng
        n = int(batch)
        lo = WALL_CENTER - WALL_THICKNESS // 2 - AGENT_R
        hi = WALL_CENTER + WALL_THICKNESS // 2 + AGENT_R
        xs = np.empty(n, np.float32)
        filled = 0
        while filled < n:
            cand = rng.uniform(POS_LO, POS_HI, n).astype(np.float32)
            cand = cand[(cand < lo) | (cand > hi)]
            take = min(n - filled, cand.size)
            xs[filled : filled + take] = cand[:take]
            filled += take
        ys = rng.uniform(POS_LO, POS_HI, n).astype(np.float32)
        return np.stack([xs, ys], axis=1)


@register
class TwoRoomTarget(_TwoRoomBase):
    """TwoRoom renderer with a visible target stored in the compact pose."""

    name = "two-room-target"
    state_dim = 4
    state_names = ("agent_x", "agent_y", "target_x", "target_y")
    reference_kwargs = {"render_target": True}
    state_layout = ((("agent", "position"), 2, 0), (("target", "position"), 2, 2))
    state_bounds = {("agent", "position"): (POS_LO, POS_HI), ("target", "position"): (POS_LO, POS_HI)}
    step_reference_groups = (("agent", (0, 1), "euclidean"), ("target", (2, 3), "euclidean"))
    # Seeded resets draw from the same tree and rejection rule as TwoRoom, so they start where
    # stable-worldmodel's TwoRoom does.
    reset_leaves = (
        ResetLeaf("agent.position", (60.0, 112.0)),
        ResetLeaf("target.position", (164.0, 112.0)),
    )
    reset_tree_factory = staticmethod(_two_room_reset_tree)

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        self._reseed(seed)
        values = self.draw_reset_values(seed, dict(options or {}))
        agent, target = np.stack(list(values["agent.position"])), np.stack(list(values["target.position"]))
        self._set_render_state(np.concatenate([agent, target], axis=1))
        self._live_rows = self.B
        return self.get_state()

    def make_variation_space(self) -> Dict:
        return Dict(
            {
                "agent": Dict(self._agent_space()),
                "target": Dict(
                    {
                        "color": RGBBox(init_value=TARGET_C),
                        "radius": Box(low=7.0, high=14.0, shape=(1,), init_value=[TARGET_R]),
                    }
                ),
                **self._structure_space(),
            }
        )

    def _step(self, action, *, active=None) -> TensorDict:
        return super()._step(action, active=active)

    def _dots(self, s, sl, k):
        return (
            torch.cat([s[:, 2:4].reshape(k, 1, 2), s[:, 0:2].reshape(k, 1, 2)], dim=1),
            torch.cat([sl("target.radius").reshape(k, 1), sl("agent.radius").reshape(k, 1)], dim=1),
            torch.cat([sl("target.color").reshape(k, 1, 3), sl("agent.color").reshape(k, 1, 3)], dim=1),
        )

    def blend_dots(self, img, s, sl, k) -> torch.Tensor:
        """Blend the target before the agent, quantizing to uint8 between blends."""
        img = self._blend_dot(img, s[:, 2:4], sl("target.radius"), sl("target.color"))
        return super().blend_dots(img, s, sl, k)

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        rng = rng or self.rng
        agent = TwoRoom.sample_states(self, batch, rng)
        target = rng.uniform(POS_LO, POS_HI, (int(batch), 2)).astype(np.float32)
        return np.concatenate([agent, target], axis=1)

    @staticmethod
    def reference_sim_state(env) -> np.ndarray:
        u = _u(env)
        return np.concatenate(
            [u.agent_position.detach().cpu().numpy(), u.target_position.detach().cpu().numpy()]
        ).astype(np.float64)

    @classmethod
    def reference_set_state(cls, u, st) -> None:
        v = np.asarray(st, np.float32)
        u.agent_position = torch.as_tensor(v[:2], dtype=torch.float32)
        u.target_position = torch.as_tensor(v[2:4], dtype=torch.float32)


class _PiecewiseBase(_TorchNavBase):
    reference_id = None
    _numba_kind = 1
    _numba_paths = ("agent.speed", "agent.radius", *(f"zones.bias_{i}" for i in range(N_ZONES)))
    state_dim = 2
    state_names = ("agent_x", "agent_y")
    state_layout = ((("agent", "position"), 2, 0),)
    state_bounds = {("agent", "position"): (POS_LO, POS_HI)}
    dynamics_variation_leaves = frozenset(
        {"agent.speed", "agent.radius", *(f"zones.bias_{i}" for i in range(N_ZONES))}
    )

    #: Discrete rendering flags that need the layered compositing path when any lane sets them.
    layer_flags: tuple[str, ...] = ("rendering.render_bias_field",)
    _layered = False
    _FIELD_CACHE = _RES.BoundedCache(16, name="navigation.bias_field")

    def _common_space(self) -> dict:
        return {
            "agent": Dict(self._agent_space()),
            "background": Dict({"color": RGBBox(init_value=BG_C)}),
            "border": Dict({"color": RGBBox(init_value=BORDER_C)}),
            "rendering": Dict({flag.split(".")[1]: Discrete(2, init_value=0) for flag in self.layer_flags}),
        }

    def _configure_variation(self, exact=None) -> None:
        super()._configure_variation(exact)
        self._layered = any(bool((self._style[flag] > 0.5).any()) for flag in self.layer_flags)

    def _render_chunk(self, s, sty, lo: int, hi: int, out=None) -> torch.Tensor:
        # The installed variation's flags are cached on the host, so the default path never synchronizes.
        if sty is self._style:
            layered = self._layered
        else:
            layered = any(bool((sty[flag][lo:hi] > 0.5).any()) for flag in self.layer_flags)
        if not layered:
            return super()._render_chunk(s, sty, lo, hi, out=out)
        k = hi - lo
        sl = lambda p: sty[p][lo:hi]
        img = self._base_image(sty, lo, hi).expand(k, IMG, IMG, 3)
        img = self._target_layer(img, s, sl, lo, hi)
        on = (sl("rendering.render_bias_field") > 0.5).view(k, 1, 1, 1)
        img = torch.where(on, self._bias_layer(img, sl, k), img)
        return self._blend_dot(img, s[lo:hi, 0:2], sl("agent.radius"), sl("agent.color"))

    def _target_layer(self, img, s, sl, lo, hi):
        return img

    def _bias_layer(self, img, sl, k) -> torch.Tensor:
        """Composite each lane's bias arrows over ``img`` in float32, truncating to uint8."""
        biases = torch.stack([sl(f"zones.bias_{i}").reshape(k, 2) for i in range(N_ZONES)], dim=1)
        host = biases.detach().cpu().numpy().astype(np.float32)
        layers = []
        for lane in range(k):
            key = (str(self.device), host[lane].tobytes())
            hit = _PiecewiseBase._FIELD_CACHE.get(key)
            if hit is None:
                rgba = torch.as_tensor(_bias_field_rgba(host[lane]))
                alpha = rgba[..., 3:].to(torch.float32) / 255.0
                hit = _PiecewiseBase._FIELD_CACHE.put(
                    key, (alpha.to(self.device), rgba[..., :3].to(torch.float32).to(self.device))
                )
            layers.append(hit)
        alpha = torch.stack([a for a, _ in layers])
        overlay = torch.stack([o for _, o in layers])
        return (img.to(torch.float32) * (1.0 - alpha) + overlay * alpha).to(torch.uint8)

    def _zone_bias_space(self) -> dict:
        return {
            f"bias_{i}": Box(low=-10.0, high=10.0, shape=(2,), init_value=list(ZONE_BIAS[i])) for i in range(N_ZONES)
        }

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample agent positions uniformly from [14, 209]^2."""
        rng = rng or self.rng
        return rng.uniform(POS_LO, POS_HI, (int(batch), 2)).astype(np.float32)

    def _advance_rows_torch(self, rows, action, sty) -> torch.Tensor:
        """Apply the current zone bias to the action, then clamp the result to the arena."""
        a = action.clamp(-1.0, 1.0)
        r = sty["agent.radius"].reshape(-1)
        pos = rows[:, 0:2]
        play = float(IMG - 2 * BORDER)
        # Compute zone indices in float64 like the source Python floats; rounding can select another zone.
        offset = pos.double() - float(BORDER)
        # Keep a tensor denominator: CUDA scalar division may use a rounded reciprocal at zone boundaries.
        denominator = torch.full_like(offset, play)
        idx = (offset / denominator * GRID_N).trunc().long().clamp(0, GRID_N - 1)
        zone = idx[:, 1] * GRID_N + idx[:, 0]
        n = int(pos.shape[0])
        biases = torch.stack([sty[f"zones.bias_{i}"].reshape(n, 2) for i in range(N_ZONES)], dim=1)
        bias = biases[torch.arange(n, device=pos.device), zone]
        nxt = pos + a * sty["agent.speed"].reshape(n, 1) + bias
        out = rows.clone()
        out[:, 0:2] = self._clamp_border(nxt, r)
        return out

    @staticmethod
    def reference_sim_state(env) -> np.ndarray:
        return np.asarray(_u(env).agent_position.detach().cpu().numpy(), np.float64)


@register
class Piecewise(_HiddenTargetSemantics, _PiecewiseBase):
    """Open room with four invisible motion-bias zones and an invisible reward target."""

    name = "piecewise"
    # Zone tint is a variation leaf, not a PiecewiseEnv constructor argument.
    reference_kwargs: dict = {}
    reset_tree_factory = staticmethod(_piecewise_reset_tree)
    reset_value_paths = frozenset(
        {
            "agent.speed",
            "target.color",
            "target.radius",
            "rendering.render_target",
            "rendering.render_bias_field",
            *(f"zones.bias_{i}" for i in range(N_ZONES)),
        }
    )

    layer_flags = ("rendering.render_target", "rendering.render_bias_field")

    def make_variation_space(self) -> Dict:
        target = {
            "color": RGBBox(init_value=TARGET_C),
            "radius": Box(low=7.0, high=14.0, shape=(1,), init_value=[TARGET_R]),
        }
        return Dict({**self._common_space(), "target": Dict(target), "zones": Dict(self._zone_bias_space())})

    def _step(self, action, *, active=None) -> TensorDict:
        return super()._step(action, active=active)

    def base_image(self, sl, k) -> torch.Tensor:
        return self._paint(self._fill(k, sl("background.color")), self._border, sl("border.color"))

    _frame_targets = None

    def _render_state_input(self, state, **kwargs) -> torch.Tensor:
        """Render full state; the target, hidden from the compact rows, is drawn from it when enabled."""
        state = self.get_state() if state is None else state
        self._frame_targets = state.get(("target", "position"), None)
        try:
            return super()._render_state_input(state, **kwargs)
        finally:
            self._frame_targets = None

    def _drawn_fields(self):
        return (*super()._drawn_fields(), (("target", "position"), 2))

    def _target_layer(self, img, s, sl, lo, hi):
        targets = self._frame_targets
        if targets is None:
            if int(s.shape[0]) != self._live_rows or s.data_ptr() != self._get_render_state().data_ptr():
                raise ValueError(
                    "Piecewise draws rendering.render_target from the hidden target position, which "
                    "compact rows do not carry; render a full State instead"
                )
            targets = self._target_position[: self._live_rows]
        pos = torch.as_tensor(targets, device=self.device, dtype=torch.float32).reshape(-1, 2)[lo:hi]
        on = (sl("rendering.render_target") > 0.5).view(-1, 1, 1, 1)
        return torch.where(on, self._blend_dot(img, pos, sl("target.radius"), sl("target.color")), img)


@register
class PiecewiseZones(_PiecewiseBase):
    """Piecewise renderer with a distinct tint for each motion-bias zone."""

    name = "piecewise-zones"
    # Seeded resets draw the agent from the same tree as Piecewise, so they start where
    # stable-worldmodel's Piecewise does.
    reset_leaves = (ResetLeaf("agent.position", (60.0, 112.0)),)
    reset_tree_factory = staticmethod(_piecewise_reset_tree)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._zones = _zone_map(self.device)

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        self._reseed(seed)
        values = self.draw_reset_values(seed, dict(options or {}))
        self._set_render_state(np.stack(list(values["agent.position"])))
        self._live_rows = self.B
        return self.get_state()

    def make_variation_space(self) -> Dict:
        z = {**self._zone_bias_space(), **{f"color_{i}": RGBBox(init_value=ZONE_C[i]) for i in range(N_ZONES)}}
        return Dict({**self._common_space(), "zones": Dict(z)})

    def _step(self, action, *, active=None) -> TensorDict:
        return super()._step(action, active=active)

    def base_image(self, sl, k) -> torch.Tensor:
        img = self._fill(k, sl("background.color"))
        for i in range(N_ZONES):
            img = self._paint(img, self._zones == i, sl(f"zones.color_{i}"))
        return self._paint(img, self._border, sl("border.color"))


# Reference figure size is 5×5 inches at 100 dpi.
SIZE = 500
WORLD = 5.0
PX_PER_UNIT = SIZE / WORLD
MAX_WALLS = 6
MIN_WALLS = 4

MAZE_AGENT_C = (255, 0, 0)
MAZE_GOAL_C = (0, 255, 0)
MAZE_WALL_C = (0, 0, 0)
MAZE_BG_C = (255, 255, 255)
MAZE_AGENT_R = 0.1
MAZE_GOAL_R = 0.2
MAZE_GOAL_POS = (4.5, 4.5)
N_WALLS = 5
ACTION_LIMIT = 0.2
MAZE_AGENT_SPEED = 1.0

# One output pixel in world units, used by the fixed-resolution axes border.
PX = WORLD / SIZE
# Fractional darkening of the antialiased axes border against the background.
SPINE_AA = 14.0 / 255.0


def _default_walls() -> tuple[np.ndarray, np.ndarray]:
    """Generate the source default wall sizes and positions.

    Draw sizes before positions from the same seeded RNG; swapping the draws
    changes the maze.
    """
    rng = np.random.default_rng(234232)
    shape = rng.uniform(low=0.5, high=1.5, size=(MAX_WALLS, 2)).astype(np.float32)
    pos = rng.uniform(low=1.0, high=3.5, size=(MAX_WALLS, 2)).astype(np.float32)
    return shape, pos


DEFAULT_WALL_SHAPE, DEFAULT_WALL_POS = _default_walls()
_GOAL_IMAGE_PATHS = (
    "background.color",
    "walls.number",
    "walls.shape",
    "walls.positions",
    "walls.color",
    "goal.radius",
    "goal.color",
    "agent.radius",
    "agent.color",
)


@register
class SimplePointMaze(_GymReferencePort):
    """Point navigation in a 5×5 arena with rectangular walls and a visible reward goal."""

    name = "simple-point-maze"
    execution_mode = "NATIVE_TORCH"
    _step_outputs_owned = True
    # The goal image is replaced at reset and restore and never written, so steps share it.
    _shared_outputs = (("info", "goal"),)
    state_dim = 4
    state_names = ("agent_x", "agent_y", "goal_x", "goal_y")
    default_size = native_size = SIZE
    reference_id = None
    reference_kwargs = {"show_goal": True}
    reference_render_args = ("rgb_array",)
    action_dim = 2
    action_low = -ACTION_LIMIT
    action_high = ACTION_LIMIT
    state_layout = (
        (("agent", "position"), 2, 0, torch.float64),
        (("goal", "position"), 2, 2),
        (("task", "cached_goal_agent_position"), 2, None),
        (("task", "state_is_float64"), 1, None, torch.bool),
    )
    state_bounds = {
        ("agent", "position"): (0.0, WORLD),
        ("goal", "position"): (0.0, WORLD),
        ("task", "cached_goal_agent_position"): (0.0, WORLD),
        ("task", "state_is_float64"): (0, 1),
    }

    reset_tree_factory = staticmethod(_simple_maze_reset_tree)
    reset_leaves = (
        ResetLeaf("agent.position", (0.5, 0.5)),
        ResetLeaf("goal.position", MAZE_GOAL_POS, sampled=False),
    )
    reset_value_paths = frozenset({"agent.speed", "goal.radius"})
    color_exactness_applicability = {
        "leaves": ("goal.color",),
        "mechanism": "the source draws the goal as a matplotlib Circle with alpha=0.5, which Agg "
        "composites as floor(bg + (c - bg) * 128/255) over the background",
        "max_shading_factor": 128 / 255,
        "official_renderer_fails_the_same_check": True,
        "official_renderer_evidence": "on the default white background the official goal centre is "
        "[135, 143, 227] for [17, 33, 200] and [227, 135, 143] for [200, 17, 33], the composite above",
        "evidence": ("tests/unit/test_envs_classic_nav.py::test_simple_point_maze_goal_is_an_agg_alpha_composite",),
    }
    dynamics_variation_leaves = frozenset(
        {
            "agent.speed",
            "agent.radius",
            "goal.radius",
            "walls.number",
            "walls.shape",
            "walls.positions",
        }
    )

    def make_variation_space(self) -> Dict:
        return Dict(
            {
                "agent": Dict(
                    {
                        "color": RGBBox(init_value=MAZE_AGENT_C),
                        "radius": Box(low=0.05, high=0.5, shape=(1,), init_value=[MAZE_AGENT_R]),
                        "speed": Box(low=0.05, high=8.0, shape=(1,), init_value=[MAZE_AGENT_SPEED]),
                    }
                ),
                "goal": Dict(
                    {
                        "color": RGBBox(init_value=MAZE_GOAL_C),
                        "radius": Box(low=0.05, high=0.5, shape=(1,), init_value=[MAZE_GOAL_R]),
                    }
                ),
                "walls": Dict(
                    {
                        "number": Discrete(MAX_WALLS - MIN_WALLS + 1, start=MIN_WALLS, init_value=N_WALLS),
                        "color": RGBBox(init_value=MAZE_WALL_C),
                        "shape": Box(low=0.5, high=1.5, shape=(MAX_WALLS, 2), init_value=DEFAULT_WALL_SHAPE),
                        "positions": Box(low=0.0, high=WORLD, shape=(MAX_WALLS, 2), init_value=DEFAULT_WALL_POS),
                    }
                ),
                "background": Dict({"color": RGBBox(init_value=MAZE_BG_C)}),
            }
        )

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._cached_goal_agent_position = torch.full((self.B, 2), 0.5, device=self.device, dtype=torch.float32)
        self._agent_position = torch.full((self.B, 2), 0.5, device=self.device, dtype=torch.float64)
        self._state_is_float64 = torch.zeros(self.B, 1, device=self.device, dtype=torch.bool)
        self._cached_goal_image = torch.zeros(self.B, SIZE, SIZE, 3, device=self.device, dtype=torch.uint8)
        self._goal_painter = GoalPainter(WORLD)
        self._drawing_reset, self._reset_walls = False, None

    def _hidden_state(self):
        return {
            ("task", "cached_goal_agent_position"): self._cached_goal_agent_position[: self._live_rows],
            ("task", "state_is_float64"): self._state_is_float64[: self._live_rows],
        }

    def _live_state(self):
        return {("agent", "position"): self._agent_position[: self._live_rows]}

    def _restore_state(self, state):
        self._set_render_state(self._pack_render_state(state))
        self._cached_goal_agent_position.copy_(
            torch.as_tensor(
                state["task", "cached_goal_agent_position"],
                device=self.device,
                dtype=torch.float32,
            ).reshape(self.B, 2)
        )
        state_is_float64 = torch.as_tensor(
            state["task", "state_is_float64"], device=self.device, dtype=torch.bool
        ).reshape(self.B, 1)
        agent = torch.as_tensor(state["agent", "position"], device=self.device, dtype=torch.float64).reshape(self.B, 2)
        agent = torch.where(state_is_float64, agent, agent.to(torch.float32).to(torch.float64))
        self._agent_position.copy_(agent)
        self._render_state[:, 0:2].copy_(agent.to(torch.float32))
        self._state_is_float64.copy_(state_is_float64)
        self._regenerate_cached_goal()

    def _set_render_state(self, states, variation=None, style=None):
        super()._set_render_state(states, variation=variation, style=style)
        n = self._live_rows
        self._agent_position[:n].copy_(self._render_state[:n, 0:2].to(torch.float64))
        self._state_is_float64[:n].zero_()
        return self

    def _wall_rows(self):
        """Host copies of the wall and radius leaves, read once per reset draw."""
        rows = self._reset_walls
        if rows is None:
            rows = {
                path: self._style[path].detach().cpu().numpy().reshape(self.B, -1)
                for path in ("walls.number", "walls.positions", "walls.shape", "agent.radius", "goal.radius")
            }
            if self._drawing_reset:
                self._reset_walls = rows
        return rows

    def draw_reset_values(self, seed=None, options=None):
        self._drawing_reset, self._reset_walls = True, None
        try:
            return super().draw_reset_values(seed, options)
        finally:
            self._drawing_reset, self._reset_walls = False, None

    def _position_collides(self, value, lane, entity="agent"):
        p = np.asarray(value, np.float32)
        rows = self._wall_rows()
        number = int(rows["walls.number"][lane, 0])
        positions = rows["walls.positions"][lane].reshape(MAX_WALLS, 2)
        shapes = rows["walls.shape"][lane].reshape(MAX_WALLS, 2)
        radius = float(rows[f"{entity}.radius"][lane, 0])
        for corner, shape in zip(positions[:number], shapes[:number], strict=True):
            lo = np.minimum(corner, corner + shape)
            hi = np.maximum(corner, corner + shape)
            closest = np.clip(p, lo, hi)
            if float(np.square(closest - p).sum()) <= radius**2:
                return True
        return False

    def _reset_value_ok(self, path, value, lane):
        if path in ("agent.position", "goal.position"):
            return not self._position_collides(value, lane, path.split(".")[0])
        return True

    def _draw_reset_lane(self, tree, lane, sampled, values):
        out = super()._draw_reset_lane(tree, lane, sampled, values)
        leaf = tree["agent"]["position"]
        for _ in range(1000):
            candidate = leaf.sample()
            if not self._position_collides(candidate, lane):
                out["task.cached_goal_agent_position"] = candidate
                return out
        raise RuntimeError("agent.position rejection sampling did not converge")

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        values = self.draw_reset_values(seed, options)
        starts, goals = values["agent.position"], values["goal.position"]
        self._set_render_state(np.concatenate([starts, goals], axis=1))
        self._agent_position.copy_(torch.as_tensor(starts, device=self.device, dtype=torch.float64))
        self._state_is_float64.zero_()
        self._cached_goal_agent_position.copy_(
            torch.as_tensor(values["task.cached_goal_agent_position"], device=self.device)
        )
        self._live_rows = self.B
        self._regenerate_cached_goal()
        return self.get_state()

    def _regenerate_cached_goal(self):
        """Rebuild the source-compatible ``info.goal`` image from state and variation.

        Use Matplotlib Agg for this cached task image; stepping and normal
        rendering use the native implementation. Step outputs share the image,
        so replace it and never write into it.
        """
        n = self.B
        style = {path: self._style[path].detach().cpu().numpy() for path in _GOAL_IMAGE_PATHS}
        rows = dict(
            background=style["background.color"] / 255.0,
            number=style["walls.number"].reshape(n),
            shapes=style["walls.shape"].reshape(n, MAX_WALLS, 2),
            positions=style["walls.positions"].reshape(n, MAX_WALLS, 2),
            wall_color=style["walls.color"] / 255.0,
            goal=self._render_state[:n, 2:4].detach().cpu().numpy(),
            goal_radius=style["goal.radius"].reshape(n),
            goal_color=style["goal.color"] / 255.0,
            agent=self._cached_goal_agent_position[:n].detach().cpu().numpy(),
            agent_radius=style["agent.radius"].reshape(n),
            agent_color=style["agent.color"] / 255.0,
        )
        images = goal_frames(self._goal_painter, rows, SIZE)
        self._cached_goal_image = torch.as_tensor(images, device=self.device)

    def scene(self, states, variation: dict | None = None, style=None) -> Scene2D:
        s = torch.as_tensor(states, device=self.device, dtype=torch.float32).reshape(-1, 4)
        n = int(s.shape[0])
        sty = self.resolve_style(style, variation, batch=n)
        sb = SceneBuilder(n, device=self.device, vertices=4)

        bg = sty["background.color"]
        shape = sty["walls.shape"].reshape(n, MAX_WALLS, 2)
        pos = sty["walls.positions"].reshape(n, MAX_WALLS, 2)
        num = sty["walls.number"].trunc()
        wcol = sty["walls.color"]
        for i in range(MAX_WALLS):
            # Keep disabled wall slots so primitive count stays fixed; rectangles use center coordinates.
            sb.add_rect(pos[:, i] + 0.5 * shape[:, i], shape[:, i], wcol, enabled=num > i)
        # Static caching is valid only when wall appearance is shared across the batch.
        sb.mark_static()

        # Agg rounds alpha=.5 to 128/255. The goal avoids walls, so pre-blend against the background.
        bg64 = bg.to(torch.float64)
        goal64 = sty["goal.color"].to(torch.float64)
        goal_blend = torch.floor(bg64 + (goal64 - bg64) * ((128.0 - 1e-9) / 255.0)).to(torch.float32)
        sb.add_disc(s[:, 2:4], sty["goal.radius"], goal_blend)
        sb.add_disc(s[:, 0:2], sty["agent.radius"], sty["agent.color"])

        # Draw the axes border last, above walls. The gray top band is row 1, below the black row 0.
        grey = bg * (1.0 - SPINE_AA)
        edges = (
            (0.0, WORLD - 2 * PX, WORLD, WORLD - PX, grey),  # row 1
            (0.0, 0.0, WORLD, PX, grey),  # row 499
            (PX, 0.0, 2 * PX, WORLD, grey),  # col 1
            (WORLD - PX, 0.0, WORLD, WORLD, grey),  # col 499
            (0.0, WORLD - PX, WORLD, WORLD, None),  # row 0   (black)
            (0.0, 0.0, PX, WORLD, None),
        )  # col 0   (black)
        black = torch.zeros(n, 3, device=self.device, dtype=torch.float32)
        for x0, y0, x1, y1, col in edges:
            sb.add_rect(((x0 + x1) * 0.5, (y0 + y1) * 0.5), (x1 - x0, y1 - y0), black if col is None else col)
        return sb.build(background=bg, world=(0.0, 0.0, WORLD, WORLD), flip_y=True)

    def _render(
        self,
        states=None,
        size: int | None = None,
        supersample: int | None = None,
        variation: dict | None = None,
        style=None,
        out=None,
    ) -> torch.Tensor:
        """Render at 500×500 only; the reference axes borders are defined in output pixels."""
        t = int(size or self.size)
        if t != SIZE:
            raise ValueError(
                f"SimplePointMaze renders at {SIZE}px only (got {t}): the reference is a 5x5 inch "
                f"matplotlib figure at 100 dpi whose axes spine is a fixed width in output pixels."
            )
        return super()._render(states, size=t, supersample=supersample, variation=variation, style=style, out=out)

    def _wall_boxes(self, sty, n):
        shape = sty["walls.shape"].reshape(n, MAX_WALLS, 2)
        pos = sty["walls.positions"].reshape(n, MAX_WALLS, 2)
        num = sty["walls.number"].trunc().view(n, 1)
        lo = torch.minimum(pos, pos + shape)
        hi = torch.maximum(pos, pos + shape)
        live = torch.arange(MAX_WALLS, device=self.device).view(1, -1) < num
        return lo, hi, live

    def _collides(self, p, sty, n) -> torch.Tensor:
        """Return whether each agent disc touches a live wall, using closest-point box distances."""
        lo, hi, live = self._wall_boxes(sty, n)
        r = sty["agent.radius"].reshape(n, 1)
        c = torch.clamp(p.unsqueeze(1), lo, hi)
        d2 = ((c - p.unsqueeze(1)) ** 2).sum(-1)
        return (live & (d2 <= r * r)).any(1)

    def sample_states(self, batch: int, rng: np.random.Generator | None = None) -> np.ndarray:
        """Sample positions uniformly from [0, 5]^2, rejecting discs that touch walls."""
        rng = rng or self.rng
        n = int(batch)
        sty = self.resolve_style(batch=1)
        out = np.empty((n, 2), np.float32)
        filled = 0
        for _ in range(1000):
            if filled >= n:
                break
            cand = rng.uniform(0.0, WORLD, (n, 2)).astype(np.float32)
            t = torch.as_tensor(cand, device=self.device)
            ok = ~self._collides(t, sty.for_batch(n, owner="SimplePointMaze.sample_states"), n)
            keep = cand[ok.cpu().numpy()]
            take = min(n - filled, len(keep))
            out[filled : filled + take] = keep[:take]
            filled += take
        if filled < n:
            raise RuntimeError(
                "agent.position rejection sampling did not converge; the wall layout leaves no free space"
            )
        goal = np.broadcast_to(np.asarray(MAZE_GOAL_POS, np.float32), (n, 2)).copy()
        return np.concatenate([out, goal], axis=1)

    def _advance_rows(self, rows, action, sty) -> torch.Tensor:
        """Move, reject collisions, then clamp to the arena.

        Check collisions before clamping the candidate position; reversing that
        order can allow movement along walls that the source rejects.
        """
        a = action.clamp(-ACTION_LIMIT, ACTION_LIMIT)
        pos = rows[:, 0:2]
        nxt = pos + sty["agent.speed"].reshape(-1, 1) * a
        blocked = self._collides(nxt, sty, int(rows.shape[0]))
        nxt = torch.where(blocked.unsqueeze(1), pos, nxt).clamp(0.0, WORLD)
        out = rows.clone()
        out[:, 0:2] = nxt
        return out

    def _outputs(self, state):
        agent = state["agent", "position"]
        goal = state["goal", "position"]
        was64 = state["task", "state_is_float64"].reshape(-1)
        distance32 = torch.linalg.vector_norm(agent.to(torch.float32) - goal, dim=1).to(torch.float64)
        distance64 = torch.linalg.vector_norm(agent - goal.to(torch.float64), dim=1)
        distance = torch.where(was64, distance64, distance32)
        terminated = distance < self._style["goal.radius"][: len(agent)].reshape(-1).to(torch.float64)
        n = int(state.batch_size[0])
        reward = torch.where(
            terminated,
            torch.ones(n, device=self.device, dtype=torch.float64),
            torch.full((n,), -0.01, device=self.device, dtype=torch.float64),
        )
        observation = agent.clone() if bool(state["task", "state_is_float64"].any()) else agent.to(torch.float32)
        return StepResult(
            {
                "state": state,
                "reward": reward,
                "terminated": terminated,
                "truncated": torch.zeros(n, device=self.device, dtype=torch.bool),
                "observation": observation,
                "info": {"goal": self._cached_goal_image if n == self.B else self._cached_goal_image[:n]},
            },
            batch_size=[n],
            device=self.device,
        )

    def _advance_agent(self, agent, action, sty, state_is_float64):
        """Advance with the source NumPy dtype-promotion rules for float16/32/64 actions."""
        action_tensor = torch.as_tensor(action, device=self.device)
        if not action_tensor.is_floating_point():
            action_tensor = action_tensor.to(torch.float64)
        if action_tensor.dtype not in (torch.float16, torch.float32, torch.float64):
            action_tensor = action_tensor.to(torch.float64)
        action_tensor = action_tensor.reshape(-1, self.action_dim)
        use64 = state_is_float64.reshape(-1) | (action_tensor.dtype == torch.float64)
        limit = float(np.float32(ACTION_LIMIT))

        def advance(dtype):
            p = agent.to(dtype)
            a = action_tensor.to(dtype).clamp(-limit, limit)
            speed = sty["agent.speed"].reshape(-1, 1).to(dtype)
            nxt = p + speed * a
            blocked = self._collides(nxt, sty, int(p.shape[0]))
            return torch.where(blocked.unsqueeze(1), p, nxt).clamp(0.0, WORLD), blocked

        value32, _ = advance(torch.float32)
        value64, blocked64 = advance(torch.float64)
        value = torch.where(use64.unsqueeze(1), value64, value32.to(torch.float64))
        action_promotes = action_tensor.dtype == torch.float64
        next_is_float64 = state_is_float64.reshape(-1) | (action_promotes & ~blocked64)
        return value, next_is_float64.reshape(-1, 1)

    def _step(self, action, *, active=None) -> TensorDict:
        self._check_open("step")
        agent, state_is_float64 = self._advance_agent(
            self._agent_position, action, self._style, self._state_is_float64
        )
        if active is not None:
            agent = torch.where(active[:, None], agent, self._agent_position)
            state_is_float64 = torch.where(active[:, None], state_is_float64, self._state_is_float64)
        self._agent_position.copy_(agent)
        self._state_is_float64.copy_(state_is_float64)
        self._render_state[:, 0:2].copy_(agent.to(torch.float32))
        return self._outputs(self.get_state())

    @classmethod
    def reference_set_state(cls, u, st) -> None:
        value = np.asarray(st, np.float32).reshape(4)
        u.state = value[:2].copy()
        u.variation_space["agent"]["position"]._value = value[:2].copy()
        u.variation_space["goal"]["position"]._value = value[2:4].copy()

    @staticmethod
    def reference_sim_state(env) -> np.ndarray:
        u = getattr(env, "unwrapped", env)
        return np.concatenate(
            [np.asarray(u.state, np.float32), np.asarray(u.variation_space["goal"]["position"].value, np.float32)]
        ).astype(np.float64)
