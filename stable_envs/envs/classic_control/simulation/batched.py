"""Classic-control dynamics stepped for every lane at once, in NumPy or in Torch.

Gymnasium environments still own resets, the PCG64 streams and the physics values, so seeded resets
and State export match :class:`ClassicControlAdapter`. Steps repeat each source ``step`` in float64,
in the same operation order. Where the source computes in float32 (a float32 action, or the float32
state of MountainCarContinuous), the kernel rounds through float32 after each such operation. Float64
has more than twice the float32 precision, so each rounded result is the float32 result.

NumPy evaluates float64 ``sin`` and ``cos`` of an array with the same results as for the source's
scalars. ``x ** 2`` of a source scalar calls the C library ``pow``, which can differ from ``x * x`` in
the last place, so the NumPy kernels call ``pow`` and ``powf`` too. NumPy steps therefore equal the
source steps, and the native contract tests check this. Torch computes these functions with its own
code on each device, so Torch steps agree with the source to rounding.
"""

from __future__ import annotations

import math
from typing import Any

import numpy as np
import torch
from tensordict import TensorDict

from ...._core.tensordicts import build_tensordict
from .local import _PHYSICS_SPACES, ClassicControlAdapter, physics_state_key

#: Time limits that ``gymnasium.make`` adds for each registered id.
MAX_EPISODE_STEPS = {
    "cartpole": 500,
    "pendulum": 200,
    "acrobot": 500,
    "mountain_car": 200,
    "mountain_car_continuous": 999,
}

_ELAPSED = ("lifecycle", "elapsed_steps")
_DYNAMIC = {
    "cartpole": (
        "x",
        "x_dot",
        "theta",
        "theta_dot",
        ("task", "steps_beyond_terminated"),
        ("task", "has_steps_beyond_terminated"),
    ),
    "pendulum": ("theta", "theta_dot", "torque", ("task", "has_last_u")),
    "acrobot": ("theta1", "theta2", "dtheta1", "dtheta2"),
    "mountain_car": ("position", "velocity"),
    "mountain_car_continuous": ("position", "velocity", ("task", "state_is_float32")),
}
_DISCRETE = {"cartpole": 2, "acrobot": 3, "mountain_car": 3}


def _f32(value: float) -> float:
    return float(np.float32(value))


#: Float32 roundings of the MountainCarContinuous bounds, for lanes that hold a float32 state.
_F32_RIGHT, _F32_LEFT, _F32_GOAL = _f32(0.6), _f32(-1.2), _f32(0.45)
#: Wrap passes of a Torch step in each direction; a step moves an angle by less than four turns.
_WRAP_PASSES = 4


class _NumPy:
    """Array operations of the kernels on host NumPy arrays; outputs move to ``device``."""

    sin, cos, where = staticmethod(np.sin), staticmethod(np.cos), staticmethod(np.where)
    minimum, maximum, remainder = staticmethod(np.minimum), staticmethod(np.maximum), staticmethod(np.remainder)
    float32 = np.float32

    def __init__(self, device: torch.device):
        self.device = device

    @staticmethod
    def clip(x, low, high):
        return np.minimum(np.maximum(x, low), high)

    @staticmethod
    def pow2(x):
        return np.float_power(x, 2.0)

    @staticmethod
    def pow2_as(x, dtype):
        """``x ** 2`` of a ``dtype`` scalar, which calls ``powf`` for float32.

        NumPy's SIMD loops and its square fast path round differently from ``powf``. They take only
        forward strides, so reversed arrays keep ``np.power`` on the C library loop.
        """
        narrow = x.astype(dtype)[::-1]
        return np.power(narrow, np.full(narrow.shape, 2, dtype)[::-1])[::-1].astype(np.float64)

    @staticmethod
    def stack(columns):
        return np.stack(columns, 1)

    @staticmethod
    def f64(x):
        return x.astype(np.float64)

    @staticmethod
    def r32(x):
        return x.astype(np.float32).astype(np.float64)

    @staticmethod
    def round_to(x, dtype):
        return x if dtype is None else x.astype(dtype).astype(np.float64)

    @staticmethod
    def full(like, value):
        return np.full(like.shape, value, np.float64)

    @staticmethod
    def flags(like, value: bool):
        return np.full(like.shape, value, np.bool_)

    @staticmethod
    def load(tensor):
        return tensor.detach().cpu().numpy().copy()

    @staticmethod
    def host(value):
        return np.asarray(value.detach().cpu() if torch.is_tensor(value) else value)

    @staticmethod
    def is_integer(a) -> bool:
        return a.dtype.kind in "iub"

    @staticmethod
    def is_bool(a) -> bool:
        return a.dtype == np.bool_

    @staticmethod
    def integers(a):
        return a.astype(np.int64)

    @staticmethod
    def floats(a):
        """Float64 values and the narrower float type that the source computes action terms in."""
        return a.astype(np.float64), (a.dtype if a.dtype.kind == "f" and a.dtype.itemsize < 8 else None)

    def column(self, x):
        """An owned ``[N, 1]`` tensor on the adapter's device."""
        return torch.from_numpy(x[:, None].copy()).to(self.device)

    def export(self, x, dtype=None):
        return torch.from_numpy(x if dtype is None else x.astype(dtype)).to(self.device)


class _Torch:
    """Array operations of the kernels on Torch tensors of the adapter's device."""

    sin, cos, where = staticmethod(torch.sin), staticmethod(torch.cos), staticmethod(torch.where)
    minimum, maximum = staticmethod(torch.minimum), staticmethod(torch.maximum)
    float32 = torch.float32

    def __init__(self, device: torch.device):
        self.device = device

    @staticmethod
    def clip(x, low, high):
        return torch.clamp(x, low, high)

    @staticmethod
    def pow2(x):
        return x * x

    @staticmethod
    def pow2_as(x, dtype):
        return (x * x).to(dtype).double()

    @staticmethod
    def remainder(x, divisor: float):
        """NumPy's float ``%``: the exact ``fmod``, moved into the divisor's sign."""
        mod = torch.fmod(x, divisor)
        return torch.where((mod != 0) & ((mod < 0) != (divisor < 0)), mod + divisor, mod)

    @staticmethod
    def stack(columns):
        return torch.stack(columns, 1)

    @staticmethod
    def f64(x):
        return x.double()

    @staticmethod
    def r32(x):
        return x.float().double()

    @staticmethod
    def round_to(x, dtype):
        return x if dtype is None else x.to(dtype).double()

    @staticmethod
    def full(like, value):
        return torch.full_like(like, value, dtype=torch.float64)

    @staticmethod
    def flags(like, value: bool):
        return torch.full_like(like, value, dtype=torch.bool)

    def load(self, tensor):
        return tensor.detach().to(self.device).clone()

    def host(self, value):
        return torch.as_tensor(value, device=self.device)

    @staticmethod
    def is_integer(a) -> bool:
        return not a.is_floating_point() and not a.is_complex()

    @staticmethod
    def is_bool(a) -> bool:
        return a.dtype == torch.bool

    @staticmethod
    def integers(a):
        return a.long()

    @staticmethod
    def floats(a):
        return a.double(), (a.dtype if a.is_floating_point() and a.dtype.itemsize < 8 else None)

    @staticmethod
    def column(x):
        return x[:, None].clone()

    @staticmethod
    def export(x, dtype=None):
        return x if dtype is None else x.to(dtype)


def _cartpole(xp, s, p, a, _):
    """``CartPoleEnv.step`` with Euler integration and the default reward."""
    x, x_dot, theta, theta_dot = s["x"], s["x_dot"], s["theta"], s["theta_dot"]
    total_mass = p["masspole"] + p["masscart"]
    polemass_length = p["masspole"] * p["length"]
    force = xp.where(a == 1, p["force_mag"], -p["force_mag"])
    costheta, sintheta = xp.cos(theta), xp.sin(theta)
    temp = (force + polemass_length * (theta_dot * theta_dot) * sintheta) / total_mass
    thetaacc = (p["gravity"] * sintheta - costheta * temp) / (
        p["length"] * (4.0 / 3.0 - p["masspole"] * (costheta * costheta) / total_mass)
    )
    xacc = temp - polemass_length * thetaacc * costheta / total_mass
    tau = p["tau"]
    x, x_dot = x + tau * x_dot, x_dot + tau * xacc
    theta, theta_dot = theta + tau * theta_dot, theta_dot + tau * thetaacc
    limit = 12 * 2 * math.pi / 360
    terminated = (x < -2.4) | (x > 2.4) | (theta < -limit) | (theta > limit)
    beyond, has_beyond = s["task", "steps_beyond_terminated"], s["task", "has_steps_beyond_terminated"]
    again = terminated & has_beyond
    new = {
        "x": x,
        "x_dot": x_dot,
        "theta": theta,
        "theta_dot": theta_dot,
        ("task", "steps_beyond_terminated"): xp.where(again, beyond + 1, xp.where(terminated, 0, beyond)),
        ("task", "has_steps_beyond_terminated"): has_beyond | terminated,
    }
    return new, [x, x_dot, theta, theta_dot], xp.where(again, 0.0, xp.full(x, 1.0)), terminated


def _pendulum(xp, s, p, u, dtype):
    """``PendulumEnv.step``; the torque terms take the action's float type, as NumPy evaluates them."""
    th, thdot = s["theta"], s["theta_dot"]
    bound = xp.round_to(p["max_torque"], dtype)
    u = xp.minimum(xp.maximum(u, -bound), bound)
    wrapped = xp.remainder(th + math.pi, 2 * math.pi) - math.pi
    if dtype is None:
        torque_cost = 0.001 * xp.pow2(u)
    else:
        torque_cost = xp.round_to(xp.round_to(xp.full(u, 0.001), dtype) * xp.pow2_as(u, dtype), dtype)
    costs = xp.pow2(wrapped) + 0.1 * xp.pow2(thdot) + torque_cost
    drive = xp.round_to(xp.round_to(3.0 / (p["m"] * xp.pow2(p["l"])), dtype) * u, dtype)
    newthdot = thdot + (3 * p["g"] / (2 * p["l"]) * xp.sin(th) + drive) * p["dt"]
    newthdot = xp.minimum(xp.maximum(newthdot, -p["max_speed"]), p["max_speed"])
    newth = th + newthdot * p["dt"]
    new = {"theta": newth, "theta_dot": newthdot, "torque": u, ("task", "has_last_u"): xp.flags(u, True)}
    return new, [xp.cos(newth), xp.sin(newth), newthdot], -costs, xp.flags(u, False)


def _acrobot(xp, s, p, a, _):
    """``AcrobotEnv.step``: one RK4 step of 0.2 s with the "book" dynamics, then wrap and bound."""
    m1, m2 = p["link_mass_1"], p["link_mass_2"]
    l1, lc1, lc2, moi = p["link_length_1"], p["link_com_pos_1"], p["link_com_pos_2"], p["link_moi"]
    # AVAIL_TORQUE[a] = [-1.0, 0.0, 1.0][a], including Python's negative indices.
    torque = xp.f64((a + 3) % 3 - 1)
    # Lane constants, grouped as Python evaluates the source expressions left to right.
    pow2 = xp.pow2
    d1_base = m1 * pow2(lc1)
    d1_lengths = pow2(l1) + pow2(lc2)
    d1_cross = 2 * l1 * lc2
    d2_base = pow2(lc2)
    d2_cross = l1 * lc2
    phi2_scale = m2 * lc2 * 9.8
    phi1_v2 = -m2 * l1 * lc2
    phi1_v12 = 2 * m2 * l1 * lc2
    phi1_g = (m1 * lc1 + m2 * l1) * 9.8
    dd2_v1 = m2 * l1 * lc2
    dd2_inertia = m2 * pow2(lc2) + moi

    def dsdt(theta1, theta2, dtheta1, dtheta2):
        cos2, sin2 = xp.cos(theta2), xp.sin(theta2)
        d1 = d1_base + m2 * (d1_lengths + d1_cross * cos2) + moi + moi
        d2 = m2 * (d2_base + d2_cross * cos2) + moi
        phi2 = phi2_scale * xp.cos(theta1 + theta2 - math.pi / 2.0)
        phi1 = (
            phi1_v2 * pow2(dtheta2) * sin2
            - phi1_v12 * dtheta2 * dtheta1 * sin2
            + phi1_g * xp.cos(theta1 - math.pi / 2)
            + phi2
        )
        ddtheta2 = (torque + d2 / d1 * phi1 - dd2_v1 * pow2(dtheta1) * sin2 - phi2) / (dd2_inertia - pow2(d2) / d1)
        ddtheta1 = -(d2 * ddtheta2 + phi1) / d1
        return dtheta1, dtheta2, ddtheta1, ddtheta2

    y = (s["theta1"], s["theta2"], s["dtheta1"], s["dtheta2"])
    dt = 0.2
    k1 = dsdt(*y)
    k2 = dsdt(*(yi + dt / 2.0 * ki for yi, ki in zip(y, k1)))
    k3 = dsdt(*(yi + dt / 2.0 * ki for yi, ki in zip(y, k2)))
    k4 = dsdt(*(yi + dt * ki for yi, ki in zip(y, k3)))
    theta1, theta2, dtheta1, dtheta2 = (
        yi + dt / 6.0 * (a1 + 2 * a2 + 2 * a3 + a4) for yi, a1, a2, a3, a4 in zip(y, k1, k2, k3, k4)
    )
    theta1, theta2 = _wrap(xp, theta1), _wrap(xp, theta2)
    dtheta1 = xp.clip(dtheta1, -4 * math.pi, 4 * math.pi)
    dtheta2 = xp.clip(dtheta2, -9 * math.pi, 9 * math.pi)
    cos1 = xp.cos(theta1)
    terminated = -cos1 - xp.cos(theta2 + theta1) > 1.0
    new = {"theta1": theta1, "theta2": theta2, "dtheta1": dtheta1, "dtheta2": dtheta2}
    observation = [cos1, xp.sin(theta1), xp.cos(theta2), xp.sin(theta2), dtheta1, dtheta2]
    return new, observation, xp.where(terminated, 0.0, xp.full(cos1, -1.0)), terminated


def _wrap(xp, x):
    """Acrobot's ``wrap(x, -pi, pi)``: subtract, then add, whole turns until x is in range."""
    turn = math.pi - (-math.pi)
    if isinstance(xp, _Torch):
        # Fixed passes read nothing back to the host; each pass repeats the source's subtraction or addition.
        for _ in range(_WRAP_PASSES):
            x = torch.where(x > math.pi, x - turn, x)
        for _ in range(_WRAP_PASSES):
            x = torch.where(x < -math.pi, x + turn, x)
        return x
    while bool((x > math.pi).any()):
        x = xp.where(x > math.pi, x - turn, x)
    while bool((x < -math.pi).any()):
        x = xp.where(x < -math.pi, x + turn, x)
    return x


def _mountain_car(xp, s, p, a, _):
    """``MountainCarEnv.step``."""
    position, velocity = s["position"], s["velocity"]
    velocity = velocity + ((a - 1) * p["force"] + xp.cos(3 * position) * (-p["gravity"]))
    velocity = xp.minimum(xp.maximum(velocity, -p["max_speed"]), p["max_speed"])
    position = xp.clip(position + velocity, -1.2, 0.6)
    velocity = xp.where((position == -1.2) & (velocity < 0), 0.0, velocity)
    terminated = (position >= 0.5) & (velocity >= 0)
    new = {"position": position, "velocity": velocity}
    return new, [position, velocity], xp.full(position, -1.0), terminated


def _mountain_car_continuous(xp, s, p, a, dtype):
    """``Continuous_MountainCarEnv.step`` with NumPy 2 scalar types.

    A lane's state is float64 after a reset and float32 after a step. A clipped force is a Python
    float, so the next sum takes the state's type. An unclipped force keeps the action's type. The
    ``wide`` masks mark lanes whose source scalar is float64 or a Python float; the others hold float32
    values, which compare with the float32 rounding of a Python constant.
    """
    position, velocity, is32 = s["position"], s["velocity"], s["task", "state_is_float32"]
    r32, power, max_speed = xp.r32, p["power"], p["max_speed"]
    cosine = 0.0025 * xp.cos(xp.where(is32, r32(3 * position), 3 * position))
    low, high = a < -1.0, a > 1.0
    clipped = low | high
    clipped_term = xp.where(low, -power, power) - cosine
    if dtype is None:
        action_term = a * power - cosine
        wide = ~is32 | ~clipped
    else:
        product = xp.round_to(a * xp.round_to(power, dtype), dtype)
        action_term = xp.round_to(product - xp.round_to(cosine, dtype), dtype)
        wide = ~is32
    narrow_velocity = r32(velocity + xp.where(clipped, r32(clipped_term), action_term))
    velocity = xp.where(wide, velocity + xp.where(clipped, clipped_term, action_term), narrow_velocity)
    limit32 = r32(max_speed)
    over = xp.where(wide, velocity > max_speed, velocity > limit32)
    velocity = xp.where(over, max_speed, velocity)
    under = ~over & xp.where(wide, velocity < -max_speed, velocity < -limit32)
    velocity = xp.where(under, -max_speed, velocity)
    wide = ~is32 | (wide & ~over & ~under)
    position = xp.where(wide, position + velocity, r32(position + r32(velocity)))
    right = xp.where(wide, position > 0.6, position > _F32_RIGHT)
    position = xp.where(right, 0.6, position)
    left = ~right & xp.where(wide, position < -1.2, position < _F32_LEFT)
    position = xp.where(left, -1.2, position)
    wide = wide | right | left
    at_wall = xp.where(wide, position == -1.2, position == _F32_LEFT)
    velocity = xp.where(at_wall & (velocity < 0), 0.0, velocity)
    terminated = xp.where(wide, position >= 0.45, position >= _F32_GOAL) & (velocity >= 0)
    reward = xp.where(terminated, 100.0, xp.full(a, 0.0)) - xp.pow2(a) * 0.1
    position, velocity = r32(position), r32(velocity)
    new = {"position": position, "velocity": velocity, ("task", "state_is_float32"): xp.flags(is32, True)}
    return new, [position, velocity], reward, terminated


_KERNELS = {
    "cartpole": _cartpole,
    "pendulum": _pendulum,
    "acrobot": _acrobot,
    "mountain_car": _mountain_car,
    "mountain_car_continuous": _mountain_car_continuous,
}


class BatchedClassicAdapter(ClassicControlAdapter):
    """Batched classic-control steps; resets and RNG stay with the Gymnasium environments.

    ``backend="numpy"`` steps on the host and ``backend="torch"`` on ``device``.
    """

    def __init__(self, domain: str, num_envs: int, device="cpu", seed: int = 0, backend: str = "numpy"):
        if backend not in ("numpy", "torch"):
            raise ValueError(f"backend must be 'numpy' or 'torch', got {backend!r}")
        super().__init__(domain, num_envs, device, seed)
        self.backend = backend
        # The kernels call one array API; NumPy and Torch overloads differ only in their stubs.
        self._xp: Any = _NumPy(self.device) if backend == "numpy" else _Torch(self.device)
        self._kernel = _KERNELS[domain]
        self._dynamic = _DYNAMIC[domain]
        self._max_steps = MAX_EPISODE_STEPS[domain]
        self._fixed: dict | None = None

    def _load(self, state: TensorDict) -> None:
        """Adopt ``state`` as the live lanes and keep the fields that steps do not change."""
        xp = self._xp
        moving = (*self._dynamic, _ELAPSED)
        self._fixed = {key: state[key].to(self.device).clone() for key in self.fields if key not in moving}
        self._live = {key: xp.load(state[key][:, 0]) for key in self._dynamic}
        self._elapsed = xp.load(state[_ELAPSED][:, 0])
        self._params = {name: xp.load(state[physics_state_key(name)][:, 0]) for name in _PHYSICS_SPACES[self.domain]}

    def configure(self, values) -> None:
        super().configure(values)
        if self._fixed is not None:
            state = self.get_state()
            for j, key in enumerate(self._physics_keys):
                state[key] = torch.from_numpy(self._physics[:, j : j + 1].copy()).to(self.device)
            self._load(state)

    def reset(self, seed: int | None = None, options: dict | None = None) -> TensorDict:
        state = super().reset(seed=seed, options=options)
        self._load(state)
        return state

    def set_state(self, state: TensorDict) -> None:
        super().set_state(state)
        # Gymnasium stores the State in its own types, for example float32 MountainCarContinuous lanes.
        self._load(ClassicControlAdapter.get_state(self))

    def get_state(self) -> TensorDict:
        self._check_open()
        if self._fixed is None:
            return super().get_state()
        tree: dict = {}
        live, fixed, column = self._live, self._fixed, self._xp.column
        for key in self.fields:
            if key in live:
                value = column(live[key])
            elif key == _ELAPSED:
                value = column(self._elapsed)
            else:
                value = fixed[key].clone()
            if isinstance(key, str):
                tree[key] = value
            else:
                tree.setdefault(key[0], {})[key[1]] = value
        return build_tensordict(tree, [self.num_envs], self.device)

    def _check_actions(self, a, live) -> None:
        n = _DISCRETE[self.domain]
        acrobot = self.domain == "acrobot"
        if not self._xp.is_integer(a):
            # Gymnasium's Discrete check rejects float actions; Acrobot indexes a Python list with them.
            message = f"{self.domain} actions must be integers, got dtype {a.dtype}"
            raise TypeError(message) if acrobot else AssertionError(message)
        bad = (a < -n) | (a >= n) if acrobot else (a < 0) | (a >= n)
        if live is not None:
            bad = bad & live
        if bool(bad.any()):
            message = f"{self.domain} actions must be in [{-n if acrobot else 0}, {n})"
            raise IndexError(message) if acrobot else AssertionError(message)

    def step(self, actions, *, active=None, capture_state=True) -> TensorDict:
        self._check_open()
        if self._fixed is None:
            raise RuntimeError("Call reset before step")
        xp = self._xp
        a = xp.host(actions)
        if tuple(a.shape) != (self.num_envs, 1):
            raise ValueError(f"{self.domain} action must have shape ({self.num_envs}, 1), got {tuple(a.shape)}")
        a = a[:, 0]
        live = None if active is None else xp.host(active)
        if live is not None and (tuple(live.shape) != (self.num_envs,) or not xp.is_bool(live)):
            raise ValueError("active must be a bool vector with shape [num_envs]")
        if self.domain in _DISCRETE:
            self._check_actions(a, live)
            a, dtype = xp.integers(a), None
        else:
            a, dtype = xp.floats(a)
        old = self._live
        new, observation, reward, terminated = self._kernel(xp, old, self._params, a, dtype)
        if live is None:
            self._live = new
            self._elapsed = self._elapsed + 1
            truncated = self._elapsed >= self._max_steps
        else:
            self._live = {key: xp.where(live, new[key], old[key]) for key in self._dynamic}
            self._elapsed = self._elapsed + live
            truncated = live & (self._elapsed >= self._max_steps)
            observation = [xp.where(live, column, 0.0) for column in observation]
            reward = xp.where(live, reward, 0.0)
            terminated = live & terminated
        observation = xp.stack(observation)
        return build_tensordict(
            {
                **({"state": self.get_state()} if capture_state else {}),
                "reward": xp.export(reward),
                "terminated": xp.export(terminated),
                "truncated": xp.export(truncated),
                "observation": xp.export(observation, xp.float32),
                "info_state": xp.export(observation, xp.float32),
            },
            [self.num_envs],
            self.device,
        )

    def close(self) -> None:
        self._fixed = None
        super().close()


#: Fewest lanes for which the CPU default and ``"auto"`` pick NumPy steps. Below them the per-call cost of the
#: batched arrays exceeds the Gymnasium loop. Measured on a Xeon Platinum 8268.
NUMPY_MIN_ENVS = {"cartpole": 2, "pendulum": 1, "acrobot": 2, "mountain_car": 1, "mountain_car_continuous": 8}
#: Fewest CUDA lanes for which ``"auto"`` picks Torch steps; below them NumPy steps on the host and the
#: outputs are copied to the GPU. On an L40S the Torch step on the GPU is faster at every batch size.
TORCH_MIN_ENVS = 1
BACKENDS = ("auto", "gym", "numpy", "torch")


def classic_backend(
    domain: str,
    device: str | torch.device,
    num_envs: int,
    backend: str | None = None,
    torch_min_envs: int = TORCH_MIN_ENVS,
) -> str:
    """Resolve ``simulation_backend`` from the device and the lane count.

    ``None`` selects Torch on CUDA and the CPU choice of ``"auto"`` on the CPU. ``"auto"`` also uses
    Torch on CUDA, but only for ``torch_min_envs`` or more lanes.
    """
    if backend is not None and backend not in BACKENDS:
        raise ValueError(f"simulation_backend must be None, 'auto', 'gym', 'numpy' or 'torch', got {backend!r}")
    if isinstance(torch_min_envs, bool) or not isinstance(torch_min_envs, int) or torch_min_envs < 1:
        raise ValueError(f"torch_min_envs must be an int >= 1, got {torch_min_envs!r}")
    cuda = torch.device(device).type == "cuda"
    if backend is None and cuda:
        return "torch"
    if backend not in (None, "auto"):
        return backend
    if cuda and num_envs >= torch_min_envs:
        return "torch"
    return "numpy" if num_envs >= NUMPY_MIN_ENVS[domain] else "gym"
