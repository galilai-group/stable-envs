"""Pendulum-v1 dynamics stepped for every lane at once in Torch, on the adapter's device.

Gymnasium environments still own resets, the PCG64 streams and the physics values, so seeded resets and
State export match :class:`ClassicControlAdapter`. Steps repeat Gymnasium's arithmetic in float64, with
the action terms in the action's own dtype as NumPy 2 evaluates them. Device ``sin``, ``cos`` and
``remainder`` may round differently from the C library, so trajectories agree to rounding, not bit for bit.
"""

from __future__ import annotations

import math

import torch
from tensordict import TensorDict

from .local import ClassicControlAdapter, physics_state_key

#: Gymnasium registers Pendulum-v1 with a 200-step time limit.
MAX_EPISODE_STEPS = 200
_DYNAMIC = ("theta", "theta_dot", "torque", ("task", "has_last_u"), ("lifecycle", "elapsed_steps"))


class PendulumTorchAdapter(ClassicControlAdapter):
    """Batched Pendulum steps on ``device``; resets and RNG stay with the Gymnasium environments."""

    def __init__(self, num_envs: int, device="cpu", seed: int = 0):
        super().__init__("pendulum", num_envs, device, seed)
        self._fixed = None
        self._params = None

    def _load(self, state: TensorDict) -> None:
        """Adopt ``state`` as the live lanes and cache the fields that steps do not change."""
        state = state.to(self.device)
        self._fixed = state.exclude(*_DYNAMIC).clone()
        self._theta = state["theta"][:, 0].clone()
        self._theta_dot = state["theta_dot"][:, 0].clone()
        self._torque = state["torque"][:, 0].clone()
        self._has_last_u = state["task", "has_last_u"][:, 0].clone()
        self._elapsed = state["lifecycle", "elapsed_steps"][:, 0].clone()
        self._params = {
            name: state[physics_state_key(name)][:, 0].clone()
            for name in ("g", "m", "l", "max_torque", "max_speed", "dt")
        }

    def configure(self, values) -> None:
        super().configure(values)
        if self._fixed is not None:
            self._load(self.get_state())

    def reset(self, seed: int | None = None, options: dict | None = None) -> TensorDict:
        state = super().reset(seed=seed, options=options)
        self._load(state)
        return self.get_state()

    def set_state(self, state: TensorDict) -> None:
        super().set_state(state)
        self._load(super().get_state())

    def get_state(self) -> TensorDict:
        self._check_open()
        if self._fixed is None:
            return super().get_state()
        state = self._fixed.clone()
        state["theta"] = self._theta[:, None].clone()
        state["theta_dot"] = self._theta_dot[:, None].clone()
        state["torque"] = self._torque[:, None].clone()
        state["task", "has_last_u"] = self._has_last_u[:, None].clone()
        state["lifecycle", "elapsed_steps"] = self._elapsed[:, None].clone()
        return state

    def check_step(self, actions, active):
        if torch.is_tensor(actions) and actions.shape != (self.num_envs, 1):
            raise ValueError(f"pendulum action must have shape ({self.num_envs}, 1), got {tuple(actions.shape)}")
        if not torch.is_tensor(actions):
            return super().check_step(actions, active)
        if active is not None and (active.shape != (self.num_envs,) or active.dtype != torch.bool):
            raise ValueError("active must be a bool vector with shape [num_envs]")
        return actions, active

    def step(self, actions, *, active=None, capture_state=True) -> TensorDict:
        self._check_open()
        u = torch.as_tensor(actions, device=self.device)
        if u.shape != (self.num_envs, 1):
            raise ValueError(f"pendulum action must have shape ({self.num_envs}, 1), got {tuple(u.shape)}")
        u = u[:, 0]
        if not u.is_floating_point():
            u = u.to(torch.float64)
        p = self._params
        th, thdot = self._theta, self._theta_dot
        # np.clip(u, -max_torque, max_torque): the Python bounds take the action's dtype.
        bound = p["max_torque"].to(u.dtype)
        u = torch.minimum(torch.maximum(u, -bound), bound)
        # angle_normalize(th) ** 2 + 0.1 * thdot**2 + 0.001 * u**2, the last term in the action's dtype.
        wrapped = _np_remainder(th + math.pi, 2 * math.pi) - math.pi
        costs = wrapped**2 + 0.1 * thdot**2 + (0.001 * u**2).to(torch.float64)
        drive = (3.0 / (p["m"] * p["l"] ** 2)).to(u.dtype) * u
        new_thdot = thdot + (3 * p["g"] / (2 * p["l"]) * torch.sin(th) + drive.to(torch.float64)) * p["dt"]
        new_thdot = torch.minimum(torch.maximum(new_thdot, -p["max_speed"]), p["max_speed"])
        new_th = th + new_thdot * p["dt"]
        live = torch.ones_like(self._has_last_u) if active is None else torch.as_tensor(active, device=self.device)
        self._theta = torch.where(live, new_th, th)
        self._theta_dot = torch.where(live, new_thdot, thdot)
        self._torque = torch.where(live, u.to(torch.float64), self._torque)
        self._has_last_u = self._has_last_u | live
        self._elapsed = self._elapsed + live.to(self._elapsed.dtype)
        observation = torch.stack([torch.cos(self._theta), torch.sin(self._theta), self._theta_dot], 1).to(
            torch.float32
        )
        observation = torch.where(live[:, None], observation, torch.zeros_like(observation))
        zero = torch.zeros((), dtype=torch.float64, device=self.device)
        return TensorDict(
            {
                **({"state": self.get_state()} if capture_state else {}),
                "reward": torch.where(live, -costs, zero),
                "terminated": torch.zeros_like(live),
                "truncated": live & (self._elapsed >= MAX_EPISODE_STEPS),
                "observation": observation,
                "info_state": observation.clone(),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def close(self) -> None:
        self._fixed = None
        super().close()


def _np_remainder(x: torch.Tensor, divisor: float) -> torch.Tensor:
    """NumPy's float ``%``: the exact ``fmod``, moved into the divisor's sign."""
    mod = torch.fmod(x, divisor)
    return torch.where((mod != 0) & ((mod < 0) != (divisor < 0)), mod + divisor, mod)


def pendulum_backend(device: str | torch.device, backend: str = "auto") -> str:
    """Torch physics for CUDA lanes and Gymnasium on CPU, unless ``backend`` names one."""
    if backend not in ("auto", "gym", "torch"):
        raise ValueError("simulation_backend must be 'auto', 'gym' or 'torch'")
    if backend == "auto":
        return "torch" if torch.device(device).type == "cuda" else "gym"
    return backend
