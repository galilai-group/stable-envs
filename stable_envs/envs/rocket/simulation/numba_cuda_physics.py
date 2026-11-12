"""CUDA build of ``numba_physics``: the same per-environment functions, one GPU thread per environment.

The state is a dict of torch CUDA tensors with the ``numba_physics.FIELDS`` names, shapes and dtypes. The calls
launch on the current torch stream and do not synchronize with the host.
"""

from __future__ import annotations

import numpy as np
import torch
from numba import cuda

from .numba_physics import (
    _ORDER,
    FIELDS,
    HULL_VERTICES,
    MAX_CONTACTS,
    NOISE_PER_SETTLE,
    NOISE_PER_STEP,
    START_FUEL,
    _agent_step,
    _settle_env,
    _start_env,
)

THREADS_PER_BLOCK = 64
_TORCH = {np.dtype(np.float64): torch.float64, np.dtype(np.int64): torch.int64, np.dtype(np.bool_): torch.bool}


@cuda.jit(cache=True)
def _step_cuda(
    pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, action, physics_steps,
    obs, ang_vel, ang_pos, lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel,
    ignition_state, throttle_state, p_ignition_state, p_throttle_state, p_render_ignition, flame_ignited,
    pad_contact, reward, step_count, termination, truncation, out_of_bounds, fatal_collision, env_complete,
    start_pos, start_orn,
    actions, noise, rng, draw, active, noise_used, verts, crel, ceiling, max_displacement, max_steps, sparse_reward,
):  # fmt: skip
    b = cuda.grid(1)
    if b < pos.shape[0] and active[b]:
        _agent_step(
            b,
            pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, action, physics_steps,
            obs, ang_vel, ang_pos, lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel,
            ignition_state, throttle_state, p_ignition_state, p_throttle_state, p_render_ignition, flame_ignited,
            pad_contact, reward, step_count, termination, truncation, out_of_bounds, fatal_collision, env_complete,
            start_pos, start_orn,
            actions, noise, rng, draw, noise_used, verts, crel, ceiling, max_displacement, max_steps, sparse_reward,
        )  # fmt: skip


@cuda.jit(cache=True)
def _settle_cuda(
    pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, action, physics_steps,
    obs, ang_vel, ang_pos, lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel,
    ignition_state, throttle_state, p_ignition_state, p_throttle_state, p_render_ignition, flame_ignited,
    pad_contact, reward, step_count, termination, truncation, out_of_bounds, fatal_collision, env_complete,
    start_pos, start_orn,
    noise, rng, draw, mask, verts, crel,
):  # fmt: skip
    b = cuda.grid(1)
    if b < pos.shape[0] and mask[b]:
        _settle_env(
            b,
            pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, action, physics_steps,
            obs, ang_vel, ang_pos, lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel,
            ignition_state, throttle_state, p_ignition_state, p_throttle_state, p_render_ignition, flame_ignited,
            pad_contact, reward, step_count, termination, truncation, out_of_bounds, fatal_collision, env_complete,
            start_pos, start_orn,
            noise, rng, draw, verts, crel,
        )  # fmt: skip


@cuda.jit(cache=True)
def _start_cuda(
    pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, action, physics_steps,
    obs, ang_vel, ang_pos, lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel,
    ignition_state, throttle_state, p_ignition_state, p_throttle_state, p_render_ignition, flame_ignited,
    pad_contact, reward, step_count, termination, truncation, out_of_bounds, fatal_collision, env_complete,
    start_pos, start_orn,
    mask, s_start_pos, s_start_orn, s_position, s_linvel, s_angvel, s_fuel,
):  # fmt: skip
    b = cuda.grid(1)
    if b < pos.shape[0] and mask[b]:
        _start_env(
            b,
            pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, action, physics_steps,
            obs, ang_vel, ang_pos, lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel,
            ignition_state, throttle_state, p_ignition_state, p_throttle_state, p_render_ignition, flame_ignited,
            pad_contact, reward, step_count, termination, truncation, out_of_bounds, fatal_collision, env_complete,
            start_pos, start_orn,
            s_start_pos, s_start_orn, s_position, s_linvel, s_angvel, s_fuel,
        )  # fmt: skip


def new_state(num_envs: int, device: str | torch.device = "cuda") -> dict[str, torch.Tensor]:
    """Zeroed ``FIELDS`` tensors of shape ``(num_envs, *shape)`` on ``device``."""
    return {
        k: torch.zeros((num_envs, *shape), dtype=_TORCH[dtype], device=device) for k, (shape, dtype) in FIELDS.items()
    }


class _Launch:
    """Numba views of one state's tensors, the device constants and the launch configuration."""

    def __init__(self, state):
        position = state["position"]
        if position.device.type != "cuda":
            raise ValueError("Rocket CUDA physics needs CUDA tensors")
        self.device = position.device
        self.num_envs = position.shape[0]
        self.key = tuple(state[k].data_ptr() for k in FIELDS)
        self.arrays = tuple(cuda.as_cuda_array(state[k]) for k in _ORDER)
        self.rng = cuda.as_cuda_array(state["rng"]).view(np.uint64)
        self.verts = cuda.as_cuda_array(torch.as_tensor(HULL_VERTICES, device=self.device))
        self.crel_tensor = torch.empty((self.num_envs, MAX_CONTACTS, 8), dtype=torch.float64, device=self.device)
        self.crel = cuda.as_cuda_array(self.crel_tensor)
        self.blocks = (self.num_envs + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        self.empty = cuda.as_cuda_array(torch.empty((1, 1), dtype=torch.float64, device=self.device))

    def stream(self):
        return cuda.external_stream(torch.cuda.current_stream(self.device).cuda_stream)


#: The views hold their tensors, so the cache keeps only the last few states.
_LAUNCHES: dict[int, _Launch] = {}
_CACHED_STATES = 4


def _launch(state) -> _Launch:
    """The cached ``_Launch`` of ``state``; a state whose tensors were replaced gets new views."""
    entry = _LAUNCHES.pop(id(state), None)
    if entry is None or entry.key != tuple(state[k].data_ptr() for k in FIELDS):
        entry = _Launch(state)
    _LAUNCHES[id(state)] = entry
    while len(_LAUNCHES) > _CACHED_STATES:
        _LAUNCHES.pop(next(iter(_LAUNCHES)))
    return entry


def forget(state: dict[str, torch.Tensor]) -> None:
    """Drop the cached views of ``state``, which hold its tensors."""
    _LAUNCHES.pop(id(state), None)


def _device(value, launch, dtype):
    return torch.as_tensor(value, dtype=dtype, device=launch.device).contiguous()


def _noise(value, launch, shape, name):
    if value is None:
        return launch.empty, True, None
    tensor = _device(value, launch, torch.float64)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"Rocket {name} expects noise {shape}")
    return cuda.as_cuda_array(tensor), False, tensor


def step(
    state: dict[str, torch.Tensor],
    actions: torch.Tensor,
    noise: torch.Tensor | None,
    active: torch.Tensor,
    *,
    ceiling: float = 500.0,
    max_displacement: float = 200.0,
    max_steps: int = 1200,
    sparse_reward: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """``numba_physics.step`` on CUDA tensors, with the same arguments and results."""
    launch = _launch(state)
    num_envs = launch.num_envs
    actions = _device(actions, launch, torch.float64)
    active = _device(active, launch, torch.bool)
    if tuple(actions.shape) != (num_envs, 7) or tuple(active.shape) != (num_envs,):
        raise ValueError("Rocket step expects actions (B, 7) and active (B,)")
    noise_array, draw, _keep = _noise(noise, launch, (num_envs, NOISE_PER_STEP), "step")
    noise_used = torch.zeros(num_envs, dtype=torch.int64, device=launch.device)
    _step_cuda[launch.blocks, THREADS_PER_BLOCK, launch.stream()](
        *launch.arrays, cuda.as_cuda_array(actions), noise_array, launch.rng, draw, cuda.as_cuda_array(active),
        cuda.as_cuda_array(noise_used), launch.verts, launch.crel,
        float(ceiling), float(max_displacement), int(max_steps), bool(sparse_reward),
    )  # fmt: skip
    info = {
        "out_of_bounds": state["out_of_bounds"].clone(),
        "fatal_collision": state["fatal_collision"].clone(),
        "env_complete": state["env_complete"].clone(),
        "landing_pad_contact": state["landing_pad_contact"].clone(),
        "noise_used": noise_used,
    }
    return state["reward"].clone(), state["termination"].clone(), state["truncation"].clone(), info


def start_state(state: dict[str, torch.Tensor], mask: torch.Tensor, start: dict[str, torch.Tensor]) -> None:
    """``numba_physics.start_state`` on CUDA tensors."""
    launch = _launch(state)
    mask = _device(mask, launch, torch.bool)
    fuel = start.get("fuel")
    fuel = (
        torch.full((launch.num_envs,), START_FUEL, dtype=torch.float64, device=launch.device)
        if fuel is None
        else _device(fuel, launch, torch.float64)
    )
    columns = [
        _device(start[k], launch, torch.float64)
        for k in ("start_pos", "start_orn", "position", "linear_velocity", "angular_velocity")
    ]
    _start_cuda[launch.blocks, THREADS_PER_BLOCK, launch.stream()](
        *launch.arrays, cuda.as_cuda_array(mask), *(cuda.as_cuda_array(c) for c in columns), cuda.as_cuda_array(fuel)
    )


def settle(state: dict[str, torch.Tensor], noise_settle: torch.Tensor | None, mask: torch.Tensor) -> None:
    """``numba_physics.settle`` on CUDA tensors."""
    launch = _launch(state)
    noise_array, draw, _keep = _noise(noise_settle, launch, (launch.num_envs, NOISE_PER_SETTLE), "settle")
    mask = _device(mask, launch, torch.bool)
    _settle_cuda[launch.blocks, THREADS_PER_BLOCK, launch.stream()](
        *launch.arrays, noise_array, launch.rng, draw, cuda.as_cuda_array(mask), launch.verts, launch.crel
    )


def observation(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """A copy of the ``RocketLandingEnv`` state vector, ``(B, OBSERVATION_DIM)`` float64."""
    return state["state"].clone()
