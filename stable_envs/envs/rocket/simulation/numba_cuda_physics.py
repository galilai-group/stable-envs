"""CUDA build of ``numba_physics``: the same per-environment functions, one GPU thread per environment.

The state is a dict of torch CUDA tensors with the ``numba_physics.FIELDS`` names, shapes and dtypes; write into
them in place. The calls launch on the current torch stream and do not synchronize with the host. A launch passes
about 50 arrays, which costs a few hundred microseconds of host time; ``StepGraph`` and ``ResetGraph`` capture one
launch in a CUDA graph and replay it for tens of microseconds.
"""

from __future__ import annotations

import numpy as np
import torch
from numba import cuda, float64

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
_START = ("start_pos", "start_orn", "position", "linear_velocity", "angular_velocity")


@cuda.jit(cache=True)
def _step_cuda(
    pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, action, physics_steps,
    obs, ang_vel, ang_pos, lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel,
    ignition_state, throttle_state, p_ignition_state, p_throttle_state, p_render_ignition, flame_ignited,
    pad_contact, reward, step_count, termination, truncation, out_of_bounds, fatal_collision, env_complete,
    start_pos, start_orn,
    actions, noise, rng, draw, active, noise_used, ceiling, max_displacement, max_steps, sparse_reward,
):  # fmt: skip
    b = cuda.grid(1)
    if b < pos.shape[0] and active[b]:
        crel = cuda.local.array((MAX_CONTACTS, 8), float64)
        _agent_step(
            b,
            pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, action, physics_steps,
            obs, ang_vel, ang_pos, lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel,
            ignition_state, throttle_state, p_ignition_state, p_throttle_state, p_render_ignition, flame_ignited,
            pad_contact, reward, step_count, termination, truncation, out_of_bounds, fatal_collision, env_complete,
            start_pos, start_orn,
            actions, noise, rng, draw, noise_used, HULL_VERTICES, crel, ceiling, max_displacement, max_steps,
            sparse_reward,
        )  # fmt: skip


@cuda.jit(cache=True)
def _settle_cuda(
    pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, action, physics_steps,
    obs, ang_vel, ang_pos, lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel,
    ignition_state, throttle_state, p_ignition_state, p_throttle_state, p_render_ignition, flame_ignited,
    pad_contact, reward, step_count, termination, truncation, out_of_bounds, fatal_collision, env_complete,
    start_pos, start_orn,
    noise, rng, draw, mask,
):  # fmt: skip
    b = cuda.grid(1)
    if b < pos.shape[0] and mask[b]:
        crel = cuda.local.array((MAX_CONTACTS, 8), float64)
        _settle_env(
            b,
            pos, quat, vel, avel, lrot, lvel, fin, gim, thr, fuel, ign, cmd, action, physics_steps,
            obs, ang_vel, ang_pos, lin_vel, lin_pos, ground_lin_vel, p_ang_vel, p_lin_vel, p_lin_pos, p_ground_lin_vel,
            ignition_state, throttle_state, p_ignition_state, p_throttle_state, p_render_ignition, flame_ignited,
            pad_contact, reward, step_count, termination, truncation, out_of_bounds, fatal_collision, env_complete,
            start_pos, start_orn,
            noise, rng, draw, HULL_VERTICES, crel,
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
    """Numba views of one state's tensors and the launch configuration."""

    def __init__(self, state):
        position = state["position"]
        if position.device.type != "cuda":
            raise ValueError("Rocket CUDA physics needs CUDA tensors")
        for key in FIELDS:
            if not state[key].is_contiguous():
                raise ValueError(f"Rocket CUDA state tensor {key} must be contiguous")
        self.device = position.device
        self.num_envs = position.shape[0]
        self.key = tuple(state[k].data_ptr() for k in FIELDS)
        self.arrays = tuple(cuda.as_cuda_array(state[k]) for k in _ORDER)
        self.rng = cuda.as_cuda_array(state["rng"]).view(np.uint64)
        self.blocks = (self.num_envs + THREADS_PER_BLOCK - 1) // THREADS_PER_BLOCK
        self.empty = cuda.as_cuda_array(torch.empty((1, 1), dtype=torch.float64, device=self.device))


#: The views hold their tensors, so the cache keeps only the last few states.
_LAUNCHES: dict[int, _Launch] = {}
_CACHED_STATES = 4
_STREAMS: dict[int, object] = {}


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


def _stream(device):
    handle = torch.cuda.current_stream(device).cuda_stream
    stream = _STREAMS.get(handle)
    if stream is None:
        stream = _STREAMS[handle] = cuda.external_stream(handle)
    return stream


def _device(value, launch, dtype):
    return torch.as_tensor(value, dtype=dtype, device=launch.device).contiguous()


def _noise(value, launch, shape, name):
    if value is None:
        return launch.empty, True
    tensor = _device(value, launch, torch.float64)
    if tuple(tensor.shape) != shape:
        raise ValueError(f"Rocket {name} expects noise {shape}")
    return cuda.as_cuda_array(tensor), False


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
    noise_array, draw = _noise(noise, launch, (num_envs, NOISE_PER_STEP), "step")
    noise_used = torch.zeros(num_envs, dtype=torch.int64, device=launch.device)
    _step_cuda[launch.blocks, THREADS_PER_BLOCK, _stream(launch.device)](
        *launch.arrays, cuda.as_cuda_array(actions), noise_array, launch.rng, draw, cuda.as_cuda_array(active),
        cuda.as_cuda_array(noise_used), float(ceiling), float(max_displacement), int(max_steps), bool(sparse_reward),
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
    columns = [_device(start[k], launch, torch.float64) for k in _START]
    _start_cuda[launch.blocks, THREADS_PER_BLOCK, _stream(launch.device)](
        *launch.arrays, cuda.as_cuda_array(mask), *(cuda.as_cuda_array(c) for c in columns), cuda.as_cuda_array(fuel)
    )


def settle(state: dict[str, torch.Tensor], noise_settle: torch.Tensor | None, mask: torch.Tensor) -> None:
    """``numba_physics.settle`` on CUDA tensors."""
    launch = _launch(state)
    noise_array, draw = _noise(noise_settle, launch, (launch.num_envs, NOISE_PER_SETTLE), "settle")
    mask = _device(mask, launch, torch.bool)
    _settle_cuda[launch.blocks, THREADS_PER_BLOCK, _stream(launch.device)](
        *launch.arrays, noise_array, launch.rng, draw, cuda.as_cuda_array(mask)
    )


def observation(state: dict[str, torch.Tensor]) -> torch.Tensor:
    """A copy of the ``RocketLandingEnv`` state vector, ``(B, OBSERVATION_DIM)`` float64."""
    return state["state"].clone()


def _capture(device, run):
    """Compile with one call of ``run`` on a side stream, then capture another call; return graph and result."""
    current = torch.cuda.current_stream(device)
    side = torch.cuda.Stream(device)
    side.wait_stream(current)
    with torch.cuda.stream(side):
        run()
    current.wait_stream(side)
    graph = torch.cuda.CUDAGraph()
    with torch.cuda.graph(graph):
        result = run()
    return graph, result


class StepGraph:
    """``step`` of one state, with thrust noise from its ``rng`` field, captured in a CUDA graph.

    A call copies ``actions`` and ``active`` into the graph's inputs and replays it. The results are the graph's
    output tensors: the next call overwrites them. Capture runs one step with no environment active, so the state
    does not change.
    """

    def __init__(self, state: dict[str, torch.Tensor], **task):
        self._launch = _launch(state)
        device, num_envs = self._launch.device, self._launch.num_envs
        self.actions = torch.zeros((num_envs, 7), dtype=torch.float64, device=device)
        self.active = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.graph, self.outputs = _capture(device, lambda: step(state, self.actions, None, self.active, **task))

    def __call__(self, actions: torch.Tensor, active: torch.Tensor):
        self.actions.copy_(actions)
        self.active.copy_(active)
        self.graph.replay()
        return self.outputs


class ResetGraph:
    """``start_state`` then ``settle`` (noise from ``rng``) of one state, captured in a CUDA graph.

    Capture runs one reset with no environment selected, so the state does not change.
    """

    def __init__(self, state: dict[str, torch.Tensor]):
        self._launch = _launch(state)
        device, num_envs = self._launch.device, self._launch.num_envs
        self.mask = torch.zeros(num_envs, dtype=torch.bool, device=device)
        self.start = {k: torch.zeros((num_envs, 3), dtype=torch.float64, device=device) for k in _START}
        self.start["fuel"] = torch.full((num_envs,), START_FUEL, dtype=torch.float64, device=device)

        def run():
            start_state(state, self.mask, self.start)
            settle(state, None, self.mask)

        self.graph, _ = _capture(device, run)

    def __call__(self, mask: torch.Tensor, start: dict[str, torch.Tensor]) -> None:
        self.mask.copy_(mask)
        for key in _START:
            self.start[key].copy_(start[key])
        fuel = start.get("fuel")
        if fuel is None:
            self.start["fuel"].fill_(START_FUEL)
        else:
            self.start["fuel"].copy_(fuel)
        self.graph.replay()
