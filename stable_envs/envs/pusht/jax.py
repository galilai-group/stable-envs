"""Batched PushT adapted from fracapuano/envX at d26c817c1124229c819be574f137cd68bda4380a.

Copyright 2026 Francesco Capuano. SPDX-License-Identifier: Apache-2.0
See licenses/envX-APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import functools
import os

import numpy as np
import torch
from tensordict import TensorDict

from ...env import Env
from ...registry import register
from ...spaces import Dict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.dlpack
import jax.numpy as jnp

from .simulation.jax_geometry import coverage, origin_from_center_of_mass
from .simulation.jax_physics import PushTParams, PushTState, simulate_control_step

_FLOAT_FIELDS = (
    ("agent_pos", 2),
    ("block_pos", 2),
    ("block_angle", 1),
    ("agent_vel", 2),
    ("block_vel", 2),
    ("block_angular_vel", 1),
    ("last_action", 2),
    ("goal", 3),
)


def _from_torch(value):
    value = value.detach().contiguous()
    alignment = 16 if value.is_cuda else 64
    if value.data_ptr() % alignment:
        value = value.clone()
    return jax.dlpack.from_dlpack(value, copy=False)


@jax.jit
@jax.vmap
def _reset(key):
    agent_key, block_key, angle_key = jax.random.split(key, 3)
    agent = jax.random.randint(agent_key, (2,), 50, 450, dtype=jnp.int32).astype(jnp.float32)
    block = jax.random.randint(block_key, (2,), 100, 400, dtype=jnp.int32).astype(jnp.float32)
    angle = jax.random.uniform(angle_key, (), minval=-jnp.pi, maxval=jnp.pi, dtype=jnp.float32)
    block = origin_from_center_of_mass(block + jnp.array([0, 45], dtype=jnp.float32), angle)
    rows = jnp.concatenate(
        (
            agent,
            block,
            angle[None],
            jnp.zeros(5, dtype=jnp.float32),
            agent,
            jnp.array([256, 256, jnp.pi / 4], dtype=jnp.float32),
        )
    )
    return rows, jax.random.split(key)[0]


@jax.jit
@jax.vmap
def _advance(values):
    state = PushTState(
        values[:2],
        values[5:7],
        values[2:4],
        values[4],
        values[7:9],
        values[9],
        jnp.int32(0),
    )
    action = jnp.clip(values[15:17], 0, 512)
    state = simulate_control_step(state, action, PushTParams())
    rows = jnp.concatenate(
        (
            state.agent_pos,
            state.block_pos,
            state.block_angle[None],
            state.agent_vel,
            state.block_vel,
            state.block_angular_vel[None],
            action,
            values[12:15],
        )
    )
    overlap = coverage(state.block_pos, state.block_angle, values[12:14], values[14])
    return rows, state.n_contacts, overlap


@functools.partial(jax.jit, static_argnames=("size", "with_target", "render_action"))
def _render(rows, *, size, with_target, render_action):
    from .jax_render import render

    return jax.vmap(lambda row: render(row, size=size, with_target=with_target, render_action=render_action))(rows)


@register
class PushTJax(Env):
    """Canonical envX circle/T task with absolute target actions and coverage reward."""

    family = domain = "pusht"
    name = "push-t-jax"
    reference_id = "envX/pusht"
    default_size = 96
    state_dim = 10
    state_names = (
        "agent_x",
        "agent_y",
        "block_x",
        "block_y",
        "block_angle",
        "goal_x",
        "goal_y",
        "goal_angle",
        "action_x",
        "action_y",
    )
    state_layout = (
        ("agent_pos", 2, 0),
        ("block_pos", 2, 2),
        ("block_angle", 1, 4),
        ("goal", 3, 5),
        ("last_action", 2, 8),
        ("agent_vel", 2, None),
        ("block_vel", 2, None),
        ("block_angular_vel", 1, None),
        ("contacts", 1, None, torch.int64),
        ("time", 1, None, torch.int64),
        ("rng", 2, None, torch.int64),
    )
    action_dim = 2
    action_low, action_high = 0.0, 512.0
    execution_mode = "JAX"

    def __init__(
        self, num_envs=1, size=None, supersample=1, device=None, seed=0, *, with_target=True, render_action=False
    ):
        if type(num_envs) is not int or num_envs < 1:
            raise ValueError("PushTJax num_envs must be a positive integer")
        if size is not None and (type(size) is not int or size < 1):
            raise ValueError("PushTJax size must be a positive integer")
        if supersample != 1:
            raise ValueError("PushTJax renders with supersample=1")
        super().__init__(num_envs, size, 1, device, "jax", seed)
        if self.device.type not in ("cpu", "cuda"):
            raise ValueError("PushTJax requires a CPU or CUDA device")
        index = self.device.index
        if index is None:
            index = torch.cuda.current_device() if self.device.type == "cuda" else 0
        self._jax_device = jax.devices("gpu" if self.device.type == "cuda" else "cpu")[index]
        self.with_target, self.render_action = bool(with_target), bool(render_action)
        self.reset(seed=seed)

    def make_variation_space(self):
        return Dict({})

    def _rows(self, state):
        return torch.cat([state[key] for key, _ in _FLOAT_FIELDS], dim=-1)

    def _state_from_rows(self, rows, **extra):
        fields, offset = {}, 0
        for key, width in _FLOAT_FIELDS:
            fields[key] = rows[:, offset : offset + width]
            offset += width
        return TensorDict({**fields, **extra}, [len(rows)], device=self.device)

    def reset(self, seed=None, options=None):
        self._check_open("reset")
        options = {} if options is None else options
        if not isinstance(options, dict) or set(options) - {"state"}:
            raise ValueError("PushTJax reset options support only a dataset state")
        if seed is not None and (
            isinstance(seed, bool) or not isinstance(seed, (int, np.integer)) or not 0 <= seed < 2**32
        ):
            raise ValueError("PushTJax seed must be an integer in [0, 2**32)")
        pose = options.get("state")
        if pose is not None:
            pose = torch.as_tensor(pose, dtype=torch.float32, device=self.device)
            if pose.shape == (5,):
                pose = pose.expand(self.B, 5)
            if pose.shape != (self.B, 5) or not bool(torch.isfinite(pose).all()):
                raise ValueError("PushTJax dataset state must be finite with shape [5] or [num_envs, 5]")
        with jax.default_device(self._jax_device):
            if seed is None and hasattr(self, "_state"):
                keys = jnp.asarray(self._state["rng"].cpu().numpy(), dtype=jnp.uint32)
            else:
                value = int(self.rng.integers(2**32)) if seed is None else int(seed)
                keys = jax.vmap(jax.random.PRNGKey)(jnp.arange(self.B, dtype=jnp.uint32) + value)
            rows, keys = _reset(keys)
            rows = torch.from_dlpack(rows).clone()
            keys = torch.from_dlpack(keys).to(torch.int64)
        if pose is not None:
            rows[:, :5] = pose
            rows[:, 10:12] = pose[:, :2]
        self._state = self._state_from_rows(
            rows,
            rng=keys,
            contacts=torch.zeros(self.B, 1, dtype=torch.int64, device=self.device),
            time=torch.zeros(self.B, 1, dtype=torch.int64, device=self.device),
        )
        return self.get_state()

    def get_state(self):
        self._check_open("get_state")
        return self._state.clone()

    def set_state(self, state):
        self._check_open("set_state")
        self._validate_state(state, batch=self.B)
        self.state_spec.validate(state, owner="PushTJax State")
        if bool(
            (state["time"] < 0).any()
            | (state["contacts"] < 0).any()
            | (state["rng"] < 0).any()
            | (state["rng"] >= 2**32).any()
        ):
            raise ValueError("Invalid PushTJax counters or RNG key")
        self._state = state.to(self.device).clone()

    def _step(self, actions, *, active=None, capture_state=True):
        self._check_open("step")
        actions = torch.as_tensor(actions, device=self.device, dtype=torch.float32)
        if actions.shape != (self.B, 2) or not bool(torch.isfinite(actions).all()):
            raise ValueError("PushTJax actions must be finite with shape [num_envs, 2]")
        if active is None:
            active = torch.ones(self.B, dtype=torch.bool, device=self.device)
        else:
            active = torch.as_tensor(active, device=self.device)
            if active.shape != (self.B,) or active.dtype != torch.bool:
                raise ValueError("PushTJax active mask must be bool [num_envs]")
        rows = self._rows(self._state)
        with jax.default_device(self._jax_device):
            updated, contacts, overlap = _advance(_from_torch(torch.cat((rows, actions), dim=-1)))
        updated, overlap = torch.from_dlpack(updated), torch.from_dlpack(overlap)
        self._state = self._state_from_rows(
            torch.where(active[:, None], updated, rows),
            rng=self._state["rng"],
            time=self._state["time"] + active[:, None],
            contacts=torch.where(
                active[:, None], torch.from_dlpack(contacts).to(torch.int64)[:, None], self._state["contacts"]
            ),
        )
        return TensorDict(
            dict(
                **({"state": self.get_state()} if capture_state else {}),
                observation=torch.cat(
                    (
                        self._state["agent_pos"],
                        self._state["block_pos"],
                        self._state["block_angle"].remainder(2 * np.pi),
                    ),
                    dim=-1,
                ).clone(),
                reward=torch.where(active, (overlap / 0.95).clamp(0, 1), 0),
                coverage=torch.where(active, overlap, 0),
                success=active & (overlap > 0.95),
                terminated=active & (overlap > 0.95),
                truncated=active & (self._state["time"][:, 0] >= 300),
            ),
            [self.B],
            device=self.device,
        )

    def _advance_without_snapshot(self, actions, *, active=None):
        return self._step(actions, active=active, capture_state=False)

    def _render(self, states=None, size=None, out=None):
        self._check_open("render")
        size = self.size if size is None else size
        if type(size) is not int or size < 1:
            raise ValueError("PushTJax render size must be a positive integer")
        rows = self._pack_render_state(self._state) if states is None else states
        with jax.default_device(self._jax_device):
            frames = _render(
                _from_torch(rows), size=size, with_target=self.with_target, render_action=self.render_action
            )
        frames = torch.from_dlpack(frames)
        if out is not None:
            if out.shape != frames.shape or out.device != frames.device or out.dtype != torch.uint8:
                raise ValueError("PushTJax output buffer must match the rendered frames")
            out.copy_(frames)
            return out
        return frames
