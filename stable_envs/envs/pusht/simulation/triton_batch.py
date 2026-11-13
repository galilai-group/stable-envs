"""PushT batches stepped by the Triton port of the contact solver on CUDA."""

import numpy as np
import torch
from tensordict import TensorDict

from . import triton_physics
from .batch import PushTBatchAdapter


class PushTTritonAdapter(PushTBatchAdapter):
    """Step PushT environments with two Triton kernel launches per step, one GPU thread per environment.

    The first kernel steps the physics; the second records the actions and computes the observation,
    reward and success. Both update the State fields in place.
    """

    def __init__(self, batch, device="cuda", seed=0, **kwargs):
        if torch.device(device).type != "cuda":
            raise ValueError("PushT Triton physics requires a CUDA device")
        super().__init__(batch, device, seed, **kwargs)

    def _initialize_device(self, fields=None, geometry=None):
        super()._initialize_device(fields, geometry)
        vertices, centers, inverse, radius, enabled = self._geometry
        self._kernel_geometry = (vertices, centers, inverse, radius, enabled.to(torch.uint8))
        self._scratch = torch.empty(
            triton_physics.FIELDS.value * self._impulse.shape[1] * self.num_envs,
            dtype=torch.float32,
            device=self.device,
        )
        self._work = triton_physics.workspace(self.num_envs, self._impulse.shape[1], self.device)
        self._everyone = torch.ones(self.num_envs, dtype=torch.uint8, device=self.device)
        self._refresh_symmetry()

    def _install_columns(self, columns):
        super()._install_columns(columns)
        self._refresh_symmetry()

    def _refresh_symmetry(self):
        shape = self._fixed["owned/variation/block/shape/current/value"].to(torch.int64)
        self._symmetry = self._symmetries[shape].contiguous()

    def _physics(self, target, single, active, everyone):
        # Resets and restores bind these fields to tensors that no State shares, so they change in place.
        triton_physics.advance(
            self._body,
            self._bias,
            self._impulse,
            self._identifiers,
            (self._tree, self._tree_bounds),
            self._tree_order,
            self._kernel_geometry,
            target.contiguous(),
            self._circle,
            self._everyone if everyone else active.to(torch.uint8),
            self._contacts,
            self._fixed["current/render_segment_count"],
            self._fixed["current/render_segments"],
            self._scratch,
            self._work,
        )

    def step(self, actions, *, active=None, capture_state=True):
        actions = torch.as_tensor(actions, device=self.device)
        discrete = self.template.discrete
        allowed = (torch.int32, torch.int64) if discrete else (torch.float32, torch.float64)
        if actions.shape != (self.num_envs, 2) or actions.dtype not in allowed:
            raise ValueError("PushT actions require shape [num_envs, 2] and a supported dtype")
        finite = torch.isfinite(actions).all()
        if not bool(finite & ((actions >= 0) & (actions < 16)).all() if discrete else finite):
            if not bool(finite):
                raise ValueError("PushT action must be finite")
            raise ValueError("PushT discrete indices must be in [0, 16)")
        everyone = active is None
        if not everyone:
            active = torch.as_tensor(active, device=self.device)
            if active.shape != (self.num_envs,) or active.dtype != torch.bool:
                raise ValueError("PushT active mask requires bool [num_envs]")
        mask = self._everyone if everyone else active.to(torch.uint8)
        actions = actions.contiguous()
        if discrete:
            polar = (actions.to(torch.float64) + 0.5) / 16
            theta = polar[:, 1] * (2 * np.pi)
            relative = torch.stack((polar[:, 0] * theta.cos(), polar[:, 0] * theta.sin()), 1).to(torch.float32)
        else:
            relative = actions
        target = (self._body[:, 0, :2].to(relative.dtype) + relative * 100).to(torch.float64)
        self._physics(target, relative.dtype == torch.float32, active, everyone)
        width = self._history_width() + 1
        if width > self._actions.shape[1]:
            capacity = max(16, 2 * self._actions.shape[1])
            self._actions = torch.cat(
                (self._actions, self._actions.new_zeros(self.num_envs, capacity - self._actions.shape[1], 2)), 1
            )
            self._codes = torch.cat(
                (self._codes, self._codes.new_zeros(self.num_envs, capacity - self._codes.shape[1])), 1
            )
        fixed = self._fixed
        reward, success = triton_physics.record(
            self._body,
            actions,
            relative,
            mask,
            self._lengths,
            self._actions,
            self._codes,
            (fixed["current/latest/value"], fixed["current/latest/type"], fixed["current/latest_present"]),
            fixed["current/observation"],
            fixed["current/goal/value"],
            self._symmetry,
        )
        self._width = width if everyone else None
        outputs = dict(reward=reward, success=success, terminated=success.clone(), truncated=torch.zeros_like(success))
        if capture_state:
            outputs["state"] = self.get_state()
        return TensorDict._new_unsafe(outputs, batch_size=torch.Size([self.num_envs]), device=self._state_device)
