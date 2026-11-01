"""MJX Reacher dynamics adapted from fracapuano/envX at d26c817c1124229c819be574f137cd68bda4380a.

Copyright 2026 Francesco Capuano. SPDX-License-Identifier: Apache-2.0
See licenses/envX-APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.
"""

from __future__ import annotations

import os
import xml.etree.ElementTree as ET

import numpy as np
import torch
from tensordict import TensorDict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax  # noqa: E402
import jax.dlpack  # noqa: E402
import jax.numpy as jnp  # noqa: E402
import mujoco  # noqa: E402
from dm_control.suite import reacher  # noqa: E402
from mujoco import mjx  # noqa: E402

from .first_family import ReacherAdapter  # noqa: E402

# Density-dependent fields, including constants from mj_setConst.
_MODEL_FIELDS = (
    "body_mass",
    "body_inertia",
    "body_ipos",
    "body_iquat",
    "body_subtreemass",
    "body_invweight0",
    "dof_invweight0",
    "dof_M0",
)


def _from_torch(value):
    value = value.detach().contiguous()
    alignment = 16 if value.is_cuda else 64
    if value.data_ptr() % alignment:
        value = value.clone()
    return jax.dlpack.from_dlpack(value, copy=False)


def _model():
    xml, assets = reacher.get_model_and_assets()
    root = ET.fromstring(xml)
    world = root.find("worldbody")
    target = world.find("geom[@name='target']")
    position = target.attrib.pop("pos")
    target.set("pos", "0 0 0")
    target.set("size", str(reacher._SMALL_TARGET))
    world.remove(target)
    # Mocap positions vary per environment without duplicating the model.
    body = ET.SubElement(world, "body", name="target_mocap", mocap="true", pos=position)
    body.append(target)
    return mujoco.MjModel.from_xml_string(ET.tostring(root, encoding="unicode"), dict(assets))


class ReacherMJXAdapter(ReacherAdapter):
    """Batched MJX Reacher dynamics."""

    def __init__(self, num_envs, device="cuda", seed=0):
        super().__init__(num_envs, device, seed)
        backend = "gpu" if self.device.type == "cuda" else "cpu"
        devices = jax.devices(backend)
        index = self.device.index
        if index is None:
            index = torch.cuda.current_device() if backend == "gpu" else 0
        self._jax_device = devices[index]
        self._native_model = _model()
        self._finger = self._native_model.geom("finger").id
        self._target = self._native_model.geom("target").id
        self._radius = self._native_model.geom_size[[self._finger, self._target], 0].sum()
        with jax.default_device(self._jax_device):
            model = mjx.put_model(self._native_model, impl="jax")
            data = mjx.make_data(self._native_model, impl="jax")
            self._data = jax.vmap(lambda _: data)(jnp.arange(self.num_envs))
            # Reset and restore must clear solver buffers.
            self._template = self._data

            def model_for(values):
                # Passing Model to jit rehashes its static arrays.
                return model.replace(
                    **{name: values[name] for name in _MODEL_FIELDS},
                    stat=model.stat.replace(meaninertia=values["meaninertia"]),
                )

            self._forward = jax.jit(jax.vmap(lambda values, data: mjx.forward(model_for(values), data)))
            self._advance = jax.jit(
                jax.vmap(lambda values, data, action: self._step_one(model_for(values), data, action))
            )
            self._select = jax.jit(jax.vmap(lambda old, new, mask: old.where(mask, new)))
        self._upload_model()
        self._load_state(super().get_state())

    def _observe(self, data):
        delta = data.geom_xpos[..., self._target, :2] - data.geom_xpos[..., self._finger, :2]
        return jnp.concatenate((data.qpos, delta, data.qvel), axis=-1), jnp.linalg.norm(delta, axis=-1) <= self._radius

    def _step_one(self, model, data, action):
        data = mjx.step(model, data.replace(ctrl=action))
        # mjx.step leaves geom_xpos at the pre-integration pose.
        data = mjx.kinematics(model, data)
        observation, success = self._observe(data)
        return data, observation, success

    def _upload_model(self):
        models = [env.physics.model.ptr for env in self.envs]
        with jax.default_device(self._jax_device):
            fields = {}
            for name in _MODEL_FIELDS:
                values = np.broadcast_to(
                    getattr(self._native_model, name), (self.num_envs, *getattr(self._native_model, name).shape)
                ).copy()
                source = np.stack([getattr(model, name) for model in models])
                values[:, : source.shape[1]] = source
                fields[name] = jnp.asarray(values, dtype=jnp.float32)
            fields["meaninertia"] = jnp.asarray([m.stat.meaninertia for m in models], dtype=jnp.float32)
            self._model_values = fields

    def _load_state(self, state):
        self._state = state.to(self.device).clone()
        self._counts = state["lifecycle", "step_count"].cpu().numpy().copy()
        self._reset_next = state["lifecycle", "reset_next_step"].cpu().numpy().copy()
        self._load_data()

    def _load_data(self):
        value = lambda key: _from_torch(self._state[key].to(torch.float32))
        warmstart = value("warmstart")
        self._data = self._forward(
            self._model_values,
            self._template.replace(
                qpos=value("qpos"),
                qvel=value("qvel"),
                time=value("time")[:, 0],
                mocap_pos=value(("task", "target_pos"))[:, None, :],
            ),
        ).replace(qacc_warmstart=warmstart)

    def reset(self, seed=None):
        state = super().reset(seed)
        if hasattr(self, "_data"):
            self._load_state(super().get_state())
            state = self.get_state()
        return state

    def configure_physics(self, values):
        super().configure_physics(values)
        if hasattr(self, "_data"):
            self._upload_model()

    def set_state(self, state):
        super().set_state(state)
        self._upload_model()
        self._load_state(state)

    def get_state(self):
        self._check_open()
        return self._state.clone() if hasattr(self, "_state") else super().get_state()

    def step(self, actions, *, active=None, capture_state=True):
        self._check_open()
        actions = torch.as_tensor(actions, device=self.device)
        if actions.shape != (self.num_envs, 2) or not bool(torch.isfinite(actions).all()):
            raise ValueError("Reacher actions must be finite with shape (num_envs, 2)")
        if active is not None and (active.shape != (self.num_envs,) or not bool(torch.as_tensor(active).all())):
            raise ValueError(
                "Accepted DMC wrappers do not expose terminal lanes; partial active masks are unsupported"
            )
        action = _from_torch(actions.to(torch.float32))
        reward = torch.zeros(self.num_envs, dtype=torch.float64, device=self.device)
        for _ in range(2):
            resetting = self._reset_next[:, 0].copy()
            if resetting.any():
                cumulative = self._state["task", "cumulative_reward"].clone()
                self._reset_lanes(np.flatnonzero(resetting))
                mask = torch.as_tensor(resetting, device=self.device)
                self._state[mask] = super().get_state()[mask]
                self._state["task", "cumulative_reward"] = cumulative
                previous = self._data
                self._load_data()
                reset_mask = _from_torch(mask)
                self._data = self._select(previous, self._data, reset_mask)
            before = self._data
            self._data, observation, success = self._advance(self._model_values, self._data, action)
            if resetting.any():
                mask = reset_mask
                self._data = self._select(self._data, before, mask)
                reset_observation, _ = self._observe(before)
                observation = jnp.where(mask[:, None], reset_observation, observation)
                success = success & ~mask
                self._state["time"][torch.as_tensor(resetting, device=self.device)] = 0
                self._state["lifecycle", "step_count"][torch.as_tensor(resetting, device=self.device)] = -1
            reward += torch.from_dlpack(success)
            self._counts = np.where(resetting[:, None], 0, self._counts + 1)
            self._reset_next = self._counts >= 1000
            if resetting.any():
                self._state["time"] += (~torch.as_tensor(resetting, device=self.device)).to(torch.float64)[
                    :, None
                ] * 0.02
            else:
                self._state["time"] += 0.02
            self._state["lifecycle", "step_count"] += 1
        for key, data in (
            ("qpos", self._data.qpos),
            ("qvel", self._data.qvel),
            ("warmstart", self._data.qacc_warmstart),
        ):
            self._state[key] = torch.from_dlpack(data).to(torch.float64)
        self._state["lifecycle", "reset_next_step"] = self._state["lifecycle", "step_count"] >= 1000
        self._state["task", "cumulative_reward"] = self._state["task", "cumulative_reward"] + reward[:, None]
        observation = torch.from_dlpack(observation)
        return TensorDict(
            {
                **({"state": self.get_state()} if capture_state else {}),
                "reward": reward,
                "observation": observation,
                "terminated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "truncated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
                "score": self._state["task", "cumulative_reward"][:, 0] / 1000,
            },
            [self.num_envs],
            device=self.device,
        )
