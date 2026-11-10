"""Cube controller and MJX-Warp stepping adapted from fracapuano/envX at d26c817.

Copyright 2026 Francesco Capuano. SPDX-License-Identifier: Apache-2.0
Modified to implement stable-envs State, Variation and DLPack contracts.
See licenses/envX-APACHE-2.0.txt and THIRD_PARTY_NOTICES.md.
"""

import copy
import hashlib
import os
from importlib.metadata import version

import numpy as np
import torch
from tensordict import TensorDict

os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")

import jax
import jax.dlpack
import jax.numpy as jnp
from mujoco import mjx

from .compatibility import COMPILED_FIELDS
from .manipulation import FLAGS, OGBManipulationAdapter


def _from_torch(value):
    value = value.detach().contiguous()
    if value.data_ptr() % (16 if value.is_cuda else 64):
        value = value.clone()
    return jax.dlpack.from_dlpack(value, copy=False)


def _quat_mul(q0: jax.Array, q1: jax.Array) -> jax.Array:
    """Hamilton product for MuJoCo ``(w, x, y, z)`` quaternions."""
    w0, x0, y0, z0 = q0
    w1, x1, y1, z1 = q1
    return jnp.asarray(
        [
            w0 * w1 - x0 * x1 - y0 * y1 - z0 * z1,
            w0 * x1 + x0 * w1 + y0 * z1 - z0 * y1,
            w0 * y1 - x0 * z1 + y0 * w1 + z0 * x1,
            w0 * z1 + x0 * y1 - y0 * x1 + z0 * w1,
        ]
    )


def _quat_conjugate(q: jax.Array) -> jax.Array:
    return q * jnp.asarray([1.0, -1.0, -1.0, -1.0], dtype=q.dtype)


def _quat_rotate(q: jax.Array, vector: jax.Array) -> jax.Array:
    """Rotate a vector without constructing a rotation matrix."""
    qvec = q[1:]
    uv = jnp.cross(qvec, vector)
    uuv = jnp.cross(qvec, uv)
    return vector + 2.0 * (q[0] * uv + uuv)


def _quat_to_rotvec(q: jax.Array) -> jax.Array:
    """MuJoCo-compatible shortest-axis quaternion logarithm."""
    q = q / jnp.maximum(jnp.linalg.norm(q), jnp.finfo(q.dtype).tiny)
    q = jnp.where(q[0] < 0.0, -q, q)
    vector_norm = jnp.linalg.norm(q[1:])
    angle = 2.0 * jnp.arctan2(vector_norm, jnp.maximum(q[0], 0.0))
    scale = jnp.where(vector_norm > 1e-8, angle / vector_norm, 2.0)
    return q[1:] * scale


def _yaw_quat(yaw: jax.Array) -> jax.Array:
    half_yaw = 0.5 * yaw
    return jnp.asarray([jnp.cos(half_yaw), 0.0, 0.0, jnp.sin(half_yaw)])


class CubeMJXAdapter(OGBManipulationAdapter):
    execution_mode = "NATIVE_JAX"
    rotation_tolerance = 1e-6

    def __init__(self, family="Cube", num_envs=1, device="cuda", seed=0, reward_task_id=None):
        if family != "Cube":
            raise ValueError("CubeMJXAdapter supports only Cube")
        device = torch.device(device)
        try:
            self._jax_device = jax.devices("gpu" if device.type == "cuda" else "cpu")[
                device.index
                if device.index is not None
                else torch.cuda.current_device()
                if device.type == "cuda"
                else 0
            ]
        except (RuntimeError, IndexError) as error:
            raise RuntimeError(
                "Cube needs JAX CUDA support; install 'stable-envs[ogbench-manipulation-gpu]'"
            ) from error
        if device.type == "cuda":
            from ...._core.warp_cuda import require_warp_cuda

            require_warp_cuda("Cube MJX physics")
        super().__init__(family, num_envs, device, seed, reward_task_id)
        try:
            self._cpu_signature = self.signature.copy()
            fingerprint = (
                self.signature.tobytes()
                + "|".join(f"{p}={version(p)}" for p in ("mujoco-mjx", "mujoco-warp", "warp-lang", "jax")).encode()
            )
            self.signature = np.frombuffer(hashlib.sha256(fingerprint).digest(), np.uint8).copy()
            with jax.default_device(self._jax_device):
                self._build_controller()
                self._native_keys = set(self.template.keys(True, True))
                # Warp reuses both velocity terms during the next implicit step.
                for name, count in (("cvel", self.envs[0].model.nbody), ("cdof_dot", self.envs[0].model.nv)):
                    self.template["mjx", name] = torch.zeros(count, 6, dtype=torch.float32, device=self.device)
                    self.fields["mjx", name] = ((count, 6), torch.float32)
                self._upload()
        except BaseException:
            self.close()
            raise

    def _build_controller(self):
        env = self.envs[0]
        self._model = mjx.put_model(env.model, device=self._jax_device, impl="warp")
        self._model_fields = tuple(
            name
            for name in COMPILED_FIELDS
            if name in self._model.__dataclass_fields__ and isinstance(getattr(self._model, name), jax.Array)
        )
        self._ik_mj_model = env._ik._model
        self._ik_model = mjx.put_model(self._ik_mj_model, device=self._jax_device, impl="jax")
        self._ik_data = mjx.make_data(self._ik_mj_model, device=self._jax_device, impl="jax")
        self._arm_qpos_ids = jnp.asarray(env.model.jnt_qposadr[env._arm_joint_ids])
        self._arm_actuator_ids = jnp.asarray(env._arm_actuator_ids)
        self._gripper_actuator_ids = jnp.asarray(env._gripper_actuator_ids)
        self._gripper_qpos_id = int(env.model.jnt_qposadr[env._gripper_opening_joint_id])
        self._pinch_site_id = int(env._pinch_site_id)
        self._right_pad_body_id = int(env.model.body("ur5e/robotiq/right_pad").id)
        self._workspace_low, self._workspace_high = jnp.asarray(env._workspace_bounds)
        self._down_quat = jnp.asarray(env._effector_down_rotation.wxyz)
        self._pinch_to_attach_quat = jnp.asarray(env._T_pa.rotation().wxyz)
        self._pinch_to_attach_pos = jnp.asarray(env._T_pa.translation())
        self._ik_site_id = int(self._ik_mj_model.site("attachment_site").id)
        self._ik_site_body_id = int(self._ik_mj_model.site_bodyid[self._ik_site_id])
        self._ik_site_quat = jnp.asarray(self._ik_mj_model.site_quat[self._ik_site_id])
        self._cube_qpos_id = int(env.model.joint("object_joint_0").qposadr[0])
        self._target_id = int(env._cube_target_mocap_ids[0])
        self._advance = jax.jit(jax.vmap(self._physics_step))
        self._select = jax.jit(jax.vmap(lambda old, new, active: old.where(active, new)))

    def _upload(self):
        models, data, parameters = {}, [], []
        for env in self.envs:
            model = env.model
            if id(model) not in models:
                models[id(model)] = mjx.put_model(model, device=self._jax_device, impl="warp")
            parameters.append({name: getattr(models[id(model)], name) for name in self._model_fields})
            data.append(
                mjx.put_data(
                    model,
                    copy.copy(env.data),
                    device=self._jax_device,
                    impl="warp",
                    naconmax=max(512, 64 * self.num_envs),
                    njmax=1024,
                )
            )
        self._parameters = jax.tree.map(lambda *values: jnp.stack(values), *parameters)
        # Warp contact buffers are shared by all worlds, rather than stacked per lane.
        batch = jax.vmap(lambda _: data[0])(jnp.arange(self.num_envs))
        self._data = jax.tree.map(
            lambda target, *rows: jnp.stack(rows) if target.ndim > rows[0].ndim else rows[0], batch, *data
        )
        self._state = super().get_state()
        # put_data recomputes kinematics; the source controller uses the saved, pre-integration pose.
        self._data = self._data.replace(
            site_xpos=self._data.site_xpos.at[:, self._pinch_site_id].set(
                _from_torch(self._state["controller", "site_xpos"][:, 0].to(torch.float32))
            ),
            site_xmat=self._data.site_xmat.at[:, self._pinch_site_id].set(
                _from_torch(self._state["controller", "site_xmat"][:, 0].to(torch.float32)).reshape(-1, 3, 3)
            ),
        )
        self._data = self._data.tree_replace(
            {
                "_impl.cfrc_ext": self._data._impl.cfrc_ext.at[:, self._right_pad_body_id].set(
                    _from_torch(self._state["output", "gripper_force"].to(torch.float32))
                )
            }
        )
        self._reward_offset = -(self._state["task", "reward_id"][:, 0] != -1).to(torch.float64)
        self._outputs = self._evaluate()
        self._sync_state()

    def _evaluate(self):
        qpos = torch.from_dlpack(self._data.qpos)
        targets = torch.from_dlpack(self._data.mocap_pos)[:, self._target_id]
        success = (qpos[:, self._cube_qpos_id : self._cube_qpos_id + 3] - targets).norm(dim=-1) <= 0.04
        reward = success.to(torch.float64) + self._reward_offset
        return TensorDict(
            dict(reward=reward, success=success, terminated=success.clone(), truncated=torch.zeros_like(success)),
            [self.num_envs],
            device=self.device,
        )

    def _sync_state(self):
        for name in ("cvel", "cdof_dot"):
            self._state["mjx", name] = torch.from_dlpack(getattr(self._data, name))
        for name in FLAGS:
            key = ("physics", name)
            if key in self._state.keys(True, True):
                value = getattr(self._data, "qacc_warmstart" if name == "warmstart" else name)
                self._state[key] = torch.from_dlpack(value).reshape_as(self._state[key]).to(self._state[key].dtype)
        self._state["controller", "site_xpos"] = torch.from_dlpack(self._data.site_xpos)[
            :, self._pinch_site_id : self._pinch_site_id + 1
        ].to(torch.float64)
        self._state["controller", "site_xmat"] = (
            torch.from_dlpack(self._data.site_xmat)[:, self._pinch_site_id : self._pinch_site_id + 1]
            .reshape(self.num_envs, 1, 9)
            .to(torch.float64)
        )
        self._state["output", "gripper_force"] = torch.from_dlpack(self._data._impl.cfrc_ext)[
            :, self._right_pad_body_id
        ].to(torch.float64)

    def get_state(self):
        self._check_open()
        return self._state.clone() if hasattr(self, "_state") else super().get_state()

    def render_columns(self):
        from .render_state import projected_keys

        self._check_open()
        return {key: self._state[key].detach().cpu().numpy().copy() for key in projected_keys("Cube")}

    def reset(self, seed=None, options=None):
        super().reset(seed, options)
        if hasattr(self, "_model"):
            with jax.default_device(self._jax_device):
                self._upload()
        return self.get_state()

    def validate(self, state, *, _batch_size=None):
        if isinstance(state, TensorDict) and set(state.keys(True, True)) == self._native_keys:
            signature = state["compatibility", "signature"]
            if (
                signature.dtype == torch.uint8
                and signature.ndim == 2
                and signature.shape[-1] == 32
                and bool((signature == torch.as_tensor(self._cpu_signature, device=signature.device)).all())
            ):
                state = state.clone(recurse=False)
                state["compatibility", "signature"] = torch.as_tensor(
                    self.signature, device=signature.device
                ).expand_as(signature)
                for name in ("cvel", "cdof_dot"):
                    shape, dtype = self.fields["mjx", name]
                    state["mjx", name] = torch.zeros(len(state), *shape, dtype=dtype, device=state.device)
        return super().validate(state, _batch_size=_batch_size)

    def prepare_restore(self, state):
        super().prepare_restore(state)
        staged = object.__new__(type(self))
        staged.__dict__ = self.__dict__.copy()
        staged.envs, staged._prepared_envs = self._prepared_envs, None
        try:
            with jax.default_device(self._jax_device):
                staged._upload()
                if "mjx" in state.keys():
                    staged._data = staged._data.replace(
                        **{name: _from_torch(state["mjx", name].to(self.device)) for name in ("cvel", "cdof_dot")}
                    )
                    staged._sync_state()
        except BaseException:
            self.discard_restore()
            raise
        self._prepared_gpu = staged

    def commit_restore(self):
        super().commit_restore()
        staged, self._prepared_gpu = self._prepared_gpu, None
        for name in ("_data", "_parameters", "_state", "_outputs", "_reward_offset"):
            setattr(self, name, getattr(staged, name))

    def discard_restore(self):
        super().discard_restore()
        self._prepared_gpu = None

    def configure(self, values):
        super().prepare_restore(self.get_state())
        super().commit_restore()
        super().configure(values)
        with jax.default_device(self._jax_device):
            self._upload()

    def step(self, actions, *, active=None, capture_state=True):
        self._check_open()
        actions = torch.as_tensor(actions, device=self.device)
        if (
            actions.shape != (self.num_envs, 5)
            or actions.dtype not in (torch.float32, torch.float64)
            or not bool(actions.isfinite().all())
        ):
            raise ValueError("Actions must be finite float32/float64 [num_envs, 5]")
        active = (
            torch.ones(self.num_envs, dtype=torch.bool, device=self.device)
            if active is None
            else torch.as_tensor(active, device=self.device)
        )
        if active.shape != (self.num_envs,) or active.dtype != torch.bool:
            raise ValueError("active must be a bool vector with shape [num_envs]")
        previous = self._outputs
        old = self._data
        new = self._advance(self._parameters, old, _from_torch(actions.to(torch.float32)))
        self._data = self._select(old, new, _from_torch(active))
        for name in ("qpos", "qvel"):
            key = ("output", "prev_" + name)
            self._state[key] = torch.where(
                active[:, None], torch.from_dlpack(getattr(old, name)).to(torch.float64), self._state[key]
            )
        post = self._evaluate()
        self._state["task", "success"] = torch.where(
            active[:, None], post["success"][:, None], self._state["task", "success"]
        )
        result = post
        self._outputs = TensorDict(
            {key: torch.where(active, value, previous[key]) for key, value in post.items()},
            [self.num_envs],
            device=self.device,
        )
        for key in result.keys():
            result[key] = torch.where(active, result[key], previous[key])
        self._sync_state()
        if capture_state:
            result["state"] = self.get_state()
        return result

    def _solve_ik(self, target_pos: jax.Array, target_quat: jax.Array, current_qpos: jax.Array) -> jax.Array:
        """Twenty fixed-shape differential-IK iterations, matching CubeEnv."""

        def iteration(_, qpos):
            data = self._ik_data.replace(qpos=qpos, qvel=jnp.zeros_like(self._ik_data.qvel))
            data = mjx.kinematics(self._ik_model, data)
            data = mjx.com_pos(self._ik_model, data)

            current_pos = data.site_xpos[self._ik_site_id]
            current_quat = _quat_mul(data.xquat[self._ik_site_body_id], self._ik_site_quat)
            pos_error = target_pos - current_pos
            rot_error = _quat_to_rotvec(_quat_mul(target_quat, _quat_conjugate(current_quat)))
            error = jnp.concatenate([pos_error, rot_error])

            jac_pos, jac_rot = mjx.jac(
                self._ik_model,
                data,
                current_pos,
                jnp.asarray(self._ik_site_body_id),
            )
            jacobian = jnp.concatenate([jac_pos.T, jac_rot.T], axis=0)
            damping = 1e-6 if qpos.dtype == jnp.float32 else 1e-12
            update = jacobian.T @ jnp.linalg.solve(
                jacobian @ jacobian.T + damping * jnp.eye(6, dtype=qpos.dtype),
                error,
            )
            update_max = jnp.max(jnp.abs(update))
            update *= jnp.minimum(1.0, jnp.deg2rad(45.0) / jnp.maximum(update_max, 1e-12))
            reached = (jnp.linalg.norm(pos_error) <= 1e-4) & (jnp.linalg.norm(rot_error) <= 1e-4)
            return jnp.where(reached, qpos, qpos + update)

        return jax.lax.fori_loop(0, 20, iteration, current_qpos)

    def _pinch_to_attach(self, pinch_pos: jax.Array, pinch_quat: jax.Array) -> tuple[jax.Array, jax.Array]:
        attach_pos = pinch_pos + _quat_rotate(pinch_quat, self._pinch_to_attach_pos)
        attach_quat = _quat_mul(pinch_quat, self._pinch_to_attach_quat)
        return attach_pos, attach_quat

    def _set_control(self, data, action: jax.Array):
        # The normalized OGBench action ranges are [0.05m, 0.3rad, 1.0].
        scaled_action = action * jnp.asarray([0.05, 0.05, 0.05, 0.3, 1.0])
        pinch_pos = data.site_xpos[self._pinch_site_id]
        pinch_matrix = data.site_xmat[self._pinch_site_id]
        pinch_yaw = jnp.arctan2(pinch_matrix[1, 0], pinch_matrix[0, 0])
        gripper_opening = jnp.clip(data.qpos[self._gripper_qpos_id] / 0.8, 0.0, 1.0)

        target_pinch_pos = jnp.clip(pinch_pos + scaled_action[:3], self._workspace_low, self._workspace_high)
        yaw = pinch_yaw + scaled_action[3]
        target_yaw = jnp.arctan2(jnp.sin(yaw), jnp.cos(yaw))
        target_pinch_quat = _quat_mul(_yaw_quat(target_yaw), self._down_quat)
        target_gripper = jnp.clip(gripper_opening + scaled_action[4], 0.0, 1.0)

        target_pos, target_quat = self._pinch_to_attach(target_pinch_pos, target_pinch_quat)
        arm_qpos = self._solve_ik(target_pos, target_quat, data.qpos[self._arm_qpos_ids])
        ctrl = data.ctrl.at[self._arm_actuator_ids].set(arm_qpos)
        ctrl = ctrl.at[self._gripper_actuator_ids].set(255.0 * target_gripper)
        return data.replace(ctrl=ctrl)

    def _physics_step(self, values, data, action):
        model = self._model.tree_replace(values)
        data = self._set_control(data, action)
        data = jax.lax.fori_loop(0, 24, lambda _, carry: mjx.step(model, carry), data)
        # Match the source's mj_rnePostConstraint after the last physics step.
        return mjx.step(model.tree_replace({"opt._impl.run_rne_postconstraint": True}), data)
