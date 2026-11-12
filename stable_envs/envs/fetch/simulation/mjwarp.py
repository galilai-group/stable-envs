"""Fetch physics on CUDA: one MuJoCo Warp world per environment, with the batched Fetch step in Torch."""

from __future__ import annotations

import copy

import numpy as np
import torch
from tensordict import TensorDict

from .batch import FetchLanes

#: Contacts per world: a grasp of the block on the table makes fewer than twenty.
CONTACTS_PER_WORLD = 32
#: Constraint rows per world: the mocap weld, joint limits and up to six rows per contact.
CONSTRAINTS_PER_WORLD = 256
#: Sides of the prism that stands in for Slide's cylinder puck on MuJoCo Warp.
PUCK_SIDES = 24


def physics_model(task):
    """A copy of the source model that MuJoCo Warp simulates close to MuJoCo.

    MuJoCo Warp makes one contact per cylinder pair and rejects nonzero geom margins under multi-contact CCD.
    A model with a cylinder (Slide's puck) gets a 24-sided prism of the same size and inertia, and zero margins,
    so that the puck rests on several contacts. The other models keep their 1 mm robot margins and step with
    multi-contact CCD and native CCD off.
    """
    import mujoco

    source = task.model
    cylinders = [g for g in range(source.ngeom) if source.geom_type[g] == mujoco.mjtGeom.mjGEOM_CYLINDER]
    if not cylinders:
        model = copy.deepcopy(source)
        model.opt.disableflags |= int(mujoco.mjtDisableBit.mjDSBL_MULTICCD) | int(
            mujoco.mjtDisableBit.mjDSBL_NATIVECCD
        )
        return model
    spec = mujoco.MjSpec.from_file(task.xml)
    for g in cylinders:
        geom = spec.geom(mujoco.mj_id2name(source, mujoco.mjtObj.mjOBJ_GEOM, g))
        radius, half = float(source.geom_size[g, 0]), float(source.geom_size[g, 1])
        angle = np.arange(PUCK_SIDES) * 2 * np.pi / PUCK_SIDES
        ring = np.stack([radius * np.cos(angle), radius * np.sin(angle)], 1)
        mesh = spec.add_mesh()
        mesh.name = f"se:prism{g}"
        rims = [np.c_[ring, np.full(PUCK_SIDES, z)] for z in (-half, half)]
        mesh.uservert = np.concatenate(rims).reshape(-1).tolist()
        geom.type = mujoco.mjtGeom.mjGEOM_MESH
        geom.meshname = mesh.name
    model = spec.compile()
    if (model.nq, model.nv, model.nbody, model.ngeom) != (source.nq, source.nv, source.nbody, source.ngeom):
        raise ValueError("Fetch prism model changed the model layout")
    # The source's inertia and its mocap weld, which gymnasium-robotics' setup resets.
    for name in ("body_mass", "body_inertia", "body_ipos", "body_iquat", "eq_data"):
        getattr(model, name)[:] = getattr(source, name)
    model.geom_margin[:] = 0.0
    return model


class FetchWarpAdapter(FetchLanes):
    """Fetch lanes on CUDA: the shared ``WarpBatch`` steps float32 MuJoCo Warp physics; the Fetch step runs in Torch.

    The step needs no host synchronization. Resets draw from each lane's PCG64 generator on the host, as the CPU
    backend does, so seeds give the same starts and goals. Physics is approximate: float32, and the contact
    changes of ``physics_model``.
    """

    execution_mode = "NATIVE_WARP"

    def __init__(self, num_envs, device="cuda", seed=0, task="PickAndPlace", reward_type="sparse"):
        from ...._execution.mjbatch import WarpBatch

        self._init_lanes(num_envs, device, task, reward_type)
        t, B = self.task, self.num_envs
        self.batch = WarpBatch(
            physics_model(t),
            B,
            device,
            frame_skip=t.n_substeps,
            outputs=("xpos", "xquat", "site_xpos"),
            batched_fields=("body_mass",) if t.has_object else (),
            naconmax=CONTACTS_PER_WORLD * B,
            njmax=CONSTRAINTS_PER_WORLD,
        )
        self.device = dev = self.batch.device
        data = self.batch.data
        self._qpos, self._xpos, self._xquat, self._site_xpos = (
            data(name) for name in ("qpos", "xpos", "xquat", "site_xpos")
        )
        f32 = dict(device=dev, dtype=torch.float32)
        self._mocap, self._body = t.welds[0]
        self._initial_qvel = torch.as_tensor(t.initial_qvel, **f32)
        self._rot = torch.tensor([1.0, 0.0, 1.0, 0.0], **f32)
        self._fingers = torch.as_tensor(t.fingers, device=dev)
        self._ctrl_qpos = torch.as_tensor(t.ctrl_qpos, device=dev, dtype=torch.long)
        self._mocap_pos = data("mocap_pos").clone()
        self._mocap_quat = data("mocap_quat").clone()
        self._grip_pos = torch.zeros(B, 3, **f32)
        self._grip_quat = torch.zeros(B, 4, **f32)
        self._elapsed = torch.zeros(B, dtype=torch.int64, device=dev)
        self._reward_dtype = torch.float32 if t.reward_type == "sparse" else torch.float64
        self._threshold = float(t.distance_threshold)
        self._words_tensor = None
        self.reset(seed=seed)

    def _body_mass(self):
        # WarpBatch batches body_mass per world; Fetch writes one value per lane rather than one model per lane.
        return self.batch._wp.to_torch(self.batch._m.body_mass)

    def _upload_lanes(self):
        dev = self.device
        self._goal64 = torch.as_tensor(self._goal, device=dev)
        self._anchor64 = torch.as_tensor(self._anchor, device=dev)
        self._goal_dev = self._goal64.to(torch.float32)
        self._limits_dev = torch.as_tensor(self._limits, device=dev)

    def _set_episode_limits(self, limits):
        super()._set_episode_limits(limits)
        self._limits_dev = torch.as_tensor(self._limits, device=self.device)

    def _read_controller(self):
        self._grip_pos.copy_(self._xpos[:, self._body])
        self._grip_quat.copy_(self._xquat[:, self._body])

    def reset(self, seed: int | None = None, variations=None) -> TensorDict:
        self._check_open()
        t, B, dev = self.task, self.num_envs, self.device
        qpos = np.empty((B, t.nq))
        self._draw_resets(seed, variations, qpos)
        self._words_tensor = None
        self._upload_lanes()
        self.batch.check_buffers()
        if t.has_object:
            self._body_mass()[:, t.object] = torch.as_tensor(self._mass, device=dev, dtype=torch.float32)
        # The template is mj_resetData's state: zero warmstart and controls, mocap bodies at their model poses.
        self.batch.reset()
        self.batch.set_physics(
            {"qpos": qpos, "qvel": self._initial_qvel.expand(B, -1), "time": np.full(B, float(t.initial_time))}
        )
        self._read_controller()
        self._elapsed.zero_()
        self._outputs = {
            "reward": torch.zeros(B, dtype=self._reward_dtype, device=dev),
            "terminated": torch.zeros(B, dtype=torch.bool, device=dev),
            "truncated": torch.zeros(B, dtype=torch.bool, device=dev),
            "success": torch.zeros(B, dtype=torch.float32, device=dev),
        }
        return self.get_state()

    def _words_dev(self):
        if self._words_tensor is None:
            self._words_tensor = torch.as_tensor(self._rng_words(), device=self.device)
        return self._words_tensor

    def get_state(self) -> TensorDict:
        self._check_open()
        words = self._words_dev()
        data = self.batch.data
        f64 = lambda x: x.to(torch.float64, copy=True)  # noqa: E731
        columns = {
            "qpos": f64(self._qpos),
            "qvel": f64(data("qvel")),
            "time": f64(data("time")[:, None]),
            ("task", "initial_gripper_xpos"): self._anchor64.clone(),
            "warmstart": f64(data("qacc_warmstart")),
            ("controller", "position"): f64(self._grip_pos),
            ("controller", "quaternion"): f64(self._grip_quat),
            "goal": self._goal64.clone(),
            ("lifecycle", "elapsed_steps"): self._elapsed[:, None].clone(),
            ("rng", "state_words"): words[:, 0:4].clone(),
            ("rng", "increment_words"): words[:, 4:8].clone(),
            ("rng", "has_uint32"): words[:, 8:9].clone(),
            ("rng", "uinteger"): words[:, 9:10].clone(),
        }
        return TensorDict(columns, [self.num_envs], device=self.device)

    def set_state(self, state: TensorDict) -> None:
        self.validate(state)
        dev = self.device
        f32 = lambda key: state[key].to(dev, torch.float32)  # noqa: E731
        self.batch.set_physics({"qpos": f32("qpos"), "qvel": f32("qvel"), "time": f32("time")[:, 0]})
        self.batch.set_physics({"qacc_warmstart": f32("warmstart")}, forward=False)
        self._grip_pos.copy_(f32(("controller", "position")))
        self._grip_quat.copy_(f32(("controller", "quaternion")))
        self._elapsed.copy_(state["lifecycle", "elapsed_steps"][:, 0].to(dev))
        host = state.cpu()
        self._goal[:] = host["goal"].numpy()
        self._anchor[:] = host["task", "initial_gripper_xpos"].numpy()
        self._restore_rngs(host)
        self._words_tensor = None
        self._upload_lanes()

    def check_step(self, actions, active):
        if tuple(actions.shape) != (self.num_envs, 4):
            raise ValueError("Fetch action must have shape (num_envs,4)")
        if active is not None and tuple(active.shape) != (self.num_envs,):
            raise ValueError("Fetch active mask must have shape (num_envs,)")
        return actions, active

    def advance(self, actions, *, active=None) -> TensorDict:
        """The Fetch step of ``FetchBatchAdapter.advance`` on device tensors, in float32."""
        self._check_open()
        t, dev = self.task, self.device
        actions = torch.as_tensor(actions, device=dev)
        if active is not None:
            active = torch.as_tensor(active, device=dev, dtype=torch.bool)
        self.check_step(actions, active)
        action = actions.to(torch.float32).clamp(-1.0, 1.0)
        ctrl = None
        if t.relative:
            gripper = torch.zeros_like(action[:, 3]) if t.block_gripper else action[:, 3]
            ctrl = self._qpos[:, self._ctrl_qpos] + gripper[:, None]
        self._mocap_pos[:, self._mocap] = self._grip_pos + action[:, :3] * 0.05
        self._mocap_quat[:, self._mocap] = self._grip_quat + self._rot
        self.batch.step(ctrl, mocap_pos=self._mocap_pos, mocap_quat=self._mocap_quat, active=active)
        if t.block_gripper:
            qpos = self._qpos.index_fill(1, self._fingers, 0.0)
            if active is not None:
                qpos = torch.where(active[:, None], qpos, self._qpos)
            self.batch.set_physics({"qpos": qpos})
        achieved = self._site_xpos[:, t.goal_site]
        d = torch.linalg.vector_norm(achieved - self._goal_dev, dim=-1)
        success = (d < self._threshold).to(torch.float32)
        reward = -(d > self._threshold).to(torch.float32) if t.reward_type == "sparse" else -d.to(torch.float64)
        out = self._outputs
        if active is None:
            self._read_controller()
            self._elapsed += 1
            out["success"], out["reward"] = success, reward
        else:
            column = active[:, None]
            self._grip_pos.copy_(torch.where(column, self._xpos[:, self._body], self._grip_pos))
            self._grip_quat.copy_(torch.where(column, self._xquat[:, self._body], self._grip_quat))
            self._elapsed += active.to(torch.int64)
            out["success"] = torch.where(active, success, out["success"])
            out["reward"] = torch.where(active, reward, out["reward"])
        truncated = self._elapsed >= self._limits_dev
        out["truncated"] = truncated if active is None else torch.where(active, truncated, out["truncated"])
        return TensorDict({key: value.clone() for key, value in out.items()}, [self.num_envs], device=dev)

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            self.batch.close()
            self.task.close()
