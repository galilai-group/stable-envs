"""Step every DMC lane in one mujoco.batch and compute task outputs for all lanes at once."""

from __future__ import annotations

import contextlib
import math

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.protocol import SimulatorAdapter
from .state import export_state

XX, XZ, ZX, ZY, ZZ = 0, 2, 6, 7, 8  # MuJoCo's row-major xmat entries
_pow = np.frompyfunc(math.pow, 2, 1)


def tolerance(x, *args, **kwargs):
    """dm_control's ``rewards.tolerance`` on lane arrays."""
    from dm_control.utils import rewards

    return rewards.tolerance(x, *args, **kwargs)


def scalar_tolerance(x, bounds=(0.0, 0.0), margin=0.0, sigmoid="gaussian", value_at_margin=0.1):
    """``rewards.tolerance`` of each lane's scalar, for the gaussian and long_tail sigmoids."""
    lower, upper = bounds
    d = np.where(x < lower, lower - x, x - upper) / margin
    # dm_control squares a scalar with libm pow, which can round differently from x * x.
    if sigmoid == "gaussian":
        value = np.exp(-0.5 * _pow(d * np.sqrt(-2 * np.log(value_at_margin)), 2.0).astype(np.float64))
    elif sigmoid == "long_tail":
        value = 1 / (_pow(d * np.sqrt(1 / value_at_margin - 1), 2.0).astype(np.float64) + 1)
    else:
        raise ValueError(f"Unsupported scalar sigmoid {sigmoid!r}")
    return np.where((lower <= x) & (x <= upper), 1.0, value)


class DMCBatchAdapter(SimulatorAdapter):
    """DMC lanes stepped together in one mujoco.batch; dm_control resets, restores and varies them."""

    def __init__(self, lanes, task, threads):
        import mujoco
        from dm_control.rl.control import PhysicsError
        from mujoco.batch import Batch

        self.lanes, self._task, self._cumulative = lanes, task, lanes._cumulative
        self.key, self.num_envs, self.device = lanes.key, lanes.num_envs, lanes.device
        self.action_dim, self.fields = lanes.action_dim, lanes.fields
        self.physics_layout, self.physics_dim = lanes.physics_layout, lanes.physics_dim
        self._models = [env.physics.model.ptr for env in lanes.envs]
        self._transform = lanes.envs[0]._transform
        self._nsub, self._limit = lanes.envs[0]._env._n_sub_steps, lanes.envs[0]._env._step_limit
        self._get_state, self._spec = mujoco.mj_getState, mujoco.mjtState.mjSTATE_INTEGRATION
        self._vel, self._error = mujoco.mjtStage.mjSTAGE_VEL, PhysicsError
        self.b = Batch(self._models[0], self.num_envs, nthread=threads)
        self.b.set_models(self._models)
        self.task = task(self.b, self._models)
        self._count, self._reset_next = np.zeros(self.num_envs, np.int64), np.zeros(self.num_envs, bool)
        self._load(np.arange(self.num_envs))

    def _load(self, ids):
        """Copy lanes from dm_control into the batch and forward them as dm_control's reset does."""
        envs = [self.lanes.envs[i] for i in ids]
        self._warnings = self.b.warning[:, :, 1].copy()
        with contextlib.ExitStack() as stack:
            for i, env in zip(ids, envs, strict=True):
                self._get_state(env.physics.model.ptr, env.physics.data.ptr, self.b.state[i], self._spec)
                self._count[i], self._reset_next[i] = env._env._step_count, env._env._reset_next_step
                stack.enter_context(env.physics.model.disable("actuation"))
            self.b.forward(ids)
        columns = export_state(envs, self._cumulative[ids], self.lanes._base_fields, "cpu")
        if len(ids) == self.num_envs:
            self._columns = columns
        else:
            self._columns[torch.from_numpy(ids)] = columns

    def configure_physics(self, values):
        self.lanes.configure_physics(values)
        self.task = self._task(self.b, self._models)

    def reset(self, seed=None):
        self.lanes.reset(seed)
        self._load(np.arange(self.num_envs))
        return self.get_state()

    def get_state(self):
        self.lanes._check_open()
        # Torch copies tensors over 32768 elements on OpenMP threads, which then spin against the batch's threads.
        b, state = self.b, self._columns.apply(lambda column: torch.from_numpy(column.numpy().copy()))
        columns = {
            "qpos": b.qpos,
            "qvel": b.qvel,
            "time": b.time[:, None],
            "warmstart": b.qacc_warmstart,
            ("lifecycle", "step_count"): self._count[:, None],
            ("lifecycle", "reset_next_step"): self._reset_next[:, None],
            ("task", "cumulative_reward"): self._cumulative[:, None],
            **({"activation": b.act} if "activation" in self.fields else {}),
            **({} if self.lanes.physics is None else self.lanes.physics.export()),
        }
        state.update({key: torch.from_numpy(np.array(value)) for key, value in columns.items()})
        return state.to(self.device)

    def validate(self, state):
        return self.lanes.validate(state)

    def set_state(self, state):
        self.lanes.set_state(state)
        self.task = self._task(self.b, self._models)
        self._load(np.arange(self.num_envs))

    def check_step(self, actions, active):
        return self.lanes.check_step(actions, active)

    def step(self, actions, *, active=None, capture_state=True):
        self.lanes._check_open()
        actions, _ = self.lanes.check_step(actions, active)
        ctrl, reward = self._transform(actions), np.zeros(self.num_envs)
        for _ in range(2):
            reset = self._reset_next.copy()
            if reset.any():
                ids = np.flatnonzero(reset)
                for i in ids:
                    self.lanes.envs[i].reset()
                self._load(ids)
            self.b.ctrl[:] = ctrl
            self.b.step(~reset, self._nsub, stage=self._vel)
            if (bad := np.flatnonzero((self.b.warning[:, :, 1] > self._warnings).any(axis=1))).size:
                raise self._error(f"Physics state is invalid in lanes {bad.tolist()}")
            reward += np.where(reset, 0.0, self.task.reward())
            self._count += ~reset
            self._reset_next = self._count >= self._limit
        self._cumulative += reward
        out = {
            **({"state": self.get_state()} if capture_state else {}),
            "reward": torch.from_numpy(reward),
            "terminated": torch.zeros(self.num_envs, dtype=torch.bool),
            "truncated": torch.zeros(self.num_envs, dtype=torch.bool),
            "observation": torch.from_numpy(self.task.observation()),
            "score": torch.from_numpy(self._cumulative / 1000.0),
        }
        if (info := self.task.info()) is not None:
            out["success"] = torch.tensor(info[0], dtype=torch.float64)
            out["info"] = TensorDict({k: torch.from_numpy(np.array(v)) for k, v in info[1].items()}, [self.num_envs])
        return TensorDict(out, [self.num_envs], device=self.device)

    def close(self):
        self.lanes.close()
        self.b = self.task = None
