from __future__ import annotations

import logging

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.process import PER_LANE
from ...._execution.protocol import SimulatorAdapter
from ...._execution.state_columns import require_columns
from . import physics as _physics
from .batch import TASKS, LaneViews
from .state import RNGColumns, export_state, model_entries, restore_lane

_INVALID_PHYSICS_STATE = "Physics state is invalid. Warning(s) raised: {warning_names}"


class DMCLanes(SimulatorAdapter):
    """Independent CPU dm_control lanes; each action advances the source twice.

    A step makes the MuJoCo calls of the source's step on every lane, then computes the task's
    observations, rewards and outputs for all lanes at once. A lane whose episode ended runs the
    source's own reset. ``device`` controls returned tensor placement. State holds physics,
    lifecycle, task RNG, episode-drawn model entries and the lane's physics Variation.
    """

    process_methods = {"configure_physics": PER_LANE}
    process_attributes = ("key", "action_dim", "fields", "physics_layout", "physics_dim")

    def __init__(self, key, num_envs, device, seed, make, fields):
        import mujoco

        self.key = key
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.envs = [make(int(seed) + i) for i in range(self.num_envs)]
        self.physics = _physics.make_lanes(key, [env.physics.model.ptr for env in self.envs])
        self.fields = {**fields, **_physics.state_fields(key)}
        self._base_fields = fields
        self.action_dim = int(self.envs[0].action_spec().shape[0])
        self._bases = [env._env for env in self.envs]
        self._views = LaneViews(
            [env.physics.model.ptr for env in self.envs], [env.physics.data.ptr for env in self.envs]
        )
        self._warnings = [d.warning.number for d in self._views.datas]
        self._task = TASKS[key](self._bases[0])
        self._model_entries = model_entries(self._views.models[0], fields)
        self._rk4 = self._views.models[0].opt.integrator == mujoco.mjtIntegrator.mjINT_RK4
        self._substeps = int(self._bases[0]._n_sub_steps)
        self._rng = RNGColumns(self.num_envs)
        self._cumulative = np.zeros(self.num_envs, np.float64)
        self._closed = False
        self.reset(seed=seed)

    def _check_open(self):
        if self._closed:
            raise RuntimeError(f"{type(self).__name__} is closed")

    @property
    def physics_layout(self):
        return None if self.physics is None else self.physics.layout

    @property
    def physics_dim(self):
        return 0 if self.physics is None else self.physics.spec.dim

    def configure_physics(self, values):
        self._check_open()
        if self.physics is None:
            raise ValueError(f"{self.key} declares no physics Variation")
        self.physics.configure(values)
        self._rng.stale[:] = True

    def reset(self, seed=None):
        self._check_open()
        for i, env in enumerate(self.envs):
            if seed is not None:
                env.task._random = np.random.RandomState(int(seed) + i)
            env.reset()
        self._rng.stale[:] = True
        self._cumulative.fill(0)
        return self.get_state()

    def get_state(self):
        self._check_open()
        extra = None if self.physics is None else self.physics.export()
        return export_state(self, self.fields, self.device, extra)

    def validate(self, state):
        """Check every lane of ``state`` and return its physics rows, or None without physics Variation."""
        self._check_open()
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError(f"{self.key} requires full TensorDict State with batch [num_envs]")
        rows = None if self.physics is None else self.physics.rows_from_state(state)
        require_columns(state, {key: ((self.num_envs, width), dtype) for key, (width, dtype) in self.fields.items()})
        return rows

    def set_state(self, state):
        rows = self.validate(state)
        host = state.cpu()
        if rows is not None:
            self.physics.configure(rows)
        for i, env in enumerate(self.envs):
            restore_lane(env, host[i], self._base_fields, self._cumulative, i)
        self._rng.stale[:] = True

    def check_step(self, actions, active):
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, np.float64)
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"DMC actions must be finite with shape {(self.num_envs, self.action_dim)}")
        if active is not None:
            active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or not active.all():
                raise ValueError(
                    "Accepted DMC wrappers do not expose terminal lanes; partial active masks are unsupported"
                )
        return actions, None

    def _advance(self, actions, control):
        """Advance every lane by one source step and return the mask of lanes that stepped physics.

        A lane that is due to reset runs the source's ``step``, which resets it. The other lanes
        make the MuJoCo calls of dm_control's legacy ``Physics.step`` and count the step toward
        the time limit.
        """
        import mujoco

        step, step1, step2 = mujoco.mj_step, mujoco.mj_step1, mujoco.mj_step2
        views, rk4, substeps = self._views, self._rk4, self._substeps
        before = np.stack(self._warnings)
        stepped = np.ones(self.num_envs, bool)
        for i, (base, m, d, ctrl) in enumerate(zip(self._bases, views.models, views.datas, views.data("ctrl"))):
            if base._reset_next_step:
                self.envs[i].step(actions[i])
                self._rng.stale[i] = True
                stepped[i] = False
                continue
            np.copyto(ctrl, control[i])
            if rk4:
                step(m, d, substeps)
            else:
                step2(m, d)
                if substeps > 1:
                    step(m, d, substeps - 1)
            step1(m, d)
            base._step_count += 1
            if base._step_count >= base._step_limit:
                base._reset_next_step = True
        raised = (np.stack(self._warnings) > before) & stepped[:, None]
        for lane in np.flatnonzero(raised.any(axis=1)):
            names = ", ".join(np.compress(raised[lane], list(mujoco.mjtWarning.__members__)))
            message = _INVALID_PHYSICS_STATE.format(warning_names=names)
            if self._bases[lane].physics._warnings_cause_exception:
                from dm_control.rl.control import PhysicsError

                raise PhysicsError(message)
            logging.warning(message)
        return stepped

    def step(self, actions, *, active=None, capture_state=True):
        self._check_open()
        if active is not None and not bool(torch.as_tensor(active).all()):
            raise ValueError(
                "Accepted DMC wrappers do not expose terminal lanes; partial active masks are unsupported"
            )
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions, np.float64)
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"{self.key} action must have finite shape (num_envs, {self.action_dim})")
        control = self.envs[0]._transform(actions)
        rewards = np.zeros(self.num_envs, np.float64)
        for _ in range(2):
            stepped = self._advance(actions, control)
            snapshot = self._views.snapshot()
            rewards += np.where(stepped, self._task.reward(snapshot), 0.0)
        self._cumulative += rewards
        observation = np.concatenate(
            [np.asarray(part).reshape(self.num_envs, -1) for part in self._task.observation(snapshot)],
            axis=1,
            dtype=np.float32,
        )
        out = {
            **({"state": self.get_state()} if capture_state else {}),
            "reward": torch.from_numpy(rewards).to(self.device),
            "terminated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "truncated": torch.zeros(self.num_envs, dtype=torch.bool, device=self.device),
            "observation": torch.from_numpy(observation).to(self.device),
        }
        score = torch.tensor(self._cumulative / 1000.0, dtype=torch.float64, device=self.device)
        info = self._task.info(snapshot)
        if info is None:
            out["score"] = score
        else:
            success, values = info
            out["success"] = torch.as_tensor(np.asarray(success, np.float64), device=self.device)
            out["score"] = score
            out["info"] = TensorDict(
                {key: torch.as_tensor(np.array(value), device=self.device) for key, value in values.items()},
                [self.num_envs],
                device=self.device,
            )
        return TensorDict(out, [self.num_envs], device=self.device)

    def close(self):
        if not self._closed:
            for env in self.envs:
                env.close()
            self._closed = True
