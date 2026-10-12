"""Frozen f0716916 OGBench row exporters for exact paired export checks.

Only function names/calls are adapted; formulas and dtype conversions are the
accepted pre-column implementation. This is test/benchmark reference code.
"""

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from stable_envs.envs.ogbench.simulation.manipulation import (
    FLAGS,
    MODEL_VALUES,
    pcg_encode,
    rngs,
    site_ids,
    task_values,
)
from stable_envs.envs.ogbench.simulation.maze import OGB_MAZE_FIELDS, _uint128_bytes


def maze_state(self):
    self._check_open()
    rows = []
    for i, env in enumerate(self.envs):
        algorithm, keys, position, has_gauss, cached = self._global_rng[i]
        assert algorithm == "MT19937"
        pcg = env.np_random.bit_generator.state
        assert pcg["bit_generator"] == "PCG64"
        values = {
            ("root", "position"): env.data.qpos.copy(),
            ("root", "velocity"): env.data.qvel.copy(),
            ("simulator", "time"): [env.data.time],
            ("simulator", "warmstart"): env.data.qacc_warmstart.copy(),
            ("task", "goal_pos"): env.model.geom("target").pos.copy(),
            ("task", "id"): [0 if env.cur_task_id is None else env.cur_task_id],
            ("rng", "env_state"): _uint128_bytes(pcg["state"]["state"]),
            ("rng", "env_inc"): _uint128_bytes(pcg["state"]["inc"]),
            ("rng", "env_has_uint32"): [pcg["has_uint32"]],
            ("rng", "env_uinteger"): [pcg["uinteger"]],
            ("rng", "global_keys"): keys.astype(np.int64),
            ("rng", "global_position"): [position],
            ("rng", "global_has_gauss"): [has_gauss],
            ("rng", "global_cached_gaussian"): [cached],
        }
        row = TensorDict({}, batch_size=[], device=self.device)
        for key, (_, dtype) in OGB_MAZE_FIELDS.items():
            row[key] = torch.as_tensor(values[key], dtype=dtype, device=self.device).clone()
        rows.append(row)
    return torch.stack(rows)


def manipulation_row(self, env):
    values = {("compatibility", "signature"): self.signature}
    for name, flag in FLAGS.items():
        size = mujoco.mj_stateSize(env.model, flag)
        if size:
            value = np.empty(size, np.float64)
            mujoco.mj_getState(env.model, env.data, value, flag)
            values["physics", name] = value.astype(bool) if name == "eq_active" else value
    pos, rot = site_ids(env)
    values.update(
        {
            ("controller", "site_xpos"): env.data.site_xpos[pos].copy(),
            ("controller", "site_xmat"): env.data.site_xmat[rot].copy(),
            ("output", "gripper_force"): env.data.body("ur5e/robotiq/right_pad").cfrc_ext.copy(),
            ("output", "prev_qpos"): env._prev_qpos.copy(),
            ("output", "prev_qvel"): env._prev_qvel.copy(),
            ("task", "goal_observation"): env._cur_goal_ob.copy(),
            ("task", "id"): np.array([-1 if env.cur_task_id is None else env.cur_task_id], np.int64),
            ("task", "reward_id"): np.array([-1 if env._reward_task_id is None else env._reward_task_id], np.int64),
            ("task", "success"): np.array([env._success], bool),
            ("lifecycle", "reset_next_step"): np.array([env._reset_next_step], bool),
        }
    )
    for key, value in task_values(env).items():
        values["task", "payload", key] = value
    if self.family == "Scene":
        for key in ("cur_button_states", "prev_button_states", "target_button_states"):
            values["scene", key] = np.asarray(getattr(env, "_" + key), np.int64).copy()
        for key in ("target_drawer_pos", "target_window_pos"):
            values["scene", key] = np.array([getattr(env, "_" + key)], np.float64)
    for name in sorted(env.supported_variations):
        node = env.variation_space._walk(name)
        values["variation", "current", name] = np.asarray(node.value, np.float64).copy()
        values["variation", "anchor", name] = np.asarray(node.init_value, np.float64).copy()
    for namespace, source in [("compiled", env._model_values), ("mjcf", env._mjcf_values)]:
        for key in MODEL_VALUES:
            values["model", namespace, key] = source[key].copy()
    for name, rng in rngs(env).items():
        for key, value in pcg_encode(rng).items():
            values["rng", name, key] = value
    _, key, position, flag, cached = env.task_rng.get_state()
    values.update(
        {
            ("rng", "task", "key"): key.astype(np.int64),
            ("rng", "task", "position"): np.array([position], np.int64),
            ("rng", "task", "has_gauss"): np.array([flag], np.int64),
            ("rng", "task", "cached_gaussian"): np.array([cached], np.float64),
        }
    )
    return TensorDict(
        {k: torch.as_tensor(v.copy(), device=self.device) for k, v in values.items()},
        batch_size=[],
        device=self.device,
    )


def manipulation_state(self):
    self._check_open()
    return torch.stack([manipulation_row(self, env) for env in self.envs])
