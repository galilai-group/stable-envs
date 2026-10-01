"""Private parameterized Fetch semantics; observation packaging belongs to SWM.

The source-proof report establishes warmstart and the gripper controller cache.
No SWM code or opaque MuJoCo state serialization is used at runtime.
"""

from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
from tensordict import TensorDict

FIELDS = {
    "qpos": (22, torch.float64),
    "qvel": (21, torch.float64),
    "time": (1, torch.float64),
    ("task", "initial_gripper_xpos"): (3, torch.float64),
    "warmstart": (21, torch.float64),
    ("controller", "position"): (3, torch.float64),
    ("controller", "quaternion"): (4, torch.float64),
    "goal": (3, torch.float64),
    ("lifecycle", "elapsed_steps"): (1, torch.int64),
    ("rng", "state_words"): (4, torch.int64),
    ("rng", "increment_words"): (4, torch.int64),
    ("rng", "has_uint32"): (1, torch.int64),
    ("rng", "uinteger"): (1, torch.int64),
}


def _words(integer):
    return [(int(integer) >> (32 * i)) & 0xFFFFFFFF for i in range(4)]


def _integer(words):
    return sum(int(word) << (32 * i) for i, word in enumerate(words))


def state_fields(nq, nv):
    fields = dict(FIELDS)
    fields.update(qpos=(nq, torch.float64), qvel=(nv, torch.float64), warmstart=(nv, torch.float64))
    return fields


class FetchAdapter:
    execution_mode = "DELEGATED_CPU"

    def __init__(self, num_envs, device="cpu", seed=0, task="PickAndPlace", reward_type="sparse"):
        import gymnasium as gym
        import mujoco
        from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv
        from gymnasium_robotics.envs.fetch.push import MujocoFetchPushEnv
        from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv
        from gymnasium_robotics.envs.fetch.slide import MujocoFetchSlideEnv

        cls = {
            "PickAndPlace": MujocoFetchPickAndPlaceEnv,
            "Reach": MujocoFetchReachEnv,
            "Push": MujocoFetchPushEnv,
            "Slide": MujocoFetchSlideEnv,
        }[task]

        self.num_envs, self.device = int(num_envs), torch.device(device)
        self._mujoco = mujoco
        self.envs = [gym.wrappers.TimeLimit(cls(reward_type=reward_type), 50) for _ in range(self.num_envs)]
        model = self.envs[0].unwrapped.model
        self.fields = state_fields(model.nq, model.nv)
        self.has_object = self.envs[0].unwrapped.has_object
        self._gripper = [
            mujoco.mj_name2id(e.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "robot0:gripper_link") for e in self.envs
        ]
        self._object = [mujoco.mj_name2id(e.unwrapped.model, mujoco.mjtObj.mjOBJ_BODY, "object0") for e in self.envs]
        self._mass = np.array(
            [e.unwrapped.model.body_mass[i] if i >= 0 else 0.0 for e, i in zip(self.envs, self._object)]
        )
        self._closed = False
        self._executor = None
        self._workers = 1
        self.reset(seed=seed)
        # The bounded 64/256-lane sweep found four workers useful and little
        # benefit from eight. Respect scheduler affinity; leave cheap batches
        # serial. Strategy remains private rather than another user API knob.
        available = len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else (os.cpu_count() or 1)
        if self.num_envs >= 64 and available > 1:
            self._set_workers(min(4, available))

    def _set_workers(self, workers):
        """Private measured executor choice; each task owns disjoint simulator lanes."""
        self._check_open()
        if type(workers) is not int or not 1 <= workers <= self.num_envs:
            raise ValueError("workers must be an integer between 1 and num_envs")
        if self._executor is not None:
            self._executor.shutdown(wait=True)
        self._workers = workers
        self._executor = ThreadPoolExecutor(max_workers=workers) if workers > 1 else None

    def _advance_shard(self, lanes, actions):
        return [(lane, self.envs[lane].step(actions[lane])) for lane in lanes]

    def _check_open(self):
        if self._closed:
            raise RuntimeError("FetchAdapter is closed")

    def configure_object_mass(self, values):
        self._mass = np.asarray(values, np.float64).reshape(self.num_envs).copy()
        # Historical Variation values are applied after the next upstream reset.

    def reset(self, seed=None, variations=None):
        self._check_open()
        for lane, e in enumerate(self.envs):
            values = {} if variations is None else variations[lane]
            u = e.unwrapped
            if "agent.start_position" in values:
                u.initial_gripper_xpos[:2] = values["agent.start_position"]
            e.reset(seed=None if seed is None else int(seed) + lane)
            changed = False
            if "block.start_position" in values:
                u.data.qpos[15:17] = values["block.start_position"]
                changed = True
            if "block.angle" in values:
                angle = float(np.asarray(values["block.angle"]).reshape(-1)[0])
                u.data.qpos[18:22] = [np.cos(angle / 2), 0, 0, np.sin(angle / 2)]
                changed = True
            if "goal.start_position" in values:
                u.goal = np.asarray(values["goal.start_position"], np.float64).copy()
                changed = True
            if "block.mass" in values:
                self._mass[lane] = float(np.asarray(values["block.mass"]).reshape(-1)[0])
            # Historical SWM applies the current mass after upstream reset.
            if self.has_object:
                e.unwrapped.model.body_mass[self._object[lane]] = self._mass[lane]
            if changed:
                self._mujoco.mj_forward(u.model, u.data)
        return self.get_state()

    def get_state(self):
        self._check_open()
        # Owned column arrays: one tensor conversion/transfer per field, not per lane.
        columns = {
            key: np.empty((self.num_envs, width), dtype=(np.float64 if dtype == torch.float64 else np.int64))
            for key, (width, dtype) in self.fields.items()
        }
        for lane, (e, i) in enumerate(zip(self.envs, self._gripper)):
            u = e.unwrapped
            r = u.np_random.bit_generator.state
            if r["bit_generator"] != "PCG64":
                raise ValueError("Fetch requires PCG64 RNG")
            values = {
                "qpos": u.data.qpos.copy(),
                "qvel": u.data.qvel.copy(),
                "time": [u.data.time],
                ("task", "initial_gripper_xpos"): u.initial_gripper_xpos.copy(),
                "warmstart": u.data.qacc_warmstart.copy(),
                ("controller", "position"): u.data.xpos[i].copy(),
                ("controller", "quaternion"): u.data.xquat[i].copy(),
                "goal": u.goal.copy(),
                ("lifecycle", "elapsed_steps"): [e._elapsed_steps],
                ("rng", "state_words"): _words(r["state"]["state"]),
                ("rng", "increment_words"): _words(r["state"]["inc"]),
                ("rng", "has_uint32"): [r["has_uint32"]],
                ("rng", "uinteger"): [r["uinteger"]],
            }
            for key in self.fields:
                columns[key][lane] = values[key]
        return TensorDict(
            {
                key: torch.as_tensor(value, dtype=self.fields[key][1], device=self.device)
                for key, value in columns.items()
            },
            [self.num_envs],
            device=self.device,
        )

    def set_state(self, state):
        self._check_open()
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError("Fetch requires full TensorDict State with batch [num_envs]")
        keys = state.keys(True, True)
        for key, (width, dtype) in self.fields.items():
            if key not in keys:
                raise ValueError(f"Full State missing {key}")
            value = state[key]
            if value.shape != (self.num_envs, width) or value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid State field {key}")
        for key in (("rng", "state_words"), ("rng", "increment_words"), ("rng", "uinteger")):
            if ((state[key] < 0) | (state[key] > 0xFFFFFFFF)).any():
                raise ValueError(f"Invalid PCG64 field {key}")
        if ((state["rng", "has_uint32"] < 0) | (state["rng", "has_uint32"] > 1)).any():
            raise ValueError("Invalid PCG64 has_uint32")
        if (state["lifecycle", "elapsed_steps"] < 0).any():
            raise ValueError("Negative elapsed_steps")
        host = state.cpu()
        for lane, (e, i) in enumerate(zip(self.envs, self._gripper)):
            row, u = host[lane], e.unwrapped
            u.data.qpos[:] = row["qpos"].numpy()
            u.data.qvel[:] = row["qvel"].numpy()
            u.goal = row["goal"].numpy().copy()
            u.data.time = row["time"].item()
            u.initial_gripper_xpos[:] = row["task", "initial_gripper_xpos"].numpy()
            # Controller overwrites mocap and ctrl before every public step.
            # Applied-force arrays remain zero under this registration's API.
            u.data.qfrc_applied[:] = 0
            u.data.xfrc_applied[:] = 0
            self._mujoco.mj_forward(u.model, u.data)
            u.data.qacc_warmstart[:] = row["warmstart"].numpy()
            u.data.xpos[i] = row["controller", "position"].numpy()
            u.data.xquat[i] = row["controller", "quaternion"].numpy()
            e._elapsed_steps = row["lifecycle", "elapsed_steps"].item()
            u.np_random.bit_generator.state = {
                "bit_generator": "PCG64",
                "state": {
                    "state": _integer(row["rng", "state_words"].tolist()),
                    "inc": _integer(row["rng", "increment_words"].tolist()),
                },
                "has_uint32": row["rng", "has_uint32"].item(),
                "uinteger": row["rng", "uinteger"].item(),
            }

    def step(self, actions, *, active=None):
        out = self.advance(actions, active=active)
        out["state"] = self.get_state()
        return out

    def advance(self, actions, *, active=None):
        """Advance task/controller/physics without materializing a full snapshot."""
        active = None if active is None else np.asarray(active.detach().cpu(), dtype=bool)
        self._check_open()
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, 4):
            raise ValueError("Fetch action must have shape (num_envs,4)")
        # Preserve public dtype: upstream multiplication before concatenation is observable.
        lanes = np.arange(self.num_envs) if active is None else np.flatnonzero(active)
        if self._executor is None:
            updates = self._advance_shard(lanes, actions)
        else:
            futures = [
                self._executor.submit(self._advance_shard, shard, actions)
                for shard in np.array_split(lanes, self._workers)
                if len(shard)
            ]
            updates = []
            # Drain all workers even on failure, so no simulator keeps mutating after return.
            errors = []
            for future in futures:
                try:
                    updates.extend(future.result())
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]
        outputs = list(self._last_outputs) if active is not None else [None] * self.num_envs
        for lane, output in updates:
            outputs[lane] = output
        self._last_outputs = outputs
        return TensorDict(
            {
                "reward": torch.as_tensor(np.array([x[1] for x in outputs]), device=self.device),
                "terminated": torch.as_tensor(np.array([x[2] for x in outputs]), device=self.device),
                "truncated": torch.as_tensor(np.array([x[3] for x in outputs]), device=self.device),
                "success": torch.as_tensor(np.array([x[4]["is_success"] for x in outputs]), device=self.device),
            },
            [self.num_envs],
            device=self.device,
        )

    def close(self):
        if not self._closed:
            if self._executor is not None:
                self._executor.shutdown(wait=True)
                self._executor = None
            for e in self.envs:
                e.close()
            self._closed = True


# Compatibility name for the accepted representative private probe.
FetchPickAndPlaceAdapter = FetchAdapter
