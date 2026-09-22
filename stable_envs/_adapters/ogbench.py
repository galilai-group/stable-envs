"""Authoritative OGBench point-maze stepping; no SWM runtime imports."""

from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

OGB_MAZE_FIELDS = {
    ("root", "position"): (2, torch.float64),
    ("root", "velocity"): (2, torch.float64),
    ("simulator", "time"): (1, torch.float64),
    ("simulator", "warmstart"): (2, torch.float64),
    ("task", "goal_pos"): (3, torch.float64),
    ("task", "id"): (1, torch.int64),
    ("rng", "env_state"): (16, torch.uint8),
    ("rng", "env_inc"): (16, torch.uint8),
    ("rng", "env_has_uint32"): (1, torch.int64),
    ("rng", "env_uinteger"): (1, torch.int64),
    ("rng", "global_keys"): (624, torch.int64),
    ("rng", "global_position"): (1, torch.int64),
    ("rng", "global_has_gauss"): (1, torch.bool),
    ("rng", "global_cached_gaussian"): (1, torch.float64),
}


def _uint128_bytes(value: int) -> np.ndarray:
    return np.frombuffer(int(value).to_bytes(16, "little"), dtype=np.uint8).copy()


def _bytes_uint128(value: torch.Tensor) -> int:
    return int.from_bytes(value.numpy().tobytes(), "little")


def _make_point_large_maze():
    """Construct the upstream environment selected by the historical empty kwargs."""
    from ogbench.locomaze.maze import make_maze_env

    return make_maze_env(
        loco_env_type="point",
        maze_env_type="maze",
        maze_type="large",
        ob_type="states",
        width=224,
        height=224,
    )


class OGBMazeAdapter:
    """B independent upstream CPU simulators for the historical point/large default."""

    def __init__(self, num_envs, device="cpu", seed=0):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.envs = [_make_point_large_maze() for _ in range(self.num_envs)]
        self._global_rng = [np.random.RandomState(int(seed) + i).get_state() for i in range(self.num_envs)]
        self._closed = False
        self.reset(seed=seed)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("OGBMazeAdapter is closed")

    def reset(self, seed=None):
        self._check_open()
        for i, env in enumerate(self.envs):
            if seed is not None:
                self._global_rng[i] = np.random.RandomState(int(seed) + i).get_state()
            self._in_legacy_rng(i, env.reset, seed=None if seed is None else int(seed) + i)
        return self.get_state()

    def _in_legacy_rng(self, lane, function, *args, **kwargs):
        """Give one lane OGBench's process-global RNG without leaking it across lanes/callers."""
        outside = np.random.get_state()
        np.random.set_state(self._global_rng[lane])
        try:
            return function(*args, **kwargs)
        finally:
            self._global_rng[lane] = np.random.get_state()
            np.random.set_state(outside)

    def get_state(self):
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

    def set_state(self, state):
        self._check_open()
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError("OGBMaze requires full TensorDict State with batch [num_envs]")
        for key, (width, dtype) in OGB_MAZE_FIELDS.items():
            if key not in state.keys(True, True):
                raise ValueError(f"Full State missing {key}")
            value = state[key]
            if value.shape != (self.num_envs, width) or value.dtype != dtype or not torch.isfinite(value).all():
                raise ValueError(f"Invalid State field {key}")
        host = state.cpu()
        decoded = []
        for row in host.unbind(0):
            task_id = row["task", "id"].item()
            if not 0 <= task_id <= self.envs[0].num_tasks:
                raise ValueError(f"Invalid OGBench task id {task_id}")
            pcg = {
                "bit_generator": "PCG64",
                "state": {
                    "state": _bytes_uint128(row["rng", "env_state"]),
                    "inc": _bytes_uint128(row["rng", "env_inc"]),
                },
                "has_uint32": row["rng", "env_has_uint32"].item(),
                "uinteger": row["rng", "env_uinteger"].item(),
            }
            np.random.PCG64().state = pcg
            legacy = (
                "MT19937",
                row["rng", "global_keys"].numpy().astype(np.uint32),
                row["rng", "global_position"].item(),
                int(row["rng", "global_has_gauss"].item()),
                row["rng", "global_cached_gaussian"].item(),
            )
            np.random.RandomState().set_state(legacy)
            decoded.append((task_id, pcg, legacy))
        import mujoco

        for i, (env, row) in enumerate(zip(self.envs, host.unbind(0), strict=True)):
            task_id, pcg, legacy = decoded[i]
            env.model.geom("target").pos[:] = row["task", "goal_pos"].numpy()
            env.cur_goal_xy = row["task", "goal_pos"][:2].numpy().copy()
            env.cur_task_id = task_id or None
            env.cur_task_info = None if task_id == 0 else env.task_infos[task_id - 1]
            env.data.qpos[:] = row["root", "position"].numpy()
            env.data.qvel[:] = row["root", "velocity"].numpy()
            env.data.time = row["simulator", "time"].item()
            mujoco.mj_forward(env.model, env.data)
            env.data.qacc_warmstart[:] = row["simulator", "warmstart"].numpy()
            env.np_random.bit_generator.state = pcg
            self._global_rng[i] = legacy

    def step(self, actions):
        self._check_open()
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, 2) or not np.isfinite(actions).all():
            raise ValueError("OGBMaze action must have finite shape (num_envs, 2)")
        outputs = [
            self._in_legacy_rng(i, env.step, action)
            for i, (env, action) in enumerate(zip(self.envs, actions, strict=True))
        ]
        infos = [o[4] for o in outputs]
        return TensorDict(
            {
                "state": self.get_state(),
                "reward": torch.tensor([o[1] for o in outputs], dtype=torch.float64, device=self.device),
                "terminated": torch.tensor([o[2] for o in outputs], dtype=torch.bool, device=self.device),
                "truncated": torch.tensor([o[3] for o in outputs], dtype=torch.bool, device=self.device),
                "observation": torch.as_tensor(np.stack([o[0] for o in outputs]), device=self.device),
                "success": torch.tensor([info["success"] for info in infos], dtype=torch.float64, device=self.device),
                "info": TensorDict(
                    {
                        key: torch.as_tensor(np.stack([info[key] for info in infos]), device=self.device)
                        for key in ("xy", "prev_qpos", "prev_qvel", "qpos", "qvel")
                    },
                    batch_size=[self.num_envs],
                    device=self.device,
                ),
            },
            batch_size=[self.num_envs],
            device=self.device,
        )

    def close(self):
        if not self._closed:
            for env in self.envs:
                env.close()
            self._closed = True
