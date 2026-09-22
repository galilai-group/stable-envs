"""Pinned historical OGBMaze oracle. Imports SWM itself, never the candidate adapter."""

import numpy as np
import torch
from tensordict import TensorDict


def _uint128_bytes(value: int) -> np.ndarray:
    return np.frombuffer(int(value).to_bytes(16, "little"), dtype=np.uint8).copy()


def _bytes_uint128(value: torch.Tensor) -> int:
    return int.from_bytes(value.numpy().tobytes(), "little")


class OGBMazeReference:
    def __init__(self, num_envs=1, seed=0):
        from stable_worldmodel.envs.ogbench.maze_env import MazeEnv

        self.num_envs = int(num_envs)
        self.envs = [MazeEnv() for _ in range(self.num_envs)]
        self._global_rng = [np.random.RandomState(int(seed) + i).get_state() for i in range(self.num_envs)]

    def _in_legacy_rng(self, lane, function, *args, **kwargs):
        outside = np.random.get_state()
        np.random.set_state(self._global_rng[lane])
        try:
            return function(*args, **kwargs)
        finally:
            self._global_rng[lane] = np.random.get_state()
            np.random.set_state(outside)

    def reset(self, seed=None):
        for i, wrapper in enumerate(self.envs):
            if seed is not None:
                self._global_rng[i] = np.random.RandomState(int(seed) + i).get_state()
            self._in_legacy_rng(i, wrapper.reset, seed=None if seed is None else int(seed) + i)
        return self.get_state()

    def get_state(self):
        rows = []
        for i, wrapper in enumerate(self.envs):
            env = wrapper.env
            algorithm, keys, position, has_gauss, cached = self._global_rng[i]
            assert algorithm == "MT19937"
            pcg = env.np_random.bit_generator.state
            assert pcg["bit_generator"] == "PCG64"
            rows.append(
                TensorDict(
                    {
                        "root": {
                            "position": torch.as_tensor(env.data.qpos.copy(), dtype=torch.float64),
                            "velocity": torch.as_tensor(env.data.qvel.copy(), dtype=torch.float64),
                        },
                        "simulator": {
                            "time": torch.tensor([env.data.time], dtype=torch.float64),
                            "warmstart": torch.as_tensor(env.data.qacc_warmstart.copy(), dtype=torch.float64),
                        },
                        "task": {
                            "goal_pos": torch.as_tensor(env.model.geom("target").pos.copy(), dtype=torch.float64),
                            "id": torch.tensor([0 if env.cur_task_id is None else env.cur_task_id], dtype=torch.int64),
                        },
                        "rng": {
                            "env_state": torch.as_tensor(_uint128_bytes(pcg["state"]["state"])),
                            "env_inc": torch.as_tensor(_uint128_bytes(pcg["state"]["inc"])),
                            "env_has_uint32": torch.tensor([pcg["has_uint32"]], dtype=torch.int64),
                            "env_uinteger": torch.tensor([pcg["uinteger"]], dtype=torch.int64),
                            "global_keys": torch.as_tensor(keys.astype(np.int64)),
                            "global_position": torch.tensor([position], dtype=torch.int64),
                            "global_has_gauss": torch.tensor([has_gauss], dtype=torch.bool),
                            "global_cached_gaussian": torch.tensor([cached], dtype=torch.float64),
                        },
                    },
                    batch_size=[],
                )
            )
        return torch.stack(rows)

    def set_state(self, state):
        import mujoco

        for i, (wrapper, row) in enumerate(zip(self.envs, state.cpu().unbind(0), strict=True)):
            env = wrapper.env
            env.model.geom("target").pos[:] = row["task", "goal_pos"].numpy()
            env.cur_goal_xy = row["task", "goal_pos"][:2].numpy().copy()
            task_id = row["task", "id"].item()
            env.cur_task_id = task_id or None
            env.cur_task_info = None if task_id == 0 else env.task_infos[task_id - 1]
            wrapper.set_state(row["root", "position"].numpy(), row["root", "velocity"].numpy())
            env.data.time = row["simulator", "time"].item()
            mujoco.mj_forward(env.model, env.data)
            env.data.qacc_warmstart[:] = row["simulator", "warmstart"].numpy()
            env.np_random.bit_generator.state = {
                "bit_generator": "PCG64",
                "state": {
                    "state": _bytes_uint128(row["rng", "env_state"]),
                    "inc": _bytes_uint128(row["rng", "env_inc"]),
                },
                "has_uint32": row["rng", "env_has_uint32"].item(),
                "uinteger": row["rng", "env_uinteger"].item(),
            }
            self._global_rng[i] = (
                "MT19937",
                row["rng", "global_keys"].numpy().astype(np.uint32),
                row["rng", "global_position"].item(),
                int(row["rng", "global_has_gauss"].item()),
                row["rng", "global_cached_gaussian"].item(),
            )

    def step(self, actions):
        outputs = [
            self._in_legacy_rng(i, wrapper.step, np.asarray(action))
            for i, (wrapper, action) in enumerate(zip(self.envs, actions, strict=True))
        ]
        infos = [o[4] for o in outputs]
        assert all(info["env_name"] == "PointMaze" for info in infos)
        return TensorDict(
            {
                "state": self.get_state(),
                "reward": torch.tensor([o[1] for o in outputs], dtype=torch.float64),
                "terminated": torch.tensor([o[2] for o in outputs], dtype=torch.bool),
                "truncated": torch.tensor([o[3] for o in outputs], dtype=torch.bool),
                "observation": torch.as_tensor(np.stack([o[0] for o in outputs])),
                "success": torch.tensor([info["success"] for info in infos], dtype=torch.float64),
                "info": TensorDict(
                    {
                        key: torch.as_tensor(np.stack([info[key] for info in infos]))
                        for key in ("xy", "prev_qpos", "prev_qvel", "qpos", "qvel")
                    },
                    batch_size=[self.num_envs],
                ),
            },
            batch_size=[self.num_envs],
        )

    def close(self):
        for env in self.envs:
            env.close()
