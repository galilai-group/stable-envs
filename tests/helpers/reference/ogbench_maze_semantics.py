"""Pinned OGBMaze oracle over stable-worldmodel's MazeEnv or the official ogbench registration.

Imports the source itself, never the candidate adapter."""

import numpy as np
import torch
from tensordict import TensorDict

from stable_envs import get_field
from stable_envs._core.state_schema import unfold
from stable_envs.envs.ogbench.simulation.maze import maze_schema


def _uint128_bytes(value: int) -> np.ndarray:
    return np.frombuffer(int(value).to_bytes(16, "little"), dtype=np.uint8).copy()


def _bytes_uint128(value: torch.Tensor) -> int:
    return int.from_bytes(value.numpy().tobytes(), "little")


def _body(model, qpos, qvel):
    """Split qpos/qvel at a leading free joint, as the public State names them."""
    import mujoco

    if int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE):
        return {"root": {"position": qpos, "velocity": qvel}}
    return {
        "root": {"position": qpos[:3], "orientation": qpos[3:7], "velocity": qvel[:6]},
        "joints": {"position": qpos[7:], "velocity": qvel[6:]},
    }


def _qpos_qvel(row):
    def field(name):
        return get_field(row, name).numpy()

    if "orientation" not in unfold(row)["root"].keys():
        return field("root.position"), field("root.velocity")
    qpos = np.concatenate([field("root.position"), field("root.orientation"), field("joints.position")])
    return qpos, np.concatenate([field("root.velocity"), field("joints.velocity")])


class _Official:
    """``gymnasium.make`` of the registered ogbench id, without its TimeLimit and checkers."""

    def __init__(self, loco, maze, width, height):
        import gymnasium
        import ogbench  # noqa: F401

        self.env = gymnasium.make(f"{loco}maze-{maze}-v0", width=width, height=height).unwrapped

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        ob, reward, terminated, truncated, info = self.env.step(action)
        return ob, reward, terminated, truncated, {**info, "env_name": None}

    def set_state(self, qpos, qvel):
        self.env.set_state(qpos, qvel)

    def close(self):
        self.env.close()


class OGBMazeReference:
    def __init__(
        self, num_envs=1, seed=0, *, width=224, height=224, loco="point", maze="large", source="swm", options=None
    ):
        self.num_envs = int(num_envs)
        self.options = options
        self.loco = loco
        if source == "swm":
            from stable_worldmodel.envs.ogbench.maze_env import MazeEnv

            self.envs = [
                MazeEnv(loco_env_type=loco, maze_type=maze, width=width, height=height) for _ in range(self.num_envs)
            ]
            self.env_name = f"{loco.capitalize()}Maze"
        else:
            self.envs = [_Official(loco, maze, width, height) for _ in range(self.num_envs)]
            self.env_name = None
        self._global_rng = [np.random.RandomState(int(seed) + i).get_state() for i in range(self.num_envs)]

    def _in_legacy_rng(self, instance, function, *args, **kwargs):
        outside = np.random.get_state()
        np.random.set_state(self._global_rng[instance])
        try:
            return function(*args, **kwargs)
        finally:
            self._global_rng[instance] = np.random.get_state()
            np.random.set_state(outside)

    def reset(self, seed=None):
        for i, wrapper in enumerate(self.envs):
            if seed is not None:
                self._global_rng[i] = np.random.RandomState(int(seed) + i).get_state()
            self._in_legacy_rng(
                i, wrapper.reset, seed=None if seed is None else int(seed) + i, options=self._instance_options(i)
            )
        return self.get_state()

    def _instance_options(self, instance):
        if self.options is None:
            return None
        values = self.options.get("variation_values", {})
        options = {**self.options, "variation_values": {k: np.asarray(v)[instance] for k, v in values.items()}}
        if isinstance(options.get("task_id"), (list, tuple)):
            options["task_id"] = options["task_id"][instance]
        return options

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
                        **_body(
                            env.model,
                            torch.as_tensor(env.data.qpos.copy(), dtype=torch.float64),
                            torch.as_tensor(env.data.qvel.copy(), dtype=torch.float64),
                        ),
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
        return maze_schema(self.loco).fold(torch.stack(rows), [self.num_envs])

    def set_state(self, state):
        import mujoco

        for i, (wrapper, row) in enumerate(zip(self.envs, unfold(state.cpu()).unbind(0), strict=True)):
            env = wrapper.env

            def field(name, row=row):
                return get_field(row, name)

            env.model.geom("target").pos[:] = field("task.goal_pos").numpy()
            env.cur_goal_xy = field("task.goal_pos")[:2].numpy().copy()
            task_id = field("task.id").item()
            env.cur_task_id = task_id or None
            env.cur_task_info = None if task_id == 0 else env.task_infos[task_id - 1]
            wrapper.set_state(*_qpos_qvel(row))
            env.data.time = field("simulator.time").item()
            mujoco.mj_forward(env.model, env.data)
            env.data.qacc_warmstart[:] = field("simulator.warmstart").numpy()
            env.np_random.bit_generator.state = {
                "bit_generator": "PCG64",
                "state": {
                    "state": _bytes_uint128(field("rng.env_state")),
                    "inc": _bytes_uint128(field("rng.env_inc")),
                },
                "has_uint32": field("rng.env_has_uint32").item(),
                "uinteger": field("rng.env_uinteger").item(),
            }
            self._global_rng[i] = (
                "MT19937",
                field("rng.global_keys").numpy().astype(np.uint32),
                field("rng.global_position").item(),
                int(field("rng.global_has_gauss").item()),
                field("rng.global_cached_gaussian").item(),
            )

    def step(self, actions):
        outputs = [
            self._in_legacy_rng(i, wrapper.step, np.asarray(action))
            for i, (wrapper, action) in enumerate(zip(self.envs, actions, strict=True))
        ]
        infos = [o[4] for o in outputs]
        assert all(info["env_name"] == self.env_name for info in infos)
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
