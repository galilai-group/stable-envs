from __future__ import annotations

import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.protocol import SimulatorAdapter
from ...._execution.state_columns import owned_columns, require_columns
from .pcg import bytes_uint128, pcg_state
from .pcg import uint128_bytes as _uint128_bytes

#: body -> (qpos fields, qvel fields, action width). Legged qpos splits at the free joint.
BODIES = {
    "point": (((("root", "position"), 2),), ((("root", "velocity"), 2),), 2),
    "ant": (
        ((("root", "position"), 3), (("root", "orientation"), 4), (("joints", "position"), 8)),
        ((("root", "velocity"), 6), (("joints", "velocity"), 8)),
        8,
    ),
    "humanoid": (
        ((("root", "position"), 3), (("root", "orientation"), 4), (("joints", "position"), 21)),
        ((("root", "velocity"), 6), (("joints", "velocity"), 21)),
        21,
    ),
}


def maze_fields(loco: str) -> dict:
    qpos, qvel, _ = BODIES[loco]
    nv = sum(width for _, width in qvel)
    return {
        **{key: (width, torch.float64) for key, width in qpos + qvel},
        ("simulator", "time"): (1, torch.float64),
        ("simulator", "warmstart"): (nv, torch.float64),
        **_TASK_RNG_FIELDS,
    }


_TASK_RNG_FIELDS = {
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
OGB_MAZE_FIELDS = maze_fields("point")


_NUMPY_DTYPES = {torch.float64: np.float64, torch.int64: np.int64, torch.bool: np.bool_, torch.uint8: np.uint8}


def _split(fields, values):
    out, start = {}, 0
    for key, width in fields:
        out[key] = values[start : start + width].copy()
        start += width
    return out


def _join(fields, row):
    return np.concatenate([row[key].numpy() for key, _ in fields])


def _make_maze(loco, maze):
    from ogbench.locomaze.maze import make_maze_env

    return make_maze_env(
        loco_env_type=loco,
        maze_env_type="maze",
        maze_type=maze,
        ob_type="states",
        width=224,
        height=224,
    )


class OGBMazeAdapter(SimulatorAdapter):
    """Independent CPU locomotion-maze simulators with owned State snapshots.

    ``device`` selects output placement. Each lane owns both its environment PCG64
    and its own copy of NumPy's legacy global RNG (``np.random`` MT19937), which
    upstream draws from; calls must be serialized.
    """

    def __init__(self, num_envs, device="cpu", seed=0, *, loco="point", maze="large"):
        self.num_envs = int(num_envs)
        self.device = torch.device(device)
        self.loco, self.maze = loco, maze
        self.fields = maze_fields(loco)
        self._qpos, self._qvel, self.action_dim = BODIES[loco]
        self.envs = [_make_maze(loco, maze) for _ in range(self.num_envs)]
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
        """Install a lane's NumPy legacy global RNG state for one call, restoring the caller's on exit."""
        outside = np.random.get_state()
        np.random.set_state(self._global_rng[lane])
        try:
            return function(*args, **kwargs)
        finally:
            self._global_rng[lane] = np.random.get_state()
            np.random.set_state(outside)

    def _values(self, lane, env):
        algorithm, keys, position, has_gauss, cached = self._global_rng[lane]
        assert algorithm == "MT19937"
        pcg = env.np_random.bit_generator.state
        assert pcg["bit_generator"] == "PCG64"
        values = {
            **_split(self._qpos, env.data.qpos),
            **_split(self._qvel, env.data.qvel),
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
        return {key: np.asarray(value, dtype=_NUMPY_DTYPES[self.fields[key][1]]) for key, value in values.items()}

    def get_state(self):
        self._check_open()
        return owned_columns((self._values(i, env) for i, env in enumerate(self.envs)), self.num_envs, self.device)

    def validate(self, state):
        self._check_open()
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([self.num_envs]):
            raise TypeError("OGBMaze requires full TensorDict State with batch [num_envs]")
        require_columns(state, {key: ((self.num_envs, width), dtype) for key, (width, dtype) in self.fields.items()})
        host = state.cpu()
        decoded = []
        for row in host.unbind(0):
            task_id = row["task", "id"].item()
            if not 0 <= task_id <= self.envs[0].num_tasks:
                raise ValueError(f"Invalid OGBench task id {task_id}")
            pcg = pcg_state(
                bytes_uint128(row["rng", "env_state"].numpy()),
                bytes_uint128(row["rng", "env_inc"].numpy()),
                row["rng", "env_has_uint32"].item(),
                row["rng", "env_uinteger"].item(),
            )
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
        return host, decoded

    def set_state(self, state):
        host, decoded = self.validate(state)
        import mujoco

        for i, (env, row) in enumerate(zip(self.envs, host.unbind(0), strict=True)):
            task_id, pcg, legacy = decoded[i]
            env.model.geom("target").pos[:] = row["task", "goal_pos"].numpy()
            env.cur_goal_xy = row["task", "goal_pos"][:2].numpy().copy()
            env.cur_task_id = task_id or None
            env.cur_task_info = None if task_id == 0 else env.task_infos[task_id - 1]
            env.data.qpos[:] = _join(self._qpos, row)
            env.data.qvel[:] = _join(self._qvel, row)
            env.data.time = row["simulator", "time"].item()
            mujoco.mj_forward(env.model, env.data)
            env.data.qacc_warmstart[:] = row["simulator", "warmstart"].numpy()
            env.np_random.bit_generator.state = pcg
            self._global_rng[i] = legacy

    def step(self, actions, *, active=None, capture_state=True):
        active = None if active is None else np.asarray(active.detach().cpu(), dtype=bool)
        self._check_open()
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if actions.shape != (self.num_envs, self.action_dim) or not np.isfinite(actions).all():
            raise ValueError(f"OGBMaze action must have finite shape (num_envs, {self.action_dim})")
        outputs = [
            self._in_legacy_rng(i, env.step, action) if active is None or active[i] else self._last_outputs[i]
            for i, (env, action) in enumerate(zip(self.envs, actions, strict=True))
        ]
        self._last_outputs = outputs
        infos = [o[4] for o in outputs]
        return TensorDict(
            {
                **({"state": self.get_state()} if capture_state else {}),
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
