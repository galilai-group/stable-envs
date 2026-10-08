from __future__ import annotations

import hashlib
from importlib.metadata import version

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from ...._execution.protocol import SimulatorAdapter
from ...._execution.state_columns import owned_columns
from .compatibility import CUBE_SAMPLES, MODEL_VALUES, CompatibleCube, CompatibleScene, nodes

PARTS = (
    "TIME",
    "QPOS",
    "QVEL",
    "ACT",
    "HISTORY",
    "WARMSTART",
    "CTRL",
    "QFRC_APPLIED",
    "XFRC_APPLIED",
    "EQ_ACTIVE",
    "MOCAP_POS",
    "MOCAP_QUAT",
    "USERDATA",
    "PLUGIN",
)
FLAGS = {p.lower(): getattr(mujoco.mjtState, "mjSTATE_" + p) for p in PARTS}
assert np.bitwise_or.reduce([int(f) for f in FLAGS.values()]) == int(mujoco.mjtState.mjSTATE_INTEGRATION)


def pcg_encode(rng):
    r = rng.bit_generator.state
    return {
        "state": np.frombuffer(r["state"]["state"].to_bytes(16, "little"), np.uint8).copy(),
        "increment": np.frombuffer(r["state"]["inc"].to_bytes(16, "little"), np.uint8).copy(),
        "has_uint32": np.array([r["has_uint32"]], np.int64),
        "uinteger": np.array([r["uinteger"]], np.int64),
    }


def pcg_decode(values):
    state = int.from_bytes(values["state"].tobytes(), "little")
    inc = int.from_bytes(values["increment"].tobytes(), "little")
    flag, cached = int(values["has_uint32"][0]), int(values["uinteger"][0])
    if inc % 2 != 1 or flag not in (0, 1) or not 0 <= cached < 2**32:
        raise ValueError("Invalid PCG64 increment/cache")
    return {"bit_generator": "PCG64", "state": {"state": state, "inc": inc}, "has_uint32": flag, "uinteger": cached}


def rngs(env):
    return {
        "environment": env.np_random,
        "action": env.action_space.np_random,
        "variation_parent": env.variation_parent,
        **{
            name: node.np_random
            for name, node in nodes(env.variation_space)
            if env.family != "Scene" or name.removeprefix("variation/").replace("/", ".") not in CUBE_SAMPLES
        },
    }


def site_ids(env):
    positions = [env._pinch_site_id]
    rotations = positions.copy()
    if env.family == "Scene":
        positions += [env._drawer_site_id, env._window_site_id, env._drawer_target_site_id, env._window_target_site_id]
        rotations += [env._drawer_site_id, env._window_site_id]
    return positions, rotations


def task_values(env, info=None):
    info = env.cur_task_info if info is None else info
    if env.family == "Cube":
        return {
            "init_xyz": np.asarray(info["init_xyzs"], np.float64),
            "goal_xyz": np.asarray(info["goal_xyzs"], np.float64),
        }
    result = {}
    for phase in ("init", "goal"):
        for key, value in info[phase].items():
            result[phase + "_" + key] = np.atleast_1d(value).astype(np.int64 if key == "button_states" else np.float64)
    return result


def task_payload(env, values):
    # task_name is diagnostic-only; no registered reset/step/output reads it.
    result = {"task_name": "restored_custom"}
    if env.family == "Cube":
        result.update(init_xyzs=values["init_xyz"].copy(), goal_xyzs=values["goal_xyz"].copy())
    else:
        for phase in ("init", "goal"):
            result[phase] = {key: values[phase + "_" + key].copy() for key in ("block_xyzs", "button_states")}
            for key in ("drawer_pos", "window_pos"):
                result[phase][key] = float(values[phase + "_" + key][0])
    return result


class OGBManipulationAdapter(SimulatorAdapter):
    """CPU Cube/Scene simulators with versioned, fully restorable checkpoints.

    State includes integration fields, controller caches, task data and RNG trees.
    Only initialized checkpoints from the same dependency profile are accepted.
    Restore builds and verifies all replacement simulators before installing them;
    ``device`` selects returned tensor placement."""

    execution_mode = "DELEGATED_CPU"

    def __init__(self, family, num_envs=1, device="cpu", seed=0, reward_task_id=None):
        if family not in ("Cube", "Scene") or type(num_envs) is not int or num_envs < 1:
            raise ValueError("Cube/Scene require a positive integer batch")
        self.family, self.num_envs, self.device = family, num_envs, torch.device(device)
        self.cls = CompatibleCube if family == "Cube" else CompatibleScene
        self.envs = [
            self.cls(seed=None if seed is None else int(seed) + i, reward_task_id=reward_task_id)
            for i in range(num_envs)
        ]
        self._closed = False
        self._prepared_envs = None
        fingerprint = "|".join(
            ["ogb_registered_state_v1", family]
            + [f"{p}={version(p)}" for p in ("ogbench", "mujoco", "numpy", "gymnasium")]
        )
        self.signature = np.frombuffer(hashlib.sha256(fingerprint.encode()).digest(), np.uint8).copy()
        self.reset(seed)
        self.template = self._row(self.envs[0])
        self.fields = {key: (tuple(value.shape), value.dtype) for key, value in self.template.items(True, True)}

    def _check_open(self):
        if self._closed:
            raise RuntimeError("OGBManipulationAdapter is closed")

    def reset(self, seed=None, options=None):
        self._check_open()
        if seed is not None and (not isinstance(seed, (int, np.integer)) or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        options = self.envs[0].validate_options(options)
        for lane, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else int(seed) + lane, options=options)
        return self.get_state()

    def _values(self, env):
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
                ("task", "reward_id"): np.array(
                    [-1 if env._reward_task_id is None else env._reward_task_id], np.int64
                ),
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
        return values

    def _row(self, env):
        return TensorDict(
            {k: torch.as_tensor(v.copy(), device=self.device) for k, v in self._values(env).items()},
            batch_size=[],
            device=self.device,
        )

    def get_state(self):
        self._check_open()
        return owned_columns((self._values(env) for env in self.envs), self.num_envs, self.device)

    def validate(self, state, *, _batch_size=None):
        batch = self.num_envs if _batch_size is None else _batch_size
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([batch]):
            raise TypeError(
                "Full Cube/Scene TensorDict State with batch [num_envs] required; RenderState cannot restore dynamics"
            )
        if set(state.keys(True, False)) != set(self.template.keys(True, False)):
            raise ValueError("Full State has missing or extra leaves")
        # Keep transfers and TensorDict traversal column-wise. The returned
        # rows view fresh host columns, so neither input nor another lane aliases.
        columns = {}
        for key, (shape, dtype) in self.fields.items():
            value = state[key]
            if value.shape != (batch, *shape) or value.dtype != dtype:
                raise ValueError(f"Invalid State shape/dtype/finite value: {key}")
            host = value.detach().cpu()
            if not torch.isfinite(host).all():
                raise ValueError(f"Invalid State shape/dtype/finite value: {key}")
            columns[key] = host.numpy().copy()
        rows = [{key: value[i] for key, value in columns.items()} for i in range(batch)]
        for row in rows:
            if not np.array_equal(row["compatibility", "signature"], self.signature):
                raise ValueError("Incompatible Cube/Scene model/dependency profile")
            if row["lifecycle", "reset_next_step"][0]:
                raise ValueError("Only initialized public checkpoints are supported")
            if row["physics", "time"][0] < 0:
                raise ValueError("Negative simulation time")
            for quat in [
                row["physics", "qpos"][
                    int(self.envs[0].model.joint("object_joint_0").qposadr[0]) + 3 : int(
                        self.envs[0].model.joint("object_joint_0").qposadr[0]
                    )
                    + 7
                ],
                row["physics", "mocap_quat"].reshape(-1, 4),
            ]:
                if not np.allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=1e-6, rtol=0):
                    raise ValueError("Invalid physics quaternion")
            for key in ("id", "reward_id"):
                if int(row["task", key][0]) not in (-1, 1, 2, 3, 4, 5):
                    raise ValueError("Invalid task ID")
            payload = task_payload(self.envs[0], {k[-1]: v for k, v in row.items() if k[:2] == ("task", "payload")})
            self.envs[0].validate_task(payload)
            task_id, reward_id = int(row["task", "id"][0]), int(row["task", "reward_id"][0])
            if reward_id != -1 and task_id != reward_id:
                raise ValueError("Reward task and active task are inconsistent")
            if task_id != -1:
                predefined = task_values(self.envs[0], self.envs[0].task_infos[task_id - 1])
                if any(not np.array_equal(value, row["task", "payload", key]) for key, value in predefined.items()):
                    raise ValueError("Task ID and numeric payload are inconsistent")
            rotations = row["controller", "site_xmat"].reshape(-1, 3, 3)
            if not np.allclose(
                rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=1e-8, rtol=0
            ) or not np.allclose(np.linalg.det(rotations), 1.0, atol=1e-8, rtol=0):
                raise ValueError("Invalid cached site rotation")
            for namespace in ("current", "anchor"):
                for name in sorted(self.envs[0].supported_variations):
                    if not self.envs[0].variation_space._walk(name).contains(row["variation", namespace, name]):
                        raise ValueError("Invalid Variation State range")
            for namespace in ("compiled", "mjcf"):
                for name in MODEL_VALUES:
                    if not (
                        min(0.01, float(np.float32(0.01)))
                        <= row["model", namespace, name][0]
                        <= max(0.03, float(np.float32(0.03)))
                        if name == "cube.size"
                        else self.envs[0].variation_space._walk(name).contains(row["model", namespace, name])
                    ):
                        raise ValueError("Invalid model configuration")
            if self.family == "Scene":
                for name in ("cur_button_states", "prev_button_states", "target_button_states"):
                    if not np.isin(row["scene", name], (0, 1)).all():
                        raise ValueError("Invalid Scene button bit")
                for name, low, high in [("target_drawer_pos", -0.16, 0.0), ("target_window_pos", 0.0, 0.2)]:
                    if not low <= row["scene", name][0] <= high:
                        raise ValueError("Invalid Scene target")
            for name in rngs(self.envs[0]):
                pcg_decode({k: row["rng", name, k] for k in ("state", "increment", "has_uint32", "uinteger")})
            key = row["rng", "task", "key"]
            if (
                np.any(key < 0)
                or np.any(key >= 2**32)
                or not key.any()
                or not 0 <= row["rng", "task", "position"][0] <= 624
                or row["rng", "task", "has_gauss"][0] not in (0, 1)
            ):
                raise ValueError("Invalid MT19937 State")
        return rows

    def _restore(self, row):
        # A temporary authoritative instance makes compilation/restore atomic even on failure.
        env = self.cls(seed=0)
        try:
            for name in MODEL_VALUES:
                env.variation_space._walk(name)._value = row["model", "compiled", name].copy()
            env._mjcf_model = env.modify_mjcf_model(env.build_mjcf_model())
            env.compile_model_and_data()
            for name in MODEL_VALUES:
                env.variation_space._walk(name)._value = row["model", "mjcf", name].copy()
            env._mjcf_model = env.modify_mjcf_model(env._mjcf_model)
            env._dirty = env._never_compiled = env._reset_next_step = False
            for name in sorted(env.supported_variations):
                node = env.variation_space._walk(name)
                node.set_init_value(row["variation", "anchor", name].copy())
                node.set_value(row["variation", "current", name].copy())
            env.cur_task_id = None if row["task", "id"][0] == -1 else int(row["task", "id"][0])
            env._reward_task_id = None if row["task", "reward_id"][0] == -1 else int(row["task", "reward_id"][0])
            env.cur_task_info = task_payload(env, {k[-1]: v for k, v in row.items() if k[:2] == ("task", "payload")})
            env._success = bool(row["task", "success"][0])
            env._cur_goal_ob = row["task", "goal_observation"].copy()
            env._prev_qpos, env._prev_qvel = row["output", "prev_qpos"].copy(), row["output", "prev_qvel"].copy()
            if self.family == "Scene":
                for name in ("cur_button_states", "prev_button_states", "target_button_states"):
                    setattr(env, "_" + name, row["scene", name].copy())
                for name in ("target_drawer_pos", "target_window_pos"):
                    setattr(env, "_" + name, float(row["scene", name][0]))
                env.model.site("drawer_handle_center_target").pos[1] = env._target_drawer_pos
                env.model.site("window_handle_center_target").pos[0] = env._target_window_pos
                env._apply_button_states()  # Upstream damping/forward only; no button transition.
            env._cube_colors_from_variation()
            # Match target visibility without calling state-changing post_step.
            for ids in env._cube_target_geom_ids_list:
                env.model.geom_rgba[ids, 3] = 0.2
            for pass_index in range(2):
                for name, flag in FLAGS.items():
                    if ("physics", name) in row:
                        mujoco.mj_setState(env.model, env.data, row["physics", name].astype(np.float64), flag)
                if pass_index == 0:
                    mujoco.mj_forward(env.model, env.data)
            pos, rot = site_ids(env)
            env.data.site_xpos[pos] = row["controller", "site_xpos"]
            env.data.site_xmat[rot] = row["controller", "site_xmat"]
            env.data.body("ur5e/robotiq/right_pad").cfrc_ext[:] = row["output", "gripper_force"]
            for name, rng in rngs(env).items():
                rng.bit_generator.state = pcg_decode(
                    {k: row["rng", name, k] for k in ("state", "increment", "has_uint32", "uinteger")}
                )
            env.task_rng.set_state(
                (
                    "MT19937",
                    row["rng", "task", "key"].astype(np.uint32),
                    int(row["rng", "task", "position"][0]),
                    int(row["rng", "task", "has_gauss"][0]),
                    float(row["rng", "task", "cached_gaussian"][0]),
                )
            )
            restored = self._row(env)
            for key, value in restored.items(True, True):
                if not np.array_equal(value.cpu().numpy(), row[key]):
                    raise ValueError(f"State could not be reconstructed exactly: {key}")
            return env
        except BaseException:
            env.close()
            raise

    def prepare_restore(self, state):
        self._check_open()
        if self._prepared_envs is not None:
            raise RuntimeError("A Cube/Scene restore is already prepared")
        rows = self.validate(state)  # All lanes before any simulator mutation/construction.
        candidates = []
        try:
            for row in rows:
                candidates.append(self._restore(row))
        except BaseException:
            for env in candidates:
                env.close()
            raise
        self._prepared_envs = candidates

    def discard_restore(self):
        candidates, self._prepared_envs = self._prepared_envs, None
        for env in candidates or ():
            env.close()

    def commit_restore(self):
        self._check_open()
        if self._prepared_envs is None:
            raise RuntimeError("No Cube/Scene restore is prepared")
        previous, self.envs = self.envs, self._prepared_envs
        self._prepared_envs = None
        for env in previous:
            env.close()

    def set_state(self, state):
        self.prepare_restore(state)
        self.commit_restore()

    @property
    def nq(self):
        return self.envs[0].model.nq

    def make_variation_space(self):
        from .compatibility import variation_tree

        tree = variation_tree(self.envs[0])
        if self.family == "Scene":
            for key in CUBE_SAMPLES:
                del tree["cube"].spaces[key.split(".")[1]]
        return tree

    def variation_values(self, layout, *, anchor=False):
        rows = [np.concatenate([np.asarray(getattr(env.variation_space._walk(name),
                               "init_value" if anchor else "value")).reshape(-1)
                               for name, _, _ in layout]) for env in self.envs]
        return torch.as_tensor(np.stack(rows), dtype=torch.float64, device=self.device)

    def configure(self, values):
        self._check_open()
        if len(values) != self.num_envs:
            raise ValueError("Variation batch mismatch")
        for env, value in zip(self.envs, values):
            env.validate_options({"variation": [], "variation_values": value})
        for env, value in zip(self.envs, values):
            for key, val in value.items():
                node = env.variation_space._walk(key)
                current = node.value.copy()
                node.set_init_value(val)
                node.set_value(current)

    def step(self, actions, *, active=None, capture_state=True):
        active = None if active is None else np.asarray(active.detach().cpu(), dtype=bool)
        self._check_open()
        if torch.is_tensor(actions):
            actions = actions.detach().cpu().numpy()
        actions = np.asarray(actions)
        if (
            actions.shape != (self.num_envs, 5)
            or actions.dtype not in (np.float32, np.float64)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Actions must be finite float32/float64 [num_envs, 5]")
        results = [env.step(action) if active is None or active[i] else self._last_outputs[i]
                   for i, (env, action) in enumerate(zip(self.envs, actions))]
        self._last_outputs = results
        return TensorDict(
            {
                **({"state": self.get_state()} if capture_state else {}),
                "reward": torch.tensor([r[1] for r in results], dtype=torch.float64, device=self.device),
                "success": torch.tensor([bool(r[4]["success"]) for r in results], device=self.device),
                "terminated": torch.tensor([bool(r[2]) for r in results], device=self.device),
                "truncated": torch.tensor([bool(r[3]) for r in results], device=self.device),
            },
            [self.num_envs],
            device=self.device,
        )

    def close(self):
        if not self._closed:
            self.discard_restore()
            for env in self.envs:
                env.close()
            self._closed = True
