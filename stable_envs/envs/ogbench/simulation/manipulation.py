from __future__ import annotations

import hashlib
from importlib.metadata import version

import mujoco
import numpy as np
import torch
from tensordict import TensorDict

from ...._core.state_schema import Field, StateSchema
from ...._execution.process import PER_INSTANCE
from ...._execution.protocol import SimulatorAdapter
from .compatibility import (
    MODEL_VALUES,
    CompatibleCube,
    CompatibleScene,
    InstancePool,
    nodes,
    target_alpha,
    variation_tree,
)
from .pcg import pcg_decode, pcg_encode

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
PCG_KEYS = ("state", "increment", "has_uint32", "uinteger")


def field(*parts):
    """The State field name of ``parts``: joined by ``/``, with a dotted Variation name split at its dots."""
    return "/".join(parts).replace(".", "/")


def state_schema(family, fields):
    """The ``StateSchema`` of Cube/Scene State ``fields``, ``{name: (shape, dtype)}`` in State order.

    A group holds the fields of one section and dtype, such as ``physics/float64``.
    """
    return StateSchema(
        f"OGB{family}",
        [
            Field(name, shape, dtype, f"{name.split('/')[0]}/{str(dtype).removeprefix('torch.')}")
            for name, (shape, dtype) in fields.items()
        ],
    )


def pcg_fields(schema, state, name):
    """``{key: value}`` of the PCG64 generator ``name`` in ``state``, a State or a ``{group: array}`` mapping."""
    return {key: schema.get(state, field("rng", name, key)) for key in PCG_KEYS}


def groups_of(schema, state):
    """``{group key: (B, size) host array}`` of a State, or of a ``{group key: array}`` mapping."""
    if isinstance(state, dict):
        return state
    return {key: state.get(group.path).numpy() for key, group in schema.groups.items()}


def rngs(env):
    return {
        "environment": env.np_random,
        "action": env.action_space.np_random,
        "variation_parent": env.variation_parent,
        **{name: node.np_random for name, node in nodes(env.variation_space)},
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


_GROUP_RANK = {"compatibility": 0, "physics": 1, "controller": 2, "output": 3, "lifecycle": 5, "scene": 7}
_TASK_RANK = {"goal_observation": 4.0, "goal_scene": 4.05, "id": 4.1, "reward_id": 4.2, "success": 4.3, "payload": 6}
_LATE_RANK = {"variation": 8, "model": 9}


def _leaf_rank(name):
    """Sort key that keeps the state field order of one combined capture; Python's sort is stable."""
    key = name.split("/")
    group = key[0]
    if group == "task":
        return (_TASK_RANK[key[1]], 0)
    if group == "rng":
        return (11, 0) if key[1] == "task" else (10, key[1] != "environment")
    return (_GROUP_RANK.get(group, _LATE_RANK.get(group, 12)), 0)


class OGBManipulationAdapter(SimulatorAdapter):
    """CPU Cube/Scene simulators with versioned, fully restorable checkpoints.

    State includes integration fields, controller caches, task data and RNG trees.
    Only initialized checkpoints from the same dependency profile are accepted.
    Restore builds and verifies all replacement simulators before installing them;
    ``device`` selects returned tensor placement.
    """

    execution_mode = "DELEGATED_CPU"
    rotation_tolerance = 1e-8
    process_methods = {"configure": PER_INSTANCE}
    process_attributes = ("family", "fields", "schema", "nq")
    action_dim = 5

    def __init__(self, family, num_envs=1, device="cpu", seed=0, reward_task_id=None):
        if family not in ("Cube", "Scene") or type(num_envs) is not int or num_envs < 1:
            raise ValueError("Cube/Scene require a positive integer batch")
        self.family, self.num_envs, self.device = family, num_envs, torch.device(device)
        self.cls = CompatibleCube if family == "Cube" else CompatibleScene
        self._pool = InstancePool()
        self.envs = [
            self.cls(seed=None if seed is None else int(seed) + i, reward_task_id=reward_task_id, pool=self._pool)
            for i in range(num_envs)
        ]
        self._closed = False
        self._prepared_envs = None
        self._order = None
        self._static_columns = None
        fingerprint = "|".join(
            ["ogb_registered_state_v2", family]
            + [f"{p}={version(p)}" for p in ("ogbench", "mujoco", "numpy", "gymnasium")]
        )
        self.signature = np.frombuffer(hashlib.sha256(fingerprint.encode()).digest(), np.uint8).copy()
        self.reset_envs(seed)
        #: ``{field name: (per-environment shape, dtype)}`` in State order.
        self.fields = {
            name: (tuple(value.shape), torch.from_numpy(value).dtype)
            for name, value in self._values(self.envs[0]).items()
        }
        self.schema = state_schema(family, self.fields)
        self._keys = set(self.schema.new(1).keys(True, False))

    def _check_open(self):
        if self._closed:
            raise RuntimeError("OGBManipulationAdapter is closed")

    def reset(self, seed=None, options=None):
        self.reset_envs(seed, options)
        return self.get_state()

    def reset_columns(self, seed=None, options=None):
        """``reset`` that returns ``columns()`` instead of a State."""
        self.reset_envs(seed, options)
        return self.columns()

    def reset_envs(self, seed=None, options=None):
        """``reset`` without building the State."""
        self._check_open()
        if seed is not None and (not isinstance(seed, (int, np.integer)) or seed < 0):
            raise ValueError("seed must be a nonnegative integer or None")
        instances = options if isinstance(options, list) else self.envs[0].instance_options(options, self.num_envs)
        self._forget_static(self.envs)
        for instance, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else int(seed) + instance, options=instances[instance])

    def _dynamic_values(self, env):
        """The state fields a step can change, by field name."""
        values = {}
        for name, flag in FLAGS.items():
            size = mujoco.mj_stateSize(env.model, flag)
            if size:
                value = np.empty(size, np.float64)
                mujoco.mj_getState(env.model, env.data, value, flag)
                values[field("physics", name)] = value.astype(bool) if name == "eq_active" else value
        pos, rot = site_ids(env)
        values.update(
            {
                "controller/site_xpos": env.data.site_xpos[pos].copy(),
                "controller/site_xmat": env.data.site_xmat[rot].copy(),
                "output/gripper_force": env.data.body("ur5e/robotiq/right_pad").cfrc_ext.copy(),
                "output/prev_qpos": env._prev_qpos.copy(),
                "output/prev_qvel": env._prev_qvel.copy(),
                "task/success": np.array([env._success], bool),
                "lifecycle/reset_next_step": np.array([env._reset_next_step], bool),
            }
        )
        if self.family == "Scene":
            for key in ("cur_button_states", "prev_button_states", "target_button_states"):
                values[field("scene", key)] = np.asarray(getattr(env, "_" + key), np.int64).copy()
            for key in ("target_drawer_pos", "target_window_pos"):
                values[field("scene", key)] = np.array([getattr(env, "_" + key)], np.float64)
        # Scene's post_step draws from the environment generator when it repairs an unstable cube.
        for key, value in pcg_encode(env.np_random).items():
            values[field("rng", "environment", key)] = value
        _, key, position, flag, cached = env.task_rng.get_state()
        values.update(
            {
                "rng/task/key": key.astype(np.int64),
                "rng/task/position": np.array([position], np.int64),
                "rng/task/has_gauss": np.array([flag], np.int64),
                "rng/task/cached_gaussian": np.array([cached], np.float64),
            }
        )
        return values

    def _static_values(self, env):
        """The state fields only reset, restore and ``configure`` change, cached on the environment until then.

        The fields are copies of the PCG64 generators, the task, the variation and the compiled model values.
        """
        cached = getattr(env, "_snv_static", None)
        if cached is not None:
            return cached
        values = {
            "compatibility/signature": self.signature,
            "task/goal_observation": env._cur_goal_ob.copy(),
            **{field("task", "goal_scene", name): value.copy() for name, value in env.goal_scene.items()},
            "task/id": np.array([-1 if env.cur_task_id is None else env.cur_task_id], np.int64),
            "task/reward_id": np.array([-1 if env._reward_task_id is None else env._reward_task_id], np.int64),
        }
        for key, value in task_values(env).items():
            values[field("task", "payload", key)] = value
        for name in sorted(env.supported_variations):
            node = env.variation_space._walk(name)
            values[field("variation", "current", name)] = np.asarray(node.value, np.float64).copy()
            values[field("variation", "anchor", name)] = np.asarray(node.init_value, np.float64).copy()
        for namespace, source in [("compiled", env._model_values), ("mjcf", env._mjcf_values)]:
            for key in MODEL_VALUES:
                values[field("model", namespace, key)] = source[key].copy()
        for name, rng in rngs(env).items():
            if name != "environment":
                for key, value in pcg_encode(rng).items():
                    values[field("rng", name, key)] = value
        env._snv_static = values
        return values

    def _values(self, env):
        values = {**self._static_values(env), **self._dynamic_values(env)}
        if self._order is None:
            self._order = tuple(sorted(values, key=_leaf_rank))
        return {key: values[key] for key in self._order}

    def _forget_static(self, envs):
        """Drop the cached static fields; the next capture recomputes them."""
        self._static_columns = None
        for env in envs:
            env.__dict__.pop("_snv_static", None)

    def columns(self):
        """The State as ``{group key: (num_envs, size) array}``, each a new array."""
        self._check_open()
        dynamic = [self._dynamic_values(env) for env in self.envs]
        if self._static_columns is None:
            rows = [self._static_values(env) for env in self.envs]
            self._static_columns = {name: np.stack([row[name] for row in rows]) for name in rows[0]}
        static = self._static_columns
        named = {
            name: static[name] if name in static else np.stack([row[name] for row in dynamic]) for name in self.fields
        }
        return self.schema.pack(named)

    def get_state(self):
        groups = self.schema.groups
        return TensorDict(
            {groups[key].path: torch.from_numpy(value).to(self.device) for key, value in self.columns().items()},
            [self.num_envs],
            device=self.device,
        )

    def render_columns(self):
        """``{field name: (num_envs, ...) array}`` of the State fields a RenderState projects, as State holds them."""
        self._check_open()
        rows = [self._render_values(env) for env in self.envs]
        return {name: np.stack([row[name] for row in rows]) for name in rows[0]}

    def _render_values(self, env):
        # The same values mj_getState copies for these flags.
        values = {
            field("physics", name): getattr(env.data, name).ravel().copy()
            for name in ("qpos", "mocap_pos", "mocap_quat")
        }
        for key in MODEL_VALUES:
            values[field("model", "compiled", key)] = env._model_values[key].copy()
        names = ("cube.color", "lock_color") if self.family == "Scene" else ("cube.color",)
        for name in names:
            value = np.asarray(env.variation_space._walk(name).value, np.float64).copy()
            values[field("variation", "current", name)] = value
        if self.family == "Scene":
            values[field("scene", "cur_button_states")] = np.asarray(env._cur_button_states, np.int64).copy()
        return values

    def validate(self, state, *, _batch_size=None):
        """Check full State and return one ``{group key: array}`` row per environment, each a view of a fresh host copy."""
        batch = self.num_envs if _batch_size is None else _batch_size
        if not isinstance(state, TensorDict) or state.batch_size != torch.Size([batch]):
            raise TypeError(
                "Full Cube/Scene TensorDict State with batch [num_envs] required; RenderState cannot restore dynamics"
            )
        schema = self.schema
        if set(state.keys(True, False)) != self._keys:
            raise ValueError("Full State has missing or extra leaves")
        schema.validate(state)
        # One transfer per group. The rows view fresh host columns, so neither the input nor another environment aliases.
        columns = {key: state.get(group.path).detach().cpu().numpy().copy() for key, group in schema.groups.items()}
        for key, value in columns.items():
            if value.dtype.kind == "f" and not np.isfinite(value).all():
                bad = next(
                    f.name for f in schema.groups[key].fields if not np.isfinite(schema.get(columns, f.name)).all()
                )
                raise ValueError(f"Invalid State shape/dtype/finite value: {bad}")
        rows = [{key: value[i] for key, value in columns.items()} for i in range(batch)]
        env = self.envs[0]
        cube = int(env.model.joint("object_joint_0").qposadr[0])
        at = schema.get
        for row in rows:
            if not np.array_equal(at(row, "compatibility/signature"), self.signature):
                raise ValueError("Incompatible Cube/Scene model/dependency profile")
            if at(row, "lifecycle/reset_next_step")[0]:
                raise ValueError("Only initialized public checkpoints are supported")
            if at(row, "physics/time")[0] < 0:
                raise ValueError("Negative simulation time")
            for quat in [
                at(row, "physics/qpos")[cube + 3 : cube + 7],
                at(row, "physics/mocap_quat").reshape(-1, 4),
                at(row, "task/goal_scene/qpos")[cube + 3 : cube + 7],
                at(row, "task/goal_scene/mocap_quat").reshape(-1, 4),
            ]:
                if not np.allclose(np.linalg.norm(quat, axis=-1), 1.0, atol=1e-6, rtol=0):
                    raise ValueError("Invalid physics quaternion")
            task_id, reward_id = int(at(row, "task/id")[0]), int(at(row, "task/reward_id")[0])
            if task_id not in (-1, 1, 2, 3, 4, 5) or reward_id not in (-1, 1, 2, 3, 4, 5):
                raise ValueError("Invalid task ID")
            payload = self._payload(row)
            env.validate_task(task_payload(env, payload))
            if reward_id != -1 and task_id != reward_id:
                raise ValueError("Reward task and active task are inconsistent")
            if task_id != -1:
                predefined = task_values(env, env.task_infos[task_id - 1])
                if any(not np.array_equal(value, payload[key]) for key, value in predefined.items()):
                    raise ValueError("Task ID and numeric payload are inconsistent")
            rotations = at(row, "controller/site_xmat").reshape(-1, 3, 3)
            if not np.allclose(
                rotations @ rotations.transpose(0, 2, 1), np.eye(3), atol=self.rotation_tolerance, rtol=0
            ) or not np.allclose(np.linalg.det(rotations), 1.0, atol=self.rotation_tolerance, rtol=0):
                raise ValueError("Invalid cached site rotation")
            for namespace in ("current", "anchor"):
                for name in sorted(env.supported_variations):
                    if not env.variation_space._walk(name).contains(at(row, field("variation", namespace, name))):
                        raise ValueError("Invalid Variation State range")
            for namespace in ("compiled", "mjcf"):
                for name in MODEL_VALUES:
                    value = at(row, field("model", namespace, name))
                    if not (
                        min(0.01, float(np.float32(0.01))) <= value[0] <= max(0.03, float(np.float32(0.03)))
                        if name == "cube.size"
                        else env.variation_space._walk(name).contains(value)
                    ):
                        raise ValueError("Invalid model configuration")
            if self.family == "Scene":
                for name in (
                    "scene/cur_button_states",
                    "scene/prev_button_states",
                    "scene/target_button_states",
                    "task/goal_scene/button_states",
                ):
                    if not np.isin(at(row, name), (0, 1)).all():
                        raise ValueError("Invalid Scene button bit")
                for name, low, high in [("target_drawer_pos", -0.16, 0.0), ("target_window_pos", 0.0, 0.2)]:
                    if not low <= at(row, field("scene", name))[0] <= high:
                        raise ValueError("Invalid Scene target")
            for name in rngs(env):
                pcg_decode(pcg_fields(schema, row, name))
            key = at(row, "rng/task/key")
            if (
                np.any(key < 0)
                or np.any(key >= 2**32)
                or not key.any()
                or not 0 <= at(row, "rng/task/position")[0] <= 624
                or at(row, "rng/task/has_gauss")[0] not in (0, 1)
            ):
                raise ValueError("Invalid MT19937 State")
        return rows

    def _payload(self, row):
        """``{key: value}`` of the task payload fields of a row."""
        prefix = "task/payload/"
        return {name[len(prefix) :]: self.schema.get(row, name) for name in self.fields if name.startswith(prefix)}

    def _restore(self, row):
        # A temporary authoritative instance makes compilation/restore atomic even on failure.
        env = self.cls(seed=0, pool=self._pool)
        at = self.schema.get
        try:
            for name in MODEL_VALUES:
                env.variation_space._walk(name)._value = at(row, field("model", "compiled", name)).copy()
            env._mjcf_model = env.modify_mjcf_model(env.build_mjcf_model())
            env.compile_model_and_data()
            for name in MODEL_VALUES:
                env.variation_space._walk(name)._value = at(row, field("model", "mjcf", name)).copy()
            env._mjcf_model = env.modify_mjcf_model(env._mjcf_model)
            env._dirty = env._never_compiled = env._reset_next_step = False
            for name in sorted(env.supported_variations):
                node = env.variation_space._walk(name)
                node.set_init_value(at(row, field("variation", "anchor", name)).copy())
                node.set_value(at(row, field("variation", "current", name)).copy())
            task_id, reward_id = int(at(row, "task/id")[0]), int(at(row, "task/reward_id")[0])
            env.cur_task_id = None if task_id == -1 else task_id
            env._reward_task_id = None if reward_id == -1 else reward_id
            env.cur_task_info = task_payload(env, self._payload(row))
            env._success = bool(at(row, "task/success")[0])
            env._cur_goal_ob = at(row, "task/goal_observation").copy()
            env.goal_scene = {
                name.removeprefix("task/goal_scene/"): at(row, name).copy()
                for name in self.fields
                if name.startswith("task/goal_scene/")
            }
            env._prev_qpos, env._prev_qvel = at(row, "output/prev_qpos").copy(), at(row, "output/prev_qvel").copy()
            if self.family == "Scene":
                for name in ("cur_button_states", "prev_button_states", "target_button_states"):
                    setattr(env, "_" + name, at(row, field("scene", name)).copy())
                for name in ("target_drawer_pos", "target_window_pos"):
                    setattr(env, "_" + name, float(at(row, field("scene", name))[0]))
                env.model.site("drawer_handle_center_target").pos[1] = env._target_drawer_pos
                env.model.site("window_handle_center_target").pos[0] = env._target_window_pos
                env._apply_button_states()  # Upstream damping/forward only; no button transition.
            env._cube_colors_from_variation()
            # Match target visibility without calling state-changing post_step.
            for ids in env._cube_target_geom_ids_list:
                env.model.geom_rgba[ids, 3] = target_alpha(env)
            for pass_index in range(2):
                for name, flag in FLAGS.items():
                    if field("physics", name) in self.fields:
                        value = at(row, field("physics", name)).astype(np.float64)
                        mujoco.mj_setState(env.model, env.data, value, flag)
                if pass_index == 0:
                    mujoco.mj_forward(env.model, env.data)
            pos, rot = site_ids(env)
            env.data.site_xpos[pos] = at(row, "controller/site_xpos")
            env.data.site_xmat[rot] = at(row, "controller/site_xmat")
            env.data.body("ur5e/robotiq/right_pad").cfrc_ext[:] = at(row, "output/gripper_force")
            for name, rng in rngs(env).items():
                rng.bit_generator.state = pcg_decode(pcg_fields(self.schema, row, name))
            env.task_rng.set_state(
                (
                    "MT19937",
                    at(row, "rng/task/key").astype(np.uint32),
                    int(at(row, "rng/task/position")[0]),
                    int(at(row, "rng/task/has_gauss")[0]),
                    float(at(row, "rng/task/cached_gaussian")[0]),
                )
            )
            for name, value in self._values(env).items():
                if not np.array_equal(value, at(row, name)):
                    raise ValueError(f"State could not be reconstructed exactly: {name}")
            return env
        except BaseException:
            env.close()
            raise

    def prepare_restore(self, state):
        self._check_open()
        if self._prepared_envs is not None:
            raise RuntimeError("A Cube/Scene restore is already prepared")
        rows = self.validate(state)  # All environments before any simulator mutation/construction.
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
        self._static_columns = None
        for env in previous:
            env.close()

    def set_state(self, state):
        self.prepare_restore(state)
        self.commit_restore()

    @property
    def nq(self):
        return self.envs[0].model.nq

    def make_variation_space(self):
        return variation_tree(self.envs[0])

    def variation_values(self, layout, *, anchor=False):
        rows = [
            np.concatenate(
                [
                    np.asarray(getattr(env.variation_space._walk(name), "init_value" if anchor else "value")).reshape(
                        -1
                    )
                    for name, _, _ in layout
                ]
            )
            for env in self.envs
        ]
        return torch.as_tensor(np.stack(rows), dtype=torch.float64, device=self.device)

    def configure(self, values):
        self._check_open()
        if len(values) != self.num_envs:
            raise ValueError("Variation batch mismatch")
        for env, value in zip(self.envs, values):
            env.validate_options({"variation": [], "variation_values": value})
        self._forget_static(self.envs)
        for env, value in zip(self.envs, values):
            env.configure_variation(value)

    def check_step(self, actions, active):
        actions = actions.detach().cpu().numpy() if torch.is_tensor(actions) else np.asarray(actions)
        if (
            actions.shape != (self.num_envs, 5)
            or actions.dtype not in (np.float32, np.float64)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Actions must be finite float32/float64 [num_envs, 5]")
        if active is not None:
            active = active.detach().cpu().numpy() if torch.is_tensor(active) else np.asarray(active)
            if active.shape != (self.num_envs,) or active.dtype != np.bool_:
                raise ValueError("active must be a bool vector with shape [num_envs]")
        return actions, active

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
        results = [
            env.transition(action) if active is None or active[i] else self._last_outputs[i]
            for i, (env, action) in enumerate(zip(self.envs, actions))
        ]
        self._last_outputs = results
        return TensorDict(
            {
                **({"state": self.get_state()} if capture_state else {}),
                "reward": torch.tensor([r[0] for r in results], dtype=torch.float64, device=self.device),
                "success": torch.tensor([bool(r[3]) for r in results], device=self.device),
                "terminated": torch.tensor([bool(r[1]) for r in results], device=self.device),
                "truncated": torch.tensor([bool(r[2]) for r in results], device=self.device),
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
