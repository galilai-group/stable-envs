from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from importlib.metadata import version

import gymnasium as gym
import mujoco
import numpy as np
from ogbench.manipspace import lie, mjcf_utils
from ogbench.manipspace.envs.cube_env import CubeEnv
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
from ogbench.manipspace.envs.scene_env import SceneEnv

from ....spaces import Box, Dict

MODEL_VALUES = ("cube.size", "agent.color", "floor.color", "camera.angle_delta", "light.intensity")

CUBE_SAMPLES = ("cube.start_position", "cube.start_yaw", "cube.goal_position", "cube.goal_yaw")
ARM_START = "agent.ee_start_position"

# set_variation applies these to the running episode. The rest are reset-only, and
# agent.color waits for the source's own compile rule.
IMMEDIATE = ("cube.size", "floor.color", "camera.angle_delta", "light.intensity", "cube.color", "lock_color")
COMPILED = ("cube.size", "floor.color", "camera.angle_delta", "light.intensity")
# Every MjModel array a compile derives from COMPILED; tex_data only for the grid texture.
COMPILED_FIELDS = (
    "body_inertia",
    "body_invweight0",
    "body_mass",
    "body_subtreemass",
    "bvh_aabb",
    "cam_mat0",
    "cam_poscom0",
    "cam_quat",
    "dof_M0",
    "dof_invweight0",
    "dof_length",
    "geom_aabb",
    "geom_rbound",
    "geom_size",
    "light_diffuse",
    "light_poscom0",
    "tex_data",
)
# MjModel arrays written per lane after compilation; lanes sharing a model swap them in.
LANE_FIELDS = ("geom_rgba", "mat_rgba", "site_pos", "dof_damping", "actuator_gainprm", "actuator_biasprm")


def legacy_seed(seed):
    """The ``np.random.seed`` argument for ``seed``; a seed wider than 32 bits passes as its 32-bit words."""
    if seed < 2**32:
        return seed
    return [(seed >> shift) & 0xFFFFFFFF for shift in range(0, seed.bit_length(), 32)]


def nodes(tree, prefix="variation"):
    yield prefix, tree
    if isinstance(tree, gym.spaces.Dict):
        for name, child in tree.spaces.items():
            yield from nodes(child, prefix + "/" + name)


def variation_tree(env):
    def box(low, high, shape, initial):
        return Box(low, high, shape=shape, dtype=np.float64, init_value=initial)

    cube = {
        "color": box(0.0, 1.0, (1, 3), env._cube_colors[:1, :3]),
        # OGBench compiles exactly 0.02, not float32(0.02).
        "size": box(0.01, 0.03, (1,), np.full(1, 0.02)),
        "start_position": box(
            env._object_sampling_bounds[0:1],
            env._object_sampling_bounds[1:2],
            (1, 2),
            [[0.3, 0.0]] if env.family == "Cube" else [np.mean(env._object_sampling_bounds, axis=0)],
        ),
        "start_yaw": box(0.0, 2 * np.pi, (1,), [0.0]),
        "goal_position": box(
            env._target_sampling_bounds[0:1],
            env._target_sampling_bounds[1:2],
            (1, 2),
            [[0.3, 0.0]] if env.family == "Cube" else [np.mean(env._object_sampling_bounds, axis=0)],
        ),
        "goal_yaw": box(0.0, 2 * np.pi, (1,), [0.0]),
    }
    angle = 10.0 if env.family == "Cube" else 5.0
    result = {
        "cube": Dict(cube),
        "agent": Dict(
            {
                "color": box(0.0, 1.0, (3,), env._colors["purple"][:3]),
                "ee_start_position": box(
                    env._arm_sampling_bounds[0],
                    env._arm_sampling_bounds[1],
                    (3,),
                    np.mean(env._arm_sampling_bounds, axis=0),
                ),
            }
        ),
        "floor": Dict({"color": box(0.0, 1.0, (2, 3), [[0.08, 0.11, 0.16], [0.15, 0.18, 0.25]])}),
        "camera": Dict({"angle_delta": box(-angle, angle, (1, 2), [[0.0, 0.0]])}),
        "light": Dict({"intensity": box(0.0, 1.0, (1,), [0.7 if env.family == "Cube" else 0.6])}),
    }
    if env.family == "Scene":
        result["lock_color"] = box(0.0, 1.0, (2, 3), [env._colors["red"][:3]] * 2)
    return Dict(result)


class _SharedModel:
    __slots__ = ("model", "key", "pristine", "active", "users")

    def __init__(self, model, key):
        self.model, self.key, self.active, self.users = model, key, None, 0
        self.pristine = {name: getattr(model, name).copy() for name in LANE_FIELDS}

    def activate(self, lane):
        if self.active is not None:
            for name, value in self.active._lane_fields.items():
                np.copyto(value, getattr(self.model, name))
        for name, value in lane._lane_fields.items():
            getattr(self.model, name)[...] = value
        self.active = lane


class _SharedTree:
    def __init__(self, root, attributes):
        self.root, self.attributes, self.active = root, attributes, None
        self.slots = [(root.find("texture", "grid"), "rgb1"), (root.find("texture", "grid"), "rgb2")]
        self.slots += [(root.find("material", "ur5e/robotiq/" + name), "rgba") for name in ("black", "pad_gray")]
        for body in ("object_0", "object_target_0"):
            self.slots += [(geom, "size") for geom in root.find("body", body).find_all("geom")]
        self.slots += [(root.find("camera", "front_pixels"), "xyaxes"), (root.find("light", "global"), "diffuse")]
        self.pristine = self.read()

    def read(self):
        return [None if (v := getattr(element, attr)) is None else np.array(v) for element, attr in self.slots]

    def activate(self, lane):
        if self.active is not lane:
            if self.active is not None:
                self.active._tree_values = self.read()
            for (element, attr), value in zip(self.slots, lane._tree_values):
                setattr(element, attr, None if value is None else value.copy())
            self.active = lane
        return self.root


class LanePool:
    """Compiled models, MJCF tree and IK solver shared by the lanes of one adapter.

    Lanes whose MJCF compiles to the same XML share one MjModel; a Variation write
    to compiled arrays moves the lane to a copy keyed by the written values. Each
    lane keeps its own LANE_FIELDS and MJCF Variation attributes, swapped in when
    it next touches the shared object. Lanes of one pool must not run concurrently.
    """

    def __init__(self):
        self.ik = None
        self.tree = None
        self.models = {}

    def join(self, lane, key, compile_model):
        self.leave(lane)
        shared = self.models.get(key)
        if shared is None:
            shared = self.models[key] = _SharedModel(compile_model(), key)
        shared.users += 1
        lane._shared = shared
        lane._lane_fields = {name: value.copy() for name, value in shared.pristine.items()}

    def derive(self, lane, values):
        """Move ``lane`` to its model with compiled arrays for ``values``; True if that model is new."""
        current = lane._shared
        key = (current.key if isinstance(current.key, str) else current.key[0], values)
        if key == current.key:
            return False
        shared = self.models.get(key)
        fresh = shared is None
        if fresh:
            shared = self.models[key] = _SharedModel(copy.copy(current.model), key)
        lane_fields = lane._lane_fields
        if current.active is lane:
            for name, value in lane_fields.items():
                np.copyto(value, getattr(current.model, name))
        self.leave(lane)
        shared.users += 1
        lane._shared, lane._lane_fields = shared, lane_fields
        return fresh

    def leave(self, lane):
        shared, lane._shared = lane._shared, None
        if shared is not None:
            if shared.active is lane:
                shared.active = None
            shared.users -= 1
            if not shared.users:
                del self.models[shared.key]

    def release(self, lane):
        self.leave(lane)
        if self.tree is not None and self.tree.active is lane:
            self.tree.active = None


class Compatibility:
    """Registered reset, Variation and output rules over OGBench 1.2.1.

    Physics, task construction, controller, IK and reset remain upstream. OGBench
    draws the task from NumPy's global RNG; each lane keeps its own copy, seeded as
    ``np.random.seed(seed)`` at each seeded reset. The arm starts at
    ``agent.ee_start_position`` instead of OGBench's sampled position only when the
    reset options name that leaf or its anchor is not the registered default.
    """

    _pool = None
    _shared = None

    def __init__(self, *, seed=0, reward_task_id=None, pool=None):
        if version("ogbench") != "1.2.1":
            raise RuntimeError("Cube/Scene CPU semantics require unmodified ogbench==1.2.1")
        if reward_task_id is not None and (type(reward_task_id) is not int or reward_task_id not in range(6)):
            raise ValueError("reward_task_id must be None or an integer in 0..5")
        self._action_box = None
        self._pool = pool
        self._arm_pinned = False
        self._seed_owned(seed)
        super().__init__(
            env_type="single" if self.family == "Cube" else "scene",
            ob_type="states",
            height=224,
            width=224,
            reward_task_id=reward_task_id,
        )
        gym.Env.reset(self, seed=self._construction_seed)
        self.variation_space = variation_tree(self)
        self.variation_space.seed(0)
        self.action_space  # Own the genuine Box before any checker/property-triggered reset.
        self.supported_variations = set(self.variation_space.names())
        self._camera_basis = None
        self._size_float32 = False
        if pool is not None:
            if pool.ik is None:
                pool.ik = self._ik
            self._ik = pool.ik

    @property
    def _model(self):
        shared = self._shared
        if shared is None:
            return self.__dict__.get("_own_model")
        if shared.active is not self:
            shared.activate(self)
        return shared.model

    @_model.setter
    def _model(self, value):
        self.__dict__["_own_model"] = value

    @property
    def _mjcf_model(self):
        if self._pool is None:
            return self.__dict__.get("_own_mjcf_model")
        return None if self.__dict__.get("_tree_values") is None else self._pool.tree.activate(self)

    @_mjcf_model.setter
    def _mjcf_model(self, value):
        if self._pool is None:
            self.__dict__["_own_mjcf_model"] = value
        elif value is None:
            self._tree_values = None
        elif value is not self._pool.tree.root:
            raise ValueError("Pooled lanes use their pool's MJCF tree")

    def build_mjcf_model(self):
        pool = self._pool
        if pool is None:
            return super().build_mjcf_model()
        if pool.tree is None:
            before = dict(vars(self))
            root = super().build_mjcf_model()
            pool.tree = _SharedTree(
                root, {k: v for k, v in vars(self).items() if k not in before or before[k] is not v}
            )
        else:
            self.__dict__.update(pool.tree.attributes)
        if pool.tree.active is self:
            pool.tree.active = None
        self._tree_values = pool.tree.pristine
        return pool.tree.activate(self)

    def close(self):
        if self._pool is not None:
            self._pool.release(self)
        super().close()

    def _seed_owned(self, seed):
        root = np.random.SeedSequence(seed)
        action, variation = root.spawn(2)
        self._construction_seed = int(root.entropy)
        self._action_seed = int(action.generate_state(1, dtype=np.uint64)[0])
        self.variation_parent = np.random.default_rng(variation)
        self.task_rng = np.random.RandomState(legacy_seed(self._construction_seed))
        if self._action_box is not None:
            self._action_box.seed(self._action_seed)

    @property
    def action_space(self):
        if self._action_box is None:
            self._action_box = ManipSpaceEnv.action_space.fget(self)
            self._action_box.seed(self._action_seed)
        return self._action_box

    def validate_options(self, options):
        if options is not None and not isinstance(options, dict):
            raise TypeError("Reset options must be a dictionary")
        options = copy.deepcopy(options or {})
        if set(options) - {"task_id", "task_info", "variation", "variation_values"}:
            raise ValueError("Unsupported registered Cube/Scene reset option (including render_goal/state)")
        if "task_id" in options and (type(options["task_id"]) is not int or options["task_id"] not in range(1, 6)):
            raise ValueError("task_id must be an integer in 1..5")
        if "task_info" in options:
            self.validate_task(options["task_info"])
        selected = options.get("variation", CUBE_SAMPLES if self.family == "Cube" else ())
        if not isinstance(selected, (list, tuple)) or any(k not in self.supported_variations for k in selected):
            raise ValueError("Unsupported registered Variation sample")
        values = options.get("variation_values", {})
        if not isinstance(values, dict) or set(values) - self.supported_variations:
            raise ValueError("Unsupported registered Variation value")
        for key, value in values.items():
            node = self.variation_space._walk(key)
            value = np.asarray(value, dtype=np.float64)
            if value.shape != node.shape or not np.isfinite(value).all() or not node.contains(value):
                raise ValueError(f"Invalid Variation {key}")
        return options

    def lane_options(self, options, batch):
        """Validated options per lane; a ``variation_values`` entry may hold one value per lane."""
        if options is not None and not isinstance(options, dict):
            raise TypeError("Reset options must be a dictionary")
        values = (options or {}).get("variation_values")
        if not isinstance(values, dict):
            return [self.validate_options(options)] * batch
        lanes = []
        for lane in range(batch):
            per_lane = {key: self._lane_value(key, value, lane, batch) for key, value in values.items()}
            lanes.append(self.validate_options({**options, "variation_values": per_lane}))
        return lanes

    def _lane_value(self, key, value, lane, batch):
        if key not in self.supported_variations:
            return value
        if hasattr(value, "detach"):
            value = value.detach().cpu().numpy()
        shape = self.variation_space._walk(key).shape
        if np.shape(value) == (batch, *shape) and np.shape(value) != shape:
            return value[lane]
        return value

    def configure_variation(self, values):
        """Install reset defaults; IMMEDIATE leaves also apply to the running episode."""
        changed = set()
        for key, value in values.items():
            node = self.variation_space._walk(key)
            current = node.value.copy()
            node.set_init_value(value)
            if key in IMMEDIATE:
                node.set_value(value)
                if not np.array_equal(node.value, current):
                    changed.add(key)
            else:
                node.set_value(current)
        if changed & set(COMPILED):
            target = dict(self._model_values)
            target.update({k: self.variation_space._walk(k).value for k in changed & set(COMPILED)})
            fields, compiled = compiled_fields(self.family, target)
            if self._pool is None or self._pool.derive(self, values_key(target)):
                for name, value in fields.items():
                    if name == "tex_data":
                        start = int(self.model.tex_adr[self.model.texture("grid").id])
                        self.model.tex_data[start : start + len(value)] = value
                    else:
                        getattr(self.model, name)[:] = value
            self._model_values = copy.deepcopy(compiled)
        if "cube.color" in changed:
            self._cube_colors_from_variation()
        if "lock_color" in changed:
            self._lock_colors()

    def validate_task(self, task):
        expected = self.task_infos[0]

        def check(value, template):
            if isinstance(template, dict):
                if not isinstance(value, dict) or set(value) != set(template):
                    raise ValueError("Invalid custom task keys")
                for key in template:
                    check(value[key], template[key])
            elif isinstance(template, str):
                if not isinstance(value, str):
                    raise ValueError("Invalid task name")
            else:
                if isinstance(template, np.ndarray) and not isinstance(value, np.ndarray):
                    raise ValueError("Custom task array fields require NumPy arrays, as upstream does")
                value = np.asarray(value)
                if (
                    value.shape != np.asarray(template).shape
                    or value.dtype.kind not in "iuf"
                    or not np.isfinite(value).all()
                ):
                    raise ValueError("Invalid task numeric payload")

        check(task, expected)
        if self.family == "Scene":
            for phase in ("init", "goal"):
                p = task[phase]
                if (
                    not np.isin(p["button_states"], [0, 1]).all()
                    or not -0.16 <= p["drawer_pos"] <= 0
                    or not 0 <= p["window_pos"] <= 0.2
                ):
                    raise ValueError("Invalid Scene task range")

    def reset(self, *, seed=None, options=None):
        options = self.validate_options(options)
        if seed is not None:
            self._seed_owned(seed)
            gym.Env.reset(self, seed=seed)
        if self._reward_task_id is None and not {"task_id", "task_info"} & options.keys():
            options["task_id"] = int(self.task_rng.randint(1, self.num_tasks + 1))
        tree = self.variation_space
        tree.seed(int(self.variation_parent.integers(0, 2**63, dtype=np.int64)))
        tree.reset()
        selected = set(options.get("variation", CUBE_SAMPLES if self.family == "Cube" else ()))
        default = np.mean(self._arm_sampling_bounds, axis=0)
        named = selected | set(options.get("variation_values", {}))
        self._arm_pinned = ARM_START in named or not np.array_equal(tree._walk(ARM_START).init_value, default)
        # SWM visits requested leaves in the tree's sorted order, not request order.
        tree.update([name for name in tree.names() if name in selected])
        explicit_size = options.get("variation_values", {}).get("cube.size")
        # NumPy 2 weak Python scalars keep the source float32 size vector in float32;
        # typed float64 samples/overrides promote it.
        self._size_float32 = explicit_size is not None and (
            not isinstance(explicit_size, np.ndarray) or explicit_size.dtype == np.float32
        )
        tree.set_value(options.get("variation_values", {}))
        return super().reset(seed=seed, options=options)

    def initialize_arm(self):
        """OGBench's sampled arm start; a pinned start consumes only the source yaw draw."""
        if not self._arm_pinned:
            return super().initialize_arm()
        eff_pos = self.variation_space["agent"]["ee_start_position"].value
        cur_ori = self._effector_down_rotation
        yaw = self.np_random.uniform(-np.pi, np.pi)
        rotz = lie.SO3.from_z_radians(yaw)
        eff_ori = rotz @ cur_ori
        T_wp = lie.SE3.from_rotation_and_translation(eff_ori, eff_pos)
        T_wa = T_wp @ self._T_pa
        qpos_init = self._ik.solve(pos=T_wa.translation(), quat=T_wa.rotation().wxyz, curr_qpos=self._home_qpos)
        self._data.qpos[self._arm_joint_ids] = qpos_init
        mujoco.mj_forward(self._model, self._data)

    def initialize_episode(self):
        if not hasattr(self, "_prev_qpos"):
            self._prev_qpos = self.data.qpos.copy()
            self._prev_qvel = self.data.qvel.copy()
        return super().initialize_episode()

    def _cube_colors_from_variation(self):
        color = self.variation_space["cube"]["color"].value[0]
        for gid in self._cube_geom_ids_list[0]:
            self.model.geom_rgba[gid] = [*color, 1.0]
        for gid in self._cube_target_geom_ids_list[0]:
            self.model.geom_rgba[gid, :3] = color

    def post_step(self):
        super().post_step()
        self._cube_colors_from_variation()

    def transition(self, action):
        """``step`` without the observation and info dictionaries; returns reward, terminated, truncated, success.

        The pre-step snapshot keeps the previous position and velocity and, for Scene, the button positions
        that ``post_step`` reads. The physics, State and returned values are those of ``step``.
        """
        if self._reset_next_step or self._success_timing not in ("pre", "post"):
            _, reward, terminated, truncated, info = self.step(action)
            return reward, terminated, truncated, info["success"]
        if self._success_timing == "pre":
            success, terminated, reward = self._success, self.terminate_episode(), self.compute_reward()
        self.set_control(np.array(action))
        if self.family == "Scene":
            self._prev_button_states = self._cur_button_states.copy()
        self._prev_qpos = self._data.qpos.copy()
        self._prev_qvel = self._data.qvel.copy()
        if self.family == "Scene":
            self._prev_ob_info = {
                f"privileged/button_{i}_pos": self._data.joint(f"buttonbox_joint_{i}").qpos.copy()
                for i in range(self._num_buttons)
            }
        mujoco.mj_step(self._model, self._data, nstep=self._n_steps)
        mujoco.mj_rnePostConstraint(self._model, self._data)
        self.post_step()
        if self._success_timing == "post":
            success, terminated, reward = self._success, self.terminate_episode(), self.compute_reward()
        return reward, terminated, self.truncate_episode(), success

    def get_reset_info(self):
        info = super().get_reset_info()
        info.pop("goal", None)
        info.pop("goal_rendered", None)
        return self._output_info(info)

    def get_step_info(self):
        return self._output_info(super().get_step_info())

    def _output_info(self, info):
        info.update(env_name=self.family, target=self._cur_goal_ob, success=self._success)
        return info

    def modify_mjcf_model(self, model):
        model = super().modify_mjcf_model(model)
        v = self.variation_space
        changes = []

        def assign(element, attr, value, equal_means_dirty=False):
            old = getattr(element, attr)
            same = old is not None and np.allclose(old, value)
            changes.append(same if equal_means_dirty else not same)
            setattr(element, attr, value)

        for i, attr in enumerate(("rgb1", "rgb2")):
            assign(model.find("texture", "grid"), attr, v["floor"]["color"].value[i])
        for name in ("black", "pad_gray"):
            material = model.find("material", "ur5e/robotiq/" + name)
            color = material.rgba.copy()
            color[:3] = v["agent"]["color"].value
            assign(material, "rgba", color, equal_means_dirty=True)
        cube_size = v["cube"]["size"].value[0]
        if self._size_float32:
            cube_size = np.float32(cube_size)
        for name in ("object_0", "object_target_0"):
            for geom in model.find("body", name).find_all("geom"):
                assign(geom, "size", cube_size * np.ones(3, np.float32))
        camera = model.find("camera", "front_pixels")
        if self._camera_basis is None:
            self._camera_basis = camera.xyaxes.copy()
        yaw, pitch = np.deg2rad(v["camera"]["angle_delta"].value[0])
        cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
        rotation = np.array([[1, 0, 0], [0, cp, -sp], [0, sp, cp]]) @ np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
        camera.xyaxes = np.concatenate([rotation @ self._camera_basis[:3], rotation @ self._camera_basis[3:]])
        changes.append(not np.allclose(camera.xyaxes, self._camera_basis))
        assign(model.find("light", "global"), "diffuse", v["light"]["intensity"].value[0] * np.ones(3, np.float32))
        if any(changes):
            self.mark_dirty()
        self._mjcf_values = {k: self.variation_space._walk(k).value.copy() for k in MODEL_VALUES}
        self._mjcf_values["cube.size"] = np.array([model.find("body", "object_0").find_all("geom")[0].size[0]])
        return model

    def compile_model_and_data(self):
        if self._pool is None:
            super().compile_model_and_data()
        else:
            self._compile_shared()
        self._model_values = copy.deepcopy(self._mjcf_values)

    def _compile_shared(self):
        root = self._mjcf_model
        getattr(root.visual, "global").offwidth = self._render_width
        getattr(root.visual, "global").offheight = self._render_height
        xml = mjcf_utils.to_string(root)
        self._pool.join(self, xml, lambda: mujoco.MjModel.from_xml_string(xml=xml, assets=mjcf_utils.get_assets(root)))
        self._data = mujoco.MjData(self._model)
        self._model.opt.timestep = self._physics_timestep
        mujoco.mj_resetData(self._model, self._data)
        mujoco.mj_forward(self._model, self._data)
        if self._renderer is not None:
            self._renderer.close()
            self._initialize_renderer()
        self.post_compilation()
        self._dirty = False


class CompatibleCube(Compatibility, CubeEnv):
    family = "Cube"


class CompatibleScene(Compatibility, SceneEnv):
    family = "Scene"

    def _apply_button_states(self):
        super()._apply_button_states()
        self._lock_colors()

    def _lock_colors(self):
        for i, state in enumerate(self._cur_button_states):
            color = self.variation_space["lock_color"].value[i] if state == 0 else self._colors["white"][:3]
            for gid in self._button_geom_ids_list[i]:
                self.model.geom_rgba[gid] = [*color, 1.0]
            self.model.material(("drawer_handle", "window_handle")[i]).rgba[:] = [*color, 1.0]


_FAMILIES = {"Cube": CompatibleCube, "Scene": CompatibleScene}
_COMPILE_LOCK = threading.Lock()
_SCRATCH: dict = {}
_COMPILED_CACHE: OrderedDict = OrderedDict()
_COMPILED_CACHE_MAX = 64


def values_key(values):
    return tuple(np.asarray(values[k], np.float64).tobytes() for k in MODEL_VALUES)


def compiled_fields(family, values):
    """Model arrays and compiled values of one MODEL_VALUES configuration, cached per distinct value."""
    key = (family, *values_key(values))
    with _COMPILE_LOCK:
        hit = _COMPILED_CACHE.get(key)
        if hit is not None:
            _COMPILED_CACHE.move_to_end(key)
            return hit
        env = _SCRATCH.get(family)
        if env is None:
            env = _SCRATCH[family] = _FAMILIES[family](seed=0)
        for k in MODEL_VALUES:
            env.variation_space._walk(k)._value = np.array(values[k], np.float64)
        env._mjcf_model = env.modify_mjcf_model(env.build_mjcf_model())
        env.compile_model_and_data()
        m = env.model
        grid = m.texture("grid").id
        start = int(m.tex_adr[grid])
        size = int(m.tex_height[grid] * m.tex_width[grid] * m.tex_nchannel[grid])
        fields = {name: np.array(getattr(m, name), copy=True) for name in COMPILED_FIELDS if name != "tex_data"}
        fields["tex_data"] = np.array(m.tex_data[start : start + size], copy=True)
        hit = _COMPILED_CACHE[key] = (fields, copy.deepcopy(env._model_values))
        if len(_COMPILED_CACHE) > _COMPILED_CACHE_MAX:
            _COMPILED_CACHE.popitem(last=False)
        return hit
