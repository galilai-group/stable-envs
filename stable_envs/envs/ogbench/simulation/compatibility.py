from __future__ import annotations

import copy
import threading
from collections import OrderedDict
from importlib.metadata import version

import gymnasium as gym
import mujoco
import numpy as np
from ogbench.manipspace import lie
from ogbench.manipspace.envs.cube_env import CubeEnv
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
from ogbench.manipspace.envs.scene_env import SceneEnv

from ....spaces import Box, Dict

MODEL_VALUES = ("cube.size", "agent.color", "floor.color", "camera.angle_delta", "light.intensity")

CUBE_SAMPLES = ("cube.start_position", "cube.start_yaw", "cube.goal_position", "cube.goal_yaw")

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


def nodes(tree, prefix="variation"):
    yield prefix, tree
    if isinstance(tree, gym.spaces.Dict):
        for name, child in tree.spaces.items():
            yield from nodes(child, prefix + "/" + name)


def variation_tree(env):
    """Build owned Variation values; Gym determines the sorted child-seeding order."""

    def box(low, high, shape, initial):
        return Box(low, high, shape=shape, dtype=np.float64, init_value=initial)

    cube = {
        "color": box(0.0, 1.0, (1, 3), env._cube_colors[:1, :3]),
        "size": box(0.01, 0.03, (1,), 0.02 * np.ones(1, np.float32)),
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


class Compatibility:
    """Registered reset, Variation and output rules over OGBench 1.2.1.

    Physics, task construction, controller and IK remain upstream. Arm reset uses
    a configured position instead of sampling three coordinates, so RNG draw order
    must be preserved when changing initialization."""

    def __init__(self, *, seed=0, reward_task_id=None):
        if version("ogbench") != "1.2.1":
            raise RuntimeError("Cube/Scene CPU semantics require unmodified ogbench==1.2.1")
        if reward_task_id is not None and (type(reward_task_id) is not int or reward_task_id not in range(6)):
            raise ValueError("reward_task_id must be None or an integer in 0..5")
        self._action_box = None
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

    def _seed_owned(self, seed):
        root = np.random.SeedSequence(seed)
        action, variation, task = root.spawn(3)
        self._construction_seed = int(root.entropy)
        self._action_seed = int(action.generate_state(1, dtype=np.uint64)[0])
        self.variation_parent = np.random.default_rng(variation)
        self.task_rng = np.random.RandomState(int(task.generate_state(1)[0]))
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
        # SWM visits requested leaves in the tree's sorted order, not request order.
        tree.update([name for name in tree.names() if name in selected])
        explicit_size = options.get("variation_values", {}).get("cube.size")
        # NumPy 2 weak Python scalars multiply the source float32 size vector in
        # float32; typed float64 samples/overrides promote it. Retain that effect.
        self._size_float32 = explicit_size is not None and (
            not isinstance(explicit_size, np.ndarray) or explicit_size.dtype == np.float32
        )
        tree.set_value(options.get("variation_values", {}))
        return super().reset(seed=seed, options=options)

    def initialize_arm(self):
        """Initialize at the configured position, consuming only the source yaw draw."""
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
        super().compile_model_and_data()
        self._model_values = copy.deepcopy(self._mjcf_values)


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


def compiled_fields(family, values):
    """Model arrays and compiled values of one MODEL_VALUES configuration, cached per distinct value."""
    key = (family, *(np.asarray(values[k], np.float64).tobytes() for k in MODEL_VALUES))
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
