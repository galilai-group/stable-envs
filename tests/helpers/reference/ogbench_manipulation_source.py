"""Source Cube/Scene classes with explicitly owned entropy, and engine/state capture.

Reset, task, physics and rendering delegate to the pinned SWM/OGBench implementations;
only the RNG boundaries are owned here.
"""

import copy

import gymnasium as gym
import mujoco
import numpy as np
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
from stable_worldmodel.envs.ogbench.scene_env import SceneEnv

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


PYTHON_FIELDS = (
    "cur_task_id",
    "cur_task_info",
    "_reward_task_id",
    "_success",
    "_target_task",
    "_target_block",
    "_target_button",
    "_target_button_states",
    "_target_drawer_pos",
    "_target_window_pos",
    "_cur_button_states",
    "_prev_button_states",
    "_cur_goal_ob",
    "_cur_goal_rendered",
    "_render_goal",
    "_prev_qpos",
    "_prev_qvel",
    "_prev_ob_info",
    "_reset_next_step",
    "_dirty",
    "_never_compiled",
    "_np_random_seed",
)


CACHE_FIELDS = ("site_xpos", "site_xmat", "cfrc_ext")


MODEL_FIELDS = ("dof_damping", "site_pos", "geom_rgba", "mat_rgba")


def encode(value):
    if isinstance(value, np.ndarray):
        return {"dtype": str(value.dtype), "shape": list(value.shape), "data": value.tolist()}
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {str(k): encode(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [encode(v) for v in value]
    return value


def engine(env):
    result = {}
    for part in PARTS:
        flag = getattr(mujoco.mjtState, "mjSTATE_" + part)
        value = np.empty(mujoco.mj_stateSize(env.model, flag), dtype=np.float64)
        mujoco.mj_getState(env.model, env.data, value, flag)
        result[part] = value
    return result


def differences(left, right, prefix=""):
    result = {}
    if isinstance(left, dict):
        for key in left:
            result.update(differences(left[key], right[key], prefix + "/" + str(key)))
    elif isinstance(left, (tuple, list)):
        for i, (a, b) in enumerate(zip(left, right, strict=True)):
            result.update(differences(a, b, prefix + "/" + str(i)))
    elif isinstance(left, np.ndarray):
        if not np.array_equal(left, right):
            result[prefix] = {
                "count": int(np.count_nonzero(left != right)),
                "max_abs": float(np.max(np.abs(left.astype(float) - np.asarray(right).astype(float)))),
            }
    elif left != right:
        result[prefix] = {"left": encode(left), "right": encode(right)}
    return result


GETTER = ManipSpaceEnv.action_space.fget


def spaces(root, prefix="variation"):
    yield prefix, root
    if isinstance(root, gym.spaces.Dict):
        for key, value in root.spaces.items():
            yield from spaces(value, prefix + "/" + key)


def space_state(space):
    rng = space._np_random  # Reading this does not lazily create entropy.
    return {
        "rng": None if rng is None else copy.deepcopy(rng.bit_generator.state),
        "value": copy.deepcopy(getattr(space, "_value", None)),
        "init_value": copy.deepcopy(getattr(space, "_init_value", None)),
    }


class OwnedVariation(swm_spaces.Dict):
    """Reuse the source's child spaces, ordering and sampling; own root seed(None)."""

    def __init__(self, original, owner):
        self.owner = owner
        super().__init__(
            original.spaces,
            init_value=original._init_value,
            constrain_fn=original.constrain_fn,
            sampling_order=original._sampling_order,
        )

    def seed(self, seed=None):
        if seed is None:
            seed = int(self.owner.variation_parent.integers(0, 2**63, dtype=np.int64))
        self.owner.variation_seeds.append(seed)
        return super().seed(seed)


class Ownership:
    """Mixin that derives the action, variation and task RNG streams from one seed."""

    def __init__(self, *args, entropy_seed=0, **kwargs):
        self.action_box = None
        self.access_count = 0
        self.variation_seeds = []
        self.settling_actions = []
        self.reset_sequence = []
        self._record_reset = False
        self._seed_owned(entropy_seed)
        super().__init__(*args, **kwargs)
        self.variation_space = OwnedVariation(self.variation_space, self)
        self.variation_space.seed(0)  # No external entropy before the first reset.
        gym.Env.reset(self, seed=self.entropy_seed)

    def _seed_owned(self, seed):
        # Explicit per-stream domain separation, all derivable from the public seed.
        root = np.random.SeedSequence(seed)
        children = root.spawn(3)
        self.action_seed = int(children[0].generate_state(1, dtype=np.uint64)[0])
        self.variation_parent = np.random.default_rng(children[1])
        self.task_rng = np.random.RandomState(int(children[2].generate_state(1)[0]))
        self.entropy_seed = int(root.entropy)
        if self.action_box is not None:
            self.action_box.seed(self.action_seed)

    @property
    def action_space(self):
        self.access_count += 1
        if self.action_box is None:
            # The source's own Box, built and seeded on first read.
            self.action_box = GETTER(self)
            self.action_box.seed(self.action_seed)
        return self.action_box

    def reset(self, seed=None, options=None, **kwargs):
        if seed is not None:
            self._seed_owned(seed)
            # Source Cube discards its argument; Gym's own seed path repairs forwarding.
            gym.Env.reset(self, seed=seed)
        options = copy.deepcopy(options) if options is not None else {}
        if self._mode == "task" and self._reward_task_id is None:
            if "task_id" not in options and "task_info" not in options:
                # Use the source's existing task_id input; no copied task/reset logic.
                options["task_id"] = int(self.task_rng.randint(1, self.num_tasks + 1))
        self.settling_actions = []
        self.reset_sequence = []
        self._record_reset = True
        try:
            return super().reset(seed=seed, options=options, **kwargs)
        finally:
            self._record_reset = False

    def step(self, action):
        if self._record_reset:
            self.settling_actions.append(np.array(action, copy=True))
            self.reset_sequence.append("step")
        return super().step(action)

    def initialize_arm(self):
        if self._record_reset:
            self.reset_sequence.append("initialize_arm")
        return super().initialize_arm()


class OwnedCube(Ownership, CubeEnv):
    pass


class OwnedScene(Ownership, SceneEnv):
    pass


def capture(env):
    state = {
        "engine": engine(env),
        "cache": {k: getattr(env.data, k).copy() for k in CACHE_FIELDS},
        "model": {k: getattr(env.model, k).copy() for k in MODEL_FIELDS},
        "python": {k: copy.deepcopy(getattr(env, k)) for k in PYTHON_FIELDS if hasattr(env, k)},
        "rng": copy.deepcopy(env.np_random.bit_generator.state),
    }
    state["variation"] = {k: space_state(v) for k, v in spaces(env.variation_space)}
    state["ownership"] = {
        name: copy.deepcopy(getattr(env, name))
        for name in ["access_count", "action_seed", "entropy_seed", "variation_seeds"]
        if hasattr(env, name)
    }
    if hasattr(env, "variation_parent"):
        state["ownership"]["variation_parent"] = copy.deepcopy(env.variation_parent.bit_generator.state)
    if hasattr(env, "task_rng"):
        state["ownership"]["task_rng"] = copy.deepcopy(env.task_rng.get_state())
    state["action_box"] = None if getattr(env, "action_box", None) is None else space_state(env.action_box)
    return state
