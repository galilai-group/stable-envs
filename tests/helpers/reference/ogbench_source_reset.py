"""Unmodified OGBench Cube/Scene with its two unseeded reset inputs fixed, and the leaves its reset sets.

OGBench draws the task from NumPy's global RNG and the settling actions from a new, unseeded action Box at
each read. Each reference env here keeps its own global RNG, seeded as ``np.random.seed(seed)`` at each seeded
reset, and one action Box seeded as a stable-envs lane seeds its action stream. Everything else is OGBench.

The registered ``cube.size`` default is ``float32(0.02)``, where OGBench compiles 0.02. With
``registered_size`` the reference compiles the same size, so physics and reset compare bit for bit.
"""

import mujoco
import numpy as np
from ogbench.manipspace.envs.cube_env import CubeEnv
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
from ogbench.manipspace.envs.scene_env import SceneEnv

from stable_envs.envs.ogbench.simulation.compatibility import legacy_seed

PARTS = ("TIME", "QPOS", "QVEL", "ACT", "WARMSTART", "CTRL", "QFRC_APPLIED", "XFRC_APPLIED", "MOCAP_POS", "MOCAP_QUAT")
FIELDS = ("cur_task_id", "_success", "_reset_next_step", "_prev_qpos", "_prev_qvel", "_cur_goal_ob")
SCENE_FIELDS = ("_cur_button_states", "_target_button_states", "_target_drawer_pos", "_target_window_pos")
MODEL_FIELDS = ("dof_damping", "site_pos")


def action_seed(seed):
    return int(np.random.SeedSequence(seed).spawn(1)[0].generate_state(1, dtype=np.uint64)[0])


class _Fixed:
    @property
    def action_space(self):
        return self.box

    def _reseed(self, seed):
        self.box.seed(action_seed(seed))
        self.legacy = np.random.RandomState(legacy_seed(seed))

    def reset(self, seed=None, options=None):
        if seed is not None:
            self._reseed(seed)
        outside = np.random.get_state()
        np.random.set_state(self.legacy.get_state())
        try:
            return super().reset(seed=seed, options=options)
        finally:
            self.legacy.set_state(np.random.get_state())
            np.random.set_state(outside)


def make(family, seed, registered_size=True):
    base = CubeEnv if family == "Cube" else SceneEnv
    kind = "single" if family == "Cube" else "scene"

    class Source(_Fixed, base):
        def __init__(self, seed):
            base.__init__(self, env_type=kind, ob_type="states", height=224, width=224)
            self.box = ManipSpaceEnv.action_space.fget(self)
            self._reseed(seed)

        def build_mjcf_model(self):
            root = super().build_mjcf_model()
            if registered_size:
                for body in ("object_0", "object_target_0"):
                    for geom in root.find("body", body).find_all("geom"):
                        geom.size = np.full(3, float(np.float32(0.02)))
            return root

    return Source(seed)


def leaves(env, task_rng=None):
    """The reset and step leaves OGBench defines, with the env's PCG64 state and its global RNG state."""
    result = {}
    for part in PARTS:
        flag = getattr(mujoco.mjtState, "mjSTATE_" + part)
        value = np.empty(mujoco.mj_stateSize(env.model, flag), np.float64)
        mujoco.mj_getState(env.model, env.data, value, flag)
        result[part] = value
    for name in ("site_xpos", "site_xmat", "cfrc_ext"):
        result[name] = getattr(env.data, name).copy()
    for name in FIELDS + (SCENE_FIELDS if isinstance(env, SceneEnv) else ()):
        result[name] = np.asarray(getattr(env, name)).copy()
    for name in MODEL_FIELDS:
        result[name] = getattr(env.model, name).copy()
    info = env.cur_task_info
    for phase in ("init", "goal") if isinstance(env, SceneEnv) else ("",):
        part = info[phase] if phase else info
        for key, value in part.items():
            result[f"task/{phase}/{key}"] = np.asarray(value).copy()
    result["np_random"] = env.np_random.bit_generator.state
    result["global"] = (task_rng if task_rng is not None else env.legacy).get_state()
    result["action"] = env.action_space.np_random.bit_generator.state
    return result


def differences(ours, theirs):
    """Names of the leaves that differ."""
    names = []
    for key in theirs:
        a, b = ours[key], theirs[key]
        if isinstance(b, tuple):
            same = a[0] == b[0] and np.array_equal(a[1], b[1]) and a[2:] == b[2:]
        elif isinstance(b, dict):
            same = a == b
        else:
            same = np.shape(a) == np.shape(b) and np.array_equal(a, b)
        if not same:
            names.append(key)
    return names
