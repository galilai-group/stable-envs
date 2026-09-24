"""Research-only entropy ownership; all simulation/task/reset methods delegate.

No process-global patches. Not a public stable-envs adapter or accepted State.
"""

from __future__ import annotations

import copy
import importlib.util
from pathlib import Path

import gymnasium as gym
import numpy as np
from ogbench.manipspace.envs.manipspace_env import ManipSpaceEnv
from stable_worldmodel import spaces as swm_spaces
from stable_worldmodel.envs.ogbench.cube_env import CubeEnv
from stable_worldmodel.envs.ogbench.scene_env import SceneEnv

PREVIOUS = Path(__file__).resolve().parent.parent / "ogb_manipulation_preflight/probe.py"
spec = importlib.util.spec_from_file_location("previous_ogb_probe", PREVIOUS)
previous = importlib.util.module_from_spec(spec)
spec.loader.exec_module(previous)
GETTER = ManipSpaceEnv.action_space.fget
BOX_SAMPLE = gym.spaces.Box.sample


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


def restore_space(space, state):
    space._np_random = None if state["rng"] is None else np.random.default_rng(0)
    if state["rng"] is not None:
        space._np_random.bit_generator.state = copy.deepcopy(state["rng"])
    if hasattr(space, "_value"):
        space._value = copy.deepcopy(state["value"])
    if hasattr(space, "_init_value"):
        space._init_value = copy.deepcopy(state["init_value"])


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
    """Mixin for authoritative classes; changes entropy boundaries only."""

    def __init__(self, *args, entropy_seed=0, design="A", **kwargs):
        self.design = design
        self.action_box = None
        self.last_box = None
        self.access_count = 0
        self.child_seeds = []
        self.variation_seeds = []
        self.settling_actions = []
        self.reset_sequence = []
        self._record_reset = False
        self._seed_owned(entropy_seed)
        super().__init__(*args, **kwargs)
        self.variation_space = OwnedVariation(self.variation_space, self)
        # The first property read constructs and seeds the genuine Box lazily.
        self.variation_space.seed(0)  # No external entropy before first checkpoint.
        gym.Env.reset(self, seed=self.entropy_seed)

    def _seed_owned(self, seed):
        # Explicit per-stream domain separation, all derivable from the public seed.
        root = np.random.SeedSequence(seed)
        children = root.spawn(3)
        self.action_seed = int(children[0].generate_state(1, dtype=np.uint64)[0])
        self.action_parent = np.random.default_rng(children[0])
        self.variation_parent = np.random.default_rng(children[1])
        self.task_rng = np.random.RandomState(int(children[2].generate_state(1)[0]))
        self.entropy_seed = int(root.entropy)
        if self.action_box is not None:
            self.action_box.seed(self.action_seed)

    @property
    def action_space(self):
        self.access_count += 1
        if self.design == "A":
            if self.action_box is None:
                self.action_box = GETTER(self)
                self.action_box.seed(self.action_seed)
            result = self.action_box
        else:
            result = GETTER(self)
            if self.design == "B":
                seed = int(self.action_parent.integers(0, 2**63, dtype=np.int64))
                result.seed(seed)
                self.child_seeds.append(seed)
        self.last_box = result
        return result

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


class LastBox:
    """Exact rejected hypothesis: merely retain the latest authoritative Box."""

    @property
    def action_space(self):
        result = GETTER(self)
        self.last_box = result
        self.access_count = getattr(self, "access_count", 0) + 1
        return result


class LastCube(LastBox, CubeEnv):
    pass


class LastScene(LastBox, SceneEnv):
    pass


def capture(env):
    state = {
        "engine": previous.engine(env),
        "cache": {k: getattr(env.data, k).copy() for k in previous.CACHE_FIELDS},
        "model": {k: getattr(env.model, k).copy() for k in previous.MODEL_FIELDS},
        "python": {k: copy.deepcopy(getattr(env, k)) for k in previous.PYTHON_FIELDS if hasattr(env, k)},
        "rng": copy.deepcopy(env.np_random.bit_generator.state),
    }
    state["variation"] = {k: space_state(v) for k, v in spaces(env.variation_space)}
    state["ownership"] = {
        name: copy.deepcopy(getattr(env, name))
        for name in ["design", "access_count", "action_seed", "entropy_seed", "child_seeds", "variation_seeds"]
        if hasattr(env, name)
    }
    for name in ["action_parent", "variation_parent"]:
        if hasattr(env, name):
            state["ownership"][name] = copy.deepcopy(getattr(env, name).bit_generator.state)
    if hasattr(env, "task_rng"):
        state["ownership"]["task_rng"] = copy.deepcopy(env.task_rng.get_state())
    state["action_box"] = None if getattr(env, "action_box", None) is None else space_state(env.action_box)
    # A diagnostic last Box is retained per instance, but not a future entropy owner.
    state["last_box"] = None if getattr(env, "last_box", None) is None else space_state(env.last_box)
    return state


def restore(env, state):
    # Same trusted-source engine restore as prior preflight, without touching globals.
    for k, v in state["python"].items():
        setattr(env, k, copy.deepcopy(v))
    for k, v in state["model"].items():
        getattr(env.model, k)[:] = v
    import mujoco

    for n in range(2):
        for part, value in state["engine"].items():
            mujoco.mj_setState(env.model, env.data, value, getattr(mujoco.mjtState, "mjSTATE_" + part))
        if n == 0:
            mujoco.mj_forward(env.model, env.data)
    for k, v in state["cache"].items():
        getattr(env.data, k)[:] = v
    env.np_random.bit_generator.state = copy.deepcopy(state["rng"])
    for path, space in spaces(env.variation_space):
        restore_space(space, state["variation"][path])
    for name, value in state["ownership"].items():
        if name in ["action_parent", "variation_parent"]:
            getattr(env, name).bit_generator.state = copy.deepcopy(value)
        elif name == "task_rng":
            env.task_rng.set_state(copy.deepcopy(value))
        else:
            setattr(env, name, copy.deepcopy(value))
    if state["action_box"] is not None:
        restore_space(env.action_box, state["action_box"])
    if state["last_box"] is not None:
        # Keep the same per-instance object where it exists; never share across envs.
        if getattr(env, "last_box", None) is None:
            env.last_box = GETTER(env)
        restore_space(env.last_box, state["last_box"])


def result(env, output):
    state = capture(env)
    # Access count and identities are instrumentation, not semantic/RNG state.
    state["ownership"].pop("access_count", None)
    state.pop("last_box")
    return {
        "state": state,
        "output": copy.deepcopy(output),
        "settling_actions": copy.deepcopy(getattr(env, "settling_actions", [])),
        "sequence": copy.deepcopy(getattr(env, "reset_sequence", [])),
    }
