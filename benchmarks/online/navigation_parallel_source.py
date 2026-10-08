"""Benchmark scalar Navigation sources through Gymnasium shared-memory workers.

The independent oracle is used only for construction and untimed State handling.
Timed calls execute the pinned source classes directly, retaining every numeric
task output, including SimplePointMaze's cached goal image.
"""

import functools
import hashlib
import importlib
import inspect
import sys
import types
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch

from benchmarks.online.classic_parallel_source import ParallelSource, SourceShard
from tests.helpers.reference.native_navigation_semantics import NativeNavigationReference

NAMES = {"two_room": "TwoRoom", "piecewise": "Piecewise", "simple_point_maze": "SimplePointMaze"}


def import_source(root):
    # Skip unrelated package initializers, without modifying the environment code.
    for name, sub in (
        ("stable_worldmodel", ""),
        ("stable_worldmodel.envs", "envs"),
        ("stable_worldmodel.envs.two_room", "envs/two_room"),
        ("stable_worldmodel.envs.piecewise", "envs/piecewise"),
    ):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(root / "stable_worldmodel" / sub)]
            sys.modules[name] = package
    importlib.import_module("stable_worldmodel.utils")
    spaces = importlib.import_module("stable_worldmodel.spaces")
    sys.modules["stable_worldmodel"].spaces = spaces


def source(domain, batch):
    ref = NativeNavigationReference(domain, num_envs=batch)
    ref.reset(17)
    return ref


def step_arrays(ref, actions, frame_indices):
    outputs, frames, validity = [], [], []
    active = np.ones(ref.num_envs, dtype=bool)
    previous = [None] * ref.num_envs
    cached_frames = [None] * ref.num_envs
    for t in range(actions.shape[1]):
        validity.append(active.copy())
        rows = [env.step(actions[i, t]) if active[i] else previous[i] for i, env in enumerate(ref.envs)]
        previous = rows
        for i in np.flatnonzero(active):
            cached_frames[i] = None
        tasks = {}
        for field, index, dtype in (
            ("observation", 0, None),
            ("reward", 1, np.float64),
            ("terminated", 2, bool),
            ("truncated", 3, bool),
        ):
            values = np.stack([r[index] for r in rows])
            tasks[field] = values.astype(dtype) if dtype else values
        keys = (
            ("goal",)
            if ref.domain == "simple_point_maze"
            else ("proprio", "state", "goal_state", "distance_to_target")
        )
        for key in keys:
            tasks[f"info.{key}"] = np.stack([r[4][key] for r in rows])
        done = tasks["terminated"] | tasks["truncated"]
        for values in tasks.values():
            values[~active] = 0
        outputs.append(tasks)
        if t in frame_indices:
            for i, env in enumerate(ref.envs):
                if cached_frames[i] is None:
                    cached_frames[i] = env.render()
            frames.append(np.stack(cached_frames))
        active &= ~done
    tasks = {key: np.stack([o[key] for o in outputs], axis=1) for key in outputs[0]}
    tasks["valid"] = np.stack(validity, axis=1)
    if frames:
        tasks["frames"] = np.stack(frames, axis=1)
    return tasks


def source_step(ref, actions, indices, device):
    return {k: torch.as_tensor(v, device=device) for k, v in step_arrays(ref, actions, indices).items()}


class NavigationShard(SourceShard):
    """Reuse classic benchmark transport/restore; replace only source call paths."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (self.shard_size, self.horizon, 2), np.float32)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        torch.set_num_threads(1)
        if self.ref is None:
            import_source(Path(self.source_root))
            self.ref = source(self.domain, self.shard_size)
        self.ref.reset(17 + self.start)
        return {k: np.zeros(space.shape, dtype=space.dtype) for k, space in self.observation_space.items()}, {}

    def step(self, action):
        return step_arrays(self.ref, action, self.frame_indices), 0.0, False, False, {}


class ParallelNavigationSource(ParallelSource):
    def __init__(self, domain, source_root, batch, workers, horizon, frame_indices):
        if workers < 1 or batch < workers or batch % workers:
            raise ValueError("Positive equal source shards required")
        self.batch, self.workers, self.horizon = batch, workers, horizon
        import_source(Path(source_root))
        prototype = source(domain, 1)
        try:
            sample = step_arrays(prototype, np.zeros((1, horizon, 2), np.float32), frame_indices)
            layout = {key: (value.shape[1:], value.dtype.str) for key, value in sample.items()}
            self.runtime_files = {}
            for cls in (type(prototype.envs[0]), gym.vector.AsyncVectorEnv):
                path = Path(inspect.getsourcefile(cls))
                self.runtime_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        finally:
            prototype.close()
        size = batch // workers
        factories = [
            functools.partial(
                NavigationShard, domain, str(source_root), size, i * size, horizon, frame_indices, layout
            )
            for i in range(workers)
        ]
        self.vector = gym.vector.AsyncVectorEnv(
            factories, shared_memory=True, copy=True, context="spawn", autoreset_mode=gym.vector.AutoresetMode.DISABLED
        )
        try:
            self.vector.reset()
        except BaseException:
            self.vector.close(terminate=True)
            raise

    def step(self, actions, device):
        shaped = actions.reshape(self.workers, self.batch // self.workers, self.horizon, 2)
        observations, _, _, _, _ = self.vector.step(shaped)
        return {
            key: torch.as_tensor(value.reshape(self.batch, *value.shape[2:]), device=device)
            for key, value in observations.items()
        }
