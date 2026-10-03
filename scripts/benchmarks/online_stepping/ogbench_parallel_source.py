"""Benchmark-only persistent OGBench source workers with shared task columns.

No rendering yet. Within each chunk, finished lanes stop advancing and task
padding is zero with an explicit validity column. A new call retains ordinary
source post-terminal behavior. Snapshot/restore RPCs are untimed preparation.
"""

import functools
import hashlib
import inspect
import os
import sys
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from scripts.benchmarks.online_stepping.classic_parallel_source import ParallelSource, SourceShard
from scripts.benchmarks.online_stepping.ogbench_source_prerequisite import import_source, ownership_module
from tests.helpers.reference.ogbench_maze_semantics import OGBMazeReference


class OGBSource:
    def __init__(self, family, root, batch, seed=17):
        if family not in ("Maze", "Cube", "Scene") or batch < 1:
            raise ValueError("Expected Maze/Cube/Scene and a positive batch")
        root = Path(root).resolve()
        if "stable_worldmodel" not in sys.modules:
            import_source(root)
        else:
            assert list(sys.modules["stable_worldmodel"].__path__) == [str(root / "stable_worldmodel")]
        self.family, self.batch = family, batch
        self.module = ownership_module()
        self.reference = None
        self.envs = []
        try:
            if family == "Maze":
                self.reference = OGBMazeReference(num_envs=batch, seed=seed)
                self.envs = self.reference.envs
                self.reference.reset(seed)
            else:
                for i in range(batch):
                    env = getattr(self.module, "Owned" + family)(entropy_seed=seed + i)
                    self.envs.append(env)
                    env.reset(seed=seed + i)
        except BaseException:
            self.close()
            raise

    def snapshot(self):
        if self.family == "Maze":
            return {key: value.numpy().copy() for key, value in self.reference.get_state().items(True, True)}
        return [self.module.capture(env) for env in self.envs]

    def restore(self, state):
        if self.family == "Maze":
            self.reference.set_state(TensorDict({key: torch.from_numpy(value) for key, value in state.items()}, [self.batch]))
        else:
            for env, row in zip(self.envs, state, strict=True):
                self.module.restore(env, row)

    def step(self, actions):
        actions = np.asarray(actions)
        dim = 2 if self.family == "Maze" else 5
        if (actions.ndim != 3 or actions.shape[0] != self.batch or actions.shape[1] < 1
                or actions.shape[2] != dim or actions.dtype not in (np.float32, np.float64)
                or not np.isfinite(actions).all()):
            raise ValueError("Require finite float32/float64 [B,T,A] source actions")
        columns, validity = [], []
        active = np.ones(self.batch, dtype=bool)
        previous = [None] * self.batch
        for t in range(actions.shape[1]):
            validity.append(active.copy())
            if self.family == "Maze":
                rows = [self.reference._in_legacy_rng(i, env.step, actions[i, t]) if active[i] else previous[i]
                        for i, env in enumerate(self.envs)]
            else:
                rows = [env.step(actions[i, t]) if active[i] else previous[i] for i, env in enumerate(self.envs)]
            previous = rows
            tasks = {"reward": np.asarray([r[1] for r in rows], np.float64),
                     "terminated": np.asarray([r[2] for r in rows], bool),
                     "truncated": np.asarray([r[3] for r in rows], bool),
                     "success": np.asarray([r[4]["success"] for r in rows],
                                           np.float64 if self.family == "Maze" else bool)}
            if self.family == "Maze":
                tasks["observation"] = np.stack([r[0] for r in rows])
                for key in ("xy", "prev_qpos", "prev_qvel", "qpos", "qvel"):
                    tasks["info." + key] = np.stack([r[4][key] for r in rows])
            done = tasks["terminated"] | tasks["truncated"]
            for values in tasks.values():
                values[~active] = 0
            columns.append(tasks)
            active &= ~done
        result = {key: np.stack([row[key] for row in columns], axis=1) for key in columns[0]}
        result["valid"] = np.stack(validity, axis=1)
        return result

    def close(self):
        for env in self.envs:
            env.close()
        self.envs = []
        self.reference = None


class OGBSourceShard(SourceShard):
    def __init__(self, *args, action_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Box(-np.inf, np.inf,
            (self.shard_size, self.horizon, action_dim), np.float64)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        torch.set_num_threads(1)
        if self.ref is not None:
            self.ref.close()
        self.ref = OGBSource(self.domain, self.source_root, self.shard_size, seed=17 + self.start)
        return {key: np.zeros(space.shape, dtype=space.dtype)
                for key, space in self.observation_space.items()}, {}

    def step(self, actions):
        return self.ref.step(actions), 0., False, False, {}

    def snapshot(self):
        return self.ref.snapshot()

    def inspection(self):
        return dict(pid=os.getpid(), torch_threads=torch.get_num_threads(),
                    start=self.start, size=self.shard_size,
                    renderers_absent=all((env.env.custom_renderer if self.domain == "Maze" else env._renderer)
                                         is None for env in self.ref.envs))

    def restore(self, state):
        if self.domain == "Maze":
            state = {key: value[self.start:self.start + self.shard_size] for key, value in state.items()}
        else:
            state = state[self.start:self.start + self.shard_size]
        self.ref.restore(state)


class ParallelOGBSource(ParallelSource):
    def __init__(self, family, root, batch, workers, horizon):
        if workers < 1 or batch < workers or batch % workers or horizon < 1:
            raise ValueError("Require positive equal source shards and horizon")
        self.family, self.batch, self.workers, self.horizon = family, batch, workers, horizon
        self.action_dim = 2 if family == "Maze" else 5
        prototype = OGBSource(family, root, 1)
        try:
            sample = prototype.step(np.zeros((1, horizon, self.action_dim), np.float64))
            layout = {key: (value.shape[1:], value.dtype.str) for key, value in sample.items()}
            classes = [*type(prototype.envs[0]).__mro__, gym.vector.AsyncVectorEnv]
            if family == "Maze":
                classes += list(type(prototype.envs[0].env).__mro__)
            self.runtime_files = {}
            for cls in classes:
                for name in ("__init__", "step", "reset", "render", "close"):
                    fn = cls.__dict__.get(name)
                    if inspect.isfunction(fn):
                        path = Path(inspect.getsourcefile(fn))
                        self.runtime_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        finally:
            prototype.close()
        size = batch // workers
        factories = [functools.partial(OGBSourceShard, family, str(root), size, i * size,
                                      horizon, (), layout, action_dim=self.action_dim) for i in range(workers)]
        self.vector = gym.vector.AsyncVectorEnv(factories, shared_memory=True, copy=True,
                                               context="spawn", autoreset_mode=gym.vector.AutoresetMode.DISABLED)
        try:
            self.vector.reset()
        except BaseException:
            self.vector.close(terminate=True)
            raise

    def step(self, actions, device="cpu"):
        actions = np.asarray(actions)
        if actions.shape != (self.batch, self.horizon, self.action_dim):
            raise ValueError("Source action shape differs from the configured batch/horizon")
        shaped = actions.reshape(self.workers, self.batch // self.workers, self.horizon, self.action_dim)
        observations, _, _, _, _ = self.vector.step(shaped)
        return {key: torch.as_tensor(value.reshape(self.batch, *value.shape[2:]), device=device)
                for key, value in observations.items()}

    def snapshot(self):
        shards = self.vector.call("snapshot")
        if self.family == "Maze":
            return {key: np.concatenate([row[key] for row in shards]) for key in shards[0]}
        return [row for shard in shards for row in shard]

    def restore(self, state):
        self.vector.call("restore", state)
