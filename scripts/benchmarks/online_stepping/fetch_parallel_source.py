"""Benchmark-only persistent Fetch source owners with shared task columns.

Actual pinned registrations/factories perform physics and observation work.
Snapshot/restore is untimed preparation. Rendering is deliberately absent until
its separate source lifecycle and device gates pass; no production route changes.
"""

import copy
import functools
import hashlib
import importlib
import inspect
import os
import sys
import types
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
import torch

from scripts.benchmarks.online_stepping.classic_parallel_source import ParallelSource, SourceShard


def import_source(root):
    root = Path(root).resolve()
    name = 'stable_worldmodel'
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(root / name)]
        sys.modules[name] = module
    assert list(sys.modules[name].__path__) == [str(root / name)], 'Wrong loaded source root'
    sys.modules[name].utils = importlib.import_module(name + '.utils')
    sys.modules[name].spaces = importlib.import_module(name + '.spaces')
    module = importlib.import_module(name + '.envs')
    assert Path(module.__file__).resolve() == root / name / 'envs/__init__.py'


def limit(env):
    while not isinstance(env, gym.wrappers.TimeLimit):
        env = env.env
    return env


def restore_world(env, state):
    """Explicit source restoration from the accepted source-only experiments.

    Rebuild derived engine fields, then restore warmstart and controller caches.
    Controlled default-variation benchmark only; never used in timed stepping.
    """
    raw = env.unwrapped
    for field in ('qpos', 'qvel', 'qacc_warmstart', 'mocap_pos', 'mocap_quat', 'ctrl', 'qfrc_applied', 'xfrc_applied'):
        getattr(raw.data, field)[:] = state[field]
    raw.data.time = float(state['time'])
    raw.goal = state['goal'].copy()
    raw.initial_gripper_xpos = state['initial_gripper_xpos'].copy()
    limit(env)._elapsed_steps = int(state['elapsed'])
    limit(env)._max_episode_steps = int(state['max_episode_steps'])
    raw.np_random.bit_generator.state = copy.deepcopy(state['rng'])
    mujoco.mj_forward(raw.model, raw.data)
    raw.data.qacc_warmstart[:] = state['qacc_warmstart']
    gripper = raw.model.body('robot0:gripper_link').id
    raw.data.xpos[gripper] = state['controller_position']
    raw.data.xquat[gripper] = state['controller_quaternion']


class FetchSource:
    def __init__(self, registration, root, batch, seed=17):
        import_source(root)
        from scripts.benchmarks.online_stepping.fetch_source_render_lifecycle import IDS, snapshot

        if registration not in IDS or type(batch) is not int or batch < 1:
            raise ValueError('Require a supported Fetch registration and positive batch')
        self.registration, self.batch, self._snapshot = registration, batch, snapshot
        self.envs = []
        try:
            for lane in range(batch):
                env = gym.make(registration)
                self.envs.append(env)
                env.reset(seed=seed + lane, options={'variation': []})
        except BaseException:
            self.close()
            raise

    def reset_episode(self, seed=None):
        for lane, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else seed + lane, options={'variation': []})

    def snapshot(self):
        return [self._snapshot(env) for env in self.envs]

    def restore(self, states):
        for env, state in zip(self.envs, states, strict=True):
            restore_world(env, state)

    def step(self, actions):
        actions = np.asarray(actions)
        if (actions.ndim != 3 or actions.shape[0] != self.batch or actions.shape[1] < 1 or actions.shape[2] != 4
                or actions.dtype not in (np.float32, np.float64) or not np.isfinite(actions).all()):
            raise ValueError('Require finite float32/64 [B,T,4] actions')
        active = np.ones(self.batch, bool)
        previous = [None] * self.batch
        columns, valid = [], []
        for t in range(actions.shape[1]):
            valid.append(active.copy())
            rows = [env.step(actions[lane, t]) if active[lane] else previous[lane]
                    for lane, env in enumerate(self.envs)]
            previous = rows
            tasks = {key: np.asarray([row[index] for row in rows]) for key, index in
                     (('reward', 1), ('terminated', 2), ('truncated', 3))}
            tasks['success'] = np.asarray([row[4]['is_success'] for row in rows])
            done = tasks['terminated'] | tasks['truncated']
            for values in tasks.values():
                values[~active] = 0
            columns.append(tasks)
            active &= ~done
        result = {key: np.stack([row[key] for row in columns], axis=1) for key in columns[0]}
        result['valid'] = np.stack(valid, axis=1)
        return result

    def close(self):
        for env in self.envs:
            env.close()
        self.envs = []


class FetchSourceShard(SourceShard):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (self.shard_size, self.horizon, 4), np.float64)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        torch.set_num_threads(1)
        if self.ref is not None:
            self.ref.close()
        self.ref = FetchSource(self.domain, self.source_root, self.shard_size, seed=17 + self.start)
        return {key: np.zeros(space.shape, space.dtype) for key, space in self.observation_space.items()}, {}

    def step(self, actions):
        return self.ref.step(actions), 0., False, False, {}

    def snapshot(self):
        return self.ref.snapshot()

    def restore(self, states):
        self.ref.restore(states[self.start:self.start + self.shard_size])

    def reset_episode(self, seed=None):
        self.ref.reset_episode(None if seed is None else seed + self.start)

    def inspection(self):
        return dict(pid=os.getpid(), start=self.start, size=self.shard_size,
                    torch_threads=torch.get_num_threads(),
                    viewers_absent=all(not env.unwrapped.mujoco_renderer._viewers for env in self.ref.envs))


class ParallelFetchSource(ParallelSource):
    def __init__(self, registration, root, batch, workers, horizon):
        if workers < 1 or batch < workers or batch % workers or horizon < 1:
            raise ValueError('Require positive equal source shards and horizon')
        self.batch, self.workers, self.horizon = batch, workers, horizon
        prototype = FetchSource(registration, root, 1)
        try:
            sample = prototype.step(np.zeros((1, horizon, 4), np.float64))
            layout = {key: (value.shape[1:], value.dtype.str) for key, value in sample.items()}
            env, classes = prototype.envs[0], [gym.vector.AsyncVectorEnv]
            while True:
                classes.extend(type(env).__mro__)
                if env is env.unwrapped:
                    break
                env = env.env
            self.runtime_files = {}
            for cls in classes:
                for name in ('__init__', 'step', 'reset', 'render', 'close', '_render_callback', '_step_callback'):
                    method = cls.__dict__.get(name)
                    if inspect.isfunction(method):
                        path = Path(inspect.getsourcefile(method))
                        self.runtime_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        finally:
            prototype.close()
        size = batch // workers
        factories = [functools.partial(FetchSourceShard, registration, str(root), size, i * size, horizon, None, layout)
                     for i in range(workers)]
        self.vector = gym.vector.AsyncVectorEnv(factories, shared_memory=True, copy=True,
                                               context='spawn', autoreset_mode=gym.vector.AutoresetMode.DISABLED)
        try:
            self.vector.reset()
        except BaseException:
            self.vector.close(terminate=True)
            raise

    def step(self, actions, device='cpu'):
        actions = np.asarray(actions)
        if actions.shape != (self.batch, self.horizon, 4):
            raise ValueError('Actions differ from configured batch/horizon')
        shaped = actions.reshape(self.workers, self.batch // self.workers, self.horizon, 4)
        observations, _, _, _, _ = self.vector.step(shaped)
        return {key: torch.as_tensor(value.reshape(self.batch, *value.shape[2:]), device=device)
                for key, value in observations.items()}

    def snapshot(self):
        return [row for shard in self.vector.call('snapshot') for row in shard]

    def restore(self, states):
        self.vector.call('restore', states)

    def reset_episode(self, seed=None):
        self.vector.call('reset_episode', seed)
