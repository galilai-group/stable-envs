"""Benchmark-only sharding of pinned wrappers through Gym's shared-memory vector API.

Each process owns a scalar-source shard. Source images and numeric task outputs
travel in Gym's shared observation buffers, not pickled pipe payloads. Oracle
State export/restore is exposed only for untimed matched-state verification.
"""

import functools
import hashlib
import inspect
from pathlib import Path

import gymnasium as gym
import numpy as np
import torch
from tensordict import TensorDict

from benchmarks.online.classic_source_pipeline import import_source, source


def step_arrays(ref, actions, frame_indices):
    outputs, frames = [], []
    discrete = ref.domain in ("cartpole", "acrobot", "mountain_car")
    for t in range(actions.shape[1]):
        rows = [env.step(actions[i, t, 0].item() if discrete else actions[i, t]) for i, env in enumerate(ref.envs)]
        outputs.append(rows)
        if t in frame_indices:
            frames.append(np.stack([env.render() for env in ref.envs]))
    tasks = {}
    for field, index, dtype in (
        ("observation", 0, None),
        ("reward", 1, np.float64),
        ("terminated", 2, bool),
        ("truncated", 3, bool),
    ):
        values = np.stack([np.stack([r[index] for r in rows]) for rows in outputs], axis=1)
        tasks[field] = values.astype(dtype) if dtype else values
    tasks["info_state"] = np.stack([np.stack([r[4]["state"] for r in rows]) for rows in outputs], axis=1)
    if frames:
        tasks["frames"] = np.stack(frames, axis=1)
    return tasks


class SourceShard(gym.Env):
    metadata = {}

    def __init__(self, domain, source_root, shard_size, start, horizon, frame_indices, layout):
        self.domain, self.source_root, self.shard_size, self.start = domain, source_root, shard_size, start
        self.horizon, self.frame_indices = horizon, frame_indices
        self.ref = None
        spaces = {}
        for key, (shape, dtype) in layout.items():
            dt = np.dtype(dtype)
            low, high = (
                (0, 1) if dt == np.dtype(bool) else ((0, 255) if dt == np.dtype(np.uint8) else (-np.inf, np.inf))
            )
            spaces[key] = gym.spaces.Box(low, high, shape=(shard_size, *shape), dtype=dt)
        self.observation_space = gym.spaces.Dict(spaces)
        shape = (shard_size, horizon, 1)
        if domain in ("cartpole", "acrobot", "mountain_car"):
            self.action_space = gym.spaces.MultiDiscrete(np.full(shape, 2 if domain == "cartpole" else 3))
        else:
            self.action_space = gym.spaces.Box(-np.inf, np.inf, shape=shape, dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        if self.ref is None:
            import_source(Path(self.source_root))
            self.ref = source(self.domain, self.shard_size)
        self.ref.reset(17 + self.start)
        return {k: np.zeros(space.shape, dtype=space.dtype) for k, space in self.observation_space.items()}, {}

    def step(self, action):
        # Actual source lane termination stays in the shared task columns.
        # This benchmark container never invents auto-resets for its lanes.
        return step_arrays(self.ref, action, self.frame_indices), 0.0, False, False, {}

    def snapshot(self):
        return {key: value.numpy() for key, value in self.ref.get_state().items(True, True)}

    def restore(self, state):
        self.ref.set_state(
            TensorDict(
                {
                    key: torch.from_numpy(value[self.start : self.start + self.shard_size])
                    for key, value in state.items()
                },
                batch_size=[self.shard_size],
            )
        )

    def close(self):
        if self.ref is not None:
            self.ref.close()
            self.ref = None


class ParallelSource:
    def __init__(self, domain, source_root, batch, workers, horizon, frame_indices):
        if batch % workers or workers < 1:
            raise ValueError("Equal source shards require a positive worker count dividing batch")
        self.batch, self.workers, self.horizon = batch, workers, horizon
        import_source(Path(source_root))
        prototype = source(domain, 1)
        try:
            action = np.full(
                (1, horizon, 1),
                1 if domain == "acrobot" else 0,
                dtype=np.int64 if domain in ("cartpole", "acrobot", "mountain_car") else np.float32,
            )
            sample = step_arrays(prototype, action, frame_indices)
            layout = {key: (value.shape[1:], value.dtype.str) for key, value in sample.items()}
            self.runtime_files = {}
            for cls in (
                type(prototype.envs[0]),
                type(prototype.envs[0]).__bases__[0],
                type(prototype.envs[0].unwrapped),
                gym.vector.AsyncVectorEnv,
            ):
                path = Path(inspect.getsourcefile(cls))
                self.runtime_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        finally:
            prototype.close()
        size = batch // workers
        factories = [
            functools.partial(SourceShard, domain, str(source_root), size, i * size, horizon, frame_indices, layout)
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
        shaped = actions.reshape(self.workers, self.batch // self.workers, self.horizon, 1)
        observations, _, _, _, _ = self.vector.step(shaped)
        # copy=True above gives owned host arrays. CPU tensors keep that owned
        # storage; CUDA delivery transfers each complete output column once.
        return {
            key: torch.as_tensor(value.reshape(self.batch, *value.shape[2:]), device=device)
            for key, value in observations.items()
        }

    def snapshot(self):
        shards = self.vector.call("snapshot")
        return TensorDict(
            {key: torch.from_numpy(np.concatenate([shard[key] for shard in shards])) for key in shards[0]},
            batch_size=[self.batch],
        )

    def restore(self, state):
        self.vector.call("restore", {key: value.cpu().numpy() for key, value in state.items(True, True)})

    def close(self):
        self.vector.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
