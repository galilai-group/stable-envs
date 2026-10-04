"""Direct pinned DMC wrappers with benchmark-only shared-memory source shards.

Source physics stays on CPU. Source render() supports software or hardware GL;
the selected GL context must be verified separately for each benchmark process.
The existing independent oracles provide only untimed complete-State handling.
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

from scripts.benchmarks.online_stepping.classic_parallel_source import ParallelSource, SourceShard
from tests.helpers.reference.dmc_semantics import DMCReference

NAMES = dict(zip(
    ("acrobot", "pendulum", "cartpole", "reacher", "cheetah", "hopper", "walker",
     "ball_in_cup", "finger", "manipulator", "quadruped", "humanoid"),
    ("DMCAcrobotBalance", "DMCPendulumSwingup", "DMCCartpoleSwingup", "DMCReacherHard",
     "DMCCheetahRun", "DMCHopperHop", "DMCWalkerWalk", "DMCBallInCupCatch", "DMCFingerTurnHard",
     "DMCManipulatorBringBall", "DMCQuadrupedWalk", "DMCHumanoidWalk"), strict=True))


def import_source(root):
    for name, sub in (("stable_worldmodel", ""), ("stable_worldmodel.envs", "envs"),
                      ("stable_worldmodel.envs.dmcontrol", "envs/dmcontrol"),
                      ("stable_worldmodel.envs.dmcontrol.custom_tasks", "envs/dmcontrol/custom_tasks")):
        if name not in sys.modules:
            package = types.ModuleType(name)
            package.__path__ = [str(root / "stable_worldmodel" / sub)]
            sys.modules[name] = package
    importlib.import_module("stable_worldmodel.utils")
    sys.modules["stable_worldmodel"].spaces = importlib.import_module("stable_worldmodel.spaces")


def source(domain, batch):
    ref = DMCReference(domain, num_envs=batch)
    ref.reset(17)
    return ref


def step_arrays(ref, actions, frame_indices):
    outputs, frames = [], []
    for t in range(actions.shape[1]):
        rows = [wrapper.step(actions[i, t]) for i, wrapper in enumerate(ref.envs)]
        # Pinned DMC wrappers deliberately keep public termination false even
        # across internal dm_control reset-next-step boundaries.
        if any(row[2] or row[3] for row in rows):
            raise AssertionError("DMC source lifecycle differs from the accepted wrapper contract")
        tasks = {}
        for key, index, dtype in (("observation", 0, None), ("reward", 1, np.float64),
                                  ("terminated", 2, bool), ("truncated", 3, bool)):
            values = np.stack([row[index] for row in rows])
            tasks[key] = values.astype(dtype) if dtype else values
        tasks["score"] = np.asarray([row[4]["score"] for row in rows], np.float64)
        # The first four accepted APIs expose score but not numeric info.
        # Source wrappers still perform their ordinary complete info computation.
        if ref.domain not in ("acrobot", "pendulum", "cartpole", "reacher"):
            tasks["success"] = np.asarray([row[4]["success"] for row in rows], np.float64)
            for key in rows[0][4]:
                if key not in ("env_name", "score", "success"):
                    tasks[f"info.{key}"] = np.stack([row[4][key] for row in rows])
        outputs.append(tasks)
        if t in frame_indices:
            frames.append(np.stack([wrapper.render(width=64, height=64) for wrapper in ref.envs]))
    result = {key: np.stack([row[key] for row in outputs], axis=1) for key in outputs[0]}
    if frames:
        result["frames"] = np.stack(frames, axis=1)
    return result


def source_step(ref, actions, indices, device):
    return {k: torch.as_tensor(v, device=device) for k, v in step_arrays(ref, actions, indices).items()}


def renderer_info(ref):
    """Read GL identity on dm_control's actual context-owning render executor."""
    from OpenGL import GL

    ref.envs[0].render(width=64, height=64)
    with ref.envs[0].env.physics.contexts.gl.make_current() as context:
        vendor = context.call(GL.glGetString, GL.GL_VENDOR).decode()
        renderer = context.call(GL.glGetString, GL.GL_RENDERER).decode()
    return dict(vendor=vendor, renderer=renderer,
                software=any(name in renderer.lower() for name in ("llvmpipe", "softpipe", "swrast")))


class DMCSourceShard(SourceShard):
    def __init__(self, *args, action_dim, **kwargs):
        super().__init__(*args, **kwargs)
        self.action_space = gym.spaces.Box(-np.inf, np.inf,
            (self.shard_size, self.horizon, action_dim), np.float64)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        torch.set_num_threads(1)
        if self.ref is None:
            import_source(Path(self.source_root))
            self.ref = source(self.domain, self.shard_size)
        self.ref.reset(17 + self.start)
        return {key: np.zeros(space.shape, dtype=space.dtype)
                for key, space in self.observation_space.items()}, {}

    def step(self, action):
        return step_arrays(self.ref, action, self.frame_indices), 0.0, False, False, {}

    def renderer_info(self):
        return renderer_info(self.ref)


class ParallelDMCSource(ParallelSource):
    def __init__(self, domain, source_root, batch, workers, horizon, frame_indices):
        if workers < 1 or batch < workers or batch % workers:
            raise ValueError("Positive equal source shards required")
        self.batch, self.workers, self.horizon = batch, workers, horizon
        import_source(Path(source_root))
        prototype = source(domain, 1)
        try:
            self.action_dim = prototype.envs[0].action_space.shape[0]
            sample = step_arrays(prototype, np.zeros((1, horizon, self.action_dim), np.float64), frame_indices)
            layout = {key: (value.shape[1:], value.dtype.str) for key, value in sample.items()}
            self.runtime_files = {}
            for cls in (type(prototype.envs[0]), type(prototype.envs[0]).__bases__[0],
                        type(prototype.envs[0].env._env), gym.vector.AsyncVectorEnv):
                path = Path(inspect.getsourcefile(cls))
                self.runtime_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        finally:
            prototype.close()
        size = batch // workers
        factories = [functools.partial(DMCSourceShard, domain, str(source_root), size, i * size,
                                       horizon, frame_indices, layout, action_dim=self.action_dim)
                     for i in range(workers)]
        self.vector = gym.vector.AsyncVectorEnv(factories, shared_memory=True, copy=True,
                                               context="spawn", autoreset_mode=gym.vector.AutoresetMode.DISABLED)
        try:
            self.vector.reset()
        except BaseException:
            self.vector.close(terminate=True)
            raise

    def step(self, actions, device):
        shaped = actions.reshape(self.workers, self.batch // self.workers, self.horizon, self.action_dim)
        observations, _, _, _, _ = self.vector.step(shaped)
        return {key: torch.as_tensor(value.reshape(self.batch, *value.shape[2:]), device=device)
                for key, value in observations.items()}

    def renderer_info(self):
        return self.vector.call("renderer_info")
