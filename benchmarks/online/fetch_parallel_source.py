"""Benchmark-only persistent Fetch source owners with shared task columns.

Actual pinned registrations/factories perform physics and observation work.
Snapshot/restore is untimed preparation. Selected source frames retain original
internal raster dimensions and require independent Oscar acceptance. PickAndPlace
frame capture is refused because its source render mutates continuation State.
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

from benchmarks.online.classic_parallel_source import ParallelSource, SourceShard


def import_source(root):
    root = Path(root).resolve()
    name = "stable_worldmodel"
    if name not in sys.modules:
        module = types.ModuleType(name)
        module.__path__ = [str(root / name)]
        sys.modules[name] = module
    assert list(sys.modules[name].__path__) == [str(root / name)], "Wrong loaded source root"
    sys.modules[name].utils = importlib.import_module(name + ".utils")
    sys.modules[name].spaces = importlib.import_module(name + ".spaces")
    module = importlib.import_module(name + ".envs")
    assert Path(module.__file__).resolve() == root / name / "envs/__init__.py"


def frame_selection(registration, indices, horizon, render_size):
    if indices is None:
        return None
    indices = tuple(indices)
    if any(type(i) is not int or not 0 <= i < horizon for i in indices) or len(set(indices)) != len(indices):
        raise ValueError("Frame indices must be unique in-range integers")
    if type(render_size) is not int or render_size < 1:
        raise ValueError("Frame capture requires a configured positive output resolution")
    if indices and "PickAndPlace" in registration:
        raise ValueError(
            "PickAndPlace source rendering changes continuation State; rendered online protocol unresolved"
        )
    return indices


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
    for field in ("qpos", "qvel", "qacc_warmstart", "mocap_pos", "mocap_quat", "ctrl", "qfrc_applied", "xfrc_applied"):
        getattr(raw.data, field)[:] = state[field]
    raw.data.time = float(state["time"])
    raw.goal = state["goal"].copy()
    raw.initial_gripper_xpos = state["initial_gripper_xpos"].copy()
    limit(env)._elapsed_steps = int(state["elapsed"])
    limit(env)._max_episode_steps = int(state["max_episode_steps"])
    raw.np_random.bit_generator.state = copy.deepcopy(state["rng"])
    mujoco.mj_forward(raw.model, raw.data)
    raw.data.qacc_warmstart[:] = state["qacc_warmstart"]
    gripper = raw.model.body("robot0:gripper_link").id
    raw.data.xpos[gripper] = state["controller_position"]
    raw.data.xquat[gripper] = state["controller_quaternion"]


class FetchSource:
    def __init__(self, registration, root, batch, seed=17, *, render_size=None):
        import_source(root)
        from benchmarks.online.fetch_source_render_lifecycle import IDS, snapshot

        if registration not in IDS or type(batch) is not int or batch < 1:
            raise ValueError("Require a supported Fetch registration and positive batch")
        if render_size is not None and (type(render_size) is not int or render_size < 1):
            raise ValueError("Require a positive source output resolution")
        self.render_size = render_size
        self._audit_render, self.render_audit_count = False, 0
        self.registration, self.batch, self._snapshot = registration, batch, snapshot
        self.envs = []
        try:
            for lane in range(batch):
                kwargs = {} if render_size is None else dict(render_mode="rgb_array", resolution=render_size)
                # Preserve upstream width/height defaults. The pinned wrapper
                # performs its own resize, and that work belongs in timing.
                env = gym.make(registration, **kwargs)
                self.envs.append(env)
                env.reset(seed=seed + lane, options={"variation": []})
        except BaseException:
            self.close()
            raise

    def reset_episode(self, seed=None):
        for lane, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else seed + lane, options={"variation": []})

    def snapshot(self):
        return [self._snapshot(env) for env in self.envs]

    def restore(self, states):
        for env, state in zip(self.envs, states, strict=True):
            restore_world(env, state)

    def _render_lane(self, lane):
        frame_selection(self.registration, (0,), 1, self.render_size)
        auditing = getattr(self, "_audit_render", False)
        before = self._snapshot(self.envs[lane]) if auditing else None
        frame = np.asarray(self.envs[lane].render())
        if auditing:
            from benchmarks.online.fetch_source_render_lifecycle import differences

            delta = differences(before, self._snapshot(self.envs[lane]))
            assert not delta, ("Source rendering mutated continuation State", delta)
            self.render_audit_count += 1
        if frame.dtype != np.uint8 or frame.shape != (self.render_size, self.render_size, 3):
            raise ValueError("Unexpected source frame shape or dtype")
        return frame

    def renderer_info(self):
        from OpenGL import GL

        self._render_lane(0)
        raw = self.envs[0].unwrapped
        raw.mujoco_renderer._viewers["rgb_array"].make_context_current()
        vendor, renderer = GL.glGetString(GL.GL_VENDOR).decode(), GL.glGetString(GL.GL_RENDERER).decode()
        return dict(
            vendor=vendor,
            renderer=renderer,
            software=any(tag in renderer.lower() for tag in ("llvmpipe", "softpipe", "swrast")),
            internal_width=raw.width,
            internal_height=raw.height,
            output_size=self.render_size,
        )

    def step(self, actions, frame_indices=None):
        actions = np.asarray(actions)
        if (
            actions.ndim != 3
            or actions.shape[0] != self.batch
            or actions.shape[1] < 1
            or actions.shape[2] != 4
            or actions.dtype not in (np.float32, np.float64)
            or not np.isfinite(actions).all()
        ):
            raise ValueError("Require finite float32/64 [B,T,4] actions")
        frame_indices = frame_selection(self.registration, frame_indices, actions.shape[1], self.render_size)
        frame_times, saved_frames, cached_frames = set(frame_indices or ()), {}, [None] * self.batch
        active = np.ones(self.batch, bool)
        previous = [None] * self.batch
        columns, valid = [], []
        for t in range(actions.shape[1]):
            valid.append(active.copy())
            rows = [
                env.step(actions[lane, t]) if active[lane] else previous[lane] for lane, env in enumerate(self.envs)
            ]
            previous = rows
            if frame_times:
                for lane in np.flatnonzero(active):
                    cached_frames[lane] = None
            tasks = {
                key: np.asarray([row[index] for row in rows])
                for key, index in (("reward", 1), ("terminated", 2), ("truncated", 3))
            }
            tasks["success"] = np.asarray([row[4]["is_success"] for row in rows])
            done = tasks["terminated"] | tasks["truncated"]
            for values in tasks.values():
                values[~active] = 0
            columns.append(tasks)
            if t in frame_times:
                for lane in range(self.batch):
                    if cached_frames[lane] is None:
                        cached_frames[lane] = self._render_lane(lane)
                saved_frames[t] = np.stack(cached_frames)
            active &= ~done
        result = {key: np.stack([row[key] for row in columns], axis=1) for key in columns[0]}
        result["valid"] = np.stack(valid, axis=1)
        if frame_indices is not None:
            result["frames"] = (
                np.stack([saved_frames[t] for t in frame_indices], axis=1)
                if frame_indices
                else np.empty((self.batch, 0, self.render_size, self.render_size, 3), np.uint8)
            )
            result["frame_valid"] = result["valid"][:, list(frame_indices)]
        return result

    def close(self):
        for env in self.envs:
            env.close()
        self.envs = []


class FetchSourceShard(SourceShard):
    def __init__(self, *args, render_size=None, **kwargs):
        super().__init__(*args, **kwargs)
        self.render_size = render_size
        self.action_space = gym.spaces.Box(-np.inf, np.inf, (self.shard_size, self.horizon, 4), np.float64)

    def reset(self, *, seed=None, options=None):
        gym.Env.reset(self, seed=seed)
        torch.set_num_threads(1)
        if self.ref is not None:
            self.ref.close()
        self.ref = FetchSource(
            self.domain, self.source_root, self.shard_size, seed=17 + self.start, render_size=self.render_size
        )
        return {key: np.zeros(space.shape, space.dtype) for key, space in self.observation_space.items()}, {}

    def step(self, actions):
        return self.ref.step(actions, self.frame_indices), 0.0, False, False, {}

    def snapshot(self):
        return self.ref.snapshot()

    def restore(self, states):
        self.ref.restore(states[self.start : self.start + self.shard_size])

    def reset_episode(self, seed=None):
        self.ref.reset_episode(None if seed is None else seed + self.start)

    def enable_render_audit(self):
        self.ref._audit_render = True

    def renderer_info(self):
        return self.ref.renderer_info()

    def inspection(self):
        return dict(
            pid=os.getpid(),
            start=self.start,
            size=self.shard_size,
            torch_threads=torch.get_num_threads(),
            render_audit_count=self.ref.render_audit_count,
            viewers_absent=all(not env.unwrapped.mujoco_renderer._viewers for env in self.ref.envs),
        )


class ParallelFetchSource(ParallelSource):
    def __init__(self, registration, root, batch, workers, horizon, *, frame_indices=None, render_size=None):
        if workers < 1 or batch < workers or batch % workers or horizon < 1:
            raise ValueError("Require positive equal source shards and horizon")
        self.batch, self.workers, self.horizon = batch, workers, horizon
        frame_indices = frame_selection(registration, frame_indices, horizon, render_size)
        prototype = FetchSource(registration, root, 1, render_size=render_size)
        try:
            sample = prototype.step(np.zeros((1, horizon, 4), np.float64))
            layout = {key: (value.shape[1:], value.dtype.str) for key, value in sample.items()}
            if frame_indices is not None:
                layout["frames"] = ((len(frame_indices), render_size, render_size, 3), np.dtype(np.uint8).str)
                layout["frame_valid"] = ((len(frame_indices),), np.dtype(bool).str)
            env, classes = prototype.envs[0], [gym.vector.AsyncVectorEnv]
            while True:
                classes.extend(type(env).__mro__)
                if env is env.unwrapped:
                    break
                env = env.env
            self.runtime_files = {}
            for cls in classes:
                for name in ("__init__", "step", "reset", "render", "close", "_render_callback", "_step_callback"):
                    method = cls.__dict__.get(name)
                    if inspect.isfunction(method):
                        path = Path(inspect.getsourcefile(method))
                        self.runtime_files[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        finally:
            prototype.close()
        size = batch // workers
        factories = [
            functools.partial(
                FetchSourceShard,
                registration,
                str(root),
                size,
                i * size,
                horizon,
                frame_indices,
                layout,
                render_size=render_size,
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

    def step(self, actions, device="cpu"):
        actions = np.asarray(actions)
        if actions.shape != (self.batch, self.horizon, 4):
            raise ValueError("Actions differ from configured batch/horizon")
        shaped = actions.reshape(self.workers, self.batch // self.workers, self.horizon, 4)
        observations, _, _, _, _ = self.vector.step(shaped)
        return {
            key: torch.as_tensor(value.reshape(self.batch, *value.shape[2:]), device=device)
            for key, value in observations.items()
        }

    def snapshot(self):
        return [row for shard in self.vector.call("snapshot") for row in shard]

    def restore(self, states):
        self.vector.call("restore", states)

    def reset_episode(self, seed=None):
        self.vector.call("reset_episode", seed)

    def renderer_info(self):
        return self.vector.call("renderer_info")

    def enable_render_audit(self):
        self.vector.call("enable_render_audit")
