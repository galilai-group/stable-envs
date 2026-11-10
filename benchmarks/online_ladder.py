"""Time playing a stored set of episodes (seeded starts, fixed action sequences) through source and stable-envs pipelines.

Every arm counts its lane-steps once per complete episode (the seeded reset and all its steps), so each timed window
holds whole episodes and the reset cost is always part of the rate.

Arms:
  src-pipelined-render  source envs in ``--procs`` ProcessAdapter workers, the transport stable-envs uses; each worker
                        steps its envs one action at a time and then draws their frames; the parent submits step t+1
                        before taking step t. The GL backend (EGL or software) selects the device; ``--src-label`` names it.
  src-pipelined-step    as above with rendering off.
  snv-step              stable-envs physics only on CPU, ``T`` actions per call.
  snv-step-gpu          as snv-step with device="cuda"; physics moves to the GPU only where the class is GPU-native.
  snv-render-cpu               stable-envs physics and frames on CPU, ``T`` actions per call; stepping and rendering overlap.
  snv-render-gpu               as snv-render-cpu with frames on CUDA.
  src-single-render     one source env in this process: step, then draw, one action at a time.
  src-sync-render       ``--procs`` source envs in a Gymnasium ``SyncVectorEnv`` in this process; frames by ``call("render")``.
  src-async-render      as src-sync-render in an ``AsyncVectorEnv`` with one worker process per env.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import multiprocessing as mp
import os
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from benchmarks.offline_sweep import _cycle, _host, _timed

#: Constructor keyword arguments of source gym ids that need them.
SOURCE_KWARGS = {"swm/PushT-v1": {"resolution": 224}, "swm/PushT-Discrete-v1": {"resolution": 224}}
#: Constructor arguments that set the frame size of source ids that otherwise draw at their own default
#: (PyFlyt 480 px, OGBench Scene 200 px).
SIZED_SOURCES = {
    "swm/PFRocketLanding-v0": lambda size: {"render_resolution": (size, size)},
}
#: Classes timed against OGBench's own pixel task, by OGBench constructor and env_type.
OGB_PIXEL_SOURCES = {
    "OGBCube": ("ogbench/visual-cube-single-v0", "cube_env", "CubeEnv", "single"),
    "OGBScene": ("ogbench/visual-scene-v0", "scene_env", "SceneEnv", "scene"),
}


#: Physics steps per action of every DM Control class, which the dm_control source repeats to do the same work.
DMC_ACTION_REPEAT = 2
#: ``reference_id`` prefixes timed against the original library rather than a stable-worldmodel id.
ORIGINAL_SOURCES = ("dm_control/", "Fetch")


def source_id(name: str) -> str:
    """The source a class is timed against.

    DM Control and Fetch classes use their original library directly. Other classes use their SWM gym id, else
    their ``reference_id`` (ogbench/...).
    """
    import stable_envs as snv
    from tests.helpers.reference.crosswalk import swm_ids_for_class

    if name in OGB_PIXEL_SOURCES:
        return OGB_PIXEL_SOURCES[name][0]
    ref = snv.registry.REGISTRY[name].reference_id
    # The original library, where stable-worldmodel's wrapper changes the episode (Fetch samples colours at reset).
    if ref and ref.startswith(ORIGINAL_SOURCES):
        return ref
    ids = swm_ids_for_class(name)
    if ids:
        return ids[0]
    if not ref:
        raise ValueError(f"{name}: no SWM id and no reference_id")
    return ref


class _DMCSource:
    """dm_control suite task with the gym calls the players use; frames from the class camera."""

    def __init__(self, domain, task, camera, size):
        from dm_control import suite

        self.env = suite.load(domain, task)
        self.camera, self.size = camera, size

    def reset(self, seed=None):
        if seed is not None:
            self.env.task.random.seed(seed)
        self.env.reset()

    def step(self, action):
        """Return whether the episode ended."""
        a = np.asarray(action, np.float64)
        for _ in range(DMC_ACTION_REPEAT):
            if self.env.step(a).last():
                return True
        return False

    def render(self):
        return self.env.physics.render(self.size, self.size, camera_id=self.camera)

    def close(self):
        self.env.close()


def _pybullet_egl(env):
    """Load PyBullet's EGL hardware renderer into the env's client; PyFlyt makes a new client on each reset."""
    import pkgutil

    client = env.unwrapped.env
    client.loadPlugin(pkgutil.get_loader("eglRenderer").get_filename(), "_eglRendererPlugin")


class _GymSource:
    """A gym env whose actions are converted to its own action space."""

    def __init__(self, env, pybullet_egl: bool = False):
        self.env = env
        self.pybullet_egl = pybullet_egl
        space = env.action_space
        self.discrete = hasattr(space, "n") and not hasattr(space, "nvec")
        self.shape, self.dtype = getattr(space, "shape", ()), getattr(space, "dtype", None)

    def reset(self, seed=None):
        self.env.reset(seed=seed)
        if self.pybullet_egl:
            _pybullet_egl(self.env)

    def step(self, action):
        a = np.asarray(action)
        a = int(a.reshape(-1)[0]) if self.discrete else a.reshape(self.shape).astype(self.dtype)
        out = self.env.step(a)
        return bool(out[2] or out[3])

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


class _OGBPixelSource(_GymSource):
    """OGBench's visual task env (ob_type="pixels", visualize_info=False) at ``size``.

    The pixel look is set when the model is built. After that the env returns state observations, so a step
    does not draw, and ``render`` draws the frame that the pixel task returns as its observation.
    """

    def __init__(self, name, size):
        import importlib

        _, module, cls, env_type = OGB_PIXEL_SOURCES[name]
        make = getattr(importlib.import_module(f"ogbench.manipspace.envs.{module}"), cls)
        env = make(env_type=env_type, ob_type="pixels", visualize_info=False, width=size, height=size)
        env.reset(seed=0)
        env.unwrapped._ob_type = "states"
        super().__init__(env)

    def render(self):
        return self.env.unwrapped.render(camera="front_pixels")


def _make_source(name: str, size: int, gpu: bool = False):
    """The source env of class ``name``; ``gpu`` draws with the source's GPU renderer (``reference_gpu_render``)."""
    import gymnasium as gym

    import stable_envs as snv

    if name in OGB_PIXEL_SOURCES:
        return _OGBPixelSource(name, size)
    sid = source_id(name)
    pybullet_egl = gpu and snv.registry.REGISTRY[name].reference_gpu_render == "pybullet-egl"
    if sid.startswith("dm_control/"):
        domain, task = sid.split("/", 1)[1].split("-", 1)
        return _DMCSource(domain, task, snv.registry.REGISTRY[name].CAMERA, size)
    if sid.startswith("ogbench/"):
        # The class's own reference: the OGBench env and the floor that SNV's frames are compared against.
        return _GymSource(snv.registry.REGISTRY[name].reference_env(size=size))
    if sid.startswith("Fetch"):
        import gymnasium_robotics

        gym.register_envs(gymnasium_robotics)
        # Gymnasium Robotics draws natively at the requested size.
        return _GymSource(gym.make(sid, render_mode="rgb_array", width=size, height=size))
    import stable_worldmodel  # noqa: F401  registers the swm/ ids

    kwargs = dict(SOURCE_KWARGS.get(sid, {}))
    if sid in SIZED_SOURCES:
        kwargs.update(SIZED_SOURCES[sid](size))
    return _GymSource(gym.make(sid, render_mode="rgb_array", **kwargs), pybullet_egl)


def cmd_dataset(args) -> None:
    import stable_envs as snv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.envs:
        env = snv.make(name, num_envs=1, device="cpu")
        spec = env.action_spec
        env.close()
        rng = np.random.default_rng(args.seed)
        shape = (args.episodes, args.length, *spec.shape[-1:])
        if getattr(spec, "discrete_n", None):
            actions = rng.integers(0, spec.discrete_n, size=shape).astype(np.int64)
        else:
            low = np.broadcast_to(np.asarray(spec.low, np.float32), spec.shape[-1:])
            high = np.broadcast_to(np.asarray(spec.high, np.float32), spec.shape[-1:])
            actions = rng.uniform(low, high, size=shape).astype(np.float32)
        seeds = np.arange(args.episodes, dtype=np.int64) + args.seed
        np.savez(out / f"{name}.npz", seeds=seeds, actions=actions, source=source_id(name))
        print(f"saved {name} actions {actions.shape} source {source_id(name)}", flush=True)


#: Steps whose frames the source render check compares with a one-env replay of the same episode.
CHECK_STEPS = (0, 1, 20, 100, 195)


def _digest(frame) -> str:
    return hashlib.sha1(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _replay_frames(name, size, seed, actions):
    """Every frame of one episode played by a single source env, and the first step that ends it.

    Each replay runs in a fresh process: some sources (OGBench, gymnasium-robotics) keep their GL
    context alive, and a second context in one process changes the frames it draws.
    """
    ctx = mp.get_context("spawn")
    with ctx.Pool(1) as pool:
        return pool.apply(_replay_here, (name, size, seed, actions))


def _replay_here(name, size, seed, actions):
    env = _make_source(name, size)
    try:
        env.reset(seed=seed)
        frames, end = [], None
        for t, a in enumerate(actions):
            done = env.step(a)
            frames.append(np.asarray(env.render()).copy())
            if done and end is None:
                end = t
        return np.stack(frames), end
    finally:
        env.close()


def _ours_frames(name, size, device, seed, actions):
    """Every frame of the same episode played by one stable-envs lane on ``device``, and the first step that ends it."""
    env = _make_ours(name, 1, size, device)
    try:
        env.reset(seed=seed)
        a = torch.as_tensor(actions[None], device=device)
        out = env.step(a, states=None, frames=slice(None))
        keys = [k for k in ("terminated", "truncated", "done") if k in out.keys()]
        ended = np.zeros(actions.shape[0], bool)
        for k in keys:
            ended |= out[k][0].reshape(actions.shape[0], -1).any(-1).cpu().numpy().astype(bool)
        hits = np.flatnonzero(ended)
        return out["frames"][0].cpu().numpy(), (int(hits[0]) if len(hits) else None)
    finally:
        env.close()


class SourceLanes:
    """Source envs as a local simulator adapter, so ``ProcessAdapter`` runs them like stable-envs lanes.

    State is only a step counter: the source envs keep their own state, and no arm restores it.
    Each worker draws its lanes' frames after stepping them (``render_live``).
    """

    process_methods: dict = {}
    process_attributes = ("frame_shape",)

    def __init__(self, name: str, size: int, render: bool, count: int, seed: int | None, gpu: bool = False):
        self.envs = [_make_source(name, size, gpu) for _ in range(count)]
        self.render = render
        self.count = count
        self.t = torch.zeros(count, 1, dtype=torch.int64)
        self.reset(seed)
        self.frame_shape = tuple(np.asarray(self.envs[0].render()).shape) if render else None

    @staticmethod
    def check_step(adapter, actions, active):
        return np.asarray(actions), active

    def probe_actions(self):
        return self._probe

    def reset(self, seed=None, *args):
        for i, env in enumerate(self.envs):
            env.reset(seed=None if seed is None else int(seed) + i)
        self.t.zero_()
        return self.get_state()

    def get_state(self):
        return TensorDict({"t": self.t.clone()}, [self.count])

    def set_state(self, state):
        raise NotImplementedError("source lanes are not restored")

    def step(self, actions, *, active=None, capture_state=True):
        for env, a in zip(self.envs, np.asarray(actions)):
            env.step(a)
        self.t += 1
        out = TensorDict({"t": self.t.clone()}, [self.count])
        if capture_state:
            out["state"] = self.get_state()
        return out

    def render_live(self):
        return np.stack([np.asarray(env.render()) for env in self.envs])

    def close(self):
        for env in self.envs:
            env.close()


def _source_factory(name, size, render, probe, count, seed, gpu=False):
    lanes = SourceLanes(name, size, render, count, seed, gpu)
    lanes._probe = probe
    return lanes


def run_source(args, name: str, render: bool) -> dict:
    """Play every stored episode through ``ProcessAdapter`` workers that own source envs; submit step t+1 before taking t."""
    from stable_envs._execution.process import ProcessAdapter

    data = np.load(Path(args.data) / f"{name}.npz")
    seeds, actions = data["seeds"][: args.episodes], data["actions"][: args.episodes]
    n, length = actions.shape[:2]
    # --source-steps times a fixed slice of each episode (its reset and first steps) where whole episodes take hours.
    if args.source_steps:
        length = min(length, args.source_steps)
    lanes = min(args.procs * args.source_lanes, n)
    cpus = sorted(os.sched_getaffinity(0))[: args.procs]
    factory = functools.partial(_source_factory, name, args.size, render, actions[:1, 0], gpu=args.src_label == "gpu")
    adapter = ProcessAdapter(factory, lanes, "cpu", int(seeds[0]), processes=args.procs, cpus=cpus)
    frame_shape = getattr(adapter, "frame_shape", None)

    def play(groups):
        for s in groups:
            adapter.reset(int(seeds[s]))
            adapter.submit_step(actions[s : s + lanes, 0], capture_state=False, capture_frame=render)
            for t in range(length):
                out = adapter.receive_step()
                if t + 1 < length:
                    adapter.submit_step(actions[s : s + lanes, t + 1], capture_state=False, capture_frame=render)
                if render:
                    out["frame"].numpy()
            yield lanes * length

    groups = list(range(0, n - lanes + 1, lanes))
    check_lanes = sorted({0, lanes // 2, lanes - 1})
    check_steps = tuple(t for t in CHECK_STEPS if t < length)
    got: dict = {}
    try:
        units = _cycle(lambda: play(groups))
        t = _timed(units, seconds=args.seconds, repeats=args.repeats)
        units.close()
        if render:
            # The workers' frames for a few lanes, spread across workers, at the check steps.
            adapter.reset(int(seeds[0]))
            for step in range(max(check_steps) + 1):
                adapter.submit_step(actions[:lanes, step], capture_state=False, capture_frame=True)
                frame = adapter.receive_step()["frame"].numpy()
                frame_shape = frame_shape or frame.shape[1:]
                if step in check_steps:
                    for lane in check_lanes:
                        got[lane, step] = _digest(frame[lane])
    finally:
        adapter.close()
    check = rollout = None
    if render:
        from stable_envs.testing import conformance as C

        bad, per_frame, shapes = [], [], set()
        # Every source replay runs before any stable-envs env exists in this process: OGBench keeps
        # its GL contexts alive, and a second context in one process corrupts the source's frames.
        theirs = {lane: _replay_frames(name, args.size, int(seeds[lane]), actions[lane]) for lane in check_lanes}
        # A second replay of the first lane: a source whose seeded reset is not reproducible cannot be checked.
        again = _replay_frames(name, args.size, int(seeds[check_lanes[0]]), actions[check_lanes[0]])[0]
        reproducible = again.shape == theirs[check_lanes[0]][0].shape and bool(
            (again == theirs[check_lanes[0]][0]).all()
        )
        for lane in check_lanes:
            bad += [f"{lane}/{step}" for step in check_steps if got[lane, step] != _digest(theirs[lane][0][step])]
        compared = []
        for lane in check_lanes:
            ours, our_end = _ours_frames(name, args.size, args.ours_device, int(seeds[lane]), actions[lane])
            source, source_end = theirs[lane]
            shapes.add((tuple(ours.shape[1:]), tuple(source.shape[1:])))
            # Stable-envs resets a lane when its episode ends and the source keeps stepping it: compare up to the end.
            stop = min(x for x in (our_end, source_end, len(source) - 1) if x is not None) + 1
            compared.append(stop)
            if ours.shape == source.shape:
                per_frame.append(C.frame_mse(ours[:stop], source[:stop]))
        # The rollout includes any physics divergence, not only rendering differences.
        rollout = {
            "episodes": len(check_lanes),
            "steps_compared": compared,
            "ours_device": args.ours_device,
            "shapes": sorted(map(str, shapes)),
        }
        if per_frame:
            rollout.update(C.mse_summary(np.concatenate(per_frame)))
        check = {
            "reproducible": reproducible,
            "lanes": check_lanes,
            "steps": list(check_steps),
            "frames": len(got),
            "mismatched": len(bad),
            "examples": bad[:5],
        }
    return {
        "source": source_id(name),
        "frame_shape": list(frame_shape) if frame_shape else None,
        "procs": args.procs,
        "lanes": lanes,
        "cpus": cpus,
        "actions_per_call": 1,
        "steps_per_episode": length,
        "steps_per_s": t["rate"],
        **t,
        # Without a reproducible source episode, neither the check nor the rollout MSE says anything.
        "rollout_mse": rollout if not check or check["reproducible"] else None,
        **(
            {"frame_check": check, "status": "invalid"}
            if check and check["reproducible"] and check["mismatched"]
            else {
                "frame_check": check,
                **(
                    {"unverified": "source seeded reset is not reproducible"}
                    if check and not check["reproducible"]
                    else {}
                ),
            }
        ),
    }


def _make_ours(name: str, lanes: int, size: int, device: str):
    """The class at ``size``, or at its native size when it declares one."""
    import stable_envs as snv

    return snv.make(name, num_envs=lanes, size=getattr(snv.registry.REGISTRY[name], "native_size", None) or size, device=device)


def run_ours(args, name: str, dev: str, render: bool, chunk: int) -> dict:
    data = np.load(Path(args.data) / f"{name}.npz")
    seeds, actions = data["seeds"][: args.episodes], data["actions"][: args.episodes]
    n, length = actions.shape[:2]
    if args.source_steps:
        length = min(length, args.source_steps)
    lanes = min(args.lanes, n)
    torch.set_num_threads(args.threads)
    env = _make_ours(name, lanes, args.size, dev)
    execution_mode = getattr(env, "execution_mode", None)
    sync = torch.cuda.synchronize if dev.startswith("cuda") else None
    acts = torch.as_tensor(actions, device=dev)
    frames = slice(None) if render else None

    def play(groups):
        for s in groups:
            env.reset(seed=int(seeds[s]))
            for t0 in range(0, length, chunk):
                env.step(acts[s : s + lanes, t0 : min(t0 + chunk, length)], states=None, frames=frames)
            yield lanes * length

    groups = list(range(0, n - lanes + 1, lanes))
    try:
        shape = None
        if render:
            env.reset(seed=int(seeds[0]))
            shape = list(env.step(acts[:lanes, :1], states=None, frames=slice(None))["frames"].shape[2:])
        t = _timed(_cycle(lambda: play(groups)), seconds=args.seconds, repeats=args.repeats, sync=sync)
    finally:
        env.close()
    return {
        "lanes": lanes,
        "device": dev,
        "execution_mode": execution_mode,
        "actions_per_call": chunk,
        "steps_per_episode": length,
        "threads": args.threads,
        "frame_shape": shape,
        "steps_per_s": t["rate"],
        **t,
    }


def _gym_lane(name, size, action_dim, threads=None):
    """A source env as a ``gymnasium.Env`` for the vector-env arms: observations are dropped, frames come from render.

    ``threads`` sets torch's thread count in the process that builds the env (one per AsyncVectorEnv worker).
    """
    import gymnasium as gym

    if threads is not None:
        torch.set_num_threads(threads)

    class GymLane(gym.Env):
        metadata = {"render_modes": ["rgb_array"]}
        render_mode = "rgb_array"

        def __init__(self):
            self.src = _make_source(name, size)
            self.action_space = gym.spaces.Box(-np.inf, np.inf, (action_dim,), np.float64)
            self.observation_space = gym.spaces.Box(0.0, 0.0, (1,), np.float32)

        def reset(self, *, seed=None, options=None):
            self.src.reset(seed=seed)
            return np.zeros(1, np.float32), {}

        def step(self, action):
            done = self.src.step(action)
            return np.zeros(1, np.float32), 0.0, False, done, {}

        def render(self):
            return self.src.render()

        def close(self):
            self.src.close()

    return GymLane()


def run_source_single(args, name: str) -> dict:
    """Play the stored episodes through one source env in this process, drawing a frame after every step."""
    data = np.load(Path(args.data) / f"{name}.npz")
    seeds, actions = data["seeds"][: args.episodes], data["actions"][: args.episodes]
    env = _make_source(name, args.size)

    def play():
        for s in range(len(seeds)):
            env.reset(seed=int(seeds[s]))
            for a in actions[s]:
                env.step(a)
                np.asarray(env.render())
            yield len(actions[s])

    try:
        t = _timed(_cycle(play), seconds=args.seconds, repeats=args.repeats)
    finally:
        env.close()
    return {"source": source_id(name), "envs": 1, "procs": 1, "steps_per_s": t["rate"], **t}


def run_source_vector(args, name: str, asynchronous: bool) -> dict:
    """Play the stored episodes through ``--procs`` source envs in a Gymnasium vector env; one frame per env per step."""
    import gymnasium as gym

    data = np.load(Path(args.data) / f"{name}.npz")
    seeds, actions = data["seeds"][: args.episodes], data["actions"][: args.episodes]
    n, length = actions.shape[:2]
    k = min(args.procs, n)
    # One torch thread per worker: k workers on k cores. Sources that draw with torch on CPU (swm TwoRoom) otherwise
    # start k threads in every worker.
    threads = 1 if asynchronous else None
    fns = [functools.partial(_gym_lane, name, args.size, actions.shape[-1], threads) for _ in range(k)]
    if asynchronous:
        venv = gym.vector.AsyncVectorEnv(fns, shared_memory=False, context="spawn")
    else:
        venv = gym.vector.SyncVectorEnv(fns)

    def play(groups):
        for s in groups:
            venv.reset(seed=[int(x) for x in seeds[s : s + k]])
            for t in range(length):
                venv.step(actions[s : s + k, t].astype(np.float64))
                frames = venv.call("render")
                np.stack([np.asarray(f) for f in frames])
            yield k * length

    groups = list(range(0, n - k + 1, k))
    try:
        t = _timed(_cycle(lambda: play(groups)), seconds=args.seconds, repeats=args.repeats)
    finally:
        venv.close()
    return {
        "source": source_id(name),
        "vector": "async" if asynchronous else "sync",
        "envs": k,
        "procs": k if asynchronous else 1,
        "worker_threads": threads,
        "steps_per_s": t["rate"],
        **t,
    }


ARMS = (
    "src-pipelined-render",
    "src-pipelined-step",
    "snv-step",
    "snv-step-gpu",
    "snv-render-cpu",
    "snv-render-gpu",
    "src-single-render",
    "src-sync-render",
    "src-async-render",
)
#: Arms that take one action per call; the others run once per ``--chunks`` value.
SINGLE_CALL = ("src-",)
#: Stable-envs arms: (device, frames). snv-step-gpu steps on CUDA, which only moves physics for GPU-native classes.
OURS = {
    "snv-step": ("cpu", False),
    "snv-step-gpu": ("cuda", False),
    "snv-render-cpu": ("cpu", True),
    "snv-render-gpu": ("cuda", True),
}


def cmd_run(args) -> None:
    res = {"host": _host(), "env": args.env, "arms": {}, "budget_s": args.budget, "repeats_max": args.repeats}
    for arm in args.arms:
        chunks = [1] if arm.startswith(SINGLE_CALL) else args.chunks
        for chunk in chunks:
            key = f"{arm}-{args.src_label}" if arm == "src-pipelined-render" else arm
            key = key if arm.startswith(SINGLE_CALL) else f"{key}-T{chunk}"
            try:
                if arm == "src-pipelined-render":
                    r = run_source(args, args.env, True)
                elif arm == "src-pipelined-step":
                    r = run_source(args, args.env, False)
                elif arm == "src-single-render":
                    r = run_source_single(args, args.env)
                elif arm in ("src-sync-render", "src-async-render"):
                    r = run_source_vector(args, args.env, arm == "src-async-render")
                else:
                    r = run_ours(args, args.env, *OURS[arm], chunk)
                r.setdefault("status", "ok")
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                r = {"status": "oom" if "out of memory" in msg.lower() else "error", "error": msg[:600]}
            print(f"{args.env} {key} {r['status']} {r.get('steps_per_s', 0):.0f} steps/s", flush=True)
            res["arms"][key] = r
    Path(args.output).write_text(json.dumps(res, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--envs", nargs="+", required=True)
    d.add_argument("--out", required=True)
    d.add_argument("--episodes", type=int, default=1024)
    d.add_argument("--length", type=int, default=196)
    d.add_argument("--seed", type=int, default=20261022)
    r = sub.add_parser("run")
    r.add_argument("--env", required=True)
    r.add_argument("--arms", nargs="+", default=list(ARMS), choices=ARMS)
    r.add_argument(
        "--src-label", default=os.environ.get("MUJOCO_GL", "gl"), help="device label of the source render arm"
    )
    r.add_argument("--chunks", type=int, nargs="+", default=[1, 16], help="actions per stable-envs step call (T)")
    r.add_argument(
        "--ours-device", default="cpu", help="device of the stable-envs lane the source rollout is compared with"
    )
    r.add_argument("--data", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--procs", type=int, default=8)
    r.add_argument("--source-lanes", type=int, default=8)
    r.add_argument("--lanes", type=int, default=1024)
    r.add_argument("--episodes", type=int, default=None, help="play only the first N stored episodes")
    r.add_argument(
        "--source-steps", type=int, default=None, help="time only the first N steps of each episode, in every arm"
    )
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--size", type=int, default=224)
    r.add_argument("--repeats", type=int, default=10)
    r.add_argument("--budget", type=float, default=300.0)
    r.add_argument("--seconds", type=float, default=10.0, help="length of one timed window")
    args = p.parse_args()
    {"dataset": cmd_dataset, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
