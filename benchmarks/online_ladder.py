"""Time playing a stored set of episodes (seeded starts, fixed action sequences) through source and stable-envs pipelines.

Arms:
  src-pipelined-render  source envs in ``--procs`` ProcessAdapter workers, the transport stable-envs uses; each worker
                        steps its envs one action at a time and then draws their frames; the parent submits step t+1
                        before taking step t. The GL backend (EGL or software) selects the device; ``--src-label`` names it.
  src-pipelined-step    as above with rendering off.
  snv-step                     stable-envs physics only, ``T`` actions per call.
  snv-render-cpu               stable-envs physics and frames on CPU, ``T`` actions per call; stepping and rendering overlap.
  snv-render-gpu               as snv-render-cpu with frames on CUDA.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import json
import os
import re
from pathlib import Path

import numpy as np
import torch
from tensordict import TensorDict

from benchmarks.offline_sweep import _host, _sweeps

#: Constructor keyword arguments of source gym ids that need them.
SOURCE_KWARGS = {"swm/PushT-v1": {"resolution": 224}, "swm/PushT-Discrete-v1": {"resolution": 224}}


def source_id(name: str) -> str:
    """The source a class is timed against: its SWM gym id, else its ``reference_id`` (dm_control/..., ogbench/...)."""
    import stable_envs as snv
    from tests.helpers.reference.crosswalk import swm_ids_for_class

    ids = swm_ids_for_class(name)
    if ids:
        return ids[0]
    ref = snv.registry.REGISTRY[name].reference_id
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
        self.env.step(np.asarray(action, np.float64))

    def render(self):
        return self.env.physics.render(self.size, self.size, camera_id=self.camera)

    def close(self):
        self.env.close()


class _GymSource:
    """A gym env whose actions are converted to its own action space."""

    def __init__(self, env):
        self.env = env
        space = env.action_space
        self.discrete = hasattr(space, "n") and not hasattr(space, "nvec")
        self.shape, self.dtype = getattr(space, "shape", ()), getattr(space, "dtype", None)

    def reset(self, seed=None):
        self.env.reset(seed=seed)

    def step(self, action):
        a = np.asarray(action)
        a = int(a.reshape(-1)[0]) if self.discrete else a.reshape(self.shape).astype(self.dtype)
        self.env.step(a)

    def render(self):
        return self.env.render()

    def close(self):
        self.env.close()


def _make_source(name: str, size: int):
    import gymnasium as gym

    sid = source_id(name)
    if sid.startswith("dm_control/"):
        import stable_envs as snv

        domain, task = sid.split("/", 1)[1].split("-", 1)
        return _DMCSource(domain, task, snv.registry.REGISTRY[name].CAMERA, size)
    if sid.startswith("ogbench/"):
        import ogbench  # noqa: F401  registers the ids

        maze = re.sub(r"(?<!^)(?=[A-Z])", "-", name).lower()
        return _GymSource(gym.make(f"{maze}-v0", render_mode="rgb_array", width=size, height=size))
    import stable_worldmodel  # noqa: F401  registers the swm/ ids

    return _GymSource(gym.make(sid, render_mode="rgb_array", **SOURCE_KWARGS.get(sid, {})))


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


def _replay_digests(name, size, seed, actions, steps) -> dict:
    """Frame digests of one episode played by a single source env in this process."""
    env = _make_source(name, size)
    try:
        env.reset(seed=seed)
        out = {}
        for t in range(max(steps) + 1):
            env.step(actions[t])
            if t in steps:
                out[t] = _digest(env.render())
        return out
    finally:
        env.close()


class SourceLanes:
    """Source envs as a local simulator adapter, so ``ProcessAdapter`` runs them like stable-envs lanes.

    State is only a step counter: the source envs keep their own state, and no arm restores it.
    Each worker draws its lanes' frames after stepping them (``render_live``).
    """

    process_methods: dict = {}
    process_attributes = ("frame_shape",)

    def __init__(self, name: str, size: int, render: bool, count: int, seed: int | None):
        self.envs = [_make_source(name, size) for _ in range(count)]
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


def _source_factory(name, size, render, probe, count, seed):
    lanes = SourceLanes(name, size, render, count, seed)
    lanes._probe = probe
    return lanes


def run_source(args, name: str, render: bool) -> dict:
    """Play every stored episode through ``ProcessAdapter`` workers that own source envs; submit step t+1 before taking t."""
    from stable_envs._execution.process import ProcessAdapter

    data = np.load(Path(args.data) / f"{name}.npz")
    seeds, actions = data["seeds"][: args.episodes], data["actions"][: args.episodes]
    n, length = actions.shape[:2]
    lanes = min(args.procs * args.source_lanes, n)
    cpus = sorted(os.sched_getaffinity(0))[: args.procs]
    factory = functools.partial(_source_factory, name, args.size, render, actions[:1, 0])
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

    groups = list(range(0, n - lanes + 1, lanes))
    check_lanes = sorted({0, lanes // 2, lanes - 1})
    check_steps = tuple(t for t in CHECK_STEPS if t < length)
    got: dict = {}
    try:
        play(groups[:1])
        t = _sweeps(lambda: play(groups), warmup=0, repeats=args.repeats, budget_s=args.budget)
        if render:
            # The workers' frames for a few lanes, spread across workers, at the check steps.
            adapter.reset(int(seeds[0]))
            for step in range(max(check_steps) + 1):
                adapter.submit_step(actions[:lanes, step], capture_state=False, capture_frame=True)
                frame = adapter.receive_step()["frame"].numpy()
                if step in check_steps:
                    for lane in check_lanes:
                        got[lane, step] = _digest(frame[lane])
    finally:
        adapter.close()
    check = None
    if render:
        bad = []
        for lane in check_lanes:
            want = _replay_digests(name, args.size, int(seeds[lane]), actions[lane], check_steps)
            bad += [f"{lane}/{step}" for step in check_steps if got[lane, step] != want[step]]
        check = {
            "lanes": check_lanes,
            "steps": list(check_steps),
            "frames": len(got),
            "mismatched": len(bad),
            "examples": bad[:5],
        }
    steps = len(groups) * lanes * length
    return {
        "source": source_id(name),
        "frame_shape": list(frame_shape) if frame_shape else None,
        "procs": args.procs,
        "lanes": lanes,
        "cpus": cpus,
        "actions_per_call": 1,
        "steps": steps,
        "steps_per_s": steps / t["median_s"],
        **t,
        **({"frame_check": check, "status": "invalid"} if check and check["mismatched"] else {"frame_check": check}),
    }


def _make_ours(name: str, lanes: int, size: int, device: str):
    """The class at ``size``, or at its one native size when it refuses another."""
    import stable_envs as snv

    try:
        return snv.make(name, num_envs=lanes, size=size, device=device)
    except ValueError as e:
        if "renders" not in str(e) or "only" not in str(e):
            raise
        return snv.make(name, num_envs=lanes, device=device)


def run_ours(args, name: str, render_device: str | None, chunk: int) -> dict:
    data = np.load(Path(args.data) / f"{name}.npz")
    seeds, actions = data["seeds"][: args.episodes], data["actions"][: args.episodes]
    n, length = actions.shape[:2]
    lanes = min(args.lanes, n)
    dev = render_device or "cpu"
    torch.set_num_threads(args.threads)
    env = _make_ours(name, lanes, args.size, dev)
    sync = torch.cuda.synchronize if dev.startswith("cuda") else None
    acts = torch.as_tensor(actions, device=dev)
    frames = None if render_device is None else slice(None)

    def play(groups):
        for s in groups:
            env.reset(seed=int(seeds[s]))
            for t0 in range(0, length, chunk):
                env.step(acts[s : s + lanes, t0 : t0 + chunk], states=None, frames=frames)

    groups = list(range(0, n - lanes + 1, lanes))
    try:
        play(groups[:1])
        shape = None
        if render_device is not None:
            env.reset(seed=int(seeds[0]))
            shape = list(env.step(acts[:lanes, :1], states=None, frames=slice(None))["frames"].shape[2:])
        t = _sweeps(lambda: play(groups), warmup=0, repeats=args.repeats, budget_s=args.budget, sync=sync)
    finally:
        env.close()
    steps = len(groups) * lanes * length
    return {
        "lanes": lanes,
        "actions_per_call": chunk,
        "threads": args.threads,
        "frame_shape": shape,
        "steps": steps,
        "steps_per_s": steps / t["median_s"],
        **t,
    }


ARMS = ("src-pipelined-render", "src-pipelined-step", "snv-step", "snv-render-cpu", "snv-render-gpu")


def cmd_run(args) -> None:
    res = {"host": _host(), "env": args.env, "arms": {}, "budget_s": args.budget, "repeats_max": args.repeats}
    for arm in args.arms:
        chunks = [1] if arm.startswith("src-") else args.chunks
        for chunk in chunks:
            key = f"{arm}-{args.src_label}" if arm == "src-pipelined-render" else arm
            key = key if arm.startswith("src-") else f"{key}-T{chunk}"
            try:
                if arm == "src-pipelined-render":
                    r = run_source(args, args.env, True)
                elif arm == "src-pipelined-step":
                    r = run_source(args, args.env, False)
                else:
                    device = {"snv-step": None, "snv-render-cpu": "cpu", "snv-render-gpu": "cuda"}[arm]
                    r = run_ours(args, args.env, device, chunk)
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
    r.add_argument("--chunks", type=int, nargs="+", default=[1, 4, 16], help="actions per stable-envs step call (T)")
    r.add_argument("--data", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--procs", type=int, default=8)
    r.add_argument("--source-lanes", type=int, default=8)
    r.add_argument("--lanes", type=int, default=1024)
    r.add_argument("--episodes", type=int, default=None, help="play only the first N stored episodes")
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--size", type=int, default=224)
    r.add_argument("--repeats", type=int, default=10)
    r.add_argument("--budget", type=float, default=300.0)
    args = p.parse_args()
    {"dataset": cmd_dataset, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
