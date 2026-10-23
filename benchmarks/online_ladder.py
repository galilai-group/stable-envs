"""Time playing a stored set of episodes (seeded starts, fixed action sequences) through source and stable-envs pipelines.

Rungs:
  L0  source: swm gym env in 8 worker processes, each stepping its envs and rendering every step into shared memory.
  L1  source: as L0 with rendering off (physics only).
  L2  stable-envs: physics only, CPU worker processes as the library schedules them.
  L3  stable-envs: physics and rendering of every step on CPU, actions submitted ``--chunk`` steps at a time.
  L4  stable-envs: physics on CPU and rendering of every step on CUDA, actions submitted ``--chunk`` steps at a time.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import os
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np
import torch

from benchmarks.offline_sweep import _host, _sweeps

SOURCE = {
    "PushT": ("swm/PushT-v1", {"resolution": 224}),
    "DMCReacherEasy": ("swm/ReacherDMControl-v0", {}),
    "OGBCube": ("swm/OGBCube-v0", {}),
    "PyFlytRocketLanding": ("swm/PFRocketLanding-v0", {}),
}


def cmd_dataset(args) -> None:
    import stable_envs as snv

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.envs:
        env = snv.make(name, num_envs=1, size=args.size, device="cpu")
        spec = env.action_spec
        env.close()
        rng = np.random.default_rng(args.seed)
        low = np.broadcast_to(np.asarray(spec.low, np.float32), spec.shape[-1:]) if hasattr(spec, "low") else None
        high = np.broadcast_to(np.asarray(spec.high, np.float32), spec.shape[-1:]) if hasattr(spec, "high") else None
        actions = rng.uniform(low, high, size=(args.episodes, args.length, len(low))).astype(np.float32)
        seeds = np.arange(args.episodes, dtype=np.int64) + args.seed
        np.savez(out / f"{name}.npz", seeds=seeds, actions=actions)
        print(f"saved {name} actions {actions.shape}", flush=True)


def _make_source(name: str):
    import gymnasium as gym
    import stable_worldmodel  # noqa: F401

    gid, kw = SOURCE[name]
    return gym.make(gid, render_mode="rgb_array", **kw)


#: Steps whose frames the L0 check compares with a one-env replay of the same episode.
CHECK_STEPS = (0, 1, 20, 100, 195)


def _digest(frame) -> str:
    return hashlib.sha1(np.ascontiguousarray(frame).tobytes()).hexdigest()


def _play(envs, episodes, seeds, actions, render, sink=None, checks=None):
    """Play ``episodes`` in groups of ``len(envs)``, stepping and rendering the group round-robin each step."""
    lanes = len(envs)
    for start in range(0, len(episodes), lanes):
        group = episodes[start : start + lanes]
        for j, ep in enumerate(group):
            envs[j].reset(seed=int(seeds[ep]))
        for t in range(actions.shape[1]):
            for j, ep in enumerate(group):
                envs[j].step(actions[ep, t])
                if render:
                    f = envs[j].render()
                    if sink is not None:
                        sink(j, f)
                    if checks is not None and t in CHECK_STEPS:
                        checks[f"{ep}/{t}"] = _digest(f)


def _source_worker(name, render, data_path, lane0, lanes, shm_name, frame_shape, conn):
    os.environ["OMP_NUM_THREADS"] = "1"
    torch.set_num_threads(1)
    data = np.load(data_path)
    seeds, actions = data["seeds"], data["actions"]
    envs = [_make_source(name) for _ in range(lanes)]
    shm = shared_memory.SharedMemory(name=shm_name) if render else None
    buf = np.ndarray((lanes * 8, *frame_shape), np.uint8, buffer=shm.buf) if render else None
    conn.send("ready")
    while True:
        msg = conn.recv()
        if msg is None:
            break

        def sink(j, f):
            buf[(lane0 + j) % len(buf)][: f.shape[0], : f.shape[1]] = f[: frame_shape[0], : frame_shape[1]]

        if isinstance(msg, tuple) and msg[0] == "check":
            checks: dict = {}
            _play(envs, msg[1], seeds, actions, render, sink, checks)
            conn.send(checks)
            continue
        _play(envs, msg, seeds, actions, render, sink)
        conn.send(len(msg))
    for e in envs:
        e.close()
    if shm is not None:
        shm.close()


def run_source(args, name: str, render: bool) -> dict:
    ctx = mp.get_context("spawn")
    data_path = str(Path(args.data) / f"{name}.npz")
    n = len(np.load(data_path)["seeds"])
    probe = _make_source(name)
    probe.reset(seed=0)
    frame_shape = tuple(np.asarray(probe.render()).shape)
    probe.close()
    lanes = args.source_lanes
    shm = shared_memory.SharedMemory(create=True, size=int(np.prod(frame_shape)) * lanes * 8) if render else None
    procs, pipes = [], []
    for k in range(args.procs):
        a, b = ctx.Pipe()
        p = ctx.Process(
            target=_source_worker,
            args=(name, render, data_path, k * lanes, lanes, shm.name if shm else None, frame_shape, b),
        )
        p.start()
        procs.append(p)
        pipes.append(a)
    for a in pipes:
        a.recv()
    shards = [list(range(k, n, args.procs)) for k in range(args.procs)]
    warm = [list(range(k, min(n, args.procs * lanes), args.procs)) for k in range(args.procs)]

    def sweep(parts=shards):
        for a, part in zip(pipes, parts):
            a.send(part)
        for a in pipes:
            a.recv()

    check = None
    try:
        sweep(warm)
        t = _sweeps(sweep, warmup=0, repeats=args.repeats, budget_s=args.budget)
        if render:
            for a, part in zip(pipes, warm):
                a.send(("check", part))
            got = {}
            for a in pipes:
                got.update(a.recv())
            check = _check_frames(name, data_path, got)
    finally:
        for a in pipes:
            a.send(None)
        for p in procs:
            p.join(timeout=60)
        if shm is not None:
            shm.close()
            shm.unlink()
    steps = n * int(np.load(data_path)["actions"].shape[1])
    out = {
        "frame_shape": list(frame_shape),
        "lanes_per_proc": lanes,
        "procs": args.procs,
        "steps": steps,
        "steps_per_s": steps / t["median_s"],
        **t,
    }
    if check is not None:
        out["frame_check"] = check
        if check["mismatched"]:
            out["status"] = "invalid"
    return out


def _check_frames(name: str, data_path: str, got: dict) -> dict:
    """Compare worker frame digests with a one-env replay of each checked episode in this process."""
    data = np.load(data_path)
    episodes = sorted({int(k.split("/")[0]) for k in got})
    env = _make_source(name)
    want: dict = {}
    try:
        _play([env], episodes, data["seeds"], data["actions"], True, checks=want)
    finally:
        env.close()
    bad = sorted(k for k in got if got[k] != want.get(k))
    per_step = {}
    for k, v in got.items():
        per_step.setdefault(k.split("/")[1], set()).add(v)
    return {
        "episodes": len(episodes),
        "frames": len(got),
        "mismatched": len(bad),
        "examples": bad[:5],
        "distinct_frames_per_step": {t: len(v) for t, v in sorted(per_step.items())},
    }


def run_ours(args, name: str, render_device: str | None) -> dict:
    import stable_envs as snv

    data = np.load(Path(args.data) / f"{name}.npz")
    seeds, actions = data["seeds"][: args.episodes], data["actions"][: args.episodes]
    n, length = actions.shape[:2]
    lanes = min(args.lanes, n)
    dev = render_device or "cpu"
    torch.set_num_threads(args.threads)
    env = snv.make(name, num_envs=lanes, size=args.size, device=dev)
    sync = torch.cuda.synchronize if dev.startswith("cuda") else None
    acts = torch.as_tensor(actions, device=dev)

    def play(groups):
        for s in groups:
            env.reset(seed=int(seeds[s]))
            for t0 in range(0, length, args.chunk):
                a = acts[s : s + lanes, t0 : t0 + args.chunk]
                env.step(a, states=None, frames=None if render_device is None else slice(None))

    groups = list(range(0, n - lanes + 1, lanes))
    try:
        play(groups[:1])
        t = _sweeps(lambda: play(groups), warmup=0, repeats=args.repeats, budget_s=args.budget, sync=sync)
    finally:
        env.close()
    steps = len(groups) * lanes * length
    return {"lanes": lanes, "chunk": args.chunk, "steps": steps, "steps_per_s": steps / t["median_s"], **t}


def cmd_run(args) -> None:
    res = {"host": _host(), "rungs": {}, "budget_s": args.budget, "repeats_max": args.repeats}
    for rung in args.rungs:
        try:
            if rung == "L0":
                r = run_source(args, args.env, True)
            elif rung == "L1":
                r = run_source(args, args.env, False)
            elif rung == "L2":
                r = run_ours(args, args.env, None)
            elif rung == "L3":
                r = run_ours(args, args.env, "cpu")
            else:
                r = run_ours(args, args.env, "cuda")
            r.setdefault("status", "ok")
        except Exception as e:  # noqa: BLE001
            msg = f"{type(e).__name__}: {e}"
            r = {"status": "oom" if "out of memory" in msg.lower() else "error", "error": msg[:600]}
        print(f"{args.env} {rung} {r['status']} {r.get('steps_per_s', 0):.0f} steps/s", flush=True)
        res["rungs"][rung] = r
    Path(args.output).write_text(json.dumps(res, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    d = sub.add_parser("dataset")
    d.add_argument("--envs", nargs="+", default=list(SOURCE))
    d.add_argument("--out", required=True)
    d.add_argument("--episodes", type=int, default=1024)
    d.add_argument("--length", type=int, default=196)
    d.add_argument("--seed", type=int, default=20261022)
    d.add_argument("--size", type=int, default=224)
    r = sub.add_parser("run")
    r.add_argument("--env", required=True)
    r.add_argument("--rungs", nargs="+", default=["L0", "L1", "L2", "L3", "L4"])
    r.add_argument("--data", required=True)
    r.add_argument("--output", required=True)
    r.add_argument("--procs", type=int, default=8)
    r.add_argument("--source-lanes", type=int, default=8)
    r.add_argument("--lanes", type=int, default=1024)
    r.add_argument("--episodes", type=int, default=None, help="play only the first N stored episodes")
    r.add_argument("--chunk", type=int, default=4)
    r.add_argument("--threads", type=int, default=8)
    r.add_argument("--size", type=int, default=224)
    r.add_argument("--repeats", type=int, default=10)
    r.add_argument("--budget", type=float, default=300.0)
    args = p.parse_args()
    {"dataset": cmd_dataset, "run": cmd_run}[args.cmd](args)


if __name__ == "__main__":
    main()
