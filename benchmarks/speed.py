"""Time stable-envs against stable-worldmodel (swm) on the same CPU cores, in the same run, and write one CSV.

    python -m benchmarks.speed --cpu --gpu --step --render --step-and-render --env PushT TwoRoom --batch 1 16 64

The script runs every combination of device, workload, env, batch B and actions per call T. Each cell times
stable-envs ("ours") and then swm, each side in a fresh process pinned to the same cores.

Workloads:
  step             physics steps without frames. Ours: B instances, T actions per ``step`` call. swm: B envs over
                   min(B, cores) worker processes, one core each; the parent sends step t+1 before it takes step t.
  step-and-render  as step, with a frame after every step.
  render           frames of stored States. Ours: ``snv.FrameLoader`` hands out B frames at a time. swm: the best
                   of 1 and ``cores`` processes that write each State into an swm env and draw it.

Timing: each side warms up for ``--warmup`` seconds and at least ``--warmup-steps`` steps per instance, then
times ``--repeats`` windows (``--repeats-small`` at B up to ``--small-max-batch``) and keeps the median rate.
A window holds ``--timed-steps`` steps per instance, rounded up to whole episodes (each episode starts with a
seeded reset), or ``--window`` seconds when ``--timed-steps`` is 0. On the GPU device, the swm side also draws
with MuJoCo EGL where swm draws with MuJoCo, and the CSV keeps the faster swm result.

When an swm step cell would take longer than ``--source-cap`` seconds, the script estimates its rate from a
partial run instead: ``--repeats`` windows of ``--partial-window`` seconds, counted in steps, not in whole
episodes. The CSV marks such cells ``estimated``.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import multiprocessing as mp
import os
import platform
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
import traceback
from multiprocessing import shared_memory
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
COLUMNS = [
    "env",
    "device",
    "workload",
    "B",
    "T",
    "ours_rate",
    "source_rate",
    "ratio",
    "source_kind",
    "estimated",
    "windows",
    "status",
    "host",
    "cpu_model",
    "gpu",
    "driver",
    "cores",
    "stable_envs_sha",
    "source_versions",
    "frame_check",
]
WORKLOADS = ("step", "step-and-render", "render")
#: Episode length and seed of the stored action sequences.
LENGTH, SEED = 64, 20261022


# Timing.


def cycle(make):
    """Endless passes over the iterable that ``make`` returns."""
    while True:
        yield from make()


def timed(
    units,
    *,
    seconds,
    repeats,
    sync=None,
    warmup_s=0.0,
    warmup_items=0,
    window_items=0,
    warmup_max_s=0.0,
    budget_s=60.0,
) -> dict:
    """Items per second over up to ``repeats`` windows; each ``next(units)`` does work and returns its items.

    The warm-up lasts ``warmup_s`` seconds and at least ``warmup_items`` items (``warmup_max_s`` stops the item
    count). With ``window_items``, a window ends at the first whole unit at or after that many items; else it lasts
    ``seconds``. After the first window, no new window starts once the windows took ``budget_s`` in total.
    """
    t0 = time.perf_counter()
    end, warmed = t0 + warmup_s, 0
    while time.perf_counter() < end or (
        warmed < warmup_items and not (warmup_max_s and time.perf_counter() - t0 > warmup_max_s)
    ):
        warmed += next(units)
    warm = {"warmup_items": warmed, "warmup_items_wanted": warmup_items, "warmup_s": time.perf_counter() - t0}
    rates, windows, items = [], [], []
    for _ in range(repeats):
        if windows and sum(windows) >= budget_s:
            break
        if sync:
            sync()
        t = time.perf_counter()
        done = 0
        while (done < window_items) if window_items else (time.perf_counter() - t < seconds):
            done += next(units)
        if sync:
            sync()
        dt = time.perf_counter() - t
        rates.append(done / dt)
        windows.append(dt)
        items.append(done)
    return {
        "rate": statistics.median(rates),
        "rates": rates,
        "window_s": windows,
        "items": items,
        "repeats": len(rates),
        **warm,
    }


def _worker_for(job) -> tuple[int, float]:
    """Draw this worker's rows 8 at a time, wrapping around, for ``seconds`` or at least ``count`` frames."""
    (s, e), seconds, render, count = job
    t = time.perf_counter()
    done, i = 0, s
    while (done < count) if count else (time.perf_counter() - t < seconds):
        j = min(i + 8, e)
        done += render((i, j))
        i = s if j >= e else j
    return done, time.perf_counter() - t


def pool_timed(pool, render, spans, *, seconds, repeats, warmup_s=0.0, warmup_items=0, window_items=0) -> dict:
    """Frames per second of all workers together: one worker per span, every worker timed over the same window."""
    if warmup_s > 0 or warmup_items:
        pool.map(_worker_for, [(sp, warmup_s, render, warmup_items) for sp in spans], chunksize=1)
    rates = []
    for _ in range(repeats):
        out = pool.map(_worker_for, [(sp, seconds, render, window_items) for sp in spans], chunksize=1)
        rates.append(sum(d / dt for d, dt in out))
    return {"rate": statistics.median(rates), "rates": rates, "repeats": repeats}


# Host.


def _sh(*cmd) -> str:
    try:
        return subprocess.run(cmd, capture_output=True, text=True, timeout=30).stdout.strip()
    except (OSError, subprocess.TimeoutExpired):
        return ""


def host_info(cores) -> dict:
    model = next((ln.split(":", 1)[1].strip() for ln in _sh("lscpu").splitlines() if ln.startswith("Model name")), "")
    gpu = _sh(
        "nvidia-smi",
        "-i",
        os.environ.get("CUDA_VISIBLE_DEVICES", "0").split(",")[0],
        "--query-gpu=name,driver_version",
        "--format=csv,noheader",
    )
    name, _, driver = gpu.partition(",")
    sha = _sh("git", "-C", str(ROOT), "rev-parse", "HEAD")
    dirty = _sh("git", "-C", str(ROOT), "status", "--porcelain", "--untracked-files=no")
    return {
        "host": platform.node(),
        "cpu_model": model or platform.processor(),
        "gpu": name.strip(),
        "driver": driver.strip(),
        "cores": ",".join(map(str, cores)),
        "stable_envs_sha": sha + ("-dirty" if dirty else ""),
    }


# Stored inputs.


def make_dataset(name: str, path: Path, episodes: int, length: int) -> None:
    """Seeds and uniform random actions of ``episodes`` episodes of ``length`` steps."""
    import stable_envs as snv

    env = snv.make(name, num_envs=1, device="cpu")
    spec = env.action_spec
    env.close()
    rng = np.random.default_rng(SEED)
    shape = (episodes, length, *spec.shape[-1:])
    if getattr(spec, "discrete_n", None):
        actions = rng.integers(0, spec.discrete_n, size=shape).astype(np.int64)
    elif not spec.dtype.is_floating_point:
        actions = rng.integers(int(spec.low), int(spec.high) + 1, size=shape).astype(np.int64)
    else:
        low = np.broadcast_to(np.asarray(spec.low, np.float32), spec.shape[-1:])
        high = np.broadcast_to(np.asarray(spec.high, np.float32), spec.shape[-1:])
        actions = rng.uniform(low, high, size=shape).astype(np.float32)
    np.savez(path, seeds=np.arange(episodes, dtype=np.int64) + SEED, actions=actions)


def make_states(name: str, path: Path, n: int, *, lanes=64, steps=1000, every=30) -> None:
    """``n`` full States saved every ``every`` random actions after seeded resets, up to ``steps`` per reset."""
    import torch

    import stable_envs as snv

    rng = np.random.default_rng(SEED)
    cls = snv.info(name)["class"]
    saved, have = [], 0
    with snv.make(name, num_envs=min(lanes, n), size=cls.native_size or 224, device="cpu") as env:
        while have < n:
            env.reset(seed=int(rng.integers(0, 2**31 - 1)))
            for t in range(1, steps + 1):
                env.step(env.action_spec.sample(env.B, rng), frames=None)
                if t % every == 0:
                    st = env.get_state().cpu()
                    saved.append(st)
                    have += int(st.batch_size[0])
                    if have >= n:
                        break
    from benchmarks.swm_source import rows

    states = _cat_states(saved)[:n]
    torch.save(states, path)
    torch.save(rows(states), rows_path(path))


def _cat_states(states):
    """Concatenate State batches whose growing leaves (PushT's action history) differ in length along axis 1.

    Shorter leaves are zero-padded; a history's ``valid`` mask is then False on the padding.
    """
    import torch

    lengths: dict = {}
    for st in states:
        for key, value in st.items(True, True):
            if value.dim() > 1:
                lengths[key] = max(lengths.get(key, 0), value.shape[1])
    padded = []
    for st in states:
        st = st.clone()
        for key, value in list(st.items(True, True)):
            if value.dim() > 1 and value.shape[1] < lengths[key]:
                pad = value.new_zeros((value.shape[0], lengths[key] - value.shape[1], *value.shape[2:]))
                st.set(key, torch.cat([value, pad], 1))
        padded.append(st)
    return torch.cat(padded)


def rows_path(states_path) -> Path:
    """The file of the per-instance field dicts (``swm_source.rows``) of a stored State file."""
    p = Path(states_path)
    return p.with_name(p.stem + "-rows.pt")


# Sides. Each runs in its own process (see ``_side``).


def _ours_env(name, lanes, device):
    import stable_envs as snv

    return snv.make(name, num_envs=lanes, size=snv.info(name)["class"].native_size or 224, device=device)


def ours_online(spec) -> dict:
    import torch

    data = np.load(spec["data"])
    seeds, actions = data["seeds"][: spec["B"]], data["actions"][: spec["B"]]
    n, length = actions.shape[:2]
    lanes, chunk, dev = min(spec["B"], n), spec["T"], spec["device"]
    torch.set_num_threads(spec["threads"])
    env = _ours_env(spec["env"], lanes, dev)
    sync = torch.cuda.synchronize if dev.startswith("cuda") else None
    acts = torch.as_tensor(actions, device=dev)
    frames = slice(None) if spec["render"] else None
    states = slice(None) if spec.get("step_states") == "all" else None

    def play(groups):
        for s in groups:
            env.reset(seed=int(seeds[s]))
            for t0 in range(0, length, chunk):
                env.step(acts[s : s + lanes, t0 : min(t0 + chunk, length)], states=states, frames=frames)
            yield lanes * length

    groups = list(range(0, n - lanes + 1, lanes))
    try:
        if spec["render"]:
            env.reset(seed=int(seeds[0]))
            env.step(acts[:lanes, :1], states=None, frames=slice(None))
        t = timed(cycle(lambda: play(groups)), sync=sync, **_timing(spec, lanes))
    finally:
        env.close()
    return {"lanes": lanes, "actions_per_call": chunk, "steps_per_episode": length, **t}


def ours_render(spec) -> dict:
    import torch

    import stable_envs as snv

    dev = spec["device"]
    if dev == "cpu":
        torch.set_num_threads(spec["threads"])
    sync = torch.cuda.synchronize if dev.startswith("cuda") else None
    states = torch.load(spec["states"], weights_only=False).to(dev)
    b = spec["B"]
    env = _ours_env(spec["env"], b, dev)
    try:
        env.render(states[:b])
        loader = snv.FrameLoader(env, states, b)

        def sweep():
            # One unit is one render-ahead chunk, so that every window pays for every frame it receives.
            got, per = 0, loader.chunk_size
            for frames in loader:
                got += int(frames.shape[0])
                if got >= per:
                    yield got
                    got = 0
            if got:
                yield got

        t = timed(cycle(sweep), sync=sync, **_timing(spec, b))
    finally:
        env.close()
    return {"frames": int(states.batch_size[0]), "chunk_size": loader.chunk_size, **t}


def _timing(spec, per_item) -> dict:
    """The ``timed`` keyword arguments of a cell whose items count ``per_item`` per step."""
    return dict(
        seconds=spec["window"],
        repeats=spec["repeats"],
        warmup_s=spec["warmup"],
        warmup_items=spec["warmup_steps"] * per_item,
        window_items=spec["timed_steps"] * per_item,
        warmup_max_s=spec["warmup_max_s"],
    )


def _lanes_worker(conn, name, size, render, start, count, seed, cpus, total):
    """Own instances ``start:start + count`` of a ``SourcePool`` and serve the parent's requests."""
    import torch

    from benchmarks.swm_source import SwmSource

    envs = shm = view = None
    try:
        if cpus is not None and hasattr(os, "sched_setaffinity"):
            os.sched_setaffinity(0, cpus)
        torch.set_num_threads(1)
        envs = [SwmSource(name, size) for _ in range(count)]
        for i, env in enumerate(envs):
            env.reset(seed=None if seed is None else seed + start + i)
        shape = envs[0].render().shape if render else None
        conn.send((True, shape))
        while True:
            command, payload = conn.recv()
            if command == "close":
                conn.send((True, None))
                break
            if command == "frames":
                shm = shared_memory.SharedMemory(name=payload)
                view = np.ndarray((total, *shape), np.uint8, shm.buf)[start : start + count]
            elif command == "reset":
                for i, env in enumerate(envs):
                    env.reset(seed=None if payload is None else payload + start + i)
            elif command == "step":
                actions, frame = payload
                for env, a in zip(envs, actions):
                    env.step(a)
                if frame:
                    view[:] = np.stack([env.render() for env in envs])
            conn.send((True, None))
    except BaseException:
        try:
            conn.send((False, traceback.format_exc()))
        except (BrokenPipeError, EOFError, OSError):
            pass
    finally:
        view = None
        if shm is not None:
            shm.close()
        for env in envs or []:
            env.close()
        conn.close()


class SourcePool:
    """swm envs split over ``processes`` spawned workers, each pinned to its share of ``cpus`` with one torch thread.

    Instance ``i`` gets seed ``seed + i``. Workers write frames to shared memory. Submit one step, then receive it
    before the next submit.
    """

    def __init__(self, name, size, render, lanes, seed, processes, cpus):
        ctx = mp.get_context("spawn")
        self._conns, self._procs, self._shm, self._frames, self._capture = [], [], None, None, False
        self._slices = [slice(int(r[0]), int(r[-1]) + 1) for r in np.array_split(np.arange(lanes), processes)]
        groups = [g.tolist() for g in np.array_split(cpus, processes)]
        try:
            for rows, group in zip(self._slices, groups):
                parent, child = ctx.Pipe()
                args = (child, name, size, render, rows.start, rows.stop - rows.start, seed, group, lanes)
                proc = ctx.Process(target=_lanes_worker, args=args)
                proc.start()
                child.close()
                self._conns.append(parent)
                self._procs.append(proc)
            shapes = self._gather()
            if render:
                shape = tuple(shapes[0])
                self._shm = shared_memory.SharedMemory(create=True, size=int(lanes * np.prod(shape)))
                self._frames = np.ndarray((lanes, *shape), np.uint8, self._shm.buf)
                for conn in self._conns:
                    conn.send(("frames", self._shm.name))
                self._gather()
        except BaseException:
            self.close()
            raise

    def _gather(self):
        replies, errors = [], []
        for conn in self._conns:
            ok, value = conn.recv()
            replies.append(value)
            if not ok:
                errors.append(value)
        if errors:
            raise RuntimeError("source worker failed:\n" + "\n".join(errors))
        return replies

    def reset(self, seed=None):
        for conn in self._conns:
            conn.send(("reset", seed))
        self._gather()

    def submit_step(self, actions, *, capture_frame=False):
        actions = np.asarray(actions)
        for conn, rows in zip(self._conns, self._slices):
            conn.send(("step", (actions[rows], capture_frame)))
        self._capture = capture_frame

    def receive_step(self):
        """Wait for the step; return a copy of its frames, or None without ``capture_frame``."""
        self._gather()
        return self._frames.copy() if self._capture else None

    def close(self):
        for conn in self._conns:
            try:
                conn.send(("close", None))
                conn.recv()
            except (BrokenPipeError, EOFError, OSError):
                pass
            conn.close()
        for proc in self._procs:
            proc.join(10)
            if proc.is_alive():
                proc.terminate()
        self._conns, self._procs, self._frames = [], [], None
        if self._shm is not None:
            self._shm.close()
            self._shm.unlink()
            self._shm = None


def source_online(spec) -> dict:
    """Play the stored episodes through a ``SourcePool``; submit step t+1 before taking step t.

    A short probe first times a few steps. When the projected cell time is over ``source_cap`` seconds, the rate
    comes from a partial run: windows of ``partial_window`` seconds, counted in steps.
    """
    data = np.load(spec["data"])
    seeds, actions = data["seeds"][: spec["B"]], data["actions"][: spec["B"]]
    n, length = actions.shape[:2]
    procs = min(spec["B"], len(spec["cores"]))
    lanes = min(procs * math.ceil(spec["B"] / procs), n)
    cpus = spec["cores"][:procs]
    render = spec["render"]
    pool = SourcePool(spec["env"], 224, render, lanes, int(seeds[0]), procs, cpus)
    out = {"procs": procs, "lanes": lanes, "cpus": cpus, "steps_per_episode": length}
    try:
        probe, t0 = 3, time.perf_counter()
        pool.reset(int(seeds[0]))
        for t in range(probe):
            pool.submit_step(actions[:lanes, t], capture_frame=render)
            pool.receive_step()
        per_step = (time.perf_counter() - t0) / probe
        # The warm-up and every window hold whole episodes.
        window = math.ceil(spec["timed_steps"] / length) * length if spec["timed_steps"] else spec["window"] / per_step
        projected = per_step * (length + window * spec["repeats"])
        out["projected_s"] = projected
        if projected > spec["source_cap"]:
            return {**out, **_partial(pool, spec, seeds, actions, lanes, render), "estimated": True}

        def play():
            pool.reset(int(seeds[0]))
            pool.submit_step(actions[:lanes, 0], capture_frame=render)
            for t in range(length):
                pool.receive_step()
                if t + 1 < length:
                    pool.submit_step(actions[:lanes, t + 1], capture_frame=render)
            yield lanes * length

        units = cycle(play)
        t = timed(units, **_timing(spec, lanes))
        units.close()
        got = _worker_frames(pool, seeds, actions, lanes) if render and spec.get("check_frames") else None
    finally:
        pool.close()
    res = {**out, **t, "estimated": False}
    if got is not None:
        res["frame_check"] = check = _frame_check(spec, got, seeds, actions, lanes)
        if check["reproducible"] and check["mismatched"]:
            res["status"] = "invalid"
    return res


def _check_steps(length: int) -> list[int]:
    """Steps whose frames the frame check compares."""
    return sorted({t for t in (0, 1, 20, length - 1) if 0 <= t < length})


def _worker_frames(pool, seeds, actions, lanes) -> dict:
    """Digests of the pool workers' frames of a few instances, spread across workers, at the check steps."""
    import hashlib

    steps = _check_steps(actions.shape[1])
    check_lanes = sorted({0, lanes // 2, lanes - 1})
    got = {}
    pool.reset(int(seeds[0]))
    for step in range(max(steps) + 1):
        pool.submit_step(actions[:lanes, step], capture_frame=True)
        frame = pool.receive_step()
        if step in steps:
            for lane in check_lanes:
                got[lane, step] = hashlib.sha1(np.ascontiguousarray(frame[lane]).tobytes()).hexdigest()
    return got


def _replay(name, seed, actions):
    """Every frame of one episode played by a single swm env, and the first step that ends it."""
    from benchmarks.swm_source import SwmSource

    env = SwmSource(name)
    try:
        env.reset(seed=seed)
        frames, end = [], None
        for t, a in enumerate(actions):
            done = env.step(a)
            frames.append(env.render().copy())
            if done and end is None:
                end = t
        return np.stack(frames), end
    finally:
        env.close()


def _replay_in_new_process(name, seed, actions):
    # Some sources keep their GL context alive, and a second context in one process changes the frames it draws.
    with mp.get_context("spawn").Pool(1) as p:
        return p.apply(_replay, (name, seed, actions))


def _frame_check(spec, got, seeds, actions, lanes) -> dict:
    """Compare the pool workers' frames with one-env replays, and ours' frames with the replays.

    A second replay of the first instance tells whether swm's seeded reset is reproducible; without that the
    comparison says nothing. Ours resets an instance when its episode ends and swm keeps stepping it, so the
    rollout MSE compares frames up to the end of the episode.
    """
    import hashlib

    import torch

    import stable_envs as snv

    name, steps = spec["env"], _check_steps(actions.shape[1])
    check_lanes = sorted({0, lanes // 2, lanes - 1})
    theirs = {lane: _replay_in_new_process(name, int(seeds[lane]), actions[lane]) for lane in check_lanes}
    again = _replay_in_new_process(name, int(seeds[check_lanes[0]]), actions[check_lanes[0]])[0]
    reproducible = bool(again.shape == theirs[check_lanes[0]][0].shape and (again == theirs[check_lanes[0]][0]).all())
    digest = lambda f: hashlib.sha1(np.ascontiguousarray(f).tobytes()).hexdigest()  # noqa: E731
    bad = [f"{lane}/{t}" for lane in check_lanes for t in steps if got[lane, t] != digest(theirs[lane][0][t])]
    device = "cuda" if spec.get("gl") == "egl" else "cpu"
    per_frame = []
    for lane in check_lanes:
        source, source_end = theirs[lane]
        with snv.make(name, num_envs=1, size=source.shape[1], device=device) as env:
            env.reset(seed=int(seeds[lane]))
            out = env.step(torch.as_tensor(actions[lane][None], device=device), states=None, frames=slice(None))
            ours = out["frames"][0].cpu().numpy()
            ended = np.zeros(len(actions[lane]), bool)
            for k in ("terminated", "truncated", "done"):
                if k in out.keys():
                    ended |= out[k][0].reshape(len(ended), -1).any(-1).cpu().numpy().astype(bool)
        hits = np.flatnonzero(ended)
        stop = min(x for x in (hits[0] if len(hits) else None, source_end, len(source) - 1) if x is not None) + 1
        if ours.shape == source[..., :3].shape:
            diff = ours[:stop].astype(np.float64) - source[:stop, ..., :3].astype(np.float64)
            per_frame.append((diff**2).mean(axis=(1, 2, 3)))
    mse = np.concatenate(per_frame) if per_frame else None
    return {
        "reproducible": reproducible,
        "instances": check_lanes,
        "steps": steps,
        "frames": len(got),
        "mismatched": len(bad),
        "examples": bad[:5],
        "ours_device": device,
        "rollout_mse_mean": None if mse is None else float(mse.mean()),
        "rollout_mse_max": None if mse is None else float(mse.max()),
    }


def _partial(pool, spec, seeds, actions, lanes, render) -> dict:
    """Rate from windows of ``partial_window`` seconds counted in steps; resets at the episode length count."""
    length = actions.shape[1]
    step = {"t": 0}

    def submit():
        if step["t"] == 0:
            pool.reset(int(seeds[0]))
        pool.submit_step(actions[:lanes, step["t"]], capture_frame=render)
        step["t"] = (step["t"] + 1) % length

    submit()
    w0, warmed = time.perf_counter(), 0
    while warmed < spec["warmup_steps"] or time.perf_counter() - w0 < spec["warmup"]:
        pool.receive_step()
        submit()
        warmed += 1
    warm_s = time.perf_counter() - w0
    rates, secs, steps = [], [], []
    for _ in range(spec["repeats"]):
        s0, k = time.perf_counter(), 0
        while True:
            pool.receive_step()
            submit()
            k += 1
            el = time.perf_counter() - s0
            if el >= spec["partial_window"]:
                break
        rates.append(k * lanes / el)
        secs.append(el)
        steps.append(k)
    pool.receive_step()
    return {
        "rate": statistics.median(rates),
        "rates": rates,
        "window_s": secs,
        "steps": steps,
        "repeats": len(rates),
        "warmup_items": warmed * lanes,
        "warmup_s": warm_s,
        "method": "partial run, windows counted in steps",
    }


_W: dict = {}


def _render_init(name, states_path):
    import torch

    from benchmarks.swm_source import SwmSource

    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    _W["src"] = SwmSource(name)
    _W["src"].reset(seed=0)
    _W["rows"] = torch.load(rows_path(states_path), weights_only=False)


def _render_rows(span) -> int:
    s, e = span
    src = _W["src"]
    for i in range(s, e):
        src.set_state(_W["rows"][i], velocity=False)
        src.render()
    return e - s


def source_render(spec) -> dict:
    """Draw the stored States with swm in ``procs`` processes, 8 States at a time."""
    procs = spec["procs"]
    if procs == 1:
        _render_init(spec["env"], spec["states"])
        n = len(_W["rows"])
        _W["src"].set_state(_W["rows"][0], velocity=False)
        shape = list(_W["src"].render().shape)
        units = cycle(lambda: (_render_rows((i, min(i + 8, n))) for i in range(0, n, 8)))
        t = timed(units, **_timing(spec, 8))
        return {"procs": 1, "frames": n, "frame_shape": shape, **t}
    import torch

    n = len(torch.load(rows_path(spec["states"]), weights_only=False))
    spans = [(n * k // procs, n * (k + 1) // procs) for k in range(procs)]
    with mp.get_context("spawn").Pool(procs, initializer=_render_init, initargs=(spec["env"], spec["states"])) as pool:
        t = pool_timed(
            pool,
            _render_rows,
            spans,
            seconds=spec["window"],
            repeats=spec["repeats"],
            warmup_s=spec["warmup"],
            warmup_items=-(-spec["warmup_steps"] * 8 // procs),
            window_items=-(-spec["timed_steps"] * 8 // procs),
        )
    return {"procs": procs, "frames": n, **t}


SIDES = {
    "ours-online": ours_online,
    "ours-render": ours_render,
    "source-online": source_online,
    "source-render": source_render,
}


def _side(spec_path: str, result_path: str) -> None:
    """Run one side of one cell in this process and write its JSON result."""
    spec = json.loads(Path(spec_path).read_text())
    if hasattr(os, "sched_setaffinity"):
        os.sched_setaffinity(0, spec["cores"])
    if spec["side"].startswith("ours"):
        # Ours imports stable_envs before torch, as a user's script does; its import-time settings then apply.
        import stable_envs  # noqa: F401
    try:
        res = {"status": "ok", **SIDES[spec["side"]](spec)}
        if spec["side"].startswith("source"):
            from benchmarks.swm_source import source_id, versions

            res.update(source=source_id(spec["env"]), source_versions=versions())
    except Exception as e:  # noqa: BLE001
        msg = f"{type(e).__name__}: {e}"
        res = {
            "status": "oom" if "out of memory" in msg.lower() else "error",
            "error": msg[:600],
            "traceback": traceback.format_exc()[-2000:],
        }
    Path(result_path).write_text(json.dumps(res))


# Driver.


def _gl_env(gpu: bool) -> dict:
    """OpenGL settings: EGL on the GPU, else software OpenGL."""
    if gpu:
        return {"MUJOCO_GL": "egl", "PYOPENGL_PLATFORM": "egl"}
    env = {"LIBGL_ALWAYS_SOFTWARE": "1", "GALLIUM_DRIVER": "llvmpipe", "PYOPENGL_PLATFORM": "glx", "MUJOCO_GL": "glfw"}
    mesa = "/usr/share/glvnd/egl_vendor.d/50_mesa.json"
    if os.path.exists(mesa):
        env["__EGL_VENDOR_LIBRARY_FILENAMES"] = mesa
    return env


class Runner:
    """Run sides in fresh pinned processes and keep their results for the run."""

    def __init__(self, args, work: Path):
        self.args, self.work, self.cache, self.count = args, work, {}, 0

    def run(self, side: str, gpu_gl: bool, **spec) -> dict:
        key = json.dumps([side, gpu_gl, spec], sort_keys=True, default=str)
        if key in self.cache:
            return self.cache[key]
        a = self.args
        spec = dict(
            spec,
            side=side,
            cores=a.cores,
            threads=len(a.cores),
            window=a.window,
            warmup=a.warmup,
            warmup_steps=a.warmup_steps,
            timed_steps=a.timed_steps,
            warmup_max_s=a.warmup_max_s,
            source_cap=a.source_cap,
            partial_window=a.partial_window,
            repeats=spec.get("repeats", a.repeats),
            check_frames=a.check_frames,
            gl="egl" if gpu_gl else "cpu",
        )
        self.count += 1
        spec_path, result_path = self.work / f"spec-{self.count}.json", self.work / f"result-{self.count}.json"
        spec_path.write_text(json.dumps(spec))
        cmd = [sys.executable, "-m", "benchmarks.speed", "_side", str(spec_path), str(result_path)]
        if not gpu_gl and shutil.which("xvfb-run") and not os.environ.get("DISPLAY"):
            cmd = ["xvfb-run", "-a", "-n", str(os.getpid() % 20000 + 100), *cmd]
        env = {
            **os.environ,
            **_gl_env(gpu_gl),
            "PYTHONPATH": str(ROOT),
            "SDL_VIDEODRIVER": "dummy",
            "PYTHONWARNINGS": "ignore",
            "PYTHONDONTWRITEBYTECODE": "1",
            "MPLBACKEND": "Agg",
        }
        try:
            proc = subprocess.run(cmd, env=env, cwd=ROOT, timeout=a.side_timeout)
            res = (
                json.loads(result_path.read_text())
                if result_path.exists()
                else {"status": "error", "error": f"exit code {proc.returncode}"}
            )
        except subprocess.TimeoutExpired:
            res = {"status": "timeout", "error": f"over {a.side_timeout} s"}
        print(
            f"{spec['env']} {side} {json.dumps({k: spec[k] for k in ('B', 'T', 'device', 'procs', 'render') if k in spec})}"
            f" gl={'egl' if gpu_gl else 'cpu'} {res['status']} {res.get('rate', 0):.1f}/s",
            flush=True,
        )
        self.cache[key] = res
        return res


def _best(results: dict) -> tuple[str, dict]:
    """The fastest of ``{kind: result}`` that finished."""
    ok = {k: r for k, r in results.items() if r.get("status") == "ok"}
    if not ok:
        return next(iter(results.items()), ("none", {"status": "skipped"}))
    return max(ok.items(), key=lambda kv: kv[1]["rate"])


def main(argv=None) -> None:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--cpu", action="store_true", help="time ours on the CPU")
    p.add_argument("--gpu", action="store_true", help="time ours on CUDA")
    p.add_argument("--step", action="store_true")
    p.add_argument("--render", action="store_true", help="frames of stored States (the default workload)")
    p.add_argument("--step-and-render", action="store_true")
    p.add_argument("--env", nargs="+", required=True)
    p.add_argument("--batch", type=int, nargs="+", default=[1, 8, 16, 64])
    p.add_argument("--actions-per-call", type=int, nargs="+", default=[1, 16], help="T of ours in step workloads")
    p.add_argument("--window", type=float, default=3.0, help="seconds per window when --timed-steps is 0")
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--repeats-small", type=int, default=7, help="windows at B up to --small-max-batch")
    p.add_argument("--small-max-batch", type=int, default=16)
    p.add_argument("--warmup", type=float, default=0.0, help="least seconds of untimed work")
    p.add_argument("--warmup-steps", type=int, default=20, help="least untimed steps per instance")
    p.add_argument("--warmup-max-s", type=float, default=300.0, help="stop the warm-up step count after this")
    p.add_argument("--timed-steps", type=int, default=100, help="steps per instance in a window; 0: --window")
    p.add_argument("--length", type=int, default=LENGTH, help="steps per stored episode")
    p.add_argument("--states-n", type=int, default=10_000, help="stored States for the render workload")
    p.add_argument(
        "--step-states",
        choices=["none", "all"],
        default="none",
        help="the states argument of ours' step calls: None, or one State per action",
    )
    p.add_argument("--source-cap", type=float, default=900.0, help="seconds above which an swm cell is estimated")
    p.add_argument("--partial-window", type=float, default=100.0, help="seconds per window of an estimate")
    p.add_argument(
        "--check-frames",
        action="store_true",
        help="step-and-render: compare the swm workers' frames with one-env replays, and ours' frames with them",
    )
    p.add_argument("--side-timeout", type=float, default=1200.0, help="seconds before one side is stopped")
    p.add_argument("--cores", default=None, help="CPU list, e.g. 0-7 (default: the first 8 cores of this process)")
    p.add_argument("--work", default=None, help="directory for the stored inputs (default: a temporary one)")
    p.add_argument("--out", default="results.csv")
    a = p.parse_args(argv)
    allowed = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else list(range(os.cpu_count() or 1))
    a.cores = _cpus(a.cores) if a.cores else allowed[:8]
    devices = [d for d, on in (("cpu", a.cpu), ("cuda", a.gpu)) if on] or ["cpu"]
    workloads = [w for w, on in zip(WORKLOADS, (a.step, a.step_and_render, a.render)) if on] or ["render"]
    work = Path(a.work or tempfile.mkdtemp(prefix="snv-speed-"))
    work.mkdir(parents=True, exist_ok=True)
    host = host_info(a.cores)
    runner = Runner(a, work)
    out = Path(a.out)
    raw = out.with_suffix(".jsonl")
    with open(out, "w", newline="") as f, open(raw, "w") as fr:
        w = csv.DictWriter(f, COLUMNS)
        w.writeheader()
        for name in a.env:
            data, states = work / f"{name}-episodes.npz", work / f"{name}-states.pt"
            if not data.exists() and set(workloads) - {"render"}:
                make_dataset(name, data, max(a.batch), a.length)
            if not states.exists() and "render" in workloads:
                _prepare_states(name, states, a.states_n)
            for dev in devices:
                for wl in workloads:
                    for b in a.batch:
                        reps = a.repeats_small if b <= a.small_max_batch else a.repeats
                        for t in a.actions_per_call if wl != "render" else [None]:
                            row, detail = _cell(runner, name, dev, wl, b, t, reps, data, states, a.step_states)
                            row.update(host)
                            w.writerow(row)
                            f.flush()
                            fr.write(json.dumps({**row, **detail}, default=str) + "\n")
                            fr.flush()
    print(f"wrote {out} and {raw}")


def _prepare_states(name: str, path: Path, n: int) -> None:
    """Make the stored States in a fresh process, so no stable-envs GL context stays in this one."""
    code = f"from benchmarks.speed import make_states; from pathlib import Path; make_states({name!r}, Path({str(path)!r}), {n})"
    subprocess.run(
        [sys.executable, "-c", code],
        env={**os.environ, **_gl_env(False), "PYTHONPATH": str(ROOT)},
        cwd=ROOT,
        check=True,
    )


def _cell(runner, name, dev, wl, b, t, reps, data, states, step_states) -> tuple[dict, dict]:
    from benchmarks.swm_source import draws_on_gpu

    gpu = dev == "cuda"
    egl = gpu and draws_on_gpu(name)
    if wl == "render":
        ours = runner.run("ours-render", gpu, env=name, device=dev, B=b, states=str(states), repeats=reps)
        src = {}
        for procs in sorted({1, len(runner.args.cores)}):
            src[f"swm cpu {procs}p"] = runner.run("source-render", False, env=name, procs=procs, states=str(states))
            if egl:
                src[f"swm egl {procs}p"] = runner.run("source-render", True, env=name, procs=procs, states=str(states))
    else:
        render = wl == "step-and-render"
        ours = runner.run(
            "ours-online",
            gpu,
            env=name,
            device=dev,
            B=b,
            T=t,
            render=render,
            data=str(data),
            repeats=reps,
            step_states=step_states,
        )
        procs = min(b, len(runner.args.cores))
        src = {
            f"swm cpu {procs}p": runner.run(
                "source-online", False, env=name, B=b, render=render, data=str(data), repeats=reps
            )
        }
        if render and egl:
            src[f"swm egl {procs}p"] = runner.run(
                "source-online", True, env=name, B=b, render=True, data=str(data), repeats=reps
            )
    kind, best = _best(src)
    ours_rate = ours.get("rate") if ours.get("status") == "ok" else None
    src_rate = best.get("rate") if best.get("status") == "ok" else None
    status = "ok" if ours_rate and src_rate else f"ours {ours.get('status')}, swm {best.get('status')}"
    row = {
        "env": name,
        "device": "gpu" if gpu else "cpu",
        "workload": wl,
        "B": b,
        "T": "" if t is None else t,
        "ours_rate": _fmt(ours_rate),
        "source_rate": _fmt(src_rate),
        "ratio": _fmt(ours_rate / src_rate if ours_rate and src_rate else None),
        "source_kind": kind if src_rate else "",
        "estimated": bool(best.get("estimated", False)),
        "windows": f"{ours.get('repeats', '')}/{best.get('repeats', '')}",
        "status": status,
        "source_versions": best.get("source_versions", ""),
        "frame_check": _check_text(best.get("frame_check")),
    }
    return row, {"ours_detail": ours, "source_detail": src}


def _check_text(check) -> str:
    if not check:
        return ""
    if not check["reproducible"]:
        return "unverified: swm seeded reset is not reproducible"
    mse = check["rollout_mse_mean"]
    tail = "" if mse is None else f"; ours vs replay MSE mean {mse:.4g} max {check['rollout_mse_max']:.4g}"
    return f"{check['mismatched']}/{check['frames']} worker frames differ from replays{tail}"


def _fmt(x):
    return "" if x is None else f"{x:.4g}"


def _cpus(text: str) -> list[int]:
    out = []
    for part in text.split(","):
        lo, _, hi = part.partition("-")
        out += list(range(int(lo), int(hi or lo) + 1))
    return out


if __name__ == "__main__":
    if sys.argv[1:2] == ["_side"]:
        _side(sys.argv[2], sys.argv[3])
    else:
        main()
