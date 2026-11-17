"""Time frames drawn from stored inputs: stable-envs on CPU or CUDA and the source renderer; and their frame MSE."""

from __future__ import annotations

import argparse
import inspect
import json
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
import traceback
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
#: Seconds of untimed work before the first timed window of a measurement.
WARMUP_S = 2.0


def _host() -> dict:
    out = {
        "host": platform.node(),
        "loadavg": os.getloadavg(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "gl": os.environ.get("MUJOCO_GL"),
    }
    try:
        out["commit"] = subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"], capture_output=True, text=True
        ).stdout.strip()
    except OSError:
        pass
    if torch.cuda.is_available():
        out["gpu"] = torch.cuda.get_device_name()
        out["driver"] = subprocess.run(
            ["nvidia-smi", "--query-gpu=driver_version", "--format=csv,noheader"], capture_output=True, text=True
        ).stdout.strip()
    return out


def _timed(
    units,
    *,
    seconds: float,
    repeats: int,
    sync=None,
    warmup_s: float = WARMUP_S,
    budget_s: float = 60.0,
    warmup_items: int = 0,
    window_items: int = 0,
) -> dict:
    """Items per second over up to ``repeats`` windows of about ``seconds``; each ``next(units)`` does work and returns
    its items. With ``window_items``, a window ends at the first whole unit at or after that many items, and
    ``seconds`` is not used. After the first window, no new window starts once the windows took ``budget_s`` in total:
    a unit of a slow source can take minutes, and one window of whole units is then enough."""
    # Warm up for at least warmup_s and warmup_items items: a kernel can still be compiling after the seconds.
    # WARMUP_MAX_S (seconds, 0 = no limit) stops the item count early for sources that take hours per item count.
    t0 = time.perf_counter()
    end, warmed, cap = t0 + warmup_s, 0, float(os.environ.get("WARMUP_MAX_S") or 0)
    while time.perf_counter() < end or (warmed < warmup_items and not (cap and time.perf_counter() - t0 > cap)):
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


def _cycle(make):
    """Endless passes over the iterable that ``make`` returns."""
    while True:
        yield from make()


def _worker_for(job) -> tuple[int, float]:
    """Draw this worker's rows 8 at a time, wrapping around, for ``seconds`` or, when ``count`` is set, for at least
    ``count`` frames; return the frames and the elapsed time."""
    (s, e), seconds, render, count = job
    t = time.perf_counter()
    done, i = 0, s
    while (done < count) if count else (time.perf_counter() - t < seconds):
        j = min(i + 8, e)
        done += render((i, j))
        i = s if j >= e else j
    return done, time.perf_counter() - t


def _pool_timed(
    pool,
    render,
    spans,
    *,
    seconds: float,
    repeats: int,
    warmup_s: float = WARMUP_S,
    warmup_items: int = 0,
    window_items: int = 0,
) -> dict:
    """Frames per second of all workers together: one worker per span, every worker timed over the same window.

    With ``window_items`` (and ``warmup_items``) each worker draws at least that many frames instead of drawing for
    ``seconds``."""
    if warmup_s > 0 or warmup_items:
        pool.map(_worker_for, [(sp, warmup_s, render, warmup_items) for sp in spans], chunksize=1)
    rates = []
    for _ in range(repeats):
        out = pool.map(_worker_for, [(sp, seconds, render, window_items) for sp in spans], chunksize=1)
        rates.append(sum(d / dt for d, dt in out))
    return {"rate": statistics.median(rates), "rates": rates, "repeats": repeats}


def _load_states(path: Path):
    blob = torch.load(path, weights_only=False)
    return blob


def _snv():
    import stable_envs as snv

    try:
        import stable_worldmodel  # noqa: F401  registers swm/ ids used by some references
    except ImportError:
        pass
    try:
        from tests.helpers.reference.certification import install_on_classes

        install_on_classes()
    except Exception:  # noqa: BLE001
        pass
    from benchmarks.swm_dmc import install

    install()
    return snv


def _class(name: str):
    import stable_envs as snv

    return snv.info(name)["class"]


def _sources():
    """How each source library draws (``tests/helpers/reference/sources.py``); the repository root is on the path."""
    from tests.helpers.reference import sources

    return sources


def source_frames(cls, ref, inputs) -> tuple[np.ndarray, np.ndarray]:
    """The source's frames of ``inputs`` and a mask of the frames it could draw.

    A source whose ``reference_render`` takes ``return_flags`` (PyFlyt Rocket) flags the inputs it cannot reach.
    """
    fn = getattr(cls, "reference_render", None)
    if fn is not None and "return_flags" in inspect.signature(fn).parameters:
        frames, flags = fn(ref, inputs, return_flags=True)
        return np.asarray(frames), ~np.asarray(flags, bool)
    frames = np.asarray(_sources().reference_render(cls, ref, inputs))
    return frames, np.ones(len(frames), bool)


def cmd_states(args) -> None:
    """Save ``n`` frame inputs per env: rows from the family sampler, and full States where a device needs them.

    A class with a sampler (``sample_states``) gives rows that spread over its whole reachable space. Classes
    without a sampler, and devices where ``env.render`` refuses rows (Rocket on CPU), use full States from seeded
    resets and random actions, saved every ``--every`` steps up to ``--steps``.
    """
    snv = _snv()
    sources = _sources()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.envs:
        path = out / f"{name}.pt"
        if path.exists():
            print(f"have {path}", flush=True)
            continue
        cls = _class(name)
        size = cls.native_size or args.size
        reference = sources.has_source_renderer(cls)
        t = time.time()
        try:
            with snv.make(name, num_envs=args.num_envs, size=size, device="cpu") as env:
                try:
                    rows = env.sample_states(args.n, rng=np.random.default_rng(args.seed))
                except NotImplementedError:
                    rows = None
                refused = [] if rows is None else _rows_refused(snv, env, rows, args.devices)
                states = inputs = None
                if rows is None or refused:
                    states, inputs = _rollouts(env, args.n, args.seed, args.steps, args.every, reference)
        except Exception as e:  # noqa: BLE001
            print(f"FAIL states {name}: {type(e).__name__}: {e}", flush=True)
            traceback.print_exc()
            (out / f"{name}.error.txt").write_text(f"{type(e).__name__}: {e}\n")
            continue
        blob = {"rows": rows, "states": states, "inputs": inputs, "name": name, "seed": args.seed, "size": size}
        blob["rows_refused"] = refused
        blob["spread"] = {k: _spread(v) for k, v in (("rows", rows), ("states", inputs)) if v is not None}
        torch.save(blob, path)
        print(f"saved {path} in {time.time() - t:.1f}s spread {blob['spread']}", flush=True)


def _refuses_rows(env, rows) -> bool:
    """Whether ``env.render`` refuses compact rows on this env's device (a family's clear error for that form)."""
    try:
        env.render(torch.as_tensor(np.asarray(rows[:1]), device=env.device))
    except (TypeError, NotImplementedError):
        return True
    return False


def _rows_refused(snv, env, rows, devices) -> list[str]:
    """The devices of ``devices`` whose ``env.render`` refuses rows; CUDA counts only where it is available."""
    refused = []
    for device in devices:
        if device == "cpu":
            refused += ["cpu"] if _refuses_rows(env, rows) else []
        elif torch.cuda.is_available():
            with snv.make(env.class_name, num_envs=1, size=env.size, device=device) as probe:
                refused += [device] if _refuses_rows(probe, rows) else []
    return refused


def _spread(rows) -> dict:
    """How far the frame inputs spread: the standard deviation and range of each render-row column. Inputs that are
    one dict per instance (PushT) use their body poses."""
    if not isinstance(rows, np.ndarray):
        keys = sorted(k for k in rows[0] if k.startswith("current/body/"))
        rows = [np.concatenate([np.ravel(r[k]) for k in keys]) for r in rows]
    rows = np.asarray(rows, np.float64)
    return {
        "std": rows.std(0).round(4).tolist(),
        "min": rows.min(0).round(4).tolist(),
        "max": rows.max(0).round(4).tolist(),
    }


def _rollouts(env, n: int, seed: int, steps: int, every: int, reference: bool):
    """``n`` full States saved every ``every`` random actions after seeded resets, up to ``steps`` actions per reset,
    and what the source draws for them."""
    sources = _sources()
    rng = np.random.default_rng(seed)
    states, inputs, have = [], [], 0
    while have < n:
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for t in range(steps + 1):
            # The first State is saved after ``every`` actions: right after a reset some classes (Rocket) hold
            # pose rows that the first step fills.
            if t and t % every == 0:
                st = env.get_state()
                if reference:
                    inputs.append(sources.reference_inputs(env, st))
                states.append(st.cpu())
                have += int(st.batch_size[0])
                if have >= n:
                    break
            env.step(env.action_spec.sample(env.B, rng), frames=None)
    if not reference:
        return _cat_states(states)[:n], None
    joined = np.concatenate(inputs) if isinstance(inputs[0], np.ndarray) else [x for part in inputs for x in part]
    return _cat_states(states)[:n], joined[:n]


def _cat_states(states):
    """Concatenate State batches whose growing fields (PushT's action history) differ in length.

    Shorter fields are zero-padded along their second axis; a history's ``valid`` mask is then False on the padding.
    """
    lengths = {}
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


def _frames_of(blob, device: str):
    """``(kind, data)`` that stable-envs draws on ``device``: the sampled rows, or the full States where ``render``
    refused rows on that device when the inputs were saved."""
    refused = blob.get("rows_refused") or []
    if blob["rows"] is not None and (blob["states"] is None or torch.device(device).type not in refused):
        return "rows", blob["rows"]
    return "states", blob["states"]


#: The input form that ours draws, as the cell JSON records it.
INPUT_FORM = {"rows": "rows", "states": "state"}


def _on_device(kind: str, data, device):
    return torch.as_tensor(data, device=device) if kind == "rows" else data.to(device)


def _source_inputs(blob, kind: str):
    return blob["rows"] if kind == "rows" else blob["inputs"]


def _chunks(n: int, b: int):
    return [(i, min(i + b, n)) for i in range(0, n, b)]


def _unsupported(e: Exception) -> dict:
    return {"status": "unsupported", "error": f"{type(e).__name__}: {e}"[:500]}


def cmd_ours(args) -> None:
    """Sweep every stored input through stable-envs at each batch size."""
    snv = _snv()
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    # Ours draws stored inputs the way the package draws offline: snv.FrameLoader renders ahead in large chunks and
    # hands out b frames at a time. --per-batch draws each batch with its own env.render call instead.
    loader_cls = getattr(snv, "FrameLoader", None)
    results = {
        "measurement": f"ours-{dev.type}",
        "host": _host(),
        "envs": {},
        "repeats_max": args.repeats,
    }
    for name in args.envs:
        path = Path(args.states) / f"{name}.pt"
        if not path.exists():
            results["envs"][name] = {"status": "skipped", "reason": "no state file"}
            continue
        blob = _load_states(path)
        kind, data = _frames_of(blob, args.device)
        data = _on_device(kind, data, dev)
        rows = []
        for b in args.batches:
            env = None
            try:
                env = snv.make(name, num_envs=b, size=blob["size"], device=args.device)
                n = len(data)
                # The stored inputs are sliced into batches before the timed windows: slicing a State is not drawing.
                pieces = [data[s:e] for s, e in _chunks(n, b)]
                frame = env.render(pieces[0])
                shape = list(frame.shape[1:])
                del frame
                loader = None if args.per_batch or loader_cls is None else loader_cls(env, data, b)

                def sweep():
                    if loader is not None:
                        # One unit is one render-ahead chunk handed out b frames at a time, so that every window
                        # holds whole chunks and pays for every frame it receives.
                        got, per = 0, loader.chunk_size
                        for frames in loader:
                            got += int(frames.shape[0])
                            if got >= per:
                                yield got
                                got = 0
                        if got:
                            yield got
                        return
                    for piece in pieces:
                        env.render(piece)
                        yield len(piece)

                t = _timed(
                    _cycle(sweep),
                    seconds=args.seconds,
                    repeats=args.repeats_small if b <= args.small_max_batch else args.repeats,
                    sync=sync,
                    warmup_s=args.warmup,
                    warmup_items=args.warmup_steps * b,
                    window_items=args.timed_steps * b,
                )
                row = {"batch": b, "status": "ok", "frames": n, "frame_shape": shape, "fps": t["rate"], **t}
                row["draw"] = "per-batch" if loader is None else "FrameLoader"
                row["render_rows"] = b if loader is None else loader.chunk_size
                if dev.type == "cuda":
                    row["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
            except torch.cuda.OutOfMemoryError as e:
                row = {"batch": b, "status": "oom", "error": str(e)[:300]}
            except (NotImplementedError, TypeError) as e:
                row = {"batch": b, **_unsupported(e)}
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                row = {"batch": b, "status": "oom" if "out of memory" in msg.lower() else "error", "error": msg[:500]}
            finally:
                if env is not None:
                    env.close()
                if dev.type == "cuda":
                    torch.cuda.empty_cache()
                    torch.cuda.reset_peak_memory_stats()
            print(f"{name} {dev.type} b={b} {row['status']} {row.get('fps', 0):.0f} fps", flush=True)
            rows.append(row)
        entry = {"size": blob["size"], "kind": kind, "input": INPUT_FORM[kind], "spread": blob["spread"]}
        results["envs"][name] = {**entry, "points": rows}
    Path(args.output).write_text(json.dumps(results, indent=1))


_W: dict = {}


def _worker_init(name: str, states: str, size: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    _snv()
    cls = _class(name)
    _W["cls"] = cls
    _W["ref"] = _sources().reference_env(cls, size)
    blob = _load_states(Path(states) / f"{name}.pt")
    _W["inputs"] = blob["rows"] if blob["rows"] is not None else blob["inputs"]


def _worker_render(span) -> int:
    s, e = span
    sources = _sources()
    for i in range(s, e, 64):
        sources.reference_render(_W["cls"], _W["ref"], _W["inputs"][i : min(i + 64, e)])
    return e - s


def _source_skip(cls, blob, gpu: bool) -> str | None:
    """Why the source cannot draw ``cls`` here, or None."""
    sources = _sources()
    if not sources.has_source_renderer(cls) or (blob["rows"] is None and blob["inputs"] is None):
        return "no source renderer"
    gpu_render = sources.reference_gpu_render(cls)
    if gpu and gpu_render != "mujoco-egl":
        return f"the offline source draws on CPU (reference_gpu_render={gpu_render})"
    return None


def cmd_source(args) -> None:
    """Sweep the stored inputs through the source renderer in ``procs`` processes; ``--gpu`` draws with EGL."""
    _snv()
    results = {
        "measurement": f"source-{'gpu' if args.gpu else 'cpu'}-{args.procs}p",
        "host": _host(),
        "envs": {},
        "repeats_max": args.repeats,
    }
    for name in args.envs:
        path = Path(args.states) / f"{name}.pt"
        if not path.exists():
            results["envs"][name] = {"status": "skipped", "reason": "no state file"}
            continue
        blob = _load_states(path)
        reason = _source_skip(_class(name), blob, args.gpu)
        if reason:
            results["envs"][name] = {"status": "skipped", "reason": reason}
            continue
        n = len(blob["rows"] if blob["rows"] is not None else blob["inputs"])
        try:
            if args.procs == 1:
                _worker_init(name, args.states, blob["size"])
                first = _sources().reference_render(_W["cls"], _W["ref"], _W["inputs"][:1])
                shape = list(np.asarray(first).shape[1:])
                rows8 = lambda: (_worker_render((i, min(i + 8, n))) for i in range(0, n, 8))  # noqa: E731
                t = _timed(
                    _cycle(rows8),
                    seconds=args.seconds,
                    repeats=args.repeats,
                    warmup_s=args.warmup,
                    warmup_items=args.warmup_steps * 8,
                    window_items=args.timed_steps * 8,
                )
            else:
                ctx = mp.get_context("spawn")
                spans = [(n * k // args.procs, n * (k + 1) // args.procs) for k in range(args.procs)]
                with ctx.Pool(
                    args.procs, initializer=_worker_init, initargs=(name, args.states, blob["size"])
                ) as pool:
                    shape = None
                    t = _pool_timed(
                        pool,
                        _worker_render,
                        spans,
                        seconds=args.seconds,
                        repeats=args.repeats,
                        warmup_s=args.warmup,
                        warmup_items=-(-args.warmup_steps * 8 // args.procs),
                        window_items=-(-args.timed_steps * 8 // args.procs),
                    )
            row = {"status": "ok", "frames": n, "fps": t["rate"], "frame_shape": shape, **t}
        except Exception as e:  # noqa: BLE001
            row = {"status": "error", "error": f"{type(e).__name__}: {e}"[:500]}
        print(f"{name} {results['measurement']} {row['status']} {row.get('fps', 0):.0f} fps", flush=True)
        results["envs"][name] = row
        _W.clear()
    Path(args.output).write_text(json.dumps(results, indent=1))


def cmd_mse(args) -> None:
    """Per-frame MSE of stable-envs frames against the source's for the first ``n`` stored inputs, on the 0-255 scale."""
    snv = _snv()
    from stable_envs.testing import conformance as C

    results = {"measurement": f"mse-{args.device}", "host": _host(), "envs": {}}
    for name in args.envs:
        blob = _load_states(Path(args.states) / f"{name}.pt")
        cls = _class(name)
        reason = _source_skip(cls, blob, gpu=False)
        if reason:
            results["envs"][name] = {"status": "skipped", "reason": reason}
            continue
        try:
            kind, data = _frames_of(blob, args.device)
            inputs = _source_inputs(blob, kind)
            n = min(args.n, len(inputs))
            # The source draws before any stable-envs env exists in this process (see online_ladder).
            ref = _sources().reference_env(cls, blob["size"])
            theirs, keep = source_frames(cls, ref, inputs[:n])
            with snv.make(name, num_envs=n, size=blob["size"], device=args.device) as env:
                ours = env.render(_on_device(kind, data[:n], env.device)).cpu().numpy()
            per_frame = C.frame_mse(ours[keep], theirs[keep])
            row = {"status": "ok", "kind": kind, "input": INPUT_FORM[kind], "spread": blob["spread"].get(kind)}
            row.update(frames=int(keep.sum()), excluded_unreachable=int((~keep).sum()))
            row.update(C.mse_summary(per_frame))
            if args.keep:
                order = np.argsort(per_frame)
                pick = [int(order[-1]), int(order[len(order) // 2]), 0]
                np.savez_compressed(
                    Path(args.keep) / f"{name}-{args.device}.npz",
                    ours=ours[keep][pick],
                    theirs=theirs[keep][pick],
                    mse=per_frame[pick],
                    labels=np.array(["worst", "median", "first"]),
                )
        except (NotImplementedError, TypeError) as e:
            row = _unsupported(e)
        except Exception as e:  # noqa: BLE001
            row = {"status": "error", "error": f"{type(e).__name__}: {e}"[:500]}
        print(f"{name} mse {args.device} {row}", flush=True)
        results["envs"][name] = row
    Path(args.output).write_text(json.dumps(results, indent=1))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--envs", nargs="+", default=[])
    common.add_argument("--states", default="states")
    common.add_argument("--repeats", type=int, default=10)
    common.add_argument("--seconds", type=float, default=10.0, help="length of one timed window")
    common.add_argument("--warmup", type=float, default=WARMUP_S, help="seconds of untimed work before the windows")
    s = sub.add_parser("states", parents=[common])
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=10_000)
    s.add_argument("--seed", type=int, default=20261022)
    s.add_argument("--size", type=int, default=224)
    s.add_argument("--num-envs", type=int, default=64, help="environments of the rollouts that save full States")
    s.add_argument("--steps", type=int, default=1000, help="most random actions after each seeded reset")
    s.add_argument("--every", type=int, default=30, help="random actions between two saved States")
    s.add_argument("--devices", nargs="+", default=["cpu", "cuda"], help="devices that draw these inputs")
    o = sub.add_parser("ours", parents=[common])
    o.add_argument("--device", default="cuda")
    o.add_argument("--batches", type=int, nargs="+", default=[1, 256, 512, 1024, 2048, 4096])
    o.add_argument("--threads", type=int, default=8)
    o.add_argument("--per-batch", action="store_true", help="one env.render call per batch, without snv.FrameLoader")
    o.add_argument("--repeats-small", type=int, default=0, help="windows of a batch up to --small-max-batch")
    o.add_argument("--small-max-batch", type=int, default=0, help="largest batch that takes --repeats-small windows")
    o.add_argument("--output", required=True)
    r = sub.add_parser("source", parents=[common])
    r.add_argument("--procs", type=int, default=1)
    r.add_argument("--gpu", action="store_true", help="the source draws with MuJoCo EGL on the GPU")
    r.add_argument("--output", required=True)
    for q in (o, r):
        q.add_argument("--warmup-steps", type=int, default=0, help="least untimed batches (source: 8-row draws)")
        q.add_argument("--timed-steps", type=int, default=0, help="batches per timed window instead of --seconds")
    m = sub.add_parser("mse", parents=[common])
    m.add_argument("--device", default="cuda")
    m.add_argument("--n", type=int, default=256)
    m.add_argument("--keep", default=None, help="save the worst, median and first frame pairs to this directory")
    m.add_argument("--output", required=True)
    args = p.parse_args()
    {
        "states": cmd_states,
        "ours": cmd_ours,
        "source": cmd_source,
        "mse": cmd_mse,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
