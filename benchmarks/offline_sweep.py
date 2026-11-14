"""Time full sweeps over a stored state file: stable-envs on CPU or CUDA, the source renderer, and JPEG/PNG decoding."""

from __future__ import annotations

import argparse
import inspect
import io
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

from benchmarks.draw import draw
from benchmarks.jax_source import CraftaxSource, is_jax_source

ROOT = Path(__file__).resolve().parents[1]
#: Seconds of untimed work before the first timed window of an arm.
WARMUP_S = 2.0


def _main_sha() -> str | None:
    """The main commit under test: ``.bench_main_sha`` in the checkout, else ``SE_MAIN_SHA``."""
    pinned = ROOT / ".bench_main_sha"
    return pinned.read_text().strip() if pinned.exists() else os.environ.get("SE_MAIN_SHA")


def _host() -> dict:
    out = {
        "host": platform.node(),
        "loadavg": os.getloadavg(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
        "main_sha": _main_sha(),
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


def _sweeps(fn, *, warmup: int, repeats: int, budget_s: float, sync=None) -> dict:
    """Run ``fn`` warmup times untimed, then up to ``repeats`` timed sweeps; drop to 3 when one exceeds budget_s."""
    for _ in range(warmup):
        fn()
    times = []
    target = repeats
    while len(times) < target:
        if sync:
            sync()
        t = time.perf_counter()
        fn()
        if sync:
            sync()
        times.append(time.perf_counter() - t)
        if times[0] > budget_s:
            target = min(repeats, 3)
    return {"median_s": statistics.median(times), "samples_s": times, "repeats": len(times), "warmup": warmup}


def _timed(
    units, *, seconds: float, repeats: int, sync=None, warmup_s: float = WARMUP_S, budget_s: float = 60.0
) -> dict:
    """Items per second over up to ``repeats`` windows of about ``seconds``; each ``next(units)`` does work and returns
    its items. After the first window, no new window starts once the windows took ``budget_s`` in total: a unit of a
    slow source can take minutes, and one window of whole units is then enough."""
    end = time.perf_counter() + warmup_s
    while time.perf_counter() < end:
        next(units)
    rates, windows, items = [], [], []
    cpu0 = time.process_time()
    for _ in range(repeats):
        if windows and sum(windows) >= budget_s:
            break
        if sync:
            sync()
        t = time.perf_counter()
        done = 0
        while time.perf_counter() - t < seconds:
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
        # CPU seconds of this process (all its threads) per wall second of the windows.
        "cpu_util": (time.process_time() - cpu0) / sum(windows),
    }


def _cycle(make):
    """Endless passes over the iterable that ``make`` returns."""
    while True:
        yield from make()


def _worker_for(job) -> tuple[int, float]:
    """Draw this worker's rows 8 at a time, wrapping around, for ``seconds``; return the frames and the elapsed time."""
    (s, e), seconds, render = job
    t = time.perf_counter()
    done, i = 0, s
    while time.perf_counter() - t < seconds:
        j = min(i + 8, e)
        done += render((i, j))
        i = s if j >= e else j
    return done, time.perf_counter() - t


def _pool_timed(pool, render, spans, *, seconds: float, repeats: int, warmup_s: float = WARMUP_S) -> dict:
    """Frames per second of all workers together: one worker per span, every worker timed over the same window."""
    if warmup_s > 0:
        pool.map(_worker_for, [(sp, warmup_s, render) for sp in spans], chunksize=1)
    rates = []
    for _ in range(repeats):
        out = pool.map(_worker_for, [(sp, seconds, render) for sp in spans], chunksize=1)
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
            with snv.make(name, num_envs=args.lanes, size=size, device="cpu") as env:
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
        draw(env, torch.as_tensor(np.asarray(rows[:1]), device=env.device))
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
    one dict per lane (PushT) use their body poses."""
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
    results = {
        "arm": f"ours-{dev.type}",
        "host": _host(),
        "envs": {},
        "budget_s": args.budget,
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
                render = lambda s, e: draw(env, data[s:e])  # noqa: E731
                n = len(data)
                spans = _chunks(n, b)
                frame = render(*spans[0])
                shape = list(frame.shape[1:])
                del frame

                def sweep():
                    for s, e in spans:
                        render(s, e)
                        yield e - s

                t = _timed(_cycle(sweep), seconds=args.seconds, repeats=args.repeats, sync=sync, warmup_s=args.warmup)
                row = {"batch": b, "status": "ok", "frames": n, "frame_shape": shape, "fps": t["rate"], **t}
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
    """Why the source arm cannot run for ``cls``, or None."""
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
        "arm": f"source-{'gpu' if args.gpu else 'cpu'}-{args.procs}p",
        "host": _host(),
        "envs": {},
        "budget_s": args.budget,
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
                t = _timed(_cycle(rows8), seconds=args.seconds, repeats=args.repeats, warmup_s=args.warmup)
            else:
                ctx = mp.get_context("spawn")
                spans = [(n * k // args.procs, n * (k + 1) // args.procs) for k in range(args.procs)]
                with ctx.Pool(
                    args.procs, initializer=_worker_init, initargs=(name, args.states, blob["size"])
                ) as pool:
                    shape = None
                    t = _pool_timed(
                        pool, _worker_render, spans, seconds=args.seconds, repeats=args.repeats, warmup_s=args.warmup
                    )
            row = {"status": "ok", "frames": n, "fps": t["rate"], "frame_shape": shape, **t}
        except Exception as e:  # noqa: BLE001
            row = {"status": "error", "error": f"{type(e).__name__}: {e}"[:500]}
        print(f"{name} {results['arm']} {row['status']} {row.get('fps', 0):.0f} fps", flush=True)
        results["envs"][name] = row
        _W.clear()
    Path(args.output).write_text(json.dumps(results, indent=1))


def cmd_source_jax(args) -> None:
    """Sweep the stored States through the batched JAX source (``benchmarks/jax_source.py``) at each batch size."""
    import jax

    _snv()
    tag = "gpu" if args.device.startswith("cuda") else "cpu"
    results = {
        "arm": f"source-jax-{tag}",
        "host": _host(),
        "envs": {},
        "budget_s": args.budget,
        "repeats_max": args.repeats,
    }
    for name in args.envs:
        path = Path(args.states) / f"{name}.pt"
        if not path.exists() or not is_jax_source(name):
            results["envs"][name] = {"status": "skipped", "reason": "no state file or no batched JAX source"}
            continue
        blob = _load_states(path)
        rows = []
        try:
            src = CraftaxSource(name, args.device)
            states = src.state_from(blob["states"])
            n = int(blob["states"].batch_size[0])
        except Exception as e:  # noqa: BLE001
            results["envs"][name] = {"status": "error", "error": f"{type(e).__name__}: {e}"[:500]}
            continue
        for b in args.batches:
            try:
                spans = _chunks(n, b)
                # Each batch is sliced before timing; the source keeps its states on its device.
                parts = [jax.tree.map(lambda x, s=s, e=e: x[s:e], states) for s, e in spans]
                shape = list(jax.block_until_ready(src.render(parts[0])).shape[1:])

                def sweep(parts=parts, spans=spans):
                    for part, (s, e) in zip(parts, spans):
                        jax.block_until_ready(src.render(part))
                        yield e - s

                t = _timed(_cycle(sweep), seconds=args.seconds, repeats=args.repeats, warmup_s=args.warmup)
                row = {"batch": b, "status": "ok", "frames": n, "frame_shape": shape, "fps": t["rate"], **t}
                del parts
            except Exception as e:  # noqa: BLE001
                msg = f"{type(e).__name__}: {e}"
                row = {"batch": b, "status": "oom" if "out of memory" in msg.lower() else "error", "error": msg[:500]}
            print(f"{name} {results['arm']} b={b} {row['status']} {row.get('fps', 0):.0f} fps", flush=True)
            rows.append(row)
        results["envs"][name] = {
            "size": blob["size"],
            "source": src.upstream_id,
            "device": str(src.device),
            "points": rows,
        }
    Path(args.output).write_text(json.dumps(results, indent=1))


def cmd_mse(args) -> None:
    """Per-frame MSE of stable-envs frames against the source's for the first ``n`` stored inputs, on the 0-255 scale."""
    snv = _snv()
    from stable_envs.testing import conformance as C

    results = {"arm": f"mse-{args.device}", "host": _host(), "envs": {}}
    for name in args.envs:
        blob = _load_states(Path(args.states) / f"{name}.pt")
        cls = _class(name)
        jax_source = is_jax_source(name)
        reason = None if jax_source else _source_skip(cls, blob, gpu=False)
        if reason:
            results["envs"][name] = {"status": "skipped", "reason": reason}
            continue
        try:
            kind, data = _frames_of(blob, args.device)
            if jax_source:
                # The upstream renderer on CPU draws the same States.
                n = min(args.n, len(data))
                src = CraftaxSource(name, "cpu")
                theirs = np.asarray(src.render(src.state_from(data[:n])))
                keep = np.ones(n, bool)
            else:
                inputs = _source_inputs(blob, kind)
                n = min(args.n, len(inputs))
                # The source draws before any stable-envs env exists in this process (see online_ladder).
                ref = _sources().reference_env(cls, blob["size"])
                theirs, keep = source_frames(cls, ref, inputs[:n])
            with snv.make(name, num_envs=n, size=blob["size"], device=args.device) as env:
                ours = draw(env, _on_device(kind, data[:n], env.device)).cpu().numpy()
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


class _Files(torch.utils.data.Dataset):
    def __init__(self, files):
        self.files = files

    def __len__(self):
        return len(self.files)

    def __getitem__(self, i):
        from PIL import Image

        with Image.open(self.files[i]) as im:
            return torch.from_numpy(np.asarray(im.convert("RGB")))


def cmd_codec(args) -> None:
    """Encode stable-envs frames as JPEG/PNG: bytes per frame for every env, decode throughput for ``--decode`` envs."""
    from PIL import Image

    snv = _snv()
    dev = torch.device(args.device)
    results = {"arm": "codec", "host": _host(), "envs": {}}
    for name in args.envs:
        path = Path(args.states) / f"{name}.pt"
        if not path.exists():
            continue
        blob = _load_states(path)
        decode = name in args.decode
        kind, data = _frames_of(blob, args.device)
        n = len(data) if decode else args.sample
        row = {"frames_encoded": n, "input": INPUT_FORM[kind]}
        try:
            with snv.make(name, num_envs=256, size=blob["size"], device=args.device) as env:
                data = _on_device(kind, data, dev)
                frames = np.concatenate([draw(env, data[s:e]).cpu().numpy() for s, e in _chunks(n, 256)])
                if kind == "rows":
                    row["state_bytes"] = int(data.shape[1]) * 4
                else:
                    row["state_bytes"] = int(sum(v.numel() * v.element_size() for v in data[0].values(True, True)))
            row["raw_bytes"] = int(frames[0].nbytes)
            for fmt, kw in (("jpeg", {"quality": 95}), ("png", {})):
                d = Path(args.out) / fmt / name
                sizes = []
                if decode:
                    d.mkdir(parents=True, exist_ok=True)
                for i, f in enumerate(frames):
                    buf = io.BytesIO()
                    Image.fromarray(f).save(buf, format=fmt.upper(), **kw)
                    sizes.append(buf.tell())
                    if decode:
                        (d / f"{i:05d}.{fmt}").write_bytes(buf.getvalue())
                row[f"{fmt}_bytes"] = float(np.mean(sizes))
                if decode:
                    files = sorted(str(p) for p in d.iterdir())
                    loader = torch.utils.data.DataLoader(
                        _Files(files), batch_size=256, num_workers=8, persistent_workers=True, prefetch_factor=4
                    )

                    def sweep():
                        for _ in loader:
                            pass

                    t = _sweeps(sweep, warmup=1, repeats=args.repeats, budget_s=args.budget)
                    row[f"{fmt}_decode"] = {"fps": len(files) / t["median_s"], **t}
                    del loader
            row["status"] = "ok"
        except (NotImplementedError, TypeError) as e:
            row = _unsupported(e)
        except Exception as e:  # noqa: BLE001
            row = {"status": "error", "error": f"{type(e).__name__}: {e}"[:500]}
        print(f"{name} codec {json.dumps({k: v for k, v in row.items() if 'decode' not in k})}", flush=True)
        results["envs"][name] = row
    Path(args.output).write_text(json.dumps(results, indent=1))


def cmd_list(args) -> None:
    snv = _snv()
    names = list(snv.list_envs())
    if args.family:
        names = [n for n in names if (snv.info(n)["family"] or "") in args.family]
    print(" ".join(names))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--envs", nargs="+", default=[])
    common.add_argument("--states", default="states")
    common.add_argument("--repeats", type=int, default=10)
    common.add_argument("--budget", type=float, default=300.0, help="seconds per sweep above which 3 repeats are kept")
    common.add_argument("--seconds", type=float, default=10.0, help="length of one timed window")
    common.add_argument("--warmup", type=float, default=WARMUP_S, help="seconds of untimed work before the windows")
    s = sub.add_parser("states", parents=[common])
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=10_000)
    s.add_argument("--seed", type=int, default=20261022)
    s.add_argument("--size", type=int, default=224)
    s.add_argument("--lanes", type=int, default=64)
    s.add_argument("--steps", type=int, default=1000, help="most random actions after each seeded reset")
    s.add_argument("--every", type=int, default=30, help="random actions between two saved States")
    s.add_argument("--devices", nargs="+", default=["cpu", "cuda"], help="devices whose arms read these inputs")
    o = sub.add_parser("ours", parents=[common])
    o.add_argument("--device", default="cuda")
    o.add_argument("--batches", type=int, nargs="+", default=[1, 256, 512, 1024, 2048, 4096])
    o.add_argument("--threads", type=int, default=8)
    o.add_argument("--output", required=True)
    r = sub.add_parser("source", parents=[common])
    r.add_argument("--procs", type=int, default=1)
    r.add_argument("--gpu", action="store_true", help="the source draws with MuJoCo EGL on the GPU")
    r.add_argument("--output", required=True)
    c = sub.add_parser("codec", parents=[common])
    c.add_argument("--out", required=True)
    c.add_argument("--decode", nargs="*", default=[])
    c.add_argument("--sample", type=int, default=256)
    c.add_argument("--device", default="cuda")
    c.add_argument("--output", required=True)
    j = sub.add_parser("source-jax", parents=[common])
    j.add_argument("--device", default="cpu")
    j.add_argument("--batches", type=int, nargs="+", default=[1, 256, 512, 1024, 2048])
    j.add_argument("--output", required=True)
    m = sub.add_parser("mse", parents=[common])
    m.add_argument("--device", default="cuda")
    m.add_argument("--n", type=int, default=256)
    m.add_argument("--keep", default=None, help="save the worst, median and first frame pairs to this directory")
    m.add_argument("--output", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--family", nargs="*", default=[])
    args = p.parse_args()
    {
        "states": cmd_states,
        "ours": cmd_ours,
        "source": cmd_source,
        "source-jax": cmd_source_jax,
        "codec": cmd_codec,
        "mse": cmd_mse,
        "list": cmd_list,
    }[args.cmd](args)


if __name__ == "__main__":
    main()
