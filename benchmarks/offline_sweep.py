"""Time full sweeps over a stored state file: stable-envs on CPU or CUDA, the source renderer, and JPEG/PNG decoding."""

from __future__ import annotations

import argparse
import io
import json
import multiprocessing as mp
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]


def _host() -> dict:
    out = {
        "host": platform.node(),
        "loadavg": os.getloadavg(),
        "affinity": sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
        "torch": torch.__version__,
        "python": sys.version.split()[0],
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


def cmd_states(args) -> None:
    """Draw ``n`` render states per env with a fixed seed and save them."""
    snv = _snv()
    from stable_envs.testing import conformance as C

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    for name in args.envs:
        path = out / f"{name}.pt"
        if path.exists():
            print(f"have {path}", flush=True)
            continue
        cls = snv.registry.REGISTRY[name]
        size = int(getattr(cls, "fixed_size", None) or args.size)
        t = time.time()
        try:
            try:
                env = snv.make(name, num_envs=args.lanes, size=size, device=args.device)
            except ValueError as e:
                if "renders only at" not in str(e):
                    raise
                size = None
                env = snv.make(name, num_envs=args.lanes, device=args.device)
            try:
                if int(getattr(env, "state_dim", 0) or 0) > 0:
                    rows = C.sample_render_rows(env, args.n, rng=np.random.default_rng(args.seed))
                    blob = {"kind": "rows", "rows": torch.as_tensor(rows)}
                    try:
                        env._render(blob["rows"][:1].to(env.device))
                    except ValueError as e:
                        if "renders only at" not in str(e):
                            raise
                        size = None
                else:
                    blob = {"kind": "state", "state": _rollout_states(env, args.n, args.seed)}
            finally:
                env.close()
        except Exception as e:  # noqa: BLE001
            print(f"FAIL states {name}: {type(e).__name__}: {e}", flush=True)
            (out / f"{name}.error.txt").write_text(f"{type(e).__name__}: {e}\n")
            continue
        blob.update(name=name, seed=args.seed, n=args.n, size=size)
        torch.save(blob, path)
        print(f"saved {path} kind={blob['kind']} in {time.time() - t:.1f}s", flush=True)


def _rollout_states(env, n: int, seed: int):
    """Full States from seeded resets and short random rollouts, stacked to ``n`` rows."""
    rng = np.random.default_rng(seed)
    parts, have = [], 0
    while have < n:
        env.reset(seed=int(rng.integers(0, 2**31 - 1)))
        for _ in range(int(rng.integers(0, 20))):
            env.step(env.action_spec.sample(env.B, rng), frames=None)
        st = env.to_render_state().cpu()
        parts.append(st)
        have += int(st.batch_size[0])
    return torch.cat(parts)[:n]


def _chunks(n: int, b: int):
    return [(i, min(i + b, n)) for i in range(0, n, b)]


def cmd_ours(args) -> None:
    """Sweep every stored state through stable-envs at each batch size."""
    snv = _snv()
    if args.device == "cpu":
        torch.set_num_threads(args.threads)
    dev = torch.device(args.device)
    sync = torch.cuda.synchronize if dev.type == "cuda" else None
    results = {"arm": f"ours-{dev.type}", "host": _host(), "envs": {}}
    for name in args.envs:
        path = Path(args.states) / f"{name}.pt"
        if not path.exists():
            results["envs"][name] = {"status": "skipped", "reason": "no state file"}
            continue
        blob = _load_states(path)
        rows = []
        for b in args.batches:
            env = None
            try:
                env = snv.make(name, num_envs=b, size=blob["size"], device=args.device)
                if blob["kind"] == "rows":
                    data = blob["rows"].to(dev)
                    render = lambda s, e: env._render(data[s:e])  # noqa: E731
                else:
                    data = blob["state"].to(dev)
                    render = lambda s, e: env.render(data[s:e])  # noqa: E731
                n = len(data)
                spans = _chunks(n, b)
                frame = render(*spans[0])
                shape = list(frame.shape[1:])
                del frame

                def sweep():
                    for s, e in spans:
                        render(s, e)

                t = _sweeps(sweep, warmup=1, repeats=args.repeats, budget_s=args.budget, sync=sync)
                row = {"batch": b, "status": "ok", "frames": n, "frame_shape": shape, "fps": n / t["median_s"], **t}
                if dev.type == "cuda":
                    row["peak_mib"] = torch.cuda.max_memory_allocated() / 2**20
            except torch.cuda.OutOfMemoryError as e:
                row = {"batch": b, "status": "oom", "error": str(e)[:300]}
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
        results["envs"][name] = {"kind": blob["kind"], "size": blob["size"], "points": rows}
    Path(args.output).write_text(json.dumps(results, indent=1))


_W: dict = {}


def _worker_init(name: str, states: str, size: int) -> None:
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    torch.set_num_threads(1)
    snv = _snv()
    cls = snv.registry.REGISTRY[name]
    _W["cls"] = cls
    _W["ref"] = cls.reference_env(size=size or cls.default_size)
    blob = _load_states(Path(states) / f"{name}.pt")
    _W["rows"] = blob["rows"].numpy() if blob["kind"] == "rows" else None


def _worker_render(span) -> int:
    s, e = span
    rows = _W["rows"][s:e]
    for i in range(0, len(rows), 64):
        _W["cls"].reference_render(_W["ref"], rows[i : i + 64])
    return e - s


def cmd_source(args) -> None:
    """Sweep every stored state through the source renderer in ``procs`` processes."""
    results = {"arm": f"source-{args.procs}p", "host": _host(), "envs": {}}
    for name in args.envs:
        path = Path(args.states) / f"{name}.pt"
        if not path.exists():
            results["envs"][name] = {"status": "skipped", "reason": "no state file"}
            continue
        blob = _load_states(path)
        if blob["kind"] != "rows":
            results["envs"][name] = {"status": "skipped", "reason": "source reference renders compact rows only"}
            continue
        snv = _snv()
        cls = snv.registry.REGISTRY[name]
        from tests.helpers.reference.certification import has_reference_renderer

        if not has_reference_renderer(cls):
            results["envs"][name] = {"status": "skipped", "reason": "no reference renderer"}
            continue
        n = len(blob["rows"])
        try:
            if args.procs == 1:
                _worker_init(name, args.states, blob["size"])
                frame = _W["cls"].reference_render(_W["ref"], _W["rows"][:1])
                shape = list(np.asarray(frame).shape[1:])
                t = _sweeps(lambda: _worker_render((0, n)), warmup=1, repeats=args.repeats, budget_s=args.budget)
            else:
                ctx = mp.get_context("spawn")
                spans = [(n * k // args.procs, n * (k + 1) // args.procs) for k in range(args.procs)]
                with ctx.Pool(
                    args.procs, initializer=_worker_init, initargs=(name, args.states, blob["size"])
                ) as pool:
                    pool.map(_worker_render, [(0, 1)] * args.procs)
                    shape = None
                    t = _sweeps(
                        lambda: pool.map(_worker_render, spans, chunksize=1),
                        warmup=1,
                        repeats=args.repeats,
                        budget_s=args.budget,
                    )
            row = {"status": "ok", "frames": n, "fps": n / t["median_s"], "frame_shape": shape, **t}
        except Exception as e:  # noqa: BLE001
            row = {"status": "error", "error": f"{type(e).__name__}: {e}"[:500]}
        print(f"{name} source {args.procs}p {row['status']} {row.get('fps', 0):.0f} fps", flush=True)
        results["envs"][name] = row
        _W.clear()
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
        n = len(blob["rows"] if blob["kind"] == "rows" else blob["state"]) if decode else args.sample
        row = {"frames_encoded": n}
        try:
            env = snv.make(name, num_envs=256, size=blob["size"], device=args.device)
            data = (blob["rows"] if blob["kind"] == "rows" else blob["state"]).to(dev)
            frames = []
            for s, e in _chunks(n, 256):
                f = env._render(data[s:e]) if blob["kind"] == "rows" else env.render(data[s:e])
                frames.append(f.cpu().numpy())
            env.close()
            frames = np.concatenate(frames)
            row["state_bytes"] = (
                int(blob["rows"].shape[1]) * 4
                if blob["kind"] == "rows"
                else int(sum(v.numel() * v.element_size() for v in blob["state"][0].values(True, True)))
            )
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
        except Exception as e:  # noqa: BLE001
            row = {"status": "error", "error": f"{type(e).__name__}: {e}"[:500]}
        print(f"{name} codec {json.dumps({k: v for k, v in row.items() if 'decode' not in k})}", flush=True)
        results["envs"][name] = row
    Path(args.output).write_text(json.dumps(results, indent=1))


def cmd_list(args) -> None:
    snv = _snv()
    names = [n for n in sorted(snv.registry.REGISTRY) if not n.startswith("Craftax")]
    if args.family:
        names = [n for n in names if (getattr(snv.registry.REGISTRY[n], "family", "") or "") in args.family]
    print(" ".join(names))


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="cmd", required=True)
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--envs", nargs="+", default=[])
    common.add_argument("--states", default="states")
    common.add_argument("--repeats", type=int, default=10)
    common.add_argument("--budget", type=float, default=300.0, help="seconds per sweep above which 3 repeats are kept")
    s = sub.add_parser("states", parents=[common])
    s.add_argument("--out", required=True)
    s.add_argument("--n", type=int, default=10_000)
    s.add_argument("--seed", type=int, default=20261022)
    s.add_argument("--size", type=int, default=224)
    s.add_argument("--lanes", type=int, default=1024)
    s.add_argument("--device", default="cuda")
    o = sub.add_parser("ours", parents=[common])
    o.add_argument("--device", default="cuda")
    o.add_argument("--batches", type=int, nargs="+", default=[1, 256, 512, 1024, 2048, 4096])
    o.add_argument("--threads", type=int, default=8)
    o.add_argument("--output", required=True)
    r = sub.add_parser("source", parents=[common])
    r.add_argument("--procs", type=int, default=1)
    r.add_argument("--output", required=True)
    c = sub.add_parser("codec", parents=[common])
    c.add_argument("--out", required=True)
    c.add_argument("--decode", nargs="*", default=[])
    c.add_argument("--sample", type=int, default=256)
    c.add_argument("--device", default="cuda")
    c.add_argument("--output", required=True)
    ls = sub.add_parser("list")
    ls.add_argument("--family", nargs="*", default=[])
    args = p.parse_args()
    {"states": cmd_states, "ours": cmd_ours, "source": cmd_source, "codec": cmd_codec, "list": cmd_list}[args.cmd](
        args
    )


if __name__ == "__main__":
    main()
