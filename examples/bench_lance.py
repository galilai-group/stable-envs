"""Phase 2: the strongest disk-based baseline in use today — JPEG blobs in Lance, batch-decoded.

Per-file ``PIL.Image.open`` is not what a stored-frame corpus does and benchmarking against it would
be setting up a straw man. The current practice this has to beat is the JPEG-q95 Lance baseline: an
episode-contiguous LanceDB table whose image column is ``pa.binary`` holding **JPEG at quality 95**,
fetched a chunk of rows at a time and decoded in a single ``torchvision.io.decode_jpeg`` call —
libjpeg-turbo, SIMD, GIL released.

**Every environment is measured in ONE process on ONE node against ONE table layout.** That is the
whole point of splitting this out. In the first pass each env's storage number came from whichever job
happened to own it, so a column that should have been a property of the filesystem varied by 18x across
envs and most of that was cross-node noise. Here the only thing that differs between rows is the one
thing that legitimately does: **compressed bytes per frame**. A flat tile-map frame is a few kB and a
textured kitchen is tens of kB, and decode cost tracks that, so the column has a real spread with a
real cause — and the aggregate at the bottom is the number to quote.

Four read paths, all from the same table:

  ``local_1``    node-local NVMe, one process, CPU decode. The favourable case for a dataset.
  ``shared_1``   the shared filesystem a cluster corpus actually lives on, otherwise identical.
  ``local_n``    node-local, ``--workers`` processes, which is how a DataLoader is really run. This is
                 the honest "fastest disk-based setup today" number.
  ``local_gpu``  node-local, nvJPEG decode on the GPU. Not the default CPU-decode path, measured
                 because it is the strongest thing the baseline could become.
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import time
from pathlib import Path

import numpy as np
import torch

FRAMES = "/mnt/vast/home/randall.balestriero/se_frames"


def build_table(blobs, root: Path, name: str):
    """A Lance table with episode/step index plus a binary ``pixels`` column."""
    import lancedb
    import pyarrow as pa

    root.mkdir(parents=True, exist_ok=True)
    db = lancedb.connect(str(root))
    n = len(blobs)
    tbl = pa.table(
        {
            "episode_idx": pa.array(np.zeros(n, np.int32)),
            "step_idx": pa.array(np.arange(n, dtype=np.int32)),
            "pixels": pa.array([bytes(b) for b in blobs], type=pa.binary()),
        }
    )
    try:
        db.drop_table(name)
    except Exception:
        pass
    return db.create_table(name, tbl)


def _decode(byte_list, device=None):
    from torchvision.io import ImageReadMode, decode_jpeg

    ts = [torch.frombuffer(b, dtype=torch.uint8) for b in byte_list]
    if device is None:
        return torch.stack(decode_jpeg(ts, mode=ImageReadMode.RGB))
    out = decode_jpeg(ts, mode=ImageReadMode.RGB, device=device)
    return torch.stack(out)


def time_path(tbl, n: int, chunk: int, reps: int, device=None) -> dict:
    """Frames/s for fetch-a-chunk-of-rows then batch-decode, plus the split between the two."""
    import pyarrow.compute as pc  # noqa: F401  (import kept beside lance for a clear failure)

    starts = [(i * chunk) % max(n - chunk, 1) for i in range(reps)]
    best_total = best_read = best_dec = None
    for s in starts:
        t0 = time.perf_counter()
        batch = tbl.to_lance().take(list(range(s, min(s + chunk, n))), columns=["pixels"])
        blobs = batch.column("pixels").to_pylist()
        t1 = time.perf_counter()
        frames = _decode(blobs, device=device)
        if device is not None:
            torch.cuda.synchronize()
        t2 = time.perf_counter()
        k = len(blobs)
        tot, rd, dc = (t2 - t0) / k, (t1 - t0) / k, (t2 - t1) / k
        best_total = tot if best_total is None else min(best_total, tot)
        best_read = rd if best_read is None else min(best_read, rd)
        best_dec = dc if best_dec is None else min(best_dec, dc)
        del frames, blobs, batch
    return {
        "fps": 1 / best_total,
        "read_fps": 1 / best_read,
        "decode_fps": 1 / best_dec,
        "read_frac": best_read / best_total,
    }


def _worker(args):
    root, name, n, chunk, reps, seed = args
    # One thread per worker, which is what a DataLoader worker gets and what makes N workers mean N
    # cores. Left at the default, each of 8 workers opened as many intra-op threads as the node has
    # cores and the aggregate came out at 121 f/s against 1,914 single-process -- a 16x LOSS from
    # adding workers, which is thread thrashing rather than anything about Lance or JPEG.
    torch.set_num_threads(1)
    import lancedb

    db = lancedb.connect(root)
    tbl = db.open_table(name)
    starts = [((i + seed) * chunk) % max(n - chunk, 1) for i in range(reps)]
    ds = tbl.to_lance()
    # One untimed pass: a spawned worker pays for importing torch and torchvision and for lance's first
    # touch of the file, and a real DataLoader worker pays that once per epoch, not once per batch.
    b = ds.take(list(range(0, min(chunk, n))), columns=["pixels"])
    _decode(b.column("pixels").to_pylist())
    t0 = time.perf_counter()
    got = 0
    for s in starts:
        batch = ds.take(list(range(s, min(s + chunk, n))), columns=["pixels"])
        blobs = batch.column("pixels").to_pylist()
        _decode(blobs)
        got += len(blobs)
    return got, time.perf_counter() - t0


_POOL = None


def get_pool(workers: int):
    """One pool for the whole sweep, not one per environment.

    A fresh ``spawn`` pool costs eight torch imports, and creating one per environment made the run
    2.4 minutes per environment -- 7.8 hours for the registry, against a job limit of 8. The workers
    are stateless apart from the lancedb handle they open per call, so reuse changes nothing measured.
    """
    global _POOL
    if _POOL is None:
        import multiprocessing as mp

        _POOL = mp.get_context("spawn").Pool(workers)
    return _POOL


def time_parallel(root: str, name: str, n: int, chunk: int, reps: int, workers: int) -> dict:
    """Aggregate frames/s with ``workers`` processes, which is how a DataLoader actually reads.

    Aggregated from the workers' OWN elapsed times, not from the wall clock around ``pool.map``. The
    first version timed the pool from outside and reported 125 f/s against 1,839 single-process --
    process spawn plus eight torch imports, charged to the read. A DataLoader pays that once per epoch;
    charging it per batch would understate the baseline this repo has to beat, which is the direction
    of error to avoid.
    """
    args = [(root, name, n, chunk, reps, w) for w in range(workers)]
    res = get_pool(workers).map(_worker, args)
    total = sum(r[0] for r in res)
    span = max(r[1] for r in res)  # the slowest worker bounds the epoch
    return {
        "fps": total / span,
        "workers": workers,
        "slowest_worker_s": span,
        "frames": total,
        "per_worker_fps": [r[0] / r[1] for r in res],
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", default=FRAMES)
    ap.add_argument("--chunk", type=int, default=256, help="rows fetched per read, as a loader would")
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--local", default="/tmp/se_lance")
    ap.add_argument("--shared", default="/mnt/vast/home/randall.balestriero/tmp_lance")
    ap.add_argument("--gpu", action="store_true", help="also time nvJPEG decode")
    ap.add_argument("--envs", default="")
    ap.add_argument("--out", default="results/storage/lance.json")
    a = ap.parse_args()

    try:
        cpu_name = (
            subprocess.run(["bash", "-c", "lscpu | grep 'Model name' | head -1"], capture_output=True, text=True)
            .stdout.split(":", 1)[-1]
            .strip()
        )
    except Exception:
        cpu_name = platform.processor()

    res = {
        "cpu": cpu_name,
        "node": platform.node(),
        "chunk": a.chunk,
        "workers": a.workers,
        "loadavg": os.getloadavg()[0],
        "cpu_count": len(os.sched_getaffinity(0)),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "envs": {},
    }
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    if outp.exists():
        try:
            res["envs"] = json.loads(outp.read_text()).get("envs", {})
        except Exception:
            pass

    files = sorted(Path(a.frames_dir).glob("*.npz"))
    if a.envs:
        keep = set(a.envs.split(","))
        files = [f for f in files if f.stem in keep]
    print(f"{len(files)} envs, chunk {a.chunk}, {a.workers} workers, node {res['node']}", flush=True)

    for f in files:
        name = f.stem
        if name in res["envs"] and res["envs"][name].get("local_1"):
            continue
        z = np.load(f, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        blobs = list(z["blobs"])
        n = len(blobs)
        entry = {
            "meta": meta,
            "n": n,
            "jpeg_bytes_per_frame": meta["jpeg_bytes_total"] / n,
            "raw_bytes_per_frame": meta["raw_bytes_per_frame"],
        }
        for label, root in (("local", a.local), ("shared", a.shared)):
            r = Path(root)
            try:
                tbl = build_table(blobs, r, name)
                entry[f"{label}_1"] = time_path(tbl, n, a.chunk, a.reps)
                if label == "local":
                    entry["local_n"] = time_parallel(str(r), name, n, a.chunk, a.reps, a.workers)
                    if a.gpu and torch.cuda.is_available():
                        try:
                            entry["local_gpu"] = time_path(tbl, n, a.chunk, a.reps, device="cuda")
                        except Exception as e:
                            entry["local_gpu"] = {"error": f"{type(e).__name__}: {e}"}
                del tbl
            except Exception as e:
                entry[f"{label}_1"] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                shutil.rmtree(Path(root) / f"{name}.lance", ignore_errors=True)
        res["envs"][name] = entry
        l1 = entry.get("local_1", {}).get("fps")
        s1 = entry.get("shared_1", {}).get("fps")
        ln = entry.get("local_n", {}).get("fps")
        lg = entry.get("local_gpu", {}).get("fps")
        print(
            f"[{name}] {meta['h']}x{meta['w']} {entry['jpeg_bytes_per_frame']:,.0f} B/frame  "
            f"local {l1:,.0f}  shared {s1:,.0f}  {a.workers}proc {ln:,.0f}  "
            f"gpu {lg:,.0f}"
            if lg
            else f"[{name}] {meta['h']}x{meta['w']} {entry['jpeg_bytes_per_frame']:,.0f} B/frame  "
            f"local {l1:,.0f}  shared {s1:,.0f}  {a.workers}proc {ln:,.0f}",
            flush=True,
        )
        outp.write_text(json.dumps(res, indent=2))

    outp.write_text(json.dumps(res, indent=2))
    ok = [e for e in res["envs"].values() if e.get("local_1", {}).get("fps")]
    if ok:
        import statistics as st

        for k in ("local_1", "shared_1", "local_n", "local_gpu"):
            v = [e[k]["fps"] for e in ok if e.get(k, {}).get("fps")]
            if v:
                print(f"{k:10s} median {st.median(v):>10,.0f} f/s   range {min(v):,.0f}-{max(v):,.0f}   n={len(v)}")


if __name__ == "__main__":
    main()
