"""Instrumented re-run of the JPEG-in-Lance arms: adds CPU accounting and read bandwidth.

The timed regions are byte-identical to ``examples/bench_lance.py``'s (``build_table``, ``time_path``
and the body of ``_worker`` are used or reproduced without a single added statement inside the timed
loop). What is new sits strictly OUTSIDE the clock:

* ``resource.getrusage(RUSAGE_SELF)`` sampled immediately before and after each worker's timed loop,
  so "8 workers" can be checked against the CPU-seconds actually burned -- ``cores_used = (utime +
  stime) / elapsed`` per worker, summed over workers;
* read bandwidth in MB/s, computed from the table's own bytes-per-frame and the read fps that
  ``time_path`` already reports, rather than by summing blob lengths inside the loop;
* the node load average before and after every arm.

Output location is recorded explicitly on every arm: these frames land as uint8 **CHW tensors in HOST
memory**. That is what ``torchvision.io.decode_jpeg`` returns and what a DataLoader worker hands back.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))
import bench_lance as BL  # noqa: E402


def sh(cmd: str) -> str:
    try:
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


def _worker(args):
    """``bench_lance._worker`` with rusage sampled outside the clock. Timed body unchanged."""
    root, name, n, chunk, reps, seed = args
    torch.set_num_threads(1)
    import lancedb
    db = lancedb.connect(root)
    tbl = db.open_table(name)
    starts = [((i + seed) * chunk) % max(n - chunk, 1) for i in range(reps)]
    ds = tbl.to_lance()
    b = ds.take(list(range(0, min(chunk, n))), columns=["pixels"])
    BL._decode(b.column("pixels").to_pylist())
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    got = 0
    for s in starts:
        batch = ds.take(list(range(s, min(s + chunk, n))), columns=["pixels"])
        blobs = batch.column("pixels").to_pylist()
        BL._decode(blobs)
        got += len(blobs)
    el = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    return {"frames": got, "elapsed_s": el,
            "utime_s": r1.ru_utime - r0.ru_utime, "stime_s": r1.ru_stime - r0.ru_stime,
            "maxrss_gb": r1.ru_maxrss * 1024 / 1e9, "pid": os.getpid()}


_POOLS: dict[int, object] = {}


def pool(workers: int):
    import multiprocessing as mp
    if workers not in _POOLS:
        _POOLS[workers] = mp.get_context("spawn").Pool(workers)
    return _POOLS[workers]


def parallel(root: str, name: str, n: int, chunk: int, reps: int, workers: int,
             bytes_per_frame: float) -> dict:
    res = pool(workers).map(_worker, [(root, name, n, chunk, reps, w) for w in range(workers)])
    total = sum(r["frames"] for r in res)
    span = max(r["elapsed_s"] for r in res)
    cpu_s = sum(r["utime_s"] + r["stime_s"] for r in res)
    return {"fps": total / span, "workers": workers, "slowest_worker_s": span, "frames": total,
            "per_worker_fps": [r["frames"] / r["elapsed_s"] for r in res],
            "per_worker_elapsed_s": [r["elapsed_s"] for r in res],
            "per_worker_cpu_s": [r["utime_s"] + r["stime_s"] for r in res],
            "per_worker_cores_used": [(r["utime_s"] + r["stime_s"]) / r["elapsed_s"] for r in res],
            "cpu_seconds_total": cpu_s,
            "cores_used_total": cpu_s / span,
            "cpu_efficiency_vs_workers": (cpu_s / span) / workers,
            "decode_MBps": total * bytes_per_frame / span / 1e6,
            "raw_out_MBps_host": None,
            "maxrss_gb_max": max(r["maxrss_gb"] for r in res),
            "output_location": "HOST memory, uint8 CHW (torchvision.io.decode_jpeg -> torch.stack)"}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--workers", default="1,8")
    ap.add_argument("--arm", action="append", default=[])
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--envs", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = [(s.partition("=")[0], s.partition("=")[2]) for s in a.arm]
    worker_counts = [int(x) for x in a.workers.split(",")]

    res = {"what": "Step 6: instrumented JPEG-in-Lance arms on Oscar "
                   "(examples/bench_lance.py timing, plus CPU accounting and read bandwidth)",
           "cpu": sh("lscpu | grep 'Model name' | head -1").split(":", 1)[-1].strip(),
           "node": platform.node(), "hostname": socket.gethostname(),
           "chunk": a.chunk, "reps": a.reps, "worker_counts": worker_counts,
           "cpu_count_affinity": len(os.sched_getaffinity(0)), "cpu_count_total": os.cpu_count(),
           "loadavg_start": os.getloadavg(),
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "slurm": {k: v for k, v in os.environ.items()
                     if k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK",
                              "SLURM_JOB_PARTITION", "SLURM_JOB_QOS")},
           "python": sys.version.split()[0], "executable": sys.executable, "torch": torch.__version__,
           "output_location_note": "every CPU arm returns uint8 CHW tensors in HOST memory; a GPU "
                                   "consumer still owes an H2D copy, which is NOT charged here",
           "arms_paths": {}, "envs": {}}
    for label, p in arms:
        Path(p).mkdir(parents=True, exist_ok=True)
        res["arms_paths"][label] = {
            "path": p, "fstype": sh(f"stat -f -c '%T' {p}"),
            "mount": sh(f"findmnt -n -o SOURCE,TARGET,FSTYPE,OPTIONS --target {p} | head -1")}

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    files = sorted(Path(a.frames_dir).glob("*.npz"))
    if a.envs:
        keep = set(a.envs.split(","))
        files = [f for f in files if f.stem in keep]
    print(json.dumps({k: v for k, v in res.items() if k != "envs"}, indent=2), flush=True)

    for fpath in files:
        name = fpath.stem
        z = np.load(fpath, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        blobs = list(z["blobs"])
        n = len(blobs)
        bpf = meta["jpeg_bytes_total"] / n
        entry = {"meta": {k: v for k, v in meta.items() if k != "provenance"}, "n": n,
                 "jpeg_bytes_per_frame": bpf, "raw_bytes_per_frame": meta["raw_bytes_per_frame"]}
        for label, rootstr in arms:
            r = Path(rootstr)
            try:
                tbl = BL.build_table(blobs, r, name)
                one = BL.time_path(tbl, n, a.chunk, a.reps)
                one["read_MBps"] = one["read_fps"] * bpf / 1e6
                one["decode_MBps"] = one["fps"] * bpf / 1e6
                one["host_out_MBps"] = one["fps"] * meta["raw_bytes_per_frame"] / 1e6
                one["output_location"] = "HOST memory, uint8 CHW"
                entry[f"{label}_1proc"] = one
                for w in worker_counts:
                    entry[f"{label}_n{w}"] = parallel(str(r), name, n, a.chunk, a.reps, w, bpf)
                    entry[f"{label}_n{w}"]["raw_out_MBps_host"] = (
                        entry[f"{label}_n{w}"]["fps"] * meta["raw_bytes_per_frame"] / 1e6)
                if label == arms[0][0] and a.gpu and torch.cuda.is_available():
                    g = BL.time_path(tbl, n, a.chunk, a.reps, device="cuda")
                    g["read_MBps"] = g["read_fps"] * bpf / 1e6
                    g["output_location"] = "DEVICE memory, uint8 CHW (nvJPEG)"
                    entry[f"{label}_nvjpeg"] = g
                del tbl
            except Exception as e:
                entry[f"{label}_error"] = f"{type(e).__name__}: {e}"
            finally:
                shutil.rmtree(r / f"{name}.lance", ignore_errors=True)
        entry["loadavg"] = os.getloadavg()
        res["envs"][name] = entry
        bits = " ".join(f"{k}={entry[k]['fps']:,.0f}" for k in sorted(entry)
                        if isinstance(entry.get(k), dict) and entry[k].get("fps"))
        cu = entry.get(f"{arms[0][0]}_n8", {}).get("cores_used_total")
        print(f"[{name}] {meta['h']}x{meta['w']} {bpf:,.0f}B/f  {bits}"
              + (f"  cores_used(8w)={cu:.2f}" if cu else ""), flush=True)
        outp.write_text(json.dumps(res, indent=2))

    res["loadavg_end"] = os.getloadavg()
    outp.write_text(json.dumps(res, indent=2))


if __name__ == "__main__":
    main()
