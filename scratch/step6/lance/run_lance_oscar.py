"""Run ``examples/bench_lance.py``'s measurement functions on Oscar, unmodified, over N storage arms.

``examples/bench_lance.py`` is imported, not copied: ``build_table``, ``time_path`` and
``time_parallel`` are the exact functions that produced ``examples/results/storage/lance.json`` on
Randall's cluster, so the only thing that differs between that run and this one is the machine, the
filesystem and the frames. Nothing in the timed region is touched.

What this adds:
  * an arbitrary number of labelled storage arms (Oscar has three that matter: node-local ``/tmp``
    on xfs, NFS-v3-over-RDMA scratch, and job-local GPFS ``/sss/jobtmp``), each reporting the
    filesystem it actually landed on as seen from INSIDE the job;
  * a worker sweep (1 / 4 / 8) on the local arm, for the scaling shape;
  * full provenance (node, lscpu, load average, cgroup cpu budget, venv, git commit).
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[3] / "examples"))
import bench_lance as BL  # noqa: E402  the committed benchmark, imported verbatim


def sh(cmd: str) -> str:
    try:
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


def fs_facts(path: str) -> dict:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    out = {
        "path": str(p),
        "stat_f_type": sh(f"stat -f -c '%T' {p!s}"),
        "df_T": sh(f"df -T {p!s} | tail -1"),
        "mount_line": sh(f"findmnt -n -o SOURCE,TARGET,FSTYPE,OPTIONS --target {p!s} 2>/dev/null | head -1"),
        "writable": os.access(str(p), os.W_OK),
    }
    try:
        st = os.statvfs(str(p))
        out["avail_gb"] = st.f_bavail * st.f_frsize / 1e9
        out["size_gb"] = st.f_blocks * st.f_frsize / 1e9
    except Exception as e:
        out["avail_gb"] = None
        out["statvfs_error"] = str(e)
    # a crude write-bandwidth probe, outside every timed region, purely descriptive
    try:
        f = p / f"_probe_{os.getpid()}.bin"
        buf = os.urandom(64 << 20)
        t0 = time.perf_counter()
        with open(f, "wb") as fh:
            fh.write(buf)
            fh.flush()
            os.fsync(fh.fileno())
        out["write_64MB_MBps"] = 64 / (time.perf_counter() - t0)
        f.unlink(missing_ok=True)
    except Exception as e:
        out["write_probe_error"] = f"{type(e).__name__}: {e}"
    return out


def new_pool(workers: int):
    """bench_lance caches one pool globally; swap it per worker count, closing the old one."""
    old = BL._POOL
    if old is not None:
        old.close()
        old.join()
    BL._POOL = None
    return workers


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--frames-dir", required=True)
    ap.add_argument("--chunk", type=int, default=256)
    ap.add_argument("--reps", type=int, default=8)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--worker-sweep", default="", help="e.g. 1,4 -- extra worker counts on the local arm")
    ap.add_argument("--arm", action="append", default=[],
                    help="label=path, repeatable. First arm is the 'local' one (gets the worker sweep).")
    ap.add_argument("--parallel-arms", default="", help="comma list of arm labels that also get --workers procs")
    ap.add_argument("--gpu", action="store_true")
    ap.add_argument("--envs", default="")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    arms = []
    for spec in a.arm:
        label, _, path = spec.partition("=")
        arms.append((label, path))
    if not arms:
        raise SystemExit("need at least one --arm label=path")
    par_arms = set(a.parallel_arms.split(",")) if a.parallel_arms else {arms[0][0]}

    res = {
        "what": "Step 6 re-run of examples/bench_lance.py on Oscar",
        "cpu": sh("lscpu | grep 'Model name' | head -1").split(":", 1)[-1].strip(),
        "lscpu": sh("lscpu | egrep 'Model name|Socket|Core\\(s\\)|Thread|CPU\\(s\\):|MHz|L3|NUMA node\\(s\\)'"),
        "node": platform.node(),
        "hostname": socket.gethostname(),
        "chunk": a.chunk, "reps": a.reps, "workers": a.workers,
        "loadavg": os.getloadavg(),
        "cpu_count": len(os.sched_getaffinity(0)),
        "cpu_count_total": os.cpu_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "slurm": {k: v for k, v in os.environ.items() if k.startswith("SLURM_")
                  and k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK",
                            "SLURM_JOB_PARTITION", "SLURM_JOB_QOS", "SLURM_MEM_PER_NODE")},
        "python": sys.version.split()[0], "executable": sys.executable,
        "torch": torch.__version__,
        "git_commit": sh(f"git -C {Path(__file__).resolve().parents[3]} rev-parse HEAD"),
        "frames_dir": a.frames_dir,
        "arms": {label: fs_facts(path) for label, path in arms},
        "envs": {},
    }
    for m in ("lancedb", "pylance", "pyarrow", "torchvision", "pillow"):
        try:
            import importlib.metadata as md
            res[m] = md.version(m)
        except Exception:
            res[m] = None
    prov = sorted(Path(a.frames_dir).glob("_provenance_*.json"))
    res["frames_provenance"] = [json.loads(p.read_text()) for p in prov]

    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)
    print(json.dumps({k: v for k, v in res.items() if k != "envs"}, indent=2), flush=True)

    files = sorted(Path(a.frames_dir).glob("*.npz"))
    if a.envs:
        keep = set(a.envs.split(","))
        files = [f for f in files if f.stem in keep]
    print(f"{len(files)} envs, chunk {a.chunk}, {a.workers} workers, node {res['node']}", flush=True)

    sweep = [int(x) for x in a.worker_sweep.split(",") if x] if a.worker_sweep else []
    local_label = arms[0][0]

    for fpath in files:
        name = fpath.stem
        z = np.load(fpath, allow_pickle=True)
        meta = json.loads(str(z["meta"]))
        blobs = list(z["blobs"])
        n = len(blobs)
        entry = {"meta": {k: v for k, v in meta.items() if k != "provenance"}, "n": n,
                 "jpeg_bytes_per_frame": meta["jpeg_bytes_total"] / n,
                 "raw_bytes_per_frame": meta["raw_bytes_per_frame"]}
        for label, rootstr in arms:
            r = Path(rootstr)
            try:
                tbl = BL.build_table(blobs, r, name)
                entry[f"{label}_1"] = BL.time_path(tbl, n, a.chunk, a.reps)
                if label in par_arms:
                    new_pool(a.workers)
                    entry[f"{label}_n"] = BL.time_parallel(str(r), name, n, a.chunk, a.reps, a.workers)
                if label == local_label:
                    for w in sweep:
                        new_pool(w)
                        entry[f"{label}_n{w}"] = BL.time_parallel(str(r), name, n, a.chunk, a.reps, w)
                    if a.gpu and torch.cuda.is_available():
                        try:
                            entry[f"{label}_gpu"] = BL.time_path(tbl, n, a.chunk, a.reps, device="cuda")
                        except Exception as e:
                            entry[f"{label}_gpu"] = {"error": f"{type(e).__name__}: {e}"}
                del tbl
            except Exception as e:
                entry[f"{label}_1"] = {"error": f"{type(e).__name__}: {e}"}
            finally:
                shutil.rmtree(r / f"{name}.lance", ignore_errors=True)
        res["envs"][name] = entry
        bits = " ".join(f"{k} {entry[k]['fps']:,.0f}" for k in sorted(entry)
                        if isinstance(entry.get(k), dict) and entry[k].get("fps"))
        print(f"[{name}] {meta['h']}x{meta['w']} {entry['jpeg_bytes_per_frame']:,.0f} B/frame  {bits}",
              flush=True)
        res["loadavg_end"] = os.getloadavg()
        outp.write_text(json.dumps(res, indent=2))

    res["loadavg_end"] = os.getloadavg()
    outp.write_text(json.dumps(res, indent=2))

    import statistics as st
    ok = [e for e in res["envs"].values() if e.get(f"{local_label}_1", {}).get("fps")]
    keys = sorted({k for e in res["envs"].values() for k in e
                   if isinstance(e.get(k), dict) and e[k].get("fps")})
    for k in keys:
        v = [e[k]["fps"] for e in ok if e.get(k, {}).get("fps")]
        if v:
            print(f"{k:14s} median {st.median(v):>10,.0f} f/s   range {min(v):,.0f}-{max(v):,.0f}   n={len(v)}",
                  flush=True)


if __name__ == "__main__":
    main()
