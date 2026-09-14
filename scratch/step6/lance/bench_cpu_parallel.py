"""Our CPU renderer at an 8-way budget, measured the way the 8-worker Lance arm is measured.

Step 2 s5.3: the CERTIFIED CPU number is 1 thread. This file produces the *supplemental* all-core
figure and never replaces it. It gives both shapes of 8-way, because they are different machines:

  ``threads8``  one process, ``torch.set_num_threads(8)`` -- intra-op parallelism inside one render.
  ``procs8``    eight ``spawn`` processes, each ``torch.set_num_threads(1)``, each rendering its own
                shard of the batch, aggregated as ``total frames / max(worker elapsed)``.
                This is structurally identical to ``bench_lance.time_parallel``: same pool, same
                aggregation, same exclusion of spawn cost, and each worker pays its own warm-up
                render before the clock starts, exactly as a Lance worker pays an untimed first pass.

``threads1`` is taken in the same process for the scaling denominator, so
``scaling_efficiency = 8-way fps / (8 x 1-thread fps)`` is a within-job ratio.

States are drawn once and live in HOST memory for every arm -- the counterpart of the Lance workers
having their table already warm. Frames land in host memory. No device is touched.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import resource
import socket
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch


def sh(cmd: str) -> str:
    try:
        return subprocess.run(["bash", "-c", cmd], capture_output=True, text=True, timeout=60).stdout.strip()
    except Exception as e:
        return f"<{type(e).__name__}: {e}>"


def _render_shard(args):
    """One process: build the env, render its shard, return frames and its own elapsed + CPU time."""
    tree, name, states, batch, reps = args
    if tree and tree not in sys.path:
        sys.path.insert(0, tree)
    torch.set_num_threads(1)
    import stable_envs as se
    env = se.make(name, batch=batch, device="cpu")
    st = torch.as_tensor(np.asarray(states, np.float32))
    env.render(st)                      # untimed warm-up, as a Lance worker gets an untimed pass
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    got = 0
    for _ in range(reps):
        env.render(st)
        got += batch
    el = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    return {"frames": got, "elapsed_s": el,
            "cpu_s": (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)}


def time_inproc(env, st, reps):
    env.render(st)
    r0 = resource.getrusage(resource.RUSAGE_SELF)
    t0 = time.perf_counter()
    for _ in range(reps):
        env.render(st)
    el = time.perf_counter() - t0
    r1 = resource.getrusage(resource.RUSAGE_SELF)
    cpu = (r1.ru_utime - r0.ru_utime) + (r1.ru_stime - r0.ru_stime)
    n = int(st.shape[0]) * reps
    return {"fps": n / el, "frames": n, "elapsed_s": el, "cpu_s": cpu, "cores_used": cpu / el}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--envs", required=True)
    ap.add_argument("--batch", type=int, default=256, help="per-arm total batch; procs8 shards it 8 ways")
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--max-cell-s", type=float, default=90.0)
    ap.add_argument("--tree", default=None, help="repo root to put on the workers' sys.path")
    ap.add_argument("--out", required=True)
    a = ap.parse_args()

    import multiprocessing as mp
    import stable_envs as se

    tree = a.tree or str(Path(__file__).resolve().parents[3])
    res = {"what": "our CPU renderer: 1 thread (denominator), 8 threads, and 8 single-thread "
                   "processes -- the supplemental all-core view, never a certification number",
           "cpu": sh("lscpu | grep 'Model name' | head -1").split(":", 1)[-1].strip(),
           "node": platform.node(), "hostname": socket.gethostname(),
           "batch": a.batch, "reps": a.reps, "workers": a.workers,
           "cpu_count_affinity": len(os.sched_getaffinity(0)),
           "loadavg_start": os.getloadavg(),
           "slurm": {k: v for k, v in os.environ.items()
                     if k in ("SLURM_JOB_ID", "SLURM_JOB_NODELIST", "SLURM_CPUS_PER_TASK")},
           "python": sys.version.split()[0], "executable": sys.executable, "torch": torch.__version__,
           "tree": tree, "states_location": "HOST memory", "frames_location": "HOST memory",
           "envs": {}}
    outp = Path(a.out)
    outp.parent.mkdir(parents=True, exist_ok=True)

    ctx = mp.get_context("spawn")
    pool = ctx.Pool(a.workers)
    shard = max(a.batch // a.workers, 1)

    for name in [n.strip() for n in a.envs.split(",") if n.strip()]:
        entry: dict = {}
        try:
            torch.set_num_threads(1)
            probe = se.make(name, batch=a.batch, device="cpu")
            pool_states = np.asarray(probe.sample_states(a.batch, rng=np.random.default_rng(0)),
                                     np.float32)
            st = torch.as_tensor(pool_states)
            f0 = probe.render(st[:1])
            entry["shape"] = f"{int(f0.shape[1])}x{int(f0.shape[2])}"
            entry["size"] = int(probe.size)
            entry["supersample"] = int(probe.ss)
            entry["backend"] = getattr(probe, "backend", None)
            t = time.perf_counter()
            probe.render(st[:1])
            per_frame = time.perf_counter() - t
            if per_frame * a.batch * a.reps > a.max_cell_s:
                reps = max(1, int(a.max_cell_s / (per_frame * a.batch)))
                entry["reps_reduced_to"] = reps
            else:
                reps = a.reps
            entry["threads1"] = time_inproc(probe, st, reps)
            del probe

            torch.set_num_threads(8)
            e8 = se.make(name, batch=a.batch, device="cpu")
            entry["threads8"] = time_inproc(e8, st, reps)
            del e8
            torch.set_num_threads(1)

            args = [(tree, name, pool_states[i * shard:(i + 1) * shard], shard, reps)
                    for i in range(a.workers)]
            got = pool.map(_render_shard, args)
            total = sum(g["frames"] for g in got)
            span = max(g["elapsed_s"] for g in got)
            cpu_s = sum(g["cpu_s"] for g in got)
            entry["procs8"] = {"fps": total / span, "frames": total, "slowest_worker_s": span,
                               "per_worker_fps": [g["frames"] / g["elapsed_s"] for g in got],
                               "per_worker_cpu_s": [g["cpu_s"] for g in got],
                               "cpu_seconds_total": cpu_s, "cores_used_total": cpu_s / span,
                               "shard_batch": shard}
            b1 = entry["threads1"]["fps"]
            entry["best_8way"] = max(entry["threads8"]["fps"], entry["procs8"]["fps"])
            entry["best_8way_arm"] = ("procs8" if entry["procs8"]["fps"] > entry["threads8"]["fps"]
                                      else "threads8")
            entry["scaling_efficiency_threads8"] = entry["threads8"]["fps"] / (8 * b1)
            entry["scaling_efficiency_procs8"] = entry["procs8"]["fps"] / (8 * b1)
            entry["scaling_efficiency_best"] = entry["best_8way"] / (8 * b1)
            print(f"[{name}] {entry['shape']} 1thr {b1:,.0f}  8thr {entry['threads8']['fps']:,.0f}  "
                  f"8proc {entry['procs8']['fps']:,.0f}  eff_best "
                  f"{entry['scaling_efficiency_best']:.2f}", flush=True)
        except Exception as e:
            entry["error"] = f"{type(e).__name__}: {e}"
            print(f"[{name}] ERROR {entry['error']}", flush=True)
        entry["loadavg"] = os.getloadavg()
        res["envs"][name] = entry
        outp.write_text(json.dumps(res, indent=2))

    res["loadavg_end"] = os.getloadavg()
    outp.write_text(json.dumps(res, indent=2))
    pool.close()
    pool.join()


if __name__ == "__main__":
    main()
