"""Measure classic CPU rendering and online execution across parent Torch threads.

This is a benchmark configuration sweep, not a library-global thread-policy
change. Acrobot simulation owners stay single-threaded. Linux CPU counters
include the parent and persistent simulator workers; no GPU speedup is claimed.
"""

import argparse
import cProfile
import hashlib
import importlib.metadata
import io
import json
import os
import platform
import pstats
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import stable_envs as se
from scripts.benchmarks.online_stepping.classic_source_pipeline import NAMES, actions_for
from tests.unit.test_classic_state_export import equal


def cpu_times(pids):
    if not Path("/proc/self/stat").exists():
        return None
    hz = os.sysconf("SC_CLK_TCK")
    result = {}
    for pid in pids:
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        result[str(pid)] = (int(fields[11]) + int(fields[12])) / hz
    return result


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--domain", choices=NAMES, required=True)
    p.add_argument("--batches", nargs="+", type=int, default=[64, 256, 1024])
    p.add_argument("--threads", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--repeats", type=int, default=3)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if min(args.batches) < 1 or min(args.threads) < 1 or args.repeats < 2 or 1 not in args.threads:
        p.error("Require positive batches/threads, a one-thread control and >=2 repeats")
    torch.set_num_threads(1)
    paths = [Path(__file__).relative_to(Path.cwd()), Path("stable_envs/envs/_classic_semantics.py"),
             Path("stable_envs/envs/classic_control.py"), Path("stable_envs/_stepping.py"),
             *Path("stable_envs/_adapters").rglob("*.py"),
             Path("stable_envs/_adapters/classic/process.py"), Path("stable_envs/_adapters/process_requests.py"),
             *Path("stable_envs/render").glob("gfxdraw*.py"),
             Path("scripts/benchmarks/online_stepping/classic_source_pipeline.py"),
             Path("scripts/benchmarks/online_stepping/classic_state_export.py"),
             Path("tests/unit/test_classic_state_export.py")]
    report = dict(status="RUNNING", domain=args.domain, device="cpu", host=platform.node(),
                  job=os.environ.get("SLURM_JOB_ID"),
                  affinity=sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None,
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "gymnasium", "tensordict")},
                  source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in paths},
                  scope="parent Torch intra-op threads; same ordinary simulator and CPU renderer; no source speedup",
                  cpu_counter_scope="Linux parent plus persistent simulator workers; quantized process CPU seconds",
                  timings=[], gates=[], profiles={})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")

    save()
    try:
        for batch in args.batches:
            torch.set_num_threads(1)
            with se.make(NAMES[args.domain], num_envs=batch, seed=17, device="cpu") as env:
                workers = env._simulator.worker_info if hasattr(env._simulator, "worker_info") else []
                pids = [os.getpid(), *[w["pid"] for w in workers]]
                initial = env.get_state()
                actions = actions_for(args.domain, batch, 31)
                if args.domain == "cartpole":
                    actions[:] = np.arange(8)[None, :, None] % 2
                operations = [("render_saved", 0, None, lambda: env.render(initial))]
                for horizon in (1, 8):
                    a = actions[:, 0] if horizon == 1 else actions
                    for selection, frames in (("none", None), ("final", slice(-1, None)), ("all", slice(None))):
                        operations.append(("online", horizon, selection,
                            lambda a=a, frames=frames: env.step(a, states=None, frames=frames)))
                for operation, horizon, selection, fn in operations:
                    torch.set_num_threads(1)
                    env.set_state(initial)
                    expected = fn()
                    for threads in args.threads:
                        torch.set_num_threads(threads)
                        env.set_state(initial)
                        actual = fn()
                        if operation == "online":
                            equal(expected, actual)
                            assert bool(actual["valid"].all())
                        else:
                            torch.testing.assert_close(expected, actual, rtol=0, atol=0)
                        report["gates"].append(dict(batch=batch, operation=operation, horizon=horizon,
                            selection=selection, threads=threads, exact_state_task_frame="PASS"))
                        del actual
                    del expected
                    samples = {threads: [] for threads in args.threads}
                    for block in range(2):
                        for threads in args.threads if block == 0 else reversed(args.threads):
                            torch.set_num_threads(threads)
                            env.set_state(initial)
                            fn()
                            for _ in range(args.repeats):
                                env.set_state(initial)
                                before = cpu_times(pids)
                                start = time.perf_counter()
                                result = fn()
                                elapsed = time.perf_counter() - start
                                after = cpu_times(pids)
                                if operation == "online":
                                    assert bool(result["valid"].all())
                                delta = {pid: after[pid] - before[pid] for pid in before} if before is not None else None
                                samples[threads].append(dict(seconds=elapsed, cpu_seconds=delta,
                                    average_busy_cores=sum(delta.values()) / elapsed if delta is not None else None))
                                del result
                    count = batch if operation == "render_saved" else batch * horizon
                    frame_count = batch if operation == "render_saved" else batch * (horizon if selection == "all" else int(selection == "final"))
                    for threads, values in samples.items():
                        rates = [count / s["seconds"] for s in values]
                        report["timings"].append(dict(batch=batch, operation=operation, horizon=horizon,
                            selection=selection, threads=threads, workers=len(workers), pids=pids,
                            count=count, frame_count=frame_count, samples=values,
                            rate_mean=statistics.mean(rates), rate_sd=statistics.stdev(rates)))
                    save()
                    print(args.domain, batch, operation, horizon, selection, "PASS", flush=True)
                if batch == 256:
                    for threads in (min(args.threads), max(args.threads)):
                        torch.set_num_threads(threads)
                        profiler = cProfile.Profile()
                        profiler.runcall(env.render, initial)
                        text = io.StringIO()
                        pstats.Stats(profiler, stream=text).sort_stats("cumulative").print_stats(35)
                        report["profiles"][str(threads)] = text.getvalue()
                        save()
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
