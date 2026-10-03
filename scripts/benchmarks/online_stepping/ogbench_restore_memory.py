"""Diagnose repeated restore memory with optional collection, without rendering.

Collection is instrumentation, not a production fix. Separate fresh processes
must be used for ordinary and explicit-collection measurements.
"""

import argparse
import gc
import hashlib
import importlib.metadata
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import torch

from stable_envs._adapters.ogbench_manipulation import OGBManipulationAdapter


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=["Cube", "Scene"], required=True)
    p.add_argument("--batch", type=int, default=2)
    p.add_argument("--repeats", type=int, default=12)
    p.add_argument("--collect", action="store_true")
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    torch.set_num_threads(1)
    files = [Path("stable_envs/_adapters/ogbench_manipulation.py"),
             Path("stable_envs/_adapters/ogbench_manipulation_compat.py"), Path(__file__).relative_to(Path.cwd())]
    report = dict(status="RUNNING", family=args.family, batch=args.batch, explicit_collection=args.collect,
                  host=platform.node(), job=os.environ.get("SLURM_JOB_ID"), samples=[],
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "mujoco", "ogbench")},
                  source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files})
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def sample(phase, **extra):
        rss = int(subprocess.check_output(["ps", "-o", "rss=", "-p", str(os.getpid())], text=True)) * 1024
        report["samples"].append(dict(phase=phase, rss_bytes=rss,
                                      gc_counts=gc.get_count(), tracked_objects=len(gc.get_objects()), **extra))
        args.output.write_text(json.dumps(report, indent=2) + "\n")
        print(phase, report["samples"][-1]["rss_bytes"], flush=True)

    adapter = None
    try:
        sample("before_construction")
        adapter = OGBManipulationAdapter(args.family, args.batch, seed=17)
        initial = adapter.get_state()
        report["state_bytes"] = sum(t.numel() * t.element_size() for t in initial.values(True, True))
        sample("constructed")
        for repeat in range(args.repeats):
            start = time.perf_counter()
            adapter.set_state(initial)
            elapsed = time.perf_counter() - start
            actual = adapter.get_state()
            for key in initial.keys(True, True):
                torch.testing.assert_close(initial[key], actual[key], rtol=0, atol=0, equal_nan=True)
            del actual
            collected = gc.collect() if args.collect else None
            sample("restore", repeat=repeat, restore_s=elapsed, collected=collected)
        sample("final_explicit_collection", collected=gc.collect())
        adapter.close()
        adapter = None
        sample("closed_and_collected", collected=gc.collect())
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        if adapter is not None:
            adapter.close()
        args.output.write_text(json.dumps(report, indent=2) + "\n")


if __name__ == "__main__":
    main()
