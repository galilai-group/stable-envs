"""Measure one production Cube/Scene population without a paired reference world.

Linux PSS snapshots apportion shared pages; summed RSS can double-count them.
These are phase snapshots, not continuous process-tree peaks. CUDA allocator
statistics cover Torch tensors, not every OpenGL/driver allocation. Run each
batch in a fresh scheduled process. This diagnostic is not a source speedup or
a replacement for independent acceptance gates.
"""

import argparse
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path

import numpy as np
import torch

import stable_envs as se


def memory(pid):
    values = {}
    for line in Path(f"/proc/{pid}/smaps_rollup").read_text().splitlines()[1:]:
        name, value, *_ = line.split()
        if name in ("Rss:", "Pss:", "Private_Clean:", "Private_Dirty:"):
            values[name[:-1]] = int(value) * 1024
    assert "Rss" in values and "Pss" in values
    return dict(pid=pid, rss_bytes=values["Rss"], pss_bytes=values["Pss"],
                private_bytes=values["Private_Clean"] + values["Private_Dirty"])


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--family", choices=("Cube", "Scene"), required=True)
    p.add_argument("--batch", type=int, required=True)
    p.add_argument("--output", type=Path, required=True)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert args.batch >= 64 and len(os.sched_getaffinity(0)) == 8
    torch.set_num_threads(1)
    files = [*Path("stable_envs/_adapters").glob("ogbench*.py"),
             *Path("stable_envs/envs/ogbench").glob("*.py"),
             Path("stable_envs/_adapters/process_requests.py"), Path("stable_envs/_stepping.py"),
             Path(__file__).relative_to(Path.cwd())]
    report = dict(status="RUNNING", family=args.family, batch=args.batch, host=platform.node(),
                  job=os.environ.get("SLURM_JOB_ID"), gpu=torch.cuda.get_device_name(),
                  affinity=sorted(os.sched_getaffinity(0)), samples=[], timings=[],
                  versions={n: importlib.metadata.version(n) for n in ("torch", "numpy", "mujoco", "ogbench")},
                  source_sha256={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
                  scope="one public production population; CUDA output; phase memory snapshots; no source or old-adapter comparison")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    workers = []

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    def sample(phase):
        torch.cuda.synchronize()
        rows = [memory(pid) for pid in [os.getpid(), *workers]]
        report["samples"].append(dict(phase=phase, processes=rows,
            total_pss_bytes=sum(r["pss_bytes"] for r in rows),
            total_rss_bytes=sum(r["rss_bytes"] for r in rows),
            cuda_torch_allocated=torch.cuda.memory_allocated(),
            cuda_torch_reserved=torch.cuda.memory_reserved(),
            cuda_torch_peak_allocated=torch.cuda.max_memory_allocated()))
        save()
        print(phase, report["samples"][-1]["total_pss_bytes"], flush=True)

    save()
    try:
        sample("before_construction")
        with se.make("OGB" + args.family, num_envs=args.batch, device="cuda", seed=17, size=64) as env:
            workers = [r["pid"] for r in env._simulator.worker_info]
            assert len(workers) == 8 and env.backend == "auto"
            report["workers"] = env._simulator.worker_info
            sample("constructed")
            initial = env.get_state()
            report["state_bytes"] = sum(t.numel() * t.element_size() for t in initial.values(True, True))
            sample("initial_state_owned")
            actions = np.random.default_rng(31).uniform(-.5, .5, (args.batch, 8, 5))
            for horizon in (1, 8):
                a = actions[:, 0] if horizon == 1 else actions
                for selection, frames in (("none", None), ("final", slice(-1, None)), ("all", slice(None))):
                    # Default all-State retention remains included. Restore is
                    # outside timing but its production memory remains visible.
                    env.set_state(initial)
                    expected = env.step(a, frames=frames)
                    assert bool(expected["valid"].all())
                    sample(f"T{horizon}_{selection}_result_owned")
                    env.set_state(initial)
                    actual = env.step(a, frames=frames)
                    for key in expected.keys(True, True):
                        torch.testing.assert_close(actual[key], expected[key], rtol=0, atol=0, equal_nan=True)
                    del expected, actual
                    seconds = []
                    for _ in range(6):
                        env.set_state(initial)
                        torch.cuda.synchronize()
                        start = time.perf_counter()
                        result = env.step(a, frames=frames)
                        torch.cuda.synchronize()
                        seconds.append(time.perf_counter() - start)
                        assert bool(result["valid"].all())
                        del result
                    rates = [args.batch * horizon / s for s in seconds]
                    report["timings"].append(dict(horizon=horizon, frames=selection, seconds=seconds,
                        steps_per_s_mean=statistics.mean(rates), steps_per_s_sd=statistics.stdev(rates),
                        frame_count=args.batch * (horizon if selection == "all" else int(selection == "final")),
                        exact_replay="PASS"))
                    sample(f"T{horizon}_{selection}_outputs_released")
            del initial
        workers = []
        sample("closed")
        report["status"] = "PASS"
    except BaseException as exc:
        report.update(status="FAIL", error=repr(exc))
        raise
    finally:
        save()


if __name__ == "__main__":
    main()
