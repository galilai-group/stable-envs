"""Direct GIL occupancy and unprofiled Fetch batch scaling on 8 + 1 CPU cores.

Benchmark-only CPython 3.12.9/Linux probe. Native observer reads the interpreter's
actual atomic lock flag, not Python frame presence or inferred wall/CPU gaps.
Reset, IPC waiting and parent copying are excluded from GIL observation windows.
Throughput includes task work, packing and transport, just as fetch_hybrid.py.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import inspect
import json
import os
import platform
import statistics
import subprocess
import sys
import sysconfig
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from fetch_hybrid import HybridShard, check
from fetch_scaling import Shard


class ProbeShard(HybridShard):
    def __init__(self, start, count, threads, cpus, observer_cpu):
        super().__init__(start, count, threads, cpus)
        self.observer_cpu = observer_cpu
        self.sampling = False

    def sample(self, enabled):
        import gil_probe

        if enabled:
            gil_probe.init(self.observer_cpu, 1000, 451 + self.start)
        else:
            gil_probe.close()
        self.sampling = enabled

    def step(self, actions):
        if not self.sampling:
            return super().step(actions)
        import gil_probe

        before = gil_probe.window(True)
        try:
            obs, r, term, trunc, info = super().step(actions)
        finally:
            after = gil_probe.window(False)
        for name, a, b in zip(("gil_samples", "gil_held", "gil_invalid"), after, before):
            info[name] = a - b
        return obs, r, term, trunc, info

    def close(self):
        if self.sampling:
            self.sample(False)
        super().close()


def controls(observer_cpu):
    import gil_probe

    result = []
    gil_probe.init(observer_cpu, 1000, 731)
    try:
        for name in ("native_released", "native_held", "python_busy", "half_held", "sleep"):
            before = gil_probe.window(True)
            if name.startswith("native"):
                gil_probe.control_work(1.5, name == "native_released")
            elif name == "python_busy":
                until = time.perf_counter() + 1.5
                value = 0
                while time.perf_counter() < until:
                    value += 1
            elif name == "half_held":
                for _ in range(75):
                    gil_probe.control_work(0.01, False)
                    gil_probe.control_work(0.01, True)
            else:
                time.sleep(1.5)
            after = gil_probe.window(False)
            n, h, invalid = (a - b for a, b in zip(after, before))
            fraction = h / n
            assert n > 500 and invalid == 0
            if name in ("native_released", "sleep"):
                assert fraction < 0.02, (name, fraction)
            elif name == "half_held":
                assert 0.43 < fraction < 0.57, (name, fraction)
            else:
                assert fraction > 0.98, (name, fraction)
            result.append(dict(control=name, total=n, held=h, fraction=fraction, invalid=invalid))
    finally:
        gil_probe.close()
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[64, 128, 256, 512, 1024])
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--sample-seconds", type=float, default=3.0)
    p.add_argument("--blocks", type=int, default=2)
    p.add_argument("--controls-only", action="store_true")
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    assert sys.version_info[:3] == (3, 12, 9)
    affinity = sorted(os.sched_getaffinity(0))
    assert len(affinity) == 9 and all(b % 8 == 0 for b in args.batches)
    workload_cpus, observer_cpu = affinity[:8], affinity[8]
    os.sched_setaffinity(0, workload_cpus)
    root = Path(__file__).resolve().parent
    include = Path(sysconfig.get_paths()["include"])
    command = [
        "gcc",
        "-shared",
        "-fPIC",
        "-O2",
        "-std=c11",
        "-pthread",
        "-Wall",
        "-Wextra",
        "-I" + str(include),
        str(root / "gil_probe.c"),
        "-o",
        str(root / ("gil_probe" + sysconfig.get_config_var("EXT_SUFFIX"))),
    ]
    subprocess.run(command, check=True)
    files = [root / n for n in ("fetch_gil_sweep.py", "gil_probe.c", "fetch_hybrid.py", "fetch_scaling.py")]
    files += [
        include / "internal" / n
        for n in ("pycore_interp.h", "pycore_ceval_state.h", "pycore_gil.h", "pycore_atomic.h")
    ]
    report = dict(
        status="RUNNING",
        base_head="1f1952c396d706bafff9ed0724dadaecd4e56a60",
        host=platform.node(),
        python=sys.version,
        affinity=affinity,
        workload_cpus=workload_cpus,
        observer_cpu=observer_cpu,
        compile_command=command,
        controls=controls(observer_cpu),
        packages={n: importlib.metadata.version(n) for n in ("gymnasium", "gymnasium-robotics", "mujoco", "numpy")},
        hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        topology={
            str(c): {
                f: Path(f"/sys/devices/system/cpu/cpu{c}/topology/{f}").read_text().strip()
                for f in ("physical_package_id", "core_id", "thread_siblings_list")
            }
            for c in affinity
        },
        rows=[],
    )
    print("GIL_CONTROLS", json.dumps(report["controls"]), flush=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for batch in [] if args.controls_only else args.batches:
        actions = np.random.default_rng(19).uniform(-1, 1, (8, batch, 4)).astype(np.float64)
        serial = Shard(0, batch)
        report["upstream_hashes"] = {
            f: hashlib.sha256(Path(f).read_bytes()).hexdigest()
            for f in {
                inspect.getfile(getattr(serial.envs[0].unwrapped, n))
                for n in ("_set_action", "_mujoco_step", "_get_obs", "compute_reward")
            }
        }
        serial.reset()
        reference = [serial.step(a)[0] for a in actions]
        serial.close()
        del serial
        configs = [(8, 1, True), (4, 2, True), (8, 1, False), (4, 2, False)]
        for block in range(args.blocks):
            order = configs if (block + args.batches.index(batch)) % 2 == 0 else configs[::-1]
            for processes, threads, shared in order:
                count = batch // processes
                cpus = np.array_split(workload_cpus, processes)
                vector = gym.vector.AsyncVectorEnv(
                    [
                        functools.partial(ProbeShard, i * count, count, threads, cpus[i].tolist(), observer_cpu)
                        for i in range(processes)
                    ],
                    shared_memory=shared,
                    copy=True,
                    context="spawn",
                    autoreset_mode=gym.vector.AutoresetMode.DISABLED,
                )
                try:
                    for sampled in (False, True):
                        if sampled:
                            vector.call("sample", True)
                        samples = []
                        accumulated = 0.0
                        # One warmup chunk per mode; reset is never sampled.
                        vector.reset()
                        for a in actions:
                            vector.step(a.reshape(processes, count, 4))
                        while len(samples) < args.repeats or (sampled and accumulated < args.sample_seconds):
                            vector.reset()
                            outputs = []
                            metrics = {
                                k: np.zeros(processes)
                                for k in (
                                    "cpu_seconds",
                                    "voluntary_switches",
                                    "involuntary_switches",
                                    "disk_blocks_in",
                                    "disk_blocks_out",
                                )
                            }
                            if sampled:
                                metrics.update(
                                    {
                                        k: np.zeros(processes, dtype=np.int64)
                                        for k in ("gil_samples", "gil_held", "gil_invalid")
                                    }
                                )
                            start = time.perf_counter()
                            for a in actions:
                                obs, _, _, _, info = vector.step(a.reshape(processes, count, 4))
                                outputs.append({k: v.reshape(batch, *v.shape[2:]).copy() for k, v in obs.items()})
                                for k in metrics:
                                    metrics[k] += info[k]
                            elapsed = time.perf_counter() - start
                            check(outputs, reference)
                            sample = dict(
                                seconds=elapsed,
                                steps_s=batch * 8 / elapsed,
                                worker_cpu_cores=float(np.sum(metrics["cpu_seconds"]) / elapsed),
                                **{k: v.tolist() for k, v in metrics.items()},
                            )
                            if sampled:
                                assert not np.any(metrics["gil_invalid"])
                                assert np.all(metrics["gil_held"] <= metrics["gil_samples"])
                                sample["gil_occupancy"] = float(
                                    np.sum(metrics["gil_held"]) / np.sum(metrics["gil_samples"])
                                )
                            samples.append(sample)
                            accumulated += elapsed
                        row = dict(
                            batch=batch,
                            block=block,
                            processes=processes,
                            threads=threads,
                            shared=shared,
                            sampled=sampled,
                            exact_trajectory="PASS",
                            samples=samples,
                            steps_s_mean=statistics.mean(s["steps_s"] for s in samples),
                            steps_s_sd=statistics.stdev(s["steps_s"] for s in samples),
                        )
                        report["rows"].append(row)
                        print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
                        save()
                        if sampled:
                            vector.call("sample", False)
                finally:
                    vector.close()
    report["status"] = "PASS"
    save()
    print("FETCH_GIL_SWEEP_PASS", flush=True)


if __name__ == "__main__":
    main()
