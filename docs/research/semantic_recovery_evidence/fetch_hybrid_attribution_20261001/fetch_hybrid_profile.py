"""Attribute Fetch hybrid overhead; separate uninstrumented and timed passes.

Benchmark only. Uses the exact source task and transport from fetch_hybrid.py.
Method wall time includes descheduling/GIL reacquisition; it is not native CPU
time. Per-thread CPU time excludes waiting. Parallel durations are not additive.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import inspect
import json
import multiprocessing
import os
import platform
import resource
import statistics
import sys
import threading
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from fetch_hybrid import HybridShard, check
from fetch_scaling import Shard

METHODS = ("_set_action", "_mujoco_step", "_step_callback", "_get_obs", "_is_success", "compute_reward")


class ProfileShard(HybridShard):
    def __init__(self, start, count, threads, cpus, pin_threads=False, switch_interval=0.005):
        super().__init__(start, count, threads, cpus)
        if multiprocessing.current_process().name != "MainProcess":
            sys.setswitchinterval(switch_interval)
        self.cpus = cpus
        self.pin_threads = pin_threads
        self.instrument = False
        self.local = threading.local()
        self.originals = []
        self.records = []

    def configure(self, instrument):
        self.instrument = instrument
        self.records = []
        if instrument and not self.originals:
            for env in self.envs:
                env = env.unwrapped
                for name in METHODS:
                    method = getattr(env, name)
                    self.originals.append((env, name, method))

                    def timed(*args, _method=method, _name=name, **kwargs):
                        metrics = getattr(self.local, "metrics", None)
                        if metrics is None:
                            return _method(*args, **kwargs)
                        wall, cpu = time.perf_counter(), time.thread_time()
                        result = _method(*args, **kwargs)
                        metrics[_name + "_cpu"] += time.thread_time() - cpu
                        metrics[_name + "_wall"] += time.perf_counter() - wall
                        return result

                    setattr(env, name, timed)
        elif not instrument and self.originals:
            for env, name, method in self.originals:
                setattr(env, name, method)
            self.originals = []

    def _advance(self, lanes, actions):
        if self.pin_threads:
            group = next(i for i, group in enumerate(self.lanes) if group is lanes)
            os.sched_setaffinity(0, [self.cpus[group]])
        if not self.instrument:
            return super()._advance(lanes, actions)
        metrics = {name + suffix: 0.0 for name in METHODS for suffix in ("_wall", "_cpu")}
        self.local.metrics = metrics
        usage = resource.getrusage(resource.RUSAGE_THREAD)
        cpu, start = time.thread_time(), time.perf_counter()
        try:
            outputs = super()._advance(lanes, actions)
        finally:
            self.local.metrics = None
        end = time.perf_counter()
        after = resource.getrusage(resource.RUSAGE_THREAD)
        metrics.update(
            start=start,
            end=end,
            task_wall=end - start,
            task_cpu=time.thread_time() - cpu,
            voluntary=after.ru_nvcsw - usage.ru_nvcsw,
            involuntary=after.ru_nivcsw - usage.ru_nivcsw,
        )
        return outputs, metrics

    def step(self, actions):
        if not self.instrument:
            return super().step(actions)
        start = time.perf_counter()
        if self.pool is None:
            result = self._advance(self.lanes[0], actions)
            submit_end = start
            groups = [result]
        else:
            futures = [self.pool.submit(self._advance, lanes, actions) for lanes in self.lanes]
            submit_end = time.perf_counter()
            groups = [f.result() for f in futures]
        joined = time.perf_counter()
        outputs = [None] * self.count
        for updates, _ in groups:
            for lane, output in updates:
                outputs[lane] = output
        pack_start = time.perf_counter()
        result = self.pack(outputs)
        end = time.perf_counter()
        self.records.append(
            dict(
                start=start,
                submit_end=submit_end,
                joined=joined,
                pack_start=pack_start,
                end=end,
                workers=[m for _, m in groups],
            )
        )
        return result, 0.0, False, False, {}

    def drain_records(self):
        result, self.records = self.records, []
        return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batch", type=int, default=256)
    p.add_argument("--repeats", type=int, default=7)
    p.add_argument("--profile-repeats", type=int, default=3)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    affinity = sorted(os.sched_getaffinity(0))
    assert len(affinity) == 8 and args.batch % 8 == 0
    scripts = [
        Path(__file__),
        Path(__file__).with_name("fetch_hybrid.py"),
        Path(__file__).with_name("fetch_scaling.py"),
    ]
    report = dict(
        status="RUNNING",
        head="722fa870b80453934b7d9803305556e3c3729954",
        host=platform.node(),
        python=sys.version,
        affinity=affinity,
        batch=args.batch,
        horizon=8,
        packages={n: importlib.metadata.version(n) for n in ["gymnasium", "gymnasium-robotics", "mujoco", "numpy"]},
        hashes={str(f): hashlib.sha256(f.read_bytes()).hexdigest() for f in scripts},
        rows=[],
    )
    report["topology"] = {
        str(c): {
            f: Path(f"/sys/devices/system/cpu/cpu{c}/topology/{f}").read_text().strip()
            for f in ("physical_package_id", "core_id", "thread_siblings_list")
        }
        for c in affinity
    }
    actions = np.random.default_rng(19).uniform(-1, 1, (8, args.batch, 4)).astype(np.float64)
    serial = Shard(0, args.batch)
    upstream = {inspect.getfile(getattr(serial.envs[0].unwrapped, name)) for name in METHODS}
    report["upstream_hashes"] = {f: hashlib.sha256(Path(f).read_bytes()).hexdigest() for f in upstream}
    serial.reset()
    reference = [serial.step(a)[0] for a in actions]
    serial.close()
    del serial
    configs = [
        ("8p1t", 8, 1, False, 0.005),
        ("4p2t", 4, 2, False, 0.005),
        ("4p2t_pin", 4, 2, True, 0.005),
        ("4p2t_switch100us", 4, 2, False, 0.0001),
    ]
    # Reverse order in block two to expose drift/order sensitivity.
    for block in range(2):
        for name, processes, threads, pin, switch in configs if block == 0 else configs[::-1]:
            count = args.batch // processes
            cpus = np.array_split(affinity, processes)
            vector = gym.vector.AsyncVectorEnv(
                [
                    functools.partial(ProfileShard, i * count, count, threads, cpus[i].tolist(), pin, switch)
                    for i in range(processes)
                ],
                shared_memory=True,
                copy=True,
                context="spawn",
                autoreset_mode=gym.vector.AutoresetMode.DISABLED,
            )
            try:
                for instrument in (False, True):
                    vector.call("configure", instrument)
                    vector.reset()
                    for a in actions:
                        vector.step(a.reshape(processes, count, 4))
                    vector.call("drain_records")
                    samples = []
                    for _ in range(args.profile_repeats if instrument else args.repeats):
                        vector.reset()
                        out, calls = [], []
                        before = time.perf_counter()
                        for a in actions:
                            call_start = time.perf_counter()
                            obs, _, _, _, _ = vector.step(a.reshape(processes, count, 4))
                            returned = time.perf_counter()
                            out.append({k: v.reshape(args.batch, *v.shape[2:]).copy() for k, v in obs.items()})
                            calls.append(dict(start=call_start, returned=returned, copied=time.perf_counter()))
                        elapsed = time.perf_counter() - before
                        check(out, reference)
                        samples.append(
                            dict(
                                seconds=elapsed,
                                steps_s=args.batch * 8 / elapsed,
                                calls=calls,
                                children=vector.call("drain_records") if instrument else [],
                            )
                        )
                    row = dict(
                        block=block,
                        config=name,
                        processes=processes,
                        threads=threads,
                        instrumented=instrument,
                        exact_trajectory="PASS",
                        samples=samples,
                        steps_s_mean=statistics.mean(s["steps_s"] for s in samples),
                        steps_s_sd=statistics.stdev(s["steps_s"] for s in samples),
                    )
                    report["rows"].append(row)
                    args.output.write_text(json.dumps(report, indent=2) + "\n")
                    print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
            finally:
                vector.close()
    report["status"] = "PASS"
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    print("FETCH_HYBRID_PROFILE_PASS", flush=True)


if __name__ == "__main__":
    main()
