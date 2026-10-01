"""Compare 4 processes x 2 threads with 8 processes x 1 thread on equal CPU resources.

Uses upstream AsyncVectorEnv shared-memory/pipe transport. Benchmark-only shard
facade: no production State/lifecycle acceptance or rendering speed claim.
"""

from __future__ import annotations

import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import resource
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gymnasium as gym
import numpy as np
from fetch_scaling import Shard


def process_usage():
    r = resource.getrusage(resource.RUSAGE_SELF)
    return np.array([time.process_time(), r.ru_nvcsw, r.ru_nivcsw, r.ru_inblock, r.ru_oublock])


class HybridShard(Shard):
    def __init__(self, start, count, threads):
        super().__init__(start, count)
        self.pool = ThreadPoolExecutor(threads) if threads > 1 else None
        self.lanes = np.array_split(np.arange(count), threads)

    def _advance(self, lanes, actions):
        return [(int(i), self.envs[i].step(actions[i])) for i in lanes]

    def step(self, actions):
        before = process_usage()
        if self.pool is None:
            updates = self._advance(self.lanes[0], actions)
        else:
            futures = [self.pool.submit(self._advance, lanes, actions) for lanes in self.lanes]
            updates, errors = [], []
            for future in futures:
                try:
                    updates.extend(future.result())
                except Exception as error:
                    errors.append(error)
            if errors:
                raise errors[0]
        outputs = [None] * self.count
        for lane, output in updates:
            outputs[lane] = output
        result = self.pack(outputs)
        delta = process_usage() - before
        info = dict(
            zip(
                ("cpu_seconds", "voluntary_switches", "involuntary_switches", "disk_blocks_in", "disk_blocks_out"),
                delta,
            )
        )
        return result, 0.0, False, False, info

    def close(self):
        if self.pool is not None:
            self.pool.shutdown(wait=True)
        super().close()


def check(outputs, reference):
    assert len(outputs) == len(reference)
    for actual, expected in zip(outputs, reference):
        assert actual.keys() == expected.keys()
        for key in expected:
            np.testing.assert_array_equal(actual[key], expected[key])


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[64, 256, 1024])
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if args.repeats < 2 or any(b < 8 or b % 8 for b in args.batches):
        raise ValueError("Need at least two repeats and batches divisible by eight")
    report = dict(
        status="RUNNING",
        host=platform.node(),
        python=platform.python_version(),
        affinity=sorted(os.sched_getaffinity(0)),
        horizon=8,
        packages={n: importlib.metadata.version(n) for n in ["gymnasium", "gymnasium-robotics", "mujoco", "numpy"]},
        source_sha256={
            p.name: hashlib.sha256(p.read_bytes()).hexdigest()
            for p in [Path(__file__), Path(__file__).with_name("fetch_scaling.py")]
        },
        rows=[],
        gates=[],
        scope="Source FetchPush task + observations/reward/flags/success/qpos/qvel transport. No exhaustive State export, GPU transfers, or rendering.",
    )
    report["cpu_topology"] = {
        str(c): {
            f: (Path(f"/sys/devices/system/cpu/cpu{c}/topology") / f).read_text().strip()
            for f in ["physical_package_id", "core_id", "thread_siblings_list"]
        }
        for c in os.sched_getaffinity(0)
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for batch in args.batches:
        actions = np.random.default_rng(19).uniform(-1, 1, (8, batch, 4)).astype(np.float64)
        serial = Shard(0, batch)
        try:
            serial.reset()
            reference = [serial.step(a)[0] for a in actions]
        finally:
            serial.close()
        del serial
        # Reverse configuration order at batch 256 to check more than one ordering.
        configs = [(8, 1, True), (4, 2, True), (8, 1, False), (4, 2, False)]
        if batch == 256:
            configs.reverse()
        # Hold only one configuration's live simulators at once to bound memory.
        for processes, threads, shared in configs:
            count = batch // processes
            start = time.perf_counter()
            vector = gym.vector.AsyncVectorEnv(
                [functools.partial(HybridShard, i * count, count, threads) for i in range(processes)],
                shared_memory=shared,
                copy=True,
                context="spawn",
                autoreset_mode=gym.vector.AutoresetMode.DISABLED,
            )
            construction = time.perf_counter() - start

            def run():
                totals = {
                    k: 0.0
                    for k in [
                        "cpu_seconds",
                        "voluntary_switches",
                        "involuntary_switches",
                        "disk_blocks_in",
                        "disk_blocks_out",
                    ]
                }
                outputs = []
                for a in actions:
                    obs, _, _, _, info = vector.step(a.reshape(processes, count, 4))
                    outputs.append({k: v.reshape(batch, *v.shape[2:]).copy() for k, v in obs.items()})
                    for k in totals:
                        totals[k] += float(np.sum(info[k]))
                return outputs, totals

            try:
                vector.reset()
                out, _ = run()
                check(out, reference)
                # Owned observations must survive both subsequent stepping and reset.
                frozen = {k: v.copy() for k, v in out[0].items()}
                vector.reset()
                run()
                for k, v in frozen.items():
                    np.testing.assert_array_equal(out[0][k], v)
                report["gates"].append(
                    dict(
                        batch=batch,
                        processes=processes,
                        threads_per_process=threads,
                        shared_memory=shared,
                        exact_task_qpos_qvel_and_ownership="PASS",
                    )
                )
                samples = []
                for _ in range(args.repeats):
                    vector.reset()
                    parent_before = time.process_time()
                    start = time.perf_counter()
                    out, metrics = run()
                    elapsed = time.perf_counter() - start
                    parent_cpu = time.process_time() - parent_before
                    check(out, reference)  # outside timing, each repeat remains an exact trajectory
                    samples.append(
                        dict(
                            seconds=elapsed,
                            steps_s=batch * 8 / elapsed,
                            worker_cpu_cores=metrics["cpu_seconds"] / elapsed,
                            parent_cpu_cores=parent_cpu / elapsed,
                            **metrics,
                        )
                    )
                row = dict(
                    batch=batch,
                    processes=processes,
                    threads_per_process=threads,
                    shared_memory=shared,
                    construction_seconds=construction,
                    repeats=args.repeats,
                    steps_s_mean=statistics.mean(s["steps_s"] for s in samples),
                    steps_s_sd=statistics.stdev(s["steps_s"] for s in samples),
                    samples=samples,
                )
                report["rows"].append(row)
                print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
                save()
            finally:
                vector.close()
        del reference
    report["status"] = "PASS"
    save()
    print("FETCH_HYBRID_PASS", flush=True)


if __name__ == "__main__":
    main()
