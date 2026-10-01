"""Diagnose Fetch CPU scaling; no GPU work and no production executor changes.

Compare disjoint threaded owners with Gymnasium AsyncVectorEnv's existing
process/shared-observation transport, using bounded shards. Engine-only rows
are a diagnostic lower bound, not an equivalent task implementation.
"""

from __future__ import annotations

import argparse
import cProfile
import functools
import io
import json
import os
import platform
import pstats
import resource
import statistics
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import gymnasium as gym
import mujoco
import numpy as np
from gymnasium_robotics.envs.fetch.push import MujocoFetchPushEnv


def usage():
    r = resource.getrusage(resource.RUSAGE_THREAD)
    return np.array([time.thread_time(), r.ru_nvcsw, r.ru_nivcsw, r.ru_inblock, r.ru_oublock])


class Shard(gym.Env):
    """Benchmark-only Gym facade for several separately seeded scalar tasks.

    Rewards/flags are retained in the observation payload so a shard does not
    collapse several terminal lanes into one Gym termination flag. The measured
    eight-step chunks never reach TimeLimit. No production lifecycle claim.
    """

    metadata = {}

    def __init__(self, start, count, engine_only=False):
        self.start, self.count, self.engine_only = start, count, engine_only
        self.envs = [gym.wrappers.TimeLimit(MujocoFetchPushEnv(), 50) for _ in range(count)]
        self.action_space = gym.spaces.Box(-1, 1, (count, 4), dtype=np.float64)
        self.observation_space = gym.spaces.Dict(
            {
                "observation": gym.spaces.Box(-np.inf, np.inf, (count, 25), dtype=np.float64),
                "achieved_goal": gym.spaces.Box(-np.inf, np.inf, (count, 3), dtype=np.float64),
                "desired_goal": gym.spaces.Box(-np.inf, np.inf, (count, 3), dtype=np.float64),
                "task": gym.spaces.Box(-np.inf, np.inf, (count, 4), dtype=np.float64),
                "qpos": gym.spaces.Box(-np.inf, np.inf, (count, self.envs[0].unwrapped.model.nq), dtype=np.float64),
                "qvel": gym.spaces.Box(-np.inf, np.inf, (count, self.envs[0].unwrapped.model.nv), dtype=np.float64),
            }
        )
        self.last = None
        self.profile = False

    def reset(self, *, seed=None, options=None):
        outputs = [e.reset(seed=17 + self.start + i)[0] for i, e in enumerate(self.envs)]
        self.last = self.pack([(x, 0.0, False, False, {"is_success": 0.0}) for x in outputs])
        return self.last, {}

    def pack(self, outputs):
        result = {k: np.stack([x[0][k] for x in outputs]) for k in ("observation", "achieved_goal", "desired_goal")}
        result["task"] = np.array([[x[1], x[2], x[3], x[4]["is_success"]] for x in outputs], dtype=np.float64)
        result["qpos"] = np.stack([e.unwrapped.data.qpos.copy() for e in self.envs])
        result["qvel"] = np.stack([e.unwrapped.data.qvel.copy() for e in self.envs])
        return result

    def step(self, actions):
        before = usage()
        prof = cProfile.Profile() if self.profile else None
        if prof:
            prof.enable()
        if self.engine_only:
            for e in self.envs:
                u = e.unwrapped
                mujoco.mj_step(u.model, u.data, nstep=u.n_substeps)
            result = self.last
        else:
            result = self.pack([e.step(a) for e, a in zip(self.envs, actions)])
        if prof:
            prof.disable()
        metrics = usage() - before
        info = {
            "cpu_seconds": metrics[0],
            "voluntary_switches": metrics[1],
            "involuntary_switches": metrics[2],
            "disk_blocks_in": metrics[3],
            "disk_blocks_out": metrics[4],
        }
        if prof:
            stream = io.StringIO()
            pstats.Stats(prof, stream=stream).sort_stats("tottime").print_stats(25)
            info["profile"] = stream.getvalue()
        return result, 0.0, False, False, info

    def close(self):
        for e in self.envs:
            e.close()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batches", nargs="+", type=int, default=[64, 256])
    p.add_argument("--workers", nargs="+", type=int, default=[1, 2, 4, 8])
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    report = {
        "status": "RUNNING",
        "host": platform.node(),
        "affinity": sorted(os.sched_getaffinity(0)),
        "mujoco": mujoco.__version__,
        "gymnasium": gym.__version__,
        "rows": [],
        "gates": [],
        "profiles": [],
        "scope": "CPU-only source task/shard transport diagnostic; no full State or renderer acceptance",
    }
    report["cpu_topology"] = {
        str(c): {
            f: (Path(f"/sys/devices/system/cpu/cpu{c}/topology") / f).read_text().strip()
            for f in ("physical_package_id", "core_id", "thread_siblings_list")
        }
        for c in os.sched_getaffinity(0)
    }

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for batch in args.batches:
        actions = np.random.default_rng(19).uniform(-1, 1, (8, batch, 4)).astype(np.float64)
        reference = None
        for kind in ("threads", "async_shared", "async_pipe", "engine_threads"):
            for workers in args.workers:
                if batch % workers:
                    continue
                n = batch // workers
                factories = [functools.partial(Shard, i * n, n, kind == "engine_threads") for i in range(workers)]
                vector = None
                pool = None
                if kind.startswith("async"):
                    vector = gym.vector.AsyncVectorEnv(
                        factories,
                        shared_memory=kind == "async_shared",
                        copy=True,
                        context="spawn",
                        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
                    )
                    shards = None
                else:
                    shards = [f() for f in factories]
                    pool = ThreadPoolExecutor(workers)

                def reset():
                    if vector:
                        vector.reset()
                    else:
                        list(pool.map(lambda s: s.reset(), shards))

                def run(profile=False):
                    if vector:
                        vector.set_attr("profile", profile)
                    else:
                        for s in shards:
                            s.profile = profile
                    totals = {
                        k: 0.0
                        for k in (
                            "cpu_seconds",
                            "voluntary_switches",
                            "involuntary_switches",
                            "disk_blocks_in",
                            "disk_blocks_out",
                        )
                    }
                    profiles = []
                    outputs = []
                    for a in actions:
                        if vector:
                            obs, _, _, _, info = vector.step(a.reshape(workers, n, 4))
                            for k in totals:
                                totals[k] += float(np.sum(info[k]))
                            if profile:
                                profiles.extend(info["profile"].tolist())
                            outputs.append({k: v.reshape(batch, *v.shape[2:]).copy() for k, v in obs.items()})
                        else:
                            results = list(
                                pool.map(lambda pair: pair[0].step(pair[1]), zip(shards, a.reshape(workers, n, 4)))
                            )
                            for _, _, _, _, info in results:
                                for k in totals:
                                    totals[k] += info[k]
                                if profile:
                                    profiles.append(info["profile"])
                            outputs.append({k: np.concatenate([r[0][k] for r in results]) for k in results[0][0]})
                    return outputs, totals, profiles

                try:
                    reset()
                    outputs, _, _ = run()
                    if kind != "engine_threads":
                        if reference is None:
                            reference = outputs
                        else:
                            for a, b in zip(reference, outputs):
                                for k in a:
                                    np.testing.assert_array_equal(a[k], b[k])
                        report["gates"].append(
                            {"batch": batch, "kind": kind, "workers": workers, "exact_task_qpos_qvel": "PASS"}
                        )
                    times = []
                    samples = []
                    for _ in range(args.repeats):
                        reset()
                        start = time.perf_counter()
                        _, metrics, _ = run()
                        elapsed = time.perf_counter() - start
                        times.append(elapsed)
                        samples.append(
                            dict(seconds=elapsed, worker_cpu_cores=metrics["cpu_seconds"] / elapsed, **metrics)
                        )
                    rates = [batch * 8 / t for t in times]
                    row = dict(
                        batch=batch,
                        workers=workers,
                        kind=kind,
                        steps_s_mean=statistics.mean(rates),
                        steps_s_sd=statistics.stdev(rates),
                        samples=samples,
                    )
                    report["rows"].append(row)
                    print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
                    if batch == 256 and workers in (1, 4, 8) and kind in ("threads", "async_shared"):
                        reset()
                        _, _, profiles = run(profile=True)
                        report["profiles"].append(
                            dict(batch=batch, workers=workers, kind=kind, first_timestep_workers=profiles[:workers])
                        )
                    save()
                finally:
                    if vector:
                        vector.close()
                    else:
                        pool.shutdown()
                        for s in shards:
                            s.close()
    report["status"] = "PASS"
    save()
    print("FETCH_SCALING_PASS", flush=True)


if __name__ == "__main__":
    main()
