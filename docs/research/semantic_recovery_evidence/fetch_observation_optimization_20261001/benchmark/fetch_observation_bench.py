"""Exact-source acceptance and paired throughput for cached Fetch observations."""

import argparse
import functools
import hashlib
import importlib.metadata
import json
import os
import platform
import statistics
import time
from pathlib import Path

import gymnasium as gym
import numpy as np
from fetch_hybrid import HybridShard, check
from fetch_observations import TASK_CLASSES
from fetch_scaling import Shard
from gymnasium_robotics.envs.fetch.pick_and_place import MujocoFetchPickAndPlaceEnv
from gymnasium_robotics.envs.fetch.push import MujocoFetchPushEnv
from gymnasium_robotics.envs.fetch.reach import MujocoFetchReachEnv
from gymnasium_robotics.envs.fetch.slide import MujocoFetchSlideEnv

SOURCE = dict(
    PickAndPlace=MujocoFetchPickAndPlaceEnv,
    Push=MujocoFetchPushEnv,
    Reach=MujocoFetchReachEnv,
    Slide=MujocoFetchSlideEnv,
)


def equal_obs(a, b):
    for key in a:
        np.testing.assert_array_equal(a[key], b[key])
        assert a[key].dtype == b[key].dtype


def gates():
    results = []
    for task, cls in SOURCE.items():
        for reward in ("sparse", "dense"):
            for dtype in (np.float32, np.float64):
                original = gym.wrappers.TimeLimit(cls(reward_type=reward), 50)
                candidate = gym.wrappers.TimeLimit(TASK_CLASSES[task](reward_type=reward), 50)
                try:
                    for seed in (17, 42):
                        obs, _ = original.reset(seed=seed)
                        got, _ = candidate.reset(seed=seed)
                        equal_obs(obs, got)
                        frozen = {k: v.copy() for k, v in got.items()}
                        actions = np.random.default_rng(seed + 100).uniform(-1.2, 1.2, (50, 4)).astype(dtype)
                        for action in actions:
                            expected, actual = original.step(action), candidate.step(action)
                            equal_obs(expected[0], actual[0])
                            for k in (1, 2, 3):
                                np.testing.assert_array_equal(expected[k], actual[k])
                            assert expected[4] == actual[4]
                            a, b = original.unwrapped, candidate.unwrapped
                            for key in ("qpos", "qvel", "qacc_warmstart", "ctrl", "mocap_pos", "mocap_quat"):
                                np.testing.assert_array_equal(getattr(a.data, key), getattr(b.data, key))
                            assert a.np_random.bit_generator.state == b.np_random.bit_generator.state
                            # Repeat observation calls must not change either physics or RNG.
                            equal_obs(a._get_obs(), b._get_obs())
                        equal_obs(got, frozen)
                        # Mutating returned observations must not poison subsequent reads.
                        actual[0]["observation"][:] = 999
                        equal_obs(original.unwrapped._get_obs(), candidate.unwrapped._get_obs())
                    results.append(
                        dict(
                            task=task,
                            reward=reward,
                            dtype=np.dtype(dtype).name,
                            seeds=[17, 42],
                            steps_per_seed=50,
                            exact="PASS",
                        )
                    )
                finally:
                    original.close()
                    candidate.close()
    return results


class CandidateShard(HybridShard):
    def __init__(self, start, count, threads, cpus, optimized):
        super().__init__(start, count, threads, cpus)
        if optimized:
            for i, env in enumerate(self.envs):
                env.close()
                self.envs[i] = gym.wrappers.TimeLimit(TASK_CLASSES["Push"](), 50)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--output", type=Path, required=True)
    p.add_argument("--batches", type=int, nargs="+", default=[1, 64, 128, 256, 512, 1024])
    p.add_argument("--repeats", type=int, default=5)
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    cpus = sorted(os.sched_getaffinity(0))
    assert len(cpus) == 8
    report = dict(
        status="RUNNING",
        base_head="30197dc376d9d5fd706d265ab714e6c1c8d0cbf8",
        host=platform.node(),
        affinity=cpus,
        packages={n: importlib.metadata.version(n) for n in ("gymnasium", "gymnasium-robotics", "mujoco", "numpy")},
        hashes={f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in Path(__file__).parent.glob("*.py")},
        gates=gates(),
        rows=[],
    )
    print("FETCH_OBSERVATION_GATES_PASS", flush=True)

    def save():
        args.output.write_text(json.dumps(report, indent=2) + "\n")

    save()
    for batch in args.batches:
        actions = np.random.default_rng(19).uniform(-1, 1, (8, batch, 4)).astype(np.float64)
        serial = Shard(0, batch)
        serial.reset()
        reference = [serial.step(a)[0] for a in actions]
        serial.close()
        del serial
        configs = (
            [(1, 1, False), (1, 1, True)] if batch == 1 else [(8, 1, False), (8, 1, True), (4, 2, False), (4, 2, True)]
        )
        for block in range(2):
            for processes, threads, optimized in configs if block == 0 else configs[::-1]:
                count = batch // processes
                groups = np.array_split(cpus, processes)
                if batch == 1:
                    direct = CandidateShard(0, 1, 1, [cpus[0]], optimized)
                    vector = None
                else:
                    direct = None
                    vector = gym.vector.AsyncVectorEnv(
                        [
                            functools.partial(CandidateShard, i * count, count, threads, groups[i].tolist(), optimized)
                            for i in range(processes)
                        ],
                        shared_memory=True,
                        copy=True,
                        context="spawn",
                        autoreset_mode=gym.vector.AutoresetMode.DISABLED,
                    )
                try:

                    def reset():
                        (vector if vector is not None else direct).reset()

                    def run():
                        outputs = []
                        cpu = 0.0
                        switches = 0.0
                        for a in actions:
                            if vector is not None:
                                obs, _, _, _, info = vector.step(a.reshape(processes, count, 4))
                                obs = {k: v.reshape(batch, *v.shape[2:]).copy() for k, v in obs.items()}
                            else:
                                obs, _, _, _, info = direct.step(a)
                            outputs.append(obs)
                            cpu += float(np.sum(info["cpu_seconds"]))
                            switches += float(np.sum(info["voluntary_switches"]))
                        return outputs, cpu, switches

                    reset()
                    out, _, _ = run()
                    check(out, reference)
                    samples = []
                    for _ in range(args.repeats):
                        reset()
                        start = time.perf_counter()
                        out, cpu, switches = run()
                        elapsed = time.perf_counter() - start
                        check(out, reference)
                        samples.append(
                            dict(
                                seconds=elapsed,
                                steps_s=batch * 8 / elapsed,
                                worker_cpu_cores=cpu / elapsed,
                                voluntary_switches=switches,
                            )
                        )
                    row = dict(
                        batch=batch,
                        block=block,
                        processes=processes,
                        threads=threads,
                        optimized=optimized,
                        exact="PASS",
                        samples=samples,
                        steps_s_mean=statistics.mean(s["steps_s"] for s in samples),
                        steps_s_sd=statistics.stdev(s["steps_s"] for s in samples),
                    )
                    report["rows"].append(row)
                    save()
                    print(json.dumps({k: v for k, v in row.items() if k != "samples"}), flush=True)
                finally:
                    (vector if vector is not None else direct).close()
    report["status"] = "PASS"
    save()
    print("FETCH_OBSERVATION_BENCH_PASS", flush=True)


if __name__ == "__main__":
    main()
