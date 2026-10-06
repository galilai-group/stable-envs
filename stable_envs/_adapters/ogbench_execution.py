import multiprocessing as mp
import os
import sys

from .ogbench_maze_process import OGBMazeProcessAdapter


def maze_process_count(num_envs, cpus):
    available = len(cpus) if cpus is not None else 1
    if sys.platform == "linux" and num_envs >= 64 and available >= 4 and not mp.current_process().daemon:
        return min(8, available)
    return 0


def make_maze_adapter(num_envs, device, seed, local_factory):
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    count = maze_process_count(num_envs, cpus)
    if count:
        return OGBMazeProcessAdapter(num_envs, device, seed, processes=count, cpus=cpus)
    return local_factory(num_envs, device, seed)


def manipulation_process_count(num_envs, cpus):
    available = len(cpus) if cpus is not None else 1
    if sys.platform == "linux" and num_envs >= 64 and available >= 4 and not mp.current_process().daemon:
        return min(8, available)
    return 0


def make_manipulation_adapter(family, num_envs, device, seed, reward_task_id, local_factory):
    cpus = sorted(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else None
    count = manipulation_process_count(num_envs, cpus)
    if count:
        from .ogbench_manipulation_process import OGBManipulationProcessAdapter

        return OGBManipulationProcessAdapter(
            family, num_envs, device, seed, reward_task_id, processes=count, cpus=cpus
        )
    return local_factory(family, num_envs, device, seed, reward_task_id)
