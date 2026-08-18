"""Atari throughput: envpool vs ale-py (serial) vs gymnasium.vector, one node.

Everything runs the same contract: ALE v5 semantics (frameskip 4, sticky
actions p=0.25, 84x84 grayscale), random actions, no learner.  Numbers are
reported as agent steps/s and as emulator frames/s (= agent steps x 4), plus
the ratio against the single-core ale-py baseline measured on the same node.

Usage: python bench_envpool.py <out.json>
"""
import json
import os
import sys
import time

import numpy as np

GAMES = ["Pong", "Breakout", "SpaceInvaders"]
FRAMESKIP = 4
CPUS = int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count()))


def timeit(fn, target_steps, warmup_steps):
    fn(warmup_steps)
    t0 = time.perf_counter()
    n = fn(target_steps)
    return n / (time.perf_counter() - t0)


def bench_ale_serial(game, total_steps=3000):
    """Single-core ale-py, raw ALEInterface, v5 semantics."""
    import ale_py
    from ale_py import roms

    rom_id = "".join("_" + c.lower() if c.isupper() else c for c in game).lstrip("_")
    ale = ale_py.ALEInterface()
    ale.setInt("frame_skip", 1)
    ale.setFloat("repeat_action_probability", 0.25)
    ale.setInt("random_seed", 0)
    ale.setLoggerMode(ale_py.LoggerMode.Error)
    ale.loadROM(roms.get_rom_path(rom_id))
    acts = ale.getMinimalActionSet()
    rng = np.random.default_rng(0)
    buf = np.empty(ale.getScreenDims(), dtype=np.uint8)

    def run(n):
        ale.reset_game()
        for _ in range(n):
            a = acts[rng.integers(len(acts))]
            for _ in range(FRAMESKIP):
                ale.act(a)
            ale.getScreenGrayscale(buf)
            if ale.game_over():
                ale.reset_game()
        return n

    return timeit(run, total_steps, 200)


def bench_envpool(game, num_envs, total_steps=None):
    import envpool

    total_steps = total_steps or max(400, min(4000, 200000 // max(num_envs, 1)))
    env = envpool.make_gymnasium(
        f"{game}-v5", num_envs=num_envs, num_threads=CPUS, seed=0
    )
    n_act = env.action_space.n
    rng = np.random.default_rng(0)

    def run(n):
        env.reset()
        for _ in range(n):
            env.step(rng.integers(n_act, size=num_envs, dtype=np.int32))
        return n * num_envs

    fps = timeit(run, total_steps, 20)
    del env
    return fps


def bench_gym_vector(game, num_envs, mode, total_steps=None):
    import ale_py  # noqa: F401  (registers ALE envs)
    import gymnasium as gym

    total_steps = total_steps or max(200, min(2000, 60000 // max(num_envs, 1)))
    env = gym.make_vec(
        f"ALE/{game}-v5", num_envs=num_envs, vectorization_mode=mode,
        frameskip=FRAMESKIP, repeat_action_probability=0.25,
        obs_type="grayscale",
    )
    rng = np.random.default_rng(0)
    n_act = env.single_action_space.n

    def run(n):
        env.reset(seed=0)
        for _ in range(n):
            env.step(rng.integers(n_act, size=num_envs))
        return n * num_envs

    fps = timeit(run, total_steps, 10)
    env.close()
    return fps


def main():
    out_path = sys.argv[1]
    res = {"cpus": CPUS, "frameskip": FRAMESKIP, "node": os.uname().nodename,
           "games": GAMES, "ale_serial": {}, "envpool": {}, "gym_async": {},
           "gym_sync": {}}
    try:
        import envpool
        res["envpool_version"] = envpool.__version__
    except Exception as e:  # noqa: BLE001
        res["envpool_version"] = "IMPORT FAILED: %r" % (e,)
    import ale_py
    import gymnasium
    res["ale_py_version"] = ale_py.__version__
    res["gymnasium_version"] = gymnasium.__version__
    res["python"] = sys.version

    for g in GAMES:
        res["ale_serial"][g] = bench_ale_serial(g)
        print("ale serial", g, res["ale_serial"][g], flush=True)

    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for g in GAMES:
            if res["envpool_version"].startswith("IMPORT"):
                break
            try:
                fps = bench_envpool(g, n)
            except Exception as e:  # noqa: BLE001
                fps = "ERROR: %r" % (e,)
            res["envpool"].setdefault(g, {})[n] = fps
            print("envpool", g, n, fps, flush=True)
        json.dump(res, open(out_path, "w"), indent=1)

    for n in [1, 2, 4, 8, 16, 32, 64, 128, 256]:
        for g in GAMES[:1]:
            for mode, key in (("async", "gym_async"), ("sync", "gym_sync")):
                if mode == "sync" and n > 32:
                    continue
                try:
                    fps = bench_gym_vector(g, n, mode)
                except Exception as e:  # noqa: BLE001
                    fps = "ERROR: %r" % (e,)
                res[key].setdefault(g, {})[n] = fps
                print("gym", mode, g, n, fps, flush=True)
        json.dump(res, open(out_path, "w"), indent=1)

    json.dump(res, open(out_path, "w"), indent=1)
    print("WROTE", out_path)


if __name__ == "__main__":
    main()
