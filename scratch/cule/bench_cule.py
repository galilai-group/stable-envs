"""CuLE throughput vs number of parallel environments, against ale-py on one core.

Two configurations, because they answer different questions:
  * "rl"  : 84x84 grayscale, frameskip 4, sticky actions off -- the standard RL
            pipeline, and what CuLE's paper reports.
  * "raw" : 210x160 RGB, frameskip 1 -- one emulated frame per step, which is
            the configuration a frame-storage argument is about.

Every number is emulator frames/s (= agent steps/s x frameskip) and agent
steps/s, both reported.  The ale-py baseline runs on one core of the same node
in the same configuration.

Usage: python bench_cule.py <out.json> [--games pong,breakout] [--max-envs 4096]
"""
import argparse
import json
import os
import time

import numpy as np
import torch

BATCHES = [1, 8, 64, 256, 1024, 4096]


def bench_ale_serial(rom, frameskip, color, total_steps):
    import ale_py
    from ale_py import roms

    ale = ale_py.ALEInterface()
    ale.setLoggerMode(ale_py.LoggerMode.Error)
    ale.setInt("frame_skip", 1)
    ale.setFloat("repeat_action_probability", 0.0)
    ale.setInt("random_seed", 0)
    ale.loadROM(roms.get_rom_path(rom))
    acts = ale.getMinimalActionSet()
    rng = np.random.default_rng(0)
    h, w = ale.getScreenDims()
    gray = np.empty((h, w), dtype=np.uint8)
    rgb = np.empty((h, w, 3), dtype=np.uint8)

    def run(n):
        ale.reset_game()
        for _ in range(n):
            a = acts[rng.integers(len(acts))]
            for _ in range(frameskip):
                ale.act(a)
            if color == "gray":
                ale.getScreenGrayscale(gray)
            else:
                ale.getScreenRGB(rgb)
            if ale.game_over():
                ale.reset_game()
        return n

    run(100)
    t0 = time.perf_counter()
    run(total_steps)
    dt = time.perf_counter() - t0
    return total_steps / dt


def bench_cule(rom, num_envs, device, frameskip, color, rescale, total_steps):
    from torchcule.atari import Env

    env = Env(rom, num_envs, color_mode=color, device=device, rescale=rescale,
              frameskip=frameskip, repeat_prob=0.0, episodic_life=False,
              max_noop_steps=30)
    env.train(frameskip)
    env.reset(initial_steps=frameskip)
    n_act = env.action_space.n

    def run(n):
        for _ in range(n):
            a = torch.randint(n_act, (num_envs,), device=device, dtype=torch.uint8)
            env.step(a)

    run(5)
    if device == "cuda":
        torch.cuda.synchronize()
    t0 = time.perf_counter()
    run(total_steps)
    if device == "cuda":
        torch.cuda.synchronize()
    dt = time.perf_counter() - t0
    mem = torch.cuda.max_memory_allocated() / 2**20 if device == "cuda" else None
    del env
    if device == "cuda":
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()
    return total_steps * num_envs / dt, mem


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("out")
    ap.add_argument("--games", default="pong,breakout,space_invaders")
    ap.add_argument("--max-envs", type=int, default=4096)
    args = ap.parse_args()

    games = args.games.split(",")
    batches = [b for b in BATCHES if b <= args.max_envs]
    res = {"node": os.uname().nodename, "batches": batches, "games": games,
           "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
           "cpus": int(os.environ.get("SLURM_CPUS_PER_TASK", os.cpu_count())),
           "configs": {}}

    cfgs = {
        "rl":  dict(frameskip=4, color="gray", rescale=True),
        "raw": dict(frameskip=1, color="rgb", rescale=False),
    }

    for cname, cfg in cfgs.items():
        res["configs"][cname] = {"cfg": cfg, "ale_serial": {}, "cule_gpu": {}, "cule_cpu": {}}
        for g in games:
            res["configs"][cname]["ale_serial"][g] = bench_ale_serial(
                g, cfg["frameskip"], cfg["color"], 2000 if cfg["frameskip"] == 4 else 5000)
            print("ale", cname, g, res["configs"][cname]["ale_serial"][g], flush=True)
        json.dump(res, open(args.out, "w"), indent=1)

        for n in batches:
            for g in games:
                steps = max(20, min(2000, 400000 // (n * cfg["frameskip"])))
                for dev, key in (("cuda", "cule_gpu"), ("cpu", "cule_cpu")):
                    if dev == "cpu" and n > 256:
                        continue
                    try:
                        fps, mem = bench_cule(g, n, dev, cfg["frameskip"],
                                              cfg["color"], cfg["rescale"], steps)
                        val = {"steps_per_s": fps,
                               "frames_per_s": fps * cfg["frameskip"],
                               "peak_mem_mb": mem, "iters": steps}
                    except Exception as e:  # noqa: BLE001
                        val = {"error": repr(e)}
                    res["configs"][cname][key].setdefault(g, {})[n] = val
                    print(cname, key, g, n, val, flush=True)
            json.dump(res, open(args.out, "w"), indent=1)

    json.dump(res, open(args.out, "w"), indent=1)
    print("WROTE", args.out)


if __name__ == "__main__":
    main()
