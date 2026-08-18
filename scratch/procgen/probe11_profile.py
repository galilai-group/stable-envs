"""Where does the render time actually go? Phase timings, plus the official renderer's own rate."""
from __future__ import annotations

import time

import numpy as np
import torch

import stable_envs as se
from stable_envs.envs.procgen import _Assets, _Geom


def phase(fn, reps=10):
    for _ in range(3):
        fn()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(reps):
        fn()
    torch.cuda.synchronize()
    return (time.time() - t0) / reps * 1000


def main():
    print("device:", torch.cuda.get_device_name(0))
    B = 4096
    for name in ("ProcgenMaze", "ProcgenHeist", "ProcgenMiner"):
        e = se.make(name, batch=B, device="cuda")
        s = torch.as_tensor(e.sample_states(B), device="cuda")
        S = e.size
        gm = _Geom.get(e.game, S, e.device)
        atlas, wh = _Assets.atlas(e.game, e.device)
        flat = atlas.reshape(-1, 4).to(torch.int32)
        sty = e.resolve_style(None, None, batch=B)
        f = e._unpack(s)
        use_bg = sty["rendering.use_backgrounds"].reshape(B) > 0.5
        bgain = sty["background_gain"].reshape(B, 3).float()
        canvas = torch.zeros(B, S, S, 3, dtype=torch.int32, device=e.device)
        n_bg = int(torch.unique(f["bg_index"]).numel())
        t_all = phase(lambda: e.render(s))
        t_bg = phase(lambda: e._draw_background(canvas, f, gm, use_bg, bgain))
        t_grid = phase(lambda: e._draw_grid(canvas, f, gm, flat, atlas.shape, None))
        t_e = {z: phase(lambda z=z: e._draw_entities(canvas, f, gm, flat, atlas.shape, wh, z, None))
               for z in (-1, 0, 1)}
        t_alloc = phase(lambda: torch.zeros(B, S, S, 3, dtype=torch.int32, device=e.device))
        print(f"\n{name} B={B}: total {t_all:.1f} ms ({B/t_all*1000:.0f} f/s)")
        print(f"   alloc {t_alloc:6.2f}  background {t_bg:6.2f} ({n_bg} distinct)  "
              f"grid {t_grid:6.2f}  ents z-1/0/1 {t_e[-1]:.2f}/{t_e[0]:.2f}/{t_e[1]:.2f}")

    # official renderer, single process
    from procgen import ProcgenGym3Env
    for game in ("maze", "heist", "miner"):
        for num in (1, 64):
            env = ProcgenGym3Env(num=num, env_name=game, distribution_mode="hard", rand_seed=0,
                                 num_threads=8)
            a = np.zeros(num, np.int32) + 4
            for _ in range(20):
                env.act(a)
                env.observe()
            t0 = time.time()
            n = 200
            for _ in range(n):
                env.act(a)
                env.observe()
            dt = time.time() - t0
            print(f"official {game:6s} num={num:3d}: {num*n/dt:9.0f} frames/s "
                  f"(step+render, 8 threads)")


if __name__ == "__main__":
    main()
