"""Throughput of the GPU renderer, against the official CPU renderer.

Measured, separately, because they are different claims:
  * index      -- state tensors -> tile-atlas index tensor (view gather, process_vis)
  * blit       -- index tensor -> pixels
  * end2end    -- both, which is what a training loop actually pays
  * cudagraph  -- end2end replayed from a captured graph (removes launch overhead,
                  which dominates at small batch because process_vis is ~100 tiny kernels)

The CPU baseline is the official renderer with a warm tile cache, i.e. its best case.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
import warnings

import numpy as np
import torch

warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import gymnasium as gym  # noqa: E402
import minigrid  # noqa: E402,F401

from mg_gpu import MiniGridGPU, batch_states, env_state  # noqa: E402


def collect(env_id, n):
    env = gym.make(env_id)
    out = []
    ep = 0
    while len(out) < n:
        env.reset(seed=ep)
        rng = np.random.default_rng(ep)
        for _ in range(64):
            out.append(env_state(env))
            if len(out) >= n:
                break
            _, _, te, tr, _ = env.step(int(rng.integers(0, 7)))
            if te or tr:
                break
        ep += 1
    env.close()
    return out


def timeit(fn, iters, warmup=5):
    for _ in range(warmup):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(iters):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / iters


def cpu_baseline(env_id, n=200):
    env = gym.make(env_id)
    env.reset(seed=0)
    u = env.unwrapped
    u.get_full_render(True, 32)          # warm Grid.tile_cache
    u.get_pov_render(32)
    rng = np.random.default_rng(0)
    acts = [int(rng.integers(0, 7)) for _ in range(n)]
    t0 = time.perf_counter()
    for a in acts:
        u.get_full_render(True, 32)
        env.step(a)
    full = (time.perf_counter() - t0) / n
    t0 = time.perf_counter()
    for a in acts:
        u.get_pov_render(32)
        env.step(a)
    pov = (time.perf_counter() - t0) / n
    env.close()
    return 1.0 / full, 1.0 / pov


def run(gpu, states, B, mode, tile, iters):
    dev = "cuda"
    pool = [states[i % len(states)] for i in range(B)]
    code, pos, dirs, car = batch_states(pool, dev)
    V, stw = states[0]["view_size"], states[0]["see_through_walls"]

    if mode == "pov":
        mk = lambda: gpu.pov_indices(code, pos, dirs, car, V, stw)  # noqa: E731
        gh, gw = V, V
    else:
        mk = lambda: gpu.full_indices(code, pos, dirs, V, stw, True)  # noqa: E731
        gh, gw = int(code.shape[2]), int(code.shape[1])

    idx = mk()
    out = torch.empty(B, gh * tile, gw * tile, 3, dtype=torch.uint8, device=dev)
    bpf = gh * tile * gw * tile * 3

    t_idx = timeit(mk, iters)
    t_blit = timeit(lambda: gpu.blit_triton(idx, out=out), iters)
    t_blit_torch = timeit(lambda: gpu.blit_torch(idx), max(3, iters // 4))
    t_e2e = timeit(lambda: gpu.blit_triton(mk(), out=out), iters)

    # CUDA-graph replay of the same end-to-end work
    t_graph = float("nan")
    try:
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s):
            for _ in range(3):
                gpu.blit_triton(mk(), out=out)
        torch.cuda.current_stream().wait_stream(s)
        g = torch.cuda.CUDAGraph()
        with torch.cuda.graph(g):
            gpu.blit_triton(mk(), out=out)
        t_graph = timeit(g.replay, iters)
    except Exception as ex:  # noqa: BLE001
        t_graph = float("nan")
        print("  cudagraph failed:", type(ex).__name__, ex, flush=True)

    return {
        "mode": mode, "B": B, "tile": tile, "frame_px": [gh * tile, gw * tile],
        "bytes_per_frame": bpf,
        "fps_index": B / t_idx,
        "fps_blit_triton": B / t_blit,
        "fps_blit_torch": B / t_blit_torch,
        "fps_end2end": B / t_e2e,
        "fps_cudagraph": B / t_graph,
        "blit_GBps": bpf * B / t_blit / 1e9,
        "e2e_GBps": bpf * B / t_e2e / 1e9,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    here = os.path.dirname(os.path.abspath(__file__))

    print(torch.cuda.get_device_name(0), flush=True)
    results = {"gpu": torch.cuda.get_device_name(0), "runs": [], "cpu": {}}

    configs = [
        # env id,                        modes,           tile sizes
        ("MiniGrid-Empty-8x8-v0", ["pov", "full"], [32, 8]),
        ("MiniGrid-FourRooms-v0", ["pov", "full"], [32]),
        ("MiniGrid-MultiRoom-N4-S5-v0", ["full"], [32]),
    ]
    atlases = {}
    for t in (32, 8):
        z = np.load(os.path.join(here, f"atlas_{t}.npz"))
        atlases[t] = MiniGridGPU(z["atlas"], t, device="cuda")

    for env_id, modes, tiles in configs:
        states = collect(env_id, 256)
        f, p = cpu_baseline(env_id)
        results["cpu"][env_id] = {"fps_full_cpu": f, "fps_pov_cpu": p}
        print(f"\n{env_id}  CPU official: full {f:8.1f} fps   pov {p:8.1f} fps", flush=True)
        for tile in tiles:
            gpu = atlases[tile]
            for mode in modes:
                for B in (256, 1024, 4096):
                    bpf_est = (7 if mode == "pov" else 25) ** 2 * tile * tile * 3
                    if bpf_est * B > 3e9:
                        continue
                    try:
                        r = run(gpu, states, B, mode, tile, iters=20)
                    except torch.cuda.OutOfMemoryError:
                        torch.cuda.empty_cache()
                        continue
                    r["env_id"] = env_id
                    results["runs"].append(r)
                    print(f"  {mode:5s} T={tile:2d} B={B:5d} {r['frame_px'][0]}x{r['frame_px'][1]}"
                          f"  idx {r['fps_index']/1e6:7.2f}M  blit {r['fps_blit_triton']/1e6:7.2f}M"
                          f"  (torch-blit {r['fps_blit_torch']/1e6:6.2f}M)"
                          f"  e2e {r['fps_end2end']/1e6:7.2f}M  graph {r['fps_cudagraph']/1e6:7.2f}M"
                          f"  {r['e2e_GBps']:6.1f} GB/s", flush=True)
                    torch.cuda.empty_cache()

    if args.out:
        with open(args.out, "w") as fh:
            json.dump(results, fh, indent=2)


if __name__ == "__main__":
    main()
