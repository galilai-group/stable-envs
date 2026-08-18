"""Throughput through the SHIPPED classes: (ratio, batch) with its crossover, never a bare peak.

For each env: our ``render`` over a batch sweep, the OFFICIAL renderer on one pinned core, and JPEG
decode of the same frames on one core — so the number that gets quoted is "x at batch B, ahead from
batch C", plus the core count a stored-frame pipeline would need to keep up.

    python scratch/griddly_pack/bench.py griddly
    python scratch/griddly_pack/bench.py overcooked
"""
import io
import json
import os
import sys
import time

import numpy as np
from PIL import Image

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
import torch                                                            # noqa: E402
import stable_envs as se                                                # noqa: E402
from stable_envs.testing import conformance as C                        # noqa: E402

BATCHES = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048, 4096, 8192]
mod = sys.argv[1] if len(sys.argv) > 1 else "griddly"
OUT = os.path.dirname(os.path.abspath(__file__))


def decode_fps(img, quality=90):
    buf = io.BytesIO()
    Image.fromarray(img).save(buf, format="JPEG", quality=quality)
    b = buf.getvalue()
    n = max(200, int(4e7 / img.size * 3))
    for _ in range(20):
        np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    t0 = time.perf_counter()
    for _ in range(n):
        np.asarray(Image.open(io.BytesIO(b)).convert("RGB"))
    return n / (time.perf_counter() - t0), len(b)


def official_fps(cls, size, states):
    """Two numbers, because they answer different questions.

    ``render`` is the official RENDERER alone — the call a data-generation loop makes after every env
    step, with the scene already on the board. ``state_to_frame`` additionally puts a stored state back,
    which is what regenerating a dataset from states through the reference would cost. Conflating them
    would credit this repo with the reference's state-restore cost, so they are reported apart.
    """
    ref = C.reference_env(cls, size)
    cls.reference_render(ref, states[:2])
    n = 60
    t0 = time.perf_counter()
    cls.reference_render(ref, np.repeat(states[:1], n, 0))
    s2f = n / (time.perf_counter() - t0)

    if mod == "griddly":
        from stable_envs.envs.griddly import _frame
        for _ in range(20):
            _frame(ref)
        n = 400
        t0 = time.perf_counter()
        for _ in range(n):
            _frame(ref)
        return n / (time.perf_counter() - t0), s2f
    if mod == "overcooked":
        from jaxmarl.environments.overcooked.common import OBJECT_TO_INDEX
        import stable_envs.envs.overcooked as oc
        viz, tile = ref
        codes = np.round(states[0]).astype(np.int64) % oc.N_CODE
        grid = np.zeros((cls.grid_h, cls.grid_w, 3), np.uint8)
        inv = np.array([OBJECT_TO_INDEX["empty"]] * 2, np.int64)
        for k, code in enumerate(codes):
            t, c, e, h = oc.ALPHABET[int(code)]
            grid[k // cls.grid_w, k % cls.grid_w] = (t, c, e)
            if t == OBJECT_TO_INDEX["agent"]:
                inv[0 if c == 0 else 1] = h
        kw = dict(tile_size=tile, highlight_mask=None,
                  agent_dir_idx=np.array([0, 1]), agent_inv=inv)
        for _ in range(5):
            viz._render_grid(grid, **kw)                  # warm the class-level tile cache
        n = 200
        t0 = time.perf_counter()
        for _ in range(n):
            viz._render_grid(grid, **kw)
        return n / (time.perf_counter() - t0), s2f
    return s2f, s2f


res = {}
for name, cls in sorted(se.REGISTRY.items()):
    if not cls.__module__.endswith("." + mod):
        continue
    env = se.make(name, batch=8, device="cuda")
    states = env.sample_states(64)
    off, s2f = official_fps(cls, env.size, states)
    fr0 = env.render(states[:1]).cpu().numpy()[0]
    dec, jb = decode_fps(fr0)
    H, W = fr0.shape[:2]
    rows = []
    base = torch.as_tensor(states, device="cuda", dtype=torch.float32)
    for B in BATCHES:
        st = base[torch.randint(0, base.shape[0], (B,), device="cuda")].contiguous()
        try:
            out = torch.empty(B, H, W, 3, dtype=torch.uint8, device="cuda")
            for _ in range(5):
                env.render(st, out=out)
            torch.cuda.synchronize()
            it = max(5, min(300, int(2e9 / (out.numel() + 1))))
            t0 = time.perf_counter()
            for _ in range(it):
                env.render(st, out=out)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / it
            rows.append({"batch": B, "fps": B / dt, "gbps": out.numel() * 2 / dt / 1e9})
            del out
        except torch.cuda.OutOfMemoryError:
            break
        torch.cuda.empty_cache()
    best = max(rows, key=lambda r: r["fps"])
    res[name] = {
        "frame": [H, W], "state_dim": env.state_dim, "size": env.size,
        "official_render_fps": off, "official_state_to_frame_fps": s2f,
        "decode_fps_per_core": dec, "jpeg_bytes": jb,
        "best_fps": best["fps"], "best_batch": best["batch"], "best_gbps": best["gbps"],
        "fps_b1": rows[0]["fps"],
        "ratio_vs_official": best["fps"] / off,
        "crossover_batch_vs_official": next((r["batch"] for r in rows if r["fps"] > off), None),
        "crossover_batch_vs_decode": next((r["batch"] for r in rows if r["fps"] > dec), None),
        "cores_of_decode_replaced": best["fps"] / dec,
        "official_cores_of_decode": off / dec,
        "compression_frame_over_state": (H * W * 3) / (env.state_dim * 4),
        "compression_packed": (H * W * 3) / env.state_dim,
        "rows": rows,
    }
    r = res[name]
    print(f"{name:26s} frame={H}x{W} state={env.state_dim} official={off:8.1f} f/s (s2f {s2f:6.1f}) "
          f"decode={dec:8.1f} f/s/core | ours B=1 {r['fps_b1']:10.0f}  best {r['best_fps']:12.0f} "
          f"@B={r['best_batch']} ({r['best_gbps']:.0f} GB/s) = x{r['ratio_vs_official']:.0f}, "
          f"crossover B={r['crossover_batch_vs_official']} | "
          f"{r['cores_of_decode_replaced']:.0f} cores of decode, official is "
          f"{r['official_cores_of_decode']:.2f}", flush=True)

with open(os.path.join(OUT, f"bench_{mod}.json"), "w") as f:
    json.dump(res, f, indent=2, default=str)
print("WROTE", os.path.join(OUT, f"bench_{mod}.json"), flush=True)
sys.stdout.flush()
