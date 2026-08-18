"""The recovered smoothscale rule as a batched GPU gather, checked on REAL CarRacing surfaces.

probe_smoothscale4 recovered the rule on synthetic images.  This checks it where it has to hold --
the actual 1000x800 surface CarRacing draws -- and measures what it costs, because blocker 1 of
docs/research/carracing.md claimed reproducing it was the thing that removes the speedup.

The rule (unit 2^14, measured):

    space = ((1 << 14) * n_src) // n_dst          # > 2^14 for a shrink
    recip = (1 << 30) // space
    acc, counter = 0, space
    for each source sample v:
        if counter > 0x4000:  acc += v;  counter -= 0x4000
        else:
            frac = 0x4000 - counter
            emit ((acc + ((v * counter) >> 14)) * recip) >> 16
            acc = (v * frac) >> 14
            counter = space - frac

Every output sample is therefore a fixed set of <=12 source samples with fixed 14-bit weights, so on
the GPU it is one gather per axis, not a filter.
"""
from __future__ import annotations

import os
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
import numpy as np
import pygame
import torch

pygame.init()
BITS = 14
U = 1 << BITS


def plan(n: int, m: int):
    """(taps, weights, recip): (m,K) int64 index/weight tables, zero-padded."""
    space = (U * n) // m
    recip = (1 << (16 + BITS)) // space
    rows, cur, counter = [], [], space
    for x in range(n):
        if counter > U:
            cur.append((x, U))
            counter -= U
        else:
            frac = U - counter
            rows.append(cur + [(x, counter)])
            cur = [(x, frac)] if frac else []
            counter = space - frac
        if len(rows) == m:
            break
    K = max(len(r) for r in rows)
    idx = np.zeros((m, K), np.int64)
    wgt = np.zeros((m, K), np.int64)
    for j, r in enumerate(rows):
        for k, (i, w) in enumerate(r):
            idx[j, k], wgt[j, k] = i, w
    return idx, wgt, recip, K


class Shrink:
    """One axis of the shrink as a gather. Contributions are ``(v*w)>>14``, or ``v`` when w == 2^14
    -- the two are NOT the same because of the truncation, which is why the weight 2^14 is kept in
    the table and branched on rather than folded into the multiply."""

    def __init__(self, n, m, device):
        idx, wgt, recip, K = plan(n, m)
        self.idx = torch.as_tensor(idx, device=device)
        self.w = torch.as_tensor(wgt, device=device)
        self.full = self.w == U
        self.recip = int(recip)
        self.m, self.K = m, K

    def __call__(self, a: torch.Tensor, axis: int) -> torch.Tensor:
        a = a.movedim(axis, 0).to(torch.int64)
        g = a[self.idx.reshape(-1)].reshape(self.m, self.K, *a.shape[1:])
        w = self.w.reshape(self.m, self.K, *([1] * (a.dim() - 1)))
        f = self.full.reshape(self.m, self.K, *([1] * (a.dim() - 1)))
        acc = torch.where(f, g, (g * w) >> BITS).sum(1)
        return (((acc & 0xFFFF) * self.recip) >> 16).clamp(0, 255).to(torch.uint8).movedim(0, axis)


def real_surfaces(n=16):
    import gymnasium
    env = gymnasium.make("CarRacing-v3", render_mode="state_pixels")
    env.reset(seed=0)
    u = env.unwrapped
    rng = np.random.default_rng(0)
    out = []
    for _ in range(n):
        for _ in range(8):
            env.step(np.array([float(rng.uniform(-0.5, 0.5)), 0.5, 0.0], np.float32))
        u._render("state_pixels")                      # populates u.surf at 1000x800, post-blit
        s = u.surf
        out.append(np.transpose(np.array(pygame.surfarray.pixels3d(s)), (1, 0, 2)).copy())
    env.close()
    return np.stack(out)


def main():
    dev = "cuda" if torch.cuda.is_available() else "cpu"
    surfs = real_surfaces()
    print(f"=== recovered smoothscale on {surfs.shape[0]} REAL CarRacing 1000x800 surfaces ===")
    ref96, ref600 = [], []
    for a in surfs:
        s = pygame.Surface((1000, 800))
        pygame.surfarray.blit_array(s, np.transpose(a, (1, 0, 2)).copy())
        ref96.append(np.transpose(np.array(pygame.surfarray.pixels3d(
            pygame.transform.smoothscale(s, (96, 96)))), (1, 0, 2)).copy())
        ref600.append(np.transpose(np.array(pygame.surfarray.pixels3d(
            pygame.transform.smoothscale(s, (600, 400)))), (1, 0, 2)).copy())
    ref96, ref600 = np.stack(ref96), np.stack(ref600)

    t = torch.as_tensor(surfs, device=dev)                                  # (B,800,1000,3)
    for (w, h, ref) in ((96, 96, ref96), (600, 400, ref600)):
        sx, sy = Shrink(1000, w, dev), Shrink(800, h, dev)
        got = sy(sx(t, 2), 1).cpu().numpy()
        d = np.abs(got.astype(np.int32) - ref.astype(np.int32))
        print(f"  -> {w}x{h}: exact {100*(d == 0).mean():.4f}%  max {d.max()}  "
              f"({d.size} pixels compared)")

    if dev == "cuda":
        print("\n=== cost of the resample alone, 1000x800 -> 96x96 ===")
        for B in (16, 64, 256):
            x = t[:1].expand(B, -1, -1, -1).contiguous()
            sx, sy = Shrink(1000, 96, dev), Shrink(800, 96, dev)
            for _ in range(3):
                sy(sx(x, 2), 1)
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            for _ in range(10):
                sy(sx(x, 2), 1)
            torch.cuda.synchronize()
            dt = (time.perf_counter() - t0) / 10
            print(f"  B={B:4d}: {1000*dt:7.3f} ms  ->  {B/dt:10.0f} resamples/s")


if __name__ == "__main__":
    main()
