"""Pixel-diff a stable-envs highway class against the official renderer, and localise disagreements.

Reports interior-pixel and all-pixel agreement and, when they are not perfect, where the wrong pixels
are (which reference colour they should have been) so a miss is attributable to a specific primitive
rather than to "rendering".
"""
import os
os.environ.setdefault("SDL_VIDEODRIVER", "offscreen")
import sys
import time
import numpy as np
import torch

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")
import stable_envs as se
from stable_envs.testing import conformance as C

NAME = sys.argv[1] if len(sys.argv) > 1 else "Highway"
N = int(sys.argv[2]) if len(sys.argv) > 2 else 32
dev = "cuda" if torch.cuda.is_available() else "cpu"

print("import errors:", se.envs.import_errors())
env = se.make(NAME, batch=N, device=dev, backend="auto" if dev == "cuda" else "torch")
cls = type(env)
print(f"{NAME}: state_dim={env.state_dim} H={env.H} W={env.W} "
      f"stripes_count={env.stripes_count} line_w={env.line_w} device={dev}")

states = env.sample_states(N)
if "--zero-heading" in sys.argv:
    states[:, 2::4] = 0.0
if "--no-crash" in sys.argv:
    states[:, 3::4] = 0.0
ref = cls.reference_env(size=env.size)
t0 = time.perf_counter()
official = cls.reference_render(ref, states)
t_ref = time.perf_counter() - t0
sc = env.scene(torch.as_tensor(states, device=dev))
print(f"primitives: {sc.n_prims}")
ours = env.render(states).cpu().numpy()
print(f"shapes ours={ours.shape} official={official.shape}   reference {N/t_ref:.1f} frames/s")

r = C.pixel_agreement(ours, official)
for k, v in r.items():
    print(f"  {k}: {v}")

d = np.abs(ours.astype(np.int16) - official.astype(np.int16)).max(-1)
m = C.interior_mask(official)
bad = np.zeros_like(d, bool)
bad[:, :-1, :-1] = (d[:, :-1, :-1] != 0) & m
print(f"\nwrong interior pixels: {bad.sum()} of {m.sum()}")
if bad.sum():
    fr = np.unique(np.nonzero(bad)[0])
    print(f"  frames affected: {len(fr)} -> {fr[:10]}")
    for f in fr[:4]:
        ys, xs = np.nonzero(bad[f])
        print(f"  frame {f}: {len(ys)} px, rows {ys.min()}..{ys.max()} cols {xs.min()}..{xs.max()}")
        for y, x in list(zip(ys, xs))[:12]:
            print(f"      ({y:3d},{x:3d}) ours={tuple(ours[f,y,x])} official={tuple(official[f,y,x])}")

alld = d != 0
print(f"\nwrong pixels overall: {alld.sum()} ({100*alld.mean():.3f}%)")
if alld.sum():
    cols, cnt = np.unique(official[alld], axis=0, return_counts=True)
    o = np.argsort(-cnt)
    print("  official colour of the disagreeing pixels:")
    for i in o[:8]:
        print(f"    {tuple(int(c) for c in cols[i])}: {cnt[i]}")
    cols, cnt = np.unique(ours[alld], axis=0, return_counts=True)
    o = np.argsort(-cnt)
    print("  our colour there:")
    for i in o[:8]:
        print(f"    {tuple(int(c) for c in cols[i])}: {cnt[i]}")
