"""Fused CarRacing against pygame itself, and the per-primitive path against pygame, on one pool.

The two disagree on one surface pixel in 19.2 M. Which of them is right is not a question either can
answer, so both are measured against the official renderer on the same states -- including the state
whose surface differs.
"""
from __future__ import annotations

import os
import sys

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import stable_envs as se
from stable_envs.envs.carracing import CarRacing


def score(name, got, want):
    d = np.abs(got.astype(np.int32) - want.astype(np.int32))
    npx = int((d > 0).any(-1).sum())
    tot = d.shape[0] * d.shape[1] * d.shape[2]
    print(f"  {name:22s} exact {100*(1-npx/tot):.6f}%  bad {npx}/{tot}  max {int(d.max())}",
          flush=True)
    return npx


def main(n=48, seed=11):
    env = se.make("CarRacing", batch=n, device="cuda")
    states = env.sample_states(n, rng=np.random.default_rng(seed))
    ref = CarRacing.reference_render(CarRacing.reference_env(), states)
    fused = env.render(torch.as_tensor(states, device="cuda")).cpu().numpy()
    a = score("fused vs pygame", fused, ref)

    # the same states through the per-primitive path, rebuilt from the pieces that still exist
    from scratch.fastraster.diag_carracing import capture, torch_surface
    s = torch.as_tensor(states, device="cuda").to(torch.float64)
    sty = env.resolve_style(None, None, batch=n)
    got = capture(env, s, sty, n)
    buf, *_ = torch_surface(env, got, s, n)
    old = env.sy(env.sx(buf, 2), 1).cpu().numpy()
    b = score("per-primitive vs pygame", old, ref)
    print(f"  {'fused vs per-primitive':22s} bad "
          f"{int((np.abs(fused.astype(np.int32) - old.astype(np.int32)) > 0).any(-1).sum())}",
          flush=True)
    print(f"=== fused {'WINS' if a < b else 'TIES' if a == b else 'LOSES'} ===", flush=True)


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), flush=True)
    main()
