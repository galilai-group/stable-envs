"""Is ``scene()`` compute or launch overhead, and which calls issue the ops?

The rasterizer docstring records ``scene()`` at ~2.0 ms flat, measured on PushT's 9 primitives. The
highway family builds 157-436, and the baseline profile puts ``scene()`` at 100-315 ms -- i.e. it is
the dominant cost for four of the six envs, not a 2 ms constant. Two questions decide what to do about
it: does it scale with the batch (compute) or not (python + launch), and which python call sites issue
the ops.
"""
from __future__ import annotations

import cProfile
import io
import os
import pstats
import sys
import time

os.environ.setdefault("SDL_VIDEODRIVER", "dummy")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

import numpy as np
import torch

import stable_envs as se


def timeit(fn, reps=5, warm=2):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        fn()
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    return float(np.median(ts))


class OpCounter(torch.overrides.TorchFunctionMode):
    def __init__(self):
        self.n = 0

    def __torch_function__(self, func, types, args=(), kwargs=None):
        self.n += 1
        return func(*args, **(kwargs or {}))


def run(name, size=256):
    env = se.make(name, batch=1, size=size, device="cuda")
    pool = env.sample_states(2048)
    scr = env.screen(size)
    print(f"\n=== {name} {scr.H}x{scr.W} ===", flush=True)
    for B in (16, 256, 1024):
        st = torch.as_tensor(pool[np.arange(B) % len(pool)], device="cuda")
        sty = env.resolve_style(None, None, batch=B)
        t = timeit(lambda: env._scene(st, sty, scr))
        sc = env._scene(st, sty, scr)
        with OpCounter() as c:
            env._scene(st, sty, scr)
        print(f"  B={B:5d} scene {1000*t:8.2f} ms  N={sc.n_prims}  torch calls {c.n}  "
              f"{1e6*t/max(c.n,1):6.1f} us/call", flush=True)
        del sc
    B = 256
    st = torch.as_tensor(pool[np.arange(B) % len(pool)], device="cuda")
    sty = env.resolve_style(None, None, batch=B)
    pr = cProfile.Profile()
    pr.enable()
    for _ in range(3):
        env._scene(st, sty, scr)
    pr.disable()
    s = io.StringIO()
    pstats.Stats(pr, stream=s).sort_stats("tottime").print_stats(18)
    print("\n".join(s.getvalue().splitlines()[4:30]), flush=True)


if __name__ == "__main__":
    print(torch.cuda.get_device_name(0), flush=True)
    for n in (sys.argv[1:] or ["Roundabout", "Highway"]):
        run(n)
