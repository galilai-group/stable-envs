"""Why do two of our own benchmarks disagree by 3.6x on DMCBallInCupCatch at 84px, batch 4096?

`examples/results/decode/ALL.json` says 978 f/s; `scratch/census/bottom_sweep.py` says 271-274 f/s for
what looks like the identical configuration. The two harnesses construct the env differently in exactly
one respect:

    bench_gpu_vs_cpu.gpu_throughput:  se.make(name, batch=B, size=s, supersample=ss, device="cuda")
    bottom_sweep:                     se.make(name, batch=B, size=s,                 device="cuda")

and `--ss` defaults to 1 in both benchmark scripts. `_DMCModel.__init__` resolves `supersample=None` to
`self.default_supersample`, which `DMControlSuite` sets to 2. So the benchmark is not measuring the env
the registry hands a user; it is measuring the same env at a quarter of the rays.

This asserts that rather than assuming it: same process, same states, same warmup, same timing, three
constructions, plus the resolved `env.ss` printed from the object itself. It also lists every registered
class whose `default_supersample` is not 1, i.e. every published row that carries the same substitution.

    python scratch/audit/ss_discrepancy.py
"""
from __future__ import annotations

import statistics
import time

import numpy as np
import torch

import stable_envs as se

NAMES = ["DMCBallInCupCatch", "DMCFingerSpin", "DMCCartpoleBalance", "DMCCheetahRun"]
BATCH = 4096
SIZE = 84


def timed(name, batch, size, ss, pool, reps=5):
    """frames/s, timed exactly the way both harnesses time it (warm up, synchronize both sides)."""
    kw = {} if ss is None else {"supersample": ss}
    env = se.make(name, batch=batch, size=size, device="cuda", **kw)
    st = torch.as_tensor(pool[:batch], device="cuda")
    f = env.render(st[:1])
    buf = torch.empty((batch, f.shape[1], f.shape[2], 3), dtype=torch.uint8, device="cuda")
    for _ in range(3):
        env.render(st, out=buf)
    torch.cuda.synchronize()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter()
        env.render(st, out=buf)
        torch.cuda.synchronize()
        ts.append(time.perf_counter() - t0)
    got = {"ss_arg": ss, "env.ss": int(env.ss), "hw": tuple(f.shape[1:3]),
           "fps": batch / statistics.median(ts)}
    del env, st, buf, f
    torch.cuda.empty_cache()
    return got


if __name__ == "__main__":
    print("classes whose default_supersample is not 1 (every published row measured at --ss 1 "
          "carries the substitution):")
    for n, cls in sorted(se.REGISTRY.items()):
        d = getattr(cls, "default_supersample", 1)
        if int(d) != 1:
            print(f"    {n:28s} default_supersample={d}")
    print()

    for name in NAMES:
        if name not in se.REGISTRY:
            print(f"{name}: not registered")
            continue
        probe = se.make(name, batch=1, size=SIZE, device="cuda")
        print(f"{name}: se.make(..., size={SIZE}) with no supersample -> env.ss={probe.ss}, "
              f"default_supersample={getattr(type(probe), 'default_supersample', 1)}")
        pool = probe.sample_states(BATCH)
        del probe
        torch.cuda.empty_cache()
        for ss in (None, 1, 2):
            try:
                r = timed(name, BATCH, SIZE, ss, pool)
                print(f"    supersample={str(ss):4s} -> env.ss={r['env.ss']}  frame {r['hw']}  "
                      f"{r['fps']:,.0f} f/s", flush=True)
            except Exception as e:                                       # noqa: BLE001
                print(f"    supersample={ss}: {type(e).__name__}: {e}", flush=True)
                torch.cuda.empty_cache()
        del pool
