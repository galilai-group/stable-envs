"""Anatomy of the serial forward-kinematics loop, and whether threads can be spent on it.

``OGBenchMaze._fk`` says in its own docstring that the legged families are FK-bound rather than
render-bound, and the storage numbers agree: pointmaze, whose FK is closed form on the GPU, runs at
42k f/s while antmaze -- SAME maze, same plate, same caster, a different body -- runs at 10.8k. So
the question is not whether the loop costs; it is WHAT in the loop costs, because the three answers
have three different fixes:

  * ``mj_kinematics`` itself      -> threads (one ``MjData`` each), if the bindings drop the GIL
  * numpy fancy-indexing per row  -> a cheaper gather, or a contiguous slice
  * interpreter overhead per row  -> fewer Python operations per row, nothing else

Threads are only admissible if they are BIT-EXACT, so the threaded result is compared to the serial
one by bit pattern, not by tolerance.
"""
from __future__ import annotations

import argparse
import time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch

import stable_envs as se

ap = argparse.ArgumentParser()
ap.add_argument("--envs", default="AntmazeMediumNavigate,HumanoidmazeMediumNavigate,"
                                 "PointmazeMediumNavigate")
ap.add_argument("--batch", type=int, default=1024)
ap.add_argument("--threads", default="1,2,4,8,16,32")
a = ap.parse_args()

import mujoco  # noqa: E402


def t_of(fn, n=5):
    fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


for name in [n for n in a.envs.split(",") if n]:
    if name not in se.REGISTRY:
        print(f"{name}: not registered")
        continue
    env = se.make(name, batch=a.batch, size=256, device="cuda")
    sc = env._scene()
    s = torch.as_tensor(env.sample_states(a.batch, rng=np.random.default_rng(0)),
                        device="cuda", dtype=torch.float32)
    m, d = sc.model, sc.data
    ids = np.asarray(sc.agent_ids)
    B = a.batch
    q = s.detach().cpu().numpy().astype(np.float64)
    P = len(ids)
    print(f"\n=== {name}  nq={m.nq} nbody={m.nbody} ngeom={m.ngeom} agent geoms={P}  B={B}")

    print(f"  {'whole _fk':32s} {t_of(lambda: env._fk(s, sc)) * 1e3:8.2f} ms")

    def kin_only():
        for i in range(B):
            mujoco.mj_kinematics(m, d)
    print(f"  {'mj_kinematics x B only':32s} {t_of(kin_only) * 1e3:8.2f} ms")

    def set_only():
        for i in range(B):
            d.qpos[:] = q[i]
    print(f"  {'qpos assign x B only':32s} {t_of(set_only) * 1e3:8.2f} ms")

    xp = np.empty((B, P, 3))
    xm = np.empty((B, P, 3, 3))

    def gather_only():
        for i in range(B):
            xp[i] = d.geom_xpos[ids]
            xm[i] = d.geom_xmat[ids].reshape(-1, 3, 3)
    print(f"  {'fancy-index gather x B only':32s} {t_of(gather_only) * 1e3:8.2f} ms")

    def to_torch():
        return (torch.tensor(xp, device="cuda", dtype=torch.float32),
                torch.tensor(xm, device="cuda", dtype=torch.float32))
    print(f"  {'host->device transfer':32s} {t_of(to_torch) * 1e3:8.2f} ms")

    if env.LOCO == "point":
        del env
        torch.cuda.empty_cache()
        continue

    # ---- threaded, one MjData per worker -----------------------------------------------------
    ref_p, ref_m = env._fk(s, sc)
    ref_p, ref_m = ref_p.cpu(), ref_m.cpu()
    for T in [int(t) for t in a.threads.split(",")]:
        datas = [mujoco.MjData(m) for _ in range(T)]
        pool = ThreadPoolExecutor(T)
        bounds = [(k * B // T, (k + 1) * B // T) for k in range(T)]

        def worker(arg):
            k, (lo, hi) = arg
            dd = datas[k]
            for i in range(lo, hi):
                dd.qpos[:] = q[i]
                mujoco.mj_kinematics(m, dd)
                xp[i] = dd.geom_xpos[ids]
                xm[i] = dd.geom_xmat[ids].reshape(-1, 3, 3)

        def run():
            list(pool.map(worker, enumerate(bounds)))

        dt = t_of(run, 3)
        got_p = torch.tensor(xp, dtype=torch.float32)
        got_m = torch.tensor(xm, dtype=torch.float32)
        bad = int((got_p.view(torch.int32) != ref_p.view(torch.int32)).sum()) + \
            int((got_m.view(torch.int32) != ref_m.view(torch.int32)).sum())
        print(f"  {'threaded T=' + str(T):32s} {dt * 1e3:8.2f} ms   "
              f"bit-diffs vs serial: {bad}")
        pool.shutdown()
    del env
    torch.cuda.empty_cache()
