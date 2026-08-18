"""Throughput: ours across a batch sweep, against the reference's fastest honest form.

Two rules this follows, because a bare "Nx" is not a claim:

  * the reference denominator includes BOTH halves of what actually produces a frame -- PGX's SVG
    string generation AND resvg's raster. Quoting only one of them would be timing a code path that
    does not yield pixels. Both are reported separately as well, so it is visible which dominates.
  * the CROSSOVER batch is reported -- the batch below which our renderer is SLOWER than the
    reference. A ratio without the batch it was measured at is not interpretable.

Neither PGX's ``to_svg`` nor resvg has a batch parameter to sweep: ``to_svg`` walks one state and
resvg rasterizes one document. So the reference's peak is its warm single-frame rate, measured after
a warm-up, and that is stated rather than implied.
"""
from __future__ import annotations

import sys
import time

import numpy as np
import torch

sys.path.insert(0, "/mnt/vast/home/randall.balestriero/code/stable-envs")

import stable_envs as se
from stable_envs.envs import pgx as P

BATCHES = (1, 8, 64, 256, 1024, 4096)


def time_it(fn, n, warm=3):
    for _ in range(warm):
        fn()
    torch.cuda.synchronize()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    torch.cuda.synchronize()
    return (time.perf_counter() - t0) / n


def reference_rate(cls, states, size):
    """Frames per second for (PGX document) + (resvg raster), and each half on its own."""
    import jax
    import jax.numpy as jnp
    game = cls.GAME
    base = P._pgx.make(game)
    st0 = jax.jit(base.init)(jax.random.PRNGKey(0))
    b = P._bundle(game)
    scale = size / float(b.native)
    rows = [np.asarray(r, np.int32) for r in states]

    def make_state(row):
        x = st0._x._replace(board=jnp.asarray(row))
        if hasattr(x, "turn"):
            x = x._replace(turn=jnp.int32(0))
        return st0.replace(_x=x)

    sts = [make_state(r) for r in rows]
    for s in sts[:2]:                                   # warm jax + svgwrite + resvg
        P._raster(s.to_svg(scale=scale), size, size)

    t0 = time.perf_counter()
    svgs = [s.to_svg(scale=scale) for s in sts]
    t_svg = (time.perf_counter() - t0) / len(sts)

    t0 = time.perf_counter()
    for sv in svgs:
        P._raster(sv, size, size)
    t_ras = (time.perf_counter() - t0) / len(svgs)
    return t_svg, t_ras


def run(names):
    for nm in names:
        cls = getattr(P, nm)
        size = cls.default_size
        env1 = cls(batch=1, device="cuda")
        st_pool = env1.sample_states(64, rng=np.random.default_rng(0))
        t_svg, t_ras = reference_rate(cls, st_pool[:32], size)
        ref_fps = 1.0 / (t_svg + t_ras)
        print(f"=== {nm}  size={size}  cells={env1.state_dim} ===")
        print(f"  reference: to_svg {1000*t_svg:7.3f} ms/frame + resvg {1000*t_ras:7.3f} ms/frame "
              f"= {ref_fps:8.1f} f/s   (no batch parameter exists on either half)")
        cross = None
        for B in BATCHES:
            env = cls(batch=B, device="cuda")
            st = env.sample_states(B, rng=np.random.default_rng(1))
            env.render(st)                                        # build plate + alpha maps
            dt = time_it(lambda: env.render(st), n=10)
            fps = B / dt
            ratio = fps / ref_fps
            if cross is None and ratio >= 1.0:
                cross = B
            print(f"    B={B:5d}  {1000*dt:8.3f} ms  {fps:10.1f} f/s   {ratio:8.2f}x")
            del env
            torch.cuda.empty_cache()
        print(f"  crossover: ours overtakes the reference at B={cross}\n")


if __name__ == "__main__":
    print(f"torch {torch.__version__}  device={torch.cuda.get_device_name(0)}")
    run(["PgxTicTacToe", "PgxConnectFour", "PgxOthello", "PgxGo9x9", "PgxGo19x19"])
