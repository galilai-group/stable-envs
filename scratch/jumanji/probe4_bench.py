"""Probe 4: throughput as a (ratio, batch) PAIR, with the crossover batch.

Three rules, and they are why this is a separate probe rather than a line in a parity script:

1. The denominator is the FASTEST HONEST form of the official renderer. A matplotlib viewer has a slow
   form (a new Figure per frame) and a fast one (one Figure reused, artists redrawn), and quoting the
   ratio against the slow form is a strawman. Both are measured; the faster is the denominator.
2. The CROSSOVER batch is reported -- the batch below which our renderer is slower. "No crossover,
   faster at every batch measured including B=1" is a real and reportable answer, not an absence.
3. Our side goes through ``env.render()`` as a caller gets it, never a prototype path.

The atlas build is EXCLUDED from the render timings and reported separately: it is a one-off per
(id, size) per machine, memoised to disk, and mixing it into a per-frame number would describe neither
the cold cost nor the warm one.
"""
from __future__ import annotations

import argparse
import os
import time
import warnings

os.environ.setdefault("MPLBACKEND", "Agg")
os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

warnings.filterwarnings("ignore")


def timeit(fn, n: int, warmup: int = 2) -> float:
    for _ in range(warmup):
        fn()
    t0 = time.perf_counter()
    for _ in range(n):
        fn()
    return (time.perf_counter() - t0) / n


def official_forms(spec, size: int, states, n: int = 8) -> dict:
    """Reused-figure and fresh-figure forms of the official viewer, in frames/s."""
    import matplotlib.pyplot as plt

    v = spec.viewer(size)
    grids = [np.rint(s).astype(np.int32) % spec.n_values for s in states[:n]]

    reused = timeit(lambda: [spec.render(v, g) for g in grids], 3) / n

    def fresh():
        plt.close(v._name)
        for g in grids:
            spec.render(v, g)

    per_fresh = timeit(fresh, 2) / n
    return {"reused_fps": 1.0 / reused, "fresh_fps": 1.0 / per_fresh,
            "denominator_fps": max(1.0 / reused, 1.0 / per_fresh)}


def main() -> None:
    import torch

    import stable_envs as se

    ap = argparse.ArgumentParser()
    ap.add_argument("--batches", nargs="*", type=int, default=[1, 8, 64, 256, 4096])
    ap.add_argument("--size", type=int, default=200)
    ap.add_argument("--envs", nargs="*", default=None)
    args = ap.parse_args()

    names = [n for n in sorted(se.REGISTRY)
             if n.startswith("Jumanji") and (args.envs is None or n in args.envs)]
    print(f"interpreter: {os.environ.get('VIRTUAL_ENV', 'conda')}")
    print(f"len(stable_envs.REGISTRY) = {len(se.REGISTRY)}  (Jumanji classes: {len(names)})")
    print(f"env-module import errors: {se.envs.import_errors()}\n")

    for name in names:
        cls = se.REGISTRY[name]
        spec = cls.SPEC
        e1 = se.make(name, batch=1, size=args.size, device="cuda")
        states = e1.sample_states(max(args.batches), rng=np.random.default_rng(0))
        off = official_forms(spec, args.size, states)
        print(f"\n=== {name}  ({spec.env_id}, {args.size} px)")
        print(f"  official reused-figure {off['reused_fps']:8.2f} f/s | fresh-figure "
              f"{off['fresh_fps']:8.2f} f/s -> denominator {off['denominator_fps']:8.2f} f/s")
        cross = None
        for B in args.batches:
            env = se.make(name, batch=B, size=args.size, device="cuda")
            st = torch.as_tensor(states[:B], device="cuda")
            out = torch.empty((B, args.size, args.size, 3), dtype=torch.uint8, device="cuda")

            def one():
                env.render(st, out=out)
                torch.cuda.synchronize()

            dt = timeit(one, 30 if B <= 256 else 5)
            fps = B / dt
            ratio = fps / off["denominator_fps"]
            if cross is None and ratio >= 1.0:
                cross = B
            print(f"    B={B:6d}  {fps:14.1f} f/s   ratio {ratio:10.1f}x")
        print(f"    crossover: {'none -- faster at every batch measured' if cross == args.batches[0] else cross}")


if __name__ == "__main__":
    main()
