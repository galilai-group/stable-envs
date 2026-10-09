# Benchmarks

Correctness and performance are separate. This page is the latter: how fast a **state becomes a
uint8 frame**, compared with decoding a stored JPEG.

## What is measured

A paired job on one card: take a state from the env's own reset distribution, produce a `(H, W, 3)`
uint8 frame. GPU path is `env.render(state)` at the class's native `default_supersample`. CPU path
is the env's official reference bridge where one exists. Both sides do the same job.

Host-bound rows travel badly. A render whose process CPU time is near 0 per second of wall clock is
one kernel on a device the job owns. Ray-cast families spend 100–206% of the timed loop on CPU, so
those numbers are a property of the node's processor under its load. Every GPU row in the generated
files carries `cpu%` (or `cpu_frac`).

OSS-facing harnesses: `python -m benchmarks.render` and `python -m benchmarks.pipeline`. The
numbers below were taken with `examples/bench_pusht_source.py`, `examples/bench_gpu_vs_cpu.py` and
`examples/bench_vs_decode.py`.

## PushT

The released PushT draws with the source's own pygame code on CPU; `device="cuda"` only places the
output. Its frames equal the pinned stable-worldmodel source, so the comparison that matters is cost
per frame against that source on the same core. `examples/bench_pusht_source.py` renders States saved
from four episodes both ways on one thread (224 px):

| CPU | batch | first `env.render` ms/frame | warm `env.render` ms/frame | source ms/frame | warm vs source |
|---|---:|---:|---:|---:|---:|
| Apple M5 Pro | 64 | 1.08 | 0.55 | 0.66 | 1.22× |
| Apple M5 Pro | 1,024 | 0.88 | 0.59 | 0.67 | 1.13× |
| Xeon Platinum 8268 | 64 | 4.98 | 3.69 | 3.04 | 0.83× |
| Xeon Platinum 8268 | 1,024 | 5.19 | 4.02 | 3.28 | 0.82× |

The first render of a batch builds one pymunk scene per lane; later renders reuse it. Batch 1 also
pays a fixed per-call cost for State validation. The pygame drawing is the same on both sides; the
difference is Python work per frame, which weighs more on the slower Xeon core.

`contrib/pusht_torch/` holds an unreleased batched torch renderer whose frames equal `env.render` on
Linux x86-64. On an RTX A5000 node it takes 0.32 ms/frame at batch 1,024, 5× `env.render` on that
node's CPU. It reproduces LGPL pygame code and is not part of the package; see its README.

## State → render vs stored JPEG decode → H2D

The stored-frame bar is one CPU core decoding *this env's own frame*, JPEG q90, already in memory —
the case most favourable to a corpus. Across a registry sweep, reading a warm local JPEG and decoding
it runs at 0.45–1.95× the speed of decoding the same bytes already in memory (median 0.96×): a
stored-frame pipeline is decode-bound, not IO-bound.

Some families named below (MiniGrid, Griddly, Sokoban, Overcooked, CarRacing, D4RL PointMaze,
HumanoidBench, Meta-World, RoboCasa, the nvdiffrast Gymnasium MuJoCo locomotion cells) are renderer-only
ports kept in the repository's `contrib/` directory, not part of the released package; their numbers are
kept as measured.

Tile-atlas families (MiniGrid, Griddly, Sokoban, Overcooked) gather from a precomputed atlas and
clear hundreds of thousands to millions of frames a second, replacing thousands of JPEG-decode cores.
Ray-cast 3-D families are slower and host-bound;
visual-tier MuJoCo / Meta-World / Fetch sit near 1–7× a stored JPEG/core.

The paper's per-family table uses eight worker processes batch-decoding JPEG-in-Lance, not one core.
See `examples/bench_lance.py`.

## Caveats / losers

* **CarRacing** is the family where the official renderer is genuinely close (historical sweep: 133 f/s
  official vs 179 ours at 96 px, before the fused-kernel remeasure). It is not the headline win.
* **Ray-cast / CPU3D families** (OGBench mazes, DM Control, D4RL PointMaze, navigation) are
  host-bound. `DMCBallInCupCatch` measured 3,664 f/s idle and 271–978 contended on the same card.
* **Visual-tier 3-D** (Fetch, HumanoidBench, Meta-World, RoboCasa, nvdiffrast locomotion) replace few
  JPEG-decode cores. Use them for live training data, not as a substitute for stored official pixels.
* The official-renderer side is single-process; those envs offer no batched entry point.
* `--ss` must be `native` (the class's `default_supersample`). Forcing `1` on the 44 classes that
  declare `2` measures a different, cheaper picture than `snv.make(name)` returns.

## How to reproduce

```bash
python -m benchmarks.render --help
python -m benchmarks.pipeline --help
python examples/bench_pusht_source.py --help
python examples/bench_gpu_vs_cpu.py --out results --sizes native
python examples/bench_vs_decode.py --help
```

`bench_gpu_vs_cpu.py` walks `snv.list_envs()`, writes after every env, and merges per env so a sweep
can split across allocations. Envs without a reference renderer get GPU columns and no speedup
column.
