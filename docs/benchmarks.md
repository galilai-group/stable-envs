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
representative numbers below were taken with `examples/bench_gpu_vs_cpu.py` and
`examples/bench_vs_decode.py`.

## Representative CPU / GPU numbers

**PushT, 96×96, NVIDIA H200** (native `default_supersample=1`). Official CPU renderer: **491 frames/s**
(`set_state` + render). GPU:

| batch | frames/s | speedup vs CPU |
|---|---|---|
| 1 | 558 | 1.14× |
| 1,024 | 380,586 | 775× |
| 16,384 | **2,914,271** | **5,931×** |
| 65,536 | 2,103,552 | 4,281× |

Batch 1 is mostly Python scene construction, not rasterization. The working point this repo quotes is
**16,384**, not the largest batch: at 65,536 the per-frame time is *worse* (0.48 µs vs 0.34 µs).

At 224×224 the official renderer is nearly resolution-independent (438 frames/s, always draws at 512
and resizes). Ours pays for the pixels: **839,360 frames/s** at batch 16,384 (1,915×).

Storage ratio before compression: a 96×96×3 frame is 27,648 B; PushT's compact render row is 20 B
(**1,382×**). Regenerating a 96 px frame costs 0.34 µs at the quoted batch.

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
2-D rasterized PushT is ~100× a stored JPEG/core. Ray-cast 3-D families are slower and host-bound;
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
* PushT is nine primitives. Nothing here says how a heavier 2-D scene scales.
* The official-renderer side is single-process; those envs offer no batched entry point.
* `--ss` must be `native` (the class's `default_supersample`). Forcing `1` on the 44 classes that
  declare `2` measures a different, cheaper picture than `snv.make(name)` returns.

## How to reproduce

```bash
python -m benchmarks.render --help
python -m benchmarks.pipeline --help
python examples/bench_gpu_vs_cpu.py --out results --envs PushT --sizes 96,224
python examples/bench_gpu_vs_cpu.py --out results --sizes native
python examples/bench_vs_decode.py --help
```

`bench_gpu_vs_cpu.py` walks `snv.list_envs()`, writes after every env, and merges per env so a sweep
can split across allocations. Envs without a reference renderer get GPU columns and no speedup
column.

Raw PushT H200 output:
[`tests/fixtures/regression/references/bench_gpu_vs_cpu.json`](https://github.com/galilai-group/stable-envs/blob/main/tests/fixtures/regression/references/bench_gpu_vs_cpu.json).
