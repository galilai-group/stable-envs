# Throughput

The PushT sections on this page were produced by
[`examples/bench_gpu_vs_cpu.py`](https://github.com/galilai-group/stable-envs/blob/main/examples/bench_gpu_vs_cpu.py)
on one NVIDIA H200 with an Intel Xeon Platinum 8558 (16 cores visible to the job), torch 2.11.0+cu130, triton 3.6.0 — and re-measured unchanged on torch 2.13.0+cu130 / triton 3.7.1 (geomean 1.003x over 12 cells, `scratch/census/triton_ab.sbatch`), at PushT's own `default_supersample` of 1. Raw output, including the host load average
around the sweep:
[`examples/results/bench_gpu_vs_cpu.json`](https://github.com/galilai-group/stable-envs/blob/main/examples/results/bench_gpu_vs_cpu.json)
and [`.md`](https://github.com/galilai-group/stable-envs/blob/main/examples/results/bench_gpu_vs_cpu.md).
The registry-wide table further down is a separate sweep and says so where it starts.

`--ss` now defaults to `native` in `examples/bench_gpu_vs_cpu.py`, `examples/bench_vs_decode.py` and
`scripts/bench.py`, so every env renders at its own `default_supersample` instead of at a forced 1. That
is a correction rather than a preference. Forty-four registered classes declare `default_supersample = 2`
— 25 OGBench mazes, 11 DM Control domains, 4 D4RL mazes, 4 MuJoCo locomotion envs — because their
reference is MSAA'd, so `se.make(name)` hands a user ss=2, and ss=2 is the picture every parity number in
this repo is measured on. Every throughput row published for those classes before this fix was taken at
`--ss 1`, which is a different picture, four times cheaper in rays. Measured in one process on an idle
node, `supersample=1` against the class's own value: DMCBallInCupCatch at 84px batch 4096 is 4,272
against 3,683 frames/s (1.16×), AntmazeLargeNavigate at 224px 70,598 against 55,947 (1.26×),
MujocoPusher at 240px 14,938 against 4,475 (3.3×). The ratio is not a constant, because these renders
are host-bound and ss=2 quadruples the per-chunk Python loop, so it grows with how contended the node is.

Both sides do the same job — take a state, produce a uint8 frame — and both are given the same states,
drawn from the env's own reset distribution. The CPU side goes through the env's own reference bridge,
which is the transformation the GPU path performs; where the reference also exposes a per-state setter and
a draw hook, the draw is timed separately so the speedup cannot be blamed on a slow setter.

!!! note "These tables are a transcription; the generated file is the source"

    Everything under this heading is copied from
    [`examples/results/bench_gpu_vs_cpu.md`](https://github.com/galilai-group/stable-envs/blob/main/examples/results/bench_gpu_vs_cpu.md),
    which `examples/bench_gpu_vs_cpu.py` rewrites on every run. When the two disagree the generated
    file is right. It also carries two columns this page does not: the supersample each row resolved
    to, and `cpu%`, the process CPU seconds per wall second of the timed loop, which is what says
    whether a row travels to another node.

## PushT, 96×96 (the native resolution)

The official CPU renderer manages **491 frames/s** (2,035 µs/frame) for `set_state` + render, and 495
frames/s for the draw alone. The setter is 3% of the total: the cost is rasterization, not bookkeeping.

| batch | ms / batch | µs / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 1.792 | 1791.54 | 558 | 0.0 | 0.22 | 100% | 1.14× |
| 64 | 2.250 | 35.15 | 28,449 | 0.8 | 0.22 | 100% | 58× |
| 256 | 2.350 | 9.18 | 108,922 | 3.0 | 0.22 | 100% | 222× |
| 1,024 | 2.691 | 2.63 | 380,586 | 10.5 | 0.25 | 100% | 775× |
| 4,096 | 3.257 | 0.80 | 1,257,509 | 34.8 | 0.34 | 100% | 2,559× |
| 16,384 | 5.622 | 0.34 | **2,914,271** | 80.6 | 0.69 | 100% | **5,931×** |
| 65,536 | 31.155 | 0.48 | 2,103,552 | 58.2 | 2.13 | 100% | 4,281× |

Two rows are worth reading before the headline one.

**Batch 1 is not a rendering measurement.** 558 frames/s against the reference's 491 is the same order of
magnitude, and 1.14× is what a batched renderer called one frame at a time is worth: 91% of that 1.792 ms
is building the scene in Python. The ratio only becomes a statement about rasterization once the launch
is saturated.

**The peak is at 16,384, not at 65,536.** The largest batch is *slower* per frame (0.48 µs against 0.34),
and at 58.2 GB/s of output against 80.6 it is not close to the bandwidth ceiling either, so it is not
simply saturating. It is recorded rather than dropped because a table that stops at its best cell hides
the shape; the working point this repo quotes is 16,384.

## PushT, 224×224

The official renderer is nearly resolution-independent (**438 frames/s**, 2,282 µs/frame) because it
always draws at 512 and resizes afterwards. Ours pays for the pixels it writes.

| batch | ms / batch | µs / frame | frames/s | GB/s | peak GB | cpu% | speedup vs CPU |
|---|---|---|---|---|---|---|---|
| 1 | 1.760 | 1759.67 | 568 | 0.1 | 0.22 | 100% | 1.30× |
| 64 | 2.393 | 37.39 | 26,744 | 4.0 | 0.23 | 100% | 61× |
| 256 | 2.741 | 10.71 | 93,382 | 14.1 | 0.26 | 100% | 213× |
| 1,024 | 3.550 | 3.47 | 288,428 | 43.4 | 0.37 | 100% | 658× |
| 4,096 | 6.741 | 1.65 | 607,585 | 91.5 | 0.84 | 100% | 1,386× |
| 16,384 | 19.520 | 1.19 | **839,360** | 126.3 | 2.71 | 100% | **1,915×** |

## Where the time goes

`render()` is two things: build a `Scene2D` from the states, then rasterize it. Timing them apart changes
what the tables above mean.

96×96:

| batch | `scene()` ms | rasterize ms | `render()` ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 1.615 | 0.054 | 1.776 | 18,536 | 0.5 |
| 256 | 2.122 | 0.162 | 2.345 | 1,581,769 | 43.7 |
| 4,096 | 2.138 | 1.009 | 3.271 | 4,060,052 | 112.3 |
| 16,384 | 2.137 | 3.350 | 5.615 | **4,890,308** | **135.2** |

224×224:

| batch | `scene()` ms | rasterize ms | `render()` ms | rasterize frames/s | rasterize GB/s |
|---|---|---|---|---|---|
| 1 | 1.625 | 0.056 | 1.760 | 17,969 | 2.7 |
| 256 | 2.123 | 0.487 | 2.729 | 526,137 | 79.2 |
| 4,096 | 2.135 | 4.483 | 6.754 | 913,734 | 137.5 |
| 16,384 | 2.122 | 17.232 | 19.503 | **950,807** | **143.1** |

## Storage

| resolution | raw frame | state | ratio |
|---|---|---|---|
| 96×96×3 | 27,648 B | 20 B (5 × float32) | **1,382×** |
| 224×224×3 | 150,528 B | 20 B | **7,526×** |

Before any compression on either side. The gallery's 120-frame side-by-side video is driven by a
`states.npy` of **2,528 bytes**; the same 120 frames as raw 96px pixels would be 3.3 MB.

The ratio is the point of the throughput numbers, not a separate claim. Regenerating a 96px frame costs
0.34 µs at batch 16,384, so the choice is between storing 27,648 bytes and spending a third of a
microsecond. Whether that beats reading the stored frame is now measured rather than left open, along
three paths:

* [`examples/bench_vs_decode.py`](https://github.com/galilai-group/stable-envs/blob/main/examples/bench_vs_decode.py)
  — one CPU core decoding *this env's own frame*, JPEG q90, already in memory, at the env's native
  resolution. The case most favourable to a stored dataset, and the column in the family table below.
* [`examples/bench_storage.py`](https://github.com/galilai-group/stable-envs/blob/main/examples/bench_storage.py)
  — real JPEG files at a fixed 256px and batch 1024, written to disk and read back through the page
  cache, from node-local disk and again from the shared filesystem a cluster corpus actually lives on.
  Distinct frames rather than one buffer re-decoded, because re-decoding one buffer keeps it in L2 and
  flatters the stored path.
* [`examples/bench_lance.py`](https://github.com/galilai-group/stable-envs/blob/main/examples/bench_lance.py)
  — the strongest disk-based baseline in use today: JPEG at quality 95 in an episode-contiguous Lance
  table, fetched a chunk of rows at a time and batch-decoded through libjpeg-turbo, on one core, on
  eight worker processes, over the shared filesystem, and through nvJPEG on the GPU.

The load half of load-and-decode turns out not to be the deciding term: across 196 envs, reading a warm
local JPEG file and decoding it runs at 0.45–1.95× the speed of decoding the same bytes already in
memory, median 0.96×. A stored-frame pipeline is decode-bound rather than IO-bound, which is why the
in-memory column is both the favourable bar and very nearly the only one. A corpus that has to cross a
network is out of scope and would do worse.

## The rest of the registry

The sweep runs over the registry at each env's native resolution, writing after every env. Three things
it is not: a like-for-like comparison across envs (they render at different resolutions, from states of
very different widths); a finished picture (the registry is under active development and several envs in
it are mid-port — those appear in the raw output with the error they raised, rather than being dropped);
and a permanent snapshot (the registry grows between runs, which is why the script merges per env instead
of replacing the file, so a sweep can resume where the last allocation stopped).

The registry is **196 environments in 24 families** at the last census (2026-08-11). No single
interpreter can import every family, so the census is the union over 15 of them and the widest single one
holds 147; `scratch/census/union.json` is the source. `LunarLander` and `Sokoban`, which an earlier sweep
could not construct, both ship.

The table below is the `## By family` table of
[`examples/results/decode/ALL.md`](https://github.com/galilai-group/stable-envs/blob/main/examples/results/decode/ALL.md),
merged from 14 benchmark shards, each env at its native frame size and its own `default_supersample`.
**default** is the env's official renderer, one frame at a time, which is the only entry point it offers;
**ours** is the GPU path at its best batch; **stored** is one CPU core decoding *this env's own frame*,
JPEG q90, already in memory. **cores replaced** is ours / stored.

!!! warning "The `ours` column predates two kernel changes"

    That sweep was taken at 05:58 on 2026-08-11, and both the fused 2-D rasterizer and the fused
    ray-cast path landed after it. It understates the 2-D families badly. Re-measured at 256 px /
    batch 1,024 the same day: `CarRacing` 179 → **21,943** frames/s, `Roundabout` 1,520 → **88,471**
    — 123x and 58x. The ray-cast families move by less at that resolution. Current per-family ratios,
    against eight worker processes batch-decoding JPEG-in-Lance rather than against one core, are
    regenerated from the storage sweep into `examples/paper/tables/family.tex`. To refresh the table
    below, re-run `examples/bench_vs_decode.py` per tag and then `scratch/census/merge_bench.py`.

| family | envs | frame | default f/s ¹ | ours f/s ¹ | stored f/s/core | **cores replaced** |
|---|---|---|---|---|---|---|
| Griddly | 7 | 144x120… | 99–145 | 2,307,114–9,438,816 | 851–5,276 | **1,789.0–3,647.9x** |
| Sokoban | 1 | 160x160 | 253 | 9,571,978 | 3,983 | **2,403.2x** |
| Overcooked | 5 | 192x224… | 41–1,595 | 3,615,605–8,145,368 | 2,914–4,399 | **1,241.0–1,860.7x** |
| MiniGrid / BabyAI | 7 | 224x224… | 117–942 | 711,991–4,479,313 | 608–4,011 | **329.7–1,225.8x** |
| MiniHack | 1 | 336x1264 | — | 244,456 | 698 | **350.0x** |
| Jumanji | 8 | 200x200 | 3–88 | 409,277–1,286,366 | 3,086–4,387 | **105.2–340.4x** |
| Procgen | 4 | 64x64 | 899–975 | 313,372–1,588,402 | 8,344–11,055 | **28.3–190.4x** |
| PushT | 1 | 96x96 | 472 | 1,256,557 | 10,259 | **122.5x** |
| OGBench | 25 | 224x224 | 82–491 | 48,978–206,740 | 1,893–4,249 | **11.9–93.6x** |
| swm nav | 5 | 224x224… | 52–772 | 71,140–187,629 | 1,083–4,985 | **15.3–86.7x** |
| Crafter | 1 | 64x64 | — | 665,836 | 8,186 | **81.3x** |
| Craftax | 1 | 112x144 | — | 569,934 | 7,039 | **81.0x** |
| PGX | 4 | 240x240… | 21–108 | 78,322–109,484 | 2,055–3,160 | **25.0–45.7x** |
| classic control | 5 | 400x600… | 249–541 | 33,216–39,843 | 1,120–1,626 | **22.3–35.6x** |
| D4RL PointMaze | 4 | 224x224 | 1,846–2,238 | 88,303–88,935 | 2,919–4,403 | **20.1–30.3x** |
| DM Control | 16 | 84x84 | 272–370 | 3,813–140,391 | 6,110–10,430 | **0.4–23.0x** |
| LunarLander | 1 | 400x600 | 336 | 18,786 | 1,165 | **16.1x** |
| highway-env | 6 | 150x600… | — | 1,520–20,389 | 660–3,205 | **2.3–7.2x** |
| Fetch | 12 | 480x480 | 323–404 | 6,008–6,164 | 896–1,399 | **4.3–6.9x** |
| HumanoidBench | 17 | 256x256 | 31–59 | 9,330–16,835 | 2,347–4,424 | **2.3–6.8x** |
| MuJoCo locomotion | 10 | 240x240 | 329–2,591 | 4,404–10,931 | 2,390–4,834 | **1.6–4.2x** |
| Meta-World | 50 | 480x480 | 8–219 | 1,039–2,554 | 655–1,066 | **1.2–3.8x** |
| RoboCasa | 4 | 256x256 | 87–560 | 5,122–11,687 | 2,511–3,718 | **1.7–3.1x** |
| CarRacing | 1 | 96x96 | 133 | 179 | 9,884 | **0.0x** |

¹ **Neither column travels unconditionally, and this page used to claim otherwise for one of them.** A
row whose `cpu%` is near 0 is one kernel on a device the job owns, and that row does repeat across nodes
to under a percent — but the 0.4% figure quoted here previously was verified on the fused 2-D rasterizer
and does not extend to the ray-cast families. Those are host-bound: they sit at 100–206% of process CPU
seconds per wall second of the timed loop, so the host is busy for the whole render and the number is a
property of the node's processor under the node's load. `DMCBallInCupCatch` measured 3,664 f/s idle and
271–978 contended, a 13× spread on a column that claimed 0.4%. The `default` column is a single-threaded
Python renderer sharing a node and has moved by more than 100× with load; read it as an order of
magnitude, and discard a baseline taken on a loaded node rather than averaging it in. Nine envs have no
`default` number because no reference package was installed in the interpreter that owns them, which is
an absence rather than a zero. The evidence is the `## Provenance and caveats` section of `ALL.md` and
the module docstring of
[`examples/bench_gpu_vs_cpu.py`](https://github.com/galilai-group/stable-envs/blob/main/examples/bench_gpu_vs_cpu.py);
the per-env `cpu%` is in the per-env table of `ALL.md`.

The spread across families tracks *how the frame is produced* rather than how big the frame is, with the
host-bound caveat above sitting on top of it. The tile-atlas and sprite envs gather from a precomputed
atlas and clear a million frames a second. The 2-D rasterized envs (PushT, classic control) sit between.
The ray-cast families cast rays against analytic primitives per pixel and drive that from a per-chunk
Python loop, which is where their `cpu%` comes from and why their rows move with the node. The older
claim that Gymnasium MuJoCo is the one family whose official renderer is competitive no longer holds:
CarRacing is the family where the official renderer is genuinely close (133 f/s against ours at 179),
while MuJoCo locomotion runs 3–31× its own renderer and D4RL PointMaze, whose official renderer is one of
the fastest measured at 1,846–2,238 f/s, is still 40–48× behind ours.

The nine envs with no `default` number are the six highway-env cells, MiniHack, Crafter and
CraftaxClassic: no reference package was installed in the interpreter that owns them, and that is
recorded as the import error rather than filled in.

The per-env rows behind the table above are in
[`examples/results/decode/ALL.md`](https://github.com/galilai-group/stable-envs/blob/main/examples/results/decode/ALL.md);
the batch-by-batch tables for the envs `bench_gpu_vs_cpu.py` has swept are in
[`examples/results/bench_gpu_vs_cpu.md`](https://github.com/galilai-group/stable-envs/blob/main/examples/results/bench_gpu_vs_cpu.md).
Two families are worth calling out for the shape of their numbers rather than their size:

* **Tile-atlas envs (MiniGrid, BabyAI)** render by gathering from a precomputed atlas rather than
  rasterizing, so they have no `scene()` / rasterize split and their throughput scales with output bytes
  almost immediately. They are also the family that reaches byte-exact parity, not merely interior-exact.
* **Large-canvas envs** (classic control renders at 400x600 and 500x500, MiniGrid FourRooms at 608x608)
  hit the output budget before they hit the kernel's ceiling: 65,536 frames at 500px is 49 GB of output,
  so the sweep records the cell as skipped rather than pretending to measure it.

## Caveats

* Nothing here says how a heavier scene scales. PushT is nine primitives with four polygon vertices, and
  both are compile-time constants in the kernel.
* The official-renderer side is single-process. The reference envs offer no batched entry point, so
  there was nothing fairer to time. The stored-frame side is a different matter, and the multi-core case
  is no longer hypothetical: `examples/bench_lance.py` measures eight worker processes batch-decoding
  JPEG-in-Lance from node-local NVMe with the page cache warm, and that eight-worker path — not one core
  — is the baseline the paper's per-family table is quoted against.
* `peak GB` is `torch.cuda.max_memory_allocated` around the timed region, which includes the
  pre-allocated output buffer. It is what a training loop would actually hold, not the kernel's working
  set.
* Figures come from a single allocation and are internally consistent. Repeating the sweep on a different
  node moved the CPU baseline by about 25% (478 frames/s on one, 602 on another) and the GPU column by
  under 10%, so treat cross-run comparisons of absolute values with more caution than the ratios. That
  last figure is the fused 2-D rasterizer and does not generalise: the host-bound rows in the family
  table above move much further than 10% with the node, which is what the footnote there records.
* Timed cells stop early once they have spent a second of wall clock, with a floor of 3 repetitions.
  Noise here is a fixed absolute quantity — launch jitter of a few hundred microseconds — so a cell that
  already takes tens of milliseconds does not need 25 repetitions to be stable.

## Reproducing

```bash
python examples/bench_gpu_vs_cpu.py --out examples/results --envs PushT --sizes 96,224
python examples/bench_gpu_vs_cpu.py --out examples/results --sizes native
```

It walks `stable_envs.REGISTRY`, so a newly registered env is benchmarked with no edit. Results merge
into the existing JSON per env, so the sweep can be split across allocations, and they are written after
every env, so a reference env that takes the process down costs one env rather than the run. Envs without
a reference renderer get the GPU columns and no speedup column, rather than a fabricated baseline.
