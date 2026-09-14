# Step 6 — `gfxdraw` (classic control): **PATCH**

- **baseline** `89fa5ca81341dc4df7186ec325cb691431a2ef17` (`sami/pre-step6-baseline`)
- **branch** `step6/gfxdraw`
- **affected configurations** (all replayed): `Acrobot`, `CartPole`, `MountainCar`,
  `MountainCarContinuous`, `Pendulum`. Other consumers of the same primitive layer, checked but not
  in the frozen corpus: `LunarLander`, `CarRacing`.
- **pixels changed: NO.** All five replay bit-identical to the frozen Step 4 oracle on CUDA *and*
  on CPU; every conformance check keeps the status it had in Step 4; the fused-path identity test
  is green; and 512 states per configuration, half of them outside the reachable set, are
  byte-identical to 89fa5ca.
- **tiers.** CPU `CPU_REGRESSION -> CPU_PARITY` on four of five (Acrobot 0.94, a hair under), from
  `S_cpu` 0.17-0.28 to 0.94-1.92, a 5.5-7.1x renderer speedup. GPU `GPU_FASTER -> GPU_FASTER` with
  `R` 37-71 -> 83-179, a 2.2-2.5x speedup at every batch size and ~1.4 GB less peak VRAM at 4096.
  Neither device reaches its FAST tier; the measured reason for both is one ticket, GFXDRAW-2.
- **semantic follow-up: none.** Section 15's first case — the frames are digest-identical, so there
  is nothing for a tripwire to see.
- **hardware** RTX 3090 (`gpu2104`) for GPU; `node2338` (batch partition, 1 torch thread) for CPU
  certification; venv `$SCR/venvs/se-step6` (torch 2.14.0+cu126). Laptop numbers are Apple M5,
  `/Users/samib/se6-venv`, and are never certification numbers.

## What the profile said

`env.render(states)` on the frozen Step 4 corpus, one torch thread, operator trace from
`scratch/step6/profile_note.py` and phase timers from `scratch/step6/gfxdraw/phase_profile.py`
(wrappers at the renderer's own seams; nothing in `stable_envs` was edited to take them).

Two regimes, and they want different fixes:

**At batch 1 the frame is dispatch.** 26,477 operator calls for one Acrobot frame, 13,892 for
Pendulum. `clip_line` alone was 33-38% of the frame in EIGHT calls of ~180 tiny operators each --
the six Cohen-Sutherland iterations cost the same whatever the edge's length, and `polygon` called
them once per edge. `blend_at` was another 34-42% across 37-42 scatters.

**At batch 16 and above the frame is `(B,H,W,3)` temporaries, one set per PRIMITIVE.** 68-74% of
the frame in three or four `StyleCanvas.paint` calls and a further 11-14% in `fill_polygon_mask`.
`paint` was `buf.copy_(torch.where(mask.unsqueeze(-1), c, buf))`: it read and rewrote the WHOLE
surface for a cart 30 rows tall, and it did it through a temporary, so a primitive moved about
12 bytes per pixel of the frame whatever its own size.

The second number is the one that decides the tier, because certification takes the best batch.

## Why the write was slow, measured rather than assumed

`torch.where` over a broadcast bool mask on an interleaved `(B,H,W,3)` surface runs at **340
Mpx/s** on one core. The same select on a single plane runs at 2.6 Gpx/s with `where` and **4.8
Gpx/s** with the bitwise identity `d ^= (d ^ s) * m`. A plain `copy_` of the same bytes runs at
89 GB/s, so the interleaved form was 30x off the machine and it was not bandwidth: it was a mask
that has to broadcast across a trailing axis of three.

## The intervention

Three changes in `render/gfxdraw.py`, and the windows each env supplies from its own geometry.

1. **The canvas is planar `(3, B, H, W)`**, `flat` correspondingly `(3, B*H*W+1)` so a scatter
   still touches all three channels in one indexing operation -- the channel axis is the one not
   being indexed. `select_` chooses `torch.where(..., out=)` on CUDA and the bitwise identity on
   the CPU; `tests/unit/test_gfxdraw_select.py` checks the two agree on the bit, because a device
   branch in a renderer claiming bit-identity is exactly the thing that silently diverges.
   Measured at batch 64 on 400x600: **45.4 ms per primitive -> 3.2**.

2. **Primitives are bounded.** `polygon` and `circle` take a window and build their mask over it.
   Every window here holds for EVERY state, not for the sampled ones: `x` moves CartPole's cart
   along the track but the cart is always 30 px tall about `CARTY`; Acrobot's second joint is one
   link length from the first in SOME direction, and a sine is bounded whatever its argument. A
   window that were merely usually right would truncate a primitive, so this is a property of the
   geometry and is written down as one (`_box`, and the `WIN_*` constants beside each `_draw`).

   A disc goes further and is drawn as **its own offset stencil** (`disc_offsets`): the write set
   of `filled_circle` is a constant of the radius, so a circle costs its own few hundred pixels
   instead of a window. That matters because a disc's window is not small -- MountainCar's wheels
   ride a sine, so their rows span 341 of 400 while each wheel is 17 px across, and the predicate
   form evaluated 205,000 pixels per wheel to paint 200.

   `kmax` is now per EDGE. A rotated rectangle alternates long and short sides and the Wu pass
   costs its bound in steps regardless, so one bound made every short edge pay the long one's
   price.

3. **One `clip_line` per polygon, not per edge** (`edge_setup`). The WRITES stay per edge and in
   draw order: two edges of a polygon can land on the same pixel at a shared vertex, and a batched
   scatter reads every destination before writing any of them, so merging the scatters would
   change the picture at corners. Only the geometry -- which touches no pixel -- is batched.

One more, from the sweep's own memory numbers: `rgb()` builds the interleaved frame one plane at
a time. `torch.stack` over the three flipped planes measures the same 11.7 ms at batch 64, but it
needs a second copy of the surface alive while it builds the third, which is 3.1 GB of transient
at batch 4096 on a 500x500 frame against 1.0.

## What was NOT done, and why

- **The fused path was not reused.** `gfxdraw_fused` is a Triton kernel over a list of 4-vertex
  quads with `pygame.draw`'s rasterizer for the tail; it has no circle, no `hline`, no blitted
  sprite, and no CPU. Classic control's frames are a quad, two discs and a row. Routing them
  through it would have meant extending the kernel to three more primitive kinds for a path that
  is CPU-bound, and it would have done nothing for the CPU rows that are the actual gap. The test
  that pins it against the per-primitive path was kept green instead, which is what makes the
  per-primitive path safe to change.
- **Data-dependent windows were rejected.** A window tightened from the batch's own extent would
  need `.item()` and a host sync per primitive (rule 6), and at the batch sizes certification uses
  the union over 4,096 states is the whole frame anyway.
- **A local-grid mask with a scatter** (evaluate a rotated quad over its own 131x131 box at a
  per-env origin, then scatter) was costed and not implemented: the int64 index is 8 bytes per
  pixel against 1 for a mask, so at these window sizes it buys about 2x on one primitive of one
  env. It is the next thing to try if the polygon mask has to come down further.

## Measurements

### Where the time went, and where it goes now

Both arms on the same laptop in the same minute, batch 64, one thread, frozen states, phase timers
at the renderer's own seams (`scratch/step6/gfxdraw/phase_profile.py`). Milliseconds per batch.

| phase | Acrobot before | Acrobot after | CartPole before | CartPole after |
|---|---|---|---|---|
| opaque write (`StyleCanvas.paint`) | 390.8 (4 calls) | **6.2** (2) | 277.9 (3) | **3.3** (2) |
| scanline mask (`fill_polygon_mask`) | 47.6 | 19.5 | 44.2 | 10.4 |
| background fill | 27.9 | 1.5 | 26.4 | 1.4 |
| circle mask (`fill_circle_mask`) | 13.3 | — drawn as offsets | 6.3 | — |
| Wu scatters (`blend_at`) | 18.0 (42 calls) | 6.7 (42) | 14.0 (41) | 5.4 (41) |
| `clip_line` | 5.3 (8 calls) | 1.4 (**2**) | 5.3 (8) | 1.4 (**2**) |
| interleave + flip (`rgb`) | 1.5 | 19.3 | 1.4 | 18.6 |
| **whole render** | **506.7** | **56.6** | **377.7** | **42.0** |

The last two rows are the trade, stated plainly. Going planar costs one transpose per FRAME — 1.5 ms
becomes 19.3 — and saves 385 ms of per-PRIMITIVE writes. It is now the single largest line in the
after column, and it is the reason CPU_FAST is out of reach for a torch-op renderer (below).


Renderer throughput is the thing this patch changed and it is reported first, because on a shared
node it is also the only number that is not a ratio of two independently noisy measurements.

### CPU, `node2338`, 1 torch thread, frozen Step 3 sweep

| config | B=1 | B=16 | B=64 | B=256 | B=1024 | best | official f/s | decode f/s/core |
|---|---|---|---|---|---|---|---|---|
| Acrobot before | 20 | 59 | 60 | 60 | - | **60** | 125 | 1150 |
| Acrobot after | 46 | 190 | 258 | 218 | 223 | **258** | 55 | 855 |
| CartPole before | 23 | 78 | 77 | 75 | - | **78** | 218 | 1165 |
| CartPole after | 50 | 341 | 396 | 375 | 317 | **396** | 218 | 961 |
| MountainCar before | 24 | 62 | 60 | 56 | - | **62** | 221 | 920 |
| MountainCar after | 49 | 316 | 337 | 296 | 278 | **337** | 115 | 923 |
| MountainCarContinuous before | 22 | 61 | 59 | 55 | - | **61** | 140 | 1174 |
| MountainCarContinuous after | 59 | 390 | 367 | 320 | 292 | **390** | 141 | 1185 |
| Pendulum before | 31 | 64 | 63 | 58 | - | **64** | 47 | 1175 |
| Pendulum after | 90 | 460 | 356 | 336 | 309 | **460** | 32 | 924 |

### GPU, RTX 3090 `gpu2104`, mandatory batches

| config | B=1 | B=16 | B=64 | B=256 | B=1024 | B=4096 | peak GB @4096 |
|---|---|---|---|---|---|---|---|
| Acrobot before | 11 | 173 | 695 | 2,611 | 8,778 | 16,661 | 10.244055552 |
| Acrobot after | 22 | 347 | 1,364 | 5,161 | 16,750 | 37,221 | 8.84584192 |
| CartPole before | 11 | 176 | 693 | 2,639 | 9,194 | 19,391 | 9.833229824 |
| CartPole after | 23 | 361 | 1,414 | 5,416 | 18,109 | 43,303 | 7.649137664 |
| MountainCar before | 12 | 197 | 776 | 2,926 | 9,259 | 16,849 | 9.832854016 |
| MountainCar after | 24 | 376 | 1,466 | 5,531 | 18,404 | 37,940 | 8.640470016 |
| MountainCarContinuous before | 12 | 196 | 778 | 2,925 | 9,231 | 16,814 | 9.832854016 |
| MountainCarContinuous after | 24 | 372 | 1,452 | 5,468 | 18,196 | 37,794 | 8.640470016 |
| Pendulum before | 21 | 332 | 1,308 | 4,585 | 12,116 | 18,622 | 10.256426496 |
| Pendulum after | 40 | 620 | 2,430 | 9,108 | 26,610 | 46,694 | 9.143596032 |

`peak GB @4096` is the cell's own peak allocation. It went DOWN, 9.8-10.3 GB to 7.6-9.1 GB: a
planar surface is the same bytes as an interleaved one, the windows cut the mask temporaries, and
`rgb()` no longer holds a second copy of the frame.

### Operator count per render, CPU, batch 64

| config | before | after |
|---|---|---|
| Acrobot | 26,477 | 12,970 |
| CartPole | 26,166 | 12,619 |
| MountainCar | 23,340 | 12,139 |
| MountainCarContinuous | 23,340 | 12,139 |
| Pendulum | 13,892 | 7,224 |

`aten::where` was the top operator in every baseline note (1,396 calls on Acrobot, 709 on
Pendulum). It is not in the top eight of any after-note.

### Operator count per render, CUDA, batch 4096, both arms on `gpu2104`

| config | before | after | profiled f/s before -> after |
|---|---|---|---|
| Acrobot | 44,307 | 20,859 | 15,728 -> 33,156 |
| CartPole | 43,812 | 20,317 | 18,005 -> 37,836 |
| MountainCar | 39,029 | 19,532 | 15,827 -> 34,830 |
| MountainCarContinuous | 39,029 | 19,532 | 15,767 -> 34,985 |
| Pendulum | 23,148 | 11,537 | 18,346 -> 43,729 |

(These are under `torch.profiler`, so they are lower than the sweep's numbers; the ratio is the
point. The before arm's notes record `stable_envs_file` under `se6/baseline`.)

### A wider bit-exactness sweep than the frozen 64

512 states per configuration, half drawn from the reachable set and half deliberately outside it —
CartPole `x` in [-8, 8] and `theta` over four turns, Acrobot and Pendulum angles over 12 turns,
MountainCar positions in [-5, 4]. All five frame-stack digests are **byte-identical** between
89fa5ca and this branch. That is the check the static windows and the per-edge `kmax` need, because
both are bounds and a corpus chosen for coverage of the reachable set cannot test a bound.

### Tiers, computed by the frozen evaluator

`scratch/step3/certify.py` reads every bound out of `scratch/certification.json`; `scratch/step6/gfxdraw/tiers.py`
only points it at the two Step 6 shards. No threshold is retyped anywhere in this workstream.

**CPU — and the certification node makes this two answers, so both are here.**

The frozen sweep (`examples/bench_step3.py`, `node2338`, both arms in one job) returns:

| config | best B | f/s before | f/s after | x | S_cpu before | S_cpu after | tier before -> after |
|---|---|---|---|---|---|---|---|
| Acrobot | 64 | 60 | 258 | 4.3x | 0.48 | 4.64 | CPU_REGRESSION -> CPU_FASTER |
| CartPole | 64 | 78 | 396 | 5.1x | 0.36 | 1.82 | CPU_REGRESSION -> CPU_PARITY |
| MountainCar | 64 | 62 | 337 | 5.4x | 0.28 | 2.94 | CPU_REGRESSION -> CPU_FASTER |
| MountainCarContinuous | 16 | 61 | 390 | 6.4x | 0.44 | 2.77 | CPU_REGRESSION -> CPU_FASTER |
| Pendulum | 16 | 64 | 460 | 7.2x | 1.38 | 14.60 | CPU_PARITY -> CPU_FASTER |

**Do not read the right-hand columns of that table.** node2338 ran at a load average of 47 and the
benchmark times the official renderer once, after the sweep: its Acrobot reference came out at 125
frames/s in the before arm and 55 in the after, and its Pendulum reference at 47 and 32 where the
same renderer does 243-261 on a quieter node and 794 on the laptop. The `S_cpu` after column is
inflated by a collapsed denominator, and `Pendulum 14.60` is that artefact, not a result. The f/s
and the speedup columns are sound — they are the numerator only.

So the same measurement was taken again, paired: reference burst, renderer burst, round after
round, in one process, median of the per-round ratio (`scratch/step6/gfxdraw/paired_cpu.py`). This
is the number to use.

**CPU, the paired measurement (`node1829`, 1 thread, reference and renderer alternating):**

| config | best B | f/s before | f/s after | x | S_cpu before | S_cpu after | tier before -> after |
|---|---|---|---|---|---|---|---|
| Acrobot | 256 | 69 | 372 | 5.5x | 0.18 | **0.94** (0.94-0.96) | CPU_REGRESSION -> **CPU_REGRESSION** |
| CartPole | 256 | 88 | 520 | 5.8x | 0.20 | **1.25** (1.22-1.31) | CPU_REGRESSION -> **CPU_PARITY** |
| MountainCar | 256 | 70 | 466 | 6.6x | 0.17 | **1.12** (1.10-1.21) | CPU_REGRESSION -> **CPU_PARITY** |
| MountainCarContinuous | 256 | 71 | 447 | 6.3x | 0.17 | **1.12** (1.07-1.17) | CPU_REGRESSION -> **CPU_PARITY** |
| Pendulum | 64 | 71 | 504 | 7.1x | 0.28 | **1.92** (1.88-1.95) | CPU_REGRESSION -> **CPU_PARITY** |

Every batch size, for completeness:

| config | B | before f/s | after f/s | x | S_cpu before | S_cpu after |
|---|---|---|---|---|---|---|
| Acrobot | 16 | 63 | 300 | 4.7x | 0.16 | 0.76 |
| Acrobot | 64 | 69 | 382 | 5.6x | 0.18 | 0.94 |
| Acrobot | 256 | 68 | 372 | 5.5x | 0.17 | 0.94 |
| CartPole | 16 | 80 | 373 | 4.7x | 0.17 | 0.82 |
| CartPole | 64 | 88 | 547 | 6.2x | 0.20 | 1.18 |
| CartPole | 256 | 90 | 520 | 5.8x | 0.20 | 1.25 |
| MountainCar | 16 | 65 | 343 | 5.2x | 0.16 | 0.84 |
| MountainCar | 64 | 70 | 462 | 6.6x | 0.17 | 1.09 |
| MountainCar | 256 | 71 | 466 | 6.6x | 0.17 | 1.12 |
| MountainCarContinuous | 16 | 66 | 346 | 5.2x | 0.16 | 0.84 |
| MountainCarContinuous | 64 | 70 | 474 | 6.8x | 0.17 | 1.11 |
| MountainCarContinuous | 256 | 71 | 447 | 6.3x | 0.17 | 1.12 |
| Pendulum | 16 | 68 | 414 | 6.1x | 0.28 | 1.61 |
| Pendulum | 64 | 71 | 504 | 7.1x | 0.27 | 1.92 |
| Pendulum | 256 | 71 | 450 | 6.4x | 0.28 | 1.85 |

Four of five move CPU_REGRESSION -> CPU_PARITY; Acrobot lands at 0.94, a hair under. `cores_cpu`
is 0.22-0.50 throughout (the decode bar is ~850-1,180 frames/s for these frames on that node), so
every row keeps `BELOW_G4` and none reaches CPU_FAST.

**GPU, class R, RTX 3090 `gpu2104`, both arms on the same card in one job:**

| config | R before | R after | cores before | cores after | tier before | tier after |
|---|---|---|---|---|---|---|
| Acrobot | 42.2 | 91.4 | 16.3 | 27.3 | GPU_FASTER | GPU_FASTER |
| CartPole | 39.9 | 90.6 | 13.2 | 29.4 | GPU_FASTER | GPU_FASTER |
| MountainCar | 37.1 | 82.7 | 14.7 | 27.1 | GPU_FASTER | GPU_FASTER |
| MountainCarContinuous | 37.4 | 84.4 | 15.1 | 27.0 | GPU_FASTER | GPU_FASTER |
| Pendulum | 70.9 | 178.9 | 17.0 | 33.6 | GPU_FASTER | GPU_FASTER |

`BATCH_SCALES` and `NO_LARGE_BATCH_REGRESSION` **pass in both arms**, and this contradicts the
brief. `fps(4096)/fps(1)` is **890x-1,890x** against a 20x bar (before: 891-1,747x). The
`blocking_qualifiers: {BATCH_SCALES: 5}` in `examples/results/step6/plan.json` is counting a
SATISFIED qualifier: `certify.classify` appends the string `"BATCH_SCALES"` when the condition
holds and `"BATCH_SCALES_FAIL"` when it does not, and the plan's summary treats the presence of the
name as a blocker. **The only thing between these five rows and GPU_FAST is R, whose class-R floor
is 500.** This patch closed that gap from ~12x to ~5.6x; it did not close it.

## Which tree each arm measured

A before/after job switches `PYTHONPATH` between `$SCR/se6/baseline` and `$SCR/se6/gfxdraw`, and a
script that does `sys.path.insert(0, ROOT)` would override that and make both arms measure the same
tree. None of the scripts used here insert the repo root (`bench_step3.py` inserts `examples/`;
`fidelity_step4.py` inserts `scratch/step4` and, inside a DM-Control-only helper, `ROOT` — neither
reached on this path and neither run in two arms). The measurements say the same thing directly:
the two arms differ by 26,166 against 12,970 operator calls, 60 against 258 frames/s on the CPU and
16,661 against 37,221 on the card. The harnesses now also record `stable_envs.__file__` and
`$PYTHONPATH` in every note, so later artifacts prove it rather than inferring it.

One wrinkle that came with it, fixed here: `tests/unit/test_repo_contracts.py` reads any
`scratch/...` substring in a readable file as a repo-relative citation and resolves it, so an
absolute cluster path inside a note reads as a broken citation — recording the measured tree took
the repo from 2 unresolved citations to 19. Prose artifacts now write `$SCR/...`, which is what
every sbatch script here calls that root, and the frozen corpus is named by its repo-relative path
rather than by whichever tree's copy was read (the copies are the same frozen file). The JSON keeps
the full path. The count is back to the 2 that were already there at 89fa5ca.

## Correctness

| check | result |
|---|---|
| `examples/fidelity_step4.py --configs <all five> --device cuda` | 5/5 `measured`; every mandatory and family-specific check has the SAME status as Step 4, and every oracle digest (`digest_ours`, `digest_ours_default_style`, `digest_ours_corpus_style`, `digest_official`), `distinct_colors` and `mean_pixel` is unchanged |
| our frames vs the OFFICIAL pygame renderer on the same host | identical on all five, CUDA |
| `tests/gpu/test_gfxdraw_fused.py` + `tests/gpu/test_fast_paths.py` (3090) | 123 passed, 166 skipped, 0 failed |
| `tests/gpu/test_conformance.py` + `test_reproducibility.py`, `-k` the five (3090) | 451 selected: **358 passed, 93 skipped, 0 failed** |
| `stable_envs/testing/conformance` mandatory + family checks, CPU, node2338 | 5/5 OK (batch invariance, batch-size invariance, chunking, order independence, render purity, style-as-state, per-env variation, backend agreement, colours apply exactly, visual fidelity, fidelity under variation) |
| `tests/unit` on Oscar | 2 failed, 504 passed. Both failures are **pre-existing at 89fa5ca** and unrelated: `test_examples[robocasa_visual/render_mjwarp.py]` and `test_every_cited_scratch_path_resolves` (three citations under `scratch/` that no longer resolve — a carracing log directory, a robocasa corpus, and the Step 6 report generator — the same three, before and after; named without their literal paths here so this file does not become a fourth) |
| 96 EXTREME states per env vs baseline AND vs the official renderer | 0 differing pixels. Cart fully off-screen, angles over nine turns, positions well outside the track — this is what checks the window bounds instead of assuming them |
| `LunarLander`'s three touch points on the canvas | its one direct `flat` scatter reproduces the interleaved semantics exactly; `paint_at` verified; the `_AARecorder` write list (order, positions, weights) is **byte-identical** at 89fa5ca and here, which is what its baked flag fringe is built from |
| `CarRacing` | uses only `G.Gfx` from this module; its kernel is unchanged and its identity test is green |

Two of those deserve their own line.

**Pendulum's frozen digest does not reproduce on a Mac, and did not before this patch either.**
Our CPU frame matches the official renderer on the laptop EXACTLY (0 of 16 M pixels), and matches
the frozen Step 4 digest on Oscar exactly, on both CPU and CUDA. The two hosts' pygame draws the
torque-arrow sprite differently. It is an upstream host difference, it predates this branch, and
the laptop replay therefore compares against a host baseline taken before the first edit.

**`kmax` per edge is a bound, and getting it backwards is caught by the oracle.** The first version
of this patch gave Acrobot's and Pendulum's quads `(120, 28, 120, 28)` when their vertex list
starts on the SHORT side. Two of five configurations went non-identical immediately; that is the
whole reason the digest replay runs after every edit rather than once at the end.

## What this does NOT reach, measured

**GPU_FAST needs R >= 500 for class R and this patch gets to 83-179.** The remaining ~5.6x is a
property of the shape, not of a constant: the frame is still ~20,000 operator launches of torch
ops, the surface is still read and written once per primitive within its window, and the Wu pass
still issues five scatters per edge. `gfxdraw_fused` is the demonstrated answer to exactly this on
exactly this primitive family — one kernel, primitives walked in registers, the surface written
once — and it reached 1,865x for PushT and made CarRacing viable. Extending it to a disc, an
`hline` and an alpha blit, and giving classic control a fused path, is a well-posed piece of work
with a precedent in this repository.

**CPU_FAST needs `S_cpu >= 5` AND `cores_cpu >= 1`, and `cores_cpu` is the binding one.** A
500x500 JPEG-q90 frame decodes at ~850-1,180 frames/s on one core of these nodes, and these
renderers now run at 258-520 there. The interleave alone is the wall: producing the caller's `(B,H,W,3)` uint8 from
three planes costs ~0.17 ms per 400x600 frame on one core in torch (10.7 ms at batch 64, and every
layout measured lands within 10%), against a 0.18 ms total budget at `S_cpu = 5` on the laptop.
A torch-op renderer cannot get under that; a fused kernel writing the frame once can.

Both point at the same ticket.

### GFXDRAW-2 (proposed, not started)

> Give classic control a fused per-pixel path, on the model of `render/gfxdraw_fused.py`, covering
> `filled_polygon`+`aapolygon`, `filled_circle`+`aacircle`, `hline`, and the alpha blit. Measured
> blockers it would remove: ~20,000 operator launches per frame; one surface pass per primitive;
> the three-plane interleave, which torch cannot do faster than ~0.17 ms per 400x600 frame on one
> core. Current distance to tier: GPU **R 83-179 against 500**; CPU **`cores_cpu` 0.22-0.50 against
> 1**. Triton has no CPU backend here, so the CPU half needs a different mechanism (a C++ extension,
> or `torch.compile` over a per-pixel form) and should be scoped separately from the GPU half.

## Other findings for the coordinator

1. **`plan.json` mislabels the gfxdraw GPU blocker.** It lists `blocking_qualifiers: {BATCH_SCALES: 5}`,
   but `certify.classify` appends the string `BATCH_SCALES` when the condition HOLDS and
   `BATCH_SCALES_FAIL` when it does not. Measured here, `fps(4096)/fps(1)` is 891-1,747x before and
   1,179-1,890x after, against a 20x bar. The real blocker is R, and the brief inherited the wrong
   one. Worth checking whether the same mislabel reaches other workstreams' rows.
2. **`scratch/step6/profile_note.py` crashed at its write step** (`n` for `note`, line 251), fixed
   here in its own commit. Because `$SCR/se6/baseline` is pristine at 89fa5ca it still carries the
   bug, so any workstream taking a profile-before by `cd`-ing into the baseline tree gets a full
   measurement and no file. Run the branch's copy of the harness against the baseline TREE instead.
3. **A shared certification node distorts `S_cpu`.** node2338 ran at a load average of 47 and the
   same official Acrobot renderer measured 125 frames/s in one arm and 55 in the other, minutes
   apart — a 2.3x swing in the denominator, wider than several tier boundaries. `paired_cpu.py`
   here alternates the reference and the renderer round by round and reports the spread; the
   coordinator may want that for every CPU row rather than just this one.
4. **No shared-helper proposal.** `render/scene.py` and `render/raster2d.py` were not touched and
   share no code with this path. The one cross-workstream observation is that finding #2's
   "each primitive is evaluated over the whole frame" applies to both, and the fix here — a static
   window per primitive plus an offset stencil for shapes whose write set is a constant — is a
   pattern raster2d could reuse without sharing code.

## Artifacts

All under `examples/results/step6/gfxdraw/`.

| what | where |
|---|---|
| profile-before, CPU (`node2314`, baseline, from the scaffolding commit) | `profile_before/*.md` |
| profile-before, CPU, same node as the after (`paired` job) | `profile_before_samenode/*.md` |
| profile-after, CPU (`node2338`) | `profile_after/*.md` |
| profile-after, CPU, same node as the before | `profile_after_samenode/*.md` |
| profile-before / profile-after, CUDA (`gpu2104`) | `profile_before_gpu/*.md`, `profile_after_gpu/*.md` |
| frozen Step 4 replay, CUDA | `replay/step4_replay_cuda.json`, `replay/digests_cuda_gpu2104.json` |
| frozen Step 4 replay, CPU | `replay/digests_cpu_node2338.json` |
| CPU sweep, both arms, one node | `sweep/cpu_1thread_{before,after}_node2338.json` |
| GPU sweep, both arms, one card | `sweep/gpu_{before,after}_gpu2104.json` |
| tiers from the frozen evaluator | `sweep/tiers_cpu_node2338.json`, `sweep/tiers_gpu_gpu2104.json` |
| paired S_cpu, both arms, alternating with the reference | `sweep/paired_{before,after}_B*_<node>.json`, `sweep/laptop_paired*.json` |
| GPU test output kept in full | `tests/conformance_reproducibility.txt`, `tests/gfxdraw_fused_fast_paths.txt` |

Tools added under `scratch/step6/gfxdraw/`: `phase_profile.py` (phase timers at the renderer's own
seams, wrappers only), `replay_digests.py` (the seconds-long replay loop), `consumer_digests.py`,
`conformance_cpu.py`, `paired_cpu.py`, `tiers.py`, and the four sbatch scripts.

## SLURM jobs

| job | id | node | what |
|---|---|---|---|
| `se6-gfxdraw-cpu` | 6346431 | node2338 | CPU sweep both arms, profile notes, `tests/unit`, CPU conformance, CPU replay |
| `se6-gfxdraw-gpu` | 6346430 | gpu2104 (RTX 3090) | Step 4 replay on CUDA, path tests, GPU sweep both arms, GPU profile-after |
| `se6-gfxdraw-gputests` | 6346933 | gpu2104 | conformance + reproducibility with their output kept |
| `se6-gfxdraw-paired` | 6346967 | (batch) | paired `S_cpu`, both arms, and the same-node profile pair |
| `se6-gfxdraw-gpuprof` | 6346999 | gpu2104 | GPU profile-BEFORE, the branch's harness against the baseline tree |
| `se6-gfxdraw-paired` | 6346954 | — | cancelled by me before it ran, to add the measured-tree provenance |

`se6-gfxdraw-paired` 6346967 landed on `node1829` (load average 30-41), the frozen sweep on
`node2338` (load average 47). Every artifact records its own host.

## Commits on `step6/gfxdraw`

- `3d20635` profile_note: fix NameError at the write step (n -> note)
- `a482d88` gfxdraw: planar canvas, bounded primitives, one clip per polygon
- `a64bd96` gfxdraw: interleave one plane at a time, and the Oscar before/after jobs
- `4609e13` tests: the two select_ branches must draw the same frame
- `22d98ec` test_gfxdraw_fused: build the reference frame through Canvas.rgb()
- `fcb9350` gfxdraw: a paired S_cpu measurement for a loaded certification node
- `183ec36` gfxdraw: rerun the GPU conformance suite with its output kept
- `31b4504` gfxdraw: profile the baseline TREE with the branch's harness, on both devices
- `b3e5578` step6 harnesses: record which TREE was measured, not which one was meant

---

## GFXDRAW-2-CPU — the design, measured. **Recommendation: do not build a PER-PIXEL-WALK CPU kernel.**

The ticket asked whether `cores_cpu >= 1` needs a non-torch mechanism and which one. It does need
one, and **the mechanism measured here is not it**: on the certification host a fused **per-pixel
walk** — the classic-control frame as one loop over pixels, each asking every primitive whether it
covers it — is **1.03-1.51x** faster than the torch path already shipped and **1.6-2.4x short** of
the bar, whether written in C++ or in numba. The architecture that looked obvious from the laptop
does not survive being measured.

**Scope of that negative, stated precisely.** It is a result about ONE design. A per-pixel walk
pays a bounding-box test and a branch per pixel PER primitive, over every pixel of the frame,
including the ~97% of the frame no primitive touches. SDL_gfx itself does not work that way and
neither does the torch path: both touch only the pixels a primitive covers. **`cores_cpu >= 1` is
therefore NOT shown to be unreachable on CPU** — what is shown is that this particular fused shape
does not reach it, and that the two mechanisms available for it (a compiled extension, numba) make
no difference to that because the shape is what costs.

### What was measured, and where

`node1831` and `node2334`, Intel Xeon Platinum 8268 @ 2.9 GHz, **one torch thread**, gcc 11.5,
the frozen Step 4 states, `$SCR/venvs/se-step6` (plus numba 0.67.0 in the Box2D venv). Four
mechanisms on the SAME frame, from `scratch/step6/gfxdraw/cost_floor.{py,cpp}` and
`numba_floor.py`:

* **torch, as shipped** — `env.render` with the planar canvas and bounded primitives, swept over
  batch, because the compiled walks have no dispatch to amortise and comparing them against torch
  at B=8 would credit a kernel with the win batching already gives.
* **compiled scalar C++** — `cost_floor.cpp`: the same op list, the same order, the same integer
  arithmetic as `render/gfxdraw_classic.py`, with nothing but the per-pixel walk left. It reads the
  op list, the clipped edge table and the disc tables from a blob the repository's own `Ops` writes,
  so it measures the walk and not a second implementation of the setup.
* **numba** — the same walk as an `@njit` loop: the same mechanism, a pip dependency instead of a
  binary in the wheel.
* **the interleave alone** — the floor of any design that keeps a planar surface.

**Every compiled and numba variant draws the byte-identical frame** (checksum against the torch
reference, 4 variants x 4 envs, all True). That is worth stating because it is the cheap half of
the bit-identity discipline `cpu-3d` §4 and the raycast3d fused cast both have to pay: the six
SDL_gfx rules are **integer arithmetic on integer geometry**, so there is no FMA contraction, no
division rounding, no `sqrt` and no fast-math to disable. A gfxdraw CPU kernel needs
`-ffp-contract=off` for nothing.

### The numbers

| config | frame | budget at `cores_cpu=1` | torch, best batch | C++ `-O3` | C++ `-march=native` | numba | best fused vs torch | short of the bar |
|---|---|---|---|---|---|---|---|---|
| CartPole | 400x600 | 0.868 ms = **3.62 ns/px** | 7.70 (B=64) | 6.82 | 7.77 | 6.73 | **1.14x** | **1.86x** |
| Acrobot | 500x500 | 1.008 ms = **4.03 ns/px** | 10.03 (B=64) | 9.78 | 11.87 | 9.83 | **1.03x** | **2.43x** |
| Pendulum | 500x500 | 0.944 ms = **3.78 ns/px** | 8.43 (B=64) | 7.47 | 9.34 | 9.39 | **1.13x** | **1.98x** |
| MountainCar | 400x600 | 0.943 ms = **3.93 ns/px** | 9.58 (B=256) | 6.33 | 7.89 | 7.64 | **1.51x** | **1.61x** |

`cores_cpu = 1` is one core of JPEG-q90 decode of the env's own frame, measured on the same host in
the same job (992-1,152 f/s). `S_cpu >= 5` is stricter still: against the official renderer's
~400-460 f/s it asks for ~1.8-2.2 ns/px, **3.1-4.5x** below the best fused scalar measured.

Torch per frame by batch (ns/px), which is why the comparison had to be swept:

| config | B=1 | B=16 | B=64 | B=256 |
|---|---|---|---|---|
| CartPole | 74.92 | 11.03 | **7.70** | 7.85 |
| Acrobot | 81.12 | 14.02 | **10.03** | 10.29 |
| Pendulum | 46.74 | 9.60 | **8.43** | 8.72 |
| MountainCar | 78.22 | 13.53 | 9.87 | **9.58** |

### Three things that fall out of this, and one that does not

**1. The fused-kernel case for CPU is dead at scalar width.** A per-pixel walk pays a bounding-box
test and a branch per pixel per primitive; the bounded torch path pays a vectorised write over each
primitive's own box and nothing at all outside it. At B >= 64 those come out level. This is the
opposite of `cpu-3d` §4.3's situation (torch 21 ms against scalar C 0.67-1.06 ms) and the reason is
structural: their bottleneck was ~200 kernels over materialised `(B,N)` temporaries per cast, and
Step 6 has already removed the equivalent here.

**2. The interleave is not the wall either, but it is a third of it.** On this host, turning three
uint8 planes into the caller's `(B,H,W,3)` costs, at B=64 on a 400x600 frame:

| spelling | ms/batch | ms/frame | ns/px |
|---|---|---|---|
| numpy, 3 strided assignments into the output | 24.5 | **0.383** | 1.60 |
| torch, 3 `copy_` of flipped planes (shipped) | 33.8 | 0.528 | 2.20 |
| torch, `stack` of flipped planes | 43.9 | 0.686 | 2.86 |
| torch, `flip().permute().contiguous()` | 65.5 | 1.024 | 4.27 |
| torch, int32-packed surface, drop the 4th byte | 111.1 | 1.736 | 7.23 |
| numpy, `stack` of reversed views | 253.1 | 3.954 | 16.48 |
| *(reference)* contiguous copy of the same bytes | <0.01 | ~0 | ~0 |

So the interleave is **44-61%** of the `cores_cpu = 1` budget on its own, and the shipped spelling
is **1.38x more expensive than the numpy one**. That is a free ~0.15 ms/frame on the CPU path, and
it is the single cheapest thing on this list — worth taking on its own, not as part of a kernel.
(It also corrects this file's earlier laptop figure; see the note above.)

**3. `-march=native` is a pessimisation on the certification host.** Plain `-O3` beat it on all
four envs (6.82 vs 7.77, 9.78 vs 11.87, 7.47 vs 9.34, 6.33 vs 7.89). The 8268 downclocks on
AVX-512, and this is a warning for the one lever that is left.

**And the two that do not fall out.**

*SIMD is untested.* Everything above is scalar. The per-pixel walk is lane-parallel across a row —
integer, no data-dependent branching within a primitive except the bounding box — and AVX-512 is 16
int32 lanes, so 3.1-4.5x is arithmetically available. Whether it is *realisable* on an 8268 that
throttles under AVX-512 (see the `-march=native` result above) is the open question, and it is the
same one `cpu-3d` §4.3 and `raycast3d`'s RAYCAST-3 both end on.

*And a whole design was not measured: a **span/scanline rasterizer writing `(H,W,3)` directly**.*
This is what SDL_gfx does and what the official renderer this family is benchmarked against IS.
Its cost is not per FRAME pixel but per PRIMITIVE pixel: fill the background once, then for each
primitive walk only its own rows and, within a row, only the span the 16.16 fill computes —
`hline(y, x0, x1)` on an interleaved surface, the Wu fringe as a short scatter, the disc as its
offset stencil. It writes the caller's layout as it goes, so **the interleave disappears entirely**
rather than costing 44-61% of the budget, and the ~97% of the frame no primitive covers is written
once by the background fill instead of being visited four or five times.

Rough arithmetic, NOT a measurement: CartPole's primitives cover roughly 8,000 of 240,000 pixels,
so the per-primitive work is ~3% of what the per-pixel walk does, and the floor becomes the
background fill plus the spans — order 0.2-0.3 ms per frame against a 0.868 ms budget. That is the
shape of the official renderer's own 400-460 frames/s (2.2-2.5 ms/frame including its per-call
surface allocation and its Python), so a compiled span rasterizer plausibly sits several times
under it. **This is the design that should be costed before anyone concludes `CPU_FAST` is out of
reach for this family**, and costing it is a `cost_floor` variant, not an integration: the same op
blob, a different loop.

What it would give up is the thing the per-pixel walk gives for free — painter's order and the
AA-before-fill interleaving are register updates there, and become real read-modify-write traffic
in a span rasterizer, which is exactly the trade `gfxdraw.py`'s own history is about. It is a real
design question and it is open.

### Recommendation

1. **Do not build a per-pixel-walk CPU kernel for gfxdraw.** Measured at 1.03-1.51x over the torch
   path and 1.6-2.4x short of the bar, in both C++ and numba, so it cannot deliver `CPU_FAST` and
   it does not even deliver a tier. The five rows stay `CPU_PARITY` (four) and `CPU_REGRESSION`
   (Acrobot, 0.94) with `BELOW_G4`. This is not a statement that CPU is finished — see 3.
2. **Take the interleave win in torch.** ~0.15 ms/frame, no new dependency, no new arithmetic: swap
   the three `copy_`s for numpy strided assignment on the CPU branch of `Canvas.rgb`. Worth roughly
   8% of the frame and it is the only free thing on this list. (Not done here: it is a change to a
   path this ticket's GPU half does not touch, and it wants its own replay.)
3. **If the coordinator still wants `CPU_FAST` for this family, cost the span rasterizer first,
   and the SIMD per-pixel walk second.** Both are `cost_floor` variants over the op blob that
   already exists, not integrations, and both are go/no-go before any integration as `cpu-3d` M2
   proposes for itself. The span form is the more promising by a wide margin and the cheaper to
   write; the SIMD form's gate is "vectorise the inner loop across 16 pixels and measure
   `<= 1.8 ns/px` on the 8268", against a `-march=native` result that is currently a pessimisation.
4. **Packaging, the shared recommendation.** `cpu-3d` §6 asks for one build decision across the
   three workstreams. This workstream's evidence adds two things to it. **numba 0.67.0 installed
   without a resolver conflict into a venv LAYERED on `se-step6`** — a fresh venv whose
   site-packages inherits `se-step6`'s through a `.pth`, so torch 2.14.0+cu126 and pygame are the
   certified ones and `se-step6` itself was never written to (`$SCR/venvs/se6-gfxdraw-box2d`,
   manifest in `characterize/venv_box2d_manifest.json`). That is evidence about a layered venv, not
   about the shipping environment's resolver. It measured *level with or better than* `-O3` C++ on
   three of four envs —
   so "a pip dependency and an LLVM JIT at import" is a live alternative to "a compiler in the
   build and a binary in the wheel", and it should be priced against `cpp_extension` rather than
   assumed away. And **this family needs none of the float discipline** the other two do, so if a
   single kernel family is built, gfxdraw is the cheapest member to make bit-stable and the worst
   one to justify: it is the only one of the three where the torch path is already at the scalar
   floor.

### Artifacts

`examples/results/step6/gfxdraw/characterize/`: `cost_floor_batched_node1831.json` (the table above),
`cost_floor_node2334.json`, `cost_floor_numba_node2334.json`, `decode_node2334.json` (the bar),
`interleave_node2334.json` and `interleave_500_node2334.json` (the interleave band on the
certification host), `interleave_laptop.json`. Tools: `scratch/step6/gfxdraw/cost_floor.py`,
`cost_floor.cpp`, `numba_floor.py`, `interleave_bench.py`, `cpu_floor.sbatch`, `cpu_floor2.sbatch`.
SLURM 6348987 (`node2334`), 6349091 (`node1831`).

---

## GFXDRAW-2-GPU — the fused per-pixel path for classic control

`stable_envs/render/gfxdraw_classic.py`, reached from `_Classic.render` when the device is CUDA and
Triton is present. One kernel, one pass, one store per pixel, already flipped.

### What it had to cover, and what `gfxdraw_fused` did not

`gfxdraw_fused` is CarRacing's, and it draws what CarRacing draws: 4-vertex gfxdraw quads plus
`pygame.draw` polygons, over a flat background. Classic control draws a different alphabet, and the
missing four are not decorations:

| primitive | where | what the kernel has to do |
|---|---|---|
| polygon, 3 or 4 vertices, + `aapolygon` | all five | the Wu inversion `gfxdraw_fused` already has, plus a variable vertex count |
| `filled_circle` + `aacircle` | CartPole axle, Acrobot joints, Pendulum hub/end/axle, MountainCar wheels | **new**: the disc as a table |
| `hline` | CartPole's track, Acrobot's ground | **new** |
| `vline` | MountainCar's flagpole | **new** |
| alpha blit of a pre-scaled sprite | Pendulum's torque arrow | **new**, and a different blend rule from SDL_gfx's |
| a pre-baked per-environment plate as the background | MountainCar's terrain | **new** |

**A disc is a table, and that is the one idea in this module that is not in `gfxdraw_fused`.**
`filled_circle`'s half-width per row and `aacircle`'s fringe weights are constants of the RADIUS,
which is a constant of the environment and never of the state or the style — so both become a dense
`(2r+1)^2` lookup at `(dy, dx)` rather than a midpoint walk. `disc_table` builds them from
`gfxdraw.circle_hw` and `gfxdraw.AACIRCLE_FRINGE`, the same two objects the per-primitive path
reads, so there is one definition of a disc in the repository rather than two that agree.

### Three implementations, two joints

    gfxdraw.py            per primitive, batched torch      the executable specification
    gfxdraw_classic.py    render_classic_torch              the same rules, per pixel, in torch
    gfxdraw_classic.py    render_classic (Triton)           the same again, in one kernel

The middle one is not a convenience. Triton does not exist on the laptop, so without it every
mistake — a window that clips a primitive, an order that swaps two, a colour from the wrong style
path — would have cost a GPU job to find. With it, the description is checked in seconds on a CPU
and the kernel only has to match a form that is already known to draw the right picture. It is the
same split `raycast3d` keeps between `cast_nearest_torch` and its Triton kernel, and
`tests/gpu/test_gfxdraw_classic.py` asserts both joints separately so a failure says which file to
open.

### Results (job 6349362, `gpu2110`, RTX 3090, both arms in one process on one card)

The "before" arm is this tree with `use_fused_render` False, which is the code 6213673 shipped;
flipping a class default rather than a `PYTHONPATH` makes the two arms same-card and same-process
by construction and removes the failure mode that costs a GPU job — measuring the same tree twice.

| config | B=1 | B=16 | B=64 | B=256 | B=1024 | B=4096 | peak GB @4096 |
|---|---|---|---|---|---|---|---|
| Acrobot before | 22 | 338 | 1,327 | 5,004 | 16,356 | 36,891 | 8.85 |
| Acrobot **after** | **89** | **1,412** | **5,519** | **20,913** | **67,454** | **160,896** | **3.10** |
| CartPole before | 22 | 348 | 1,364 | 5,227 | 17,594 | 42,746 | 7.65 |
| CartPole **after** | **94** | **1,513** | **5,774** | **22,590** | **75,193** | **186,137** | **2.98** |
| MountainCar before | 23 | 362 | 1,416 | 5,349 | 17,866 | 37,640 | 8.64 |
| MountainCar **after** | **87** | **1,365** | **5,322** | **20,241** | **64,492** | **146,514** | **2.98** |
| MountainCarContinuous before | 23 | 362 | 1,417 | 5,345 | 17,922 | 37,631 | 8.64 |
| MountainCarContinuous **after** | **87** | **1,362** | **5,368** | **20,257** | **64,952** | **146,466** | **2.98** |
| Pendulum before | 39 | 609 | 2,377 | 8,812 | 26,112 | 46,639 | 9.14 |
| Pendulum **after** | **91** | **1,451** | **5,662** | **21,558** | **71,199** | **176,078** | **3.09** |

**3.8-4.4x at every batch size**, batch 1 included — the kernel removes launches and passes rather
than amortising them. **Peak VRAM at 4,096 falls from 7.65-9.14 GB to 2.98-3.10 GB**, a 2.9x cut,
because the kernel stores the frame directly: no planar surface, no transient, no interleave.
`BATCH_SCALES` (1,687-1,990x against a 20x bar) and `NO_LARGE_BATCH_REGRESSION` (7.2-8.2 against
0.9) pass in both arms.

The **VRAM figures above come from the sweep shards**, where `bench_gpu_vs_cpu.gpu_throughput`
resets the peak counter per cell. The `peak_gb` recorded in the first committed `paired_gpu` run
does NOT mean the same thing — it was a process high-water mark, so its "after" arm inherited the
"before" arm's peak and says nothing about the kernel. `paired_gpu.py` now resets per arm and
records `peak_gb_is_per_arm`; the claim rests on the sweep either way.

Cumulatively against the pre-Step-6 baseline on a 3090: Acrobot 16,661 -> 160,896 frames/s, **9.7x**.

### Correctness — every gate green

| check | result |
|---|---|
| `tests/gpu/test_gfxdraw_classic.py` (both joints, 4 state classes, per-row style, chunk/order invariance, the CPU-path assertion) | **55 passed** |
| `tests/gpu/test_fast_paths.py -k gfxdraw-classic` (the two new cases: `env.render` with the kernel on vs off) | **10 passed** |
| `tests/gpu/test_gfxdraw_fused.py` (CarRacing's own identity test), Box2D venv | **4 passed** |
| frozen Step 4 replay **under the kernel**, cuda | **5/5**: every check status and every oracle digest, `distinct_colors` and `mean_pixel` **identical to Step 4** |
| `replay_digests.py --device cuda` | **5/5 bit-identical** |
| LunarLander and CarRacing on CUDA, baseline tree vs branch | **byte-identical**, both |

**Pixels changed: no.** Semantic follow-up: **none** — section 15's first case.

### R, settled by pairing (job 6351758, `gpu2264`, RTX 3090)

In job 6349362 R's denominator moved by a factor of two **between the two arms of one job on one
node** — Acrobot 428.2 then 229.8 frames/s, CartPole 488.4 then 246.1, while MountainCar's barely
moved — and the class-R `GPU_FAST` bar of 500 sat inside the resulting range. The reference is a
single-threaded pygame loop on the same node whose CPU is feeding the GPU, and `bench_step3` times
it in a block after the sweep rather than interleaved.

`paired_gpu.py` interleaves it: a reference burst, then a render burst, round after round, in one
process, both arms. The denominator is stable to within 2% across arms once paired (Acrobot 407.2
against 410.6, CartPole 517.9/527.8, MountainCar 483.6/492.5, MCC 490.1/488.2, Pendulum
254.7/253.8), which is what makes the ratio a measurement.

**B = 4,096, the headline batch:**

| config | f/s before | f/s after | x | official f/s | **R before** | **R after** | tier before -> after |
|---|---|---|---|---|---|---|---|
| Acrobot | 37,199 | 160,066 | 4.30x | 410.6 | 91.6 | **389.8** (385-394) | GPU_FASTER -> GPU_FASTER |
| CartPole | 42,854 | 185,436 | 4.33x | 527.8 | 82.3 | **347.0** (341-354) | GPU_FASTER -> GPU_FASTER |
| MountainCar | 37,522 | 144,691 | 3.86x | 492.5 | 77.5 | **295.0** (289-306) | GPU_FASTER -> GPU_FASTER |
| MountainCarContinuous | 37,685 | 145,176 | 3.85x | 488.2 | 76.6 | **296.8** (290-300) | GPU_FASTER -> GPU_FASTER |
| Pendulum | 46,694 | 175,462 | 3.76x | 253.8 | 183.1 | **691.4** (681-699) | GPU_FASTER -> **GPU_FAST** |

At B = 1,024 the same pairing gives R 136-283 after, against 34-103 before: the ratio is still
4.2-4.3x but the absolute R is lower, which is why 4,096 is the headline batch the spec names.

`cores` is 105-163 against a bar of 1, so `BELOW_G4` does not apply and R was the only open
question. **`frames identical: True` for all five between the two arms**, checked inside the timing
job itself rather than inferred from a separate replay.

**So: Pendulum reaches the target tier, and the other four do not.** R 295-390 against 500 is a
shortfall of **1.28-1.69x**, down from the 5.8-6.5x this ticket started at. The honest summary is
that the fused kernel closed most of the gap and not all of it, and the remaining factor is not a
constant to be tuned: at B = 4,096 Acrobot now writes 160,066 x 250,000 x 3 bytes = **120 GB/s** of
frame, which is ~13% of a 3090's peak, so there is headroom — but it is in the per-pixel walk (five
op types with a bounding-box test each, a Wu inversion per edge, two table lookups per disc), not in
the store. Taking it further means fewer ops per pixel, which means per-env specialisation of the
kernel, which is a different and much less general piece of work than this one.

Also settled here: **CarRacing's own fused identity test, 4 passed** in the Box2D venv. In job
6349362 it collected 4 cases and SKIPPED all four for want of Box2D, which the suite's coverage
guard correctly reported as a failure rather than a pass.

### Two things to carry to Step 8 (recorded, not fixed — both off the certification path)

**`PLATE_B` is a `tl.constexpr`.** It is the number of rows in MountainCar's baked terrain plate:
1 when the batch shares one appearance, `B` when it does not. As a constexpr it takes part in
Triton's specialisation key, so **every distinct batch size rendered with more than one distinct
`(background.color, terrain.color)` pair triggers a fresh JIT compile** — a few hundred milliseconds
each, once per (batch size, style-diversity) pair per process. It is a constexpr because as a
runtime scalar both the modulo that picks the plate row and the int64 widening beside it are
mixed-type expressions whose lowering varies by Triton version, and the kernel had not run on a
device when that choice was made. Now that it has, the runtime form can be tried and measured.
Nothing on the certification path hits this: every sweep, replay and parity run uses one
appearance, so `PLATE_B` is 1 and there is one specialisation.

**`plate.contiguous()` copies `(3, B, H, W)` per render** when the plate is per-environment
(`render_classic`, the `plate.contiguous() if plate is not None` argument). `_terrain_base` builds
it with `index_select(...).transpose(0, 1)`, which is not contiguous, so the copy is real: **2.9 GB
of transient at B = 4,096** on a 500x500 frame, on top of the 3.0 GB the kernel already peaks at.
Again not on the certification path — a shared appearance takes the `(3,1,H,W)` broadcast branch,
which is already contiguous — but a caller rendering 4,096 environments with per-row terrain would
double the kernel's memory. The fix is to build the gathered plate contiguous in `_terrain_base`
rather than to copy it at the call site, which is a change to MountainCar's bake and wants its own
replay.

### The one thing that is decided already

`_use_fused` is CUDA-and-Triton only, and `tests/gpu/test_gfxdraw_classic.py` asserts it: a
`device="cpu"` render takes the per-primitive path by construction, so **the CPU path's pixels and
the CPU tiers in this document are untouched by GFXDRAW-2** whatever the kernel measures.
