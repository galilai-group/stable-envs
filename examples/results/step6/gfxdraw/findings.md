# Step 6 — `gfxdraw` (classic control): **PATCH**

- **baseline** `89fa5ca81341dc4df7186ec325cb691431a2ef17` (`sami/pre-step6-baseline`)
- **branch** `step6/gfxdraw`
- **affected configurations** (all replayed): `Acrobot`, `CartPole`, `MountainCar`,
  `MountainCarContinuous`, `Pendulum`. Other consumers of the same primitive layer, checked but not
  in the frozen corpus: `LunarLander`, `CarRacing`.
- **pixels changed: NO.** All five replay bit-identical to the frozen Step 4 oracle on CUDA, and
  the fused-path identity test is green.
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

## Correctness

| check | result |
|---|---|
| `examples/fidelity_step4.py --configs <all five> --device cuda` | 5/5 `measured`; every mandatory and family-specific check has the SAME status as Step 4, and every oracle digest (`digest_ours`, `digest_ours_default_style`, `digest_ours_corpus_style`, `digest_official`), `distinct_colors` and `mean_pixel` is unchanged |
| our frames vs the OFFICIAL pygame renderer on the same host | identical on all five, CUDA |
| `tests/gpu/test_gfxdraw_fused.py` + `tests/gpu/test_fast_paths.py` (3090) | 123 passed, 166 skipped, 0 failed |
| `tests/gpu/test_conformance.py` + `test_reproducibility.py`, `-k` the five (3090) | 451 selected: **358 passed, 93 skipped, 0 failed** |
| `stable_envs/testing/conformance` mandatory + family checks, CPU, node2338 | 5/5 OK (batch invariance, batch-size invariance, chunking, order independence, render purity, style-as-state, per-env variation, backend agreement, colours apply exactly, visual fidelity, fidelity under variation) |
| `tests/unit` on Oscar | 2 failed, 504 passed. Both failures are **pre-existing at 89fa5ca** and unrelated: `test_examples[robocasa_visual/render_mjwarp.py]` and `test_every_cited_scratch_path_resolves` (it names `scratch/carracing/logs/`, `scratch/robocasa/out/corpus` and `scratch/step6/report.py`; the same three, before and after) |
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

**CPU_FAST needs `S_cpu >= 5` AND `cores_cpu >= 1`, and `cores_cpu` is the binding one.** On
node2338 a 500x500 JPEG-q90 frame decodes at ~850-1,180 frames/s on one core, and these renderers
now run at 258-460. The interleave alone is the wall: producing the caller's `(B,H,W,3)` uint8 from
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
| `se6-gfxdraw-paired` | 6346954 | — | cancelled by me before it ran, to add the measured-tree provenance |

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
