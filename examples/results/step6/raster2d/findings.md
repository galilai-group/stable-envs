# Step 6E — `raster2d` / Scene2D CPU

**`PATCH`.** One correctness commit that makes the CPU path reachable at all on the certification
cluster, and one structural commit that removes the largest measured bottleneck in it. Pixels are
unchanged everywhere, held by digest rather than by argument. `CPU_FAST` is **not** reached; the
remaining bottleneck is isolated, named and measured below, which is this workstream's other
stop condition.

| | |
|---|---|
| baseline | `89fa5ca81341dc4df7186ec325cb691431a2ef17` on `sami/pre-step6-baseline` |
| branch | `step6/raster2d` (HEAD when this was written: `d1f481b65e07`; this file and the
  artifact directory are committed after it, so the tip is one or two commits later) |
| commits | `80b962c` shared harness · `e6c2af3` correctness · `dd67eb2` performance · `54dd65f`, `89408cf`, `755f2d4`, `def0893` measurement scripts |
| configs patched | `PushT`, `SimplePointMaze` |
| configs profiled, diagnosed, NOT patched | `TwoRoom`, `TwoRoomTarget`, `Piecewise`, `PiecewiseZones` — these are `swm_nav` torch arithmetic, not Scene2D, and need their own fix (below) |
| other live consumers of the same file | `Highway`, `Intersection`, `Merge`, `Parking`, `Racetrack`, `Roundabout` — all six replay byte-identical |
| CPU hardware | Oscar `batch`, `node2339`/`node2340`/`node2342`/`node2344`, Intel Xeon Platinum 8268 @ 2.90 GHz, 1 torch thread, `OMP_NUM_THREADS=1` |
| GPU hardware | RTX 3090 on `gpu2105`, driver 575.57.08 — before and after in the same job on the same card |
| venvs | `se-step6` (profile, pytest, fuzz) · `se-step5` (stable_worldmodel + matplotlib references) · `se-step3` (gym_pusht reference) — all three carry torch 2.14.0+cu126 and triton 3.8.0 |
| SLURM jobs | `6344858` profile-before (wrote PushT, then cancelled: see below) · `6346289` profile-before, the rest · `6346290` profile-after · `6346291` CPU certification sweep, before+after · `6346300` GPU: fuzz, replay, tests, sweep before+after · `6346490` PushT profile-before, remeasured |

`6344858` was cancelled at `SimplePointMaze`, batch 256. One call at the baseline is 72 s at batch
64 and the cell projected past four hours for the ten repetitions and three warm-ups, which would
have bought a number the batch-64 cell already gives and cost the other five envs their notes to the
wall clock. `SimplePointMaze` is therefore profiled to batch 64 on BOTH sides; the certification
sweep, which does have to reach the frozen batches, caps each cell by projection instead and records
the projection in place of the missing cell. `6346490` was cancelled once its two artifacts were
written, to drop a duplicate digest run that `6346289` also performs.

---

## 1. Correctness: the CPU path did not exist on any host with Triton (commit `e6c2af3`)

This is finding #1 of `examples/results/step6/findings.md`, confirmed and fixed.

`render()` chose its backend from `HAVE_TRITON`, which is a property of the **interpreter**, and never
asked where the scene lived. On a host where `import triton` succeeds, a `device="cpu"` scene was
routed into `render_triton`, whose first act is `with torch.cuda.device(scene.kind.device)` — so every
CPU render of a Scene2D env raised `ValueError: Expected a cuda device, but got: cpu`.

- Step 3 recorded exactly this: `PushT` and `SimplePointMaze` on `node2349` are
  `CPU_UNMEASURED · blocked: ValueError: Expected a cuda device, but got: cpu`
  (`docs/research/step3_benchmarks.md:135,137`). Their CPU tiers came from the laptop, which has no
  Triton — they could not have been produced where they are certified.
- All three Oscar venvs used here carry triton 3.8.0, so this was not one unlucky interpreter.

The predicate now reads the scene's device: torch unless Triton is importable **and** the scene is on
CUDA. An explicit `backend="triton"` on a CPU scene raises a message naming the real cause.

**A second instance of the same defect, in a file this workstream does not own.**
`stable_envs/testing/conformance.py:381` guards `check_backend_agreement` with `if not HAVE_TRITON`
and then calls `render_triton(sc, ...)` unconditionally — so running the Step 4 conformance suite with
`device="cpu"` on a Triton host fails that check for the same reason. Not edited; it is shared
regression machinery. Coordinator's call.

## 2. Profile-before: where the time actually went

`scratch/step6/profile_note.py` for the public path, plus `scratch/step6/raster2d/attribute.py` for
the phases an operator trace cannot separate (a grid subtraction and a disc-centre subtraction are
both `aten::sub`). `attribute.py` times a **replica** of `render_torch`'s body and digest-checks it
against the real function on every run; `replica_matches_render_torch` is `true` in every artifact
here, so the phases describe the real code shape.

Artifacts: `profile_before/`, `profile_after/` (JSON + MD, host-tagged).

### Scene2D, frames per second, Oscar batch partition, 1 thread, frozen Step 4 states

| config | tree | host | B=1 | B=16 | B=64 | B=256 |
|---|---|---|---|---|---|---|
| PushT (96 px) | before | `node2339` | 66.8 | 128.3 | 114.3 | 98.1 |
| PushT | **after** | `node2342` | **151.5** | **513.0** | **472.1** | **454.2** |
| SimplePointMaze (500 px) | before | `node2340` | 3.3 | 2.4 | 0.9 | — |
| SimplePointMaze | **after** | `node2342` | **91.1** | **118.8** | **119.4** | — |

### The mechanism, measured

`render_torch` evaluated **all three primitive families over the whole frame for every slot**, then
threw away the two that could not apply. PushT's scene is 4 polygons, 4 capsules and 1 disc; at B=64
the before-attribution (`attribution_PushT_cpu_node2339.json`, raster 557.24 ms) splits as

| block | ms | slots that are this kind | ms that cannot reach a pixel |
|---|---|---|---|
| polygon | 398.9 | 4 of 9 | 221.6 |
| capsule | 75.2 | 4 of 9 | 41.8 |
| disc | 68.8 | 1 of 9 | 61.2 |
| grid / canvas / output | 4.4 | — | — |

**58% of the raster was primitive families the slot does not have.** The rest was mostly frame it
could not touch: the same artifact records each primitive's world-space bounding box as a fraction of
the frame — mean **1.7%** at B=1, 0.155 frames summed over all nine primitives against the 9.0 frames
the loop actually evaluated.

`SimplePointMaze` is the same disease at 27x the pixel count: 12 polygons, 1 disc, 1 disabled slot,
**zero** capsules, and the capsule block ran 13 times per frame anyway. 324.75 ms for one frame at
B=1 (`6344858`, `node2351`); 72.4 s for a batch of 64 (`node2340`). Its 13 primitives cover 1.7% of
the frame each. Its before-attribution at B=64 (raster 74,779 ms) splits as

| block | ms | slots that are this kind | ms that cannot reach a pixel |
|---|---|---|---|
| polygon | 55,463 | 12 of 13 | 4,266 |
| capsule | 9,508 | **0** of 13 | 9,508 |
| disc | 6,428 | 1 of 13 | 5,933 |
| canvas + output conversion | 214 | — | — |

— 26% of it evaluating primitive families the scene does not contain at all, and the other 74%
mostly evaluating polygons over the 98% of the frame they cannot cover.

This is finding #2 of the pre-implementation notes — "each primitive is evaluated over the whole
frame" — confirmed, and with a second cause beside it that the note did not have: the primitive is
also evaluated as all three things it is not.

### A hypothesis the profile killed

Finding #2 proposed pre-baking `SimplePointMaze`'s static maze, and its `static_prefix` is already
declared (6 of 13). Measured, the bake is the wrong fix on CPU: the six walls' bounding boxes cover
0.216 frames in total, so drawing them bounded is cheaper than the one full-frame copy that painting
a baked plate into the batch would cost. No cache was added, which is also why render purity needed
no argument.

## 3. The patch (commit `dd67eb2`)

One pre-pass, `_prim_plan`, and four things it lets the loop skip:

1. **primitive families no environment has in that slot.** `kind` is per environment, so this is
   measured, not assumed; a slot disabled in every row is skipped outright.
2. **the polygon outline where no row has `line_w > 0`, and the inner disc where no row has
   `inner_r > 0`.** The outline is the expensive half of a polygon — the inside test is one dot
   product per edge, the outline is a projection, a clamp, a rejection vector and a squared length.
3. **polygon edges degenerate in every row** (padding to `V` by repeating the last vertex), and the
   `torch.where` sentinel on the edges where no row is degenerate.
4. **every pixel outside the primitive's own world-space bound**, unioned over the batch.

The bound is the encoding `_bbox_kernel` already uses on the GPU, special cases included — an empty
box for `kind == 0`, an infinite one for a polygon whose shoelace is exactly zero (no interior, so
its halfplane intersection is unbounded). Using the same rule on both devices means the repo has one
culling rule, the one `tests/gpu/test_fast_paths.py` already holds against every registered env,
rather than a second one nothing tests. The shoelace is evaluated in float64 **and** in the geom
dtype, and either zero marks the primitive uncullable, so the CPU can never call bounded something
the kernel calls unbounded.

Second half: with `supersample == 1` nothing between a primitive's colour and the output is an
average, so `round -> clamp -> uint8` applies to the `(B,3)` colours once instead of to a
`B x 3 x Rh x Rw` float canvas at the end, and the canvas is the output's own `(B,Rh,Rw,3)` layout.
That removes the float canvas (one byte a channel instead of four), the round, the clamp, the cast
and the final permute+copy — five passes over the frame stack. `ss > 1` genuinely averages and keeps
the float canvas and the conversion it always had.

**No cache, no pre-bake, no state.** The plan is a pure function of the scene handed in and is
recomputed every call, so render purity and batch/order invariance hold by construction rather than
by policy. The window being a union over the batch is a throughput property only: a window larger
than necessary evaluates pixels whose mask then rejects them.

## 4. Pixels: unchanged, four independent ways

| evidence | result |
|---|---|
| frozen Step 4 replay, all six configs, on cuda where the frozen digests were taken (`replay/raster2d_swm.json`, `replay/raster2d_pusht.json`, job `6346300`) | `digest_ours`, `digest_ours_default_style`, `digest_ours_corpus_style` and `digest_official` **all identical**; all 17 conformance checks per config identical in status |
| CPU digests of the six configs, before and after, on Oscar and on the laptop (`cpu_digest_before.json` `node2340`, `cpu_digest_after.json` `node2342`) | all six identical across the patch, and identical to the laptop's arm run of both trees; five of the six also equal the frozen cuda digest |
| every registered Scene2D consumer on CPU — PushT, SimplePointMaze and the six highway-env envs (130 to 604 primitives, outlines, padded vertices) | 8 of 8 digests identical |
| differential fuzz against a verbatim copy of the 89fa5ca8 implementation (`scratch/step6/raster2d/fuzz_render_torch.py`) | byte-identical in every case. The counts are what the runs printed: **400 + 5x250 = 1,650 on CPU** at the time of this commit and **400 + 250 = 650 on the 3090** in job `6346300`, whose log carries both `400/400 identical` and `250/250 identical`. RASTER2D-2 re-ran it at 8x300 = 2,400 on CPU. Cases: collinear polygons, all-degenerate edges, padded triangles, negative radii, rings larger than their disc, primitives off the frame and larger than the world, `flip_y`, non-square worlds and outputs, supersample 2 |

Batch invariance, batch-size invariance, order invariance and render purity are re-checked in
`cpu_digest_*.json` for all six configs: `b1_matches`, `order_invariant`, `pure` all true.

## 5. Tests

- `tests/unit`, local, 487 passed / 6 skipped / 2 failed. Both failures are pre-existing at
  `89fa5ca8` and unrelated to this workstream: `test_every_cited_scratch_path_resolves` names three
  probe paths that the docs cite and this worktree does not contain (the Step 6 report generator, a
  carracing log directory and a robocasa corpus), and `robocasa_visual/render_mjwarp.py --help`
  cannot import its stack here. Verified by `git stash`: both fail identically at the baseline. Their
  paths are deliberately NOT spelled out above — this file is inside that test's search roots, and
  quoting a dead citation adds a site to it.
- `tests/gpu/test_conformance.py`, `test_reproducibility.py`, `test_fast_paths.py` restricted to the
  six envs, on the 3090 (job `6346300`): **183 passed, 75 skipped, 0 failed**. The skips are the
  `reference`-marked tests whose official renderers are not in the pytest interpreter.
  `test_backend_agreement` — the fused kernel against this very function — is among the passes.

## 6. Sweeps

### CPU certification, 1 thread, `batch` partition, `node2344`, before and after in one job (`6346291`)

Two shards, one per interpreter, exactly as Step 3 did it, because no Oscar interpreter holds every
official renderer. `--states-out` / `--states-in` so both halves render the same states.

| config | best B | before f/s | after f/s | speed-up | official f/s (before/after run) | `S_cpu` before | `S_cpu` after | `cores_cpu` after | tier |
|---|---|---|---|---|---|---|---|---|---|
| PushT | 16 | 115 | **590** | **5.1x** | 337 / 395 | 0.34 | **1.50** | 0.08 | CPU_REGRESSION -> **CPU_PARITY** |
| SimplePointMaze | 64 | 3 | **112** | **41.2x** | 48 / 51 | 0.06 | **2.21** | 0.13 | CPU_REGRESSION -> **CPU_FASTER** |
| TwoRoom | 16 | 235 | 269 | 1.1x | 298 / 308 | 0.79 | 0.87 | 0.09 | CPU_REGRESSION (unchanged) |
| TwoRoomTarget | 16 | 186 | 187 | 1.0x | 185 / 190 | 1.01 | 0.98 | 0.06 | CPU_PARITY/CPU_REGRESSION — see below |
| Piecewise | 16 | 415 | 417 | 1.0x | 352 / 362 | 1.18 | 1.15 | 0.11 | CPU_PARITY (unchanged) |
| PiecewiseZones | 64 | 223 | 226 | 1.0x | 250 / 258 | 0.89 | 0.87 | 0.06 | CPU_REGRESSION (unchanged) |

**The four `swm_nav` rows are the same source in both trees** — this patch does not touch them — so
their before/after spread is this sweep's noise floor, and it is ±5% on our renderer and ±3% on the
official one. `TwoRoomTarget` crossing the `S_cpu = 1` line on 1.01 -> 0.98 is that noise, not a
change; it is quoted rather than rounded away because the alternative is a tier that moves when
nothing did.

`SimplePointMaze`'s before row only reached B=16: B=64 and above were skipped by the per-cell budget
at *"projected 30 s per repetition at the 2 frames/s already measured"*. The after row reaches
B=1024.

**`CPU_FAST` is not reached.** It needs `S_cpu >= 5` **and** `cores_cpu >= 1`, and `cores_cpu` is the
harder half by a long way: one-core JPEG-q90 decode of these frames is 7,535 f/s at 96 px and 876 f/s
at 500 px, so PushT would have to reach 7,535 f/s single-threaded — 0.13 ms per 96x96 frame including
scene construction — and `SimplePointMaze` 876 f/s at 500 px, which is 1.3 GB/s of output alone.
`cores_cpu` moved 0.02 -> 0.08 (PushT) and 0.00 -> 0.13 (SimplePointMaze); the frozen ladder's
`BELOW_G4` qualifier still applies to every row in this workstream, as it does to every measured CPU
row in the project.

### GPU, before and after on the same card in the same job (`6346300`, RTX 3090, `gpu2105`)

| config | before best f/s | after best f/s | ratio |
|---|---|---|---|
| PushT | 1,933,431 | 1,926,247 | 0.996 |
| SimplePointMaze | 43,682 | 43,317 | 0.992 |
| TwoRoom | 39,628 | 39,562 | 0.998 |
| TwoRoomTarget | 23,784 | 23,736 | 0.998 |
| Piecewise | 54,750 | 54,661 | 0.998 |
| PiecewiseZones | 47,038 | 47,029 | 1.000 |

`render_triton` is untouched; this is the measurement that says so. No healthy GPU row regressed.

## 7. What is left, isolated

### 7a. The remaining Scene2D bottleneck: a per-environment primitive's union window is the frame

The window is a union over the batch because one torch op covers the whole batch. A primitive that is
the same in every environment keeps a tight window at any batch size; one whose position is part of
the state widens to whatever the batch spans, and for a corpus of reachable states that is the whole
frame. After the patch, at B=64 (`attribution_*_cpu_node2342.json`):

| config | raster ms | the moving primitives | share |
|---|---|---|---|
| SimplePointMaze | 518.5 | agent disc, 301.0 ms | **58%** |
| PushT | 143.1 | the two block quads, 104.4 ms | **73%** |

and the bounding-box coverage confirms the cause rather than inferring it: summed over all
primitives, 0.218 frames at B=1 against 1.216 at B=64 for `SimplePointMaze`, 0.155 against 2.491 for
`PushT`.

Measured on the real agent positions of the frozen corpus
(`scratch/step6/raster2d/window_experiment.py --mode chunk`, output in
`experiments/window_chunk.txt`), splitting the batch into chunks and taking a window per chunk
collapses that cost — 93.7 ms at one chunk of 64, 82.1 at chunks of 8, 58.8 at 4, 25.5 at 2 and
**2.5 ms at 1** — but chunk-of-1 is a Python loop over the batch, which is the first structural
problem on section 7's own list and would be `O(B)` dispatch at B=4096. The shape that keeps it
batched is the one the GPU already uses: bin primitives against **screen tiles** and, per tile,
select the rows whose box reaches it, which is flat in the batch.

**This was done: see §9 (RASTER2D-2).**

Beside it, at B=64 and 500 px, the uint8 canvas fill is 69.9 ms (13%) and is irreducible — it is the
output — and the `_prim_plan` pre-pass is 1.6 ms, flat in the batch (24% of the raster at B=1, 1% at
B=64).

### 7b. `TwoRoom` / `Piecewise` are a different path and need a different fix

They are not Scene2D — `swm_nav` overrides `render()` with per-pixel torch arithmetic, and
`scene()` raises to say so. They were profiled anyway, per the brief. The structural problem there is
real but **not the same one**, and the patch in this ticket cannot help them:

| config | render ms @ B=64 | `base_image` | share | state-independent? |
|---|---|---|---|---|
| TwoRoom | 257.3 | 127.1 | 49% | yes (measured) |
| TwoRoomTarget | 375.3 | 119.6 | 32% | yes |
| Piecewise | 183.8 | 42.4 | 23% | yes |

`base_image` — background fill, wall stripes, doors, border — is a pure function of the style, is
rebuilt from scratch on every call, and is materialised at `(k, 224, 224, 3)` per environment even
when the style is batch-uniform. That is section 7's "repeated construction of static scene data",
and the fix is the pre-bake that `SimplePointMaze` did *not* need: build it for the distinct style
rows only and expand.

**The dot's support was claimed unbounded here, and that was wrong.** The first version of this
section said "an alpha of 2e-9 still moves a pixel", concluded the Gaussian had unbounded support and
ruled bounding out. Measured instead against `swm_nav._alpha_blend` itself
(`scratch/step6/raster2d/alpha_bound.py`, output in `experiments/alpha_bound.txt`), the smallest
alpha that changes any pixel over every uint8 contrast is **2.98e-8 = 2^-25**, and it is reached at
`img=1, colour=0` — not because 1/255 has been crossed but because `img * (1 - a)` goes through
`1 - a`, which rounds to exactly 1.0 in float32 for every `a` below the spacing at 1.0. The two
directions are not symmetric: painting a bright colour onto a dark pixel needs `a >= 1/255`, and
darkening a bright one is four orders of magnitude more sensitive. A bound has to hold for the
easiest pixel to move, so 2^-25 is the number, and the corresponding radius of a normalised Gaussian
is **5.89 sigma** — 41 px at the smallest declared `agent.radius` and 82 px at the largest, against a
224 px frame. The dot IS boundable.

Two things the bound has to carry, and both are in the implementation rather than in this paragraph:
the normalisation divides by the frame's own maximum, so the threshold radius is measured from the
peak and the peak is the sample nearest the centre (half a pixel, which the formula charges for);
and if a caller passes a state whose agent is outside the frame, the in-frame maximum is no longer
next to the centre, so the margin has to grow by that distance. See §10.

### 7c. Which tree each leg measured, proved rather than assumed

The `raycast3d` workstream reported that a script doing `sys.path.insert(0, <its own root>)`
overrides `$PYTHONPATH`, so a before/after job that switches `PYTHONPATH` between two checkouts can
silently measure one tree twice. Checked here:

- `scratch/step6/profile_note.py` inserts nothing; `attribute.py`, `cpu_digest.py` and
  `fuzz_render_torch.py` insert the root derived from their own `__file__`, and every leg runs the
  copy of the script that lives **inside the tree it is measuring** (`cd $TREE`, `$TREE/examples/...`
  for `bench_step3.py`), so the inserted root and the intended tree are the same path;
- confirmed by import under each leg's exact invocation:
  `…/se6/raster2d-preB/stable_envs/__init__.py`, `_prim_plan` absent, against
  `…/se6/raster2d/stable_envs/__init__.py`, `_prim_plan` present;
- the artifacts already carry the proof independently. Every before attribution records
  `"replica": "replica"` and every after one `"replica": "replica_v2"`, and which replica runs is
  decided by `hasattr(raster2d, "_prim_plan")` — a before note could not say `replica` if it had
  imported the patched tree;
- and the sweep's own control says the same thing: the four `swm_nav` rows, whose source is
  byte-identical in both trees, moved by ≤5%, while the two Scene2D rows moved 5.1x and 41.2x. One
  tree measured twice cannot produce that.

The three scripts now also record `stable_envs.__file__` (and `attribute.py` records
`has_prim_plan`) in every note they write, so a future artifact states which code produced it rather
than leaving it to the invocation. The artifacts in this directory predate that field and are
covered by the four points above.

### 7d. Two defects in shared tooling, found by using it

- `scratch/step6/profile_note.py` wrote `n["host"]` in `main()`, where `n` is `render_note`'s
  parameter and is unbound. Every run profiled, printed and then raised `NameError` **before writing
  any file**. Fixed in its own commit (`80b962c`) at the coordinator's request; every workstream
  needs it.
- `rsync --delete` from the Mac to `$SCR/se6/<ws>/` deletes the results a running job has just
  written there, because the worktree does not contain them. It cost PushT's first Oscar
  profile-before, which was remeasured (`6346490`). Now documented in the sbatch header; the safe
  form is `--exclude examples/results/step6/<ws>` or pull-before-push.

## 8. Semantic follow-up: none required

Every affected frozen frame is digest-identical, on cuda against the frozen Step 4 oracle and on CPU
against the pre-patch renderer, so section 13's first branch applies: no representation tripwire, no
semantic smoke, no paired Gate H.

Saying this explicitly for `PushT`, because it is the representative path of the Step 5 parity gate
(PushT + LeWM: official 88.0%, stable-envs 87.7%, paired delta -0.33 pp, 95% CI [-3.47, 2.80]): its
`digest_ours` on cuda is `cecda14a…`, unchanged, and its GPU renderer was not modified at all — the
patch is in `render_torch`, which the default cuda path does not call. There is nothing for a
downstream rerun to detect.


---

# Round 2 — the three authorized follow-ups

`1103326` went to independent review as `APPROVE-WITH-CHANGES`. What follows is on top of it, on
`step6/raster2d` after `git merge step6/integration` (a merge, not a rebase; the one conflict was
`profile_note.py`, where both branches had made the same one-line fix and only this branch carried
the comment explaining it).

## 8. RASTER2D-1c — the second instance of the device-dispatch defect (`48c221a`)

`check_backend_agreement` guarded on `HAVE_TRITON` and then called `render_triton` unconditionally,
so on a Triton host a `device="cpu"` env died with the same `ValueError: Expected a cuda device` and
took its conformance row with it. Reproduced on a host without Triton by forcing
`raster2d.HAVE_TRITON = True`, which is what makes the old predicate fire: PushT and SimplePointMaze
both raise before the commit and both return a skip naming the device after it.

It is a defect rather than a slow path. A CPU env has ONE backend, so the question the check asks
does not arise, and "not applicable" is the same answer, in the same shape, that the tilemap branch
beside it already returns for the other reason a Scene2D comparison cannot be made. On cuda nothing
changes.

## 9. RASTER2D-2 — row grouping (`c376f34`)

The plan from §7a, implemented. Per-row boxes of a moving primitive are all the same SIZE and differ
only in ORIGIN; quantise the origin onto a grid of that size and each cell's rows share a window at
most two cells across. The cell count is bounded by the FRAME, so the loop is flat in the batch:
SimplePointMaze's disc has **64 groups at batch 64 and the same 64 at batch 1024**, against a union
window costing 64 and 1024 frames.

**Both constants of the cost model are measured, and the first guess at one of them was wrong by 5x
and silently refused the grouping on every scene that wanted it.** One group is 56-68 us of
dispatch, flat in the batch, timed on a scene built so the grouped pixel work is fixed and only the
group count varies; a window pixel is 6-20 ns; so a group buys itself at 10,000 window pixels. The
accept margin is 1.25. Scanned against wall time over the frozen corpora and a synthetic worst case
of many groups and almost no pixel work, that pair is never slower than not grouping. Experiments and
their output: `scratch/step6/raster2d/window_experiment.py`, `experiments/window_*.txt`.

Interleaved in one process against the `render_torch` at `1103326`, one thread, frozen states:

| config | B=1 | B=16 | B=64 | B=256 |
|---|---|---|---|---|
| PushT | 1.01x | 0.97x | **2.08x** | **3.10x** |
| SimplePointMaze | 0.99x | **2.46x** | **2.44x** | **2.43x** |

Pixels identical: 2,400 random scenes against the verbatim 89fa5ca8 implementation, the six frozen
CPU digests, and all eight registered Scene2D consumers.

**The interface, for gfxdraw.** `raster2d._row_groups(box, r0, r1, c0, c1)` takes a `(B,4)` int64
array of per-row pixel boxes and the union window already computed from them, and returns a list of
`(rows, r0, r1, c0, c1)` — `rows is None` meaning the whole batch through a view, otherwise an index
array. It has no Scene2D in it and nothing 2-D-primitive-specific; `swm_nav` already uses it for its
Gaussian dots (§10). If `gfxdraw` wants the same partition it can call it as-is. **`gfxdraw.py` was
not touched** — that workstream is mid-GFXDRAW-2.

## 10. SWMNAV-1 — bake the base image, bound the dot (`d862e2a`)

Two changes on the `swm_nav` path, which is not Scene2D and needed its own fix.

**The bake.** The base image is a pure function of the style; rows are grouped by the bytes of their
style row and each distinct row is built once. The key is the WHOLE row rather than the leaves an env
happens to read — a key listing the relevant leaves would go stale the day somebody adds a leaf that
feeds the base image and forgets to declare it, and a wrong key is a stale picture. Keying on
everything can only cost a miss.

**The dot window, and the claim this findings document got wrong the first time.** §7b said "an alpha
of 2e-9 still moves a pixel" and concluded the Gaussian could not be cropped. Measured against
`_alpha_blend` itself, the smallest alpha that changes any pixel over every uint8 contrast is
**2.98e-8 = 2^-25**, at `img=1, colour=0` — `img * (1 - a)` goes through `1 - a`, which rounds to
exactly 1.0 in float32 below the spacing at 1.0, and that is four orders of magnitude more sensitive
than the 1/255 the other direction needs. Below that alpha the blend is the identity, so the dot is
boundable at **5.89 sigma**.

The subtlety is the normalisation. `_gaussian_dot` divides by the frame's own maximum, so for a
centre outside the frame — which `render(states)` permits, since only `step` clamps — the whole dot
is rescaled until its largest in-frame sample is 1.0, and a window derived from the un-normalised
tail alone would crop a dot painting at full strength. The window therefore grows by the distance
from the centre to the nearest in-frame sample, and it always contains that sample, which is what
lets the maximum be taken over the window and still be the frame's.

Against the full-frame composition it replaces, one thread, frozen states, pixels identical:

| config | B=1 | B=16 | B=64 | B=256 |
|---|---|---|---|---|
| TwoRoom | 8.81x | 7.55x | **6.88x** | 7.08x |
| TwoRoomTarget | 6.31x | 5.39x | **4.10x** | 4.28x |
| Piecewise | 4.36x | 3.78x | **3.08x** | 3.24x |
| PiecewiseZones | 11.99x | 9.86x | **8.43x** | 8.95x |

## 11. The regression tests the review asked for

`tests/unit/test_raster2d_torch_backend.py` and `tests/unit/test_swm_nav_render.py`, both CPU-only,
both differential against the arithmetic they replaced, and both mutation-checked rather than
assumed green:

| mutation | caught by |
|---|---|
| delete the exactly-zero shoelace collinear cull | 7 raster2d cases |
| shrink a window by one pixel | 2 raster2d cases |
| widen `_ALPHA_FLOOR` to 1e-3 | 33 swm_nav cases |
| drop the out-of-frame margin from the dot window | 1 swm_nav case |
| restore the `contiguous()` that aliased the cached plate | 20 swm_nav cases |
| key every style row on row 0 | 4 swm_nav cases |

The raster2d oracle is 89fa5ca8's `render_torch`, loaded from `fuzz_render_torch.py` so there is one
copy rather than two that can drift, and it **stays** 89fa5ca8's under RASTER2D-2 — comparing the
current renderer against itself would pass no matter what it drew.

Two of those mutations are bugs this work actually had, not hypotheticals. The `contiguous()` one
shipped for about twenty minutes: `expand` to a tensor's own shape returns it, `contiguous()` on
something already contiguous returns it again, so at batch 1 the agent was painted into the cached
plate and every later render of that style came back with a stale dot. The order-invariance and
purity columns of `cpu_digest.py` went False and that is how it was found.

## 12. Provenance of a sweep

`examples/bench_step3.py` now records `stable_envs.__file__` and the tree root in every shard and
resolves the SHA through `STABLE_ENVS_GIT_COMMIT` or a `GIT_COMMIT` file, the way `fidelity_step4`
already does — the measured copy on the cluster is an rsync without `.git`, so every sweep artifact
before this recorded `git_commit: null`, and a before/after that switches `PYTHONPATH` between two
checkouts could not otherwise prove which tree each leg timed.

## 13. Round-2 verification on Oscar

Jobs `6349273` (CPU sweep, `node1828`, Xeon Platinum 8268, 1 thread, before and after in one job on
one node) and `6349274` (GPU, RTX 3090, before and after on one card).

### CPU certification, 1 thread — before is `1103326`, the approved PATCH

| config | best B | before f/s | after f/s | speed-up | official f/s | `S_cpu` before | `S_cpu` after | `cores_cpu` after | tier |
|---|---|---|---|---|---|---|---|---|---|
| PushT | 1024 | 621 | **2,046** | 3.3x | 365 / 373 | 1.70 | **5.49** | 0.29 | CPU_PARITY -> `S_cpu` bar cleared |
| SimplePointMaze | 256 | 114 | **243** | 2.1x | 47 / 52 | 2.42 | **4.69** | 0.24 | CPU_FASTER |
| TwoRoom | 16 | 261 | **1,623** | 6.2x | 290 / 320 | 0.90 | **5.08** | 0.55 | CPU_REGRESSION -> `S_cpu` bar cleared |
| TwoRoomTarget | 16 | 189 | **731** | 3.9x | 200 / 200 | 0.95 | **3.66** | 0.22 | CPU_REGRESSION -> CPU_FASTER |
| Piecewise | 16 | 425 | **1,435** | 3.4x | 376 / 378 | 1.13 | **3.79** | 0.41 | CPU_PARITY -> CPU_FASTER |
| PiecewiseZones | 16 | 215 | **1,401** | 6.5x | 266 / 265 | 0.81 | **5.28** | 0.51 | CPU_REGRESSION -> `S_cpu` bar cleared |

**Against the Step-5 baseline** — round 1 and round 2 composed, same partition, same CPU model:
PushT **115 -> 2,046 f/s (17.8x)**, SimplePointMaze **3 -> 243 (81x)**, TwoRoom 235 -> 1,623 (6.9x),
TwoRoomTarget 186 -> 731 (3.9x), Piecewise 415 -> 1,435 (3.5x), PiecewiseZones 223 -> 1,401 (6.3x).
Every row that was `CPU_REGRESSION` has left it.

**`CPU_FAST` is still not awarded, and the reason has changed.** Three rows now clear `S_cpu >= 5`
(PushT 5.49, PiecewiseZones 5.28, TwoRoom 5.08) where before none reached 2. The tier is a
conjunction, and `cores_cpu >= 1` is the half that does not fall: one-core JPEG-q90 decode of these
frames is 7,015 f/s at 96 px, 2,773-3,536 f/s at 224 px and 992 f/s at 500 px, and `cores_cpu` has
moved from 0.02-0.13 to 0.22-0.55. The frozen `BELOW_G4` qualifier caps every row in this workstream
at `CPU_FASTER` regardless of `S_cpu`, as it does every measured CPU row in the project. Going
further is a different argument from this one: at 224 px and one thread, 2,773 f/s is 417 MB/s of
uint8 output, so the remaining factor of two is bandwidth on the write, not arithmetic.

### GPU, before and after on the same card (`6349274`)

| config | before best f/s | after best f/s | ratio |
|---|---|---|---|
| PushT | 1,937,192 | 1,917,702 | 0.99 |
| SimplePointMaze | 43,201 | 42,724 | 0.99 |
| TwoRoom | 39,629 | **55,845** | **1.41** |
| TwoRoomTarget | 23,818 | **26,989** | **1.13** |
| Piecewise | 54,831 | 53,823 | 0.98 |
| PiecewiseZones | 46,993 | **53,730** | **1.14** |

**This table was best-batch only and it was wrong to stop there — see §15.** Per batch, the same
artifact shows TwoRoom 0.48x at B=16 and Piecewise 0.19x. SWMNAV-1 was held at review on exactly
that, correctly.

### Correctness, round 2

| evidence | result |
|---|---|
| frozen Step 4 replay, all six configs on cuda (`replay_r2/`) | `digest_ours`, `digest_ours_default_style`, `digest_ours_corpus_style`, `digest_official` all identical; every check unchanged in status |
| differential fuzz on the 3090 against the 89fa5ca8 function | 650/650 identical |
| `tests/unit/test_raster2d_torch_backend.py` + `test_swm_nav_render.py` on Oscar | 56 passed |
| targeted GPU suite for the six envs | 183 passed, 81 skipped, 0 failed |
| CPU digests of the six configs | unchanged |

Each sweep shard now names the tree it timed: the before legs resolve to
`$SCR/se6/raster2d-preB/stable_envs/__init__.py` and the after legs to `$SCR/se6/raster2d/...`. The
first run of this stamped both shards with the after tree's SHA because the environment variable was
exported once for the job rather than per leg — fixed, and recorded here because `stable_envs_file`
being right is what made it visible.

## 14. Still open

- **`cores_cpu`**, the G4 bar, is now the only thing between these rows and `CPU_FAST`, and it is a
  memory-bandwidth argument rather than an arithmetic one.
- **`raster2d._row_groups` is shared machinery now** — `swm_nav` calls it for its Gaussian dots. If
  `gfxdraw` wants the same partition the interface is in §9. Not touched here.
- **`stable_envs/testing/conformance.py`** was edited by this workstream (§8) after the coordinator
  confirmed it; `swm_nav.py` likewise, under SWMNAV-1.


---

# Round 3 — SWMNAV-1 held, and why

## 15. The GPU regression I reported as "no regression"

Round 2's §13 tabulated best-batch f/s and concluded no healthy GPU row regressed. The per-batch
numbers, from the artifact I committed alongside that sentence (job `6349274`, `gpu2113`, now in
`sweep/superseded_round2_node1828/`):

| config | B=1 | B=16 | B=64 | B=256 | B=1024 | B=4096 |
|---|---|---|---|---|---|---|
| TwoRoom | 1.25 | **0.48** | **0.55** | 1.41 | 1.40 | 1.41 |
| TwoRoomTarget | 0.86 | **0.32** | **0.40** | 1.11 | 1.13 | 1.13 |
| Piecewise | 0.58 | **0.19** | **0.35** | 0.99 | 0.98 | 0.98 |
| PiecewiseZones | 0.85 | **0.27** | **0.43** | 1.14 | 1.14 | 1.14 |

A 5.3x regression at batch 16 that a best-batch summary does not show is still a regression, and the
summary was mine. Best-batch is the right statistic for a tier, because the tier is defined at the
certified batch; it is the wrong statistic for "did anything get worse", and I used it for both.

**Mechanism.** Both SWMNAV-1 optimisations decide on the HOST from values living on the device: one
`.cpu()` for the bake key, one plus four scalar reductions for the dot windows, doubled for
TwoRoomTarget's two dots. On a CPU those are copies. On a GPU they are synchronisations in the middle
of a render, and they cost more than the work they save at every batch where the saved work is small.
Section 7 names host/device synchronisation in a hot loop as a structural defect in its own right,
not a cost to net off against a gain elsewhere. `raster2d._row_groups` does not rescue it: both its
constants were measured on one CPU core, so on cuda it accepts groupings that lose.

**Fix.** On `device.type == "cuda"` both paths run the arithmetic they replaced, unchanged. The GPU
rows are identical code rather than an argument about which batches the qualifiers sample. Recovering
the large-batch gain there — 1.41x on TwoRoom was real — needs the decisions made on the device or
the cost model re-calibrated on the card, and that is its own ticket.

## 16. The full per-batch GPU table, and how it was taken

Two SLURM legs turned out not to be able to answer this. A GPU render at batch 1 is host-launch
bound, the legs run minutes apart on a shared node, and the spread *within* one tree's two runs came
out the same size as the difference between the trees: PiecewiseZones read **1,031 and 1,075 f/s for
the same code** at batch 1. Job `6352385` ran the pair once each and every row came back 0.96-1.00
*including PushT and SimplePointMaze, whose cuda code is byte-identical between the two trees* —
which is how the bias was spotted rather than argued about. Job `6353295` repeated the pair with the
order reversed and the low cells moved with the position.

So the decisive measurement is in ONE process, interleaved, on one warmed card
(`scratch/step6/raster2d/gpu_same_process_ab.py`, job `6353691`, `gpu2264`, artifact
`sweep/gpu_same_process_ab.json`). `swm_nav`'s two methods take their branch from
`self.device.type`, so binding them to the expressions they replaced *is* the pre-SWMNAV-1 render:

| config | B=1 | B=16 | B=64 | B=256 | B=1024 | B=4096 |
|---|---|---|---|---|---|---|
| TwoRoom | 0.992 | 0.991 | 0.995 | 1.000 | 1.000 | 0.999 |
| TwoRoomTarget | 0.987 | 0.987 | 0.997 | 0.999 | 1.000 | 1.000 |
| Piecewise | **0.979** | 0.981 | 0.992 | 0.999 | 1.000 | 0.999 |
| PiecewiseZones | 0.985 | 0.986 | 0.992 | 0.999 | 1.002 | 1.000 |

**Worst 0.979, bar 0.97, and every cell reports its frames identical.** The residual is explained
rather than absorbed: at batch 1 a Piecewise GPU render is 0.667 ms and the cuda branch adds one
Python call for `_base_image`, one for `_blend_dot` and two attribute compares — about 14 us, which
is 2.1% of 0.667 ms and nothing at all by batch 256. It could be removed by binding the
implementation once at construction instead of branching per call; that trades a real interaction
with subclass init and with the tests that rebind these methods for 2% at batch 1 on a path that is
not the certified one, so it was not done.

And the four-leg cross-process table (`sweep/gpu_{before,after,after_first,before_second}.json`, job
`6353295`, `gpu2101`), for completeness — this is the one whose noise floor is the effect size:

| config | B=1 | B=16 | B=64 | B=256 | B=1024 | B=4096 |
|---|---|---|---|---|---|---|
| PushT (control, identical cuda code) | 0.99 | 0.98 | 0.99 | 0.99 | 0.99 | 0.99 |
| SimplePointMaze (control) | 1.00 | 0.99 | 1.02 | 0.98 | 0.99 | 1.00 |
| TwoRoom | 0.98 | 0.98 | 0.98 | 1.00 | 1.00 | 1.00 |
| TwoRoomTarget | 0.98 | 0.99 | 0.99 | 1.00 | 1.00 | 1.00 |
| Piecewise | 0.97 | 0.99 | 0.98 | 1.00 | 1.00 | 1.00 |
| PiecewiseZones | 0.94 | 0.95 | 0.98 | 1.00 | 1.00 | 1.00 |

The two control rows sit at 0.98-1.02 for code that cannot have changed, which is the measurement's
own floor and why the in-process table above is the one to read.

## 17. CPU gains re-confirmed on the shipping code

Job `6352386`, `node2334`, load average 40.3, both legs in one job, before = `1103326`,
after = this branch. Provenance per leg: the before shards carry `1103326` and
`.../raster2d-preB/stable_envs/__init__.py`, the after shards `c96f886` and `.../raster2d/...`.

| config | best B | before f/s | after f/s | speed-up | `S_cpu` after | `cores_cpu` after |
|---|---|---|---|---|---|---|
| PushT | 1024 | 621 | **1,981** | 3.2x | **5.51** | 0.24 |
| SimplePointMaze | 16 | 106 | **242** | 2.3x | 4.91 | 0.22 |
| TwoRoom | 16 | 290 | **1,359** | 4.7x | 4.32 | 0.41 |
| TwoRoomTarget | 1 | 197 | **638** | 3.2x | 3.20 | 0.16 |
| Piecewise | 16 | 426 | **1,243** | 2.9x | 3.21 | 0.37 |
| PiecewiseZones | 16 | 233 | **1,205** | 5.2x | 4.48 | 0.34 |

Compared with round 2 on `node1828`, `S_cpu` for TwoRoom reads 4.32 against 5.08 and PiecewiseZones
4.48 against 5.28 — **the rows that straddle the `S_cpu = 5` line straddle it by node**, so the
honest statement is that PushT clears it on both nodes and TwoRoom and PiecewiseZones clear it on one
of the two. `cores_cpu` is unchanged in its verdict: still well under 1, so `BELOW_G4` still caps
every row at `CPU_FASTER`.

Pixels, round 3: all six configs replay identical on cuda (`replay_r3/`, `replay_r4/`), 650/650 fuzz
on the card, 60 unit tests (the two new files) and 183 GPU tests passed, four CPU digests unchanged.

## 18. Where the shards live

`sweep/` now holds exactly one measurement of each leg, because the Step 6 report generator labels a
tier from every Step-3-shaped shard it finds there and two rounds side by side would give one env two
tiers. Earlier rounds are in `sweep/superseded_*/`, each with a README saying what it measured and
why it is not the one to read; `sweep/README.md` is the index. Nothing was edited — a shard is a
record of a run, and correcting the record is worse than recording that it was wrong, which is what
`superseded_round2_node1828/PROVENANCE_NOTE.md` does for the SHA that was stamped once per job
instead of once per leg.

Full job list for this workstream: `6322230` (wave-0 profile), `6344858`, `6346289`, `6346290`,
`6346291`, `6346300`, `6346490` (round 1), `6349273`, `6349274` (round 2), `6352385`, `6352386`
(round 3), `6353295` (order-reversed GPU), `6353691` (same-process GPU A/B).
