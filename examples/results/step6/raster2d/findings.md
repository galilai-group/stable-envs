# Step 6E — `raster2d` / Scene2D CPU

**`PATCH`.** One correctness commit that makes the CPU path reachable at all on the certification
cluster, and one structural commit that removes the largest measured bottleneck in it. Pixels are
unchanged everywhere, held by digest rather than by argument. `CPU_FAST` is **not** reached; the
remaining bottleneck is isolated, named and measured below, which is this workstream's other
stop condition.

| | |
|---|---|
| baseline | `89fa5ca81341dc4df7186ec325cb691431a2ef17` on `sami/pre-step6-baseline` |
| branch | `step6/raster2d` |
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
the frame each.

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
| CPU digests of the six configs, before and after, on Oscar and on the laptop (`cpu_digest_before.json`, `cpu_digest_after.json`) | identical across the patch AND across arm/x86; five of the six also equal the frozen cuda digest |
| every registered Scene2D consumer on CPU — PushT, SimplePointMaze and the six highway-env envs (130 to 604 primitives, outlines, padded vertices) | 8 of 8 digests identical |
| differential fuzz against a verbatim copy of the 89fa5ca8 implementation (`scratch/step6/raster2d/fuzz_render_torch.py`) | **1,650 scenes on CPU + 650 on the 3090, byte-identical in every case** — collinear polygons, all-degenerate edges, padded triangles, negative radii, rings larger than their disc, primitives off the frame and larger than the world, `flip_y`, non-square worlds and outputs, supersample 2 |

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

Measured on the real agent positions of the frozen corpus, splitting the batch into chunks and
taking a window per chunk collapses that cost — 59.5 ms at one chunk of 64, 35.3 at chunks of 4,
15.1 at 2, **1.5 ms at 1** — but chunk-of-1 is a Python loop over the batch, which is the first
structural problem on section 7's own list and would be `O(B)` dispatch at B=4096. The shape that
keeps it batched is the one the GPU already uses: bin primitives against **screen tiles** and, per
tile, select the rows whose box reaches it, which is flat in the batch. That is a larger change than
this ticket authorises and it is the next thing worth doing on this path.

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

**Bounding is not available to them.** `_gaussian_dot` normalises by the frame's own maximum and
`_alpha_blend` truncates to uint8, so an alpha of 2e-9 still moves a pixel by one count — the dot's
support is the whole frame and cropping it would change pixels. Any patch there has to come from the
static half, not the dynamic one. Recommend a separate ticket; owner unclear, since `swm_nav.py` is a
per-env renderer rather than a shared one.

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
