# Step 6 — `tilemap` (tile / atlas / gather rendering)

**Classification: `PATCH`.**

| | |
|---|---|
| baseline SHA | `89fa5ca81341dc4df7186ec325cb691431a2ef17` (`sami/pre-step6-baseline`) |
| merged in | `step6/integration` at `00ffe697c733` — no file overlap with this workstream |
| affected configs | **`CraftaxClassic` only** — 1 of the 49 frozen Step-4 render-distinct configurations |
| affected swm ids | `swm/CraftaxClassicPixels-v1` (Gate A `PARTIAL`, unchanged) |
| pixels changed | **no**, on both devices, at every tile size, at the default appearance and under every variation |
| semantic follow-up | **none**. Every frozen digest is byte-identical, so there is nothing for a tripwire to see |
| CPU tier | `CPU_REGRESSION` (`NO_REFERENCE` `BELOW_G4`) → **`CPU_EXCEPTIONAL`** (`NO_REFERENCE`) |
| GPU tier | `GPU_FAST` (`NO_REFERENCE`) → **`GPU_EXCEPTIONAL`** (`NO_REFERENCE`); both frozen qualifiers still pass, by a wider margin |
| SLURM jobs | `6349384` CPU sweep · `6350070` CPU confirm · `6349394` GPU sweep · `6351158` GPU confirm |

---

## 1. What was measured before anything was edited

The brief forbids editing code that looks slow, so nothing was edited until the attribution below
existed. Two harnesses produced it, both run on **one Oscar batch node, in one job, at one torch
thread, with the two legs interleaved**, each leg recording the `stable_envs` it actually imported:

- `scratch/step6/profile_note.py` — the operator trace. Extended here to record
  `stable_envs.__file__` and the node's load average, because a `PYTHONPATH`-switched before/after
  is otherwise indistinguishable in the artifact: the harness's own git commit answers a different
  question from "which checkout did this leg import".
- `scratch/step6/tilemap/attribute.py` — the section-8H phase vocabulary (atlas indexing, gather,
  layout, compositing, recolouring, allocation, output conversion, static vs dynamic). It
  re-executes the renderer's own stages one at a time and **asserts the reconstruction is
  pixel-identical to `env.render` before reporting any timing**, so a per-stage number attributes
  time to the real path rather than to a lookalike.

| artifact | path |
|---|---|
| profile-before, operators | `profile_before/CraftaxClassic_cpu_node1831.md` (plus the coordinator's earlier `..._node2314.md`) |
| profile-before, phases | `profile_before/attribution_before_cpu_node1831.md` |
| profile-after, operators | `profile_after/CraftaxClassic_cpu_node1831.md` |
| profile-after, phases | `profile_after/attribution_after_cpu_node1831.md` |
| CPU certification sweep | `sweep/cpu_before.json`, `sweep/cpu_after.json`, `sweep/tier_cpu.json` |
| CPU sweep re-run on the final commit | `sweep/cpu_before_final.json`, `sweep/cpu_after_final.json`, `sweep/tier_cpu_final.json` |
| Step-4 replay | `replay/cpu_before.json`, `replay/cpu_after.json`, `replay/cpu_before_vs_after.json`, `replay/cpu_vs_frozen.json` |

---

## 2. The bottleneck, and why throughput FELL with batch

`CraftaxClassic.render` composited every frame from scratch: gather the block texture into a
`B x 7t x 9t x 3` float32 buffer, untile it, blend the player into the centre tile, build a
**full-frame** mob texture layer and a **full-frame** mob alpha layer, blend those over everything,
multiply by `channel_gain`, round, cast. About a dozen image-sized float32 temporaries per frame:
**1.94 MB of them against a 48.4 KB frame** at the default 16 px tile.

`node1831` (Intel Xeon, 1 thread; L1d 1.5 MiB / 48 instances, L2 48 MiB / 48 instances,
L3 71.5 MiB / 2 instances, so ~35.8 MiB of last-level cache per socket):

| B | blend-pipeline float temporaries | fits L3? | `env.render` | fps |
|---|---|---|---|---|
| 1 | 1.9 MB | yes | 0.798 ms | 1,253 |
| 16 | 31.0 MB | **just** | 8.527 ms | **1,876** |
| 64 | 123.9 MB | no (3.5x) | 38.476 ms | 1,663 |
| 256 | 495.5 MB | no (14x) | 230.060 ms | **1,113** |

That is the answer to the question the brief asked. Throughput did not fall because of per-row
Python (`set_state` is 0.04–0.13 ms and essentially flat, and the operator count is a **constant
216 per render at every batch**), and not because of allocation (`torch.empty` of a whole float
frame measures 0.002–0.010 ms — under 0.5% at every batch). It fell because **the working set
crossed the last-level cache between B=16 and B=64**, and the pipeline is memory-bound on
temporaries that carry no information.

Every stage degrades in step, which is what a capacity miss looks like and what a dispatch or
allocation problem does not:

| stage, as frames/s in isolation | B=16 | B=64 | B=256 |
|---|---|---|---|
| tile gather | 22,268 | 22,401 | 13,671 |
| gather + layout (untile) | 15,717 | 12,459 | 7,513 |
| mob alpha layer | 10,620 | 9,085 | 5,346 |
| mob composite | 5,013 | 4,066 | 2,467 |
| channel gain (recolour) | 19,328 | 17,178 | 11,616 |
| output conversion | 20,318 | 18,854 | 10,049 |

Attribution at the certified end of the curve (B=256, before): **compositing 66.3%** (player 0.4,
mob alpha layer 20.8, mob blend 45.1), **layout 6.7%** on top of an **8.1% gather**, **recolouring
9.6%**, **output conversion 11.1%**, **allocation 0.0%**, **atlas indexing 0.1%** (decoding class
ids out of the state).

**Static versus dynamic.** Only that last 0.1% is dynamic. A tile's pixels are a function of exactly
three small integers — block id, mob id, and which player sprite (if any) stands on it — so the
entire composite is static data that was being recomputed per frame, per batch, forever.

---

## 3. The intervention

One structural change, in `stable_envs/envs/craftax.py`: **bake the composite into a 680-entry tile
table and gather from it straight into the uint8 output.**

`_Composite` holds `5 player slots x 17 blocks x 8 mobs` already-composited tiles, built once per
`(device, tile size)` — 522 KiB as uint8 at 16 px, which sits in L2 and stays there.

- **`mobs_visible = 0` folds into mob id 0**, whose alpha table is exactly zero — asserted, not
  assumed (`test_mob_zero_alpha_is_exactly_zero`). Scaling the alpha to zero and looking up the
  "no mob" composite are then the same float32 expression, `x*(1-0) + tex*0`.
- **`player_visible = 0` and every non-centre tile fold into player slot 4**, which is the block
  texture untouched.
- Both fold into the **gather index, not into a branch**, so one batch may still carry a different
  appearance per environment. A fast path guarded on "the whole batch is at the default" would have
  been simpler and would have quietly broken `style-as-state`.

At the default `channel_gain` the render is then **one `index_select` from a uint8 table into the
uint8 output, with no float buffer anywhere**. The index is built at (frame, tile row, scanline
within the tile, tile column) granularity, which *is* the output's own row order, so the gather
writes the final raster directly instead of building a tile-major buffer and transposing it.

`channel_gain` cannot fold in — it multiplies *before* the rounding, so a uint8 table cannot carry
it. A batch with any gain other than exactly 1 gathers the pre-rounding float table and then applies
the gain: still one gather and three elementwise passes rather than a dozen. Which table is used can
change only the speed and not the pixels, because the uint8 table is *defined* as
`round -> clamp -> uint8` of the float one, in that order, with the ops the renderer used.

The pre-Step-6 body is kept verbatim as `CraftaxClassic._render_blend`. It is the readable
definition of what the table bakes, and it is the path taken above `ATLAS_MAX_TILE_PX = 64`, where
the table (`680 * t*t*3` bytes as uint8, four times that as float32 — 522 KiB at 16 px but 134 MiB
at `size=256`, which `tests/unit/test_benchmarks.py` does exercise) stops being a small resident
table and becomes the largest allocation in the process. The cache keeps at most 8 `(device, size)`
entries, the rule `render/tilemap.py` already uses for its recoloured atlas.

---

## 4. Pixels: unchanged, shown four ways

**(a) Frozen Step-4 replay — `CraftaxClassic`, the only affected configuration.** Every digest the
oracle froze is reproduced byte for byte. The frozen digests were taken on CUDA and are reproduced
here on the CPU as well, so the two devices agree with each other and with the oracle:

| digest | frozen (Step 4, RTX 3090) | replay (branch) |
|---|---|---|
| `digest_ours` | `1aeabbe8bc6a9a2e…` | identical |
| `digest_ours_default_style` | `1aeabbe8bc6a9a2e…` | identical |
| `digest_ours_corpus_style` | `e80818fa414e2b13…` | identical |
| `style_digest` | `0514b876ec45f00a…` | identical |
| `digest_official` | `1aeabbe8bc6a9a2e…` | **unavailable** — `jax` is in no Oscar Step-6 interpreter |

`frames_shape` `[64, 112, 144, 3]`, `mean_pixel` 93.84458414713542 and `distinct_colors` 277 match
the frozen record exactly. `replay/cpu_vs_frozen.json`, and `replay/cpu_before_vs_after.json` for
baseline-versus-branch on the same node in the same job.

**(b) The full conformance battery, before and after, same node, same job.** Every mandatory check
reports the same evidence on both trees: batch invariance 0, batch-size invariance 0, chunked render
0 over chunks `[1, 7, 64, 4096]`, order independence 0 for both duplication and permutation, render
purity 0 on all four probes, `style_is_state` 0, no dead state dimension, no inert variation,
`check_resolution_consistency` `exact_frac` 1.0 / `max_abs` 0 — that one renders at 32 px as well as
16, so the nearest-resampled-texture path is covered too. `check_backend_agreement` stays
`not_applicable` (this env renders without a `Scene2D`) and
`render_parity_with_official_renderer` stays `blocked_environment` on the missing `jax`, exactly as
before.

**(c) A committed differential fuzz — `tests/unit/test_craftax_atlas.py`.** The atlas path against
`_render_blend` (the HEAD render function, kept verbatim) at tile sizes 7, 16, 32 and 64, over three
state families — `sample_states`, uniformly random class ids, and deliberately out-of-range
fractional states that exercise the documented wrap — crossed with seven appearances: default,
random gain, an extreme `[0, 1, 2]` gain, each visibility flag off, both off, and a
**per-environment mixture of all three within one batch**. 84 cells, `max|diff| == 0` in every one.
The module also asserts the table is never written through by a render, that slot 4 is the block
texture untouched, that the cache stays bounded under a size sweep, that `out=` receives the frame
on both paths, that leading dimensions survive, and batch / chunk / permutation invariance of the
new gather. 16 tests.

**(d) A cross-tree render diff** against the baseline checkout: 90 frame blocks (6 tile sizes,
including two above the atlas budget so the fallback is covered, x 3 state families, x 5
appearances) rendered by both trees and compared block by block — **0 differ**, and the concatenated
SHA-256 over all 90 blocks is
`1c6e47d6ef0fdcb534ce70fc850af1fce416b1d71189fe3d8c38e892161e9dcf` on both.

**Gate A stays `PARTIAL`.** The documented exclusions — the 2-row inventory strip, night frames, a
sleeping player, two entities on one tile — are untouched. No coverage was added; that is Step 7.

---

## 5. CPU certification: before and after, one node, one job

`examples/bench_step3.py --device cpu --threads 1 --envs CraftaxClassic`, both legs on `node1831`
in job `6349384`, labelled by **Step 3's own evaluator** — `scratch/step6/tilemap/tier.py` imports
`scratch/step3/certify.py:classify` rather than restating the ladder, and was validated by
reproducing the frozen Step-3 labels for this env on both hosts before being used here.

Craftax's official renderer needs `jax`, which no Oscar Step-6 interpreter carries, so `S_cpu` is
undefined and the **frozen `NO_REFERENCE` fallback ladder** applies: `CPU_FASTER` at
`cores_cpu >= 1`, `CPU_FAST` at `>= 3`, `CPU_EXCEPTIONAL` at `>= 10`. That selection is made by the
shard, which records the importer's own error — not by this workstream.

| | before | after |
|---|---|---|
| fps at B=1 | 1,218 | 3,067 |
| fps at B=16 | **1,867** | 33,967 |
| fps at B=64 | 1,711 | 81,507 |
| fps at B=256 | 1,144 | 115,166 |
| fps at B=1024 | 1,134 | **116,448** |
| certified batch | 16 | 1,024 |
| one-core JPEG-q90 decode of its own frame | 4,345 f/s | 4,976 f/s |
| `cores_cpu` | **0.430** | **23.40** |
| qualifiers | `NO_REFERENCE` `BELOW_G4` | `NO_REFERENCE` |
| **tier** | **`CPU_REGRESSION`** | **`CPU_EXCEPTIONAL`** |

**62.4x at the certified batch**, and the curve no longer falls with batch. `BELOW_G4` is gone: the
renderer now beats one core of JPEG decode of its own frame by 23x where it previously lost to it by
more than half. The decode bar is re-measured per run and came out 15% *higher* in the after leg, so
the ratio is if anything conservative.

The workstream's frozen stop condition — "current covered tile/atlas paths meet the CPU target
without changing coverage or fidelity semantics" — is met, and met at `CPU_FAST`'s bar of 3 by a
factor of 7.8.

**Re-measured on the final commit**, on a different, much less loaded batch node — job `6350070`,
`node2333`, load average 7.8 against `node1831`'s 38 — because the first sweep measured the tree one
commit earlier (the difference is a two-line guard bounding the table cache, which cannot execute in
a process that renders one tile size; "cannot execute" is an argument, not a measurement, so it was
re-measured rather than asserted):

| | before | after |
|---|---|---|
| fps at B=1 / 16 / 64 / 256 / 1024 | 580 / **2,528** / 1,583 / 879 / 781 | 3,487 / 38,915 / 99,769 / **154,086** / 146,632 |
| certified batch | 16 | 256 |
| decode bar | 4,818 f/s | 5,534 f/s |
| `cores_cpu` | 0.525 | **27.85** |
| **tier** | **`CPU_REGRESSION`** (`NO_REFERENCE` `BELOW_G4`) | **`CPU_EXCEPTIONAL`** (`NO_REFERENCE`) |

61.0x at the certified batch, the same verdict on a different node under a seventh of the load. The
before leg's falling-with-batch curve is *sharper* on the quiet node (2,528 → 781 f/s from B=16 to
B=1024) than on the busy one, which is what a cache-capacity limit does and what contention noise
does not.

Corroboration on a third machine and a different microarchitecture (Apple M5 Pro, 1 torch thread —
**not** a certification number): 5,237 f/s before, 186,685 f/s after, 35.6x. The Mac does *not* show
the falling-with-batch curve at all, which is what the cache-capacity explanation predicts on a
machine with a very different memory system and what an algorithmic explanation would not.

### Where the remaining 2.47 ms at B=256 goes

| phase | ms | % of render |
|---|---|---|
| `index_select` gather into the uint8 output | 0.999 | 40.4 |
| gather-index build (the `(B,H,t,W)` int32 index) | 0.552 | 22.3 |
| decode: class ids out of the state | 0.173 | 7.0 |
| style resolution, views, `where`/`full`, dtype casts | ~0.75 | ~30 |
| allocation | 0.002 | 0.1 |

Operator calls per render fell from **216 to 186** — a far smaller change than the 62x in
throughput, which is the point: this was never a dispatch problem, and a patch that had only cut
launches would have moved almost nothing.

---

## 6. Tests run

| suite | where | result |
|---|---|---|
| `tests/unit` (whole suite) | `node1831` job `6349384`; re-run `node2333` job `6350070` | **558 / 559 passed, 9 skipped, 2 failed (both pre-existing, see below)** |
| `tests/unit/test_craftax_atlas.py` (new, 16 tests) | local + Oscar | all passed |
| `examples/fidelity_step4.py --configs CraftaxClassic` | CPU, both trees, same job | `measured`; every check identical |
| `tests/gpu/{test_conformance,test_reproducibility,test_fast_paths}.py -k CraftaxClassic` | 3090, job `6349394` | section 8 |

The **2 failures are pre-existing and belong to other workstreams.** Both were reproduced at the
merge commit with this branch's changes stashed:

- `tests/unit/test_repo_contracts.py::test_every_cited_scratch_path_resolves` — two scratch paths
  cited by `examples/results/step6/texture-fidelity/findings.md` (`scratch/carracing/logs/` and
  `scratch/robocasa/out/corpus`) do not resolve in a worktree;
- `tests/unit/test_examples.py::test_every_argparse_script_can_print_its_own_help[robocasa_visual/render_mjwarp.py]`.

Both arrived with `step6/integration`; they are the coordinator's to route. This workstream touched
neither file.

---

## 7. Residual, as a narrow ticket

**`TILEMAP-2` — the `channel_gain` path is ~21x slower than the default path and is now this env's
worst case.** Measured on `node1831` at B=256: default appearance 103,585 f/s, non-default
`channel_gain` 4,809 f/s. The mechanism is the one this patch fixed everywhere else: a gain other
than exactly 1 multiplies before the rounding, so the gather has to produce a float32 frame (1.94 MB
per frame, 495 MB at B=256 — the same L3 overflow as the old pipeline) before the gain and the
rounding can be applied. It is still 4.3x the pre-patch number, and it is off the certification path
(the certified appearance is the default one, and `check_variations_do_something` passes), so it is
a ticket rather than a blocker. The obvious fix is a **per-gain uint8 table** baked when the style is
batch-shared. It is not in this patch because forming that cache key means reading the gain to the
host, which is a device→host sync, and that trade needs its own measurement rather than an
assumption.

**Recorded, smaller:** `render` calls `torch.all(gain == 1.0)` to choose the table — free on the CPU,
but on CUDA a 3-element reduction plus one synchronisation per render. It cannot be hoisted without
either caching a boolean against a mutable style tensor's identity or reading the gain to the host,
both worse than the thing they avoid. Its GPU cost is reported in section 8.

**Not a ticket, but worth the coordinator's eye:** the same shape of fix applies to every other
gather-style env in the repo — `crafter.py`, `griddly.py`, `minihack`, the Jumanji family are all
per-cell gathers over a small finite composite. None is in this workstream's scope or in its Step-4
replay set, so none was touched.

---

## 8. GPU — before and after on the same card

Job `6349394`, `--partition=gpu --qos=norm-gpu --gres=gpu:1 --constraint=geforce3090`, **one
allocation, one card**: `gpu2106`, NVIDIA GeForce RTX 3090. Both legs measured in that allocation,
each recording the checkout it imported. GPU was already healthy for this env, so the job of this
run was to prove it *stayed* healthy — and in particular that the two frozen qualifiers still hold,
because removing most of the per-render work lifts small batches more than large ones and
`BATCH_SCALES` is `fps(4096) >= 20 * fps(1)`. A patch can make every batch faster and still fail
that bar.

### Certification sweep, all six mandatory batches

| | before | after |
|---|---|---|
| fps at B=1 | 1,487 | 1,892 |
| fps at B=16 | 23,687 | 30,108 |
| fps at B=64 | 74,753 | 119,150 |
| fps at B=256 | 127,090 | 381,039 |
| fps at B=1024 | 141,484 | 836,415 |
| fps at B=4096 | **145,110** | **1,206,822** |
| one-core JPEG-q90 decode | 5,636 f/s | 4,599 f/s |
| `cores` | 25.7 | **262.4** |
| `BATCH_SCALES` (bar: 20x) | **pass**, 97.6x | **pass**, 637.9x |
| `NO_LARGE_BATCH_REGRESSION` (bar: 0.9) | **pass**, 1.142 | **pass**, 3.167 |
| `HOST_BOUND` (bar: `cpu_frac >= 1.5`) | no, 0.995 | no, 0.994 |
| **tier** | **`GPU_FAST`** (`NO_REFERENCE`) | **`GPU_EXCEPTIONAL`** (`NO_REFERENCE`) |

**8.3x at the certified batch, and both qualifiers pass by a wider margin than before**, because
B=1 barely moved: at B=1 this renderer is bound by Python dispatch (232 operator calls), not by the
device, so the patch cannot lift it much — which is exactly what `BATCH_SCALES` is designed to
reward. The before leg reproduces Step 3's frozen 3090 number for this env (144,738 f/s) to within
0.3%, which is the check that the two legs are comparable at all.

### Peak device memory, which is the same finding said differently

| B | peak VRAM before | after |
|---|---|---|
| 256 | 243 MiB | 16 MiB |
| 1024 | 954 MiB | 58 MiB |
| 4096 | **3,790 MiB** | **221 MiB** |

17x less at B=4096. The old pipeline's peak *was* the image-sized float32 temporaries; the new one's
peak is the output frame plus the gather index, and the 522 KiB table.

Operator calls per render fell from 271–277 to 232–238. The device→host synchronisation introduced
by `torch.all(gain == 1.0)` is inside those numbers and is not measurable against them: B=1 rose
from 1,487 to 1,892 f/s *despite* it.

### Frozen Step-4 replay on CUDA — the oracle's own device

`PIXELS UNCHANGED`. `digest_ours`, `digest_ours_default_style`, `digest_ours_corpus_style` and
`style_digest` are all byte-identical to the frozen record, as are `frames_shape`
`[64, 112, 144, 3]`, `mean_pixel` 93.84458414713542 and `distinct_colors` 277. The baseline leg was
replayed on the same node for the same comparison and also came back `PIXELS UNCHANGED`, so a
difference would have indicted the patch rather than the node. `digest_official` is `unavailable`
(no `jax`). Files: `replay/cuda_vs_frozen.json`, `replay/cuda_before_vs_frozen.json`.

### Targeted GPU tests

`tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py tests/gpu/test_fast_paths.py -k
CraftaxClassic` → **30 passed, 14 skipped, 3481 deselected**. No failures.

### Re-measured on the final commit, on a second 3090

The first GPU job finished in 58 seconds and three docstring-only commits landed in the same minute,
so its artifact does not pin which text the after leg imported. Job `6351158` re-took the sweep, the
CUDA replay and the GPU tests on the offered commit, on `gpu2104` (a different RTX 3090, driver
575.57.08), with `md5sum stable_envs/envs/craftax.py` printed into the log and matching the offered
tree byte for byte (`ec41decf3fb8ad2b5fb7e0054884ef26`):

| | before | after |
|---|---|---|
| fps at B=1 / 16 / 64 / 256 / 1024 / 4096 | 1,504 / 23,823 / 74,963 / 127,240 / 141,334 / **145,113** | 1,918 / 30,639 / 120,649 / 381,568 / 849,747 / **1,229,787** |
| `cores` | 31.0 | **268.4** |
| `BATCH_SCALES` | pass, 96.5x | pass, **641.3x** |
| `NO_LARGE_BATCH_REGRESSION` | pass, 1.140 | pass, **3.223** |
| **tier** | **`GPU_FAST`** | **`GPU_EXCEPTIONAL`** |

8.5x, the same verdict on a second card. `PIXELS UNCHANGED` against the frozen oracle; the targeted
GPU tests again 30 passed, 14 skipped. Files: `sweep/gpu_before_final.json`,
`sweep/gpu_after_final.json`, `sweep/tier_gpu_final.json`, `replay/cuda_final_vs_frozen.json`.

---

## 9. What this does not claim

- It does not claim a parity number against the official renderer **measured in Step 6**. `jax` is
  absent from every interpreter available here, so `render_parity_with_official_renderer` was
  `blocked_environment` on both legs. What it claims instead is that the frames are byte-identical
  to the frames Step 4 certified against the official renderer — a stronger statement about this
  patch and a weaker one about the renderer.
- It does not move `CraftaxClassic` toward `ENVIRONMENT_DONE`. Gate A is `PARTIAL`, and the frozen
  ladder is explicit that no throughput number retires a declared exclusion. The reachable state is
  `BACKEND_CERTIFIED_PARTIAL(cpu)`, and this supplies the `cpu_tier_at_least_CPU_FAST` edge that was
  missing.
- It does not change any threshold, declaration, frozen artifact, or file another workstream owns.
  The one shared file touched is `scratch/step6/profile_note.py`, additively: two recorded fields
  (`stable_envs_file`, `load_average`) and one line in the generated note.
