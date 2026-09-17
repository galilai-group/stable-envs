APPROVE

# Step 9 — independent review of Workstream G (2-D rendering backends)

**Reviewed** `step9/render-2d` at **8afa2f4** (delta `git diff 5befc46 8afa2f4`) · parent `5befc460`
(step9/integration, Wave 0 + H) · Step-8 behavioural baseline `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`.
**Reviewer host** Apple M5 Mac `cis1b1zs2j3`, local CPU only, `MUJOCO_GL=glfw`, `-p no:cacheprovider`,
venvs `/Users/samib/se6-venv`, `/Users/samib/se8-vm-venv`, `/Users/samib/se7-craftax-venv`.
Nothing was edited, committed, stashed, reset or pushed; Oscar was not touched. (The oracle harness
creates and removes its own detached `git worktree` of `21cb306` under `$TMPDIR`; it cleaned up.)

## Verdict

**APPROVE**, including the fused-kernel paths, with the post-Wave-1 GPU oracle to follow (ruling 8).
Reasoning in §Judgement below. Three documentation-only defects are listed at the end; none blocks.

## The ten §17 items

1. **Files moved/removed/created** — 17 files: 8 owned render modules edited, 1 new test file
   (`tests/unit/test_render_2d_shared.py`), 3 new harnesses under `scratch/step9/render-2d/`,
   5 artefacts under `examples/results/step9/render-2d/`. `git diff --stat 5befc46 8afa2f4 --
   stable_envs/envs tests` = the new test file only. Zero consumer edits. Verified.
2. **API changes** — additive: `raster2d.tile_shape`, `raster2d.TILE_WASTE_MAX`, `raster2d.row_groups`,
   `gfxdraw.edge_geometry`, `scene.Batched`. `raster2d._row_groups` kept as an alias to the *same
   object* (`row_groups is _row_groups` → True), so `envs/navigation.py:872` is untouched.
   `edge_setup` keeps its exact 14-tuple; `_fit`, `_TILE_WASTE_MAX`, `gfxdraw_classic._blend_t`
   removed with no surviving caller (`grep -rn _blend_t stable_envs tests scratch` → only the new
   test's `assert not hasattr(GC, "_blend_t")`). No signature changed except annotations.
3. **Step-8 IDs affected** — the 12–13 SWM ids the findings enumerate, plus Crafter/highway/tile
   families. Confirmed by re-running the instrumented discovery, not read off imports: 20 CPU cells,
   39 skipped with dependency reasons byte-identical in both arms, 22 registered configs classified
   `not_a_2d_cell` (all DMC/Mujoco 3-D, plus MiniHack — see defect G-1).
4. **Targeted tests** — `tests/unit` on this branch: **1405 passed, 195 skipped, 112 deselected**
   (83.7 s), exactly the expected count. The 2-D suites in `se8-vm-venv`: **269 passed, 2 skipped**.
5. **Certification checks** — none touched: no threshold, declaration, tier, frozen digest or corpus
   is in the delta. The applicable GPU harnesses (`test_gfxdraw_classic`, `test_gfxdraw_fused`,
   `test_navdot_fused`, `test_fast_paths`, `test_conformance`) are selected in
   `scratch/step9/render-2d/gpu_oracle.sbatch`, which is prepared and not submitted.
6. **Default-frame comparison** — re-run by me from scratch into a temp dir; **20 cells, 64 states
   each, every compared key identical to the committed record, `unexplained_differences: {}`**. Also
   re-ran the two Craftax cells in `se7-craftax-venv` (which now imports): PASS.
7. **Variation/state comparison** — untouched by construction; `states_sha256`, `size`,
   `frames_shape`, `frames_from_rows_sha256` all identical across arms and across my re-run;
   `test_style.py`, `test_swm_matrix.py`, `test_swm_nav_render.py` pass.
8. **Performance** — no repeatable regression; worst committed 1.077 (SimplePointMaze) is below the
   1.10 bar and is inside this laptop's demonstrated same-tree floor. See §Timings.
9. **Resource lifetime** — none of the eight modules owns a process resource; caches
   (`raster2d._BASE_CACHE`, `gfxdraw.Gfx`, `tilemap.MiniGridAtlas`, `gfxdraw_classic.DiscTables`)
   are unchanged by the delta. The unit run's safety plugin reports 16 EGL contexts, all from the
   DM Control tests, none from these modules.
10. **Unexplained oracle differences: zero** on CPU, reproduced independently. GPU not yet measured
    (ruling 8: one post-Wave-1 GPU oracle).

## The three consolidations, checked expression by expression

### (a) `raster2d.tile_shape` vs the old `_fit` + navdot `_tile`

Read both old spellings at `git show 5befc46:stable_envs/render/{raster2d,navdot}.py`. Old
`_fit(target, W)` + `_tiles`'s `bh = max(1, min((8*128)//bw, 1 << (H-1).bit_length()))`, and navdot's
transcription with the literal `0.125` and `_BLOCK = 1024`, are the same arithmetic as
`tile_shape(H, W, lanes, max_width)` (`raster2d.py:955`), differing only in the width cap.
`_BLOCK` is still 1024; `TILE_WASTE_MAX` is still 0.125.

Re-ran the exhaustive equivalence myself over **(H, W) in 1..1100 × 1..1100 (1,210,000 pairs), both
call sites**:

```
tile_shape exhaustive 1..1100 x 1..1100 mismatches: 0
TILE_WASTE_MAX 0.125  _BLOCK 1024  row_groups is _row_groups: True
```

`navdot._tile` also still returns the same `(bh, bw, ntx, nt)` 4-tuple (`ntx`/`nt` unchanged).

### (b) `gfxdraw.edge_geometry` vs the old `edge_setup` and `gfxdraw_fused.edge_table`

`edge_geometry`'s body (`gfxdraw.py:634-681`) is **character-identical** to the old `edge_setup`
body — same 16 expressions, same order, same dtypes (int64 + bool), only the return tuple widened by
the clipped far endpoint `(x2, y2)`. The old `edge_table` body was the same 14 lines transcribed;
`edge_table` now unpacks `G.edge_geometry` and computes `kind` from the returned `vert/horiz/ymaj`,
which is order-irrelevant integer work.

Bit-identity measured against a standalone import of the **baseline** `gfxdraw`/`gfxdraw_fused`
(extracted with `git show 5befc46:` into a temp package), not against prose:

```
random quads: 300 geometries OK (edge_table bit-identical, edge_setup 14-tuple bit-identical)
degenerate cases: 9 OK          # zero-length edges, exactly horizontal, exactly vertical,
                                # wholly-clipped, clipped both sides, exact corners, off-by-one
                                # bounds, 45 deg (dx == dy)
edge_geometry returns 16 quantities; dtypes: ['torch.bool', 'torch.int64']
edge_setup returns 14; its elements are the SAME tensor objects as edge_geometry's
edge_table cols: torch.Size([9, 4, 16]) torch.int32  EG_W = 16
kinds over random quads: [0, 1, 2, 4]   # K_GEN, K_VERT, K_HORIZ, K_GEN+4 (ymaj) all reached
```

Comparison covered every column of the table (`draw, x1, y1, xx0, yy0, dx, dy, xdir, kind, major, E,
lo, ex0, ey0, ex1, ey1`), its dtype (int32) and its shape `(M, 4, EG_W)`.

### (c) `M32`, `_blend_t`, the constexpr twins, and the Triton source text

* `gfxdraw_fused.M32 is gfxdraw.M32` → True, value `(1<<32)-1` (unchanged); `cM32 = _c(M32)` so the
  compiled literal is unchanged.
* `_blend_t` was a one-line pass-through to `G.blend`; both call sites now call `G.blend` directly.
  No caller anywhere else.
* `gfxdraw_classic` no longer redefines `cEG_*`/`cK_*`; it imports them from `gfxdraw_fused`, whose
  definitions are **the same expressions** (`_c(EG_*)`, `cK_YMAJ = _c(K_GEN + 4)`) at module level
  inside the same `if HAVE_TRITON:` guard, from the same `EG_*`/`K_*` ints (unchanged in the delta).
  AST check: every `cEG_*`/`cK_*`/`EG_*`/`K_*` name loaded in `gfxdraw_classic.py` (32 of them) is
  bound — no missed import, so the module will import under Triton.
* **Kernel bodies diffed old vs new**: `awk '/@triton.jit/,0'` over `gfxdraw_classic`,
  `gfxdraw_fused`, `raster2d`, `navdot`, `tilemap` — every difference is host-side (annotations,
  `tile_shape`, docstrings). **No Triton source line changed in any of the five modules.** Same
  source text + same constexpr values ⇒ same compiled kernels.

## CPU oracle — re-run by the reviewer

```
cd /Users/samib/stable-envs-step9-render-2d
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python scratch/step9/render-2d/oracle.py \
    --out <tmp> --tag review
...
ORACLE: PASS      # exit 0
committed: identical True  head_sha de0ba10  baseline 21cb306  n 64
mine     : identical True  head_sha 8afa2f4
ok cells committed: 20   mine: 20   cell status sets equal: True
cross-run digest diffs: NONE
unexplained committed: {}   mine: {}
```

Cells (all `/cpu`): Acrobot, CartPole, CraftaxClassic, CraftaxFull, Crafter, Highway, Intersection,
Merge, MountainCar, MountainCarContinuous, Parking, Pendulum, Piecewise, PiecewiseZones, PushT,
Racetrack, Roundabout, SimplePointMaze, TwoRoom, TwoRoomTarget. I compared **every** per-cell key of
both arms of my run against both arms of the committed record — frame digests, frames-from-rows
digests, state digests, sizes, shapes, backends, variants, status, reason: **no differences**.

Discovery is genuinely by instrumentation (`oracle.py:208-266`: construct at batch 2, render, keep
the cell iff the counters fired; the discovered list and the registry are themselves compared). The
39 skips are optional-dependency or declared-no-CPU-backend reasons, identical in both arms. The
2-D families the Mac cannot import (CarRacing, LunarLander, MiniGrid/BabyAI, Sokoban, Overcooked,
Griddly, PGX) are CPU-uncovered here and are picked up automatically by the GPU arm.

Craftax in `se7-craftax-venv` (which now has tensordict/loguru):

```
MUJOCO_GL=glfw /Users/samib/se7-craftax-venv/bin/python scratch/step9/render-2d/oracle.py \
    --python /Users/samib/se7-craftax-venv/bin/python --only CraftaxClassic,CraftaxFull ...
"identical": true, "cells_ok": 2, "unexplained_differences": {}   ORACLE: PASS
```

## Timings — noise, not a cost

Committed: `timings_cpu_local.json` worst **1.0773** (SimplePointMaze, bar 1.10, `blocks_merge:
false`); `timings_cpu_local_vs_parent.json` worst **1.1156** (CraftaxFull, `blocks_merge: true`, but
with a 0.39 baseline-leg spread); `timings_cpu_local_selftime.json` (head vs head) worst 1.0073.

My independent paired runs, same harness, 5 alternating rounds × 5 reps, load average ~4:

```
bench.py --tag reviewA      (baseline 21cb306 vs head): worst 1.1287 = Roundabout (spread 0.259);
                             SimplePointMaze 1.0586, CraftaxFull 1.0051, PushT 0.936
bench.py --tag reviewFloor  (--arms head,head, SAME TREE): worst 1.1125 = PushT (spread 0.892)
```

The same-tree floor crossed 1.10 with **zero code difference**, and a different cell tops the table
in every run (SimplePointMaze → Roundabout → PushT → CraftaxFull). Nothing survives repetition, so
neither the 1.077 nor the vs-parent 1.1156 is a cost. That agrees with the mechanism: on CPU the only
changed code on a hot path is the `row_groups` rebinding (identical call graph, which the author
confirmed with cProfile) and, on the `gfxdraw` path, one extra Python call + tuple unpack per
*batched* `edge_setup` call (`gfxdraw.py:676-681`) — 8 calls per CartPole frame, measured at
1.009/1.013/0.990 across runs. `tile_shape` is Triton-path only and is not called on CPU at all.
Authoritative paired numbers remain the GPU arm's.

## Tests

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit -x \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
  -> 1405 passed, 195 skipped, 112 deselected, 26619 warnings in 83.72s
     resources: 16 EGL contexts created, 16 live (peak 16); renderers 15 created / 0 closed
        (all from the DM Control tests; unchanged by this branch)

MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_render_2d_shared.py tests/unit/test_gfxdraw_{blend_at_contract,bounds,select}.py \
  tests/unit/test_raster2d_torch_backend.py tests/unit/test_craftax_prng.py \
  tests/unit/test_swm_nav_render.py tests/unit/test_envs_classic_nav.py tests/unit/test_style.py
  -> 269 passed, 2 skipped in 2.64s
```

No `/oscar` path in any committed file of this delta except
`scratch/step9/render-2d/gpu_oracle.sbatch:9,34` (`--output=/oscar/scratch/sboughan/logs/...`,
`SCR=/oscar/scratch/sboughan`), which is a SLURM script and matches every existing
`scratch/step{3,4,6,7}/*.sbatch`. No `/oscar` in `stable_envs/render/` from this branch.
Dependency direction (contract §8) holds: `render/*` imports nothing from `envs/`, the core or SWM;
the one new edge is `navdot -> raster2d`, and `raster2d` does not import `navdot` (no cycle).

## Judgement — merge now, or wait for the GPU oracle?

**Merge now.** Ruling 8 removes the per-workstream GPU job, so this static reading is the pre-merge
gate on the fused kernels, and it is unusually strong for this particular patch:

1. **No Triton source line changed.** The kernel text of all five fused modules is byte-identical to
   the baseline; the kernels' only inputs from this patch are the constexpr twins and the edge table.
2. **The constexpr values are provably unchanged** — same `_c(...)` expressions over the same
   unchanged `EG_*`/`K_*`/`M32` ints, now imported rather than re-declared, with an AST check that
   every name the kernel loads is bound.
3. **The host-side table the kernels read is bit-identical**, measured against a real checkout of the
   baseline over 300 random quad geometries and 9 hand-built degenerate families (zero-length,
   exactly horizontal, exactly vertical, clipped on one and both sides, exact corners, dx == dy),
   with all four `kind` branches demonstrably reached — dtype, shape and all 16 columns.
4. **The tile law is exhaustively identical** over more than a million frame geometries, which is the
   only other quantity a kernel launch derives from this patch.

There is no path by which a fused kernel can see a different input or compile to different code.
The residual GPU risk is limited to import-time wiring (a name that exists only under Triton), which
items 2 and the AST check address, and which the post-Wave-1 GPU oracle will catch immediately. If
that oracle returns any 2-D difference, it lands on this branch and the findings' §13 recommendation
(paired semantic check before closure) applies.

## Defects (documentation only, none blocking)

* **G-1** `examples/results/step9/render-2d/findings.md:92` lists **MiniHack** under `tilemap` and
  among the families "the Mac cannot import". Neither is true: MiniHack imports and renders locally
  and the oracle classified it `not_a_2d_cell`, because it uses its own glyph-atlas gather
  (`stable_envs/envs/minihack.py:747`) and `stable_envs/envs/minihack.py` does not import
  `render/tilemap.py`. The instrumented discovery is right; the prose overstates the coverage claim.
  Same overstatement at `findings.md:255` and `findings.md:300`.
* **G-2** `findings.md:107` and `findings.md:365` state that `/Users/samib/se7-craftax-venv` cannot
  import `stable_envs`. That venv now has `tensordict` and `loguru`; I ran both Craftax cells there
  against the baseline and got a PASS with zero differences. Coordinator request #2 is satisfied and
  can be closed.
* **G-3** `examples/results/step9/render-2d/timings_cpu_local_vs_parent.json` carries
  `"blocks_merge": true` (worst 1.1156, CraftaxFull, baseline-leg spread 0.39). The flag is the
  harness's mechanical bar test, not a verdict, and the record is contradicted by the same cell at
  0.989/1.005 in the two other runs and by a same-tree floor of 1.1125 on this host. A reader who
  greps `blocks_merge` will reach the wrong conclusion; a one-line note in that record (or a
  `"noise": true` annotation) would be worth having. Not a reason to hold the branch.

Nothing above changes the verdict: **APPROVE**.
