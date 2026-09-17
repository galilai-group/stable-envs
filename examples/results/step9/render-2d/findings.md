# Step 9 — Workstream G (the 2-D rendering backends)

**Branch** `step9/render-2d` · **parent** `5befc460` (step9/integration, Wave 0 + Workstream H merged) ·
**Step-8 behavioural baseline** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc` ·
**host** Apple M5 Mac (`cis1b1zs2j3`), `/Users/samib/se6-venv` (torch 2.14 CPU, pygame, pymunk) and
`/Users/samib/se8-vm-venv` for the swm subset.

**Result: PATCH.** Six commits, 184 insertions and 108 deletions across the eight modules this
workstream owns, **no consumer or test call site touched**. Three laws that were each written out
twice in modules certified against one another pixel for pixel are now one definition each. On CPU:
**20 cells, 64 frozen states each, zero differences** in ten compared keys — including the cell list
itself, which is discovered by instrumentation rather than declared. Worst paired CPU timing ratio
**1.077** against a 1.10 bar, on a laptop whose load average was 9.8 because the other Step-9
workstreams were running there too; the one cell above 1.05 has a **provably identical call graph**
in the two trees (§8). The GPU arm is prepared and **not submitted**: `scratch/step9/render-2d/
gpu_oracle.sbatch`.

---

## 1. Files created / moved / removed

| file | what happened |
|---|---|
| `stable_envs/render/raster2d.py` | `_fit` + the tile-height expression became **`tile_shape(H, W, lanes, max_width)`**, the one tile law both 2-D fused kernels pick their tile by; `_TILE_WASTE_MAX` is `TILE_WASTE_MAX`; `_row_groups` is **`row_groups`** (public, with the old name as an alias — it has a caller outside the module); types on `render`, `render_torch`, `render_triton`, `_static_base`, `_destination`, `_tiles`, `_pixel_grid`. |
| `stable_envs/render/navdot.py` | `_tile` now calls `raster2d.tile_shape` instead of transcribing it; the docstring says which argument is the only thing that differs between the two kernels. |
| `stable_envs/render/gfxdraw.py` | The Wu walk's setup — `clip_line` plus the twelve constants — is **`edge_geometry`**, the one place that arithmetic is written down; `edge_setup` keeps its exact tuple and is now a re-ordering of it. Types on the twelve host-side functions, docstrings on `_round16` and `_clip_encode`. |
| `stable_envs/render/gfxdraw_fused.py` | `edge_table` calls `G.edge_geometry` instead of re-transcribing it; `M32` is `G.M32`; types on `bin_primitives`, `render_surface`, `shrink`. |
| `stable_envs/render/gfxdraw_classic.py` | `_blend_t` (a one-line pass-through to `gfxdraw.blend`) removed; the edge-column **constexpr twins come from `gfxdraw_fused`**, which had already made them, instead of being re-wrapped here; the module's three scattered imports of the fused layout are one import at the top; types on the two render entry points. |
| `stable_envs/render/scene.py` | The builder's broadcast contract is the **`Batched`** alias in every signature rather than a sentence in the class docstring; return types; four missing docstrings (`batch`, `n_prims`, `vertices`, `to`, `build`, `_enabled_n`). |
| `stable_envs/render/tilemap.py`, `prng.py` | Types on `blit`, `blit_torch`, `uniform`, `_counter`. Nothing else: both are already single-definition modules. |
| `tests/unit/test_render_2d_shared.py` | **new**, 20 tests — see §4. |
| `scratch/step9/render-2d/oracle.py`, `scratch/step9/render-2d/bench.py`, `scratch/step9/render-2d/gpu_oracle.sbatch` | **new** evidence harness. Spelled out rather than brace-expanded, because `test_every_cited_scratch_path_resolves` opens every cited path and a `{a,b}` glob is not one. |
| `examples/results/step9/render-2d/` | **new** artefacts: `oracle_cpu_local.json`, `timings_cpu_local.json`, `timings_cpu_local_vs_parent.json`, `timings_cpu_local_selftime.json`, `findings.md`. |

**Consumer call sites changed: none.** `git diff 5befc460 HEAD --stat` is eight files under
`stable_envs/render/`, one new test file, `scratch/step9/render-2d/` and
`examples/results/step9/render-2d/` — nothing else. The two renames both kept the old spelling:
`row_groups`/`_row_groups` are the same object (`envs/navigation.py:872` imports the private one and
still works), and `edge_setup` keeps its exact 14-tuple because `aaline_emit` and `polygon` unpack it
positionally. A rename with no alias would have cost edits in files this workstream does not own and
bought nothing the import line does not already say.

**Files NOT merged into their parent, deliberately — see §12.** Plan §3 names `scene.py`,
`primitives.py`, `raster2d.py`, `gfxdraw.py` and `tilemap.py` as the target files. `gfxdraw_classic.py`
and `gfxdraw_fused.py` are the two CUDA fused kernels the brief says to keep as private modules or
sections, `navdot.py` is a third, and `prng.py` is not a renderer at all. None of their numeric
expressions is character-identical to anything in a parent module, so none of them qualified for the
merge the brief permits.

## 2. API changes

Additive, plus two renames that both kept the old name.

* New: `raster2d.tile_shape`, `raster2d.TILE_WASTE_MAX`, `raster2d.row_groups`,
  `gfxdraw.edge_geometry`, `scene.Batched`.
* Renamed, with the Step-8 spelling kept as an alias to the same object: `raster2d._row_groups`
  → `row_groups`. Renamed with no alias because nothing outside the module ever used it:
  `raster2d._fit` (absorbed into `tile_shape`), `raster2d._TILE_WASTE_MAX` → `TILE_WASTE_MAX`,
  `gfxdraw_classic._blend_t` (removed; it was `gfxdraw.blend`).
* Removed: nothing that had a caller. Signatures: annotations only; no argument added, removed,
  reordered or defaulted differently.
* `render/` is internal (Wave-0 contract §1 puts nothing from it at top level) and still imports
  nothing from `envs/`, the core or SWM. `navdot` now imports `raster2d`, which is the only new
  edge in the dependency graph and points the same way everything else in `render/` does; the
  parametrised `test_a_renderer_knows_nothing_about_environments_or_stable_worldmodel` passes.

## 3. Step-8 IDs affected

Every cell that reaches one of the eight modules. **Measured, not read off the imports**: every
registered config is constructed and rendered with 24 entry points wrapped in counters, and these are
the ones that called something (the same probe runs inside each oracle arm, and the call set is part
of the comparison).

| SWM ID | cell(s) | module | covered on |
|---|---|---|---|
| `swm/PushT-v1`, `swm/PushT-Discrete-v1` | PushT | `scene` + `raster2d` | cpu ✓, cuda (GPU arm) |
| `swm/SimplePointMaze-v0` | SimplePointMaze | `scene` + `raster2d` | cpu ✓ |
| `swm/TwoRoom-v1` | TwoRoom, TwoRoomTarget | `navdot` + `raster2d` | cpu ✓ |
| `swm/Piecewise-v0` | Piecewise, PiecewiseZones | `navdot` + `raster2d` | cpu ✓ |
| `swm/CartPoleControl-v1` | CartPole | `gfxdraw` (+ `gfxdraw_classic` on cuda) | cpu ✓ |
| `swm/AcrobotControl-v1` | Acrobot | `gfxdraw` | cpu ✓ |
| `swm/MountainCarControl-v0` | MountainCar | `gfxdraw` | cpu ✓ |
| `swm/MountainCarContinuousControl-v0` | MountainCarContinuous | `gfxdraw` | cpu ✓ |
| `swm/PendulumControl-v1` | Pendulum | `gfxdraw` (+ the sprite blit) | cpu ✓ |
| `swm/CraftaxClassicPixels-v1` | CraftaxClassic | `prng` | cpu ✓ |
| `swm/CraftaxPixels-v1` | CraftaxFull | `prng` | cpu ✓ |

That is **12 of the 42 Step-8 SWM IDs** (13 counting `PushT-Discrete`). Plus the non-42 2-D families
this workstream also serves: **Crafter** (`tilemap`) and the six **highway** configs — Highway,
Intersection, Merge, Parking, Racetrack, Roundabout (`scene` + `raster2d`) — all covered on cpu; and
**CarRacing** (`gfxdraw_fused`), **LunarLander** (`gfxdraw`), the six **MiniGrid** configs plus
**BabyAI**, **Sokoban**, **Overcooked**, **Griddly** and **MiniHack** (`tilemap`), none of which the
Mac can import. They are the GPU arm's job, and the oracle picks them up **automatically** because it
discovers its cells rather than reading a list.

Not affected and explicitly not touched: every 3-D cell (Workstreams H and I), `rocket_landing`
(`nvdiff`), `env.py`, and every family's own logic.

## 4. Targeted tests

| suite | venv | result |
|---|---|---|
| `tests/unit -x --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"` | se6-venv | **1405 passed**, 195 skipped, 112 deselected (104 s) |
| the same, before this branch's test file | se6-venv | **1385 passed** — the brief's baseline, reproduced |
| `test_swm_nav_render`, `test_swm_matrix`, `test_envs_classic_nav`, `test_raster2d_torch_backend`, the three `test_gfxdraw_*`, `test_foundation_pusht`, `test_style` | se8-vm-venv | **265 passed**, 1 skipped |
| `test_render_2d_shared.py` | se8-vm-venv | **20 passed** |
| the Craftax subset | se7-craftax-venv | **BLOCKED, not a failure**: that venv has neither `loguru` nor `tensordict`, which Wave 0 made core dependencies, so `import stable_envs` fails before any test runs. Nothing was installed. The two Craftax cells are covered instead in se6-venv, where they render and are byte-identical (§6), and the GPU arm runs them again in `$SCR/venvs/se7-craftax-gpu`. **Coordinator item for Workstream L.** |

1385 → 1405 is exactly the 20 new tests; a collected-ID diff confirms nothing disappeared.

The new tests pin the specific way each consolidation could come apart — not "the two agree today",
because two copies of an expression agree today and that is why nobody noticed:

* `navdot._tile` **is** `raster2d.tile_shape` at its own lane budget, over 324 frame geometries, and
  the waste bound holds as a property (power of two, within `TILE_WASTE_MAX`, lanes ≤ 1024);
* `edge_table`'s twelve shared columns **are** `edge_setup`'s tuple, edge for edge, over quads that a
  second test **asserts reach all four edge kinds and both sides of the clip** — a generator that
  quietly stopped producing vertical edges would otherwise leave the comparison passing on a third of
  the rule;
* `M32` is one constant and `row_groups`/`_row_groups` are one object, both **by identity** (a second
  copy would pass a value comparison, which is the failure mode being removed);
* `gfxdraw_classic` has no second name for the blend, `raster2d` has no private twin of the width fit,
  and the builder's `Batched` contract is on the methods.

## 5. Certification checks

Nothing in this patch touches a threshold, a declaration, a tier, a frozen digest or a corpus. The
applicable harnesses are `tests/gpu/test_gfxdraw_classic.py` (which diffs the classic-control Triton
kernel against `render_classic_torch` and diffs THAT against `gfxdraw`'s per-primitive path — both
joints of the chain this patch consolidated), `test_gfxdraw_fused.py` (the CarRacing surface kernel
against the per-primitive path on random polygons), `test_navdot_fused.py`, `test_fast_paths.py`
(which diffs the binned scene kernel against the unbinned one on the environments themselves) and
`test_conformance.py`. All five are in the GPU arm, §9, and the selection is written into the sbatch.

## 6. Default-frame comparison (the oracle)

`scratch/step9/render-2d/oracle.py` → `examples/results/step9/render-2d/oracle_cpu_local.json`.

Both arms run in the same venv on the same host, one with a detached `git worktree` of `21cb306` on
`PYTHONPATH` and one with this branch, through the public path only (`cls(batch=n, device=...)`,
`set_state`, `render()`), over the **frozen corpora** — `examples/results/step4/states/*.npz` for
PushT, SimplePointMaze, the five classic-control cells and the four swm_nav cells;
`examples/results/step7/craftax-classic-full/corpus/states.npz` and
`examples/results/step7/craftax-full/corpus/states.npz` for the two Craftax cells; seeded
`sample_states` for Crafter and the six highway configs, which never had one — 64 states each.

> **CraftaxClassic's Step-4 corpus is stale and this is the first place it shows.** It is 127 wide
> and the cell has been 149 wide since Step 7 widened it, so `set_state` refuses it. The oracle uses
> the Step-7 corpus, which is frozen, 64 states and the right width. Both arms are handed the same
> file, and the record carries its digest. Recorded here because a later reader will reach for the
> Step-4 path first, as this workstream did.

**20 cells, 64 states each, zero differences** in any compared key: the frame SHA-256, the frames
rendered from explicit rows, the state digest each arm was handed, the frame shape, the registered
size, the skip reason where a cell was skipped, and the *set of 2-D entry points the cell actually
called*. The other 39 registered cells were skipped in both arms with byte-identical reasons
(`ogbench`, `minigrid`, `gymnasium_robotics` absent; the two OGB cells have no CPU backend by
declaration).

Two choices in the harness are worth recording because both were arrived at by being wrong first:

* **The cell list is discovered, and the discovery is compared.** A hand-written list cannot cover
  what the Oscar venv can import and the Mac cannot, and a cell that quietly stops reaching a backend
  would drop out of a declared list silently. Here a cell present in one arm and not the other is a
  difference, and so is the registry itself.
* **`raster2d.render` and `scene.SceneBuilder` are deliberately NOT instrumented.** Both are bound by
  name at module import time in a consumer — the Step-8 `core/env.py` does
  `from ..render.raster2d import render as render_scene` at module scope, and Wave 0 moved that
  import inside the method — so wrapping them reports a difference between the two trees' **import
  style** rather than between their pictures, which is exactly what the first run of this oracle
  reported. Nothing is lost: `render` only dispatches to `render_torch`/`render_triton`, which it
  reaches through its own module globals at call time, and a scene that was not built cannot be
  rasterized.

**The non-fused paths.** Five of these modules ship a fused CUDA kernel and a torch expression it must
equal. On `cpu` the torch expression IS what runs, so all 20 cells above exercise it by construction —
`render_torch` (not `render_triton`) for every Scene2D cell, `gfxdraw`'s per-primitive Wu/scanline
path (not `gfxdraw_classic`'s kernel) for the five classic-control cells, `tilemap.blit_torch` for
Crafter, and `navigation.py`'s torch blend (not `navdot.blend_dots`) for the four swm_nav cells. On
`cuda` the oracle renders each cell a second time through whichever of `backend_torch`, `cull_off`,
`cull_on`, `gfx_per_primitive`, `tilemap_torch` and `navdot_torch` its call set says it has; those
digests are in the GPU arm's record and are compared like any other key.

## 7. Variation / state comparison

Untouched by construction: none of the eight modules knows what a variation is. Colours, radii, the
anti-alias gate and the per-environment background arrive as tensors and pass through expressions
this patch did not modify, and the oracle's per-cell run uses each env's registered default style,
whose frames are identical. `SceneBuilder`'s `_style` hook, its `mark_static` refusal and
`raster2d._static_base`'s sharedness measurement are unchanged; `test_style.py` and
`test_swm_matrix.py` pass in both venvs.

## 8. Performance

`scratch/step9/render-2d/bench.py` → `timings_cpu_local.json`, `timings_cpu_local_vs_parent.json`,
`timings_cpu_local_selftime.json`. Seven **alternating** baseline/head rounds (a before-block followed
by an after-block measures the machine's mood; interleaving does not), five timed `render()` calls per
leg, separate interpreter per leg, single-threaded, frozen states, at the registered frame sizes.

Against the **Step-8 baseline `21cb306`** (which is what the contract asks for, and therefore includes
Wave 0 and Workstream H as well as this patch):

| cell | px | batch | hot path | baseline ms | head ms | ratio | same-arm spread |
|---|---|---|---|---|---|---|---|
| PushT | 96 | 64 | `raster2d`, static bake | 18.68 | 17.93 | **0.960** | 3.5 % |
| SimplePointMaze | 500 | 16 | `raster2d` | 18.27 | 19.68 | **1.077** | 4.3 % |
| Roundabout | 600 | 8 | `raster2d`, 436 primitives | 142.54 | 142.51 | 1.000 | 2.7 % |
| TwoRoom | 224 | 32 | `navdot` window + `raster2d` | 6.99 | 6.95 | 0.995 | 3.8 % |
| CartPole | 400x600 | 16 | `gfxdraw` per-primitive | 13.75 | 13.87 | 1.009 | 2.9 % |
| Acrobot | 500 | 16 | `gfxdraw` | 17.78 | 17.66 | 0.994 | 11.4 % |
| Pendulum | 500 | 16 | `gfxdraw` + sprite blit | 13.88 | 13.77 | 0.993 | 3.5 % |
| Crafter | 64 | 32 | `tilemap` blit | 4.68 | 4.66 | 0.994 | 2.8 % |
| CraftaxClassic | 144 | 32 | `prng` + atlas | 17.56 | 17.52 | 0.998 | 5.5 % |
| CraftaxFull | 130x110 | 32 | `prng` + atlas | 7.57 | 7.48 | 0.989 | 4.2 % |

**Worst 1.077 against the 1.10 bar. No blocking regression.** Against the branch **parent** `5befc460`
(which isolates this patch from Wave 0 and H) the picture is the same: worst 1.116 on CraftaxFull —
with a 39 % same-arm spread, and 0.989 in the run above — and SimplePointMaze again at 1.074.

**The one cell worth a paragraph is SimplePointMaze**, which came back at 1.084, 1.081, 1.045, 1.077,
1.060 (15 rounds) and 0.997 across six runs. It is above the 1.05 the other nine sit in, and it is the
only ratio here that survived repetition, so it was chased rather than averaged away:

* **cProfile over 20 renders in each tree gives identical call graphs** — the same functions, the same
  call counts, the only differences being `_row_groups` → `row_groups` (20 calls in both) and the
  address of a built-in `__new__`. **No python work was added**, which is what a structural regression
  would look like.
* The time is in `torch.where` — 1,120 calls in both trees, the same shapes, 0.212 s against 0.227 s.
  That is per-op wall time on identical work.
* The same-tree floor on the same cell, measured by `bench.py --arms head,head` in the same
  alternating shape, is **1.003 to 1.045** depending on the run.
* **This laptop is shared.** `uptime` during the measurements read a load average of **9.8**, because
  the other Step-9 workstreams run their suites on it. The swapped-arms run
  (`--arms head,baseline`) landed on a moment with 36–65 % same-arm spreads and reported 0.818 on this
  cell, i.e. the machine can produce a 20 % swing in either direction on demand.

So: nothing was added to the render path for this cell, the ratio does not reproduce in one run out of
six, and the bar was never crossed. **The authoritative paired numbers are the GPU arm's** — same
card, same job, one tenant — and they are §9's.

## 9. GPU arm — prepared, NOT submitted

`scratch/step9/render-2d/gpu_oracle.sbatch`: `--partition=gpu --qos=norm-gpu --gres=gpu:1
--cpus-per-task=4 --mem=48G --time=01:00:00`, job name `se9-render-2d-oracle`, log to
`$SCR/logs/%x_%j.out`, tree at `$SCR/se9/render-2d` (**must be a real git clone whose history contains
`21cb306`** — both scripts `git worktree add --detach` it and fail loudly if it is missing, rather than
silently comparing the branch against itself). Four stages in the one job on the one card:

1. **the same oracle, `--devices cpu,cuda`, 64 frozen states per cell**, venv
   `$SCR/venvs/se-step6/bin/python`. This is where the fused kernels are actually diffed:
   `raster2d`'s binned scene kernel, `gfxdraw_classic`'s classic-control kernel, `gfxdraw_fused`'s
   CarRacing surface kernel, `tilemap`'s blitter and `navdot`'s two passes — each against the Step-8
   tree AND against its own torch expression through the `variants` above. Because the cell list is
   discovered, this stage also covers CarRacing, LunarLander, the six MiniGrid configs, BabyAI,
   Sokoban, Overcooked, Griddly, PGX and MiniHack, whichever the venv can import.
2. **the two Craftax cells in `$SCR/venvs/se7-craftax-gpu`**, which is the only interpreter where that
   family imports.
3. **paired alternating GPU timings** (`bench.py --set gpu`, 5 rounds x 5 reps, both legs on this card
   in this job), plus a short `--arms head,head` run so the same-tree floor is measured beside them
   rather than assumed.
4. **`tests/gpu/test_gfxdraw_classic.py test_gfxdraw_fused.py test_navdot_fused.py
   test_fast_paths.py test_conformance.py -k "<this workstream's cells and modules>"`**, with the
   **whole** pytest output to `$OUT/gpu_tests_<tag>.log` and only the summary echoed. That is
   Workstream H's lesson, taken rather than repeated: H's job 6464319 piped pytest into `tail -40` and
   MuJoCo's EGL teardown — dozens of `Exception ignored in __del__` lines printed *after* pytest's
   verdict — pushed the summary out of the window, so the run produced no test counts at all.

**Expected pre-existing failures, not this branch's.** Workstream H's §9c established that every
`tests/gpu` selection currently fails for renderer-only cells through
`testing/render_worker.py` → `rollout.py`'s `has_dynamics` guard, a Wave-0 item since fixed at
`e1da2f6`/`430bb6d`; and that missing optional packages (`warp`, `gymnasium_robotics`, `minigrid`)
produce collection errors. Neither is in this workstream's files.

## 10. Resource lifetime

None of the eight modules owns a process resource: no EGL context, no CUDA context, no file handle,
no MuJoCo renderer. What they own is:

* **`raster2d._BASE_CACHE`**, the baked static-prefix images — keyed on the primitives' raw bytes, the
  resolution AND the device, capped at `_BASE_CACHE_MAX = 32` and cleared wholesale on overflow.
  Unchanged by this patch.
* **`gfxdraw.Gfx`**, a per-(size, device) constant cache owned by the env that made it and released by
  refcount. Unchanged.
* **`tilemap.MiniGridAtlas`**, a per-(tile size, device) atlas with an on-disk cache. Unchanged.
* **`gfxdraw_classic.DiscTables`** and `Ops`, built per frame list and released by refcount.

Nothing here needs a `close()` and nothing depends on `__del__`, so plan §2's resource rule is
already satisfied and Workstream I's work does not extend to these files. The unit run's safety
plugin reports the same EGL ledger before and after (16 created / 16 live, all from the DM Control
tests, none from these modules).

## 11. Unexplained oracle differences

**CPU: zero** — 20 cells, 64 frozen states each, ten compared keys, including the discovered cell list
and the backend call set. **GPU: not yet measured**; the arm is §9's and the coordinator submits it.

Uncovered on cpu and covered elsewhere rather than omitted: the 23 cells whose optional dependency is
absent from every local venv (`ogbench`, `minigrid`, `gymnasium_robotics`) and the 2-D families the
Mac cannot import at all (CarRacing, LunarLander, MiniGrid, BabyAI, Sokoban, Overcooked, Griddly, PGX,
MiniHack) — all of them in §9 stage 1. Nothing was installed.

## 12. Deliberate non-consolidations (plan §16, recorded so they are not re-proposed)

* **`gfxdraw_classic.py` and `gfxdraw_fused.py` were not merged into `gfxdraw.py`.** They are two
  Triton kernels over two different primitive alphabets — five classic-control ops over a flat
  background or a pre-baked plate, against quads plus `pygame.draw` polygons with a screen-bin
  pre-pass — and their per-pixel code is not character-identical to anything in `gfxdraw.py` or to
  each other. The brief permits a merge only where every numeric expression is identical, and says
  the fused kernels stay as separate code paths. What WAS shared between them is now shared: the edge
  law, the blend mask, the constexpr column layout.
* **`gfxdraw.blend` (torch) and `gfxdraw_fused._blend` (Triton) stay separate.** Same rule, two
  languages, and the Triton one deliberately widens to int64 for one short-lived value rather than
  keeping an int64 canvas (27.7 ms against 12.4 on an 8x128 tile, recorded in its docstring). The
  same is true of the two `_round16`s. They now at least share `M32`.
* **`gfxdraw.fill_polygon_mask` and `gfxdraw_classic._poly_torch`'s scanline fill** are the same
  16.16 rule at different shapes: one reduces over an edge axis with `amin`/`amax` on `(B,n,H)`, the
  other accumulates `minimum`/`maximum` per edge over `(B,H,W)`. Different reduction order is
  entitled to a different last bit, and both are certified against pygame. Not a place to be tidy.
* **The `try: import triton / except ImportError: HAVE_TRITON = False` idiom in `raster2d`,
  `gfxdraw_fused`, `tilemap` and `navdot`** looks like four copies of one thing and is not: each
  module's flag is independently flipped, by its own tests and by this workstream's oracle, to force
  the torch expression behind that kernel and only that one. Folding them into a shared flag would
  remove exactly the control the non-fused comparison in §6 depends on.
* **`navdot.available(device)` and `raster2d.render`'s inline "Triton importable AND the scene is on
  cuda"** are the same question about two different Triton imports, for the same reason.
* **`SceneBuilder._b`/`_bn` and `_enabled`/`_enabled_n`** are the one-primitive and n-primitive forms
  of one broadcast. Merging them is safe in principle (they are reshapes, not arithmetic) but would
  change the reshape path on the one-primitive form, and the module docstring records that an
  `einsum` or a `norm` over an extra axis is entitled to a different last bit — one ulp on a vertex is
  a pixel wherever that vertex lands on a boundary. Ten lines saved is not worth finding out.
* **The kernel/oracle pairs** — `render_triton`/`render_torch`, `blit`/`blit_torch`,
  `visibility_scan`/`visibility_scan_torch`, `render_classic`/`render_classic_torch` — are not
  duplication. Each readable twin exists to be diffed against its kernel, and an oracle that shares an
  optimisation with the thing it checks is not one.

## 13. Semantic follow-up recommendation

**None.** No pixel moved on any of the 20 CPU cells or on either path behind any fast-path switch, so
there is nothing for a DINO tripwire or a semantic smoke to be sensitive to. If the GPU arm returns a
difference, that cell needs a paired semantic check before merge and this recommendation changes.

## 13a. Provenance of the records

`oracle_cpu_local.json` carries `head_sha de0ba10` (the findings commit). The three timing records
carry `cd97cd3`, the commit before it; `git diff cd97cd3 HEAD -- stable_envs tests` is empty, so they
describe the code at HEAD.

## 14. Commits

```
05b3fc0  render 2-D: one tile-geometry law for both fused kernels
f40eba2  gfxdraw: one Wu edge-setup law behind all three anti-aliased paths
f02a4d4  render 2-D: name the builder's broadcast contract, publish row_groups
d722235  render 2-D: types on every host-side entry point, two missing docstrings
dfc7cdc  step9 render-2d: oracle, paired timing harness, GPU sbatch, CPU records
cd97cd3  render 2-D: tests for the three consolidated laws
<this>   step9 render-2d: findings
```

## 15. Requests for the coordinator

1. **Submit `scratch/step9/render-2d/gpu_oracle.sbatch`** and return the job id. It needs
   `$SCR/se9/render-2d` to be a real clone containing `21cb306`, and `$SCR/venvs/se7-craftax-gpu` for
   stage 2.
2. **`/Users/samib/se7-craftax-venv` cannot import `stable_envs` any more** — Wave 0 made `loguru` and
   `tensordict` core dependencies and that venv has neither. Every workstream whose brief names that
   venv will hit this. A Workstream L item; nothing was installed here.
3. **`examples/results/step4/states/CraftaxClassic.npz` is stale** (127 wide against the cell's 149
   since Step 7). It is frozen and read-only, so this is a note rather than a change: anything still
   citing it as CraftaxClassic's corpus should cite
   `examples/results/step7/craftax-classic-full/corpus/states.npz` instead.
4. **Ownership note, no edit made:** `envs/navigation.py:872` imports `raster2d._row_groups`, a
   private name, from inside a function. The public `row_groups` now exists and the alias keeps the
   call site working; whoever owns `navigation.py` next (Workstream A or K) can switch the one line.
