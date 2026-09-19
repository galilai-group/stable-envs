# Step 9 — Workstream I (MJWarp / nvdiffrast / renderer resource ownership)

**Branch** `step9/render-mjwarp-nvdiff` · **original parent** `b3f3a81` · **current integration
merged through** `d3cb4a8` · **HEAD:** use `git rev-parse HEAD` (this findings update follows the
review-fix commit `6a8f3b8`) ·
**Step-8 behavioural baseline** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc` ·
**host** Apple M5 Mac (`cis1b1zs2j3`), `/Users/samib/se6-venv` (torch 2.14 CPU, mujoco 3.12.0,
dm_control 1.0.45, no CUDA, no warp, no nvdiffrast, no mujoco_warp).

**Result: PATCH — GPU pixels established, job exit is classified.** Explicit resource ownership is in
place across all four backends plus the new `render/_resources.py`; the static read-through shows
**not one numeric expression, upload order or cache key changed**. CUDA evidence is job **6509711**
(RTX A6000, `gpu1501`, HEAD `fc05512`) after the `bash -e` harness abort on **6507919**:

| arm | frames vs 21cb306 | lifetime | worst bench ratio (bar 1.10) |
|---|---|---|---|
| vis311 DMC trackcom (5 cells) | identical | ok | 0.999 |
| OGB Cube/Scene | identical | ok | 1.006 |
| Rocket / nvdiff (3 cells) | identical | ok | within bar |
| Fetch (4 cells) | identical | ok | 1.017 |

Direct probes: `mjwarp` PASS, `nvdiff` PASS (ray-cache LRU overflow). `rendernet` failed because the
probe constructed `LoadedCorrector` around a **CPU** `ESPCN` (`cuda` input vs CPU weights). Production
`load` / `unsealed` move weights; the probe now calls `unsealed`. Broad `tests/gpu` in family venvs
still fail on missing optional packages (`warp` / `gymnasium_robotics`) and Fetch's venv has no
`pytest` — classified as harness/venv, not pixel regressions. 2D and 3D sections of `6507919` already
passed frames/timings (3D worst ratio 1.003).

Plan §17 unexplained oracle differences on the cells this workstream can construct: **zero**.

### Independent-review response (`review_i.md`)

The first review was `APPROVE-WITH-CHANGES` because the committed GPU instrument could print a
pass without exercising or validating release. Commit `6a8f3b8` closes that false-green path:

- release failures are counted, recorded and logged; they are not counted as successful closes;
- cache callback failures remain non-throwing at an unrelated insertion but are observable;
- the old `_TEX_KEEP.extend` fallback remains non-refusing past the retention tripwire;
- `Scene.clip` and renderer setup/convenience calls reject use after close;
- a CPU test constructs and releases a real `Scene`, and every construction path pins `_opened()`;
- `oracle.py` compares backend calls, requires all selected cells and backend modules, checks the
  exact closed-env error and release-failure counters;
- `bench.py` rejects empty selections, preserves baseline and head memory separately, and exits
  nonzero above the 1.10 gate;
- `gpu_oracle.sbatch` aggregates every return code into one verdict, treats missing coverage as a
  failure, runs explicit GPU test modules, redirects the rendernet cache, and runs direct lifetime
  probes over all five backend objects plus nvdiff cache overflow.

The CUDA result remains owed; this section records that the measuring instrument can now fail for
the defects it is supposed to detect, not that it has already passed.

---

## 1. The design, in ten lines

1. Resource lifetime is explicit and central: `stable_envs/render/_resources.py`, stdlib only.
2. Three kinds of resource, three different correct answers — the whole design is that they differ.
3. **Exclusively-owned device memory** is released on `close()`: `Closeable` gives idempotent
   `close()`, `with`, a `_release()` hook and `_check_open` → `UseAfterCloseError`.
4. **Shared cached scenery** is released by its *cache*, never by a user of it: `BoundedCache`
   (capacity, LRU, one eviction at a time, optional `on_evict` release).
5. **An EGL/GL context is never released at all**, deliberately — `testing/resource_guard.py`
   records four incidents where freeing one silently corrupted every other live context.
6. So `env.close()` frees the env's device buffers and makes it refuse to render; it does **not**
   return the process to zero EGL contexts, and the findings say so where a reader would assume it.
7. **No `__del__` anywhere.** An unclosed object leaks visibly and countably (`_resources.live`),
   which is a better failure than a nondeterministic one that runs after the CUDA bindings are gone.
8. Five backend objects are now `Closeable`: `mjwarp.BatchRenderer`, `nvdiff.Scene`,
   `nvdiff.BatchRenderer`, `rendernet._Runner`, `rendernet.LoadedCorrector`.
9. Ownership that was implicit is now stated and settable: `nvdiff.BatchRenderer.owns_scene`.
10. Two module-level caches stop being unbounded or cliff-edged: `mjwarp_lightspec._CASTS_CACHE`
    (was an unbounded dict) and `nvdiff.BatchRenderer._ray_cache` (was `clear()`-on-full).

---

## 2. Files created / changed

| file | what happened |
|---|---|
| `stable_envs/render/_resources.py` | **new**. `Closeable`, `BoundedCache`, `UseAfterCloseError`, `retain_process_lifetime`, `counts()/live()/delta()`, and observable release-failure accounting. A failed `_release` is not counted closed; cache callback failures remain non-throwing but are counted, recorded and logged. `close()` is thread-safe. Imports stdlib only. |
| `stable_envs/render/mjwarp.py` | `BatchRenderer` is `Closeable` (`_release` drops graph → context → `Data` → `Model`); `render` guards with `_check_open`; the anonymous `_TEX_KEEP` leak becomes a counted `retain_process_lifetime`; the compatibility fallback warns and retains past the tripwire because the old list never refused construction; new `release_render_context(rc)`. |
| `stable_envs/render/mjwarp_lightspec.py` | `_CASTS_CACHE` becomes a `BoundedCache(64)`. |
| `stable_envs/render/nvdiff.py` | `Scene` and `BatchRenderer` are `Closeable`; `owns_scene` records and can be set; `_ray_cache` becomes a `BoundedCache(8)`; `Scene.clip` and every public renderer entry point guard with `_check_open`. |
| `stable_envs/render/rendernet.py` | `_Runner` and `LoadedCorrector` are `Closeable`; the captured CUDA graph is `reset()` before its static buffers are dropped. |
| `tests/unit/test_render_resources.py` | **new**, now 33 IDs (32 pass, 1 skips without `mujoco_warp`): failed-release accounting, concurrent close, both `Scene` constructors calling `_opened`, and CPU construction/release of every owned `Scene` tensor. §5. |
| `scratch/step9/render-mjwarp-nvdiff/{oracle.py,bench.py,resource_probe.py,gpu_oracle.sbatch}` | **new** evidence harness. `resource_probe.py` directly constructs/closes all five backend objects across compatible GPU venvs and exercises ray-cache overflow; none of the CUDA arms has run yet (§7). |
| `examples/results/step9/render-mjwarp-nvdiff/findings.md` | this file. |

**No file outside those was touched.** Nothing removed, nothing renamed, no public symbol deleted:
every change is additive except the two cache objects, whose call sites are inside the same files.

---

## 3. The resource-ownership table

| object | owner | created where | released where | tested by |
|---|---|---|---|---|
| MJWarp `Model` (`put_model`) | `mjwarp.BatchRenderer` | `BatchRenderer.__init__` | `BatchRenderer._release` (`self.m = None`) | shape test + GPU cycles arm |
| MJWarp `Data` (`put_data(nworld=B)`) | `mjwarp.BatchRenderer` | `BatchRenderer.__init__` | `BatchRenderer._release` | shape test + GPU cycles arm |
| `mjw` render context (`rgb_data`, per-world geom buffers, textures) | whoever called `create_render_context` — `BatchRenderer`, **or a consumer's `_CTX_CACHE`** | `mjwarp.create_render_context` | `BatchRenderer._release` → `release_render_context`; for a cached one, **the cache** on eviction (§4) | GPU cycles arm; the consumer half is §4 and not yet made |
| `tex_lod` downsampled textures (torch, given to `wp.from_torch`) | the context they are attached to (`rc._se_tex_lod_keep`) | `create_render_context` | `release_render_context` clears the attribute | `test_retention_past_the_cap_raises…` (the fallback path) |
| …same, when the context refuses the attribute | **the process, deliberately** | `retain_process_lifetime(..., TEX_KEEP_REASON, overflow="retain")` | never — counted; warns past 256 but preserves the old unbounded keepalive behavior rather than freeing live foreign storage | both retention-cap tests |
| captured warp graph (`ScopedCapture`) | `mjwarp.BatchRenderer` | `BatchRenderer._capture` | `_release`, **first**, before the buffers it replays into | GPU cycles arm |
| `light_castshadow` memo | `mjwarp_lightspec._CASTS_CACHE` (module, process-scoped) | `light_casts` | LRU eviction at 64 keys; nothing device-side to free | `test_the_module_level_cast_memo_is_bounded`, `test_a_key_in_use_survives_an_unrelated_insertion` |
| nvdiffrast `RasterizeCudaContext` | `nvdiff.BatchRenderer`, always | `BatchRenderer.__init__` | `_release` (`self.ctx = None`), last | shape test + GPU cycles arm |
| `nvdiff.Scene` (triangle soup, sparse transforms, appearance, texture pyramids) | `BatchRenderer` **iff `owns_scene`**, else the caller | `Scene.__init__` / `Scene.from_triangles` | `Scene._release`; via `BatchRenderer._release` only when `owns_scene` | CPU construction closes and verifies all 16 tensor attributes; GPU probe measures the CUDA path |
| skybox texture pyramid | `nvdiff.BatchRenderer` | `set_skybox` | `_release` | GPU cycles arm |
| ray-direction grids | `BatchRenderer._ray_cache` (per renderer) | `_ray_dirs` | LRU eviction at 8; all of them on `_release` | `test_the_per_renderer_ray_cache_is_bounded`, LRU tests |
| shadow hook (a closure, usually capturing a second renderer) | the caller that installed it | consumer (`mujoco_locomotion._ShadowPass`) | `BatchRenderer._release` drops the reference | — (consumer-side) |
| corrector weights (`Corrector` module) | `rendernet.LoadedCorrector` | `rendernet.load` | `LoadedCorrector._release` | shape test |
| corrector CUDA graph + static in/out buffers | `rendernet._Runner` | `_Runner.__init__` (on first use) | `_Runner._release`: `graph.reset()`, then the buffers | shape test; GPU arm for the RoboCasa corrected cell |
| **EGL / GL context** | **the process** | mujoco/dm_control, via a consumer's reference env or plate bake | **never** | `test_nothing_in_the_render_backends_releases_an_egl_context` (source scan, comments stripped) + the GPU cycles arm asserting `context_freed == 0` |

---

## 4. Per-consumer change list — the exact one line each owner must make

No consumer file was edited: C (Fetch), D (OGBench), E (Rocket), F (MuJoCo wrappers) and B
(DMControl) are running, and `envs/*` is theirs.

**First, the finding that shapes this list.** These eight families hold **no per-instance backend
handle at all** — measured, not assumed: `grep -nE "self\._[a-z_]+ = (torch|np\.|_|[A-Z])"` over
`dmcontrol, fetch, ogb_manip, robocasa, rocket_landing, mujoco_locomotion, humanoidbench, metaworld`
returns two RoboCasa index tensors and nothing else. Every renderer, context, scene and corrector
lives in a **module-level cache keyed by `(cell, res, nworld, device)` and shared by every env with
that key**. So an env "releasing what it acquired" would free a context another live env is still
rendering through, and **no `_close` is needed for the env to refuse to render**: `env.close()` sets
`_closed`, and the shared `render` wrapper already calls `_check_open` (`env.py:237`).

**What explicit ownership does require** is that those caches release on eviction, because their
values are now `Closeable` and an eviction that merely drops the reference leaves the free to the
collector — which is the dependency plan §2 forbids, one level up from where it was fixed.

| file | line | the change (one line, inserted immediately **before** the existing `clear()`) |
|---|---|---|
| `envs/fetch.py` | 453 | `for _c in _CTX_CACHE.values(): MW.release_render_context(_c.rc)` |
| `envs/ogb_manip.py` | 471 | `for _c in _CTX_CACHE.values(): MW.release_render_context(_c.rc)` |
| `envs/robocasa.py` | 336 | `for _c in _CTX_CACHE.values(): MW.release_render_context(_c.rc)` |
| `envs/humanoidbench.py` | 520 | `for _c in _CTX_CACHE.values(): MW.release_render_context(_c.rc)` |
| `envs/metaworld.py` | 760 | `for _c in _CTX_CACHE.values(): MW.release_render_context(_c.rc)` |
| `envs/dmcontrol.py` | 2631 | `for _c in _TC_CACHE.values(): MW.release_render_context(_c.rc)` |
| `envs/mujoco_locomotion.py` | 1640 | `for _c in _NVD_CACHE.values(): _c.br.close()` |
| `envs/rocket_landing.py` | 729 | `for _c in _NVD_CACHE.values(): _c.br.close()` |

Two of those eight carry a second, equally one-line rider:

* `envs/rocket_landing.py:739` — after `c.br = ND.BatchRenderer(None, ..., scene=scene, ...)`, add
  `c.br.owns_scene = True`. It is the only call site in the repository that builds a `Scene` purely
  to hand it to one renderer and keeps no other reference to it, so without this line the
  `br.close()` above drops the scene's tensors to the collector instead of freeing them. The default
  is deliberately the other way round (a renderer must never close a scene its caller still uses);
  the attribute exists to be set, and `nvdiff.BatchRenderer`'s docstring says so.
* `envs/robocasa.py:926-928` — `_CORRECTOR_CACHE` is an **unbounded dict** keyed by class name and
  therefore never evicts, so it needs no release line today. Recommended (not required): make it a
  `_resources.BoundedCache(4, name="robocasa.correctors", on_evict=lambda c: c.close())`. A corrector
  holds a captured CUDA graph and two static uint8 buffers of `net_batch × out_size²×3`; at the
  shipped 128×256²×3 that is ~25 MB a side, per cell, forever.

None of the twelve lines changes a pixel: each one runs only on a path that was already discarding
the value, and `release_render_context` is a no-op on anything that has no `tex_lod` tensors attached.

**Recommended follow-up, not requested here:** the independent review found 23
`clear()`-on-full sites in `envs/*`; eight are already on the consumer list above, leaving **15**
additional candidates. Step 6 already replaced that with an `OrderedDict` LRU in
`envs/ogbench.py:396-403` after measuring the thrash; `_resources.BoundedCache` is that policy with a
release hook, and the migration is mechanical. It belongs to each file's owner, and it is a
behaviour-preserving change only in the sense that a miss recomputes the identical value — which is
true for every one of those caches but should be checked per file by the owner rather than asserted
here.

---

## 5. Targeted tests

| suite | venv | result |
|---|---|---|
| `tests/unit -x --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"` at the parent `b3f3a81` | se6-venv | **1385 passed**, 195 skipped, 112 deselected (142 s) — the brief's baseline, reproduced |
| the same command after merging current integration and the review fixes | se6-venv | **1652 passed, 223 skipped**, 112 deselected (88.56 s) |
| `tests/unit/test_render_resources.py` alone | se6-venv | **32 passed, 1 skipped** (the skip is the `mujoco_warp` cast-memo check; the package is not installed locally) |
| `tests/unit/test_render_resources.py tests/unit/test_nvdiff_triangles.py` | se6-venv | **50 passed, 1 skipped** |
| `test_repo_contracts.py::test_every_cited_scratch_path_resolves` (the test the brief deselects) | se6-venv | **fails, pre-existing**: 14 unresolved citations, every one an `/oscar/scratch/sboughan/...` path cited by Step-6/7/8 findings and by `examples/results/step9/envs-classic-nav/findings.md`. **None is this workstream's**, checked by grepping the failure for `render-mjwarp-nvdiff` (no hit). The three scratch paths this branch cites are spelled out individually rather than brace-expanded, so they resolve. |

**The original delivery delta was fully accounted for, by a collected-ID diff and not by arithmetic.**
`pytest --collect-only -q` in each tree (the parent in a detached worktree, the same venv, the same
`-k`) gives 1579 → 1608, **+29 IDs, none removed**:

* **+28** in `tests/unit/test_render_resources.py` — 26 plain tests plus two parametrised ones
  (`test_every_backend_object_honours_the_close_protocol` ×5 backend classes,
  `test_nothing_in_the_render_backends_releases_an_egl_context` ×5 owned files);
* **+1** because `test_foundation_imports.py::test_a_renderer_knows_nothing_about_environments_or_
  stable_worldmodel` is parametrised over `render/*.py` and now includes `_resources.py`.

That was +28 passed and +1 skipped in the delivery run, exactly 1385 → 1413 and 195 → 196. The
review fix adds five resource tests; the larger current total also includes C/D/E and their tests
from integration. (Both
trees collect one ID more than they report run+skipped, 1579 vs 1580 and 1608 vs 1609; the
off-by-one is identical on both sides and therefore pre-existing, not this patch's.)

**The EGL counters printed by the safety plugin are byte-identical at the parent and at HEAD:**
`16 EGL contexts created, 0 released, 16 live (peak 16); renderers 15 created / 0 closed`. That is
the local, CPU-only half of the never-release evidence — adding `close()` to five backend classes did
not cause a single context or renderer to be released anywhere in the suite.

What the new tests pin — each one is a way the design could still be wrong:

* `close()` is idempotent and `_release` runs at most once (the natural call site is `with` inside a
  function that also closes on the error path);
* a released object raises `UseAfterCloseError` naming the object and the method, rather than
  reading a freed pointer;
* `__exit__` closes on the **exception** path too;
* **nothing releases from `__del__` or GC** — the inverse of the usual test, asserted with two
  `gc.collect()` rounds and `not hasattr(Closeable, "__del__")`;
* a constructor that raises before `_opened()` leaves the object closed, so `close()` on a half-built
  object is a no-op and not a double free;
* the counters survive repeated create/close cycles (5 cycles, created == closed, live back to base);
* concurrent `close()` calls release once; a failed `_release` is counted as failed, is not counted
  closed, and remains live in the accounting; a cache callback failure is likewise observable;
* eviction is **least-recently-used and one entry at a time**, and *a key in use survives an
  unrelated insertion* — the property `clear()`-on-full does not have, and the one that makes the
  `_CASTS_CACHE` change safe rather than merely tidy;
* eviction and **replacement** both release through `on_evict`; `pop` transfers ownership and does
  not; a failing release does not fail the unrelated insertion that triggered it;
* all five backend classes are `Closeable`, declare their own `resource_kind`, define `_release`,
  call `_opened()` on every construction path, and define no `__del__`;
* `nvdiff.Scene.from_triangles` is constructed on CPU, closed, checked for all 16 released tensor
  attributes, and then checked to raise `UseAfterCloseError` from `clip`;
* **no owned module reaches `GLContext`, `.free()`, `eglDestroyContext` or `mujoco.Renderer`** —
  checked over the *tokenised* source with comments and strings stripped, because these files
  explain the rule at length in prose and a substring search finds the explanation.

---

## 6. Static evidence: nothing numeric moved

The independent review recounted the original delivery as **thirteen removed code lines**, not
eleven (eight substantive lines plus five class headers). They are:

| removed | replaced by | why it cannot move a pixel |
|---|---|---|
| `_TEX_KEEP: list = []` | `retain_process_lifetime(t, TEX_KEEP_REASON, overflow="retain")` in the same fallback branch | the tensor kept alive is the same; the explicit overflow policy preserves the old list's non-refusing behavior and logs the tripwire instead |
| `_CASTS_CACHE: dict = {}` | `BoundedCache(64, …)` | same key, same value tuple |
| `hit = _CASTS_CACHE[key] = (casts, m.light_castshadow)` | `hit = _CASTS_CACHE.put(key, (casts, m.light_castshadow))` | `put` returns the value it stored |
| `self._ray_cache: dict = {}` | `BoundedCache(8, …)` | same key `(int(R), mat.tobytes(), float(cam.fovy))` |
| `if len(self._ray_cache) >= 8: self._ray_cache.clear()` + `t = self._ray_cache[key] = torch.as_tensor(d, …)` (3 lines) | `t = self._ray_cache.put(key, torch.as_tensor(d, dtype=torch.float32, device=self.device))` | identical tensor expression, identical dtype and device; the only change is **which** entry an overflow drops |
| `class BatchRenderer:` ×2, `class Scene:`, `class _Runner:`, `class LoadedCorrector:` | the same with `(_resources.Closeable)` | a base class with no `__init__`, no `__del__` and no name these classes already use |

Enumerated positively, the things a reviewer should check are unchanged and are: `effective_alpha`'s
resolution order; `prepare_model` / `restore_model`; `neutralise_illegal_spots` and the
`GL_SPOT_CUTOFF_*` constants; `texture_gaps`, `unsamplable_texture_materials`,
`bake_unsamplable_textures`, `bake_primitive_textures`, `texture_mean_rgb`; `needs_com_pos`;
`create_render_context`'s hoisted blob (**the upload order is the same three statements in the same
order**: one `wp.array(model.tex_data)` on first use, `wp.zeros((h, w, 4))`, then
`wp.launch(_convert_texture_data, dim=(w, h), …)`, and the `tex_lod` `F.interpolate(..., mode="area")`
after it); `gl_rounding`'s `pack_rgba_round`; the four/five-call frame loop and its `needs_com`
condition; the `[..., [2, 1, 0]]` channel swap; all of `mjwarp_lightspec`'s vendored kernel and its
`MAX_SHININESS`/`DEFAULT_MAT_*`/`NO_LIGHT_AMBIENT_FALLBACK` constants; `QUANT`, `mvp_rows`,
`offset_clip`, `_sphere`/`_capsule`/`_cylinder`/`_box`/`_plane`, `texgen` and `TEXGEN_OFFSET`,
`effective_rgba`, `half_extent`, `prepare_texture*`, `_to_pow2`, `Scene._assemble/_sparse/_appearance/
_mesh/clip`, `Camera.matrices`, `Lights.from_scene`, `_shade`, `_texture`, `_sample`, `_background`,
`_ray_dirs`'s grid arithmetic; and every line of `rendernet`'s `Corrector`/`ESPCN`/`UNetTiny`,
`_Runner.__call__`'s tail-fill, and the seal/load digest.

**The documented x86 sparse-stage one-ULP limitation is preserved as documented.** It is a property
of `torch.sparse.mm` in `Scene.clip` and of `nvdiff.mvp_rows`; neither was touched, and
`envs/rocket_landing.py`'s `DOCUMENTED_LIMITATION` block (CLOSURE-1) and
`examples/results/step8/certification/nvdiff_ulp_classification.json` are unchanged and unreferenced
by this patch. Nothing here re-freezes anything.

---

## 7. What is owed — the GPU oracle

`scratch/step9/render-mjwarp-nvdiff/gpu_oracle.sbatch` (partition `gpu`, `--qos=norm-gpu`,
`--gres=gpu:1`, `--cpus-per-task=4`, `--mem=48G`, `--time=01:30:00`). **Not submitted**, per the
cadence ruling: it is the section of the coordinator's single post-Wave-1 GPU job that covers these
four backends. Four arms on **one card in one job**, because no interpreter holds all four
dependency stacks but a paired before/after is only a before/after on the same card:

| arm | venv | cells |
|---|---|---|
| `vis311` | `$SCR/venvs/se-vis311` (mujoco_warp 3.11) | DMCCheetahRun, DMCHopperHop, DMCHumanoidStand, DMCWalkerWalk, DMCQuadrupedWalk, RoboCasaCounterToSink(+Corrected) |
| `ogb` | `$SCR/venvs/se7-ogb` | OGBCube, OGBScene |
| `rocket` | `$SCR/venvs/se7-rocket` | PFRocketLanding, MujocoHalfCheetah, MujocoAnt |
| `fetch` | `$SCR/venvs/se-s3b-fetch311` | FetchReach, FetchPush, FetchSlide, FetchPickAndPlace |

Each arm runs, and each of these is a question the others cannot answer:

1. **frame identity** — baseline tree vs this branch, 64 states from the frozen corpus where one
   exists, through the public path only; frame SHA-256, frames-from-explicit-rows SHA-256, the state
   digest each arm was handed, the frame shape, the registered size, and **which owned entry points
   the cell actually called** (so a cell that silently stops using the backend fails rather than
   passes);
2. **fresh-process reproducibility** — `--arms head,head`: the same tree twice, in two processes,
   which is what tells "not reproducible across processes" apart from "this branch changed it";
3. **repeated environment create/close cycles** (5), asserting identical frames, zero recorded
   release failures, and `resource_guard`'s opposite rule (`context_freed == 0`,
   `renderer_closed == 0` — closing an env must **never** release a GL context);
4. **direct backend create/close probes** for all five `Closeable` classes, asserting created ==
   closed, live returns to baseline, failed releases remain zero, allocated memory falls after
   close, and the nvdiff ray LRU rebuilds the same grid after overflowing;
5. **batch-size invariance** at 1 / 2 / 8 / 64, comparing row 0 — a cache with a new eviction policy
   is exactly the kind of change that could break this at one width and not another;
6. **a closed env refuses with the exact Wave-0 `RuntimeError`**, and closing twice is a no-op;
7. **paired alternating throughput** on the same card at **B=1 and B=4096** (stepping down
   4096→1024→256→64 on OOM, recording the width both legs used), one cell per backend path, bar
   1.10 (plan §8). B=1 is where a `move_to_end` could conceivably show; B=large should be 1.000.
8. all four explicit GPU test modules in every compatible venv. Exit 5, missing
   interpreters/dependencies/baseline, absent backend coverage, skipped selected cells, oracle
   failures and >10% benchmark regressions all increment a final failure count; the sbatch exits
   nonzero unless every section passes.
9. `STABLE_ENVS_RENDERNET_CACHE` is redirected to `$SCR/cache/rendernet`; absent RoboCasa assets or
   the sealed corrector fail coverage instead of silently producing a partial pass. Full pytest logs
   are retained because MuJoCo's teardown noise arrives after the verdict.

**Unexplained oracle differences: none observed, because none could be — the oracle has not run.**
That is a statement about evidence, not about the code, and it is the one open item on this branch.

### The one number this patch could plausibly move, stated in advance

Batch-1 frame time, by at most the cost of one `OrderedDict.move_to_end` plus a `threading.Lock`
acquire per `_ray_dirs` call (nvdiff only; MJWarp's path has no per-frame cache). That is tens of
nanoseconds against a measured 6.08 ms batch-1 frame at 240 px. If the bench shows anything above
noise at B=1 on an nvdiff cell, the lock is the first thing to look at and it can be dropped for the
per-renderer cache (which is single-threaded by construction) without touching the policy.

---

## 8. Certification, declarations, tiers, variation

Nothing in this patch touches a threshold, a declaration, a fidelity tier, a frozen digest or a
frozen corpus. No `examples/results/step4|step5|step7|step8` file is modified. The applicable
harnesses are `tests/gpu/test_conformance.py`, `test_reproducibility.py` and `test_fast_paths.py`,
all in the GPU arm. Variation semantics are untouched by construction: `mat_rgb`, `size_scale` and
the style tensor reach these backends as tensors and pass through expressions this patch did not
modify.

---

## 9. Requests for the coordinator

1. **Submit `gpu_oracle.sbatch` as a section of the post-Wave-1 GPU job** and return
   `oracle_*_gpu_*.json`, `fresh_*`, `timings_*`. Until then this branch is APPROVE-blocked by §17's
   zero-unexplained-differences rule on evidence grounds, not on a known defect.
2. **Assign the twelve consumer lines in §4** to C / D / E / F / B, or to a single follow-up
   workstream after they merge. They are the half of "explicit resource ownership" that lives in
   files this workstream does not own; without them the *design* is complete and the *practice* still
   leaves an evicted context to the collector.
3. **Foundation request (`env.py`, not edited here):** none required. `Env.close`/`_close`/`__enter__`
   /`__exit__` and the `render` guard at `env.py:237` are exactly the protocol these backends needed,
   and this workstream built on them rather than around them. One optional suggestion: `Env.close`
   could log at DEBUG through the Wave-0 Loguru logger when `_close` is absent, so "this family
   releases nothing because it owns nothing" is visible rather than inferred.
4. **Ruling wanted:** should `_resources.BoundedCache` become the repository-wide cache? The
   independent review found **15 additional** `clear()`-on-full sites beyond the eight consumer
   sites already listed. Apply them as one coordinated Wave-2 sweep, ordered with the release
   callbacks, rather than as independent edits.
5. **No recount needed:** the collected-ID diff in §5 was run (1579 → 1608, +29, none removed) and
   closes exactly against 1385 → 1413 passed and 195 → 196 skipped. It is reproducible with
   `pytest --collect-only -q` in each tree if the reviewer wants it independently.
