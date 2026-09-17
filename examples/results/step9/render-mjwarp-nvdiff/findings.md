# Step 9 — Workstream I (MJWarp / nvdiffrast / renderer resource ownership)

**Branch** `step9/render-mjwarp-nvdiff` · **parent** `b3f3a81` (`step9/integration` after H merged;
`5741b62` names the same tree plus two dashboard commits — `git diff b3f3a81 5741b62` is
`examples/results/step9/dashboard.yaml`, 47 lines, nothing else) · **HEAD** `86e782d`, 7 commits ·
**Step-8 behavioural baseline** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc` ·
**host** Apple M5 Mac (`cis1b1zs2j3`), `/Users/samib/se6-venv` (torch 2.14 CPU, mujoco 3.12.0,
dm_control 1.0.45, no CUDA, no warp, no nvdiffrast, no mujoco_warp).

**Result: PATCH — design change delivered, GPU evidence owed.** Explicit resource ownership is in
place across all four backends plus the new `render/_resources.py`; the static read-through shows
**not one numeric expression, upload order or cache key changed**; the local suite is
**1385 → 1413 passed, 195 → 196 skipped** (+28 new lifetime tests, +1 from the parametrised
dependency-direction check picking up the new file; collected-ID diff +29, none removed). **Every backend this workstream owns is CUDA-only and the local
machine has no CUDA, so no frame in this workstream has been rendered.** Frame identity,
fresh-process reproducibility, the create/close counters, batch invariance and paired throughput are
all owed to the coordinator's post-Wave-1 GPU oracle; `scratch/step9/render-mjwarp-nvdiff/gpu_oracle.sbatch`
is this workstream's section of that job and has **not** been submitted (Sami's cadence ruling).
Until it reports, plan §17's "unexplained oracle differences: zero" is **unestablished, not
satisfied**, and this branch should not merge on the local evidence alone.

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
| `stable_envs/render/_resources.py` | **new**, 333 lines. `Closeable`, `BoundedCache`, `UseAfterCloseError`, `retain_process_lifetime`, `counts()/live()/delta()`. Imports stdlib only — no torch, warp, mujoco or nvdiffrast — so it loads in every interpreter and passes the dependency-direction test unchanged. |
| `stable_envs/render/mjwarp.py` | `BatchRenderer` is `Closeable` (`_release` drops graph → context → `Data` → `Model`); `render` guards with `_check_open`; the anonymous `_TEX_KEEP` leak becomes a counted, capped `retain_process_lifetime`; new `release_render_context(rc)`. |
| `stable_envs/render/mjwarp_lightspec.py` | `_CASTS_CACHE` becomes a `BoundedCache(64)`. |
| `stable_envs/render/nvdiff.py` | `Scene` and `BatchRenderer` are `Closeable`; `owns_scene` records and can be set; `_ray_cache` becomes a `BoundedCache(8)`; `render` guards with `_check_open`. |
| `stable_envs/render/rendernet.py` | `_Runner` and `LoadedCorrector` are `Closeable`; the captured CUDA graph is `reset()` before its static buffers are dropped. |
| `tests/unit/test_render_resources.py` | **new**, 28 test IDs (27 pass, 1 skips without `mujoco_warp`). §5. |
| `scratch/step9/render-mjwarp-nvdiff/{oracle.py,bench.py,gpu_oracle.sbatch}` | **new** evidence harness, none of it run (§7). |
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
| …same, when the context refuses the attribute | **the process, deliberately** | `retain_process_lifetime(..., TEX_KEEP_REASON)` | never — counted and capped at 256 instead | `test_process_lifetime_retention_records_its_reason_and_is_capped` |
| captured warp graph (`ScopedCapture`) | `mjwarp.BatchRenderer` | `BatchRenderer._capture` | `_release`, **first**, before the buffers it replays into | GPU cycles arm |
| `light_castshadow` memo | `mjwarp_lightspec._CASTS_CACHE` (module, process-scoped) | `light_casts` | LRU eviction at 64 keys; nothing device-side to free | `test_the_module_level_cast_memo_is_bounded`, `test_a_key_in_use_survives_an_unrelated_insertion` |
| nvdiffrast `RasterizeCudaContext` | `nvdiff.BatchRenderer`, always | `BatchRenderer.__init__` | `_release` (`self.ctx = None`), last | shape test + GPU cycles arm |
| `nvdiff.Scene` (triangle soup, sparse transforms, appearance, texture pyramids) | `BatchRenderer` **iff `owns_scene`**, else the caller | `Scene.__init__` / `Scene.from_triangles` | `Scene._release`; via `BatchRenderer._release` only when `owns_scene` | shape test; `owns_scene` is checked by inspection, GPU arm measures the memory |
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

**Recommended follow-up, not requested here:** eight `_CACHE`/`_CACHE_MAX` pairs in `envs/*` still
spell `if len(C) >= MAX: C.clear()`. Step 6 already replaced that with an `OrderedDict` LRU in
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
| the same command at HEAD `86e782d` | se6-venv | **1413 passed, 196 skipped**, 112 deselected (96 s) |
| `tests/unit/test_render_resources.py` alone | se6-venv | **27 passed, 1 skipped** (the skip is the `mujoco_warp` cast-memo check; the package is not installed locally) |

**The delta is fully accounted for, by a collected-ID diff and not by arithmetic.**
`pytest --collect-only -q` in each tree (the parent in a detached worktree, the same venv, the same
`-k`) gives 1579 → 1608, **+29 IDs, none removed**:

* **+28** in `tests/unit/test_render_resources.py` — 26 plain tests plus two parametrised ones
  (`test_every_backend_object_honours_the_close_protocol` ×5 backend classes,
  `test_nothing_in_the_render_backends_releases_an_egl_context` ×5 owned files);
* **+1** because `test_foundation_imports.py::test_a_renderer_knows_nothing_about_environments_or_
  stable_worldmodel` is parametrised over `render/*.py` and now includes `_resources.py`.

That is +28 passed and +1 skipped in the run, which is exactly 1385 → 1413 and 195 → 196. (Both
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
* eviction is **least-recently-used and one entry at a time**, and *a key in use survives an
  unrelated insertion* — the property `clear()`-on-full does not have, and the one that makes the
  `_CASTS_CACHE` change safe rather than merely tidy;
* eviction and **replacement** both release through `on_evict`; `pop` transfers ownership and does
  not; a failing release does not fail the unrelated insertion that triggered it;
* all five backend classes are `Closeable`, declare their own `resource_kind`, define `_release`, and
  define no `__del__` (shape, not construction: four of the five need a GPU to build);
* **no owned module reaches `GLContext`, `.free()`, `eglDestroyContext` or `mujoco.Renderer`** —
  checked over the *tokenised* source with comments and strings stripped, because these files
  explain the rule at length in prose and a substring search finds the explanation.

---

## 6. Static evidence: nothing numeric moved

`git diff b3f3a81 -- stable_envs/render/` with the prose and comments filtered gives **eleven
removed lines in total**, and they are:

| removed | replaced by | why it cannot move a pixel |
|---|---|---|
| `_TEX_KEEP: list = []` | `retain_process_lifetime(t, TEX_KEEP_REASON)` in the same fallback branch | the tensor kept alive is the same tensor; only the list it is appended to changed |
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
3. **repeated create/close cycles** (5), asserting the frames are identical on every cycle **and**
   both counters: `_resources` (created == closed, nothing live) and `resource_guard`
   (`context_freed == 0`, `renderer_closed == 0` — closing an env must **never** release a GL
   context, and that asymmetry is the design);
4. **batch-size invariance** at 1 / 2 / 8 / 64, comparing row 0 — a cache with a new eviction policy
   is exactly the kind of change that could break this at one width and not another;
5. **a closed env refuses to render**, and closing twice is a no-op;
6. **paired alternating throughput** on the same card at **B=1 and B=4096** (stepping down
   4096→1024→256→64 on OOM, recording the width both legs used), one cell per backend path, bar
   1.10 (plan §8). B=1 is where a `move_to_end` could conceivably show; B=large should be 1.000.
7. the GPU test selection this code can reach, **with the whole pytest output to a file and only the
   summary echoed** — Workstream H lost a run's counts because MuJoCo's EGL teardown noise arrives
   after pytest's verdict and a `| tail` showed only the noise.

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
4. **Ruling wanted:** should `_resources.BoundedCache` become the repository-wide cache, i.e. should
   the eight remaining `clear()`-on-full caches in `envs/*` migrate in Wave 2? It is behaviour-
   preserving for every one of them (a miss recomputes the identical value) but it is eight files
   with eight different owners, so it is a coordination question rather than a technical one.
5. **No recount needed:** the collected-ID diff in §5 was run (1579 → 1608, +29, none removed) and
   closes exactly against 1385 → 1413 passed and 195 → 196 skipped. It is reproducible with
   `pytest --collect-only -q` in each tree if the reviewer wants it independently.
