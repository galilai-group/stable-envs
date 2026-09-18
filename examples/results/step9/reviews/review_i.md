APPROVE-WITH-CHANGES

# Step 9 Wave 1 — Workstream I (render/_resources.py, mjwarp, mjwarp_lightspec, nvdiff, rendernet) — independent review

**Reviewed SHA:** `50acdf81eb98a7c64cb81b8979d414149c09d298` (`step9/render-mjwarp-nvdiff` HEAD) =
author delivery `5aeee3712907ebf8984647b0b4767efea46b4094` (9 commits) merged with
`step9/integration` = `312a631` by the coordinator. `git merge-base --is-ancestor 312a631 HEAD`
holds; the merge is clean and contributes no code of its own
(`git diff 312a631..50acdf8` touches exactly the ten files below).
**Code delta reviewed:** `git diff 312a631..50acdf8` (1930 insertions, 18 deletions, 10 files) and
`git diff 21cb306..50acdf8 -- stable_envs/render/` for the behavioural question.
**Step-8 behavioural baseline:** `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`.
**Reviewer environment:** Apple M5 Mac, `cis1b1zs2j3.ad.brown.edu`, local CPU only —
**no CUDA, no warp, no mujoco_warp, no nvdiffrast**. `/Users/samib/se6-venv/bin/python`
(torch 2.14 CPU, mujoco 3.12.0, dm_control 1.0.45), `MUJOCO_GL=glfw`, `-p no:cacheprovider`.
No edits, commits, stashes, resets, rebases or pushes. **The sbatch was read, never submitted;
Oscar was not contacted.** Every number below I produced myself.

---

## Verdict

**APPROVE-WITH-CHANGES.** The design is right, the static argument is very nearly right, and the
patch is additive in a way I could confirm symbol by symbol. What it does not have is an oracle, and
— this is the finding that matters most — **the GPU job that is supposed to supply one cannot
currently fail in the way it would need to fail.** I audited it as an instrument and it has six
holes, two of which make it report `PASS` over a real regression. So this is an approval contingent
on a *repaired* GPU job passing, not on the job as committed.

What I agree with:

* **No `__del__`, deliberately, is the correct call** and the code honours it: `grep -rn "__del__"
  stable_envs/render/` returns six hits and every one is prose. `Closeable` defines no `__del__`,
  no owned class defines one, and the shape test asserts it per class. `Closeable.close()` is
  idempotent and `_release` runs at most once — I exercised both.
* **The never-release rule for EGL contexts is the right rule and is preserved.** `resource_guard.py`
  is byte-identical at `21cb306` and at HEAD (`git diff --stat` over that path is empty) and its four
  incidents justify it. My own two full unit-suite runs print **identical** safety counters at the
  branch and at integration: `20 EGL contexts created, 0 released, 20 live (peak 20); renderers 19
  created / 0 closed`. Adding `close()` to five classes released nothing anywhere in 1561 tests.
* **Adding `Closeable` as a base to the five classes is safe.** I walked all five ASTs for collisions
  against the ten protocol names (`resource_kind`, `_se_closed`, `_se_counted`, `_opened`, `closed`,
  `_check_open`, `_release`, `close`, `__enter__`, `__exit__`): the only overlaps are the intended
  `_release` and `resource_kind`. None of the five has `__slots__`. `grep -rn "\.closed"` over
  `stable_envs/{envs,render}/` finds exactly one hit and it is a docstring, so the new read-only
  `closed` property shadows nothing and no consumer assigns to it.
* **The re-export surface holds.** Verified by import, not by grep:
  `stable_envs.render.mjwarp` still exposes `HIDDEN_GROUP` (`5`), `DEFAULT_GROUPS` (`(0, 1, 2)`),
  `gl_rounding`, `create_render_context`, plus the new `release_render_context` and
  `TEX_KEEP_REASON`. Every consumer reaches them as `MW.<name>` (`fetch.py:170-171`,
  `metaworld.py:272-273`, `humanoidbench.py:267-268`, `ogb_manip.py:1305`, `robocasa.py:375`,
  `fetch.py:468`, `metaworld.py:811`) and all resolve. The one name that disappeared, `_TEX_KEEP`, has
  no reader anywhere in the tree (`scratch/altrender/probe_budget_frontier.py` defines its own).

What must change before merge: items 1–4 of the defect list. Item 1 is the GPU condition.

---

## The §17 ten items

**1. Files moved / removed / created.** `git diff 312a631..50acdf8 --stat` — ten files, no renames, no
deletions:

| file | lines |
|---|---|
| `stable_envs/render/_resources.py` | **new**, +333 |
| `stable_envs/render/mjwarp.py` | +74 / −7 |
| `stable_envs/render/mjwarp_lightspec.py` | +18 / −2 |
| `stable_envs/render/nvdiff.py` | +95 / −6 |
| `stable_envs/render/rendernet.py` | +51 / −2 |
| `tests/unit/test_render_resources.py` | **new**, +308 |
| `scratch/step9/render-mjwarp-nvdiff/{oracle.py,bench.py,gpu_oracle.sbatch}` | **new**, +737 |
| `examples/results/step9/render-mjwarp-nvdiff/findings.md` | **new**, +315 |

`git diff 312a631..50acdf8 -- stable_envs/envs stable_envs/env.py stable_envs/testing tests/gpu` is
**empty** (zero output): no consumer, no foundation file, no GPU test touched. The findings' "no file
outside those was touched" is accurate.

**Scope note, not a defect.** Plan §13 gives Workstream I `render/mjwarp.py`, `render/nvdiff.py`,
`render/_resources.py` — three files. It also edited `render/mjwarp_lightspec.py` and
`render/rendernet.py`. Neither is in any other workstream's `Own:` block and neither is in §10's
coordinator-owned list, so nothing was trespassed; but the coordinator should record both as I's so a
Wave-2 workstream does not collide.

**2. API changes.** Additive only, verified name by name. New: `_resources.{Closeable, BoundedCache,
UseAfterCloseError, retain_process_lifetime, retained_reasons, counts, live, delta, MAX_RETAINED}`;
`mjwarp.{release_render_context, TEX_KEEP_REASON}`; `mjwarp_lightspec.CASTS_CACHE_MAX`;
`nvdiff.RAY_CACHE_MAX`; `nvdiff.BatchRenderer.owns_scene`; `close()/closed/__enter__/__exit__` on the
five classes; `resource_kind` per class. Removed: `mjwarp._TEX_KEEP` (no reader, above). Changed
**type**: `mjwarp_lightspec._CASTS_CACHE` and `nvdiff.BatchRenderer._ray_cache` from `dict` to
`BoundedCache`; both are private and every access is inside the defining file (checked by grep — the
only external references are in the new test file).

`_resources.py` imports `threading`, `collections.OrderedDict` and `typing` and nothing else, so the
§8 dependency rule holds by construction. The parametrised
`test_foundation_imports.py::test_a_renderer_knows_nothing_about_environments_or_stable_worldmodel`
picks the new file up and **passes** — I ran it alone: `17 passed, 9 deselected`. (The findings say
this new param accounts for "+1 skipped"; it does not, it accounts for +1 **passed**. The totals
still reconcile — see item 4 — so this is a mislabel, not an error in the arithmetic.)

**3. Step-8 IDs affected.** Every cell that reaches MJWarp or nvdiffrast: the 5 DMC trackcom cells,
OGBCube/OGBScene, the 4 Fetch tasks (× Dense/Dict variants sharing the renderer), RoboCasa
(+Corrected), PFRocketLanding and the visual MuJoCo locomotion cells — the 17 in `oracle.py:CELLS`
plus their variants. **Behaviourally none changed** on the evidence available, which is static only
(item 6).

**4. Targeted tests — my own numbers, same venv, same command, both trees.**

| tree | result |
|---|---|
| `step9/integration` = `312a631` | **1533 passed, 198 skipped, 112 deselected**, 154.72 s |
| this branch = `50acdf8` | **1561 passed, 199 skipped, 112 deselected**, 114.83 s |
| delta | **+28 passed, +1 skipped, ±0 deselected** |

Command in both: `MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider
tests/unit -x --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"`.

The delta closes exactly: `tests/unit/test_render_resources.py` collects **28 IDs → 27 passed,
1 skipped** (the skip is `test_the_module_level_cast_memo_is_bounded`, `importorskip("mujoco_warp")`),
plus **+1 passed** from the dependency-direction param on `_resources.py`. 27+1 = +28 passed,
+1 skipped. Confirmed independently, not reasoned from the author's table.

The author's own reported baseline (1385 → 1413) was against `b3f3a81`; integration has moved since
(G merged), which is why my numbers are 1533 → 1561. The delta is the same.

**5. Applicable certification checks.** No threshold, declaration, fidelity tier, frozen digest or
frozen corpus is touched: the diff contains no change to any numeric constant, and
`examples/results/step4|5|7|8` is untouched. The findings name
`tests/gpu/{test_conformance,test_reproducibility,test_fast_paths}.py` as the applicable harnesses —
**and the sbatch's `-k` filter excludes two of the three.** See the sbatch audit, hole S-3.

**6. Default-frame comparison. UNESTABLISHED, not satisfied.** No frame in this workstream has been
rendered, by the author or by me: every backend here is CUDA-only and neither of our machines has
CUDA. I confirmed there is no CPU fallback to lean on — `mjwarp.BatchRenderer.__init__` imports
`mujoco_warp` unconditionally, `nvdiff.BatchRenderer.__init__` builds
`dr.RasterizeCudaContext`, and `rendernet._Runner.__init__` captures a `torch.cuda.CUDAGraph`. The
author's wording is the honest one and I am holding it: **the behavioural oracle for this workstream
does not exist yet.** The positive evidence is the static read-through (item 11 below) and the
never-release counters, and that is all it is.

**7. Variation / state comparison.** Untouched by construction: `mat_rgb`, `size_scale` and the style
tensor reach these backends as tensors through expressions the diff does not modify. No
`state_layout`, `state_names`, `pack`/`unpack` or `variation_space` code is in the diff.

**8. Performance comparison for hot paths.** Not measured, same reason. One path plausibly moved and
the author called it in advance correctly: `_ray_dirs` now does an `OrderedDict.move_to_end` under a
`threading.Lock` on every cache hit where it previously did a bare `dict` lookup. I confirmed the
timed path in `bench.py` actually contains it (3 warm-up renders then 5 timed renders on the same
state and camera → every timed render is a cache **hit**), so the B=1 leg would see it if it were
visible. Against a stated 6.08 ms batch-1 frame it will not be. MJWarp's path has no per-frame cache;
`_CASTS_CACHE` is touched twice per renderer construction, not per frame. Nothing else in the diff is
on a hot path.

**9. Resource-lifetime comparison. This is the item with the substance, and it has a hole the
findings do not state.** I traced every construction site of the five `Closeable` classes:

| class | constructed by | `close()` called by |
|---|---|---|
| `mjwarp.BatchRenderer` | **nothing in `stable_envs/`, nothing in `tests/`, nothing in the workstream's own oracle** | nothing |
| `nvdiff.Scene` | `nvdiff.BatchRenderer.__init__`; `rocket_landing.py:735` (`from_triangles`); `tests/unit/test_nvdiff_triangles.py` on `device="cpu"` | `BatchRenderer._release` only, iff `owns_scene` |
| `nvdiff.BatchRenderer` | `mujoco_locomotion.py:1653` (owns its scene), `rocket_landing.py:739` (scene handed in) | nothing |
| `rendernet._Runner` | `LoadedCorrector.runner` (lazy) | `LoadedCorrector._release` only |
| `rendernet.LoadedCorrector` | `robocasa.py:928` via `RN.load` | nothing |

And `grep -rn "def _close" stable_envs/envs/` returns **three** hits — `pusht.py`,
`classic_control.py`, `dmcontrol/base.py` — none of which is one of the eight MJWarp/nvdiff families.
`dmcontrol/base.py:136`'s `_close` pops a CPU scene dict and documents that it releases neither the
`mujoco.Renderer` nor the `_TC_CACHE` context, on purpose.

**So today, on this branch, `close()` is never called on any of the five classes by any code path in
the package.** Every `_release` body is dead code until the twelve consumer lines land. The findings
say this obliquely ("without them the *design* is complete and the *practice* still leaves an evicted
context to the collector"); it deserves to be stated flatly, because it is what makes the GPU job
unable to validate the patch (hole S-1).

**10. Unexplained oracle differences: not zero — unmeasured.** `identical: true` has not been produced
by anything. The §17 requirement is not met and cannot be met on this machine.

---

## 11. The removed lines, re-derived from scratch

The findings claim **eleven** removed lines. My own derivation —
`git diff 312a631..50acdf8 -- stable_envs/render/ | grep '^-' | grep -v '^---'`, with comments and
blanks stripped — gives **thirteen**:

| # | removed | replaced by | my finding |
|---|---|---|---|
| 1 | `_TEX_KEEP: list = []` | module-level list in `_resources` | ✅ no numeric, no upload, no key |
| 2 | `_TEX_KEEP.extend(keep)` | `for t in keep: retain_process_lifetime(t, TEX_KEEP_REASON)` | ⚠️ **not behaviour-preserving** — see defect 4 |
| 3 | `_CASTS_CACHE: dict = {}` | `BoundedCache(64, …)` | ✅ same key `(ptr, nlight, use_shadows)`, same value tuple |
| 4 | `hit = _CASTS_CACHE[key] = (casts, m.light_castshadow)` | `hit = _CASTS_CACHE.put(key, (…))` | ✅ `put` returns `value` (`_resources.py:290`) — identical |
| 5 | `self._ray_cache: dict = {}` | `BoundedCache(8, …)` | ✅ |
| 6 | `if len(self._ray_cache) >= 8:` | the cache's own bound | ✅ same bound (8), different victim |
| 7 | `self._ray_cache.clear()` | LRU `popitem(last=False)` | ✅ values identical on a miss |
| 8 | `t = self._ray_cache[key] = torch.as_tensor(d, dtype=torch.float32, device=self.device)` | `t = self._ray_cache.put(key, torch.as_tensor(d, dtype=torch.float32, device=self.device))` | ✅ tensor expression **character-for-character identical** |
| 9–13 | `class BatchRenderer:` ×2, `class Scene:`, `class _Runner:`, `class LoadedCorrector:` | the same `(_resources.Closeable)` | ✅ no name collision (see Verdict) |

Eight substantive code lines plus five class headers; the author's "eleven" undercounts by two
(row 1 of their table folds two lines into one, and row 6 counts `class BatchRenderer:` ×2 as one
entry). The count is cosmetic. **The substance holds for twelve of the thirteen and fails for one**,
which is line 2 — the coordinator's prediction was correct.

I also re-derived the two claims the table rests on:

* **No cache key changed.** `_CASTS_CACHE`: key is `(int(m.light_castshadow.ptr), int(m.nlight),
  bool(rc.use_shadows))` at `21cb306` and at HEAD — I diffed the two versions of `light_casts`
  directly and only the `put` line differs. `_ray_cache`: key is `(int(R), mat.tobytes(),
  float(cam.fovy))`, unchanged.
* **No upload order changed.** `create_render_context`'s hoisted `create_warp_texture` is the same
  three statements in the same order (`wp.array(model.tex_data)` once, `wp.zeros((h, w, 4))`,
  `wp.launch(ru._convert_texture_data, dim=(w, h), …)`) with the `lod > 1` `F.interpolate(…,
  mode="area")` after it; the diff touches only the `except AttributeError` branch after `rc` exists.
  MJWarp's `_frame()` is the same five calls in the same order with the same `needs_com` condition;
  the `[..., [2, 1, 0]]` swap is unchanged; the only insertion in `render` is `_check_open`.

**One correctness question the table does not raise, which I checked and cleared.** `_CASTS_CACHE`'s
own comment at baseline states the invariant *"the value keeps a reference to the array so the
pointer it is keyed on cannot be freed and reissued to a different model."* Bounding an unbounded
pointer-keyed cache is exactly how you break that. It does **not** break here: `BoundedCache.put`
removes the key from `_d` and hands the old value to `_free` in the same critical section, so a key
is never in the cache without its array reference alive, and a pointer can only be reissued after its
key is gone (which produces a miss and a recompute, never a wrong answer). The author's separate
argument — that LRU cannot evict the warm-up entry before the capturing read, because the warm-up
`put` makes it MRU and nothing inserts between the two calls — I also followed and it holds.

**And one I checked and cleared for the ray cache.** `_ray_dirs` returns the cached tensor (as an
`expand` view). Its single consumer is `_background`, whose next statement is
`torch.stack([d[...,0], d[...,2], -d[...,1]], -1)` — a new tensor. Nothing writes through the cached
grid, so returning the *same* tensor object across an overflow (LRU) rather than a *rebuilt* one
(`clear()`) cannot change a pixel.

---

## 12. §16 — no abstraction without demand

`BoundedCache` **passes** §16, but only half of it does.

* **The LRU-bound half has two real, demanding call sites today**: `mjwarp_lightspec._CASTS_CACHE`
  (was unbounded — a genuine leak, one entry per model forever) and
  `nvdiff.BatchRenderer._ray_cache` (was `clear()`-on-full). Both are in files this workstream owns,
  both had a defect the class fixes, and `envs/ogbench.py:396-403` is a third, independent instance
  of the same policy that Step 6 wrote by hand. That is demand, not speculation.
* **The `on_evict` half has zero call sites.** Neither of the two constructions passes `on_evict`, and
  the only proposed user — `robocasa._CORRECTOR_CACHE` — is a *recommendation* in §4 of the findings,
  not applied. So the release-on-eviction machinery (`_free`, the eviction counter, the swallow, the
  "a cache must not become the thing that fails" policy) is a design for a call site that does not
  exist yet. It is ~25 lines and it is the justification for the whole "the cache is the owner"
  argument, so I would not strip it — but the coordinator should know that if the twelve consumer
  lines are never assigned, `on_evict` is dead abstraction and §16 would then say remove it.

`Closeable` passes §16 comfortably (five subclasses). `retain_process_lifetime` has one caller, on a
`# pragma: no cover` fallback branch — borderline, and it is where defect 4 lives.

---

## 13. The sbatch as a measuring instrument — six coverage holes

I read `scratch/step9/render-mjwarp-nvdiff/gpu_oracle.sbatch` (123 lines), `oracle.py` (402) and
`bench.py` (212) as one instrument and asked the only question that matters: *would this report PASS
while a regression this refactor could cause hid?* Six times, yes.

### S-1 (the big one). The instrument cannot see the thing under test, and it would say PASS.

The sbatch's stated measurement 3 is "`stable_envs.render._resources`: created == closed, nothing
left live" (lines 26). **`oracle.py` never asserts it.** `LIFETIME_ASSERTIONS` (lines 309–317) has
five entries — `cycles_identical`, `egl_released_across_cycles`, `renderers_closed_across_cycles`,
`closed_env_refuses`, `batch_invariant` — and none of them is `se_created`/`se_closed`. Those two are
computed at lines 223–228, written into the JSON, and never compared to anything.

That is not an oversight that can simply be fixed by adding the assertion, because **the assertion
would fail for a reason that has nothing to do with this patch**: per §17-item-9 above, the cycles arm
calls `e3.close()` on the *env*, the eight families have no `_close`, the renderers and contexts live
in module-level caches keyed by `(cell, res, nworld, device)`, and nothing calls `close()` on any of
the five classes. So `se_closed` would be `{}` on every cycle and `live()` would be non-zero forever.

**Consequence:** as committed, no arm of this job exercises a single `_release` body. The design
change is invisible to its own oracle, and the oracle prints `ORACLE: PASS`.

**What to fold in instead.** Either (a) run the GPU job *after* the twelve consumer lines land, and
then assert `se_created == se_closed` per kind across cycles; or (b) add an arm that constructs and
closes each of the five classes directly, in-process, independently of any env — `MW.BatchRenderer(m,
batch=8)`, `ND.BatchRenderer(m, res=64)`, `ND.Scene(m)` both owned and handed-in, `RN.load(...)`
followed by `.runner` — and around each, `torch.cuda.synchronize(); torch.cuda.memory_allocated()`
before and after `close()`. (b) is the one I would insist on, because of S-2.

### S-2. Nothing anywhere measures memory *after* a close, so a `_release` that frees nothing passes.

`oracle.py` contains no reference to `torch.cuda.memory_allocated` or `memory_reserved`. `bench.py`
does record `torch.cuda.max_memory_allocated()` — but (i) it is read *before* `env.close()` and is
therefore peak-during-render, a different quantity, and (ii) `_paired` overwrites `meta` on **every**
leg (`meta = got`, line 136), so the `peak_bytes` that reaches the record is the **head** leg of the
last round only and the baseline's is discarded. The one memory number the harness collects cannot be
used as a before/after.

This matters specifically because three of the five `_release` bodies are **hand-maintained lists of
attribute names**: `mjwarp.BatchRenderer._release` names `graph, _bgra, rc, d, m`;
`nvdiff.Scene._release` names sixteen (`vert, norm, uv, vgeom, face, fgeom, mjgeom, S3, S4, S3p,
mat_rgb, alpha, spec, shininess, emission, texid`) behind `if hasattr(self, name)`;
`nvdiff.BatchRenderer._release` names five plus the conditional scene. A misspelled or forgotten name
is **silently skipped** by the `hasattr` guard, leaves the buffer alive, and is invisible to every
counter, to every unit test, and to every arm of this job. That is the highest-probability bug class
in the whole patch and it is the one thing nothing measures.

(The `hasattr` guard is load-bearing, not merely defensive: `Scene.from_triangles` builds through
`cls.__new__(cls)` and does not set `mjgeom`, so the guard is doing real work — which is exactly why a
typo hides in it.)

### S-3. The pytest arm deselects two of the three harnesses the findings declare applicable.

`gpu_oracle.sbatch:113-114` runs `pytest tests/gpu -k "mjwarp or nvdiff or rendernet or resource or
lifetime or reproducib"`. I ran that exact selection locally:

```
no tests collected (104 deselected) in 0.15s        # rc 5
```

Locally the two relevant modules are `importorskip`-ed away, so 0 collected is expected here. But the
keyword set is the problem regardless. `-k` matches node names, so:

* `test_rendernet.py` → selected whole (filename contains `rendernet`) ✅
* `test_reproducibility.py` → selected whole (filename contains `reproducib`) ✅
* **`test_conformance.py` → entirely deselected.** Its filename matches nothing and none of its test
  names do either: `test_batch_invariance`, `test_determinism`, `test_backend_agreement`,
  `test_render_parity_with_official_renderer`, `test_style_is_state`,
  `test_fidelity_tier_is_fully_declared`, … **`test_render_parity_with_official_renderer` is the
  actual pixel oracle for these backends and this job does not run it.**
* **`test_fast_paths.py` → entirely deselected.** `test_the_fast_path_is_bit_identical_to_the_path_it_
  replaces`, `test_the_ellipsoid_twins_are_bit_identical_on_both_casts`, … none matches.
* `resource` and `lifetime` match nothing in `tests/gpu` at all.

Findings §8 says all three are "in the GPU arm". Two are not. Replace the `-k` with explicit paths:
`tests/gpu/test_conformance.py tests/gpu/test_reproducibility.py tests/gpu/test_fast_paths.py
tests/gpu/test_rendernet.py`.

Also: only two of the four venvs (`vis311`, `rocket`) run pytest at all; `ogb` and `fetch` get none.

### S-4. Nothing in the job produces a verdict, and several failure modes are indistinguishable from a pass.

`set -uo pipefail` without `-e` is right for a report job, but the report never aggregates:

* `arm()` echoes `rc=$?` per step and **no rc is ever collected**. There is no failure counter, no
  final `PASS`/`FAIL` line, and the script always reaches `ALLDONE`.
* A missing interpreter prints `MISSING interpreter … -- arm skipped` and `return`s 0 (line 72).
  **A skipped arm looks like a passing arm** in the stdout stream.
* `ensure_deps` failure is a `WARNING` only (line 65). The handoff warns that compute nodes may lack
  internet; if `tensordict`/`loguru` do not install, `import stable_envs` fails, `_run_probe` raises
  `SystemExit`, the arm produces nothing, and the job still says `ALLDONE`.
* `baseline_present` is computed and printed at line 54 and **never checked**. (`oracle.py` does fail
  if the SHA is absent — `git worktree add … check=True` raises — but `capture_output=True` discards
  git's message, so the log shows a bare `CalledProcessError`. Not "fails loudly" as line 36 claims.)
* **pytest exit code 5 is treated as a pass.** `grep -E '[0-9]+ (passed|failed|error)'` does not match
  `no tests collected`, so the echoed line is `rc=5 ::` with nothing after it. The handoff's own
  job-script habits say "report exit code 5 as a warning, never as a pass".
* `bench.py` computes `blocks_merge: bool(worst > 1.10)` and then **`return 0` unconditionally**
  (line 197). A >10 % regression prints the word `REGRESSION` into a log file and exits clean.

**Also latent:** `oracle.py` refuses an empty `--cells` match (line 338) but `bench.py` does not — an
empty `cells` list yields `rows=[]`, `worst=0.0`, `"within bar"`, exit 0. **A vacuous pass.** The
coordinator will hit this the moment he re-slices the cell lists for the merged job.

### S-5. One passing cell makes an arm pass, and the two classes with a CUDA graph can get zero coverage.

`ok = record["identical"] and record["lifetime_ok"] and ran` where `ran` is *any* cell with
`status == "ok"`. A cell whose dependency is missing is `status: "skipped"` on **both** arms, so it
produces no diff and is not a failure. There is no minimum-cell count and no per-backend coverage
check, even though `oracle.py:CELLS` carries a `backend` column expressly for that purpose.

The sharp case: `rendernet._Runner` and `LoadedCorrector` are reachable from exactly one cell,
`RoboCasaCounterToSinkCorrected`, which needs the 24 GB RoboCasa asset library **and** a sealed
corrector checkpoint. `rendernet.CACHE_ROOT` is
`os.path.expanduser(os.environ.get("STABLE_ENVS_RENDERNET_CACHE", "~/.cache/stable-envs/rendernet"))`
— it ignores `XDG_CACHE_HOME`, which is the only cache variable the sbatch redirects, and the sbatch
says at line 43 that `HOME` is at 100 % of quota and nothing may be written there. **The sbatch does
not export `STABLE_ENVS_RENDERNET_CACHE`.** If the checkpoint is not already under `$HOME`, that cell
skips, `_Runner`/`LoadedCorrector` get zero GPU coverage, and the arm reports PASS on the other cells.

Add `export STABLE_ENVS_RENDERNET_CACHE=$SCR/cache/rendernet` (and stage the checkpoint there), and
make `oracle.py` fail when the union of `backend` strings over `status == "ok"` cells does not cover
`mjwarp`, `lightspec`, `nvdiff` and `rendernet`.

### S-6. The one policy this patch actually changed in a render path is never exercised, and two recorded signals are never compared.

* **`_ray_cache` eviction is unreachable in this job.** Seeing LRU differ from `clear()`-on-full needs
  ≥9 distinct `(R, camera-orientation, fovy)` keys inside **one** renderer's lifetime, with a
  previously-used key re-requested after the overflow. Every arm renders one camera orientation per
  env and builds a fresh renderer per cycle and per width, so the cache never exceeds one or two
  entries. **The `popitem(last=False)` path never runs on a GPU.** This is the cheapest and most
  valuable addition: loop ≥9 rotated `cam.mat` values through one renderer on a fixed state and assert
  every frame equals the frame from a freshly-built renderer for the same camera. Same for
  `_CASTS_CACHE` at 65 keys, though there the consequence is only a recompute.
* **`backends_called` is recorded and never compared.** `oracle.py`'s docstring (lines 61–62) and the
  probe's comment (115–116) both claim that "a cell that stops reaching its module is itself a
  difference". `COMPARED` (lines 295–306) does **not** contain `backends_called`. The stated defence
  against a cell silently ceasing to use the backend is not wired up. One line fixes it. (I verified
  every instrumented name exists in its module, so the set itself is sound; and note that none of the
  five `Closeable` classes is instrumented, so `HITS` cannot report whether the owned *objects* were
  used, only the free functions.)
* **`closed_env_error` is recorded and never asserted.** Lines 239–244 catch bare `Exception`, so an
  `AttributeError: 'NoneType' object has no attribute 'qpos'` — a read through a released `_release`
  target, i.e. precisely the failure the design exists to prevent — counts as `closed_env_refuses:
  True`. Assert `closed_env_error == "UseAfterCloseError"`.
* **Memory residency after an overflow is not peak-bounded the way it was.** `clear()`-on-full
  oscillated 1…8 grids; LRU sits at 8. Peak is identical (both cap at 8), so this cannot OOM where
  the baseline did not — but a large-batch workload with ≥8 camera orientations now holds up to
  8 × `B·R·R·3·4` bytes steadily where it used to periodically return to one. At B=4096, R=240 that
  is ~2.8 GB per grid. The B=4096 bench leg uses one orientation and cannot see it. Low risk, worth
  one line in the job.

**What I would fold into the post-Wave-1 job**, in priority order: a direct create/close arm with
`memory_allocated()` before and after (S-1b, S-2); explicit test paths instead of `-k` (S-3); an
aggregated `FAIL_COUNT` with a final verdict line, exit-5 handling, and `bench.py` returning
non-zero on `blocks_merge` (S-4); `STABLE_ENVS_RENDERNET_CACHE` plus a per-backend coverage
assertion (S-5); a ≥9-orientation ray-cache loop and the three record-but-never-compare fixes (S-6).

---

## 14. Ruling asked for: should `BoundedCache` replace the `clear()`-on-full caches in `envs/*` in Wave 2?

**My recommendation: YES, but the ruling should be issued over the right number and with two
preconditions. It is not eight caches; it is fifteen.**

I scanned `stable_envs/envs/**/*.py` for the exact idiom (`if len(X) >= MAX: X.clear()`, regex over
source, capture group tying the two mentions together):

* **23 `clear()`-on-full sites** in `stable_envs/envs/`.
* **8** of them are the ones already on the consumer change list (`fetch.py:452`, `ogb_manip.py:470`,
  `robocasa.py:335`, `humanoidbench.py:519`, `metaworld.py:759`, `dmcontrol/trackcom.py:58`,
  `mujoco_locomotion.py:1639`, `rocket_landing.py:728`).
* **15 remain**: `classic_control.py:1410` (`_TERRAIN_CACHE`), `craftax_full.py:607` (`cls._cache`),
  `dmcontrol/suite.py:183` (`_SCENE_CACHE`), `fetch.py:242` (`_CAM_CACHE`, note `>` not `>=`),
  `fetch.py:667` (`_PLATE_CACHE`), `griddly.py:339` (`self._rgb_cache`), `maze2d.py:352`
  (`_SCENE_CACHE`), `metaworld.py:1101` (`_PLATE_CACHE`), `mujoco_locomotion.py:757`
  (`_SCENE_CACHE`), `navigation.py:834` (`_BAKE_CACHE`), `ogb_manip.py:822` (`_PLATE_CACHE`),
  `overcooked.py:385`, `pgx.py:609` (`_PLATE_CACHE`), `pgx.py:761`, `sokoban.py:337`.

(`envs/craftax.py:699` also matches but its guard is `if have is None and len(...) >= MAX`, i.e. it
already declines to clear on a hit; it is in the 23 and behaves like `classic_control`'s.)

**Is a bounded LRU the right semantics for them?** Yes. I looked for a property `clear()`-on-full
has that LRU loses and could not find one:

* *Lower average residency* is a cost property, not a correctness one, and **peak residency is
  identical** at the same capacity, so an LRU swap cannot OOM where `clear()` did not.
* *Pointer pinning* — the one real hazard, and the reason I checked `_CASTS_CACHE` so carefully — is
  preserved: `BoundedCache.put` drops key and value in the same critical section, so a key never
  outlives its referent.
* *Determinism* is preserved and slightly improved: a miss recomputes the identical value in every
  one of the 15 (each is a pure function of its key: a baked plate, a compiled scene, a terrain
  surface, a camera pose, an RGB frame), and LRU removes the "which unrelated call happened to
  overflow the cache" dependency that `clear()` introduces.
* `clear()`-on-full's *only* distinctive property is that it can throw away the entry the caller is
  holding right now. That is the defect, not a feature. The author's
  `test_a_key_in_use_survives_an_unrelated_insertion` pins exactly the right thing.

**Two preconditions on the ruling, both verified:**

1. **`BoundedCache` is not a `dict` and the sweep is not a one-liner per site.** It has
   `get/put/pop/clear/keys/__contains__/__len__/__iter__/__repr__` and deliberately **no
   `__getitem__`, no `__setitem__`, no `values()`, no `items()`**. I audited every one of the
   candidate caches for dict-surface use: all of them write via `C[key] = v` (`ogb_manip.py:533`,
   `fetch.py:485`, `robocasa.py:391`, `humanoidbench.py:576`, `metaworld.py:851`,
   `trackcom.py:105`, `_SCENE_CACHE[key] = sc` ×4, `_PLATE_CACHE[key] = …` ×4, `_NVD_CACHE[key] = c`
   ×2, `_BAKE_CACHE[ck] = hit`, `_CAM_CACHE[key] = (pos, q)`, `_TERRAIN_CACHE[key] = …`) and
   **`classic_control.py:1426` additionally reads via `return _TERRAIN_CACHE[key]`**. **None** uses
   `.values()`, `.items()`, `del`, or iteration. So the sweep is **two** mechanical edits per site
   (construction + `C[key] = v` → `C.put(key, v)`) plus one `__getitem__` read in
   `classic_control.py`. That is genuinely small and genuinely behaviour-preserving — but it is not
   "replace the constructor", and the ruling should say two lines, not one.
2. **The consumer change list's own lines assume a `dict`.** `for _c in _CTX_CACHE.values(): …` works
   today because `_CTX_CACHE` is a `dict`; it would **break** the moment that cache became a
   `BoundedCache`. So the twelve consumer lines and the K/O sweep must be **ordered**: either apply
   the release lines first and convert later (rewriting them to `on_evict`, which is the better end
   state and retires `.values()`), or convert first and write the release lines as `on_evict`
   directly. Do not let two workstreams do these independently. Likewise
   `robocasa._CORRECTOR_CACHE → BoundedCache(4, on_evict=close)` is **not** the one-liner the findings
   call it: `robocasa.py:928` is `hit = _CORRECTOR_CACHE[cls.__name__] = RN.load(…)` and must become
   `hit = _CORRECTOR_CACHE.put(cls.__name__, RN.load(…))`.

**And add a third, from S-2:** before the sweep, give `BoundedCache._free` an observable failure path
(below). A repo-wide cache whose release failures are `except Exception: pass` is a worse default than
the `clear()` it replaces, because `clear()` at least could not pretend to have released anything.

I recommend the sweep be assigned to **K** (one owner, one diff, one review) rather than split across
the fifteen files' owners — the author's §4 note that "it belongs to each file's owner" would produce
fifteen reviews of a two-line change and fifteen chances to get the ordering above wrong.

---

## 15. Consumer change list — verified line by line

I opened each cited line. All of them still exist and all of them are the right line.

| cited | what is actually there | verdict |
|---|---|---|
| `fetch.py:453` | `_CTX_CACHE.clear()` (guard on 452) | ✅ insert before 453 |
| `ogb_manip.py:471` | `_CTX_CACHE.clear()` | ✅ **still `envs/ogb_manip.py`** — D has not merged; there is no `envs/ogbench/` package on integration yet. It will move when D lands. |
| `robocasa.py:336` | `_CTX_CACHE.clear()` | ✅ |
| `humanoidbench.py:520` | `_CTX_CACHE.clear()` | ✅ |
| `metaworld.py:760` | `_CTX_CACHE.clear()` | ✅ |
| `mujoco_locomotion.py:1640` | `_NVD_CACHE.clear()` | ✅ `_c.br.close()` is the right call |
| `rocket_landing.py:729` | `_NVD_CACHE.clear()` | ✅ |
| `rocket_landing.py:739` | `c.br = ND.BatchRenderer(None, …, scene=scene, …)` — the call spans 739–741 | ✅ the cited line is exact; insert `c.br.owns_scene = True` after **741**, before `c.br.background` on 742 |
| dmcontrol `_TC_CACHE` | **moved.** It is no longer `envs/dmcontrol.py:2631`. Since B merged, the object is defined at **`stable_envs/envs/dmcontrol/_model.py:87`** (`_TC_CACHE`, `_TC_CACHE_MAX = 3` on 88), re-exported from `dmcontrol/__init__.py:86`, and **the `clear()` to insert before is at `stable_envs/envs/dmcontrol/trackcom.py:59`** (guard on 58; the write is `trackcom.py:105`). | ✅ retargeted |
| robocasa `_CORRECTOR_CACHE` | defined `robocasa.py:899`, used 926–928 | ⚠️ not a one-liner — see ruling precondition 2 |

I independently confirmed the two claims the list rests on:

* **`rocket_landing.py` is the only `scene=` call site in the repository.** `grep -rn "BatchRenderer("
  stable_envs/envs/` returns exactly two constructions: `rocket_landing.py:739` (scene handed in) and
  `mujoco_locomotion.py:1653` (`ND.BatchRenderer(m, …)`, builds its own). And `scene` at
  `rocket_landing.py:735` is a local that is read once more at 743 (`scene.V`) and then dropped, so
  `owns_scene = True` is correct and necessary there. ✅
* **`release_render_context` cannot move a pixel.** It does `rc._se_tex_lod_keep = None` inside a bare
  `try/except Exception: pass` and nothing else; it touches no EGL context, no `rgb_data`, no warp
  array, and it is idempotent. ✅

---

## 16. Ruling 4 — `dynamics.has_dynamics`, never `hasattr(env, "step")`

**Satisfied, and Workstream I introduces nothing that threatens it.** The harness asks
`has_dynamics`: `testing/render_worker.py:73`, `testing/conformance.py:1075`, `rollout.py:120`,
`_interop/worldmodel.py:558,821`, `registry.py:312`, `env.py:1142`. The only
`hasattr(env, "step")` in the package is `dynamics.py:49`, which is the documented fallback for
duck-typed stand-ins and is the file the repo-wide test excludes by name. That test —
`tests/unit/test_foundation_api.py::test_testing_harness_asks_has_dynamics_not_hasattr_step`, a regex
sweep over `stable_envs/**/*.py` — **passes in my run**. I also confirmed the diff adds no `hasattr`
of any kind.

---

## 17. The 28 lifetime tests — do they test release, or only that `close()` returns?

**Mostly the former.** These are better than the question implies, and I checked them for
unfalsifiability rather than reading the names. The ones that can genuinely fail:

* `test_close_releases_once_and_is_idempotent` — asserts `log == ["released"]` after three `close()`
  calls. An observable side effect, not a return value. Real.
* `test_use_after_close_raises_instead_of_reading_freed_state` — `pytest.raises(UseAfterCloseError)`
  **and** asserts the message names the object and the method. Real.
* `test_the_context_manager_closes_…including_on_an_exception` — asserts the release log on the
  `ValueError` path. Real.
* `test_correctness_never_depends_on_del` — drops the only reference, two `gc.collect()` rounds,
  asserts the log is **empty** and `not hasattr(Closeable, "__del__")`. This is the inverse test and
  it is the right one; it would fail the moment anybody adds a `__del__`. Real, and the most valuable
  test in the file.
* `test_a_half_built_object_is_not_open` — pins the `_se_closed = True` class default. Real.
* `test_the_counters_track_create_and_close_cycles` — 5 objects, double-close each, asserts
  `live()` returns to base and `delta` is 5/5. Real.
* `test_eviction_is_least_recently_used_and_one_entry_at_a_time` — asserts the surviving key set
  **and** `c.evictions == 1`. The `== 1` is what distinguishes LRU from `clear()`. Real.
* `test_a_key_in_use_survives_an_unrelated_insertion` — 20 interleaved `get`/`put` at capacity 2.
  The correctness property. Real.
* `test_eviction_can_release_…and_a_replacement_counts_as_a_drop`, `test_pop_hands_ownership_back_
  without_releasing`, `test_clear_releases_everything`, `test_a_failing_release_does_not_fail_the_
  insertion` — all assert `closed` on specific objects and not on others. Real.

Where they fall short, and it lines up exactly with the sbatch holes:

* **The five-class test is shape only** — `issubclass(Closeable)`, own `resource_kind`, `_release` in
  `__dict__`, no `__del__` in `__dict__`. It cannot detect a `_release` that frees the wrong thing or
  the wrong name. Given S-2, nothing else can either.
* **Nothing asserts that a class calls `_opened()`.** This is not cosmetic. `_se_closed` defaults to
  `True`, so a construction path that omits `_opened()` produces an object that is never counted,
  whose `close()` returns immediately **without running `_release`**, and whose guarded methods raise
  `UseAfterCloseError`. Two of the five have more than one construction path
  (`Scene.__init__` / `Scene.from_triangles`; `_Runner` only on first render). All five are correct
  today — I checked each — but the test cannot see it, and `from_triangles` is one `_opened()` line
  away from silently disabling release for the whole rocket path.
* **`nvdiff.Scene` is constructible without CUDA and the author did not use that.** The test file's
  header says the classes are checked by shape "rather than by construction, which needs a GPU" —
  **not true for `Scene`**: `tests/unit/test_nvdiff_triangles.py` already builds
  `ND.Scene.from_triangles(…, device="cpu")` in five places. A CPU test that builds a `Scene`, closes
  it, and asserts (i) every name in the `_release` list is `None`, (ii) `textures == {}`, and
  (iii) the name list *covers* every `torch` attribute `__init__`/`from_triangles` assign, would
  close the highest-risk gap in the patch for about fifteen lines and no GPU. This is the change I
  most want.
* **`test_nothing_in_the_render_backends_releases_an_egl_context` is a spelling tripwire, not a
  proof.** `_code_of` strips COMMENT and STRING tokens and then substring-searches for `". free ("`,
  `"GLContext"`, `"eglDestroyContext"`, `"mujoco . Renderer"`. `getattr(ctx, "free")()` passes,
  because the string is stripped — and that getattr-by-name idiom is *already in this patch*
  (`_Runner._release` does `getattr(g, "reset", None)`). `from mujoco import Renderer` passes. It is
  worth keeping as a tripwire; it should not be read as evidence. K will inherit it.
* `test_process_lifetime_retention_records_its_reason_and_is_capped` appends to the module-level
  `_RETAINED` with **no teardown**, so every suite run leaves `"a unit test"` in
  `retained_reasons()` for the rest of the session. Harmless now; it is a global-state mutation in a
  test and it will read as a real retention in any later diagnostic.

---

## Defects

**1. The GPU oracle cannot validate this patch, and it reports PASS. — SEVERITY: HIGH. BLOCKS: YES
(this is the merge condition).**
Nothing in `stable_envs/` calls `close()` on any of the five `Closeable` classes (§17-item-9), so no
arm of `gpu_oracle.sbatch` exercises a single `_release` body; `oracle.py` never asserts the
`_resources` counters the sbatch says it asserts (`LIFETIME_ASSERTIONS`, lines 309–317, versus the
sbatch's line 26); and no arm measures device memory after a close. **Merge condition:** the GPU
section must be repaired per §13 (at minimum: a direct create/close arm over the five classes with
`torch.cuda.memory_allocated()` around each `close()`, and explicit test paths instead of `-k`) and
must then pass, **or** the twelve consumer lines must land first so that `se_created == se_closed`
becomes an assertable property and the existing cycles arm becomes meaningful. As things stand a
green run of this job is not evidence.

**2. Counters report a failed release as a successful one, and `BoundedCache` swallows the failure
entirely. — SEVERITY: HIGH. BLOCKS: NO (fix before the Wave-2 sweep).**
`Closeable.close()` bumps `_CLOSED` in a `finally`, so `_release()` raising still counts as closed;
and `BoundedCache._free` is `except Exception: pass` with no counter, no log and no reason recorded.
I proved both with a probe class whose `_release` raises:

```
close() re-raised: the free failed
delta: {'created': {'probe.Boom': 1}, 'closed': {'probe.Boom': 1}, …}
live('probe.Boom') = 0        <-- counters say closed; nothing was freed

routed through on_evict:  exception surfaced? NO
  delta: {'created': {'probe.Boom': 2}, 'closed': {'probe.Boom': 1}, 'evicted': {'probe.cache': 1}}
  a.closed = True             <-- reported closed; its resource was never freed
```

So the counter arm the sbatch relies on cannot distinguish "released" from "tried and threw", and an
`on_evict` that fails systematically leaks every entry with zero signal. Fix: bump a
`_RELEASE_FAILED[kind]` counter (or log at WARNING through the Wave-0 logger) in both places, and
record the reason; then `counts()` can be asserted for real. `BoundedCache` must not become the
repo-wide cache (§14) with a silent swallow.

**3. Three record-but-never-compare signals, and one that cannot distinguish designed from broken. —
SEVERITY: MEDIUM. BLOCKS: NO (fold into the GPU job).**
`backends_called` is absent from `COMPARED` despite `oracle.py`'s docstring claiming it is the
defence against a cell ceasing to use the backend; `se_created`/`se_closed` are recorded and never
asserted; `bench.py`'s `peak_bytes` is overwritten per leg so only the head's survives;
`closed_env_error` is recorded and never asserted, and the surrounding `except Exception` makes an
`AttributeError` through released state count as a pass. All four are one-line fixes.

**4. The one place out of thirteen where the static argument is wrong: `retain_process_lifetime` can
raise where `_TEX_KEEP.extend` could not. — SEVERITY: MEDIUM. BLOCKS: NO.**
`mjwarp.create_render_context`'s `except AttributeError` fallback changed from
`_TEX_KEEP.extend(keep)` to a loop of `retain_process_lifetime(t, …)`, and that function **raises
`ResourceWarning` past `MAX_RETAINED = 256`** — a global, never-decreasing, all-reasons-shared cap.
`keep` holds one downsampled tensor per texture with `tex_lod > 1`, so a texture-rich model
(RoboCasa), or four cached contexts of a moderately textured one, can exceed 256 and **abort context
creation on a path that previously always succeeded**. The findings' table row says only "the tensor
kept alive is the same tensor; only the list it is appended to changed", which is incomplete: the new
list *refuses*. The branch is `# pragma: no cover` and dead on the pinned `mujoco_warp` 3.11 (the
attribute assignment succeeds), which is why nothing has noticed — but it is a latent hard failure
introduced by a refactor that claims to introduce none. Fix: cap per reason, or warn-and-retain
instead of raising on this path, and say so in the findings table.

**5. `nvdiff.Scene` has no `_check_open` on any method, so a closed `Scene` fails as an
`AttributeError` on `None` rather than as `UseAfterCloseError`. — SEVERITY: MEDIUM. BLOCKS: NO.**
`_resources`' stated promise is that "every later call through the object raises `UseAfterCloseError`
instead of reading a freed pointer". `Scene.clip` (`nvdiff.py:882`) is the method
`BatchRenderer.render` drives and it is unguarded, so the `owns_scene=False` case — a caller closes
its scene while a renderer still holds it, which is exactly the scenario `owns_scene`'s conservative
default exists to make impossible — surfaces as `'NoneType' object has no attribute …` from inside
the shading law. Add `self._check_open("clip")` at the top of `Scene.clip`. (`BatchRenderer.set_lights`
and `set_skybox` are likewise unguarded and will silently "work" on a closed renderer; lower stakes.)

**6. No test asserts that each construction path calls `_opened()`, and `Scene`'s release is
CPU-testable but untested. — SEVERITY: MEDIUM. BLOCKS: NO.**
See §17. All five are correct today; the failure mode if one regresses is an object that is never
counted and whose `close()` silently skips `_release`. And `Scene` — the class with sixteen
hand-listed attribute names behind a `hasattr` guard — is constructible on `device="cpu"`
(`tests/unit/test_nvdiff_triangles.py` does it five times), so the release test the author declined
to write as "needs a GPU" needs no GPU. Fifteen lines; highest value per line in the patch.

**7. The sbatch produces no verdict; five failure modes are indistinguishable from a pass. —
SEVERITY: MEDIUM. BLOCKS: NO (fold into the GPU job).**
No rc aggregation, no final PASS/FAIL, missing interpreter returns 0, dependency install failure is a
warning, `baseline_present` printed and unchecked, pytest exit 5 silently accepted, `bench.py`
`return 0` on a declared `blocks_merge`, and `bench.py` gives a vacuous pass on an empty cell match.
Details in §13/S-4.

**8. `STABLE_ENVS_RENDERNET_CACHE` is not exported, so the only cell that reaches two of the five
classes can silently skip. — SEVERITY: MEDIUM. BLOCKS: NO (fold into the GPU job).**
`rendernet.CACHE_ROOT` ignores `XDG_CACHE_HOME` and defaults under `$HOME`, which the sbatch itself
says is full. One `export` plus staging the checkpoint under `$SCR`, and a per-backend coverage
assertion so the skip fails rather than passes. Details in §13/S-5.

**9. The `-k` filter deselects `test_conformance.py` and `test_fast_paths.py`, two of the three
harnesses the findings declare applicable — including the official-renderer parity check. —
SEVERITY: MEDIUM. BLOCKS: NO (fold into the GPU job).**
Verified by running the selection: `no tests collected (104 deselected)`. Replace with explicit paths.
Details in §13/S-3.

**10. `examples/results/step9/render-mjwarp-nvdiff/findings.md` has three numbers I could not
reproduce. — SEVERITY: LOW. BLOCKS: NO.**
(a) "eleven removed lines" — I count **thirteen** code lines (§11). (b) "the eight remaining
`clear()`-on-full caches in `envs/*`" — I count **23** such sites, 8 of them already on the consumer
list, leaving **15** (§14); the ruling is being asked over roughly twice the scope stated.
(c) §5 attributes the "+1 skipped" to the dependency-direction param, which in fact contributes +1
**passed**; the skip is the `mujoco_warp` cast-memo check. The totals reconcile either way. None of
these changes a conclusion, but (b) materially changes the size of the Wave-2 ruling.

**11. `Closeable.close()` is not thread-safe while the rest of the module is. — SEVERITY: LOW.
BLOCKS: NO.**
`_resources` guards its counters with `_LOCK` and `BoundedCache` with a per-instance lock, but
`close()` reads and writes `_se_closed` unprotected, so two threads can both run `_release`. Nothing
in this repository closes a renderer from two threads, and `_release` is idempotent in practice — but
the inconsistency will read as an oversight to the next person. Either lock it or say in the docstring
that a `Closeable` is single-threaded by contract.

**12. `delta()`'s return shape is inconsistent. — SEVERITY: LOW. BLOCKS: NO.**
`created`/`closed`/`evicted` map kind → count; `retained` maps `{"retained": n}`. Harmless, and
`oracle.py` does not consume it, but it will trip the first person who writes a generic consumer.

---

## Commands run (reviewer)

```bash
# provenance
cd /Users/samib/stable-envs-step9-render-mjwarp-nvdiff
git rev-parse HEAD                       # 50acdf81eb98a7c64cb81b8979d414149c09d298
git merge-base --is-ancestor 312a631 HEAD && echo ancestor
git log --stat step9/integration..HEAD
git diff 312a631..50acdf8 --stat
git diff 312a631..50acdf8 -- stable_envs/envs stable_envs/env.py stable_envs/testing tests/gpu   # empty
git diff --stat 21cb306..50acdf8 -- stable_envs/testing/resource_guard.py                        # empty

# the suite, both trees, same venv, same command
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit -x \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
#   at /Users/samib/stable-envs-step9-integration (312a631): 1533 passed, 198 skipped, 112 deselected, 154.72s
#   at this branch                          (50acdf8): 1561 passed, 199 skipped, 112 deselected, 114.83s
#   safety footer IDENTICAL on both: 20 EGL created / 0 released / 20 live (peak 20); renderers 19 / 0

MUJOCO_GL=glfw … -m pytest -q tests/unit/test_render_resources.py -rs        # 27 passed, 1 skipped
MUJOCO_GL=glfw … -m pytest -q tests/unit/test_foundation_imports.py \
  -k renderer_knows_nothing                                                  # 17 passed, 9 deselected

# the sbatch's pytest selection, run as written
MUJOCO_GL=glfw … -m pytest -q tests/gpu \
  -k "mjwarp or nvdiff or rendernet or resource or lifetime or reproducib" --collect-only
#   -> no tests collected (104 deselected)   [rc 5]

# the removed lines, re-derived
git diff 312a631..50acdf8 -- stable_envs/render/ | grep '^-' | grep -v '^---' \
  | sed 's/^-//;s/^[[:space:]]*//' | grep -v '^#' | grep -v '^$' | cat -n      # 13 lines

# the clear()-on-full census (regex ties len(X) to X.clear())
python - <<'PY'   # over stable_envs/envs/**/*.py
# 23 sites; 8 on the consumer change list; 15 remaining
PY

# protocol name-collision audit over the five classes (AST: own defs + self-assigned attrs)
# -> only _release and resource_kind overlap; no __slots__ on any of the five

# the counter-vs-failed-release proof
python - <<'PY'
class Boom(R.Closeable):
    resource_kind = "probe.Boom"
    def __init__(self): self._opened()
    def _release(self): raise RuntimeError("the free failed")
PY
# -> live()==0 and delta.closed==1 although nothing was freed; via on_evict the exception vanishes

# import-level verification of the re-export surface
python -c "import stable_envs.render.mjwarp as MW; [print(n, getattr(MW,n,'MISSING')) for n in
 ('HIDDEN_GROUP','DEFAULT_GROUPS','gl_rounding','create_render_context','release_render_context')]"
```

Read-only throughout: no edits, no commits, no stashes, no resets, no rebases, no pushes. Oscar was
not contacted and `gpu_oracle.sbatch` was not submitted.
