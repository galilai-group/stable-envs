# Step 9 — Wave 1, Workstream B (DM Control) — independent review

APPROVE

**Reviewed SHA:** `82ce271b49ce485c91f1aa85777f5ab222f6715f` (branch `step9/envs-dmcontrol`)
**Delta reviewed:** `git diff b3f3a81 82ce271` (42 files, +22,359 / −3,391). The branch merged
`step9/integration` at `7c5c7d8`; foundation and Workstream-H code is out of scope and was not reviewed.
**Reviewer host:** Apple M5 Mac (`cis1b1zs2j3`), CPU only. Venvs `/Users/samib/se8-vm-venv`
(dm_control 1.0.44 + mujoco 3.11 + swm + tensordict) and `/Users/samib/se6-venv`.
`MUJOCO_GL=glfw`, `-p no:cacheprovider`. No edits, no commits, no Oscar access.

Every claim below was checked against an artifact or a command I ran, not against the findings prose.

---

## Verdict summary

The migration is behaviour-preserving to the byte on every axis I could measure locally, and the
evidence is reproducible: I re-ran the author's CPU oracle at HEAD into a temp directory and the
resulting record is **identical to the committed `oracle.json` in every leaf but `head_sha`**, and I
re-ran the T3 bake-vs-recompile probe for one colour and got the committed digest. The two suites
reproduce the claimed counts exactly (1513 and 436). Scope is clean, the template rules hold, the four
hand-written tables are provably equal to the Step-8 literals, and the two re-pointed tests are
strengthened rather than weakened. GPU evidence is real, carries no `/oscar` path, and its
carry-forward argument is sound.

---

## 1. Scope

`git diff --stat b3f3a81 82ce271` touches exactly:

* `stable_envs/envs/dmcontrol/` — 22 modules (`ls` confirms 22 `.py` + `__pycache__`);
* `stable_envs/envs/dmcontrol.py` — **removed** (3,376 lines);
* `tests/unit/test_declarations.py`, `tests/unit/test_dmcontrol_tendon.py` — re-pointed;
* `tests/unit/test_envs_dmcontrol.py` — new (131 collected);
* `examples/results/step9/envs-dmcontrol/*` (incl. `gpu/`), `scratch/step9/envs-dmcontrol/*`.

**No** edit to `stable_envs/render/*`, `stable_envs/env.py`, `stable_envs/registry.py`,
`_plate_texels`, or any variation map. Confirmed by the file list; nothing else appears in the diff.

**Module→package removal (handoff §4 ruling 6).** The ruling's obligation is a test that every Step-8
module-level name still resolves. `oracle.py` reads the names off the baseline SOURCE by AST
(`21cb306:stable_envs/envs/dmcontrol.py`) rather than from a transcribed list, so the list cannot
drift: `step8_module_names_checked: 79`, `step8_module_names_missing: []`, one declared
`expected_differences` entry (`_make_classes`, the private `type()` loop, with its reason). I confirm
the reasoning that a shim is impossible here — a module and a package cannot share a name — so the
`swm_nav.py` pattern does not apply and the 79-name check is the right substitute. Obligation met.

**Re-pointed tests — nothing weakened.**

* `tests/unit/test_declarations.py:649-687` — the AST walk moved from one file to every module of the
  package, `_cells.py` first so the shared `_AT`/`_AT_VERSIONS`/`_AT_MEASURED_WITH` names are bound
  before the per-domain tables that reference them, `__init__.py` skipped because its
  `VISUAL_FIDELITY` is a view of the classes rather than a literal. Same `want` set, same
  never-import discipline. It is **stronger**: `assert table, "the walk has drifted"` (line 685) turns
  a drifted glob into a failure instead of a vacuous pass.
* `tests/unit/test_dmcontrol_tendon.py:158-188` — the substitution now happens where each name is
  looked up (`D.suite._chain_capsules_batch`, `D._tendon.catenary`) instead of on the package object,
  which after the split would have left the real implementations running on both sides. The
  substitution is **counted and asserted** (`assert calls["chain"] and calls["cat"]`), so the vacuous
  comparison is closed structurally, not once. The reference implementations and the comparison
  tolerances are untouched.

## 2. Behaviour preservation (plan §12 B)

**Oracle re-run, independently, at the reviewed HEAD.**

```
cd /Users/samib/stable-envs-step9-envs-dmcontrol
MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python scratch/step9/envs-dmcontrol/oracle.py \
  --out $SCRATCH/oracle_rerun --name oracle.json --cache $SCRATCH/corpus
```
```
 "identical": true,
 "unexplained_differences": {},
 "step8_module_names_checked": 79,
 "step8_module_names_missing": [],
 "expected_differences": {"_make_classes": "...the private factory..."},
 "functional_ok": true
  DMCAcrobotSwingup          frames b0a1daa2adf19cc0  states 64
  ...
  DMCSwimmerSwimmer6         frames None  states None
  DMCWalkerWalk              frames d8d30e1c869a9e1b  states 64
ORACLE: PASS
```

Leaf-by-leaf diff of my record against the committed `examples/results/step9/envs-dmcontrol/oracle.json`:

```
total differing leaves: 1
  /head_sha | 7c5c7d8d6026d2904981931289adfa718d88b029 -> 82ce271b49ce485c91f1aa85777f5ab222f6715f
compared keys: 35 = 35
```

So: **35 compared keys, 17 cpu frame digests, all identical to the committed record and to the
baseline tree; unexplained differences `{}`**. The 17 digests also match the table in findings §6
hex for hex. I read `oracle.py`'s comparison logic (lines 320-462): absent keys are compared as
`"<absent>"`, so a migrated-side render failure cannot hide; `corpus_sha256` is compared so both arms
demonstrably replay one array (the two corpus-less cells are drawn by the baseline arm and cached for
the migrated arm); `registry_row` is compared in full including `module`. I found no vacuity.

**mujoco 3.11 provenance.** From the oracle records: `DMCCheetahRun`/`DMCQuadrupedWalk`
`reference_versions == {"mujoco": "3.11.0"}`, `measured_with == {"mujoco_warp": "3.11.0"}`;
`DMCSwimmerSwimmer6` (cuda arm) `{"mujoco": "3.11.0"}` / `{"warp": "1.16.0"}`. Identical to baseline
in both trees and pinned again as literals in `tests/unit/test_envs_dmcontrol.py:181-185`.

**T3 `floor.color`.** Present on **12/12** parity cells (counted off `variation_names` in the oracle
record). I re-ran the probe for one colour:

```
PYTHONPATH=<worktree> MUJOCO_GL=glfw se8-vm-venv/bin/python scratch/step8/capability/bake_vs_recompile.py \
  --mode recompile|substitute --probe 0 --cells DMCCartpoleBalance ... --compare ...
VERDICT IDENTICAL
 recompile_frames_sha256 = substitute_frames_sha256 = 0d8b1ba9299ec1e307d1e3ddb6ec32cfada8053509d7ae2373e0b9e16241feac
 tex_identical: true, frames_identical: true
```
which is the exact digest in the committed `t3_bake_vs_recompile.json`. The quantised cache key, the
texel substitution and the in-place upload on the existing context still equal a full swm recompile.

**Manipulator dead dims.** `RENDER_INVARIANT_DIMS == {"DMCManipulatorBringBall": ("ball_y",
"target_ball_quat_qw", "target_ball_quat_qy")}` and `state_dims_note` byte-equal between the trees,
and asserted as literals plus "nobody else's" at `tests/unit/test_envs_dmcontrol.py:192-196`.

**trackcom / `_TCtx` / cpu3d.** I diffed the migrated `DMControlTrackCom` class body against the same
region of `git show 21cb306:stable_envs/envs/dmcontrol.py`: **one substantive line differs**,
`reason = CPU_UNCERTIFIED.get(type(self).__name__)` → `reason = type(self).CPU_UNCERTIFIED_REASON or None`
(the rest is trailing blank lines). `DMControlSuite` differs by **nothing** but trailing blank lines
over 1,111 lines. `_TCtx`, `_tc_context`, `_tc_frame` moved verbatim.

**Variation maps.** `tests/unit/test_variation_maps_dmc_control.py` (which asserts per-cell
`classification` counts against the declaration, the 161-leaf total, isolation, round-trip and the
registered-default reproduction) passes against live `stable_worldmodel` in `se8-vm-venv` — see §5.

**Fidelity declarations / parity floors.** Compared in the oracle (`visual_fidelity` in full,
`parity_interior_min`, `parity_all_exact_min`, `parity_note`, `fidelity_tier`, `render_decoder`,
`docstring`) — identical. Independently, I AST-extracted the Step-8 literals and compared them to the
package's runtime values (see §3): **0 differences** across `VISUAL_FIDELITY`, `_AT`, `_SWIMMER_AT`,
`_QUADRUPED_AT`, all four `*_VERSIONS`/`*_MEASURED_WITH`, `PARITY_NOTE`, `VISUAL_NOTE`,
`SPECULAR_NOTE`, `QUADRUPED_NOTE`, `ALL_EXACT_FLOOR`, `TRACKCOM_RENDER_FLAGS`, `SHAPE_KNOB_SPECS`,
`RENDER_INVARIANT_DIMS`, `STATE_DIMS_NOTE`. **`SPECULAR_NOTE` is byte-identical to Step 8** — untouched,
as required.

**`_bundle` one-Renderer-per-(domain,size).** The caches are single objects across the package; I
asserted the identities at runtime myself (`_ENV_CACHE`/`_SCENE_CACHE`/`_TC_CACHE` in `suite`,
`trackcom`, `base` and the package are the `_model` objects) — no divergence. `_close` drops only the
per-instance `_cpu_scenes`; neither tier defines `_close` (`base.py:141-152`). The policy of never
releasing the shared EGL context is preserved and now asserted
(`tests/unit/test_envs_dmcontrol.py:208-221`).

**CPU refusal.** `DMCSwimmerSwimmer6` refuses `device="cpu"` in both trees with the identical message
(compared as `construct_refused`), and the refusal now reads the cell's own
`CPU_UNCERTIFIED_REASON` rather than a name lookup in a side table — same string, one fewer place to
drift.

## 3. Template conformance

* **No `state` override, no `_layout` override.** `"_layout" in vars(_DMCModel)` → `False`;
  `"state" in vars(_DMCModel)` → `False`. `state_layout` is assigned in `__init__`
  (`base.py:103-109`), which is exactly handoff ruling 2. Also no `_hidden_state`/`_live_state`;
  the oracle's functional arm confirms `no_hidden_fields` for all 18.
* **`num_envs=` everywhere.** `_DMCModel.__init__(num_envs=1, ..., batch=None)` (`base.py:88-113`),
  forwarded to `Env`; 18/18 accept it in the oracle's functional checks and in
  `test_takes_the_step9_constructor_and_the_lifecycle`.
* **Renderer-only refusal.** `has_dynamics` is `False`; I called `step` and `transition` on
  `DMCCartpoleBalance` and both raise the contract's `NotImplementedError` naming the cell and the
  reason ("…ships a renderer and no batched simulator… PushT has dynamics; DMCCartpoleBalance does
  not…"). `reset`/`render` work. Pinned for all eighteen.
* **Three-level hierarchy, data over subclasses.** `DMControlSuite.__bases__ == (_DMCModel,)`,
  `DMControlTrackCom.__bases__ == (_DMCModel,)`, `_DMCModel.__bases__ == (Env,)`. I enumerated all 18
  leaves: **no leaf namespace holds a callable**, and the only extra base is `_SWMQuadrupedModel` on
  `DMCQuadrupedWalk` — which is the Step-8 `TRACKCOM_BASES` entry, i.e. unchanged, not new.
  Every cell's `__module__` is `stable_envs.envs.dmcontrol`, so the registry `family` column and the
  `module` column are unchanged and truthful (the package does re-export all eighteen).
* **The four tables are views, and the views equal the baseline tables.** I verified this
  independently of the author's test: AST-exec of the Step-8 literals vs the package's runtime values —

  ```
  TABLE DIFFS: 0
  ```
  covering `DEFAULT_SIZES` (`{"DMCManipulatorBringBall": 224}`), `CPU_UNCERTIFIED`
  (`{"DMCSwimmerSwimmer6": "no CPU certification (Step 7 allow-list; …)"}`), `TRACKCOM_BASES`
  (`{"DMCQuadrupedWalk": (_SWMQuadrupedModel,)}`), `VISUAL_FIDELITY` in full, and the declared
  provenance/note strings listed in §2.

## 4. GPU evidence (job 6469088)

`examples/results/step9/envs-dmcontrol/gpu/` carries the artifacts and they say what the findings say:

* `provenance_head.txt`: `host gpu3001.oscar.ccv.brown.edu`, `job 6469088`, `sha 0f610cf…`,
  `NVIDIA L40S, 575.57.08`, `mujoco 3.11.0 torch 2.14.0+cu126`, `python $SCR/venvs/se-vis311/bin/python`.
* `oracle_head_cuda.json`: `compare_device cuda`, `identical true`, `unexplained_differences {}`,
  `functional_ok true`, 79 names checked / 0 missing; `DMCSwimmerSwimmer6` has a cuda frame digest
  `6f1387abd1f74c89` — so MJWarp and the Swimmer digest were genuinely compared.
  `oracle_head_cpu.json` likewise `identical true`.
* `pytest_conformance.txt` **239 passed, 0 failed**; `pytest_reproducibility.txt` **345 passed,
  0 failed**; `pytest_fast_paths.txt` **78 passed, 0 failed**; both the conformance and reproducibility
  runs print `reference-renderer={'env': 'DMCAcrobotSwingup', 'size': 84, 'max_abs_diff': 0}`.
* `performance_cuda.json`: −0.24 % (`DMCCartpoleBalance`), −0.23 % (`DMCReacherHard`), +0.40 %
  (`DMCCheetahRun`), with overlapping per-repeat lists. Far inside plan §8.
* `grep -rl "/oscar" examples/results/step9/envs-dmcontrol/gpu/` → **no hits**. Paths are scrubbed to `$SCR`.

**Carry-forward judgement: accepted — no GPU re-run is required before merge.**
`git diff 0f610cf 82ce271 -- stable_envs/envs/dmcontrol` is one file, `base.py`, +18/−25, and I read
all of it: three docstring/comment paragraphs, plus moving the identical layout computation out of the
deleted `_layout` method into an `__init__` assignment that produces the same tuple. No renderer, no
backend selection, no `device`/`cuda`/`cpu` branch, no render import. My CPU oracle at the reviewed
head is byte-identical to the one taken at `0f610cf`, which is the independent confirmation. The one
genuine gap the author states plainly — Workstream H's raycast3d/cpu3d consolidation arrived after the
job's clone and has not been exercised on DMC cells on GPU — is **exactly what Sami's cadence ruling
(handoff §4 item 8: one GPU oracle over all 42 ids after the Wave-1 merges) is for**, and H's own GPU
oracle already covers those files. So: **wait for the post-Wave-1 GPU oracle**; `gpu.sbatch` is
committed and re-runnable in ~12 minutes if the coordinator wants it pinned earlier.

## 5. Tests (reproduced by the reviewer)

```
cd /Users/samib/stable-envs-step9-envs-dmcontrol
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit -x \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
1513 passed, 198 skipped, 112 deselected, 29521 warnings in 131.65s (0:02:11)
```
```
MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_variation_maps_dmc_control.py tests/unit/test_step8_gate_record.py \
  tests/unit/test_step8_provenance.py tests/swm tests/unit/test_envs_dmcontrol.py \
  tests/unit/test_dmcontrol_tendon.py tests/unit/test_dmcontrol_episode_quat.py \
  tests/unit/test_cpu3d.py tests/unit/test_declarations.py
436 passed, 4 skipped, 32719 warnings in 70.88s (0:01:10)
```
Both match the expected 1513 and 436 exactly. (The brief's shorter se8 selection — without the tendon,
episode-quat and cpu3d files — gives `385 passed, 4 skipped`; the author's 436 is the same run plus
those three files, and I ran the author's exact selection above to reproduce the number.)
`tests/unit/test_envs_dmcontrol.py` collects **131**. I read the structural tests: they compare each
table entry against the class it was collected from, assert the leaf namespaces hold no callables,
assert the cache object identity, and assert `"_layout" not in vars(_DMCModel)` — none is vacuous, and
the three local skips are the honest `DMCSwimmerSwimmer6` CPU-refusal skips in the three behavioural
tests (its refusal is asserted, not skipped, in the lifecycle test).

## 6. The §17 ten items

1. **Files moved/removed/created** — PASS. `envs/dmcontrol.py` (3,376 ln) → `envs/dmcontrol/` (22
   modules, 3,714 ln); one new test file; scratch + artifacts. Only `_make_classes` is not carried,
   declared with its reason and never referenced outside the old module.
2. **API changes** — PASS, additive. `num_envs=` (`batch=` still accepted), a two-key `State`
   (`qpos` + `episode`), `close()`/context manager, four hand-written tables replaced by collected
   views. No `state`/`_layout`/`_hidden_state`/`_live_state` override.
3. **Step-8 IDs affected** — PASS, behaviourally none. 18 cells unchanged in name, tier, decoder,
   `state_dim`, `reference_id`, `module` and registry row.
4. **Targeted tests** — PASS. 131 new tests; 1513 local / 436 SWM-venv reproduced by me; two
   source-reading tests re-pointed, one materially strengthened.
5. **Applicable certification checks** — PASS. Corpus replay for all 18, T3 `floor.color` re-verified
   by me for one colour, variation-map suite green against live swm, three GPU suites green at
   mujoco 3.11 with `max_abs_diff: 0` against dm_control's own renderer.
6. **Default-frame comparison** — PASS. 17 cpu digests identical (re-derived by me, hex for hex) and
   all 18 on cuda, including Swimmer, which cpu cannot draw by declaration.
7. **Variation/state comparison** — PASS. Style layout, default appearance, taxonomy leaf for leaf,
   packed layout, `set_state`/`get_state` round-trip, `sample_states`, every fidelity and provenance
   declaration: identical.
8. **Performance comparison for hot paths** — PASS. CPU −0.44 % (parity/raycast3d) and −1.75 %
   (cpu3d); CUDA −0.24 % / −0.23 % / +0.40 % (incl. MJWarp). All inside run-to-run spread, two orders
   of magnitude under the §8 bar. The absence of a `step` number is correct: no DMC cell has dynamics.
9. **Resource-lifetime comparison** — PASS. One `mujoco.Renderer` per (domain, size), never closed;
   `_close` drops only the per-instance CPU scene cache; cache-object identity now asserted. The
   EGL-context counts in the suite footer are unchanged Step-8 behaviour.
10. **Unexplained oracle differences** — **zero**, confirmed by my own re-run.

## 7. Defects and notes (none blocking)

No correctness defect found. Four minor notes, all cosmetic or coordinator-facing:

1. `examples/results/step9/envs-dmcontrol/oracle.json` — `head_sha` is `7c5c7d8`, the *parent* of the
   commit whose tree was measured (`d5992f7`, which changed `base.py`). The author's convention ("the
   commit that carries a record cannot be the one it measured") explains it, but here it reads as if
   the record predates the R1 change. My re-run at `82ce271` settles it: every other leaf is identical.
   Suggest recording the measured tree explicitly (`--head-sha`) in future records.
2. `examples/results/step9/envs-dmcontrol/findings.md:36,40` — "3,721 lines" / "`base.py` 548" are
   stale by 7 lines after the R1 deletion (actual: 3,714 total, `base.py` 541). Cosmetic.
3. R4 remains open by coordinator deferral: `SPECULAR_NOTE` (reaching `DMCSwimmerSwimmer6.visual_note`
   and therefore `describe()`) self-cites `stable_envs/envs/dmcontrol.py`, which is now a directory.
   Correctly deferred to Wave 3 — editing it would break the byte-identity the oracle rests on — but it
   should stay on the Wave-3 list, and the same wording problem will recur for Workstream D.
4. `scratch/step9/envs-dmcontrol/oracle.py:305-317` — `_baseline_worktree` returns an existing temp
   path without confirming it is a registered worktree, and only removes what it created. A stale
   `/private/tmp/step9-oracle-baseline-21cb30678b7a` worktree registration is left behind in the repo
   (pre-existing; I did not create or remove it). Housekeeping for the coordinator, scratch-only.

**Verdict: APPROVE.** Merge does not need to wait for a GPU re-run; the DM Control cells' GPU coverage
at the merged head belongs to the single post-Wave-1 GPU oracle per Sami's cadence ruling.
