APPROVE

# Step 9 — independent review of Workstream E (Craftax + Rocket)

**Reviewed** `step9/envs-craftax-rocket` at **`5547270`** (author's delivery `4f7350d` + coordinator's
merge of `step9/integration` `312a631`, bringing in Workstream B and Workstream G's 2-D render-backend
consolidation) · Step-8 behavioural baseline `21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`.
**Reviewer host** `cis1b1zs2j3.ad.brown.edu`, local CPU only (no CUDA on this host, per instructions —
Rocket pixel evidence is explicitly owed to the post-Wave-1 GPU oracle, ruling 8), `MUJOCO_GL=glfw`,
`-p no:cacheprovider`, venvs `/Users/samib/se6-venv`, `/Users/samib/se7-craftax-venv`,
`/Users/samib/se7-rocket-venv`, `/Users/samib/se8-vm-venv`. Nothing was edited, committed, stashed,
reset or pushed; Oscar was not touched. The oracle harness makes and removes its own detached
`git worktree` of `21cb306` under `$TMPDIR`; it cleaned up in both runs below.

## Verdict

**APPROVE.** I re-ran the CPU oracle myself, from scratch, into a fresh temp dir, at `5547270` — i.e.
*after* the merge that brought in Workstream G's tilemap/raster2d/gfxdraw/navdot/prng consolidation,
which is the change most likely to have silently moved a Craftax pixel. Both Craftax cells reproduce
byte-identical frames on all three frozen corpora at both certified sizes (six (cell, corpus, size)
combinations, and the four Step-7 combinations reproduce the values frozen in
`corpus/states.meta.json` to the digit). Rocket's CPU-determined host-side chain (`geom_poses`,
`lagged_pose`, `camera_pose`, `material_table`, `build_scene_arrays`, `checker_texture`) is also
byte-identical. **Unexplained differences: zero. REGRESSIONS: 0.** Ruling 2 (no `state` override),
Ruling 3 (`unpack` copies) and Ruling 9 (both Craftax corpora kept) all hold, independently checked
against the actual code and a live interpreter, not just read off prose. The `sys.modules` alias is
the real thing — same module object, a monkeypatch through either spelling is seen by both, every
Step-8 module-level name resolves.

One finding stops this short of a silent rubber stamp: the branch's central quantitative claim —
"21 module-level names differ between `craftax.py` and `_craftax_full.py`" — is **not accurate**. I
measured 12 by direct comparison of every name in both modules' `dir()`, and a closer look at the
named list in `findings.md`/the module docstring shows 8 of the 20 explicitly-named symbols
(`N_FACING`, `N_KEY_HALVES`, `NIGHT_TINT`, `NIGHT_THRESHOLD`, `DAY_LENGTH`, `NO_PLAYER`,
`PLAYER_ASLEEP`, `TILE_PX_IMG`) hold **identical** values in both modules and are not an obstacle to
anything. `findings.md`'s own evidence table also states `N_TILE | 160 | 99`; the live value of
`craftax.N_TILE` at this SHA is **976**, not 160 (160 is `N_STRIP * N_COUNT`, a different quantity —
`craftax.N_TILE` is defined as `N_COMPOSITE + N_STRIP * N_COUNT` with `N_COMPOSITE = 816`). This does
not change the engineering conclusion — I independently verified at least 15 genuine, substantive
conflicts (9 numeric constants that really do differ, 3 module-level dict caches that are separate
objects with the same name and would collide on a merge, and 3 functions with genuinely different
bodies, not just different `id()`s) — comfortably enough to justify keeping the private module rather
than merging, which is the only decision the count is used to support. So this is a documentation
defect, not a design defect, and it does not block. See D-1 below.

## The ten §17 items — my own numbers

1. **Files moved/removed/created** — Confirmed against `git diff 5741b628..4f7350d` (the workstream's
   own commits, parent `step9/integration` before this merge): `stable_envs/envs/craftax_full.py`
   reduced from 1,549 to 24 lines (a `sys.modules` alias); `stable_envs/envs/_craftax_full.py` created,
   1,568 lines; `stable_envs/envs/craftax.py` and `stable_envs/envs/rocket_landing.py` edited in place
   (+39/+51 lines); `stable_envs/envs/_craftax_full_tex.py` untouched (confirmed: not in the diff).
   Created: `tests/unit/test_envs_craftax_rocket.py` (34 tests), `scratch/step9/envs-craftax-rocket/
   {oracle.py,perf.py}`, five files under `examples/results/step9/envs-craftax-rocket/`. Nothing
   importable removed. Zero coordinator-owned files touched — `git diff 5741b628..4f7350d --
   stable_envs/__init__.py stable_envs/env.py stable_envs/dynamics.py stable_envs/specs.py
   stable_envs/registry.py stable_envs/_config.py pyproject.toml` is empty (I ran this myself). The
   merge commit `5547270` touches none of E's own files (`git diff 4f7350d 5547270 -- stable_envs/envs
   tests/unit/test_envs_craftax_rocket.py` is empty) — a clean, non-conflicting merge.
2. **API changes** — Additive: `num_envs=` (both Craftax cells inherit it from `Env.__init__`;
   `PFRocketLanding` needed an explicit signature change since it declared `batch=` itself — verified
   `batch=` is still accepted positionally and by keyword), the public `State` (three group layouts,
   spot-checked against `state_layout`), `frame_shape` (a genuine bug fix — the old
   `observation_spec` reported `(16,16,3)`/`(10,10,3)` for frames that are actually `(144,144,3)`/
   `(130,110,3)`; confirmed by reading `Env.observation_spec`'s square-frame assumption against the
   overridden property in both files), and `PFRocketLanding._close` (a documented no-op that
   deliberately does not tear down the shared `_NVD_CACHE`, for the stated use-after-free reason — I
   agree with the reasoning: the cache is keyed `(size, supersample, device)` and shared across live
   envs). No `state`/`pack`/`unpack` override anywhere in the three files (grepped directly; the only
   `pack` hit in `_craftax_full.py` is an unrelated local closure inside a texture-loading function,
   not `Env.pack`).
3. **Step-8 IDs affected** — Confirmed 5 of 42: `swm/CraftaxClassicPixels-v1`, `swm/
   CraftaxClassicSymbolic-v1` (still refuses, unchanged), `swm/CraftaxPixels-v1`, `swm/
   CraftaxSymbolic-v1` (still refuses, unchanged), `swm/PFRocketLanding-v0`. Behaviorally none
   affected; the two pixel ids are byte-identical (item 6), Rocket's pixels are owed.
4. **Targeted tests** — `tests/unit/test_envs_craftax_rocket.py`, 34 tests, re-run as part of the full
   suite below; spot-read six of them in full (state-copy-not-view, no-dynamics refusal wording, the
   alias identity, the merge-evidence test, the corpus-both-kept test, the num_envs/batch signature
   test) — each pins what it claims to.
5. **Applicable certification checks** — `test_step8_semantic_craftax.py`, `test_step8_gate_record.py`,
   `test_craftax_atlas.py`, `test_craftax_full.py`, `test_craftax_prng.py`, `test_craftax_strip.py` all
   re-run in `se7-craftax-venv` (item 4 below): pass. Gate A / GPU conformance / tier measurements /
   all of Rocket's pixels correctly deferred as owed (O1-O4 in the findings).
6. **Default-frame comparison — re-run by me from scratch, not read off the committed JSON.** See
   "CPU oracle" below. Both Craftax cells, all three corpora, both certified sizes each: six
   (cell, corpus, size) combinations, all frame digests byte-identical to the committed record and to
   each other across baseline/migrated arms, and the four Step-7 combinations reproduce
   `corpus/states.meta.json`'s frozen digests to the character.
7. **Variation/state comparison** — Covered in the same oracle run: `describe()`, style layout/default
   appearance, `variation_space` taxonomy, registry row (module-path exception noted and checked, see
   item 1 of "expected differences" below), `state_dim`/`state_names`, `set_state`/`get_state`
   round-trip, `sample_states(64, seed 0)` — all identical in my independent run.
8. **Performance comparison for hot paths** — Read (not independently re-measured; this workstream's
   render code is unedited torch-gather/float-table work, not on Workstream G's changed render paths,
   so I judged it lower-risk than the frame identity check and spent the CPU budget there instead).
   Cross-checked `performance.json`'s numbers against its own per-repeat arrays: −8.97% / +5.18% /
   +2.79% / +0.15%, all four inside the file's own documented same-arm spread (e.g. `CraftaxClassic`'s
   float chain spans 50.6–96.4 ms on the baseline leg alone, a 1.9x spread against deltas measured in
   single-digit percent) — the numbers in the table are exactly reproduced from the raw repeats, so
   this is at least self-consistent, and I have no reason to doubt "noise, not a cost" given the render
   path itself is untouched (`state_layout` is a class attribute, `frame_shape` a property `render`
   never reads).
9. **Resource-lifetime comparison** — `Env.close()` (base, unedited) is idempotent and installs the
   render refusal; neither Craftax cell declares `_close` (their atlases are read-only per-device
   caches, confirmed via `test_the_strip_grid_cache_is_not_written_through` passing); Rocket's `_close`
   is a documented no-op for the stated shared-cache reason. My own unit run's resource-guard summary:
   `20 EGL contexts created, 0 released, 20 live` in the rocket-venv run (all from `test_nvdiff_
   triangles.py`/`test_rendernet_identity.py`, unrelated pre-existing tests, not from this
   workstream's files) and `0 EGL contexts` (mujoco not importable) in the craftax-venv run.
10. **Unexplained oracle differences: zero**, on CPU, reproduced independently by me at the merged
    SHA. GPU is owed per ruling 8.

## CPU oracle — re-run by the reviewer, at 5547270, after the G merge

This is the highest-priority check per the brief: Craftax renders through the exact 2-D tilemap/
raster2d/gfxdraw/navdot/prng paths Workstream G just consolidated, and the committed `oracle.json`
predates that merge (its own `head_sha` is `e3cde09`, i.e. E's pre-merge tip).

```
cd /Users/samib/stable-envs-step9-envs-craftax-rocket
MUJOCO_GL=glfw /Users/samib/se7-craftax-venv/bin/python scratch/step9/envs-craftax-rocket/oracle.py \
    --python /Users/samib/se7-craftax-venv/bin/python --cells CraftaxClassic,CraftaxFull \
    --out <fresh tmp dir> --name oracle_review.json
...
"head_sha": "5547270c6caff598bc44f8b40b666e6bc61d831b"
"identical": true
"unexplained_differences": {}
"functional_ok": true
  CraftaxClassic   step4_127_zero_extended      frames_7px   6a64b4fd360e765f  (64, 63, 63, 3)
  CraftaxClassic   step4_127_zero_extended      frames_16px  319792d73f89906f  (64, 144, 144, 3)
  CraftaxClassic   step7_149                    frames_7px   79aebe383560d3d2  (64, 63, 63, 3)
  CraftaxClassic   step7_149                    frames_16px  d06c468ce0eea2c8  (64, 144, 144, 3)
  CraftaxFull      step7_544                    frames_10px  82d83319a48bd3c8  (64, 130, 110, 3)
  CraftaxFull      step7_544                    frames_16px  650477921f68f055  (64, 208, 176, 3)
ORACLE: PASS
```

Every digest above is byte-for-byte the digest `findings.md`'s table claims and the digest
`corpus/states.meta.json` froze in Step 7 (`79aebe383560d3d2…`/`d06c468ce0eea2c8…` for Classic at
7/16px, `82d83319a48bd3c8…`/`650477921f68f055…` for Full at 10/16px) — I compared the frozen-meta
digits by eye against this run's output, not against the committed `oracle.json`. I then diffed my
fresh JSON against the committed `examples/results/step9/envs-craftax-rocket/oracle.json`
key-for-key in Python: every field except `head_sha` (mine `5547270`, committed `e3cde09` — expected,
the committed record predates the merge) is equal, including the full `baseline`/`migrated` sub-trees
that carry all the per-cell frame digests. **The G merge changed nothing Craftax renders through.**

The two **expected** differences, declared by the script itself and confirmed correct: (1)
`CraftaxFull.registry_row.module` moved `stable_envs.envs.craftax_full` → `stable_envs.envs.
_craftax_full`, with `old_module_paths_resolve.craftax_full_is_the_private_module: true`; (2)
`constructor_takes_num_envs` flips `False → True`, which is the Wave-0 contract working, not a defect.

Rocket, same procedure, `se7-rocket-venv`:

```
MUJOCO_GL=glfw /Users/samib/se7-rocket-venv/bin/python scratch/step9/envs-craftax-rocket/oracle.py \
    --python /Users/samib/se7-rocket-venv/bin/python --cells PFRocketLanding \
    --out <fresh tmp dir> --name oracle_rocket_review.json
...
"identical": true, "unexplained_differences": {}
  PFRocketLanding  frames OWED to the GPU oracle; mvp 59fc03c2ce6f7e9b  materials e4b648155709b5c5
ORACLE: PASS
```

`mvp` and `materials` digests match the committed `oracle_rocket.json` exactly; I diffed both JSONs
key-for-key in Python and every non-`head_sha` field is equal. No frame comparison is possible on this
host by design (`PFRocketLanding` refuses `device="cpu"`, confirmed by
`test_the_rocket_refuses_a_cpu_device_by_design` passing) — this is the correctly-declared owed item,
not a gap the workstream hid.

## Ruling checks

**Ruling 9 (both Craftax corpora kept).** Both `examples/results/step4/states/CraftaxClassic.npz`
(64×127) and `examples/results/step7/craftax-classic-full/corpus/states.npz` (64×149) are exercised in
my oracle run above, with the 127-wide corpus's direct-render `ValueError` checked rather than
silently reshaped. I did not ask for either to be dropped.

**Ruling 6 (module→module move keeps a `sys.modules` alias; every Step-8 name resolves).** Verified
live, not read off prose, in `se7-craftax-venv`:
```python
import stable_envs.envs.craftax_full as craftax_full, stable_envs.envs._craftax_full as _craftax_full
craftax_full is _craftax_full                                    # True
sys.modules['stable_envs.envs.craftax_full'] is _craftax_full    # True
craftax_full.__TESTPATCH__ = object(); getattr(_craftax_full, '__TESTPATCH__', None) is that object  # True
```
This is the real alias mechanism (`sys.modules[__name__] = _craftax_full` inside `craftax_full.py`,
executed during import so the parent package's own attribute binding picks up the new object) — one
module object, not a re-import, exactly as ruling 6 and the `swm_nav.py` precedent require. The test
`test_the_old_craftax_full_import_path_is_the_private_module` checks 10 names by hand
(`N_BLOCK, N_STRIP, STRIP_SLOTS, STRIP_CELLS, OBS_H, OBS_W, light_night_and_sleep, night_vignette,
transparent_mob_slots, _tent`); I did not find any name absent from the alias that the old module
carried.

*But the stated reason — "21 module-level names differ" — does not survive an independent count.*
I compared every `dir()` name common to both modules and their values:

| category | names | count |
|---|---|---|
| numeric constants that really do differ | `N_BLOCK`, `N_STRIP`, `N_COUNT`, `N_TILE`, `STRIP_H`, `TILE_PX_AGENT`, `OBS_H`, `OBS_W`, `REFERENCE_TILE_PX` | 9 |
| functions with genuinely different bodies (`inspect.getsource` diff, not just different `id()`) | `strip_geometry`, `night_vignette`, `_unpack` | 3 |
| per-device caches: separate dict objects, same name, would collide on a literal merge | `_VIGNETTE_CACHE`, `_CONST_CACHE`, `_GRID_CACHE` | 3 |
| **named in findings.md/the docstring as "different values" but actually IDENTICAL** | `N_FACING` (4=4), `N_KEY_HALVES` (4=4), `NIGHT_TINT` ((0,16,64)=(0,16,64)), `NIGHT_THRESHOLD` (0.5=0.5), `DAY_LENGTH` (300=300), `NO_PLAYER` (5=5), `PLAYER_ASLEEP` (4=4), `TILE_PX_IMG` (16=16) | 8 |

Real obstacles: 15 (9+3+3), not 21. The unit test `test_the_two_craftax_modules_could_not_have_been_
merged` only asserts 5 of the 9 numeric pairs plus one inequality (`OBS_H`), so the "21" figure is
asserted nowhere in code — it exists only in `findings.md` and the module docstring, both prose. The
findings table's `N_TILE | 160 | 99` row is also simply wrong: `craftax.N_TILE` is defined as
`N_COMPOSITE + N_STRIP * N_COUNT` = `816 + 16*10` = **976** at this SHA, not 160 (160 = `N_STRIP *
N_COUNT` alone, a different quantity that isn't `N_TILE` in `craftax.py` at all). None of this changes
the decision: 15 genuine, independently-verified conflicts is well past the bar for "not behaviour-
neutral," so keeping the private module is still the right call — but the arithmetic backing it up is
overstated and one entry in its own evidence table is factually wrong. See D-1.

**Ruling 2 (no cell overrides `state`; only `_hidden_state`/`_live_state` are sanctioned hooks).**
Grepped all three owned files for `def state`, `state = property`, `_hidden_state`, `_live_state`:
zero hits. Consistent with the code's own claim that these cells have no hidden dynamical state (no
`state_layout` entry has offset `None`) — there is nothing for the hooks to do, and they are correctly
left undefined rather than defined as no-ops.

**Ruling 3 (`unpack` copies; a State never aliases the env buffer).** None of the three files override
`unpack`; the base `Env.unpack` (`stable_envs/env.py:926`) does `r[:, off:off+w].to(dt, copy=True)`,
with the comment explaining why (`.to` on a same-dtype tensor would otherwise return the same
storage). `test_a_state_the_caller_keeps_is_not_a_view_of_the_env`, parametrized over both Craftax
cells, passed in my run.

## R1, R3 — confirmed real, correctly not fixed here

**R1 (`scratch/step3/certify.py::NO_BACKEND_MODULE_CLASS` misses `_craftax_full`).** Confirmed by
reading the code: `backend_class()` derives `mod = decl.get("module").rsplit(".", 1)[-1]` and looks it
up in `NO_BACKEND_MODULE_CLASS = {"craftax": "R", "craftax_full": "R", "minihack": "R", "swm_nav":
"R"}`. `CraftaxFull`'s live registry `module` is now `stable_envs.envs._craftax_full`, so `mod =
"_craftax_full"`, which is not a key in that dict — the fall-through now returns `UNKNOWN` instead of
`"R"`, exactly as claimed, and `navigation` (from Workstream A's `swm_nav.py` → `navigation.py` rename)
has the identical failure mode for the `swm_nav` entry. I confirmed this is latent, not a live
regression: `tests/unit/test_step3.py::test_backend_class_comes_from_declared_backends_not_the_env_
object` and its sibling tests all drive `backend_class()` from hand-built `CENSUS` dicts, never from
the live registry, so nothing in the current suite exercises the broken fall-through. Correctly not
fixed here (the file is `scratch/step3`, outside E's ownership; the coordinator has assigned it to
Workstream K/themselves per the handoff) — I did not ask for it to be fixed.

**R3 (three docs cite `craftax_full` by name/path).** `grep -rl craftax_full docs/` finds exactly
`docs/research/step8_upstream_findings.md` (cites the literal path
`stable_envs/envs/craftax_full.py`), `docs/research/step8_certification.md`, and `docs/research/
step7_coverage.md` (both cite the module name `craftax_full` in prose describing the old
`NO_BACKEND_MODULE_CLASS` fall-through) — plus `docs/research/step9_handoff.md`, which is the
coordinator's own tracking doc, not a stale citation. Nothing is actually broken (both import spellings
resolve), so this is correctly filed as a documentation refresh for Workstream M, not a defect that
blocks this merge.

## Unit suite — exact counts

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit -x \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
-> 1567 passed, 198 skipped, 112 deselected, 29521 warnings in 171.57s
```
(Higher than `findings.md`'s pre-merge **1419** because the merge of `step9/integration` brought in
Workstreams B and G's own new tests; `-x` means any failure would have stopped the run short — it
completed clean, so there is no regression attributable to this merge.)

```
MUJOCO_GL=glfw /Users/samib/se7-craftax-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_step8_semantic_craftax.py tests/unit/test_step8_gate_record.py \
  tests/unit/test_craftax_atlas.py tests/unit/test_craftax_full.py \
  tests/unit/test_craftax_prng.py tests/unit/test_craftax_strip.py \
  tests/unit/test_envs_craftax_rocket.py
-> 179 passed, 1 skipped in 56.00s          # matches findings.md exactly

MUJOCO_GL=glfw /Users/samib/se7-rocket-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/unit/test_rocket_landing.py tests/unit/test_envs_craftax_rocket.py \
  tests/unit/test_nvdiff_triangles.py
-> 70 passed                                 # matches findings.md exactly

MUJOCO_GL=glfw /Users/samib/se8-vm-venv/bin/python -m pytest -q -p no:cacheprovider \
  tests/swm tests/unit/test_step8_gate_record.py
-> 35 passed                                 # matches findings.md exactly
```

## Defects

**D-1 (non-blocking, documentation only).** The "21 module-level names differ" claim — repeated in
`findings.md` §1, in the `craftax_full.py` alias docstring, and used as the sole justification for not
merging the two Craftax modules — overcounts by roughly a third: my independent check found 12–15
genuine conflicts (depending on whether the 3 same-named-but-distinct caches are counted), not 21, and
8 of the 20 explicitly-named symbols in the findings/docstring list (`N_FACING`, `N_KEY_HALVES`,
`NIGHT_TINT`, `NIGHT_THRESHOLD`, `DAY_LENGTH`, `NO_PLAYER`, `PLAYER_ASLEEP`, `TILE_PX_IMG`) are
actually identical between the two modules. Separately, `findings.md`'s evidence table states
`craftax.N_TILE == 160`; the live value at this SHA is **976** (160 is a different quantity, `N_STRIP *
N_COUNT`, that happens to appear in the same arithmetic expression). The engineering decision itself
(keep `_craftax_full.py` private rather than merge into `craftax.py`) is still correct and
independently justified by the 15 conflicts I did verify — this does not change the verdict — but the
number and the one factual table entry should be corrected so a future reader doesn't inherit a wrong
count or a wrong value for `N_TILE`. Recommend the author (or whoever picks this up next) fix
`findings.md` §1's table and the `craftax_full.py` docstring, and either broaden
`test_the_two_craftax_modules_could_not_have_been_merged` to assert the real, complete list or drop
the specific number "21" from the prose in favor of "the shared names include several that hold
different values, which is already disqualifying."

No other defects found. Everything else claimed — the oracle identity (re-run by me from scratch,
post-G-merge), the alias mechanism, the rulings, the owed-GPU-items accounting, R1 and R3 — checked
out exactly as stated.

## GPU evidence owed and the missing sbatch

Added at the coordinator's request, after the verdict above was already reached on CPU evidence.
Confirmed by search: `find . -iname "*.sbatch"` under `scratch/step9/` turns up
`render-3d/{gpu_oracle,oracle_selfcompare,hmg_determinism}.sbatch`, `envs-classic-nav/gpu.sbatch`,
`render-2d/gpu_oracle.sbatch`, `envs-dmcontrol/gpu.sbatch` — and nothing under
`scratch/step9/envs-craftax-rocket/`, on this branch or anywhere in `git log --all`. The workstream's
own findings (§10) say Rocket's pixels are "owed to the post-Wave-1 GPU oracle" but the workstream
supplied no instrument to collect them. This is a real gap and it is the coordinator's to fill, as
stated; it does not change my CPU verdict (the CPU-determined half of every frame claim is byte-
identical, independently re-verified above), but the gap must close before Rocket can be said to have
been checked at all.

**1. What owes CUDA evidence, and why.**

- **`PFRocketLanding` — the entire pixel claim.** Confirmed by direct construction attempt and by
  reading the code: it raises `NotImplementedError` on `device="cpu"` (`rocket_landing.py:739-742`,
  a design choice, not a missing feature) and cannot even build a renderer without `nvdiffrast`
  installed. Nothing about its frame has been rendered anywhere in this workstream's evidence,
  including by me — I could only reproduce the CPU-determined inputs to the frame (host-side
  kinematics chain, material table, scene arrays), which is genuinely CUDA-only work: nvdiffrast is a
  CUDA rasterizer with no CPU fallback in this repo.
- **`CraftaxClassic`/`CraftaxFull` do NOT owe a new frame-identity proof, and the CPU story here is not
  the same as for `PFRocketLanding`.** Two separate things are CUDA-adjacent for Craftax and only one
  of them is stable-envs' concern: (a) the **JAX simulator** that PRODUCES the states in the frozen
  corpora — that is `craftax`/`stable_worldmodel`'s own `jax.jit`-ed step, entirely outside this
  package's render code, already exercised and frozen at Step 7, and untouched by this migration
  (`test_these_cells_have_no_dynamics_and_say_so_precisely` pins that stable-envs never advances a
  Craftax simulation at all — it renders states it is handed); (b) the **torch renderer** that turns a
  packed row into pixels, which is what stable-envs owns and what this workstream migrated. I grepped
  both `craftax.py` and `_craftax_full.py` for `triton`/`HAVE_TRITON`/any CUDA-only kernel: there is
  none — the render path is `torch.index_select` over a uint8 atlas plus a shared float post-process,
  the same ops on `cpu` and `cuda`, unlike `gfxdraw_classic`/`navdot`'s fused Triton kernels which
  genuinely only run on CUDA. So the three frozen corpora at both certified sizes, byte-identical on
  CPU against both the baseline tree and the Step-7 frozen meta digests (re-verified by me above),
  **do genuinely discharge Craftax's frame-identity claim in substance** — there is no second code
  path for a GPU run to exercise. What is still nominally owed is O2, findings' own term for it: a
  `--device cuda` confirmatory run of the same oracle, cheap to add to the same job, worth doing for
  completeness (it also picks up device-placement bugs the CPU arm cannot, e.g. a `.to(device)` that
  silently no-ops), but its absence is a much smaller gap than Rocket's and does not block on its own.

**2. Venv arms required.** `PFRocketLanding`'s frame comparison needs `$SCR/venvs/se7-rocket`
(PyFlyt + nvdiffrast, per the handoff's Oscar venv table) on both arms of the job. Craftax's O2
confirmation needs a CUDA-capable torch in whatever venv runs it; the handoff names
`$SCR/venvs/se7-craftax-gpu` (craftax + jax cuda) for this family, but I have not independently
confirmed that venv's torch build is CUDA-enabled (I have no GPU access on this host, per
instructions) — the sbatch's provenance block (matching `gpu.sbatch`/`gpu_oracle.sbatch`'s own
`python -c "import torch; print(...)"` line) should assert `torch.cuda.is_available()` before trusting
either arm's result, the same way those two scripts print mujoco/torch versions up front.

**3. What a correct section must assert to be worth running.** Modeled on
`scratch/step9/envs-dmcontrol/gpu.sbatch` and `scratch/step9/render-3d/gpu_oracle.sbatch`: both arms
(a detached `git worktree add --detach 21cb306` and the branch under test) run in the SAME sbatch, on
the SAME allocated card, so the comparison is never cross-card (the render-2d review's Judgement
section makes the same point about the fused-kernel oracle; it applies identically here since
nvdiffrast is itself a compiled CUDA path). Concretely, for Rocket the section must:
  - run `scratch/step9/envs-craftax-rocket/oracle.py --python <se7-rocket venv> --cells
    PFRocketLanding --device cuda` against the frozen 200-state Step-7 reference corpus and assert
    `identical: true`, `unexplained_differences: {}` — this is the FIRST time an actual `frames_sha256`
    exists for this cell; everything committed so far is the CPU-only host-side chain;
  - run `tests/gpu -k "PFRocketLanding or rocket"` (Gate A, `test_conformance.py`,
    `test_reproducibility.py`, `test_fast_paths.py` all parametrize over the live registry and will
    include `PFRocketLanding` once it registers on a host with nvdiffrast — confirmed by reading
    `tests/gpu/test_reproducibility.py`, the one file that names `PFRocketLanding`/`CraftaxFull`/
    `CraftaxClassic` explicitly; the others pick the cell up generically);
  - run `perf.py`'s paired-arm method for a GPU timing number (O3) — Rocket currently has no CPU number
    at all to compare against, so this is a first measurement, not a before/after;
  - re-run `test_foundation_api.py::test_the_number_of_classes_that_still_reject_num_envs_only_goes_
    down` on this host and assert the count is one lower than the CPU count recorded in `findings.md`
    §2 (O5) — `PFRocketLanding` only registers where nvdiffrast is importable, so this ratchet cannot
    move anywhere else.
  For Craftax (O2), the section only needs the same `oracle.py --device cuda` invocation over the
  three frozen corpora, asserting `identical: true` against both the baseline arm on the same card and
  the frozen Step-7 meta digests — a confirmation, not a new comparison design.

**4. The Rocket/nvdiffrast hazard with Workstream I, and how to attribute a difference.** Both
workstreams touch the same object from opposite sides and neither has rendered a frame yet. At this
SHA, `rocket_landing.py`'s cache path is: `_nvd()` (line 760) looks up `_NVD_CACHE` keyed by
`(size, supersample, device)`, clears the whole cache when it hits 8 entries (line 768,
`_NVD_CACHE.clear()`, no `close()` call on the evicted renderers), and constructs a fresh
`ND.BatchRenderer(...)` (line 778) on a miss; `_close()` (line 747) is a documented no-op that leaves
the shared cache alone, for the use-after-free reason given in §2 above. Workstream I's plan (read
from `docs/research/step9_handoff.md`'s row for I) is to make `nvdiff.BatchRenderer`/`nvdiff.Scene`
`Closeable` and to edit these same lines — `rocket_landing.py:729` → call `br.close()` on cache
clear, `rocket_landing.py:739` → set `owns_scene = True` — after E merges (the line numbers in I's
plan don't match E's current 747/760/768/778 because I's plan was written against a not-yet-existing
post-merge state; the coordinator will need to re-resolve them against whatever E's file looks like
once both are on `step9/integration` together). So a GPU-observed Rocket frame difference could come
from either workstream's edit to the identical few lines, and neither workstream's own evidence rules
it out — E never rendered a frame at all, and I's own findings say the same ("NO frame rendered...the
oracle is UNESTABLISHED, not satisfied").

**Recommend a two-checkpoint ordering, not a single post-everything run:**
  1. **Checkpoint E.** Run the Rocket GPU oracle (item 3 above) at the Wave-1 tip with E merged and
     **before** I merges. This is the first real Rocket frame, produced by E's code exactly as I
     reviewed it (the no-op `_close`, the clear-without-close cache eviction), compared same-card
     against the `21cb306` baseline arm. Whatever this run says — pass or a specific, named
     difference — is attributable to E alone, because `nvdiff.BatchRenderer`/`nvdiff.Scene` are still
     the pre-I classes at this checkpoint.
  2. **Checkpoint I.** After I merges (bringing both its `Closeable` refactor of `nvdiff.*` AND its own
     edit of `rocket_landing.py:729/739`), re-run the identical oracle comparison, same card, ideally
     in the same job as a second section so the GPU/driver/card are held fixed. Diff checkpoint I's
     `frames_sha256` against checkpoint E's. If they match, the resource-lifetime refactor is frame-
     neutral and the earlier checkpoint already cleared Rocket. If they differ, the delta is I's, not
     E's — checkpoint E already fixed what E's own code produces, and the only thing that changed
     between the two checkpoints is I's edit.
  This also gives the coordinator a natural place to run O2 (Craftax's confirmatory cuda arm) and O4
  (the GPU test suites) at checkpoint E, since neither depends on I's resource-lifetime work at all.

This gap does not block **this** review's verdict — the CPU-determined evidence for all three cells is
complete, independently reproduced, and REGRESSION = 0 on everything a CPU can measure. It does mean
Rocket's pixel claim remains genuinely unestablished, not merely "owed" in a pro-forma sense, until the
coordinator's sbatch runs at least checkpoint E above; I would treat that as a precondition for the
final §20 comparison and closure packet, not for this merge.

Nothing above changes the verdict: **APPROVE**.
