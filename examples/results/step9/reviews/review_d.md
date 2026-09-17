APPROVE

# Step 9 — independent review of Workstream D (OGBench)

**Reviewed** `step9/envs-ogbench` at **`e2a6aed5362100a349f8a8833e5b70aa839fbab8`** (the author's delivery
`7f90a06` plus the coordinator's merge of `step9/integration` `312a631`, which brings in Workstream B
(DMControl as a package) and Workstream G (2-D render backend consolidation: gfxdraw, raster2d, scene,
navdot, tilemap, prng)). Delta reviewed: `git diff 312a631...e2a6aed` (10 files, +15,306/−1,588) and
`git log --stat step9/integration..HEAD`. Step-8 behavioural baseline
`21cb30678b7a7d0f3344b59f176f33d3f3b09ffc`.

**Reviewer host** Apple M5 Mac `cis1b1zs2j3`, local CPU only. `MUJOCO_GL=glfw`, `-p no:cacheprovider`.
Venvs `/Users/samib/se7-ogb-venv` (mujoco 3.12.0, ogbench, stable_worldmodel, gymnasium 1.3.0,
tensordict 0.14.2 — the interpreter ogbench itself is installed in) and `/Users/samib/se6-venv` (the
CPU-only unit-suite interpreter). Nothing was edited, committed, stashed, reset, rebased or pushed; no
GPU job, no Oscar, no ssh. The oracle and plate-probe harnesses create and remove their own detached
`git worktree`s of `21cb306`; I created two more myself (for independent AST-name enumeration and a
pre-D holdout count) and removed both with `git worktree remove --force` when done.

## Verdict

**APPROVE.** I re-ran the CPU oracle from scratch into a temp directory (not the committed
`oracle.json`) and it is identical to the committed record in 8,499 of 8,500 flattened leaves — the
one difference is `head_sha`, which is expected (the record was written one commit before the SHA it
measures, and my re-run is at the post-merge tip). Zero unexplained differences, zero regressions, on
25 maze cells over 11 frozen scene corpora, after the merge that added B and G. I independently
enumerated the Step-8 module-level names against a fresh `21cb306` worktree (not the author's list)
and got 17 + 41 = 58, all 58 resolving off the migrated tree; confirmed the `ogb_manip` alias is a
real `sys.modules` identity (not a re-import) by patching an attribute through one path and reading it
through the other; confirmed `unpack` copies (different tensor storage; mutating the packed row after
`unpack` does not change the returned State); confirmed no OGBench cell overrides `state`, `_layout`,
`_hidden_state` or `_live_state`; independently re-ran `plate_probe.py` against both the head tree and
a fresh baseline worktree and got the same digests the author reports; and confirmed the num_envs
ratchet's *live* count is 14 post-D (verified against a fresh 41-holdout count on `step9/integration`
pre-D, with the 27 D fixed enumerating exactly to Antmaze×11 + Humanoidmaze×6 + OGBCube/OGBScene×2 +
Pointmaze×8 = 27, and the 14 remaining splitting exactly into highway 6 / maze2d 4 /
mujoco_locomotion 4, matching the coordinator's independent count). One non-blocking defect below: the
ratchet's enforced ceiling constant (`NUM_ENVS_HOLDOUTS`) was not edited by this workstream, so the
"41 → 14" in the findings/handoff describes the measured count, not a tightened bound — worth a
coordinator note, not a merge blocker. Full unit suite at `e2a6aed`: **1575 passed, 215 skipped, 112
deselected, 0 failed.**

## The ten §17 items (my own numbers)

1. **Files moved/removed/created.** `git diff --stat 312a631...e2a6aed`: `stable_envs/envs/ogbench.py`
   → `stable_envs/envs/ogbench/maze.py` (via rename, +78/−0 net diff noise from the package move),
   `stable_envs/envs/ogb_manip.py` (1,582 → 23 lines, rewritten as a `sys.modules` alias),
   `stable_envs/envs/ogbench/manipulation.py` (new, 1,667 lines), `stable_envs/envs/ogbench/__init__.py`
   (new, 79 lines), `tests/unit/test_envs_ogbench.py` (new, 393 lines), plus
   `scratch/step9/envs-ogbench/{oracle.py,plate_probe.py}` and
   `examples/results/step9/envs-ogbench/{oracle.json,plate_probe.json,findings.md}`. No file outside
   this list touched (confirmed by the diff stat — no `stable_envs/render/*`, no `stable_envs/env.py`,
   no other family's `envs/*`, no `_plate_texels.py` edit).
2. **API changes** — additive, verified live: `num_envs=`/`batch=` both work
   (`PointmazeLargeNavigate(num_envs=64)` and `.../batch=64` both construct); `env.close()` +
   `with snv.PointmazeLargeNavigate(...) as env:`; `env.state` returns a `TensorDict`; `pack`/`unpack`
   round-trip. `env.step(...)`/`env.transition(...)` raise `NotImplementedError` naming the cell and
   the reason (message reproduced below); `render()` after `close()` raises `RuntimeError` naming the
   cell. No signature narrowed, nothing removed that a Step-8 caller used.
3. **Step-8 IDs affected** — behaviourally none of the three SWM ids this family serves
   (`swm/OGBMaze-v0`, `swm/OGBCube-v0`, `swm/OGBScene-v0`); all declarations unchanged, confirmed by
   my own oracle re-run (registry row compared in full, `module` included, identical).
4. **Targeted tests** — `tests/unit/test_envs_ogbench.py` collects **59** tests. Full suite at
   `e2a6aed`, reproduced by me: **1575 passed, 215 skipped, 112 deselected, 0 failed**, 121.4 s
   (this number is larger than the author's own reported 1427 because that count was taken on the
   author's branch *before* the coordinator's merge of B and G into integration; the merge itself adds
   tests unrelated to this workstream. No OGBench test regressed either count.)
5. **Applicable certification checks** — the Step-4 frozen-corpus replay for all 25 maze cells and the
   plate-bake probe for both manipulation cells, both re-run by me from scratch (below). Gate A / GPU
   conformance / `check_visual_fidelity` need CUDA and are correctly recorded as owed (§10 of the
   findings), consistent with ruling 8 (one post-Wave-1 GPU oracle, not a per-workstream job).
6. **Default-frame comparison** — re-run by me into a temp directory (command and diff below): **25
   maze cells, 11 distinct scene digests, identical to the committed record and to my own independent
   digest table**; `unexplained_differences: {}`, `identical: true`.
7. **Variation/state comparison** — `describe()`, style layout, default appearance, the variation
   taxonomy leaf for leaf with init values, `state_dim`/`state_names`, the packed round trip,
   `sample_states`, all fidelity declarations: identical in my re-run (same 8,499-leaf diff as §6).
8. **Performance for hot paths** — none claimed and none needed: `git diff` shows the render/bake/
   composite functions untouched byte for byte except the plate-bake path, which runs once per
   cache miss, not per render (verified by reading `_apply_floor`/`substitute` — cache-keyed).
9. **Resource-lifetime comparison** — module caches (`_ENV_CACHE`, `_SCENE_CACHE`, `_CTX_CACHE`,
   `_PLATE_CACHE`, `_PLATE_RIGS`, `_BUNDLE_CACHE`, `_REF_CACHE`, `_KEEPALIVE`) are the same objects
   before and after the package split (`test_the_module_caches_are_one_object_each_across_the_package`
   passed in my run); nothing closed a resource that Step 8 kept open. `close()` is explicit,
   idempotent, and a closed env refuses `render` (verified live, above).
10. **Unexplained oracle differences: zero**, reproduced independently by me, after the merge that
    brought in B and G — this is the item ruling 8 asked me to re-establish, since the committed
    record predates that merge.

## Re-running the CPU oracle myself (not reading the committed record)

```
MUJOCO_GL=glfw /Users/samib/se7-ogb-venv/bin/python scratch/step9/envs-ogbench/oracle.py \
  --out <scratchpad>/d_review_oracle --name oracle_review.json
...
 "identical": true,
 "unexplained_differences": {},
 "step8_module_names_checked": {"maze": 17, "manip": 41},
 "step8_module_names_missing": {"maze": [], "manip": []},
 "owed_to_the_gpu_oracle": {"OGBCube": "...", "OGBScene": "..."},
 "maze_is_package": true,
 "cache_objects_are_shared": true,
 "functional_ok": true
ORACLE: PASS
```

Per-scene digests reproduced hex for hex (pointmaze-medium `22185b713581b94a`, pointmaze-large
`b35a6631e7d20247`, pointmaze-giant `577646f1835f113f`, pointmaze-teleport `3cde1e67c10fb539`,
ant-medium `3234e84199fd6e7b`, ant-large `d648171bd3bf0bb3`, ant-giant `368f7277bf4b3adc`,
ant-teleport `da777169b3c52a2f`, humanoid-medium `be9993beeca4641d`, humanoid-large
`b4dd4403238962d7`, humanoid-giant `55534ade7b9bd328` — all 11, matching the committed table exactly).

**Leaf-by-leaf diff of my JSON against the committed `examples/results/step9/envs-ogbench/oracle.json`**
(flattened both records, 8,499 keys each, compared value for value):

```
total differing leaves: 1
  head_sha  MINE=e2a6aed5362100a349f8a8833e5b70aa839fbab8  COMMITTED=f00ddcf
compared leaves total: 8499 8499
```

**Classification: no difference to classify.** REGRESSION = 0. The single differing leaf is exactly
the record's own `head_sha`, and the author's stated convention (the commit carrying a record cannot
be the commit it measures) already explains it — my re-run at the actual post-merge tip settles that
the merge changed nothing else. This directly answers the review's central question: the maze cells,
which render through `render/raycast3d.py` and MuJoCo's own plate bake and do **not** touch any of
Workstream G's owned files (`raster2d`, `gfxdraw`, `scene`, `navdot`, `tilemap`, `prng` — confirmed by
`grep -rl` finding no import of those modules in `stable_envs/envs/ogbench/`), are unaffected by G's
merge, and the oracle proves it rather than assuming it.

## Ruling 6 — the 58/58 claim, checked against my own enumeration of `21cb306`, not the author's list

I built a fresh detached worktree of `21cb306` myself and ran my own AST walk (FunctionDef/ClassDef/
Assign/AnnAssign at module level, same rule the author's `oracle.py:389-397` uses — I did not read
their list, I wrote my own extractor and pointed it at the baseline files):

```
ogbench.py module-level names (mine): 17   ['BODY_STATE','CELLS','MAT_SPECULAR','OGBenchMaze',
  'SHADE_AMBIENT','SHADE_DIFFUSE','SHADE_EXPONENT','SHADE_SPECULAR','_ANT_JOINTS','_ENV_CACHE',
  '_FREE','_HUMANOID_JOINTS','_NOMINAL','_SCENE_CACHE','_SCENE_CACHE_MAX','_Scene','_build_reference',
  '_make_classes','_mat_and_tex','_offsets','_scene_env']  # 21 raw incl. imports; 17 excl. imports
ogb_manip.py module-level names (mine): 41  # 52 raw incl. imports; 41 excl. imports
```

(A raw AST walk including `Import`/`ImportFrom` gives 28/52; subtracting the 11/11 pure re-exported
imports each — `Box`, `Dict`, `np`, `os`, `torch`, `annotations`, `register`, `StableEnv`, etc. — lands
on 17/41, matching the author's counted total exactly, and matching my own independent count using
only the same function/class/assignment rule the oracle applies.)

Checked every one of the 58 against the live migrated tree:

```python
import stable_envs.envs.ogbench as pkg
import stable_envs.envs.ogb_manip as manip_alias
missing_maze  = [n for n in maze_names  if not hasattr(pkg, n)]          # -> []
missing_manip = [n for n in manip_names if not hasattr(manip_alias, n)]  # -> []
```

**58/58 confirmed, by my own enumeration and my own resolution check, not the author's.**

**The alias is a real `sys.modules` identity, checked behaviourally, not just by `is`:**

```python
import stable_envs.envs.ogb_manip as A
import stable_envs.envs.ogbench.manipulation as B
A._SOME_TEST_ATTR = "patched-via-A"
getattr(B, "_SOME_TEST_ATTR", "MISSING")   # -> "patched-via-A"
sys.modules["stable_envs.envs.ogb_manip"] is sys.modules["stable_envs.envs.ogbench.manipulation"]  # -> True
```

An attribute set through one import path is visible through the other, which is what `sys.modules[__name__]
= _manipulation` (`stable_envs/envs/ogb_manip.py:23`) buys and a set of re-exports would not — this is
the exact property `tests/unit/test_ogb_manip.py`'s monkeypatches depend on, and I verified it directly
rather than trusting the docstring's claim.

## Ruling 5 — the registry-row test, read myself

`tests/unit/test_foundation_registry.py::test_every_canonical_class_name_resolves_to_the_same_class_as_at_the_baseline`
(foundation-owned, not D's, but what D's cells are graded by) does exactly what the ruling requires:
semantic columns are compared with `{k: v for k, v in now[name].items() if k != "module"} == {...}`,
and `module` is compared separately — only if it *changed* does the test fall back to importing the
old path and checking `getattr(old, name, None) is R.REGISTRY[name]` (lines 84-93). For this
workstream that fallback branch is never exercised: I confirmed live that `OGBCube.__module__ ==
"stable_envs.envs.ogb_manip"` and `PointmazeLargeNavigate.__module__ == "stable_envs.envs.ogbench"`,
both byte-identical to the recorded values in `examples/results/step9/foundation/registry_baseline.json`
(checked for all four sample classes). So D's own claim — "this workstream does not hit the
reachability fallback, `module` is compared like every other column and is identical" — is accurate;
D achieves the *stronger* bar (no move at all) rather than relying on the reachability concession.

## Ruling 2 — no cell overrides `state`

```
grep -n "def state\b|def _layout\b|def _hidden_state|def _live_state" stable_envs/envs/ogbench/{maze,manipulation}.py
```
returns nothing. `pack`/`unpack`/`state`/`_layout` are all inherited from `Env`; `state_layout` is
assigned in `__init__` (the R1 instance-method pattern), never overridden as a method.

## Ruling 3 — `unpack` copies, checked with a live storage-pointer test

```python
env = snv.PointmazeLargeNavigate(num_envs=4, device="cpu")
rows = env.pack(env.reset(seed=0))
st = env.unpack(rows)
st["root","position"].data_ptr() == rows.data_ptr()   # -> False (different storage)
rows.fill_(999.0)
st["root","position"][0,0]                             # -> unchanged (3.08...), not 999.0
```
`env.py:934-939`'s `.to(dt, copy=True)` is what makes this true; it is foundation code, not D's, but D
does not override `unpack` anywhere, so the guarantee reaches these 27 cells intact. Confirmed live,
not by reading the comment.

## The plate-texel dedup and the 38150c3 upload-order fix

`mj_quantise = _quantise` where `_quantise` is `from .._plate_texels import quantise as _quantise`
(`manipulation.py:113,340`) — object identity by construction, not by convention.
`renderer._gl_context.make_current()` precedes `mujoco.mjr_uploadTexture(...)` in
`_plate_texels.substitute` (`stable_envs/envs/_plate_texels.py:130-131`), preserving the Step-8
`38150c3` ordering fix.

I re-ran `plate_probe.py` myself, once against the head tree and once against a **fresh** baseline
worktree I built (copying the unmodified probe script in, since the baseline never had one):

```
head:     Cube plate_default=... plate_custom differs (55.47 -> 140.10 area), restored after custom: True
baseline: Cube identical digests/means on every pre-existing key
Scene:    identical on every pre-existing key, both trees
diffs (excl. the two NEW keys texel_slots adds): {}
```
The only keys that differ between trees are `grid_texels`/`grid_covered`, which the shared
`texel_slots` helper adds and the old duplicated code never reported (`None`/`-1` at baseline vs.
`262144`/`261121` at head) — exactly what the findings describe as the measurement's byproduct, not a
behavioural change. Every plate digest, cache key and restore-after-custom check is identical.

## The ratchet — live count verified independently; one gap found

I built a second temporary worktree at `step9/integration`'s pre-D tip (`b3f3a81`) and ran the same
holdout query the foundation test uses:

```
pre-D (b3f3a81):  41 holdouts — the 27 OGBench cells (11 Antmaze + 6 Humanoidmaze + 2 OGBCube/Scene +
                   8 Pointmaze) plus the same 14 non-OGBench cells listed below
post-D (e2a6aed): 14 holdouts — Highway, Intersection, Merge, MujocoInvertedDoublePendulum,
                   MujocoInvertedPendulum, MujocoPusher, MujocoReacher, Parking, PointMazeLarge,
                   PointMazeMedium, PointMazeOpen, PointMazeUMaze, Racetrack, Roundabout
```
This splits exactly into highway 6 (Highway, Intersection, Merge, Parking, Racetrack, Roundabout) /
maze2d 4 (PointMazeLarge/Medium/Open/UMaze) / mujoco_locomotion 4 (the four `Mujoco*` classes) — F's
14, matching the coordinator's own independent count in the brief. **D's 27 are genuinely and
completely fixed**; none of the 14 remaining is an OGBench cell, and none of D's 27 reappears.

**Gap found (non-blocking):** `tests/unit/test_foundation_api.py`'s `NUM_ENVS_HOLDOUTS = 41` is
**unedited** by this workstream — `git diff 312a631...e2a6aed` does not touch this file at all, and
the constant reads `41` at `e2a6aed`, not `14`. The assertion is `len(holdouts) <= NUM_ENVS_HOLDOUTS`
(14 ≤ 41), so the test passes, but the *ceiling* the docstring calls a one-way ratchet ("pinned here
so it can only go DOWN") was not tightened — a future regression that pushed the live count back up
to, say, 30 would still pass silently. The brief's framing ("F is also lowering this same constant
right now, so the line-level conflict is mine to resolve") assumes both workstreams touched the same
line; in fact **D never touched it**, so there is no line-level conflict from D's side — only F's edit
(if it makes one) will apply. This is a defect in the finding's phrasing ("ratchet lowered 41 → 14"
describes the *measured* count, not the *enforced* bound) rather than in behaviour: I recommend the
coordinator set `NUM_ENVS_HOLDOUTS = 14` centrally when resolving Wave-1, in the same pass that
resolves F's edit, since `tests/unit/test_foundation_api.py` is a shared foundation file no single
Wave-1 workstream should be editing unilaterally anyway.

## R2 / R3 — checked, both real and correctly deferred

**R2** (`ogb_manip` alias + `manipulation._MODULE` + `scratch/step4/universe.py::shard_of` must move
together in Wave 3): confirmed. `manipulation._MODULE = "stable_envs.envs.ogb_manip"` pins
`__module__` on the generated classes (`manipulation.py:1629,1637`); `shard_of`
(`scratch/step4/universe.py:148-152`) maps `"ogbench" -> "ogb"` with no `"ogb_manip"` entry, so
`shard_of` on these two cells' configurations falls through to the default `"raster"` shard **today,
already, at Step 8 and unchanged by this patch** — exactly as R2 states. Both files are outside this
workstream's ownership (`stable_envs/envs/ogbench/` only), so deferring rather than editing them now
is correct: editing `_MODULE` now would break the registry-row byte-identity this workstream's whole
oracle rests on.

**R3** (`docs/api/envs.md:26` `::: stable_envs.envs.ogbench` now only documents the package
`__init__`'s re-exports): confirmed — line 26 is exactly that directive, and `docs/` is not this
workstream's file. Correctly deferred to M.

## Unit suite (reproduced by the reviewer, exact command from the brief)

```
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q -p no:cacheprovider tests/unit -x \
  --deselect tests/unit/test_examples.py -k "not test_every_cited_scratch_path_resolves"
resources: 20 EGL contexts created, 0 released, 20 live (peak 20); renderers 19 created / 0 closed
1575 passed, 215 skipped, 112 deselected, 29521 warnings in 121.41s (0:02:01)
```
Zero failures. (The count is higher than the author's own reported 1427 because that number was taken
on the author's branch before the coordinator's merge of B and G; nothing about this discrepancy is
this workstream's defect — it is exactly what merging two more Wave-1 workstreams into the same tree
does to the total.)

## GPU evidence owed and the missing sbatch

D committed no `scratch/step9/envs-ogbench/gpu*.sbatch`, unlike A, B, G, H and I, which each committed
a GPU section for the single post-Wave-1 oracle. This is a genuine gap in what D delivered (the
findings' §10 "owed" table describes what is owed in prose but the workstream leaves no runnable
instrument for it, unlike every sibling workstream), and I was asked to scope it precisely rather than
write it.

**1. Exactly which cells owe CUDA evidence, and why.**

* **`OGBCube`, `OGBScene` — genuinely CUDA-only, zero evidence on any device from any Step-9 job so
  far.** `device="cpu"` raises `ValueError: ... has no CPU backend (CPU3D-OGB-1)` unconditionally
  (`manipulation.py:661,1660`) — there is no honest way to render them on a CPU host at all, which is
  why D's own oracle records them under `owed_to_the_gpu_oracle` rather than skipping silently. I
  checked whether Workstream H's already-completed GPU job (`scratch/step9/render-3d/gpu_oracle.sbatch`,
  jobs 6464319/6464500/6464601, venv `$SCR/venvs/se-step6`) picked these two up as a side effect: it
  did not — `examples/results/step9/render-3d/findings.md:193-198` records `stable_worldmodel` as
  missing from `se-step6`, so `"swm/OGBCube-v0" and "swm/OGBScene-v0" skipped on cuda ... these two
  IDs are not covered at all"`. **These two cells have no GPU (or any-device) measurement anywhere in
  Step 9 as of this review**, on top of owing the CUDA conformance gate (`check_visual_fidelity`),
  which needs frames to grade against the declared 4.3 / 8.7 mean-abs ceilings.
* **The 25 maze cells also have a CUDA-only code path that D's own oracle never exercised** — the
  committed and my re-run records both have `"compare_device": "cpu"` throughout; there is no cuda
  leaf for any maze cell in `oracle.json`. These cells are not CUDA-*only* (they render on CPU today,
  proven by both my re-run and D's), but `raycast3d`'s CUDA/Triton path is a genuinely different code
  path from its CPU path and D's ticket only exercised the latter — findings §10 says this in prose
  ("the maze CUDA arm ... is unproved by this ticket") but nothing in `oracle.json`'s structure tracks
  it as owed the way `OWED_ON_CPU` tracks the manipulation pair, so a reader of the JSON alone would
  not discover the gap.
  * This part is **not starting from zero**: H's completed job already ran
    `scratch/step9/render-3d/oracle.py --devices cpu,cuda` over a 38-cell selection that included at
    least 12 of these 25 maze IDs on cuda (HumanoidmazeGiantNavigate plus "3 other humanoidmazes, 4
    antmazes, 4 pointmazes" per `findings.md:184-198`) and found all but one byte-identical to
    baseline (see the hazard below). I confirmed `git diff 21cb306 -- stable_envs/envs/ogbench.py
    stable_envs/envs/ogbench/maze.py` touches **only** `__init__` signature/`STATE_LAYOUT`/`__module__`
    pinning — no line inside `render`, `_fk`, `_bake`, `_scene`, `_mat_and_tex` or any camera/geometry
    expression changed — so H's job is strong supporting evidence the raycast3d cuda path is
    unaffected by D's patch specifically. It is not a substitute for a run on the post-D **merged**
    tree, though (ruling 8's own reasoning: per-workstream jobs don't compose, only the one
    post-Wave-1 oracle on the fully merged tip counts as the authoritative record), and it does not
    cover the 13 maze IDs outside that 12, though those are all siblings of a covered scene and share
    its frame digest by construction (`test_the_dataset_siblings_of_one_scene_draw_the_same_frame`).

**2. Venv.** `$SCR/venvs/se7-ogb` (ogbench + `mujoco_warp`, per the handoff's Oscar venv table) is the
only viable one. H's `se-step6` cannot cover `OGBCube`/`OGBScene` at all (missing
`stable_worldmodel`); D's own local venv, `/Users/samib/se7-ogb-venv`, is this same arm's Mac twin and
is CPU-only (no CUDA device on this host), which is exactly why D's own evidence stops at the CPU
arm. `tensordict`/`loguru` are already confirmed present in that venv (R4, verified as part of this
review's oracle re-run).

**3. What a correct section must assert.** `scratch/step9/envs-ogbench/oracle.py` already supports
this with **no script edit** — `--device cuda` runs both the baseline-worktree arm and the head arm as
two subprocesses of the *same* `main()` invocation, under the *same* interpreter, which is the
same-job/same-card form `gpu.sbatch` (B) and `gpu_oracle.sbatch` (H) use; there is no valid
cross-card alternative (a cross-card digest comparison is not a valid oracle, per the brief). A correct
section, modelled on those two:

```
$PY scratch/step9/envs-ogbench/oracle.py --python "$PY" --device cuda --head-sha "$SHA" \
    --name oracle_head_cuda.json --out "$OUT"
```
run once with the full 27-cell default `--cells` (covers the maze CUDA arm AND gets
`OGBCube`/`OGBScene` their first-ever frames+declarations+registry-row comparison in one pass — the
script already handles a CUDA-refusing... here CUDA-*requiring* cell exactly as it handles the CPU
refusal, by recording `construct_refused` if a device genuinely can't serve a cell, so no `--cells`
split is needed). This alone gives: 25 maze frame digests on cuda vs. baseline (same job, same card),
and `OGBCube`/`OGBScene` frames, `describe()`, registry row and CPU-refusal-message-on-the-other-device
sanity, all compared byte for byte in one process pair. Beyond that, the section must add what the
oracle script does **not** check:
* **`check_visual_fidelity`** (or the equivalent Gate/conformance harness) for `OGBCube`/`OGBScene`
  against their declared ceilings — this is an absolute quality bar, not a same-tree diff, and is the
  piece findings §5/§10 name as separately owed ("it grades the declaration this patch preserves, and
  it needs frames").
* a `tests/gpu` selection scoped to this family, e.g. `-k "OGB or Antmaze or Humanoidmaze or Pointmaze"`
  (mirroring B's `-k "DMC or dmc-"`), with the exit-5/no-tests-selected check every other section uses
  and the full output redirected to a file with only the summary echoed (the EGL-teardown-noise
  lesson `gpu_oracle.sbatch`'s own comment records).
* a provenance block (host, job id, `git rev-parse HEAD` of `$SCR/se9/envs-ogbench`, GPU name/driver,
  python/mujoco/ogbench/mujoco_warp versions) and a loud `FATAL` exit if `mujoco_warp` or
  `stable_worldmodel` is missing from the venv, exactly as B's script guards `mujoco_warp` — this is
  precisely the gap that silently cost H's job its OGBCube/OGBScene coverage, and it must not recur
  here where those two cells are the entire point of the run.
* the tree submitted must be **the merged `e2a6aed` tip** (or later), not the author's pre-merge
  `7f90a06` — the whole reason this review re-ran the CPU oracle was that the committed record predates
  the B+G merge, and the same reasoning applies to any GPU record.

**4. The `HumanoidmazeGiantNavigate` hazard.** That cell is one of D's 25 (the Humanoidmaze-Giant
scene, shared with `HumanoidmazeGiantStitch`; digest `55534ade7b9bd328` in both the committed record
and my re-run). H already characterized its cuda instability precisely
(`examples/results/step9/render-3d/findings.md:370-388`): four observations across three jobs and two
cards — job 6464319 baseline `929f7c09…` vs head `6c5385e4…` on an RTX A6000 (the one difference in
that job); job 6464500, baseline ×3 and head ×3, **all** `6c5385e4…`, on an A40; job 6464601, a
same-card 3-leg self-compare (baseline-vs-baseline, head-vs-head, baseline-vs-head) all identical on
the original RTX A6000. Classification: **DOCUMENTED_LIMITATION** — the baseline tree itself is not
process-reproducible on this specific cell on cuda (`_BOUND_SLACK_NOISE`: a ~100 m camera and a 5 cm
capsule put `disc > 0` on float32 rounding noise), and the head tree has produced exactly one digest on
every run and every card so far. **D's refactor cannot plausibly be the cause of a recurrence**: the
diff to `maze.py` (confirmed above) touches no camera, capsule-bounds or rounding expression — only
`__init__`/state-layout/module-pinning code that runs once at construction, never inside `render()`.
So a correct GPU section for D must **not** treat a lone `HumanoidmazeGiantNavigate`/cuda mismatch as
a fresh D regression: it should (a) cite H's classification and jobs directly rather than re-deriving
it, and (b) escalate to REGRESSION only if the pattern is qualitatively new — e.g. the *head* tree
itself becomes non-reproducible (more than one digest across repeated runs), or *every* run of this
job disagrees with baseline rather than the one-in-several-runs pattern H already recorded. Every other
cell disagreeing, even once, should be treated as a candidate regression and investigated, since D's
merge base for the render path is otherwise untouched.

**Does this block?** No. This gap does not change my APPROVE verdict for the merge into integration.
The CPU evidence for this workstream is complete and independently reproduced by me (25/25 maze cells,
11/11 scenes, zero unexplained differences, at the merged tip); the CUDA deferral is consistent with
ruling 8's cadence (no per-workstream GPU job; one post-Wave-1 oracle) and with how B and G were
approved before their own GPU confirmation. What is different about D is that, unlike A/B/G/H/I, it
leaves no committed instrument for the coordinator to fold into that one job — so this section is the
specification the coordinator should use to write `scratch/step9/envs-ogbench/gpu.sbatch` before that
consolidated run, not a reason to withhold approval now.

## Defects

1. **[Minor, non-blocking] `NUM_ENVS_HOLDOUTS` ceiling not tightened.** See "The ratchet" above.
   `tests/unit/test_foundation_api.py:130` still reads `41`; the live count this workstream produces
   is `14`, correctly under the ceiling, but the ceiling itself was not lowered by this patch, so the
   "ratchet" is not actually tightened by D's delivery — only measured. Fix: coordinator sets the
   constant to `14` (or the final post-F number) when resolving the shared-file merge; not this
   workstream's file to edit unilaterally, so not a reason to withhold approval.
2. **[Cosmetic] Findings §1 phrasing.** `findings.md:126-130` says "The ratchet ... measures 14 on
   this branch, down from 41" — true of the *live measurement*, but a reader could take it to mean the
   enforced constant moved, which it did not (defect 1). Worth a one-line clarification in a future
   edit; not blocking.
3. **[Minor, non-blocking] No `scratch/step9/envs-ogbench/gpu*.sbatch` committed.** Unlike A, B, G, H
   and I, this workstream leaves no GPU instrument for the coordinator's consolidated post-Wave-1
   oracle, despite its own findings recording two cells (`OGBCube`, `OGBScene`) whose frames are
   CUDA-only and, as of this review, unmeasured by any completed Step-9 GPU job (H's job could not
   reach them: missing `stable_worldmodel` in `se-step6`). See "GPU evidence owed and the missing
   sbatch" above for the full scope (which cells, which venv, what a correct section must assert, and
   the `HumanoidmazeGiantNavigate` non-determinism hazard already characterized by Workstream H). Not
   a reason to withhold approval: ruling 8 defers all CUDA evidence to one post-Wave-1 job, and this
   workstream's CPU evidence is complete and independently reproduced.

None of the three defects touches correctness, pixels, state, resource lifetime, or the registry.
**REGRESSION count over everything I checked: 0.**

## Judgement

This is the same standard of evidence review_g and review_b set: every number above is one I produced
— a fresh CPU oracle run, a fresh AST enumeration of the Step-8 baseline, a fresh pre-D holdout count,
a fresh plate-probe run against a fresh baseline worktree, and live interpreter checks of the alias
identity, the `unpack` copy semantics, the `state` non-override, and the step/transition/render/close
error messages. Nothing here was taken from the author's prose. The merge of B and G into this
workstream's parent changed nothing observable in the OGBench family: the maze cells do not import any
of G's owned render modules, and the oracle proves the 25-cell/11-scene frame identity holds at the
merged tip, not just at the pre-merge commit the committed record measures. **APPROVE.**
