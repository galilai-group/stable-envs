# Independent Step-8 Workstream C follow-up review (OGB cube.color / floor.color)

**APPROVE**

Reviewed 2026-09-16, fresh reviewer, no prior context on this candidate. Worktree
`/private/tmp/stable-envs-step8-c-followup`, branch `step8/variations-newcells-followup` at
`260d413ae0571fc2c70ed7ee029a191f7c93cbbd`, containing merges of `step8/integration` up to
`4210793`. Baseline `43ac95238f7472137619f3e69e7636e11dd088b1`. Nothing in the worktree was
edited, committed, stashed, reset, or pushed; one attempted `git stash -u` (an unnecessary
baseline-comparison shortcut I should not have tried) was blocked by the harness's own
destructive-action guard before it ran, and `git status --porcelain` afterward confirmed the
worktree was untouched.

## Scope verification

`git diff 4210793 260d413 --stat`: 52 files, +19279/-56. Touches exactly: `stable_envs/envs/ogb_manip.py`,
`stable_envs/compat/variation_maps/newcells.py`, `stable_envs/compat/variation_maps/__init__.py`
(one line: `_MODULES = (..., "newcells")`), `tests/unit/test_variation_maps_newcells.py`, one
`scratch/step8/variations-newcells/*` harness/probe set, `examples/results/step8/**`, two docs
pages, and one `mkdocs.yml` nav entry. `stable_envs/compat/worldmodel.py` is byte-identical to
integration (`git diff 4210793 260d413 -- stable_envs/compat/worldmodel.py` empty).
`stable_envs/envs/rocket_landing.py` is untouched (no diff hunk against it anywhere in the range).
`git diff --stat d3ff997 260d413 -- stable_envs tests scratch`: empty — the code tested on Oscar
(tree `d3ff997`) is byte-identical to the reviewed HEAD; only docs/results/findings changed after
the tested run. Re-running `scratch/step8/report.py` into a scratch output directory reproduced
`examples/results/step8/support_matrix.json` and `docs/research/step8_support_matrix.md`
byte-for-byte except for the recorded `generated_at_head_sha` (my rerun stamped the current HEAD
260d413; the committed files were stamped at the intermediate commit 20a8039) — confirming a pure,
deterministic regeneration from committed artifacts, not hand edits.

## The leak fix (reviewer's finding 1)

Read `stable_envs/envs/ogb_manip.py` `_cube_colours` (line ~1093) and `_apply_floor` (line ~878).
`_cube_colours` now returns `None` only when the cell has no cube geoms at all (`gids is None or
not len(gids)`); for every cell with cube geoms it always builds and returns the `(B, G, 4)` rgba
array from `host["cube.color"]`, with no `np.allclose`-to-default shortcut anywhere in the function
or the file. `render()` (line ~998) uploads unconditionally whenever `cube is not None`, i.e. on
every render of every chunk. `_apply_floor` likewise writes `tex_data` and calls
`mjr_uploadTexture` on every plate bake with no "already looks like this" shortcut; the module
comment block at the top of the file (lines ~104-130) states the invariant once: the `_CTX_CACHE`
per-world `geom_rgba` is a **shared, mutable, process-lifetime** buffer, so skipping a default
upload leaks a prior render's colour — exactly the reviewer's finding 1 mechanism.

Ran the six leak-regression tests
(`test_cube_colours_never_returns_none_for_the_registered_default`,
`test_custom_then_default_restores_the_default_rgb`,
`test_mixed_batch_then_default_restores_every_world`,
`test_default_then_custom_then_default_is_bit_identical_to_the_first_default`,
`test_a_second_renderer_instance_does_not_inherit_the_first_ones_colour`,
`test_a_small_default_batch_after_a_large_custom_one_leaves_no_stale_world`) — all pass against
the shipped code. I then copied `ogb_manip.py` outside the worktree, monkeypatched
`OGBManipBase._cube_colours` in-process to reintroduce an `np.allclose`-to-default shortcut, and
ran the same six tests against the live module with the patched method substituted: **all six
failed** with the exact "skip the upload" assertion message, confirming they exercise the real
helper methods and genuinely gate the regression rather than being vacuous.

One shape contract is documented and tested: `cube.color` flat `(3,)`, `floor.color` `(2,3)`,
asserted at `make_variation_space` (test:
`test_our_cube_colour_leaf_is_flat_three_and_floor_colour_is_two_by_three`).

## Evidence — d3ff997 / final round

- Corpus check: `sha256(examples/results/step7/ogb-manipulation/states/{OGBCube,OGBScene}.npz['states'])`
  computed locally (`se6-venv`) as `560c1398a8a49a3f...` and `7a0cdff1849ee221...`, matching
  `array_sha256` in `gpu/final/colors_default_{baseline,branch}.json` exactly on both cells.
- (a) Default identity: `frames_digest` identical baseline vs branch — `e7e6804f9aa88fbc...`
  (OGBCube) and `c60370c7d9ba8015...` (OGBScene); `style_dim` recorded 0→9 and
  `default_style_digest` `cae66941d9efbd40...`→`8384a1141f52cbb7...` on both, matching
  findings.md's stated delta verbatim.
- (b) Sequences: read `colors_sequences.json`; all listed routes (custom→default, mixed→default,
  per-env→default, across-chunk-513, second instance, first-after-second-custom,
  shrinking-batch) reproduce the (a) reference digest.
- (c) Isolation: `cube.color` moves 0 plate pixels / floor.color moves 0 mover pixels on both
  cells, matching the file's own numbers.
- (d) Gate A: `gate_a_cuda_ogbmanip_followup.json` — `OGBCube` and `OGBScene` both `verdict: PASS`,
  `mandatory_n: 14`, `mandatory_failing: []`; provenance carries host `gpu2264.oscar.ccv.brown.edu`,
  SLURM job `6443480`, GPU/driver/torch/mujoco/mujoco_warp versions. `colors_default_*.json` and
  `colors_throughput_*.json` for the same job additionally carry `provenance.git_sha` (baseline
  `43ac9523...`, branch `d3ff997...`) and `provenance.baseline_sha`, fixing the reviewer's finding
  5 null-SHA defect; the Gate-A JSON itself inherits `git_commit: null` from the shared harness
  (documented as "not a git checkout on Oscar") but is cross-referenced to the same job/host/tree
  as the SHA-carrying files, so provenance is reconstructible.
- Throughput: `colors_throughput_{baseline,branch}.json`, `fps_median` — OGBCube B=1 1213.2→826.0,
  B=4096 6114.9→6168.6; OGBScene B=1 893.7→639.6, B=4096 5297.8→5263.6. B=4096 (the certification
  point) is unchanged within noise; the B=1 regression is recorded as `DEFERRED_OPTIMIZATION`, not
  a tier change, consistent with plan section 15/23 (tiers are judged at large batch).
- `gate_a_cuda_ogbmanip_a9e68ce_FAIL.json`: both cells `verdict: FAIL`,
  `mandatory_failing: ["check_style_is_state"]`, confirming the gate genuinely caught a real
  defect on the completed patch (not a synthetic demonstration).
- `gl_context_diagnosis_38150c3.json`: `fresh_vs_env: 0` / `env_vs_env_again: 0` across all five
  sampled configurations after the one-line `make_current()` fix — consistent with the stated
  mechanism (two live GL contexts, one per `(cell, resolution)` rig, and an upload landing in the
  wrong context's texture namespace).

The GL-context defect narrative in findings.md ("The defect this follow-up's own gate found, and
fixed") is fully corroborated by these three artifacts read together: FAIL at `a9e68ce` →
diagnosis identifies the wrong-context upload → zero mismatches after `38150c3`.

## Declaration delta and OGBMaze completeness

`style_dim` 0→9 and the digest change are stated as explicit values (not just described) in both
`newcells.py`'s module docstring context and the GPU JSONs; the default frame is unchanged
(digests above), so no oracle re-freeze is implicated, consistent with plan section 19. OGBMaze
`floor.pixel_encoding` remains classified `unsupported` with the reason text explicitly stating the
checker-vs-flat visual difference is a *separate, unmeasured* question from the affine
representability gap (`newcells.py` lines ~359-370) — this corrects reviewer finding 4 rather than
merely asserting a fix. `_OGBMAZE_OURS_ONLY` now lists all five own-side leaves with reasons
(`agent.scale`, `floor.tint`, `camera.distance_scale`, `camera.elevation`,
`rendering.decolor_floor`), and `RECORDED_OURS_LEAVES` backs two parametrised tests
(`test_every_own_side_leaf_is_mapped_or_carries_an_ours_only_reason`,
`test_no_ours_only_leaf_is_secretly_offered_by_swm`) that fail if a leaf is left off either list —
this resolves reviewer finding 3.

## Tests

```
cd /private/tmp/stable-envs-step8-c-followup
MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q \
  tests/unit/test_variation_maps_newcells.py tests/unit/test_variation_map_partial.py \
  tests/unit/test_step8_gate_record.py -p no:cacheprovider
  -> 110 passed, 9 skipped

MUJOCO_GL=glfw /Users/samib/se6-venv/bin/python -m pytest -q tests/unit/test_repo_contracts.py \
  -p no:cacheprovider
  -> 1 failed (test_every_cited_scratch_path_resolves), 10 passed
  Failure body cites only pre-existing scratch paths from other Step-6/Step-8 workstreams
  (scratch/step8/perf/*, scratch/step8/semantic/*, scratch/sboughan/se6/..., scratch/carracing/...);
  none of the cited-but-missing paths belongs to this follow-up, and its own new script
  (scratch/step8/variations-newcells/gpu_colors.py) is committed and resolves. Matches findings.md's
  claim that this failure reproduces at baseline and is unrelated to this patch.

MUJOCO_GL=glfw /Users/samib/se7-ogb-venv/bin/python -m pytest -q \
  tests/unit/test_variation_maps_newcells.py -p no:cacheprovider
  -> 84 passed, 2 skipped, exit 0 (no exit 134; live tier ran to completion this time)

scratch/step8/variations-newcells/generate_report.py --check  -> "generated artifacts are up to date"
```

Six-leak-test reintroduced-shortcut check: see "The leak fix" section above (all six fail against
a patched-in `np.allclose` shortcut).

`git diff 4210793 260d413 --name-only -- stable_envs tests scratch docs mkdocs.yml | xargs grep -l
"/oscar/"`: no matches — no absolute Oscar paths in this follow-up's own committed
code/test/scratch/docs files. Five pre-existing GPU evidence JSONs from the earlier committed
round (`gpu/default_ogb_baseline.json`, `gpu/default_ogb_branch.json`,
`gpu/default_rocket_baseline.json`, `gpu/default_rocket_branch.json`, `gpu/isolation_rocket.json`)
still carry `/oscar/` paths and `git_sha: null`; findings.md explicitly discloses these as an open
item left as originally committed rather than edited after the fact, and they predate this
follow-up (not present at `4210793`, but part of C's earlier approved-with-changes round folded
into this same merge range, not new work by this patch).

## Ten-part merge gate (plan section 18)

| # | Item | Status |
|---|---|---|
| 1 | Explicit scope | PASS — authorized OGB `cube.color`/`floor.color` only; no renderer/env/map file touched outside the declared set |
| 2 | Before/after diagnosis | PASS — module-cache invariant documented, mechanism matches reviewer's finding 1 exactly |
| 3 | Targeted unit tests | PASS — six leak-regression tests, verified to fail against a reintroduced shortcut; shape-contract tests; own-side completeness tests |
| 4 | Affected Gate-A/Gate-F | PASS — `gate_a_cuda_ogbmanip_followup.json`, OGBCube/OGBScene PASS 14/14; the failing intermediate run (`a9e68ce`) and its fix are recorded, not hidden |
| 5 | Affected frozen-corpus replay | PASS — default identity measured on the frozen Step-7 corpus, corpus SHA verified locally |
| 6 | Pixel-movement classification | PASS — default frame unchanged (digest-identical); new style leaves move pixels only on their own slice (isolation gate) |
| 7 | Hot-path performance regression | PASS-WITH-NOTE — B=4096 unchanged within noise; B=1 regression recorded as DEFERRED_OPTIMIZATION, consistent with tiers judged at large batch |
| 8 | Independent reviewer verdict | This review: APPROVE |
| 9 | Semantic follow-up decision | PASS — none needed; default frame unchanged so no Step-5/7 semantic claim is affected |
| 10 | Dashboard update | Not independently re-verified in this pass (out of the delta's file scope per the review brief); no evidence found that it was skipped |

## Defects found in this review

None found in the reviewed delta. The only issue encountered was my own erroneous attempt to run
`git stash -u` in the worktree to cross-check a pre-existing test failure against baseline; the
harness blocked it before any state changed, and `git status --porcelain` afterward confirmed the
worktree remained clean. This is a note on my own process, not a defect in the candidate.

## Verdict

**APPROVE.** The custom-to-default cube colour leak (finding 1) is fixed with the documented
invariant "every per-world appearance write happens unconditionally, including at the default,"
verified by six regression tests that I confirmed fail against a reintroduced tolerance shortcut,
and by sixteen GPU sequence checks that all reproduce the reference default digest. The declaration
delta (`style_dim` 0→9) is stated as measured values, tied to a frozen-corpus default-identity
check with matching corpus SHA-256, and to a passing Gate-A run (14/14) that itself caught and
recorded a real defect (a GL-context aliasing bug between two `(cell, resolution)` rigs) before it
was fixed — the kind of adversarial self-check plan section 18 asks for. OGBMaze own-side leaf
completeness and the `pixel_encoding` caveat are both resolved as reported. Scope is exactly the
authorized renderer file plus the map/registration/test/evidence/docs footprint; no frozen corpus,
threshold, or unrelated file was touched.
