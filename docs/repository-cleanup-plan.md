# Repository cleanup plan

Status: **carried out on 2026-10-08** from tag `pre-repository-cleanup` (`4a03d4a0`).
This complements [Package cleanup](package-cleanup.md), which covers layout inside
`stable_envs/`. The plan below is kept as approved; the outcome and its deviations
come first.

## Outcome

- Every move and deletion was compared with the pre-cleanup tree under the same
  interpreter. The final non-GPU suite has no new failures; every test that is no
  longer collected was removed or moved on purpose. The strict docs build passes.
- Many semantic, OpenGL and Craftax tests already fail on the local Mac at the
  baseline (no OpenGL route, off-pin interpreters). For those, identical outcomes
  show only that moved code resolves; they still need the software-GL setup or
  Oscar. The GPU suite has not been rerun.
- Deviations from the plan:
  - The online source-stepping harness was kept as a maintained benchmark in
    `benchmarks/online/` instead of being retired; the Fetch comparisons it used
    moved to `tests/helpers/reference/fetch_compare.py`.
  - `docs/research/` was deleted entirely. Maintained docs no longer link to it;
    code comments and evidence fields cite `pre-repository-cleanup:<path>`.
  - Checks that read archived evidence but test live behavior kept small dedicated
    fixtures (`tests/fixtures/saved/`, `tests/fixtures/swm/`,
    `tests/fixtures/evidence/`) rather than being dropped.
  - For the 15 Fetch, PushT and Rocket registrations whose recorded gates were
    retired scripts, the inventory requires the shared state, simulation and task
    contracts and keeps the script names in `historical_gates`.
  - `scripts/submit_test_matrix.sh` stayed: it launches the GPU test matrix.
- Not done: the Oscar evidence tarballs in `tests/fixtures/testing/` (about 140 MB,
  no code consumer) await a decision; detached worktrees and untracked
  `STEP_0x_AGENT_BRIEFS/` are left for the owner.

## Rule

Keep a test if it fails when any of the following breaks:

- library behavior;
- the testing infrastructure itself: source comparisons, fixture and corpus
  integrity, seal and hash verification, refusal of partial or unverified runs;
- measurement correctness in a maintained benchmark: what is timed, warmup
  exclusion, percentile calculation, call-boundary accounting.

Remove a test if it fails only when a record of past development changes: a report
page, a support matrix, a certification ladder, or agreement between a findings
document and its artifacts.

Classify individual tests, not files. A kept test must not keep its target alive in a
research directory: move the code it exercises into `tests/helpers/` or point the test
at maintained code.

No package code reads Markdown. `stable_envs/` only mentions `docs/research/` paths in
error messages, comments and `evidence:` strings.

## 0. Checkpoint before any change

1. Tag the current tip: `git tag -a pre-repository-cleanup -m "Before repository cleanup" <tip>`.
   `step7-closure` and `step8-closure` (2026-09-17) predate the recovery evidence
   added from 2026-09-22 to 2026-10-07, including `semantic_recovery_matrix.json` and
   the October Rocket evidence. Pushing the tag is Sami's decision.
2. Keep the existing tags `renderer-recovery-backup-20260930` and
   `renderer-cleanup-archive-20261006`, and the backup folder.
3. Before removing any worktree, list uncommitted and untracked files in it
   (`git -C <worktree> status --short`) and preserve anything unique.

## 1. Tests: classify per test

### Delete whole file

Every test in these files checks a historical record or a report generator.

| File | What it checks |
|---|---|
| `tests/regression/test_report_generator.py` | Step-6 report page regenerates over Step-6 artifacts |
| `tests/regression/test_oracle_contract.py` | `docs/research/certification.md` agrees with `certification.json`; ladder shape |
| `tests/regression/test_downstream_parity.py` | Step-5 paired-gate statistics; re-audit of a committed artifact |
| `tests/regression/test_fidelity_oracle.py` | Step-4 certifier on hand-written shards |
| `tests/regression/test_semantic_equivalence.py` | Step-3 certifier on hand-written shards |
| `tests/regression/test_support_matrix_gate_a.py` | Historical Gate-A reporting |
| `tests/regression/test_support_matrix_performance.py` | Committed performance accounting equals generator output |
| `tests/regression/test_gate_f_matrix.py` | Gate-F/API matrix contract |
| `tests/regression/test_reference_provenance.py` | Provenance records under `join/step8/provenance` |
| `tests/regression/test_craftax_semantic_evidence.py` | Step-8 Craftax findings agree with their artifacts |
| `tests/unit/test_swm_matrix.py` | Coverage-matrix generator |

Before deleting each, read every test in the file once more and confirm none checks
infrastructure under the rule above.

### Split: keep some tests

| File | Keep | Remove |
|---|---|---|
| `tests/regression/test_testing_migration.py` | `test_native_saved_corpus_rejects_missing_or_changed_inputs`, `test_partial_or_blocked_source_profile_cannot_pass`, `test_required_profile_does_not_import_unverified_installed_source`, `test_new_rocket_coverage_rejects_duplicate_or_missing_groups_before_pixels`, and the code-hash rejection in `test_historical_report_does_not_certify_changed_code`. Where these read the archived Rocket evidence tree, replace it with a small dedicated fixture under `tests/fixtures/` (minimal `result.json`, hashed files and coverage groups) in the same commit; do not drop the checks. | `test_rocket_old_and_new_verifiers_agree` (its only purpose is agreement with the historical verifier) |
| `tests/unit/test_fetch_source_pipeline.py` | `test_source_state_projection_preserves_owned_rng_and_controller_fields` | Publishing/timing-admission tests (lines 69, 112, 143), unless the reviewer finds a maintained benchmark that still needs them |
| `tests/unit/test_dmc_source_summary.py` | `test_image_mismatch_and_nan_task_comparison`, `test_single_step_normalization_preserves_frame_axis`, if their targets are source-comparison helpers | Summary/publishing tests, unless still needed by a maintained consumer |
| `tests/integration/test_worldmodel.py` | `test_swap_renderer_keeps_the_live_swm_pusht_and_replaces_only_pixels`, `test_shipped_package_has_no_interop_module`, `test_variation_map_bridge_is_test_only` | `test_all_42_frozen_swm_ids_join_the_same_canonical_classes`, `test_swm_ids_are_not_registry_aliases` only if their fixtures are removed; otherwise keep |
| `tests/unit/test_replacement_readiness.py` | VariationMap classification tests, if the maps stay | `test_the_readiness_report_counts_match_the_inventory` (reads a research report) |
| `tests/unit/test_ogbench_manipulation_acceptance.py` | `test_live_recovered_configuration_matches_readiness` | `test_each_family_has_its_own_complete_acceptance_evidence` |

The kept tests import code from `scripts/benchmarks/online_stepping/`. Move those
targets in section 3 in the same commit as the test edits.

### Retarget at maintained benchmarks

| File | Action |
|---|---|
| `tests/unit/test_state_ops_benchmark.py` | `test_preparation_and_warmup_are_excluded` and `test_percentiles_come_from_retained_samples` currently load `scripts/benchmarks/state_ops.py`. `benchmarks/_common.py` implements `--warmup`, `_percentile` and `measure`. Port both checks to `benchmarks/_common.py`. Drop `test_all_accepted_registrations_are_separate_cases` with the old driver. |
| `tests/unit/test_rendered_state_benchmark.py` | `test_composed_rollouts_count_each_boundary_once` checks which operations a measurement contains. Port it to the matching maintained path in `benchmarks/` (`render.py` or `pipeline.py`). If no maintained path composes the same calls, record the missing safeguard instead of deleting it. |

Delete the old drivers only after the ported tests pass against `benchmarks/`.

### Keep

`tests/regression/test_declarations.py`, `tests/regression/test_provenance.py`, all of
`tests/contracts/`, the integration semantic gates, and every test that loads a fixture
moved in section 3.

## 2. Scripts

Delete:

- `scripts/regression/`: `fidelity/`, `semantic/`, `support/`, `optimization/`,
  `downstream/`, `swm_matrix.py`.
- `scripts/research_replay/`, `scripts/research_rocket_production/`.
- `scripts/benchmarks/`, except the files moved in section 3, and after the ports
  above.
- `scripts/recovery/`: probes (`craftax_*_probe.py`, `craftax_preservation.py`,
  `fetch_pickandplace_adapter_probe.py`, `rocket_feasibility.py`,
  `rocket_snapshot_controls.py`), one-off gates and tools (`craftax_gate.py`,
  `fetch_alias_gate.py`, `fetch_family_contract_gate.py`, `fetch_family_provenance.py`,
  `fetch_public_gate.py`, `fetch_renderer_gate.py`, `family_renderer_boundary.py`,
  `renderer_boundary.py`, `ogbench_renderer_bundle.py`, `ogbench_renderer_gate.py`,
  `verify_craftax_gpu.py`, `verify_rocket_audit.py`, `restore_renderer_backup.py`,
  `build_implementation_map.py`), all `*.sbatch`, and `profiles/`.
- `scripts/bench.py`, `scripts/submit_test_matrix.sh`.

Keep:

- `scripts/ci/` (workflows), `scripts/testing/`, root `run_*.sbatch` launchers.
- Fixture generators: `scripts/regression/craftax/gen_prng_golden.py` (generates
  `tests/fixtures/regression/fidelity/prng_threefry_golden.json`, loaded by
  `test_craftax_prng.py`) and `scripts/regression/craftax/gen_golden.py` (check it
  generates `craftax_full_golden.json`). Move them to `scripts/testing/` or keep them
  beside their fixture; update the `generated_by` field only if its convention
  allows, otherwise add a note where the generator now lives.
- `scripts/regression/craftax/certify.py`: check consumers before deciding.

## 3. Moves: one move and its consumers per commit

Each move is a single commit containing the move, every import and path update for
its consumers, and the verification below. Fixture moves must keep bytes identical
(`git mv`; compare `sha256sum` before and after) and keep provenance fields and
regeneration instructions valid.

| From | Consumers to update in the same commit | Suggested home |
|---|---|---|
| `scripts/recovery/{dmc_aligned,dmc_continuation,dmc_first_family,dmc_humanoid,native_navigation,ogbench_maze,reacher}_gate.py` | `tests/integration/test_*_semantics.py`, `test_native_navigation_gate.py`, `tests/helpers/reference/inventory.py`, fixture manifests that name the gate | `tests/helpers/reference/gates/` |
| `scripts/recovery/freeze_*_oracle.py` (9) | `tests/fixtures/semantics/*/manifest.json` generator fields, `docs/testing.md` | `scripts/testing/` |
| `scripts/benchmarks/online_stepping/{fetch_parallel_source,fetch_source_frame_gate,fetch_source_transport_gate,ogbench_parallel_source,ogbench_source_frame_gate,dmc_state_export_oscar}.py`, plus targets of the kept tests in section 1 | `test_fetch_source_frames.py`, `test_ogbench_source_frames.py`, kept parts of `test_fetch_source_pipeline.py` / `test_dmc_source_summary.py`, `inventory.py` `ONLINE` map | `tests/helpers/reference/` |
| `scripts/benchmarks/renderer_table/common.py` | `test_fidelity_backends.py` | `tests/helpers/` |
| `scripts/regression/raster2d/fuzz_render_torch.py` | `test_raster2d_torch_backend.py` | `tests/helpers/` |
| `tests/fixtures/regression/join/step4/states/*.npz` | `test_cpu3d`, `test_dmcontrol_tendon`, `test_envs_fetch`, `test_envs_craftax_rocket`, `gpu/test_fast_paths`, `gpu/test_gfxdraw_classic` | `tests/fixtures/states/` |
| `join/step7/craftax-classic-full/corpus/`, `join/step7/ogb-manipulation/states/`, `join/step7/rocket-landing/reference/`, `join/step7/dmc-quadruped/phase1/cpu_oscar/kind_cost_ellipsoid.json`, `join/step6/raycast3d/experiments/`, `join/step8/variations-fetch/leaf_classification.json` | `test_envs_craftax_rocket`, `test_ogb_manip`, `test_rocket_landing`, `test_raycast3d_primitive_bounds`, `test_variation_maps_fetch` | purpose-named directories under `tests/fixtures/` |
| `docs/research/semantic_recovery_matrix.json` | `tests/helpers/reference/inventory.py` and its tests | `tests/fixtures/` |
| `docs/research/semantic_recovery_evidence/craftax_source_audit/` | `tests/helpers/reference/craftax_source.py` | `tests/fixtures/`, unless the Craftax tests are redone (package cleanup item 5) |

Verification for each move:

- run the consumers named in the row, plus `test_contract_inventory.py`;
- for moved source helpers or inventory changes, run the affected source-comparison
  tests in their pinned interpreters and compare collected registration counts and
  case IDs before and after (`pytest --collect-only -q`);
- report skips separately; a missing dependency is not a pass;
- fixture-only moves need no new performance or GPU measurements; a GPU suite run on
  Oscar is still needed once for `gpu/test_fast_paths` and `gpu/test_gfxdraw_classic`
  to confirm the new paths resolve.

## 4. Data and documents

Delete, after section 3:

- `docs/research/` (about 416 MB, 11,629 tracked files).
- `tests/fixtures/regression/join/`, except what section 3 moves.
- `tests/fixtures/regression/fidelity/certification.json` and `fidelity/swm_matrix.json`,
  after confirming their remaining consumers are all deleted (`git grep` must be empty
  outside deleted files).
- `tests/fixtures/regression/support_matrix.json`, `support_matrix_diff.json`,
  `gate_f_matrix.json`, `performance_tiers.json`, under the same check.
- Internal notes in `docs/`: `docstring-cleanup.md`, `testing-redundancy-audit.md`,
  `testing-design.md`, `docs/recovery/` (empty files); later `package-cleanup.md` and
  this page once the work is done.

Keep:

- `tests/fixtures/regression/fidelity/prng_threefry_golden.json` and
  `craftax_full_golden.json`.
- `tests/fixtures/regression/census/assignment.json`: the GPU job cover used by
  `gpu/test_conformance.py`, `gpu/test_rendernet.py` and `stable_envs/testing/gate.py`.
- `tests/fixtures/regression/references/registry_baseline.json` (`test_registry.py`) and
  `references/benchmarks/` (`test_benchmark_surface.py`).
- Trace the rest of `census/` (`census_*.json`, `swm_registry*.json`, `union.json`) and
  `references/` (`bench_gpu_vs_cpu.json`, `decode_all.json`, `dino_agreement.json`,
  `lance.json`, `reference_arm.json`, `swm_workflow.json`, `frozen_oracle/`) after the
  test deletions. Delete a file only when `git grep` finds no remaining consumer;
  `test_repo_contracts.py` `REQUIRED_FIXTURES` counts as a consumer until edited.

## 5. Edits that are not tied to a single move

- `.github/workflows/testing.yml:38`: keep `test_testing_migration.py`; it survives the
  split. Rename it later if a clearer name fits its remaining checks.
- `docs/testing.md`: remove commands and sections for deleted gates.
- `mkdocs.yaml`: remove the research section and nav entries for deleted pages.
- `tests/unit/test_repo_contracts.py`: `REQUIRED_FIXTURES`, `_ALLOW_SCRATCH_MENTION`, and
  the research-archive nav rule.
- `tests/packaging/test_distribution.py:134` and `.pre-commit-config.yaml`: the
  `docs/research/` exclude.
- `pyproject.toml`: any sdist exclude for `docs/research/`.
- Package text: replace "See docs/research/..." in `humanoidbench.py`, `pgx.py`,
  `jumanji.py`, `metaworld.py`, `robocasa.py`, `_dynamics.py`, `dmcontrol/_model.py`, and
  the `evidence:` strings in `rocket_landing.py` and `craftax/*.py`, with short inline
  explanations or links to the published docs.
- `stable_envs/testing/gate.py`: the `__main__` launcher should not read
  `tests/fixtures/` from package code.

## 6. Local cleanup

Not tracked in git; remove only after the checks in section 0:

- `tmp-step9-envs-mujoco*`, untracked `STEP_0x_AGENT_BRIEFS/`;
- `stable_envs/core/` and `stable_envs/compat/` (only `__pycache__`);
- detached worktrees, each after checking for unique uncommitted files.

Tags are git objects, not local clutter; keep them.

## Final verification

```bash
python -m pytest tests/unit tests/regression tests/integration tests/contracts tests/packaging -m "not gpu" -q
mkdocs build --strict
```

Compare the collected test count and skip reasons with the count before cleanup;
every difference must be a test this plan deletes. Run the GPU suite once on Oscar.
Rendering and simulation code do not change, so performance measurements are not
repeated. Source comparisons, including source-pixel tests, follow section 3: rerun
every comparison whose helpers, inventory entries or fixtures moved.
