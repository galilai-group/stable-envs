# Onboarding and OGBench manipulation checkpoint

Stop here for review. Contributor onboarding is complete. Cube and Scene remain
**INCOMPLETE** after a source-level reset gate blocker; Scene additionally retains
its pre-existing frozen-renderer constructor blocker. Final matrix remains
**33 VERIFIED / 9 INCOMPLETE**. No production implementation or promotion commit
was made, and no GPU job was submitted.

## Exact commits and worktree

- Starting SHA and recovered containing PushT checkpoint:
  `e1b7e5a364b02386e1d3ed6ace29c3d22c7826f3`.
- Earlier PushT evidence: `6873b7d608cfe0fb7777fb35680e859e2160f511`.
- Earlier accepted Fetch promotion: `b45500ca84d8c6b3216d90196acac747e8fb3d5b`.
- Documentation checkpoint: `03adc1ab79bae71e3e62db2e5c4cb19f20723166`.
- OGB implementation SHA: **none**; stopped before production edits.
- OGB source evidence/schema and incomplete matrix/map update:
  `013883e87ce64d8e484bd48ebe0202ddedde4c3d`.
- This document is the containing checkpoint, whose exact SHA is reported in the
  final handoff. It cannot embed its own commit hash.

Branch throughout: `repair/environment-semantics`. Worktree:
`/private/tmp/stable-envs-semantic-recovery`. Starting and intermediate committed
checkpoints were clean; final clean status is checked after this containing commit.
No push occurred. The unrelated main checkout and other family worktrees were not
modified. No production/shared contract/test/renderer/fixture code changed. No
shared-contract change is requested. SWM learner observation integration remains
deferred; no Observation class or filtering layer was added.

## Results

The [onboarding report](onboarding_checkpoint.md) records four new guides,
a compatibility page, navigation and concept/API updates, and a site-only hook
for archived source/evidence links. It contains no production changes. All 24
files in that commit remained byte-identical during OGB work.

The [source diagnostic report](semantic_recovery_evidence/ogb_manipulation_preflight/README.md)
separates positive contact restoration from failed reset replay:

| Gate | Exact outcome |
| --- | --- |
| Source resets, seeds 0/17/42 × all five task IDs, fixed Variation | Cube 15/15 pairs differ; Scene 15/15 pairs differ |
| Persistent environment/global RNG conditioned resets, task 2 | 3/3 pairs differ per registration |
| Candidate independent/different-history roundtrip | 3/3 exact per registration |
| Eight-action continuation after contact/controller histories | 24/24 exact per registration, including candidate State and all source step outputs |
| Candidate restored-anchor unseeded reset | 3/3 differ per registration |
| Additional four-action propagation from conditioned source resets | Cube qpos/qvel equal in all 12 comparisons; Scene seed 17 differs in all four, seeds 0/42 equal |
| Frozen constructor | Cube constructs without rendering; Scene TypeError: missing env_type |
| Public adapter / exhaustive validated State / malformed input / public nonaliasing | NOT_RUN_PREFLIGHT_STOP |
| Full task success/reward boundaries, default-task distribution and aliases | NOT_RUN_PREFLIGHT_STOP; task IDs 1–5 exercised only in source reset pairs |
| Full action dtype/bounds/clipping/scaling/25-substep acceptance | Source trace plus limited mixed-dtype actions; full gate NOT_RUN_PREFLIGHT_STOP |
| Full lifecycle and raw TimeLimit gate | Source and registration inspection only; no raw horizon; full gate NOT_RUN_PREFLIGHT_STOP |
| Reset anchors and seeded/unseeded RNG acceptance | Blocked; source controls/warmstart differ despite persistent RNG alignment |
| Persistent/reset-scoped and physics/visual Variation acceptance | NOT_RUN_PREFLIGHT_STOP |
| Full-State/private RenderState frame equality and nonmutation | NOT_RUN_PREFLIGHT_STOP |
| Compact-state refusal and new public adapter wiring | No new adapter; existing renderer-only refusal retained |
| GPU frozen regression / full task-render coverage | NOT_RUN_CPU_PREFLIGHT_STOP; no availability or GPU physics claim |
| Promotion per registration | Neither promoted |

These are source-on-source diagnostics, not a candidate-versus-oracle public
adapter certificate. The named candidate includes complete MuJoCo integration
components, controller-read site caches, mutable model/task state and persistent
RNG. The [candidate schema](ogbench_manipulation_candidate_state.md) gives every
proposed field, shape, relevance/exclusion, unresolved fields and the restore order.
It does not assert that a public exhaustive reset State has been established.

Restore order tested: task/model effects first, all engine integration components,
forward, reapply integration components, restore source site/contact caches, then
RNG. The source is not restored during the eight-action continuation comparison.
Ordinary steps call the authoritative controller and MuJoCo directly. Scene lock
state alters damping and must precede forward. Cube's controller can read a cached
pinch pose from before the final qpos update; preserving that cache matters.

The reset blocker is separate: Cube drops its seed parameter, and both task
initializers create unseeded action-space generators during two goal-settling
steps. Those future generators are absent from an environment checkpoint. This
run did not replace that randomness with the renderer's class-descriptor patch.
Exact source reset-State replay therefore remains unproved. It is not a general
claim that MuJoCo contact state is inaccessible or that Cube step physics failed.

## Authority and source identities

SWM `176d54532b3c0a251a4f509b74f2fdacc143a63d`, tracked clean. CPU profile:
Python 3.12.11, OGBench 1.2.1, MuJoCo 3.8.1, Gymnasium 1.2.3, NumPy 2.2.6,
dm-control 1.0.41, dm-env 1.6, dm-tree 0.1.9, SciPy 1.17.1, lxml 6.1.1,
labmaze 1.0.6. Torch 2.10.0/TensorDict 0.14.2 and the pinned SWM import dependencies
are listed in the runtime provenance. `MUJOCO_GL=disable` scoped execution to CPU
physics. No accepted dependency profile was changed.

Inspected paths include SWM Cube/Scene overrides, OGBench Cube/Scene upstream,
ManipSpaceEnv and CustomMuJoCoEnv, MJCF helpers, IK and Lie utilities, SWM Variation
reset, Gym registration and wrappers, SWM World horizon ownership, and the frozen
renderer construction/projection boundary. Full renderer acceptance stopped at
the CPU prerequisites; source inspection is not pixel evidence.

| Source | SHA256 |
| --- | --- |
| `swm/stable_worldmodel/envs/ogbench/cube_env.py` | `143e5e81f424ff7a4d7fdde287dee88b722c09082a6fc9e68721839788e94c19` |
| `swm/stable_worldmodel/envs/ogbench/scene_env.py` | `d0eebdce0131f86e3247736c601f97c2b11d0dbfad56fca998df3192f08a7b9f` |
| `swm/uv.lock` | `643d1774c8b527dfab26070c4b16a454e501591001a97da13370f42a6e895e10` |
| `ogbench/manipspace/envs/cube_env.py` | `6f8457ffd634f6620411c788c86e871f44ea5b5b5eecf76cd694fc04d12d678d` |
| `ogbench/manipspace/envs/scene_env.py` | `b03395bc50d7a65fcd45a3141fe6cca419f07cbe7422d397003d824fdfcb7c54` |
| `ogbench/manipspace/envs/manipspace_env.py` | `03524eb1c95155bbc2ec3d29e9836fccffe8eb9fce1be15b6f0e7062cc1bb5fd` |
| `ogbench/manipspace/envs/env.py` | `c656f4733a1ad4cb07289bc58ac1ac0f7da378e1f2714dc80a5ed8c0b3028c38` |
| `ogbench/manipspace/controllers/diff_ik.py` | `4914ea5fbf6428daa4f475a0490a8786514fafd15072c5e18c5d4a5213b8f1be` |
| `gymnasium/envs/registration.py` | `3053db8b8d4716a2eaa56dc7a5b5d65e80085a091a21dd9fb63e060d524f6d01` |
| `gymnasium/wrappers/common.py` | `bc0ebcd8d1cb65286fc2d610c43bd262accfef2f06869460861a9ec0b1d86824` |
| `mujoco/include/mujoco/mjdata.h` | `1be8300051e0ecfb69c4c88b2d190d7f23c32a713434fb19cec7b06d254ecd16` |

The complete per-file source/native-library/asset inventory is in
`semantic_recovery_evidence/ogb_manipulation_preflight/provenance.json` and
`dependency_hashes.json`. Exact archived evidence hashes:

| Artifact | SHA256 |
| --- | --- |
| results.json | `72e7a48165b237033f9449565822ad938443c0c7f3030b1e170cf57eee6380ba` |
| reset_effects.json | `87be151139fd8fddf2288592c7cd47b08778f26e5c7d9e21a63456d3b53f84c2` |
| provenance.json | `f9c9c4f964d41819b0e0f55db80905358077167f9bf07d613c2409c6df07895b` |
| dependency_hashes.json | `cac4fdd685a04db0274d296ab063af072babee95d82b08fd7414c8288cc10abd` |
| closure.json | `656f7b2f76eebb1fe2f250fa224aec3f80baed57a6981ebc90aea390108d4ad1` |
| SHA256SUMS | `0ff90558c87512465717110cec3d87d0a1a1e9851061dbe505ce995a2f71d43a` |

The manifest seals 18 source-diagnostic, raw log, validation and provenance files.
Its two diagnostic completion markers explicitly say NO_PROMOTION and are never
acceptance markers. Per-registration map entries preserve public parity statuses
as UNTESTED and accepted class lists empty.

## Verification and accepted-family preservation

- Initial shared readiness/matrix/State/harness suite: **42 passed**, one optional
  ALE warning.
- Documentation targeted suite: **57 passed**, one optional ALE warning.
- Eight independently executed contributor Python examples: **8 passed**.
- Documentation strict MkDocs build: **PASS**. The pre-change build failed on ten
  archived links; the site-only hook resolves them without altering old evidence.
- OGB checkpoint shared suite: **57 passed**, one optional ALE warning.
- Final readiness/matrix/selected legacy OGB structure suite: **71 passed,
  14 deselected**, one optional ALE warning. Deselects comprise the known broken
  manipulation Variation constructor test and 13 behavioral/render tests; they
  are not counted as passed or as new Maze rendering acceptance.
- OGB strict MkDocs build: **PASS**, including link/anchor checks.
- Ruff check/format and whitespace checks: **PASS**.
- Independent final production-diff inspection: **empty**. Renderer bytes and
  therefore AST unchanged; no threshold edits.
- Final closure: **33 accepted / 170 artifact hashes / 11 terminal logs /
  75 Fetch-manifest files** all intact. Non-target matrix/map rows, old research
  evidence, accepted Maze production/evidence and unrelated family metadata are
  unchanged. Only the two incomplete manipulation rows acquire source-preflight
  records and additional limitations; count and promotion fields are preserved.

Reproduction commands for the probes are in the diagnostic README. Exact shared
and final test commands (run from the integration worktree):

```sh
.venv/bin/python -m pytest -q tests/unit/test_repo_contracts.py tests/unit/test_full_state_contract.py tests/unit/test_semantic_harness.py tests/unit/test_native_classic_semantics.py tests/unit/test_replacement_readiness.py tests/unit/test_swm_matrix.py
.venv/bin/python -m pytest -q tests/unit/test_replacement_readiness.py tests/unit/test_swm_matrix.py tests/unit/test_envs_ogbench.py -k 'not manipulation_variation_taxonomy and not takes_the_step9_constructor and not private_projection_packs and not renderer_only_cell_refuses and not private_render_takes and not dataset_siblings'
.venv/bin/python docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/audit.py
.venv/bin/python -m mkdocs build --strict --config-file mkdocs.yaml --site-dir /private/tmp/stable-envs-ogb-preflight-site
.venv/bin/ruff check docs/_hooks.py docs/research/semantic_recovery_evidence/ogb_manipulation_preflight
.venv/bin/ruff format --check docs/_hooks.py docs/research/semantic_recovery_evidence/ogb_manipulation_preflight
git diff --check
```

The pre-map closure reused the old PushT audit. After these intentional incomplete
row updates, use the new audit: the old one additionally asserts that the entire
map equals its earlier baseline. The new audit checks every accepted and other
non-target row, all original artifact/terminal hashes, the 24 documentation files,
and source identity without rewriting old evidence.

## Expected versus actual changed files

The expected scope was contributor documentation followed by OGB-only work. Actual
OGB changes are research/evidence and two incomplete metadata rows; no implementation.
The separate containing commit adds only this file.

Documentation commit (24 files):

- `CONTRIBUTING.md`
- `README.md`
- `docs/_hooks.py`
- `docs/adding-an-env.md`
- `docs/adding/environment.md`
- `docs/adding/index.md`
- `docs/adding/renderer.md`
- `docs/adding/task.md`
- `docs/api/env.md`
- `docs/concepts/dynamics.md`
- `docs/concepts/lifecycle.md`
- `docs/concepts/physics.md`
- `docs/concepts/state.md`
- `docs/concepts/worldmodel.md`
- `docs/index.md`
- `docs/quickstart.md`
- `docs/research/onboarding_checkpoint.md`
- `docs/research/semantic_recovery_evidence/onboarding/SHA256SUMS`
- `docs/research/semantic_recovery_evidence/onboarding/examples.log`
- `docs/research/semantic_recovery_evidence/onboarding/mkdocs-baseline.log`
- `docs/research/semantic_recovery_evidence/onboarding/mkdocs-strict.log`
- `docs/research/semantic_recovery_evidence/onboarding/tests.log`
- `docs/testing.md`
- `mkdocs.yaml`

OGB evidence commit (22 files):

- `docs/research/ogbench_manipulation_candidate_state.md`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/README.md`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/SHA256SUMS`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/audit.py`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/closure-before-map.log`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/closure.json`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/closure.log`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/dependency_hashes.json`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/final-readiness-legacy.log`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/formatting.log`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/mkdocs-strict.log`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/probe.log`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/probe.py`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/production_diff_review.json`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/provenance.json`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/reset_effects.json`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/reset_effects.log`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/reset_effects.py`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/results.json`
- `docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/tests.log`
- `docs/research/semantic_recovery_implementation_map.json`
- `docs/research/semantic_recovery_matrix.json`

## Remaining incomplete registrations and exact resumption

1. `swm/CraftaxClassicPixels-v1`
2. `swm/CraftaxClassicSymbolic-v1`
3. `swm/CraftaxPixels-v1`
4. `swm/CraftaxSymbolic-v1`
5. `swm/OGBCube-v0`
6. `swm/OGBScene-v0`
7. `swm/PFRocketLanding-v0`
8. `swm/PushT-Discrete-v1`
9. `swm/PushT-v1`

Resume only after review of the family-specific reset entropy/oracle-conditioning
policy. That decision must preserve source action distributions, expose all
replay-relevant RNG, isolate lanes/instances, and prove reset State parity without
pretending the unmodified source honors its seed. Do not weaken the shared State
contract, omit an observed engine field merely to hide a reset mismatch, or
silently reuse the renderer's temporary class mutation as semantic acceptance.
For Scene, separately review the pre-existing constructor/signature compatibility
with a frozen-renderer fix or a compatible rendering profile; neither is certified
here. Cube can proceed independently if its own gates are resolved.

Then implement the delegated adapter from the candidate schema, validate the full
input before mutation, and complete every outstanding CPU gate in the table.
Only after CPU acceptance: inspect all six Oscar skills, stage one immutable
source/archive with local/remote hashes, and submit one authorized GPU gate.
Require frozen comparison criteria, exact full/private equality and nonmutation,
per-task/configuration coverage, raw evidence and terminal success; download and
verify before promotion. If queued/unavailable, leave that registration incomplete
and provide job/stage-specific resumption details. There is no pending job from
this checkpoint to poll. Keep PushT parked and do not begin Craftax or Rocket.
