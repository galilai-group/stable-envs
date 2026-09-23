# Fetch family CPU checkpoint — Oscar gate pending

The family implementation and CPU acceptance work are complete. Final backend
promotion is blocked by Oscar job **6649686**, PENDING **QOSGrpGRES**, no estimated
start or GPU terminal marker. Leave the authorized job queued and collect its
real evidence before promotion. This is the explicit GPU-blocker stop condition.

- Starting branch: `repair/environment-semantics`, clean at `c3e03f198e91c632a42ea81e69b474093d0a7e84`.
- Ending branch: `repair/environment-semantics`; the containing checkpoint commit follows the two commits below.
- Implementation commit: `0dd0271f47d5b01dde1630a8cdbb1ea0072a5f92`.
- Evidence/map commit: `0038f004a98b47366081211e5735ee4fd7b3cf7b`.
- Newly accepted registrations: **none**; the original 22 remain accepted.
- Final matrix: **22 VERIFIED / 20 INCOMPLETE**. Eleven Fetch candidates have complete CPU evidence but await GPU acceptance.
- Shared code changed: **no** (Fetch-family common implementation changed; shared State/RenderState/SimulatorAdapter/parity/taxonomy/promotion machinery did not).
- Shared-contract change requested: **no**. No push.
- SWM observation integration: **DEFERRED** for all twelve Fetch registrations.

## Per-task result

| Task | Authoritative implementation | Full State widths qpos/qvel/warmstart | CPU sparse + dense | CPU aliases | GPU |
|---|---|---|---|---|---|
| PickAndPlace | Robotics 1.4.2 `pick_and_place.MujocoFetchPickAndPlaceEnv` | 22/21/21 | PASS 24 cases + lifecycle/boundaries/Variation | PASS 12 cases | Family rerun pending; original representative accepted |
| Reach | Robotics 1.4.2 `reach.MujocoFetchReachEnv` | 15/15/15 | PASS 24 cases + lifecycle/boundaries/Variation | PASS 12 cases | Pending |
| Push | Robotics 1.4.2 `push.MujocoFetchPushEnv` | 22/21/21 | PASS 24 cases + lifecycle/boundaries/Variation | PASS 12 cases | Pending |
| Slide | Robotics 1.4.2 `slide.MujocoFetchSlideEnv` | 22/21/21 | PASS 24 cases + lifecycle/boundaries/Variation | PASS 12 cases | Pending |

All also retain time, reset anchor, warmstart, controller position/quaternion,
goal, elapsed steps and complete PCG64 fields. Common restore order is unchanged.
PickAndPlace has an active gripper and possibly airborne goals; Reach has no
object; Push/Slide lock the gripper and target the table. Slide uses the source
puck model and wider/offset target distribution. Sparse reward is `-float(d>.05)`,
dense `-d`, success `float(d<.05)`, terminated false and truncation at step 50.
Restored simulator RNG replays unseeded resets; visual seeding is independent.
Physical mass/placements and visual-only changes have exact bounded CPU proofs;
Reach tests goal/reset-anchor effects and table color, with no object/mass leaf.

[Full recovery report](fetch_family_recovery.md) contains every task/reset/RNG/
Variation detail, State field and dtype, threshold-evidence distinction for
Reach, limitations, all test commands, upstream source paths and hashes, and
renderer acceptance requirements. The CPU profile is pinned MuJoCo3.8.1;
CPU full rendering is unsupported. No GPU-profile physics claim is made.

Dense/Dict exact covering proofs are the four `*-alias.json` files: same next
full State, success and done, direct dense distance rewards, exact sparse/Dict
reward and future, reset/RNG and RenderState equivalence. GPU frame-equivalence
coverage is prepared but **not yet proven**. No Dict observation container was
implemented or tested.

## Evidence and regression protection

- CPU: 96 task/reward cases, 48 alias cases, 12 public-contract declarations.
- Shared readiness/taxonomy/promotion and legacy checks: 62 passed / 56 optional skips; final readiness/promotion rerun 29 passed. Ruff and diff whitespace checks pass.
- Original 22: 94 evidence hashes and ten actual terminal logs remain intact.
- All other families' production code and metadata rows remain unchanged.
- Independent final production diff inspection: only three Fetch production files changed; all 15 renderer definitions retain identical ASTs.
- Frozen renderer baseline: `6d80b68b8b3ccd07d4207be94cbf4f0760758a06`, unchanged. `bf9101cb` is the prior driver revision, not the baseline.
- Oscar baseline Fetch source SHA256 matches the local immutable baseline: `6006720faca081cceba7c13895383e428036bb74e22073787dc1f3275de228a2`.
- Archive SHA256: `9829af3404ba7458ecb2e5e8195c3ac98f943a1e8c1f83114fb5a7b4305c4bbb`, checked on Oscar.
- [Coverage manifest](semantic_recovery_evidence/fetch_family/coverage.json), SHA256 `d0b5042eda5f2fb86d75482139a46f165d9ee01ae3108024a95d9c24bdb82255`.
- [Evidence hashes](semantic_recovery_evidence/fetch_family/SHA256SUMS), SHA256 `385800bbaf3db9c196fa847b49c14384193358f7bc666554bd413beb6e4b721b`.

## Remaining incomplete registrations

- `swm/CraftaxClassicPixels-v1`
- `swm/CraftaxClassicSymbolic-v1`
- `swm/CraftaxPixels-v1`
- `swm/CraftaxSymbolic-v1`
- `swm/FetchPickAndPlaceDense-v3`
- `swm/FetchPickAndPlaceDict-v3`
- `swm/FetchPush-v3`
- `swm/FetchPushDense-v3`
- `swm/FetchPushDict-v3`
- `swm/FetchReach-v3`
- `swm/FetchReachDense-v3`
- `swm/FetchReachDict-v3`
- `swm/FetchSlide-v3`
- `swm/FetchSlideDense-v3`
- `swm/FetchSlideDict-v3`
- `swm/OGBCube-v0`
- `swm/OGBScene-v0`
- `swm/PFRocketLanding-v0`
- `swm/PushT-Discrete-v1`
- `swm/PushT-v1`

## Resume the required GPU gate

Read Oscar job 6649686 and its log at
`/oscar/scratch/sboughan/se-semantic-recovery/logs/fetch-family-6649686.out`.
The immutable source stage is `fetch-family-0dd0271f`. Require four per-task
`*-baseline.json` / `*-candidate.json` exact comparisons and `*-gate.json`
full/private nonmutation plus Dense/Dict frame equivalence, and the terminal
`FETCH_FAMILY_GPU_GATE_PASS fetch-family-0dd0271f`. Download the evidence and
compare local/remote hashes. Only after those pass may the eleven candidates
be promoted to 33 VERIFIED / 9 INCOMPLETE. Preserve all original representative
evidence. Do not modify the frozen renderer or shared contract to bypass a failure.

## Changed files before this checkpoint

- `docs/research/fetch_family_recovery.md`
- `docs/research/semantic_recovery_evidence/fetch_family/PickAndPlace-alias.json`
- `docs/research/semantic_recovery_evidence/fetch_family/PickAndPlace-alias.out`
- `docs/research/semantic_recovery_evidence/fetch_family/PickAndPlace.json`
- `docs/research/semantic_recovery_evidence/fetch_family/PickAndPlace.out`
- `docs/research/semantic_recovery_evidence/fetch_family/PickAndPlaceDense.json`
- `docs/research/semantic_recovery_evidence/fetch_family/PickAndPlaceDense.out`
- `docs/research/semantic_recovery_evidence/fetch_family/Push-alias.json`
- `docs/research/semantic_recovery_evidence/fetch_family/Push-alias.out`
- `docs/research/semantic_recovery_evidence/fetch_family/Push.json`
- `docs/research/semantic_recovery_evidence/fetch_family/Push.out`
- `docs/research/semantic_recovery_evidence/fetch_family/PushDense.json`
- `docs/research/semantic_recovery_evidence/fetch_family/PushDense.out`
- `docs/research/semantic_recovery_evidence/fetch_family/Reach-alias.json`
- `docs/research/semantic_recovery_evidence/fetch_family/Reach-alias.out`
- `docs/research/semantic_recovery_evidence/fetch_family/Reach-invalid-fixture.out`
- `docs/research/semantic_recovery_evidence/fetch_family/Reach.json`
- `docs/research/semantic_recovery_evidence/fetch_family/Reach.out`
- `docs/research/semantic_recovery_evidence/fetch_family/ReachDense.json`
- `docs/research/semantic_recovery_evidence/fetch_family/ReachDense.out`
- `docs/research/semantic_recovery_evidence/fetch_family/SHA256SUMS`
- `docs/research/semantic_recovery_evidence/fetch_family/Slide-alias.json`
- `docs/research/semantic_recovery_evidence/fetch_family/Slide-alias.out`
- `docs/research/semantic_recovery_evidence/fetch_family/Slide.json`
- `docs/research/semantic_recovery_evidence/fetch_family/Slide.out`
- `docs/research/semantic_recovery_evidence/fetch_family/SlideDense.json`
- `docs/research/semantic_recovery_evidence/fetch_family/SlideDense.out`
- `docs/research/semantic_recovery_evidence/fetch_family/canonical-PickAndPlace.json`
- `docs/research/semantic_recovery_evidence/fetch_family/canonical-Push.json`
- `docs/research/semantic_recovery_evidence/fetch_family/canonical-Reach.json`
- `docs/research/semantic_recovery_evidence/fetch_family/canonical-Slide.json`
- `docs/research/semantic_recovery_evidence/fetch_family/coverage.json`
- `docs/research/semantic_recovery_evidence/fetch_family/diff-review.json`
- `docs/research/semantic_recovery_evidence/fetch_family/final-readiness.out`
- `docs/research/semantic_recovery_evidence/fetch_family/oscar-queue.json`
- `docs/research/semantic_recovery_evidence/fetch_family/oscar-submit.json`
- `docs/research/semantic_recovery_evidence/fetch_family/provenance.json`
- `docs/research/semantic_recovery_evidence/fetch_family/provenance.out`
- `docs/research/semantic_recovery_evidence/fetch_family/public-contract.json`
- `docs/research/semantic_recovery_evidence/fetch_family/public-contract.out`
- `docs/research/semantic_recovery_evidence/fetch_family/shared-legacy-tests.out`
- `docs/research/semantic_recovery_evidence/fetch_family/start_audit.json`
- `docs/research/semantic_recovery_implementation_map.json`
- `docs/research/semantic_recovery_matrix.json`
- `scripts/recovery/fetch_alias_gate.py`
- `scripts/recovery/fetch_family_contract_gate.py`
- `scripts/recovery/fetch_family_provenance.py`
- `scripts/recovery/fetch_public_gate.py`
- `scripts/recovery/fetch_renderer_gate.py`
- `scripts/recovery/oscar_fetch_family_gate.sbatch`
- `stable_envs/_adapters/fetch_pickandplace.py`
- `stable_envs/envs/_fetch_semantics.py`
- `stable_envs/envs/fetch.py`
- `tests/unit/test_envs_fetch.py`
