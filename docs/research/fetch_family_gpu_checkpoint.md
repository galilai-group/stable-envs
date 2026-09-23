# Fetch-family GPU acceptance checkpoint

All twelve Fetch backend registrations are VERIFIED within the recorded bounded
scope. Eleven candidates are newly promoted. **33 VERIFIED / 9 INCOMPLETE**.
SWM observation integration remains **DEFERRED**. Stop here for review; no new
family work was begun.

## Git and scope

- Branch: `repair/environment-semantics`.
- Starting HEAD: `6f3cc85d6ef40007ae5eae99ea66e0fa60abf446`, clean.
- Ending HEAD: the commit containing this report; its exact SHA is returned in the handoff.
- Commit: `Accept Fetch family GPU evidence and promote eleven registrations`.
- Production code changed: **no**. Test drivers, fixtures, thresholds and shared machinery unchanged.
- Shared-contract change requested: **no**. Push performed: **no**.
- Original 22 closure: **PASS**, 94 artifact hashes and ten terminal logs rechecked; all original rows remain VERIFIED. Original Fetch representative raw evidence is preserved.

## Jobs and immutable inputs

| Job | Partition / QOS | Final state | GPU | Elapsed | Exact family marker |
|---|---|---|---|---|---|
| 6649686 (primary required job) | 3090-gcondo / cs-3090-gcondo | COMPLETED 0:0 | NVIDIA RTX A5000 | 2:48 | Yes |
| 6652487 (explicitly authorized duplicate) | gpu / norm-gpu | COMPLETED 0:0 | NVIDIA RTX A5500 | 4:49 | Yes |

Both logs contain exactly:
`FETCH_FAMILY_GPU_GATE_PASS fetch-family-0dd0271f`.
Neither job was cancelled. The duplicate isolates its result files in
`/oscar/scratch/sboughan/se-semantic-recovery/fetch-normgpu-6652487/`.
The source, driver, fixtures and comparisons are identical to the original gate.

- Source stage: `fetch-family-0dd0271f`; implementation `0dd0271f47d5b01dde1630a8cdbb1ea0072a5f92`.
- All **189 staged files** matched the immutable archive.
- Archive SHA256: `9829af3404ba7458ecb2e5e8195c3ac98f943a1e8c1f83114fb5a7b4305c4bbb`.
- Frozen baseline: `6d80b68b8b3ccd07d4207be94cbf4f0760758a06`.
- Oscar baseline Fetch source SHA256: `6006720faca081cceba7c13895383e428036bb74e22073787dc1f3275de228a2`.
- GPU profile: Torch 2.14.0+cu126, NumPy 2.5.3, Gymnasium 1.3.0, Robotics 1.4.2, MuJoCo/MJWarp 3.11.0, Warp 1.17.0, TensorDict 0.14.2.
- CPU physics authority remains MuJoCo 3.8.1; no GPU-profile physics claim. CPU full rendering remains `CPU_BACKEND_MISSING`.

## Per-task GPU outcome (both jobs)

| Task | Frozen baseline/candidate | Full/private equality | Snapshot/live nonmutation | Visual Variation | Dense/Dict frames |
|---|---|---|---|---|---|
| PickAndPlace | Byte-identical | PASS | PASS | object.color PASS | Canonical + varied exact |
| Reach | Byte-identical | PASS | PASS | table.color PASS | Canonical + varied exact |
| Push | Byte-identical | PASS | PASS | object.color PASS | Canonical + varied exact |
| Slide | Byte-identical | PASS | PASS | object.color PASS | Canonical + varied exact |

Every task also refused compact RenderState restoration. The canonical CPU State
fixture hashes match the GPU gate records. Deduplicated frozen alias coverage is
justified by identical scene, projection, renderer and default appearance plus
explicit Dense/Dict frame comparisons in both runs. No observation container
integration is claimed. Frozen frames also matched across A5000 and A5500.

## Evidence, hashes and checks

All raw data is retained under
`docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/`:
`original/`, `duplicate/`, both complete logs, 24 GPU JSON records, the duplicate
runner, scheduler record, source identity, and remote/local hash manifests.
Every downloaded raw file matches its remote SHA256.

- [Acceptance](semantic_recovery_evidence/fetch_family/gpu_acceptance/acceptance.json), SHA256 `6b037d62b08d14ec89a07fee36bdc672276f9b42d6d30e735c9dc43ef447329c`.
- [Original remote hashes](semantic_recovery_evidence/fetch_family/gpu_acceptance/original-remote-hashes.json).
- [Duplicate remote hashes](semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate-remote-hashes.json).
- [All family hashes](semantic_recovery_evidence/fetch_family/SHA256SUMS), manifest SHA256 `c18099ca28e322565a3faac80649c4d8caca33f586fdb43e0259e0072e876cd2`.
- Regression/readiness/taxonomy/promotion/legacy checks: **62 passed, 56 optional skips**. Final metadata readiness/promotion rerun: **29 passed**. Independent diff, closure and hash checks PASS. No production diff from starting HEAD.

Commands (integration worktree):

```sh
.venv/bin/python -m pytest -q tests/unit/test_replacement_readiness.py tests/unit/test_full_state_contract.py tests/unit/test_semantic_harness.py tests/unit/test_variation_maps_fetch.py tests/unit/test_envs_fetch.py tests/unit/test_swm_matrix.py
.venv/bin/python -m pytest -q tests/unit/test_replacement_readiness.py tests/unit/test_swm_matrix.py
git diff --check
```

[Recovery report](fetch_family_recovery.md) retains all CPU State, source, RNG,
lifecycle and Variation details. [Final review](semantic_recovery_evidence/fetch_family/gpu_acceptance/final-review.json)
records scope and closure checks. The former CPU checkpoint remains a historical
record of the queued gate; this report closes that blocker.

## Newly promoted

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

## Remaining incomplete

- `swm/CraftaxClassicPixels-v1`
- `swm/CraftaxClassicSymbolic-v1`
- `swm/CraftaxPixels-v1`
- `swm/CraftaxSymbolic-v1`
- `swm/OGBCube-v0`
- `swm/OGBScene-v0`
- `swm/PFRocketLanding-v0`
- `swm/PushT-Discrete-v1`
- `swm/PushT-v1`

## Changed files

- `docs/research/fetch_family_gpu_checkpoint.md`
- `docs/research/fetch_family_recovery.md`
- `docs/research/semantic_recovery_evidence/fetch_family/SHA256SUMS`
- `docs/research/semantic_recovery_evidence/fetch_family/coverage.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/acceptance.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate-remote-hashes.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/fetch-family-0dd0271f-PickAndPlace-baseline.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/fetch-family-0dd0271f-PickAndPlace-candidate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/fetch-family-0dd0271f-PickAndPlace-gate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/fetch-family-0dd0271f-Push-baseline.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/fetch-family-0dd0271f-Push-candidate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/fetch-family-0dd0271f-Push-gate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/fetch-family-0dd0271f-Reach-baseline.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/fetch-family-0dd0271f-Reach-candidate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/fetch-family-0dd0271f-Reach-gate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/fetch-family-0dd0271f-Slide-baseline.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/fetch-family-0dd0271f-Slide-candidate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/fetch-family-0dd0271f-Slide-gate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/fetch-family-6652487.out`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/duplicate/oscar_fetch_family_normgpu-0dd0271f.sbatch`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/final-readiness.out`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/final-review.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original-remote-hashes.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original/fetch-family-0dd0271f-PickAndPlace-baseline.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original/fetch-family-0dd0271f-PickAndPlace-candidate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original/fetch-family-0dd0271f-PickAndPlace-gate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original/fetch-family-0dd0271f-Push-baseline.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original/fetch-family-0dd0271f-Push-candidate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original/fetch-family-0dd0271f-Push-gate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original/fetch-family-0dd0271f-Reach-baseline.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original/fetch-family-0dd0271f-Reach-candidate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original/fetch-family-0dd0271f-Reach-gate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original/fetch-family-0dd0271f-Slide-baseline.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original/fetch-family-0dd0271f-Slide-candidate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original/fetch-family-0dd0271f-Slide-gate.json`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/original/fetch-family-6649686.out`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/regression.out`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/scheduler.txt`
- `docs/research/semantic_recovery_evidence/fetch_family/gpu_acceptance/source-identity.json`
- `docs/research/semantic_recovery_implementation_map.json`
- `docs/research/semantic_recovery_matrix.json`
