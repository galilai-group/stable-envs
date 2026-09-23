# Post-C acceptance and representative Fetch checkpoint

Checkpoint **A** reached: the classic and source-aligned DMC Oscar jobs passed
independently. The matrix is now **17 VERIFIED / 25 INCOMPLETE** (previously
10 / 32). Four CPU-passed rows still require Oscar acceptance after failed jobs;
there are no replacement jobs queued by this checkpoint. No shared contract,
production implementation, parity tolerance, or promotion rule changed here.

## Oscar outcomes

| Slice | Job | Result | Promotion |
| --- | --- | --- | --- |
| Remaining classic | 6626587 | PASS | AcrobotControl-v1, MountainCarControl-v0, MountainCarContinuousControl-v0 |
| Source-aligned DMC | 6626669 | PASS | BallInCupDMControl-v0, FingerDMControl-v0, ManipulatorDMControl-v0, QuadrupedDMControl-v0 |
| Native navigation | 6626600 | FAILED | None |
| Humanoid Walk | 6626742 | FAILED in gate diagnostic | None |

All historical names in this table have the `swm/` prefix. Promotion applies only
to their tested historical defaults and recorded bounded Variation scopes.
Sibling tasks and the complete historical Variation surface remain uncertified.
Both accepted slices use **DELEGATED_CPU** physics with CUDA State/action and
renderer coverage; these are not GPU physics or performance certifications.

Classic source `98c65cbe6690404bf04a463e72003ba924b6c989` completed 24
semantic cases (18 trajectories and six boundary/lifecycle cases), plus the
16-case CartPole/Pendulum regression. Source-aligned DMC source
`57c10c17f1bf74ce565ff80e5f44292e3fc827ad` completed 32 semantic cases
(24 trajectories and eight lifecycle cases), plus Reacher (10), first-family
DMC (30), and continuation DMC (30) regression cases. Reported parity errors are
zero. The trajectory grids use seeds 0/17/42, batches 1/3, 32 steps and eight
steps after different-history restoration. DMC lifecycle cases cover 502 public
steps, changed RNG on implicit reset and exact replay.

Both real terminal logs contain their expected final and frozen-renderer pass
markers. Baseline and candidate frozen-renderer JSON files are byte-identical:
three classic and four DMC cells, four frozen rows per class, compared within
each recorded CUDA profile. This does not imply equality across CPU/CUDA or
across dependency versions. Required regression JSON is present and complete.
All eight aligned-DMC omission experiments retain positive observable divergence
when a field is omitted and zero error with the complete schema. The Ball-in-Cup
prefix-six warmstart witness is 2.942091015256665e-15 on Oscar; no tolerance was
introduced. Each aligned-DMC physical Variation has nonzero divergence from the
default and zero error against its varied reference. Visual-only preservation
also passes. Finger's NaN success entry remains the established unavailable
placeholder; its positive sparse reward is 2.0, not a finite success claim.

The [acceptance manifest](semantic_recovery_evidence/post_c_acceptance/acceptance.json)
records immutable source/archive/runner/renderer hashes, package profiles,
renderer hashes, raw terminal logs and JSON checksums. Original evidence remains
unchanged, including superseded failures and checkpoint C's queued-job snapshot.

Navigation failed `compare_state` in `_boundary_case` on
`info.distance_to_target`: absolute difference **0.7803192138671875**, relative
**0.004922586150551877**. The terminal trace does not label the specific domain
or probe. Root cause is not established. No completed semantic JSON, candidate
frozen-renderer JSON or final pass marker was produced; all three navigation
registrations remain INCOMPLETE. Their CPU acceptance is retained.

Humanoid failed in `dmc_humanoid_gate._physical_error`: the physical Variation
diagnostic directly subtracts CUDA candidate and CPU nominal reward tensors.
This identifies a gate diagnostic device bug, not a demonstrated production
semantic mismatch. There is only baseline renderer evidence, no completed
semantic JSON, no candidate frozen comparison and no completed required
regressions for this job. **Humanoid remains CPU complete, INCOMPLETE, and
unaccepted on Oscar.** No fix or requeue was made before returning checkpoint A.

## Fetch start

The new bounded branch `repair/fetch-representative` at
`/private/tmp/stable-envs-fetch-recovery` starts from reviewed integration
`907c6e0f45f0763ac736cea006fb762c6e802ba2`. Its initial audit commit is
`cae0ffbe5682fb733ec24c3d4088d3fa5c1ce49a`, incorporated into integration as
`01149c100f50d4484680b0aa542d914f501975be` before this acceptance update.

Selected registration: **swm/FetchPickAndPlace-v3**, mapped by pinned SWM to
**FetchPickAndPlace-v4 / MujocoFetchPickAndPlaceEnv**, sparse reward,
`flatten=True`, object present and gripper unblocked. The authority is pinned
SWM, then Gymnasium Robotics/MuJoCo; no production SWM import was added.

Candidate investigation fields are qpos[22], qvel[21], time, mocap position and
quaternion, goal/task state, elapsed TimeLimit, warmstart[21], ctrl/applied forces
and upstream RNG. The compiled model has nu=2, na=0 and nmocap=1. These are
**candidates, not an established sufficient State schema**. Omission experiments
and different-history restoration have not passed or been used to select fields.

Source inspection finds four-dimensional actions clipped to [-1,1], XYZ scaled
by 0.05, fixed rotation [1,0,1,0], duplicated gripper command, actuator/mocap
application and 20 physics substeps. Float32/float64 arithmetic is not assumed
interchangeable. Source inspection also identifies object/goal randomization,
sparse reward `-(distance > 0.05)`, success `distance < 0.05`, false base done
flags and a 50-step TimeLimit. These are **source findings, not runtime parity**.
Reset/reward/done/TimeLimit/RNG replay, exact action parity, physical/visual
Variation, full-State rendering and frozen renderer preservation remain unrun.

The concrete blocker occurs before reset in the existing local profile:
Robotics 1.4.2, Gymnasium 1.3.0, MuJoCo 3.12.0, NumPy 2.5.3. Both SWM and direct
upstream constructors fail in `mujoco_utils.set_joint_qpos` for `robot0:slide0`.
An isolated probe shows asymmetric NumPy joint-type / MuJoCo-enum equality:
NumPy-scalar equality succeeds, reversed enum equality and enum tuple membership
fail despite equal integer value 2. No source or installed-library patch was
used. A compatible, explicitly recorded upstream profile is the next prerequisite.
This is not evidence of a shared State/SimulatorAdapter limitation.

The [initial Fetch audit](fetch_representative_initial_audit.md) records source
hashes, original probe scripts, stdout/stderr and the comparison diagnostic.
No Fetch production adapter, CPU acceptance gate or Oscar job was added.

## Parked and preserved work

Both historical PushT registrations remain INCOMPLETE with an explicit
supported-interface blocker: exact arbitrary restoration requires engine state
not exposed by the supported Pymunk interface. C3 evidence is preserved. This is
not an impossibility claim. No private ABI/CFFI work, opaque engine serialization,
approximate restoration, tolerance change or replay-from-reset substitution was
introduced. Resuming private/native integration requires coordinator review.

OGBench manipulation, Craftax, Rocket and final paired benchmarking remain
unstarted. Phase 0.5/Reacher acceptance is preserved and Reacher passed again in
the accepted DMC job. Product `v0-draft` remains
`9471a619e08aedb0b53a90b901c9e26ffff57c11`; frozen renderer baseline remains
`6d80b68b8b3ccd07d4207be94cbf4f0760758a06`. The Phase 0 audit SHA-256 remains
`f27769ec10d843145c5642d9e98ee12d679a9f01206ad1e29c7c24c69ca61c0a`.

Validation for this documentation/evidence update: all 42 historical IDs retained,
exactly seven status promotions, complete artifact checksums, preserved C3 and
shared/source/fixture files, and eight replacement-readiness unit tests passing.
The containing acceptance commit is the final integration checkpoint commit.
