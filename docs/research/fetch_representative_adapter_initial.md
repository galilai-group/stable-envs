# Representative Fetch: initial private adapter

The Navigation Oscar completion reached the user's independent stopping
checkpoint while this adapter was being proved. This commit contains a working,
independently tested private adapter, not a completed representative Fetch CPU
gate. No public Fetch registration or renderer is changed. Fetch remains
INCOMPLETE; the twelve registrations are not generalized or promoted.

`stable_envs/_adapters/fetch_pickandplace.py` constructs one independent
Gymnasium Robotics `MujocoFetchPickAndPlaceEnv(reward_type='sparse')` with a
50-step Gymnasium TimeLimit per lane. Production imports no stable-worldmodel.
Execution is `DELEGATED_CPU`; the device only controls tensors returned and
accepted by the adapter. Tests ran on CPU in the pinned authority profile:
Python 3.12.11 / NumPy 2.2.6 / Gymnasium 1.2.3 / Robotics 1.4.2 / MuJoCo 3.8.1.

## Provisional explicit State

The following fields are implemented and tested for the bounded default task.
This is not yet the final family schema approved by the full CPU gate.

| Field | Width / dtype | Evidence |
| --- | --- | --- |
| qpos | 22 float64 | Source omission changes observation |
| qvel | 21 float64 | Source omission changes observation |
| warmstart | 21 float64 | Source omission changes public output even at very small absolute scale |
| controller.position | 3 float64 | Explicit cached gripper body position, independently necessary after mj_forward |
| controller.quaternion | 4 float64 | Explicit cached gripper body quaternion, independently necessary after mj_forward |
| goal | 3 float64 | Public goal/observation and reward task state |
| lifecycle.elapsed_steps | 1 int64 | Restoring at step 49 controls 50/51 truncation |
| rng.state_words | 4 int64 | Explicit little-endian 32-bit words encoding PCG64's 128-bit state |
| rng.increment_words | 4 int64 | Explicit little-endian 32-bit words encoding PCG64's 128-bit increment |
| rng.has_uint32 | 1 int64 | Explicit PCG64 cached-integer flag |
| rng.uinteger | 1 int64 | Explicit PCG64 cached-integer value |

The PCG64 fields represent named generator state, not an opaque serialized
object. Simulator RNG omission is independently witnessed by an unseeded reset;
individual cached-integer components have not each been shown necessary for this
uniform-sampling-only task. Their necessity/minimality remains a schema review
item. The adapter retains them for lossless named PCG64 restoration.

Mocap and control values are overwritten by the public controller and omitted.
Applied forces remain zero under the registration's supported operations and
are reset to zero on restoration. The simulator clock has no independent public
output witness and is omitted; elapsed public steps are retained separately.
The bounded adapter tests match across different simulator-clock histories.
None of these exclusions generalizes to external force injection, other Fetch
tasks or untested physical Variation.

## Passing adapter-only checks

The independent probe uses the pinned SWM historical registration as oracle and
extracts source State without importing candidate extraction helpers. It
asserts the pinned SWM revision and tracked-clean source, compares every numeric
output with exact value, dtype and shape equality, and compares all candidate
State fields exactly. Visual variations are explicitly disabled in this probe.

- Twelve cases: seeds 0/17/42, batches 1/3, action dtype float32/float64.
- Exact seeded reset State; independent snapshot mutation check.
- Each case: 32 action steps, then restore candidate-only into an adapter with
  11 different-history steps and continue for eight source-matched steps.
- Inputs include exact -1/+1, simultaneous XYZ/gripper components and random
  out-of-range [-2,2] values. The adapter preserves dtype and delegates clipping,
  scaling and order to upstream without normalization.
- Exact full State roundtrip, repeated seeded reset and unseeded reset RNG replay.
- Exact reward/success values on these trajectories and lifecycle at 49/50/51,
  including restoring step 49 into a different-history adapter. Stepping after
  truncation is allowed and continues to match source.
- Compact render row restoration is refused.

Complete raw results and stderr are under
[`semantic_recovery_evidence/fetch_adapter_initial`](semantic_recovery_evidence/fetch_adapter_initial/).
The executable probe is `scripts/recovery/fetch_pickandplace_adapter_probe.py`.
Source necessity experiments remain separately recorded in
[`fetch_representative_state_experiments.md`](fetch_representative_state_experiments.md).

## Outstanding acceptance work

Public environment integration, candidate reward/success threshold tests,
meaningful physical Variation, visual-only Variation, Variation-space RNG
continuation, full-State/private-render equality, render non-mutation and frozen
renderer preservation remain outstanding. The private mass configuration method
is preparatory and has not yet passed a Variation gate; it is not claimed as
accepted behavior.

The existing Fetch renderer has no CPU mover implementation: its tests explicitly
identify `CPU_BACKEND_MISSING`, its module requires MuJoCo >=3.11/MJWarp, and its
mesh mover pass invokes MJWarp. The authority simulator uses MuJoCo 3.8.1.
The CPU plate alone cannot satisfy the frozen full-frame gate. No dependency
patch, replacement renderer, process abstraction or new shared contract was
introduced. This is an existing rendering/platform limitation, not evidence
that the shared State contract is insufficient.

PushT remains parked. No other simulator family, shared contract or promotion
rule is changed.
