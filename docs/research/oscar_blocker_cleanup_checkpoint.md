# Oscar blocker cleanup and Fetch authority checkpoint

Checkpoints **A and B** reached: Humanoid's complete Oscar rerun passes, and
Navigation's exact failing registration/probe and arithmetic cause are isolated.
The matrix is now **18 VERIFIED / 24 INCOMPLETE**. All twelve historical DMC
default registrations are accepted within their recorded bounded scopes. The
three Navigation registrations remain CPU-passed and unaccepted on Oscar.

## Humanoid acceptance

Job **6628919**, immutable source
`07d709841ea2e57fae334e7ef541559a6617236a`, changed only the family gate's
physical-Variation diagnostic: reward and observation operands are detached and
copied to CPU before subtraction. The new runner records hashes of the gate,
renderer driver and submitted runner. Production Humanoid, State, Variation,
fixtures, shared contracts and tolerances were unchanged.

The complete local CPU gate passed. Its first sandboxed attempt was blocked by
macOS CoreGraphics access; the same command outside that sandbox passed.
The Oscar rerun used immutable archive SHA-256
`d5f510f0e8168570b81ecfb67584f2e8f84975b72e99e42a4855f05c34230d7d`,
runner `oscar_dmc_humanoid_gate-c7b5299.sbatch` and unchanged immutable driver
`family_renderer_boundary-12c866f.py`. Remote hashes were rechecked before launch.

The actual log contains both expected pass markers. Complete semantic JSON has
ten cases: six trajectories at seeds 0/17/42 and batches 1/3, two positive-reward
cases and two lifecycle cases. Trajectories run 32 steps plus eight steps after
different-history restoration; State/output errors are zero. Lifecycle replay
covers public steps 499–502 and DMC implicit reset. Warmstart omission produces
observable error 2.5192292696374352e-11; full restoration gives zero error.
Meaningful gravity Variation matches the varied reference exactly and diverges
from default; visual-only preservation passes.

Candidate and baseline frozen-renderer JSON are byte-identical, with four private
Stand-corpus rows and frame SHA-256
`2c020229a8eb0c6748033f93fa4085b431308f7530f0784d89050d2ef67cc168`.
Public accepted identity is **Walk**. Required Reacher (10 cases), first-family
DMC (30), continuation DMC (30), and source-aligned DMC (32) regressions all have
complete JSON and zero reported parity error. All 155 reported error fields in
the gate and regressions were checked, excluding intentional omission witnesses.

Only `swm/HumanoidDMControl-v0` is newly promoted. Execution remains
**DELEGATED_CPU**, with CUDA State/action and renderer acceptance, not GPU physics
or benchmark certification. The remaining historical Variation surface and
nondefault reset/task options remain uncertified.

## Navigation root cause

Diagnostic revision `1f717d12847cbcb998ad6c0bcc7ce7386249e560`, job **6628945**,
reproduced the original 0.7803192138671875 distance error with unchanged production
semantics. The gate now observes reset, restore and step operations and emits
registration/domain, seed, batch/lanes, named probe, per-side step, action,
restore inputs, complete State, outputs, goals and dtype/device information.
Large image outputs retain metadata and checksums rather than giant pixel lists.
The shared parity harness was not changed. The full CPU gate still passes.

The earlier diagnostic-only job 6628935 encountered the original mismatch but
its reporter could not serialize StyleTensor. The reporter was corrected to
serialize each Variation leaf, then checked using an intentional local exception
before submitting the immutable replacement. Both logs are retained.

Exact failure:

| Property | Candidate | Reference |
| --- | --- | --- |
| Historical registration | swm/Piecewise-v0 | swm/Piecewise-v0 |
| Seed / batch / lane / step | 5 / 1 / 0 / 1 | 5 / 1 / 0 / 1 |
| Probe | zone_boundary_right | zone_boundary_right |
| Restored agent, float32 | [112, 50] | [112, 50] |
| Restored goal, float32 | [190, 190] | [190, 190] |
| Public action, float32 | [0, 0] | [0, 0] |
| Selected zone bias | [2, 0] | [0, 2] |
| Resulting agent | [114, 50] | [112, 52] |
| Distance | 159.2984619140625 | 158.5181427001953 |

This is a **zone-selection arithmetic error**, not goal restoration, goal-image
separation or the previous norm-ordering issue. Restored State and goals are
identical. The default Variation's zone biases explain the distinct next States.

The family expression computes
`((pos.double() - 14.0) / 196.0 * 2).trunc()` before selecting a bias.
Isolated Oscar job **6628994** confirms that at x=112, CUDA scalar-denominator
float64 division returns `0x1.fffffffffffffp-2`, immediately below 0.5. Multiplying
by two and truncating produces column 0. CPU and Python source division return
exactly 0.5 and column 1. On the same GPU, a float64 tensor denominator returns
exactly 0.5 and selects column 1; the immediate float32 neighbours also match
source indices. This probe changes no production code and is not a complete
corrected gate.

**Correction status:** production remains unchanged at this review checkpoint.
The arithmetic probe identifies a small family-local correction to validate:
use tensor-denominator division for this zone lookup. It still needs the complete
CPU gate, a new immutable revision and the complete Oscar Navigation semantic and
frozen-render gate. No Navigation promotion or corrected acceptance rerun is
claimed. No pass marker, complete semantic JSON or candidate renderer comparison
exists for diagnostic job 6628945.

## Fetch authority profile and remaining proof

The dependency blocker is resolved with a clean isolated profile:
**Python 3.12.11, NumPy 2.2.6, Gymnasium 1.2.3, Gymnasium Robotics 1.4.2,
MuJoCo 3.8.1**. The simulator versions come from pinned SWM's lockfile. Direct
upstream and pinned SWM both construct and reset unmodified at seeds 0/17/42;
installed Robotics utility source matches its wheel RECORD. No library patch or
source monkeypatch was used. The former failing profile is unchanged.

The representative remains `swm/FetchPickAndPlace-v3` mapped to
`FetchPickAndPlace-v4` / `MujocoFetchPickAndPlaceEnv`, sparse reward. Runtime
reference-only probes confirm clipped-equivalent out-of-range float64 actions,
reward/success threshold behavior, truncation at step 50 with continued stepping
at 51, 20 substeps, dt=0.04 and PCG64 RNG. These results do not certify a
stable-envs candidate. Full action ordering and float32/float64 behavior still
need the representative parity gate.

Candidate State fields remain hypotheses: qpos[22], qvel[21], time, mocap pose,
goal/task state, elapsed TimeLimit, warmstart[21], ctrl/applied forces and PCG64
state. Model na=0, so there is no activation array. **No final State schema,
omission witnesses, different-history result, candidate reset/reward/done/
TimeLimit/RNG parity or renderer preservation exists yet.** Physical/visual
Variation, render non-mutation and lossy-restore refusal are also outstanding.
Checkpoint A occurred before candidate implementation. No Fetch promotion or
sibling generalization is included.

The [Fetch profile report](fetch_authoritative_profile.md) includes exact
reproduction commands, complete package freeze, constructor/reset JSON and
reference-only runtime evidence. Its bounded worktree is
`/private/tmp/stable-envs-fetch-recovery`, branch `repair/fetch-representative`.

## Preservation and validation

Both PushT registrations remain parked with the supported-interface blocker and
all C3 evidence preserved. No private ABI work, opaque serialization, tolerance
change or replay-from-reset substitute was introduced. OGBench manipulation,
Craftax, Rocket and final paired benchmarking remain unstarted.

The [acceptance manifest](semantic_recovery_evidence/blocker_cleanup/acceptance.json)
and adjacent raw logs/JSON preserve job/revision hashes, the full Navigation
failure context and isolated arithmetic evidence. Existing evidence is retained.
Eight readiness tests pass. All 42 IDs, exactly one new promotion, twelve accepted
DMC defaults, unchanged production/shared files/fixtures, immutable audit hash and
unchanged product/baseline branch pointers were checked. The containing commit is
the final `repair/environment-semantics` integration checkpoint.
