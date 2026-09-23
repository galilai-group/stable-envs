# Oscar acceptance checkpoint B

Observed 2026-09-23 00:16:40 UTC. Integration base:
`8f0bbc22959c239f8274f4d93ace93ce34971095` on `repair/environment-semantics`.
Pinned SWM source: `176d54532b3c0a251a4f509b74f2fdacc143a63d`.

The inventory is now **10 VERIFIED / 32 INCOMPLETE**, previously 5 / 37.
This meets the requested checkpoint B. PushT and Humanoid have exploratory
findings below; neither has completed CPU recovery or been promoted.
The [current matrix](semantic_recovery_matrix.json) supersedes the queue status
in earlier checkpoint documents, which remain historical records.

## Accepted bounded slices

| Oscar job | Newly VERIFIED historical registrations | Immutable source commit |
| --- | --- | --- |
| 6605941 | `swm/CartPoleControl-v1`, `swm/PendulumControl-v1` | `446f8608bc800c2d61701f675a2536df0dcbaa3f` |
| 6605999 | `swm/CheetahDMControl-v0`, `swm/HopperDMControl-v0`, `swm/WalkerDMControl-v0` | `6470521e84a47d5f95a5bd5f0c5a338ab293619f` |

Both actual logs contain their expected `FAMILY_GPU_GATE_PASS <stage>` terminal
marker. Their semantic JSON, baseline and candidate renderer JSON, and required
regressions were inspected. Remote source archive hashes were rechecked.
[Acceptance manifest and artifact hashes](semantic_recovery_evidence/oscar_backlog_20260923/acceptance.json)
record these checks; the same directory preserves the raw logs and JSON.

- Classic: 16 cases, including 12 domain/seed/batch trajectory cells and four
  lifecycle cells. DMC: 30 cases, including 18 trajectory cells, six positive
  reward cells and six lifecycle cells. Reported State/output parity errors are
  **zero**, with seeds 0/17/42, batches 1/3, 32-step trajectories and eight-step
  continuation after different histories.
- The required Reacher regression has 10 cases and first-family DMC regression
  has 30 cases, with all reported parity errors zero. Phase 0.5/Reacher acceptance
  remains intact.
- Classic gates cover unseeded reset replay, typed actions, reward boundaries,
  500/200-step TimeLimits and nondefault physics. DMC covers implicit-reset RNG
  replay, positive reward, nondefault gravity, and visual-only default
  preservation. Cheetah runs through public step 1252; Hopper/Walker through 502.
- DMC warmstart omission produces nonzero observable drift (Cheetah
  `1.4654943925052066e-14`, Hopper `2.042810365310288e-14`, Walker
  `2.2493118478905672e-13`); restoring it gives exact continuation. The Oscar
  construction census compares 583 numeric model arrays exactly.
- Five frozen renderer cells match baseline **byte for byte**, four frozen rows
  per accepted class. Previous CPU preservation evidence and new CUDA
  preservation evidence are comparisons within each device profile; they are
  not a claim that CPU frames equal CUDA frames. Full-State render projection,
  non-mutation and lossy restore refusal remain part of the bounded gates.

These five implementations remain `DELEGATED_CPU`: CUDA State and rendering
acceptance does not mean GPU physics. Certification covers each recorded
historical default and tested Variation subset, not all sibling classes,
reset options or the complete historical Variation surface.

Oscar semantic profile: Linux x86_64, Torch 2.14.0+cu126, NumPy 2.5.3,
Gymnasium 1.3.0, TensorDict 0.14.2, dm-control 1.0.44 and MuJoCo 3.11.0.
The source pin, rather than the installed SWM distribution version, identifies
the reference.

## CPU-passed registrations whose Oscar acceptance failed

All ten retain `INCOMPLETE` and `CPU_PASSED_OSCAR_PENDING`. Here “pending” means
acceptance is outstanding; **these jobs are finished failures, not queued jobs**.

| Job | Registrations | Observed blocker |
| --- | --- | --- |
| 6608718 | `swm/AcrobotControl-v1`, `swm/MountainCarControl-v0`, `swm/MountainCarContinuousControl-v0` | Gate diagnostic subtracts CUDA candidate and CPU reference tensors. |
| 6610032 | `swm/BallInCupDMControl-v0`, `swm/FingerDMControl-v0`, `swm/ManipulatorDMControl-v0`, `swm/QuadrupedDMControl-v0` | Omission assertion fails without identifying domain, field or measured errors. |
| 6611139 | `swm/TwoRoom-v1`, `swm/Piecewise-v0`, `swm/SimplePointMaze-v0` | Exact `info.distance_to_target` mismatch, absolute error `7.62939453125e-06`. |

All three completed baseline rendering but failed before a completed semantic
JSON and candidate frozen-render comparison. Printed baseline hashes do not
establish candidate renderer preservation. CPU preservation remains recorded;
CUDA preservation and the remaining semantic checks are unaccepted.

Concrete next diagnostics, without changing shared contracts or tolerances:

1. Classic: fix both device-mismatched diagnostic comparisons in
   `_continuous_dtype_flag_necessity` by comparing on the host. This is a gate
   defect, not evidence that production dynamics failed. A new reviewed revision
   and complete rerun are required.
2. DMC: report domain/key and `error_without`/`error_with` in
   `_omission_experiments`. The existing log cannot distinguish a zero omission
   witness from a faulty restore. A profile-dependent witness is only a
   hypothesis: the local Ball-in-Cup witness was near machine precision, and
   local ARM/MuJoCo 3.12 differs from Oscar x86/MuJoCo 3.11. If restored error is
   nonzero, investigate the family restore; if both errors are zero, establish a
   stronger deterministic witness. Do not weaken tolerances or remove fields
   based on this assertion alone.
3. Navigation: the failing step passed State comparison before the numeric
   output assertion. Source inspection identifies CUDA `torch.linalg.vector_norm`
   versus the historical per-environment CPU `torch.norm` distance path as the
   likely cause. A family-local CPU norm with source ordering is a proposed fix,
   not a proven result. Verify distance and threshold termination on Oscar.
   Later cases, including the SimplePointMaze Agg profile, remain untested by
   this failed run.

Failed immutable source commits:
`b3b6da79555020a3fe9ae046c3f6b4a2bef8a153`,
`e4d5c2cd99d035fc57d84dd5eab1ea70e9b26941`, and
`9fed5402631660c0764229182ed2a71c9a7a8e83`, respectively.
No archives were modified and no jobs were requeued at this checkpoint.

## PushT findings: exploratory, not certified

[Probe scripts, output provenance and dependency profile](semantic_recovery_evidence/checkpoint_b_exploratory/pusht/profile.txt)
are preserved with checksums. Stdout files transcribe observed tool output; the
profile distinguishes original standalone scripts from the saved inline body.
These probes were not rerun while packaging the evidence.

The pinned source registers **`swm/PushT-v1` and `swm/PushT-Discrete-v1`**.
The v0 spellings in the request are not registrations at this pin. The historical
inventory is preserved, and continuous/discrete certification will be separate.

Continuous actions use a target fixed at public-step entry:
`agent.position + 100 * action`. The source runs ten 0.01-second PD-controlled
substeps (`kp=100`, `kv=20`). Although the advertised space is float32 Box[-1,1],
the source does not validate or clip the action; the observed input [2,0] is
accepted. Discrete actions are two 16-way polar bins:
`r=(rbin+0.5)/16`, `theta=(tbin+0.5)*2*pi/16`, then float32 Cartesian decoding
into that relative controller. Bin [0,0] is not a zero-motion action.

Reset samples agent position in [50,450]^2, block position in [100,400]^2 and
block angle in [-2*pi,2*pi]. A second draw supplies a seven-dimensional goal.
Goal installation and actual-state installation each advance physics once.
The visible seven-vector contains agent position, block position, wrapped block
angle and agent velocity. Reward is minus the norm of this vector to the sampled
goal. Success tests the first four coordinates against distance 20 and the
source's raw-goal/wrapped-current angular expression against pi/9. Preserve its
angle quirks. The fixed renderer `goal_pose` is distinct from the sampled reward
goal. There is no registered TimeLimit: truncated is false and post-termination
steps remain allowed. Step itself draws no RNG; unseeded reset uses fresh
external entropy through the Variation tree.

Exploratory contact histories show that body fields alone are insufficient:
zeroing agent velocity, block velocity or block angular velocity changes future
outputs. Restoring accessible body poses/velocities/forces/torques and goal data
into a different-history simulator still changes continuation. A controlled
wall-contact probe with matching accessible pre-state produces a next block-y
difference of about **0.552379** and reward difference about **0.259348**.
This establishes a missing engine-state dependency; it does **not** isolate
contact impulses, body bias, broadphase bookkeeping or another specific field.
Pymunk exposes contact/arbiter internals that can be investigated. No final
State schema or opaque serialization has been introduced. Default bodies were
awake with infinite sleep threshold; sufficiency outside that observation is
not yet proved. Cached goal output also requires explicit investigation because
reset advances physics around its capture.

Variation, lifecycle, full-State restoration, both action gates and renderer
preservation still require complete candidate acceptance, including Oscar.

## Humanoid findings: historical Walk, exploratory State evidence

[Probe provenance and dependency profile](semantic_recovery_evidence/checkpoint_b_exploratory/humanoid/humanoid_audit_manifest.json)
accompany five scripts and observed-result JSON files. These are transcriptions
of executed inline bodies and tool outputs; original standalone files did not
exist. They were not rerun during packaging and are exploratory evidence only.

`swm/HumanoidDMControl-v0` has empty registration kwargs, and its wrapper defaults
to **walk**, with `move_speed=1`, `pure_state=False`, 25-second horizon, control
interval 0.025, five physics substeps, and action repeat two with reward summation.
Actions are 21 float64 normalized controls. Observation has 67 float32 entries.
The model has nq=28, nv=27, nu=21, na=0, with no mocap, user or plugin State.

The existing `DMCHumanoidStand` uses stand/move_speed=0. At seed 0, Walk and Stand
have identical reset and next zero-action physics/observation, but public-step
reward differs: **0.14940584923773587** versus **0.8811451420084**. Stand therefore
cannot be certified as this historical registration. A future public Walk class
must use the established adapter while preserving historical Stand renderer
evidence privately.

An exploratory restored-source probe (seed 0, one prefix step, nine steps of
another history, eight continuation steps) restored qpos/qvel/time/lifecycle,
cumulative reward and MT19937 state. Omitting warmstart produced maximum
observable error **5.861977570020827e-13**; restoring it produced zero error.
Different old controls did not affect the next overwritten action. This is
source-simulator evidence guiding implementation, not a completed candidate
State gate or a proof that every other omitted field is unnecessary.

The source implicitly resets after DMC step 1000: public steps 499/500/501/502
leave DMC counts 998/1000/1/3. The reset occurs within action repeat, RNG changes,
public terminated/truncated remain false, and cumulative reward keeps
accumulating. A 585-array numeric model census against stock Walk/Stand found
only geom_friction differences after source reset, from float32 MJCF rounding
of secondary floor friction. Source defaults and model construction must be
matched exactly, not replaced with nominally identical decimal defaults.

The full omission matrix, seed/batch/reset/Variation coverage, candidate gates,
renderer preservation and Oscar acceptance remain outstanding.

## Change and validation boundary

This checkpoint changes research evidence and matrix acceptance only. The shared
State/RenderState/SimulatorAdapter/parity/taxonomy, production source and frozen
fixtures are unchanged. Fetch, OGBench manipulation, Craftax, Rocket and final
paired benchmarks have not started.

The Phase 0 audit hash remains
`f27769ec10d843145c5642d9e98ee12d679a9f01206ad1e29c7c24c69ca61c0a`.
`baseline/renderer-v0` remains
`6d80b68b8b3ccd07d4207be94cbf4f0760758a06`; product `v0-draft` remains
`9471a619e08aedb0b53a90b901c9e26ffff57c11`.
The local readiness suite passed **8 tests**. Artifact checksums, matrix counts,
accepted/failed membership and baseline/candidate JSON equality were checked.
