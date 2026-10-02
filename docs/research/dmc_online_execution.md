# DMC online execution audit

Starting checkpoint: `ce3ec1ffd01698409fe14a7c6dc88a78a207e9a0`, clean
`repair/render-config-fixes` worktree. Fetch's accepted process/pipeline evidence
is preserved. The semantic matrix remains 39 VERIFIED / 3 INCOMPLETE; this work
does not promote any registration. Craftax is deferred, and PushT/Rocket retain
their correctness prerequisites.

## Scope and routing

The non-Craftax routing is 12 completed Fetch registrations, 12 DMC and three
OGBench scalar CPU registrations to evaluate with persistent process owners,
five cheap classic-control registrations whose process overhead must first be
measured, and three natively batched Torch Navigation registrations. The three
remaining non-Craftax registrations are correctness-blocked. A process pool is
not the intended default for native GPU batching or for tasks where it loses.

DMC is the first cohort after Fetch. Its accepted public classes map to five
adapter implementations:

| Historical registrations | Public class / adapter route |
| --- | --- |
| Acrobot, Pendulum, Cartpole DMControl | `DMCAcrobotBalance`, `DMCPendulumSwingup`, `DMCCartpoleSwingup` → `FirstFamilyAdapter` |
| Reacher DMControl | `DMCReacherHard` → `ReacherAdapter` |
| Cheetah, Hopper, Walker DMControl | `DMCCheetahRun`, `DMCHopperHop`, `DMCWalkerWalk` → `DMCContinuationAdapter` |
| BallInCup, Finger, Manipulator, Quadruped DMControl | `DMCBallInCupCatch`, `DMCFingerTurnHard`, `DMCManipulatorBringBall`, `DMCQuadrupedWalk` → `DMCAlignedAdapter` |
| Humanoid DMControl | `DMCHumanoidWalk` → `DMCHumanoidAdapter` |

The public `step` shape/slice dispatcher calls each semantics mixin's `_step`,
which delegates to its adapter. Each lane calls the accepted dm_control task
twice per public action, retaining task-specific control substeps, rewards,
observations and numeric info. State export follows those calls. CUDA selects
output/render placement; these simulators execute on CPU. Raw `mj_step` or
MuJoCo rollout would omit task logic and is not a drop-in replacement.

The original backlog's suggestion to fix DMC mixed-terminal masks was not
supported by this audit: all accepted adapters expose false `terminated` and
`truncated` flags, including internal dm_control episode-reset boundaries.
The wrapper's reset-next-step and counter are part of State. Preserve those
source semantics; do not invent new public termination behavior. Some source
tasks intentionally expose `success=NaN`; equivalence requires matching NaN
positions and exact equality for finite values.

## First measured bottleneck and candidate

The frozen baseline builds a TensorDict per lane, with one tensor conversion
and clone per field, then stacks lanes. CUDA output adds per-lane, per-field
transfers. The first local Acrobot B64 component diagnostic measured 3.19 ms
State export within a 5.69 ms adapter step; its separate cProfile identified
TensorDict construction/stacking as substantial costs.

The candidate `dmcontrol_state.export_state` allocates fresh typed NumPy
columns, copies each owner's physical/task/RNG/lifecycle fields into them, and
creates/transfers one tensor per complete column. Each returned State owns its
storage across future exports, steps and resets. Reacher's target, Finger's
target site, Manipulator's target body pose, and Quadruped's actuator activation
remain included. No State schema, physics, restore algorithm, public API, or
renderer was changed. Aligned adapters now also reject export after close.

This removes duplication across the five exporters, but does not yet add a
process executor. Process sizing/default selection requires subsequent paired
measurements after this overhead is removed.

## Evidence and acceptance in progress

`dmc_profile.py` measures synchronized adapter step, export and restore,
then records cProfile separately from throughput measurements. It is explicitly
a component diagnostic, not an upstream speedup or public pipeline result.

`dmc_state_export.py` loads the legacy `get_state` methods from the frozen
baseline archive. It compares complete public outputs for one/eight actions,
independent output slices, reverse/empty selections, mixed internal reset
boundaries, cached Gaussian RNG state, restoration, and ownership after later
step/reset calls. Frames compare exactly against the accepted legacy path;
this is not a fresh source-renderer MSE certification. Paired timings alternate
legacy/candidate order on the same host, restore outside timing, synchronize
CUDA, retain all selected States, and report six raw samples plus mean/SD.

Local adapter tests pass 26 CPU cases. The same suite passes all 38 CPU/CUDA
cases on Oscar, including each of the twelve registrations on CUDA.
All twelve registrations pass the local paired State/task/RNG/lifecycle/ownership
gates for one/eight actions. Three also run frame comparisons using Apple GL;
CPU software-GL and NVIDIA GPU frame gates remain separate Oscar work.

The [local CSV](semantic_recovery_evidence/dmc_online_20261002/local/comparison.csv)
contains 192 aggregate timing cells. At B64/T8, frames disabled, measured public
steps/s are:

| Domain | Legacy steps/s ± SD | Column steps/s ± SD | Ratio |
| --- | ---: | ---: | ---: |
| acrobot | 10,974 ± 53 | 18,527 ± 144 | 1.69× |
| ball_in_cup | 8,430 ± 256 | 12,441 ± 12 | 1.48× |
| cartpole | 11,478 ± 72 | 19,474 ± 135 | 1.70× |
| cheetah | 11,335 ± 318 | 20,416 ± 112 | 1.80× |
| finger | 7,814 ± 257 | 12,047 ± 27 | 1.54× |
| hopper | 9,955 ± 167 | 15,628 ± 522 | 1.57× |
| humanoid | 4,342 ± 59 | 5,194 ± 34 | 1.20× |
| manipulator | 4,006 ± 640 | 5,358 ± 18 | 1.34× |
| pendulum | 13,023 ± 97 | 25,065 ± 502 | 1.92× |
| quadruped | 5,127 ± 69 | 6,473 ± 81 | 1.26× |
| reacher | 11,032 ± 326 | 19,127 ± 1,043 | 1.73× |
| walker | 6,423 ± 24 | 8,253 ± 225 | 1.28× |

These are Mac CPU-output results. They do not establish cluster throughput,
multiprocessing gains, or renderer performance.


Oscar job bookkeeping and source manifests are in
[the checkpoint](semantic_recovery_evidence/dmc_online_20261002/checkpoint.json).
The isolated Acrobot smoke (`6961313`) completed with 38 adapter tests passing,
ten exact public State/task/frame cases per device, and 48 aggregate timing
cells. Its CPU GL proof is Mesa llvmpipe; its GPU is NVIDIA RTX A6000, not a
3090. Both frame gates have MSE 0 versus the legacy library path. Raw logs retain
EGL destructor warnings during interpreter shutdown; both application reports
are PASS and the job exited 0. The 3090 baseline array `6960514` and final
all-registration candidate array `6961314` are queued. Full acceptance is pending.

 The first attempted array used the Fetch-specific
MuJoCo 3.11 interpreter with dm-control 1.0.45 and failed before simulation due
to the missing `light_softness` field. Remaining tasks were canceled. Separate
baseline/candidate arrays use the matching MuJoCo 3.12 / dm-control 1.0.45
environment; this version choice is shared by both controls and does not
rewrite historical acceptance evidence.

## Remaining execution work

After exporter acceptance, profile the remaining task/export/restore costs and
evaluate bounded persistent pure-process shards at equal CPU allocations. Reuse
the one-transition lookahead protocol and owned snapshots already used by Fetch.
The common scheduler/transport should handle nested task columns without
duplicating a whole executor for DMC. Family hooks retain construction,
configuration, reset, State validation and source lifecycle behavior.

Every registration still needs independent process correctness and public API
throughput gates, selected-frame pipeline timings, same-device source baselines,
and batch/horizon sweeps before changing its default execution route. The full
35-registration goal remains active.
