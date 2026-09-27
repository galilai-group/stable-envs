# Rocket public renderer blocker checkpoint

**39 VERIFIED / 3 INCOMPLETE. No registrations promoted. No push.** The user’s
explicit Rocket renderer stop condition was reached. Randall’s matched renderer
table, PushT dynamic replay implementation and the 42/42 freeze were not started.

The sequence began from clean commit
`ccb997c51f7faa56036b0502c50f233647754f52`, which also contains the preceding
`rocket_recovery_checkpoint.md`. Branch: `repair/environment-semantics`.
Worktree: `/private/tmp/stable-envs-semantic-recovery`. The root checkout was
not changed. The closing commit containing this report is the ending checkpoint;
its exact SHA and clean status are recorded in the chat handoff.

## Decisive failure

The final candidate production is
`d2516cb67b0ef23b0dedc2f378d6e1c6a93f93d8`. Oscar job **6732132** ran on
**gpu1504**, NVIDIA **A40**, dual **AMD EPYC 7413** host. The job had four CPU
cores, 32 GB RAM, `gpu` partition / `norm-gpu` QOS, no requeue. Dynamics and the
authoritative TinyRenderer execute on CPU; the candidate renderer uses CUDA.
PyFlyt 0.29.0, PyBullet 3.2.7, NumPy 2.5.3, Gymnasium 1.3.0, Torch
2.14.0+cu126 and TensorDict 0.14.2 were recorded by the running process.

All **204 public semantic checkpoints** completed, including exact source
numeric diagnostics, immediate source pixels, independent and same-instance
restore, invalid-action/State rejection, full-State versus RenderState frames,
restore-frame equality, batch/singleton and chunk equality. All **41 visual
groups** were evaluated; **five groups failed** the frozen mean-absolute limit:

| Checkpoint | Frames | Frozen mean-absolute metric | Limit | Within 8 | SSIM |
|---|---:|---:|---:|---:|---:|
| Horizon 600 | 6 | 10.60954861 | ≤10.0 | 0.94942274 | 0.93893971 |
| Horizon 900 | 6 | 10.36119358 | ≤10.0 | 0.95060330 | 0.93911661 |
| Horizon 1201 | 6 | 10.88325521 | ≤10.0 | 0.94827691 | 0.93708959 |
| Horizon 1202 | 6 | 10.66644531 | ≤10.0 | 0.94932292 | 0.93703626 |
| Horizon 1204 | 6 | 10.66644531 | ≤10.0 | 0.94932292 | 0.93703626 |

Each group covers seeds 0/17/42 and both float32/float64. Within-eight and SSIM
pass their unchanged limits of ≥0.91 and ≥0.80. The policy’s `mean_abs` is the
mean per-pixel **maximum absolute RGB-channel difference**, not RGB-channel MAE
or MSE. The source-derived high-ceiling configuration (`ceiling=500000`,
`accelerate_drop=False`) keeps the rocket airborne to exercise its registered
1,202-action bound. These are source-supported diagnostic trajectories, not
ordinary default flights; default zero-control flight crashes around step 303.

The smallest retained decisive witness is one frame pair, index **19**:
seed 0, float32, **600 valid all-zero actions**, 480×480 RGB, no Variation.
It is the earliest failing **sampled** checkpoint, not a claim of the globally
minimal failure step. The source numeric State and immediate authoritative
pixels match exactly; the frozen candidate image exceeds the visual limit.
The witness is preserved without resizing, cropping, shifting or color changes.

- [Exact recipe and frame hashes](semantic_recovery_evidence/rocket_public_gpu_6732132/witness.json)
- [Authoritative frame](semantic_recovery_evidence/rocket_public_gpu_6732132/oracle-witness.png)
- [Candidate frame](semantic_recovery_evidence/rocket_public_gpu_6732132/ours-witness.png)
- [All frames and measurements](semantic_recovery_evidence/rocket_public_gpu_6732132/results/result.json)
- [Independent recomputation](semantic_recovery_evidence/rocket_public_gpu_6732132/verification.json)
- [Terminal accounting](semantic_recovery_evidence/rocket_public_gpu_6732132/sacct.txt)

Job 6732132 ended **FAILED, 2:0**, after **31:20**. Dependent performance job
**6732153** ended **CANCELLED**, elapsed **0:00**, no assigned node or start time.
It did not run. Conditional final-revision persistent-Variation and independent
source-suffix followups also did not run because the public gate exited 2.
Earlier local versions of those gates passed and remain revision-qualified;
they must not be represented as final-revision cluster acceptance.

The frozen renderer file is byte-identical to the starting checkpoint. No
threshold changed. The earlier 102-frame / 17-group preflight remains intact;
it did not include these extended high-ceiling checkpoints. The failure’s root
cause is not established by the pixel discrepancy alone. The next decision is
whether to authorize a separate Rocket-local renderer investigation under the
**existing** limits. This sequence performs no renderer repair or scope waiver.

## Completed action and State work

The normalized public action boundary accepts finite float32/float64 controls
within the source’s declared bounds, including exact boundaries. Fins/gimbals
use [-1,1]; ignition/throttle use [0,1]. Invalid shape, dtype, nonfinite or
out-of-range controls are rejected before entering PyFlyt. Whole-batch validation
precedes any lane mutation. Rejections do not advance physics, RNG, lifecycle,
replay history, caches or rendering, and subsequent valid continuation matches
an environment that never received the rejected call. The original source’s
partially mutating rejection witnesses remain preserved as provenance.

[Atomic-boundary evidence](semantic_recovery_evidence/rocket_atomic_boundary_v1/result.json)
covers **936 valid controls** and **864 rejected calls**, including exact and
adjacent boundaries, all 128 corners, repeated invalid calls and equal-count
cancellation witnesses. These are bounded source comparisons, not an exhaustive
enumeration of all finite controls. Source call-path inspection establishes the
prevalidation boundary; valid-action processing is unchanged.

The public candidate uses official unpatched PyFlyt/PyBullet and no production
stable-worldmodel imports. Its fixed numeric State has **187 leaves**, including
reset anchors, owned RNGs, persistent/episode Variation, action history and
original action dtype codes, lifecycle and native diagnostics. The 1,202-row
physical-action capacity excludes rejected calls; post-terminal calls have a
compressed count/control representation. Restoration stages independent engines,
replays real public steps, compares every supplied numeric leaf and swaps only
after all lanes succeed. Shared State/RenderState contracts were not changed.

At B1 / 480×480: **768,719 State bytes**, **825,183 uncompressed NPZ bytes**.
The retained named goal cache is **691,200 bytes**; no complete exact proof
justifies omitting it. Replay restore is O(H), bounded by 1,202 physical actions
for the registration. Seeded resets are exactly replayable, owned RNG is
restorable, and unseeded resets retain source fresh external entropy. The
candidate CPU renderer remains **UNSUPPORTED**.

The requested B1/B16 performance campaign did not execute. Correctness-run B1
restore durations remain in the raw report as diagnostics; they are not a
matched performance benchmark or a basis for speedup claims. No B16, serialized
latency, branch-cadence or warm render performance claims are made.

## Evidence and validation scope

- Final candidate Rocket unit tests: **18 passed**.
- Shared State/harness/readiness/repository/registry/renderer math checks:
  **95 passed, 2 skipped**. Combined current unit checks: **113 passed, 2 skipped**.
- Earlier revision CPU trajectory gate: **204 checkpoints passed**.
- Earlier revision independent-source continuation: **108 cases / 1,296 future
  steps passed**.
- Scalar-fix revision persistent Variation: **90 cases passed**, before the final
  external configuration sampler addition.
- Final candidate Oscar gate: **204 semantic checkpoints**, **36/41 visual groups
  passed**, overall **BLOCKED**. Independent frame/metric recomputation verified
  the same decision.
- Changed Rocket files pass Ruff and formatting; whitespace and strict documentation
  pass. Repository-wide checks expose **36 existing lint errors in two files**
  and **58 existing files requiring formatting**. Every affected file is
  byte-identical to the starting commit; no out-of-scope repairs were made.
  [Baseline comparison](semantic_recovery_evidence/rocket_public_blocker_checkpoint/preexisting_lint.json).
  The earlier
  checkpoint’s broader legacy run (253 passed / 61 skipped / 112 pre-existing
  failures) remains historical evidence; no complete broad-suite pass is claimed.

[Revision-qualified CPU evidence](rocket_public_cpu_evidence.md) distinguishes
these runs. Jobs **6731648**, **6731765** and **6731919** were deliberately
cancelled as superseded candidates; all logs/accounting remain preserved and
none counts as acceptance. Their respective elapsed times were 0:39, 14:08 and
20:44. No active recovery job remains.

## Deferred milestones

Randall’s complete per-registration table was not produced because the sequence
stopped at Rocket. Its requested normalized RGB definition is
`mean((I_stable/255 - I_oracle/255)^2)` over H×W×3. The one retained failing frame
has raw MSE **2213.12969473** and normalized MSE **0.03403505874**. Per-frame raw
and normalized MSE, RGB MAE and SSIM are retained in independent verification;
normalized mean/median/p95/max are also retained. These quality diagnostics are
not a substitute for the requested table across all accepted registrations.

No warm B1 timing table was run. The intended definitions remain full
State→RenderState→render for our CPU/GPU paths, including GPU transfer/sync,
and actual original CPU render at equal state/resolution. Hardware would be
matched on Oscar; Apple CPU timings would not be presented as direct GPU
speedups. Rocket’s CPU-render column would be UNSUPPORTED. Other registrations’
CPU support/noncomparability was not reassessed in this stopped sequence.

Neither PushT registration was modified or promoted. No dynamic shared State
extension, dynamic-history schema, O(T) performance measurements, or continuous/
discrete acceptance gates were implemented. The three remaining registrations
are `swm/PFRocketLanding-v0`, `swm/PushT-v1`, and `swm/PushT-Discrete-v1`.
No 42/42 closure or semantic-freeze claim is made.

## Preservation and implementation inventory

All **208 accepted artifact hashes**, **112 accepted production hashes**, prior
accepted manifests/terminal logs and historical renderer baselines remain
unchanged. The matrix and implementation map stay byte-identical. Every tracked
file is present. No PyFlyt/PyBullet package or private experimental engine was
modified. The checkpoint directory records repository growth and changed-file
hashes, separating production from evidence and research harness additions.

Production changes relative to the starting commit:

| File | Change |
|---|---|
| `stable_envs/_adapters/rocket_actions.py` | Prevalidate supported controls atomically |
| `stable_envs/_adapters/rocket_numeric.py` | Named numeric diagnostics |
| `stable_envs/_adapters/rocket_replay.py` | Complete bounded replay recipe and validation |
| `stable_envs/_adapters/rocket.py` | Batch lifecycle and atomic staging/swap |
| `stable_envs/envs/rocket_semantics.py` | Public candidate registration, State, Variation and render projection |
| `stable_envs/_adapters/rocket_task.py` | Add local action validator before authoritative step |
| `pyproject.toml` | Pin official Rocket runtime dependencies in its optional extra |

No shared-contract or shared-renderer implementation changed. The original
private task/space candidate and rejected-control evidence are preserved.
[Closing verification and growth](semantic_recovery_evidence/rocket_public_blocker_checkpoint/verification.json)
and [changed-file inventory](semantic_recovery_evidence/rocket_public_blocker_checkpoint/changed-files.json)
record the final scope. Commit history is preserved alongside them. No push.
