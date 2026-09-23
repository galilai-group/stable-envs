# Semantic recovery checkpoint C

Continuation from `7b2a42eaf2db457fbac9164b4e4413c579f9f513` on
`repair/environment-semantics`. Reference SWM remains
`176d54532b3c0a251a4f509b74f2fdacc143a63d`.

**Checkpoints C2 and C3 are reached.** Humanoid Walk completes its corrected CPU
gate and frozen renderer check (details below). The bounded PushT investigation
found exact single-contact reconstruction but no sufficient supported-interface State for
simultaneous contacts. The remaining private engine candidates require review.
This is not evidence that exact restoration is globally impossible, and it is
not a shared architecture failure.

Oscar queue last checked **2026-09-23 01:07:01 UTC**; all four acceptance
jobs are PENDING / QOSGrpGRES.

The inventory remains **10 VERIFIED / 32 INCOMPLETE**. No new promotions have
occurred. All complete GPU reruns below await acceptance, so no new GPU terminal
marker, semantic result or candidate renderer hash is claimed.

## Three acceptance repairs and immutable reruns

| Slice | New complete GPU job | Immutable source | Status at checkpoint |
| --- | --- | --- | --- |
| Remaining classic control | 6626587 | `98c65cbe6690404bf04a463e72003ba924b6c989` | PENDING / QOSGrpGRES |
| Source-aligned DMC | 6626669 | `57c10c17f1bf74ce565ff80e5f44292e3fc827ad` | PENDING / QOSGrpGRES |
| Navigation | 6626600 | `bb517bdd6e9bac07dfb56fde42b0bc9f249a5110` | PENDING / QOSGrpGRES |

[Submission manifest](semantic_recovery_evidence/checkpoint_c/oscar_reruns.json)
records exact archive hashes, immutable runner/renderer driver versions and
expected terminal markers. Older source archives and drivers were not modified.

Classic changes only the two diagnostic subtractions in
`_continuous_dtype_flag_necessity`: both operands are detached and compared on
CPU. Production dynamics, State, tolerances and fixtures are unchanged. The full
CPU gate passes 24 cases with zero reference errors. The GPU runner requires its
full semantic JSON, candidate/baseline renderer equality and the prior
CartPole/Pendulum regression before printing its terminal marker.

DMC revision `c6ae835421e166437b6384ff5582f56b3a504929` first made omission
failures report domain, omitted key, `error_without` and `error_with`.
CPU diagnostic job **6626607** on Oscar then failed specifically at Ball-in-Cup
warmstart: **omitted error 0, restored error 0**. Thus this tested history had no
observable omission witness; the restoration was exact in that experiment.
It was not a demonstrated restoration bug.

A bounded deterministic search, CPU job **6626635**, tested contact histories on
the same Oscar package profile. All restored errors were zero. Moving the
Ball-in-Cup snapshot from prefix step 5 to **6** gives one contact/four constraints
and nonzero observable omission error on both observed profiles:

| Profile | Omitted warmstart error | Restored error |
| --- | ---: | ---: |
| Mac ARM, MuJoCo 3.12 | `1.1102230246251565e-16` | 0 |
| Oscar CPU x86, MuJoCo 3.11 | `2.942091015256665e-15` | 0 |

These remain small floating-point witnesses, not a claim of large physical
drift or universal hardware robustness. The fixed contact history is validated
on both observed profiles; the complete GPU gate remains mandatory. The search
stopping target of `1e-10` was only a search-strength target, never an acceptance
tolerance. Gate assertions still require omitted error > 0 and restored error
== 0. No State field was removed. All four domains pass the complete revised CPU
gate (32 cases); other omission fields remain demonstrated.

GPU job **6626590**, the diagnostic-only revision's pending full gate, was
cancelled before execution after the CPU diagnosis and replaced by **6626669**.
It has no acceptance result. The replacement runs frozen renderer comparison,
Reacher, first-family DMC and continuation DMC regressions as well as the four
new domains. Raw diagnostic/search logs and scripts are preserved in the
[checkpoint evidence directory](semantic_recovery_evidence/checkpoint_c/).

Navigation reproduces the historical per-environment CPU `float(torch.norm(...))`
reduction, then places distance and strict-threshold outputs on the requested
device. The production change is local to hidden-target navigation. Its original
family commit is `9b6b655b6c39b097d3751aa06bbdabef6a2be74d`, integrated as
`bb517bdd6e9bac07dfb56fde42b0bc9f249a5110`. All 18 CPU trajectory cells pass
exactly, including TwoRoom/Piecewise strict success thresholds and SimplePointMaze
Agg cached-goal bytes/dtype boundaries; nine focused tests pass. CUDA correctness
and candidate renderer preservation await the complete rerun.

The ten backlog rows still awaiting Oscar acceptance are:

- `swm/AcrobotControl-v1`, `swm/MountainCarControl-v0`,
  `swm/MountainCarContinuousControl-v0`.
- `swm/BallInCupDMControl-v0`, `swm/FingerDMControl-v0`,
  `swm/ManipulatorDMControl-v0`, `swm/QuadrupedDMControl-v0`.
- `swm/TwoRoom-v1`, `swm/Piecewise-v0`, `swm/SimplePointMaze-v0`.

All remain `INCOMPLETE / CPU_PASSED_OSCAR_PENDING`. Their earlier failed jobs are
retained as provenance rather than erased by the new queued attempts.

## Humanoid Walk CPU recovery: checkpoint C2

The corrected full CPU gate also reaches C2. Family commit
`6af63ce57dd82f7531f21f175155375d93804293` is integrated as
`97436d6ae92835d34a01fd5bd758cbc30ff0cca7`; immutable Oscar job **6626742**
uses that integrated source and separately hashed runner/renderer drivers. The public class is
`DMCHumanoidWalk` for `swm/HumanoidDMControl-v0`; the old Stand name is retained
only in private/test-side frozen-renderer translation. The candidate remains
`DELEGATED_CPU` and **INCOMPLETE / CPU_PASSED_OSCAR_PENDING**.

Task semantics are `Humanoid(move_speed=1, pure_state=False)`, 25-second horizon,
0.025-second controls, five physics substeps, 21 float64 normalized controls,
and action repeat two with reward summation. The observation is float32[67].
All 585 comparable numeric model arrays match the pinned source, including the
actual float32-to-MJCF floor-friction rounding.

The proven bounded State contains:

- float64 qpos[28], qvel[27], qacc_warmstart[27] and time;
- DMC step counter and reset-next-step flag;
- cumulative reward;
- complete MT19937 keys, position, Gaussian-cache flag and cached value.

The model has no activation state. Different old controls remain unrestored but
are overwritten by the next action; the gate proves exact continuation despite
those different controls. Warmstart omission produces
`5.861977570020827e-13` observable drift; full restoration produces zero error.

The final gate has six seed/batch trajectory cells (seeds 0/17/42, batches 1/3,
32 steps and eight-step different-history continuation), two positive-reward
cells and two lifecycle cells. State and output errors are zero. The lifecycle
replay explicitly restores public-step-499 State and crosses the implicit reset,
including RNG changes, rather than merely replaying after reset. It also covers
snapshot independence, set/get roundtrip, full-State/private-render equality,
render non-mutation and refusal of lossy restoration.

Nondefault gravity matches SWM exactly for 32 steps while diverging from the
nominal task; visual-only Variation preserves nominal gravity and eight-step
continuation. These tests certify a bounded Variation subset, not the complete
historical reset/Variation surface.

Four frozen Stand-corpus rows produce byte-identical baseline/candidate CPU
frames (shape [4,84,84,3]):

- Rows SHA-256: `216358698225a5a9daaacba852be12048f7e4ce660390e51ffe3fe97cac9f948`.
- Frame SHA-256: `2348bd286192b622d5a8f527abaceecd9794e40324f14d704aaff4181b861047`.

[Humanoid recovery report](dmc_humanoid_recovery.md) and its linked CPU JSON,
frozen-frame JSON and reference-only oracle preserve the evidence. Its separate
Oscar runner requires Humanoid semantics and renderer equality plus Reacher,
first-family, continuation and aligned-DMC regressions. GPU acceptance remains
outstanding; this adds Humanoid to the ten backlog rows, for **11 CPU-passed
registrations awaiting Oscar** and still **10 VERIFIED / 32 INCOMPLETE**.

## PushT: bounded restoration experiments and the remaining boundary

Both `swm/PushT-v1` and `swm/PushT-Discrete-v1` remain INCOMPLETE. No public State,
production implementation or opaque engine serializer has been introduced.
Continuous actions remain understood as the source's relative target at step
entry, `position + 100 * action`, followed by ten 0.01-second PD substeps
(kp=100, kv=20), with no source clipping/validation. Discrete actions use the
source's 16-by-16 polar-bin midpoints and float32 Cartesian decoding; [0,0] is
not a no-op. Sampled reward goal and renderer goal_pose remain distinct.

The [probe profile](semantic_recovery_evidence/checkpoint_c/pusht/PROFILE.md)
links eight saved scripts, raw execution stdout/stderr, inspected CFFI symbols,
invalid-run labeling and SHA-256 checksums. These are direct CPU experiments on
Pymunk 7.3.0 against the pinned source, not certification of either registration.

1. Restoring accessible body poses/velocities/forces/torques, sampled goal and
   cached goal image leaves the controlled wall mismatch at about 0.552379
   block-y and 0.259348 reward. Adding exposed Space timestamp, current timestep
   and shape-counter scalars does not resolve it.
2. Restoring the cached arbiter through Pymunk's helpers round-trips its declared
   numerical state exactly: e/u/surface velocity, normal/count/swapped/stamp/state,
   contact r1/r2, nMass/tMass, bounce, jnAcc/jtAcc/jBias, bias and hash. However,
   the inserted cache does not establish active body membership: source count 1,
   restored count 0. Pymunk's serializer omits active thread linkage.
3. One supported detection step followed by exact body restoration reconstructs
   live membership. The controlled **single-wall** case then has zero next-output
   and reward error. Header, geometry and impulse ablations also remain exact
   in that case, so they are not established necessary there.
4. For a simultaneous wall plus agent–block contact, reconstructed cached-arbiter
   and per-body semantic traversal order match the source. Three-step continuation
   still differs. Without arbiter overwrite, maximum errors are
   `[0.0808164379388927, 0.07850967358524485, 0.054683375469721796]`.
   Restoring all tested exposed arbiter/contact state gives
   `[0.10992291186727243, 0.10205843906925338, 0.07806326210055659]`.
   Header/geometry restoration alone does not improve the first sequence;
   restoring accumulated impulses changes and worsens it. Copying those values
   is therefore not a demonstrated sufficient State extension.
5. As a separate diagnostic, supported `Space.copy()` starts with exact exposed
   body values and a cached arbiter but diverges on the next wall step (maximum
   error 37.5275). This corroborates a copy limitation; it is not a proof of
   impossibility or authorization to serialize the engine.

The shipped CFFI exposes cpArbiter/cpContact fields while cpBody/cpSpace are opaque.
Body bias velocities (`v_bias`, `w_bias`), internal solver membership, broadphase
collision IDs/cache and other private Space bookkeeping remain candidates, not
experimentally isolated causes. Reaching those internals would require a
separate native/private-ABI investigation. The present evidence supports stopping
at C3 for review: **no general sufficient explicit State was found within the
tested supported surface**. Single-wall success cannot certify general PushT.

## Preservation boundary

No shared State/RenderState, SimulatorAdapter, parity harness, execution taxonomy
or promotion rule changed. No tolerances were relaxed. Existing frozen evidence
is preserved. No Fetch, OGBench manipulation, Craftax, Rocket or final paired
benchmark work started.

The Phase 0 audit remains byte-identical; its SHA-256 is
`f27769ec10d843145c5642d9e98ee12d679a9f01206ad1e29c7c24c69ca61c0a`.
`baseline/renderer-v0` remains `6d80b68b8b3ccd07d4207be94cbf4f0760758a06`;
product `v0-draft` remains `9471a619e08aedb0b53a90b901c9e26ffff57c11`.

## Final validation

Humanoid focused unit/oracle tests: 4 passed; full integration gate: 1 passed.
Existing DMC migration checks: 116 passed, 3 expected skips. Navigation focused
tests: 9 passed. Integrated replacement-readiness tests: 8 passed. The complete
classic and corrected aligned-DMC CPU gates pass, as recorded above. Matrix
counts, artifact checksums, referenced paths, exact Humanoid errors, corrected
lifecycle replay and frozen-frame equality were checked. Historical baseline,
product branch and Phase 0 audit remain unchanged.
