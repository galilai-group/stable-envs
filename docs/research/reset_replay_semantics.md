# Reset reproducibility at external entropy boundaries

This metadata clarification leaves the Phase 0.5 State/RenderState structures,
SimulatorAdapter protocol, renderer policy, and every accepted registration's
existing acceptance scope unchanged.

`set_state()` must restore exact continuation and all environment-owned RNG
used by that continuation. Explicit `reset(seed=N)` must reproduce the seeded
reset within the registration's accepted configuration. A source operation
that explicitly acquires fresh external entropy at `reset(seed=None)` has no
promise that two independent calls receive the same entropy. It must actually
resample; replaying a captured seed on a later unseeded reset is incorrect.
Capturing an episode's reset anchor for restoration does not change this rule
for the next reset. Hidden owned RNG cannot be treated as external entropy.

The additive [registration metadata](reset_replay_classification.json) records:

- `SEEDED_AND_UNSEEDED_REPLAYABLE`: owned reset streams are restored, within the
  **existing accepted scope**, including its fixed persistent Variation and
  configuration requirements. This label adds no new acceptance coverage.
- `SEEDED_REPLAYABLE_FRESH_UNSEEDED_ENTROPY`: exact seeded reset and continuation;
  future unseeded reset intentionally obtains fresh source entropy.
- `UNCLASSIFIED_INCOMPLETE`: no accepted reset replay claim.

Rocket uses the fresh entropy category as a **source requirement, not promotion**.
Pinned SWM `RocketLandingEnv.reset` calls `variation_space.seed(seed)` before
sampling. The inherited Gymnasium `Dict.seed(None)` calls `seed(None)` on each
child; leaf generators acquire new entropy. The main Gymnasium generator,
Aviary, rocket and boosters share an owned PCG64 stream and are still subject
to exact restoration. The previous research witness's unequal independent
unseeded resets is preserved unchanged; it is no longer itself a blocker under
this clarified boundary. It does not certify a production adapter.

Evidence: [source audit](rocket_recovery_audit.md),
[replay feasibility](replay_feasibility.md), and the immutable matrix's
per-registration `reset_parity`, `rng_lifecycle`, and acceptance evidence.
The three accepted navigation registrations already explicitly record fresh
external entropy in their `rng_lifecycle`; their guarantees are unchanged.
Cube/Scene's accepted owned task/Variation RNG compatibility layer also remains
unchanged. PushT has no new classification or acceptance claim.
