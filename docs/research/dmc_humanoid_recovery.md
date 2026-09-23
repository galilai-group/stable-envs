# Historical DMC Humanoid Walk recovery

`swm/HumanoidDMControl-v0` defaults to `task="walk"`. The recovered public
identity is therefore `DMCHumanoidWalk`; `DMCHumanoidStand` remains only as a
test-side translation for immutable renderer corpora recorded under the old
name. Walk uses dm_control's `Humanoid(move_speed=1, pure_state=False)`, a
25-second horizon, 0.025-second controls made from five physics substeps, and
two controls per public step. Its 21 normalized controls and rewards exactly
match pinned SWM commit `176d54532b3c0a251a4f509b74f2fdacc143a63d`.

The production adapter builds the pinned wrapper's actual MJCF rather than
using `suite.load`. All 585 comparable numeric `MjModel` arrays match. This
includes floor friction `[1.0, 0.004999999888241291,
9.999999747378752e-05]`, produced by SWM's float32 MJCF write. The model has
`nq=28`, `nv=27`, `nu=21`, `na=0`; the flattened observation has 67 float32
values.

Full State contains `qpos[28]`, `qvel[27]`, `qacc_warmstart[27]`, time, DMC
lifecycle counter and reset flag, cumulative reward, and complete MT19937
state. In a candidate-only different-history restore, omitting warmstart caused
an observable continuation error of `5.861977570020827e-13`; restoring it made
all State and output leaves exactly equal. Controls remained different after
two histories and exact continuation still resumed because the next public
action overwrites them. The model has no activation state, and the task's speed
and observation mode are immutable construction values rather than mutable
episode caches.

The CPU gate covers seeds 0, 17, and 42 at batches 1 and 3, exact reset and 32
steps, snapshot independence, roundtrip, different-history continuation,
action behavior, positive reward, rendering from full State, render
non-mutation, and refusal of lossy projections. At public step 500 the internal
counter reaches 1000 and schedules reset; step 501 resets and advances once.
Restoring the public-step-499 State into two different histories and replaying
four steps crosses that boundary exactly, including RNG change.

Gravity `[1.25, -0.75, -7.5]` matches the varied SWM reference for 32 steps and
diverges from the same-seed default. A visual-only light change preserves exact
nominal gravity and eight-step future behavior. Frozen Stand-named rows and the
Walk renderer produce the same CPU frame digest
`2348bd286192b622d5a8f527abaceecd9794e40324f14d704aaff4181b861047`.

Evidence is in `semantic_recovery_evidence/dmc_humanoid_cpu.json`, the two
renderer JSON files beside it, and the immutable reference-only oracle under
`tests/fixtures/semantics/dmc_humanoid`. GPU acceptance remains an Oscar step;
`scripts/recovery/oscar_dmc_humanoid_gate.sbatch` runs the Humanoid gate,
frozen renderer comparison, and all earlier recovered DMC regressions.
