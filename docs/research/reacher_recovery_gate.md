# Phase 0.5 and Reacher stopping gate

This checkpoint implements the shared full-State boundary and the first delegated
proof. It does not certify the remaining 41 historical registrations or begin
paired benchmarking. The Phase-0 audit and existing renderer fixtures remain
unchanged. Shared contract and harness commit: `7346db7`.

## Contract and implementation

`get_state()` and `state` return independent full TensorDict snapshots;
`set_state(State)` restores native or delegated simulator state and returns None.
Native CartPole/Pendulum/Acrobot/MountainCar, PushT and MiniGrid restore their
previously discarded hidden fields. This repairs their own restoration contract;
it does not certify their historical SWM reset/reward/lifecycle semantics.
Renderer-only cells explicitly refuse full-State reset/restoration until recovered.
Compact buffers, projection helpers and renderer entry points are private.

The tiny `SimulatorAdapter` protocol has reset/get/set/step/close and num_envs, no
renderer or required transition. Reacher composes B independent dm_control CPU
simulators. `device="cuda"` puts State and rendering on CUDA; it does not claim GPU
physics. The adapter imports upstream dm_control/MuJoCo, never SWM. The independent
test bridge imports the actual pinned SWM wrapper at
`176d54532b3c0a251a4f509b74f2fdacc143a63d`.

Reacher State contains float64 qpos/qvel/time/warmstart, target xyz, cumulative
reward, integer step counter, reset-next-step flag and MT19937 state. A different-
history test initially exposed a 7.366e-16 velocity difference with qpos/qvel alone
plus lifecycle/RNG. Restoring `qacc_warmstart` eliminated it; the final gate uses
zero tolerance. Prior actuator control was not needed because the next action
replaces it. No opaque whole-simulator serialization is used.

Target position changes during SWM's implicit reset inside a public step after
the dm_control time limit. It is therefore carried in State rather than assumed
constant throughout the public trajectory. Densities are persistent Variation and
are applied to the simulator; appearance remains owned by the existing renderer.
SWM's hidden-target default is now the public Reacher default. Frozen renderer
comparison explicitly supplies the old visible-target Variation on both versions;
the four-column layout and frame function remain unchanged.

## Evidence

- Reset, full roundtrip, one-step and 32-step parity: seeds 0/17/42, batches 1/3.
- Different-history restore: eight subsequent steps per seed/batch, all output
  fields checked, including observation and score.
- Positive reward case: both underlying rewards sum to 2; termination/truncation
  match the wrapper.
- Lifecycle boundary: 502 public steps, including implicit reset and target/RNG
  changes, followed by different-history replay across that boundary.
- Physical Variation: arm density 650 and finger density 1350, reset and 32 steps.
- Full-State rendering equals private projection rendering and leaves simulator
  state unchanged. Raw/incomplete projections are refused by restoration.
- All semantic comparisons have maximum error **0**, on local CPU (MuJoCo 3.12.0,
  dm_control 1.0.45) and Oscar RTX 3090 State/rendering (MuJoCo 3.11.0,
  dm_control 1.0.44), with a same-profile SWM reference on each host.

Machine evidence: `semantic_recovery_evidence/reacher_cpu.json` and
`semantic_recovery_evidence/reacher_cuda.json`. The additive independent oracle is
`tests/fixtures/semantics/reacher/`, including a content hash and dependency pins.
Its generator refuses to overwrite an existing oracle.

Oscar jobs: `6597759` passed the shared boundary comparison; `6597787` passed the
Reacher semantic gate and renderer comparison. `6597722` failed before testing the
candidate because its profile lacked MJWarp; the existing MJWarp profile was then
used. No thresholds or fixtures were changed to obtain a pass.

Frozen render comparisons use four existing rows each for CartPole, Pendulum,
PushT, TwoRoom, DMCReacherHard, DMCCartpoleBalance and DMCHopperHop. Baseline and
candidate frames are byte-identical on CPU and RTX 3090, including MJWarp Hopper.
This is **seven-cell evidence**, not an all-42 `REGRESSION=0` claim. CPU hashes are
in `phase05_cpu_renderer.json`; GPU hashes are in `renderer_cuda.json`.

Final local suites: 315 regression/shared/oracle tests passed (two skipped);
146 integrated Reacher/native tests passed; 37 State/spec tests passed. These
sets overlap and are not added together as a total. The earlier shared dataset/API
run passed 180 tests with four optional skips. Ruff and `git diff --check` pass.

## Remaining scope

The canonical `semantic_recovery_matrix.json` records per-registration status.
Reacher's default hard-task semantic gate is passed; its complete historical
Variation surface (including shape), reset options and renderer similarity against
SWM across those options remain unverified. Easy/qpos-match and other DMC tasks
are outside this proof. Broader family recovery and reference/stable CPU/GPU paired
benchmarks have not started. Pure transition/unroll is optional for delegation.

Reproduce with the selected pinned profile:

```sh
PYTHONPATH=. python scripts/recovery/reacher_gate.py --device cpu --output /tmp/reacher-cpu.json
# On an Oscar GPU allocation, with pinned SWM on PYTHONPATH:
PYTHONPATH=".:$PYTHONPATH" python scripts/recovery/reacher_gate.py --device cuda --output /tmp/reacher-cuda.json
python -m pytest tests/unit/test_full_state_contract.py tests/unit/test_semantic_harness.py tests/unit/test_reacher_oracle.py
```
