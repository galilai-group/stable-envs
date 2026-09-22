# Acrobot correction and second bounded recovery checkpoint

No new registration is promoted to `VERIFIED` at this checkpoint. Five additional
historical defaults have passed exact CPU gates and are integrated into
`repair/environment-semantics`, but required Oscar acceptance is blocked by
cluster scheduling. The matrix remains **5 VERIFIED / 37 INCOMPLETE**, including
**5 CPU_PASSED_OSCAR_PENDING** rows within INCOMPLETE. These are distinct from the
historical architecture-readiness counts.

The Acrobot identity correction is complete: canonical `DMCAcrobotBalance`
implements `Balance(sparse=True)`. Its former public Swingup name is removed;
private reference machinery translates immutable historical fixture names.
The corrected three-default CPU gate and Oscar job **6605333** passed, including
three-cell frozen renderer preservation and the original Reacher regression.
The shared Phase-0.5 contract and renderer baseline remain unchanged.

## Exact additions awaiting Oscar

| Historical ID | Canonical class | Exact task |
| --- | --- | --- |
| `swm/CheetahDMControl-v0` | `DMCCheetahRun` | CustomCheetah run, move_speed=10; default superclass reward branch |
| `swm/HopperDMControl-v0` | `DMCHopperHop` | CustomHopper hop, hopping=True; default superclass reward branch |
| `swm/WalkerDMControl-v0` | `DMCWalkerWalk` | CustomPlanarWalker walk, move_speed=1; default superclass reward branch |
| `swm/CartPoleControl-v1` | `CartPole` | Gymnasium CartPole-v1 with SWM float32 physical defaults |
| `swm/PendulumControl-v1` | `Pendulum` | Gymnasium Pendulum-v1 with SWM float32 physical defaults |

No sibling task or renderer variant inherits certification. The existing five
VERIFIED scopes remain Reacher hard/default, Acrobot sparse Balance, Pendulum
SwingUp, Cartpole sparse SwingUp, and PointMaze large Navigate.

## State, adapters and exact parity

The new DMC State holds float64 qpos/qvel/warmstart (widths 9/7/9 by domain), time,
DMC step/reset-next-step counters, cumulative reward and explicit MT19937 fields.
Each task independently produced observable error when saved warmstart was omitted:
Cheetah 2.75e-14, Hopper 2.13e-14, Walker 1.69e-13. Restoring warmstart reduced all
three to exact zero against the same uninterrupted reference output sequence.
Control is overwritten by the next action; qacc is derived; all three models have
no activation state. No opaque serialization or tolerance was introduced.

CartPole State holds float64 x/x-dot/theta/theta-dot, elapsed steps,
steps-beyond-termination plus its presence bit, and explicit PCG64 limbs/cache.
Pendulum holds float64 theta/theta-dot/torque, a last-u presence bit, elapsed
steps, and the same PCG64 representation. Float64 torque preserves accepted
higher-precision actions exactly. The obsolete physical-only transition/unroll
and sim_state/set_sim_state helpers explicitly refuse on these two classes.

Both family adapters own independent authoritative CPU simulators per lane and
report `DELEGATED_CPU`. CUDA State/output placement and rendering do not imply
GPU physics. Production does not import stable-worldmodel. The shared State and
SimulatorAdapter contracts were not redesigned.

Pinned SWM source is `176d54532b3c0a251a4f509b74f2fdacc143a63d`, with clean tracked
source verified by every gate. All State/output leaves compare exactly at seeds
0/17/42, batches 1/3, one and 32 steps. Evidence includes reset parity, direct
snapshot mutation, candidate-only restore versus an uninterrupted reference,
different-history restoration, RNG replay, rewards/done and lifecycle boundaries,
render non-mutation and rejection of compact/incomplete restoration.

DMC lifecycle checks cover 1252 public Cheetah steps and 502 Hopper/Walker steps,
including implicit reset. Gravity `[1.25,-0.75,-7.5]` is exercised for 32 steps
per task; visual-only Variation preserves exact default gravity and eight steps.
CartPole checks the 500-step TimeLimit and terminal rewards `[1,0,0]`; Pendulum
checks the 200-step TimeLimit, upright zero reward, negative cost and clipping.
All six physical leaves per classic task are changed together in a nondefault
32-step case. Full historical Variation and reset-option surfaces remain uncertified.

## Validation and renderer scope

Coordinator combined CPU checks: **338 passed, 3 expected Swimmer CPU skips**.
Standalone gates produced **30 DMC cases and 16 classic cases**, all exact.
New independent reference-only NPZ corpora include hashes and typed schemas.

Frozen renderer preservation reran four rows per new class: Cheetah, Hopper,
Walker, CartPole and Pendulum, byte-identical to the unchanged baseline on CPU.
The Acrobot correction separately reran Acrobot/Pendulum/Cartpole DMC on CPU and
Oscar. Thus this checkpoint has eight distinct CPU renderer cells rerun and
three completed Oscar renderer cells; the other five Oscar cells remain pending.
This is not an all-42 renderer pass or a paired performance benchmark.

## Oscar scheduling blocker and continuation

As of 2026-09-22 14:58 UTC, both jobs are `PENDING (Priority)`:

| Job | Source stage | Requested time | SLURM estimated start, cluster local time |
| --- | --- | --- | --- |
| 6605941 | native-classic-446f860 | 5 minutes | 2026-09-23 22:30 |
| 6605999 | dmc-continuation-6470521 | 10 minutes | 2026-09-23 22:40 |

Estimates may change. Source archives were checksum-verified in the approved
`/oscar/scratch/sboughan/se-semantic-recovery/` directory. Existing running
training allocations were left untouched. No NVIDIA work ran on the Mac.

Next, inspect actual logs `logs/family-<job>.out`, not just SLURM completion.
Require `FAMILY_GPU_GATE_PASS <stage>`, exact gate JSON, and identical baseline /
candidate frozen renderer JSON. The DMC job also runs the original Reacher and
first-three-default regression gates. Outputs are `<stage>-gate-cuda.json`,
`<stage>-renderer-baseline-cuda.json`, and `<stage>-renderer-cuda.json` under the
same scratch directory. Diagnose any failure before promotion. Only after these
checks pass may the five pending rows become VERIFIED, bringing the count to 10.

## Visible remaining blockers

The native audit covers all ten historical native registrations. Its remaining
eight rows are unverified. PushT still needs relative continuous actions, separate
polar discrete decoding, exact goal/reward/reset behavior and experimentally
proven contact restoration; contact-cache handling is an open proof requirement,
not a demonstrated impossibility. Navigation still needs task/reset/collision
proof. SimplePointMaze additionally has a default `show_goal=True` versus `False`
identity mismatch. The other three classic tasks need independent recovery.

Historical Humanoid is Walk; the Stand renderer cannot be certified as its
replacement. Ball-in-cup Catch, Finger Turn Hard, Manipulator Bring Ball and
Quadruped Walk are source-aligned candidates only. OGBench Scene remains explicitly
blocked by its missing required `env_type`; no global skip or family-complete
claim was added. Fetch, OGBench manipulation, Craftax, Rocket and paired benchmarks
were not launched in this bounded checkpoint.

## Revisions and evidence

- Acrobot identity: `7221759596f8648bbbd84e885231dfbcd4ab9e50`;
  acceptance record: `5292c12bc125dac317fcb5fb08c01accf44e4a1c`.
- Classic implementation/oracle: `446f8608bc800c2d61701f675a2536df0dcbaa3f`;
  native audit: `a5f6eda8c5d768657a2998e51e28817ca731c296`.
- DMC continuation implementation/oracle/report: `6470521e84a47d5f95a5bd5f0c5a338ab293619f`.
- Integration merges: native `5c6c7a2`, DMC `94a46e9`.
- Product branch `v0-draft` remains `9471a619e08aedb0b53a90b901c9e26ffff57c11`.
- Renderer baseline remains `6d80b68b8b3ccd07d4207be94cbf4f0760758a06`.

See [DMC evidence](dmc_continuation_recovery.md), [native audit](native_verification.md),
[recovery matrix](semantic_recovery_matrix.json), and the JSON files in
`semantic_recovery_evidence/`. Historical renderer corpora, performance fixtures,
Phase-0 audit and previous provenance are unchanged. No branch was pushed.
