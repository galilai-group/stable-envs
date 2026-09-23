# Fetch representative acceptance checkpoint

Starting branch: `repair/environment-semantics`, clean at
`a5ef2e47662d352022468a1765f06d261497033a`. Final branch is the same; the containing
commit records this checkpoint. Implementation is `32849a4f`; accepted renderer
driver revision is `bf9101cb`. No push was performed.

Work was performed in `/private/tmp/stable-envs-semantic-recovery`. The main
checkout `/Users/samib/stable-envs` and the prior
`/private/tmp/stable-envs-fetch-proof` worktree were inspected and left unchanged.
The clean pinned oracle is `/Users/samib/stable-worldmodel` at
`176d54532b3c0a251a4f509b74f2fdacc143a63d`.

The 42-row implementation map was committed separately as `f51d26eb`:
[semantic_recovery_implementation_map.json](semantic_recovery_implementation_map.json).
Its original closure audit confirms all 21 previously accepted registrations:
12 DMC historical defaults, CartPole/Pendulum/Acrobot/MountainCar/continuous
MountainCar, TwoRoom/Piecewise/SimplePointMaze, and OGBench Point/Large Maze.
Every evidence hash resolves, terminal success markers exist, renderer evidence
matches, and accepted production code is unchanged at the latest covering gate
except equivalent Maze formatting. No closure discrepancy required a rerun.
Five older original terminal logs were archived locally to complete packaging.

Exactly one new registration is accepted: **swm/FetchPickAndPlace-v3**.
The matrix and map now record **22 VERIFIED / 20 INCOMPLETE**.

Production changes are restricted to `stable_envs/_adapters/fetch_pickandplace.py`,
`stable_envs/envs/_fetch_semantics.py`, and `stable_envs/envs/fetch.py`. Supporting
changes are the implementation-map builder, public CPU and GPU renderer gates,
Oscar runner, legacy Fetch source tests and research/evidence records. Shared
State/RenderState, SimulatorAdapter, parity, taxonomy and promotion machinery
are unchanged.

The sufficient public State is float64 qpos[22], qvel[21], time[1], warmstart[21],
controller.position[3], controller.quaternion[4], goal[3],
task.initial_gripper_xpos[3]; int64 elapsed_steps[1] and complete named PCG64
state/increment words[4 each], has_uint32[1], uinteger[1]. It is a sufficient
named schema, not a minimal-byte claim. Complete PCG64 is atomic. Warmstart and
controller-cache omission witnesses remain preserved in the earlier source
proof. The reset anchor follows directly from source reads on future resets.

Restore validates full State, writes qpos/qvel/time, goal and reset anchor,
clears applied forces, calls mj_forward, restores warmstart and controller
caches, then lifecycle and PCG64. Normal sequential step does not restore.
The source trace covers pinned SWM Fetch/spaces, upstream Robotics robot/fetch/
pick-and-place/mujoco-utils, and Gymnasium TimeLimit; hashes are in provenance.

| Required acceptance | Result |
| --- | --- |
| Public adapter wiring and execution | PASS, DELEGATED_CPU; no production SWM import |
| Reset, independent snapshots, State roundtrip | PASS for seeds0/17/42, B1/3, both action dtypes |
| Different-history continuation | PASS, 32 steps then 8 after 11 different-history steps |
| Reward/success boundaries | PASS, exact0.05 and immediate float64 neighbours; reward0/success0 at threshold |
| Action semantics | PASS, float32/64, +/-1, clipping and simultaneous XYZ/gripper |
| Lifecycle | PASS, terminated false; truncated false/true/true at49/50/51; restore at49 |
| RNG | PASS, complete simulator PCG64 replay under fixed Variation; independent seeded visual RNG |
| Bounded Variation | PASS, mass8 changes reference dynamics, candidate exact; start/goal/angle ordering and reset-anchor replay; persistent versus reset mass; object color preserves physics |
| CPU full rendering | Unsupported: CPU_BACKEND_MISSING |
| GPU rendering and frozen baseline | PASS Oscar6646530 on A5000; full/private equality, non-mutation, color effect, four frozen480x480 rows identical |
| Compact restore refusal | PASS on CPU and GPU |
| Readiness | 8 passed |
| Legacy Fetch source checks | 10 passed; 7 renderer-construction skips in general local profile lacking Robotics |

CPU semantics use the pinned MuJoCo3.8.1 profile. GPU rendering uses a separate
MuJoCo/MJWarp3.11.0 profile; serialized CPU State bridges them. No physics parity
under the GPU renderer profile is claimed. Detailed results and limitations are
in [fetch_public_recovery.md](fetch_public_recovery.md) and the
[acceptance manifest](semantic_recovery_evidence/fetch_public/acceptance.json).

There is no remaining blocker within this accepted representative scope. CPU
full rendering stays unsupported. Broader Variation/default appearance
randomization, alternate tasks, eleven Fetch siblings and benchmarks are not
certified. Unseeded SWM visual resampling uses fresh external entropy and has no
replay claim. PushT remains parked; Cube/Scene, Craftax and Rocket were not started.
Shared-contract change requested: **no**. Stop here for review before expansion.
