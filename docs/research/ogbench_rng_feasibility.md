# Cube/Scene RNG feasibility checkpoint

**Recommendation: Design A plus per-instance seed, task-selection and Variation
ownership.** Exact reset replay, property-read independence, and independent
interleaved/concurrent use pass in the pinned CPU research profile. This is an
RNG-feasibility result, not public State or registration acceptance. Both
registrations remain INCOMPLETE; the matrix remains **33 VERIFIED / 9 INCOMPLETE**.
Stop here for review, before production implementation.

## Starting point and boundaries

Start: clean `repair/environment-semantics` at
`f9e62ac9f3d9f9179eebfacbf3c4c45a9898fcc3`, worktree
`/private/tmp/stable-envs-semantic-recovery`. The documentation checkpoint
`03adc1ab79bae71e3e62db2e5c4cb19f20723166` and the
[candidate-State artifact](ogbench_manipulation_candidate_state.md) were present.
Both manipulation adapters were NOT_IMPLEMENTED. No discrepancy in that handoff.
The containing research/evidence commit and exact ending SHA are reported in the
final handoff; a commit cannot embed its own hash.

SWM remains at `176d54532b3c0a251a4f509b74f2fdacc143a63d`. Its tracked files are
clean. It has unrelated untracked user files/directories, which were left alone.
No production stable-envs, source oracle, renderer, matrix/map, promotion metadata,
shared contract, or accepted family files changed. No GPU/renderer work or push.

All new files are this report and
`semantic_recovery_evidence/ogb_rng_feasibility/`. The executable prototypes live
only there. No public `get_state`/`set_state` implementation is proposed as complete.
No learner observation abstraction or SWM integration was added.

## Authority, versions and exact paths

Python 3.12.11, macOS arm64; OGBench 1.2.1, MuJoCo 3.8.1, Gymnasium 1.2.3,
NumPy 2.2.6, dm-control 1.0.41, dm-env 1.6, dm-tree 0.1.9, SciPy 1.17.1,
lxml 6.1.1 and labmaze 1.0.6. Existing isolated CPU profile:
`/private/tmp/ogb-manip-authority-swm-lock/bin/python`. No dependency profile changed.
`MUJOCO_GL=disable` avoids a graphical context. No pixels were generated.

| Source and relevant line | SHA256 |
| --- | --- |
| SWM `envs/ogbench/cube_env.py:776` reset; `:1158` settling | `143e5e81f424ff7a4d7fdde287dee88b722c09082a6fc9e68721839788e94c19` |
| SWM `envs/ogbench/scene_env.py:316` reset; `:690` settling | `d0eebdce0131f86e3247736c601f97c2b11d0dbfad56fca998df3192f08a7b9f` |
| SWM `spaces.py:13` reset helper | `73079b2a66dace087abc7c417691e3cc339887866915553e21d0936a362229eb` |
| OGBench `manipspace/envs/manipspace_env.py:145` getter; `:282` reset; `:304` task draw | `03524eb1c95155bbc2ec3d29e9836fccffe8eb9fce1be15b6f0e7062cc1bb5fd` |
| Gym `spaces/box.py:346` sampling | `b5ad274e40f84f197c84ea9ebf05b108d2ae5da1ccdee6e090adce417bcd3231` |
| Gym `spaces/space.py:69` lazy RNG; `:110` seeding | `e9f12f9bc7bd26e686a2cf557ccfa23d7d0c5e6472fc11bf743eb88049c16108` |
| Gym `spaces/dict.py:114` recursive seeding | `7101051852bc9b2ac6dfacbcea444b0fe5c07b3b4f4f9e78387060dd5c17486a` |
| Gym `utils/seeding.py` | `eb2a223b92604b7d1df2a859338e4a175c60f3fb11b87f8f316f306f1b97de53` |
| Gym `core.py:117` Env.reset | `92226890db5bec675bce52b6b9ceba350baef86d86e6f279d02e96f1a39c2258` |

[provenance.json](semantic_recovery_evidence/ogb_rng_feasibility/provenance.json)
records absolute paths, exact definition line ranges, versions, hashes and the
previous complete source/asset inventory. The audit rechecks those source hashes.

## Entropy call graph and raw trace

`trace_entropy.py` uses scoped `unittest.mock.patch` wrappers and pass-through RNG
proxies to record actual calls, inputs, outputs, object identities, and before/after
bit-generator states. It does not replace physics, controller, task calculations,
rewards, success or observations. Raw records are in `entropy_traces.json`.

For the direct source classes:

1. Construction creates SWM Variation Boxes (11 Cube, 12 Scene), initially without
   RNGs. No Python RNG entropy is observed in direct construction or compilation.
2. A read of `action_space` constructs a genuine new Box: shape `(5,)`, bounds
   `[-1,1]`, dtype float32. Its RNG is still `None`. Reading metadata does not itself
   seed it. Every access creates a different object.
3. Wrapper reset calls `reset_variation_space(..., seed=None)`. Gym Dict.seed(None)
   recursively seeds all leaves from fresh entropy; it does not seed the Dict
   nodes themselves. This happens even with `variation=[]`.
4. Default task selection calls global `np.random.randint(1, num_tasks + 1)` before
   Gym Env.reset. Explicit task ID, custom task info, or reward-task configuration
   bypass that draw. It uses the process-global MT19937, not the environment PCG64.
5. Cube drops its public seed; Gym receives None. Scene forwards seed 17 correctly.
   Environment RNG calls include permutation and uniform arm yaw, cube jitter/yaw;
   Scene also draws drawer/window jitter. Each actual call and state is traced.
6. Goal initialization calls the getter and Box.sample twice, in order. First
   sampling lazily creates each Box's PCG64 from fresh entropy. The untouched Box
   sampler invokes its normal/exponential/uniform branches (including zero-length
   branches), then returns the float32 action. Both sampled actions are recorded.
7. Both actions pass through the authoritative controller, IK, and 25-substep
   MuJoCo step. The source restores initial qpos/qvel and retains settling effects
   including controls, warmstart and goal cache. No source output is mocked.

Counts below include one deliberate unsampled metadata read before the caller's
reset. Thus direct rows have three accesses, of which exactly two sample.
`seed(None)` counts mean calls to Gym Space/Dict.seed, not all factory invocations.

| Source case, public seed 17 | Getter accesses | Settling samples | Variation samples | Space/Dict.seed(None) | Global task draws |
| --- | ---: | ---: | ---: | ---: | ---: |
| Cube direct default | 3 | 2 | 4 | 19 | 1 |
| Cube direct variation=[] | 3 | 2 | 0 | 19 | 1 |
| Cube direct one sampled leaf + task 2 | 3 | 2 | 1 | 19 | 0 |
| Scene direct default | 3 | 2 | 0 | 20 | 1 |
| Scene direct variation=[] | 3 | 2 | 0 | 20 | 1 |
| Scene direct one sampled leaf + task 2 | 3 | 2 | 1 | 20 | 0 |
| Cube gym.make + caller reset | 7 | 4 | 8 | 38 | 2 |
| Scene gym.make + caller reset | 7 | 4 | 0 | 40 | 2 |

**Registration construction matters:** Gym PassiveEnvChecker reads
`observation_space`; the source getter performs an implicit reset when its model
is absent. Consequently gym.make causes an additional unseeded reset and two
settling samples before the caller's reset. Cube's two Gym seed arguments are
None/None; Scene's are None/17. All constructed Boxes, including observation-space
Boxes, are recorded (21 Cube / 22 Scene in the gym.make traces). These traces do
not justify treating metadata reads of `observation_space` as nonmutating. The
required action-space read-independence gate is separate. No native OS/syscall
entropy audit is claimed; the trace covers identified Python RNG boundaries.

## Designs and source-native injection points

**Retain the last Box: rejected.** `LastCube`/`LastScene` call the original getter,
store the returned Box on that instance and increment an access counter. The
experiment captures S, resets, restores S and the retained Box's RNG, then resets
again. Both resets allocate different Boxes; the retained Box's RNG remains
unchanged while the new reset consumes new generators. Both full reset comparisons
fail. Box identities, the first post-sample RNG, retained checkpoint and differing
outputs are in each family's `last_box` record. No module-global last Box exists.

**Design A: passes.** On the first property access, call the original getter,
retain that genuine Box per environment, seed it with the preassigned owned action
seed, and return it thereafter. The object is still exactly Gym Box and uses its
original sampling method. Constructing/seeding it on a metadata read does not
advance any parent stream. Tests cover 1 and 5 reads, both before the first reset
and after a captured checkpoint. Sampling advances the retained RNG normally.

**Design B, eager child seeds: rejected.** Each access calls the original getter
and draws a child seed from an owned parent PCG64. It exactly replays when the
access sequence is identical, but 1 or 5 extra unsampled reads change parent state,
settling actions and reset results in both families. A lazy variant would need a
new sampling-time ownership hook or proxy; the existing source getter offers no
such child-entropy hook. It was not implemented or certified. Design A already
meets read independence without modifying the Box sampling path.

**Design C, existing source-native boundaries: useful but not a complete substitute
for A.** Inspection found no constructor argument/setter for a persistent action
space or reset-settling RNG. Seeding a temporary getter result cannot repair it.
The following existing interfaces do solve the other sources, and are evaluated
in the combined prototype:

- `gym.Env.reset(self, seed=...)` establishes environment PCG64 and seed metadata.
  Calling it at the owned reset boundary repairs Cube's dropped seed; Scene's own
  forwarding remains intact. No reset implementation is copied.
- A per-instance `np.random.RandomState` uses the same `randint` primitive and
  bounds as the source to choose the default task, then supplies the supported
  `options['task_id']`. Explicit/custom/reward task paths bypass this draw. Task
  tables, payload selection, goal construction and reward remain authoritative.
  Entropy plumbing preselects the option before source reset; source simulation
  initialization and settling order remain unchanged.
- `OwnedVariation`, a subclass of the source Dict, retains the original child
  spaces, constraints, values and sampling order. Its instance seed(None) draws
  an integer from its own parent generator, then calls the original Dict.seed(int)
  recursion. The source reset helper and all Variation sampling remain unchanged.
  There is no process-global function replacement or runtime class monkeypatch.

The research bootstrap uses SeedSequence to separate action, Variation-parent and
task streams. Explicit seeds reproduce the complete configuration. With no
constructor seed, one root entropy value is drawn during construction, retained,
and used to initialize the environment RNG before any reset. All future reset
streams already exist or derive deterministically from owned state. Unseeded
construction followed by same-instance and independent-instance replay is tested.
This is a specified ownership policy, not a claim that the historical broken
seeded source already generated the same trajectories.

## Replay, semantic preservation and isolation results

- **Design A, each family:** 11 cases, each with exact same-instance and independent
  different-history reset replay. Seeds 0/17/42 cover default, fixed Variation,
  and sampled cube start/size; explicit task 5 and custom task payload add two
  cases. Snapshots remain unchanged while the original simulator advances.
- **Task routes, each family:** eight exact replay cases cover explicit/custom
  task inputs with reward_task_id None/0/1/5. The task RNG does not advance when
  those routes bypass default selection. Source ID-0 alias behavior is retained.
- **Unseeded construction:** both families capture generated bootstrap entropy
  and exactly replay into the same and an independently initialized simulator.
- **Property independence:** 1/5 warm reads and 1/5 cold reads pass for A, each
  family. B's four warm-read cases fail as expected. B's no-extra-read replay
  passes for both families, isolating the metadata-read defect.
- **Semantic preservation:** six cases (both families × seeds 0/17/42) record real
  settling actions, then force both the original source and the candidate to use
  that exact tape. Only entropy boundaries are patched. Physics/integration state,
  controller-visible state, task payload, goals/cache, observations/info, success,
  model damping/colors, environment RNG and Variation RNGs match exactly. The
  forced candidate also equals its naturally sampled semantic result.
- **Source order:** both settling calls remain between the first and second arm
  initialization. Box bounds, shape, dtype and sampling method remain authoritative.
  Structural identity establishes the sampling algorithm; finite samples are not
  claimed to prove distributional identity.
- **Isolation:** two Cube plus two Scene instances match for seed 42, in forward
  and reversed execution order, with cross-instance metadata reads. Four distinct
  worker threads synchronize at a barrier; each per-environment result matches
  its sequential result exactly. Independent seeds 17/42 also pass an explicit
  round-robin schedule and a four-thread run. Every nonacting environment remains
  unchanged after each round-robin reset, covering both directions across families.
  Global NumPy state remains unchanged. Concurrency is across independent instances,
  not simultaneous reset calls on the same mutable instance.
- **Numeric serialization:** six recorded checkpoints replay after validated
  numeric RNG roundtrips; six explicit-seed replay comparisons also pass. Eight
  malformed RNG encodings are rejected. Seed metadata includes None, -1, zero,
  ordinary seeds and integers beyond 128 bits without truncation.

Comparisons include all MuJoCo integration components, cached site poses/contact
forces, active/reward task IDs, task/reset payload, goal cache, qpos/cube/buttons/
drawer/window, complete source observations/info, success, model damping/colors/
target-site positions, lifecycle flags, environment/action/task/Variation RNGs and
seed metadata. Captures are research records, not public State acceptance.

All final replay, codec and isolation gates run in fresh processes without scoped
patches. Diagnostic fixed-tape patches are removed at context exit. No permanent
Box.sample, global NumPy, global entropy factory or shared RNG mutation is used.
Two early harness defects (unequal diagnostic-list comparison and removal of an
already-omitted last_box field) are retained in development logs; both were fixed
before final gates. They were harness exceptions, not physics divergence.

## Per-source classification and future numeric State leaves

| Source | Classification / resolution |
| --- | --- |
| Environment PCG64 | 1: already environment-owned; capture complete state and seed metadata |
| Cube seed argument | 2: adapt through Gym's existing seed path before authoritative reset |
| Fresh settling Box entropy | 2: memoized authoritative per-instance Box, complete PCG64 snapshot |
| Global MT19937 default task draw | 2: per-instance legacy RandomState plus source task_id option; no global swapping |
| Variation seed(None) | 2: owned seed parent and authoritative Dict.seed(int); preserve every node RNG |
| Seed metadata / initial root entropy | 3 after bootstrap: retained constructor/reset configuration; arbitrary-width numeric encoding |
| Direct model/IK compilation | No traced Python entropy; deterministic configuration in these witnesses |
| External future reset entropy | No unresolved identified source in the tested task-mode scope |

Required RNG schema for a future Design-A implementation:

- `rng.environment`, `rng.action`, `rng.variation_parent`: PCG64 records.
- `rng.variation/<path>`: initialized flag and complete PCG64 for **every** Dict
  and Box node, including root/intermediate nodes created by integer seeding
  (17 nodes Cube, 18 Scene). Preserve Variation values/init values separately.
- `rng.task`: legacy MT19937 record; no process-global task state.
- `rng.action_initialized` and the preassigned action seed for lazy construction.
- Environment `_np_random_seed` plus retained root/public seed metadata, including
  None/-1 where meaningful, represented without narrowing large integers.

PCG64 numeric leaves: `state:uint8[16]`, `increment:uint8[16]` (little endian),
`has_uint32:uint8[]` in {0,1}, `uinteger:uint32[]`; validate odd increment and exact
keys/shapes/dtypes. MT19937 leaves: `key:uint32[624]`, `position:int64[]` in 0..624,
`has_gauss:uint8[]`, `cached_gaussian:float64[]`. Seed metadata uses kind:uint8[],
length:int64[] and canonical little-endian magnitude bytes:uint8[length]. A public
batched representation would pad those bytes and validate lengths; this checkpoint
does not implement that public serializer. Root entropy does not need a live
SeedSequence after child streams are materialized: no future spawning is used.

The prototype additionally records B's action parent and diagnostic access/seed
histories/last-Box aliases. Those are not proposed future Design-A State leaves.
No shared-contract change is requested.

## Remaining work and upstream recommendation

No RNG-feasibility blocker remains for the tested registered task-mode defaults.
The proposed implementation is **A + the three narrow per-instance boundaries**,
not a process-global monkeypatch or a copied task/reset simulator. It merits a
separate reviewed production checkpoint. Full State validation/restoration,
remaining task/configuration/Variation gates, renderer acceptance, and GPU evidence
are still outstanding. Scene's pre-existing frozen-renderer constructor/signature
blocker is untouched. These results do not certify data-collection/multiview or
other untested configurations.

An upstream OGBench change should be proposed: retain one action-space object per
instance and expose owned seeding/task RNG hooks. SWM should forward Cube's seed
and pass controlled Variation entropy instead of unconditional seed(None). These
are recommendations only; no upstream issue, PR or message was sent. Preserve
conditional source semantics and record the intentional seed-policy repair.

## Commands, evidence and preservation

From the integration worktree, each research command uses:

```sh
MUJOCO_GL=disable SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel:. /private/tmp/ogb-manip-authority-swm-lock/bin/python docs/research/semantic_recovery_evidence/ogb_rng_feasibility/trace_entropy.py
MUJOCO_GL=disable SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel:. /private/tmp/ogb-manip-authority-swm-lock/bin/python docs/research/semantic_recovery_evidence/ogb_rng_feasibility/semantic_preservation.py
MUJOCO_GL=disable SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel:. /private/tmp/ogb-manip-authority-swm-lock/bin/python docs/research/semantic_recovery_evidence/ogb_rng_feasibility/gates.py
MUJOCO_GL=disable SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel:. /private/tmp/ogb-manip-authority-swm-lock/bin/python docs/research/semantic_recovery_evidence/ogb_rng_feasibility/codec_gate.py
MUJOCO_GL=disable SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel:. /private/tmp/ogb-manip-authority-swm-lock/bin/python docs/research/semantic_recovery_evidence/ogb_rng_feasibility/schedules.py
MUJOCO_GL=disable SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel:. /private/tmp/ogb-manip-authority-swm-lock/bin/python docs/research/semantic_recovery_evidence/ogb_rng_feasibility/audit.py
.venv/bin/python -m pytest -q tests/unit/test_full_state_contract.py tests/unit/test_semantic_harness.py tests/unit/test_replacement_readiness.py tests/unit/test_swm_matrix.py
.venv/bin/ruff check docs/research/semantic_recovery_evidence/ogb_rng_feasibility
.venv/bin/ruff format --check docs/research/semantic_recovery_evidence/ogb_rng_feasibility
.venv/bin/python -m mkdocs build --strict --config-file mkdocs.yaml --site-dir /private/tmp/stable-envs-ogb-rng-site
git diff --check
```

Raw traces/results are `entropy_traces.json`, `replay_results.json`,
`semantic_preservation.json`, `numeric_rng_results.json`, and
`schedule_results.json`; corresponding logs contain exact terminal markers.
Source identities and final preservation are in `provenance.json`/`closure.json`.
`SHA256SUMS` seals the files, including raw logs and research scripts. Rerunning
research commands overwrites results; preserve the committed run first. Negative
last-Box and source traces deliberately use fresh entropy, so their raw values
need not recur on another run.

Shared checks: **42 passed**, one optional ALE warning. Ruff, whitespace and strict
MkDocs/link checks pass. Initial lint import-order findings were fixed. The final
preservation audit verifies all **1,981 starting tracked files byte-for-byte**,
including documentation, candidate-State report, both matrix/map files, frozen
renderer, shared code and all accepted-family code/evidence. Closure remains
33 accepted / 170 artifact hashes / 11 terminal logs / 75 Fetch-manifest files.
No promotion and no push. Stop for review.

Exact changed-file inventory (all new; no existing files modified):

- `docs/research/ogbench_rng_feasibility.md`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/SHA256SUMS`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/audit.py`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/closure.json`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/closure.log`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/codec.log`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/codec_gate.py`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/development-comparator.log`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/development-semantic-view.log`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/entropy_traces.json`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/final_diff.json`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/formatting.log`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/gates.log`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/gates.py`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/mkdocs.log`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/numeric_rng_results.json`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/ownership.py`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/provenance.json`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/replay_results.json`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/rng_codec.py`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/schedule_results.json`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/schedules.log`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/schedules.py`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/semantic-preservation.log`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/semantic_preservation.json`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/semantic_preservation.py`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/shared-tests.log`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/trace.log`
- `docs/research/semantic_recovery_evidence/ogb_rng_feasibility/trace_entropy.py`
