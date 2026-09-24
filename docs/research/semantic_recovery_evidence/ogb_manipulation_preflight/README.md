# Cube/Scene source preflight: no promotion

This evidence is a bounded investigation of the pinned authoritative sources,
not a replacement-backend acceptance suite. Both registrations remain INCOMPLETE.
The [candidate schema and restore order](../../ogbench_manipulation_candidate_state.md)
were written before production code; no production code was subsequently changed.

## Profile and reproducibility

SWM: `176d54532b3c0a251a4f509b74f2fdacc143a63d`. CPU interpreter:
`/private/tmp/ogb-manip-authority-swm-lock/bin/python`, Python 3.12.11, macOS arm64.
The isolated profile installs OGBench 1.2.1, dm-control 1.0.41, SciPy 1.17.1,
lxml 6.1.1, dm-env 1.6, dm-tree 0.1.9 and labmaze 1.0.6. Its `authority-base.pth`
reads the previously pinned PushT and Fetch site-packages, in that order, supplying
MuJoCo 3.8.1, Gymnasium 1.2.3, NumPy 2.2.6 and the SWM import dependencies.
Those existing environments were not modified. Exact runtime versions, source,
assets, native-library hashes and file paths are in `provenance.json` and
`dependency_hashes.json`; the latter includes SWM's `world/world.py`.

CPU imports require `MUJOCO_GL=disable` on this non-GUI execution session.
Without it, dm-control's GLFW initialization aborted an initial exploratory import;
no physics ran in that failed import. The documented disabled renderer backend
allows CPU physics. No CPU or GPU frames were generated. Logs retain the optional
ALE, Gym precision/casting, and duplicate SDL class warnings from SWM imports.

From the integration worktree, run:

```sh
MUJOCO_GL=disable SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel:. /private/tmp/ogb-manip-authority-swm-lock/bin/python docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/probe.py
MUJOCO_GL=disable SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel:. /private/tmp/ogb-manip-authority-swm-lock/bin/python docs/research/semantic_recovery_evidence/ogb_manipulation_preflight/reset_effects.py
```

These research scripts overwrite their result JSON files. Copy the evidence
folder first to preserve the recorded run. Because the source asks for fresh OS
entropy, a rerun is not expected to reproduce those raw random numbers or their
hashes. The structural source findings do not depend on particular draws.

The exact terminal markers mean **diagnostic completion only**:

- `OGB_MANIPULATION_PREFLIGHT_COMPLETE_NO_PROMOTION`
- `OGB_RESET_EFFECTS_RECORDED_NO_PROMOTION`

They must never be used as acceptance markers.

## Results and limits

| Check | Cube | Scene |
| --- | --- | --- |
| Independent source resets: seeds 0,17,42 × task IDs 1–5, fixed Variation | 15/15 differ | 15/15 differ |
| Same reset after conditioning both persistent PCG64s and global task RNG | 3/3 differ | 3/3 differ |
| Source candidate roundtrip into independent different-history target | 3/3 exact | 3/3 exact |
| Eight ordinary actions after each restored contact/controller history | 24/24 exact candidate States and complete source step outputs | 24/24 exact candidate States and complete source step outputs |
| Unseeded reset after restoring candidate engine/task/persistent RNG anchors | 3/3 differ | 3/3 differ |
| Frozen reference constructor on pinned profile | Constructs; no frame claim | TypeError: missing `env_type` |
| Public adapter, public input validation/nonaliasing, complete tasks/Variation gates | NOT_RUN_PREFLIGHT_STOP | NOT_RUN_PREFLIGHT_STOP |
| Full/private rendering, rendering nonmutation, frozen GPU regression | NOT_RUN_PREFLIGHT_STOP | NOT_RUN_PREFLIGHT_STOP |

The 16-action source history descends to the cube and closes the gripper while
the target follows a different task/controller history. Contact names are retained.
All Cube histories have gripper–cube and floor–cube contacts. Scene seeds 0 and 17
have gripper–cube contacts; Scene 42 has floor–cube and other robot/scene contacts,
so it must not be described as a third gripper–cube witness.
Only the target is restored before the eight-action comparison. The source
continues untouched. Actions alternate float64/float32 with simultaneous
translation, yaw and gripper components. This is not a full action-boundary gate.

The diagnostic captures all 14 named MuJoCo integration components, site caches,
contact-force diagnostics, enumerated Python task bookkeeping, mutable model
arrays and environment/global RNGs. It replays under fixed default Variation.
It does not provide a production TensorDict schema, input validator, per-lane
randomness isolation, general custom-task support, or Variation RNG serializer.
The source resets Variation generators with `seed(None)` even when no leaves are
sampled; those unused draws are excluded from comparisons. That exclusion does
not explain the observed differences in MuJoCo controls and warmstart.

## Reset blocker, distinct from contact restoration

Cube's wrapper accepts `seed` but does not forward it. Scene forwards the seed.
Both task initializers settle the goal with two `action_space.sample()` calls;
`ManipSpaceEnv.action_space` creates a new unseeded Box on every access. Seeding
`env.action_space` therefore seeds a discarded object. Future generators do not
exist in a pre-reset environment snapshot. Goal settling leaves its controls,
warmstart and goal cache behind even after initial qpos/qvel are reinstated.

The conditioned-reset diagnostic explicitly aligns persistent PCG64/global RNG
state. This removes Cube's dropped-seed difference but does not align freshly
created action-space generators. Seed 0 Cube still differs in controls by
`179.794279634507` and warmstart by `78.37517704531541` in the recorded run.
Scene seed 0 similarly differs in controls by `0.7052544822424807` and warmstart
by `23.23104609966932`. Raw arrays and all differences are retained.

`reset_effects.json` separately tests four zero-action physical transitions after
new conditioned reset pairs. All three Cube pairs and Scene seeds 0/42 have equal
qpos/qvel for all four steps in this run. Scene seed 17 differs at step one by
`2.0990154059319366e-16` in qpos and `6.367129046225273e-14` in qvel; thresholds
were not relaxed. Reward/success/done agree in these bounded cases. Thus there is
no demonstrated Cube ordinary-step physics failure in this evidence, and no claim
that every fresh warmstart difference necessarily changes later qpos/qvel.

The required exact reset-State gate is not satisfied by the unmodified sources.
Deterministic per-environment control of newly created entropy sources, or a
revised oracle-conditioning policy, needs explicit review before a family adapter
can claim that gate. The renderer already has a temporary class-descriptor
`_seeded_action_space` patch for sampling render corpora; this experiment does not
silently promote that workaround into authoritative reset semantics. No shared
contract change is requested. This is a bounded reset acceptance blocker, not a
proof that MuJoCo state cannot be restored.

Scene additionally requires a reviewed frozen-renderer constructor fix or a
reviewed compatible renderer profile. The omitted `env_type` defect predates this
checkpoint and is already recorded in `first_family_recovery.md`; this run
reproduces it. The upstream Scene `set_state` signature also differs from the
frozen renderer's per-button keyword call; further compatibility inspection is
required, not assumed fixed by the constructor argument alone.

## Remaining source-derived requirements

Each raw registration has no `max_episode_steps`; SWM World's default 100-step
wrapper belongs to downstream orchestration. Raw truncation is false. Success
terminates by default. Default reward is 1 on success, else 0; explicit reward
task IDs use number of successful entities minus total entities. ID 0 aliases 2;
IDs 1–5 and custom task info must receive separate task acceptance coverage.
Default omitted task ID uses global NumPy task sampling, not seeded env PCG64.

The controller unnormalizes using the exact source affine expression with ranges
(.05,.05,.05,.3,1), then clips workspace/yaw/gripper targets. It does not first
clip normalized actions. Dynamics advance .05/.002 = 25 substeps. Cube success
uses position distance <= .04 and ignores orientation. Scene also requires
button equality and drawer/window distance <= .04. Button crossing is previous
position > -.02 and current <= -.02; it changes damping between 1e6 and 2.
Scene instability recovery consumes env RNG. These statements come from source;
threshold-neighbor, default-task, button-toggle and instability acceptance tests
were not completed after the reset blocker.

Physics-affecting Variation includes cube size and reset end-effector position.
Visual leaves include colors, camera angle and light intensity; visual physics
nonmutation, persistent/reset-scoped behavior, and renderer support remain
unproven. Data-collection mode, non-single Cube configurations, custom task info,
rendered goals and multiview remain outside these bounded experiments. They must
not be certified by association.

The full production/GPU gates remain outstanding. No GPU job was submitted,
queued or cancelled. CPU preconditions were not met, so there is no claim of
GPU unavailability or GPU-profile physics parity.
