# Representative Fetch: initial source audit

Selected historical registration: `swm/FetchPickAndPlace-v3`.
Pinned SWM: `176d54532b3c0a251a4f509b74f2fdacc143a63d`.
New bounded worktree: `/private/tmp/stable-envs-fetch-recovery`, branch
`repair/fetch-representative`, based on reviewed integration
`907c6e0f45f0763ac736cea006fb762c6e802ba2`.

This task exercises object manipulation and an unblocked gripper. Only this
registration is selected; no sibling adapter or certification has been added.
Oscar classic acceptance reached the requested checkpoint A during this initial
audit, so the report deliberately stops before a Fetch implementation or State
sufficiency claim.

## Authority and task identity

The pinned SWM registration passes `env_id='FetchPickAndPlace-v4'` to
`FetchWrapper`, retaining its default `flatten=True`. Installed Gymnasium
Robotics 1.4.2 registers that upstream ID to `MujocoFetchPickAndPlaceEnv` with
sparse reward and a 50-step TimeLimit. Production recovery should import
Gymnasium Robotics/MuJoCo directly, never SWM. The historical v3 suffix is not
an instruction to load an upstream v3 task.

Source constructor settings are has_object=True, block_gripper=False,
n_substeps=20, gripper_extra_height=0.2, target_in_the_air=True, obj_range=0.15,
target_range=0.15 and distance_threshold=0.05. SWM concatenates the 25-component
upstream observation and desired goal, casting the resulting 28-vector to
float32. Its info retains upstream proprio and desired-goal arrays. This is an
observation format, not a simulator snapshot.

## Source-defined action, reset and lifecycle behavior

These findings come from inspected implementation, not a passed candidate gate:

- Upstream checks four-dimensional action shape and applies `np.clip` to [-1,1]
  before the controller. It does not coerce all inputs to float32: the action
  arithmetic can retain float32 versus float64 precision. Both need parity tests.
- XYZ control is multiplied by 0.05. Fixed rotation `[1,0,1,0]` and the duplicated
  gripper scalar are concatenated into the control vector. Actuator controls are
  applied, mocap is reset against the welded body and incremented by position and
  quaternion control; MuJoCo advances 20 substeps.
- Reset calls `mj_resetData`, restores initial time/qpos/qvel, samples object XY
  within +/-0.15 of initial gripper XY until distance is at least 0.1, then
  forwards physics. Goal sampling uses a separate offset, fixes height to the
  object height, and with probability 0.5 adds height sampled in [0,0.45].
- Sparse reward is `-(distance > 0.05)` as float32; success is
  `(distance < 0.05)` as float32. At exactly the threshold, the source formulas
  imply zero reward and false success. Dense reward is negative distance, but
  the dense sibling is outside this first scope.
- Base termination and truncation return false. Upstream registration supplies
  the 50-step TimeLimit. Post-truncation behavior and RNG replay still require
  execution after resolving the construction issue below.
- SWM's default reset variations are visual. Its mass write is always applied;
  explicit object/goal position changes call `mj_forward`. Agent placement can
  modify `initial_gripper_xpos` before upstream reset. These mutations must be
  handled under the established persistent-Variation contract, not silently
  replaced with nominal defaults.

## Candidate State investigation scope — not a finalized schema

The compiled model, observed before the constructor failed, has nq=22, nv=21,
nu=2, na=0, nmocap=1. Candidate fields to test include qpos[22], qvel[21], time,
mocap position/quaternion, goal, elapsed TimeLimit counter, warmstart[21], ctrl,
applied forces, and upstream RNG state. No actuator activation exists in this
observed model. The need for each other field remains unproved; the RNG algorithm
has not yet been captured. Mutable goal/configuration effects also require
inspection. No DMC schema has been copied.

Omission experiments, different-history restoration, candidate reset/step/reward
parity, lifecycle/RNG replay, physical/visual Variation and frozen renderer
preservation have **not run successfully**. No Fetch row is promoted and no
production code changed.

## Concrete local reference-construction blocker

The existing `/Users/samib/se8-fetch-venv/bin/python` profile contains Gymnasium
Robotics 1.4.2, Gymnasium 1.3.0, MuJoCo 3.12.0, NumPy 2.5.3, Torch 2.14.0 and
TensorDict 0.14.2. Both the SWM wrapper and direct upstream construction fail at
`mujoco_utils.set_joint_qpos` before reset.

For the first joint, `robot0:slide0`, model joint type is `numpy.int32(2)` and
`int(mjJNT_SLIDE)` is also 2. The isolated comparison probe records:

```text
numpy scalar == enum: true
enum == numpy scalar: false
numpy scalar in (HINGE enum, SLIDE enum): false
integer tuple membership: true
```

The upstream tuple-membership assertion therefore rejects this slide joint in
this profile. No library monkeypatch or site-package edit was made to bypass it.
The diagnostic wrapper only logged arguments before calling the original failing
function. This is a concrete dependency/interface compatibility issue, not
evidence that the shared SimulatorAdapter/State contract cannot express Fetch.
A compatible, explicitly recorded upstream profile is the next prerequisite.

Scripts and original stdout/stderr are preserved in
[`semantic_recovery_evidence/fetch_initial`](semantic_recovery_evidence/fetch_initial/).
`source_probe.stdout.txt` is empty because construction failed before its JSON
result; it must not be interpreted as a passed source audit. The construction
diagnostic and enum comparison localize the failure. No private-ABI PushT work,
other simulator family or final benchmark was started.
