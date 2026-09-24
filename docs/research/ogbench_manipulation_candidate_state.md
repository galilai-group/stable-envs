# Cube/Scene candidate State, before production implementation

This is a source-derived research proposal, not an accepted public schema. It was
written at documentation checkpoint `03adc1ab79bae71e3e62db2e5c4cb19f20723166`,
before any manipulation production edits. The authority is stable-worldmodel
`176d54532b3c0a251a4f509b74f2fdacc143a63d`, with OGBench 1.2.1, MuJoCo 3.8.1,
Gymnasium 1.2.3 and NumPy 2.2.6. Neither registration is promoted.

## Engine and controller fields

All floating fields below use float64. Shapes are per simulator, before a batch
dimension. The candidate covers the registered single Cube and one-cube Scene.

| Named field | Cube | Scene | Reason |
| --- | --- | --- | --- |
| physics.time | scalar | scalar | Integrator clock |
| physics.qpos | 21 | 25 | Robot, gripper, objects, Scene joints |
| physics.qvel | 20 | 24 | Generalized velocities |
| physics.act | 0 | 0 | No activation state in these models |
| physics.history | 0 | 0 | No actuator history in these models |
| physics.qacc_warmstart | 20 | 24 | Constraint solver initial state |
| physics.ctrl | 7 | 7 | Arm and gripper actuator controls |
| physics.qfrc_applied | 20 | 24 | Applied generalized forces |
| physics.xfrc_applied | 26 × 6 | 38 × 6 | Applied body forces |
| physics.eq_active | 3 bool | 3 bool | Active equality constraints |
| physics.mocap_pos | 1 × 3 | 1 × 3 | Cube goal position |
| physics.mocap_quat | 1 × 4 | 1 × 4 | Goal orientation; retain even when default identity |
| physics.userdata / plugin | 0 / 0 | 0 / 0 | Empty in pinned models |
| controller.site_xpos | 5 × 3 | 11 × 3 | Cached pinch pose read before integration |
| controller.site_xmat | 5 × 9 | 11 × 9 | Cached pinch orientation read before integration |

Use MuJoCo's named `mjSTATE_*` components, whose union is
`mjSTATE_INTEGRATION`, rather than guessing a subset from Fetch. The diagnostic
also retains `cfrc_ext` (26 × 6 / 38 × 6) for source diagnostic nonmutation and
exact source-output comparison; it is not established as future physics state.

Cube does not call `mj_forward` in `post_step`; its controller reads the cached
site pose from the end of the last integration stage. Forwarding during restore
and keeping only the recomputed cache can change the next action's target.
Scene calls `mj_forward` through `_apply_button_states` each post-step.

`_target_effector_pose` is overwritten before use by each `set_control`.
`DiffIKController.solve` starts from supplied current joint positions; its error,
Jacobian and update workspaces are recomputed before use. These are not assumed
to require persistent controller state. The diagnostic excludes that overwritten target pose; it does not declare it
necessary for subsequent source transitions.

## Task, reset, lifecycle and RNG

The candidate needs active task ID (nullable, integer), reward task ID (nullable,
integer), task/reset payload (`cur_task_info`), success (bool), target task enum,
block index, and the fixed mode, success timing and termination configuration.
For predefined Cube tasks the payload contains init/goal arrays (1 × 3 each).
For Scene it contains init/goal cube positions (1 × 3), button bits (2), drawer
and window positions (scalars). Custom task payload validation is unproven.

Scene additionally needs current and target button states (2 integer bits), target
button index, target drawer/window positions, and target-site positions derived
from those targets. Current buttons determine drawer/window damping (1e6 locked,
2 unlocked), and button/handle colors. Restore those model effects before forward.

The source goal cache `_cur_goal_ob` is 28 / 40 float64 values in default mode.
It is returned as `info['target']` and must be preserved if that output is exposed.
It does not drive the task reward or next physical transition. A source diagnostic
may retain it; no learner observation class or filtering API is proposed.
Rendered goal caches remain SWM-owned; `render_goal=True` is not tested here.

`_prev_qpos`, `_prev_qvel`, `_prev_ob_info`, and Scene `_prev_button_states` are
replaced in `pre_step` before `post_step` consumes them. They matter to immediate
source info reads, so the diagnostic retains them. A production schema may omit
only after proving the precise public output boundary. Reset anchors are the
persistent configuration and active task payload, plus complete relevant RNG.

Retain `_reset_next_step`, `_dirty`, `_never_compiled` booleans and validate that
public checkpoints represent initialized, compiled models. No raw registration
TimeLimit or elapsed-step counter exists. SWM World applies its separate default
100-step cap; do not invent that counter in this raw simulator.

Persistent randomness includes environment PCG64 state and seed metadata, global
NumPy MT19937 task-selection state, and Variation values/RNGs. A public TensorDict
encoding would require validated named numeric leaves or a validated byte encoding
for RNG state (as in accepted adapters), not arbitrary Python object handles.

**Unresolved source entropy:** Cube drops the public `seed` argument. Both sources
call `self.action_space.sample()` twice during task goal settling. Each property
access constructs a new unseeded Gym Box, so the RNG does not exist at checkpoint
time and cannot be captured from the environment. Variation reset also calls
`seed(None)`. Fixed `options={'variation': []}` removes Variation sampling from
the reset witness but does not remove the goal-settling entropy. Do not silently
borrow the renderer's class-descriptor seeding patch for authoritative semantics.
An exhaustive replayable reset State is not established until this is resolved.

## Proposed restore order and validation

1. Validate all fields, keys, shapes, dtypes, finite ranges, RNG payloads, task IDs,
   button bits and matching persistent model/configuration before mutation.
2. Construct/compile the independently owned authoritative model with matching
   persistent Variation. Reject unsupported model/configuration mismatches.
3. Restore task/lifecycle values and Scene buttons/targets; derive damping and
   target-site positions using authoritative code. Do not call `post_step`, which
   can toggle buttons, consume RNG, and change success.
4. Restore every integration component with MuJoCo's state API; run `mj_forward`
   to rebuild derived engine data.
5. Reapply integration components (especially warmstart changed by forward), then
   source cached site poses and retained diagnostics. Restore RNG last.
6. Compare the complete snapshot, then independent one/multi-step continuation
   from different contact/controller histories. Ordinary step calls authoritative
   `step` directly and never restores a checkpoint implicitly.

The research prototype tests this order only for bounded source instances; it is
not a validated production `set_state` implementation. Whole-State rejection,
public adapter wiring, all configurations, Variation, render equality and GPU
acceptance remain gates, even if a bounded continuation probe passes.
