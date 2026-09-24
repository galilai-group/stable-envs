# Cube/Scene CPU compatibility implementation plan

Recorded before production edits, under the reviewed narrow-subclass decision.
Starting clean worktree `/private/tmp/stable-envs-semantic-recovery`, branch
`repair/environment-semantics`, HEAD/dependency audit
`75ef812977423fa059b857b36542119e101a430f`. RNG checkpoint
`434cc72974360ab98e47cad7aebcc8b1b62bb1de`. Both adapters currently
NOT_IMPLEMENTED; matrix 33 VERIFIED / 9 INCOMPLETE. No push.

Oracle remains immutable SWM `176d54532b3c0a251a4f509b74f2fdacc143a63d`;
upstream dependency remains unmodified OGBench 1.2.1. Exact definition lines and
hashes are recorded in the dependency-boundary inventory. In particular:

| Source | SHA256 |
|---|---|
| SWM Cube | 143e5e81f424ff7a4d7fdde287dee88b722c09082a6fc9e68721839788e94c19 |
| SWM Scene | d0eebdce0131f86e3247736c601f97c2b11d0dbfad56fca998df3192f08a7b9f |
| Upstream Cube | 6f8457ffd634f6620411c788c86e871f44ea5b5b5eecf76cd694fc04d12d678d |
| Upstream Scene | b03395bc50d7a65fcd45a3141fe6cca419f07cbe7422d397003d824fdfcb7c54 |
| Upstream ManipSpaceEnv | 03524eb1c95155bbc2ec3d29e9836fccffe8eb9fce1be15b6f0e7062cc1bb5fd |

Categories: 1 excluded registration mode/API; 2 existing configuration/hook;
3 delegate with pre/post adjustment; 4 authorized adapted semantic body.

| Differing SWM method (both families unless marked) | Category | Planned treatment |
|---|---:|---|
| `__init__` | 2 | Single/scene task mode, states, fixed defaults; stable-envs Variation tree; reject other modes |
| `set_tasks` | 2 | Inherit unchanged upstream numeric tables, including reward-task alias |
| `reset` | 3 | Validate options, seed owned streams, select default task through task_id, configure Variation, delegate; raw historical state option not supported |
| `initialize_arm` | 4 | Shared 11-normalized-line source body: replace uniform position draw with Variation value; retain yaw, upstream Lie transforms/IK, assignment and forward |
| `initialize_episode` | 3 | Pre-initialize immediate-output cache as SWM does, delegate whole task reset/settling, apply required cube colors |
| `set_new_target` | 1 | Data-collection-only; reject mode, inherit unused implementation |
| `add_objects` | 2 | Delegate upstream assembly, add only required camera configuration via hook; no assets copied |
| `compile_model_and_data` (inherited lifecycle hook) | 3 | Delegate compilation, retain numeric configuration used by compiled model; source may change MJCF without recompiling |
| `modify_mjcf_model` | 2 | Existing hook applies physical size and appearance, with source-compatible dirty lifecycle |
| `post_step` | 3 | Delegate all physics/success/button logic, then restore configured cube colors |
| `get_reset_info`, `get_step_info` | 3 | Delegate upstream observation/info generation; adapt goal/env-name/success keys |
| `render` | 2 | Configure front_pixels camera; rendering acceptance remains deferred |
| `render_multiview` | 1 | Not ported |
| Cube `set_target_pos`; Scene `set_cube_target_pos`, `set_target_button_state`, `set_target_drawer_pos`, `set_target_window_pos` | 1 | No convenience setters ported; State restore uses numeric fields |
| Scene `set_state` | 2 | Use upstream button-array signature internally; public State restore is adapter-owned |
| Scene `_apply_button_states` | 3 | Delegate damping/forward; apply lock color adjustments afterwards |

Methods with identical executable AST remain inherited: post-compilation object
lookup, success/reward, object info, observation/oracle observation; Scene
pre_step and drawer containment. Normal step is inherited directly. No task,
controller, IK, physics, observation construction, XML or mesh duplication.

Only the arm-reset body is initially authorized as category 4. Its source body
is 19 physical lines / 11 normalized lines, identical between SWM families; one
position-source statement differs from upstream. A super call cannot inject that
position without extra RNG draws before yaw or repairing already-computed goal
settling afterwards. Exact controlled reset and trajectory parity will verify
this override, including seeds 0/17/42 and fixed/sampled start positions.
Other hooks configure values or delegate transitions; they must not grow into
copied episode or post-step algorithms. Any additional category-4 need requires
review rather than silently expanding this plan.

Variation leaves are restricted to registered physical/reset/appearance values.
Historical Cube start/goal leaves are retained for its default reset sampling
and replay even though task placement ignores them. Scene's corresponding
inactive leaves may be retained privately only to preserve Gym Dict child-seed
allocation; they are not promised data-collection controls. No SWM spaces code
will be copied. Numeric State captures current/reset-anchor values and all owned
RNGs. Unsupported options/modes fail explicitly before mutation.

Implementation files will be confined to new OGBench manipulation adapter,
compatibility/Variation/RNG helpers, public semantics module and family package
registration wiring. `manipulation.py` remains byte-identical. Evidence/tests and
this plan/report are separate from accepted evidence; matrix/map stay unchanged.
The final report will record actual overrides/line counts, schema, restore order,
commands/results and every file changed.
