# Cube/Scene dependency-boundary audit

Stage A **FAILS the requested boundary** on OGBench 1.2.1. Stage B was not started.
The pinned SWM classes contain their own concrete reset/target implementations;
they are siblings of upstream Cube/Scene, not wrappers around those concrete
classes. Much of that code is unchanged upstream code. The blocking changes are
small at individual sampling sites but lie inside reset/task algorithms that
this checkpoint expressly forbids stable-envs to reproduce. This is not a claim
that SWM has a different physics engine, or that every copied line is necessary.

Review requested: approve an upstream dependency remedy before production work.
No shared-contract change is requested.

## Starting point and disposition

- Branch: `repair/environment-semantics`.
- Worktree: `/private/tmp/stable-envs-semantic-recovery`.
- Starting HEAD and RNG research/evidence checkpoint:
  `434cc72974360ab98e47cad7aebcc8b1b62bb1de`; initially clean.
- Documentation checkpoint `03adc1ab79bae71e3e62db2e5c4cb19f20723166`
  is an ancestor and its files remain unchanged.
- Oracle: `/Users/samib/stable-worldmodel`,
  `176d54532b3c0a251a4f509b74f2fdacc143a63d`; tracked source unchanged.
- Installed audit profile: OGBench **1.2.1**, MuJoCo **3.8.1**, Gymnasium
  **1.2.3**, NumPy **2.2.6**, dm-control **1.0.41**.
- Matrix remains **33 VERIFIED / 9 INCOMPLETE**. Both adapters remain
  `NOT_IMPLEMENTED`, CPU `NOT_ACCEPTED`; no promotion occurred.
- No production, shared contract, renderer, matrix, accepted Maze, or other
  family code changed. No simulator/controller/task implementation was copied
  into production. AST inspection found no `stable_worldmodel` production import.
- No push was performed. Previous checkpoint records also say no push; this
  audit does not claim remote-server history was independently inspected.
- One separate audit/evidence commit contains this report and its evidence.
  Its exact SHA is given in the final handoff (a commit cannot embed its own SHA).
  No implementation commit exists. Stage-B State schema, restore order, public
  RNG encoding and CPU acceptance results are **not implemented / not run**.
  The earlier candidate schema and Design A research remain unchanged.

## Actual runtime graph and source ownership

SWM registration is in `stable_worldmodel/envs/__init__.py:40` and `:45`, with
empty kwargs, no registered TimeLimit, and entry points:

- `stable_worldmodel.envs.ogbench.cube_env:CubeEnv` (default `env_type='single'`).
- `stable_worldmodel.envs.ogbench.scene_env:SceneEnv` (default `env_type='scene'`).

The complete MRO of each SWM class is:

```text
stable_worldmodel.envs.ogbench.{cube_env.CubeEnv|scene_env.SceneEnv}
ogbench.manipspace.envs.manipspace_env.ManipSpaceEnv
ogbench.manipspace.envs.env.CustomMuJoCoEnv
gymnasium.core.Env
typing.Generic
abc.ABC
builtins.object
```

Upstream concrete classes are
`ogbench.manipspace.envs.cube_env.CubeEnv` and
`ogbench.manipspace.envs.scene_env.SceneEnv`; their MRO has that upstream concrete
class first, followed by exactly the six shared entries above. Runtime
`issubclass(SWMCube, UpstreamCube)` and the Scene equivalent are both false.
There is no hidden intermediate SWM simulator superclass.

Actual default `gym.make` wrapper order, outermost first, is
`gymnasium.wrappers.common.OrderEnforcing` →
`gymnasium.wrappers.common.PassiveEnvChecker` → the SWM concrete class.
The checker inspects `observation_space`, which invokes reset if no model
exists. The audit observed a compiled model during construction in both families;
direct class construction leaves `_model is None`. No `TimeLimit` is installed;
SWM World's separate 100-step cap is not part of these raw registrations.

| Component | Actual owner | Difference from upstream concrete environment |
|---|---|---|
| Model XML, meshes, textures | Upstream `manipspace/descriptions`; base `build_mjcf_model`, compiler `CustomMuJoCoEnv` | Same source assets; SWM modifies assembled MJCF, including physical cube size |
| Object assembly | SWM `add_objects`, copied from concrete upstream | Extra cameras; same object source files and attachment logic |
| Control and IK | Upstream `ManipSpaceEnv.set_control`; `controllers.diff_ik.DiffIKController.solve`; upstream `lie` | No SWM controller/IK replacement |
| `step()` | Upstream `ManipSpaceEnv.step` | Same controller → pre-step → `mj_step(nstep=25)` → `mj_rnePostConstraint` → post-step sequence at defaults; virtual hooks differ |
| `reset()` | SWM wrapper → upstream `ManipSpaceEnv.reset` → `CustomMuJoCoEnv.reset` → Gym seed path | Variation reset; Cube drops seed; optional raw-state override; Scene forwards seed |
| Episode/arm initialization | SWM `initialize_episode` / `initialize_arm` | Position sources and RNG consumption differ; no upstream sampling injection hook |
| Task tables | SWM `set_tasks`, copied from upstream concrete class | Numeric tables identical for all five Cube sizes and Scene; reward-task 0 aliases 2 |
| Task selection | Upstream `ManipSpaceEnv.reset` | Same global `np.random.randint`, explicit task/custom/reward routes |
| New target selection | SWM `set_new_target` | Data-collection goal position/yaw supplied by Variation, not environment uniform draws |
| Reward/success | SWM concrete `compute_reward` / `_compute_successes`, unchanged AST; default reward upstream base | Formulas/thresholds unchanged; initial states differ |
| Post-step | SWM concrete `post_step` | Removes success recoloring; Scene button toggles, instability recovery and success logic retained |
| Action/observation spaces | Upstream `ManipSpaceEnv` properties | Fresh action Box on every access; observation property can reset during construction |
| Observation arrays/info | SWM concrete observation functions, unchanged AST; base `compute_ob_info` | Arrays unchanged at identical input state; SWM info adds `env_name`, `target`, `success` instead of reset `goal`/`goal_rendered` |
| Variation | SWM `spaces.Box`, `spaces.Dict`, `reset_variation_space`; Gym sampling/seeding underneath | No upstream OGBench Variation object or equivalent value consumers |
| Rendering | SWM render wrappers → upstream `ManipSpaceEnv.render` → `CustomMuJoCoEnv.render` | SWM defaults to `front_pixels`, supports multiview, alters cameras/light/materials; no rendering performed here |

The runtime record lists **every method/property owner**, its exact source path,
definition line and file hash, not just the components in this table:
[dependency graph](semantic_recovery_evidence/ogb_dependency_boundary/dependency_graph.json).
The [provenance inventory](semantic_recovery_evidence/ogb_dependency_boundary/provenance.json)
rechecks all 483 prior source/dependency/asset hashes and adds wrapper/registration
file hashes. It includes SWM spaces/utilities and upstream base, concrete classes,
controller, Lie/MJCF utilities, Gym and MuJoCo sources.

Both concrete files import `mujoco`, `numpy`, `dm_control.mjcf`, upstream
`ogbench.manipspace.lie` and `ManipSpaceEnv`. SWM additionally imports its own
`spaces` and `envs.utils.perturb_camera_angle`. They do not import the upstream
concrete Cube/Scene classes.

## Exact differences and copy size

Complete zero-context unified diffs preserve documentation, imports and all code:
[Cube exact diff](semantic_recovery_evidence/ogb_dependency_boundary/cube.exact.diff),
[Scene exact diff](semantic_recovery_evidence/ogb_dependency_boundary/scene.exact.diff).
Complete method diffs remove docstrings/comments and normalize Python formatting;
methods added relative to the concrete class are compared against their inherited
upstream implementation where one exists:
[Cube normalized diff](semantic_recovery_evidence/ogb_dependency_boundary/cube.normalized.diff),
[Scene normalized diff](semantic_recovery_evidence/ogb_dependency_boundary/scene.normalized.diff).

| Measure | Cube | Scene |
|---|---:|---:|
| SWM full source lines, including documentation | 1,670 | 1,366 |
| Upstream concrete full source lines | 815 | 709 |
| SWM directly defined methods | 20 | 27 |
| Exact executable AST matches (methods / normalized lines) | 6 / 51 | 8 / 101 |
| Task table differs only in dict spelling (methods / normalized lines) | 1 / 13 | 1 / 4 |
| Changed methods (count / normalized lines) | 11 / 243 | 13 / 351 |
| SWM-only methods (count / normalized lines) | 2 / 14 | 5 / 22 |
| All method normalized lines | 321 | 478 |

These are reproducible `ast.unparse` line counts, not estimates of unique new
logic. Long task tables and Variation definitions collapse onto single lines.
A wholesale transplant would copy **3,036 source lines / 799 normalized method
lines**, including unchanged code and documentation. That is unnecessary and
prohibited. Nor do the 594 lines in changed methods represent 594 changed lines.

The reset/target methods that contain the semantic changes total **283 normalized
lines**: Cube `initialize_arm` 11 + `initialize_episode` 62 + `set_new_target` 39;
Scene 11 + 86 + 74. Copying those six bodies would duplicate substantial reset,
IK setup and target-selection logic. Most statements in them are unchanged.
For default task mode alone, the immediate missing behavior is concentrated in
one **11-line normalized / 19-line physical `initialize_arm` body**, identical
between SWM families, with **one changed position-source statement** relative to
upstream. This audit does **not** claim the entire 283-line surface must be copied
for default task mode. Even that smaller copy reproduces the reset pose/IK setup
procedure, rather than invoking an upstream operation accepting the desired pose,
which fails the explicit no-reset-algorithm-reimplementation rule.

Configuration/model hooks and output wrappers could handle several other
differences narrowly. Those differences alone would not justify a failed gate.

### Semantically consequential differences

1. **Arm reset, including default task mode.** Upstream
   `ManipSpaceEnv.initialize_arm(self)` samples 3-vector position and then yaw.
   SWM uses `variation_space['agent']['ee_start_position'].value`, default
   `[0.425, 0, 0.275]`, and draws only yaw. The method then performs pose transforms,
   IK, joint assignment and forward. There is no position argument or delegated
   pose-initialization operation. Task reset invokes it before goal settling and
   again before the start state. Adjusting the pose after reset cannot repair the
   already generated goal observation or the consumed draws.

   Degenerate position bounds preserve position but consume three draws. At seed
   17, the SWM yaw primitive yields `2.168168867985815`; upstream with degenerate
   bounds yields `-0.8288781722573246`. Seeds 0 and 42 also differ. This bounded
   primitive test is a concrete configuration counterexample, **not** an end-to-end
   parity claim. Spoofing `Generator.uniform` or rewinding the generator around
   selected calls would change the validated RNG integration design, not supply
   the missing upstream configuration hook; it was not implemented.

2. **Data-collection reset/target construction.** Both SWM families read cube
   start XY/yaw and non-stacked target XY/yaw from Variation. Upstream draws them
   inline from its environment RNG. Scene retains its drawer-placement/stacking
   branches; only the ordinary target branch uses Variation XY. SWM target yaw
   always uses Variation. Task-mode cube positions still come from the same task
   tables, with the same jitter; sampled cube start/goal Variation is not applied
   to that task-mode placement. Data collection is a constructor-exposed source
   path, not an accepted stable-envs feature. The default arm divergence already
   blocks the gate independently of whether that broader mode is promised.

3. **Physical model configuration.** Both use the very same 70-file upstream
   descriptions tree (13 XML files); the inventory hashes each asset. Base model
   assembly uses `floor_wall.xml`, UR5e `ur5e.xml`, Robotiq `2f85.xml`; Cube adds
   `cube_outer.xml` and per-cube `cube_inner.xml`; Scene adds `cube.xml`,
   `buttons.xml`, `drawer.xml`, `window.xml`. XML references supply shared meshes
   and button components. No SWM-local alternate mesh/XML is loaded.
   SWM `modify_mjcf_model` writes Variation cube sizes into physical and target
   geoms, then marks the model dirty. Default float32-derived size is
   `0.019999999552965164` versus upstream `0.02`. Actual compiled default models
   differ in mass (`0.07935999467849743` versus `0.07936000000000001`) and inertia,
   in both families. This is a measurable physical difference, not just rendering.
   It can use the existing upstream model-modification hook; no physics rewrite
   is needed. The source's `agent_color_changed = np.allclose(...)` also causes
   dirty/recompile behavior even when the agent color is unchanged.

4. **RNG/Variation lifecycle.** Cube drops its explicit `seed`; Scene forwards it.
   Both call SWM Variation reset with `seed=None`. SWM Dict supplies ordered dotted
   sampling, reset-to-initial values and reset-scoped overrides; Gym Dict supplies
   recursive seeding. Cube defaults to four start/goal samples; Scene to none.
   Upstream has no such tree. The validated research `OwnedCube`/`OwnedScene`
   inherit **SWM** concrete classes and `OwnedVariation` inherits **SWM** Dict.
   Their evidence remains valid for that research boundary; it proves neither
   upstream concrete-class parity nor a SWM-free Variation dependency.

5. **Output/render/configuration differences.** Extra camera counts are 4 versus
   2 for Cube and 3 versus 2 for Scene. SWM changes floor, agent, cube, lock colors,
   lighting and camera angles, removes cube success recoloring, adds multiview and
   target setters, and changes info keys. Scene's lock damping (1e6 / 2), button
   transition test and `mj_forward` remain unchanged; only colors change there.
   Scene `set_state` accepts per-button kwargs rather than upstream's button-state
   array, and its raw `options.state` route omits those required kwargs. Upstream
   constructors require `env_type`; SWM supplies defaults and applies `ob_type`
   after the base constructor. These are distinct from the reset sampling blocker.

The six task-table comparisons cover 30 tasks (25 Cube across five sizes, five
Scene) with exact numeric payload equality. Reward, success and observation
methods are executable-AST matches; the audit makes no claim that different reset
states produce equal trajectories merely because those formulas match.

## Remedy and review boundary

Preferred remedy: add and test **upstream OGBench reset sampling hooks**, then pin
an explicit reviewed release/commit. In particular, let upstream `initialize_arm`
accept or obtain a supplied end-effector position while keeping its yaw draw,
transforms, IK and forward sequence upstream. Equivalent overridable sampling
hooks can supply data-collection object/target XY/yaw without consuming replacement
environment draws. Retain upstream task tables, settling sequence, step, success,
reward and controller. Existing MJCF modification hooks can configure size and
visual properties without copying physics.

If upstream cannot host this, review a separately maintained reusable dependency
subclass, based on upstream concrete Cube/Scene, owning the necessary reset
extension and source-compatible Variation value interface. Merely relocating the
full SWM files into another package would not resolve the duplication. The
Variation dependency also needs a SWM-free boundary before the research wrapper
can be used in production. This report does not authorize that new dependency or
alter the accepted RNG design.

Resume only after review of this dependency remedy. Re-run Stage A against the
chosen exact upstream dependency, then carry out the requested Stage B using
Design A. Public exhaustive State, atomic validation/restore and CPU contact/parity
gates remain outstanding. Renderer repair and Oscar/GPU acceptance remain later,
separate gates; the pre-existing Scene renderer constructor/signature problems
are unchanged. No GPU submission or renderer threshold change occurred here.

## Audit execution and preservation

Evidence is under `semantic_recovery_evidence/ogb_dependency_boundary/`.
Commands run from the recovery worktree:

```sh
MUJOCO_GL=disable SDL_VIDEODRIVER=dummy PYTHONPATH=/Users/samib/stable-worldmodel:. /private/tmp/ogb-manip-authority-swm-lock/bin/python docs/research/semantic_recovery_evidence/ogb_dependency_boundary/audit.py
.venv/bin/python -m pytest -q tests/unit/test_full_state_contract.py tests/unit/test_semantic_harness.py tests/unit/test_replacement_readiness.py tests/unit/test_swm_matrix.py
.venv/bin/ruff check docs/research/semantic_recovery_evidence/ogb_dependency_boundary/*.py
.venv/bin/ruff format --check docs/research/semantic_recovery_evidence/ogb_dependency_boundary/*.py
.venv/bin/python -m mkdocs build --strict --config-file mkdocs.yaml --site-dir /private/tmp/stable-envs-ogb-boundary-docs
.venv/bin/python docs/research/semantic_recovery_evidence/ogb_dependency_boundary/closure.py
git diff --check
```

The audit logs default wrapper construction, both model configurations, six task
tables, exact diffs, provenance and three degenerate-bound counterexamples.
Preservation checks compare **all 2,010 files tracked at the starting SHA byte for
byte**, plus the 170 accepted artifacts, 11 terminal logs and 75 Fetch manifest
entries; the 27-file RNG evidence manifest is rechecked separately. No old audit
was rerun in a way that overwrites its evidence. Check results and the exact added
file inventory follow below.

## Complete method classification

Source ranges refer to the pinned SWM files; upstream definition owners and lines are in the runtime graph. Counts exclude docstrings and blank lines after AST normalization.

### Cube

| Method | SWM source lines | AST comparison | Classification |
|---|---:|---|---|
| `__init__` | 95–321 | changed | Configuration, Variation declarations; Scene previous-button initialization |
| `set_tasks` | 323–774 | dict_syntax_only | Unchanged task payloads; dict spelling only |
| `reset` | 776–842 | changed | RNG/Variation wrapper and raw-state reset route |
| `add_objects` | 844–921 | changed | Rendering camera additions; assembly otherwise unchanged |
| `post_compilation_objects` | 923–949 | identical_AST | Unchanged compiled object ID lookup |
| `modify_mjcf_model` | 951–1064 | changed | Physical size + rendering configuration + recompile lifecycle |
| `initialize_episode` | 1066–1195 | changed | Reset semantics in data collection; colors/cache initialization |
| `initialize_arm` | 1197–1215 | changed | Reset pose source/RNG consumption; IK setup copied |
| `set_new_target` | 1217–1316 | changed | Task target sampling semantics in data collection |
| `_compute_successes` | 1318–1344 | identical_AST | Unchanged success definitions |
| `set_target_pos` | 1346–1358 | SWM_only | Additional target configuration setter |
| `post_step` | 1361–1393 | changed | Rendering recoloring removed; physical/task logic unchanged |
| `get_reset_info` | 1395–1414 | changed | Output boundary keys |
| `get_step_info` | 1416–1435 | changed | Output boundary keys |
| `add_object_info` | 1437–1489 | identical_AST | Unchanged observation/info values |
| `compute_observation` | 1491–1540 | identical_AST | Unchanged observation construction |
| `compute_oracle_observation` | 1542–1570 | identical_AST | Unchanged oracle observation construction |
| `compute_reward` | 1572–1601 | identical_AST | Unchanged reward definition |
| `render` | 1603–1629 | changed | Rendering default camera |
| `render_multiview` | 1631–1670 | SWM_only | Rendering output configuration |

### Scene

| Method | SWM source lines | AST comparison | Classification |
|---|---:|---|---|
| `__init__` | 22–220 | changed | Configuration, Variation declarations; Scene previous-button initialization |
| `set_state` | 222–232 | changed | Button input signature/list representation; upstream physics restore delegated |
| `set_tasks` | 234–314 | dict_syntax_only | Unchanged task payloads; dict spelling only |
| `reset` | 316–345 | changed | RNG/Variation wrapper and raw-state reset route |
| `add_objects` | 347–399 | changed | Rendering camera additions; assembly otherwise unchanged |
| `post_compilation_objects` | 401–441 | identical_AST | Unchanged compiled object ID lookup |
| `_apply_button_states` | 443–482 | changed | Rendering lock colors; physical damping/forward unchanged |
| `modify_mjcf_model` | 484–583 | changed | Physical size + rendering configuration + recompile lifecycle |
| `initialize_episode` | 585–746 | changed | Reset semantics in data collection; colors/cache initialization |
| `_is_in_drawer` | 748–753 | identical_AST | Unchanged task spatial test |
| `set_new_target` | 755–910 | changed | Task target sampling semantics in data collection |
| `initialize_arm` | 912–930 | changed | Reset pose source/RNG consumption; IK setup copied |
| `pre_step` | 932–934 | identical_AST | Unchanged previous-state capture |
| `_compute_successes` | 936–965 | identical_AST | Unchanged success definitions |
| `set_cube_target_pos` | 967–979 | SWM_only | Additional target configuration setter |
| `set_target_button_state` | 981–987 | SWM_only | Additional target configuration setter |
| `set_target_drawer_pos` | 989–991 | SWM_only | Additional target configuration setter |
| `set_target_window_pos` | 993–995 | SWM_only | Additional target configuration setter |
| `post_step` | 997–1073 | changed | Rendering recoloring removed; physical/task logic unchanged |
| `get_reset_info` | 1082–1087 | changed | Output boundary keys |
| `get_step_info` | 1089–1094 | changed | Output boundary keys |
| `add_object_info` | 1096–1204 | identical_AST | Unchanged observation/info values |
| `compute_observation` | 1206–1257 | identical_AST | Unchanged observation construction |
| `compute_oracle_observation` | 1259–1281 | identical_AST | Unchanged oracle observation construction |
| `compute_reward` | 1283–1297 | identical_AST | Unchanged reward definition |
| `render` | 1299–1325 | changed | Rendering default camera |
| `render_multiview` | 1327–1366 | SWM_only | Rendering output configuration |

## Recorded gate results

- Dependency graph/model audit: completed, two families, six task tables, two real default wrapper chains. Architecture decision: **FAIL**.
- Degenerate-bound counterexamples: expected mismatch for seeds 0, 17, 42.
- Shared contract, semantic harness, readiness/taxonomy and matrix suite: **42 passed, 1 optional ALE warning**; 0.89 seconds.
- Ruff lint and format checks: pass, two research scripts.
- Strict documentation build: pass.
- Whole-tree/evidence preservation: pass; exact counts in `closure.json`.
- Staged whitespace check: pass.
- Stage-B CPU parity/contact/State tests: **not run**, because Stage A failed. Neither Cube nor Scene is accepted.

## Exact changed-file inventory

All additions; no existing tracked file is edited. Two scripts inspect source/dependencies and preserve evidence; four `.diff` files are research comparisons, not vendored production modules. `SHA256SUMS` seals the report and evidence, excluding itself.

- `docs/research/ogbench_dependency_boundary.md`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/audit.py`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/audit.log`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/closure.py`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/closure.json`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/closure.log`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/cube.exact.diff`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/cube.normalized.diff`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/dependency_graph.json`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/provenance.json`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/scene.exact.diff`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/scene.normalized.diff`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/shared-tests.log`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/mkdocs.log`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/formatting.log`
- `docs/research/semantic_recovery_evidence/ogb_dependency_boundary/SHA256SUMS`
