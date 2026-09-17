# Step 9 — Refactor, Productize, and Prepare `stable-envs` for OSS

> **Coordinator:** Fable 5.1 High  
> **Delegation:** Opus 5 Medium by default; Opus 5 High for shared rendering/resource work and ambiguous integration work  
> **Baseline:** accepted Step-8 functional-completion milestone  
> **Goal:** turn the certified implementation into a clean, coherent, OSS-quality `stable-*` library without changing its behavior

---

## 0. Mission

Steps 1–8 proved that `stable-envs` works.

Step 9 is different:

> **Do not discover new behavior. Do not redesign semantics. Do not optimize for benchmark badges. Make the already-certified system simple, coherent, idiomatic, and maintainable.**

The final repository should feel like a sibling of:

- `stable-pretraining`
- `stable-datasets`
- `stable-worldmodel`

A contributor familiar with one of those repos should immediately understand:

- where public APIs live;
- where environments live;
- where rendering machinery lives;
- how tests are organized;
- how to run verification;
- how to add an environment;
- how `stable-envs` composes with `stable-worldmodel`.

The Step-8 support/certification matrix is the behavioral oracle.

Step 9 may move, rename, simplify, consolidate, and delete code aggressively **only if the Step-8 behavior remains intact**.

---

## 1. Baseline and acceptance rules

Before any refactor work:

1. resolve the accepted Step-8 milestone/tag;
2. verify it points to the accepted Step-8 closure HEAD;
3. create:
   - `step9/integration`
   - `sami/pre-step9-baseline`
   from that exact SHA;
4. record the baseline SHA in:
   - `examples/results/step9/dashboard.yaml`
   - every Step-9 verification artifact;
5. verify the Step-8 closure packet/support matrix is available and reproducible.

The coordinator must treat the accepted Step-8 support matrix as immutable input.

### Git rules

All project commits:

```text
author/committer: sami-bg <samibghanem@gmail.com>
```

No AI attribution trailers.

Do not push Step-9 branches/tags unless Sami explicitly asks.

Do not delete Step-8 evidence until equivalent canonical fixtures/artifacts are safely preserved.

---

## 2. Binding architectural decisions

These decisions are already made for Step 9 unless a concrete implementation impossibility is discovered.

### Public API

Primary, class-first API:

```python
import stable_envs as snv

env = snv.PushT(
    num_envs=1024,
    device="cuda",
)
```

Dynamic/config-driven API is secondary:

```python
env = snv.make("swm/PushT-v1", device="cuda")
```

Strings exist for config, serialization, registries, and interop. Concrete environment classes are canonical.

### Stateful stepping

```python
state = env.reset(seed=0)
out = env.step(actions)
```

`step(actions)` is stateful. Do not overload `step` with a functional signature.

### Functional dynamics

```python
next_state = env.transition(state, action)
trajectory = env.unroll(state, actions)
```

`transition` and `unroll` are functional/pure with respect to environment state.

`unroll` means known/open-loop actions, not policy rollout.

### Rollout ownership

`stable-envs` owns:

```text
reset
step
transition
unroll known actions
render
state/action/variation semantics
```

`stable-worldmodel` owns:

```text
policy rollouts
MPC
CEM / MPPI
planning
dataset collection
evaluation
async orchestration
auto-reset policy semantics
```

Do not introduce `env.rollout(policy, ...)`.

### Structured state

Use TensorDict where state structure is genuinely useful.

Example:

```python
state["agent", "position"]
state["object", "rotation"]
```

Do not require TensorDict ceremony for trivially shaped tensors.

The public state representation should describe the environment, not a renderer kernel's preferred packed layout.

### Specs

Use one shared conceptual spec abstraction:

```python
env.state_spec
env.action_spec
env.observation_spec
env.variation_spec
```

Do not create separate class hierarchies for `StateSpec`, `ActionSpec`, `VariationSpec` unless distinct behavior is genuinely necessary.

Follow TensorDict/TensorSpec conventions where practical instead of inventing a large custom framework.

### Logging/config

Mirror the stable-* ecosystem:

```python
snv.set(verbose="INFO")
snv.get_config()
```

Use Loguru internally. Keep global config deliberately small.

### Interoperability

Use:

```text
stable_envs/interop.py
```

not `worldmodel.py`, `compat/`, or public `utils.py`.

`interop.py` only contains code that exists because stable-envs and another package must understand one another.

The ideal long-term state is that `interop.py` shrinks because SWM consumes `Env` natively.

### No speculative transforms layer

Do not create `transforms.py` unless actual repeated usage during Step 9 proves it useful.

### Resource ownership

Resource lifetime must be explicit:

```python
env.close()

with snv.PushT(...) as env:
    ...
```

Correctness must not depend on `__del__`.

### Testing

Use pytest + markers, not a homegrown verification framework.

Expected markers:

```text
unit
integration
gpu
certification
slow
```

Benchmarks remain separate from correctness.

---

## 3. Target repository structure

```text
stable-envs/
├── stable_envs/
│   ├── __init__.py
│   ├── __about__.py
│   ├── _config.py
│   ├── env.py
│   ├── dynamics.py
│   ├── specs.py
│   ├── registry.py
│   ├── interop.py
│   ├── envs/
│   │   ├── __init__.py
│   │   ├── classic_control.py
│   │   ├── pusht.py
│   │   ├── navigation.py
│   │   ├── craftax.py
│   │   ├── rocket_landing.py
│   │   ├── fetch.py
│   │   ├── dmcontrol/
│   │   │   ├── __init__.py
│   │   │   ├── base.py
│   │   │   ├── acrobot.py
│   │   │   ├── ball_in_cup.py
│   │   │   ├── cartpole.py
│   │   │   ├── cheetah.py
│   │   │   ├── finger.py
│   │   │   ├── hopper.py
│   │   │   ├── humanoid.py
│   │   │   ├── manipulator.py
│   │   │   ├── pendulum.py
│   │   │   ├── quadruped.py
│   │   │   ├── reacher.py
│   │   │   └── walker.py
│   │   ├── ogbench/
│   │   │   ├── __init__.py
│   │   │   ├── maze.py
│   │   │   └── manipulation.py
│   │   └── mujoco/
│   │       └── ...
│   └── render/
│       ├── __init__.py
│       ├── scene.py
│       ├── primitives.py
│       ├── raster2d.py
│       ├── gfxdraw.py
│       ├── tilemap.py
│       ├── raycast3d.py
│       ├── cpu3d.py
│       ├── mjwarp.py
│       ├── nvdiff.py
│       └── _resources.py
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── gpu/
│   ├── certification/
│   ├── fixtures/
│   └── conftest.py
├── benchmarks/
│   ├── README.md
│   ├── render.py
│   └── pipeline.py
├── examples/
│   ├── render.py
│   ├── batched_env.py
│   ├── functional_dynamics.py
│   └── worldmodel.py
├── docs/
│   ├── index.md
│   ├── quickstart.md
│   ├── concepts/
│   ├── environments/
│   ├── benchmarks.md
│   └── api/
├── scripts/
│   └── ...
├── .github/
│   └── workflows/
│       ├── testing.yml
│       ├── pre-commit.yml
│       ├── docs.yml
│       └── release.yml
├── pyproject.toml
├── mkdocs.yaml
├── README.md
├── LICENSE
└── CHANGELOG.md
```

This is a target, not an excuse to invent empty abstractions. If a file/module is unnecessary after migration, omit it.

---

## 4. Dependency-direction rules

Required principles:

- core API modules must not depend on SWM;
- render backends must not know SWM IDs;
- render code should not import concrete environment classes;
- environments may depend on core API + renderer machinery;
- `interop.py` may depend on stable-envs public concepts and optional external packages;
- top-level imports expose the happy path, not backend internals.

Avoid circular imports by design rather than patching them late.

---

## 5. Public namespace

Expected top-level shape:

```python
import stable_envs as snv

snv.PushT
snv.Craftax
snv.DMCReacher
snv.OGBCube

snv.make
snv.spec
snv.list_envs
snv.set
snv.get_config
```

Do not expose backend machinery at top level.

---

## 6. Core module responsibilities

### `env.py`

Defines the main environment abstraction and common lifecycle behavior.

Owns:

- `reset`;
- stateful `step`;
- `render`;
- context manager;
- `close`;
- common device/batch semantics;
- references to specs.

Keep it small.

### `dynamics.py`

Defines the pure batched dynamics protocol:

```python
transition(state, action)
unroll(state, actions)
```

Do not put policy/planner logic here.

### `specs.py`

Defines or re-exports the chosen spec vocabulary.

Goals:

- clear state/action/observation/variation description;
- sampling/validation;
- device/shape/dtype metadata;
- TensorDict compatibility;
- minimal custom machinery.

### `registry.py`

Owns:

```python
make(...)
spec(...)
list_envs(...)
```

Class definitions remain canonical. Registry should not become the source of environment semantics.

### `interop.py`

Owns only cross-package glue.

Likely initial contents:

- SWM ID aliases;
- VariationMap bridging;
- `swap_renderer` migration path;
- state/variation conversion needed for current SWM.

Reduce this module as native SWM integration improves.

---

## 7. Step-8 behavioral oracle

Step 9 must preserve:

```text
42 SWM IDs
40 pixel-producing
2 symbolic / renderer-free
```

Preserve:

- registered default frames;
- fidelity tiers;
- Gate A behavior;
- VariationMap semantics/taxonomy;
- SWM integration;
- state semantics;
- device support honesty;
- determinism;
- batch invariance;
- order/chunk invariance;
- resource lifetime correctness;
- semantic carry-forward;
- CPU/GPU performance within reasonable noise.

Do not weaken thresholds, remove checks, silently reduce support, or re-freeze because the refactor changed pixels.

---

## 8. Performance preservation rule

Step 9 is not an optimization phase, but hot paths must not materially regress.

For changes affecting rendering/dynamics:

- compare before/after on the same hardware;
- use paired measurements;
- report relevant batch sizes;
- tolerate ordinary noise;
- investigate clear repeatable regressions.

A roughly >10% repeatable regression on a core hot path should block merge unless causally explained and accepted.

---

## 9. Coordinator operating model

Coordinator: **Fable 5.1 High**

Default subagent: **Opus 5 Medium**

Use **Opus 5 High** for:

- shared renderer internals;
- CUDA/EGL/resource lifetime;
- MJWarp/nvdiff;
- ambiguous SWM integration;
- final architecture review.

The coordinator must:

- establish/freeze the skeleton before broad migration;
- assign exclusive file ownership;
- create/manage worktrees;
- resolve shared-file conflicts;
- run independent reviews;
- merge only approved work;
- maintain the dashboard;
- prevent agents from redesigning public APIs independently;
- compare final behavior against Step 8.

No subagent may push.

---

## 10. File-ownership rule

Family migration agents do not edit coordinator-owned shared files unless explicitly authorized.

Coordinator-owned by default:

```text
stable_envs/__init__.py
stable_envs/env.py
stable_envs/dynamics.py
stable_envs/specs.py
stable_envs/registry.py
stable_envs/_config.py
pyproject.toml
```

Family agents implement/migrate their assigned modules and report what must be wired globally.

---

## 11. Wave 0 — establish the canonical skeleton

**Owner:** Fable 5.1 High, optionally one Opus 5 High helper

Do this serially before broad migration.

Implement and approve:

```text
env.py
dynamics.py
specs.py
registry.py
_config.py
interop.py
```

Establish:

- stateful vs functional semantics;
- TensorDict convention;
- spec convention;
- device/batch semantics;
- result shape from `step`;
- lifecycle/close behavior;
- logging/config;
- registry declaration mechanism;
- import/dependency rules;
- pytest markers.

### Representative environment

Migrate **PushT** end-to-end as the canonical example.

Required demonstration:

```python
import stable_envs as snv

env = snv.PushT(num_envs=64, device="cuda")

state = env.reset(seed=0)
step = env.step(action)

next_state = env.transition(state, action)
trajectory = env.unroll(state, action_sequence)

pixels = env.render(state)

env.close()
```

Independent architecture review required before Wave 1. Once approved, this becomes the migration template.

---

## 12. Wave 1 — environment-family migration

Run in parallel after Wave 0 approval.

### Workstream A — classic control + simple navigation

**Model:** Opus 5 Medium

Own:

```text
stable_envs/envs/classic_control.py
stable_envs/envs/navigation.py
```

Migrate:

- Acrobot
- CartPole
- MountainCar
- MountainCarContinuous
- Pendulum
- SimplePointMaze
- Piecewise
- TwoRoom

No shared-renderer rewrite.

### Workstream B — DMControl

**Model:** Opus 5 Medium  
**Reviewer:** Opus 5 High

Own:

```text
stable_envs/envs/dmcontrol/
```

Preserve:

- 3.11 certification semantics/provenance;
- T3 floor-color support;
- task-specific state exclusions;
- trackcom behavior;
- Step-8 variation mappings.

No giant inheritance hierarchy merely because DMC has many tasks.

### Workstream C — Fetch

**Model:** Opus 5 Medium

Own:

```text
stable_envs/envs/fetch.py
```

Goals:

- remove avoidable duplication;
- preserve exact registered defaults;
- preserve current partial `background.color`;
- preserve CPU/GPU support status;
- preserve variation taxonomy.

Do not re-freeze Fetch defaults.

### Workstream D — OGBench

**Model:** Opus 5 Medium

Own:

```text
stable_envs/envs/ogbench/
```

Migrate maze, OGBCube, and OGBScene.

Preserve plate-bake behavior, Step-8 variations, button bake, per-world arrays, CPU refusal, and fidelity declarations.

### Workstream E — Craftax + Rocket

**Model:** Opus 5 Medium

Own:

```text
stable_envs/envs/craftax.py
stable_envs/envs/rocket_landing.py
```

Preserve Craftax full/classic semantics, day/night, JIT-wrapper declaration, Rocket TinyRenderer law, flame/state semantics, GPU-only support, and Step-8 declarations.

### Workstream F — MuJoCo/nvdiff environment wrappers

**Model:** Opus 5 Medium

Own only environment-family/wrapper code. Shared nvdiff implementation belongs to Workstream I.

---

## 13. Wave 1 — rendering migration

### Workstream G — 2D rendering

**Model:** Opus 5 Medium

Own:

```text
render/scene.py
render/raster2d.py
render/gfxdraw.py
render/tilemap.py
```

Goals:

- clean boundaries;
- remove duplicate helpers;
- preserve exact output;
- improve naming/types/docs;
- no abstraction unless at least two current backends use it.

### Workstream H — raycast3d / CPU3D / primitives

**Model:** Opus 5 High

Own:

```text
render/primitives.py
render/raycast3d.py
render/cpu3d.py
```

Preserve exact output, ellipsoid behavior, and backend semantics. Hot-path regression checks required.

### Workstream I — MJWarp / nvdiff / resources

**Model:** Opus 5 High

Own:

```text
render/mjwarp.py
render/nvdiff.py
render/_resources.py
```

Goals:

- explicit resource ownership;
- context-manager/close semantics;
- no correctness dependency on GC;
- centralize reusable lifetime/cache handling;
- preserve Step-8 pixels/certification;
- preserve documented x86 sparse-stage ULP limitation;
- no re-freeze.

Required tests:

- fresh-process reproducibility;
- repeated create/close;
- resource leak/lifetime behavior;
- batch-size invariance;
- same-card hot-path throughput.

---

## 14. Wave 2 — integration/product work

### Workstream J — SWM interoperability

**Model:** Opus 5 High

Own:

```text
stable_envs/interop.py
tests/integration/test_worldmodel.py
```

Preserve 42-ID resolution, VariationMaps, current `swap_renderer` compatibility, and move toward native:

```python
env = snv.PushT(...)
world = swm.World(env=env, ...)
```

Do not move MPC/rollout logic into stable-envs.

### Workstream K — tests/certification cleanup

**Model:** Opus 5 High

Target:

```text
tests/
├── unit/
├── integration/
├── gpu/
├── certification/
├── fixtures/
└── conftest.py
```

Markers:

```text
unit
integration
gpu
certification
slow
```

Desired commands:

```bash
pytest -m unit
pytest -m integration
pytest -m "certification and not gpu"
pytest -m "certification and gpu"
pytest
```

If a tiny `scripts/certify.py` remains useful, it may only orchestrate pytest/configurations and summarize status. It must not implement certification semantics.

### Workstream L — packaging / dependencies / CI

**Model:** Opus 5 Medium

Own, once granted:

```text
pyproject.toml
.github/
.pre-commit-config.yaml
mkdocs.yaml
```

Align with sibling stable-* conventions.

Requirements:

- editable/dev install;
- package build;
- optional backend extras where sensible;
- graceful missing-dependency behavior;
- pre-commit;
- unit/integration CI;
- GPU/certification strategy;
- docs build;
- release workflow.

### Workstream M — docs + examples

**Model:** Opus 5 Medium

Own:

```text
README.md
docs/
examples/
```

Required examples:

```text
render.py
batched_env.py
functional_dynamics.py
worldmodel.py
```

Quickstart must use only final public API.

### Workstream N — benchmarks

**Model:** Opus 5 Medium

Own:

```text
benchmarks/
```

Separate correctness from performance. Keep renderer throughput, batch scaling, and optional production-pipeline comparisons clean and OSS-facing.

---

## 15. Wave 3 — archaeology/dead-code cleanup

**Model:** Opus 5 Medium

Only after canonical tests/docs/fixtures exist.

Audit:

```text
scratch/
examples/results/
old closure drivers
superseded compatibility modules
duplicated renderers
stale scripts
dead imports
old aliases
```

Classify each item:

```text
KEEP_PUBLIC
KEEP_INTERNAL
ARCHIVE_RESEARCH
DELETE_SUPERSEDED
```

Do not destroy the Step-8 behavioral oracle.

---

## 16. No-abstraction-without-demand rule

> Do not introduce an abstraction unless at least two current environments/backends/packages need it, or it materially simplifies the public API.

Avoid unnecessary:

```text
core/
interfaces/
protocols/
adapters/
managers/
factories/
utils/
```

Prefer a small number of obvious modules.

---

## 17. Review/merge gate

Every workstream patch receives independent review.

Reviewer returns:

```text
APPROVE
APPROVE-WITH-CHANGES
BLOCK
```

Only `APPROVE` merges.

Every migration reports:

1. files moved/removed/created;
2. API changes;
3. Step-8 IDs affected;
4. targeted tests;
5. applicable certification checks;
6. default-frame comparison;
7. variation/state comparison;
8. performance comparison for hot paths;
9. resource-lifetime comparison if relevant;
10. unexplained oracle differences: zero.

Coordinator merges into `step9/integration`.

---

## 18. Dashboard

Maintain:

```text
examples/results/step9/dashboard.yaml
```

Suggested record:

```yaml
workstream:
  branch:
  owner:
  reviewer:
  model:
  status:
  files_owned:
  ids_affected:
  baseline_sha:
  head_sha:
  tests:
  certification:
  pixel_diff:
  api_diff:
  variation_diff:
  performance_diff:
  resource_diff:
  oracle_diff:
  review:
  merge_sha:
```

---

## 19. Continuous verification

After each major merge:

```bash
pytest -m unit
```

After each family batch:

```bash
pytest -m "unit or integration"
```

After shared renderer/resource changes:

```bash
pytest -m "certification and not gpu"
pytest -m "certification and gpu"
```

At major checkpoints, regenerate the Step-8-derived support/certification matrix and compare it against baseline.

---

## 20. Final oracle comparison

Generate a Step-9 support matrix and compare against Step 8 for all 42 IDs.

Every difference is exactly one of:

```text
API_ONLY
INTENTIONAL_IMPROVEMENT
REGRESSION
```

For successful closure:

```text
REGRESSION = 0
```

No unexplained differences.

---

## 21. Final performance check

Do not rerun every historical benchmark.

Run targeted paired before/after comparisons for:

- shared 2D hot paths;
- raycast/CPU3D;
- MJWarp;
- nvdiff;
- any dynamics code moved into new abstractions.

Clear repeatable regressions above roughly 10% on important hot paths block closure unless accepted with a concrete reason.

---

## 22. Final OSS quality gate

### Package/API

- [ ] clean install succeeds
- [ ] build succeeds
- [ ] public imports intentional
- [ ] top-level namespace small
- [ ] no import cycles
- [ ] optional dependencies fail gracefully
- [ ] close/context-manager semantics documented

### Tests

- [ ] unit green
- [ ] integration green
- [ ] CPU certification green
- [ ] GPU certification green on supported paths
- [ ] fresh-process/resource tests green
- [ ] no unexplained Step-8 oracle differences

### Docs

- [ ] README quickstart final API
- [ ] environment listing available
- [ ] state/action/spec concepts documented
- [ ] functional vs stateful documented
- [ ] SWM ownership boundary documented
- [ ] backend/device limitations documented
- [ ] benchmark methodology documented

### Repo hygiene

- [ ] root resembles sibling stable-* repos
- [ ] research archaeology no longer dominates ordinary paths
- [ ] dead code removed
- [ ] no generated junk in user-facing directories without reason
- [ ] no AI trailers
- [ ] changelog/release notes prepared

---

## 23. Closure recommendation

Return one of:

```text
STEP 9 OSS-READY
STEP 9 OSS-READY WITH DOCUMENTED LIMITATIONS
CONTINUE STEP 9 FOR <finite explicit tickets>
```

A limitation already accepted in Step 8 does not block OSS readiness merely because it still exists.

---

## 24. Step-9 closure packet

Produce:

```text
docs/research/step9_closure.md
examples/results/step9/support_matrix.json
```

Include:

1. Step-8 baseline SHA;
2. Step-9 final HEAD;
3. final repository tree;
4. public API summary;
5. 42-ID support comparison;
6. oracle differences;
7. test/certification totals;
8. performance before/after;
9. resource-lifetime status;
10. SWM interoperability;
11. packaging/build/CI;
12. docs/examples;
13. deleted/archived archaeology;
14. remaining limitations;
15. suggested first OSS release notes.

---

## 25. Recommended branches/worktrees

Coordinator:

```text
step9/integration
```

Foundation:

```text
step9/foundation
```

Parallel migrations:

```text
step9/envs-classic-nav
step9/envs-dmcontrol
step9/envs-fetch
step9/envs-ogbench
step9/envs-craftax-rocket
step9/envs-mujoco
step9/render-2d
step9/render-3d
step9/render-mjwarp-nvdiff
```

Product/integration:

```text
step9/interop
step9/testing
step9/packaging-ci
step9/docs-examples
step9/benchmarks
step9/cleanup
```

Use fewer branches if ownership is cleaner that way.

---

## 26. First coordinator actions

1. verify accepted Step-8 milestone/tag and exact SHA;
2. create/verify `step9/integration`;
3. create Step-9 dashboard;
4. snapshot Step-8 oracle;
5. inventory package structure/import graph;
6. inventory environment families/backends;
7. build shared-file ownership table;
8. implement Wave 0;
9. migrate PushT as canonical example;
10. assign independent Opus 5 High architecture reviewer;
11. merge foundation only after APPROVE;
12. launch Wave-1 environment/render workstreams in parallel;
13. integrate continuously;
14. launch Wave 2 once enough migrated code is stable;
15. launch cleanup only after canonical tests/docs/fixtures exist;
16. finish with full oracle comparison.

First user-facing update:

```text
Step-9 baseline: <SHA>
Foundation branch: <SHA/status>
Canonical PushT migration: <status>
Active parallel workstreams: <N>
Step-8 oracle regressions: <count>
```

Then proceed autonomously unless a true public-API or behavior decision requires Sami.

---

## 27. Core principle

> **The architecture may change dramatically. The behavior may not.**

Step 8 proved the system.

Step 9 makes it pleasant to use, understand, extend, and release.
