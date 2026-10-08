# Adding an environment, task, or renderer

Use these guides when extending stable-envs. Identify the layer being changed
before choosing an implementation or a test plan.

A registration is approximately:

```text
Registration = Environment + Task + renderer/default configuration
```

These are responsibilities, not three required Python base classes. The existing
API has `Env` and a private simulator-adapter protocol; it does not require a new
`Task`, `Renderer`, or `Observation` class. Several registrations can share an
implementation while retaining different acceptance evidence.

| Change being added | Guide |
|---|---|
| New simulator, dynamics, action mechanism, or State schema | [Environment](environment.md) |
| New reward, goal, reset distribution, success rule, or task configuration on an existing backend | [Task](task.md) |
| New pixels, camera, visual backend, or rendering implementation without changing semantics | [Renderer](renderer.md) |
| New model inputs, observation packaging, or learner-visible filtering | [stable-worldmodel](../concepts/worldmodel.md) |

## Where code belongs

- Keep the package root for shared public types and their immediate implementation:
  `Env`, State/specs, spaces, registration and configuration.
- Put family-specific code in `envs/<family>/`. Craftax's `prng.py`, for example,
  exists to reproduce its night noise; it is not a shared rendering backend.
- Keep reusable rendering engines in `render/`. A helper used by one environment
  family belongs with that family, even when it deals with graphics.
- Put a family's simulator adapter and State conversion in `envs/<family>/simulation/`.
  Shared worker-process code lives in `_execution/`, shared internals in `_core/`.
- Put reference comparisons and corpus utilities under `testing/`; experiment
  drivers belong under `scripts/`.

A leading underscore marks an internal module or helper. It should communicate
privacy, not development history. Public support is defined by the documented
API, not by the absence of an underscore. Create a subpackage when several files
share an owner or backend; avoid directories that merely rename a single file.

## Ownership and minimum interface

Stable-envs owns exhaustive simulator State, `get_state()`, `set_state()`, reset,
step, actions, reward/success/done, RNG, bounded Variation, State-to-RenderState
projection and rendering. Calling the authoritative upstream simulator is the
intended implementation. A second implementation of its dynamics is unnecessary.

Stable-worldmodel owns learner-visible pixels/proprioception/goals, flat or dict
packaging, model input keys and dataset filtering. Stable-envs supplies frames and
semantic goal state; the learner's selection and packaging belong downstream.
Do not introduce an Observation class or State-filtering layer here. Observation
integration remains deferred; no runtime SWM dependency is required.

| Value | Meaning | Can restore dynamics? |
|---|---|---|
| `State` | Independent TensorDict checkpoint of every future-relevant simulator/task/RNG/lifecycle field, under the documented persistent configuration | Yes, on a certified simulator |
| `RenderState` | Private projection sufficient for a claimed renderer; often a compact float32 row | Only if independently proven complete; a lossy projection must be refused |
| `Frame` | Rendered uint8 pixels, usually `(B,H,W,3)` | No |
| Learner observation | Downstream selection/packaging of frames, proprioception, goals or other authorized inputs | Not necessarily |

State can contain privileged information. Returning it from a simulator does not
make it the default input to a learner. Pixel equality does not establish State
sufficiency. PushT's parked recovery is a concrete failure case: matching body
transforms and ordinary velocities did not restore hidden contact history.

The [environment interface](../api/env.md) supplies reset/get/set/step/close,
State/action/frame specifications and `render(full_state)`. Renderer-only cells
must refuse simulator operations. Implement a private `_step` kernel; the shared
public `step` provides sequence execution and capture.

## Complete small starting example

This uses accepted CartPole semantics and its renderer. Run it from a checkout
installed with the development dependencies:

```python
import numpy as np
import stable_envs as se

with se.CartPole(num_envs=1, device="cpu") as env:
    saved = env.reset(seed=17)
    expected = env.step(np.array([[1]], dtype=np.int64))
    assert env.set_state(saved) is None
    actual = env.step(np.array([[1]], dtype=np.int64))
    assert (actual == expected).all()
    frame = env.render(saved)
    assert frame.dtype.is_floating_point is False
```

Follow the layer-specific guide for implementation examples and deeper checks.
This smoke test is not a promotion gate.

## Establish authority before implementation

Read the complete construction path: registration, wrappers, reset, controller,
physics, task reward/done, RNG and rendering. Pin upstream commits or package
versions and hash the actual source/assets used. Include dependency versions that
change arithmetic or engine behavior. Record wrapper ownership of a time limit
rather than assuming a trainer's episode cap belongs in the raw environment.

An official implementation can contain unusual dtype conversions, no-op knobs or
counterintuitive task rules. Preserve the claimed behavior and document it.
Runtime witnesses verify the source interpretation; guessing from rendered images
does not replace reading the source.

## Development versus promotion

Fast local development checks:

```bash
python -m pip install -e '.[dev,docs]'
python -m pytest -q tests/unit/test_full_state_contract.py tests/unit/test_semantic_harness.py
python -m pytest -q tests/unit/test_repo_contracts.py
python -m mkdocs build --strict --config-file mkdocs.yaml
ruff check .
ruff format --check .
git diff --check
```

Install family dependencies in a compatible, explicitly pinned profile; see
[installation](../install.md). Run the new family's targeted tests as well.
`mkdocs --strict` checks documentation links and anchors. The repository-contract
tests check navigation and fixture references.

Final certification needs source closure, independent reference comparisons,
State restoration across different histories, task/configuration coverage and
renderer gates on every claimed device/backend. Run the relevant integration,
regression and GPU suites on machines supporting those profiles. A skipped gate
is not a pass. The [testing page](../testing.md) lists the existing runners;
cluster-specific submission commands are not a contributor requirement.

## Evidence, definition of done, and review

Keep a reproducible driver, immutable source identity, dependency/platform record,
raw logs, per-check results and artifact hashes. A remote gate needs local/remote
source hashes and an exact success marker in its retained terminal log. Keep
baseline and candidate results distinct. Never edit a frozen fixture or threshold
to obtain acceptance. Update registration status only after its required gates
pass; one sibling's result cannot promote another automatically.

A contribution is done when its bounded claim is implemented, every required
check passes, unsupported paths refuse explicitly, and a reviewer can trace each
claim to evidence. A documented blocker leaves the registration incomplete.
Common failures are omitting hidden State, making learner packaging part of the
simulator, using renderer parity to infer physics parity, or reporting optional
skips as certification.

PR-review checklist:

- [ ] The layer and every affected registration are named.
- [ ] Authority, versions, source hashes and supported configuration are explicit.
- [ ] State, RenderState, frames and learner observations remain distinct.
- [ ] The relevant environment/task/renderer checklist is complete.
- [ ] Fast tests and final acceptance results are reported separately.
- [ ] Existing evidence and frozen baselines are preserved.
- [ ] No unsupported device, configuration or sibling inherits a claim by association.
