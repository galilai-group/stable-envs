# Adding a component

New contributors start with
[CONTRIBUTING.md](https://github.com/galilai-group/stable-envs/blob/main/CONTRIBUTING.md). It covers
access, branches, tests and pull requests.

First identify which layer you are changing. A registration combines three responsibilities:

```text
Registration = Environment + Task + renderer/default configuration
```

The API has `Env` and a private simulator-adapter protocol. There is no `Task`, `Renderer` or
`Observation` class to subclass.

| you are adding | guide |
|---|---|
| a simulator, dynamics, action mechanism or State schema | [Adding an environment backend](environment.md) |
| a reward, goal, reset distribution, success rule or task configuration on an existing backend | [Adding a task](task.md) |
| pixels, a camera or a rendering backend, without changing semantics | [Adding a renderer](renderer.md) |
| model inputs, observation packaging or learner-visible filtering | [Sibling projects](../concepts/worldmodel.md) |

## Where code goes

| code | location |
|---|---|
| shared public types: `Env`, State, specs, spaces, registration, configuration | package root |
| family-specific code | `envs/<family>/` |
| a family's simulator adapter and State conversion | `envs/<family>/simulation/` |
| reusable rendering engines | `render/` |
| shared worker-process code | `_execution/` |
| shared internals | `_core/` |
| reference comparisons and corpus utilities | `testing/` |
| experiment drivers | `scripts/` |

A helper used by one family lives with that family, even if it is graphics code. A leading underscore
marks an internal module. Create a subpackage only when several files share an owner.

## Values and who owns them

stable-envs owns full State, `get_state`, `set_state`, reset, step, actions, reward, success, done,
RNG, Variation and rendering. Call the upstream simulator for dynamics. stable-worldmodel owns learner
observations. Do not add an Observation class or a State-filtering layer here.

| value | meaning | restores dynamics? |
|---|---|---|
| `State` | TensorDict checkpoint of every simulator, task, RNG and lifecycle field needed for the future | yes |
| `RenderState` | private renderer input, often a compact float32 row | no (`set_state` must refuse it) |
| `Frame` | uint8 pixels, `(B,H,W,3)` | no |
| learner observation | downstream selection of frames, proprioception, goals | not necessarily |

Matching pixels do not prove that State is complete.

The [Env interface](../api/env.md) provides reset, get/set, step, close, the specs and
`render(full_state)`. Implement the private `_step` kernel. The shared public `step` handles action
sequences and capture. Renderer-only classes must refuse simulator operations.

## Small example

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
    assert not frame.dtype.is_floating_point
```

## Read the source first

Read the full construction path: registration, wrappers, reset, controller, physics, reward and done,
RNG and rendering. Pin upstream versions and record hashes of the source and assets you use. Include
dependency versions that change arithmetic. Record which wrapper owns the time limit.

Keep unusual upstream behaviour (dtype conversions, no-op knobs, odd task rules) and document it. Then
check your reading with runtime comparisons. See [Fidelity methodology](../concepts/fidelity-methodology.md).

## Local checks

```bash
python -m pip install -e '.[dev,docs]'
python -m pytest -q tests/unit/test_full_state_contract.py tests/unit/test_semantic_harness.py
python -m pytest -q tests/unit/test_repo_contracts.py
python -m mkdocs build --strict --config-file mkdocs.yaml
ruff check .
ruff format --check .
git diff --check
```

Install the family's pinned profile (see [Install](../install.md)) and run its tests. [Testing](../testing.md)
describes the registry-wide checks in CI.

## Definition of done

A new registration is done when all of these conditions are true:

- The claim is implemented and every required check passes.
- Unsupported paths refuse explicitly.
- Each claim traces to a test or a recorded comparison.

A skipped required check is not a pass. Do not edit a frozen fixture or a threshold to make a check
pass. A sibling's result does not cover another registration.

Keep these records for review: the driver script, source versions and hashes, dependency and platform versions, raw
logs and per-check results.

## Review checklist

- [ ] The layer and every affected registration are named.
- [ ] Source versions, hashes and supported configuration are stated.
- [ ] State, RenderState, frames and learner observations stay separate.
- [ ] The environment, task or renderer checklist is complete.
- [ ] Fast tests and final acceptance results are reported separately.
- [ ] Frozen fixtures and baselines are unchanged.
- [ ] No device, configuration or sibling class inherits a claim it was not tested for.
