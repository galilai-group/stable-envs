# Adding an environment backend

Use this guide for new dynamics, a simulator or controller, an action mechanism, or a State schema.
For a reward or reset change on an existing backend, see the [task guide](task.md). For pixels only,
see the [renderer guide](renderer.md).

## Interface

The backend owns simulation and every field needed to resume it. Call the upstream engine and
controller. A simulating `Env` provides:

| operation | contract |
|---|---|
| `reset(seed=None, options=...)` | return an independent full State. Keep the source's reset and seeding semantics. |
| `get_state()` / `state` | independent TensorDict snapshot with documented field shapes and dtypes |
| `set_state(state)` | validate the whole input before changing anything. Restore every field. Return `None`. |
| `step(actions)` | advance the upstream simulator. Return State and reward, success, termination, truncation. |
| `state_spec`, `action_spec` | match the actual fields and actions. Document constraints the spec cannot express. |
| `render(state=None, ...)`, `frame_spec` | render without changing the live simulation |
| `close()` / context manager | idempotent cleanup of owned resources. Later calls refuse. |

The private protocol `stable_envs._execution.protocol.SimulatorAdapter` has `num_envs`, reset, get, set,
step and close. It has no rendering method. Examples:

- `ClassicControlAdapter` in `stable_envs/envs/classic_control/simulation/local.py`, which
  `ClassicControlSemantics` in `stable_envs/envs/classic_control/semantics.py` uses
- the Fetch adapter in `stable_envs/envs/fetch/simulation/adapter.py`

Family mixins combine these backends with renderers. Register public classes with `@register` from
`stable_envs.registry`. Class names are the lookup keys. Set family, domain and default configuration
explicitly. `import_errors()` reports missing optional dependencies.

Implement `_step(actions)` and masked stepping for terminated lanes. The shared `step` handles action
sequences and capture. Do not snapshot and restore inside ordinary `step()`. That can discard
controller or warm-start history and change the trajectory.

## Example: restoring into an independent instance

```python
import numpy as np
import stable_envs as se

with se.CartPole(num_envs=2, device="cpu") as source, \
     se.CartPole(num_envs=2, device="cpu") as restored:
    source.reset(seed=17)
    action = np.array([[0], [1]], dtype=np.int64)
    source.step(action)
    saved = source.get_state()
    held = saved.clone()

    restored.reset(seed=42)
    restored.step(1 - action)  # unrelated history
    assert restored.set_state(saved) is None
    assert (restored.get_state() == saved).all()
    for actions in (action, 1 - action, action):
        assert (source.step(actions) == restored.step(actions)).all()
    assert (saved == held).all()

    before = restored.get_state()
    saved["x"].add_(1)  # snapshots do not share storage with the simulator
    assert (restored.get_state() == before).all()
```

Each engine needs its own schema. Do not copy CartPole's or Fetch's fields without tracing the new
family's physics, controller and task.

## Designing State

Read the registration defaults, wrappers, reset, step, controller helpers and the engine's state API.
Find every field that affects the future:

- integration state, control targets and actuator state
- applied forces and contact caches
- task bookkeeping and reset anchors
- RNG state

Record the restore order: which engine call recomputes derived arrays, which fields must be set after
it, and which setters have side effects.

Validate before changing any lane: missing or extra keys, batch shape, dtype, finite values, ranges and
semantic constraints. A bad field must not leave earlier lanes half-restored.

A static model or configuration can stay outside State only if restoring requires the same
configuration and the documentation states that requirement.

## Local checks

- Field validation, and rejection before any change.
- Snapshots are independent of each other and of the simulator.
- Reset matches the source on fixed seeds and fixed Variation.
- One-step and multi-step continuation, then restoration after a different history.
- Action dtypes, bounds, and clipping or rejection.
- `set_state` refuses compact render rows.
- Persistent physics Variation versus one-reset overrides.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/unit/test_full_state_contract.py tests/unit/test_semantic_harness.py
python -m pytest -q tests/unit/test_native_classic_semantics.py
python -m pytest -q tests/integration/test_native_classic_gate.py
ruff check stable_envs tests
ruff format --check stable_envs tests
git diff --check
```

The integration test is an example for CartPole, not coverage of a new class. Write an equivalent
driver for your registration with pinned dependencies. Do not generate its reference data with the
candidate code.

## Final acceptance

- Compare source and candidate reset, next State, reward and done for a fixed seed set. Compare every
  State field.
- Cover contacts, controller history, random draws and episode boundaries.
- Check seeded and unseeded reset wherever the registration promises replay. Store reset anchors and generator states
  without loss.
- Decide whether `TimeLimit` belongs to the registration or to a caller's wrapper. If the backend owns
  it, restore elapsed steps. Compare termination and truncation separately.
- Test physics Variation.
- Run the renderer checks separately.

```bash
python -m pytest -q tests/unit/test_replacement_readiness.py
python -m pytest -m 'regression and not gpu'
```

Run the family and GPU checks on a GPU where you claim GPU support. Record skips, missing dependencies
and device limits. See [Definition of done](index.md#definition-of-done).

Common mistakes:

- saving only `qpos`
- dropping reset RNG or anchors
- wrong restore order
- a setter that changes hidden history
- casting float64 source arithmetic to float32
- silently clipping actions

## Review checklist

- [ ] The backend calls the upstream simulator and controller. Sequential `step` is unchanged.
- [ ] Schema, dtypes, validation and restore order come from the source.
- [ ] Restoration tests include different histories with contacts and controller state.
- [ ] Tests cover Task, RNG, lifecycle and physics Variation.
- [ ] Compact render rows cannot restore the simulator.
- [ ] Results cover every claimed profile. Required skips block acceptance.
