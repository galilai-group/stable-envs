# Adding an environment backend

Use this guide for new dynamics, a simulator/controller, an action mechanism or
an exhaustive State schema. For a reward/reset change on the same backend, use
the [task guide](task.md); for pixels alone, use the [renderer guide](renderer.md).

## Responsibility and interface

The backend owns authoritative simulation and all state needed to resume it.
Delegate to the upstream engine and controller. Stable-envs owns the public
State contract and its validation; stable-worldmodel owns observation selection.
The [layer overview](index.md) defines that boundary.

A simulator-backed `Env` exposes:

| Operation | Contract |
|---|---|
| `reset(seed=None, options=...)` where supported | Return independent full State; preserve source reset and seeding semantics |
| `get_state()` / `state` | Independent TensorDict snapshot; named field shapes/dtypes documented |
| `set_state(state)` | Validate the entire input before mutation; restore all relevant fields; return `None` |
| `step(actions)` | Advance the live authoritative simulator; return State and defined reward/success/termination/truncation values |
| `state_spec`, `action_spec` | Match actual field/action shapes and dtypes; document constraints beyond the descriptive spec |
| `render(state=None, ...)`, `frame_spec` | Project internally to the renderer; do not change live simulation |
| `close()` / context manager | Idempotent owned-resource cleanup; operations after close refuse |

The private `stable_envs._execution.protocol.SimulatorAdapter` protocol comprises `num_envs`,
reset/get/set/step/close. It has no rendering or transition method. See the accepted
`ClassicControlAdapter` and `ClassicControlSemantics` in
`stable_envs/envs/classic_control/semantics.py`, and the delegated Fetch adapter in
`stable_envs/envs/fetch/simulation/adapter.py`. Family mixins compose those backends
with existing renderers. New public classes use `@register` from
`stable_envs.registry`; unique class names are lookup keys. Family, domain, slug
and default configuration should be explicit. Module discovery is automatic;
`import_errors()` reports unavailable optional dependencies.

Ordinary sequential `step()` must not snapshot and restore internally on every
step. That can discard warmstart/controller history and silently change the
source trajectory. Implement `_step(actions)` and masked advancement for terminal
lanes; shared `step` handles action sequences and capture.

## Complete example: independent restoration using an accepted backend

This executable example exercises the integration pattern in accepted CartPole.
It uses the shipped implementation rather than sketching a second physics engine.
The implementation to adapt is the adapter/mixin pair named above, including
its full-field validator, RNG serializer and wrapper lifecycle restoration.

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
    saved["x"].add_(1)  # snapshots do not alias live simulator storage
    assert (restored.get_state() == before).all()
```

Source-derived schemas differ by engine. Do not copy CartPole's or Fetch's fields
into a new family without tracing that family's physics, controller and task.
A static model/configuration can remain outside State only under a documented
same-configuration restore precondition. Changing supported persistent Variation
must configure the receiving simulator consistently.

## Authority and State design

Read registration defaults, all wrappers, reset and step, integration/controller
helpers and the engine's state API. Identify continuation-relevant integration
state, control targets, actuator state, applied forces, contact caches, task
bookkeeping, RNG state and reset anchors. Include a field when the source or an
omission witness shows relevance; do not minimize away necessary bytes.

Record restore order: which engine synchronization recomputes derived arrays,
which fields must be restored after it, and which setters have side effects.
Validate missing/extra keys, batch shape, dtype, finite values/ranges and semantic
constraints before changing any lane. A malformed later field must not leave an
earlier lane partially modified. Specs alone do not prove the runtime validator.

PushT remains incomplete: body transforms/velocities plus inspected numeric
contact fields did not reproduce a different contact history. Rebuilding two
spaces until they agree is insufficient; the restored continuation must match
an untouched source simulator. If a safe engine-supported mechanism is missing,
stop without weakening State or excluding contacts.

## Fast development checks

Add targeted tests alongside the accepted family tests:

- Field key/shape/dtype/range validation and rejection before mutation.
- Independent snapshots and independence of source/target simulator instances.
- Reset parity on fixed seeds and fixed Variation, and full State roundtrip.
- One-step and several-action continuation, then different-history restoration.
- Action dtypes, bounds, clipping/rejection/scaling and simultaneous components.
- Explicit refusal of raw compact rows and lossy decoded RenderState.
- Persistent physics Variation versus one-reset overrides; unsupported leaves refuse.

```bash
python -m pip install -e '.[dev]'
python -m pytest -q tests/unit/test_full_state_contract.py tests/unit/test_semantic_harness.py
python -m pytest -q tests/unit/test_native_classic_semantics.py
python -m pytest -q tests/integration/test_native_classic_gate.py
ruff check stable_envs tests
ruff format --check stable_envs tests
git diff --check
```

The last integration test is an accepted example, not automatic coverage of your
new class. Pin its oracle dependencies and add an equivalent independent driver
for your registration; do not regenerate an oracle with the candidate code.

## Final certification

Require independent source/candidate reset and next-State/reward/done comparisons
for the declared seed set (including 0, 17 and 42 where following the existing
recovery gates). Compare every State field. Exercise contacts, controller history,
random draws and lifecycle boundaries, not only motion in free space.

Prove seeded reset and unseeded reset/RNG replay wherever promised. A reset(None)
that requests fresh external entropy upstream needs an explicit bounded claim;
saving one generator must not be presented as capturing all entropy. Record
reset anchors and complete generator states losslessly.

Determine whether `TimeLimit` belongs to the registration or a caller's wrapper.
Restore elapsed steps if owned by the backend; compare termination and truncation
separately at the boundary. Test physics-affecting Variation and meaningful
omission witnesses. Run the renderer guide's gates separately, and shared
readiness/taxonomy/promotion checks:

```bash
python -m pytest -q tests/unit/test_replacement_readiness.py
python -m pytest -m 'regression and not gpu'
```

Then run the applicable family and frozen-renderer GPU gates on a GPU when claimed.
These commands do not replace a new registration's semantic evidence. Record
all skips, missing dependencies and device restrictions.

## Evidence and definition of done

Use the [common evidence requirements](index.md#evidence-definition-of-done-and-review).
Include a field/dtype table, restore-order explanation, pinned source hashes,
action/controller details, reset/lifecycle ownership, per-Variation scope and
raw comparisons. Done means exact branching under the declared profile and
configuration plus the required task/renderer gates. State sufficiency cannot be
inferred from a matching image or an in-place roundtrip.

Common failures: serializing only qpos, dropping reset RNG/anchors, wrong restore
order, a setter changing hidden history, casting source float64 arithmetic to
float32, silently clipping actions, or exposing exhaustive State as learner input.

PR-review checklist:

- [ ] Upstream simulator/controller is delegated; sequential step is untouched.
- [ ] Schema, dtypes, validation and restore order are source-derived.
- [ ] Independent and different-history tests include contacts/controller histories.
- [ ] Task, RNG, lifecycle and physics Variation are covered.
- [ ] Lossy render projections cannot restore the simulator.
- [ ] Evidence covers every claimed profile; required skips block promotion.
