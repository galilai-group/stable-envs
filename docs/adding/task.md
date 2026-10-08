# Adding a task or task registration

Use this guide when reward, goal, reset distribution, success, termination or task
configuration changes on an existing simulator. New dynamics/controller/action
mechanisms need the [environment guide](environment.md). Learner flat/dict
packaging belongs to stable-worldmodel, not a new simulator abstraction.

## Responsibility and minimum interface

A task supplies the source-defined reset/goal rules, reward/success and lifecycle
semantics through the same `Env.reset`, `step`, `get_state` and `set_state`
interface. Goal, task identifier, reset anchors and counters belong in State when
needed for future transitions/resets. Bounded task Variation must be routed to
its real source consumer. There is no required public Task base class.

A registration selects a backend, task and renderer/default configuration.
Shared code does not erase per-registration obligations. Fetch's accepted family
uses one `FetchAdapter`, with task-specific configuration and sparse/dense reward
selection. Its twelve classes compose `FetchSemantics` with `FetchBase`; the
`CELLS` table and `_make_classes` in `stable_envs/envs/fetch.py` define the public
classes. The native registry uses class names, not historical SWM IDs.

## Complete example: shared Fetch backend, distinct task semantics

Install the compatible robotics profile described in [Install](../install.md).
This CPU example needs no renderer or GPU. It executes three accepted registrations
and verifies their shared transition while preserving the dense/sparse distinction:

```python
from contextlib import ExitStack
import numpy as np
import torch
import stable_envs as se

with ExitStack() as stack:
    names = ("FetchPickAndPlace", "FetchPickAndPlaceDense", "FetchPickAndPlaceDict")
    envs = [stack.enter_context(se.make(n, num_envs=1, device="cpu")) for n in names]
    states = [env.reset(seed=17) for env in envs]
    for state in states[1:]:
        assert (state == states[0]).all()
    action = np.array([[0.1, -0.2, 0.05, -1.0]], dtype=np.float32)
    outputs = [env.step(action) for env in envs]
    for out in outputs[1:]:
        assert (out["state"] == outputs[0]["state"]).all()
        assert torch.equal(out["terminated"], outputs[0]["terminated"])
        assert torch.equal(out["truncated"], outputs[0]["truncated"])
    assert torch.equal(outputs[0]["reward"], outputs[2]["reward"])
    assert outputs[0]["reward"].item() in (-1.0, 0.0)
    assert outputs[1]["reward"].item() <= 0.0
```

The Dict class retains historical registration correspondence; this example does
not construct a dict learner observation. The source's goal-distance threshold
and exact sparse/dense formulas are tested separately in the family gate. These
assertions do not certify another Fetch task by association.

## Find and pin the authoritative task

Trace the upstream registration, constructor defaults, wrapper stack, goal/reset
sampler, reward and success functions, and TimeLimit. Read the controller and
physics path far enough to prove that selecting the task still delegates to the
same engine. Record version/commit and source/asset hashes, including any wrapper
that changes semantics. Fetch uses authoritative gymnasium-robotics/MuJoCo;
its pinned historical wrapper also supplies reset/Variation behavior. Importing
stable-worldmodel at runtime is not part of the accepted backend design.

Document goal state separately from rendered goal decoration and learner goal
images. Document the default task and every task ID/configuration exposed by the
registration. Distinguish constructor configuration, persistent Variation and
one-reset overrides. Refuse unsupported options instead of silently ignoring them.

## Fast development tests

For each configuration, test:

- Reset distribution, deterministic fixed-seed parity and goal/reset anchors.
- Reward/success formulas, task IDs and default selection.
- Threshold equality and immediate representable neighbours below/above; use the
  source arithmetic dtype. Include simultaneous boundaries and angle wrap if used.
- Termination versus truncation, horizon ownership and elapsed-step restoration.
- Seeded/unseeded RNG behavior, task-affecting Variation and reset scope.
- Dense/sparse/alias equivalence of physics and expected differences in reward.
- The public class actually calls the real simulator/controller, including during
  consecutive steps; tests must detect a duplicate hand-written transition path.

Accepted local examples and shared checks:

```bash
python -m pip install -e '.[dev,robotics]'
python -m pytest -q tests/unit/test_envs_fetch.py tests/unit/test_variation_maps_fetch.py
python -m pytest -q tests/unit/test_semantic_harness.py tests/unit/test_replacement_readiness.py
ruff check stable_envs tests
ruff format --check stable_envs tests
git diff --check
```

Run a family-specific driver in the pinned source/profile with a new output
location; inspect its entry point before running archived drivers that write fixed evidence paths. Optional-dependency skips in the unit suite do not prove Fetch
semantics. The [accepted family report](../research/fetch_family_recovery.md)
shows a concrete evidence structure; its local setup is not a contributor requirement.

## Promotion and certification

Run independent source-versus-candidate comparisons for reset and multi-step
continuation under each registration/configuration. Compare full State, reward,
success and both done flags. Restore into independent simulators with different
contact/controller histories. Replay reset anchors and RNG wherever promised.
Cover task-affecting and visual-only Variation separately.

Shared evidence may be referenced when the exact code path and configuration are
identical. Still execute each public registration, validate its constructor/action
spec/lifecycle/reward configuration, and record registration-specific results.
A new reward branch needs its own boundaries; a different reset distribution
needs its own seeded witnesses; a renderer-changing task needs rendering evidence.
One accepted representative is not family-wide acceptance.

```bash
python -m pytest -q tests/unit/test_full_state_contract.py tests/unit/test_replacement_readiness.py
python -m pytest -m 'regression and not gpu'
```

Also run your registration's independent semantic driver and required renderer/GPU
gates. GPU renderer acceptance proves rendering under that profile, not physics
parity under a different dependency profile. Missing required gates leave that
registration incomplete; siblings may be accepted independently.

## Evidence, definition of done, and review

Follow the [common provenance rules](index.md#evidence-definition-of-done-and-review).
Keep a per-registration table of task IDs, defaults, source functions, aliases,
reset/reward/lifecycle tests and renderer coverage. Record exact source versions,
hashes, comparison records, terminal results and unsupported scope.

Done means the registration's complete task behavior is proven on the real backend
and all required acceptance gates pass. Typical failures are confusing visual and
reward goals, changing `<` to `<=`, dropping velocity terms from a dense reward,
using trainer horizons as simulator horizons, and certifying siblings by name.

PR-review checklist:

- [ ] Backend delegation is unchanged and tested.
- [ ] Goal/reset/reward/success and lifecycle ownership are source-derived.
- [ ] Threshold neighbours use the correct source dtype.
- [ ] Every exposed task/configuration and public registration executes.
- [ ] Shared evidence has an explicit identical-path justification.
- [ ] Variation/reset/RNG and renderer differences have their own witnesses.
- [ ] Observation packaging remains downstream; status changes are evidence-backed.
