# Adding a task

Use this guide when the reward, goal, reset distribution, success rule, termination or task
configuration changes on an existing simulator. New dynamics, controllers or action mechanisms need
the [environment guide](environment.md). Learner observation packaging belongs to stable-worldmodel.

## Interface

A task supplies reset and goal rules, reward, success and lifecycle through the same `Env.reset`,
`step`, `get_state` and `set_state`. There is no Task base class.

- Goal, task id, reset anchors and counters go in State when future transitions or resets need them.
- Task Variation must reach the code in the source that uses it.

A registration picks a backend, a task and a renderer configuration. Shared code does not remove
per-registration checks. Example: the twelve Fetch classes share one `FetchAdapter` with
task-specific configuration and sparse or dense reward. They combine `FetchSemantics` with `FetchBase`.
The `CELLS` table and `_make_classes` in `stable_envs/envs/fetch/envs.py` define the classes.

## Example: shared Fetch backend, different rewards

Install the robotics profile from [Install](../install.md). This runs on CPU without a renderer:

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

The Dict class does not build a dict observation here. The family's own tests check the
goal-distance threshold and the exact reward formulas.

## Read the source task

Trace the upstream registration, constructor defaults, wrappers, goal and reset sampler, reward and
success functions, and `TimeLimit`. Check that the task still calls the same engine. Record versions
and hashes of the source and any wrapper that changes semantics. Fetch uses gymnasium-robotics and
MuJoCo.

- Keep goal State separate from the rendered goal marker and from learner goal images.
- Document the default task and every task id the registration exposes.
- Distinguish constructor configuration, persistent Variation and one-reset overrides.
- Refuse unsupported options. Do not ignore them.

## Local checks

For each configuration, test:

- reset distribution, fixed-seed parity and reset anchors
- reward and success formulas, task ids and defaults
- thresholds at the boundary and at the nearest representable values on each side, in the source's
  dtype
- termination versus truncation, who owns the horizon, and elapsed-step restoration
- seeded and unseeded RNG, and task Variation
- that dense, sparse and alias classes share physics and differ only in reward
- that the public class calls the real simulator on every step

```bash
python -m pip install -e '.[dev,robotics]'
python -m pytest -q tests/unit/test_envs_fetch.py tests/unit/test_variation_maps_fetch.py
python -m pytest -q tests/unit/test_semantic_harness.py tests/unit/test_replacement_readiness.py
ruff check stable_envs tests
ruff format --check stable_envs tests
git diff --check
```

Skips for missing optional dependencies do not prove Fetch semantics. Run the family tests in the
pinned profile.

## Final acceptance

- Compare source and candidate reset and multi-step continuation for each registration and
  configuration: full State, reward, success and both done flags.
- Restore into simulators with different contact and controller histories.
- Replay reset anchors and RNG wherever the registration promises replay.
- Test task Variation and visual-only Variation separately.

You may reuse results only when the code path and configuration are identical. Still run each public
class and check its constructor, action spec, lifecycle and reward configuration. A new reward branch
needs its own boundary tests. A new reset distribution needs its own seeded tests.

```bash
python -m pytest -q tests/unit/test_full_state_contract.py tests/unit/test_replacement_readiness.py
python -m pytest -m 'regression and not gpu'
```

GPU rendering results do not prove physics under another dependency profile. See
[Definition of done](index.md#definition-of-done).

Common mistakes:

- confusing the visual goal with the reward goal
- changing `<` to `<=`
- dropping velocity terms from a dense reward
- using a trainer's horizon as the simulator's
- accepting siblings by name

## Review checklist

- [ ] The backend still calls the upstream simulator, and a test checks it.
- [ ] Goal, reset, reward, success and lifecycle come from the source.
- [ ] Threshold tests use the source dtype.
- [ ] Every exposed task, configuration and public class runs.
- [ ] Shared results have an identical-path justification.
- [ ] Variation, reset, RNG and renderer differences have their own tests.
- [ ] Observation packaging stays downstream.
