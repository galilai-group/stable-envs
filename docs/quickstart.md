# Quickstart

Install the core and construct an environment by class name. See [Install](install.md) for extras
and pinned profiles.

```bash
pip install stable-envs
```

```python
import numpy as np
import stable_envs as snv

with snv.CartPole(num_envs=2, device="cpu") as env:
    state = env.reset(seed=17)
    pixels = env.render(state)
    action = np.array([[0], [1]], dtype=np.int64)
    out = env.step(action)
    assert env.set_state(state) is None
    assert (env.step(action) == out).all()
```

`num_envs` is the simulator batch size. `close()` is idempotent, and the context manager calls it.
A renderer may accept a smaller batch. That does not change the batch that `set_state` requires.

## State and frames

`get_state()` returns an independent TensorDict snapshot of the full State. `state` is an alias.
Fields include physical, task, lifecycle and RNG state, as the family needs. `state_spec` describes
those fields. `action_spec` describes actions. `frame_spec` describes uint8 `(H,W,3)` output without
the batch axis.

`render(state)` converts the State to the renderer's private input internally. `set_state` refuses
compact renderer rows, because they can omit fields that simulation needs. See
[State](concepts/state.md).

`step` accepts `[B,A]` actions or `[B,T,A]` action sequences. It returns the final State, the task
outputs, the State history and the final frame by default. Use `frames=None` to skip rendering.
Renderer-only classes refuse simulator operations. See [Stepping](concepts/dynamics.md).

## Registration and Variation

- `snv.make("CartPole", num_envs=2, device="cpu")` constructs by class name.
- `snv.list_envs()` lists the classes registered in this interpreter.
- `snv.import_errors()` explains missing families.
- `snv.info("CartPole")` returns metadata without constructing the environment.

Variation covers appearance, physics, reset and task configuration. Check what the family supports
before you use `set_variation` or reset options. See [Variations](concepts/variation-spaces.md).

State is a replay checkpoint and may contain privileged fields. stable-worldmodel selects learner
observations. See [Sibling projects](concepts/worldmodel.md).

Next: [Lifecycle](concepts/lifecycle.md), [Environments](environments.md), and
[Adding a component](adding/index.md).
