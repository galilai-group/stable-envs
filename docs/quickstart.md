# Quickstart

Install the core and construct an accepted environment by class name. See
[Install](install.md) for optional extras and pinned dependency profiles.

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

`num_envs` is the simulator batch size. `close()` is idempotent; a context manager
closes its owned resources. Renderer short-batch support does not change a
backend's full-State batch requirements.

## State and frames

`get_state()` returns an independent exhaustive TensorDict snapshot. `state` is
its alias. Named fields include physical, task, lifecycle and RNG state as needed
by the family. `state_spec` describes those fields; `action_spec` describes actions;
`frame_spec` describes uint8 `(H,W,3)` output without the batch axis.

`render(state)` projects internally to private RenderState. Compact renderer rows
can omit fields needed for continuation and are refused by `set_state`. There is
no public `pack_render_state` or `unpack_render_state` API. See [State](concepts/state.md).

`step` accepts `[B,A]` actions or `[B,T,A]` action sequences. It returns the final
State and defined task outputs, plus a frame by default. Sequences also capture
State history by default; use `capture="states"` for physics-only execution. Renderer-only cells refuse full simulator operations.
See [dynamics](concepts/dynamics.md).

## Registration, Variation and downstream use

Class names are canonical: `snv.make("CartPole", num_envs=2, device="cpu")`.
`snv.list_envs()` lists imports available in the current interpreter;
`snv.import_errors()` explains missing optional families. `snv.info("CartPole")`
returns metadata without constructing the environment.

Variation includes appearance, physics, reset and task configuration. Consult the
family's bounded support and scope before using `set_variation` or reset options;
see [Variation](concepts/variation-spaces.md).

Stable-worldmodel owns learner input selection, observation packaging and dataset
filtering. Full State is a replay checkpoint and may contain privileged fields;
observation integration remains deferred. See [sibling projects](concepts/worldmodel.md).

Next: [lifecycle](concepts/lifecycle.md), [environment listing](environments.md),
and [adding an environment, task or renderer](adding/index.md).
