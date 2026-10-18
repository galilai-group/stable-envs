# Quickstart

Install the core and construct an environment by class name. See [Install](install.md) for extras
and pinned profiles.

```bash
pip install stable-envs
```

```python
import numpy as np
import stable_envs as snv

env = snv.CartPole(num_envs=2, device="cpu")
state = env.reset(seed=17)
pixels = env.render(state)
action = np.array([[0], [1]], dtype=np.int64)
out = env.step(action)
assert env.set_state(state) is None
assert (env.step(action) == out).all()
env.close()
```

`num_envs` is the number of environments that run together. You can think of it as the batch size.
`reset`, `step` and `set_state` use this batch. For most classes, `render(state)` also accepts a State
with a different batch. `PyFlytRocketLanding`, `OGBCube` and `OGBScene` refuse a State whose batch is
not `num_envs`.

`env.close()` stops the simulator's worker processes and frees the environment's own buffers.
Renderer contexts that other environments share stay cached. A second call has no effect. An
environment is also a context manager: `with snv.CartPole(num_envs=2) as env:` closes it at the end
of the block.

## State and frames

`get_state()` returns an independent TensorDict snapshot of the full State. `state` is an alias.
Fields include physical, task, lifecycle and RNG state, as the family needs. `state_spec` describes
those fields. `action_spec` describes actions. `frame_spec` describes uint8 `(H,W,3)` output without
the batch axis.

For the CartPole example above, with `num_envs=2`:

```python
state.batch_size                      # torch.Size([2])
list(state.keys())
# ['x', 'x_dot', 'theta', 'theta_dot', 'lifecycle', 'task', 'rng', 'physics']
state["x"].shape, state["x"].dtype    # (torch.Size([2, 1]), torch.float64)
state["rng", "state"].shape           # torch.Size([2, 4])
list(state["physics"].keys())         # ['gravity', 'masscart', 'masspole', 'length', 'force_mag', 'tau']

env.action_spec                       # Spec(shape=(1,), dtype=int64, n=2)
env.frame_spec                        # Spec(shape=(400, 600, 3), dtype=uint8, bounds=[0, 255])
pixels.shape, pixels.dtype            # (torch.Size([2, 400, 600, 3]), torch.uint8)
```

CartPole renders only at the native 400x600 size of the Gymnasium renderer.

`render(state)` converts the State to the renderer's private input internally. `set_state` refuses
compact renderer rows, because they can omit fields that simulation needs. See
[State](concepts/state.md).

`step` accepts `[B,A]` actions or `[B,T,A]` action sequences. It returns the final State, the task
outputs, the State history and the final frame by default. Use `frames=None` to skip rendering.
Renderer-only classes refuse simulator operations. See [Stepping](concepts/dynamics.md).

```python
list(out.keys())
# ['state', 'reward', 'terminated', 'truncated', 'observation', 'info_state',
#  'valid', 'states', 'state_valid', 'frames', 'frame_valid']
out["reward"].shape                   # torch.Size([2])
out["states"].batch_size              # torch.Size([2, 1])
out["frames"].shape                   # torch.Size([2, 1, 400, 600, 3])
```

The task outputs differ by family. `states` and `frames` keep a time axis, even for one action.

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
