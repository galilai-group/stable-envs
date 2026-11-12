# Specs and spaces

A [spec](../terms.md#spec) is a `Spec` or `CompositeSpec` that describes the shape, dtype, device, bounds and column names of the values in one [batch row](../terms.md#batch-row). Every environment has four, for its [state](../terms.md#state), action, [frame](../terms.md#frame) and [variation](../terms.md#variation), and its variation comes from a tree of space classes in `stable_envs.spaces`.

```python
import numpy as np
import stable_envs as snv

env = snv.TwoRoom(num_envs=4, device="cpu")
print(env.action_spec)                          # Spec(shape=(2,), dtype=float32, bounds=[-1, 1])
print(env.frame_spec.batch_shape(env.num_envs)) # (4, 224, 224, 3)
print(env.state_spec.keys(nested=True))         # [('agent', 'position'), ('target', 'position')]
print(env.state_spec["agent", "position"].names)  # ('agent_x', 'agent_y')

actions = env.action_spec.sample(env.num_envs, rng=np.random.default_rng(0))
print(actions.shape, actions.dtype)               # torch.Size([4, 2]) torch.float32
print(env.action_spec.contains(actions * 3))      # False, outside [-1, 1]
print(env.state_spec.contains(env.reset(seed=0))) # True
env.close()
```

## Specs

| Spec | Shape | Describes |
|---|---|---|
| `env.state_spec` | a `CompositeSpec` with one leaf per state field | The state that `reset`, `get_state` and `step` return, with each field's dtype and, where the family declares them, bounds and names. |
| `env.action_spec` | `(action_dim,)` | One action. A discrete action has `discrete_n` set and an integer dtype. |
| `env.frame_spec` | `(H, W, 3)`, uint8, bounds `[0, 255]` | One frame at the environment's size. |
| `env.variation_spec` | `(variation_dim,)` | One compiled variation row, with one name per column, such as `agent.color[0]`. |

A spec's `shape` leaves out the batch axis: `action_spec.shape == (2,)` means `step` takes `(B, 2)` or `(B, T, 2)` actions. `batch_shape(B)`, `zeros(B)` and `sample(B)` add it back. `validate` checks the trailing shape, casts to the spec's dtype, and refuses NaN, infinity and values outside the bounds or the discrete values, and `contains` returns False where `validate` would raise. `Spec.sample` needs bounds or `discrete_n`. For state fields without bounds, draw states with `env.sample_states(n)` instead.

`CompositeSpec` nests specs by key path, the way state nests tensors: `spec["agent", "position"]` is the leaf for `state["agent", "position"]`. Its `zeros` and `sample` return a TensorDict with batch size `(B,)`, and its `validate` also refuses missing and unexpected keys.

::: stable_envs.specs.Spec
    options:
      heading_level: 3

::: stable_envs.specs.CompositeSpec
    options:
      heading_level: 3

## Spaces

`stable_envs.spaces` holds Gymnasium `Discrete`, `Box` and `Dict` spaces that also keep a default and a current value. A family builds one tree of them in `make_variation_space`, and `env.variation_space` is that tree for the instance:

```python
from stable_envs import spaces

space = spaces.Dict({
    "agent": spaces.Dict({
        "color": spaces.RGBBox(init_value=(255, 0, 0)),
        "radius": spaces.Box(0.05, 0.2, shape=(1,), init_value=[0.1]),
    }),
    "door": spaces.Dict({"number": spaces.Discrete(3, init_value=1)}),
}, seed=0)

print(space.names())                  # ['agent.color', 'agent.radius', 'door.number']
print(space.value["agent"]["color"])  # [255   0   0]
space.update(["agent.radius"])        # resample one leaf
print(space.check())                  # True
space.reset()                         # back to the defaults
print(space.flat_dim)                 # 5
```

Leaves are addressed by dotted paths such as `"agent.color"`, the same names that `env.set_variation` and reset options take. `flat_layout`, `pack` and `unpack` convert between the tree and the flat float32 row that `env.get_variation()` returns for each batch row. In most code you set variation through the environment rather than the space.

- `Dict(spaces, sampling_order=[...])` draws the listed keys first, then the others in insertion order. `sample()`, `update()` and `ordered_names()` follow this order, and an unknown key in `sampling_order` raises `ValueError`.
- `Box(..., exact=True)` keeps values as given: a Python float stays float64. Without it, the leaf casts values to its `dtype` and shape.
- `set_init_value(value, reset_value=True)` replaces a leaf's default. With `reset_value=False` the leaf keeps its current value.

::: stable_envs.spaces.Discrete
    options:
      heading_level: 3
      inherited_members: false
      members: [sample, contains]

::: stable_envs.spaces.Discrete.value
    options:
      heading_level: 4

::: stable_envs.spaces.Discrete.init_value
    options:
      heading_level: 4

::: stable_envs.spaces.Discrete.set_value
    options:
      heading_level: 4

::: stable_envs.spaces.Discrete.set_init_value
    options:
      heading_level: 4

::: stable_envs.spaces.Discrete.reset
    options:
      heading_level: 4

::: stable_envs.spaces.Discrete.check
    options:
      heading_level: 4

::: stable_envs.spaces.Box
    options:
      heading_level: 3
      inherited_members: false
      members: [sample, contains, set_value, set_init_value]

::: stable_envs.spaces.Box.value
    options:
      heading_level: 4

::: stable_envs.spaces.Box.init_value
    options:
      heading_level: 4

::: stable_envs.spaces.Box.reset
    options:
      heading_level: 4

::: stable_envs.spaces.Box.check
    options:
      heading_level: 4

::: stable_envs.spaces.RGBBox
    options:
      heading_level: 3
      inherited_members: false
      members: false

::: stable_envs.spaces.Dict
    options:
      heading_level: 3
      inherited_members: false
      members: [sample, value, init_value, set_value, set_init_value, reset, check, update, names, ordered_names, flat_layout, flat_dim, pack, unpack]

::: stable_envs.spaces.reset_variation_space
    options:
      heading_level: 3
