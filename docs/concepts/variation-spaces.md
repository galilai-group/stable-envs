# Variations

**Variation** is persistent environment configuration. It is separate from the instantaneous
**State**. A leaf can change appearance, shape, dynamics, reset or task.

```text
render-only          background.color, agent.color
dynamics-only        gravity, joint damping, agent.speed
reset / task         spawn distribution, task.min_steps
render + dynamics    block.scale, morphology
```

A frame is `f(state, variation)`. Where native dynamics exist, `next_state = g(state, action, variation)`.
Each env declares only the factors its source actually has.

```python
import stable_envs as snv

env = snv.CartPole(num_envs=64)
state = env.reset(seed=0)
frames = env.render(state, variation={"pole.color": [220, 40, 40]})
```

The defaults are the official configuration. Rendering without `variation=` gives the reference
picture.

## Inspect the space

Every env has a `variation_space` tree. The public tensor form is a flat row.

```python
env.variation_space.names()
# ['agent.color', 'agent.inner_color', 'agent.scale', 'background.color', …]

env.variation_spec          # a Spec: shape (variation_dim,), names per column
env.variation_dim           # row width
env.variation_names         # one path per leaf (an RGB colour is one name)
```

`env.variation_space` is a `stable_envs.spaces.Dict` of nested `Dict`, `Box`, `RGBBox` and `Discrete`
leaves ([API](../api/spaces.md)):

- `Dict(spaces, sampling_order=[...])` draws the listed keys first, then the others in insertion order.
  `sample()`, `update()` and `ordered_names()` follow this order. An unknown key in `sampling_order`
  raises `ValueError`.
- `Box(..., exact=True)` keeps values as given, so a Python float stays float64. Without it, values are
  cast to the leaf's `dtype` and shape.
- `set_init_value(value, reset_value=True)` replaces a leaf's default. With `reset_value=False` the
  current value is kept.

## Defaults and samples

```python
row = env.default_variation()     # (variation_dim,): official defaults
rows = env.sample_variation(64)   # (64, variation_dim): independent draws
bg = env.sample_variation(64, paths=["background.color"])
```

`paths=` randomises only the named leaves. The others stay at the default. An unknown path raises an
error.

`env.set_variation(rows)` installs rows on the env. `get_variation()` returns the installed
`(n, variation_dim)` tensor.

## Reset

`reset(seed, options)` follows stable-worldmodel's rule:

- Leaves named in `options["variation"]` are sampled per lane. Lane `i` uses seed `seed + i`.
- Leaves in `options["variation_values"]` are set, with one value for the batch or one per lane.
- Every other leaf returns to its default, including physics leaves.

The defaults are the registered values until `set_variation` replaces them. A reset with no options
returns to the values `set_variation` last installed. Values chosen through reset options last for that
episode only.

```python
env.set_variation({"gravity.z": [-5.0]})
env.reset(seed=0, options={"variation": ["gravity.z"]})   # sampled for this episode
env.reset(seed=1)                                         # gravity.z is -5.0 again
```

PushT, Rocket and OGBench Cube/Scene wrap the stable-worldmodel source. `set_variation` changes the
source's defaults, which apply at the next reset. Some leaves also apply to the running episode:

- PushT: appearance leaves (`PushT.immediate_variation_leaves`: colours, goal pose,
  `rendering.render_goal`). Shapes, scales and start poses wait for the next reset.
- Rocket: colour leaves. Start leaves (drop height, lateral offset, tilt) wait for the next reset.
- OGBench Cube/Scene: cube size and colour, floor colour, camera angle, light intensity and lock
  colour. Placements wait for the next reset.

## One Variation or B Variations

| you pass | meaning |
|---|---|
| nothing | current configuration, broadcast to the batch |
| one row `(variation_dim,)` or a dict of scalars | one configuration shared by every lane |
| a `(B, variation_dim)` tensor | one configuration per lane |

```python
shared = env.default_variation()
frames = env.render(state, variation=shared)          # one look, B frames

many = env.sample_variation(env.num_envs, rng=env.rng)
frames = env.render(state, variation=many)            # B looks, one call
```

Some renderers apply one appearance per call, not per lane. Count distinct frames if this matters for
your experiment.

`render(..., variation=)` and `set_variation(...)` accept a mapping of leaf overrides, a flat row, or
batched rows. With a mapping, unnamed leaves keep their current values. Do not pass a row together
with the developer `style=` argument.

## Storage

Store the Variation row next to the full State:

```python
snapshot = state.clone()
variation_row = env.get_variation()           # (n, variation_dim)
frames = env.render(snapshot, variation=variation_row)
```

To resume simulation, restore the full State under a compatible configuration.

`sample_variation(..., rng=)` is deterministic for a given generator. `reset(seed=0)` reseeds
`env.rng`. Pass your own `rng` to keep Variation draws independent of reset.

## Coverage against stable-worldmodel

Each class with a stable-worldmodel counterpart that has a variation space has a `VariationMap` in
`tests/helpers/reference/variation_maps/`. It places every stable-worldmodel leaf in one group:
mapped (identical, renamed, scaled, affine, colour or enum), unsupported, dead upstream, not
applicable, or swm-only. The maps are tests, not a shipped API.

## Next

- [State, Action, Variation and Frame](state.md)
- [API: spaces](../api/spaces.md)
