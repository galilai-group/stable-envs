# Variation

**Variation** is persistent environment configuration. It is not appearance-only. A leaf may
change how the world looks, how it is shaped, how it integrates, how an episode is reset, or
which task is being solved — without being folded into the instantaneous **State**.

```text
render-only          background.color, agent.color
dynamics-only        gravity, joint damping, agent.speed
reset / task         spawn distribution, task.min_steps
render + dynamics    block.scale, morphology
```

Do not invent knobs. Declare the factors the environment actually has. Variation may live on
the env rather than on every method signature; `set_variation` / `render(..., variation=)`
are the public surface.

The world (`state["agent", "position"]`, …) is still the instantaneous dynamic state. A frame
is `f(state, variation)`. Where native dynamics exist, `next_state = g(state, action, variation)`.

```python
import stable_envs as snv

env = snv.PushT(num_envs=64)
state = env.reset(seed=0)
frames = env.render(state, variation={"block.color": [220, 40, 40]})
```

Defaults are the official configuration: rendering with no `variation=` (or with
`default_variation()`) reproduces the reference picture. Render fidelity and contact-solver
fidelity remain separate claims — a gravity leaf does not make a reimplemented solver bit-exact.

## Inspect the space

Every env declares a `variation_space` tree. The public tensor contract is the **flat row**.

```python
env.variation_space.names()
# ['agent.color', 'agent.inner_color', 'agent.scale', 'background.color', …]

env.variation_spec          # one Spec: shape (variation_dim,), names per column
env.variation_dim           # width of that row
env.variation_names         # leaf paths, one per leaf (an RGB colour is still one name)
```

`variation_spec` is a `snv.Spec`, not environment metadata. Use `snv.info("PushT")` for
family / domain / slug.

`env.variation_space` is a `stable_envs.spaces.Dict` of nested `Dict`, `Box`, `RGBBox` and
`Discrete` leaves ([API](../api/spaces.md)):

- `Dict(spaces, sampling_order=[...])` draws top-level keys in that order, with unlisted keys after
  them in insertion order; `sample()` and `update()` follow it, and `ordered_names()` yields every
  dotted path in that order. A key in `sampling_order` that is not in `spaces` is a `ValueError`.
- `Box(..., exact=True)` keeps `init_value` and later values as given, so a Python float stays
  float64; without it values are cast to the leaf's `dtype` and shape. A `Box` given an explicit
  `init_value` holds it as a 0-d array when its shape is `()`.
- `set_init_value(value, reset_value=True)` replaces a leaf's default; with `reset_value=False` the
  current value is left alone. `Dict.set_init_value` forwards both arguments.

## Defaults and samples

```python
row = env.default_variation()     # (variation_dim,) — official defaults
rows = env.sample_variation(64)   # (64, variation_dim) independent draws
bg = env.sample_variation(64, paths=["background.color"])
```

`sample_variation(batch)` draws **B independent** configurations. `paths=` randomises only
those leaves; everything else stays at the official default. An unknown path is an error,
not a silent no-op.

Install a row on the env with `env.set_variation(rows)` or pass it into `render` (below).
`get_variation()` returns the currently installed `(n, variation_dim)` tensor.

## Reset

`reset(seed, options)` follows stable-worldmodel's rule. Leaves named in
`options["variation"]` are sampled per lane (lane `i` seeded with `seed + i`); leaves in
`options["variation_values"]` are set, with one value shared by the batch or one per lane.
Every other leaf returns to its default, including physics leaves on the simulator and in
State. The defaults are the registered values until `set_variation` replaces them, so a reset
with no options returns to whatever `set_variation` last installed. Values a reset's options
choose, or a physics value adopted from a restored State, last for that episode only.

```python
env.set_variation({"gravity.z": [-5.0]})
env.reset(seed=0, options={"variation": ["gravity.z"]})   # sampled for this episode
env.reset(seed=1)                                         # gravity.z is -5.0 again
```

Families that wrap the stable-worldmodel source (PushT, Rocket, OGBench Cube/Scene) apply
`set_variation` to the source's defaults, which take effect at the next reset, and each also
applies some leaves to the running episode:

- PushT: its appearance leaves (`PushT.immediate_variation_leaves`: colours, goal pose,
  `rendering.render_goal`). Body shapes and scales need a rebuilt physics space and start poses
  exist only at reset, so those wait for the next reset.
- Rocket: its colour leaves; every later reset draws them, so `get_variation` always matches the
  drawn colours. Its start leaves (drop height, lateral offset, tilt) wait for the next reset.
- OGBench Cube/Scene: cube size and colour, floor colour, camera angle, light intensity and lock
  colour. Reset-time placements wait for the next reset, and `agent.color` follows the source's
  compile rule.

## Coverage against stable-worldmodel

For each class with a stable-worldmodel counterpart that has a variation space, a `VariationMap`
in `tests/helpers/reference/variation_maps/` (and `tests/helpers/reference/variation.py`) places
every swm leaf in exactly one of: mapped (identical, renamed, scaled, affine, colour or enum),
unsupported, dead upstream, not applicable or swm-only. The maps are test evidence, not a shipped
API, and `CONTRIBUTING.md` requires one for a new class whose source varies.

- DM Control: `floor.color` is mapped on all 13 mapped cells. On the five trackcom cells with a
  map the checker is drawn from a texture of per-texel palette weights times each row's colours, so
  one MJWarp context serves every colour. `DMCWalkerWalk` also maps `floor.rotation_y`, a physics
  leaf that tilts the floor geom and that every renderer draws per row.
- OGBench Cube and Scene: `camera.angle_delta`, `light.intensity`, `cube.size` and, on Scene,
  `lock_color` are per-world inputs of one MJWarp context per family, so a single batch renders
  any mix of their values; Cube maps 11 swm leaves and Scene 12.

## One Variation vs B independent Variations

`num_envs` is how many copies the object holds. Configuration follows the same batch axis:

| you pass | meaning |
|---|---|
| nothing | current / default configuration, broadcast to the batch |
| one row `(variation_dim,)` or a dict of scalars | **one** configuration, shared by every environment |
| a `(B, variation_dim)` row | **B independent** configurations, one per environment |

```python
shared = env.default_variation()
frames = env.render(state, variation=shared)          # one look, B frames

many = env.sample_variation(env.num_envs, rng=env.rng)
frames = env.render(state, variation=many)            # B looks, one launch
```

Not every family paints a different appearance per environment. Tile-atlas families
(MiniGrid, BabyAI) bake one palette per **call**, so a batched row still yields one look
for that render. Scene-built families (PushT, …) do vary per environment. Count distinct
frames if the distinction matters for your experiment.

## Dict overrides vs flat / batched rows

`render(..., variation=)` and `set_variation(...)` accept `VariationLike`: a mapping of
leaf overrides, **or** a flat / batched tensor or numpy array.

```python
# Mapping: named leaves. Unmentioned leaves stay at the current / default values.
frames = env.render(state, variation={"block.color": [220, 40, 40], "agent.scale": 1.4})

# Flat row: every column, official width.
row = env.default_variation()
frames = env.render(state, variation=row)

# Batched rows: one configuration per environment.
rows = env.sample_variation(env.num_envs)
frames = env.render(state, variation=rows)
```

Do not pass both a non-mapping row and the developer `style=` argument.

## Reproducibility and storage

Variation is data. Store its row alongside the full State snapshot:

```python
snapshot = state.clone()                     # complete, owned State
variation_row = env.get_variation()           # (n, variation_dim)
# later
frames = env.render(snapshot, variation=variation_row)
```

Compact RenderState packing is private. To resume simulation, restore the full
State under compatible persistent configuration; a render-only projection cannot
restore missing velocities, RNG or lifecycle fields.

`sample_variation(..., rng=)` is deterministic given that generator. `reset(seed=0)`
reseeds `env.rng`; pass an explicit `rng` when you want Variation draws independent of
reset.

Defaults (`default_variation()`) are the official configuration. Saving them is optional;
omitting `variation=` at render time reproduces the same picture.

## Next

- [State, Variation, Frame](state.md) — public State vs compact render state
- [API: spaces](../api/spaces.md) — the `variation_space` tree
