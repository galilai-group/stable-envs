# Variation

**Variation** is appearance and layout that can change without changing State. The world
(`state["agent", "position"]`, …) stays put; colours, scales, toggles, and similar knobs
do not.

```python
import stable_envs as snv

env = snv.PushT(num_envs=64)
state = env.reset(seed=0)
frames = env.render(state, variation={"block.color": [220, 40, 40]})
```

A frame is a deterministic function of compact render state **plus** Variation. Defaults
are the official appearance: rendering with no `variation=` (or with `default_variation()`)
reproduces the reference picture.

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

## Defaults and samples

```python
row = env.default_variation()     # (variation_dim,) — official appearance
rows = env.sample_variation(64)   # (64, variation_dim) independent draws
bg = env.sample_variation(64, paths=["background.color"])
```

`sample_variation(batch)` draws **B independent** appearances. `paths=` randomises only
those leaves; everything else stays at the official default. An unknown path is an error,
not a silent no-op.

Install a row on the env with `env.set_variation(rows)` or pass it into `render` (below).
`get_variation()` returns the currently installed `(n, variation_dim)` tensor.

## One Variation vs B independent Variations

`num_envs` is how many copies the object holds. Appearance follows the same batch axis:

| you pass | meaning |
|---|---|
| nothing | current / default appearance, broadcast to the batch |
| one row `(variation_dim,)` or a dict of scalars | **one** appearance, shared by every environment |
| a `(B, variation_dim)` row | **B independent** appearances, one per environment |

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

# Batched rows: one appearance per environment.
rows = env.sample_variation(env.num_envs)
frames = env.render(state, variation=rows)
```

Do not pass both a non-mapping row and the developer `style=` argument.

## Reproducibility and storage

Variation is data. Store the flat row next to the compact render state:

```python
render_row = env.pack_render_state(state)     # (n, render_state_dim)
variation_row = env.get_variation()           # (n, variation_dim)
# later
state = env.unpack_render_state(render_row)
frames = env.render(state, variation=variation_row)
```

`sample_variation(..., rng=)` is deterministic given that generator. `reset(seed=0)`
reseeds `env.rng`; pass an explicit `rng` when you want Variation draws independent of
reset.

Defaults (`default_variation()`) are the official appearance. Saving them is optional;
omitting `variation=` at render time reproduces the same picture.

## Next

- [State, Variation, Frame](state.md) — public State vs compact render state
- [API: spaces](../api/spaces.md) — the `variation_space` tree
- Contributor note: [variation collapses](../research/variation_collapses.md) — why
  per-environment appearance is measured, not assumed
