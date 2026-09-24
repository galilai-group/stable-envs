# `stable_envs.env`

Concrete classes such as `snv.CartPole` are the primary API. The shared `Env`
provides the simulator/rendering boundary; family adapters implement authoritative
simulation without adding a shared Observation class.

- `reset`, `get_state` and `state` return independent exhaustive State snapshots
  where simulator semantics are supported.
- `set_state(full_state)` restores all required fields and returns `None`.
- `step` returns State and the task's defined reward/success/done values.
- `transition` and `unroll` are optional; delegated families may explicitly refuse.
- `render(full_state)` projects internally and returns a uint8 Frame.
- `close` and context-manager methods implement resource ownership.
- State/action/frame/Variation specs describe shapes and dtypes; family validators
  enforce complete schemas and semantic constraints before restoration.

State is a TensorDict checkpoint, potentially privileged, not the learner
observation. Variation can affect physics, reset, task or appearance.

Contributor/internal interfaces include `_pack_render_state`,
`_unpack_render_state`, `_get_render_state`, `_set_render_state`, `_render`,
`state_dim`/`state_names`, `scene`, and style helpers. Compact RenderState layouts
are not full State schemas. The old public `pack_render_state`/`unpack_render_state`
methods are absent. Renderer input shape/finite-value checks do not replace
full-State restoration validation.

See [State](../concepts/state.md), [lifecycle](../concepts/lifecycle.md) and
[adding a component](../adding/index.md). Existing source comments about compact
renderer layouts should be read within this private rendering boundary.

::: stable_envs.env
