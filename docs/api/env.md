# `stable_envs.env`

The contract every environment implements. Concrete classes (`snv.PushT`, …) are the primary API; this module is the shared `Env` they inherit.

Public vocabulary:

- `Env` — `reset`, stateful `step`, functional `transition` / `unroll`, `render`, `close`
- `State` — instantaneous dynamic world state; alias for `tensordict.TensorDict`
- `Action` — intervention at a timestep
- `Variation` / `VariationLike` — persistent environment configuration (not appearance-only); `render(..., variation=)` accepts a mapping or a flat / batched row
- `Frame` — pixels from `render`; described by `frame_spec`
- `StepResult` — TensorDict `step` returns; always has `"state"`
- `pack_render_state` / `unpack_render_state` / `render_state_dim` / `render_state_names` — public State ↔ compact render row (not always lossless)

Developer / internal (used by env implementations and tests; not taught in the user guide):

- `StyleTensor`, `resolve_style`, `style_dim` / `style_names` / `default_style` / `get_style` / `set_style` / `sample_style`
- `pack` / `unpack` / `state_dim` / `state_names` — the same compact render row, implementation names
- `scene()` — Scene2D builders only; tile and ray-cast families override `render` instead

See [State, Action, Variation, Frame](../concepts/state.md) and [Lifecycle](../concepts/lifecycle.md).

`check_states` is the second thing to read. Every entry point that turns numbers into pixels used to go through `torch.as_tensor(states, ...).reshape(-1, state_dim)`, and that reshape is the most dangerous line in the package: hand it a frame stack whose size happens to divide by `state_dim` and it succeeds, then renders a plausible-looking picture built out of pixel bytes read as coordinates. The rule that replaces it is on the SHAPE rather than on the size: a state array is 2-D and its last axis is exactly `state_dim`. `render` relaxes only the rank, because "arbitrary leading dims" is its documented contract. NaN and inf are refused with the offending rows and named columns, and a batch SHORTER than the env's is accepted and rendered, because that is the last batch of every ordinary epoch.

::: stable_envs.env
