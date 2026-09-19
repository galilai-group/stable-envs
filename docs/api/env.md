# `stable_envs.env`

The contract every environment implements. Concrete classes (`snv.PushT`, …) are the primary API; this module is the shared `Env` they inherit.

Public vocabulary:

- `Env` — `reset`, stateful `step`, `render`, `close`, context manager
- `State` — alias for `tensordict.TensorDict`
- `StepResult` — alias for the TensorDict `step` returns
- `pack` / `unpack` — public State ↔ packed float32 row (the unchanged Step-8 layout)

See [State and specs](../concepts/state.md) and [Lifecycle](../concepts/lifecycle.md).

`check_states` is the second thing to read. Every entry point that turns numbers into pixels used to go through `torch.as_tensor(states, ...).reshape(-1, state_dim)`, and that reshape is the most dangerous line in the package: hand it a frame stack whose size happens to divide by `state_dim` and it succeeds, then renders a plausible-looking picture built out of pixel bytes read as coordinates. The rule that replaces it is on the SHAPE rather than on the size: a state array is 2-D and its last axis is exactly `state_dim`. `render` relaxes only the rank, because "arbitrary leading dims" is its documented contract. NaN and inf are refused with the offending rows and named columns, and a batch SHORTER than the env's is accepted and rendered, because that is the last batch of every ordinary epoch.

`stable_envs.core.env` still re-exports the same objects until Wave 3.

::: stable_envs.env
