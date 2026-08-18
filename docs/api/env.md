# `stable_envs.core.env`

The contract every environment implements. Read this first: `scene()` is the method that carries the
repo's claim, and everything else exists to make that claim checkable.

`check_states` is the second thing to read. Every entry point that turns numbers into pixels used to go
through `torch.as_tensor(states, ...).reshape(-1, state_dim)`, and that reshape is the most dangerous
line in the package: hand it a frame stack whose size happens to divide by `state_dim` and it succeeds,
then renders a plausible-looking picture built out of pixel bytes read as coordinates — no exception, no
warning. The rule that replaces it is on the SHAPE rather than on the size, which is what makes it
impossible rather than unlikely: a state array is 2-D and its last axis is exactly `state_dim`.
`render` and `render_states` relax only the rank, because "arbitrary leading dims" is their documented
contract. NaN and inf are refused with the offending rows and named columns, and a batch SHORTER than
the env's is accepted and rendered, because that is the last batch of every ordinary epoch.

::: stable_envs.core.env
