# `stable_envs.compat.worldmodel`

Adapters that let `stable_worldmodel` drive a stable-envs renderer without changing the caller's code.
The mismatch is structural: `swm.World` loops over a list of single `gym.Env` objects, while a
stable-envs env has no `reset`, a `step` that takes `(B, action_dim)` and a `render` that is a pure
function of a `(B, state_dim)` stack.

Three adapters are offered, and they are different trades rather than three spellings of one thing —
drop-in single-env compatibility, a batched pool, and a direct renderer swap.

::: stable_envs.compat.worldmodel
