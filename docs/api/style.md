# `stable_envs.core.style`

Appearance as state. A `StyleTensor` is the batched form of a `spaces.Dict`: one flat `(B, style_dim)`
float32 tensor plus the layout that says which columns are `block.color`, so the same code serves one
shared appearance and 4,096 different ones.

Read the module docstring before changing anything here. It records **the batch rule** — the single
operation that is allowed to change a style's batch, and the three shipped defects that existed because
that operation was open-coded at each call site and the wrong way to do it is one slice shorter than the
right way.

::: stable_envs.core.style
