# `stable_envs._style`

Developer / internal. A `StyleTensor` is the compiled form of Variation: one flat
`(B, variation_dim)` float32 tensor plus the layout that says which columns are
`block.color`. Ordinary callers use `variation_spec` / `default_variation` /
`sample_variation` / `render(..., variation=)`.

This module records **the batch rule** — the single operation that is allowed to change a
compiled row's batch, and the three shipped defects that existed because that operation
was open-coded at each call site.

::: stable_envs._style
