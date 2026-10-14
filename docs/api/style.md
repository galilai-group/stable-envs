# `stable_envs._core.style`

Internal. A `StyleTensor` is the compiled form of Variation: one flat `(B, variation_dim)` float32
tensor plus a layout that maps columns to leaves such as `block.color`. Normal code uses
`variation_spec`, `default_variation`, `sample_variation` and `render(..., variation=)`.

The module defines the batch rule: the only operation allowed to change a compiled row's batch size.

::: stable_envs._core.style
