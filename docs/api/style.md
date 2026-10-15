# `stable_envs._core.style`

This module is internal. A `StyleTensor` is the compiled form of Variation: one flat `(B, variation_dim)` float32
tensor plus a layout that maps columns to leaves such as `block.color`. Normal code uses
`variation_spec`, `default_variation`, `sample_variation` and `render(..., variation=)`.

The module defines the batch rule. The batch rule is the only operation that may change the batch
size of a compiled row.

::: stable_envs._core.style
