# Fidelity methodology

Set each constant from a measurement against the reference renderer. Read the source to form a
hypothesis. Then compare pixels to decide.

Constants that read plausibly from the source can still be wrong. The table shows some examples from this repository.

| what the source suggested | what was true | how it showed |
|---|---|---|
| PushT flips y (`512 - y`) | no flip: `positive_y_is_up` is False | agent at row 348 instead of 162 |
| the T rotates about its centre of gravity, `cog = (0, 45)` | it rotates about the body origin | 44.6 px error |
| block fill `(142, 163, 183)`, truncating a 1.2× colour scale | `(143, 163, 184)`: `as_int()` rounds | off by one on every interior pixel of the block |
| walls are `draw_fat_segment(radius=2)`, half-width 2.0 | half-width 2.5, because `r > 2` adds caps | 8,012 wall pixels instead of 10,009 |
| Fetch's render state is `qpos + qvel` | it also needs `mocap_pos`, `mocap_quat` and the goal site | 29–138/255 error instead of 0 |
| MJWarp textures are the limit | MJWarp ignores alpha and paints invisible markers | mean error 79.34 → 7.47 after a model pre-pass |

## Practice

1. Document each constant with the measurement that fixed it, next to the constant. "8,012 wall pixels
   against the reference's 10,009" is useful. "Matches pygame" is not.
2. Time a speedup on the code path a caller actually reaches.
3. Test on states where the question arises. For example, draw order only matters where two bodies
   overlap.
4. Split the metric. Interior pixels (a flat 3×3 neighbourhood in the reference) must match exactly.
   Report all-pixel agreement separately with a looser bar. A wrong corner shape can cost 2% of
   interior pixels and nothing visible in an all-pixel mean.
5. Record rejected environments with the check that rejected them.

[Fidelity tiers](fidelity-tiers.md) describes the two tiers of claim. [Adding a component](../adding/index.md)
contains the contributor guides.
