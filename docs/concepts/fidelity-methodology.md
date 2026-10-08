# Calibrate against the reference, never against the source

This repo reproduces other people's renderers bit for bit. The single rule that makes that possible is:

> **A constant is right when a measurement says so. Reading the reference's source is how you form the
> hypothesis, not how you decide.**

It is stated this strongly because it is not a style preference — it is the accumulated shape of every
defect this repo has shipped. Every environment in it had at least one constant that read plausibly off
the source and was wrong, and in each case the plausible reading produced a picture that looked fine:

| what the source suggested | what was true | how the difference showed up |
|---|---|---|
| PushT flips y (`512 - y`), per a long docstring in the draw override | no flip: `positive_y_is_up` is False | the agent drawn at row 348 where the reference has it at 162 |
| the T rotates about its centre of gravity, since it sets `cog = (0, 45)` | it rotates about the body origin | 44.6 px against 5.7e-14 for the right form, over 6 random poses |
| block fill `(142, 163, 183)`, truncating a 1.2× colour scale | `(143, 163, 184)` — `as_int()` **rounds** | off by one on every interior pixel of the block |
| walls are `draw_fat_segment(radius=2)`, so half-width 2.0 | half-width 2.5, because `r > 2` adds caps and an extended quad | 8,012 wall pixels against the reference's 10,009 |
| Fetch's render-relevant state is `qpos + qvel` | plus `mocap_pos`, `mocap_quat` and the goal site | 29–138/255 vs 0/255 |
| MJWarp renders textured surfaces flat, so textures are the limit | it never reads **alpha**, and paints invisible markers opaque | mean 79.34 → 7.47 from a one-line model pre-pass |
| a `(B, w)` tensor with a non-zero row stride holds per-environment values | it holds whatever it holds; stride says nothing | the static-prefix bake was dead code at every batch above 1 |

The last row is the general case, not an outlier: the guard *read* as if it were checking the right
thing, it had a docstring explaining why, and it never fired. Nobody had counted.

Concretely, that means:

1. **Every constant carries the measurement that fixed it**, in the docstring, next to the constant.
   "8,012 wall pixels against the reference's 10,009" is the useful form; "matches pygame" is not.
2. **Never assert a speedup you have not timed on the path a caller reaches.** The bake above was
   honestly measured at 1.83× — on a code path no batched render could enter.
3. **Search over states where the thing you are resolving actually occurs.** PushT's draw order came
   back a meaningless near-tie (0.9127 vs 0.9110) over random rollout poses, because the agent and the
   block almost never overlap there. It resolved instantly once overlapping poses were added. A parity
   search over states that do not exercise the question returns confident noise.
4. **Split the metric so a real error cannot hide in an aggregate.** Interior pixels (3×3 flat
   neighbourhood in the reference) must be 100% exact; all-pixel agreement is reported separately and
   held to a looser bar. A mitred outline instead of a round one costs 2.2% of interior pixels and
   nothing visible in an all-pixel mean.
5. **A negative result is a result.** Atari, RoboCasa, CarRacing, LunarLander and Procgen are written up
   with the gate that rejected them, not omitted.

The two published tiers — identity against the official renderer, versus a different rasterizer of
the same scene — are [Fidelity tiers](fidelity-tiers.md). The worked port that applies this method
is [Adding an environment](../adding-an-env.md).
