# Contributing

## The method: calibrate against the reference, never against the source

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
   with the gate that rejected them, not omitted. See
   [when does this pay off](docs/research/when_does_this_pay_off.md).

## Adding an environment

[`docs/adding-an-env.md`](docs/adding-an-env.md) is the worked procedure, with a checklist at the end.
The short version: drop a module in `stable_envs/envs/` (auto-discovered), implement `scene()` or
override `render()`, bridge to the official renderer with `reference_env` / `reference_set_state` /
`reference_render`, and run the suite — it is parameterised over `REGISTRY`, so a new env is covered
without editing a test.

## Running things

Nothing in this repo is measured on a login node, and nothing that measures is measured on a busy one.

```bash
sbatch scripts/run_smoke_tests.sbatch    # one env per family, ~3.4 min
sbatch scripts/run_gpu_tests.sbatch      # full conformance suite
pytest tests/ -v -rs                     # the same, if you already hold a GPU
```

The suite requires CUDA by design and skips loudly rather than going green without it. Run it through
`stable_envs.testing.supervisor` (which is what the sbatch scripts do): a segfault or `SIGABRT` out of
MuJoCo, EGL or the driver ends the interpreter, and a bare pytest then reports nothing at all about
which test caused it.

**CPU baselines are not reproducible under contention** and GPU numbers are. The same official-renderer
measurement gave 422 frames/s on an idle node and 3 frames/s with three other jobs resident, while the
GPU column reproduces across nodes to 0.4%. Take a CPU baseline on a node you have to yourself, quote
the ratio as an order of magnitude rather than to three digits, and say which it was.

## What not to do

- Do not relax `parity_interior_min` to make a test pass. It may be lowered only together with a
  `parity_note` naming the mechanism, which the test then prints.
- Do not release an EGL context or close a `mujoco.Renderer`. Releasing one corrupts rendering in every
  other live context in the process, silently, and the damage surfaces later as pixel differences in an
  unrelated env. `stable_envs/testing/resource_guard.py` fails the test that does it.
- Do not create contexts, renderers or envs per state, per style value or per call. Context creation
  that scales with the batch means a per-environment value is being turned into a per-environment
  *scene*.
- Do not commit measurement artifacts. Results belong in `examples/results/` as JSON written by the
  script that produced them, so a number in the docs can be traced to a run.
