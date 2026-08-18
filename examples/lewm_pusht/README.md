# A pretrained agent cannot tell our frames from the official ones

Pixel parity is a claim about bytes. This is the claim about *consequences*: take the official
[LeWorldModel](https://huggingface.co/quentinll/lewm-pusht) PushT checkpoint — a ViT world model trained
end to end on frames from the official pygame/pymunk renderer — run **its own** evaluation unchanged, and
feed it frames from `stable_envs.make("PushT").render(states)` instead. If the success rate does not move,
the frames are the official frames as far as anything downstream is concerned.

It does not move.

| renderer | seeds | successes | success rate | 95% CI | paired vs official |
|---|---|---|---|---|---|
| **official** (`swm/PushT-v1`, pygame) | 42–47 | 264/300 | **88.0 %** | [83.8, 91.2] | — |
| **stable-envs**, every pixel | 42–47 | 263/300 | **87.7 %** | [83.5, 90.9] | 12 / 11 discordant, **p = 1.000** |
| blank frames (control) | 42 | 2/50 | 4.0 % | [1.1, 13.5] | 46 / 0 discordant, **p < 0.001** |

The two arms are *paired*: for each seed the episode indices, start steps, goal offset and CEM seed are
identical, so episode *i* is the same task in both. Across 300 paired episodes, 12 were won only by the
official renderer and 11 only by ours — a difference of −0.3 points, McNemar exact p = 1.000. Seed-to-seed
spread of the official arm alone is 80–96 %, so the effect being tested for is an order of magnitude
smaller than the noise it would have to beat.

The blank-frame row is why that null means something. Success here is computed from the pymunk state and
never reads a pixel, so "the score did not change when I changed the renderer" is only evidence if the
score *can* change. Replacing every frame with a white image costs 84 points and flips 46 of 50 episodes
to failure. The evaluation is acutely pixel-dependent; it simply cannot tell these two renderers apart.

## Which pixels were swapped

Three surfaces feed the model an image, and swapping only some of them would leave it comparing our
frames against theirs:

1. `PushT._render_frame` — the observation at every step, reached via `AddPixelsWrapper -> env.render()`.
2. `World._extract_init_goal`'s `pixels` — the **stored corpus frame** at the start step, which becomes
   the planner's first observation before the env is ever stepped.
3. `World._extract_init_goal`'s `goal` — the **stored corpus frame** at `start + 25`, the target the world
   model's cost is measured against, re-injected on every step.

(2) and (3) are re-rendered from the corpus's own `state` column, which is the point of the repo: the
stored images are redundant with the numbers stored beside them. Splitting the swap by surface turns a
null result into a diagnosis:

| what was swapped | render calls | seeds | success rate | paired vs official |
|---|---|---|---|---|
| live observation only (surface 1) | 1487 / run | 42–47 | **88.0 %** (264/300) | 2 / 2 discordant, Δ = **+0.0** pts, p = 1.000 |
| corpus frames only (surfaces 2–3) | 2 / run | 42–47 | 86.3 % (259/300) | 17 / 12 discordant, Δ = −1.7 pts, p = 0.458 |
| everything | 1516 / run | 42–47 | 87.7 % (263/300) | 12 / 11 discordant, Δ = −0.3 pts, p = 1.000 |

Replacing **every live frame the agent ever sees** — ~1500 renders per run, driving a 300-sample × 30-iteration
CEM at every replan — leaves the success count exactly where it was, with four episode outcomes moving in
each direction out of 300. What little movement there is comes from surface 2–3, and the next section
explains why that is a statement about the corpus rather than about either renderer.

## Pixel parity at the resolution the checkpoint consumes

48 random corpus states, rendered three ways at 224 px (`parity_probe.py`):

| comparison | all pixels exact | interior pixels exact | mean abs |
|---|---|---|---|
| ours (direct 224 raster) vs live official env | 95.6 % | **100.00 %** | 1.18 |
| ours (512 raster + `cv2.resize`) vs live official env | 97.3 % | **100.00 %** | 0.58 |
| stored corpus frame vs live official env | 84.6 % | 91.9 % | 0.69 |

"Interior" is a pixel whose 3×3 neighbourhood in the reference is one flat colour; the remainder is edge
anti-aliasing, where two independent rasterisers are entitled to disagree and neither is ground truth.
Every interior pixel matches, under both resampling policies — so the direct 224 raster is not merely
close to the reference's own 512-then-`cv2.resize` path, it is exact everywhere the question is
well posed.

The third row is the interesting one: **the corpus's own stored frames agree with the official renderer
less well than ours do.** Two independent causes, both diagnosed rather than assumed.

**`_set_state` does not put the env where you asked.** It ends with a physics substep, so re-driving the
env to a stored state lands up to **2.32 px** away from the pose that was recorded. That is a
state-restoration error, not a rendering one, which is why the probe measures and reports it separately —
folded into an aggregate it would masquerade as a renderer discrepancy. (This is the mild cousin of the
`gym_pusht` bug documented in `stable_envs/envs/pusht.py`, where setting position before angle displaces
the block by up to 70 px. `swm/PushT-v1` already sets angle first, so only the substep remains.)

**The corpus is lossily compressed.** Its frames carry **1582 distinct colours each**, against 146 for the
live pygame+`cv2` path and **7** for a flat-shaded raster — background, walls, goal, agent outer, agent
inner, block fill, block outline, the entire palette. Interiors survive (99.99 % exact) but every edge
carries a wide ringing band. `resize_probe.py` rules out all nine plausible single-stage filters (cv2
linear/area/nearest/cubic, PIL bilinear/bicubic/nearest/box/lanczos) and `corpus_origin_probe.py` rules
out a render-small-then-upscale round trip at eight intermediate resolutions: none reproduces the corpus,
and every one comes out *sharper* than it, which no purely resampling pipeline can explain.

So the 46 GB of stored pixels are not merely redundant with 7 floats per step — they are a **degraded**
copy of what the renderer produces, and regenerating them from the states is strictly closer to the
reference than reading them off disk. Surface 2–3 therefore hands the model a *cleaner* goal image than
the one the official run uses, which is the honest explanation for the −1.7-point drift in that arm: it is
a change of image statistics inherited from the corpus, not a rendering error, and it is not significant
at n = 300 anyway.

## What was run

The eval is `le-wm`'s own `eval.py` with its own config, copied here unmodified except for a `renderer`
key (and `video=None`, whose imageio path is broken in this environment and which the success rate does
not depend on). Dataset, episode-sampling RNG, the `StandardScaler` fitted over all 2.34 M corpus rows,
the ImageNet-normalised transform, the CEM solver (300 samples, 30 iterations, top-30) and its seed, the
plan config (horizon 5, receding 5, action block 5), the 50-step budget and the success criterion
(`PushT.eval_state`: within 20 px and π/9) are all theirs.

    LeWM PushT, dataset-driven: 50 episodes/seed, goal 25 steps ahead, 50-step budget, CEM 300x30, 224 px

The official arm reproduces the archived number for this protocol **exactly** — 96.0 % at seed 42, failing
the same two episodes (14 and 15) as the historical run in
`~/.stable_worldmodel/checkpoints/pusht/pusht_results.txt`.

## Reproducing

Everything runs under SLURM; nothing here touches `stable_envs/` or `tests/`.

```bash
cd examples/lewm_pusht

sbatch run_probe.sbatch                                          # pixel parity + colour census
sbatch --export=ALL,RENDERER=official,SEED=42    run_eval.sbatch
sbatch --export=ALL,RENDERER=stable_envs,SEED=42 run_eval.sbatch
python summarize.py                                              # the tables above
```

`RENDERER` is one of `official`, `stable_envs`, `stable_envs_resize512` (512 raster then `cv2.resize`,
matching the reference's own downsampling path), `stable_envs_live`, `stable_envs_dataset`, `blank`.

Requires the official checkpoint at `~/.stable_worldmodel/checkpoints/pusht/lewm/` (`quentinll/lewm-pusht`)
and the corpus at `~/.stable_worldmodel/datasets/pusht_expert_train.h5` (`quentinll/lewm-pusht`).

| file | what it is |
|---|---|
| `eval_lewm_pusht.py` | `le-wm/eval.py`, plus a `renderer` key |
| `config/eval/` | their config tree, copied |
| `swap_renderer.py` | the monkeypatches: three pixel sources, nothing else |
| `parity_probe.py` | pixel agreement and colour census at 224 px |
| `resize_probe.py`, `corpus_origin_probe.py` | why the corpus is blurrier than the renderer |
| `summarize.py` | paired McNemar table over the per-run JSONs |
| `results/` | one JSON per run: per-episode outcomes, episode indices, start steps |
| `results/parity_224.png` | stored \| reference \| ours-direct \| ours-resize512, four states |
