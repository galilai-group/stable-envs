# An opt-in fast path for Meta-World

Meta-World is the largest population of environments in this repository still below the
stored-frame bar: **50 tasks at a median 0.456x** of what eight worker processes read out of their
own Lance table. This directory measures whether a cheaper render plus a small learned corrector can
clear that bar, what it costs in fidelity, and whether the answer is one model or fifty.

Nothing here touches `stable_envs/`. Everything is measured on H200s through SLURM, batch 1024, and
the official frame is always `mujoco.Renderer` at 256 px through the model camera `corner`.

> **This is NOT a replacement for the shipped render, and it must never be presented as one.**
> Meta-World is a PARITY family: it declares that its frame IS MuJoCo's frame, with an interior-exact
> floor of 0.978 and an all-pixel-exact floor of 0.80. **Reducing the internal resolution voids that
> claim at the first step** -- all-pixel exactness falls 0.879 -> 0.524 at 160 px, before any network
> is involved -- and a learned corrector cannot restore it (best measured 0.83, median 0.68). So the
> deliverable is an **opt-in fast path alongside the exact one**, carrying its own declared numbers,
> for callers generating training data rather than evaluating a policy on official pixels. The
> decision belongs to the caller, and section 8 gives them the numbers to make it on.

---

## 1. The result

`unet_96t128_w48_d2_u2` (441 k parameters, 16.4 us/frame in fp16 from a CUDA graph) on a 96 px
render, upsampled to 256 px. **All 17 tasks measured clear their stored-frame bar**, the tightest by
1.10x, and the mean error against MuJoCo is **2.3-3.2x smaller than the cheap render it corrects and
smaller than JPEG q95 of the official frame**.

`MetaWorldDoorLock`, held-out episodes (512 states from 128 episodes the training loop never saw,
section 9), scored in blocks of 64:

| | f/s | x bar | mean \|diff\| | within-8 | SSIM | interior-exact | all-exact | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped 256 px (parity) | 6,850 | 0.42 | 0.995 | 0.9721 | 0.9806 | **1.0000** | **0.881** | 73 | 255 |
| bilinear 96 -> 256 | 24,509 | 1.51 | 4.137 | 0.8678 | 0.9118 | 0.9889 | 0.456 | 88 | 255 |
| **corrector** | **18,186** | **1.12** | **1.589** | **0.9542** | **0.9773** | 0.9419 | 0.688 | **73** | 255 |
| corrector + shrink 1.0 | 18,186 | 1.12 | 1.861 | 0.9534 | 0.9746 | **0.9797** | 0.451 | 73 | 255 |
| *JPEG q95 of official -- the storage baseline* | *16,202* | *1.00* | *2.120* | *0.9609* | *0.9935* | *0.8837* | *0.396* | *38* | *237* |

Read that table twice, because two of its rows disagree.

* **In mean \|diff\| the corrector is closer to MuJoCo than the JPEG a stored dataset would hand the
  same training loop** (1.589 against 2.120), and it is closer on interior-exactness (0.942 against
  0.884). It is 1.6x further from MuJoCo than the shipped parity render, at 2.7x the speed.
* **In SSIM and in the error tail JPEG is better** (0.9935 against 0.9773; p99.9 of 38 against 73).
  JPEG's error is a uniform faint ring around every edge; the corrector's is concentrated on the
  arm's silhouette, where the shipped render also disagrees.
* **All-pixel exactness is gone and no configuration brings it back.** 0.881 -> 0.688, and the
  shrinkage that protects interiors costs it further (0.451). This is the parity claim, and it is the
  thing being traded.

The recommendation clears the bar on every task measured, **with the lightspec kernel switch off**;
turning it on adds 5-7 %:

| tightest measured tasks | bar | shipped | fast path | x shipped | x bar |
|---|---:|---:|---:|---:|---:|
| Soccer | 15,620 | 6,218 | 17,147 | 2.76 | **1.10** |
| PlateSlide | 16,077 | 6,690 | 17,788 | 2.66 | **1.11** |
| DoorLock | 16,202 | 6,850 | 18,186 | 2.66 | **1.12** |
| PlateSlideBack | 15,922 | 6,589 | 17,899 | 2.72 | **1.12** |
| SweepInto (easiest measured) | 16,301 | 8,028 | 20,310 | 2.53 | **1.25** |

17 of 50 tasks were timed with a corrector, chosen as the 10 slowest in the family plus 7 spread
across it; the ones not timed are all faster at 96 px than the ones that were.

**One corrector serves several tasks.** Trained on six tasks at once it costs 6 % of its mean error
against six specialists, and -- unlike a single-task corrector, which is a wash or harmful on a task
it never saw -- it transfers to a seventh, unseen task, recovering 45 % of the cheap render's error
there (section 6).

---

## 2. What the lever is, and why it is a different lever here

Meta-World's frame is a **composite**, not one kernel: MuJoCo's own raster of the episode's static
scene (the plate, ~64 % of the pixels, exact by construction, cached per episode) with
`mujoco_warp`'s ray-cast movers composited over it through the segmentation buffer. Lowering the
internal resolution lowers both halves at once, because the env already takes the resolution as an
argument and already bakes a plate per (episode, appearance, **resolution**). That is why the whole
study runs through `env.render(states, size=r)` and edits no env module.

The ray-cast is where the time is, and the lever reaches it:

| | 256 px | 160 px | 128 px | 112 px | 96 px |
|---|---:|---:|---:|---:|---:|
| captured graph's share of the frame (median, 50 tasks) | 92 % | 85 % | 80 % | 76 % | 72 % |

so at 96 px roughly a quarter of the frame is the torch composite and the per-world uploads, which is
what makes the resolution curve sublinear. The per-episode MuJoCo bake is **6.5-7.1 ms and does not
move with resolution** (MuJoCo's raster is nearly resolution-independent at these sizes); it is paid
once per episode and is out of the timed region here for the same reason it is out of it in the
shipped measurement -- a 500-step rollout pays it once.

---

## 3. The resolution frontier, all 50 tasks

`probe_frontier.py`, 64 states per task from the env's own `sample_states`, batch 1024, the upsample
to 256 px inside the timed region. Median over the 50 tasks, [min, max] across them:

| | r256 (shipped) | r160 | r128 | r112 | r96 |
|---|---:|---:|---:|---:|---:|
| f/s | 7,237 [3,162, 8,026] | 13,432 [6,339, 14,537] | 18,237 [9,018, 19,829] | 21,391 [10,910, 23,267] | 25,655 [13,515, 28,249] |
| x stored bar | 0.456 [0.201, 0.493] | 0.840 [0.403, 0.892] | 1.142 [0.573, 1.216] | 1.337 [0.694, 1.427] | 1.605 [0.859, 1.733] |
| mean \|diff\| | 1.022 [0.976, 1.386] | 2.633 [2.435, 3.264] | 3.290 [3.070, 3.948] | 3.627 [3.390, 4.347] | 4.153 [3.877, 4.883] |
| SSIM | 0.980 | 0.956 | 0.939 | 0.929 | 0.913 |
| interior-exact | 1.000 [0.989, 1.000] | 0.999 [0.987, 1.000] | 0.997 [0.982, 0.998] | 0.994 [0.978, 0.995] | 0.991 [0.971, 0.992] |
| all-pixel exact | 0.879 [0.819, 0.884] | 0.524 [0.453, 0.542] | 0.490 | 0.479 | 0.464 |
| clears the bar | 0/50 | 0/50 | 47/50 | 48/50 | 48/50 |

Four things this says, and two of them were not expected.

* **The median task needs 2.2x and 128 px already gives it.** A caller who wants speed and will
  accept mean 3.3/255 needs no network at all.
* **Interior-exactness barely moves** -- 1.000 -> 0.991 at 96 px -- because a downsample followed by
  an upsample reproduces a flat region exactly, and Meta-World's interiors are flat. The declared
  floor of 0.978 survives resolution alone on 48 of 50 tasks.
* **All-pixel exactness collapses at the first step off 256** (0.879 -> 0.524 at 160 px) and is flat
  thereafter. The parity claim dies at 224 px, not at 96 px; there is no gentle part of this curve.
* **The two window tasks do not clear at any resolution measured** (0.86-0.87x at 96 px, 1.11-1.12x
  at 80 px). They need section 4.

`STABLE_ENVS_LIGHTSPEC=1`, which deletes the three dead any-hit BVH traversals the megakernel
compiles for lights that do not cast, is worth **1.064x median [1.045, 1.078] over 38 tasks** and is
bit-identical on every task except the two window ones (1/255 on 2 of 4,194,304 pixels, a fact the
env module already records). A path that has given up bit-identity should take it; nothing in this
report depends on it.

---

## 4. The two window tasks: the second graph replay is the whole gap

`window-open-v3` and `window-close-v3` render at 3,162 and 3,171 f/s against a ~15,600 bar -- 4.9x, the widest
gap in the family -- and the decomposition says why: at 256 px the captured graph alone runs at
7,348 f/s while the env delivers 3,162, i.e. **the frame pays for two graph replays**. `_blend`
renders the scene a second time with the translucent geoms' BVH slots re-pointed at a static geom,
because `window_glass` (`rgba="0 .3 .4 .1"`) must be blended against what is behind it, and what is
behind it is another mover that the plate does not contain.

Dropping that second pass -- `_blend` returns its input, which is what `mujoco_warp`'s own shader
does anyway, painting the pane opaque -- is measured in `probe_window.py`, patched into the class
object inside the process and removed afterwards:

| `MetaWorldWindowOpen` | f/s | x bar | mean \|diff\| | SSIM | interior-exact |
|---|---:|---:|---:|---:|---:|
| shipped 256 px | 3,163 | 0.20 | 1.356 | 0.9770 | 0.9888 |
| 256 px, no second pass | 6,833 | 0.44 | 2.572 | 0.9575 | 0.9888 |
| 96 px | 13,643 | 0.87 | 4.534 | 0.9093 | 0.9714 |
| **112 px, no second pass** | **20,892** | **1.34** | 5.168 | 0.9060 | 0.9782 |
| 96 px, no second pass | 25,197 | 1.61 | 5.699 | 0.8904 | 0.9714 |

and a corrector trained on that cheap frame recovers most of it: **5.122 -> 1.797 mean**, within-8
0.8623 -> 0.9471, SSIM 0.9055 -> 0.9751, at **16,692 f/s (1.07x bar, 2.4x shipped)**. The pane is a
local, always-in-the-same-place defect, which is the kind a corrector is good at; the arithmetic that
the parity path spends a whole extra ray-cast on, the network reproduces for 16 us.

This is the one place in the study where the corrector is doing something other than restoring
resolution, and it is the only route by which these two tasks reach the bar at all.

---

## 5. Correctors, per task

`train.py`, the recipe from `scratch/altrender/rendernet/train.py` unchanged: architectures from its
`nets.py`, zero-initialised head (so the model **is** the bilinear baseline at step 0 and every
number is a departure from a measured point), L1 on the un-rounded output, bf16 autocast with
gradient clipping, OneCycle, 400 epochs over 4,096 pairs, ~4 minutes per task on one GPU. Held-out
episodes, median over the tasks trained, [min, max]:

| corrector | render | us/frame | tasks | bilinear | mean \|diff\| | within-8 | all-exact | interior-exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| `unet_112t128_w64_d3_u2` | 112 | 21.5 | 7 | 3.73 | **1.23** [0.95, 1.57] | 0.9637 | 0.792 | 0.9630 |
| `unet_96t128_w64_d3_u2` | 96 | 21.5 | 7 | 4.26 | 1.25 [1.02, 1.58] | 0.9646 | 0.760 | 0.9495 |
| `unet_112t128_w48_d2_u2` | 112 | 16.6 | 8 | 3.79 | 1.56 [1.24, 1.96] | 0.9545 | 0.718 | 0.9645 |
| **`unet_96t128_w48_d2_u2`** | **96** | **16.4** | **7** | 4.26 | **1.59** [1.33, 2.09] | 0.9542 | 0.684 | 0.9421 |
| `unet_112t128_w32_d2_u2` | 112 | 12.6 | 7 | 3.73 | 1.83 [1.60, 2.37] | 0.9491 | 0.627 | 0.9734 |
| `espcn_112t128_w48_d4_u2` | 112 | 10.6 | 7 | 3.73 | 2.56 [2.31, 3.05] | 0.9288 | 0.484 | 0.9056 |

* **The UNet dominates the sub-pixel ESPCN at every budget.** At 10.6 us the ESPCN recovers 31 % of
  the bilinear error; at 12.6 us the UNet recovers 51 %. RoboCasa's study found the two comparable;
  on this content the extra receptive field of one downsample is worth more than any width.
* **Resolution and capacity trade almost exactly.** 96 px with `w48_d2` (1.59) and 112 px with
  `w32_d2` (1.83) cost the same in throughput terms and the first is better, which is why the
  recommendation gives away resolution rather than parameters.
* **`w64_d3` is the fidelity winner and it does not fit.** 1.23 median, and 21.5 us puts it below the
  bar on the family's slowest ordinary tasks by too little to trust (PlateSlide 1.02x, Soccer 1.05x
  without lightspec, against `w48_d2`'s 1.11x and 1.10x). It is the row to quote for what a corrector
  *can* do, not the row to ship.
* On `MetaWorldAssembly`, `unet_112t128_w64_d3_u2` scores **0.981 mean against the shipped render's
  1.034**, with better SSIM (0.9849 vs 0.9786) and better within-8 -- i.e. a 112 px render plus a
  920 k-parameter network is *closer to MuJoCo on three of five metrics than the parity composite at
  256 px*, at 2.4x the speed. It still loses on the two exactness metrics, which is the whole point.

### Interior-exactness is a knob, and it has to be turned deliberately

The cheap frame's interior is 0.989-0.995 bit-exact before any network touches it, and a dense
residual destroys that: the corrector alone drops it to 0.942 while improving everything else. The
residual soft-shrink from the RoboCasa study -- zeroing residuals below a threshold, so "the
corrector has nothing to say here" produces a bit-identical pixel -- buys it back:

| `MetaWorldDoorLock`, held out | mean | within-8 | SSIM | interior-exact | all-exact |
|---|---:|---:|---:|---:|---:|
| corrector, no shrink | **1.589** | 0.9542 | **0.9773** | 0.9419 | **0.688** |
| shrink 0.25 | 1.611 | 0.9541 | 0.9771 | 0.9620 | 0.662 |
| shrink 0.5 | 1.771 | 0.9539 | 0.9762 | 0.9708 | 0.506 |
| shrink 1.0 | 1.861 | 0.9534 | 0.9746 | **0.9797** | 0.451 |

RoboCasa found shrinkage worthless at 300 epochs because the trained residual becomes sparse on its
own. **Here it is not worthless and the reason is the baseline**: RoboCasa's interior-exactness was
0.48 and had nothing to protect, Meta-World's is 0.99 and has everything to protect. Shrink 1.0 puts
interior-exactness back above the family's declared 0.978 floor on 3 of the 4 tasks measured
(0.9797, 0.9812, 0.9836; ButtonPress 0.9700) for +0.27 of mean error. It costs a further 0.24 of
all-pixel exactness, which was already lost.

---

## 6. One corrector for six tasks

The question that decides whether this is a technique or a chore. `multi6` is one checkpoint trained
on the concatenated training splits of six tasks, 200 epochs over 24,576 pairs, evaluated per task on
each task's own held-out episodes, against the six specialists of the same architecture:

| `unet_96t128_w48_d2_u2` @ 96 px | specialist | one model | cost |
|---|---:|---:|---:|
| Assembly | 1.331 | 1.335 | +0.3 % |
| ButtonPress | 1.821 | 2.034 | +11.7 % |
| DoorLock | 1.587 | 1.633 | +2.9 % |
| PlateSlide | 1.735 | 1.802 | +3.8 % |
| Soccer | 2.087 | 2.125 | +1.8 % |
| SweepInto | 1.341 | 1.546 | +15.3 % |
| | | **mean +6.0 %, worst +15.3 %** | |

and with more capacity the gap widens rather than closes (`w64_d3` @ 112 px: **+9.9 % mean, +15.9 %
worst**), which is what "the specialists were using the capacity" looks like.

**One model covers six tasks for 6 % of its mean error.** It is still 2.2-2.7x better than the cheap
render on every one of them, it clears the same bars (the cost is in the weights, not the shape), and
it is the configuration to prefer for a caller who wants one artefact rather than fifty.

### And training on six tasks buys transfer to a SEVENTH that one task does not

`MetaWorldHammer`, which no checkpoint in this ladder was trained on, all four rows at 96 px with the
same `unet_96t128_w48_d2_u2` architecture, on Hammer's own held-out episodes:

| what produced the frame | mean \|diff\| | within-8 | SSIM | interior-exact | all-exact |
|---|---:|---:|---:|---:|---:|
| bilinear 96 -> 256 (do nothing) | 4.260 | 0.8646 | 0.9125 | 0.9903 | 0.445 |
| a corrector trained on ONE other task (`DoorLock`) | 3.321 | 0.8904 | 0.9351 | 0.9041 | 0.504 |
| **the SIX-task corrector, zero-shot** | **2.350** | **0.9290** | **0.9619** | 0.9300 | 0.579 |
| Hammer's own specialist | 1.59 | 0.9573 | — | 0.9546 | 0.684 |
| — the shipped parity render | 1.186 | 0.9656 | 0.9791 | 1.0000 | 0.857 |

**Task diversity is what makes a corrector transfer.** One task's corrector recovers 22 % of the
bilinear error on an unseen task and wrecks interior-exactness (0.990 -> 0.904); six tasks' corrector
recovers **45 %** and holds interior-exactness higher (0.930), while improving within-8 and SSIM --
the two metrics that caught RoboCasa's hallucination. The larger `w64_d3` at 112 px does better
still: 3.734 -> **1.855** zero-shot, within-8 0.8833 -> 0.9469, SSIM 0.9280 -> 0.9724 -- 50 % of the
error recovered on a task it never saw, against the 69 % that Hammer's own 112 px `w64_d3`
specialist (1.14) achieves.

This is the lead `rendernet/README.md` ranked third and never tested ("one corrector TRAINED across
the cells of one kitchen"), and on this family it works. It does not make a corrector universal --
the zero-shot rows are still worse than the specialists on every metric, and interior-exactness is
still the casualty -- but it changes the shape of the offer for a 50-task family: **train one model
on a spread of tasks, not fifty models.**

---

## 7. Where it fails, and the three controls

### Foreign task, zero-shot

Same checkpoint, a task it was never trained on. `bilinear` is "do nothing but resize", so a corrector
worse than bilinear is actively harmful:

| corrector | applied to | mean | within-8 | SSIM | interior-exact |
|---|---|---:|---:|---:|---:|
| Assembly `w64_d3` @112 | `SweepInto` -- bilinear | 3.722 | 0.9013 | 0.9210 | 0.9941 |
| | `SweepInto` -- **corrector** | **3.602** | **0.9003** | **0.9200** | **0.9188** |
| DoorLock `w48_d2` @96 | `Soccer` -- bilinear | 4.834 | 0.8608 | 0.9007 | 0.9916 |
| | `Soccer` -- corrector | 2.539 | 0.9382 | 0.9615 | 0.9354 |
| | `Hammer` -- bilinear | 4.260 | 0.8646 | 0.9125 | 0.9903 |
| | `Hammer` -- corrector | 3.321 | 0.8904 | 0.9351 | 0.9041 |
| WindowOpen `w48_d2` @112nb | `WindowClose` -- bilinear | 4.742 | 0.8646 | 0.9099 | 0.9809 |
| | `WindowClose` -- corrector | 2.137 | 0.9343 | 0.9642 | 0.9197 |

Two different failures, and both matter:

* **Assembly -> SweepInto is the RoboCasa signature exactly** (the bold row above). The mean barely
  moves (3.722 -> 3.602) while within-8, SSIM and interior-exactness all get worse.
  `sbs_pt_MetaWorldAssembly_unet_112t128_w64_d3_u2_r112__MetaWorldSweepInto_heldout.png` shows what
  it is doing: **it paints a diagonal wood-grain streak across a tabletop that carries no such
  texture in the official frame**, visible in the difference map and invisible in the mean. This is
  the picture to keep, and it is the reason this report scores five metrics and two tail statistics.
* **The other three transfer usefully on mean, within-8 and SSIM and destroy interior-exactness**
  (0.99 -> 0.90-0.94). A borrowed corrector is not free even when its mean improves, and on a family
  whose declaration is an exactness floor, that is the metric that decides.

**A corrector belongs to the tasks it was trained on** -- unless it was trained on several, which is
the one thing that changes the answer (section 6). A task with no corrector of its own must fall back
to the exact render or to a MULTI-TASK corrector, never to a single neighbour's.

### Appearance variation: train on it, or do not use it

Meta-World declares per-task camera azimuth (+/-25 deg), elevation (+/-20 deg), distance scale
(0.6-1.6), light diffuse and four colour knobs. `MetaWorldDoorLock`, `unet_112t128_w48_d2_u2` at
112 px, the 2x2 (the official frames are re-rendered with the same style written into the reference
model, and the default row round-trips to **exactly 0.0**):

| trained at | evaluated at the default | evaluated under the declared variation |
|---|---:|---:|
| the default appearance | **1.485** | **4.015** |
| the declared variation | 1.674 | 3.111 |
| — bilinear (do nothing) | 3.608 | 3.754 |
| — shipped 256 px | 0.995 | 1.317 |

* **A corrector trained at the default appearance is worse than not correcting at all under the env's
  own declared variation** -- 4.015 against bilinear's 3.754, with within-8 0.8765 against 0.8844 and
  interior-exactness 0.8415 against 0.9914. `MetaWorldAssembly` reproduces it exactly: 4.351 against
  bilinear's 3.703, within-8 0.8635 against 0.8870, interior-exactness 0.8168 against 0.9930.
  Shipping a default-trained corrector to a caller who uses the variation knobs would silently make
  their frames worse.
* **Training on the variation fixes the sign and not the size.** 3.111 against bilinear 3.754 is a
  17 % reduction, where the default-appearance corrector achieves 59 %. Meta-World's variation space
  is far wider than RoboCasa's (+/-25 deg of azimuth against +/-8 deg of pan), and a 441 k-parameter
  network asked to restore resolution for every camera pose in that space does much less well at each
  one. It also costs 0.19 at the default (1.674 against 1.485).

**So the fast path is declared at the default appearance.** A caller using the variation knobs either
retrains on their own slice of the space or uses the exact render; both numbers are above.

### Blank frame: nothing is drawn from the weights

A constant input, to see what the network emits with no image to read:

| level | residual mean | max | pixels moved > 8/255 |
|---|---:|---:|---:|
| 0 (black) | 1.40 - 2.46 / 255 | 8 - 15 | 0.0 - 1.4 % |
| 128 (grey) | **0.12 - 0.40 / 255** | 12 - 22 | **0.0 - 0.3 %** |
| 255 (white) | 0.71 - 1.27 / 255 | 19 - 68 | 0.1 - 2.0 % |

RoboCasa's corrector moved **69 % of the frame by more than 8/255** on this control, because it had
learned to synthesise a texture its input did not contain. **These correctors emit almost nothing**,
and that is the mechanism difference in one measurement: here the missing information is *resolution*
(present but blurred) rather than *texture* (absent), so there is far less to hallucinate -- which is
also why the out-of-distribution damage in the table above is smaller than RoboCasa's, and why it is
still real.

### Dynamic against static pixels

Split the frame by how much the OFFICIAL pixel varies across the evaluation states; the top quartile
is where error can only be earned by reading the input:

| `MetaWorldDoorLock` | mean, all | dynamic 25 % | static 75 % |
|---|---:|---:|---:|
| shipped 256 px | 0.995 | 3.62 | 0.12 |
| bilinear 96 -> 256 | 4.137 | 7.58 | 2.99 |
| corrector | 1.589 | 5.63 | 0.24 |

**Static pixels improve 12x, moving pixels 1.35x.** Most of the win is on the parts of the scene that
do not move -- and on the pixels that carry the state, the fast path is 1.6x further from MuJoCo than
the parity render (5.63 against 3.62). A consumer who cares about the arm and the object specifically,
rather than about the frame's average, should read the dynamic column and not the first one.

---

## 8. Recommendation, and the numbers it would carry

**Ship an opt-in fast path, per task or per group of tasks, at 96 px, alongside the exact render.**

| | configuration |
|---|---|
| render | `env.render(states, size=96)`, bilinear to 256, `STABLE_ENVS_LIGHTSPEC=1` optional (+5-7 %) |
| corrector | `unet_96t128_w48_d2_u2`, 441 k parameters, fp16, `channels_last`, CUDA-graph replay, 16.4 us/frame at batch 1024 |
| the two window tasks | 112 px **and** the translucent second pass dropped, `unet_112t128_w48_d2_u2` |
| trained on | that task's own episodes at the DEFAULT appearance, 4,096 pairs, 400 epochs, ~4 min/task |
| one model or many | one checkpoint over a group of tasks costs +6 % of mean error and is the better artefact |

Measured, per task, held out (specialists; the one-model numbers are section 6):

f/s columns are lightspec ON for the fast path and the shipped parity render as it ships (lightspec
is off there by design, because it is not bit-identical on two tasks); that is exactly the comparison
a caller choosing between the two paths faces. Without lightspec the fast path is 4-6 % slower and
still clears every bar (section 1).

| task | shipped f/s | fast f/s | x shipped | x bar | mean now -> fast | within-8 | interior-exact | all-exact |
|---|---:|---:|---:|---:|---|---:|---:|---:|
| Assembly | 7,680 | 20,104 | 2.62 | 1.23 | 1.03 -> **1.33** | 0.9643 | 0.954 | 0.721 |
| SweepInto | 8,028 | 21,174 | 2.64 | 1.30 | 0.99 -> **1.34** | 0.9656 | 0.954 | 0.681 |
| Hammer | 7,167 | 19,866 | 2.77 | 1.25 | 1.19 -> **1.59** | 0.9573 | 0.955 | 0.684 |
| DoorLock | 6,850 | 18,998 | 2.77 | 1.17 | 1.00 -> **1.59** | 0.9542 | 0.942 | 0.689 |
| PlateSlide | 6,690 | 18,532 | 2.77 | 1.15 | 1.00 -> **1.74** | 0.9503 | 0.933 | 0.683 |
| ButtonPress | 7,071 | 19,542 | 2.76 | 1.23 | 1.15 -> **1.82** | 0.9487 | 0.916 | 0.659 |
| WindowOpen* | 3,162 | 19,174 | 6.06 | 1.23 | 1.35 -> **1.88** | 0.9465 | 0.906 | 0.617 |
| Soccer | 6,218 | 17,858 | 2.87 | 1.14 | 1.02 -> **2.09** | 0.9466 | 0.942 | 0.691 |

SSIM was measured on four of these (full five-metric tables in section 1 and section 7): shipped
0.9786-0.9806 against the corrector's 0.9730-0.9849.

\* the window row is the 96 px **no-second-pass** configuration throughout, against the parity
render's own 3,162 f/s -- which is why its speedup is 6.06x and not 2.8x, the second ray-cast being
the thing removed. The 112 px variant recommended above is 16,692 f/s at mean 1.797. (In
`fps_MetaWorld*_noblend.json` the `shipped_256` row is also a no-second-pass render, because the
patch is applied for the whole process; the parity number quoted here is from
`window_MetaWorldWindowOpen.json`, which measures both.)

The sentence such a path has to carry:

> **This is not the parity render.** These frames are `mujoco_warp`'s composite at 96 px followed by a
> 441 k-parameter convolutional corrector trained on this task (or on a group containing it),
> upsampled to 256 px. Against
> `mujoco.Renderer` they carry mean \|diff\| 1.3-2.1/255, within-8 0.947-0.966, SSIM ~0.977,
> interior-exact 0.91-0.95 and all-pixel-exact 0.62-0.72 -- against the exact path's 1.0-1.4, 0.95-0.97,
> 0.98, **0.989-1.000** and **0.84-0.88**. **The all-pixel exactness the family declares (>= 0.80) is gone
> and cannot be recovered**; interior-exactness can be held above the declared 0.978 floor by the
> residual shrink at a cost of +0.27 mean.
>
> The frame is still a pure deterministic function of `(state, style)` -- the corrector is a fixed
> checkpoint applied to a render that was already such a function -- so `render` is still
> `f(get_state())`, still batch-invariant, still reproducible by anyone holding the weights. What
> changed is the decoder, not the contract.
>
> It is **specific to the tasks it was trained on and to the DEFAULT appearance**. A corrector trained
> on ONE task and applied to another is a wash on the mean and worse on within-8, SSIM and
> interior-exactness, and can paint texture that is not there; one trained on six tasks transfers to a
> seventh at about two thirds of a specialist's benefit. Under the env's own declared camera
> variation a default-trained corrector is **worse than not correcting at all**.

### What is NOT recommended

* **Not for evaluating a policy on official Meta-World pixels.** Use the exact path. That is the
  whole reason this is opt-in.
* **Not the 128 px plain render as a fast path with a corrector.** At 128 px the render alone is
  1.14x the bar, which leaves 7 us for a network, and nothing useful fits in 7 us.
* **Not a SINGLE-task corrector borrowed by another task** (section 7): on `SweepInto` the
  Assembly corrector is a wash on the mean and worse on within-8, SSIM and interior-exactness.
  A MULTI-task corrector used zero-shot is a different and much better proposition (section 6),
  and is the fallback to prefer for a task with no corrector of its own.
* **Not `w64_d3`**, however good it looks in section 5: 21.5 us puts the family's slowest ordinary
  tasks below their bar.

### Residual leads, and what was NOT measured

1. **The frozen-encoder check was not run.** `rendernet/README.md` section 5 is the one place its
   pixel metrics oversold: correctors that were 2.5x better in mean |diff| sat between the shipped
   render and the bilinear baseline in DINOv2/ResNet-18/DINOv3 cosine distance. `probe_repr.py`
   exists and would answer the same question here in about an hour. **Nothing in this report licenses
   "a model would not notice"**, and the argument that it is less likely to here -- the corrector
   synthesises almost nothing, section 7's blank-frame control -- is an argument, not a measurement.
2. **One corrector over more than six tasks.** Six tasks already transfer to a seventh; 12, 25 and 50
   were not tried, and the trend is the thing that decides whether the artefact is one file or ten.
3. **A hybrid: bake the plate at 256 px and ray-cast only the movers at 96.** The static ~64 % of the
   frame would stay bit-exact, which is the metric the resolution lever destroys. The decomposition
   in section 2 says the ray-cast is 72-92 % of the frame, so the arithmetic is promising, but the
   composite would then run at 256 px and this was **not measured** -- it needs a change inside
   `render()` that this directory deliberately does not make.
4. **Style-augmented multi-task training** (both hard factors at once) was not tried; the two were
   measured separately.
5. **r80 with the translucent pass kept** clears the window tasks' bar at 1.11-1.12x with no network
   at all (mean 5.25). Not pursued because the no-second-pass route at 112 px is both faster and much
   more accurate once a corrector is present.

---

## 9. Method

### The split, which is why any of these numbers mean anything

Meta-World writes the episode's placement into `model.body_pos`/`model.site_pos` at reset, so **two
states of one episode share their entire static scene**. Splitting states would put the same scene on
both sides and would measure a lookup table. The split is therefore on the EPISODE SEED, and there
are three disjoint ranges:

| split | episodes | states | seeds |
|---|---:|---:|---|
| train | 1,024 | 4,096 | `_seeded_reset` 20,000+ |
| held out | 128 | 512 | 30,000+ |
| `decl` -- the draw a declaration is measured on | the shipped pool | 256 (4 draws x 64) | 10,000-10,063, via the env's own `sample_states` |

`common.episode_states` is `metaworld._state_pool`'s own generative process with the seed base made
an argument -- same `_seeded_reset` (the only route that reproduces a draw on 50/50 tasks), same 0-25
random actions per state, same random yaw on every free joint.

**`decl` is disjoint from training here**, which is stronger than the RoboCasa study, where `decl`
overlapped the training pool by construction. It shows: held-out 1.589 against `decl` 1.607 on the
same checkpoint (worst of 4 draws: 1.639). **There is no state-memorisation effect to subtract.**

### Measurement rules

* Every metric is computed in **blocks of 64 frames and averaged**, because
  `testing.conformance.ssim` runs a 3-D uniform filter that smooths along the frame axis, so SSIM on
  512 frames is not the same quantity as SSIM on the 64 a declaration is measured at.
* The corrector is measured in **fp16** because that is what it runs in; fp32 costs 21.6 us instead of
  16.4 and would change which candidates clear their bar.
* Throughput is batch 1024, median of 15-25 repetitions, with the upsample and the corrector inside
  the timed region and the plate cache warm. Interquartile spread is under 1 % on every row reported.
  Two independent harnesses agree: `probe_fps.py` (an untrained network of the same architecture,
  every candidate in one process) and `eval.py` (the actual checkpoint, alongside its fidelity) give
  DoorLock 18,186 / 18,151, Soccer 17,147 / 17,024 and Hammer 19,252 / 19,217 f/s.
* The shipped baseline is rendered at the same appearance as the official frame it is scored against
  (styled evaluations thread the style through `env.render`), which is the harness bug
  `rendernet/README.md` section 8 records.

### Two things found on the way

* **A shared warp kernel cache races.** 18 concurrent jobs compiling the same megakernel into
  `~/.cache/warp` produced `WarpCodegenError` on 17 of 50 tasks; a per-job `WARP_CACHE_PATH` fixes it
  and all 50 then complete. Worth knowing for any fan-out over this family.
* **One node reported the corrector 31 % slower** (21.55 us against 16.42-16.60 on the other 16
  runs, same architecture, same dtype, all H200). It moved `MetaWorldPlateSlide` from 1.15x its bar
  to 1.05x. Candidates within a few percent of a bar must be re-measured on another node before they
  are believed.

### Files

| file | what it is |
|---|---|
| `common.py` | states, official frames (with and without style), metrics, the runtime ZOO extension |
| `probe_frontier.py` | the resolution frontier, the ray-cast/composite split, the plate bake cost |
| `probe_window.py` | the two window tasks and the translucent second pass |
| `gen_pairs.py` | `(cheap render, official frame)` pairs on disjoint EPISODE sets |
| `train.py` | the fit; `--envs` takes a list, so one run trains one corrector for several tasks |
| `eval.py` | five metrics, tails, dynamic/static, blank-frame and foreign-task controls, side-by-sides |
| `probe_fps.py` | every candidate timed back to back in one process, against that task's bar |
| `summarise.py` | every table in this README, from the JSON the others wrote |

Data under `/mnt/vast/home/randall.balestriero/data/mwfast/`, results, checkpoints and images under
`/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/mwfast/`.

Three pictures belong with any recommendation:

1. `sbs_pt_MetaWorldDoorLock_unet_96t128_w48_d2_u2_r96__MetaWorldDoorLock_heldout.png` -- official /
   shipped / bilinear / corrector / JPEG with amplified difference maps.
2. `sbs_pt_MetaWorldAssembly_unet_112t128_w64_d3_u2_r112__MetaWorldSweepInto_heldout.png` -- the same
   layout on a task the corrector never saw, where it paints wood grain that is not there.
3. `blank_pt_MetaWorldAssembly_unet_112t128_w64_d3_u2_r112.png` -- what it draws from its weights
   alone, which here is very nearly nothing.
