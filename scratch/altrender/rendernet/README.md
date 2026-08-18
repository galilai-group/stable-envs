# A learned render corrector for the visual tier

A small convolutional network that maps a **cheap MJWarp render** to the **official
`mujoco.Renderer` frame**. The visual tier is rendered at 112 px, delivered at 256 px, and comes out
**2.5x closer to MuJoCo than the frame it ships today while running 2.6x faster** — clearing the
stored-frame throughput bar that no change of renderer could reach (`scratch/altrender/README.md`).

Nothing here touches `stable_envs/envs/`. Everything is measured on one H200 through SLURM.

> **This changes the claim the tier makes.** Today the visual tier is *MuJoCo's renderer, reproduced
> approximately, with declared error*. With a corrector it is **a learned approximation of MuJoCo's
> renderer, with declared error** — a different thing, and the paper has to say so in those words.
> Most of the win is that the network **paints a marble texture its input does not contain**:
> `mujoco_warp` draws RoboCasa's textured BOX primitives as one flat colour, so the detail the
> corrector supplies came out of its weights, not out of the frame. That is a decoder that has
> learned the scene, and it is only as good as the distribution it was trained on — see
> [§4](#4-where-it-fails-and-it-fails-hard).
>
> What it does **not** change is the repository's central thesis. A corrector is a deterministic pure
> function of the cheap frame, which is itself a pure function of `(state, style)`, so `render` is
> still `f(get_state())`, still batch-invariant, still reproducible bit-for-bit by anyone holding the
> checkpoint. **It changes the decoder, not the contract.**
>
> Both halves of that have to travel together. Someone who intends to evaluate a policy on official
> RoboCasa pixels needs to know whether the frames they trained on came from *MuJoCo's renderer
> approximated* or from *a learned approximation of MuJoCo's renderer* — and the answer is not visible
> in the picture, only in the declaration. §7 gives the sentence to ship.

---

## 1. The result

`RoboCasaCounterToSink`, **held-out states** — a disjoint set of recorded trajectory states the
training loop never saw (§8) — 256 frames scored in blocks of 64, the block size every
`visual_fidelity` entry in this repo is measured at. Throughput is end to end at batch 1024 (MJWarp
at 112 px into a buffer, then the network replayed from a CUDA graph), measured for every row in
**one process on one node** (`probe_final_fps.py`, 25 reps, IQR < 0.4 %), because per-checkpoint
numbers vary 5 % between nodes and several candidates sit within 2 % of the bar.

| | f/s | x shipped | corrector µs | mean \|diff\| | within-8 | SSIM | interior-exact | p99.9 | max |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| shipped 256 px MJWarp | 5,754 | 1.00 | — | 6.72 | 0.824 | 0.895 | 0.473 | 167 | 206 |
| bilinear 112 -> 256 | 19,705 | 3.42 | — | 7.78 | 0.800 | 0.870 | 0.480 | 139 | 204 |
| Lanczos-3 112 -> 256 | 19,067 | 3.31 | 1.6 | 8.76 | 0.766 | 0.847 | 0.360 | 157 | 212 |
| FXAA on bilinear 112 -> 256 | 16,163 | 2.81 | 11.1 | 8.08 | 0.781 | 0.866 | 0.476 | 123 | 204 |
| **corrector** `unet_112t128_w48_d2_u2` style-aug. | **14,889** | **2.58** | 16.6 | **2.65** | **0.929** | **0.958** | **0.738** | **78** | **205** |
| **corrector** `espcn_112t128_w64_d6_u2` style-aug. | **15,412** | **2.68** | 14.3 | **2.86** | 0.926 | 0.946 | 0.531 | 78 | 207 |
| corrector `unet_112t128_w64_d3_u2` style-aug. | 13,855 | 2.41 | 21.4 | 2.49 | 0.934 | 0.966 | 0.768 | 79 | 207 |
| corrector `espcn_112t128_w96_d6_u2` default-trained | 14,325 | 2.49 | 19.1 | 3.25 | 0.913 | 0.939 | 0.573 | 92 | 236 |
| *JPEG q95 of official — the storage baseline* | *15,937* | *2.77* | — | *1.44* | *0.997* | *0.994* | *0.131* | *13* | *71* |

The JPEG row's f/s is not a render rate at all — it is what eight worker processes read out of a
Lance table (`examples/results/storage/lance.json`), i.e. the thing this whole exercise has to beat,
and it is in the table so the target has a face. The Lanczos and FXAA rows are the measured render
f/s composed with the measured filter cost; every other row was timed end to end.

**Bold rows clear the 14,343 f/s stored-frame target.** The corrector beats the shipped render on all
five declared quantities at once and on the error tail, at 2.6x the speed. Against the aspiration —
JPEG q95, what a stored dataset actually delivers — it closes **77 % of the mean-error gap**
(6.72 -> 2.65 against 1.44) and **overtakes JPEG on interior-exactness** (0.738 against 0.131; JPEG's
DCT destroys bit-exactness on every flat surface, MJWarp does not).

The two non-learned baselines both **lose to plain bilinear**: Lanczos-3 sharpens the aliasing it is
meant to fix (+0.98 mean) and FXAA blurs silhouettes without touching the texgen gap (+0.31 mean, and
it costs 11.1 µs — two thirds of the corrector's whole budget). Aliasing is not the problem. The
missing information is not high-frequency, it is **absent**, and no resampling filter can invent it.

### Against the declaration

`decl` is the exact draw a `visual_fidelity` entry is measured on — 64 states x 4 seeds, worst of the
four. The harness reproduces the shipped declaration to the digit
(`FIDELITY[("L52S03", "robot0_agentview_left")]` = 6.41 / 0.833 / 0.907 / 1.27 / 0.408), which is what
says the corrector's row is measured the same way and not a different way that happens to look good:

| worst of 4 draws | mean \|diff\| | within-8 | SSIM | interior mean | interior-exact |
|---|---:|---:|---:|---:|---:|
| shipped (declared, and re-measured here) | 6.41 | 0.833 | 0.907 | 1.27 | 0.408 |
| corrector `unet_112t128_w48_d2_u2` style-aug. | **2.45** | **0.935** | **0.970** | **0.24** | **0.789** |
| corrector `espcn_112t128_w64_d6_u2` style-aug. | 2.79 | 0.928 | 0.953 | 0.51 | 0.574 |
| corrector `espcn_112t128_w96_d6_u2` default | 2.81 | 0.929 | 0.955 | 0.60 | 0.522 |
| JPEG q95 of official | 1.46 | 0.997 | 0.995 | 1.10 | 0.135 |

**Read the `decl` row with its caveat attached.** `sample_states` draws from a 400-row pool built from
200 recorded trajectory states, and the corrector trained on 150 of those 200 — so three quarters of a
`decl` draw shares its arm pose (not its object yaws) with a training frame. The honest
generalisation number is the **held-out** one, and the gap between them is small: **2.45 in
distribution against 2.65 held out** for the same checkpoint. That 0.20/255 is the entire size of the
state-memorisation effect, and it is a fifth of the gap the corrector closes.

### HumanoidBench

`HumanoidBenchH1HandBalanceSimple` — the cell that needed the most speed (1.73x) and had the second
worst shipped fidelity — and `H1Walk`, the easiest:

| cell | | f/s | x shipped | mean \|diff\| | within-8 | SSIM | interior-exact |
|---|---|---:|---:|---:|---:|---:|---:|
| H1HandBalanceSimple | shipped 256 px | 11,186 | 1.00 | 5.34 | 0.908 | 0.861 | 0.930 |
| | bilinear 112 -> 256 | 31,877 | 2.85 | 6.28 | 0.886 | 0.839 | 0.929 |
| | **corrector** `espcn_112t128_w96_d6_u2` | **19,838** | **1.77** | **3.30** | **0.912** | **0.875** | **0.947** |
| | *JPEG q95 of official* | *21,181* | *1.89* | *0.34* | *1.000* | *0.999* | *0.922* |
| H1Walk | shipped 256 px | 19,368 | 1.00 | 2.86 | 0.934 | 0.844 | 0.930 |
| | **corrector** `espcn_112t128_w96_d6_u2` | **25,606** | **1.32** | **2.82** | 0.928 | 0.838 | **0.944** |
| | *JPEG q95 of official* | *21,055* | *1.09* | *0.29* | *1.000* | *0.999* | *0.932* |

**HandBalanceSimple clears its 19,063 f/s target and improves every metric** (5.34 -> 3.30, a 38 %
reduction). **H1Walk is a speed win and a fidelity wash**: 2.86 -> 2.82 mean, within-8 and SSIM
fractionally worse, interior-exact better, at 1.32x. Both are honest and they are different results —
a cell whose shipped render is already good has less for a corrector to fix.

---

## 2. Which defect is being fixed, and how much of it is memorisation

`sbs_RoboCasaCounterToSink_unet_112t128_w48_d2_u2_sty_e300__r112_tex2_decl.png` is the same state and
the same layout as `scratch/altrender/sbs_RoboCasaCounterToSink_r112_tex2.png`, so the two are
directly comparable, and it is the recommended checkpoint. On that frame: shipped **6.83**,
bilinear-112 **7.84**, **corrector 2.45**, JPEG **1.41**. The marble backsplash and dark granite
counter that MJWarp draws as one flat white are back, and the difference map stops being structural
and becomes edge-localised like JPEG's. `..._r112_tex2_sty_heldout.png` is the same comparison with a
camera and light drawn from the env's declared variation space.

### The resolution lever is nearly free once a corrector exists

Train the same kind of corrector on the **shipped 256 px frame** — no resolution to recover, only the
texgen gap — and the decomposition falls out. Held out, same states:

| corrector input | best held-out mean | f/s | what is being fixed |
|---|---:|---:|---|
| 256 px (shipped frame itself) | **2.41** | 5,162 | texgen gap only |
| 128 px + tex LOD 2 | 2.99 | 13,509 | texgen + 2.0x resolution |
| 112 px + tex LOD 2 | **2.49** | 13,855 | texgen + 2.3x resolution |
| 96 px + tex LOD 2 | 3.21 | 16,937 | texgen + 2.7x resolution |
| — no corrector, shipped 256 px | 6.72 | 5,754 | |

**The texgen gap is 4.31/255 of the shipped render's 6.72 — 64 % of it — and giving away resolution
from 256 px to 112 px costs 0.08/255 once a corrector is present.** That is the whole argument for
the speed lever, and it is also why the 256 px row cannot ship: its render alone is 5,754 f/s against
a 14,343 target, so **a residual corrector that changes no resolution is impossible on RoboCasa at
any network cost.** (The 112 px row beating the 128 px row is the training recipe, not the
resolution: the 112 px model is style-augmented and the 128 px one is not — see §3.)

### Three controls on where the texture comes from

**Dynamic versus static pixels.** Split the frame by how much the OFFICIAL value varies across the
evaluation states; the top quartile is where error can only be earned by reading the input.

| | mean, all | dynamic 25 % | static 75 % |
|---|---:|---:|---:|
| shipped 256 px | 6.72 | 11.24 | 5.21 |
| bilinear 112 | 7.78 | 14.06 | 5.68 |
| corrector (style-augmented) | 2.65 | 7.05 | 1.18 |

Static pixels improve **4.4x**, dynamic ones **1.6x**. Most of the win is on the parts of the scene
that do not move — exactly what "it learned this room's textures" predicts — but the moving parts
improve too, which a lookup table cannot do.

**Explicit screen coordinates buy 0.24/255.** `espcn_112t128_w96_d6_u2_coord` gets two constant
position planes, making screen-space lookup free, and scores **3.01 against 3.25** held out. If the
plain network were already a positional stamp, coordinates would be worth nothing; if it were purely
image-driven, they would be worth a lot. 0.24 is the size of the positional component.

**Blank input: nonzero, but structureless.** Feed a constant frame and see what the network draws from
its weights alone. The recommended checkpoint moves 69 % of pixels by more than 8/255 on mid-grey,
mean residual **12.7/255**, max 27 — and `blank_RoboCasaCounterToSink_unet_112t128_w48_d2_u2_sty_e300.png`
shows what it is: a smooth field plus a regular sub-pixel dither from the final `PixelShuffle`. **No
counter, no sink, no marble, no room.** (The default-trained `espcn_112t128_w96_d6_u2` is worse on this
control — mean residual 30.4/255, 92 % of pixels moved — and still draws no structure.) It is an
output prior firing on an input far outside its training distribution, not a memorised picture. **A
corrector that had memorised the room would draw the room here. Neither of them does.**

**Foreign input: the residual is not a fixed overlay.** Pearson correlation between the mean residual
map on its own kitchen and on another scene: **0.14** (different camera, same kitchen), **0.03**
(different kitchen), **-0.20** (wrist camera); on HumanoidBench, **0.01** across cells. A screen-space
stamp would score near 1. The network reads its input; what it has learned is this scene's
*statistics*, not this scene's *pixels*.

---

## 3. Camera and light variation: train on it, and it is free

RoboCasa's `make_variation_space` declares per-environment camera offset (±6 cm), pan and tilt (±8°),
light scale (0.45-1.6) and tint. That is part of what the env promises, so a corrector has to survive
it. Two checkpoints of one architecture, one trained at the default appearance and one on draws from
the declared space (25 % of training rows pinned at the default), each evaluated on both:

| `espcn_112t128_w96_d6_u2` trained on | at the default appearance | under the declared variation |
|---|---:|---:|
| default only | **3.25** | **9.06** |
| declared variation | **2.65** | **3.27** |
| — bilinear baseline | 7.78 | 8.91 |
| — shipped 256 px | 6.72 | 7.93 |

* **A corrector trained at one camera pose does not survive the env's own declared camera variation.**
  3.25 -> 9.06, which is worse than not correcting at all (8.91) and worse than the render it
  replaces (7.93). Shipping the default-trained model would silently break every consumer who uses
  the variation knobs — and the variation knobs are the reason this family exists.
* **Training on the variation costs nothing and improves the default case**: 2.65 against 3.25 at the
  default appearance. Domain randomisation over the env's own declared space is the regulariser, and
  the augmented model is better everywhere. `unet_112t128_w64_d3_u2` trained the same way reaches
  **2.49**, the best number in this study.

**This is the configuration to ship.** Anything else is a corrector that works only where it was
pointed.

---

## 4. Where it fails, and it fails hard

Zero-shot, same checkpoint, no retraining. `bilinear` is "do nothing but resize", so a corrector worse
than bilinear is *actively harmful*:

| `CounterToSink` corrector (`espcn_112t128_w96_d6_u2`, L52S03 agentview_left) applied to | corrector | bilinear | shipped 256 | verdict |
|---|---:|---:|---:|---|
| `RoboCasaCounterToSinkRight` — same kitchen, camera moved | 6.71 | 6.14 | 4.75 | **worse than doing nothing** |
| `RoboCasaCounterToSinkWrist` — same kitchen, wrist camera | 11.35 | 4.39 | 3.94 | **2.6x worse than doing nothing** |
| `RoboCasaCounterToSinkL45S33` — different kitchen | 31.54 | 32.85 | 32.20 | mean ties; **SSIM 0.733 vs 0.815, within-8 0.291 vs 0.365 are worse** |
| `HumanoidBenchH1HandSpoon` from a `H1HandBalanceSimple` corrector | 11.37 | 9.63 | 8.35 | **worse than doing nothing** |
| `HumanoidBenchH1Stand` from a `H1HandBalanceSimple` corrector | 3.23 | 3.83 | 2.86 | beats bilinear, loses to shipped |

`sbs_*__OOD_RoboCasaCounterToSinkL45S33.png` shows what "mean ties, SSIM is worse" looks like: the
network paints **marble-like speckle onto plain cabinet fronts that carry no texture in the official
frame**. That is a hallucination, it is visible, and **the mean did not catch it — SSIM and within-8
did**. The same picture is why interior-exactness collapses out of distribution (0.015-0.086 against
0.50-0.55 for bilinear): almost no pixel survives untouched. It is the reason this report insists on
four metrics plus tails and not on a mean.

**Style augmentation helps out of distribution and does not rescue it.** The recommended
style-augmented checkpoint scores **5.94** on the moved camera against the default-trained model's
6.71 — now marginally better than bilinear's 6.14 instead of worse — and **33.19** on the different
kitchen against 31.54, with SSIM 0.763 against bilinear's 0.792. Training on ±8° of pan does not
generalise to a camera 40 cm away, and it generalises to a different kitchen not at all.

The one partial transfer is `H1Stand` from `H1HandBalanceSimple` — same robot, similar scene, no
table — which beats bilinear but not the shipped render. Similarity of scene is what transfers, and
nothing else does.

**A corrector is per cell.** One checkpoint per (scene, camera), trained on that cell, declared on
that cell. Not a hardship — every cell already compiles its own MJCF and builds its own render
context — but it must be stated, and a cell without a trained corrector must fall back to the plain
render rather than borrow a neighbour's.

---

## 4b. Depth and segmentation: spent, and they do NOT close the gap

§4 has one obvious cause. The thing the network is missing — *this surface is a textured granite box
that `mujoco_warp` drew as one flat colour* — is not in the RGB frame; it has to be **inferred** from
image statistics, and image statistics are what change when the camera or the kitchen changes.
`mujoco_warp` will hand it over directly, out of the same megakernel. So it was handed over.

`aux.py` adds two per-pixel channels and the geom id is turned into a **row of a table**, built two
ways:

* **`learned`** — an `nn.Embedding` fitted with the network. Maximum capacity, **structurally unable
  to cross scenes**: geom 742 is a different object in a different kitchen.
* **`descriptor`** — `geom_features`, computed from whatever `MjModel` is loaded: the mean colour of
  the texture that *should* be on the surface, how textured it is, the material's specular terms,
  and a **`texgen_gap` flag that is 1.0 exactly on the geoms `render/mjwarp.py` documents as broken**.
  A table built this way is *defined for a kitchen the corrector has never seen*, which is the only
  reason the transfer test is askable at all. It found **195 flagged geoms on L52S03, 215 on L45S33,
  and 0 on HumanoidBench**. Every out-of-distribution row below was run with the table **swapped to
  the target scene's own** (`geom_table_swapped_to_target_scene: true` in the JSON).

### In distribution it works, and it works for the reason claimed

Architecture-matched, RoboCasa held-out states, style-augmented `unet_112t128_w48_d2_u2`:

| aux | mean \|diff\| | within-8 | SSIM | interior-exact |
|---|---:|---:|---:|---:|
| none | 2.65 | 0.929 | 0.9584 | 0.738 |
| depth | 2.60 | 0.930 | 0.9593 | **0.767** |
| **descriptor** | **2.51** | **0.934** | **0.9629** | 0.751 |

and on HumanoidBench (`espcn_112t128_w96_d6_u2`), where the `texgen_gap` column is identically zero:

| aux | mean \|diff\| | within-8 | SSIM |
|---|---:|---:|---:|
| none | 3.30 | 0.912 | 0.8748 |
| depth | 2.95 | 0.923 | 0.8812 |
| descriptor | 2.87 | 0.924 | 0.8807 |

**The descriptor helps in proportion to the defect it names.** On RoboCasa, which has 195 broken
geoms, it is worth 0.09 over depth alone; on HumanoidBench, which has none, it is worth 0.08 over
depth while depth itself is worth 0.35. That is the mechanism check: the channels are not helping
because they are extra channels.

### Out of distribution it does not close, and the descriptor makes it worse

Architecture-matched, style-augmented `unet_112t128_w48_d2_u2`, zero-shot:

| target | aux | mean | within-8 | SSIM | verdict |
|---|---|---:|---:|---:|---|
| `…Right`, camera moved | — bilinear | 6.14 | 0.840 | 0.8997 | |
| | — shipped 256 | 4.75 | 0.867 | 0.9262 | |
| | none | 5.94 | 0.847 | 0.9030 | still below shipped |
| | depth | 5.75 | 0.845 | 0.8998 | mean -0.19, SSIM -0.003 |
| | descriptor | 6.06 | 0.839 | 0.8949 | **worse than no aux** |
| `…L45S33`, other kitchen | — bilinear | 32.85 | 0.336 | 0.7920 | |
| | — shipped 256 | 32.20 | 0.365 | 0.8146 | |
| | none | 33.19 | 0.342 | 0.7626 | |
| | depth | 32.89 | 0.325 | 0.7558 | |
| | **descriptor** | **32.51** | **0.264** | **0.6969** | **mean best, SSIM and within-8 worst** |
| `…Wrist` | — bilinear | 4.39 | 0.859 | 0.9659 | |
| | none | 5.45 | 0.838 | 0.9604 | |
| | depth | 5.22 | 0.842 | 0.9666 | |
| | **descriptor** | **35.60** | 0.610 | **0.6501** | **collapse** |

Read the cross-kitchen block twice. The descriptor gives the **best mean of any variant** and the
**worst SSIM and within-8 of any variant** — 0.697 against no-aux 0.763, bilinear 0.792 and shipped
0.815. This is the same signature as §4 and it is worth restating in the same words, because it is
what a reviewer will assume was not checked:

> **The out-of-distribution failure is caught by SSIM and within-8 and NOT by the mean.** A corrector
> can lower the average error of a frame while making more of its pixels wrong, and only the
> structural metrics say so.

### Why, and the control that says why

The blank-frame probe answers it in one line. Fed a constant grey frame with zero depth and geom id 0:

| aux | residual mean | pixels moved > 8/255 |
|---|---:|---:|
| none (RGB only) | 12.7 / 255 | 69 % |
| depth | **0.6 / 255** | **0 %** |
| descriptor | **86.3 / 255** | **100 %** |

Depth makes the corrector *better behaved* than RGB-only on an empty input — told there is nothing
there, it does nothing. The descriptor does the opposite, and the reason is in the generation log:
RoboCasa's camera sees no sky, so **`background frac 0.000`** — the descriptor model was never once
shown the background row of its own table. It has never seen a table row it did not train on. When
the table is swapped to another kitchen, every row is a vector it has never seen, and the residual
energy it emits on a foreign scene goes to **34.2/255 against 6.8 at home**. It is not transferring;
it is being confidently wrong on new inputs, which is precisely what §4 diagnosed and precisely what
the aux channels were supposed to cure.

### And it costs throughput

`probe_final_fps.py`, one process, 25 reps, with the two extraction kernels and their conversions
**inside** the timed region, because a consumer pays for those on every frame:

| | f/s | corrector µs |
|---|---:|---:|
| render only, 112 px + tex LOD 2 | 19,757 | — |
| render + depth + segmentation, written **and extracted** | **18,437** | — |
| `espcn_112t128_w64_d6_u2`* RGB only | **15,412** ✓ | 14.3 |
| `espcn_112t128_w64_d6_u2`* + depth | **14,484** ✓ | 14.8 |
| `espcn_112t128_w64_d6_u2`* + descriptor | 13,656 ✗ | 19.0 |
| `espcn_112t128_w64_d6_u2`* + learned embedding | 12,265 ✗ | 27.3 |
| `unet_112t128_w48_d2_u2`* RGB only | **14,889** ✓ | 16.6 |
| `unet_112t128_w48_d2_u2`* + descriptor | 13,243 ✗ | 21.3 |

**The buffers cost 3.6 µs/frame end to end, 6.7 % of the render — not the 0.4-2.3 % measured
earlier.** That earlier figure was the megakernel *writing* them; `get_depth`, `get_segmentation`
and the uint8/int32 conversions are the rest of it. Against a 14,343 f/s target that turns a 7.5 %
margin into 1.0 % for the depth variant and into a **miss** for the two that carry geometry identity.

### Verdict

**Not recommended, and the lead is spent.** Depth is defensible on fidelity alone — 2.65 -> 2.60 in
distribution, better interior-exactness, better blank-frame behaviour, and it still clears the bar on
one architecture — but it is 0.05/255 for a 6 % throughput margin, and it does not move §4. The
descriptor is a **real in-distribution gain (2.65 -> 2.51) and an out-of-distribution regression**,
and the learned embedding misses the throughput bar outright.

The honest framing is the narrow one: **a corrector is per (scene, camera). Handing the network the
renderer's own geometry does not change that, and here is the evidence it cannot be avoided cheaply.**
What it costs to train is small — 20 minutes of official frames and 6 minutes of fitting per cell, on
one GPU — and that, not a transferable corrector, is the thing to plan around.

---

## 5. The frozen-encoder check, and it does not agree with the pixels

`probe_repr.py`'s test with the corrector added as a variant: cosine distance to the official frame in
a frozen pretrained encoder, expressed as a **fraction of the distance between two different states**.
The blank-frame control fires at 14-70x, so the test can fail.

Each block is one run, so its `shipped_256` row is the reference measured against the same scale (the
scale is a mean over random state pairs and moves ±9 % between runs, because `probe_repr.embed`'s
permutation is unseeded — compare within a block, never across blocks).

| RoboCasaCounterToSink, held-out states | DINOv2 ViT-B/14 | ResNet-18 | DINOv3 ViT-B/16 |
|---|---:|---:|---:|
| **run A** — `unet_112t128_w64_d3_u2`, style-augmented | | | |
| JPEG q95 of official | 0.10 | 0.05 | 0.45 |
| shipped 256 px | 0.70 | 1.08 | 2.63 |
| bilinear 112 -> 256 | 2.08 | 4.00 | 9.95 |
| **corrector** | **0.89** | **1.05** | **2.26** |
| blank frame (control) | 19.6 | 13.9 | 69.6 |
| **run B** — `unet_112t128_w48_d2_u2`, style-augmented (the recommendation) | | | |
| shipped 256 px | 0.76 | 1.09 | 3.05 |
| bilinear 112 -> 256 | 2.26 | 4.04 | 11.51 |
| **corrector** | **1.28** | **1.32** | **2.72** |
| **run C** — `espcn_112t128_w96_d6_u2`, default-trained | | | |
| shipped 256 px | 0.83 | 1.02 | 2.75 |
| **corrector** | **1.47** | **1.57** | **3.99** |

**This is the one place the pixel metrics oversell, and it has to be said in the same breath as the
headline.** The style-augmented correctors sit **between the shipped render and the bilinear
baseline**, much closer to shipped: run A is worse than shipped on DINOv2 (0.89 vs 0.70), at parity on
ResNet-18 (1.05 vs 1.08) and better on DINOv3 (2.26 vs 2.63); run B is 1.2-1.7x worse on the two
convolutional-scale encoders and better on DINOv3. The default-trained model (run C) is clearly worse
than shipped on all three. Meanwhile every one of them is **2.5x better in pixel mean**.

Note what the recommendation costs on this axis. `w64_d3` is better than `w48_d2` in pixels
(2.49 vs 2.65) *and* better in representation space (0.89 vs 1.28), and it is not recommended, because
it is 13,873 f/s against a 14,343 target. **The throughput bar, not the fidelity, is what picks the
model** — and the encoder gap between the two is larger than the pixel gap, so the price of clearing
the bar is bigger than mean \|diff\| makes it look.

The mechanism is the obvious one: the corrector's texture is *synthesised*. It has the right
statistics and the wrong details, and a pretrained encoder is sensitive to exactly that while
mean \|diff\| is not. Both are measurements and they disagree:

> **In pixels the corrector cuts the tier's error by 2.5x. In a frozen encoder it moves the
> representation about as far as the tier already does — not further, but not less.** Nothing here
> licenses "a model would not notice".

This is a **proxy and is labelled one**, exactly as in `scratch/altrender/README.md`: the real test is
a pretrained world model with its evaluation unchanged and only the pixels swapped, and no such model
exists for either family. The absolute fractions here are larger than that README's because the scale
— the distance between two states — is measured on a *held-out* state set whose 50 recorded poses are
less diverse than a full `sample_states` draw. Compare columns within this table, not across the two.

---

## 6. The frontier, and what it cost to find

### The speed budget is the design constraint

`probe_final_fps.py`, one process, 25 reps, IQR under 0.4 % on every row:

| render | f/s | corrector budget to reach RoboCasa's 14,343 |
|---|---:|---:|
| 256 px (shipped) | 5,754 | negative — impossible |
| 128 px, tex LOD 2 | 16,012 | 7.3 µs |
| **112 px, tex LOD 2** | **19,705** | **19.0 µs** |
| 96 px, tex LOD 2 | 25,070 | 29.7 µs |

Measured corrector cost, batch 1024, fp16, `channels_last`, CUDA-graph replay. The µs is a property
of the architecture — two checkpoints of one architecture agree to 0.02 µs — so a cell's f/s is that
cell's measured render rate composed with it; the RoboCasa column and the `espcn_112t128_w96_d6_u2`
and `espcn_112t128_w64_d6_u2` HumanoidBench entries were also timed end to end directly and agree.

| corrector | params | GFLOP/frame | µs/frame | RoboCasa f/s | HB HandBalance f/s |
|---|---:|---:|---:|---:|---:|
| `espcn_112t128_w48_d4_u2` | 68 k | 0.55 | 10.6 | 16,305 ✓ | 23,827 ✓ |
| `espcn_112t128_w64_d6_u2` | 182 k | 1.49 | 14.3 | **15,412 ✓** | 21,788 ✓ |
| `unet_112t128_w48_d2_u2` | 441 k | 1.32 | 16.6 | **14,889 ✓** | 20,890 ✓ |
| `espcn_112t128_w128_d4_u2` | 364 k | 2.98 | 16.7 | 14,782 ✓ | — |
| `espcn_112t128_w96_d6_u2` | 384 k | 3.14 | 19.1 | 14,325 ≈ | 19,838 ✓ |
| `unet_112t128_w64_d3_u2` | 920 k | 2.55 | 21.4 | 13,855 ✗ | 18,938 ✗ |
| `unet_112t128_w96_d3_u2` | 2.0 M | 5.52 | 31.8 | 12,114 ✗ | — |
| `espcn_96t128_w96_d6_u2` (96 px in) | 384 k | 3.14 | 19.2 | 16,937 ✓ | — |

`espcn_112t128_w96_d6_u2` lands within 0.1 % of RoboCasa's target — too close to claim either way,
which is why the recommendation is a model with margin. **Bigger is not better**: the 2.0 M-parameter
UNet is 31.8 µs and loses on fidelity to models a third its size.

### Two implementation facts worth 3.3 µs and 4.4 µs

* **The output must stay in the model's dtype.** An earlier `_forward_u8` wrote `(base + res).float()`
  before quantising, materialising an fp32 `(1024, 3, 256, 256)` tensor — 805 MB — for a step fp16
  performs exactly (every integer to 2048 is representable). It cost **3.3 µs per frame, 18 % of the
  budget**, and was invisible until the end-to-end timing disagreed with the standalone one by exactly
  that much.
* **`PixelUnshuffle` in front of the trunk is most of the throughput.** At 128 px with C=32 one 3x3
  conv moves 2.1 GB per layer at batch 1024. Folding 2x of resolution into channels leaves the
  arithmetic identical, cuts traffic 4x and doubles the receptive field per layer in output pixels.
  `espcn_128_w32_d4` (0.75 GFLOP) costs **14.5 µs**; `espcn_128_w64_d4_u2` (0.89 GFLOP — *more*
  arithmetic) costs **10.1 µs**.

### What was tried and lost

* **Lanczos-3 and FXAA**, timed and scored in §1. Both worse than bilinear on this content.
* **ReLU.** The first sweep took held-out mean 7.77 -> 4.26 over six epochs and then went back to
  **7.79, the bilinear baseline to three digits** — the trunk had died, and the zero-initialised head
  gave it a perfectly good fallback to fall back to, so the loss curve read as a plateau rather than a
  failure. SiLU plus gradient clipping fixes it. The zero-init that made the failure quiet is also
  what makes every number a departure from a measured point, so it stays.
* **Residual soft-shrinkage** (zeroing residuals below half a level to preserve bit-exact interiors)
  was worth mean 3.305 -> 3.272 and interior-exact 0.416 -> 0.493 at 60 epochs and is **worthless at
  300** — the trained residual becomes sparse on its own (interior-exact 0.57-0.77 with no shrinkage).
  Kept as a flag, defaulted off.
* **A texgen-only corrector at 256 px.** Best fidelity in the study (2.41) and 5,162 f/s, which is
  **slower than the shipped render**. It is an ablation, not a configuration.

---

## 7. Recommendation

**Ship a corrector per cell, trained on that cell's own declared variation space, at 112 px.**

| family | configuration | f/s | target | mean \|diff\| now -> with corrector |
|---|---|---:|---:|---|
| RoboCasa room cameras | 112 px + tex LOD 2, `unet_112t128_w48_d2_u2`, style-augmented, **RGB only** | **14,889** | 14,343 | 6.72 -> **2.65** |
| RoboCasa, more margin | same, `espcn_112t128_w64_d6_u2`, style-augmented | **15,412** | 14,343 | 6.72 -> **2.86** |
| HumanoidBench, heavy cells | 112 px, `espcn_112t128_w96_d6_u2` | **19,838** | 19,063 | 5.34 -> **3.30** |
| HumanoidBench, light cells | 112 px, `espcn_112t128_w64_d6_u2` | 21,788-29,000 | 18,950-19,801 | 2.86 -> **2.82** (speed only) |

**RGB only.** Depth and segmentation were built, trained on and measured (§4b); they buy 0.14/255,
cost 3.6 us/frame, and do not fix the limitation they were aimed at.

and declare it as what it is:

> `visual_fidelity` is measured against `mujoco.Renderer` for a frame produced by MJWarp at 112 px
> followed by a 441 k-parameter convolutional corrector trained on this cell.
>
> **The frame is still a pure deterministic function of `(state, style)`.** The corrector is a fixed
> checkpoint applied to a render that was already such a function, so `render` is still
> `f(get_state())`, still batch-invariant, still reproducible bit-for-bit by anyone holding the
> weights. What changed is the DECODER, not the contract: this tier is no longer *MuJoCo's renderer
> approximated*, it is **a learned approximation of MuJoCo's renderer, with declared error**. Anyone
> who intends to evaluate a policy on official RoboCasa pixels needs to know which of the two they
> trained on, and this sentence is where they find out.
>
> It is **specific to this scene and this camera** — applied to another cell it is worse than not
> correcting at all, and giving it the renderer's own depth and geometry does not fix that (§4b).
> The detail it restores on textured primitives is **synthesised**, not measured: it has the right
> statistics and not the right pixels, which a frozen pretrained encoder can tell apart even though
> the pixel metrics improve by 2.5x.

Three pictures must go with it, all in `logs/stable_envs_logs/rendernet/`:

1. `sbs_<ckpt>__<tag>_decl.png` — official / shipped / bilinear / corrector / JPEG with amplified
   difference maps, on the draw the declaration is measured on.
2. `sbs_<ckpt>__OOD_RoboCasaCounterToSinkL45S33.png` — the same layout on a kitchen the corrector
   never saw, where it hallucinates texture onto plain cabinets.
3. `blank_<ckpt>.png` — what it draws from its weights alone.

### Residual leads, ranked

1. ~~Depth and segmentation buffers.~~ **Spent — see §4b.** They cost 3.6 us/frame end to end (not
   the 0.4-2.3 % the megakernel alone suggested), they are worth 0.14/255 in distribution, and they
   do not close the out-of-distribution gap; the per-geom descriptor makes the cross-kitchen SSIM
   and within-8 *worse*. Do not re-open this without a mechanism that fixes the diagnosed cause,
   which is that the network never sees a table row it did not train on.
2. **Read the BGRA buffer directly.** `env.render(..., out=)` does `c.bgra[..., [2,1,0]]` — an
   advanced-index gather and a copy — and then the corrector permutes again. A corrector that took the
   BGRA view and did the swap inside its first convolution would save a full pass over
   1024x112x112x4 bytes. That needs a change in `stable_envs/envs/`, which is not this module's to
   make.
3. **One corrector TRAINED across the cells of one kitchen.** L52S03's three cameras share a scene,
   and §4b showed that handing a model the geometry is not enough — but it never showed that a model
   *trained on three cameras at once* fails, which is a different and much cheaper question than
   cross-kitchen transfer. It is now the top lead: if one checkpoint covers a kitchen's cameras, the
   per-cell training cost falls by 3x on RoboCasa.
4. **Randomise the aux channels the way the style is randomised.** The descriptor's diagnosed failure
   is that it never saw an unfamiliar table row (`background frac 0.000` in training). Dropping rows
   to the background vector, jittering the descriptor columns, and training over several kitchens'
   tables at once are the obvious repairs, and none of them was tried.
5. **Do not attempt cross-kitchen transfer as a zero-shot swap.** Measured with the target scene's
   own descriptor table in place: mean ties, SSIM 0.697 against bilinear's 0.792. There is no signal
   to build on without retraining.

---

## 8. Method

### The splits, which are the whole reason these numbers mean anything

* **RoboCasa.** `_yawed_pool` rows `2i` and `2i+1` are the same recorded state at two object yaws, so
  splitting the 400 rows would put the same arm pose on both sides. The split is on the recorded
  index: **0-149 train, 150-199 held out**. A training draw re-randomises the yaws with a different
  generator — the same generative process the env uses — and 30 % of draws interpolate the
  non-quaternion `qpos` entries between two training rows. 16,384 training pairs, 2,048 held out.
* **HumanoidBench.** The 512-state pool splits **0-383 train, 384-511 held out**; half of training
  draws interpolate two training rows with the root quaternion renormalised, because
  `reference_render` asserts the `qpos` round-trip to 1e-5 and an unnormalised quaternion is a hard
  error rather than a silent one. 8,192 training pairs, 2,048 held out.
* **`decl`** is the env's own `sample_states` at seeds 0-3 — the draw a declaration is measured on. It
  overlaps training by construction and is labelled in-distribution everywhere it appears.
* **Out of distribution** is a different registered cell entirely: another camera on the same kitchen,
  a wrist camera, a different kitchen, another HumanoidBench task.

### Two harness bugs found and fixed, both of which produced plausible numbers

* **SSIM is stack-size dependent.** `testing.conformance.ssim` runs
  `scipy.ndimage.uniform_filter(x, size=7)` over the whole `(N, H, W)` stack, and a 3-D uniform filter
  smooths along the **frame axis**. SSIM on 512 frames is a different quantity from SSIM on the 64
  every `visual_fidelity` entry was measured at, so every metric here is computed in **blocks of 64
  and averaged**.
* **The shipped baseline has to be rendered at the same appearance.** On a split generated with the
  env's declared camera variation, rendering the shipped configuration at the *default* camera
  compares a frame of one scene against an official frame of another. It reported mean 41.6 and SSIM
  0.22 — a number about the harness, not the renderer. Fixed by threading the per-sample style through
  `shipped_frames`.

### Files

| file | what it is |
|---|---|
| `nets.py` | the correctors: sub-pixel ESPCN and a tiny UNet, both with a bilinear skip and a zero-initialised head |
| `probe_speed.py` | inference cost alone, plus the non-learned baselines (bilinear, Lanczos-3, FXAA) |
| `probe_render_fps.py` | render-only throughput per resolution, and what depth/segmentation cost |
| `gen_pairs.py` | `(cheap render, official frame)` pairs on disjoint state sets, optionally under the env's declared variation |
| `gen_lr.py` | re-render only the cheap half at another resolution, hard-linking the official half |
| `train.py` | the fit, and the held-out number after every epoch |
| `eval_corrector.py` | four metrics, tails, throughput end to end, out of distribution, hallucination controls, frozen encoders |
| `probe_final_fps.py` | every candidate timed back to back in one process, 25 reps |
| `summarise.py`, `final_table.py` | one table over every checkpoint and every evaluation |
| `data.py` | loading, and the dynamic/static pixel masks |
| `aux.py` | depth and per-geom identity out of the same megakernel, and the two ways to turn a geom id into features |
| `gen_aux.py` | add depth and geom-id planes to a split that already has its two halves |
| `aux_table.py` | the aux question in one table: does the out-of-distribution failure close? |

Data under `/mnt/vast/home/randall.balestriero/data/rendernet/`, results, images and checkpoints
under `/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/rendernet/`.
