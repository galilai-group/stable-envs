# Depth-reprojected plates (RoboCasa) and two-rate rendering (HumanoidBench)

Everything here is a measurement of one idea: **"there is no static view to bake" was concluded from
EXACT camera poses.** RoboCasa's room cameras move 2.2-3.4 cm over the env's own state pool, which is
tiny, and real-time graphics has standard machinery for exactly that -- bake a plate WITH its depth
buffer and reproject it to the new camera. Nothing here touches `stable_envs/`.

Results and images land in `/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/reproj/`.

## Verdict in one line each

* **RoboCasa: the plate works and is worth 1.50-1.58x, at a fidelity cost of ZERO against the
  official renderer** -- 5.934 mean |diff| against the shipped renderer's 5.925, SSIM 0.9089 against
  0.9091. It does not reach the 2.66-2.84x bar, and the reason is measured rather than argued: the
  second half of the plan, windowing the dynamic pass, is blocked by two independent facts about
  where a ray tracer's time goes (Section 5).
* **Composed with internal-resolution reduction it reaches 2.01x at r192 and 2.39x at r144 and stops
  there** (Section 6). The two levers compose with a decay the arithmetic predicts -- the plate's
  marginal value falls 1.59x -> 1.03x from r256 to r128, because what it deletes scales with `r^2`
  and what it costs does not -- so the product 1.59 x 1.75 is not available. What IS available is the
  better frontier: **at matched speed the composition is 0.45-0.67/255 more accurate than resolution
  alone**, i.e. 2.0x at r192 instead of r144, at a resolution where nothing is visibly soft.
* **HumanoidBench: no. The ceiling on ANY scheme that keeps the robot's own pixels is 1.29-1.34x,**
  measured by timing the robot's silhouette rays directly, against a median 1.60x needed. The
  realised two-rate configuration is 1.10x and the adaptive-refinement one is 0.85x -- slower than
  the frame it refines.

## The one mechanism everything rests on

`mujoco_warp`'s megakernel reads its primary ray as `ray[rayid]` -- a FLAT array of directions in the
CAMERA's own frame, shared by every world -- and transforms it by `cam_xmat[worldid]` about
`cam_xpos[worldid]` (`mujoco_warp/_src/render.py`, the `use_precomputed_rays` branch). Two things
follow, and this whole directory is built on them:

* a context built at `cam_res=(S,S)` whose `rc.ray` is overwritten with an **arbitrary gather** of
  another table's directions renders exactly those pixels -- a rectangle, the complement of a
  rectangle, every second pixel, or the union of 71 boxes. It only has to be a SET, and the same set
  for every world in the launch;
* the camera pose is per world, so **one launch bakes 64 plates from 64 different viewpoints**.

`rc.depth_data` is planar depth (`dist * -ray_z`) and exactly 0.0 on a miss, which is what makes both
the reprojection and the depth composite possible.

## 1. The fidelity budget: what the STORED baseline already costs

Every target here is "beat reading the env's own frames back off disk", and those frames are JPEG-q95
in Lance. So the incumbent is measured on the same ruler first (`probe_jpeg_budget.py`, 64 states,
256 px). **JPEG-q95 of the OFFICIAL frame, against the official frame:**

| cell | mean abs | within 8 | exact | SSIM |
|---|---:|---:|---:|---:|
| RoboCasaCounterToSink | 0.977 | 0.9968 | 0.193 | 0.99549 |
| RoboCasaCounterToSinkRight | 1.020 | 0.9958 | 0.181 | 0.99558 |
| RoboCasaCounterToSinkWrist | 0.627 | 0.9988 | 0.293 | 0.99982 |
| RoboCasaCounterToSinkL45S33 | 0.847 | 0.9947 | 0.419 | 0.99532 |
| HumanoidBenchH1Walk | 0.294 | 1.0000 | 0.801 | 0.99867 |

**That is the budget an approximation may spend: about 1.0/255 and 0.005 of SSIM on RoboCasa, 0.3/255
and 0.0013 on HumanoidBench.** Two consequences worth stating plainly:

* it is far tighter in SSIM than the declared ceilings suggest -- CounterToSink declares SSIM 0.907
  against a floor of 0.86, which is 0.047 of headroom, 9x the JPEG budget;
* **`exact_frac` is not a metric a stored-frame consumer has.** JPEG-q95 alone takes whole-frame
  exactness to 0.19-0.42 and `interior_exact_frac` to 0.02-0.06 on the room cameras. An approximation
  that costs exactness and nothing else costs a downstream consumer nothing.

`jpeg_rc_*.png` / `jpeg_hb_*.png` are official | JPEG-q95 | ours | 8x |jpeg-official| |
8x |ours-official|, so the two error terms can be compared by eye.

## 2. The hole-fraction curve

`probe_delta.py`. Bake one plate at one camera, then move the camera by exactly *d* mm along eight
random directions (or rotate it by exactly *a* degrees) and compare against a TRUE render from the
displaced pose. `camera_error` asserts MJWarp put the camera where it was asked to (measured 1e-7 m).
Plate at the target resolution, holes filled by a 3x3 dilation:

| delta | holes | mean abs | SSIM | | rotation | holes | mean abs | SSIM |
|---|---:|---:|---:|---|---|---:|---:|---:|
| 0 mm | 0 % | **0.00** | **1.000** | | 0.0 deg | 0 % | 0.00 | 1.000 |
| 1 mm | 0.00 % | 1.13 | 0.958 | | 0.1 deg | 0.03 % | 2.03 | 0.928 |
| 2 mm | 0.10 % | 2.00 | 0.922 | | 0.25 deg | 0.67 % | 2.69 | 0.902 |
| 5 mm | 1.01 % | 2.54 | 0.910 | | 0.5 deg | 1.19 % | 2.34 | 0.918 |
| 10 mm | 1.56 % | 2.25 | 0.919 | | 1 deg | 2.43 % | 2.53 | 0.918 |
| 20 mm | 3.09 % | 2.59 | 0.916 | | 2 deg | 4.85 % | 2.93 | 0.909 |
| 50 mm | 7.60 % | 3.69 | 0.886 | | 5 deg | 11.6 % | 6.42 | 0.832 |
| 100 mm | 14.5 % | 6.74 | 0.818 | | | | | |
| 300 mm | 37.9 % | 19.03 | 0.704 | | | | | |

**The holes are not the problem and never were** -- at the pool's own 11 mm mean delta they are 1.6 %
of the frame and a 3x3 dilation removes all of them. What costs is RESAMPLING: the error saturates at
~2.5/255 anywhere between 2 and 20 mm, because a forward splat rounds each source pixel to the
nearest target pixel and a sub-pixel shift moves ~30 % of them by one. The curve therefore reads
backwards from the expectation: **the quantiser has to be fine not to avoid holes but to avoid
resampling**, and at 1 mm the whole thing costs 1.13/255, the same order as JPEG's own 0.98.

`delta_*_20mm.png` is truth | reprojected+filled | reprojected raw | 8x difference at a deliberately
large 20 mm, where the raw disocclusions are visible as streaks and the filled frame is not.

### The supersampled plate is a measured NEGATIVE

Baking the plate at 2x or 4x and splatting down is the standard answer to magnification. It makes the
holes smaller (0.77 % against 1.56 % at 10 mm) and **the error worse at every delta**, including
delta = 0, where a 1x plate is exact and a 2x plate scores 2.53/255. The mechanism: MJWarp's 256 px
frame is a POINT SAMPLE at each pixel centre, so a 512 px plate splatted down delivers whichever of
its four sub-samples wins the z-test, and that is a different sample from the one the frame is scored
against. Supersampling helps when the target is an area average, and this target is not one.

## 3. The quantised plate cache

`probe_holes.py` over the env's own `sample_states`, and the cache in `rc_pipeline.py`. The camera is
a deterministic function of the state, so the cache is keyed on the camera pose quantised onto a grid
and the **cell centre** is what gets baked -- keying on the sample that missed would be right for that
sample and off by a whole cell for the next.

Measured over the pool (`RoboCasaCounterToSink`, 64 states): `cam_xpos` spread 2.4 cm, mean delta from
the pool medoid 11.0 mm, max 26.4 mm, rotation max 0.94 deg, 56 distinct poses in 64 states.

| plates | delta to plate (mean/max) | holes | mean abs vs true static | SSIM |
|---|---|---:|---:|---:|
| 1 | 11.0 / 26.4 mm | 2.20 % | 2.11 | 0.928 |
| 4 | 0.72 / 3.7 mm | 0.12 % | 1.04 | 0.964 |
| 16 | 0.10 / 0.45 mm | 0 % | **0.22** | **0.993** |

**Hit rate, measured honestly** (`rc_pipeline.py --fresh`): the timed repetitions re-render one batch
and hit 100 % by construction, so FRESH draws from `sample_states` with fresh seeds were run against a
warm cache. At a 1 mm / 0.05 deg grid the cache holds **58 plates for CounterToSink, 68 for Right, 64
for L45S33 and 198 for the wrist**, and every fresh draw hits **100 %** with **zero** new plates. The
camera pose is a function of the arm's qpos and the arm's qpos comes from a recorded pool, so the pose
set is finite and the cache saturates. That is a property of the state DISTRIBUTION, not of the env,
which is why the worst case is measured too (Section 4, `--nocache`).

A plate is 1.0 MB on device (a `(65536,3)` float32 world-point buffer and a `(65536,3)` uint8 colour
buffer), so 200 plates is 200 MB.

## 4. The RoboCasa frontier

`rc_pipeline.py`, batch 1024, 64 worlds per launch, 256 px, H200. `fused` is the same code path with
one full-scene launch per chunk and is within 2 % of the env surface. Fidelity against the OFFICIAL
MuJoCo raster on 64 states; `vs fused` is the approximation's own cost, on 256 frames.

| config | f/s | x fused | mean abs vs official | within 8 | SSIM | vs fused: mean / SSIM |
|---|---:|---:|---:|---:|---:|---|
| **CounterToSink** shipped | 5,730 | 1.00 | 5.925 | 0.8349 | 0.9091 | — |
| plate, 1 mm grid, mask window | 8,413 | 1.49 | **5.934** | 0.8349 | **0.9089** | 0.42 / 0.9861 |
| plate, 1 mm, no sort | 8,913 | **1.58** | 5.934 | 0.8349 | 0.9089 | 0.42 / 0.9861 |
| plate, 1 mm, rectangle window | 8,600 | 1.53 | 5.934 | 0.8349 | 0.9089 | 0.42 / 0.9861 |
| plate, 5 mm grid | 7,116 | 1.27 | 6.086 | 0.8337 | 0.9023 | 1.23 / 0.9551 |
| plate, 20 mm grid | 5,661 | 1.01 | 6.237 | 0.8320 | 0.8997 | 1.55 / 0.9511 |
| plate, cache DISABLED (worst case) | 4,239 | 0.75 | 5.934 | 0.8349 | 0.9089 | 0.42 / 0.9861 |
| declared gate | | | <= 8.5 | >= 0.78 | >= 0.86 | |
| **Right** shipped | 5,661 | 1.00 | 5.124 | 0.8472 | 0.9219 | — |
| plate, 1 mm | 8,345 | **1.50** | 5.128 | 0.8471 | 0.9213 | 0.39 / 0.9879 |
| **L45S33** shipped | 5,835 | 1.00 | 30.885 | 0.3673 | 0.8230 | — |
| plate, 1 mm | 8,969 | **1.54** | 30.888 | 0.3673 | 0.8231 | 0.30 / 0.9849 |
| **Wrist** shipped | 13,341 | 1.00 | 4.338 | 0.8648 | 0.9492 | — |
| plate, 1 mm | 12,467 | **0.96** | 4.346 | 0.8647 | 0.9491 | 0.22 / 0.9976 |

Three things to read off it:

* **the approximation is free against the declaration.** Every cell moves the declared metrics by
  0.001-0.009 of a level and the third decimal of SSIM, and its own residual against the frame it
  replaces (0.22-0.42 mean |diff|) is **less than half what JPEG-q95 already costs** (0.63-1.02). Its
  SSIM cost, 0.986-0.998, is the one column worse than JPEG's 0.995-0.9998. `pipe_q_rc.png` is
  official | fused | ours | the three difference maps: the ours-vs-fused map is black except for
  silhouette edges, and the ours-vs-official map is indistinguishable from the fused-vs-official one.
* **the grid is the knob and finer is both faster and better.** 20 mm is 1.01x because holes appear
  and have to be filled; 1 mm has no holes at all and skips the fill.
* **the cache is load-bearing.** Disabled -- every state a pose never seen -- the scheme is 0.75x, so
  this is an optimisation for a state distribution with a finite camera set, not a free win. It does
  degrade gracefully in fidelity (identical pixels) but not in speed.

## 5. Why the dynamic half cannot be windowed: two mechanisms, both measured

The plan was plate + windowed dynamic pass, and `scratch/perf/window_dyn.py` says a rectangle window
scales with the ray count (0.25 of the rays -> 3.01x). The pipeline opens a window covering the movers
and gets **1.03x** for 0.55 of the rays. `probe_coherence.py` holds the ray COUNT fixed at 0.517 of
the frame and varies only which rays and in what order -- same scene, same worlds, same thread count:

| ray set | movers-only BVH: ms | x | whole scene: ms | x |
|---|---:|---:|---:|---:|
| full frame (1.000 of the rays) | 4.920 | 1.00 | 11.348 | 1.00 |
| `toprows` -- the first 132 rows | 1.052 | **4.68** | 3.116 | **3.64** |
| `stride2` -- every second pixel, in order | 3.522 | 1.40 | 8.060 | 1.41 |
| `rect` -- one rectangle over the movers | 3.739 | 1.32 | 7.487 | 1.52 |
| `mask` -- the union of the movers' own boxes | 4.506 | 1.09 | 8.651 | 1.31 |
| `random` -- the same count, scrambled | 5.751 | **0.86** | 13.739 | **0.83** |

Both columns say the same thing, so it is a property of the traversal and not of one BVH.

1. **Ray coherence is worth 1.63x.** `stride2` and `random` draw the SAME rays -- every second pixel
   against a uniform sample -- and differ only in the order the threads read them: 3.52 ms against
   5.75 (8.06 against 13.74 on the whole scene). Half the rays, scrambled, is **slower than all of
   them in order**. An arbitrary ray SET is therefore not free even though the gather is.
2. **A mover window keeps the expensive rays.** `mask` and `rect` are the same count and `mask` costs
   20 % more, because the pixels a mover occupies are exactly the ones whose rays descend into a mesh
   BVH; `toprows` is 4.68x for the same count because the top of this frame is cabinet doors. Per-ray
   cost varies about 4x across the frame and a window aimed at the movers aims at the peak.

Between them the two account for the whole gap: the dynamic half is 0.55 of the rays and 0.9 of the
cost, so the frame is the plate's 1.5x and nothing more.

## 6. Composing the plate with internal-resolution reduction

`scratch/altrender/` measures a second lever on these exact cells -- render at `r` px, bilinear
upsample to 256, upsample inside the timed region. The two attack different terms: the plate removes
the static rays, the resolution lever shrinks every remaining one. `rc_pipeline.py --render-res r`
runs both and reports the resolution lever ALONE beside the composition in the same job, on the same
states, with `probe_budget_frontier.py`'s own `upsample` function so the two are on one frontier and
not two. Cells' own env surface is the denominator; `plate marginal` is composed / resolution-only.

| RoboCasaCounterToSink | composed f/s | x env | res-only x env | plate marginal | mean abs | SSIM |
|---|---:|---:|---:|---:|---:|---:|
| shipped, 256 px | 5,740 | 1.00 | 1.00 | — | 5.925 | 0.9091 |
| plate, r256 | 8,922 | 1.56 | 1.00 | **1.59x** | 5.934 | 0.9089 |
| plate, r224 | 10,200 | 1.78 | 1.16 | 1.53x | **5.880** | **0.9148** |
| plate, r192 | 11,556 | 2.01 | 1.43 | 1.41x | 6.135 | 0.9096 |
| plate, r176 | 12,282 | 2.14 | 1.60 | 1.34x | 6.255 | 0.9061 |
| plate, r160 | 13,015 | 2.26 | 1.82 | 1.24x | 6.458 | 0.9019 |
| plate, r152 | 13,220 | 2.31 | 1.95 | 1.19x | 6.538 | 0.8994 |
| plate, r144 | 13,665 | 2.39 | 2.02 | 1.18x | 6.667 | 0.8968 |
| plate, r128 | 14,570 | 2.54 | 2.47 | **1.03x** | 7.056 | 0.8892 |
| **the bar** | **15,937** | **2.78** | | | <= 8.5 | >= 0.86 |
| JPEG-q95 budget, same states | | | | | 0.974 | 0.9955 |

**They compose, and the composition never reaches 2.81x.** The best is r128 at 2.54x, and by then the
plate is worth 3 %. The decay is monotone and its mechanism is arithmetic: the plate deletes the
static rays, whose count falls with `r^2`, while everything the plate costs -- the pose pass
(0.36 ms/chunk at every resolution, measured), the splat, the composite and the upsample -- is
independent of `r`. 1.59x -> 1.53 -> 1.41 -> 1.34 -> 1.24 -> 1.19 -> 1.18 -> 1.03 is that ratio
closing.

**What the composition IS worth is the frontier, not the peak.** At matched speed it is more accurate
than the resolution lever alone, everywhere from r256 to about r144:

| at this speed | resolution alone | plate + resolution | the plate is worth |
|---|---|---|---|
| 1.8x | ~r150, mean ~6.55 | **r224, mean 5.880** | 0.67/255, and 5.880 is BETTER than shipped |
| 2.0x | r144, mean 6.662 | **r192, mean 6.135** | 0.53/255, 54 % of the JPEG budget |
| 2.26x | ~r134, mean ~6.9 | **r160, mean 6.458** | ~0.45/255 |
| 2.5x | r128, mean 7.050 | r128, mean 7.056 | nothing |

Read the other way: **the composition reaches a given speed at a MUCH milder resolution** -- 2.0x at
r192 instead of r144, 2.26x at r160 instead of ~r134 -- which is the whole point, because the
resolution lever's cost is a visible blur and the `altrender` README measures r112 moving a frozen
encoder's representation by 1-6 whole state changes. `pipe_cmp_rc_r192.png` is official / shipped /
composed / the three difference maps; the composed frame and the shipped frame are not separable by
eye and the ours-vs-shipped map is silhouette edges only.

**The plate's own share of that error is nil.** Composed minus resolution-only, per cell: **+0.014
(r224), +0.011 (r192), +0.021 (r176), +0.016 (r160), +0.005 (r152, r144, r128)** -- 0.5-2 % of the
0.974 JPEG budget. Every one of these rows' error is the resolution lever's; the plate is invisible
inside it, and gets MORE invisible as the resolution drops because the downsample averages away the
resampling residual that is its whole cost.

### The two cautions, both measured

* **Cache the plate at the RENDER resolution, not at full resolution.** `--plate-res 256` with
  `--render-res 176` is mean 6.510 against 6.255, and 11,933 f/s against 12,282: **worse on both
  axes**. Same mechanism as the supersampled-plate negative in Section 2 -- MJWarp point-samples per
  pixel, so a 256 px plate splatted into a 176 px grid delivers whichever sub-sample wins the z-test
  rather than the sample the 176 px frame is scored against. r192 repeats it: 6.372 against 6.135,
  10,939 f/s against 11,556.
* **The 1 mm quantiser does not need re-tuning, and coarsening it is worse.** The grid lives in
  camera-pose space, not pixel space, so the cache is resolution-independent: **58 plates and a
  100 % hit rate on fresh seeded draws at every `r` from 224 down to 128**, the same 58 as at full
  resolution. Coarsening to 5 mm at r176 costs mean 6.414 against 6.255 AND 10,922 f/s against 12,282,
  because holes reappear and the fill has to run.

### The other three cells at r144

| cell | composed f/s | x env | res-only | plate marginal | bar | mean abs (shipped) | verdict |
|---|---:|---:|---:|---:|---:|---|---|
| CounterToSink | 13,665 | 2.39 | 2.02x | 1.18x | 15,937 | 6.667 (5.925) | short |
| Right | 13,612 | 2.40 | 2.14x | 1.12x | 15,863 | 5.912 (5.124) | short |
| L45S33 | 13,694 | 2.29 | 2.06x | 1.11x | 15,723 | 31.073 (30.885) | short |
| **Wrist** | 20,268 | 1.52 | **2.05x** | **0.74x** | 18,743 | 4.620 (4.338) | clears, but WITHOUT the plate |

**The wrist is the one cell that clears, and it clears without me.** Resolution alone at r144 is
27,422 f/s -- 2.05x, against the 1.40x it needs -- and adding the plate makes it **slower** (0.74x),
for the reason its full-resolution row already showed: the wrist's static half is a third of its
frame rather than half, its camera moves 59 cm so the cache holds 198 plates rather than 58, and the
gather over that cache costs more than the rays it saves.

## 7. HumanoidBench: the two-rate ceiling

`hb_fovea.py`. The premise on record is "78 % of the kernel is inside the 25 % of the frame the robot
covers". `rc.ray` lets both halves be timed directly instead of inferred: the robot's box as a
rectangle, the complement of it as an arbitrary set, every second pixel of the complement, and the
union of the robot's own SILHOUETTES -- the tightest fovea the shared ray table allows.

| H1Walk (full frame 3.002 ms) | rays | ms | x |
|---|---:|---:|---:|
| box, the sorted-chunk union rectangle | 0.473 | 2.604 | 1.15 |
| outer field, the complement | 0.528 | 0.314 | 9.55 |
| outer field at every second pixel | 0.132 | 0.136 | 22.1 |
| whole frame at half resolution | 0.250 | 1.230 | 2.44 |
| **union of the robot's silhouettes** | 0.291 | 2.329 | **1.29** |
| two-rate: box at full rate + outer at half | | 2.741 | 1.10 |
| refinement: silhouettes at full + whole frame at half | | 3.559 | **0.84** |

| cell | needs | ceiling if the far field were FREE | tightest fovea | realised two-rate |
|---|---:|---:|---:|---:|
| H1Walk | 1.12x | 1.15x | 1.29x | 1.10x |
| H1HandStair | 1.82x | 1.16x | 1.30x | 1.11x |
| H1HandBalanceSimple | 1.92x | 1.17x | 1.34x | 1.10x |

**The far field is 53 % of the rays and 10 % of the time, so there is 10 % to play with and the fovea
is the other 90 %.** The union of the robot's silhouettes -- 29 % of the frame, and there is nothing
tighter that one shared ray table can express -- costs 78 % of the frame on its own, which caps every
scheme in this family at 1.29-1.34x against a median 1.60x needed. Adaptive refinement is worse than
useless (0.84x) because the refined set is the expensive set and a coarse-plus-refine scheme pays it
twice.

The fidelity is not the problem and is recorded so: the two-rate frame is 0.078-0.202 mean |diff| from
the full frame at SSIM 0.987-0.992, and against the official raster it is 2.921 against the shipped
2.860 with a declared ceiling of 3.4. It passes. It is simply not fast enough.

One thing worth handing on: **the whole frame at half resolution is 2.44-2.50x on these cells**, which
would clear every one of the seventeen. That is internal-resolution scaling, which is not this
directory's lane, but the number is here because it is the only configuration measured on
HumanoidBench that beats the bar.

## Recommendation

**Take the plate on the three RoboCasa room cameras, composed with whatever internal resolution the
consumer will accept, down to about r144. Do not take the plate for the wrist. Do not take anything
for HumanoidBench.**

Three points on the composed frontier, all of them inside every declared gate and inside the JPEG-q95
budget as an increment:

| | speed | mean abs vs official | SSIM | the case for it |
|---|---:|---:|---:|---|
| plate, r256 | 1.56x | 5.934 (shipped 5.925) | 0.9089 (0.9091) | free: +0.009/255, 1 % of the budget |
| **plate, r192** | **2.01x** | 6.135 | 0.9096 | +0.21/255, 22 % of the budget, SSIM unchanged; the mildest thing that doubles throughput |
| plate, r144 | 2.39x | 6.667 | 0.8968 | +0.74/255, 76 % of the budget; the most the composition offers before the plate stops paying |

None of them reaches 2.81x. `scratch/altrender`'s own r112 + texture LOD 2 does (3.17-3.32x), and the
choice between them is a fidelity judgement rather than a throughput one: **plate + r192 gets 2.01x
at a resolution where nothing is visibly soft, where the frozen-encoder probe in that README shows no
representation shift, and where the whole increment is 22 % of what the storage format already
spends.** Its texture LOD 2 is orthogonal to everything here -- 1.05x and BETTER than shipped on mean
abs -- and composes on top of any row above; it was not re-measured here because it is that
directory's lever and it is free.

The integration is small and lives entirely inside `robocasa.py`'s `render`, and it is orthogonal to
the internal-resolution knob -- build the plate and dynamic contexts at whatever `r` that knob picks,
never at 256 (Section 6's first caution):

1. two extra render contexts per `(kitchen, camera, res)` -- static-only with `render_depth=True`, and
   movers-only -- built beside the existing one in `_context`. The mover set is `moving_geoms`:
   measured against the env's OWN `sample_states`, not `body_weldid`, which calls 176 of 344 pinned
   cabinet doors movers;
2. a `_PLATE_CACHE` keyed on `(kitchen, camera, res, quantised cam_xpos/cam_xmat)` at 1 mm / 0.05 deg,
   holding one `(res*res,3)` float32 world-point buffer and one `(res*res,3)` uint8 colour buffer per
   cell, bounded at a few hundred entries;
3. `kinematics` + `camlight` once per chunk, then the plate lookup, `splat`, the movers-only draw at
   full frame, and the depth composite. `rp_common.splat` is 30 lines of torch and costs 0.1 ms per
   64 frames -- no custom kernel is needed, and one was not written because the measurement said the
   reprojection was never the cost;
4. keep the fused path as a fallback for a cache miss rate above some threshold, since `--nocache` is
   0.75x.

It buys **1.50-1.58x at 0.22-0.42 mean |diff| from the frame it replaces**, which is under half of
what the JPEG-q95 storage the numbers are compared against already spends, and it moves no declared
metric by more than 0.009. Composed with the resolution lever it buys 2.01x at r192 and 2.39x at
r144, adding 0.005-0.021/255 of its own at either. It does not reach 2.66-2.84x on the room cameras
alone or composed; Section 5 is why the window cannot supply the rest, and Section 6 is why the
resolution lever cannot either without going below r144.

## Scripts

| script | what it measures |
|---|---|
| `rp_common.py` | contexts, the ray-table camera model, unproject/splat/hole-fill, the four metrics |
| `probe_jpeg_budget.py` | **the budget**: JPEG-q95 against the official frame, in the same four metrics |
| `probe_delta.py` | **the hole-fraction curve**: error against a CONTROLLED camera delta, mm and degrees |
| `probe_holes.py` | the same over the env's own state pool, and the K-plate frontier |
| `rc_pipeline.py` | **the prototype**: plate cache + reprojection + windowed dynamic pass, timed and scored. `--render-res` composes it with internal-resolution reduction and reports that lever alone beside it; `--plate-res` re-measures the supersampled plate at reduced resolution |
| `probe_coherence.py` | **the negative**: identical ray counts, different ray sets and orders |
| `diag_boxes.py` | the movers' screen boxes against the pixels they paint |
| `hb_fovea.py` | HumanoidBench: the robot's rays and the far field, timed separately |
| `job.sbatch` | `sbatch --job-name=<tag> job.sbatch <venv> <script.py> [args]` |

Two bugs found by measurement rather than by reading, recorded so they are not repeated:

* `clip_to_near` concatenated the box corners and the near-plane crossings along a flat point axis and
  reshaped afterwards, which INTERLEAVED the geoms: geom 0 got its own corners and geom 1's. A 6 mm
  gripper part then claimed 58 % of the frame, every chunk chose the full window, and the window
  silently never fired. `window_sides {256: 16}` in the output is what caught it.
* `geom_rbound` as the mover bound is unusable here -- it is the radius about the geom's frame origin
  and on a cabinet-scale mesh that is metres. `geom_aabb` was checked against the meshes' own vertex
  bounds and agrees, so it is the right source and the first suspicion was wrong.
