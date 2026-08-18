# Alternative render backends for RoboCasa and HumanoidBench

Everything here is a measurement of whether something other than `mujoco_warp`'s batch ray tracer can
render these two visual-tier families fast enough to beat reading their own frames back off disk
(JPEG-q95 in Lance, 8 worker processes, 256 px, batch 1024). Nothing here touches `stable_envs/`.

The bar, from `examples/results/storage/lance.json` (`local_n`) against `storage_se-hb_*.json` (`gpu`):

| env | now (f/s) | stored (f/s) | needed |
|---|---:|---:|---:|
| RoboCasaCounterToSink | 5,681 | 15,937 | 2.81x |
| RoboCasaCounterToSinkL45S33 | 5,909 | 15,723 | 2.66x |
| RoboCasaCounterToSinkRight | 5,593 | 15,863 | 2.84x |
| RoboCasaCounterToSinkWrist | 13,273 | 18,743 | 1.41x |
| HumanoidBench, 17 cells | 11,014-18,765 | 20,613-22,197 | 1.12-1.92x (median 1.60x) |

## Verdict

**Nothing here clears the bar, and the reason is one number.** A 256 px frame is 65,536 pixels and
these scenes carry **295k-668k triangles**, 95-99.7 % of them smaller than one pixel. A ray tracer's
cost is O(pixels x log triangles); a rasterizer's is O(triangles) + O(fragments). At 4.5-10 triangles
per pixel the rasterizer does 5-30x more primitive work than there are pixels to paint, and the ray
tracer is barely sensitive to the triangle count at all -- **8.6x fewer triangles buys 1.19x**
(measured, `probe_decimate.py`). MJWarp is on the right side of that crossover and every alternative
here is on the wrong side of it.

## Scripts

| script | what it measures |
|---|---|
| `probe_decompose.py` | per-stage and per-render-feature cost, scene census, what a camera resolves |
| `probe_knobs.py` | the six `create_render_context` compile-time switches, and the world-count sweep |
| `probe_rays.py` | AABBs pierced per ray, hit distance, in-frustum triangle census and projected areas |
| `probe_fidelity.py` | throughput at batch 1024 AND the four declared fidelity metrics, per render config |
| `raster_tri.py` | **the prototype**: a batched z-buffer triangle rasterizer in Triton |
| `probe_decimate.py` | mesh decimation vs MJWarp throughput and the four metrics |
| `probe_madrona.py` | does `madrona_mjx` run, and on what |
| `job.sbatch`, `job_madrona.sbatch` | SLURM wrappers (`open_research`, one GPU, faulty nodes excluded) |

Results land in `/mnt/vast/home/randall.balestriero/logs/stable_envs_logs/altrender/`.

## The free control: why the wrist camera is 2.3x faster

Same compiled kitchen, different camera. `probe_decompose.py` confirms the scene half -- identical
`ngeom 1213 / visible 344 / 442,466 triangles / 73 textures` for both -- so the whole difference is
inside the `render` megakernel. `probe_rays.py` names the camera half, and it is **not** the top-level
BVH:

| | agentview_left | eye_in_hand |
|---|---:|---:|
| geom AABBs pierced per ray | 28.1 | **29.4** |
| AABBs in front of the hit | 22.3 | 25.3 |
| rays that hit nothing | **0.0 %** | **43.1 %** |
| triangles in the frustum | **166,413** | **24,187** |
| mean hit distance | 1.24 m | 1.35 m |
| captured graph, 64 worlds | 10.80 ms | 7.63 ms |

The wrist camera opens *more* geom boxes and 6.9x less triangulated surface, and 43 % of its rays
leave the scene without descending into any mesh BVH. **The cost is leaf-level triangle work, not
top-level traversal** -- and 6.9x less of it buys only 1.42x, which is the log again.

## What each candidate cost, and why it lost

Bare captured-graph time at 64 worlds / 256 px, against the budget the stored-frame bar allows
(`64 / target_fps`, before the env surface's own 4-11 % overhead):

| | RoboCasa agentview | RoboCasa wrist | HB h1-walk | HB h1hand-balance |
|---|---:|---:|---:|---:|
| MJWarp as shipped | 10.80 ms | 7.63 ms | 3.05 ms | 5.27 ms |
| **budget to beat stored frames** | **4.02** | **3.42** | **3.04** | **3.02** |
| Triton raster, small triangles only | 2.47 | 2.16 | 1.73 | 2.24 |
| Triton raster, + large triangles (depth) | 4.39 | 3.58 | 2.85 | 3.38 |
| Triton raster, + triangle ids | 5.69 | 4.35 | 3.38 | 4.04 |
| MJWarp, shadows off | 10.79 | 7.63 | 1.98 | 3.34 |
| MJWarp, meshes decimated ~9x | 8.56 (2.2x) | — | 2.60 | 4.17 |
| madrona_mjx | aborts | — | — | — |

`raster_tri.py` beats MJWarp's *complete shaded frame* by 1.9x on RoboCasa with depth and triangle
ids -- and is still **42 % over the budget for the whole frame**, with no shading written. That is the
result: a rasterizer is not the wrong algorithm by a small margin here, it is short by a margin that
the shader has to fit into and cannot.

### The rasterizer's geometry is right, which is why the timing means anything

`raster_tri.py` checks its own depth buffer against `mujoco.Renderer`'s, over four states:

| | hit/miss mask agrees with MuJoCo | mean relative depth error | within 1 % |
|---|---:|---:|---:|
| RoboCasa agentview | 99.788 % | 0.0013 | 99.8 % |
| RoboCasa wrist | 99.999 % | 0.0033 | 95.0 % |
| HB h1-walk | 99.997 % | 0.0043 | 91.9 % |
| HB h1hand-balance | 99.996 % | 0.0039 | 94.6 % |

Two bugs were found by that check and not by reading the kernel, which is rule 5 of
`docs/concepts/bit-exact-triton.md` twice over. A 4x4 per-lane bounding box **silently dropped every
large triangle**, so the first version reported 15.5 G triangle/s while painting 3.7 % of a frame whose
real coverage is 74.5 %. And with no near-plane clipping a single 400 m floor quad always has a vertex
behind the camera and is rejected whole, so the floor -- 70 % of a HumanoidBench frame -- vanished; it
is cut into a 128x128 grid instead, and those 32 k extra triangles are charged in every timing above.

### Decimation: the cleanest statement of the mechanism

`probe_decimate.py`, MJWarp throughput and the four declared metrics against the OFFICIAL renderer.
The `keep=1.0` row is the same MJCF recompiled through its spec with nothing decimated -- it does not
carry the env module's colour surgery, so the surgery's worth is (shipped - 1.0) and decimation's cost
is (row - 1.0):

| env | keep | triangles | ms | mean abs | within 8 | SSIM |
|---|---|---:|---:|---:|---:|---:|
| HB h1-walk | shipped | 295,328 | 3.00 | 3.50 | 0.933 | 0.849 |
| | 1.0 | 295,328 | 3.09 | 4.77 | 0.900 | 0.640 |
| | 0.1 | 42,744 | 2.77 | 10.17 | 0.872 | 0.496 |
| | 0.03 | **34,450** | **2.60** | 10.27 | 0.872 | 0.494 |
| HB h1hand-balance | 1.0 | 366,704 | 5.47 | 7.28 | 0.875 | 0.642 |
| | 0.03 | **36,789** | **4.17** | 12.71 | 0.849 | 0.506 |
| RoboCasa agentview | 1.0 | 467,084 | 11.40 | 6.37 | 0.834 | 0.909 |
| | 0.03 | **210,426** | **8.56** | 17.54 | 0.722 | 0.691 |

**8.6x fewer triangles buys 1.19x. 10x fewer buys 1.31x.** On RoboCasa only the 87 untextured meshes
of 201 can be touched (`fast_simplification` does not carry texture coordinates), so 2.2x fewer
triangles buys 1.33x -- and costs SSIM 0.909 -> 0.691. This is the log in the ray tracer's cost
function, measured directly, and it is the same fact that condemns the rasterizer.

### HumanoidBench: shadows are 35 % of the frame

The one large shader term in either family, and it is HumanoidBench's alone -- RoboCasa's single light
has `castshadow=0`, so `use_shadows=False` there changes 0 pixels and 0.1 % of the time. Measured at
the env surface, batch 1024, against the official renderer:

| cell | shipped f/s | shadows off f/s | gain | needs | mean abs | SSIM |
|---|---:|---:|---:|---:|---|---|
| H1Stand | 18,760 | 26,508 | 1.41x | 1.17x | 2.86 -> 3.74 | 0.849 -> 0.795 |
| H1Walk | 18,688 | 26,561 | 1.42x | 1.12x | 6.37 -> 7.25 | 0.801 -> 0.750 |
| H1HandStair | 11,669 | 16,443 | 1.41x | 1.82x | 2.76 -> 3.50 | 0.931 -> 0.889 |
| H1HandBalanceSimple | 11,056 | 16,716 | 1.51x | 1.92x | 5.40 -> 6.71 | 0.861 -> 0.813 |

It clears the **eight easiest** of the seventeen cells and misses the nine that need it most, because
the heavy cells gain the same 1.4-1.5x while needing 1.6-1.9x. It also breaks three of the four
declared `mean_abs_max` ceilings and one `ssim_min`, so it is a re-declaration and not a free switch.

### The six untested `create_render_context` switches

`probe_knobs.py`. `enable_backface_culling=False` is the only one that is free: **0 pixels changed**
on all four cells, worth 1.02x on RoboCasa (5,663 -> 5,782 f/s at the env surface) and nothing on
HumanoidBench. `enable_specular`, `enable_emission`, `enable_per_light_ambient`,
`use_ambient_lighting` are all under 1.01x and three of them move pixels a lot.
`use_precomputed_rays=False` and `use_fast_math=False` are 1.03-1.34x **slower**. Tuning is done.

### madrona_mjx

`docs/research/backend3d.md` records it as "builds, does not run", failing at B=8/64 px. That is now
wrong in both directions, and neither correction rescues it:

* **It does run.** Two-geom scene, H200, driver 590.48.01: B=2/32 px, B=8/64 px, B=16/256 px and
  **B=32/256 px** all render (0.63-1.01 ms per batch, i.e. a ~1 ms fixed floor). The earlier failure
  was `MADRONA_MWGPU_KERNEL_CACHE` pointed at a *directory*; it names a **file**, and pointed at a
  directory the executor aborts in `checkAndLoadMegakernelCache` before it builds anything.
* **B=64 aborts at every resolution** (64, 128, 256 px) in `MWCudaExecutor::run`, cuda_exec.cpp:2759 —
  so the ceiling is batch, not resolution, and train and eval batch must equal the renderer's.
* **RoboCasa never gets a frame, for two independent reasons.** `mjx.put_model` refuses the kitchen —
  `NotImplementedError: (mjGEOM_CYLINDER, mjGEOM_MESH) collisions not implemented` — because the
  renderer takes an `mjx.Model` and MJX validates what it can *simulate* even when the caller only
  wants to draw. Zeroing `contype`/`conaffinity` gets past that, and then `MWCudaExecutor::initCUDA`
  (cuda_exec.cpp:2317) aborts at B=32 and B=64, with a 32 GB heap, an 8 GB heap, and
  `MADRONA_DISABLE_CUDA_HEAP_SIZE=1` — so it is the CUDA primary-context retain and not the heap
  limit, and the two-geom scene in the same job configuration survives it. The difference is that
  `mjx.put_model` on a 1,213-geom model has already put JAX in possession of the device.

Deprecated upstream since 2026-02-17. **Do not adopt** — same verdict as before, three attributed
failure points instead of one guess.

## Recommendation

**Keep MJWarp. Neither family can be made to beat its own stored frames by changing the renderer.**

State the ceiling with the mechanism: *RoboCasa tops out near 5,700 f/s at 256 px because 442k-668k
triangles have to be resolved into 65,536 pixels, and every algorithm that is linear in the triangles
loses to one that is logarithmic in them.* The evidence is three independent measurements of the same
crossover -- 8.6x fewer triangles buys 1.19x (decimation), 6.9x fewer triangles in the frustum buys
1.42x (the wrist camera), and a rasterizer that must touch all of them needs 4.4-5.7 ms for visibility
alone against a 3.0-4.0 ms budget for the entire frame.

If someone wants to spend more on this, the ranked residual leads, with what each is worth:

1. **Finish the Triton rasterizer for RoboCasa only.** It is the one candidate that beat MJWarp:
   1.90x on depth+ids at 64 worlds. Three things would move it -- pack depth and id into one 64-bit
   `atomic_min` instead of two 32-bit ones (the id pass is 1.30 ms of 5.69), bin the large triangles
   per tile instead of one program per triangle (the large pass is 1.92 ms of 4.39), and clip at the
   near plane so the plane grid is not needed. An optimistic 3.0-3.5 ms for visibility plus a shader
   priced at MJWarp's own shading share (<=6 % of its frame) lands at **2.0-2.5x**, short of 2.81x, and
   the shader is where all of RoboCasa's fidelity risk lives: 61 % of MJWarp's residual is already
   textured meshes, and this one would have to sample 377 MB of textures with its own UV rule.
   HumanoidBench is not worth attempting -- its visibility pass alone is 2.85 ms against a 3.04 ms
   budget, and it would need a second depth pass for the shadow map that MJWarp spends 35 % of its
   frame on.
2. **Shadows off on the eight light HumanoidBench cells**, if a re-declaration is acceptable: 1.41x for
   mean abs +0.9 and SSIM -0.05.
3. **Not madrona, not decimation, not the render-context switches.** All measured above.

## Two things found on the way that belong to whoever owns the env modules

* **`h1-walk`'s declared fidelity does not reproduce.** `H1Stand` reproduces its declaration exactly
  (measured 2.859 against 2.86 declared, and 18,760 f/s against 18,765 recorded), and so do
  `H1HandStair` (2.7598 vs 2.76) and `H1HandBalanceSimple` (5.4024 vs 5.40). `H1Walk` shares
  `H1Stand`'s entry and measured **6.37** on one seeded draw of `sample_states` and **3.50** on
  another, against a declared 2.86 with a declared draw spread of [2.81, 2.87].
* **RoboCasa's declared `interior_exact_frac` is pessimistic** on every cell: measured 0.430 / 0.682 /
  0.457 / 0.153 against 0.129 / 0.079 / 0.158 / 0.048 declared, on 64 states at 256 px, with the other
  three metrics reproducing. Different state draws, same direction on all four cells.

---

# Part 2: the JPEG budget, and what it buys

The verdict above answers "can a different renderer be faster". This part answers a different
question -- **were we holding the renderer to the right fidelity budget at all?** The thing our
frames replace is not MuJoCo's output, it is a **JPEG q95 blob in a Lance table**, and that is not
MuJoCo's output either. `probe_budget_frontier.py` measures the storage baseline's own error in the
four units a visual-tier class declares, through the repo's own write and read paths
(`PIL.Image.save(quality=95)` from `examples/bench_storage.py`, `torchvision.io.decode_jpeg` from
`examples/bench_lance.py`), on the same official frames.

## Task 1: the budget, and it does not go the way we hoped

**JPEG q95 is far MORE faithful to MuJoCo than our renderer is.** All 21 cells, 64 states:

| | JPEG q95 of official | our shipped render |
|---|---|---|
| mean abs (HumanoidBench) | **0.29 - 0.90** | 2.62 - 8.40 |
| mean abs (RoboCasa) | **1.02 - 1.47** | 5.43 - 31.90 |
| within-8 | **0.986 - 1.000** | 0.370 - 0.958 |
| SSIM | **0.9953 - 0.9998** | 0.822 - 0.931 |
| interior-exact | 0.087 - 0.942 | 0.153 - 0.943 |
| bytes/frame | 10.3 - 18.6 kB | (196.6 kB raw) |

The hypothesis was that the two would be the same order of magnitude and JPEG might be worse. **It is
not.** JPEG is 4-27x smaller in mean |diff| on every one of the 21 cells, and better on within-8 and
SSIM on every one. The single place the hoped-for result appears is **interior-exactness on the two
RoboCasa agentview cells** -- ours 0.430 and 0.682 against JPEG's 0.135 and 0.087 -- because JPEG's
DCT destroys bit-exactness on every flat surface while MJWarp reproduces flat interiors exactly. On
the other 19 cells JPEG wins that metric too or ties it.

So the strict reading of the argument -- *ship anything whose total error is no larger than the
stored frame's* -- **fails before any lever is applied**: the visual tier as it ships is already 4-27x
outside it. That has to be said plainly, and it is the honest answer to Task 1.

What survives is the **incremental** reading, which is what the "the field accepted PNG -> JPEG"
analogy actually supports: a consumer who has already accepted a lossy step should not object to a
second step no larger than the first. That is the yardstick every row below is scored against --
**Δ(mean abs) from a lever must not exceed the mean abs that JPEG q95 introduces on its own.**

## Task 2: the frontier

Full per-cell tables in `budget_*.json`. Two levers dominate everything measured before:

**Internal resolution.** Render at `r` px, bilinear upsample to 256, upsample inside the timed region.
Cost is close to linear in pixels and the error is a mild blur on a frame the tier already
approximates. On RoboCasaCounterToSink: `r224` is **1.22x with a LOWER mean error and a HIGHER SSIM
than shipped**, `r192` is 1.51x at Δmean +0.21, `r128` is **2.70x** at Δmean +1.19 -- inside the 1.46
JPEG budget.

**Texture LOD.** `context_with_tex_lod` box-filters every texture by 2x or 4x before it reaches the
GPU. On RoboCasa `tex2` alone is **1.05x and better than shipped on mean abs (6.37 -> 6.04), within-8
(0.834 -> 0.858) and interior-exact** -- because `mujoco_warp` builds textures at `num_mip_levels=1`
and has a documented minification gap (`render/mjwarp.py`, "the second gap"), so a box-filtered
texture is closer to what the GL reference produces than the base level is. It is a rare lever that
buys speed and fidelity at once. It does nothing on HumanoidBench, which has `ntex=2` and no textured
primitive.

**Levers that lost, measured:**

* **Shadows off is strictly dominated by resolution.** On H1HandBalanceSimple, `r256_noshadow` is
  1.53x for Δmean +1.31 / ΔSSIM -0.048, while `r144` is **1.97x for Δmean +0.53 / ΔSSIM -0.013** --
  more speed for a third of the error. And `sbs_HumanoidBenchH1Walk_r160_noshadow.png` shows why:
  the humanoid's shadow is simply gone. Withdrawn as a recommendation.
* **Upsample filter: bilinear wins.** `nearest` costs 2-3x more mean error at the same speed;
  `bicubic` is close on SSIM but worse on mean and on interior-exact.
* **Specular is camera-dependent, not a lever.** `enable_specular=False` costs +8.17 mean on
  agentview and *gains* -0.80 mean and +0.079 SSIM on the wrist camera. Not shipped either way.
* **Precision is not reachable.** `mujoco_warp`'s render megakernel is written against `float`
  (float32) with no precision switch anywhere in `_src/render.py` or `_src/types.py`; the only
  exposed precision knob is `use_fast_math`, which is **already on** (turning it off costs 8-34%).
  fp16 ray math would mean vendoring the megakernel, which is a fork rather than a configuration.

## Recommended configuration, with its measured re-declaration

`recommended_declaration.json` holds this per cell, in the shape `visual_fidelity` takes, measured
the way the repo measures declarations -- **64 states x 4 independent draws, worst of the four**
(`probe_declare.py`), not the single draw the frontier sweep used.

| family | configuration | speedup | needed |
|---|---|---:|---:|
| RoboCasa, all 4 cells | render **112 px** -> bilinear -> 256, **texture LOD 2**, backface culling off | **3.17 - 3.32x** | 1.46 - 2.81x |
| HumanoidBench, 15 cells | render **160 px** -> bilinear -> 256, backface culling off | **1.63 - 1.81x** | 1.10 - 1.79x |
| HumanoidBench, HandBalanceSimple + HandStair | render **144 px** | **1.95x** | 1.78 - 1.88x |

**Every one of the 21 cells clears its stored-frame target.** What it costs, worst of four draws:

| cell | mean abs was -> is | Δ | JPEG budget | SSIM was -> is | within-8 was -> is |
|---|---|---:|---:|---|---|
| RoboCasaCounterToSink | 6.41 -> 7.53 | +1.12 | 1.46 | 0.907 -> 0.880 | 0.833 -> 0.807 |
| RoboCasaCounterToSinkRight | 5.52 -> 6.82 | +1.30 | 1.47 | 0.921 -> 0.892 | 0.846 -> 0.819 |
| RoboCasaCounterToSinkL45S33 | 32.01 -> 32.76 | +0.75 | 1.20 | 0.822 -> 0.799 | 0.364 -> 0.332 |
| RoboCasaCounterToSinkWrist | 9.69 -> 9.88 | +0.19 | 1.07 | 0.857 -> **0.858** | 0.826 -> **0.837** |
| HumanoidBench (range, 17) | 2.67-8.40 -> 3.04-8.89 | +0.19 to +0.66 | 0.29-0.91 | -0.002 to -0.014 | -0.002 to -0.019 |

**Δ(mean abs) is inside the JPEG budget on all four RoboCasa cells and on about half the
HumanoidBench cells. ΔSSIM and Δwithin-8 are 2-10x OUTSIDE it everywhere.** The wrist cell is the
only one that is strictly better than shipped on SSIM and within-8 while being 3.17x faster.

## What a consumer gives up — stated in the same units, and shown

Three things, and none of them is a free lunch:

1. **Against MuJoCo, the frame gets worse by roughly the amount JPEG q95 already costs, on mean
   |diff| only.** On SSIM and within-8 it gets worse by several times that amount. Numbers above.
2. **It is visible.** `sbs_RoboCasaCounterToSink_r112_tex2.png` puts official / shipped / recommended /
   JPEG-of-official side by side with amplified difference maps. JPEG's map is nearly black (mean
   1.41, max 38) and edge-localised; the shipped render's is structural (mean 6.83, max 173 -- the
   granite counter is drawn flat white, which is MJWarp's textured-box gap and not something this
   proposal introduced); the recommended config adds a further soft blur (mean 7.84, max 181) that a
   reader can see in the faucet and the sink details. **Do not call this imperceptible.** The honest
   claim is narrower: the marginal change is smaller than the approximation the tier had already made,
   and much larger than what the storage baseline does.
3. **A frozen pretrained encoder notices, and so does it already notice the shipped tier.**
   `probe_repr.py`, cosine distance to the official frame expressed as a fraction of the distance
   between two *different states*, three encoders, blank-frame control:

   | | DINOv2 ViT-B/14 | ResNet-18 | DINOv3 ViT-B/16 |
   |---|---:|---:|---:|
   | JPEG q95 | 0.07 | 0.03 | 0.33 |
   | **shipped render** | 0.40 | 0.57 | 1.53 |
   | recommended (r128 shown) | 1.10 | 2.01 | 5.89 |
   | texture LOD 4 | 0.42 | 0.57 | 1.54 |
   | blank frame (control) | 11.8 | 7.8 | 43.9 |

   RoboCasaCounterToSink; HumanoidBench is the same shape (`repr_*.json`). The control fires, so the
   test can fail. **JPEG is invisible to these encoders and the resolution lever is not** -- it moves
   the representation by 1-6 whole state-changes. Texture LOD is indistinguishable from shipped.

   This is a **proxy and is labelled one**. The real test in this repo's tradition is a pretrained
   world model run with its evaluation unchanged and only the pixels swapped; **no such model exists
   for RoboCasa or HumanoidBench**, and training one costs more than this whole investigation. The
   proxy is the strongest evidence available and it argues for caution, not for confidence.

## If you want the cautious point on the frontier instead

`r176_tex2` on RoboCasa is **1.69 - 1.75x** and is inside the JPEG budget on **all three** aggregate
metrics simultaneously -- mean abs moves by -0.00 to +0.10, within-8 *improves* by up to +0.004, SSIM
moves by -0.0025. It does not reach the stored-frame target, but it is the largest speedup this
investigation found that nothing can object to. On the wrist cell `r112_tex2` is both -- fully inside
the budget on all three metrics AND 3.17x, because that camera resolves so little of the kitchen that
resolution costs it almost nothing.

## Scripts added in Part 2

| script | what it does |
|---|---|
| `probe_budget_frontier.py` | the JPEG budget and the whole speed/error frontier, per cell |
| `probe_declare.py` | 4-draw worst-case re-declaration for one configuration, in `visual_fidelity` shape |
| `probe_repr.py` | frozen-encoder representation distance, with blank and JPEG controls |
| `dump_sidebyside.py` | official / shipped / candidate / JPEG plus amplified difference maps |
| `job_array.sbatch`, `job_declare.sbatch` | SLURM array wrappers over the 21 cells |
| `recommended_declaration.json` | **the deliverable**: per-cell config + measured `visual_fidelity` |
