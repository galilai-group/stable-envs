# Step 6D — `step6/fetch-gpu` findings

**Classification: `FINDING` (primary) + `PATCH_CANDIDATE` (secondary, and it does not reach the bar).**

- baseline SHA: `89fa5ca81341dc4df7186ec325cb691431a2ef17` (`sami/pre-step6-baseline`)
- branch: `step6/fetch-gpu`
- authorisation: PROFILE ONLY. No renderer or env code was modified. Everything under
  `scratch/step6/fetch-gpu/` and `examples/results/step6/fetch-gpu/` is new; the one edit outside
  those is a one-word NameError fix in the shared harness (§9).
- hardware: NVIDIA GeForce RTX 3090 (82 SM, 1536 threads/SM, 24 GiB), `gpu2116.oscar.ccv.brown.edu`,
  venv `$SCR/venvs/se-s3b-fetch311` — python 3.12.9, torch 2.14.0+cu126, mujoco 3.11.0,
  mujoco-warp 3.11.0, warp-lang 1.17.0, gymnasium-robotics 1.4.2.
- SLURM: **6343470** (stage profile, `gpu2116`), **6344491** (context ablation + profile_note rerun).
  Step 3's frozen Fetch sweep was job 6113848 on `gpu2101`; the earlier profile-before was 6322235 on
  `gpu2104`. All three are `geforce3090`.

---

## 1. The render-distinct Fetch set, and what collapses

`examples/results/step4/certification.json` carries **12 Fetch configurations**, all
`role = registered_default`, all `tier = parity` (`parity_interior_min = 0.99`), all 480 px,
`supersample = 1`. Every one has `siblings: []` — Step 4 recorded no collapse — but the frozen
conformance oracle shows the collapse anyway: `examples/results/step4/conformance/fetch.json`
`oracle.digest_ours` takes **four** distinct values over the twelve, one per scene.

| render class | configs | `digest_ours` | state corpus `sha256` | state_dim | reference |
|---|---|---|---|---|---|
| PickAndPlace | `FetchPickAndPlace`, `…Dense`, `…Dict` | `ad2507dc907dca20…` | `ec78e35a4db9…` | 25 | `FetchPickAndPlace-v4` |
| Push | `FetchPush`, `…Dense`, `…Dict` | `f729d776c3107997…` | `93d8ca8025ba…` | 25 | `FetchPush-v4` |
| Reach | `FetchReach`, `…Dense`, `…Dict` | `c8855f3a7bb213ba…` | `f5a519689d33…` | 18 | `FetchReach-v4` |
| Slide | `FetchSlide`, `…Dense`, `…Dict` | `e4aa8398b069587b…` | `17375ce1403e…` | 25 | `FetchSlide-v4` |

`Dense` selects a reward and `Dict` an observation container; neither reaches a pixel, and the three
ids per scene share one frozen state corpus and render byte-identically. So a pixel change on this
path has **4 distinct frames to check and 12 configurations to replay** — the replay obligation is
12, not 4, because each carries its own `describe_digest` and its own contract row.

`examples/results/step6/plan.json` widens it further: the workstream's `step4_replay_configs` is
**26** — the 12 Fetch plus 14 `DMC*`, because `certification.json` lists `mjwarp` among the DMC
backends. Any change inside `stable_envs/render/mjwarp.py` therefore obliges all 26. A change
confined to `stable_envs/envs/fetch.py` obliges only the 12. **That distinction decides the cost of
every candidate in §7 and is the reason none of them is proposed against `mjwarp.py`.**

---

## 2. Where the two qualifiers actually stand (measured, B=4096 included)

The earlier profile-before stopped at B=1024 and reported a "7–16x" range. Both envs were re-measured
across the full mandatory batch list on one 3090:

| env | fps(1) | fps(16) | fps(64) | fps(256) | fps(1024) | fps(4096) | `BATCH_SCALES` | `NO_LARGE_BATCH_REGRESSION` |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| FetchReach | 548 | 3,170 | 4,726 | 5,146 | 5,231 | 5,209 | **9.50x** (bar 20) FAIL | **1.012** (bar 0.90) **PASS** |
| FetchPickAndPlace | 538 | 3,160 | 4,682 | 5,088 | 5,179 | 5,157 | **9.59x** FAIL | **1.014** **PASS** |

and Step 3's frozen sweep (job 6113848, `gpu2101`) agrees across all twelve: `BATCH_SCALES`
7.6–11.7x, `NO_LARGE_BATCH_REGRESSION` 1.005–1.018, peak VRAM 3.55 GB at B=4096.

**`NO_LARGE_BATCH_REGRESSION` holds, with margin, on every Fetch configuration.** There is no
large-batch regression to fix. `BATCH_SCALES` is the only blocking qualifier, exactly as
`plan.json` says (`BATCH_SCALES_FAIL x12`, nothing else), and it caps all twelve rows at
`GPU_PARITY` when their ratio `R = 5.3–6.1` against a class-S bar of 4 would otherwise give
`GPU_FAST`.

Artifacts: `profile_stages/Fetch{Reach,PickAndPlace}_stages_gpu2116.{json,md}`,
`profile_before_4096/`, and the pre-existing `profile_before/*_gpu2104.*`.

---

## 3. The wall partition: the plateau is COMPUTE, not un-amortised overhead

`scratch/step6/fetch-gpu/profile_stages.py` charges every microsecond of `env.render(states)` to a
named stage or to a named gap between two stages, so the partition is exhaustive rather than a
residual. FetchReach, 480 px, median of 10 renders:

| B | wall µs/frame | device µs/frame | **device % of wall** | megakernel µs/frame | composite µs/frame | host ms/render |
|---:|---:|---:|---:|---:|---:|---:|
| 1 | 1823.7 | 401.3 | 22.0 % | 286.7 | 34.6 | 1.453 |
| 16 | 315.4 | 226.9 | 71.9 % | 203.2 | 15.1 | 1.705 |
| 64 | 211.6 | 189.7 | 89.6 % | 172.5 | 12.3 | 2.370 |
| 256 | 194.3 | 188.6 | 97.1 % | 172.6 | 12.0 | 5.298 |
| 1024 | 191.2 | 188.5 | 98.6 % | 172.5 | 12.0 | 9.173 |
| 4096 | 192.0 | 189.8 | **98.9 %** | 173.8 | 12.0 | 26.417 |

**From B=256 the GPU is busy for 98–99 % of the wall clock.** The plateau at ~5.2k f/s is not a fixed
cost that refuses to amortise; it is per-frame device work that is already proportional to B. The
per-frame device cost is flat from B=64 (189.7 µs) to B=4096 (189.8 µs) — the path has reached its
asymptote by B=64 and there is nothing left to amortise above it.

Device partition at B=4096 (one profiled render, kineto trace, 385 kernel launches for 4096 frames):

| bucket | µs/frame | share of device |
|---|---:|---:|
| MJWarp ray-trace megakernel | 173.8 | **90.5 %** |
| torch composite / output conversion | 12.0 | 6.3 % |
| other (index materialisation, fills, D2D copy) | 2.09 | 1.1 % |
| MJWarp scene update (`kinematics`, `camlight`, `refit_bvh`) | 0.25 | 0.1 % |

Wall partition at B=4096 (786.3 ms total): `warp_sync` 716.8 ms (91.2 %, the GPU wait for the
megakernel graph), `torch_sync` 43.2 ms (5.5 %, draining the previous chunk's composite),
`gap:enter->plate` 8.7 ms, `upload` 6.4 ms, `gap:warp_sync->context` 3.7 ms (issuing the composite),
`camera_batch` 3.4 ms, everything else < 0.3 ms. **Total host work: 26.4 ms of 786.3 ms = 3.4 %.**

### Things the brief asked to separate, each answered with a number

- **Python loops over batch/env/body: none.** The chunk loop is the only loop in `render`, and it runs
  `ceil(B/256)` times — 1 at B≤256, 16 at B=4096. `_upload` writes the whole chunk with ~10 indexed
  tensor writes; there is no per-env or per-body Python. `sync` counters confirm `_upload` and
  `_context` are entered once per chunk, never per env.
- **Host↔device synchronisation: exactly two D2H round trips per render, at any B.**
  `Tensor.cpu` = 2 and `Tensor.numpy` = 2 at every batch size — the plate-appearance key
  (`render`, `key.cpu().numpy()` then `np.unique`) and the camera key (`_camera_batch`). Plus
  `torch.cuda.synchronize` and `warp.synchronize` once per chunk each. No `.item()`, no `.tolist()`.
- **H2D copies: 4 calls / 60 bytes at B=1, 19 calls / 147,840 bytes at B=4096** — i.e. 36 bytes per
  frame. **Nothing static is re-uploaded per frame.** Geometry, meshes, BVH and textures are uploaded
  once when `_context` builds and the context is cached (`_CTX_CACHE`), and the frame is replayed
  from a captured CUDA graph, so the per-frame host→device traffic is only the style/camera indices.
- **Per-frame writes into the per-world model arrays are 600 bytes/chunk** (`mat_rgba` 35×4,
  `light_diffuse` 1×3, `cam_pos`, `cam_quat`) against 72 bytes/chunk of genuinely dynamic state
  (`qpos` 15, `mocap_pos` 3) for Reach. Under the default style those 600 bytes are the same numbers
  on every render — a re-upload of static appearance — but they are **600 bytes**, and the cost of
  `_upload` is its ~10 kernel launches (≈400 µs per chunk, host-side), not its bandwidth.
- **D2D copies: 691,200 bytes per frame = exactly one full `(480,480,3)` uint8 frame.** That is
  `buf[lo:hi] = comp` plus the once-per-render `torch.stack(plates)`. ~1.7 µs/frame of bandwidth.
- **Allocation:** the composite materialises the channel-swapped RGB, the int64 index for
  `index_select`, the mover mask and the `where` output per chunk. Visible as `index_elementwise`
  (4.4 µs/frame), `elementwise` add (3.2), `where` (3.0) and `memtile_value_kernel<long>` (2.1).
  Footnote: `_mover_lut`'s docstring says `index_select` on int32 avoids "a 64-bit copy of the whole
  frame stack", but `index_select` requires an int64 index, so the 64-bit temporary is materialised
  anyway — that is what `memtile_value_kernel<long>` at 2.1 µs/frame is. Real, and worth 1 %.

---

## 4. The mechanism, and the controlled experiment that proves it

`fps(4096)/fps(1)` is a ratio of a per-frame cost to a per-CALL cost. Decompose the render as

    T(B)  =  H  +  B · D          →      fps(1) = 1/(H + D₁),   fps(4096) ≈ 1/D∞
    BATCH_SCALES  =  fps(4096)/fps(1)  =  (H + D₁)/D∞

- `H` = fixed host cost per `render()` call = **1.476 ms**, i.e. **81 % of the whole B=1 render**.
  Its four terms are `camera_batch` 353 µs, `_upload` 348 µs, the style key + `np.unique` before the
  first `_plate` 318 µs, and issuing the composite 217 µs.
- `D₁` = device cost of one frame = 401 µs. `D∞` = 190 µs. So the occupancy headroom `D₁/D∞` is only
  **2.1x**: one 480×480 frame is **230,400 rays**, and a 3090 holds 82 × 1536 = **125,952** threads
  resident, so a single Fetch frame is already **1.83 full waves** — the device is essentially full
  at B=1 and there is almost nothing for a larger batch to fill.

The decisive experiment is the **same code path at four frame sizes** (`render(size=…)` is a public
argument; nothing was edited). FetchReach, one job, one card:

| frame | px/frame | fps(1) | fps(256) | fps(4096) | µs/frame @4096 | Gpixel/s | `BATCH_SCALES` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 84 px | 7,056 | **531** | 46,200 | 52,425 | 19.1 | 0.370 | **98.6x** PASS |
| 120 px | 14,400 | **540** | 31,476 | 34,219 | 29.2 | 0.493 | **63.4x** PASS |
| 240 px | 57,600 | **540** | 13,355 | 14,067 | 71.1 | 0.810 | **26.1x** PASS |
| 480 px | 230,400 | **513** | 5,106 | 5,209 | 192.0 | **1.200** | **10.2x** FAIL |

**`fps(1)` is 513–540 at every resolution — a 33x change in frame area moves it by 5 %.** That is `H`
measured directly: at B=1 this path spends 81 % of its time on host work that does not know how big
the frame is. `fps(4096)` meanwhile falls in proportion to the pixels. `BATCH_SCALES` on this path is
therefore, to within a few percent, **`fps(4096)/530` — a disguised measurement of the per-frame
device cost, with the batching behaviour divided out.** The identical code batches at 98.6x when the
frame is small and 10.2x when it is large.

Note also the last column: pixel throughput *rises* with frame size (0.370 → 1.200 Gpixel/s). The
480 px configuration is the **most** efficient point of this path per pixel, and it is the one the
qualifier fails.

### Against the rest of the class-S GPU corpus

Step 3's frozen 3090 sweep, sorted by pixels per frame:

| env | px/frame | fps(1) | fps(4096) | `BATCH_SCALES` | Gpixel/s |
|---|---:|---:|---:|---:|---:|
| CartPole (gfxdraw) | 360,000 | **8** | 17,444 | 2113.7x PASS | 6.280 |
| Acrobot (gfxdraw) | 250,000 | **8** | 15,216 | 1875.7x PASS | 3.804 |
| SimplePointMaze (raycast3d) | 250,000 | **155** | 41,250 | 265.7x PASS | 10.313 |
| **Fetch × 12 (mjwarp)** | **230,400** | **443–680** | 5,162–5,267 | **7.6–11.7x FAIL** | **1.19–1.21** |
| PointmazeLargeNavigate (raycast3d) | 200,704 | 202 | 95,962 | 475.3x PASS | 19.260 |
| DMC × 11 (84 px) | 7,056 | 109–1,070 | 2,459–76,735 | 18.9x–380x | 0.017–0.541 |

### The same renderer, MJWarp against MJWarp

`stable_envs/envs/dmcontrol.py` draws the four `TRACKCOM_DOMAINS` with **the same MJWarp megakernel
Fetch uses** (`DMControlTrackCom`; the fixed-camera DMC cells use `raycast3d` + a plate and are not
this comparison). Same backend, same card, same frozen sweep — only the frame size differs:

| env | backend | frame | fps(1) | fps(4096) | µs/frame @4096 | **ns/pixel** | `BATCH_SCALES` |
|---|---|---:|---:|---:|---:|---:|---:|
| DMCCheetahRun | mjwarp | 84 px | 1,069 | 76,589 | 13.1 | 1.85 | 71.6x **PASS** |
| DMCHopperHop | mjwarp | 84 px | 1,070 | 76,735 | 13.0 | 1.85 | 71.7x **PASS** |
| DMCWalkerWalk | mjwarp | 84 px | 1,045 | 73,503 | 13.6 | 1.93 | 70.3x **PASS** |
| DMCHumanoidStand | mjwarp | 84 px | 938 | 64,493 | 15.5 | 2.20 | 68.7x **PASS** |
| FetchReach | mjwarp | 480 px | 548 | 5,209 | 192.0 | **0.83** | 9.5x **FAIL** |
| FetchReach (`size=84`) | mjwarp | 84 px | 531 | 52,425 | 19.1 | 2.71 | **98.6x PASS** |

Two things fall out of that table:

1. **Fetch's per-pixel cost is the best of the five: 0.83 ns/pixel against 1.85–2.20.** The row that
   fails the qualifier is the most efficient use of the megakernel in the corpus. The DMC MJWarp rows
   pass with a *higher* `fps(1)` (938–1,070 against Fetch's 548) purely because their frame is 33x
   smaller, so their `fps(4096)` is 13x larger.
2. **Fetch's own code passes at 84 px** (last row, from the resolution sweep in this job) and fails at
   480 px. One binary, one card, one state corpus, one run: the only variable is the frame size.

Every row that passes does so because its `fps(1)` is small — 8 f/s for the pygame rasterisers,
109–202 for the raycasters. **Fetch has the highest `fps(1)` of any large-frame row in the corpus.**
Its B=1 path is the best-engineered one in the table, and that is precisely why it fails: the
qualifier rewards a bad `fps(1)`.

The frozen spec says so itself (`docs/research/certification.md` §6.7): the 20x bar is
*"set below the observed minimum across 129 rows (22.8x) so it fails only a renderer that is not
actually batching — a per-frame loop scores ≈1x."* Fetch **is** actually batching: one megakernel
launch per 256 worlds, no per-env Python, per-frame cost 1823.7 → 192.0 µs (a 9.5x fall) and flat
from B=64. The qualifier's stated intent is satisfied; its numeric proxy is not, because the proxy is
relative to `fps(1)` and the corpus it was calibrated on contained no path whose single frame already
fills the device.

---

## 5. What it would take to pass, arithmetically

`fps(1) = 548` is fixed by `H`, and `H` is host work the device size cannot touch. So

    BATCH_SCALES ≥ 20   ⟺   fps(4096) ≥ 20 × 548 = 10,960   ⟺   µs/frame ≤ 91.2

against a measured **192.0 µs/frame**. The whole non-megakernel device cost is 14.1 µs/frame, so
even deleting the composite, the scene update and every copy leaves 177.9 µs and `BATCH_SCALES` at
10.3x. **At least 86.7 µs must come out of the megakernel's 173.8 µs — a ≥ 2.0x speedup of the
MJWarp ray-trace at 480 px, holding the pixels identical.** Including the composite, the requirement
on the whole device path is **2.11x**.

And the requirement tightens if the path is made better: every microsecond removed from `H` raises
`fps(1)` and raises the bar with it. Removing the two D2H round trips and the ~400 µs/chunk of
`_upload` launches — the single most user-visible improvement available here, worth ~3.3x at B=1 and
~1.3x at B=16 — would take `fps(1)` to roughly 1,800 and make `BATCH_SCALES` **≈ 2.9x**.

**This is the finding the coordinator needs: on the Fetch path the natural optimisation and the
frozen qualifier point in opposite directions.** The bar is not weakened here and no threshold change
is proposed; the measurement is handed over as it stands.

---

## 6. Answers to the brief's specific questions

- *Is it compute-bound or overhead-bound?* **Compute-bound from B≈64 up** (device 89.6–98.9 % of
  wall), **overhead-bound at B=1** (device 22.0 % of wall). Both, at different ends of the sweep —
  which is exactly the shape that makes the ratio small.
- *Why does the plateau sit at ~5.2k f/s at 480 px?* Because 1/(190 µs) = 5,263 f/s, and 173.8 µs of
  that 190 is one MJWarp megakernel launch over 256 worlds at 230,400 rays each. The plateau is the
  ray-tracer's throughput, reached at B=64 and held to B=4096.
- *Is static geometry / texture / camera re-uploaded per frame?* **No.** 36 bytes/frame of H2D. The
  context, its meshes, its BVH and its textures are built once and cached; the frame is a replayed
  CUDA graph. The per-world *appearance* arrays (600 bytes/chunk) are rewritten every render even
  when unchanged, which is a redundant write but not a measurable cost.
- *Kernel count and time.* 385 kernel launches per render at B=4096 — 16 megakernels (one per chunk)
  and 369 composite/index/fill kernels. Kernel time 770.7 ms against a 786.3 ms wall.
- *Output conversion.* 12.0 µs/frame on device (6.3 %), all on-device: channel swap, mover lookup,
  `where` against the baked plate, and one full-frame D2D copy. There is **no** float→uint8 pass and
  **nothing** goes through the host.
- *Can `BATCH_SCALES` be met by reducing fixed cost at B=1?* **No — reducing fixed cost at B=1 makes
  it strictly worse**, because `fps(1)` is the denominator and is 81 % fixed host cost. It can only
  be met by roughly halving the per-frame device cost at 480 px.

---
## 7. How much of the device cost is actually reachable — measured, not estimated

`scratch/step6/fetch-gpu/ablate_context.py` flips one flag on `mujoco_warp`'s render context (or one
context manager this repo already ships) from the OUTSIDE, clears `_CTX_CACHE` so the flag reaches a
fresh context, then times the **ordinary public `env.render(states)`** and takes a sha256 of the
frames over the frozen Step 4 corpus. A cell only counts if its digest matches the control, because
Fetch declares the parity tier. FetchReach, RTX 3090 `gpu2101`, jobs 6344491 and 6344691 (two
independent runs; the second column pair is the rerun):

| cell | µs/frame @4096 | vs control | `BATCH_SCALES` | pixels identical to control |
|---|---:|---:|---:|:--|
| control | 190.3 / 191.8 | 1.000x | 8.5x / 8.7x | — |
| `render_skybox=False` | 183.8 / 185.7 | 1.035x / 1.033x | 8.9x / 8.7x | **YES** |
| `use_textures=False` | 184.4 / 188.6 | 1.032x / 1.017x | 8.9x / 8.6x | **YES** |
| **both of the above** | — / 185.0 | — / 1.037x | — / 9.1x | **YES** |
| both + `MJW_CHUNK=512` | — / 184.4 | — / 1.040x | — / 8.6x | **YES** |
| `enable_specular=False` | 184.9 / 189.4 | 1.029x / 1.013x | 8.7x / 8.6x | NO |
| `MJW_CHUNK=512` | 187.8 / 191.8 | 1.013x / 1.000x | 8.1x / 8.3x | YES |
| `MJW_CHUNK=1024` | 189.2 / 191.7 | 1.006x / 1.000x | 8.3x / 8.2x | YES |
| `mjwarp_lightspec` | 198.4 / 203.2 | **0.959x / 0.944x** | 8.1x / 8.1x | YES |
| `render_seg=False` | — | — | — | raises: the composite needs the buffer |

Three results worth keeping:

- **Two savings are free of pixels.** `render_skybox=False` is pixel-identical because the composite
  replaces every non-mover pixel with the MuJoCo-baked plate, and a sky pixel is by construction a
  non-mover pixel. `use_textures=False` is pixel-identical because every textured surface in a Fetch
  scene (floor, skybox) is also a plate surface — no mover geom carries a texture. Together they are
  worth **3.7 %**, and they are **not additive** (3.3 % + 1.7 % → 3.7 %): both are removing the same
  texture-sampling work.
- **`mjwarp_lightspec` is a NO-OP here, and slightly negative.** It is worth 1.12x on Meta-World
  because that model has three lights and a headlight carrying four compiled shadow traversals.
  A Fetch model has `nlight = 1`, `use_shadows` is already `False`, and the context is already built
  with `light_castshadow=false`, so there is nothing to elide — and the different inlining context
  costs **4–6 %**. Reproduced on two runs. Hypothesis killed by measurement; do not adopt it here.
- **`MJW_CHUNK` is exhausted.** 512 and 1024 are within noise of 256 at B=4096 and *raise* `fps(1)`,
  so they make `BATCH_SCALES` worse. The constant's docstring already said throughput saturates at
  64 worlds; this confirms it end to end.

**Total pixel-identical headroom available without touching a kernel: 3.7 %, taking 191.8 µs/frame to
184.4. The requirement is 91.2.**

---

## 8. `PATCH_CANDIDATE`s — and neither reaches the bar

Neither of these is implemented; this workstream is authorised to profile only.

### PC-1 — `render_skybox=False, use_textures=False` in `fetch._context` (3.7 %)

- **Mechanism.** `fetch.render` composites the MJWarp frame over a MuJoCo-baked static plate using the
  segmentation buffer, so every pixel the ray-tracer spends on the skybox or on a textured floor is
  computed and then discarded. Turning both off deletes the texture fetch and the cubemap arithmetic
  from the megakernel for those rays.
- **Blast radius.** Two keyword arguments in `stable_envs/envs/fetch.py::_context`. **Nothing in
  `stable_envs/render/mjwarp.py` changes**, so the replay obligation is the 12 Fetch configurations,
  not the 26 that a change to the shared renderer would require.
- **Expected effect:** fps(1) unchanged (device is 22 % of a B=1 render), fps(256) +3.4 %,
  fps(4096) +3.7 %. `BATCH_SCALES` 8.7x → 9.1x — **still FAIL**. `R` 5.34 → 5.54.
- **Pixel risk:** measured digest-identical over the 64 frozen FetchReach states on two runs. It must
  still be digested on the other three scenes before anyone believes it, because the argument
  ("no mover geom is textured, no mover pixel sees the sky") is a property of each scene's MJCF.
  `FetchPickAndPlace` and `FetchSlide` add an `object` geom that Reach does not have.
- **Honest assessment: not worth the replay on its own.** It buys 3.7 % on a metric that already
  passes and does not move the one that fails.

### PC-2 — overlap the composite with the ray-trace (≈5 %, B > 256 only)

- **Mechanism.** The chunk loop brackets each graph launch with `torch.cuda.synchronize()` and
  `wp.synchronize()`, which fully serialises torch's composite stream against warp's render stream.
  Measured cost: `torch_sync` is **43.2 ms of the 786.3 ms** at B=4096 — chunk *i*'s composite
  (49.3 ms of device time in total) is drained before chunk *i+1*'s ray-trace is allowed to start.
  Replacing the two full-device synchronisations with CUDA **event** dependencies in both directions
  preserves the exact ordering the comment in `render` is defending while letting the two streams
  overlap across chunk boundaries.
- **Expected effect:** nothing at B ≤ 256 (one chunk); up to +5.5 % at B ≥ 512. `BATCH_SCALES`
  8.7x → ~9.2x — **still FAIL**.
- **Risk:** this is the determinism guarantee. `check_determinism`, `check_chunked_render_matches`,
  `check_batch_size_invariance` and `check_order_independence` are the frozen checks it must clear,
  on all 12 configurations, before it could be believed.

### PC-1 + PC-2 together

≈ 9 % → `BATCH_SCALES` ≈ 9.5x against a bar of 20. **The gap is not closable by anything short of
roughly halving the MJWarp ray-trace at 480 px**, which means the megakernel — the one thing the
brief rules out of the first intervention, which is upstream `mujoco_warp` code, and which would have
to stay bit-identical against a `parity` declaration measured at interior 0.9988.

### What is NOT proposed, and why

Removing the fixed host cost `H` (§4) is the largest real improvement available on this path — the
two D2H round trips, the two `np.unique` sorts, and the ~400 µs/chunk of `_upload`'s ten small kernel
launches are worth roughly **3.3x at B=1** and **1.3x at B=16**, which is the batch range an
interactive or evaluation caller actually uses. It is not proposed **because it would take
`BATCH_SCALES` from 8.7x to about 2.9x** and drop all twelve rows further below the qualifier. The
coordinator should decide that trade; a workstream should not make it silently.

---

## 9. Things found on the way that belong to someone else

1. **`scratch/step6/profile_note.py` was broken at its write step on the baseline.**
   `host = n["host"]` where `n` is not in scope — a `NameError` raised after all the timing is done,
   so a run measures everything and writes nothing. Job 6343470 hit it on both envs. This is the
   shared harness every Step 6 workstream is told to reuse, so it was blocking all of them. Fixed in
   this worktree (`note["host"]`, one word) and re-run; **the coordinator should propagate that fix
   rather than have each workstream discover it.**
2. **`FetchBase._mover_lut`'s docstring overstates its own optimisation.** It says `index_select` on
   the int32 segmentation avoids "a 64-bit copy of the whole frame stack", but `index_select` requires
   an int64 index, so torch materialises one anyway — visible as `memtile_value_kernel<long>` at
   2.1 µs/frame (≈1 % of the frame) in the B=4096 trace. Not worth a patch on its own; worth not
   believing the comment.
3. **`BATCH_SCALES` on this path is node-dependent.** The same twelve configurations measure 7.6–11.7x
   across four runs on three 3090s. `fps(4096)` is stable to ±1.5 % (5,157–5,310) while `fps(1)` moves
   ±21 % (443–680), because `fps(1)` is 81 % host work and the nodes' processors differ. The
   qualifier's value is therefore mostly a measurement of the login-node-class CPU on whichever node
   SLURM picked — which is exactly what `HOST_BOUND` exists to say about a row, and Fetch does not
   trip `HOST_BOUND` (`cpu_frac` 0.995 against a 1.5 threshold) because its host is busy *spinning in
   a synchronise*, not doing work.

---

## 10. Status against the brief's success condition

> *All relevant Fetch default configs preserve contracts, reach GPU_FAST+, pass BATCH_SCALES, and pass
> NO_LARGE_BATCH_REGRESSION.*

| condition | status |
|---|---|
| contracts preserved | **yes** — nothing was changed; no replay was needed |
| `NO_LARGE_BATCH_REGRESSION` | **PASS** on all 12 (1.005–1.021 against a 0.90 bar) |
| `R ≥ 4` (class S) | **PASS** on all 12 (5.34–6.09) |
| `cores ≥ 1` | **PASS** on all 12 (4.10–5.40) |
| `BATCH_SCALES ≥ 20` | **FAIL** on all 12 (7.6–11.7x), and **measured unreachable** — it needs the per-frame device cost to fall from 192 µs to 91 µs at 480 px, while every flag-level saving available totals 3.7 % |
| `GPU_FAST` | blocked by that one qualifier alone |

**Stop condition met by the second branch:** the batching failure is not fixed, and the precise
remaining bottleneck is measured and handed over — it is the MJWarp ray-trace megakernel at
173.8 µs per 480×480 frame, 90.5 % of the device time, in upstream `mujoco_warp`.

- pixels changed: **no** (nothing implemented)
- tests run: none required — no code under `stable_envs/` was modified
- Step 4 replay: not required; the 12-config list and the 26-config list are recorded in §1 for
  whichever candidate the coordinator authorises
- semantic follow-up: **none** — frames are untouched
