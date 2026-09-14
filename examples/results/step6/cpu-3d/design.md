# Step 6G — `cpu-3d`: a genuine CPU path for the MJWarp-only scenes (Phase 1, design)

**Classification: FINDING for the four DM Control trackcom cells (design ready for approval, with the
tier it can reach stated per architecture) and BLOCKER for `CPU_FAST` on the twelve Fetch cells
(a narrow, measured one: the decode bar at 480 px is out of reach of any one-core mesh renderer by
5–20×; a genuine CPU path is still possible there, at `CPU_PARITY`).**

- Baseline SHA `89fa5ca81341dc4df7186ec325cb691431a2ef17`; branch `step6/cpu-3d`; nothing under
  `stable_envs/` was modified (Phase 1 is design and characterisation only).
- Every number below is either read off a frozen artifact (cited) or measured by a script under
  `scratch/step6/cpu-3d/` whose output is under `examples/results/step6/cpu-3d/`. The certification
  numbers are from **SLURM job 6345025**, `node2339` (Intel Xeon Platinum 8268 @ 2.9 GHz, `batch`
  partition, no GPU, `OMP_NUM_THREADS=1`, `torch.set_num_threads(1)`), venvs `se-step6` /
  `se-step5` / `se-s3b-fetch311` as noted. Apple M5 numbers appear only as a scaling sanity check
  and are marked as such.

---

## 0. The verdict in one paragraph

The affected set is 16 render-distinct configurations: the four `DMControlTrackCom` cells (84 px,
visual tier, drawn by MJWarp because their camera rides the body) and the twelve Fetch cells (480 px,
parity tier, drawn by MJWarp because the arm is 18 moving meshes). Neither has a CPU path today;
Step 3's "cpu" rows for them are either `CPU_BACKEND_MISSING` or — for the DMC four — the record of
a `device="cpu"` env that silently rendered on the GPU (§1.2). The two halves are different
problems. **The DMC four are 6–19 capsules and spheres over one textured plane under a skybox, with
a camera and a spotlight that translate together and never rotate**: in camera space the skybox is
a constant image (99.4–99.7 % of skybox pixels bit-identical across the frozen corpus in our own
MJWarp frames), the floor is an analytic plane with one bilinear texture tap, and the movers cover
2.5–7.2 % of the pixels. That is a small analytic scene and every piece of it except the skybox
sampler and the spotlight already exists in `raycast3d`/`dmcontrol.py`. A vectorised-torch hybrid
(baked skybox plate + analytic floor + compacted capsule cast + bounded shadow rays) is estimated at
400–1,100 fps at 84 px on one core, which is `CPU_FASTER` (S_cpu 15–40 against the same-host
official renderer's 11–27 fps) and **not `CPU_FAST`**, because the second conjunct — one core of
JPEG decode, measured at **6,629 fps** at 84 px on the certification host — needs 21 ns per pixel,
and torch's one-thread floor is 2 µs per operator and 0.9–2.2 ns per element per operator. A
compiled per-pixel kernel measured in scalar C on the same host reaches 95–150 ns/px (943–1,484
fps); `CPU_FAST` would need a further 4.5–7× from SIMD, which is plausible for the plane/sky/shade
half and unproven for the primitive and shadow half. **Fetch cannot reach `CPU_FAST` on one core
under any architecture**: 31,720 triangles per frame at 480 px against a decode bar of 815 fps is
39 ns per triangle including fill, and the measured scalar floor is 540–1,640 ns per triangle; the
official renderer on the same host is Mesa llvmpipe at 46 fps, and a CPU mover pass would at best
tie it (`CPU_PARITY`). The certification contract is the other half of the answer: a CPU renderer
cannot be held to the frozen MJWarp digests (MJWarp is not digest-stable against itself: 0.02 % of
pixels move run to run, and the baseline tree already misses the four trackcom digests by
0.16-0.61/255 in `mean_pixel` on the same card - the digests are a soft oracle), so a CPU backend
needs either a cross-backend bound that does not exist yet in `certification.json` or its own
visual-tier declaration under the 3.11 reference — a coordinator decision, §8. The fused CPU
primitive cast that both this design and the raycast3d workstream need is one kernel behind
`raycast3d.cast_nearest`, owned by `step6/raycast3d`, with this workstream's whole-frame kernel
layered on it (§6).

---

## 1. The affected set, exactly

### 1.1 What `certification.json` says (frozen Step 4, read-only)

| config(s) | `backends` | tier / bound | size | on-path renderer (measured, `backend_census` in the raycast3d workstream) | Step-4 Gate A |
|---|---|---|---|---|---|
| `DMCCheetahRun`, `DMCHopperHop`, `DMCWalkerWalk`, `DMCHumanoidStand` | `['mjwarp','raycast3d']` | **visual**: `mean_abs_max` 6.5 / 5.7 / 10.5 / 8.2, `within8_min` .84/.86/.65/.69, `ssim_min` .62/.64/.77/.82, declared under mujoco 3.11.0 + mujoco_warp 3.11.0 | 84 | `DMControlTrackCom` → **MJWarp only**; `raycast3d` is a module-level attribution, no raycast3d entry point is called | 3 FAIL (3.12 reference drift, closed by texture-fidelity as BLOCKED-not-FAIL), Humanoid PASS |
| `Fetch{Reach,Push,Slide,PickAndPlace}` × `{'', Dense, Dict}` (12) | `['mjwarp']` | **parity**: `parity_interior_min` 0.99, all-pixel ≥ 0.90, `mean_abs_max` 4.0 | 480 | `FetchBase` → baked MuJoCo plate (91 % of the frame) + MJWarp mover pass, composited by MJWarp's segmentation buffer | 12 PASS (interior 0.99880–0.99901) |
| `DMCAcrobotSwingup … DMCReacherHard` (10 `DMControlSuite` cells) | `['mjwarp','raycast3d']` | parity, interior 1.0 | 84 | **`raycast3d`** (three baked plates + analytic movers) — **already a genuine CPU path**, owned by `step6/raycast3d` | 10 PASS |

The ten `DMControlSuite` cells are **not** in this workstream's problem: they render on CPU today
(Step 3, `node2349`, llvmpipe official 13–38 fps at 84 px: `CPU_PARITY`/`CPU_FASTER`, every row
`BELOW_G4` with cores 0.003–0.06). Their speed is the raycast3d workstream's. What this note reuses
from them is the design, not the rows.

### 1.2 What the "cpu" rows actually recorded (why "76 never benchmarked")

`examples/results/step3/cpu/oscar_x86_mjwarp_1thread.json` is a `device: cpu` shard, run on
`gpu2106`, whose DMC trackcom rows report **1,373 fps at B=1 and 77,457 fps at B=1024 with peak
device memory 1.2–1.4 GB** — that is MJWarp on the RTX 3090 behind a `device="cpu"` request, and
`certification.json` labels it `CPU_UNMEASURED` with `CPU_BACKEND_MISSING, DEVICE_NOT_HONOURED`.
The Fetch "cpu" shards are blocked at construction (`Expected all tensors to be on the same device,
cuda:0 and cpu`). So the honest CPU state of all 16 is: **no path exists**, and the DMC four have a
path that lies about its device. Any Phase-2 design must make `device="cpu"` refuse MJWarp rather
than fall into it.

### 1.3 The bars, on the certification host (job 6345025, `node2339`, 1 thread)

`CPU_FAST := S_cpu ≥ 5 AND cores_cpu ≥ 1`, `S_cpu` against the official renderer **on the same
host**, `cores_cpu` against one core of JPEG-q90 decode of the env's own frame
(`docs/research/certification.md` §5.3). A batch node has no GPU, so the official renderer there is
Mesa **llvmpipe** (`GL_RENDERER = llvmpipe (LLVM 19.1.7, 256 bits)`, Mesa 24.2.8 via EGL), which is
a different denominator from the hardware-EGL numbers Step 3 took on GPU nodes:

| config | official fps, same host (llvmpipe, 1 core) | official fps, Step 3 GPU-node EGL (`gpu2106`/`gpu2101`) | decode fps/core, same host | **`S_cpu ≥ 5` needs** | **`cores_cpu ≥ 1` needs** | ns/px for `cores ≥ 1` |
|---|---:|---:|---:|---:|---:|---:|
| DMCCheetahRun 84 px | 24.4 | 385 | 6,629 (2,495 B JPEG) | 122 fps | **6,629 fps** (151 µs/frame) | **21.4** |
| DMCHopperHop | 27.1 | 391 | 6,629 | 136 | 6,629 | 21.4 |
| DMCWalkerWalk | 25.5 | 378 | 6,629 | 128 | 6,629 | 21.4 |
| DMCHumanoidStand | 10.7 | 355 | 6,629 | 54 | 6,629 | 21.4 |
| FetchReach 480 px | 45.7 render-only / 38.6 from a stored state | 910 | 815 (19,750 B JPEG) | 229 | **815 fps** (1.23 ms/frame) | **5.3** |
| FetchPush 480 px | see §3.2 | 854 | see §3.2 | | 815-class | 5.3 |

(The decode bar was measured on a real official frame with the Step-3 protocol, `PIL` from memory;
Step 3 recorded 6,885–8,054 at 84 px and 976–1,264 at 480 px on the GPU nodes' faster cores. The
`cores` conjunct is the binding one on every row by 25–50×: the same-host `S_cpu` bar is a few
hundred fps because llvmpipe is slow, the decode bar is thousands.)

For reference, the GPU path these rows must be compared against per the fetch-gpu finding: MJWarp on
an RTX 3090 costs **1.85–2.2 ns/px** on the DMC four (76.6 k fps at B=4096) and **0.83 ns/px** on
Fetch at 480 px (5.2 k fps) with the whole card. `cores_cpu ≥ 1` asks one Xeon core for 1/11 of a
3090's per-pixel throughput at 84 px and 1/6.4 of it at 480 px.

---

## 2. Characterisation — the DM Control trackcom four

Scripts: `scratch/step6/cpu-3d/census_dmc.py` (compiled model + frozen corpus, no GL; run locally
under mujoco 3.12 — geometry, materials and texture bytes are byte-identical to 3.11 per
texture-fidelity §2) → `census/DMC*_cis1b1zs2j3.json`; `frames_census.py` over the texture-fidelity
workstream's uncommitted frame stacks (`localize312/frames/*.npz`: `frozen_ours` = our MJWarp frames
at the frozen states, `frozen_official`, `frozen_seg` = MuJoCo's segmentation buffer) →
`census/frames_census.json`, `census/shadow_candidates.json`.

### 2.1 Geometry, camera, lights, materials

| | Cheetah | Hopper | Walker | Humanoid |
|---|---|---|---|---|
| visible geoms | 1 plane + **8 capsules** | 1 plane + **6 capsules** | 1 plane + **7 capsules** | 1 plane + **16 capsules + 3 spheres** |
| meshes / hfields / drawn sites | 0 / 0 / 0 (sites are group 3, hidden) | 0 / 0 / 0 | 0 / 0 / 0 | 0 / 0 / 0 (25 sites, all group 3) |
| static geoms | the plane only | the plane only | the plane only | the plane only |
| capsule radii / half-lengths (m) | 0.046 / 0.07–0.5 | 0.03–0.065 / 0.035–0.165 | 0.04–0.07 / 0.1–0.3 | 0.027–0.09 / 0–0.17 |
| camera 0 | `side`, **trackcom**, fovy 45, offset (0,−3,0) from the torso COM | `cam0`, trackcom, (0,−2.8,0.8) | `side`, trackcom, (0,−2,0.7), pitched 30° | `back`, trackcom, (−3,0,1), pitched 26.6° |
| `cam_xmat` over the 64 frozen states | **constant** | constant | constant | constant |
| `cam_xpos` spread over the corpus (m) | x 0.62, z 0.09 | x 0.91, z 1.12 | x 1.45, z 1.56 | x 1.13, y 0.84, z 1.66 |
| model light | spot at (0,0,2) above the torso COM, **trackcom**, dir (0,0,−1), cutoff 45°, exponent 10, attenuation (1,0,0), ambient 0, diffuse .7, specular .3, **castshadow** | same | same | same |
| light position relative to the camera | **constant** (spread 0.000 on all three axes) | constant | constant | constant |
| headlight | active, ambient .4, diffuse .8, specular .1, no shadow | same | same | same |
| materials in the frame | `grid` (plane): rgba (1,1,1,1), texture `grid`, texrepeat (1,1), **texuniform 1**, reflectance .2, specular .5, shininess .5; `self` (bodies): (0.7,0.5,0.3), no texture, specular .5, shininess .5 | same | same | same (+ spheres wear `self`) |
| textures | `skybox`: type SKYBOX, 800×4800 (6 faces of 800²), 307 distinct colours (gradient + random white marks); `grid`: 300×300, **2 colours** (checker with edge marks), mean (38,64,89) | same | same | same |
| `mj_kinematics` / `mj_comPos` / `mj_camlight` / `mj_forward` per state, `node2339` | 1.4 / 0.6 / 0.5 / 13.7 µs | 1.7 / 0.8 / 0.7 / 10.0 | 2.5 / 1.2 / 0.9 / 14.2 | 6.3 / 1.7 / 1.0 / 37.2 |

Three structural facts fall out of the table and decide the design:

1. **Nothing rotates.** `mode="trackcom"` positions the camera at `subtree_com + offset` with a
   fixed orientation, and the light is `trackcom` on the same body, so the light is a **constant
   offset from the camera**. In camera space the scene is: a fixed ray-direction grid, a plane whose
   pose relative to the camera is a function of `cam_z` alone, a skybox that is a **constant image**,
   a spotlight at a constant position, and 6–19 primitives.
2. **The primitive set is tiny and the floor is one plane with one bilinear texture tap.** No
   meshes, no textured primitives (the bodies' `self` material has no texture, so MJWarp's
   PLANE/MESH-only UV rule is not even exercised on them), no reflection pass in MJWarp (it has none),
   no sites drawn.
3. **Forward kinematics is free**: 3–10 µs per state for the whole chain (`kinematics + comPos +
   camlight`), so a per-state Python loop over `mj_kinematics` costs ~1 % of the frame budget at the
   `CPU_FASTER` level and is not a design variable (the plate families already do exactly this).

### 2.2 What the frozen frames are made of (our MJWarp frames at the frozen states, 64 per cell)

| | Cheetah | Hopper | Walker | Humanoid |
|---|---|---|---|---|
| share of pixels: sky / floor / body | .826 / .132 / .042 | .854 / .121 / .025 | .548 / .380 / .072 | .000 / .962 / .038 |
| body pixels per frame (median) | 296 | 179 | 514 | 272 |
| movers' screen bbox, share of frame (median / max) | .188 / .217 | .073 / .182 | .277 / .541 | .115 / .250 |
| **skybox pixels bit-identical across states, ours** | **.995** | **.997** | **.994** | — |
| skybox pixels bit-identical across states, official | .826 | .631 | .519 | — |
| floor pixels bit-identical across states, ours | .000 | .000 | .000 | .000 |
| distinct colours in ours: sky / floor / body | 313 / 246 / 461 | 337 / 231 / 283 | 293 / 389 / 622 | — / 1,052 / 551 |
| horizon row (first floor row, 84 px) over the corpus | 57–59 | 46–75 | 34–71 | 0 |
| floor pixels inside the movers' dilated bbox (shadow candidates), median / max | .119 / .137 | .065 / .178 | .397 / .463 | .530 / .753 |
| ours vs official, mean |diff| sky / floor / body (3.12 reference) | 4.1 / 19.3 / 12.9 | 4.0 / 20.4 / 12.3 | 4.1 / 33.0 / 11.0 | — / 5.9 / 17.6 |

The skybox row is the important one. In **our** frames 99.4–99.7 % of skybox pixels never change
across 64 states (the 0.3–0.6 % that do are MJWarp's known run-to-run 1-level noise, texture-fidelity
§2), because MJWarp samples the cube map by ray direction and the direction grid is fixed. So for the
three sky-bearing cells **55–85 % of the frame is a per-(camera orientation, resolution) plate that
is exact against MJWarp by construction** — the same argument the OGBench plate makes, applied in
camera space instead of world space. (The official renderer's skybox is *less* constant, 52–83 %,
which is its own multisampled cube-map sampling; a CPU path matching MJWarp inherits MJWarp's 4.1/255
skybox residual against the reference, which is inside the declared bounds and is what the
declaration measured.) The floor is never constant — the checker phase moves with `cam_x` — so it
has to be computed, but it is one plane.

### 2.3 What MJWarp computes per pixel (what a CPU path must reproduce)

Read off `stable_envs/render/mjwarp_lightspec.py` (the vendored megakernel; the DMC context runs it
with `use_textures=True, use_shadows=True, render_skybox=True`, `gl_rounding()` active,
`enable_specular` default on — swimmer alone turns it off — and no reflection pass because MJWarp
has none) and off `characterize/mujoco_warp_sampling_source.txt` (job 6345025, mujoco_warp 3.12.0,
byte-identical to 3.11.0 on these functions per texture-fidelity):

```
ray      d_cam = normalize(left + (right-left)(px+0.5)/W, top + (bottom-top)(py+0.5)/H, -znear),  half_h = znear·tan(fovy/2)
         d = cam_xmat @ d_cam ;  o = cam_xpos
hit      nearest geom by BVH (here: 1 plane + K primitives), normal n, hit point p = o + t d
miss     skybox: (rx,ry,rz) = (d.x, d.z, -d.y); major axis → face ∈ 0..5, (sc,tc); s = (sc/ma+1)/2, t = clamp((tc/ma+1)/2, 0.5/800, 1-0.5/800);
         v = (face + t)/6;  colour = bilinear(tex_800x4800, (s, v))  → packed, NO lighting
colour   base = mat_rgba[matid] (matid ≥ 0) else geom_rgba
texture  PLANE only: local = R^T (p - pos); u = 0.5·local.x·texrepeat_x − 0.5 ; v = −0.5·local.y·texrepeat_y − 0.5 ; wrap; base *= bilinear(tex_300x300, (u, v))
         (no mip level: one bilinear tap of the base level — the minification residual the tier declares)
shade    result = base·headlight_ambient(.4)                                  [+ per-light ambient: 0 here]
         per light: spot: L = normalize(lightpos − p); cosθ = dot(−L, lightdir); if cosθ < cos(cutoff) → 0
                    att = cosθ^exponent (attenuation is the default (1,0,0) → no distance term)
                    ndotl = max(0, n·L); if 0 → skip
                    shadow: any-hit from p + 1e-4·n toward L, max_t = dist − 1e-3 → hit ⇒ visible = 0.3   (NO_LIGHT_AMBIENT_FALLBACK, not 0)
                    w = att·visible;  result += base·(diffuse·ndotl·w) + specular·mat_spec·max(0, n·H)^(128·mat_shininess)·w,  H = normalize(L + view)
         headlight: directional L = −cam_fwd (toward the camera), no shadow, diffuse .8, specular .1, same specular law
         clamp to [0,1], ×255, round-to-nearest (gl_rounding), pack; then the BGRA→RGB swap in the env
```

Two MJWarp-specific details a CPU path has to copy rather than "do right": the shadow term is
**0.3, not 0**, and `wp.texture_sample(LINEAR)` is **CUDA hardware bilinear filtering**, which
computes its weights in 9-bit fixed point (8 fractional bits). A float32 bilinear on CPU is within a
level of it, not equal to it; emulating the 1/256 weight quantisation is cheap and is the first
thing to try when the CPU frames are first diffed against ours. The DMC context also compiles the
megakernel with warp's default `fast_math`, and `mujoco_locomotion.md` §6.5 measured
`use_fast_math=False` to change the numbers "in the sixth decimal" — that is where the last 1-level
disagreements will live and it is not closable from the CPU side.

### 2.4 What `raycast3d` gives and what it lacks for these scenes

| needed | in `stable_envs/render/raycast3d.py` today | in `envs/dmcontrol.py` (`DMControlSuite`) today | missing |
|---|---|---|---|
| plane / sphere / capsule / cylinder / box intersection, world-space normals | `_plane, _sphere, _capsule, _cylinder, _box`, `intersect`, `cast_nearest_torch`, `any_blocking_torch` (bit-identical to the fused CUDA kernel, tested) | — | nothing (no mesh needed here) |
| per-pixel ray grid | `Camera.rays` (frozen dataclass, memoised) — **one camera pose per call** | `_per_chunk_constants` caches per scene | **per-environment origin**: the trackcom camera translates per state. Direction grid stays shared; the origin becomes `(B,3)`. Trivial, and it is exactly the `oo = o.expand(B,N,3)` broadcast already in `composite`. |
| screen-space compaction of the mover pass | `screen_span`, `_tile_origin`, `_expand_tile` (fixed square tile ladder) | `_mover_idx` (tile ladder), `COVER_COMPACTION`, `SHADOW_COMPACTION` | a per-environment **rectangular** window (the humanoid's bbox is 0.12–0.25 of the frame; a square tile of the max extent wastes 2×) |
| shading: spotlight with cutoff/exponent, per-light shadow masking, `128·shininess` exponent | `shade` is **directional lights only**, one exponent | `_shade` + `_light_geometry` implement spot/attenuation/cutoff exactly as MuJoCo's fixed function, with `unlit` masking and `SPECULAR_SCALE=0.988` (a MuJoCo-calibrated constant that MJWarp does **not** apply) | a **MJWarp-law** shader: headlight as a directional light at the camera, shadow factor 0.3, no `SPECULAR_SCALE`, specular from `mat_specular`, round-to-nearest. The composition is the same; the constants differ, so it is a third small shader rather than a change to either existing one |
| skybox | none (the plate families bake it through MuJoCo) | none | cube-map sampler with MJWarp's face/axis convention and the `t` clamp — ~30 lines; or, better, a **baked per-(orientation, res) plate** computed once by that sampler (exact against MJWarp, 55–85 % of the frame becomes a copy) |
| textured plane | `tex_rgb` is one flat colour per primitive | plates hold the floor | a bilinear texture tap with MJWarp's PLANE UV law (§2.3) — the 300×300 checker as a `(300,300,3)` tensor and four `gather`s |
| CPU speed | `cast_nearest_torch` = a Python loop over K primitives × ~150–250 elementwise kernels over the **whole** ray set; measured on `node2339` at **210–225 ns per ray-primitive test** at B ≥ 16 and 336–532 ns at B = 1 (§4.1) | the parity cells pay 3 full-frame passes of it: 47–65 fps at 84 px | the fused CPU cast the raycast3d workstream is designing. **Shared file, shared fix**: a CPU-side fusion of `cast_nearest` belongs to `step6/raycast3d`; this design compacts rays before calling it so that it is not on the critical path. |

---

## 3. Characterisation — Fetch

Script: `scratch/step6/cpu-3d/census_fetch.py` on `node2339` in `se-s3b-fetch311` (mujoco 3.11.0,
gymnasium-robotics; MJWarp never constructed) → `characterize/census_fetch_node2339.json`,
`Fetch*_official_480.npy`.

### 3.1 Geometry, camera, lights

| | Reach | Push / Slide / PickAndPlace |
|---|---|---|
| visible geoms | 1 plane + 1 static box (table) + **15 meshes + 2 boxes (fingers) + 1 sphere (goal)** movers | + the block (box) / puck; see §3.2 |
| triangles, visible movers / static | **31,720 / 0** (all 15 meshes: `estop_link` 7,736, `laser_link` 7,116, `base_link` 4,720, `head_pan` 2,118, `gripper` 1,506, `shoulder_pan` 1,312, seven links of 900–1,000, `wrist_roll` 520, `torso_fixed` 116) | same robot |
| camera | fixed model camera `se:cam` (the official free camera's pose recovered from the two `mjvGLCamera` eyes), fovy 45, **mode 0 — never moves** | same |
| light | one **directional** light (type 1), `castshadow=False`, ambient .2 / diffuse .8 / specular .3; headlight ambient .1 / diffuse .4 / specular .5 | same |
| materials | one flat material per geom (the `_build_bundle` surgery), reflectance 0, no texture on any drawn geom | same |
| textures | skybox 32×192 (a 6-face gradient of 32² each); `texture_block` unused | same |
| `mj_kinematics` / `mj_forward` per state, `node2339` | 2.5 / 19.1 µs | same class |
| plate (floor + table + skybox = 91 % of the frame) | baked through `mujoco.Renderer` on the same host (llvmpipe): **0.375 s cold, 10.8 ms warm**; `plate == official` on **89.5 %** of pixels at state 0 (the other 10.5 % are the movers) | same |
| mover pixel share (official − plate ≠ 0) | **10.5 %** = 24.1 k pixels of 230.4 k | ~9–10 % |
| movers' screen bbox at state 0 | **89 % of the frame** (the robot spans base to head) | same |
| official renderer, same host | **45.7 fps** render-only, 38.6 fps from a stored state (`reference_render` incl. `mj_forward`) | see §3.2 |
| decode bar, same host, real frame | **815 fps/core** (JPEG-q90 = 19,750 B) | see §3.2 |

The Fetch scene is the easy half of what MJWarp does here (flat materials, no shadows, no textured
primitives, a fixed camera — the plate is exact and free) attached to the hard half (a 31.7 k-triangle
robot whose every joint moves). Because the light is directional with no shadow and the camera is
fixed, a mover pixel's colour depends only on its own mesh's pose — there is no cross-geometry term at
all; a CPU renderer's whole cost is *which mesh, which triangle, which normal* at 24 k pixels, and
the per-frame triangle count is what sets it.

### 3.2 The four scenes side by side (job 6345025, `node2339`, 1 thread, llvmpipe)

| scene | visible movers | mover triangles | mover px share | mover bbox share (state 0) | official fps render-only / from state | decode fps/core (JPEG bytes) | plate warm (ms) | `cores >= 1` needs | `S_cpu >= 5` needs |
|---|---|---:|---:|---:|---:|---:|---:|---:|---:|
| Reach | 15 meshes + 2 boxes + 1 sphere | 31,720 | 10.5 % | 89 % | 45.7 / 38.6 | 815 (19,750) | 10.8 | 815 fps | 229 |
| Push | + the block (3 boxes) | 31,720 | 10.5 % | 87 % | 44.1 / 37.6 | 776 (20,455) | 11.5 | 776 | 221 |
| Slide | + the puck (cylinder) | 31,720 | 8.3 % | 50 % | 39.6 / 32.0 | 881 (20,853) | 13.5 | 881 | 198 |
| PickAndPlace | + the block | 31,720 | 10.5 % | 87 % | 53.3 / 37.4 | 806 (20,204) | 10.7 | 806 | 267 |

The `Dense` and `Dict` siblings share each scene's model and frames, so these four rows are the
twelve configurations. Every scene carries the same 31,720-triangle robot; the object adds one
primitive; nothing structural changes between them, and the `cores >= 1` bar is 776-881 fps on
every one - **1.13-1.29 ms per frame for 31.7 k triangles, i.e. 36-41 ns per triangle including
fill**.

---

## 4. Cost model, and the constants it rests on

### 4.1 Measured constants, `node2339`, 1 thread (`characterize/cpu_micro_node2339.json`, `cost_floor_node2339.jsonl`)

| constant | value | how |
|---|---|---|
| torch dispatch floor (a 1-element op) | **2.0 µs** | `a*a` on a 1-element tensor |
| torch elementwise cost, N = 7,056 (one 84 px frame) | **6.3 µs/op** = 0.90 ns/elem (13 GB/s, in L2) | 10-op `x*b+c` chain |
| N = 112,896 (16 frames) | 107 µs/op = 0.95 ns/elem | |
| N = 451,584 (64 frames) | 1,009 µs/op = **2.24 ns/elem** (5.4 GB/s, out of L2) | |
| N = 230,400 (one 480 px frame) | 332 µs/op = 1.44 ns/elem | |
| `raycast3d.cast_nearest_torch`, one capsule, 84 px, B=1 / 16 / 64 | 3.75 / 27.3 / 106.7 ms → **532 / 242 / 236 ns per ray-primitive test** | synthetic trackcom-shaped scene |
| the same, K = 8 (cheetah) B=1 / 16 / 64 | 19.0 / 189.8 / 798.9 ms → 336 / 210 / 221 ns | |
| the same, K = 19 (humanoid) B=1 / 16 / 64 | 45.3 / 469.6 / 1,917.8 ms → 338 / 219 / 224 ns | |
| `any_blocking_torch` (shadow rays), same K, B | within 5 % of the cast | |
| one full-frame plane pass, B=64 | 25.6 ms = **0.40 ms/frame** | `intersect(PLANE)` |
| one full-frame 2-light `shade`, B=64 | 63.3 ms = **0.99 ms/frame** | `raycast3d.shade` (directional lights; a spot costs ~1.3× more) |
| compiled scalar C, DMC kernel (plane + skybox + checker + 2 lights + shadow + K capsules), 84 px | K=8: brute **258 ns/px** (548 fps), tiled **95 ns/px (1,484 fps)**; K=19: brute 476 (297 fps), tiled **150 ns/px (943 fps)** | `cost_floor dmc`, `-O3 -march=native`, no SIMD intrinsics, no fast-math |
| the same at 224 px, K=8 | brute 237 ns/px, tiled 87 ns/px (229 fps) | |
| compiled scalar C, triangle z-buffer raster + Gouraud + headlight, 480 px | **540–600 ns/tri at 3 px edges, 790–860 at 6 px, 1,270–1,640 at 12 px** (T = 10 k–100 k; e.g. T=30 k, 6 px: 25.0 ms/frame = 40 fps) | `cost_floor tri` |
| Apple M5 scaling check (not certification): DMC K=8 tiled 28.6 ns/px, tri 6 px 301 ns/tri | the Xeon 8268 is 2.8–3.3× slower per core than the M5 on this code | |

Two things these constants say before any architecture is priced:

* **Torch on CPU is bandwidth- and dispatch-bound, not compute-bound, and batching does not help.**
  Per element an op costs 0.9 ns while the working set fits in L2 (one to sixteen 84 px frames) and
  2.2 ns once it does not (64 frames). The "batch large to amortise dispatch" reflex of the GPU
  path is wrong here: on one core the best per-frame cost is at B ≈ 4–16 (dispatch amortised,
  still in cache), and it is **6.3 µs per full-frame op at 84 px** — so a per-pixel pipeline of `n`
  elementwise ops costs at least `6.3·n` µs per frame, and `cores_cpu ≥ 1` (151 µs) allows **n ≤ 24
  full-frame ops** including the texture gathers, the shading and the output pack. A spotlight +
  headlight Blinn-Phong alone is ~60 elementwise ops.
* **The unfused `raycast3d` CPU cast is ~220 ns per ray-primitive test**, which is 60–100 flops of
  arithmetic paid as ~200 kernels over materialised `(B,N)` temporaries (the raycast3d workstream's
  diagnosis, reproduced here on the trackcom shape). A full-frame cast of the humanoid's 19
  primitives is therefore 19 × 7,056 × 220 ns = 29 ms per frame — 30× over the `CPU_FASTER` budget
  by itself. Every viable torch design below compacts rays to the movers' window *before* casting.

### 4.2 Per-frame cost model at 84 px (DMC), one core

Notation: `N` = 7,056 pixels; `s` = sky share; `f` = floor share; `w` = movers' window share
(median bbox, §2.2); `c` = shadow-candidate share (floor pixels in the dilated window, §2.2); `K`
primitives; torch op cost `τ` = 6.3 µs per full-frame op at B ≤ 16 (in cache), scaled by the share of
pixels the op touches (compacted tensors are cheaper in proportion, down to the 2 µs dispatch floor).

| stage | ops | Cheetah (s .83, f .13, w .19, c .12, K 8) | Hopper (.85, .12, .07, .07, 6) | Walker (.55, .38, .28, .40, 7) | Humanoid (.00, .96, .12, .53, 19) |
|---|---|---|---|---|---|
| FK (`mj_kinematics + comPos + camlight`, Python loop) | — | 3 µs | 3 | 5 | 9 |
| skybox: **baked plate copy** (torch (d)) | 1 | 6 µs | 6 | 6 | 0 |
| floor: `t = −o_z/d_z`, hit point, PLANE UV, wrap, 4-tap bilinear gather, modulate | ~25 on `f·N` | 6.3·25·.13 = 20 µs (floor at dispatch floor ⇒ ~50) | 50 | 60 | 150 |
| floor + mover shading (headlight + spot, no shadow term yet) | ~60 on `(f+w)·N` | ~120 µs | 100 | 250 | 400 |
| mover cast, compacted to the window: `K` × ~45 ops on `w·N` | 45K | 8×45×2 µs (dispatch-bound) = 720 µs | 540 | 630 | 1,700 |
| shadow rays from floor + mover hits toward the spot, on `c·N`, `K` prims | 40K | 640 µs | 480 | 560 | 1,500 |
| composite, clamp, round, pack | ~8 | 50 µs | 50 | 50 | 50 |
| **total per frame** | | **≈ 1.6 ms → ~600 fps** | **≈ 1.2 ms → ~800 fps** | **≈ 1.6 ms → ~650 fps** | **≈ 3.8 ms → ~260 fps** |
| `S_cpu` (same host) / `cores_cpu` | | 25 / **0.09** | 30 / 0.12 | 25 / 0.10 | 24 / 0.04 |

The two primitive stages dominate and they are **dispatch-bound** (every op runs on 500–2,000 rays
per frame, where the 2 µs floor is most of the cost). Batching B = 16 frames of the same cell into one
compacted tensor takes those two stages to bandwidth-bound (0.9 ns/elem × 45K ops × w·N) — for cheetah
that is 8 × 45 × 1,300 × 0.9 ns = 0.42 ms, hardly better — so the model is robust to that choice:
**a vectorised-torch hybrid lands at 250–800 fps at 84 px, i.e. `CPU_FASTER` (S_cpu 24–30 against
llvmpipe) with `BELOW_G4` (cores 0.04–0.12).** The uncertainty is ±2× (the shade-op count and
the compaction efficiency); the conclusion — an order of magnitude short of `cores ≥ 1` — does not
move within it. Note that even the pure floor+sky+shade half (no movers at all) costs ~0.6 ms at
84 px in torch, i.e. the tier would be capped at `CPU_FASTER` *before a single capsule is drawn*.

Validation against a measured point: the raycast3d workstream's `DMCFingerSpin` CPU row (7 movers,
84 px, three full-frame torch passes over all K, no compaction) measures 47.8 fps at B=64 = 21 ms per
frame; this model with `w = c = 1` and K=7 gives 7×45×6.3 + 7×40×6.3 + 0.99 ms ≈ 25 ms. Same number,
same mechanism, and the 10–20× between it and the table above is exactly the compaction.

### 4.3 The compiled floor at 84 px

`cost_floor dmc` is the whole DMC frame in one scalar C loop (§4.1): **95 ns/px (K=8, 1,484 fps)
and 150 ns/px (K=19, 943 fps) with per-tile primitive lists and shadow rays restricted to the
movers' dilated window; 258 / 476 ns/px brute force.** Against the 21.4 ns/px that `cores_cpu ≥ 1`
demands on this host, the tiled scalar kernel is **4.5× short at K=8 and 7× short at K=19**. What is
left is SIMD: the plane/skybox/shade half is a per-pixel map (AVX-512 on the 8268 is 16 float lanes;
4–8× is the usual realised gain on such loops), the capsule tests inside a tile are also
lane-parallel across pixels, and the shadow loop is the branchy part (early-out on the first
occluder). A 4.5–7× gain from vectorisation is *plausible* and is *not demonstrated*; the honest
statement is that **`CPU_FAST` for the DMC four is reachable, if at all, only by a fused SIMD kernel,
and the go/no-go for that is a stand-alone micro-benchmark, not an integration**. For calibration,
the 3090 does these frames at 1.85 ns/px with 82 SMs; 21 ns/px on one core is 1/11 of that.

### 4.4 Per-frame cost model at 480 px (Fetch), one core

The plate is free and exact (10.8 ms to bake once per appearance, on the same llvmpipe the official
frame comes from — see the risk in §8.3). The mover pass has to produce the robot's 24 k pixels
from 31,720 triangles per frame (plus 2 boxes and a sphere):

| architecture | per-frame cost, measured floor | fps | vs `S_cpu ≥ 5` (229) | vs `cores ≥ 1` (815) |
|---|---|---|---|---|
| (b) CPU z-buffer raster of the 31.7 k triangles, scalar C | 31.7 k × 600–800 ns (the robot's triangles are 1–3 px at 480 px: 24 k covered pixels / 31.7 k tris) = **19–25 ms** | **40–50** | ✗ (≈ `CPU_PARITY`, ties llvmpipe at 46) | ✗ by 16–20× |
| (b) the same, with a well-vectorised binning rasteriser (3–5× over scalar; not measured) | 4–8 ms | 125–250 | borderline | ✗ by 3–6× |
| (a) analytic/BVH ray cast restricted to the movers' window — the bbox is **89 % of the frame** (the robot spans it), per-mesh bboxes ~10–15 % | 40–60 k rays × 300–500 ns BVH traversal (literature-class number, not measured here) = 12–30 ms | 30–80 | ✗ | ✗ by 10–25× |
| (g) Mesa llvmpipe itself as the mover rasteriser (render movers only over the plate) | the official renderer's own 21.9 ms/frame; hiding 91 % of the geometry saves little because llvmpipe's cost here is per-frame overhead + fill, not triangle count (its 84 px and 480 px DMC numbers are within 20 %) | ~45 | ✗ (`CPU_PARITY` by construction) | ✗ by 18× |
| (e) torch, vectorised over triangles | 31.7 k × ~40 ops over their bboxes ≥ 4 M element-ops per op-pass = 1.3 M elems × 40 × 1.4 ns ≈ 70 ms + gather/scatter | ~10–15 | ✗ | ✗ by 50× |
| decimated meshes (e.g. 4 k triangles) | changes the pixels; the parity contract (interior ≥ 0.99 vs official) is measured on the 31.7 k-triangle silhouette and shading, and MuJoCo shades **per vertex** — fewer vertices is a different picture | — | — | not admissible without a contract decision |

**`cores_cpu ≥ 1` at 480 px requires 1.23 ms per frame for 31.7 k triangles: 39 ns per triangle
including transform, setup, fill, z-test, Gouraud and pack, on one core.** That is below the
measured scalar floor by 14–40× and below what a state-of-the-art SIMD binning rasteriser achieves
on triangles this small (they are fill-and-setup bound at 1–3 px). This is the narrow blocker: **not
"a CPU renderer for Fetch is impossible" — one at `CPU_PARITY` is a week's work — but "`CPU_FAST`
for Fetch at its certified 480 px is not reachable on one core by any architecture in the list"**,
and the same arithmetic at 224 px (5.7 k covered pixels, decode ~2.5 k fps → 400 µs/frame, still
31.7 k triangles → 12.6 ns/tri) is worse, because the triangle count does not shrink with the frame.

---

## 5. Architecture comparison

Estimated per-frame cost on `node2339`, 1 thread, at the certified resolution, and fidelity against
the frozen contract. "vs MJWarp" is agreement with our own frozen frames (the digests); "vs official"
is the tier's declared bound.

| | architecture | DMC 84 px cost → fps → tier | Fetch 480 px cost → fps → tier | fidelity vs MJWarp frames | fidelity vs official (declared bound) | risk |
|---|---|---|---|---|---|---|
| (a) | **analytic raycast extension of `raycast3d`**: add skybox sampler, PLANE texture tap, MJWarp-law spot shader; full-frame torch cast of 1 plane + K prims; mesh via BVH for Fetch | 8–19 prims × 7,056 × 220 ns = **12–29 ms → 35–80 fps → `CPU_FASTER`/`PARITY`** (S 3–7) | needs a mesh intersector `raycast3d` does not have; BVH cast ~12–30 ms → 30–80 fps → `CPU_PARITY` | same arithmetic as MJWarp: expected > 99 % within 1/255, bit-exact fraction unknown (§2.3) | DMC: inside declared bounds (MJWarp's residual + ≤ 1 level) under the 3.11 reference; Fetch: interior 0.99 plausible | the unfused CPU cast; a shared-file change owned by raycast3d |
| (b) | **lightweight CPU mesh rasteriser** (z-buffer, torch/numpy) | n/a for capsules unless tessellated (28×16 sphere/capsule meshes — then MuJoCo's own silhouettes, not MJWarp's analytic ones: a *worse* match to the frozen frames) | scalar C 19–25 ms → 40–50 fps → `CPU_PARITY`; torch ~70 ms → 10–15 fps | Fetch: MJWarp ray-casts the same triangles; a rasteriser with the same per-pixel Blinn-Phong lands within a level; silhouette rule differs (edge-function vs ray-triangle) — 1-px silhouettes | Fetch: same class of residual as MJWarp's declared one (per-pixel vs Gouraud), interior 0.99 plausible | needs C to be even parity-class; no path to `CPU_FAST` (§4.4) |
| (c) | **hybrid: primitives by analytic cast, meshes by raster** | = (d) for DMC (no meshes) | = (b) for the arm + (a) for the block/goal | as (a)/(b) | as (a)/(b) | two code paths for one family |
| (d) | **pre-rendered static plate + dynamic overlays, in camera space**: baked skybox plate; analytic floor; compacted capsule cast; bounded shadow rays; torch | **1.2–3.8 ms → 260–800 fps → `CPU_FASTER` + `BELOW_G4`** (§4.2) | plate is already the design; the overlay is the mesh problem of (b) | skybox plate exact vs MJWarp by construction (99.5 %); rest as (a) | as (a) | ±2× on the estimate; cannot reach `cores ≥ 1` |
| (e) | **vectorised torch/numpy only, no compaction** | ≥ 12–29 ms as (a) | as (b) torch row | as (a) | as (a) | strictly dominated by (d) |
| (f) | **C++/SIMD fused per-pixel kernel** behind the same `render(states)` surface (torch C++ extension or a small shared library, CPU only) | scalar measured **0.67–1.06 ms → 943–1,484 fps**; with SIMD 4.5–7× needed for **`CPU_FAST`**, plausible/unproven | scalar raster 19–25 ms; SIMD binning 4–8 ms → `CPU_FASTER` at best; `CPU_FAST` needs 39 ns/tri: **no** | same laws in C: as (a), and easier to make bit-stable than torch (fixed op order, `-ffp-contract=off`, no fast-math) | as (a) | a compiled dependency for the package (build/CI policy); the raycast3d workstream's fused-CPU-cast question is the same question |
| (g) | **Mesa llvmpipe as the CPU renderer** (MuJoCo's own rasteriser; plate + mover-only render) | 24–27 fps → `CPU_PARITY` at best | ~45 fps → `CPU_PARITY` | it is the *official* rasteriser: exact vs official, **not** vs the MJWarp digests (the DMC four would be *better* than their declaration) | exact vs official (interior 1.0 class) | zero new renderer code; honours `device="cpu"`; no speed; not batched |

The ranking for the DMC four is (d) ≫ (a) ≈ (e) as torch designs — the compaction is worth 10–20×
and the skybox plate a further 1.5–3× on the three sky cells — and (f) is the only route past
`BELOW_G4`. For Fetch every row stops at `CPU_PARITY`/`CPU_FASTER`, and (g) is the cheapest honest
one.

---

## 6. Recommendation and the smallest Phase-2 plan

**Recommend (d) for the DMC four, implemented in torch on top of `raycast3d`'s intersectors, with
(f) as an explicitly gated second milestone; record Fetch `CPU_FAST` as a measured blocker and offer
(g) only if the coordinator wants `device="cpu"` honoured on Fetch at `CPU_PARITY`.**

### M1 — `DMControlTrackCom` gets a genuine CPU backend (target: `CPU_FASTER`, S_cpu ≥ 20 same-host)

Files:

- **new** `stable_envs/render/cpu3d.py` (env-agnostic, CPU-first, no CUDA import anywhere):
  - `SkyboxPlate.bake(m, cam_xmat, res)` — MJWarp's `sample_skybox` law + bilinear (with the 1/256
    weight quantisation as a switchable option) → `(H,W,3)` float, cached per (texture sha, xmat
    bytes, res). Exact-by-construction against MJWarp for the sky.
  - `plane_texture_tap(hit_local_xy, tex, texrepeat)` — the PLANE UV law of §2.3, four gathers.
  - `shade_mjwarp(...)` — the megakernel's composition with MJWarp's constants (headlight as a
    directional light at the camera, spot cutoff/exponent, shadow factor 0.3, `128·shininess`,
    round-to-nearest). Kept separate from `raycast3d.shade` (OGBench law) and `dmcontrol._shade`
    (MuJoCo law with `SPECULAR_SCALE`), because the three are three measured laws.
  - `render_trackcom(cam_origin (B,3), cam_xmat, dirs, plane, prims, poses (B,K,…), light_pos (B,3),
    sky_plate, tex)` — the per-frame pipeline: floor `t`, sky/floor select, mover window
    (`raycast3d.screen_span`, rectangular per env), compacted `raycast3d.cast_nearest_torch` on the
    window, shading, shadow rays via `raycast3d.any_blocking_torch` on the candidate set, composite.
    Uses only `raycast3d` functions that already run on CPU; adds nothing to `raycast3d.py`.
- `stable_envs/envs/dmcontrol.py`: `DMControlTrackCom.__init__` builds the MJWarp context only when
  `self.device.type == "cuda"`; `render` dispatches on device. The CUDA path is untouched byte for
  byte (so the four frozen `digest_ours` values are unaffected), the CPU path runs `_fk`-style
  `mj_kinematics + mj_comPos + mj_camlight` per state (3–10 µs) and `cpu3d.render_trackcom`.
  `device="cpu"` must **never** reach `_tc_context`.
- Tests: `tests/gpu/test_conformance.py` already parameterises over the registry; add a CPU
  construction/render smoke in `tests/unit` (no GL, no GPU), and a `tests/gpu` cross-backend diff
  (cuda MJWarp vs cpu cpu3d on the frozen states) that *reports* exact/within-1/mean and *asserts*
  whatever bound §8 decides.
- Replay obligation: the four DMC trackcom configs on **cuda** at B=64 against `digest_ours` (must
  be bit-identical: no CUDA code changes) — `examples/fidelity_step4.py` unchanged; plus a new CPU
  shard written under `examples/results/step6/cpu-3d/replay/` recording the CPU digests and the
  cross-backend agreement. Nothing else is affected: `raycast3d.py` is not edited and
  `DMControlSuite` is not touched, so the 10 parity cells and the 11 OGBench cells need no replay
  (if the coordinator prefers, the raycast3d-on-path set can be replayed as a no-op control).
- Profile: `scratch/step6/profile_note.py` on `node23xx`, `--device cpu --threads 1`, B = 1, 16, 64,
  256, for the four configs (the baseline harness has a `NameError` at its write step —
  `host = n["host"]` should be `note["host"]`; `step6/raycast3d` commit `01fb70c` fixes it on their
  branch, and this worktree will take the same one-line fix when the harness is first used here).
- Sweep: the Step-3 CPU protocol (`examples/bench_step3.py --device cpu --threads 1`) on a batch node
  for the same-host official and decode numbers.

Effort: 3–5 days implementation (most of it the shader-law diff loop against our frozen MJWarp
frames, which needs one short GPU job to regenerate frames only if the texture-fidelity stacks are
insufficient — they hold all 64 frozen states for all four cells, so probably none), 1–2 days
certification runs. Expected outcome: `CPU_FASTER` on all four, cores 0.04–0.12.

### M2 — fused SIMD kernel (target: `CPU_FAST`; gated)

Gate: a stand-alone C++ micro-benchmark of the `render_trackcom` pipeline (the `cost_floor` probe,
vectorised) must measure **≤ 21 ns/px at K=19 on the certification host** before any integration
work starts. If it does not, the DMC four stay `CPU_FASTER` and the note says why with the number.
If it does: a torch C++ extension (or `ctypes` shared library) `stable_envs/render/_cpu3d_kernel`
implementing exactly `render_trackcom` for planes + capsules + spheres, bit-compared against M1's
torch path (which becomes the readable reference, as `cast_nearest_torch` is for the Triton
kernel). Effort: 1–2 weeks plus the build/packaging decision (a compiled extension is new for this
package). The raycast3d workstream's own fused-CPU-cast decision is the same decision; one kernel
family should serve both, so this milestone should be scheduled with them, not in parallel.

### The shared fused CPU cast — what it would look like, and who owns it

Both this workstream and `step6/raycast3d` arrive at the same object from opposite ends: the
raycast3d finding (§2 there) measures the unfused CPU cast at 145-240 ns per ray-primitive test,
attributes 76-95 % of every multi-mover CPU render to it, and estimates that "an ideal fused, SIMD
CPU cast at one core is worth roughly 2,000 fps scalar / ~16,000 fps at 8-wide on the 4-primitive
cells (about 2.4 MFLOP per frame at 84 px) - enough for `DMCReacherEasy/Hard`, `Cartpole`, `Acrobot`,
`Pendulum`, not enough for `DMCBallInCupCatch` at K=36 (about 1,800 fps, cores about 0.27)". The
`cost_floor` probe here measures the scalar half of that estimate directly on the certification
host - 95-150 ns/px for a whole trackcom frame at K=8-19, i.e. 943-1,484 fps - so the two numbers
agree, and the SIMD factor (4.5-8x) is the shared unknown.

What one kernel would have to be, to serve both:

- **Interface = `raycast3d.cast_nearest` / `any_blocking`, unchanged.** `(kinds, sizes (B,K,3),
  pos (B,K,3), R (B,K,3,3), o (B,N,3), d (B,N,3)) -> (t, normal, index)` and `(..., lim, skip) -> blocked`.
  The Triton `_tri_cast` already has exactly this shape with a runtime K loop and one program per
  block of rays; the CPU kernel is the same loop with SIMD lanes across rays. Dispatch in
  `cast_nearest` becomes: CUDA -> Triton, CPU with the extension present -> fused, else the torch
  loop (the readable reference both must match).
- **Bit-identity discipline is the same one the Triton kernel needed**: no FMA contraction
  (`-ffp-contract=off`), IEEE division and square root, the torch loop's association order
  (`raycast3d.py`'s "the fused cast" block lists the four counts of float32 patterns that differ
  otherwise). With that discipline the CPU kernel and `cast_nearest_torch` can be asserted
  bit-identical on the frozen corpora, as `tests/gpu/test_fast_paths.py` asserts for Triton.
- **Two consumers, two ray sets.** `DMControlSuite` calls it on compacted mover windows and on
  shadow/reflection candidate sets; `cpu3d.render_trackcom` (M1) would call it on the mover window
  and the shadow candidates. Neither needs the plane in the primitive list on CPU (the plane is
  cheaper as the analytic `t = -o_z/d_z` pass), so the kernel's primitive set is spheres, capsules,
  cylinders, boxes - what `_tri_isect` has.
- **What it does *not* cover**: the per-pixel *shading* of the trackcom frame (floor texture tap,
  skybox, two-light Blinn-Phong), which is 60-80 % of this design's torch cost (§4.2) and is
  outside `raycast3d`'s cast interface. A fused cast alone takes the DMC four from ~260-800 fps to
  perhaps 400-1,200 fps; `CPU_FAST` needs the *whole* per-pixel pipeline fused (M2), which is a
  second, larger kernel that would naturally embed the first.

Ownership: **`stable_envs/render/raycast3d.py` and any fused CPU cast behind its interface belong to
`step6/raycast3d`** (it is their file, their measured bottleneck, and their reachable tier on
Pointmaze/Reacher/Cartpole); this workstream consumes it through the unchanged `cast_nearest`
signature and contributes the trackcom-shaped benchmark (`cost_floor dmc`, and M1's torch pipeline
as the reference). The M2 whole-frame kernel would be this workstream's, built on top of theirs, and
should be scheduled after theirs so there is one SIMD/bit-identity discipline and one build
decision, not two. That split is proposed, not decided - it is the coordinator's call, and it is
the same call as the "compiled extension in the package" policy question.

### Fetch — recorded blocker, optional (g)

- Record in `certification`-facing terms: `CPU_FAST` unreachable at 480 px (39 ns/tri required,
  540–1,640 measured scalar; the official same-host renderer is llvmpipe at 46 fps).
- If `device="cpu"` must be honoured for the twelve: (g) — `FetchBase.render` on a CPU device uses
  `mujoco.Renderer` (same host, same GL) for the mover pass over the cached plate, with the
  segmentation buffer for the composite exactly as today. ~1 week including the composite plumbing
  and the CPU replay; tier `CPU_PARITY` (S_cpu ≈ 1, cores ≈ 0.05); fidelity **better** than the
  frozen digests (it is the official rasteriser) and therefore *not* digest-matching — another
  instance of the §8 contract question.

---

## 7. Risks

1. **Pixel-exactness against the frozen MJWarp digests is not achievable by a different renderer,
   and should not be the bar - the digests are a soft oracle even for MJWarp.** MJWarp is not
   digest-stable against itself (texture-fidelity §2: 99.98 % exact run to run, "known 1-level
   noise"; the lightspec note: 2 px in 4.2 M move under a different inlining of the same
   arithmetic), and the reacher-correctness workstream found that **the four trackcom cells do not
   reproduce their frozen Step-4 `digest_ours` from the pre-Step-6 baseline itself** (job 6344904,
   RTX 3090, same driver, same mujoco 3.12.0 / mujoco_warp 3.12.0 / warp 1.17.0, a different venv
   and node: `mean_pixel` off by 0.16-0.61/255 and the distinct-colour count by ~1 % on all four,
   while all 11 parity cells reproduce their digests exactly). So the frozen trackcom digests
   describe *a* MJWarp run, not *the* MJWarp frame; the reference for a CPU path is "MJWarp's picture
   to within its own run-to-run and venv-to-venv spread", which the texture-fidelity frame stacks
   (`localize312/frames`, 64 frozen states per cell, rendered in `se-step5` on gpu2108/gpu2113) and
   the reacher-correctness `gpu/pixels_all_baseline.json` together bound at <= 1/255 on the affected
   pixels. A CPU reimplementation of the same laws in float32
   will differ by one level on some fraction of pixels from: CUDA's 8-bit-fraction bilinear weights,
   `powf` implementations, fast-math reassociation, and summation order. Expected: > 99 % of pixels
   within 1/255, bit-exact somewhere between 30 % and 90 % depending on how much of the CUDA
   filtering is emulated. §8 says what to certify instead.
2. **The floor's minification residual is inherited, deliberately.** MJWarp samples one bilinear
   tap of the base level; matching *our* frames means matching that, not the reference's mipmap.
   The visual-tier bounds were declared on that residual, so it is inside the contract.
3. **Reference version.** The DMC declarations now carry `reference_versions = {mujoco 3.11.0,
   mujoco_warp 3.11.0}` (texture-fidelity, commit `8682223`) and the certifier BLOCKS a judgement
   under 3.12. A CPU backend does not use `mujoco_warp` at all, so the declaration as written
   cannot be *the* declaration for it: either the CPU path is certified by cross-backend agreement
   against the MJWarp frames (no reference needed) or it needs its own `visual_fidelity` entry and
   `_AT` (measured under mujoco 3.11.0 + dm_control 1.0.44, in `se-vis311` + a CPU renderer).
4. **The same-host official denominator is llvmpipe.** `S_cpu` for these rows is 11–27 fps on a
   batch node against 355–391 on a GPU node's hardware EGL. The frozen spec says "same host", so the
   batch-node number is the certification one, and it makes `S_cpu ≥ 5` easy and `cores ≥ 1` the
   only real bar. Worth stating in the certification record so a later reader does not think the
   CPU path was 15× faster than MuJoCo's *hardware* renderer.
5. **Fetch plate on CPU nodes.** The plate is baked by `mujoco.Renderer`; on a batch node that is
   llvmpipe, whose rasterisation is not bit-identical to the GPU-node EGL bake behind the frozen
   Fetch digests — but it *is* identical to the official reference rendered on the same node, which
   is what parity is measured against. Parity would be host-consistent; the digest would not.
6. **`raycast3d.py` is a shared file.** M1 as specified adds nothing to it; if the implementation
   finds it needs a per-env rectangular window helper there rather than in `cpu3d.py`, that is a
   coordinator-assigned edit per AGENT_COMMON, not one this workstream makes.
7. **Batch/order invariance on CPU.** The compaction window is per environment and the cast is
   per ray; `cast_nearest_torch` is elementwise by construction (the `_mv` unrolling argument in
   `raycast3d`), so batch invariance holds as it does on the parity cells. Any torch reduction
   (`.sum(-1)`) in the new shader must be written unrolled for the same reason.
8. **Memory/threads.** All estimates are at 1 torch thread; a caller with 8 threads would see
   roughly linear gains on the bandwidth-bound stages and none on the dispatch-bound ones. The
   certification number is the 1-thread one and the design does not rely on threads.
9. **Per-row visibility, for the record.** The reacher-correctness workstream established that
   MJWarp cannot carry a per-environment visibility knob (its shader never reads alpha; zero-alpha
   elements are hidden by `prepare_model` at context creation and the graph is captured), hence
   `DMControlTrackCom.VISIBILITY_KNOB = False`. A CPU path built on `raycast3d` intersectors has no
   such limit - the mover list is per environment - and the four trackcom cells draw nothing per
   episode, so nothing here depends on it; it is noted because a CPU backend would be the first
   trackcom path that *could* honour such a knob, and offering it would make the two devices'
   variation spaces differ, which `check_fidelity_declaration` does not currently contemplate.
   Not proposed for M1.

---

## 8. Contract: what a CPU path would be certified under (needs a decision)

`certification.json` has exactly one cross-backend bound, `backend_agreement_exact_frac_min = 0.99`
(`tests/gpu/test_conformance.py::test_backend_agreement`), and it is the Scene2D Triton-vs-torch
check — two implementations of the *same* arithmetic that are bit-identical by design — and it is
`not_applicable` on all 16 affected configs ("env renders without a Scene2D"). It is the wrong bound
for MJWarp-vs-CPU (risk 1) and stretching it would fail a correct CPU renderer.

The options, in order of how much they change:

| option | what is asserted | inside the frozen contract? | who decides |
|---|---|---|---|
| A. **visual-tier bound, per backend**: the CPU path measures its own `mean_abs / within8 / ssim` against the official renderer under the declared 3.11 reference and declares them (a `visual_fidelity` entry keyed by device, or a second class) | the same claim the GPU path makes, for the CPU path | the *mechanism* exists (`check_visual_fidelity`, `reference_versions`); a per-device declaration does **not** — `visual_fidelity` is one dict per class | coordinator (schema) + Sami (a new declaration = a new number) |
| B. **cross-backend agreement bound, new**: `within1_frac ≥ x`, `mean_abs ≤ y` between the CPU frames and the frozen MJWarp frames on the frozen states, plus the existing self-consistency checks (determinism, batch invariance, order independence, chunk agreement) on the CPU device | "the CPU path draws MJWarp's picture to within a level" | **no** — a new bound in `bounds`, a new check in `conformance.py`; but it needs no reference renderer and no reference-version pin, and it is the only option that certifies against *our* frames as the coordinator instructed | coordinator |
| C. **parity tier for Fetch, unchanged**: interior ≥ 0.99 vs official on the same host | the same claim as today | yes, if the CPU mover pass reaches it (plausible for (b)/(g), certain for (g)) | none |
| D. digest equality with the frozen `digest_ours` | bit-identity with MJWarp | technically the strongest; **not achievable** (risk 1), and MJWarp does not achieve it against itself - the baseline tree already misses the four trackcom digests by 0.16-0.61/255 in `mean_pixel` on the same card (reacher-correctness, job 6344904) | — |

Recommendation: **B for the DMC four (primary, against our frames, as instructed) and A alongside it
(so the CPU path also has a stated distance from the official reference), C for Fetch.** The thresholds
for B should be set from a measurement in M1, not now: render the 64 frozen states through the CPU
path, diff against `frozen_ours`, and declare `within1_min` / `mean_abs_max` from the spread — the
same way every other bound in this repo was set. Any of A/B changes `certification.json`'s schema or
an env declaration and is therefore **outside what this workstream may do without approval**.

Semantic follow-up (per AGENT_COMMON): none for the GPU frames (unchanged); for a CPU path that
lands within a level of the MJWarp frames, the DINO tripwire on the CPU frames is the proportionate
check (the tripwire bound for the visual tier is 0.25 and the MJWarp frames sit far inside it); no
Gate-H rerun.

---

## 9. Effort and sequencing

| item | effort | depends on |
|---|---|---|
| M1 torch CPU path for the DMC four (`CPU_FASTER`) | 3–5 days + 1–2 days certification | approval of this note; §8 decision on the bound (can be measured first, declared after) |
| M2 SIMD kernel gate (micro-benchmark only) | 1–2 days | M1's pipeline definition |
| M2 kernel + integration (`CPU_FAST`, if the gate passes) | 1–2 weeks | gate; packaging decision; coordination with `step6/raycast3d` |
| Fetch (g) llvmpipe mover pass (`CPU_PARITY`, `device="cpu"` honoured) | ~1 week | whether the coordinator wants it at all |
| Fetch `CPU_FAST` | **blocked** (§4.4) | a change to the bar or the resolution — not a renderer task |

---

## 10. Artifacts, scripts, jobs

- `scratch/step6/cpu-3d/census_dmc.py`, `frames_census.py`, `cpu_micro.py`, `census_fetch.py`,
  `mjw_dump.py`, `cost_floor.c`, `characterize_cpu.sbatch`.
- `examples/results/step6/cpu-3d/census/` (DMC census JSON ×4, `frames_census.json`,
  `shadow_candidates.json`), `examples/results/step6/cpu-3d/characterize/` (`cpu_micro_node2339.json`,
  `cost_floor_node2339.jsonl`, `census_fetch_node2339.json`, `mujoco_warp_sampling_source.txt`,
  official frames `*_official_84.npy` / `*_official_480.npy`).
- SLURM: **6345025** (`se6-cpu-3d-char`, `node2339`, `batch`/`normal`, 4 CPUs, 1 thread). No GPU job
  was needed: the reference frames came from the texture-fidelity workstream's frozen-state stacks.
- Sibling findings read and not depended on: `step6/raycast3d` (CPU phase profile — its
  220 ns/ray-primitive number is reproduced here), `step6/fetch-gpu` (the GPU per-pixel numbers),
  `step6/texture-fidelity` (reference-version drift and the new provenance certifier),
  `step6/reacher-correctness` (edits `DMControlSuite` target visibility; does not touch the trackcom
  cells).
