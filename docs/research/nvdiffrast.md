# nvdiffrast as a MuJoCo backend: the two operations MJWarp does not have

**Verdict: the mechanism works and the pictures are right, and it is not parity.** A prefiltered,
mipmapped sampler takes Gymnasium's minified checker floor from **0.0001 to 0.66-0.81 of far-band
pixels bit-exact** — the number that ruled the MJWarp backend out for six locomotion domains — and
user-supplied UVs turn RoboCasa's textured boxes from one flat colour into a textured surface. On
OGBench, the gate this had to clear first, it reaches **0.9984-0.9997 of interior pixels bit-exact**.
But whole-frame exactness on the official locomotion scene is 0.46-0.64 and interior 0.54-0.86,
because three things this renderer does not do — shadows, transparency, and non-power-of-two
mipmapping — are each worth more than the residual it closed.

It is also **3.3-4.9× slower than MJWarp at 224 px and 8-30× slower at 64 px**, which contradicts the
52,481 fps in [`backend3d.md`](backend3d.md) §4 and does not contradict the measurement: that figure
was for a renderer with a two-line shader and the wrong colours. A renderer that reproduces MuJoCo's
lighting, textures and skybox costs what §5 below reports.

All numbers measured on this cluster: one NVIDIA H200 (sm_90, 150 GB, driver 590.48.01), via `srun`
on `open_research`, `MUJOCO_GL=egl`, one env per process. Code: `stable_envs/render/nvdiff.py` and
`scratch/nvdiff/`.

---

## 1. Install

nvdiffrast 0.4.0 builds a CUDA extension at `pip install` time and needs two things it does not find
on its own. `scratch/nvdiff/install_venv.sh` is the script; it was run from scratch and verified.

```bash
BASE=/mnt/vast/home/randall.balestriero/miniconda3/envs/test     # nvcc 13.0.88, torch 2.11.0+cu130
VENV=/mnt/vast/home/randall.balestriero/venvs/senvdiff
"$BASE/bin/python" -m venv --system-site-packages "$VENV"
INC="$BASE/lib/python3.11/site-packages/nvidia/cu13/include"     # torch's OWN bundled CUDA headers
CUDA_HOME="$BASE" CPATH="$INC" TORCH_CUDA_ARCH_LIST=9.0 \
  "$VENV/bin/pip" install --no-build-isolation \
  "git+https://github.com/NVlabs/nvdiffrast.git@v0.4.0"
```

* **`CUDA_HOME` needs a real `nvcc`**, and it must match torch's CUDA major version (13 here).
* **`CPATH` must point at torch's bundled CUDA headers.** nvdiffrast compiles against torch's
  extension headers, which include `<cusparse.h>`; the conda toolkit does not ship it and the build
  fails on a missing include that has nothing to do with nvdiffrast. This is the note
  [`backend3d.md`](backend3d.md) left behind, re-verified.
* `--system-site-packages` over the `test` conda env is deliberate: that env already carries the exact
  reference stack this repo's OGBench and locomotion numbers were taken under (mujoco 3.9.0,
  ogbench 1.2.1, gymnasium 1.2.3, torch 2.11.0+cu130), and a venv never writes into its base, so the
  shared env is untouched.

---

## 2. What had to be measured before anything worked

Four of these were wrong on the first pass and every one of them was silent.

### 2.1 The texgen law, which is not what MJWarp implements

MuJoCo generates texture coordinates for **every** primitive by ONE object-linear projection along
the geom's local z: `u` is a function of local x alone and `v` of local y alone, on every face of
every type. Not a per-face unwrap, not a cube map, and not dependent on `numquads` (slopes identical
at 1, 2, 4 and 8 — the tessellation hypothesis, refuted).

    u = +0.5 * texrepeat_x * x * scale_x + 0.5
    v = -0.5 * texrepeat_y * y * scale_y + 0.5
    scale = 1                 if texuniform         (repeat per unit length)
          = 1 / half_extent   otherwise             (repeat across the object)

`half_extent` is `geom_size` per axis for box, ellipsoid and plane; `(r, r, r)` for a sphere; and
`(r, r, ·)` for a **capsule and a cylinder — the radius on both texture axes, not the half-length**.

| case | measured slope | law |
|---|---:|---:|
| box (0.4, 0.3, 0.2), repeat 0.5 | +0.6241 / −0.8350 | +0.625 / −0.8333 |
| sphere r = 0.4, repeat 0.5 | +0.6209 / −0.6273 | ±0.625 |
| capsule r = 0.3, repeat 0.5 | +0.8289 / −0.8358 | ±0.8333 |
| planes, half-size 0.5 … 32 | `scale = size` to within 0.05 % | — |
| **OGBench's real 16 m floor** | **0.031256, offset 0.49988** | 0.031250, 0.5 |

`mujoco_warp`'s `sample_texture` implements `s = 0.5*repeat*x − 0.5` for planes with **no size term**,
which is right exactly when `texuniform` is true (Gymnasium's floor) and twice the slope when it is
false (OGBench's floor, `texrepeat="1"` on a 16 m plane).

> **The trap that cost an afternoon.** A slope read off a WRAPPED texture coordinate comes back at
> exactly **half** its true value whenever the unwrap misses a period, and half a slope is a
> plausible-looking number. Synthetic planes first read as `0.5*repeat/(2*size)` with offset 0.25 for
> precisely that reason, and it took the real OGBench floor — where the repeat is small enough that u
> never leaves [0,1] — to show `0.5*repeat/size` with offset 0.5. **Measure texgen with a repeat that
> cannot wrap**, and paint the texture with a LINEAR ramp rather than a wrapped index, because the
> average of a ramp is still the ramp and the average of a wrapped index is noise.
> (`scratch/nvdiff/probe_plane.py`, `probe_texgen{,2,3}.py`, `probe_ogfloor.py`.)

A CUBE texture is indexed by a **direction**, not by that uv pair: the offset that centres a 2D map
would tilt every lookup toward one corner. The direction is the local position, rotated into cube
space MuJoCo's way (`S=x, T=z, R=−y`, the same convention its skybox uses). Getting this right took
Hopper's body pixels from **mean 40.5 / 0.0000 exact to mean 12.7 / 0.1954 exact / 0.81 within 8**.

### 2.2 Clamp AFTER the texture, never before

    primary = mat_rgb * (A + sum_l D_l max(0, n·l)) + sum_l S_l * spec * max(0, n·h)^p     # WHITE, ADDED
    final   = clamp(primary * tex_rgb)

The primary colour routinely **exceeds 1** — on a floor lit head-on it is `A + D + specular ≈ 1.75` —
and MuJoCo multiplies the texture into that and clamps the product. Clamping the primary first caps
the floor at its raw texel value, costs ~65 % of an OGBench frame, and leaves every flat interior
pixel exact, so it reads like a texture bug and is a composition bug.

### 2.3 Round, do not truncate

Same rule `mjwarp.gl_rounding()` applies to MJWarp. Measured here on six OGBench cells, rounding vs
truncating: interior exactness **0.998-1.000 vs 0.949-0.999**, wall exactness **0.951-0.977 vs
0.886-0.943**. Rounding wins on every cell.

### 2.4 `d.cam_xpos[i]` is a VIEW

Appending `np.asarray(d.cam_xpos[i], np.float64)` in a loop gives every state the **last** state's
camera, because the next `mj_forward` overwrites the buffer. On a `track` camera that is a
plausible-looking picture of the right scene from the wrong place: frame mean **34.8/255 against
3.1**, floor mean 72.7 against 5.7, and the body pixels destroyed. It is the same class of silent
error as [`mujoco_locomotion.md`](mujoco_locomotion.md) §6.2's missing `com_pos`, and it produced no
warning of any kind.

### 2.5 nvdiffrast will not mipmap a non-power-of-two texture

`RuntimeError: Mip-map size error - cannot downsample an odd extent greater than 1`. It fires on
300×300 (OGBench's grid), 100×100 (Gymnasium's checker) and 127×762 (MuJoCo's `texgeom`), i.e. on
most textures MuJoCo actually ships, and **MuJoCo itself does not resample them**: a 300×300 texture
painted with its own texel index shows 297 of its 300 distinct rows in a magnified frame. So the base
level has to be resampled to a power of two, and that is a declared change to the picture.

Its cost is entirely texture-dependent and was measured both ways: on OGBench's per-texel gradient the
resample is worth **0.03-0.05/255 of floor mean** (`exact_tex` vs `full` below), and on a cube map it
is worth a lot — resampling the locomotion skybox 100→128 costs interior exactness **1.0000 → 0.8329**
and mean 0.04 → 0.42 over 53 % of the frame, for nothing, because a cube map is sampled with a plain
linear filter and never needs a pyramid. `prepare_texture` therefore resamples 2D textures only.

---

## 3. OGBench: the sanity gate

224 px, 8 states per cell, `offsamples=0`, one env per process. Reference is `mujoco.Renderer` at
OGBench's own top-down `custom_camera`. Scored per surface off MuJoCo's segmentation buffer.

| cell | **interior exact** | all exact | mean /255 | walls interior | floor mean / within 8 | skybox exact | agent exact (px) |
|---|---:|---:|---:|---:|---:|---:|---:|
| pointmaze-medium | **0.9997** | 0.354 | 2.23 | 0.9997 | 3.49 / 0.9995 | — | 0.160 (539) |
| pointmaze-large | **0.9984** | 0.567 | 1.63 | 0.9999 | 3.95 / 0.9998 | 0.980 | 0.170 (171) |
| antmaze-medium | **0.9997** | 0.354 | 2.30 | 0.9997 | 3.60 / 0.9988 | — | 0.246 (484) |
| antmaze-large | **0.9984** | 0.567 | 1.64 | 0.9999 | 3.97 / 0.9997 | 0.980 | 0.173 (133) |
| humanoidmaze-medium | **0.9997** | 0.354 | 2.19 | 0.9997 | 3.42 / 1.0000 | — | 0.126 (135) |
| humanoidmaze-large | **0.9987** | 0.569 | 1.62 | 0.9999 | 3.93 / 1.0000 | 1.000 | 0.054 (74) |

The gate is cleared: an independent rasterizer, driven only by `qpos`, agrees with MuJoCo on
**99.84-99.97 % of interior pixels**, against the 1.0000 the repo's analytic plate + ray-caster
reaches on the same cells. Untextured walls are 0.9997-0.9999 interior-exact and the skybox is
0.98-1.00 exact with interior 1.0, so geometry, camera, lighting composition and quantisation are all
right.

**The floor is the residual, and it is not the resample.** It is 0 % bit-exact at mean 3.4-4.0/255
with 99.9 % of pixels within 8, and using the exact 300×300 base instead of the 256×256 resample moves
that by 0.03-0.05. Recovering MuJoCo's shading scalar directly — the floor's material colour is
(1,1,1), so its value is `round(255·clamp(S·tex))` and sampling our own texture at the same pixels
inverts it — gives **S_ref median 1.6203 against ours 1.6458, a ratio of 0.9839**: our shading scalar
is 1.6 % too high. That is the same discrepancy [`ogbench.md`](ogbench.md) records from a completely
different direction (it fits the specular weight at 0.2505-0.2612 where the light table predicts
`(0.5 + 0.1)·0.5 = 0.30`), now confirmed by a second, independent measurement. Ablating each light's
specular on the reference shows why the light table over-predicts: zeroing the **model** light's
specular changes 8.0 % of pixels at mean 0.25/255, while zeroing the **headlight's** changes 91.9 % at
mean 15.9 — the second light's specular barely arrives.

**It is deliberately not corrected in the code.** A constant fitted on one scene's floor is exactly
the kind of number this repo keeps finding to be wrong later, and the locomotion control in §4 shows
the declared constants are interior-exact 0.95-1.00 on a scene where the specular weight is different
(`MatPlane` has `specular="1"`). The residual is declared instead.

---

## 4. MuJoCo locomotion: does mipmapping close the checker floor?

**Yes as a mechanism, no to bit-exactness.** 240 px, 8 states per env from the env's own reset
distribution plus a 0-60 step random-policy rollout, reference `mujoco.Renderer` at the model's
`track` camera. `mip` and `nomip` are the SAME pipeline with `filter_mode` switched, so the difference
between the two rows is the prefiltered sampler and nothing else.

### 4.1 The deciding number: the far tenth of the floor

| env | far-band exact, **mip** | far-band exact, **nomip** | MJWarp ([mujoco_locomotion.md](mujoco_locomotion.md) §6.4) | luminance std ours (mip) | reference |
|---|---:|---:|---:|---:|---:|
| HalfCheetah-v5 | **0.7734** | 0.0001 | 0.000069 | 4.69 | 0.00 |
| Hopper-v5 | **0.8094** | 0.0001 | 0.000028 | 3.11 | 0.00 |
| Walker2d-v5 | **0.6977** | 0.0001 | 0.000041 | 6.99 | 0.86 |
| Ant-v5 | **0.6622** | 0.0001 | 0.000018 | 9.01 | 1.86 |
| Humanoid-v5 | **0.8042** | 0.0000 | 0.000017 | 3.50 | 0.00 |
| Swimmer-v5 (not minified) | 0.4034 | 0.4027 | 0.2028 | 117.3 | 117.1 |

Without mipmapping this backend lands **exactly in MJWarp's regime** — 0.0000-0.0001 exact, luminance
spread 95 against a reference of 0 — which is the control that says the two are failing for the same
reason. With it, two thirds to four fifths of those pixels are bit-identical and the spread drops from
95 to 3-9. `mujoco_locomotion.md`'s "seventeen millionths" becomes 0.66-0.81.

It is not 1.0, and the gap is a filter difference rather than a missing operation: nvdiffrast's
trilinear pyramid and GL's are not the same pyramid, and the 100×100 checker has to be resampled to
128×128 to have one at all.

**Anisotropic filtering was the obvious next hypothesis and it is refuted.** The reference's near-band
luminance spread (95.11) sits BETWEEN isotropic trilinear (94.32) and unfiltered (95.34), which is
what anisotropy looks like, so a proper reconstruction was implemented — N taps along the major
footprint axis at the minor axis's level of detail (`BatchRenderer(aniso=N)`). It makes things worse:
far-band exactness 0.7569 → **0.2774 (N=4) / 0.2355 (N=16)**, and the near band does not move
(0.3835 → 0.3840). MuJoCo is doing plain isotropic trilinear. `aniso=1` is the default; the knob is
kept because a negative result is worth being able to re-run.

### 4.2 The whole frame, and the ladder that says what is left

Every rung modifies **both** renderers identically. Interior exact / all-pixel exact, `mip`:

| rung | HalfCheetah | Hopper | Walker2d | Ant | Humanoid | Swimmer |
|---|---|---|---|---|---|---|
| L0 the official scene | .603 / .643 | .857 / .615 | .804 / .635 | .616 / .637 | .862 / .620 | .537 / .464 |
| L1 `reflectance = 0` | .608 / .648 | .849 / .621 | .802 / .641 | .604 / .637 | .858 / .622 | .538 / .478 |
| L2 L1 + shadows off | .705 / .675 | .844 / .623 | .872 / .676 | .627 / .656 | .858 / .625 | .497 / .444 |
| **L3 L2 + floor texture removed** | **.951 / .920** | **1.000 / .950** | **.921 / .897** | **.953 / .878** | **1.000 / .955** | **1.000 / .957** |
| *MJWarp at the same L3* | .939 / .682 | .931 / .667 | .874 / .612 | .864 / .602 | .927 / .662 | 1.000 / .944 |

Read down the column. **L3 is the control and it is the important row**: with the floor texture gone
on both sides this renderer is interior-exact 0.92-1.00 and whole-frame exact 0.88-0.96, better than
MJWarp on five of six and far better on whole-frame exactness. So the camera (including `TRACKCOM`),
the tessellation, the shading composition, the constants and the quantisation are all right, and
everything between L0 and L3 is texture filtering plus shadows.

L0 → L2 splits what is left:

* **The mirror is worth nothing.** `MatPlane` declares `reflectance="0.5"` and zeroing it moves
  interior exactness by −0.012 to +0.005 — in both directions, i.e. noise. Same conclusion
  `mujoco_locomotion.md` §6.3 reached for MJWarp.
* **Shadows are worth a lot, on three of six.** Turning them off on the reference takes the NEAR band
  from 0.427 → 0.753 (HalfCheetah), 0.438 → 0.854 (Walker2d), 0.397 → 0.614 (Ant), and does nothing on
  Hopper, Humanoid or Swimmer. This renderer has no shadow pass; a shadow map is the single largest
  remaining term on the near floor and it is ordinary work rather than a limitation of the backend.
* The skybox is 52-55 % of the frame and reaches **0.947-0.958 exact with interior 1.0000** — one
  level of disagreement on a smooth gradient, not a wrong picture.
* Bodies are 2.5-10.3 % of the frame at 0.12-0.28 exact. On Ant, where they are 10.3 %, their interior
  exactness is **1.0000**.

**So blocker 2 is answered and blocker 1 is not what stands in the way any more.** What now caps
locomotion is shadows and a filter that agrees to within 3-9 levels of luminance instead of exactly.

---

## 5. Throughput and memory: 3-5x slower than MJWarp, and it OOMs first

Frames per second, and peak device memory, at one camera with textures, skybox and full Phong. Geom
frames are taken as given on device (producing them is MJWarp's or MuJoCo's job and costs 0.6 % of a
frame). Every row is `(fps, batch)`; the ratio against MJWarp is quoted at the batch where both
saturate.

**224 × 224**

| scene | B=64 | B=256 | B=1024 | B=4096 | peak at B=1024 | MJWarp at saturation | ratio |
|---|---:|---:|---:|---:|---:|---:|---:|
| antmaze-medium (8.5k verts) | 7,690 | 12,992 | 15,780 | **15,994** | 12.9 GB | 53,549 | **0.30×** |
| humanoidmaze-medium (11.8k) | 12,329 | 15,109 | 15,957 | **16,147** | 13.1 GB | 60,132 | **0.27×** |
| cube-single (579k verts) | 3,425 | **4,186** | *crash* | — | 13.6 GB at B=256 | 20,343 | **0.21×** |

**64 × 64**

| scene | B=256 | B=1024 | B=4096 | peak at B=4096 | MJWarp | ratio |
|---|---:|---:|---:|---:|---:|---:|
| antmaze-medium | 45,118 | 60,149 | **64,918** | 6.3 GB | 488,587 | **0.13×** |
| humanoidmaze-medium | 46,459 | 56,271 | **59,089** | 7.2 GB | 619,484 | **0.10×** |
| cube-single | 5,409 | *crash* | — | 11.7 GB at B=256 | 163,533 | **0.03×** |

**RoboCasa L52S03** — the case where "geometry is per world" bites hardest, at 1.73 M vertices after
de-indexing:

| B | 1 | 8 | 32 | 64 | 128 | 256 |
|---|---:|---:|---:|---:|---:|---:|
| fps, 224 px | 100 | 619 | 1,269 | 1,453 | **1,561** | *crash* |
| peak, 224 px | 3.1 GB | 3.9 GB | 7.2 GB | 11.7 GB | 20.6 GB | — |
| fps, 64 px | 154 | 829 | 1,511 | 1,663 | **1,772** | *crash* |
| peak, 64 px | 3.1 GB | 3.8 GB | 7.0 GB | 11.2 GB | 19.7 GB | — |

It saturates at **1,561 fps by B=128 and dies at 256**, against MJWarp's 18,126 fps at 4,096 worlds
on the same asset set (`robocasa.md` §7) — **0.086×**, on 1/32 of the batch. This backend does not
compete there and should not be asked to.

The two resolutions are the diagnosis: dropping from 224 px to 64 px is a **12× cut in pixels for a
1.14× gain in throughput and a 4.6 % cut in memory**, so a kitchen is bound by its 1.73 M vertices and
not by its fill. Primitive scenes behave the opposite way (antmaze gains 4.1× from the same change),
which is why one ratio against MJWarp does not describe this backend.

Its **setup** is also slow on that scene, and for a reason that is this module's rather than
nvdiffrast's: `Scene.__init__` walks 1,032 geoms in Python
and de-indexes 585 k triangles one mesh at a time, which takes minutes where `robocasa.md` §12 gets
MJWarp's whole context down to 1.34 s. Nothing about it is intrinsic — it is a loop that has never
needed to be fast — but it is what a caller would hit first.

Three things a caller needs from this table:

1. **The crossover is around B=256 and the ceiling is ~16k fps at 224 px**, on primitive scenes. It
   saturates by B=1024 and B=4096 buys 1.4 %.
2. **[`backend3d.md`](backend3d.md)'s 52,481 fps does not survive a real shader.** That benchmark
   rasterized the same geometry with a two-line headlight and admitted the colours were wrong; adding
   MuJoCo's lighting, texture sampling with derivatives, and a cube-mapped skybox background over the
   whole frame costs 3.3×. The honest number for a MuJoCo-matching nvdiffrast renderer is 16k, not 52k.
3. **It runs out of memory long before MJWarp does, and it does not do so politely.** nvdiffrast's
   instanced mode wants the full `(B, V, 4)` clip-space vertex buffer, so memory is linear in
   batch × scene vertices with no sharing across worlds: **12.6 MB per world at 224 px on
   antmaze-medium** (51.6 GB at B=4096 on a 150 GB card), 52 MB on `cube-single`, 140 MB on a
   RoboCasa kitchen. And it does not raise — `cube-single` at B=1024 and RoboCasa at B=256 both
   return **`CUDA error: an illegal memory access was encountered` and kill the interpreter**, so a
   sweep has to run one process per scene or it loses every row after the first that does not fit.
   `scratch/nvdiff/bench.py` writes each row to disk before the next attempt for that reason.

Usable batch, per resolution, on this 150 GB card: **primitive scenes (10 k verts) 4096 at either
resolution; mesh-heavy scenes (0.5 M verts) 256 at either resolution; a RoboCasa kitchen (1.7 M
verts) 128.** MJWarp does 4096 on all of them.

Marginal device memory per world, measured at 224 px: **12.59 MB** (antmaze, 8.5 k verts),
**12.80 MB** (humanoidmaze, 11.8 k), **52.03 MB** (cube-single, 579 k), **139.58 MB** (RoboCasa,
1.73 M). A two-term fit on the first three — **239 bytes per output pixel per world plus 69 bytes per
scene vertex per world** — predicts the fourth at 131.6 MB against 139.6 measured, i.e. to within
6 % on a scene 200× the vertex count it was fitted on. The first term is the rasterizer's own buffers
plus the interpolated attributes and dominates small scenes; the second dominates large ones. Divide
the card by that before choosing a batch, because the alternative is finding the limit by killing the
interpreter.

---

## 6. RoboCasa: do user-supplied UVs close the textured boxes?

**Yes, and on the best of three kitchens the textured boxes reach 0.976 of interior pixels
bit-exact.** Kitchens read from `scratch/robocasa/out/corpus` (expanded MJCF, absolute asset paths,
compiles standalone). Visible-geom census, which reproduces `robocasa.md` §7's count from the other
side:

| kitchen | geoms | visible | zero-alpha hidden | **BOX + texture** | MESH + texture | untextured BOX | textures |
|---|---:|---:|---:|---:|---:|---:|---:|
| L52S03 | 1,251 | 1,032 | 219 | **176** | 98 | 598 | 72 |
| L24S54 | 2,202 | 1,894 | 308 | **401** | 177 | 982 | 104 |
| L45S33 | 1,396 | 1,181 | 215 | **196** | 131 | 660 | 95 |

### 6.1 Transparency has to come out first, and finding that out is the finding

Scored as-is, the class that looks broken is **untextured BOX**: 0.000 exact at mean 52.8, which is
not a filtering residual and not something texgen could cause. Chasing it to individual pixels names
the geoms — `counter_corner_2_main_group_1_top_0` and `stack_2_right_group_1_1_left_door_g1`, both
with **alpha = 0.5**. MuJoCo draws transparent geoms in a second, blended pass that lands on top of
the opaque ones regardless of depth; this renderer depth-tests them like everything else, so MuJoCo
shows a translucent green placement box over a white counter and we show the white counter.
`robocasa.md` §7 counts five such geoms and treats them as a footnote; on this camera they are large
and near, and they were hiding the answer to the actual question.

Two ablations that came back **exactly zero**, recorded so they are not re-run: `mjRND_SHADOW` off
and `mat_reflectance = 0` on the reference change this kitchen's score in no digit. RoboCasa's
residual is transparency and texture filtering, not shadows and not the mirror.

### 6.2 With transparency removed from both sides

128 px, `robot0_agentview_left`, partially transparent geoms moved out of the rendered groups on both
sides. `uv` is this renderer with texgen on; `notex` is the same renderer with texture modulation off,
which is the closest available stand-in for MJWarp's "samples texel (0, 0)" behaviour.

| kitchen / class | px | `uv` mean | `uv` exact | `uv` within 8 | `uv` interior exact | `notex` mean | `notex` within 8 |
|---|---:|---:|---:|---:|---:|---:|---:|
| **L45S33 BOX + texture** | 16,438 | **0.86** | 0.378 | **0.999** | 0.564 | 71.1 | 0.183 |
| **L52S03 BOX + texture** | 9,003 | **2.11** | 0.398 | **0.998** | **0.976** | 18.6 | 0.021 |
| **L24S54 BOX + texture** | 19,058 | 8.03 | 0.086 | 0.571 | 0.000 | 58.2 | 0.002 |
| L45S33 MESH + texture | 9,654 | 7.57 | 0.487 | 0.785 | 0.589 | 107.0 | 0.003 |
| L52S03 MESH + texture | 16,985 | 2.88 | 0.358 | 0.919 | 0.626 | 111.1 | 0.000 |
| L24S54 MESH + texture | 6,911 | 92.9 | 0.186 | 0.436 | 0.098 | 153.7 | 0.000 |
| L45S33 MESH, untextured | 6,442 | 10.26 | 0.593 | 0.873 | **1.000** | 10.3 | 0.873 |
| L52S03 MESH, untextured | 6,552 | 10.72 | 0.598 | 0.873 | **1.000** | 10.7 | 0.873 |
| L24S54 MESH, untextured | 6,571 | 10.93 | 0.584 | 0.869 | **1.000** | 11.0 | 0.869 |
| L45S33 CYLINDER | 230 | 3.98 | 0.713 | 0.878 | 0.621 | 44.8 | 0.465 |
| L52S03 CYLINDER | 222 | 1.35 | 0.671 | 0.978 | 0.529 | 79.2 | 0.279 |
| **L45S33 whole frame** | 32,768 | **4.71** | 0.455 | **0.910** | 0.574 | 69.5 | 0.268 |
| **L52S03 whole frame** | 32,768 | **4.23** | 0.419 | **0.932** | 0.634 | 65.4 | 0.182 |
| **L24S54 whole frame** | 32,768 | 26.50 | 0.210 | 0.602 | 0.051 | 69.1 | 0.178 |

**The UVs are load-bearing and they produce a textured box.** Turning texture modulation off takes
`BOX + texture` from 99.9 / 99.8 / 57.1 % of pixels within 8 to **18.3 / 2.1 / 0.2 %**, so those
surfaces are genuinely textured rather than painted a lucky flat colour — which is what MJWarp cannot
do at all (`robocasa.md` §7: MuJoCo paints 167 distinct colours on a checkered box, MJWarp paints 2).
For scale, MJWarp's best on this class is mean 5.19 at 25.0 % exact; two of the three kitchens here
are **mean 0.86 and 2.11**, and L52S03's textured boxes are **97.6 % interior-exact**.

**The spread between kitchens is large** — the same 5× spread `robocasa.md` §11 reports for MJWarp.
Whole-frame mean is 4.71 / 4.23 / 26.50 and within-8 is 0.910 / 0.932 / 0.602, and the outlier is
driven by L24S54's textured meshes (mean 92.9 against 2.88 and 7.57 on the other two). Untextured
meshes are interior-exact **1.0000 on all three**, so neither the geometry nor the shading is what
separates them; the asset set is.

## 7. What this backend does not do

In the order they cost pixels on the scenes measured here.

1. **No shadows.** Worth 0.427 → 0.753 on the locomotion near floor (HalfCheetah), and nothing at all
   on Hopper/Humanoid/Swimmer or on a RoboCasa kitchen. A shadow map is ordinary work, not a
   limitation of nvdiffrast.
2. **No transparency.** Partially transparent geoms are depth-tested and drawn opaque instead of
   blended in a second pass. This is what caps RoboCasa. Zero-alpha geoms ARE handled — `Scene` drops
   them, the same pre-pass `mjwarp.prepare_model` performs.
3. **Non-power-of-two textures cannot be mipmapped**, so their base level is resampled (2D only; cube
   maps are left alone). Declared in `prepare_texture(...)["resampled"]` per texture.
4. **The trilinear pyramid is not GL's**, which is the 3-9 luminance-spread residual on the minified
   floor. Anisotropic filtering is implemented, measured, and refuted (§4.1).
5. **No reflections.** Measured worth: nothing, on every scene here.
6. **Silhouettes differ by ~1 px** because MuJoCo multisamples by default (`offsamples=4`); every
   number above is against `offsamples=0` for that reason. `BatchRenderer(supersample=N)` exists.
7. **A mesh is de-indexed to one vertex per triangle corner**, because MuJoCo indexes position, normal
   and texcoord with three separate per-face arrays and there is no way to carry that on shared
   vertices. It is exact and it triples the vertex count, which is part of why §5's memory is what it
   is.
8. **Heightfields and SDF geoms are skipped**, silently. No scene measured here has one.

---

## 8. Recommendation

**Do not replace `stable_envs/render/mjwarp.py`.** MJWarp is 3-5× faster at 224 px and 8-30× at 64, holds 4096 worlds on scenes
where this OOMs at 256, and is parity on flat-material scenes. This module is worth having for exactly
three things:

1. **The two blockers are now measured rather than assumed.** Minification is closeable —
   0.0001 → 0.66-0.81 on the far band — and primitive texgen is closeable, and neither needs upstream
   to move. If `mujoco_warp` ever gains mip-level selection and primitive UVs, §2.1's law and §4.1's
   numbers are what to check it against.
2. **OGBench is at 0.9984-0.9997 interior**, from a renderer that draws the whole scene every frame with
   no baked plate. The repo's plate + ray-caster is still better (1.0000) and much faster, but this is
   the first evidence that a general MuJoCo rasterizer reaches that band here.
3. **It is a `visual`-tier renderer for scenes the plate cannot serve.** Locomotion at L0 is
   whole-frame **mean 2.50-5.45/255 with 84.5-94.6 % of pixels within 8** — a picture that is right
   without being the official picture, which is exactly what `core/env.py`'s `visual` tier is for. No
   env is switched to it here: the tier's `visual_fidelity` contract wants `mean_abs`,
   `within8_frac` AND `ssim` measured under stated conditions, and `ssim` is not measured in this
   study. **That measurement was since taken and the six locomotion domains DID land**, with a shadow
   pass on top of what is measured here — [`mujoco_locomotion.md`](mujoco_locomotion.md) §7 has the
   declared numbers (SSIM 0.947-0.981), the shadow ablation, and two throughput findings that this
   section's benchmark could not see because it drove a shared camera rather than a tracking one. The
   two numbers this study does have, per env at 240 px over 8 states:

   | env | frame mean /255 | within 8 |
   |---|---:|---:|
   | HalfCheetah-v5 | 4.55 | 0.875 |
   | Hopper-v5 | 4.56 | 0.851 |
   | Walker2d-v5 | 4.00 | 0.878 |
   | Ant-v5 | 5.45 | 0.888 |
   | Humanoid-v5 | 4.63 | 0.845 |
   | Swimmer-v5 | 2.50 | 0.946 |

What would change the recommendation: a shadow pass (worth up to +0.33 on the near floor), and either
NPOT mipmapping in nvdiffrast or a way to reproduce GL's pyramid exactly. Neither is speculative work;
both are bounded. Throughput would still be a third of MJWarp's.

---

## Reproducing

```
venv:  /mnt/vast/home/randall.balestriero/venvs/senvdiff   (nvdiffrast 0.4.0, mujoco 3.9.0,
                                                            ogbench 1.2.1, gymnasium 1.2.3,
                                                            torch 2.11.0+cu130)
run:   srun --partition=open_research --account=open_research --qos=normal --gres=gpu:1 \
            --cpus-per-task=8 --mem=64G ... MUJOCO_GL=egl
logs:  /mnt/vast/home/randall.balestriero/stable_envs_logs/nvdiff/
```

One process per scene, for two independent reasons: two MuJoCo envs alive together can return
byte-identical frames from cameras that provably differ, and nvdiffrast answers "too big" by killing
the interpreter.

| script | what it answers |
|---|---|
| `scratch/nvdiff/install_venv.sh` | the install recipe, run from scratch |
| `scratch/nvdiff/probe_ogbench.py` | §3: the sanity gate, per surface, over four texture/quantisation variants |
| `scratch/nvdiff/probe_texgen{,2,3}.py` | §2.1: the texgen law, by inverting a painted texel index |
| `scratch/nvdiff/probe_plane.py` | §2.1: the plane scale, with a repeat small enough that it cannot wrap |
| `scratch/nvdiff/probe_ogfloor.py`, `probe_uvcheck.py` | §2.1 on the REAL floors, ours against MuJoCo's per pixel |
| `scratch/nvdiff/probe_npot.py` | §2.5: which texture sizes nvdiffrast will mipmap |
| `scratch/nvdiff/probe_loco.py`, `run_ladder.sh` | §4: the far band, the near band, and the four-rung ladder |
| `scratch/nvdiff/probe_spec.py`, `probe_S.py` | §3: which light's specular arrives, and MuJoCo's shading scalar |
| `scratch/nvdiff/probe_robocasa.py` | §6: per-geom-class residual on three kitchens |
| `scratch/nvdiff/bench.py`, `run_bench.sh` | §5: throughput and peak memory, one process per scene |
