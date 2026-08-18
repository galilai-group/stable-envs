# RoboCasa, side by side

The deliverable of this directory is a picture, not a number: one RoboCasa trajectory rendered twice
from the same `qpos` — MuJoCo's own rasterizer on the left, `mujoco_warp`'s batch raytracer on the
right, and 4× the absolute difference on the far right so the divergence is visible rather than
asserted.

| | |
|---|---|
| `media/sidebyside_L52S03.mp4` | 200 frames, 256 px, `robot0_agentview_left`, layout 52 / style 3 — a **typical** kitchen |
| `media/sidebyside_L45S33.mp4` | the same, layout 45 / style 33 — the **worst** of the four kitchens measured |
| `media/sidebyside_*.gif` | the same clip for viewers that will not run a `<video>` element |
| `media/contact_*.png` | six evenly spaced frames as a still: official / MJWarp / 4×\|diff\| |
| `media/dr_one_batch.png` | 12 worlds, **one render context, one graph launch**, camera and light randomised per world |
| `media/kitchen_grid.png` | 12 different compiled kitchens — **12 contexts, 12 launches**; MJWarp batches worlds of one model, not models |
| `media/manifest.json` | every number below, plus the ffprobe line for each video |

Both kitchens are shown on purpose. Registering only the good one would make the tier's numbers a
selection effect, and the spread between kitchens is a factor of five.

## What the video shows

The geometry, the camera, the robot, the objects and their motion agree. The artefact is one mechanism
and it is easy to name once you have seen it: **surfaces whose texture is painted on a BOX or a CYLINDER
come out one flat colour.** MJWarp's `sample_texture` derives UVs for `PLANE` and `MESH` only, so every
other geom type samples texel (0, 0). On L52S03 that is the marble backsplash and the sink basin; on
L45S33 it is the counter and the whole run of upper cabinet doors, which is why that kitchen is five
times worse. The robot — 18 textured *meshes* — is essentially exact.

## Measured

256 px, 200 trajectory states, one H200, mujoco 3.11.0 / mujoco-warp 3.11.0. Scored against the frames
robosuite itself produced, `env.sim.render` at geom group 1.

| kitchen | camera | mean \|diff\| | within 8/255 | SSIM | interior mean | interior exact |
|---|---|---|---|---|---|---|
| L52S03 | agentview_left | 6.60 | 82.8 % | 0.894 | 1.66 | 13.1 % |
| L52S03 | agentview_right | 5.75 | 84.4 % | 0.902 | 1.30 | 7.9 % |
| L52S03 | eye_in_hand | 6.36 | 84.6 % | 0.932 | 3.72 | 17.5 % |
| L45S33 | agentview_left | **32.50** | **35.6 %** | **0.796** | 45.45 | 4.7 % |
| L45S33 | agentview_right | 31.06 | 37.5 % | 0.799 | 46.01 | 2.8 % |
| L45S33 | eye_in_hand | 15.14 | 63.5 % | 0.849 | 24.30 | 12.0 % |

`interior exact` is the number every **parity** env in this registry declares at 1.0. It is reported
here because it is the honest measure of what the visual tier gave up, not as something to pass.

The registered classes declare slightly different numbers ([`calibrate_tier.py`](calibrate_tier.py),
copied into `FIDELITY` in `stable_envs/envs/robocasa.py`) because they are measured on a different
distribution: `sample_states` is this trajectory with every free joint's yaw randomised, which is a
state the env reaches the moment the gripper touches an object and which the recorded rollout barely
visits. It moves the room-scale cameras by ~0.05/255 and the **wrist** camera from 6.36 to 9.89, with a
spread of 4.55–9.89 over four draws — a camera 10 cm from an object is mostly showing that object.

The reference itself is measured, not assumed. A bare `mujoco.Renderer` restricted to geom group 1
reproduces robosuite's `env.sim.render` with **max abs diff 0** on the live model; recompiled from the
dumped MJCF it reaches 99.7 % of pixels exact at mean 0.004/255 under the same MuJoCo, and 99.4 % at
mean 0.041/255 under MuJoCo 3.11 — three orders of magnitude below the residual being measured, and the
reason a RoboCasa env can carry a reference renderer at all.

**A mitigation that does not pay, recorded because it was tried.** Replacing the texture on the geoms
MJWarp cannot UV with that texture's *mean* colour instead of its corner texel
(`stable_envs.render.mjwarp.bake_primitive_textures`) moves mean \|diff\| by 6.60 → 6.68 on L52S03 and
32.50 → 31.83 on L45S33, and SSIM down in both. RoboCasa's box materials are often near-uniform, so the
corner texel is already about right, and only 38 of ~137 textured materials are worn *exclusively* by
primitives — the other 99 are shared with meshes, which MJWarp textures correctly and a bake would
break. It is not the shipped path.

## Throughput and setup

A peak frames/s is not an answer, so throughput is quoted as a `(ratio, batch, resolution)` surface with
its crossovers. The official column is MuJoCo's own rasterizer at geom group 1, one process, warmed,
best of two passes. The decode column is one CPU core of PIL JPEG decode from memory
(`scratch/robocasa/bench_decode.py`), which is the *favourable* case for a stored dataset and therefore
the honest bar for regenerating one.

Frames/s, L52S03, `robot0_agentview_left`, one H200:

| resolution | official EGL | MJWarp b=1 | b=16 | b=64 | b=256 | 1 core JPEG decode |
|---|---|---|---|---|---|---|
| 128 px | 932 | 2,010 | 11,973 | 14,409 | **16,701** | 7,483 |
| 224 px | 855 | 1,929 | 6,028 | 6,489 | **6,948** | 6,762 |
| 256 px | 812 | 1,846 | 5,161 | 5,436 | **5,666** | 5,486 |

Two ratios, two crossovers, and they are not the same story:

* **Against the official renderer: (17.9×, batch 256) at 128 px, (7.0×, batch 256) at 256 px.** It is
  ahead already at batch 1 (2.2×) — there is no crossover to wait for — and saturates by batch 64,
  which is why `MJW_CHUNK = 64`.
* **Against reading a stored dataset: (2.2 cores, batch 256) at 128 px, (1.03 cores, batch 256) at
  256 px**, with the crossover at roughly **batch 8** at 128 px and **batch ~128** at 256 px. At 256 px
  one H200 replaces about one CPU core of decode, which is why this tier is a training-throughput
  argument and not a storage argument.

Resolution moves the second ratio harder than batch does, and in the direction intuition gets wrong: a
raytraced frame costs 4× as many rays at 256 px as at 128, while a JPEG costs only 1.4× as much to
decode.

**Setup, per compiled kitchen** (256 px, batch 256, warm):

| stage | seconds |
|---|---|
| MJCF compile | 0.81–1.25 |
| `put_model` | 0.75–1.41 |
| `put_data` | 0.87–1.24 |
| `create_render_context` | 0.56–0.58 |
| CUDA-graph capture | 0.03 |
| **total** | **3.0–3.5 s** (3.1–8.7 across the 12 kitchens in the grid, which vary 1.2k–2.3k geoms) |

Against the official renderer, which pays its own ~1.7 s of compile plus EGL context, MJWarp's *extra*
setup is ~1.8 s and it saves (1/812 − 1/5,666) s per frame, so a kitchen pays for itself after
**≈1,700 frames — about 9 episodes of 200 steps at one camera.** Against a stored dataset at 256 px it
essentially never does, because the steady-state rates are within 3 % of each other; at 128 px it takes
≈48,000 frames (~240 episodes).

Two things a live-generation user has to know about that setup number. It is **warm**: the first context
built for a given render configuration on a machine also pays a one-time Warp megakernel JIT of
**140–175 s**, cached in `WARP_CACHE_PATH` and never paid again. And `create_render_context` is 0.56 s
here rather than the 8.98 s upstream costs, because `stable_envs.render.mjwarp.create_render_context`
hoists `mujoco_warp`'s per-texture re-upload of the entire texture blob out of its loop — on a
109-texture kitchen that is 38.8 GB of host-to-device traffic to use 356 MB, and removing it is
bit-identical.

## Reproducing

```bash
sbatch --partition=open_research --account=open_research --qos=normal --gres=gpu:1 \
       --cpus-per-task=8 --mem=96G --time=03:00:00 \
       --wrap='bash examples/robocasa_visual/run.sh'
```

| stage | interpreter | what it does |
|---|---|---|
| `dump_trajectory.py` | `robocasa` conda env (mujoco 3.3.1) | rolls 200 steps, keeps `qpos` + the **official** frames + the expanded MJCF |
| `render_mjwarp.py` | `~/venvs/se3d` (mujoco 3.11 + mujoco-warp) | re-renders the same `qpos`, both bake variants, throughput, the two grids |
| `check_reference_bridge.py` | both | is a bare `mujoco.Renderer` at group 1 the official renderer, on both MuJoCos? |
| `make_video.py` | `robocasa` conda env (cv2 + ffmpeg) | composes the triptych, scores fidelity, encodes H.264 |
| `calibrate_tier.py` | `~/venvs/se-fetch` | measures the registered classes on their **own** state distribution, over 4 draws |

The two interpreters are not a convenience. `robocasa/__init__.py` asserts
`mujoco.__version__ == "3.3.1"` and `mujoco_warp` requires ≥ 3.11, so **no interpreter can hold both**.
The expanded MJCF carries absolute asset paths and compiles standalone, which is what lets the halves
stay separate.

## Two traps this pipeline hit, both silent

**The official frame is rendered from stale kinematics.** `mj_step` evaluates the forward dynamics at
the old `qpos` and then integrates, so on return `data.qpos` is the new state while `data.xpos` — what
`mjv_updateScene` reads — still describes the old one. Recording `qpos` and the frame at that moment
gives a pair no renderer can reproduce: re-rendering the recorded `qpos` through `mj_forward` on the
*same live model* disagrees on 7.7 % of pixels, 39.0 % on the wrist camera. `dump_trajectory.py` calls
`sim.forward()` first and then **asserts** the round trip is bit-exact before writing anything.

**`cv2.VideoWriter`'s `mp4v` fourcc is MPEG-4 Part 2**, which no browser plays in an HTML5 `<video>`
element — the page shows a blank player with controls, which reads as a broken build rather than as a
codec choice. Everything here is encoded through ffmpeg with
`-c:v libx264 -pix_fmt yuv420p -profile:v baseline -movflags +faststart`, and the stream is probed and
the probe recorded in `manifest.json`:
`Video: h264 (Constrained Baseline) (avc1 / 0x31637661), yuv420p`.

## The envs

The four classes this directory calibrates are in
[`stable_envs/envs/robocasa.py`](../../stable_envs/envs/robocasa.py) and are **visual tier** — see
[`docs/concepts/fidelity-tiers.md`](../../docs/concepts/fidelity-tiers.md) for what that licenses and,
more importantly, what it does not.
