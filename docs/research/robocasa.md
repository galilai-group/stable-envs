# RoboCasa / robosuite — feasibility for `stable-envs`

> **Fourth pass, 2026-08-10 — the NO-GO below is unchanged, and RoboCasa has landed in a NEW,
> SEPARATELY DECLARED tier.** Sections 1-16 answer "can RoboCasa join the parity tier", and the answer
> is still no, for the reasons and at the numbers recorded there. Section 17 asks a different question —
> is a *fast, self-consistent, non-exact* render of a RoboCasa kitchen worth having for **live training-data
> generation** — measures it, shows it, and lands four classes in
> `stable_envs/envs/robocasa.py` under the
> [visual tier](../concepts/fidelity-tiers.md). Nothing in Section 17 relaxes anything in Sections 1-16:
> `test_render_parity_with_official_renderer` still demands 1.0 of every parity env, and a visual-tier
> env is skipped by it and asserted by its own test against its own declared numbers.

**Verdict: NO-GO — on G4, measured on the render LOOP, with setup assumed away.** This is the third
pass, and the reason has moved again. It is now the one that does not depend on anything the
consumer of a dataset controls:

> Every environment in this repo — every one in the registry, including the DM Control family landed
> this week — declares `parity_interior_min = 1.0`, and none has ever used the relaxation mechanism. For RoboCasa exactly one renderer can meet that, MuJoCo's own
> EGL rasterizer, because it *is* the official renderer. With the model already compiled and
> resident, and stripped to only the geometry it draws (verified bit-exact), that renderer reaches
> **1,925 frames/s across 16 processes at 128 px against 5,137 f/s for one CPU core of JPEG decode —
> 0.37 cores** (§10). Regenerating a RoboCasa frame exactly is 2.7× more expensive than reading one
> off disk, at K = ∞, before a single second of setup.
>
> The fast alternative is MJWarp at **16,700–26,400 f/s — 3.2–5.1 cores** — and **6–23 % of interior
> pixels exact** against the official frame, worst-kitchen within-8 agreement **0.188** (§11).

**Exact costs more than reading; fast is not exact.** Both halves are properties of the render loop,
so no amount of bucketing, caching, model pooling or setup engineering moves either one.

Everything the two previous passes rested on has now been re-measured, and most of it was wrong:

| previous claim | status | corrected |
|---|---|---|
| MJWarp "fails to apply RoboCasa's materials", 7.6 % exact | **retracted** (2nd pass) | it never reads alpha; 154 invisible marker geoms painted opaque (§7) |
| G4 loses 34.7× end to end | **retracted** (2nd pass) | cold compile averaged in, `mj_forward` per frame, wrong geom groups (§5) |
| `ep_meta` reproduces only **94.95 %** of pixels, "re-draws the object instance" | **retracted** | **99.6 %**, and the entire residual is one robot-base geom. The object half replays exactly. The old number came from restoring state by INDEX across two compilations that order joints differently (§9) |
| per-episode setup is a fixed 10 s tax | **retracted** | **10.04 s → 1.41 s** with three fixes, all verified bit-identical; the big one is a one-line upstream bug — `create_render_context` re-uploads the *entire* texture blob once per texture (§12). Break-even K falls from 145 to **21** |
| K = 1 is what RoboCasa gives you | **half right** | K per *kitchen* is **12 / 240 / 2,500** in the three shipped corpora, well above break-even; K per *compiled model* is still 1, and that is a different claim (§8) |
| G3 is the deciding gate | **superseded** | G3 still fails, but G4 now fails *independently*, on the loop. That is the durable answer (§10) |
| *(new lead, tested and closed)* MJWarp's truncate-vs-round bug might close the fidelity gap | **falsified for this scene** | it is the biggest fidelity win on that backend (0.791 → 0.9988 on flat materials) and caps RoboCasa at **0.142–0.647** interior-exact, because only that fraction of the residual has a truncation signature (§11) |

All numbers measured on an H200 via SLURM, robocasa 1.0.1 / robosuite 1.5.2 / mujoco 3.3.1
(MJWarp probes: mujoco 3.11.0 / mujoco-warp 3.11.0 in `~/venvs/se3d`).
Probes: `scratch/robocasa/probe_*.py`. Raw JSON and frames in `scratch/robocasa/out/`.

Six measurement traps were hit and corrected across the three passes; all are recorded because each
would have produced a confident wrong answer:
- **`mujoco.Renderer` is not what RoboCasa shows.** It defaults to geom groups 0–2 and paints
  robosuite's 730 collision primitives over the visual meshes. RoboCasa renders group 1 only. The
  corrected baseline reproduces `env.sim.render` bit-identically (§3).
- **`mjRND_SHADOW`/`REFLECTION`/`SKYBOX` live on `MjvScene.flags`, not `MjvOption.flags`.** Setting
  them on the option object silently does nothing, and every ablation reads a perfect 0.0 — which
  looks exactly like "this feature is unused" (§6).
- **MJWarp ignores alpha.** Its shader reads `color[0..2]` and packs a constant `255`, so a geom
  MuJoCo draws as fully transparent is drawn opaque. On a scene that uses invisible marker geometry
  this reads as a catastrophic material failure and is nothing of the kind (§7).
- **`mujoco.Renderer`'s segmentation buffer is `(objid, objtype)`, not `(objtype, objid)`.** Reading
  it the other way round yields an empty mask for every geom, which looks like "nothing is visible".
- **The zero-alpha pre-pass has to cover SITES, not just geoms.** Sites are in MuJoCo's default
  visible groups and MJWarp draws them opaque for the same reason; each RoboCasa kitchen carries 7
  invisible ones. A geom-only pre-pass — which is what §7 originally described — is half the fix
  (§11). `stable_envs.render.mjwarp.prepare_model` does both.
- **Restoring a state across two compilations must be done by JOINT NAME.** `set_state_from_flattened`
  copies by index, and two compilations of the same scene need not order joints the same way, so a
  replay scored that way measures a mis-*posed* scene and reads as a mis-*built* one. This is what
  produced the retracted 94.95 % (§9), and it is the single most expensive mistake in this document's
  history.

---

## 1. Install status

Already installed; no install work was needed.

| item | value |
|---|---|
| conda env | `/mnt/vast/home/randall.balestriero/miniconda3/envs/robocasa` (Python 3.10.20) |
| robocasa | 1.0.1, editable from `/mnt/vast/home/randall.balestriero/robocasa_ws/robocasa` |
| robosuite | 1.5.2, editable from `/mnt/vast/home/randall.balestriero/robocasa_ws/robosuite` |
| mujoco | 3.3.1 |
| assets | 24.07 GB, already downloaded under `robocasa/models/assets` |
| not installed | NVISII, mimicgen, robosuite_models (all optional; EGL rendering works without them) |

The shared `test` env does **not** have robocasa/robosuite — use the `robocasa` env.

One portability trap: this robocasa version renamed the task classes. `PnPCounterToSink` (used by the
older `counterfactual-WM` scripts and by `stable_worldmodel/envs/robocasa.py`) no longer exists; it is
now `PickPlaceCounterToSink`. Anything ported from the prior work needs the new names.

## 2. What state is needed to reproduce a frame

### The pose half: exact, and small

With the compiled model held fixed, the frame is a pure function of the MuJoCo state, exactly:

```
restore sim.set_state_from_flattened(s0) after 20 steps away
  -> max pixel diff 0/255, mean 0.0     (probe_scene.py)
```

Not "visually identical" — bit-identical, every pixel, every channel. The pose state is small:

| quantity | value |
|---|---|
| `mjSTATE_FULLPHYSICS` | 180 float64 (time + nq 91 + nv 88) |
| `nq` / `nv` | 91 / 88 |
| mocap | 2 bodies (6 pos + 8 quat floats) |
| qpos as float32 | 364 bytes/step |

### The scene half: not a small id — this is the blocker

RoboCasa rebuilds the kitchen on **every** reset (`hard_reset=True`), and the rebuild changes the
model's *shape*, not just its contents:

| reset | nq | ngeom | nmesh | nmeshface | ntex | layout/style | XML |
|---|---|---|---|---|---|---|---|
| 0 | 133 | 2203 | 350 | 856,029 | 105 | 24 / 54 | 659 KB |
| 1 | 103 | 1492 | 297 | 728,968 | 85 | 32 / 54 | 469 KB |
| 2 | 103 | 1397 | 281 | 659,040 | 95 | 45 / 33 | 454 KB |
| 3 | 131 | 2341 | 408 | 688,578 | 101 | 29 / 28 | 690 KB |

Pinning layout **and** style does not fix this. Four resets at layout 52 / style 3 gave
ngeom 1201 / 1215 / 1231 / 1238, nmesh 189 / 201 / 219 / 226, and four *different* XMLs — RoboCasa
resamples which of the 3,430 object instances and 705 fixture variants get placed.

RoboCasa does ship a compact per-episode descriptor, `ep_meta` — 6,387 bytes JSON, 1,662 gzipped,
containing exactly the fields a "scene id + variation" would want (`layout_id`, `style_id`,
`object_cfgs`, `fixture_refs`, `gen_textures`, `cam_configs`). It is not sufficient:

| replay path | descriptor size | XML matches | pixels exact | mean \|diff\| | max \|diff\| |
|---|---|---|---|---|---|
| **A** `ep_meta` only | 6.4 KB (1.7 KB gz) | no | ~~94.95 %~~ → **99.6 %** (§9) | ~~2.79~~ → 0.66 | 208 / 255 |
| **B** `ep_meta` + full MJCF | 385 KB (31.7 KB gz) | no (ids renamed) | **99.90 %** | 0.0042 / 255 | 43 / 255 |

> **Superseded by §9 — both rows above are wrong, and the sentence that followed them is retracted.**
> Path A was scored after restoring state by INDEX across two compilations that do not order joints
> the same way, i.e. it scored a mis-*posed* scene rather than a mis-*built* one. Mapped by joint
> name, `ep_meta` alone reaches **99.6 % of pixels exact**, the object geom signature matches the
> reference exactly, and the whole residual is 64 pixels on one robot-base geom. The original text
> read: *"Path A produces a kitchen that looks right at a glance and has a different canister on the
> counter — it re-draws the object instance."* It does not; it draws the same canister.

This is corroborated by RoboCasa's own code rather than only by my measurement:
`robocasa/utils/robomimic/robomimic_env_wrapper.py::reset_to` applies `set_ep_meta(...)`, resets,
**and then** calls `reset_from_xml_string(state["model"])` with the full MJCF stored per demo
(`collect_demos.py` writes `ep_data_grp.attrs["model_file"] = xml_str`). The authors did not treat
the 6 KB descriptor as sufficient either.

**So the state decomposition is:** `frame = f(expanded_MJCF_385KB + 24GB_asset_library, qpos_91..133)`.
The per-step part is genuinely tiny. The per-episode part is a half-megabyte scene graph whose
decoder is a 24 GB mesh/texture library and a MuJoCo compile.

## 3. Scene complexity (real numbers from `MjModel`)

`PickPlaceCounterToSink`, layout 52 / style 3 — a *typical*, not a worst-case, kitchen:

| | in model | **actually rendered** |
|---|---|---|
| bodies / joints | 226 / 73 (3 free, 45 hinge, 25 slide) | — |
| `nq` / `nv` / `nu` | 91 / 88 / 13 | — |
| geoms | 1,251 (984 box, 214 mesh, 51 cylinder, 2 sphere) | **517** (group 1) |
| collision geoms hidden | — | 730 (group 0) |
| mesh geoms | 214 | **166** (165 unique meshes) |
| **triangles** | 554,430 | **523,772** |
| largest single mesh | 79,468 faces | — |
| textures | 76 (1 skybox, 72 2D, 3 cube), up to 2048² | **71** |
| **texture data** | 388.7 MB (129.6 Mpx) | **377.0 MB** |
| materials | 234 — 170 textured, 49 reflective (max 0.8), 224 specular | 230 |
| lights / cameras | **1** directional light (`castshadow=0`) + headlight / 6 cameras | — |
| MJCF | 385,031 chars | — |

The "actually rendered" column matters and is easy to get wrong. `robosuite/environments/base.py`
sets `vopt.geomgroup[0] = render_collision_mesh` (default `False`) and `vopt.geomgroup[1] =
render_visual_mesh` (default `True`), so an official RoboCasa observation draws **group 1 only**. A
bare `mujoco.Renderer` defaults to groups 0–2 and paints robosuite's 730 collision primitives over
the visual meshes, producing a flat garish frame that RoboCasa never shows. Restricting to group 1
reproduces `env.sim.render(...)` **bit-identically** (mean 0.0, max 0, 100.00 % pixels exact), which
is how the mask was verified rather than assumed.

Note the triangle count barely moves (554k → 524k): hiding the collision geoms removes two thirds of
the *geoms* but almost none of the *triangles*, because the hidden ones are boxes.

Across resets this reaches ngeom 2,341 and 856,029 mesh faces.

### Do distinct kitchens share assets? Yes — mostly

The per-scene figures above are misleading if read as additive, so the union was measured directly:
10 resampled kitchens (10 different layout/style pairs), hashing every raw `tex_data` and mesh
payload.

| | |
|---|---|
| mean texture data per scene | 361.0 MB |
| naive additive total for 10 scenes | 3,610 MB |
| **actual union over 10 scenes** | **486.8 MB** (161 unique textures) |
| sharing factor | **7.4×** |
| marginal texture cost per additional kitchen | **~35 MB** |
| union mesh data over 10 scenes | 62.5 MB (1,421 unique meshes) |

This is a genuine mark *against* the memory objection, and it is recorded as such: an asset pool
covering many kitchens is a few GB, not a few hundred. GPU memory is not what stops batching here —
heterogeneous geometry is (below).

Asset library backing it:

| | |
|---|---|
| total | **24.07 GB** |
| mesh files | 108,784 (16.79 GB) |
| texture files | 7,732 (7.23 GB) + 900 generative textures |
| MJCF files | 4,206 |
| object catalogue | 302 categories, **3,430 instances** |
| fixtures | 705 XMLs (69 cabinets, 50 fridges, 50 sinks, 50 microwaves, …) |
| layouts × styles | 60 × 60 = 3,600 |

## 4. Renderer and cost

MuJoCo 3.3.1's own OpenGL rasterizer through EGL — robosuite's `MjRenderContextOffscreen`
(`binding_utils.py`), or raw `mujoco.Renderer`. NVISII is not installed and is not on the path used
by any RoboCasa dataset script.

Throughput on one H200, single process:

| resolution | robosuite `sim.render` | raw `mujoco.Renderer` |
|---|---|---|
| 64² | 467 fps | — |
| 128² | **766 fps** | 483 fps |
| 224² | **710 fps** | 474 fps |
| 256² | 711 fps | 468 fps |
| 512² | 558 fps | — |

3 cameras @128²: 323 observation-sets/s.

Multi-process on one GPU, 224², pinned scene:

| processes | per-proc fps | aggregate fps |
|---|---|---|
| 1 | 469 | 469 |
| 2 | 437 | 875 |
| 4 | 294 | 1,177 |
| 8 | 183 | 1,467 |

GPU memory for one env + render context: **1,345 MB**.

**The decisive cost comparison — rendering is not the bottleneck:**

| stage | cost | per env step |
|---|---|---|
| physics step (20 Hz control = 25 × 2 ms `mj_step`) | **41.9 ms** | 96.7 % |
| render 224² | 1.41 ms | **3.3 %** |
| hard reset (rebuild + recompile model) | **8.8 s** (6.2–18.6 s observed) | once per *episode* |

A perfect, free renderer would speed up RoboCasa data generation by 3.3 %.

## 5. Regenerate-from-state vs. store-the-images (re-measured)

The thesis is a trade, so both sides are priced on the same clock. The first pass reported one
end-to-end number — 4.03 s to regenerate 500 frames against 0.116 s to decode them, 34.7× — and that
number was wrong in three ways, each of which inflated the regeneration side:

1. **It averaged a cold compile into the compile term.** The three measurements were 3.85 / 1.23 /
   1.02 s and their mean, 2.03 s, was reported. A warm compile is **0.97 s**; the 3.85 s is the
   first read of 288 asset files off NFS and is paid once per process, not once per episode.
2. **It ran `mj_forward` on every frame.** Regenerating from stored states needs no dynamics at all.
   `mj_kinematics` + `mj_camlight` is **0.029 ms**; `mj_forward` is **0.954 ms** — as expensive as
   the render itself. Half the old "render loop" was physics for a state already in hand.
3. **It rendered geom groups 0–2**, i.e. with robosuite's 730 collision primitives painted over the
   scene — a frame RoboCasa never produces, which also mis-sizes the JPEG baseline it is compared to.

Corrected, 500 steps at 128², one camera, group 1, `robot0_agentview_left` (`probe_g4_egl.py`):

| stage | mujoco 3.3.1 | mujoco 3.11.0 |
|---|---|---|
| MJCF compile, cold / warm | 3.90 s / **0.97 s** | 1.43 s / **0.97 s** |
| renderer init (EGL context + texture upload) | 0.83 s | 1.25 s |
| `mj_kinematics` + `mj_camlight`, per frame | 0.029 ms | 0.029 ms |
| `update_scene`, per frame | 0.046 ms | 0.046 ms |
| `render`, per frame | 0.903 ms | 1.176 ms |
| render loop, 500 frames | **0.452 s** (1,107 fps) | 0.450 s (1,111 fps) |
| **regenerate total, one episode** | **2.245 s** | 2.667 s |
| decode 500 stored JPEGs (q90), one core | **0.097 s** (5,137 fps) | 0.115 s (4,365 fps) |
| penalty, end to end | **23.1×** | 23.3× |
| penalty, render loop only | **4.6×** | 3.9× |

**80 % of the corrected cost is setup, not rendering** — which is the whole point, because setup is
what G3 says RoboCasa pays per episode.

### Can the compile be bought off with storage? No.

MuJoCo can dump a compiled model as a binary `.mjb` that loads without re-reading any asset. That
trades the compile for disk, and the trade is terrible:

| | |
|---|---|
| expanded MJCF | 385 KB (**31.7 KB** gzipped) |
| compiled `.mjb` | **528.6 MB** (128.9 MB gzipped) |
| `mj_loadModel` from `.mjb` | 0.744 s |
| XML compile, warm | 0.966 s |

0.22 s saved for 528 MB per episode, against a 3.51 MB JPEG episode. The escape hatch is closed by
arithmetic.

### On the GPU: MJWarp batch regeneration

MJWarp renders the unmodified kitchen at ~18,000 fps across thousands of worlds (§7), so the render
loop moves to the right side of the ledger. Setup does not. Measured with a CUDA-graph-captured
`kinematics → camlight → refit_bvh → render` and a device-side BGRA→RGB gather
(`probe_g4_mjwarp.py`, 128², one camera, group 1):

| batch | put_data | render context | **setup total** | fps (qpos → RGB on device) | 500 frames |
|---|---|---|---|---|---|
| 1 | 1.04 s | 4.35 s | 10.41 s | 2,582 | 0.194 s |
| 64 | 0.94 s | 4.06 s | **10.02 s** | 16,758 | 0.030 s |
| 256 | 1.02 s | 4.02 s | 10.04 s | 17,756 | **0.028 s** |
| 512 | 1.11 s | 4.15 s | 10.27 s | 17,935 | 0.028 s |
| 1,024 | 1.37 s | 4.39 s | 10.77 s | 18,057 | 0.028 s |
| 4,096 | 2.68 s | 5.33 s | 13.02 s | 18,139 | 0.028 s |

Setup is `compile 1.43 + put_model 3.56 + put_data + create_render_context + graph capture 0.03`.
GPU memory runs 3,272 MB at one world to 4,698 MB at 4,096 — the marginal cost of a world is
~0.35 MB, so batching is not what constrains this. Copying the batch back to pinned host memory
costs another 2 %.

So the per-episode cost is a straight amortization question, with `K` the number of episodes that
share one compiled kitchen:

```
episode_seconds(K) = 10.04 / K + 0.028          (MJWarp, batch 256)
episode_seconds     = 2.245                      (MuJoCo EGL, single world)
decode_seconds      = 0.097                      (one CPU core, PIL, q90)
```

| K (episodes per compiled kitchen) | 1 | 10 | 100 | **145** | 1,000 | ∞ |
|---|---|---|---|---|---|---|
| MJWarp seconds per episode | 10.05 | 1.03 | 0.129 | **0.097** | 0.038 | 0.028 |
| vs. one core of JPEG decode | 103× slower | 10.6× | 1.33× | **break-even** | 2.5× faster | **3.5× faster** |

Three readings of that table, in order of how much they matter:

- **The render loop wins.** 500 frames in **27.7 ms** against **97 ms** to decode them: batched GPU
  regeneration of RoboCasa frames is **3.5× cheaper than reading a stored dataset** on one core. The
  objection that "even a free renderer loses to `ls`" is false once the renderer is batched — it was
  an artifact of measuring MuJoCo's single-world EGL rasterizer.
- **The episode loses, and loses harder on the GPU.** At K = 1 — which is RoboCasa's actual regime,
  since every reset resamples the object set and its own replay code stores a fresh MJCF per demo —
  MJWarp costs **10.05 s per episode against EGL's 2.245 s**, because `put_model` + render-context
  construction is 7.6 s where an EGL context is 0.83 s. Moving to the GPU makes the wrong term worse.
- **Break-even needs ~145 episodes per compiled kitchen.** That is exactly the pinned-scene regime,
  and pinning the scene is what G3 rejects. G4 is a corollary of G3 here, not an independent gate.

> **Superseded by §§10–12.** Both bullets above are about *setup*, and setup turned out to be
> reducible by 7× (10.04 s → 1.41 s, break-even K 145 → 21), while real corpora bucket 12–2,500
> episodes per kitchen. Neither changes the verdict, because G4 fails on the **loop** once the
> parity requirement is applied: the only renderer that reproduces the official frame exactly runs
> at 1,925 f/s against 5,137 f/s for a core of decode. "G4 is a corollary of G3" is withdrawn —
> it is now the independent gate.

Storage goes the other way, and this part of the thesis does hold:

| | per 500-step episode |
|---|---|
| JPEG q90, 1 camera | 3.51 MB (2.65 MB for the group-1 frame) |
| qpos float32 | 0.182 MB |
| scene MJCF, gzipped | 0.032 MB |
| **ratio** | **16.4× smaller** (≈49× at the 3 cameras the datasets actually use, since state is camera-count-independent) |

## 6. What a from-scratch rasterizer would have to match

Good news, and it is the one genuinely encouraging finding: **RoboCasa's shading is simple.** Ablated
every render feature across 4 resampled kitchens × 6 cameras = 24 viewpoints, worst case over all,
then re-run against the verified group-1 baseline:

| feature disabled | mean \|diff\| | max | pixels changed |
|---|---|---|---|
| shadows | 0.0 | 0 | **0.00 %** |
| reflections | 0.0 | 0 | **0.00 %** |
| skybox | 0.0 | 0 | **0.00 %** |
| haze | 0.0 | 0 | **0.00 %** |
| **textures** | **61.2** | 209 | **82.0 %** |

Shadows are off because the single light has `castshadow = 0`; the 46–63 reflective materials never
land in any camera's view. The shading model is therefore textured Phong with one directional light
plus MuJoCo's headlight (ambient 0.1 / diffuse 0.4 / specular 0.5), 4× MSAA (`offsamples = 4`). That
is a tractable port — no shadow maps, no reflection passes, no global illumination.

Textures move 82 % of the pixels, which reads as "the simplified-shading escape hatch is closed"
(see `out/vis_no_texture.png`). Read carefully, it is weaker than that: **disabling** a texture drops
the surface to a fallback colour, which is not the same as **baking** it. §7 measures the bake —
every textured material replaced by the mean colour of its own texture — at mean **7.99/255** with
94.0 % of pixels within 32. So the honest statement is that textures carry 8/255 of *colour* detail
on a frame whose flat-colour layout is already 92 % right, not that a flat-shaded RoboCasa is a
different environment. That still does not reach this repo's parity bar, but the ablation number
alone overstates the gap by 7.7×.

## 7. MJWarp: the batched GPU renderer already exists — and it nearly matches, once alpha is handled

> **Read §11 with this section.** The "interior mean 4.96 / 89.4 % within 8" below excludes a
> 1-pixel band around silhouettes. Scored the way this repo's conformance suite defines *interior* —
> a pixel whose 3x3 neighbourhood in the reference is one flat colour — the same pipeline reaches
> **5.9-23.1 % of interior pixels exact** across four kitchens, against the 100 % every env in the
> registry declares. The section below is right about the alpha bug and optimistic about what
> follows from fixing it.

> **Standalone summary — this section is written to be liftable by the 3-D backend survey.**
>
> - `madrona_mjx` is **deprecated**, pointing users at MJWarp: *"High-throughput batch rendering is now
>   natively supported within the MuJoCo ecosystem via MJWarp… We recommend all users migrate to the
>   official MJX renderer."*
> - `pip install mujoco-warp==3.11.0` works standalone (isolated venv, no robosuite needed) and loads
>   an **unmodified** RoboCasa MJCF: 1,251 geoms, 213 meshes, 553,830 faces, 388.7 MB of textures.
> - **Throughput: 18,126 fps at 4,096 worlds, 128², one H200** — 12× MuJoCo's EGL rasterizer across 8
>   processes (1,467 fps aggregate). Marginal memory **~0.35–0.7 MB per world**: mesh BVHs and the
>   texture pool are shared across worlds, only the scene BVH is per-world. Context creation
>   4.0–5.3 s, plus 3.6 s for `put_model`.
> - **It does reproduce MuJoCo's appearance, once the model is made renderable.** The earlier result
>   (7.7 % exact, mean 77.8/255) was caused by **MJWarp ignoring alpha**: RoboCasa puts 154 invisible
>   marker geoms (`rgba="0 1 0 0"`) in geom group 1, and MJWarp paints them opaque green over ~40 % of
>   the frame. Excluding them takes the disagreement to **mean 7.47, 82.8 % of pixels within 8/255**,
>   and the best swept config reaches **interior mean 4.96 / 89.4 % within 8** on the agent view.
> - **The one real renderer gap is primitive texgen.** `sample_texture` derives UVs for `PLANE` and
>   `MESH` only; a textured `BOX`, `CYLINDER`, `SPHERE` or `CAPSULE` samples texel (0, 0) and comes out
>   one flat colour. Minimal repro: MuJoCo paints 167 distinct colours on a checkered box, MJWarp
>   paints **2**. Still true on `mujoco_warp` `main` as of 2026-08-10; 3.11.0 (2026-07-28) is the
>   latest release.
> - **Implication for a 3-D backend:** batched GPU rendering of MuJoCo scenes at thousands of worlds is
>   solved and fast, and heavily-textured asset sets are **not** the barrier they were recorded as.
>   Two model-side pre-passes — hide zero-alpha geoms, and re-express textured primitives as meshes
>   with texcoords — are what stand between MJWarp and this class of scene.
>
> Probes: `scratch/robocasa/probe_{mjwarp,mjwarp_calib,mjwarp_texdiag,texgen,green,alpha,parity}.py`;
> frames in `out/parity_*.png`, `out/alpha_sheet.png`, `out/texgen_sheet.png`.

Partway through this investigation the premise changed, so it was tested rather than argued about.

`madrona_mjx` — the batched MuJoCo renderer this project would otherwise have targeted — now carries
a **deprecation notice** pointing at MJWarp: *"High-throughput batch rendering is now natively
supported within the MuJoCo ecosystem via MJWarp… We recommend all users migrate to the official MJX
renderer."* MJWarp ships a first-party GPU batch **raytracer** with UV-mapped mesh textures, and its
mesh BVHs and texture pool are shared across all worlds (only the top-level scene BVH is per-world),
which is exactly the architecture this thesis wants.

Installed `mujoco-warp==3.11.0` into an isolated venv (`~/venvs/mjwarp`) and pointed it at the
**unmodified** RoboCasa kitchen dumped by the other probes — 1,251 geoms, 213 meshes, 553,830 faces,
388.7 MB of textures. It loads and renders.

**Throughput at 128², one camera, on one H200:**

| worlds | fps | GPU memory |
|---|---|---|
| 1 | 34 | 3,051 MB |
| 16 | 532 | 3,051 MB |
| 64 | 1,753 | 3,083 MB |
| 256 | 9,135 | 3,243 MB |
| 1,024 | 15,725 – 22,366 | 3,787 MB |
| 4,096 | **18,126** | — |

Context creation is 3.5–3.9 s. Marginal memory is **~0.7 MB per world** — assets really are shared,
confirming the §3 finding from the other direction. So batched state→frame rendering of a real
RoboCasa kitchen at thousands of envs is **not** hypothetical: it is ~18,000 fps today, **12× the
1,467 fps** MuJoCo's EGL rasterizer manages across 8 processes, and **3.5× a CPU core's JPEG decode**.

Those are eager launches. §5 re-measures the same sweep as a CUDA-graph-captured
`kinematics → camlight → refit_bvh → render` with the qpos upload and the BGRA→RGB gather included —
16,758 fps at 64 worlds rising to 18,139 at 4,096, so the curve saturates by batch 64 rather than
1,024, and the per-episode setup is broken out separately there.

**At first it rendered a different picture.** A 32-configuration sweep (skybox, ambient, specular,
background, geom groups) against the verified group-1 OpenGL reference reached only mean 77.8/255
and 7.7 % exact pixels, with MJWarp painting counters and cabinetry saturated speckled green. That
was recorded as a texture-path failure. It is not one.

### The green is invisible geometry, not missing textures

Three measurements narrow it down, and the third is decisive.

**The channel reading is not the cause.** MJWarp packs `(a<<24)|(r<<16)|(g<<8)|b`, so a byte view of
`rc.rgb_data` is BGRA and the original probe read it as BGR. Re-scored here through `mjw.get_rgb`,
which unpacks correctly, against the same pinned scene, qpos, camera and group-1 convention:

| reading | mean \|diff\| | pixels exact | within 8 |
|---|---|---|---|
| raw byte view (BGR) | 81.67 | 7.57 % | 29.0 % |
| channel-corrected / `get_rgb` | 79.34 | **7.60 %** | 42.0 % |

The swap is worth 2.9 % of the mean error. It is a real bug and it is not this bug — independently
confirming what `backend3d.md` found.

**Baking the textures away does not help either.** Replacing all 170 textured materials with the
mean colour of their texture costs MuJoCo almost nothing (mean 7.99/255, 94.0 % of pixels within 32
of the textured original — so textures are worth ~8/255 here, not the 61.2 the texture-*off*
ablation in §6 suggested, because that ablation falls back to a default colour rather than a bake).
MJWarp renders the baked, entirely texture-free kitchen just as green: mean 78.5 against the baked
MuJoCo reference. A flat-material scene is exactly the case `backend3d.md` measures at 98.9 % exact
on OGBench, so something structural was wrong.

**Forcing every colour in the model to one grey makes the two renderers agree** — mean **1.85**,
31.9 % exact, 95.1 % within 8. Geometry, camera, BVH, lighting and shading are all correct. What was
wrong is *which geoms get drawn*.

MJWarp's shader reads `color[0..2]` and writes a constant alpha:

```python
base_color = wp.vec3(color[0], color[1], color[2])       # render.py — color[3] never read
...
rgb_out[...] = pack_rgba_to_uint32(hit_color[0]*255, hit_color[1]*255, hit_color[2]*255, 255.0)
```

Alpha is never consulted, so a geom MuJoCo draws as fully transparent is drawn opaque. RoboCasa's
kitchen is full of them — placement regions, gripper axis markers, region interiors — all written as
`rgba="0 1 0 0"` and left in geom group 1:

| group-1 geoms | 517 |
|---|---|
| effective alpha 0 (invisible to MuJoCo) | **154** |
| effective alpha in (0, 1) | 5 (0.085, 0.7, 0.8×3) |
| effective alpha 1 | 358 |

Those 154 geoms are ~40 % of the frame in MJWarp's output and 0 % in MuJoCo's. Moving them into a
geom group the render context does not enable is a one-line model-side pre-pass:

```python
alpha = mjm.geom_rgba[:, 3].copy()
has_mat = mjm.geom_matid >= 0
alpha[has_mat] = mjm.mat_rgba[mjm.geom_matid[has_mat], 3]
mjm.geom_group[alpha <= 0] = 3        # 3 is not in enabled_geom_groups
```

| RoboCasa kitchen, 128², group 1 | mean | exact | within 8 | within 32 |
|---|---|---|---|---|
| as-is (the old result, channel-corrected) | 79.34 | 7.60 % | 42.0 % | 46.3 % |
| **zero-alpha geoms hidden** | **7.47** | **14.4 %** | **82.8 %** | **92.8 %** |
| also hiding the 5 partial-alpha geoms | 7.77 | 14.4 % | 82.7 % | 92.6 % |

A 10.6× reduction in mean error from excluding geometry that was never supposed to be visible.

### Best achievable parity, swept

Re-running the calibration sweep on the corrected model (shadows in the grid this time, which the
original omitted), scoring interior pixels separately from silhouettes as `backend3d.md` does for
every other scene:

| camera | best config | interior mean | interior exact | within 8 | within 32 |
|---|---|---|---|---|---|
| `robot0_agentview_left` | shadows, specular, skybox, ambient all on | **4.96** | 14.7 % | **89.4 %** | 96.3 % |
| `robot0_agentview_right` | same | 5.83 | 11.8 % | 85.7 % | 94.2 % |
| `robot0_eye_in_hand` | as above but specular off | 8.64 | 6.4 % | 66.5 % | 94.1 % |

Interior pixels are 84–89 % of the frame. Shadows *on* is best, matching the OGBench finding — the
original sweep pinned `use_shadows=False` and so never tested it.

That places RoboCasa in `backend3d.md`'s **"lightly-textured: close but not identical"** band, next
to `cube-single` (20.8 % exact, MAE 5.93) and `humanoidmaze` textured (MAE 4.64 at its best config)
— not in a failure category of its own. It is still not the ≥98 % bit-exact parity the flat-material
scenes reach, and this repo's parity tests demand that, but the gap is now ordinary rather than
catastrophic.

### The one real renderer gap: no texgen for primitives

`mujoco_warp/_src/render.py::sample_texture` derives UVs for exactly two geom types:

```python
if geom_type[geom_id] == GeomType.PLANE:  ...          # procedural texgen, matches render_gl3.c
if geom_type[geom_id] == GeomType.MESH:   ...          # barycentric interpolation of mesh_texcoord
u = uv[0]*tex_repeat[0] + offset[0]                    # uv is still (0,0) for every other type
```

For `BOX`, `CYLINDER`, `SPHERE`, `CAPSULE` and `ELLIPSOID` the UV stays `(0, 0)`, so the whole
surface samples the texture's corner texel. Minimal two-geom repro (`probe_texgen.py`):

| checkered box + grey plane, 128² | MuJoCo | MJWarp |
|---|---|---|
| distinct colours painted on the box | 167 | **2** |
| box pixels: mean \|diff\| / exact / within 8 | — | 50.30 / **0 %** / 7.7 % |
| plane pixels: mean \|diff\| / exact | — | 0.45 / **99.2 %** |

The untextured plane is essentially exact; the textured box is flat. Re-expressing the same box as
an explicit 12-triangle mesh with cube texcoords restores it — box-only mean **50.30 → 6.43**, exact
**0 % → 62.6 %**, 98 distinct colours instead of 2 — so this is fixable model-side without waiting
for upstream. One caveat, measured rather than assumed: a naive per-face `[0,1]²` UV assignment is
*not* MuJoCo's own box texgen (the mesh version differs from the primitive-box reference by mean
21.07), so a bake that must stay pixel-faithful has to replicate MuJoCo's cube mapping exactly.
That is unfinished work here.

On this particular kitchen the gap costs less than it sounds: textured boxes are 30.6 % of the frame
but only 21.2 % of the residual error (mean 5.19, 25.0 % exact), because most RoboCasa box materials
use near-uniform textures — `textures/flat/light_gray.png` and friends — for which the corner texel
happens to be right. Residual after the alpha fix, split by geom type:

| geom class | share of frame | mean \|diff\| | exact | share of total error |
|---|---|---|---|---|
| `MESH` + texture | 50.6 % | 9.04 | 2.2 % | 61.2 % |
| `BOX` + texture | 30.6 % | 5.19 | 25.0 % | 21.2 % |
| `MESH`, untextured | 18.2 % | 6.66 | 30.5 % | 16.2 % |
| `CYLINDER` | 0.1 % | 23.88 | 29.4 % | 0.3 % |

The dominant residual is textured *meshes*, where MJWarp does have a UV path — that is texture
filtering (MuJoCo mipmaps, MJWarp point-samples a linear filter) plus MSAA on silhouettes, the same
residual every textured scene in `backend3d.md` shows.

### Has MJWarp moved?

No. **3.11.0, released 2026-07-28, is the latest release on PyPI**, and `sample_texture` on
`google-deepmind/mujoco_warp` `main` still handles only `PLANE` and `MESH` as of 2026-08-10. Alpha
is still unread. The DeepMind in-repo `benchmarks/kitchen` is still the RoboCasa-derived scene with
textures *"replaced by uniform colors"* and mesh *"normals and UV coordinate information removed"* —
but that is now better read as the reason nobody upstream has hit the primitive-texgen gap than as
evidence the textured path is broken.

## 8. Episodes per kitchen in the corpora that actually exist

The K = 1 in §5 was measured on a *stream of resets*, which is one access pattern. A dataset is
written once and read many times and the reader picks the order, so the honest question is how many
episodes a real corpus puts in one bucket. RoboCasa answers it in its own code and its own paper.

A "kitchen scene" is a `(layout_id, style_id)` pair — `robocasa/models/scenes/scene_registry.py`
splits layouts 1–10 / 11–60 and styles the same way, so `split="pretrain"` is 50 × 50 = 2,500
scenes, and the target split is *paired* rather than crossed
(`env_utils.py:466`, `list(zip(range(1,11), range(1,11)))`) = exactly 10 scenes. The RoboCasa365
paper (arXiv 2603.04356, §3.2 and App. H.1) states the same numbers.

| shipped corpus | tasks | demos/task | kitchens | eps per kitchen **per task** | eps per kitchen **pooled** | download |
|---|---|---|---|---|---|---|
| pretrain, human | 300 | 100 | 2,500 | 0.04 | **12** | 300 tars, 52 MB – 632 MB each |
| pretrain, MimicGen | 60 | 10,000 | 2,500 | 4 | **240** | 60 tars, 2.7–4.4 GB each |
| target, human | 50 | 500 | **10** | 50 | **2,500** | 50 tars, **57.3 GB** total |
| *ablation* `mg_5x5` | 17 | 10,000 | 25 (5 layouts × 5 styles) | 400 | — | **registry only — 0 of 34 have a Box link** |
| *ablation* `mg_5x1` | 17 | 10,000 | 5 (5 layouts × 1 style) | 2,000 | — | **registry only** |

`KITCHEN_SCENES_5X5` / `_5X1` are real and are in `env_utils.py:48-54`; they exist for the paper's
"environment diversity ablation" (Table 8: zero-shot 29.6 / 39.6 / 44.7 at 5 / 25 / 2,500 scenes) and
their tars are not published.

**So the premise is right and the conclusion does not follow.** Bucketing by kitchen gives K = 12 to
2,500, and 240 and 2,500 are both above the 145 the old table needed. But "episodes per kitchen" and
"episodes per compiled model" are different quantities, and it is the second one that setup
amortizes over.

### The compiled model is still per-episode, even inside one bucket

24 resets at a pinned layout 52 / style 3 (`probe_bucket.py`):

| | |
|---|---|
| distinct expanded MJCFs | **24 / 24** |
| distinct object instance in each of the 3 slots | **24 / 24, 24 / 24, 24 / 24** |
| `nq` | 91 on all 24 |
| `ngeom` | 1,198 – 1,270 |
| geoms belonging to the KITCHEN | **1,185 on all 24** |
| geoms belonging to the 3 sampled OBJECTS | 13 – 85 |
| distinct `gen_textures` / `cam_configs` | 1 / 1 |
| distinct robot base pose | 24 / 24 |
| mean reset (rebuild + recompile) | 6.88 s (4.10 – 18.78) |

Every episode draws three fresh objects from the 3,430-instance catalogue, and `mj_saveModel` of the
result is a different model. That is corroborated by RoboCasa's own generators rather than only by
this measurement: MimicGen's `data_generator.py` calls `env.reset()` per generated demo (there is no
`reset_to(model=...)` in its loop), `collect_demos.py` does the same per human demo, and
`kitchen.py:591` re-draws `(layout_id, style_id)` from the RNG whenever `ep_meta` is absent.

### What *is* constant: the kitchen you can see

This is the finding that makes the rest of the section worth measuring. Splitting the group-1 geoms
by effective alpha, over **64 independent episodes** of one kitchen (`probe_union.py`, layout 52 /
style 3):

| | 16 episodes | 64 episodes |
|---|---|---|
| group-1 geoms | 507 | 507 |
| — VISIBLE (effective alpha > 0) | **357** | **357** |
| — invisible placement markers (`rgba="0 1 0 0"`) | 150 | 150 |
| distinct signatures of the whole kitchen | **16** | **64** |
| distinct signatures of the **visible** kitchen | **1** | **1** |

The signature covers each geom's name, type, group, mesh, size, pose, orientation and colour. So
RoboCasa does re-place fixtures on every `_load_model` — but everything it moves is invisible. **The
drawn kitchen is a constant of `(layout, style)`, and the entire per-episode delta in the picture is
three objects and the robot.** Nothing in either previous pass suggested this, and it is what makes
the next section a question rather than a non-starter.

## 9. `ep_meta` is a sufficient scene descriptor — the 94.95 % was a measurement artifact

`env_utils.create_obj` honours `cfg["info"]["mjcf_path"]` with a comment saying it is there "to do
deterministic playback", and `ObjCat.get_mjcf_kwargs` returns a fixed scale per category, so the
object half of `ep_meta` ought to replay exactly. It does. The earlier probe restored state with
`set_state_from_flattened`, i.e. **by index**, across two compilations that do not have to order
joints the same way — so it scored a mis-*posed* scene, not a mis-*built* one.

Re-measured with qpos mapped by JOINT NAME, `ep_meta` JSON round-tripped the way a corpus stores it
(`probe_epmeta.py`, three independent replays, 128 px):

| replay | pixels exact | mean \|diff\| | max |
|---|---|---|---|
| **`ep_meta` + qpos by name** | **99.62 % / 97.20 % / 99.61 %** | 0.66 / 2.02 / 0.66 | 204 |
| + mocap restored | identical to the above | — | — |
| + full `mjSTATE_FULLPHYSICS` | identical to the above | — | — |
| object geom signature vs reference | **matches** | — | — |
| kitchen geom signature vs reference | differs (the 150 invisible markers) | — | — |

Not 94.95 % — **99.6 %**. And the residual is not diffuse: segmenting the disagreeing pixels by geom
id gives **one geom, `mobilebase0_fixed_support`, 64 px = 0.39 % of the frame**. RoboCasa applies a
random base offset to the mobile robot at reset (`generate_random_robot_pos`, `env.rng.uniform`) on
top of the `init_robot_base_pos` anchor that `ep_meta` does store, and that offset is not stored —
which also explains why two replays of the *same* `ep_meta` differ from each other.

So the scene descriptor is **6.4 kB (1.7 kB gzipped) plus about five floats**, not a 385 kB MJCF.
Retract "path A produces a kitchen with a different canister on the counter" — it produces the same
canister, and a robot standing a centimetre to the left.

RoboCasa v1.0 ships both forms per episode anyway: `robocasa/utils/lerobot_utils.py` writes
`extras/episode_<id>/{ep_meta.json, model.xml.gz, states.npz}` alongside the LeRobot tree.

**This does not rescue G3.** The descriptor being compact was never the problem; the problem is that
decoding it runs a MuJoCo compile against a 24 GB library. But it does mean a bucketing pass can be
planned from metadata alone, without reading a single 385 kB XML.

## 10. The deciding number: the ceiling of the bit-exact path

Every environment in the registry declares `parity_interior_min = 1.0` — 100 % of interior pixels
bit-identical to the official renderer. The relaxation mechanism exists and, as of this week, no env
uses it, including the five DM Control domains that had every reason to.

For RoboCasa exactly one renderer can meet that bar by construction: MuJoCo's own EGL rasterizer,
because it *is* the renderer `env.sim.render` calls. So the question collapses to one measurement —
**with the model already compiled and resident, so that G3 and every setup cost are assumed away, is
MuJoCo's rasterizer faster than a CPU core decoding stored JPEGs?**

Measured at 128 px, group 1, on a real kitchen, with `mj_kinematics + mj_camlight + update_scene +
render` in the loop and no dynamics (`probe_exact_ceiling.py`). The model is first stripped to the
geometry it draws — 738 geoms and 85 assets deleted, every body's inertial written out explicitly so
it still compiles — and the stripped model is verified **pixel-identical (max abs diff 0)** before
its throughput is believed:

| processes | per-process fps | aggregate fps |
|---|---|---|
| 1 | 950 | 950 |
| 2 | 557 | 1,115 |
| 4 | 303 | 1,210 |
| 8 | 224 | 1,789 |
| 16 | 120 | **1,925** |

**1,925 f/s against 5,137 f/s for one CPU core of JPEG decode = 0.37 cores.** An entire H200
replaces 0.37 of one CPU core; regenerating the frame exactly costs **2.7× more than reading it**.

Three things this rules out, which is why it is the deciding measurement rather than another
data point:

- It is measured at **K = ∞**. No amount of bucketing, model caching, context pooling or setup
  engineering can improve it, because setup is not in it.
- It is measured across **processes**, so it does not depend on the GIL. MuJoCo 3.11 adds
  free-threading support and the obvious next idea is N threads without a GIL; the contention here
  is in the driver and the GL context, not in Python, and 16 processes already have no GIL between
  them. Free-threading can at best reach this same 1,925.
- It is measured on the **smallest model that draws the same picture**, and that buys nothing in the
  loop: 912 f/s on the full model, 799 in the same process on the stripped one (a second live EGL
  context costs more than the geometry saves), 950 on the stripped model in a fresh process. MuJoCo's
  rasterizer is not geometry-bound at 128 px — it is bound by per-frame scene construction and
  readback, neither of which batches. Stripping pays enormously in *setup* (§12) and not at all here.

## 11. And the fast path is not exact — measured against the official frame

MJWarp with the zero-alpha pre-pass reaches **16,690–26,400 f/s** at 128 px, batch 256, shadows on,
across kitchens (`probe_fastsetup.py`, `probe_setup_final.py`) — 3.2–5.1 cores of decode, and 22–34×
robosuite's own `sim.render`. Its agreement is scored against the frame RoboCasa itself produced,
saved by `dump_corpus.py` at collection time, on four different kitchens, with "interior" defined the way `stable_envs.testing.conformance` defines
it (a pixel whose 3×3 neighbourhood in the reference is one flat colour), best of an 8-config sweep.
The pre-pass is `stable_envs.render.mjwarp.prepare_model`, and the whole table reproduces to the
digit whether the effective alpha is resolved MuJoCo's way (material only when the geom carries the
compiler default rgba) or the cruder way, and whether or not the 7 invisible sites are hidden —
which is worth knowing, because it means none of the residual is a pre-pass mistake:

| kitchen | interior share of frame | interior exact | within 8 | within 32 | interior mean | all-pixel exact |
|---|---|---|---|---|---|---|
| L52 S03 | 19.8 % | 0.090 | **0.992** | 0.998 | 1.38 | 0.125 |
| L10 S59 | 21.2 % | 0.231 | 0.621 | 0.630 | 17.39 | 0.161 |
| L24 S54 | 16.2 % | 0.069 | 0.401 | 0.999 | 8.35 | 0.076 |
| L45 S33 | 24.6 % | 0.059 | 0.188 | 0.198 | 47.68 | 0.085 |

**6–23 % of interior pixels exact against a required 100 %, and the spread between kitchens is a
factor of 5** — one kitchen is genuinely close (within-8 0.992, mean 1.38) and another is not
(within-8 0.188, mean 47.7). Toggling `use_shadows` changes nothing at all, which is consistent with
RoboCasa's single directional light carrying `castshadow=0`; `enable_specular` is the only flag that
moves the score, and it moves it in opposite directions on different kitchens.

The spread has a named cause, and it is the §7 texgen gap rather than anything subtle.
`texture_gaps` counts the TEXTURED geoms whose UVs MJWarp cannot derive — everything that is not a
`PLANE` or a `MESH` samples texel (0, 0) and comes out one flat colour:

| kitchen | visible group-1 geoms | textured primitives MJWarp cannot UV | invisible geoms / sites hidden |
|---|---|---|---|
| L52 S03 | 357 | **178** | 170 / 7 |
| L45 S33 | — | **198** | 198 / 7 |
| L10 S59 | — | **295** | 273 / 7 |
| L24 S54 | — | **401** | 295 / 7 |

Between a third and half of everything drawn is a textured box or cylinder rendered as one colour,
and the kitchens with more of them score worse. §7 measured this as costing "only" 21.2 % of the
residual because RoboCasa's box materials are often near-uniform — that holds on average and not per
kitchen, which is exactly what the 5× spread is.

Worth recording separately: the pre-pass also hides **7 invisible SITES** per kitchen. Sites are in
MuJoCo's default visible groups and MJWarp draws them opaque for the same reason it draws geoms
opaque, so a geom-only alpha pre-pass — which is what this document's §7 described — is half the
fix. `stable_envs.render.mjwarp.prepare_model` does both.

### The truncate-vs-round fix does not rescue it — measured, not assumed

While this was being written another agent found the largest single fidelity bug on this backend:
**MJWarp truncates float→uint8 where the GL pipeline MuJoCo renders through rounds**, so MJWarp is
never above MuJoCo and is exactly one level below wherever the fractional part reaches 0.5. On Fetch
(flat materials) correcting it takes interior agreement from 0.791 to **0.9988**.

That is the most promising remaining lead for RoboCasa, so it was tested rather than hoped for. A
truncation-only residual has a signature that cannot be faked: `reference − mjwarp` is in `{0, +1}` on
every channel and **never negative**. Measuring that fraction bounds what a shader-side round could
possibly recover, without needing the fix:

| kitchen | interior exact today | interior mean | **residual consistent with truncation** | signed `ref − got` at 0 / +1 / ≤ −4 / ≥ +4 |
|---|---|---|---|---|
| L52 S03 | 0.090 | 1.38 | **0.647** | 0.230 / 0.149 / 0.160 / 0.162 |
| L24 S54 | 0.069 | 8.35 | **0.352** | 0.122 / 0.102 / 0.569 / 0.064 |
| L10 S59 | 0.231 | 17.39 | **0.284** | 0.177 / 0.087 / 0.282 / 0.272 |
| L45 S33 | 0.059 | 47.68 | **0.142** | 0.092 / 0.076 / 0.387 / 0.336 |

**A perfect round-instead-of-truncate leaves 35–86 % of interior pixels wrong.** The signed histograms
say why: they are two-sided and heavy-tailed — 16–57 % of pixels are 4 or more levels *below* the
reference and 6–34 % are 4 or more *above*, where a truncation residual is one-sided and bounded at
one level. For contrast, the Fetch measurement that motivated the fix is `{0: 86.8 %, ±1: 13.1 %}`
and nothing else above 0.01 %.

So the ranking of causes on this scene is settled: the texgen gap and texture filtering dominate, and
the rounding rule is a real bug that matters enormously on flat-material scenes and marginally here.
Even the kitchen it helps most (L52 S03, where the residual is already only 1.38/255) would cap at
0.65 interior-exact against a required 1.0.

This is a harsher number than §7's "interior mean 4.96 / 89.4 % within 8" and the difference is the
definition, not a regression: §7 excluded a 1-pixel band around silhouettes, which on a kitchen full
of mesh detail is a much weaker exclusion than "the reference is locally flat".

For calibration, another agent ran the same comparison on DM Control this week and got 19–31 % exact
on the side-camera domains and 0.5 % top-down, then abandoned MJWarp and built a plate + analytic
ray-cast renderer that reaches interior-exact 1.0. That route is not available here: DM Control's
movers are untextured primitives, RoboCasa's are ~100 textured mesh geoms, and §7's residual
breakdown already attributes 61 % of the remaining error to textured meshes — mipmapped GL texturing
against a point-sampled ray, which is precisely the part a hand-written mover pass would have to
reproduce exactly.

## 12. Setup cost: not a fixed tax, and 85 % of one term is an upstream one-line bug

Both previous passes amortized the ~10 s and neither asked what it was made of. It decomposes, and
three of the four terms come down. All numbers on one H200, batch 256, 128 px, and every variant is
checked **bit-identical** against the unmodified path before its timing is used.

**T1 — `create_render_context` re-uploads the entire texture blob once per texture.**
`mujoco_warp._src.render_util.create_warp_texture` builds one texture and does
`wp.array(mjm.tex_data, dtype=wp.uint8)` *inside itself*, but `mjm.tex_data` is MuJoCo's single
concatenated blob for every texture in the model. On a kitchen with 109 textures over 356 MB that is
**38.8 GB of host-to-device traffic to use 356 MB**. Hoisting the array out of the loop:

| stage of `create_render_context` | upstream | hoisted | + BVH pruned |
|---|---|---|---|
| texture upload | **7.600 s** | **0.106 s** | 0.109 s |
| mesh BVH build | 1.231 s | 1.138 s | **0.869 s** |
| everything else | 0.149 s | 0.092 s | 0.093 s |
| **total** | **8.980 s** | **1.336 s** | **1.072 s** |
| transient device memory | 2,338 MB | 168 MB | — |
| frames vs upstream | — | max diff **0** | max diff **0** |

**T2 — mesh BVHs are built for every mesh**, while the scene BVH only ever contains geoms in
`enabled_geom_groups`. 413 meshes built, 234 reachable from group 1; skipping the rest is the
1.23 → 0.87 s above.

**T3 — `put_model` and `put_data` are dominated by the collision half of the model**, which the
renderer never touches. `put_model` on a two-geom model is 0.012 s, so it is not fixed overhead; on
the full kitchen it is 3.70–6.07 s and on the same kitchen stripped to its drawn geometry
(1,251 → 513 geoms, 239 → 162 meshes) it is **0.048–0.053 s — 70–115× less**. `make_data` replaces
`put_data` at 0.98 s against 5.68 s, because `put_data` tiles a host `MjData` `nworld` times. The
stripped model renders **bit-identical** frames, so this is free.

Two of the four addendum angles do **not** pay, and are recorded as measured negatives:

- **Overlapping setup with rendering is worth 1.06×.** Building the next kitchen on a second thread
  while the current one renders: render alone 0.582 s, builder alone 7.937 s, wall 8.005 s against a
  serial 8.519 s. The builder's allocations and BVH launches contend for the same device.
- **A context pool works but does not scale to a corpus.** Four kitchens held live at once all
  render, the oldest unchanged, at 6.28 GB total (1.5–2.5 GB each) — so roughly 20 kitchens fit on
  an H200 at 128 px, against 2,500 in the pretrain split and 10 in the target split. It is the right
  structure for the target split and useless for the pretrain one.

`stable_envs.render.mjwarp.create_render_context` now carries T1 for every caller in this repo; T2
and T3 are model surgery and are left to the caller.

### The corrected setup, end to end

All three fixes applied at once, on one kitchen at 128 px / batch 256, against the same kitchen
stripped to its drawn geometry — and the two render **bit-identical frames** (`probe_setup_final.py`,
max abs diff 0), which is what makes the second column usable rather than a different scene:

| term | as recorded (§5) | full model, fixed | **stripped model, fixed** |
|---|---|---|---|
| MJCF compile, warm | 1.43 | 0.90 | **0.63** |
| `put_model` | 3.56 | 6.07 | **0.053** |
| `put_data` → `make_data` | ~1.0 | 1.05 | **0.181** |
| `create_render_context` | 4.02 | 1.43 | **0.513** |
| CUDA-graph capture | 0.03 | 0.036 | 0.029 |
| **setup per compiled model** | **10.04 s** | 9.53 s † | **1.41 s** |
| loop, 500 frames @ 16,690 f/s | 0.028 s | 0.030 s | 0.030 s |
| **break-even K vs one core of decode** | **145** | 142 | **21** |

† `put_model` is the noisy term: 3.70 s and 6.07 s on the *same* file in two runs, against 0.048 s
and 0.053 s for the stripped model in the same two runs. The stripped figure is stable because there
is almost nothing left to upload; treat the full-model row as 4-6 s.

One caution about reading the two rightmost columns against §5's 10.04 s: they are the same class of
kitchen (`PickPlaceCounterToSink`, layout 52 / style 3, 1,251 geoms, 72 textures) as the pinned scene
§5 used, while §12's 17.75 s baseline is a *larger* kitchen (1,995 geoms, 109 textures) where the
strip was not applied. Like for like, **10.04 s recorded → 1.41 s measured, and break-even K from 145
to 21.** Every shipped corpus clears 21 when bucketed by kitchen (§8: 12, 240, 2,500 — the human
pretrain split is the only marginal one).

And it does not matter. The ceiling this converges to is **3.2–5.1 cores** of decode
(16,690–26,400 f/s across kitchens) **at 6–23 % interior-exact**, or **0.37 cores at 100 %**. Setup
work moves you along the curve; it does not move the curve.

## 13. Can the compiled model be made a per-kitchen constant? Yes, and it still loses

§8 shows the drawn kitchen is constant, so the natural construction is a **union model**: the
kitchen, plus every object body any episode in the bucket uses, with the ones an episode does not
need parked below the floor. Parking is a free-joint qpos write, which is exactly the batched
interface this repo wants. Built for real out of the corpus MJCFs, sharing byte-identical assets
(`probe_union.py`, one kitchen, all 64 of its episodes):

| episodes folded in | ngeom | nmesh | ntex | nq | texture data | **compile** | compile / episode |
|---|---|---|---|---|---|---|---|
| 1 | 1,201 | 189 | 72 | 91 | 361 MB | 0.90 s | — |
| 4 | 1,390 | 369 | 80 | 154 | 393 MB | 0.93 s | 0.23 s |
| 8 | 1,540 | 504 | 91 | 238 | 424 MB | 1.56 s | 0.20 s |
| 12 | 1,620 | 571 | 103 | 322 | 455 MB | 4.72 s | 0.39 s |
| 16 | 1,825 | 760 | 114 | 406 | 490 MB | 6.00 s | 0.38 s |
| 20 | 2,013 | 920 | 122 | 490 | 505 MB | 6.52 s | 0.33 s |
| 24 | 2,193 | 1,078 | 133 | 574 | 535 MB | 8.75 s | 0.36 s |
| 28 | 2,363 | 1,199 | 139 | 658 | 552 MB | 9.88 s | 0.35 s |
| 32 | 2,539 | 1,310 | 151 | 742 | 584 MB | 10.50 s | 0.33 s |
| 36 | 2,739 | 1,498 | 162 | 826 | 612 MB | 13.93 s | 0.39 s |
| 40 | 2,934 | 1,681 | 173 | 910 | 639 MB | 16.22 s | 0.41 s |
| 44 | 3,183 | 1,885 | 184 | 994 | 674 MB | 19.91 s | 0.45 s |
| 48 | 3,313 | 1,990 | 196 | 1,078 | 705 MB | 21.14 s | 0.44 s |
| 52 | 3,433 | 2,067 | 207 | 1,162 | 732 MB | 24.07 s | 0.46 s |
| 56 | 3,644 | 2,201 | 215 | 1,246 | 750 MB | 24.84 s | 0.44 s |
| 60 | 3,847 | 2,388 | 228 | 1,330 | 783 MB | 27.01 s | 0.45 s |
| 64 | 4,010 | 2,535 | 239 | 1,414 | 816 MB | **29.95 s** | **0.47 s** |

The construction is sound: the union renders its base episode **bit-identically** (max abs diff 0)
with the other 45 objects parked at z = −50 (measured on the 16-episode union; the 64-episode
build was priced but its per-episode render pass was cut for time). The other 15 episodes come out
at 91.4–97.1 % exact,
and the residual is the same `mobilebase0_fixed_support` mechanism as §9 — RoboCasa bakes the robot
base pose into the *model*, so a union built on one episode's kitchen cannot express another's.
Fixable by bucketing on the base pose too, or by exposing it as a joint.

It does not matter, and the reason is the last column. **Compile time is LINEAR in the bucket**, at a
marginal cost that settles between **0.33 and 0.47 s per episode folded in** — against **0.097 s to
decode the whole 500-frame episode**. So `setup(K)/K` does not converge toward zero; it converges
toward ~0.45 s, which is **4.6× what the union is trying to save**, and it drifts up rather than down.
Every count in that table scales the same way (ngeom 1,201 → 4,010, nmesh 189 → 2,535, nq 91 → 1,414
over 64 episodes) because each episode contributes three objects nothing else uses.

The compiled-model cache that §5 dismissed at 528 MB *per episode* does not come back either: the
64-episode union's `.mjb` is **2,285 MB**, so caching one per kitchen costs 2.3 GB for 64 episodes,
against 3.51 MB of JPEG per episode.

The asymptote is worse still: with `obj_groups="all"`, every slot can draw any of the 3,430 catalogue
instances, so a bucket of 240 or 2,500 episodes converges on "the kitchen plus the entire object
catalogue" rather than on a small pool. A union model is a real construction and it does not
amortize.

MJWarp's per-world `Model` fields are the one construction that could avoid the blow-up —
`geom_dataid`, `geom_matid`, `geom_size` and `geom_rgba` all carry a leading `*` batch dimension in
`mujoco_warp._src.types.Model`, settable through `put_model(batch_sizes=...)`, and
`refit_scene_bvh` reads `geom_dataid` and
`geom_size` per world — so one model with N object *slots* and a shared mesh pool could select a
different instance per world without any union of bodies at all. That is a genuinely attractive
design and it is **not pursued**, because it can only ever land on the §11 row: it is an MJWarp
render, and an MJWarp render of a RoboCasa kitchen is 6–23 % interior-exact.

## 14. Renderer survey, re-run 2026-08-10

The brief asked for this explicitly, so the field was re-checked rather than assumed. **Nothing
released since the last survey displaces MJWarp**, and MuJoCo 3.11.0 / mujoco-warp 3.11.0
(2026-07-28) are still the newest releases; warp-lang 1.16.0 (2026-08-03) contains no rendering
changes.

| candidate | version / date | installs headless CUDA 12.9–13 | MJCF path | measured or published parity vs `mujoco.Renderer` | why not |
|---|---|---|---|---|---|
| **mujoco-warp** | 3.11.0, 2026-07-28 | yes | native | **6–23 % interior-exact here** (§11); upstream claims SSIM 0.917–0.994 (PR #1364) | the only viable fast option, and not exact |
| **MuJoCo EGL** | 3.11.0 | yes | native | **1.000 by construction** | 1,925 f/s ceiling (§10) |
| Newton | 1.4.0, 2026-07-16 | yes | `add_mjcf()` | none — a second Warp raycaster with no lighting-parity work | no parity story, no absolute throughput published |
| ManiSkill3 / SAPIEN | 3.0.1 / 3.0.3, 2026-04 / 2026-03 | needs Vulkan; open unanswered H100 hang (SAPIEN #290) | loader drops static actors and cameras | none; their own RoboCasa port required manual texture re-exports | renders SAPIEN scenes, not MuJoCo models |
| Isaac Lab / Isaac Sim | 3.0-beta2 / 6.0.1, 2026-07 / 2026-06 | container-scale; Kit + RTX renderer proprietary | MJCF → USD, lossy | none; RTX + MDL | 23 s warm / 230 s cold per scene, and the wrong dependency for this repo |
| Genesis | 1.3.2, 2026-08-07 | core yes; batch path `gs-madrona` has an unfixed Vulkan init bug | treats MJCF as one entity, ignores world options | a known **negative** (issue #236: MuJoCo textured, Genesis not) | `scene.build()` reported at 14 min |
| Madrona / madrona_mjx | deprecation notice 2026-02-18, last functional commit 2025-06-21 | pinned CUDA ≤ 12.5.1, jax < 0.6 | via MJX | never claimed | deprecated upstream, and does not run here (§ `backend3d.md`) |
| gsplat / 3DGS | 1.5.3, 2025-07-04 | yes | **none — no MJCF→splat baker exists** | every published metric is against photographs, none against a rasterizer | 554k triangles ≈ 10⁶–10⁷ Gaussians, and the target is incoherent |
| Mitsuba 3 / Dr.Jit | 3.9.1 / 1.5.0, 2026-08-07 | yes (OptiX) | MJCF → USD (experimental) → Blender → Mitsuba | no `phong` BSDF exists in Mitsuba 3 | no scene-batch axis at all: N states = N renders |
| nvdiffrast | 0.4.0, 2025-12-08 | git install | none | you write the shading model | 0.4.0 **removed the OpenGL backend**, so the CUDA rasterizer that OOM'd is now the only path |
| PyTorch3D | 0.7.9, 2025-11-28 | no Linux CUDA wheels | none | you write the shading model | 47 ms at 326k triangles vs nvdiffrast's 9 ms |
| Kaolin | 0.18.0, 2025-08-08 | wheels to cu129 | none | SH/PBR, further from Phong than writing it | its fast path *is* nvdiffrast |
| mujofil | 0.2.6, 2026-06-27 | yes, surfaceless EGL | native | PBR + IBL + SSAO **by design** | pins mujoco < 3.10; 3,200 cam/s claimed |
| PyBatchRender | arXiv 2601.01288, 2026-01-03 | git install | none — scenes authored in Python | none | headline is CartPole at 64 px |

Three things worth carrying out of the survey independently of RoboCasa:

- **`mujoco_warp`'s own `render_test.py` has no RGB parity test.** It asserts exact segmentation and
  depth at `atol=1e-2`; the RGB assertion compares eager against graph-captured output, i.e. the
  renderer against itself. Upstream's parity claim is an SSIM number from a PR description.
- **"Lambertian diffuse only" is stale.** PR #1364 (merged 2026-06-02, so already in 3.11.0) added
  full Phong specular/shininess/emission, headlight, and per-light attenuation/spot/ambient.
  `backend3d.md` repeats the old line and it is corrected there.
- Alpha and primitive texgen are **still** absent on `main` today, with no open issue on either.

## 15. Verdict

**NO-GO, on G4, measured on the render loop.** Two numbers decide it and neither is reachable by
engineering the parts around them:

| path | frames/s, 128 px | CPU cores of JPEG decode replaced | interior pixels exact vs the official frame |
|---|---|---|---|
| MuJoCo EGL, 1 process | 950 | 0.18 | **1.000** (it *is* the official renderer) |
| MuJoCo EGL, 16 processes | 1,925 | **0.37** | **1.000** |
| MJWarp, batch 256, alpha pre-pass | 16,690 – 26,400 | **3.2 – 5.1** | **0.059 – 0.231** |
| *required by every env in the registry* | — | *> 1* | ***1.000*** |

**Exact costs 2.7× more than reading. Fast is 4–17× short of the parity bar.** Both are measured with
the model already compiled and resident, so both hold at K = ∞ and neither depends on G3, on how a
corpus is ordered, or on any of the setup work in §12.

What each of the four gates now says, so the record is not "it seems slow":

1. **G1 purity — passes.** Restoring the state on a fixed compiled model reproduces the frame with
   max abs diff 0/255 (§2).
2. **G2 compression — passes, and by more than recorded.** 364 B/step against a 49,152 B frame, and
   the per-episode scene descriptor is `ep_meta` at **1.7 kB gzipped**, not the 385 kB MJCF: it
   replays at 99.6 % of pixels with the whole residual on one robot-base geom (§9).
3. **G3 shared decoder — still fails, for a narrower reason than recorded.** The compiled model is
   per-episode in every shipped corpus, 24/24 distinct at a pinned kitchen. But the *drawn* kitchen
   is a constant of `(layout, style)` — 357 visible geoms identical across 64 episodes — and the
   whole per-episode delta is three objects and the robot base (§8). The decoder can be made
   per-kitchen (§13), and its build cost then converges to ~0.4 s per episode against the 0.097 s it
   costs to decode that episode, so the amortization never gets below 4x decode.
4. **G4 economics — fails on the LOOP, independently.** This is new. RoboCasa is the first family in
   `when_does_this_pay_off.md` whose G4 failure is not downstream of G3: with setup deleted
   entirely, the only renderer that meets the parity bar is 2.7× more expensive than reading the
   stored frames.

### Answering the three questions the brief asked

- **Episodes per kitchen in real corpora: 12 (pretrain human), 240 (pretrain MimicGen), 2,500
  (target human)** — pooled across tasks, with kitchen = `(layout_id, style_id)`. The hypothesis that
  K = 1 is an artifact of episode ordering is **correct**. Two of the three shipped corpora sit well
  above any break-even this document has ever computed.
- **The bucketed-pipeline crossover: K ≈ 21**, down from 145, once setup is corrected from 10.04 s to
  **1.41 s** on the same kitchen (§12). Real corpora sit above it — 240 and 2,500 comfortably, 12 marginally. It changes
  nothing, because the ceiling the crossover converges to is 3.2–5.1 cores at 6–23 % interior-exact,
  or 0.37 cores at 100 %.
- **Renderer survey: nothing displaces MJWarp** (§14). ManiSkill3/SAPIEN has no usable MJCF path and
  an open H100 hang; Newton is a second Warp raycaster with no parity work; Genesis has a *published*
  texture mismatch against MuJoCo; Isaac is proprietary and 23 s/scene; no MJCF→splat baker exists;
  Mitsuba has no scene-batch axis and no Phong BSDF. The only two renderers that can draw a RoboCasa
  kitchen correctly are MuJoCo's own and MJWarp's, and they are the two rows of the table above.

### Engaging with the pinned-kitchen refusal

The previous pass declined a pinned-kitchen subset for three reasons. Bucketing changes one of them
and sharpens the other two:

| previous reason | what the measurements do to it |
|---|---|
| "the win is 3.5 cores, against 7.7–899 for every shipped env" | **now 3.2–5.1 cores** with the alpha pre-pass and shadows on — better, still an order of magnitude below the family's own examples, and only at 6–23 % interior-exact. At 100 % exact it is **0.37**. |
| "it cannot be pixel-exact; 14.7 % of interior pixels" | **worse than recorded** on this repo's own interior definition: 5.9–23.1 % across four kitchens, and a 5× spread between them. `parity_interior_min` would have to be declared at ~0.06 where every other env declares 1.0. |
| "24 GB outside the package, nothing imports in CI" | unchanged, and now second-order. `dmcontrol.py` shows an optional-dependency env is landable; the parity floor is what is not. |
| "it isn't RoboCasa" | **partly answered.** With the drawn kitchen constant per `(layout, style)` (§8), a pinned kitchen still varies its three objects, which is real diversity and would be a legitimate `variation_space`. That argument is now available and does not rescue the other two rows. |

### What would change the answer

Stated as falsifiable conditions, because two previous verdicts rested on numbers that turned out to
be wrong:

1. **A renderer that is bit-exact against `mujoco.Renderer` on textured meshes and batches.** Two
   named causes, and one of them is already excluded: correcting MJWarp's truncate-vs-round rule —
   the biggest fidelity win available on that backend, worth 0.791 → 0.9988 on flat materials — caps
   RoboCasa at **0.142–0.647** interior-exact, because only that fraction of the residual has a
   truncation signature (§11). What is left is primitive texgen (178–401 textured boxes and cylinders
   per kitchen sampling one texel) and texture filtering (MuJoCo mipmaps; MJWarp point-samples a
   linear filter — `wp.Texture2D` gained `num_mip_levels` in Warp 1.15 but the shader computes no
   LOD). If MJWarp reached ≥ 99 % interior exact, RoboCasa becomes a GO for kitchen-bucketed corpora
   at K ≥ 21.
2. **MuJoCo's rasterizer above ~5,200 f/s at 128 px on one GPU.** It saturates at 1,925 across 16
   processes today, and the bottleneck is per-frame scene construction and readback, neither of which
   batches. A batched `mjr_render` would be a MuJoCo change, not a stable-envs one.
3. **A shipped corpus with a fixed compiled model.** `mg_5x1` (5 kitchens, 10,000 demos/task) would
   give K ≈ 2,000, but its tars are not published — 0 of 34 entries have a download link — and it
   still re-samples objects per demo.

None of the three is a matter of effort inside this repo, which is the sense in which this is a
NO-GO rather than a backlog item.

### What was landed anyway

The RoboCasa work produced one change that ships, because it is not RoboCasa-specific:
`stable_envs.render.mjwarp.create_render_context` removes MJWarp's per-texture re-upload of the whole
texture blob. **Context creation 8.98 s → 1.34 s, transient device memory 2,338 MB → 168 MB, frames
bit-identical.** It pays for every textured MuJoCo scene in this repo — DM Control's three-plate bake
included — and is worth reporting upstream as a one-line change.

## 16. Reproducing

Run on a GPU node; the pinned scene XML carries absolute asset paths and loads without robosuite.

```
bash scratch/robocasa/run_g4.sh egl        # G4 decomposition, mujoco 3.3.1 and 3.11
bash scratch/robocasa/run_g4.sh mjwarp     # batch setup/throughput/amortization
bash scratch/robocasa/run_g4.sh texgen     # primitive texgen repro, bake, BGRA re-check
python scratch/robocasa/probe_green.py     # bisection that found the alpha bug
python scratch/robocasa/probe_alpha.py     # alpha census + the fix + residual by geom type
python scratch/robocasa/probe_parity.py    # 16-config sweep, interior vs silhouette, 3 cameras
```

| probe | output |
|---|---|
| `probe_g4_egl.py` | `out/g4_egl_mj{311,331}.json` |
| `probe_g4_mjwarp.py` | `out/g4_mjwarp.json` |
| `probe_texgen.py` | `out/texgen.json`, `out/texgen_{mini,mesh,kitchen*}_*.png` |
| `probe_green.py` | `out/green.json`, `out/green_const_*.png` |
| `probe_alpha.py` | `out/alpha.json`, `out/alpha_sheet.png` |
| `probe_parity.py` | `out/parity.json`, `out/parity_<camera>.png` |

Third pass (this one), on a GPU node:

```
python scratch/robocasa/probe_bucket.py --n 24        # episodes per compiled model, kitchen/object split
python scratch/robocasa/dump_corpus.py                # a kitchen-bucketed corpus: MJCF + qpos + ep_meta + frame
python scratch/robocasa/probe_epmeta.py               # ep_meta sufficiency, residual localised by geom
python scratch/robocasa/probe_exact_ceiling.py        # THE deciding number: MuJoCo EGL ceiling, 1-16 procs
python scratch/robocasa/probe_parity_now.py           # MJWarp vs the official frame, repo interior definition
python scratch/robocasa/probe_fastsetup.py            # setup decomposed; texture hoist, BVH prune, pool, overlap
python scratch/robocasa/probe_minsetup.py             # put_model / put_data: content or fixed overhead?
python scratch/robocasa/probe_setup_final.py          # corrected setup end to end + break-even K
python scratch/robocasa/probe_union.py                # per-kitchen union model: exactness and growth
```

| probe | output |
|---|---|
| `probe_bucket.py` | `out/bucket.json` |
| `dump_corpus.py` | `out/corpus/L<ll>S<ss>/ep*.{xml,_qpos.npy,_meta.json,_frame.npy}` |
| `probe_epmeta.py` | `out/epmeta.json` |
| `probe_exact_ceiling.py` | `out/exact_ceiling.json`, `out/stripped_scene.xml` |
| `probe_parity_now.py` | `out/parity_now2.json` |
| `probe_fastsetup.py` | `out/fastsetup.json` |
| `probe_minsetup.py` | `out/minsetup.json` |
| `probe_setup_final.py` | `out/setup_final.json` |
| `probe_union.py` | `out/union.json`, `out/corpus/*/union.xml` |

The MJWarp probes run in `~/venvs/se3d` (mujoco 3.11, mujoco-warp 3.11); everything that needs
robosuite runs in the `robocasa` conda env (mujoco 3.3.1). The corpus MJCFs carry absolute asset
paths and compile without robosuite, which is what lets the two halves stay separate.

## 17. Fourth pass: the same measurements, a different question — and a new tier

Sections 1-16 ask whether RoboCasa can join the **parity** tier. They answer no, twice over, and both
halves of that answer stand: the only renderer that reproduces the official frame exactly is MuJoCo's
own, at **1,925 f/s across 16 processes against 5,137 f/s for one CPU core of JPEG decode at 128 px**
(§10), and the fast alternative is MJWarp at **6-23 % of interior pixels exact** (§11).

Bit-exactness is the right bar for the claim this repo is built on — *the regenerated frame IS the frame
you would have stored*. It is the wrong bar for a second, legitimate use that the same measurements
support: **generating training frames live**, where what matters is that the picture is right and
self-consistent rather than that it matches MuJoCo's rasterizer pixel for pixel. So the repo now carries
two declared tiers (`docs/concepts/fidelity-tiers.md`), and RoboCasa is the first env in the second one.

Everything in this section is at **256 px** — the resolution a robot-learning consumer actually trains
at, and one where the arithmetic is less favourable than §§10-11's 128 px, which is why it is quoted
rather than the more flattering number.

### 17.1 What it looks like — the deliverable is a video, not a table

`examples/robocasa_visual/` renders one 200-step trajectory twice from the same `qpos`: MuJoCo's own
rasterizer, MJWarp, and 4x the absolute difference. Both the best and the **worst** of the four kitchens
§11 measured are shipped, because registering only the good one would make every number below a
selection effect.

The geometry, the camera, the robot, the objects and their motion agree. The artefact is one mechanism,
and it is the §7/§11 texgen gap seen rather than inferred: **surfaces whose texture is painted on a BOX
or a CYLINDER come out one flat colour.** On L52 S03 that is the marble backsplash and the sink basin;
on L45 S33 it is the counter and the entire run of upper cabinet doors, which is exactly why that
kitchen scores five times worse. The robot — 18 textured *meshes*, where MJWarp does have a UV path — is
essentially exact.

### 17.2 Measured, against the frames robosuite itself produced

256 px, 200 trajectory states, one H200, scored against `env.sim.render` at geom group 1:

| kitchen | camera | mean \|diff\| | within 8/255 | SSIM | interior mean | interior exact |
|---|---|---|---|---|---|---|
| L52 S03 | agentview_left | 6.60 | 82.8 % | 0.894 | 1.66 | 13.1 % |
| L52 S03 | agentview_right | 5.75 | 84.4 % | 0.902 | 1.30 | 7.9 % |
| L52 S03 | eye_in_hand | 6.36 | 84.6 % | 0.932 | 3.72 | 17.5 % |
| L45 S33 | agentview_left | **32.50** | **35.6 %** | **0.796** | 45.45 | 4.7 % |
| L45 S33 | agentview_right | 31.06 | 37.5 % | 0.799 | 46.01 | 2.8 % |
| L45 S33 | eye_in_hand | 15.14 | 63.5 % | 0.849 | 24.30 | 12.0 % |

The interior-exact column is the same quantity §11 reports and is consistent with it (§11's four
kitchens ran 5.9-23.1 % at 128 px). It is carried into the env's `visual_fidelity` and **reported but
not asserted**: it is the honest measure of what the tier gave up, and asserting a low floor on it would
turn a number reported for honesty into a target.

### 17.3 Two measurement errors this pass found, both silent, both in the PREVIOUS numbers' favour

**The official frame is rendered from stale kinematics.** `mj_step` evaluates the forward dynamics at
the old `qpos` and then integrates, so on return `data.qpos` is the new state while `data.xpos` — what
`mjv_updateScene` reads — still describes the old one. Recording `qpos` and the frame at that moment
gives a pair no renderer can reproduce: restoring the recorded `qpos` into a fresh `MjData` on the
**same live model** and re-rendering disagrees on **7.7 % of pixels (39.0 % on the wrist camera)**, mean
0.41/255. Every fidelity number measured that way charges a state mismatch to the renderer.
`dump_trajectory.py` now calls `sim.forward()` first and **asserts** the round trip is bit-exact over
eight spaced frames before it writes anything. (Its effect on the aggregate here was small — mean 6.62
to 6.60 — because the texgen residual dominates. It was not small on the state-sufficiency claim, which
is what makes an env contract possible at all.)

**§3's "bit-identical" holds for the renderer and not for the round trip.** A bare `mujoco.Renderer`
restricted to geom group 1 reproduces `env.sim.render` with **max abs diff 0** on the live model
(`probe_visual_reference.py`, case A) — so §3 is right about what it measured. Re-compiled from the
MJCF that `sim.model.get_xml()` writes, the same comparison is **99.74 % of pixels exact at mean
0.0044/255** under the same MuJoCo, because the XML writer rounds coordinates to ~6 significant digits
(`body_pos` moves by 5e-6 m) and silhouettes land differently. Across MuJoCo versions, 3.11.0 against
frames recorded under 3.3.1, it is **99.4 % at mean 0.041/255**. Both are three orders of magnitude
below the residual being measured, which is what licenses the env's reference bridge — and neither is
the 1.000 §3's wording implies.

### 17.4 Throughput, as a surface with its crossovers

Official = MuJoCo's rasterizer at group 1, one process, warmed, best of two passes. Decode = one CPU
core of PIL JPEG from memory (`bench_decode.py`), the favourable case for a stored dataset.

| resolution | official EGL | MJWarp b=1 | b=16 | b=64 | b=256 | 1 core JPEG decode |
|---|---|---|---|---|---|---|
| 128 px | 932 | 2,010 | 11,973 | 14,409 | **16,701** | 7,483 |
| 224 px | 855 | 1,929 | 6,028 | 6,489 | **6,948** | 6,762 |
| 256 px | 812 | 1,846 | 5,161 | 5,436 | **5,666** | 5,486 |

- **vs the official renderer: (17.9x, batch 256) at 128 px and (7.0x, batch 256) at 256 px.** Ahead
  already at batch 1 (2.2x), so there is no crossover to wait for; saturates by batch 64.
- **vs reading a stored dataset: (2.2 cores, batch 256) at 128 px and (1.03 cores, batch 256) at
  256 px**, crossing one core at roughly batch 8 (128 px) and batch ~128 (256 px).

The 128 px row is consistent with §11's 16,690-26,400 across kitchens. The 256 px row is the one that
matters for the tier's honesty: **at the resolution a robot-learning consumer trains at, an H200
replaces about one CPU core of decode.** This tier is a training-throughput argument against the
official renderer, not a storage argument against a dataset — which is the same conclusion §15 reaches,
reached again at a resolution that makes it starker.

### 17.5 Setup, per compiled kitchen, and the break-even a live user actually cares about

256 px, batch 256, warm, all fixes from §12 applied:

| stage | seconds |
|---|---|
| MJCF compile | 0.81-1.25 |
| `put_model` | 0.75-1.41 |
| `put_data` | 0.87-1.24 |
| `create_render_context` | 0.56-0.58 |
| CUDA-graph capture | 0.03 |
| **total** | **3.0-3.5 s** (3.1-8.7 across 12 kitchens of 1.2k-2.3k geoms) |

§12's 1.41 s is reproduced in spirit and not in number, and the difference is stated rather than
smoothed: §12 measured a kitchen **stripped to its drawn geometry** at 128 px, and this is the
unstripped model at 256 px. `create_render_context` — the term §12's texture hoist fixed, and the one
this repo owns — is 0.56 s against upstream's 8.98 s, unchanged.

The break-even a live-generation user needs is not §12's `K` against decode. It is against the renderer
they would otherwise call. MuJoCo pays its own ~1.7 s of compile plus EGL context, so MJWarp's *extra*
setup is ~1.8 s and it saves (1/812 − 1/5,666) s per frame: **≈1,700 frames, about 9 episodes of 200
steps at one camera.** Against a stored dataset at 256 px it never amortises, because the steady-state
rates are within 3 % of each other; at 128 px it takes ≈48,000 frames (~240 episodes).

One cost that is easy to omit and is paid once per machine: the **first** context built for a given
render configuration also compiles the Warp megakernel, **140-175 s**, cached in `WARP_CACHE_PATH`
thereafter. Every setup number above is warm.

### 17.6 A mitigation that does not pay, recorded as a measured negative

§7 notes that re-expressing a textured box as a mesh with cube texcoords recovers most of the texgen
gap, and that a naive per-face UV is not MuJoCo's own box mapping. The cheaper version was tried
instead: replace the texture on the geoms MJWarp cannot UV with that texture's **mean colour** rather
than its corner texel (`stable_envs.render.mjwarp.bake_primitive_textures`). §7 measures the mean-colour
bake at 7.99/255 against MuJoCo's textured frame, so it ought to help.

It does not. Mean \|diff\| moves 6.60 to 6.68 on L52 S03 and 32.50 to 31.83 on L45 S33, and SSIM falls in
both. Two reasons, both measurable: RoboCasa's box materials are often near-uniform
(`textures/flat/light_gray.png` and friends), so the corner texel is already about right; and only **38
of ~137** textured materials are worn *exclusively* by primitives — the other 99 are shared with meshes,
which MJWarp textures correctly and which a flat bake would break. Fixing those needs a per-geom material
clone and a recompile, and is not done. The function ships with this result in its docstring.

### 17.7 What landed, and what it does NOT license

Four classes, one compiled kitchen and one camera each, in `stable_envs/envs/robocasa.py`. The contract
is the repo's usual one: `@register`, `state_names`, a `variation_space` whose defaults are the official
appearance, `resolve_style` for per-environment randomisation, `reference_env`/`reference_render`, and an
import-time guard so the family is cleanly absent where `mujoco_warp` or the 24 GB asset library are not.

Three things are specific to this family and worth carrying:

- **The state is the `qpos` this camera can see, measured.** A pinned kitchen has `nq` 91-133 and most
  of it is fixture hinges behind the camera or inside a closed cabinet. The live set is measured once
  per `(kitchen, camera)` by the same rule `check_state_dims_matter` uses, but stricter (a dimension must
  move the frame for *every* probe state, not any), and cached beside the scene: **40, 44, 13 and 37**
  of 91/91/91/103. The wrist camera's 13 are its own arm.
- **The kitchen is built by a different interpreter.** `robocasa/__init__.py` asserts
  `mujoco.__version__ == "3.3.1"` and `mujoco_warp` needs >= 3.11, so **no interpreter holds both**. The
  expanded MJCF is produced once by the robocasa env and cached under
  `~/.cache/stable-envs/robocasa/<kitchen>/`; the render side never imports robocasa.
- **Domain randomisation is one launch.** `cam_pos`, `cam_quat` and `light_diffuse` are per-world MJWarp
  model fields, so 12 cameras and 12 lightings cost one graph replay (`media/dr_one_batch.png`, 3.1 ms
  for 12 worlds at 256 px). Different *kitchens* are different compiled models and therefore N contexts
  and N launches (`media/kitchen_grid.png`) — MJWarp batches worlds of one model, not models, and the
  two figures are labelled separately so the first claim is not read onto the second.

And the boundary, which is the reason the tier exists:

> **Suitable for generating training frames live. NOT a substitute for a stored dataset when a policy
> will be evaluated on official RoboCasa pixels.** The train and eval pixel distributions differ by the
> amounts in §17.2, and nothing in this repo measures what that difference costs a policy.

`parity_interior_min` is `None` on these classes, `register()` refuses the class if it is not, and
`test_render_parity_with_official_renderer` skips them rather than being relaxed for them.
