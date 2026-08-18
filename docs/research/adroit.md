# Adroit hand (gymnasium-robotics) — feasibility for `stable-envs`

**Verdict: NO-GO — on the backend. One deciding gate, plus one separate upstream-worthy finding.**

**The deciding gate: `mujoco_warp` cannot draw these scenes.** It has no alpha path at all, and
derives texture UVs for `PLANE` and `MESH` only. Adroit's scenes are built from exactly the
primitives that leaves out, and the two sub-limits kill *different* scenes — a distinction that
matters because they have different fixes (§4):

| | killed by | moving geometry lost |
|---|---|---|
| **Door** | textured primitives | **20.31 %** of the frame |
| **Hammer** | textured primitives | 3.25 % (+ 0.41 % sites) |
| **Pen** | **alpha** | 1.52 % — the pen (α 0.60) and its target ghost (α 0.30) |
| **Relocate** | **both** | 0.40 % textured sphere **+ 3.06 % translucent target site** (α 0.125) |

Cube-texcoord re-expression, the known partial workaround, would recover part of Door and Hammer and
**does nothing whatsoever for Pen**, whose loss is entirely alpha.

**The separate finding: the SHIPPED default appearance is unreproducible.** Adroit's `rgb_array`
frame is history-dependent — its free camera's `lookat` is frozen at `median(data.geom_xpos)` the
first time `render()` is called — so the same state renders **41.1 %–80.9 % differently** depending
on what preceded that first call (§2). This is a real defect worth reporting upstream, and it is
**not** independently fatal to a port: `camera_name="fixed"` is officially supported and measured
bit-identical across seeds and warmups (10/10 pairs, §3), and under it the thesis holds for Adroit —
G1, G2 and G3 all pass. A port would simply have to ship that camera with a loud note that it is not
the picture the Adroit papers show. The port dies on the backend, not on the camera.

The family is otherwise an excellent fit — a 24–30-joint hand, a fixed camera available, and 61–72 %
of every frame provably static and bakeable — which is why both the camera and the backend were
pushed to a measured number rather than an argument.

All numbers: H200 via SLURM, `MUJOCO_GL=egl`, gymnasium-robotics 1.4.2, gymnasium 1.2.3, mujoco
3.11.0 / mujoco-warp 3.11.0 (`~/venvs/se-fetch`), 480 px — the family's own `DEFAULT_SIZE`.
Probes: `scratch/adroit/probe*.py`. Raw JSON: `/mnt/vast/.../se_logs/adroit/probe*.json`.

---

## 1. What the family is

Eight registered ids over four scenes; `Sparse` changes the reward only, so it would have shared a
scene exactly as Fetch's `Dense`/`Dict` do.

| scene | ids | per-episode randomisation | where it lives |
|---|---|---|---|
| Door | `AdroitHandDoor[Sparse]-v1` | door frame position | `model.body_pos["frame"]` |
| Hammer | `AdroitHandHammer[Sparse]-v1` | nail board height | `model.body_pos["nail_board"]` |
| Pen | `AdroitHandPen[Sparse]-v1` | desired pen orientation | `model.body_quat["target"]` |
| Relocate | `AdroitHandRelocate[Sparse]-v1` | object and target position | `model.body_pos["Object"]`, `model.site_pos["target"]` |

The first thing worth recording is that **the per-episode randomisation is written into the MODEL,
not into `qpos`**. Fetch's goal is a `data` quantity; Adroit's is `model.body_pos` / `body_quat` /
`site_pos`. That is not itself a blocker — `mujoco_warp`'s `put_model` accepts `body_pos`,
`body_quat` and `site_pos` in `batch_sizes`, so they can be per-world — but it means the state vector
is `qpos + (body_pos | body_quat | site_pos slice)` and it means a naive port that stores only `qpos`
silently loses the layout.

## 2. The shipped default appearance is not a function of the state

This section is a finding about the *reference*, not the deciding gate — §3 shows it has an
officially-supported rescue. It is recorded in full because it affects anyone using Adroit pixels at
all, whether or not they use this repo.

`gymnasium/envs/mujoco/mujoco_rendering.py`, `OffScreenViewer._init_camera`:

```python
self.cam.type = mujoco.mjtCamera.mjCAMERA_FREE
for i in range(3):
    self.cam.lookat[i] = np.median(self.data.geom_xpos[:, i])
self.cam.distance = self.model.stat.extent
```

`MujocoRenderer._set_cam_config` then applies `DEFAULT_CAMERA_CONFIG` on top. Fetch overrides all
four parameters there, so its camera is a constant of the family and `FetchBase._check_camera` can
assert it. **Every Adroit env overrides only `distance` and `azimuth`** — so `lookat` keeps the
median of `geom_xpos`, `elevation` keeps `MjvCamera`'s default (measured: −45.0), and the viewer is
built lazily, at the first `render()`, and then frozen for the life of the env.

Two consequences, both measured with **one env per process** (§2.1 explains why that matters).

**Same state, different number of actions before the first render** (`probe4_purity.py`):

| scene | lookat shift (m) | differing pixels | mean \|diff\| | max |
|---|---|---|---|---|
| Door | 0.2161 | **80.9 %** | 58.3 | 255 |
| Relocate | 0.0449 | **59.5 %** | 21.9 | 199 |
| Pen | 0.0134 | **53.9 %** | 7.5 | 255 |
| Hammer | 0.0201 | **41.1 %** | 9.5 | 255 |

**Same state, different reset seed before the first render** — the most charitable convention a port
could declare, "the camera is the one frozen at a reset state" (`probe5_seedsweep.py`, 6 seeds, 15
pairs):

| scene | pairs bit-identical | lookat spread (m) | worst pair |
|---|---|---|---|
| Hammer | **15 / 15** | 0.000000 | — |
| Relocate | 6 / 15 | 0.003539 | 33.5 % of pixels |
| Door | 4 / 15 | 0.004172 | **48.4 %** of pixels |
| Pen | 4 / 15 | 0.002000 | 45.3 % of pixels |

So even that convention is ill-defined for three of the four scenes. Only Hammer's median is exactly
invariant across resets, because its `reset_model` moves a single body along one axis and a median
over 37 geoms does not notice.

The sensitivity is the reason such tiny shifts matter. Through **one** viewer, moving `lookat` by
1 mm (`probe3_camsens.py`):

| scene | dy = 0.001 m | dy = 0.005 | dy = 0.020 | dy = 0.216 |
|---|---|---|---|---|
| Pen | 37.3 % | 51.8 % | 56.7 % | 63.6 % |
| Door | 29.4 % | 48.0 % | 59.4 % | 81.1 % |
| Relocate | 21.9 % | 34.6 % | 44.3 % | 63.1 % |
| Hammer | 14.8 % | 25.3 % | 36.9 % | 53.9 % |

A millimetre of camera is a fifth to a third of the frame. Door's cross-process 80.9 % lands within
0.002 of the 81.1 % that a pure 0.216 m lookat shift produces through one viewer, which is what
confirms the camera is the whole effect and nothing else drifted.

**This is a property of the reference, not of any GPU renderer**, and it is the same shape as the
LunarLander no-go (`render()` mutates what it draws) — with one decisive difference: LunarLander has
no rescue, and this does (§3). It applies to anyone using Adroit pixels at all: two
dataset-collection runs of the same policy produce systematically different images, and nothing in a
stored state records which camera was in force.

### 2.1 A measurement trap that produced a confident wrong answer

The obvious version of this test — two envs in one process — is **not measurable**. gymnasium's
`OffScreenViewer.render` never binds its GL context (`MujocoRenderer.render` calls
`make_context_current()` only when a single renderer owns more than one `render_mode`), so with two
live viewers both renders resolve against whichever EGL context was created last.

Measured: Hammer, Pen and Relocate returned **byte-identical** frames from cameras that were re-read
immediately before each render and differed by 0.013–0.045 m — shifts the sensitivity table above
says must move 32–55 % of pixels. Door, in the same probe, gave a correct-looking 80.9 %. A probe
that had covered only Hammer/Pen/Relocate would have concluded the family was pure; one that covered
only Door would have got the right answer for the wrong reason. One env per process removes it.

## 3. The rescue that works, and how far it goes

Every Adroit XML defines `<camera name="fixed">` and `<camera name="vil_camera">`, and gymnasium's
`MujocoEnv` accepts `camera_name=`. A model camera is a constant of the model, so it is immune to all
of §2. Measured across 5 cases spanning reset seeds **and** warmup counts, 10 pairs
(`probe6_fixedcam.py`):

| scene | cam type / id | pairs bit-identical |
|---|---|---|
| Door, Hammer, Pen, Relocate | 2 (`mjCAMERA_FIXED`) / 0 | **10 / 10** |

Verified not vacuous (`probe7_notblank.py`): the fixed-camera frames carry 11,198–12,961 unique
colours over the full 0–255 range, against 7,595–11,349 for the free-camera frames. This is the
"parity against a blank surface passes" trap, checked rather than assumed.

**So under a declared fixed camera the thesis holds for Adroit**: the frame is a function of
`qpos` + the per-episode model slice. G1, G2 and G3 all pass. The port dies on the next gate.

## 4. The deciding gate — `mujoco_warp` cannot draw these scenes

`stable_envs.render.mjwarp` documents two structural limits: `sample_texture` derives UVs for `PLANE`
and `MESH` only, so a textured `BOX`/`SPHERE`/`CYLINDER`/`CAPSULE` samples texel (0,0) and renders as
one flat colour; and the shader reads `color[0..2]` and packs a constant alpha of 255, so there is no
translucency path at all. Adroit's scenes are built almost entirely from exactly those primitives,
carrying `MatWood` / `MatFoil` / `tablecube` / `MatWoodB` textures.

Pixel shares through the fixed camera, over states the env reaches (reset + 0–25 random actions, 12
states per scene, `probe8_coverage.py`):

| scene | **moving & undrawable** | moving & OK (hand meshes) | static (bakeable) | sites | background |
|---|---|---|---|---|---|
| **Door** | **20.32 %** | 8.06 % | 66.24 % | 0.00 % | 5.39 % |
| **Hammer** | **3.25 %** | 13.01 % | 61.46 % | 0.41 % | 21.88 % |
| **Pen** | **1.52 %** | 4.38 % | 72.00 % | 0.00 % | 22.11 % |
| **Relocate** | **0.40 %** | 9.98 % | 65.50 % | **3.06 %** | 21.06 % |

The static column is *not* a problem — it is baked through MuJoCo itself, exactly as Fetch does, and
is exact by construction.

### 4.1 Which sub-limit kills which scene

The undrawable share splits cleanly by cause, and the split is the part worth carrying forward,
because the two causes have completely different fixes:

| scene | textured primitives | alpha | which upstream gap must close |
|---|---|---|---|
| **Door** | **20.31 %** | 0.00 % | UVs only |
| **Hammer** | **3.25 %** (+ sites, mixed) | 0.00 % | UVs only |
| **Pen** | 0.00 % | **1.52 %** | **alpha only** |
| **Relocate** | 0.40 % | **3.06 %** (target site) | **both** |

What is in each:

| scene | what is undrawable, and why |
|---|---|
| Door | the door panel (`geom54`, **9.4 % of the frame alone**, textured BOX), both frame posts and both door posts (textured CYLINDER), the latch barrel and its two handles (textured CYLINDER/CAPSULE) — 8 geoms, all `MatWood`/`MatFoil`, **none translucent** |
| Hammer | the nail board (2.05 %, textured BOX), the hammer handle (0.76 %, textured CAPSULE), the two nail cylinders (`MatFoil`); its 0.41 % of sites is mixed — `nail_goal` is opaque but `MatWood`-textured, `S_target` is α = 0.2 |
| Pen | **the pen itself** (α = 0.60) and **its target ghost** (α = 0.30) — the two objects the task is about, and **nothing textured at all** |
| Relocate | the object sphere (0.40 %, textured SPHERE) **and** a target *site* at α = 0.125 covering 3.06 %, which `mujoco_warp` does not draw at all and which re-expression as a geom does not rescue, because the residual problem is the alpha |

Door is decisive on magnitude: one fifth of every frame would be flat-shaded where MuJoCo paints wood
grain. The other three are decisive on *what* is lost rather than how much — a port that renders the
pen opaque, or omits Relocate's target, reproduces a picture the env never produces, in exactly the
region a policy looks at. Unlike Fetch's translucent mocap markers (1.2–1.5 % of all pixels but
0.000002 of *interior* pixels, because a translucent bar over a flat surface leaves no flat 3×3
neighbourhood), Adroit's undrawable geoms are large opaque textured surfaces whose interiors are
precisely what the interior metric measures.

Note that a *site* is not a third limitation of the same kind: `mujoco_warp` draws no sites, but a
site can be re-expressed as a geom, which is exactly what the Fetch port does for `target0`. That
re-expression works for Hammer's opaque `nail_goal` (leaving it a textured-cylinder problem) and does
not help Relocate's `target`, which is translucent whatever primitive carries it.

### What was considered and rejected

| option | why not |
|---|---|
| re-express textured primitives as meshes with cube texcoords | `render/mjwarp.py` measures this at mean error 50.30 → 6.43 — a large recovery, still not interior-exact. On 20 % of the Door frame that is not a `parity_note`, it is a different picture. And per §4.1 it addresses **only** Door, Hammer and 0.40 % of Relocate: it does nothing for Pen, whose entire loss is alpha. |
| bake the door/latch/nail/pen into the plate | they move with the state; that is the definition of a mover |
| render everything through MuJoCo | exact, but it *is* the official renderer — no speedup, fails G4 |
| hybrid `raycast3d` (analytic primitives, textures, alpha) + `mujoco_warp` (meshes) | the only real fix. Needs textured-primitive UVs, alpha blending, and correct cross-backend depth compositing — new shared rendering infrastructure with its own fidelity story, not an env module |

## 5. What would change the verdict

> **Upstream re-checked 2026-08-10, and it has not moved.** MuJoCo's newest release is still 3.11.0
> (2026-07-28) and `mujoco_warp`'s `main` at `c822833` (2026-08-07) differs from the installed 3.11.0
> `render.py` by **six lines**, all kernel-launch configuration (`default_grid_stride`,
> `grid_stride=False`). `sample_texture` still derives UVs for `PLANE` and `MESH` only; the shader
> still packs a constant alpha of 255. Neither gap has an upstream issue or PR. The last functional
> renderer work was PR #1364 (merged 2026-06-02), which brought OpenGL LIGHTING parity — SSIM against
> `mujoco.Renderer` from ~0.3–0.5 to 0.917–0.994 — and is already in 3.11.0, i.e. already in every
> number this document reports.
>
> The cube-texture workaround that unblocked [Franka Kitchen](kitchen.md) and
> [Meta-World](metaworld.md) does **not** transfer here, and the reason is the §4.1 split: Adroit's
> losses are moving textured PRIMITIVES (Door's 20.31 %) and ALPHA (Pen, Relocate). Neutralising a
> texture only helps where the pixel is discarded anyway — a static surface taken from the plate.
> Adroit's are movers, drawn by the ray-caster and kept.

- **Alpha and non-mesh texture UVs in `mujoco_warp` — both, not either.** Both are upstream gaps,
  present in 3.11.0 and on `main`, and §4.1 shows they are not interchangeable: UVs alone would
  unblock Door and Hammer and leave Pen exactly as broken as it is now; alpha alone would unblock Pen
  and leave Door's 20.31 % untouched. Only Relocate needs both to be fixed on its own. **With both
  closed, all four scenes become straightforward Fetch-pattern ports** — the state is known
  (`qpos` + the per-episode model slice), the camera is available and pure, and 61–72 % of each frame
  already bakes exactly through MuJoCo.
- **A declared fixed camera would have to ship with it** — `camera_name="fixed"`, plus a loud note
  that the DEFAULT free-camera appearance is unreproducible and why. That is a legitimate reference
  (`gymnasium.make(id, render_mode="rgb_array", camera_name="fixed")` is officially supported and
  measured bit-identical here), but it is *not* the picture the Adroit papers show.

## 6. Upstream-worthy findings

1. **Adroit's `rgb_array` output is history-dependent** (§2). Worth reporting to
   gymnasium-robotics: adding `lookat` and `elevation` to each `DEFAULT_CAMERA_CONFIG` would make
   the camera a constant of the family, cost nothing, and make the pixels reproducible. Fetch already
   does exactly this.
2. **`OffScreenViewer.render` does not bind its GL context** (§2.1) — any process holding two MuJoCo
   viewers silently reads the wrong framebuffer.
3. **`MujocoRenderer.render(..., segmentation=True)` is broken under numpy 2**: it evaluates
   `rgb_img[:, :, 1] * (2**8)` on a `uint8` array, raising
   `OverflowError: Python integer 256 out of bounds for uint8`. The official segmentation path does
   not run at all on this stack; probes here go through `mujoco.Renderer` directly.
4. **MuJoCo's segmentation buffer is `(H, W, 2)` = (object id, object type)**, not (type, id). The
   swapped reading is plausible, silent, and reports every pixel as background — it cost one probe
   here (`probe6` → `probe8`).
