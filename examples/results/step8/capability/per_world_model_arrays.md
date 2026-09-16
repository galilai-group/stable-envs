# Step 8 — Capability-gated SWM leaves across Workstreams A/B/C, and the minimal shared capability

**Analysis only.** Nothing is implemented, nothing outside this file is modified. Aggregates the
three Wave-1 VariationMap workstreams under Sami's ruling: *do not classify a semantically required
SWM leaf as unsupported merely because supporting it requires a renderer change; first aggregate
A/B/C evidence; if multiple families need the same per-world model-array capability, implement one
minimal shared capability.*

**Headline, stated first because it changes the shape of the decision:** the "per-world MODEL array"
capability Workstream C escalated **already exists in this repository and in `mujoco_warp`**. It is
`mjw.put_model(m, batch_sizes=...)`, used today by five families
(`stable_envs/envs/fetch.py:458`, `robocasa.py:349`, `dmcontrol.py:2471`, `metaworld.py:785`,
`humanoidbench.py:530`, `ogb_manip.py:366`), and the render megakernel already indexes it per world
(`stable_envs/render/mjwarp_lightspec.py:611`:
`light_diffuse_worldid = light_diffuse[worldid % light_diffuse.shape[0]]`). No renderer *capability*
is missing for mechanism class (i); what is missing is per-cell wiring in `ogb_manip.py`.

---

## 1. Every capability-gated leaf across A/B/C

Mechanism classes: **(i)** per-world model array in the mjwarp context; **(ii)** per-world/per-value
texture; **(iii)** alpha/translucency in the kernel; **(iv)** re-derived lighting law; **(v)** other.

| # | swm id(s) | leaf | upstream write (measured) | upstream pixel effect | class | ours today |
|---|---|---|---|---|---|---|
| 1 | `swm/OGBCube-v0`, `swm/OGBScene-v0` | `light.intensity` | `light_diffuse` row 0, nothing else | lights both halves of the frame; OGBMaze analogue 43,931/50,176 px at maxdiff 84 | **(i)** + plate re-bake | `unsupported` (C findings.md:116-119) |
| 2 | `swm/OGBCube-v0`, `swm/OGBScene-v0` | `camera.angle_delta` | `cam_mat0` + `cam_quat` on `front_pixels` | whole frame (camera move) | **(i)** + plate re-bake | `unsupported` |
| 3 | `swm/OGBCube-v0`, `swm/OGBScene-v0` | `cube.size` | `geom_size` on 2 MOVER geoms (+12 derived inertia arrays) | mover silhouette | **(i)** | `unsupported` |
| 4 | `swm/OGBCube-v0`, `swm/OGBScene-v0` | `floor.color` | `tex_data` only (builtin `grid` rgb1/rgb2); floor is STATIC | plate-only | **(ii)** plate | `unsupported`; patch in flight |
| 5 | 12 DMC ids / 13 cells (Acrobot, BallInCup, Cartpole×2, Cheetah, Finger, Hopper, Humanoid, Manipulator, Pendulum, Quadruped, Reacher, Walker) | `floor.color` | `tex_data` only — **the same builtin `grid` rgb1/rgb2** | 6,408–321,610 px over 8 states | **(ii)** | `unsupported`, 0 % of range reachable (A findings.md:77-84) |
| 6 | 12 `swm/Fetch*-v3` | `background.color` **skybox half** | skybox texture + `floor_mat` geoms | floor 9.2–9.4 %, **skybox a further 61.7–73.0 %** | **(ii)** plate | `mapped` to `floor.color` with a stated residual (B findings.md:70-80) |
| 7 | 12 `swm/Fetch*-v3` | `rendering.transparent_arm` | material alpha | 9.3–9.4 % of pixels, maxdiff 136 | **(iii)** | `unsupported` |
| 8 | `swm/OGBMaze-v0` | `light.intensity` | `light_diffuse` row 0 | 43,931 px at maxdiff 84 (intensity 0 and 0.5); default 1.0 moves 0 | **(iv)** | `unsupported` |
| 9 | `swm/OGBCube-v0`, `swm/OGBScene-v0` | `cube.color` | `geom_rgba`, 2–3 MOVER geoms, **no recompile** | mover colour | **(v)** none — array already per-world | `unsupported` only because the cell's style space is empty |
| 10 | `swm/OGBScene-v0` | `lock_color` | `mat_rgba`, 2 materials on 6 MOVER geoms | 1.0–2.2 % of pixels (`ogb_manip.py:70-71`) | **(v)** none | same as #9 |
| 11 | `swm/OGBMaze-v0` | `floor.pixel_encoding` | a switch we already render, inverted sense | switch | **(v)** — **already resolved** by `VariationMap.affine` (A findings.md:86-101) | no longer capability-gated |

Not in this table, deliberately: DMC `ball.size`, `spinner.color`, `target.shape`,
`floor.rotation_y`, `physics.link_length_*` are **cell-declaration** gaps (the cell declares no such
knob), not renderer-capability gaps — the raycaster already carries per-primitive `size`/`rgb`
(`dmcontrol.py:1722-1735`). Fetch `camera.angle_delta` and Rocket `environment.sky_color` are
**dead upstream** (measured 0 px) and must not be implemented.

Provenance for every row: `leaf_classification.json` in each workstream's
`examples/results/step8/variations-{dmc-control,fetch,newcells}/` directory, `reason` + `evidence`
fields, plus the probe JSONs each `evidence` string names.

---

## 2. Mechanism class by class

### (i) Per-world model array in the mjwarp context — **already a capability; 6 leaves, 2 ids, 1 family**

**Leaves:** #1, #2, #3 (OGBCube + OGBScene × light/camera/size). **Semantics are identical** across
the two ids — same MJCF family, same `_context` in `ogb_manip.py`, same three arrays.

**It is not missing.** `mujoco_warp._src.types.Model` carries a leading batch dimension on
`geom_dataid`, `geom_matid`, `geom_size`, `geom_rgba` (`docs/research/robocasa.md:977-980`, which
also records that `refit_scene_bvh` reads `geom_size` **per world**), on `light_diffuse`, `cam_pos`,
`cam_quat` (`stable_envs/envs/fetch.py:458`, `robocasa.py:349`) and on `body_pos`/`body_quat`/
`site_pos` (`docs/research/adroit.md:52-55`). The kernel's per-world index is at
`mjwarp_lightspec.py:611`. Fetch already ships the full pattern end to end: batched arrays at
`fetch.py:458`, one hoisted host→device camera upload at `fetch.py:774+`, one launch for the whole
batch (`fetch.py:704-707`).

**Minimal shared implementation:** extend `ogb_manip.py:366` from
`batch_sizes={"geom_rgba": nworld, "mat_rgba": nworld}` to also carry `light_diffuse`, `cam_pos`,
`cam_quat`, `geom_size`; add a per-world upload alongside `_upload_button_colours`
(`ogb_manip.py:774-790`) written in exactly its shape (`wp.to_torch(c.wm.<array>)` +
`index_copy_`); and extend `PLATE_PATHS` / `_plate` key for the two leaves that also move the static
plate (`light.intensity`, `camera.angle_delta`), copying `fetch.py:639-651` and
`fetch.py:723-735`. `cube.size` touches no plate (movers only).

**Invariants.** (a) The batch/chunk split is a property of the *context*, not of the appearance —
the batch must still be grouped by plate key and rendered in `MJW_CHUNK` pieces, never split per
style value (`fetch.py:706-708`). (b) `_CTX_CACHE`/`_PLATE_CACHE` keys must gain the appearance
vector, quantised, so two requests that compile to the same picture share one bake (the in-flight
patch already does this: `_floor_key`, `/private/tmp/stable-envs-step8-c-followup`). (c) The camera
must be written as `cam_pos`/`cam_quat` and left to `mjw.camlight` to derive `cam_xmat`, as Fetch
does — do not batch `cam_mat0`. (d) **Default byte-identity:** at the registered default every
batched row is the pristine value broadcast, so the uploaded arrays are bit-equal to today's shared
arrays and the frame must be byte-identical. This is testable and was the exact form of C's own GPU
evidence (C findings.md:64-72, three cells IDENTICAL).

**Cost: S–M.** S for `light.intensity` and `cube.size` (one array each, upload mirrors an existing
method). M for `camera.angle_delta`, because it moves the plate *and* the mover pass and the plate
is "a picture taken FROM this camera", so the two must agree to the pixel — Fetch's `_camera` /
`_camera_batch` is the worked precedent and can be lifted almost verbatim.

**GPU validation:** (1) default-style digest identity vs baseline on the same card/job; (2)
per-leaf isolation — each leaf moves pixels and only its own style slice, other dims unchanged
(C's Rocket table is the format); (3) Gate A re-run with the new `style_dim`/`style_digest` — these
cells are certified with an **empty** style space today (`ogb_manip.py:607-620`), so adding the
first style dims is a declaration move needing a deliberate re-baseline, not a quiet edit
(C findings.md:121-125); (4) paired throughput check at the default style, since the added uploads
are per chunk and must not show outside noise.

### (ii) Per-world / per-value texture — **26 leaves, 26 ids, 3 families; semantics identical in 2 of 3**

**Leaves:** #4 (OGB `floor.color`, 2 ids), #5 (DMC `floor.color`, 12 ids), #6 (Fetch
`background.color` skybox residual, 12 ids).

**#4 and #5 are the same operation.** Both write the **builtin `grid` texture's rgb1/rgb2** and
recompile upstream; both change **exactly one** model array, `tex_data`; in both the floor is
STATIC, i.e. plate-only. The in-flight Workstream-C patch already proves the method and the result:
substitute the rgb1/rgb2 texels in a *private* copy of `tex_data` and call `mjr_uploadTexture` — no
recompile, no second context — and the plate is **byte-identical to letting swm recompile the whole
model at the same `floor.color`** (0 bytes of `tex_data` differing, 0 of 50,176 px, both cells;
`_grid_texture` docstring in `git diff a09b407 HEAD -- stable_envs/envs/ogb_manip.py` at
`/private/tmp/stable-envs-step8-c-followup`). It also pins the quantisation: MuJoCo's texture
compiler **truncates** `int(255*v)`, it does not round (`mj_quantise`, same diff).

**The DMC split matters.** Of the 12 DMC ids, **7 (8 cells)** are `DMControlSuite` parity cells with
a baked plate whose cache key is *already* the whole variation vector (`dmcontrol.py:1443-1458`) —
the same fix lands there. The other **5** (Cheetah, Walker, Hopper, Humanoid, Quadruped) are
`DMControlTrackCom` (`dmcontrol.py:309-319`): trackcom camera, **no plate**, whole scene handed to
mjwarp, textures uploaded at `create_render_context`. There a per-value floor texture *is* one
context per value and stays blocked.

**#6 (Fetch skybox) is the same class but a different texture, and it is cheaper than B believed.**
B's reason — "mujoco_warp uploads textures when a render context is built, so a per-value skybox is
one context per value" (`fetch.py:589-592`) — is true of the *mjwarp* pass, but the Fetch skybox is
not drawn by the mjwarp pass in the delivered frame: the floor/table/**skybox** are the MuJoCo-baked
plate (`fetch.py:653-654`, `parity_note` at `fetch.py:545`), the mjwarp result is composited only
where the mover mask is set (`comp = torch.where(mv.unsqueeze(-1), rgb, pl)`, `fetch.py:763`), and
the plate cache key is already `PLATE_PATHS` (`fetch.py:503-506`, `fetch.py:639-641`). So a
per-value skybox is a **texel substitution + `mjr_uploadTexture` in `_bake_plate` plus one entry in
`PLATE_PATHS`** — the identical shape of change as #4/#5, and no extra render context.
**This claim is inferred from the composite path and must be measured before it is relied on.**

**Minimal shared implementation:** one helper, e.g.
`stable_envs/render/mjwarp.py::retexture(m, texid, {colour -> colour})` + `mj_quantise`, paired with
a "rebind the texture on this GL bake context" call; three call sites (`ogb_manip._bake_plate`,
`dmcontrol._bake`, `fetch._bake_plate`), each adding its leaf to the family's existing plate cache
key. **The one genuinely new thing** is DMC's `_bundle` policy: one `mujoco.Renderer` per
(domain, size), kept for the life of the process, with the comment "only a TEXTURE edit would [need
a second GL context], and none is offered" (`dmcontrol.py:1041-1044`). A texel edit + upload on the
*existing* context is exactly the escape that comment does not consider, and is what the OGB patch
does; a second context is not required.

**Invariants:** the private `tex_data` copy must never be written back to a shared `MjModel`
(DMC's `_bundle` model is shared by every instance and mutated by every bake, `dmcontrol.py:419`);
masks must be taken once from the *default* texture so two requested colours colliding cannot
corrupt them; `markrgb` texels must be left alone; the plate key must use the **quantised** bytes;
and at the registered default the substitution must be the identity — swm's `init_value` for DMC
`floor.color` *is* `common/materials.xml`'s own grid texture (A findings.md:81-83), so the default
plate hash must not move.

**Cost: M** for OGB (patch essentially written) and Fetch (one texture, one key entry); **M–L** for
the 7 DMC parity cells, because it touches the per-(domain,size) `Renderer` policy and 8 certified
cells' style declarations at once. **L/blocked** for the 5 trackcom cells.

**GPU validation:** default plate sha256 identity per cell (DMC already tracks this,
`dmcontrol.py:1150`); a bake-vs-recompile byte-equality test at ≥3 probe colours (the OGB patch's
existing form, extended per family); isolation per leaf; Gate A with the new style dims; paired
throughput — the risk here is **cache thrash**, not kernel cost: a batch with N distinct floor
colours bakes N plates, so the throughput check must include a high-cardinality style batch, and
`_PLATE_CACHE_MAX` (32 in `fetch.py:186`) must be re-examined.

### (iii) Alpha / translucency in the kernel — **12 leaves, 12 ids, 1 family**

Fetch `rendering.transparent_arm`. `mujoco_warp`'s render kernel reads `color[0..2]` and packs a
**constant 255** (`stable_envs/render/mjwarp.py:19`, `fetch.py` `parity_note` ~line 540). There is no
alpha path to parameterise; adding one is an upstream kernel change, squarely inside plan §2's "no
renderer rewrites". **Genuinely unsupported. `DOCUMENTED_LIMITATION`.**

### (iv) Re-derived lighting law — **1 leaf, 1 id, 1 family**

OGBMaze `light.intensity`. `envs/ogbench.py` does not run mjwarp for the agent: it composites an
analytic raycast over a MuJoCo-baked plate under a **fitted** shading law whose four constants were
least-squares'd over 13,069 agent-interior pixels at the default intensity
(`ogbench.py:74-89`). Scaling `light_diffuse` is not demonstrably the same operation the official
renderer performs, so honouring the leaf means re-deriving the law as a function of intensity and
re-certifying the family's parity floor. **One family only → one-off by construction.** Cost **L**
(a new fit + re-certification, not a code change). Recommend deferring as a
`DOCUMENTED_LIMITATION`; note the default (1.0) already moves 0 px, so the registered default is
exact and only the *range* is gapped — the same shape as A's `floor.color` argument.

### (v) Other — **3 leaves needing no capability, 1 leaf already fixed**

`cube.color` (×2) and `lock_color` (×1) write `geom_rgba`/`mat_rgba`, which `ogb_manip.py:366`
**already batches per world** and `_upload_button_colours` already writes per world. They are
`unsupported` purely because the cells' style space is empty and adding the first dim moves a
certified declaration. Cost **S**; the work is a re-baseline, not an implementation.
`floor.pixel_encoding` is no longer gated: `VariationMap.affine` (A findings.md:86-101) expresses
`ours = 1 - theirs`.

---

## 3. What stays genuinely unsupported even with the shared capability

| leaf | ids | why |
|---|---|---|
| `rendering.transparent_arm` | 12 Fetch | no alpha path in the mjwarp kernel; `mjwarp.py:19` |
| `floor.color` | 5 DMC trackcom (Cheetah, Walker, Hopper, Humanoid, Quadruped) | no plate; textures are uploaded at context creation, so a per-value floor is one context per value (`dmcontrol.py:2466-2480`, `dmcontrol.py:309-319`) |
| `light.intensity` | `swm/OGBMaze-v0` | fitted lighting law, class (iv); not a capability gap |
| `agent.color` | OGBCube/OGBScene | **dead upstream** — inverted dirty flag in `CubeEnv.modify_mjcf_model` (C findings.md:51-53) |
| `environment.sky_color` | `PFRocketLanding-v0` | dead upstream, re-measured 0/691,200 px |
| `camera.angle_delta` | 12 Fetch | dead upstream — wrapper moves model cameras, env renders the free camera |
| DMC `ball.size`, `spinner.color`, `target.shape`, `floor.rotation_y`, `physics.link_length_*` | various | cell-declaration gaps, not renderer gaps; each is a separate semantic decision |

---

## 4. Recommendation

**Semantically required** here means: a *registered SWM appearance leaf* that moves *official*
pixels. Rows 1–7 and 9–10 qualify. Dynamics/state leaves (`block.mass`, `agent.start_position`,
`cube.start_position`, every `physics.*` density/COM leaf) do **not** — they are dimensions of our
state or of a reference integrator, correctly `not_applicable`/`unsupported`, and nothing here
changes them.

**The single smallest capability that unlocks the most semantically required leaves across ≥2
families is (ii): per-value texel substitution on the plate bake (`retexture` + `mj_quantise` +
`mjr_uploadTexture`), with the family's existing plate cache key extended.**

- It unlocks **21 leaves across 21 swm ids in 3 families**: OGB `floor.color` (2), DMC parity-tier
  `floor.color` (7 ids / 8 cells), Fetch `background.color`'s skybox residual (12) — the single
  largest appearance term in the Fetch family at 61.7–73.0 % of the frame.
- It requires **no new renderer mechanism**: the method is already implemented and measured
  byte-exact in the in-flight Workstream-C patch, and all three families already key a plate cache
  on the variation vector.
- Its one policy change is DMC's one-`Renderer`-per-(domain,size) rule, and the change is *weaker*
  than A assumed: an in-place texel upload on the existing context, not a second context.

**Second, and nearly free: (i) is not a capability at all.** The per-world MODEL array C escalated
already exists (`put_model(batch_sizes=...)`, five families use it, the kernel indexes per world).
Wiring it into `ogb_manip.py` unlocks **6 more leaves on 2 ids**. Because this is one family it is
formally a one-off — but it is a one-off that *reuses* the shared capability rather than inventing a
hack, which is what the ruling asks for. Recommend doing it directly after (ii).

**Cheapest of all: the 3 leaves in class (v)** need no code beyond a style-declaration re-baseline.

**One-off, and recommend deferring:** OGBMaze `light.intensity` (class iv — a re-fit and
re-certification, one family, default already exact).

**Do not implement:** class (iii), and every dead-upstream leaf in §3.

**Caveat to verify before acting:** the Fetch skybox conclusion in §2(ii) is inferred from
`fetch.py:653-654` + `fetch.py:763` (skybox lives in the MuJoCo-baked plate; mjwarp output is used
only under the mover mask). It contradicts the *reason* recorded at `fetch.py:589-592`, so it needs
one measurement — bake a plate with a substituted skybox texture and diff against a full recompile —
before it is quoted as fact.
