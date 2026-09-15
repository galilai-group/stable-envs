# Q11 for Sami — OGBScene's button cube-map: the three options, priced

`swm/OGBScene-v0` cannot render through MJWarp. This is the evidence and the cost of each option.
**No option has been implemented and none is recommended here; the decision is a fidelity decision.**

## What the crash is

Reproduced in two jobs, the second with one env per process so it is not contamination
(6395461, 6395736), then bisected (6396447, each configuration in its own subprocess):

| configuration | result |
|---|---|
| baseline | ❌ `Warp CUDA error 700: an illegal memory access` |
| `capture=False` (no CUDA graph) | ❌ — not a graph-capture bug |
| `skybox=False` | ❌ |
| `shadows=False` | ❌ |
| `textures=False` | ✅ renders |
| detach only the cube map, keep every 2-D texture and the skybox | ✅ renders |

**Root cause, read from `mujoco_warp/_src/render.py` and confirmed on the model.** `sample_texture`'s
MESH branch reads

```python
mesh_texcoord[mesh_texcoord_offsets[mesh_id] + mesh_facetexcoord[face_adr][i]]
```

with no guard. OGBScene's two button geoms wear mesh `stopbuttonrim`, which carries a **CUBE**
texture and **`texcoordadr = -1`, `texcoordnum = 0`** — no UVs at all — so that index is computed
from a negative offset and reads off the allocation. A cube-mapped mesh normally has no UVs, because
MuJoCo generates cube texgen from position and normal and never needs them; that is why the two
symptoms travel together. `OGBCube` has **zero** textured mesh geoms, which is why it does not crash.

## Option (b) — bake `T_btn` to its mean colour: **measured**

MuJoCo against MuJoCo, 8 real states, 200 px, so this is the cost of the approximation alone with no
MJWarp term mixed in (`q11_button_bake.json`, `probe_q11_button_bake.py`; needs no GPU).
`btn_metal`'s mean RGB is `(0.523, 0.516, 0.512)` — a mid grey, which is what a brushed-metal cube
map averages to.

| quantity | value |
|---|---:|
| the two button geoms | **36 px/frame = 0.090 % of the frame** |
| pixels changed by the bake | **55.5 px/frame = 0.139 %** |
| of which are the buttons themselves | 64.9 % |
| max abs difference | **37/255** |
| mean abs on the button pixels | 21.6/255 |
| mean abs over the whole frame | **0.023/255** |
| pixels worse than 8/255 | 0.110 % |
| pixels worse than 32/255 | 0.005 % |
| **`interior_exact` (baked vs official)** | **1.00000** |
| interior mean abs | **0.0** |

**The residual touches no interior pixel at all.** Every changed pixel fails the 3x3-flat test, so a
bake changes nothing the interior metric can see. For scale, the alpha `parity_note` already accepted
for Fetch is 1.2–1.5 % of pixels; this is **0.139 %**, roughly a tenth of it, and Fetch's did not
score 1.00000 on the interior.

**What option (b) actually costs in code, which is not zero.** `bake_primitive_textures` in its
shipped form refuses this material **twice**: it skips non-2-D textures, and it classifies a material
worn by MESH geoms as `mixed` and leaves it alone (MJWarp can UV a mesh — normally). So (b) is a
change to that shared helper, not a call to it. Smallest form: let a caller opt into baking a
material whose texture the sampler cannot address even on a UV-able geom — the same predicate
`texture_gaps` now computes. Roughly a dozen lines plus tests, in a file five families share.

## Option (a) — patch `mujoco_warp`'s mesh sampling

The guard is one condition in `sample_texture`: if the mesh has no texcoords, return the material's
flat colour instead of indexing `mesh_texcoord`. That is ~3 lines in a `@wp.func` in a third-party
package, so the real cost is not the edit but the route: a vendored patch in this repo (rejected
elsewhere as a fork), an upstream PR on someone else's release cadence, or a monkeypatch of the kind
`create_render_context` already applies in `stable_envs/render/mjwarp.py`. A monkeypatch is the only
one that lands inside Step 7, and it would be the second in that module. It also fixes the *general*
case — any textured UV-less mesh in any future family — where (b) fixes only this scene, and unlike
(b) it costs **no** pixels: the buttons would then sample the cube map the way MuJoCo does, which
option (b) approximates away.

## Option (c) — ship `OGBCube` only

Coverage goes to 39/40 rather than 40/40. Nothing else in Step 7 depends on OGBScene, and its census,
state contract, plate and camera are already measured, so (c) is reversible at any later date.

## The one thing worth weighing

(b) is cheap, measured, and provably invisible to the interior metric — but it is an approximation
applied to make a **crash** go away, and the frozen record would carry "these two button faces are a
flat grey because the renderer could not read their texture" rather than "MJWarp cannot sample a
UV-less mesh". (a) removes the cause; (b) removes the symptom on one scene at 0.139 % of pixels;
(c) removes the scene. The numbers say (b) is affordable. Whether affordable is the right test is
yours.
