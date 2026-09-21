# Step 9 — leftover items after v0-draft

Tracked against `v0-draft`. These are not release blockers and are not Step-9
regressions. Each row is a different *kind* of remaining work: a freeze you
would choose to reopen, an upstream API, hardware-dependent numerics, or a
process-global EGL constraint.

User-facing summaries stay in [limitations](../concepts/limitations.md) and
[worldmodel](../concepts/worldmodel.md). This file is the tracker: why it was
left, what a fix actually is, and how you would know it is done.

Discovery-naming shims from an earlier draft of this list (`envs/craftax_full.py`,
`ogb_manip`, `swm_nav`, `core/`, `compat/`) are gone: `discover()` loads
`_craftax_full` explicitly, and the unpublished Step-8 modules were deleted
before the first tagged 0.1.0.

| # | item | class | in-tree afternoon patch? |
|---|---|---|---|
| 1 | Fetch `background.color` skybox | product freeze | no — re-certify Fetch pixels |
| 2 | Native `swm.World(env=snv.PushT(...))` | upstream API | no — change stable-worldmodel |
| 3 | `HumanoidmazeGiantNavigate` CUDA process-repro | hardware numerics | investigate; may not close |
| 4 | EGL contexts never released | platform constraint | no — `close()` is unsafe here |

---

## 1. Fetch `background.color` — skybox half

**Status.** Variation map **partial**. Floor half is mapped and reproduced;
skybox residual is measured and not rendered.

**Why v0 left it.** Supporting the skybox would move the family's frozen
default frames. Step 9 does not re-freeze certified pixels.

swm's "background" is two things at once: the `floor_mat` geoms **and** the
skybox texture (recoloured by overwriting `model.tex_data` and re-uploading).
Measured split at 480 px (`docs/research/step8_variations_fetch.md`):

| scene | floor only | floor + skybox | skybox alone |
|---|---|---|---|
| Reach | 9.18 % | 81.74 % | 73.03 % |
| Push | 9.18 % | 81.90 % | 73.19 % |
| Slide | 9.45 % | 70.69 % | 61.68 % |
| PickAndPlace | 9.18 % | 81.73 % | 73.03 % |

A second, independent reason not to offer a skybox knob: `mujoco_warp` uploads
textures when a render context is built, so a per-value skybox costs **one
render context per value**. `FetchBase.make_variation_space` already records
that.

**Path.** Implement the skybox (or accept a different mapping), re-measure all
twelve Fetch ids, re-freeze default frames and the variation-map
classification (`partial` → `complete` if the residual is gone). That is a
Step-8-style certification event, not a silent patch under `21cb306`.

**Done when.** Fetch `background.color` is `complete` on the support matrix;
frozen default frames and the Fetch variation study are updated; the context
budget is still bounded (or the study names a new cost).

---

## 2. Native `World(env=snv.PushT(...))` on the sibling package

**Status.** Not a stable-envs API. Installed `stable-worldmodel` takes
`env_name: str` and always `gym.make`s it. `stable_envs.interop` is gone:
no `StableEnvGym`, no `BatchedEnvPool`, no `snv.make("swm/...")`.

**Why v0 left it.** The dependency direction is the other way. stable-envs
does not wrap, register, or adapt SWM. If a later SWM release wants to
construct `World` from a live `snv.Env`, that change belongs in
`stable-worldmodel`.

Historical renderer-swap evidence lives under
[research artifacts](artifacts/README.md), not in `examples/`.

**Path.** Change `stable-worldmodel.World` (or equivalent) to accept a live
env. Do not restore an interop layer here.

**Done when.** A released SWM accepts a live `snv.Env` (or equivalent). That
is an SWM change, not a restoration of `stable_envs.interop`.

---

## 3. `HumanoidmazeGiantNavigate` on CUDA — not process-reproducible

**Status.** `DOCUMENTED_LIMITATION` on at least `gpu1503` (RTX A6000,
driver 575.57.08). Passed on an A40 (`gpu1504`). CPU arm of the same cell is
fine. Not a Step-9 regression: the baseline tree already emitted two digests
across jobs.

**Why v0 left it.** The patch under test did not change the arithmetic. The
oracle difference is not evidence about Step 9.

**Mechanism.** `raycast3d._BOUND_SLACK_NOISE`: camera ~100 m from a 5 cm
capsule, so `c = olx² + oly² − r²` loses the radius in float32 and `disc > 0`
is decided on rounding noise. Step-6 job **6348207** moved one pixel of
3,211,264 by 22/255 on this exact configuration. Probe:
`scratch/step9/render-3d/hmg_determinism.py`. Jobs: 6464319 / 6464500 /
6464601.

**Path.** Isolate Triton vs torch vs `PRIMITIVE_BOUNDS=False`; pin CUDA
determinism; or keep the cell out of the process-repro gate on that hardware.
May be card/driver reduction order rather than a logic bug.

**Done when.** Same frozen corpus, three fresh processes, one digest on the
cards that matter; **or** an explicit gate carve-out naming the cell, device,
and hardware, with the baseline's self-disagreement still on record.

---

## 4. EGL / GL contexts are never released

**Status.** Intentional. `close()` drops MJWarp tex_lod keepalives and
nvdiffrast CUDA state. It must not call `mujoco.Renderer.close`,
`GLContext.free`, or `eglDestroyContext`. Enforced by
`tests/unit/test_render_resources.py` (source scan) and
`stable_envs/testing/resource_guard.py` (zero tolerance).

**Why v0 left it.** Releasing one EGL context has corrupted every other live
EGL context in the process — silent pixel damage in an unrelated env. Four
separate incidents in this repository. Fetch `_RENDERER_CACHE` and DMC
`_ENV_CACHE` are unbounded on purpose: one `mujoco.Renderer` per
(scene, resolution), built once.

**Path.** Not `close()` in this process. Options: one GL context per process
and exit to free it; subprocess isolation for teardown; or an upstream MuJoCo
/ EGL fix that makes contexts independently releasable. Until one of those
exists, "leak until process exit" is the correctness policy.

**Done when.** A release path is demonstrated that does **not** move pixels in
any other live env in the same process, under the resource-guard, across a
sweep that constructs and closes several families.
