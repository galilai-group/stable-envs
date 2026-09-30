# Rendering fidelity and performance campaign — 2026-09-28

Worktree: `repair/render-config-fixes`, starting at
`df80c9c54590afa9fc0482a4dcb241d21f6b63a3`. No push. This work does not
promote simulator semantics: the semantic matrix remains 39/42 verified.

The target is normalized RGB MSE <= 0.001, with 0.0005 as a stretch target.
Per-frame MSE is the mean squared RGB difference after division by 255.
The campaign records 1,000 frame observations per measurable registration,
mean/sample SD, maximum error, raw render timings, image hashes and full-State
witnesses. Duplicate State hashes are counted explicitly. Two symbolic Craftax
registrations expose no RGB output. PushT's two registrations still lack the
required source-compatible full-State witness and cannot earn a matched-State
rendering acceptance label.

Source CPU/GPU and candidate CPU/GPU run in separate processes. Source GPU
means hardware rendering of the source simulator where exposed, and JAX GPU
for Craftax. A GPU tensor obtained by running a CPU renderer is not accepted as
GPU rendering. Explicit OpenGL CUDA backends use NVIDIA GL and include their
internal host readback/upload in the measured call. CPU GL requires verified
Mesa software rendering. Cross-device Craftax comparisons require exact State
hashes; incompatible platform-specific States remain noncomparable.

The source pin is `176d54532b3c0a251a4f509b74f2fdacc143a63d`.
Oscar campaign root:
`/oscar/scratch/sboughan/se-semantic-recovery/render-parity-df80c9c5-20260928`.
Local launchers and downloaded summaries are under
`tmp/render-parity-20260928`. Each source revision and output directory is
immutable. The prior accepted evidence is preserved.

## Findings and implementation

Scene's edge roughness partly comes from differing anti-aliasing. Supersampling
alone changes worst-frame MSE from 0.00919953 to 0.00819573 at 2x and 0.00799482
at 4x. It does not repair the larger effects. The static-background compositor
drops moving-object shadows, and MJWarp treats the moving translucent pane as
opaque. No z-fighting cause has been established by these probes.

Explicit `backend="opengl"` implementations now exist for Cube/Scene, Fetch,
and DMC trackcom rendering. They draw candidate-owned models from supplied
state, without invoking a source environment renderer or replaying physics.
They restore source shadows, transparency, anti-aliasing and reflections. Their
defaults remain unchanged pending the fidelity/performance tradeoff. Contexts
follow the existing process-lifetime policy to avoid invalidating sibling GL
contexts. Cube/Scene supplied-state tests passed on CPU and GPU (jobs 6755494,
6755495); Fetch/Walker CPU tests passed after adding closed-environment guards
(6755882, 6755883).

On the earlier stress corpus, explicit OpenGL's worst GPU/source-GPU error was
3.84e-10 for Scene (132 frames) and 0.000179 for Cube (57 frames). These are
bounded preliminary results, not the final 1,000-frame acceptance results.
The fresh 1,000-frame Fetch CPU errors were <= 0.000174 independently for all
12 registrations. PickAndPlace's three variants remain reset-only because
source rendering mutates continuation caches after actions; the CSV must say so.

A two-pass MJWarp composition probe restores moving shadows and blends the
moving alpha layer. A process-local shader ablation established that MJWarp
3.12's 0.3 shadow transmission is another discrepancy. That experimental
global change is **not** in production. The production Cube-only
`backend="mjwarp_fidelity"` instead compensates the static-background shadow
ratio locally. Its research implementation passed all 1,000 fresh Cube frames
without supersampling: worst MSE 0.000894665 (job 6756038). Production
batch-invariance and timing gates are separate: batching passed, including 257 rows and mixed appearances (6756180 and bounded-cache regression 6756452). The production 1,000-frame images are complete (6756179); independent image aggregation follows.
The same compositor approximation did not pass Scene's full stress set;
Scene therefore does not expose this backend.

Rocket has two concrete projection defects: visible links use the current
body pose while the camera uses a cached link pose; the old renderer inferred
one velocity lag for both. At the high-altitude stress checkpoints, differing
float precision and matrix multiplication order also move edges. A private
TinyRenderer world fed explicit numeric poses exactly matched all 204 source
frames (6755986). The corrected CUDA probe passed the 0.0005 stretch target
with worst MSE 0.000428221 (6756015). The implementation adds the missing body,
camera and RGBA leaves to Rocket's family-specific render projection; it does
not change shared State or simulator restoration. CPU and GPU both use those
RGBA leaves; the legacy pose/style leaves serve compact legacy consumers.
The CPU path never steps physics or calls the source environment renderer.

TinyRenderer's arithmetic is visible in the upstream
[rendering implementation](https://github.com/bulletphysics/bullet3/blob/master/examples/TinyRenderer/TinyRenderer.cpp)
and [float matrix implementation](https://github.com/bulletphysics/bullet3/blob/master/examples/TinyRenderer/geometry.h).
The installed PyFlyt camera code was also inspected directly on Oscar.
The production Rocket 1,000-frame gates are jobs 6756157/6756158; its CPU
State/lifecycle regressions passed (18 tests, job 6756187). No registration
promotion is implied by a successful probe.

## Performance constraints

Do not replace the fast defaults merely because OpenGL passes the image gate.
The same-GPU public RenderState benchmark (jobs 6755248) measured:

| Family, batch | MJWarp ms/frame | OpenGL ms/frame |
|---|---:|---:|
| Scene, 1 | 15.3175 | 7.4484 |
| Scene, 8 | 2.1894 | 1.9553 |
| Scene, 64 | 0.6526 | 1.2895 |
| Scene, 256 | 0.6439 | 1.2139 |
| Cube, 1 | 8.1305 | 4.3698 |
| Cube, 8 | 1.1533 | 1.0252 |
| Cube, 64 | 0.2756 | 0.6060 |
| Cube, 256 | 0.1875 | 0.5618 |

Thus OpenGL improves small-batch latency, but costs about 1.9x for Scene and
3x for Cube at batch 256. A research two-pass+2x-supersampling path similarly
cost 2.3x/3.2x at that batch. Cube without supersampling measured 0.442 versus
0.371 ms/frame (6755928), but this is a **lower bound** because the research
path omits public validation/grouping. It is not the production slowdown.
Production Cube timing (6756204, RTX A6000) measured 0.5496 versus 0.3315
ms/frame at batch 256, a **66% increase**, rather than the research lower bound's
19%. At batch 1 it measured 18.585 versus 14.813 ms. It remains optional.
Rocket's same-GPU before/after and exact batch consistency gate (6756254)
measured 0.4433 versus 0.4357 ms/frame at batch 256 (**1.75% increase**), and
5.485 versus 6.592 ms at batch 1 (**16.8% faster**). These numbers use the public
RenderState boundary. The 1,000-frame comparison CSV additionally includes
full-State projection, so its times are a different, explicitly labeled workload. Validation and configuration grouping
now transfer Cube/Scene metadata to CPU once, avoiding per-leaf GPU scalar
synchronization; the corresponding production benchmark includes that change.

## Export and remaining gates

`export_parity_campaign.py` independently reloads saved RGB arrays, verifies
their hashes, and joins only exact matching full-State/case/seed/step witnesses.
It reports six CPU/GPU/source comparisons, numeric mean/SD/max, timing mean/SD,
frame counts, unique State counts and backend names. Missing/failed rows are
never zero-filled. Acceptance requires all 1,000 frames below threshold on
both candidate devices. The gallery keeps a source-left/candidate-right worst
frame for each measured row. Twenty local exporter/verifier tests passed.

The current manifest is `tmp/render-parity-20260928/manifest-v6.json`.
The remaining main GPU comparison array 6756048 was released after the bounded
gates. No unrelated job was touched. The final independent export is 6756643, merged by 6756658 and packaged by
6756668 after the successful CUDA retries in 6756626.

Independent CPU and GPU-host source trajectories differed by floating-point
roundoff in Cube/Scene and Quadruped. The exporter rejected their full-State
hashes rather than treating nearby poses as the same State. New source-only
GPU runs install the recorded CPU numeric State into the actual pinned source,
recapture every full-State leaf exactly, and verify rendering nonmutation.
Cube/Scene models are constructed by the source from the original reset recipe;
no candidate model or candidate image is copied. Their compiled and pending
MJCF configuration is checked against the State. Quadruped uses its existing
accepted reference set_state bridge. Jobs: 6756505 and 6756466 respectively.
Earlier unsuccessful bridge revisions remain preserved (space accessor, metadata
capture dtype, and GL vendor-query ordering fixes).

The semantic acceptance matrix and shared State abstractions are unchanged.
No push was made.

## Selecting the measured backends

```python
import stable_envs as se

scene = se.make("OGBScene", device="cuda", backend="opengl")
cube = se.make("OGBCube", device="cuda", backend="mjwarp_fidelity")
walker = se.make("DMCWalkerWalk", device="cuda", backend="opengl")
image = scene.render(scene.to_render_state(scene.get_state()))
```

The explicit OpenGL backend also supports Cube/Scene and Fetch CPU rendering in
a Linux process configured for Mesa software GL before MuJoCo is imported.
The measured CPU launchers use Xvfb, `MUJOCO_GL=glfw`,
`PYOPENGL_PLATFORM=glx`, `LIBGL_ALWAYS_SOFTWARE=1`, and
`GALLIUM_DRIVER=llvmpipe`. Hardware OpenGL measurements use EGL and one assigned
NVIDIA GPU per process. All internal readback/upload costs are included.
The GPU defaults for Cube/Scene and Walker remain their existing fast paths.
Thus passing a row with an explicit fidelity backend does **not** mean the
registration's default renderer has reached that same error bound.

The generic `original_backend` text in older collector frame records is not a
device witness. The campaign role, GL vendor/renderer checks, CUDA allocation,
and package/hardware profile identify actual execution. For source replay
records, the State archives reside in the original source CPU corpus referenced
by the manifest and `corpus_sha256`; images and new timings reside in the replay
output. No source State archive is inferred from a candidate image.

## Completed results

[Readable 42-row CSV](semantic_recovery_evidence/renderer_parity_1000_20260928/render_differences_42.csv),
[numeric CSV](semantic_recovery_evidence/renderer_parity_1000_20260928/renderer_comparison_42.csv),
and [image gallery](semantic_recovery_evidence/renderer_parity_1000_20260928/images.html).

All 38 measurable RGB registrations pass normalized MSE <= 0.001 on every one
of their 1,000 measured frames on both candidate devices, with the explicitly
listed backends and coverage restrictions. 24 also pass <= 0.0005 on every
frame. 36 have per-registration means <= 0.0005 on both devices; Cheetah and
Hopper means are 0.0005066 and 0.0005405. PushT remains unestablished and
symbolic Craftax is NO_RGB. This does not establish parity of all defaults.

Scene's requested seed 17, case 20, step 0, frame 114 has **zero GPU/source-GPU
MSE** with the explicit OpenGL backend. Across all 1,000 Scene frames its
worst GPU/source-GPU error is 5.2544e-9; CPU/source-CPU is exactly zero. The
source CPU/GPU gap reaches 0.000163686. The image pair and its full-State
hash are preserved in `scene_requested_frame_114.png/.json` in the artifact
folder. Cube's optional batched fidelity path has worst GPU/source-GPU
MSE 0.000784674 and GPU/source-CPU MSE 0.000889995. Rocket CPU/source-CPU is
exactly zero; corrected GPU/source-CPU worst MSE is 0.000428221.

The artifact README explains timing scope, software/hardware profiles,
reset-only coverage, incompatible Craftax cross-device State witnesses,
cache-versus-recompute timing, and optional backend selection. Main CSV timings
include full-State projection/validation; the separate RenderState throughput
probe does not. Final source/production hashes and SLURM terminal evidence are
preserved. All 13 changed production files match Oscar source-v12. No push.
