# Cube and Scene: completed semantic and renderer acceptance

`swm/OGBCube-v0` and `swm/OGBScene-v0` independently pass their required gates.
The semantic-recovery matrix is **35 VERIFIED / 7 INCOMPLETE**. Oscar retry
**6669909** completed with exit **0:0** on `gpu3002`, NVIDIA L40S, in 6m43s.
It produced `OGB_CUBE_GPU_GATE_PASS`, `OGB_SCENE_GPU_GATE_PASS`, and
`OGB_CUBE_SCENE_GPU_GATE_PASS`. No external gate remains pending for this checkpoint.

The separate acceptance records are [Cube](semantic_recovery_evidence/ogb_renderer_retry/Cube-acceptance.json)
and [Scene](semantic_recovery_evidence/ogb_renderer_retry/Scene-acceptance.json).
Each references its own CPU results, frames, comparisons, and resource measurements.
Neither promotion depends on the other family's pass.

## Checkpoint and provenance

Branch: `repair/environment-semantics`; worktree: `/private/tmp/stable-envs-semantic-recovery`.
The CPU sequence started clean at `72612563590f12a081d940394496795a6c024577`.
The retry started clean at `b59df7599eac1951dfd1ac5682b41452db0ac8f5`.
CPU implementation: `14456339ec7f1b28ad698fc3158fe9e4ed8f6d1e`;
CPU evidence: `72612563590f12a081d940394496795a6c024577`.
Pinned immutable SWM oracle: `176d54532b3c0a251a4f509b74f2fdacc143a63d`.
The earlier dependency/RNG/documentation checkpoints are recorded in the
[CPU closeout](ogbench_cpu_closeout.md).

Commits created in the sequence, in order:

| Commit | Content |
|---|---|
| `0aaa72fed5cc1a63c46af02aa448bebf6e279838` | CPU bookkeeping and restore performance |
| `2f4b2a6e5c0bc1ed8455d8235c1eb0e26ce70ff7` | Unified profile, RenderState and frozen-renderer integration |
| `457afa0b401a3fa757c94eb8a8b6a3de3074585c` | Local renderer evidence and Oscar runner |
| `294675f039e9b7f867c4bff15aac2b5f2e572bf6` | Complete independent CPU test selectors; tested source |
| `b59df7599eac1951dfd1ac5682b41452db0ac8f5` | Failed first job and corrected bundle preflight |
| `3af3c63446f6b6ef1f261a8fb0b85e584db4e749` | Verified retry GPU evidence |

This report and the promotion records belong to the subsequent containing commit;
its exact SHA is supplied in the final handoff rather than a self-referential hash.
The worktree is to be clean at handoff; no push is performed.

The first job, **6669720**, failed during test collection because its archive
omitted a transitive fixture. No renderer ran in that attempt. Its actual failure
archive and marker remain unchanged in the [previous closeout](ogbench_renderer_closeout.md).
The user explicitly authorized further submissions. Exactly one retry was submitted,
using the corrected complete tracked-source archive, the same production source,
runner, oracle, comparisons, and profile. Both attempts used `gpu` / `norm-gpu`
and disabled requeue. The retry required no further runtime fix.

## Profile and supported scope

The selected architecture is **one unified MuJoCo 3.12.0 profile**. Both CPU
semantics and GPU rendering passed within that profile. Dynamics remain
**DELEGATED_CPU**. The earlier exact CPU acceptance under 3.8.1 is preserved;
there is no claim of bit identity between MuJoCo versions or of GPU physics parity.

Install `stable-envs[ogbench-manipulation]` for CPU dynamics or
`stable-envs[ogbench-manipulation-gpu]` for the MJWarp renderer. Both extras pin
OGBench 1.2.1, MuJoCo 3.12.0, NumPy 2.2.6, Gymnasium 1.2.3 and dm-control 1.0.41;
the GPU extra also pins mujoco-warp 3.12.0 and warp-lang 1.17.0.
The accepted Linux run used Python 3.12.8, Torch 2.14.0+cu126 and TensorDict 0.14.2.
Its [profile record](semantic_recovery_evidence/ogb_renderer_retry/remote/evidence/profile.json)
asserts every required version equals the installed version.

Acceptance covers the registered Cube `single` and Scene `scene` task defaults,
state observations, supported task IDs and CPU-tested reward-task routes, seeds
0/17/42, public batches 1/3 and float32/float64 actions. Rendering is bounded to
`front_pixels`, Cube 224 and Scene 200 output sizes, supersample 1, MJWarp on CUDA.
CPU rendering, historical dataset sampling, data collection, multiview and
render-goal modes remain explicit refusals. The named public State is exhaustive;
RenderState is a separate owning numeric TensorDict and cannot restore dynamics.

Cube has 11 mapped public Variation leaves and is **READY** in the separate
historical Variation inventory. Scene has eight mapped leaves and remains
**PARTIAL** there: `cube.start_position`, `cube.start_yaw`, `cube.goal_position`
and `cube.goal_yaw` are explicitly refused. They are not declared dead upstream.
This broader inventory distinction does not invalidate Scene's accepted registered
task-default scope. Both use `DELEGATED_CPU`; all other readiness rows are unchanged.
The source's conditional agent-color dirty behavior, including its inverted
`np.allclose` test and pending-versus-compiled distinction, is preserved.

## RenderState and unchanged drawing

The [schema](semantic_recovery_evidence/ogb_renderer_closeout/renderstate-schema.json)
has 14 Cube leaves and 16 Scene leaves. Both contain a numeric renderer signature,
size, supersample and camera; qpos (21/25); goal mocap position and quaternion;
compiled cube size, agent color, camera angle, light and floor colors; current
cube color and target alpha. Scene additionally contains lock colors and button
bits. The schema records every exact key, shape and dtype. Leaves own their
storage and support public batching. RNG, task, lifecycle and physics caches are
excluded. Validation checks family/profile compatibility and malformed inputs;
`set_state(RenderState)` refuses explicitly.

The frozen Scene constructor omitted `env_type` when constructing OGBench Scene.
The narrow model-provider caller now supplies the compatibility-configured model;
no constructor or drawing algorithm in the frozen renderer was edited. Geometry,
materials, lights and the selected camera matched independently configured source
inputs in all 35 local cases. The caller also supplies the recorded mocap quaternion.

Frozen renderer SHA-256, unchanged:
`bac4b1ea3e29a04faa5eeec33de0c212cb87bbf488ae0466ea762bd3a140576a`.

## Actual acceptance measurements

| Gate | Cube | Scene |
|---|---:|---:|
| Linux unified-profile CPU tests | 45 passed | 48 passed |
| GPU frame cases | 13 passed | 22 passed |
| Full State versus named RenderState versus frozen baseline | exact | exact |
| Different pixel channels / maximum absolute error | 0 / 0 | 0 / 0 |
| Created EGL contexts / released contexts | 32 / 0 | 51 / 0 |
| Retained process-lifetime plate rigs | 19 | 29 |

The CPU selections execute 93 cases covering 92 unique tests; one interleaving
case is intentionally run for both families. There are no CPU skips or failures.
Renderer cases include all three seeds, representative tasks, appearance and size
endpoints, Scene button patterns, restored drawer/window/cube placements, and
real contact trajectories. All required visual changes changed pixels;
visual-only fixtures held physics constant. Full/private rendering, repetition,
batch order, chunk size, public batch three, independent environments, input/live
nonmutation, no global RNG mutation and no hidden step/reset calls all passed.

The exact baseline is the unchanged frozen drawing configured independently from
the pinned SWM oracle. Direct source MuJoCo images are also retained. Their exact
pixel fraction ranges are **0.864258–0.882514** for Cube and
**0.695200–0.830525** for Scene. These existing visual-tier differences are not
relabelled as exact source-pixel fidelity, and no threshold was relaxed.

The frozen process-lifetime EGL policy remains in force. Instance close releases
instance tables; private EGL plate rigs are retained until process exit. No context
was freed or failed release during the gate. Repeated and independent calls with
existing configurations added no rig. New static configurations can retain further
rigs, so unbounded configuration sweeps are not claimed to have bounded memory.
Dynamic MJWarp caches retain the frozen LRU behavior. These measurements include
the independent source/baseline rigs, not just the candidate.

Restore performance is unchanged: same-configuration median restore is Cube
181.389 ms (B1) / 591.219 ms (B3), Scene 211.752 / 705.623 ms, about 150–193 times
normal step. [CPU closeout](ogbench_cpu_closeout.md) includes warm/cold median,
p95, throughput, changed-configuration costs and the proposed later fast path.
No restore optimization was included in this checkpoint.

## Evidence, commands and preservation

The actual remote archive was downloaded and checked against the remote SHA-256.
[Independent verification](semantic_recovery_evidence/ogb_renderer_retry/verified-gpu.json)
then checked all **158 remote file hashes**, reloaded every frame, recomputed exact
comparisons and source residuals, and checked CPU XML counts and terminal markers.
The committed [GPU manifest](semantic_recovery_evidence/ogb_renderer_retry/GPU-SHA256SUMS)
seals 173 files; the acceptance manifests add per-family evidence chains.

| Artifact | SHA-256 |
|---|---|
| Tested source archive | `a002c3aa72ea16a841858ef1d53123009f40ff681e2afa0dcf594235b9fe46c2` |
| Immutable oracle archive | `9d8fefcd19337522f3f5cf37379703c48fc2974e958f22f9c849a8c3a9b33b7d` |
| Oscar runner | `88aa1d237de31ab2f9a783d1ab26473e9186af39e1b1cdd1014dc7272790b134` |
| Bundle metadata | `18e0a6f194a26e9789e0535d2b4c14861b1d20ab3e910f9329d280e6f3f39beb` |
| Downloaded GPU evidence archive | `6f584dc2f29f592b75c77eaf2237e4879d7b07eb5ec5add517fd8a5147581cd4` |

Earlier complete [commands and results](semantic_recovery_evidence/ogb_renderer_closeout/commands.json)
cover CPU closure (86), unified CPU (86), unified CPU plus RenderState (92), local
model inputs (35), fixtures (35), and the original regression/documentation checks.
Retry and final commands are recorded in
[final commands](semantic_recovery_evidence/ogb_renderer_retry/final-commands.json).
The original initial readiness failure is retained: a test-only frozen Scene helper
still used the incompatible constructor. Routing that helper through the actual
configured model provider corrected the test without changing production or the
frozen renderer. Final full-dependency readiness tests passed **96, with 2 skips**;
shared State/harness/readiness/taxonomy/matrix tests passed **131, with 11 skips**.
The latter environment lacks optional dependency extras; the full-dependency and
remote runs supply the required Cube/Scene coverage. Formatting, lint, whitespace
and strict documentation checks also pass.

The [preservation audit](semantic_recovery_evidence/ogb_renderer_retry/promotion-preservation.json)
verifies all 33 previous accepted rows and all 40 non-target rows in both semantic
records are unchanged, plus 170 prior artifacts, 11 terminal logs, Fetch/RNG/CPU/
dependency manifests, 217 prior renderer-checkpoint hashes and the 173 GPU hashes.
Historical mutable map/source entries are verified at their recorded checkpoint,
not misrepresented as unchanged current data. All 485 audited source files and
the immutable SWM checkout remain unchanged. All non-target Variation maps and
readiness rows match the pre-promotion snapshot.

Production and packaging have no changes from the retry starting tip. Fetch,
OGBench Maze, PushT, Craftax, Rocket, shared contracts and unrelated renderers remain
untouched. Production has no stable-worldmodel import; OGBench 1.2.1 is unmodified
and no complete SWM implementation was copied. The audit records expected versus
actual changes; the [complete sequence inventory](semantic_recovery_evidence/ogb_renderer_retry/changed-files-sequence.txt)
records every changed path. The [final manifest](semantic_recovery_evidence/ogb_renderer_retry/SHA256SUMS)
seals the resulting report, metadata, helpers, tests and new evidence.

The actual remaining incomplete registrations, verified from the matrix, are:

- `swm/PushT-v1`
- `swm/PushT-Discrete-v1`
- `swm/CraftaxClassicPixels-v1`
- `swm/CraftaxClassicSymbolic-v1`
- `swm/CraftaxPixels-v1`
- `swm/CraftaxSymbolic-v1`
- `swm/PFRocketLanding-v0`

The sequence stops for review here. There is no queued acceptance job to resume,
no additional family has been started, and no push has occurred.
