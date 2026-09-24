# Cube/Scene renderer integration checkpoint

Both registrations remain **INCOMPLETE**. CPU semantics and exhaustive State passed;
renderer integration is implemented. Oscar job **6669720** failed at a test-bundle
packaging prerequisite; neither GPU renderer ran. GPU acceptance remains pending. The matrix
remains **33 VERIFIED / 9 INCOMPLETE**. No other family is in this checkpoint.

## Profiles and supported execution

The accepted CPU checkpoint used MuJoCo 3.8.1. The proposed unified profile is
MuJoCo **3.12.0**, OGBench **1.2.1**, NumPy **2.2.6**, Gymnasium **1.2.3**, and
dm-control **1.0.41**. Install the `ogbench-manipulation` extra for this CPU profile;
`ogbench-manipulation-gpu` additionally pins mujoco-warp **3.12.0** and warp-lang
**1.17.0**. Oscar uses Python 3.12 and CUDA-enabled Torch 2.14.0+cu126 inherited
from the existing isolated `se7-ogb` environment. The job records every package.

The complete 86-test CPU suite passed independently under both MuJoCo versions.
The final 3.12 suite including six renderer-boundary tests passed **92 tests**.
These are exact candidate-versus-oracle comparisons within each profile, not a
claim that trajectories are bit-identical across MuJoCo versions. The single
Oscar job repeats these tests under its Linux profile before attempting frames.
Until GPU evidence passes, 3.12 remains a candidate unified profile. Physics is
always `DELEGATED_CPU`, including when State tensors and rendering use CUDA.

```python
from stable_envs.envs.ogbench import OGBCube

with OGBCube(device="cuda", seed=17) as env:
    full = env.get_state()
    pixels = env.to_render_state(full)
    frame = env.render(pixels)  # GPU gate pending
```

CPU rendering refuses explicitly. Only `auto`/`mjwarp` rendering and the registered
`front_pixels` camera are supported. Historical data collection, multiview and
render-goal modes remain outside the certified registration. No CPU-to-GPU
cross-version bridge claim is needed or made at this stage.

## RenderState schema and model ownership

`to_render_state` returns an owning numeric TensorDict with batch `[B]`.
`render(full)` projects the same schema accepted by `render(render_state)` and
private `_render(render_state)`. The older flat qpos/mocap/button packing remains
an internal compatibility projection; it is not sufficient input to this renderer.
`set_state` explicitly rejects RenderState as incomplete dynamics State.

| Key | Per-row shape | dtype |
| --- | --- | --- |
| renderer.signature | 32 | uint8 |
| renderer.size, supersample, camera | 1 each | int64 |
| pose.qpos | Cube 21 / Scene 25 | float64 |
| pose.mocap_pos, mocap_quat | 3, 4 | float64 |
| appearance.cube.size | 1 | float64 |
| appearance.agent.color | 3 | float64 |
| appearance.camera.angle_delta | 1 × 2 | float64 |
| appearance.light.intensity | 1 | float64 |
| appearance.floor.color | 2 × 3 | float64 |
| appearance.cube.color | 1 × 3 | float64 |
| appearance.target_alpha | 1 | float32 |
| appearance.lock_color (Scene) | 2 × 3 | float64 |
| pose.buttons (Scene) | 2 | int64 |

Cube has 14 leaves and Scene 16. Validation covers exact nested keys, shapes,
dtypes, finite values, color/intensity and size ranges, unit quaternions, button
bits, family/asset/schema signature, one resolution per batch, and the supported
camera and target alpha (0.2). The schema signature is serialization-compatible;
the runtime separately enforces exact renderer package versions.

Size, agent color, camera, lighting and floor come from **compiled** configuration,
not pending MJCF edits. Cube/target RGB and Scene locks/buttons come from current
immediate appearance. Target visibility is fixed by the registered mode. There
are no RNG, warmstart, task, lifecycle, or arbitrary Python-object leaves.

The renderer compiles an independent upstream OGBench model through the audited
stable-envs compatibility hooks, without reset or step. Immutable static
configuration keys namespace the existing frozen model/context/plate caches.
Cube and floor RGB use the existing per-world color paths. The mocap upload hook
receives the explicit orientation. No production code imports stable-worldmodel.

## Scene diagnosis and frozen baseline

The frozen `_ogbench_make("Scene", ...)` calls upstream `SceneEnv` without its
required `env_type`; OGBench 1.2.1 raises `TypeError`. This is constructor wiring,
not an unsupported registered task. The new model-provider boundary supplies
`CompatibleScene` (registered scene configuration), then calls the inherited
frozen drawing implementation. The frozen constructor is not patched globally.

`stable_envs/envs/ogbench/manipulation.py` remains SHA-256
`bac4b1ea3e29a04faa5eeec33de0c212cb87bbf488ae0466ea762bd3a140576a`.
Its render, plate, camera corrections, thresholds and rasterization are unchanged.
The local 35-case model-input audit matches source geometry, materials, lights,
meshes and the selected camera. The source's unused extra cameras are irrelevant
to the selected registered view. Default output sizes remain Cube 224 / Scene 200.

The GPU comparison is **exact candidate versus the unchanged frozen drawing
algorithm configured from an independently constructed pinned SWM model**.
Real source MuJoCo frames are also retained with their residual metrics. The
historical frozen renderer has visual-tier differences from source MuJoCo pixels;
these are not relabelled exact and no threshold is relaxed. Any difference from
the frozen comparison fails the gate and preserves the frames.

## Context lifetime and scope

The frozen renderer deliberately retains private EGL plate rigs for process
lifetime, because closing one can invalidate another context. That established
policy is preserved. Dynamic MJWarp contexts keep the frozen bounded LRU policy;
instance close releases instance-owned tables. Rendering is serialized at the
new caller boundary. Repeated calls and independent environments using an
existing configuration must add no plate rig. New static configurations retain
additional process-lifetime rigs; this checkpoint does not claim an unbounded
configuration sweep has bounded memory. GPU evidence must record actual context
counts and release failures. Changing the frozen lifetime policy requires review.

## Evidence and acceptance

The local evidence directory is
[ogb_renderer_closeout](semantic_recovery_evidence/ogb_renderer_closeout/).
The CPU performance report is [CPU closeout](ogbench_cpu_closeout.md).
Fixture generation produced 35 cases across seeds 0/17/42, task IDs 1/2/5,
appearance and size endpoints, all Scene button patterns, and real contact
trajectories for cube/button/drawer/window. Visual-only fixtures retain identical
physics State. The Oscar gate records full-State, named/private, frozen-baseline
and source frames, mutation checks, variation changes, batch/chunk/order checks,
and independent-instance/resource results separately for each family.

One immutable source/oracle bundle and one non-requeued Oscar job are authorized.
Only exact per-family evidence, the complete terminal marker, hash verification,
and readiness/taxonomy gates can justify promotion. A pending or failed job is
not acceptance. GPU status and exact resumption commands are recorded in the
checkpoint handoff after submission. No push is authorized or performed.

## Oscar outcome and resubmission boundary

Job **6669720** ran once on `gpu2708` (NVIDIA L40S), `gpu` / `norm-gpu`,
no requeue. It ended **FAILED**, exit `1:0`, after 57 seconds. Both CPU
selectors exited 4 during `conftest` loading. The archive lacked
`tests/fixtures/regression/support_matrix.json`, read transitively through
`tests.helpers.reference.crosswalk`. This was an archive preparation error,
not evidence of a CPU semantic or renderer mismatch. Both renderer exits are
99 (not started); no baseline or candidate frames were produced. The exact
terminal marker is `OGB_CUBE_SCENE_GPU_GATE_FAIL`.

The actual remote archive and all 12 contained files were downloaded and their
hashes verified against the remote hashes. See
[verification](semantic_recovery_evidence/ogb_renderer_closeout/remote-verification.json)
and [outcome](semantic_recovery_evidence/ogb_renderer_closeout/gpu-outcome.json).
No success marker exists, and neither registration is promoted.

The correction changes only archive composition: `ogbench_renderer_bundle.py`
archives the entire tracked source tree, including transitive harness fixtures.
The corrected bundle uses the same committed production code, oracle, runner,
profile and comparisons. It is prepared locally at
`/private/tmp/ogb-renderer-294675f0-complete`; no second job has been submitted.
A fresh extraction is tested locally before the resubmission review.
Exact commands and hashes are in the checkpoint handoff.
