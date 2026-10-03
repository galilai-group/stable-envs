# Fetch direct-source prerequisite and render-callback lifecycle audit

Base checkpoint: `cc255691298dc877dd4d547e1011ab695781dbb0`. Production code,
public API, process routing and accepted renderer behavior are unchanged. All
measurements here are CPU source/adapter correctness or constructor inspection;
**no image, GL, GPU, FPS or speedup result is produced**.

## Source/adapter State prerequisite: all twelve pass

`cpu-prerequisite.json` records 48 independent B2 cases: all 12 registrations,
seeds 17/42, float32/float64 actions. Each runs eight matched transitions, restores
complete State into an adapter with a different seed and five-step history, then
checks eight more transitions against actual pinned source environments. The
comparison independently reads qpos/qvel/time, warmstart, gripper-controller
position/quaternion, goal/reset anchor, lifecycle and RNG. Task reward, success,
termination and truncation match exactly. Retained snapshots remain owned;
unseeded resets reproduce the source RNG. Ordinary step is guarded against
calling set_state. This uses real accepted adapters, not public Env construction
or process transport, and does not certify visuals or timing.

## Render callback: nine unchanged, three change continuation

`callback-summary.csv` summarizes **576 actual source cases**, 48 per registration:
seeds 0/17/42, float32/64, zero/random actions, prefixes 0/1/8/49. Two independent
source worlds execute the same prefix. Only one then executes the unmodified
upstream `_render_callback`; both execute eight more identical actions. Full
per-step differences and source call paths are preserved losslessly in
`callbacks.json.gz`. The real renderer is never called; GL paths are guarded.

| Registrations | Cases | Immediate State changes | Subsequent State/output changes |
|---|---:|---:|---:|
| Reach / ReachDense / ReachDict | 144 | 0 | 0 |
| Push / PushDense / PushDict | 144 | 0 | 0 |
| Slide / SlideDense / SlideDict | 144 | 0 | 0 |
| PickAndPlace / Dense / Dict | 144 | 108 | 108 |

Each PickAndPlace registration changes in all 36 post-step cases; its 12 reset-only
cases do not change. Only the gripper controller position/quaternion fields change
immediately in the audited continuation snapshot. Subsequent physical states and
task outputs also diverge, so this is not just an image or harmless display-cache
difference. No tolerance, cache repair or source function patch is applied.

The inspected robotics `render()` invokes `_render_callback()` before drawing.
That callback moves the visual target and calls `mj_forward`, recomputing derived
body poses. Reach/Push/Slide have `block_gripper=True`; their normal `_step_callback`
already calls `mj_forward`. PickAndPlace has `block_gripper=False`, so it omits
that recomputation after a step. Its next action consumes the gripper body-pose
cache; rendering therefore changes its subsequent controller input. This explains
the observed difference and is consistent with the earlier retained full-render
PickAndPlace mutation diagnostics. This callback-only run does not replace actual
CPU/GPU render nonmutation checks for any registration.

The next benchmark must preserve this distinction. The nine unaffected registrations
can proceed to ordinary online source-image gates. For PickAndPlace, a naïve
step/render loop cannot simultaneously claim unchanged no-render dynamics and
matched trajectories. Measure no-frame stepping and saved-State rendering separately
while resolving that online protocol. If a separate source render world is used,
its State transfer/restore work must be counted and labeled; never silently repair
source caches, teleport the candidate between steps, or advertise such an adapted
source pipeline as an untouched native baseline. A native-source pipeline may
still be timed, but its differing continuation semantics must not be hidden behind
a matched-trajectory/MSE claim. No production semantic contract is redesigned here.

## Source image dimensions also need explicit provenance

`source-dimensions.json` inspects all 12 registered constructors in three configurations
without drawing (36 checks):

| Source configuration | Internal raster size | Wrapper output size |
|---|---:|---:|
| Default | 480x480 | 224x224 |
| `resolution=64` | 480x480 | 64x64 |
| `resolution=64, width=64, height=64` | 64x64 | 64x64 |

The pinned wrapper calls `cv2.resize` after upstream rendering. Changing just the
output size does not reduce the source's internal raster work. The two 64-output
configurations are not assumed pixel-equivalent: actual MSE/quality checks are
required before comparing their timing. Future source rows must record both sizes,
include native source resize/readback/delivery in timing, and preserve the source
configuration used by the matched-image oracle. No source slowdown or speedup is
inferred from constructor inspection alone.

## Dependency profile and reproduction

The source is stable-worldmodel `176d54532b3c0a251a4f509b74f2fdacc143a63d`, tracked
clean during the callback audit. Source and runtime file hashes are recorded.
The CPU run uses Gymnasium Robotics 1.4.2, Gymnasium 1.3.0, NumPy 2.5.3 and MuJoCo
3.11.0, matching the accepted Fetch pipeline's MuJoCo version.

The local base venv now has MuJoCo 3.12.0, where Fetch construction fails in
`mujoco_utils.set_joint_qpos`: tuple membership compares the joint enum against a
NumPy scalar in the opposite direction from the successful direct comparison.
`mujoco312-compatibility.json` retains the failure and equality diagnostic. A
MuJoCo 3.11.0 wheel was installed into an isolated temporary directory with
`--no-deps`; the base venv and upstream files were not modified. The dependency
manifest records the loaded wheel's Python/binary hashes. This is a local profile
compatibility observation, not a general cross-platform MuJoCo bug claim.

```bash
uv pip install --python /path/to/fetch-python --target /tmp/fetch-mujoco311 --no-deps mujoco==3.11.0
PYTHONPATH=/tmp/fetch-mujoco311:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /path/to/fetch-python -m scripts.benchmarks.online_stepping.fetch_source_render_lifecycle \
  --output /tmp/new-fetch-callbacks.json
PYTHONPATH=/tmp/fetch-mujoco311:. OMP_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1 MKL_NUM_THREADS=1 \
  /path/to/fetch-python -m scripts.benchmarks.online_stepping.fetch_source_cpu_prerequisite \
  --source-manifest docs/research/semantic_recovery_evidence/fetch_source_prerequisite_20261003/source_manifest.json \
  --output /tmp/new-fetch-cpu-prerequisite.json
```

Both commands refuse existing output files. The callback archive decompresses to
the original JSON byte-for-byte; its uncompressed hash is recorded in `review.json`.
`profile_probe.py` is the recorded constructor/dependency inspection, not a public
API or GPU benchmark. The overall matrix remains 39 VERIFIED / 3 INCOMPLETE and
all twelve Fetch registrations remain accepted; this is a source-baseline protocol
constraint, not a new semantic demotion. Source CPU/GPU timing, shared transport,
full-render correctness and larger/sustained workloads remain open. Oscar still
rejects SSH authentication; existing submitted jobs were not restarted.
