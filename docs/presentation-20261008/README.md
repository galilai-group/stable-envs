# Presentation measurements — October 8, 2026

Start with [the readable summary](summary.md). This is a live collection, not an all-passing result. `PENDING` is not zero FPS.
Craftax is excluded. The requested batches are 1, 64, 256, 1024 and 2048.

| File | What it measures |
| --- | --- |
| [render-only-cpu-gpu.csv](render-only-cpu-gpu.csv) | Frames from already prepared, saved states; no rollout or state preparation in the timer. |
| [online-cpu-gpu.csv](online-cpu-gpu.csv) | Public stepping and all requested frames together, at horizons 1 and 8. |
| [step-cpu-gpu.csv](step-cpu-gpu.csv) | Public stepping without frames, including the candidate API's final State. |
| [slower-registration-plan.csv](slower-registration-plan.csv) | Per-registration slowdown tally and next steps; measured causes are distinguished from hypotheses. |
| [slower-render-cells.csv](slower-render-cells.csv) | Every measured render cell numerically slower than its source comparator; small differences can be noise. |
| [gpu-utilization-source-vs-ours.csv](gpu-utilization-source-vs-ours.csv) | Matched Nsight Systems source/ours SM activity, instruction issue, shader activity and DRAM bandwidth at each requested batch. |
| [hardware-utilization.csv](hardware-utilization.csv) | Separate Nsight Compute replay at batch 256; CUDA kernel compute and DRAM throughput. |

The corresponding files without `-cpu-gpu` contain the full measurements,
standard deviations, device identities, resource samples and evidence paths.
Use `status=COMPLETE` for conclusions, and check the actual render devices.
GPU-output comparisons against a CPU-only source include its upload and are
labelled as fallbacks. PushT currently draws on the CPU even with CUDA output.

## Comparability

- CPU columns deliver CPU images; GPU columns deliver GPU images. Render-only
  detail files also retain the alternative GPU-render-to-CPU measurements.
- Render-only preserves the previous advisor protocol: one CPU thread, pinned
  source, identical saved poses, alternating timing order. Existing small-batch
  results are reused explicitly; new large-batch results use `65ec1f38`.
- Online/step use eight allocated CPU cores and up to eight source processes.
  Candidate execution uses the production adapter's policy. Each side runs in
  a fresh process. Initialization and per-sample resets are outside timers.
  Source stepping does not export candidate-shaped State inside its timer.
  Step rates count valid transitions; nominal masked-lane rates remain in detail files.
- All pinned source simulation in this non-Craftax cohort runs on the CPU.
  The step table's GPU configuration includes GPU result delivery; it is not
  a source GPU physics measurement. Fetch CPU uses explicit `backend="opengl"`.
- Online captures all H frames. Task values and validity masks must agree;
  the pixel gate requires every compared frame to have normalized MSE <= 0.001.
  `MEASURED_PIXEL_GATE_FAILED` keeps timings and full MSE distributions while
  explicitly failing image acceptance. Five timed repeats
  follow warmup. These checks do not replace full snapshot acceptance tests.
- Online sizes are 64 for Fetch/DMC/OGBench/PushT, 96 for Rocket, and native
  sizes otherwise. Saved-render corpus sizes can differ. Do not subtract these
  tables' rates to estimate stage costs across different resolutions/workloads.
- The saved render corpus has 16 independent poses for most rows, 1,024 for
  PushT. Repeated copies fill larger batches. They are not new independent poses.
- Source Fetch PickAndPlace rendering mutates continuation: its rendered-online
  protocol remains unsupported here. Do not fill it with step-only results.

## GPU utilization is not MFU

MFU normally means useful *model* FLOPs divided by theoretical hardware FLOPs
available over elapsed time. This renderer suite has no defined useful-model
FLOP count, so `mfu_percent` is empty with `UNDEFINED_USEFUL_FLOPS`, not zero.

The measured alternatives are:

- Cross-API device counters: Nsight Systems samples every millisecond during
  labelled, synchronous render calls for source and ours. These cover OpenGL
  and CUDA, with SM activity, instruction issue, shader activity, and DRAM read/
  write bandwidth reported separately. Samples are device-wide, not process
  counters; brief calls have fewer samples. These are separate diagnostic runs,
  not FPS measurements. High activity alone is not evidence of efficiency, and
  graphics fixed-function work is not fully described by SM activity.

- Sampled GPU busy time and VRAM: NVIDIA-SMI every 200 ms. Whole-run samples
  include setup, warmup and checks; they are labelled accordingly. Timed-window
  samples are separate and can be absent for short calls. Sampled peaks can miss
  short allocations. Torch allocation peaks cover Torch, not all GL/Warp memory.
- CUDA compute/DRAM throughput: duration-weighted Nsight kernel counters in a
  separate replay. These omit CPU idle gaps, transfers and OpenGL shaders, and
  must not be presented as whole-application efficiency or source OpenGL MFU.
  Profiler-inflated timing is never used as table FPS.

See NVIDIA's [utilization definition](https://docs.nvidia.com/deploy/nvidia-smi/index.html)
, [cross-API GPU counters](https://docs.nvidia.com/nsight-systems/UserGuide/index.html),
and [roofline guidance](https://docs.nvidia.com/nsight-compute/ProfilingGuide/index.html).

## Reproduction and preservation

[provenance.json](provenance.json) hashes the measurement scripts and collected
archives. Local scripts and evidence are preserved under
`branch-backups/presentation-20261008/`; the working collection is
`/private/tmp/se-presentation-20261008`. Oscar outputs live under
`/users/sboughan/se-presentation-20261008`. Large temporary data use node-local
storage; no scratch evidence has been removed or rewritten.

The historical rendering harness is staged outside the checkout. It has not
been restored as maintained production code. The online runner reuses pinned
source bridges and adds a shared process/timing layer; it remains campaign code
until its matrix and failure handling have been reviewed.
