# Scene GPU: residual costs and batch scaling

This experiment profiles the accepted Scene OpenGL implementation after the
render-only kinematics optimization. It does not change the renderer, quality
settings, or accepted evidence. Oscar job 6810313 runs on an RTX 3090; the
immutable candidate snapshot and harness hashes are recorded in
`semantic_recovery_evidence/scene_gpu_scaling_20260929/`.

The sweep uses batches 256 through 16,384, doubling each time, with CPU and CUDA
image destinations measured separately. Every lane executes rendering. Sixteen
distinct saved States are repeated to fill each batch; this is not a new
16,384-State quality corpus. Disk loading, source State installation, and
RenderState projection are outside the timer. Public rendering, output stacking,
copies, and synchronization are inside it. Source and candidate use the same
GPU, batch size, and image destination, with at least seven alternating pairs.
The source starts with its installed derived geometry; our render call performs
its own pose updates. This is the existing prepared-input API comparison.

## What the current path does

The benchmark explicitly selects Scene's OpenGL fidelity backend. It loops over
worlds, updates each scene, draws, reads the image back to host memory, and uploads
the assembled batch to satisfy its CUDA tensor return contract. For the CPU-image
comparison the harness then copies that tensor back to CPU; source already
returns host images. For CUDA images both implementations upload their host
images. These API conversion costs stay inside the timer. Increasing the outer batch
does not make those worlds render simultaneously. Cube's accepted batched
MJWarp compositor is a different implementation.

At B256, the fresh instrumented candidate render averages about 193 ms.
Approximately 48% is inside native drawing and 28% inside pixel readback.
Those calls include CPU driver work and GPU waits, so these are wall-time
attributions, not GPU execution durations. Button colors cost about 3.2%, scene
updates 2.3%, kinematics 1.0%, and validation 0.7%. Full dynamics is absent from
the measured fast path. Source rendering is dominated by the same draw/readback
operations.

Separate sustained phases initially measure about 37% GPU busy time for ours
and 38% for source, with roughly one process CPU core occupied. Memory-busy time
is about 11%. These counters describe activity, not MFU, SM occupancy, peak
bandwidth utilization, or separate graphics-engine saturation. The persistent
Xorg server is also visible on the device (5 MiB); the raw process snapshots
retain that possible device-counter contamination. Telemetry timings are not
used for the throughput results. CPU duty includes driver spin/wait; it does
not establish that Python computation is the bottleneck or quantify achievable
GPU headroom. VRAM measurements include both warmed implementations and allocator
caches in the same process, so they are not isolated renderer memory footprints.

## A genuinely batched Scene path remains a separate optimization

The existing batched implementation has concrete fidelity gaps: its baked
background composition loses moving shadows, and transparent moving geometry
requires additional handling because the pinned MJWarp path ignores alpha.
Shadow shading and antialiasing also differ. No z-fighting cause was established
in the earlier probes.

An earlier experimental two-pass compositor with 2× supersampling and corrected
shadow transmission passed all 132 tested frames at maximum normalized MSE
0.000933. That is useful feasibility evidence, but the shader modification was
process-global and the prototype never passed the production 1,000-frame gate.
Its recorded B256 cost was 1.289 ms/frame versus 0.557 ms/frame for that run's
older MJWarp baseline. It repeated one pose and excluded some public API work;
it is not evidence of beating source GPU. The original probe and raw results
are preserved in this experiment's `prototype_reference/` directory.

The next substantial Scene optimization therefore belongs in the existing
batched shading/composition path: context-local shadow handling, correct
transparent-layer composition, and less expensive antialiasing/multiple-pass
work. Both quality and same-device speed need independent acceptance gates.
The present sweep determines how far the current serial path scales; it cannot
establish the eventual performance of that different implementation.

## Final measurements

Job 6810313 completed successfully in 19m07s. All 14 measurements passed
State/input nonmutation and batch-lane checks. Mean normalized MSE is between
zero and 1.60e-11 across the sweep; the sixteen telemetry phases completed.

| Batch | Ours FPS, CUDA output | Source FPS, CUDA output | CUDA-output speedup | CPU-output speedup |
|---:|---:|---:|---:|---:|
| 256 | 1,384.1 | 1,372.1 | 1.0088× | 0.9866× |
| 512 | 1,377.2 | 1,364.4 | 1.0094× | 0.9700× |
| 1,024 | 1,382.8 | 1,376.8 | 1.0044× | 0.9719× |
| 2,048 | 1,388.4 | 1,372.6 | 1.0115× | 0.9759× |
| 4,096 | 1,383.0 | 1,336.9 | 1.0345× | 0.9770× |
| 8,192 | 1,392.2 | 1,376.5 | 1.0114× | 0.9807× |
| 16,384 | 1,396.9 | 1,378.8 | 1.0131× | 0.9859× |

There is no CPU-output crossover through B16,384. CUDA output has a small
measured advantage, but increasing the batch 64-fold from 256 to 16,384 raises
our FPS by only 0.9%. The 3.45% relative lead at B4,096 coincides with a slower,
noisier source sample (2.06% latency SD), rather than improved candidate FPS.
The B1,024 paired timing interval includes a tie. These observations do not
support a claim of substantial batch scaling.

With CUDA output, GPU busy time ranges from 36.4–38.7% for ours and 37.2–39.2%
for source across the measured telemetry phases. At B16,384 it is 38.7±5.6%
versus 37.2±8.4%; process CPU duty is about 100% of one core on both sides.
The B4,096 attribution reproduces B256: native drawing 47.7%, readback 29.1%,
kinematics 1.0%, validation 0.08%. The dominant serial work remains per-image.

`semantic_recovery_evidence/scene_gpu_scaling_20260929/scene_scaling.csv` retains
all batches, both destinations, latency/FPS/MSE SD, descriptive paired timing
intervals, and telemetry. The main fixed-B256 table selects the fresh B256 row;
it does not substitute the most favorable batch. The final evidence manifest
hashes the harness, raw results, process snapshots, profiler traces, and logs.
