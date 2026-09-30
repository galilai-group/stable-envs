# Scene GPU scaling evidence

Oscar job **6810313**, RTX 3090, immutable accepted Scene kinematics source.
See `job.json`, `harness/plan.json`, and the preserved scheduler output for
allocation and source provenance. No old evidence was overwritten.

- `results/30/result.json`: uninstrumented paired render timings and MSE for
  B256,512,1024,2048,4096,8192,16384, with CPU and CUDA image destinations.
- `results/30/profile_b*/`: separate instrumented attribution and cProfile data.
- `results/30/telemetry/`: separate sustained ours/source phases; raw sensor
  readings, process snapshots, and summaries.
- `results/30/tool_inventory/`: available diagnostics and driver-counter policy.
- `prototype_reference/`: copied historical Scene batched feasibility evidence;
  these measurements are not part of the fresh GPU sweep.
- `scene_scaling.csv`: generated after successful completion by `summarize.py`.
- `sha256.json`: hashes of the completed evidence and harness.

The 16 distinct saved States are repeated to fill each batch; every image is
rendered, not fetched from an image cache. Public rendering, output assembly,
copies, and synchronization are timed; disk loading and State preparation are
not. The CUDA renderer's public return is a CUDA tensor, so the CPU-output row
includes the copy back to CPU. Source uses the same GPU, batch, and requested
image destination. CPU processing uses one configured Torch/BLAS worker.

FPS is `total_frames / total_time`; `fps_sample_sd` is the sample SD of individual
batch throughput measurements. Latency SD and MSE SD are separate columns.
Paired timing intervals are descriptive Student-t intervals, with no guarantee
of independence under clock drift or serial correlation. Small percentage
differences must not be presented as large throughput gains.

GPU busy and memory busy are sampled activity, not MFU, occupancy, or bandwidth
saturation. Device-wide VRAM includes both warmed implementations and allocator
caches. Xorg is present on the measured GPU; raw process snapshots retain that
fact. CPU duty includes graphics-driver work and spinning/waiting. Telemetry
phase durations do not supply the headline FPS. NVIDIA counter definitions:
https://docs.nvidia.com/deploy/nvidia-smi/index.html#utilization

The companion interpretation is `../../scene_gpu_scaling_20260929.md`.
