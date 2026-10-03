# Default-renderer DMC pipeline measurement smoke

Job 6968051 runs Humanoid at B64, T1/8 on an RTX 3090 allocation with eight
physical CPUs. This is a bounded validation of the measurement harness before
expanding across the eleven accepted DMC process routes and B64/256/1024.
Quadruped is excluded pending its independent renderer diagnostic.

Public construction is unpatched and uses backend=auto at 64x64. CPU runs in a
separate software-GL interpreter; CUDA runs with NVIDIA EGL. A probe records
and verifies that configuration. The ordinary pipeline is compared with the
same owners and fused State transport under deferred dispatch. Deferring the
next submit until receive removes overlap without changing the number of
requests or exports. The script asserts exact complete State/task/frame equality
for each batch, horizon and frame selection before retaining paired timings.

The component measurements are task stepping without State capture (including
task packaging, IPC and output transfer), State export including IPC/transfer,
and public rendering of saved States. The first is not a bare physics-kernel
measurement. Components are timed independently and are not additive. The
pipeline always retains all States; it separately measures no frames, the final
frame, and all frames. Steps/s counts transitions and FPS counts emitted images.
For T8/final-only these counts differ by eight.

Each cell uses six synchronized samples, with order reversed for the paired
pipeline controls. Construction, warmup, restore, State preparation for isolated
rendering, and equality checks are outside timing. These are library scheduling
comparisons, not upstream speedups or a new source-image MSE certification.
Same-device upstream comparisons remain required by the overall goal.

The local deferred-dispatch smoke verifies that submit does not start an owner,
receive dispatches exactly once, and the patched control is restored afterward.
Actual CPU/CUDA rendering and throughput require the Oscar reports. Summarize
terminal PASS reports using `summarize_dmc_pipeline.py RESULTS NEW_OUTPUT
--source FROZEN_SOURCE --batches 64`; it requires all component, pipeline and
frame-check cells on both devices. Source manifest and the two harness overrides
are preserved here; no production implementation changes are included.

## V1 failure and isolated V2 retry

Job 6968051 terminated FAILED (1:0), after 4:51. CPU completed all six equality
gates and twelve pipeline cells. CUDA failed before timing because se-step6
lacks mujoco_warp. The raw outputs and log remain in `failed-v1`; CPU-only
success does not establish two-device acceptance.

V2 retains the same frozen production source and benchmark workload. It adds
explicit renderer package versions to the report and installs exactly
mujoco-warp 3.12.0 and warp-lang 1.17.0 with --no-deps into a new scratch target.
The se-step6 environment is not modified, and its MuJoCo/dm-control/Torch/NumPy
versions are retained. The installation's pip report records downloaded hashes.
Both CPU and CUDA rerun with this one fresh profile; reports are not spliced
across attempts. `retry-v2` records the launcher, source manifest and reporting
override. No benchmark success is inferred from submission.
