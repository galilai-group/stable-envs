# Renderer performance follow-up, 2026-09-29

This checkpoint addresses Scene GPU overhead, the four measured CPU DMC regressions, and source-GPU availability for eight registrations. Craftax optimization, Rocket recovery, PushT recovery, fidelity presets, and semantic acceptance changes are outside this round. No push.

## Scene

The first candidate removes the duplicate full forward pass, groups exact appearance configurations per batch, extracts host arrays once, and renders into preallocated output. RTX3090 job 6805950 passed saved-State and image checks at B64/128/256. At B256 CPU output, ours improved from 853.97 to 1,140.58 FPS within that allocation; paired source was 1,369.00 FPS, so the candidate remained 0.833× source.

A separate candidate replaces full forward dynamics with guarded render-only kinematics and camera/light updates. It sets the same final button/lock colors on renderer-owned model data. Unsupported camera, tendon, model, or visualization features retain full-forward fallback. Geometry equality across 128 local poses and final image equality on Oscar support this change; no poses or images are cached.

RTX3090 job 6806327 passed all tested batches and output destinations. B256 CPU output: ours 1,355.79 FPS, source 1,370.20 FPS (0.9895×), normalized MSE 0. Mean±SD is 0.73758 ± 0.00449 ms/image versus source 0.72982 ± 0.00842 ms/image. CUDA output: ours 1,375.66 FPS, source 1,355.71 FPS (1.0147×), MSE 1.60e-11. This is approximately source performance, not a large source-relative win. All CPU regression gates passed. The exact tested V2 implementation is now adopted for both devices.

## CPU DMC

The measured cost is ray arithmetic, not State preparation. In instrumented B64 runs, bounding rejection accounted for ~73% of BallInCup and ~52% of Finger rendering time. Pose/FK preparation was under 0.2%. Humanoid also spends substantial time in bounds, capsule intersections and shading. Profiling overhead makes these diagnostic wall times unsuitable as headline FPS.

Chunk sweeps do not support one global smaller batch. Humanoid group 4 improved B64 from 8,423 ms to 5,846 ms (1.44×) with identical pixels. Finger improved at most ~8%, Manipulator ~16%, and BallInCup regressed with smaller groups. The Humanoid-specific working-set change passed its complete same-allocation gate (job 6806273): at B256, baseline 7.603 FPS improved to 10.764 FPS (1.416×), versus source 10.811 FPS (0.9956×). This is a near tie with source. Full-batch image hashes remained identical; normalized MSE is 0.0004328633 ± 0.0000732948. Current Humanoid lifecycle/projection/render tests also passed on Oscar.

A separate CPU Inductor experiment fuses only the existing bounding predicate. Its exact predicate/image and compilation-count checks are recorded, but its modest gains do not yet justify adding default CPU compiler/platform/startup complexity. It remains diagnostic-only; no raycast compiler dispatch is changed.

## Source GPU availability

The pinned source implementations for Acrobot, CartPole, both MountainCars and Pendulum render to CPU pygame surfaces; SimplePointMaze uses CPU Matplotlib/Agg. Their GPU-candidate versus CPU-source comparisons are legitimate fallbacks and explicitly labeled separately from same-device comparisons.

Piecewise and TwoRoom have CUDA-configurable upstream Torch rasterizers. Source-owned tensors are placed before timing; unchanged public render still returns CPU NumPy, and that transfer remains timed. Both passed source State and image gates on RTX3090, array 6805885, at B64/128/256. At B256 CPU output: Piecewise 52,539.84 versus source 806.79 FPS (65.12×), TwoRoom 25,444.33 versus 653.20 FPS (38.95×); MSE approximately 8e-9. These are comparisons to an unchanged source public API, which renders batch lanes serially. Source GPU versus saved CPU pixels also differ by only approximately 8e-9 MSE. No timing ratio across distinct CPU/GPU allocations is inferred.

## Scope and provenance

Main CSV: `semantic_recovery_evidence/paired_saved_rendering_20260929/paired_rendering_summary.csv`. Its explicit `export_inputs.json` selects evidence; `regenerate.py` recreates exports without rerunning experiments. Every replacement supplies both ours and source from the same paired run. Original evidence remains untouched. Supplemental A5500/L40S results are separate from RTX3090 data.

These are warm rendering measurements from 16 distinct disk-saved States repeated into batches, not 1,000 independently sampled States and not simulation throughput. Disk loading, source State installation, RenderState projection and compilation are outside timers. Source geometry is already prepared; our public renderer performs its internal geometry update during rendering. Timing SD and measured pixel MSE/SD are retained.
