# Renderer profiling — completed 2026-09-28

**All 42 registrations accounted for: 97 completed renderer cases and four
NO_RGB device entries.** CPU and GPU profiles cover public full-State
rendering, validation/preparation, prepared rendering, and prepared-input
batches 1, 8, 64 and 256. All rendered cases passed batch-repeat and
nonmutation checks. The two PushT entries remain diagnostic only; symbolic
Craftax entries have no RGB renderer.

- [Findings, priorities and methodology](semantic_recovery_evidence/renderer_profiles_20260928/README.md)
- [Interactive table](semantic_recovery_evidence/renderer_profiles_20260928/dashboard.html)
- [42-row CSV](semantic_recovery_evidence/renderer_profiles_20260928/per_environment_42.csv)
- [All timings](semantic_recovery_evidence/renderer_profiles_20260928/timings.csv)
- [Python function profiles](semantic_recovery_evidence/renderer_profiles_20260928/functions.csv)
- [Torch/CUDA operator profiles](semantic_recovery_evidence/renderer_profiles_20260928/torch_operators.csv)

The principal optimization targets are Craftax's validation/conversion path,
Cube/Scene/Rocket full-State preparation, BallInCup CPU ray casting, classic
control's small GPU launches, and the large-batch OpenGL fidelity cost—most
notably Walker's 27.54× observed cost relative to its fast GPU backend.

Worktree `/Users/samib/stable-envs/.worktrees/render-config-fixes`, branch
`repair/render-config-fixes`, HEAD
`df80c9c54590afa9fc0482a4dcb241d21f6b63a3`. All 13 accepted renderer overlay
file hashes remain unchanged. No production optimization, semantic promotion,
or push occurred. Existing uncommitted renderer work and prior evidence were
preserved.

Oscar root:
`/oscar/scratch/sboughan/se-semantic-recovery/renderer-profiles-df80c9c5-20260928`.
Measured production source: immutable parity `source-v12`. Pilots 6759929 /
6759930, initial CPU bulk 6759954, CPU retries 6760005, norm-GPU sweep 6760006.
Final packaging job 6760082 completed successfully and validated all 101
statuses, all 291 profiler records, production hashes and backend-allocation
consistency. Both archives were downloaded and SHA-256 verified locally.

A condo array remained blocked by `QOSGrpGRES` and was cancelled before any
measurement. A diagnostic helper filename collided with Python's `inspect`
module, causing 29 CPU jobs to fail before measurements; renaming the helper
and retrying resolved this. Failure and retry logs remain in the raw archive.
No completed output was overwritten. The unrelated training job 6492167_1
was left untouched.

The five harness tests and lint/compilation checks pass. Performance samples
are 3–30 warmed calls per probe, with sample SD and retained outliers; this is
not a rerun of the prior 1,000-frame source-MSE campaign. Higher-batch timings
exclude State preparation. GPU hardware differs between some environment
jobs, and profiling overhead is excluded from latency measurements. Full
limitations and observed backend tradeoffs are in the report.
