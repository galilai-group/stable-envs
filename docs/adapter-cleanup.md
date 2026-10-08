# Adapter cleanup

Work starts at `24fa12f7` on an isolated `adapter-cleanup` worktree while the
repository cleanup proceeds separately. Public environment and stepping APIs
are unchanged. Craftax simulation and performance remain out of scope.

## Decisions

- Share process-selection policy; retain family eligibility, affinity, the Linux
  and daemon checks, B64/four-core thresholds, and the eight-process cap.
- Use one DMC factory for local and worker construction. Quadruped stays local.
- Describe asynchronous submit/receive/abort through `StepPipeline`. Family
  configuration remains separate from the common lifecycle protocol.
- Expose Fetch object masses as owned metadata instead of reading private storage.
- Share identical buffer allocation, views, and snapshot copies across Classic,
  DMC and OGBench. Fetch retains its distinct flat buffers and task shapes.
- Keep family worker loops, validation, and OGBench prepare/commit restoration.
  Their differences do not justify a universal worker framework.
- Group simulator implementation by family; keep common execution and transport
  separate. Move rendering helpers to their owning environment implementation.
- Retain family-specific source space types: precision, sampling order, and
  default/current-value behavior differ from public spaces. No deduplication
  without independent semantic evidence.

## Validation in progress

Execution selection: 251 routing cases passed across all four families.
Initial process/sequence tests: 290 passed, 86 skipped locally; skips are not
acceptance. Fetch's existing Mac environment fails inside Gymnasium Robotics
with MuJoCo 3.12 on both the baseline and candidate. A temporary MuJoCo 3.11
overlay is used for the pinned Fetch checks. Static interface assignments pass. The pinned Fetch run passed all 31 tests;
OGBench routing/process checks passed 73 tests with one CUDA skip.

Required before closure: relocated imports and registry checks, family process
and source contracts, CUDA output and pipeline checks, and same-host baseline
versus candidate online timings at horizons 1 and 8. Rerun the saved-state advisor
table independently: MSE, source and candidate CPU/GPU FPS, time mean/SD, equal
batches and output destinations, explicit CPU-only reference qualifications.
Use fresh output directories; preserve previous measurements and fixture bytes.
